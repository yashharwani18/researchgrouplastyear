import os
import gc
import sys
import json
import time
import math
import argparse
import random
import csv
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple

import torch
try:
    import torchvision.ops
except Exception:
    pass

if hasattr(torch, 'library') and hasattr(torch.library, 'define'):
    try:
        torch.library.define("torchvision::nms", "(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
    except Exception:
        pass

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import timm
except ImportError:
    raise ImportError("Please install timm: pip install timm")

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split


# ==============================================================================
# 0. GLOBAL SETUP & CONSTANTS
# ==============================================================================

def get_safe_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_safe_device()

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else ("/kaggle/working" if os.path.exists("/kaggle/working") else os.getcwd())
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_ablation")
LOG_DIR = os.path.join(BASE_DIR, "logs")
MASTER_SUMMARY_CSV = os.path.join(LOG_DIR, "ablation_summary.csv")
MASTER_TRACKER_JSON = os.path.join(LOG_DIR, "ablation_tracker.json")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

DEFAULT_DATA_DIR = r"/kaggle/input/datasets/bitsnpieces/ovarian-cancer-and-subtypes-dataset-histopathology/OvarianCancer"

CLASS_NAMES = ['Clear_Cell', 'Endometri', 'Mucinous', 'Non_Cancerous', 'Serous']
NUM_CLASSES = len(CLASS_NAMES)


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==============================================================================
# 1. ATOMIC CHECKPOINTING & EXPERIMENT TRACKER
# ==============================================================================

def atomic_torch_save(obj: Any, filepath: str):
    """Saves torch objects atomically using a temp file to survive power cuts."""
    tmp_path = filepath + ".tmp"
    torch.save(obj, tmp_path)
    if os.path.exists(filepath):
        os.replace(tmp_path, filepath)
    else:
        os.rename(tmp_path, filepath)


class ExperimentTracker:
    """Manages experiment records, resume state, and summary CSVs."""
    def __init__(self, tracker_path: str = MASTER_TRACKER_JSON, summary_csv: str = MASTER_SUMMARY_CSV):
        self.tracker_path = tracker_path
        self.summary_csv = summary_csv
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.tracker_path):
            try:
                with open(self.tracker_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Tracker] Warning loading {self.tracker_path}: {e}. Initializing clean tracker.")
        return {"trials": {}}

    def save(self):
        tmp_path = self.tracker_path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        if os.path.exists(self.tracker_path):
            os.replace(tmp_path, self.tracker_path)
        else:
            os.rename(tmp_path, self.tracker_path)

    def is_trial_fully_completed(self, trial_id: int, num_folds: int = 5) -> bool:
        t_key = f"trial_{trial_id:02d}"
        if t_key not in self.data["trials"]:
            return False
        trial_info = self.data["trials"][t_key]
        completed_folds = trial_info.get("completed_folds", {})
        if len(completed_folds) >= num_folds:
            return all(completed_folds.get(str(f), {}).get("done", False) for f in range(num_folds))
        return False

    def is_fold_completed(self, trial_id: int, fold: int) -> Optional[Dict[str, Any]]:
        t_key = f"trial_{trial_id:02d}"
        if t_key in self.data["trials"]:
            completed_folds = self.data["trials"][t_key].get("completed_folds", {})
            f_data = completed_folds.get(str(fold))
            if f_data and f_data.get("done", False):
                return f_data
        return None

    def record_fold_completion(self, trial_id: int, trial_name: str, fold: int, metrics: Dict[str, Any]):
        t_key = f"trial_{trial_id:02d}"
        if t_key not in self.data["trials"]:
            self.data["trials"][t_key] = {
                "trial_id": trial_id,
                "trial_name": trial_name,
                "completed_folds": {}
            }
        self.data["trials"][t_key]["completed_folds"][str(fold)] = {
            "done": True,
            "metrics": metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save()

    def record_trial_summary(self, trial_id: int, trial_name: str, summary: Dict[str, Any], config_dict: Dict[str, Any]):
        t_key = f"trial_{trial_id:02d}"
        if t_key in self.data["trials"]:
            self.data["trials"][t_key]["summary"] = summary
            self.data["trials"][t_key]["config"] = config_dict
            self.save()
        self._append_summary_csv(trial_id, trial_name, summary, config_dict)

    def _append_summary_csv(self, trial_id: int, trial_name: str, summary: Dict[str, Any], config_dict: Dict[str, Any]):
        file_exists = os.path.exists(self.summary_csv)
        row = {
            "trial_id": trial_id,
            "trial_name": trial_name,
            "mean_acc": f"{summary.get('acc_mean', 0.0)*100:.2f}%",
            "std_acc": f"{summary.get('acc_std', 0.0)*100:.2f}%",
            "mean_macro_f1": f"{summary.get('f1_mean', 0.0):.4f}",
            "std_macro_f1": f"{summary.get('f1_std', 0.0):.4f}",
            "mean_weighted_f1": f"{summary.get('wf1_mean', 0.0):.4f}",
            "mean_kappa": f"{summary.get('kappa_mean', 0.0):.4f}",
            "backbone": config_dict.get('backbone_type', 'densenet121'),
            "dim": config_dict.get('dim', 256),
            "num_blocks": config_dict.get('num_blocks', 4),
            "use_sssm": config_dict.get('use_sssm', True),
            "use_gcaf": config_dict.get('use_gcaf', True),
            "use_hierarchical_cnn": config_dict.get('use_hierarchical_cnn', True),
            "use_peg": config_dict.get('use_peg', True),
            "use_distillation": config_dict.get('use_distillation', True),
            "use_clahe": config_dict.get('use_clahe', True),
            "loss_type": config_dict.get('loss_type', 'focal'),
            "lr_backbone": config_dict.get('lr_backbone', 2e-5),
            "lr_head": config_dict.get('lr_head', 1e-4),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Rewrite CSV cleanly to avoid duplicates
        rows = []
        if file_exists:
            with open(self.summary_csv, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get('trial_id') != str(trial_id):
                        rows.append(r)
        rows.append(row)
        rows.sort(key=lambda x: int(x['trial_id']))

        with open(self.summary_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(rows)


# ==============================================================================
# 2. DATASET & PREPROCESSING
# ==============================================================================

class CLAHE_Transform:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        if cv2 is None:
            return img
        try:
            img_np = np.array(img)
            if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
                img_np = clahe.apply(img_np)
            return Image.fromarray(img_np)
        except Exception:
            return img


def get_transforms(img_size: int = 224, mode: str = 'train', use_clahe: bool = True, mean=None, std=None):
    if mean is None: mean = [0.485, 0.456, 0.406]
    if std is None:  std  = [0.229, 0.224, 0.225]

    tfs = []
    if use_clahe:
        tfs.append(CLAHE_Transform(clip_limit=2.0, tile_grid_size=(8, 8)))

    tfs.append(v2.Resize((img_size, img_size)))

    if mode == 'train':
        tfs.extend([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(degrees=45),
            v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
            v2.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        ])
    else:
        tfs.extend([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])

    return v2.Compose(tfs)


def find_dataset_dir(data_dir: str = ".") -> str:
    candidate_dirs = [
        data_dir,
        os.path.join(data_dir, "OvarianCancer") if data_dir else None,
        "/kaggle/input/datasets/bitsnpieces/ovarian-cancer-and-subtypes-dataset-histopathology/OvarianCancer",
        "/kaggle/input/ovarian-cancer-and-subtypes-dataset-histopathology/OvarianCancer",
        "/kaggle/input/datasets/bitsnpieces/ovarian-cancer-and-subtypes-dataset-histopathology",
        "/kaggle/input/ovarian-cancer-and-subtypes-dataset-histopathology",
        "./OvarianCancer",
        "."
    ]
    for c_dir in candidate_dirs:
        if c_dir and os.path.exists(c_dir) and os.path.isdir(c_dir):
            if any(os.path.exists(os.path.join(c_dir, cls)) for cls in ['Clear_Cell', 'Endometri', 'Endometrioid', 'Mucinous', 'Non_Cancerous', 'Serous']):
                return c_dir
    return data_dir


class OvarianDataset(Dataset):
    def __init__(self, data_dir: str = "."):
        self.samples = []
        target_dir = find_dataset_dir(data_dir)

        if os.path.exists(target_dir):
            detected = sorted([
                d for d in os.listdir(target_dir)
                if os.path.isdir(os.path.join(target_dir, d)) and not d.startswith('.')
            ])
            global CLASS_NAMES, NUM_CLASSES
            if detected:
                CLASS_NAMES = detected
                NUM_CLASSES = len(CLASS_NAMES)

        self.class_names = CLASS_NAMES
        label_map = {name: i for i, name in enumerate(CLASS_NAMES)}
        for root, _, files in os.walk(target_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    parent = os.path.basename(root)
                    if parent in label_map:
                        self.samples.append((os.path.join(root, file), label_map[parent]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return path, label


class SubsetWithTransform(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as img:
            return self.transform(img.convert('RGB')), label


# ==============================================================================
# 3. LOSS FUNCTIONS
# ==============================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, num_classes: int = 5, alpha: Optional[torch.Tensor] = None, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.alpha = alpha if alpha is not None else torch.ones(num_classes, device=DEVICE)

    @torch.no_grad()
    def update_alpha(self, class_f1s: List[float]):
        if not class_f1s:
            return
        f1_tensor = torch.tensor(class_f1s, device=self.alpha.device).clamp(min=0.1)
        target_alpha = 1.0 / f1_tensor
        target_alpha = target_alpha / target_alpha.mean()
        self.alpha = 0.9 * self.alpha + 0.1 * target_alpha

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = loss * self.alpha[labels]
        return loss.mean()


class LabelSmoothedCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels, label_smoothing=self.smoothing)


# ==============================================================================
# 4. MODULAR ABLATION MODEL ARCHITECTURE
# ==============================================================================

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


def stochastic_depth(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if not training or drop_prob == 0.0:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    return x / keep_prob * torch.floor(torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob)


class PEG(nn.Module):
    """Positional Encoding Generator via depthwise convolution."""
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H = W = int(round(N ** 0.5))
        if H * W != N:
            return x
        feat = x.permute(0, 2, 1).view(B, C, H, W)
        return self.proj(feat).view(B, C, N).permute(0, 2, 1)


class SPPFBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 5):
        super().__init__()
        mid = in_c // 2
        self.cv1 = nn.Sequential(nn.Conv2d(in_c, mid, 1), nn.BatchNorm2d(mid), nn.SiLU())
        self.pool = nn.MaxPool2d(k, 1, k // 2)
        self.cv2 = nn.Sequential(nn.Conv2d(mid * 4, out_c, 1), nn.BatchNorm2d(out_c), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class HierarchicalConvBlock(nn.Module):
    def __init__(self, in_c: int, dim: int, kernel_sizes=(3, 5, 7)):
        super().__init__()
        branch = dim // len(kernel_sizes)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_c, branch, k, padding=k // 2), nn.BatchNorm2d(branch), nn.SiLU())
            for k in kernel_sizes
        ])
        self.sppf = SPPFBlock(branch * len(kernel_sizes), dim)
        self.norm = nn.GroupNorm(8, dim)
        self.drop = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [F.adaptive_avg_pool2d(c(x), (14, 14)) for c in self.convs]
        return self.drop(self.norm(self.sppf(torch.cat(feats, dim=1)))).flatten(2).transpose(1, 2)


class SimpleConvBlock(nn.Module):
    """Ablation fallback: Standard single-scale Conv2d branch."""
    def __init__(self, in_c: int, dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        feat = F.adaptive_avg_pool2d(feat, (14, 14))
        return feat.flatten(2).transpose(1, 2)


class BackboneExtractor(nn.Module):
    BACKBONES = {
        'densenet121': lambda: timm.create_model('densenet121', pretrained=True, num_classes=0, global_pool=''),
        'densenet169': lambda: timm.create_model('densenet169', pretrained=True, num_classes=0, global_pool=''),
        'densenet201': lambda: timm.create_model('densenet201', pretrained=True, num_classes=0, global_pool=''),
    }
    BACKBONE_DIMS = {
        'densenet121': 1024,
        'densenet169': 1664,
        'densenet201': 1920,
    }

    def __init__(self, b_type: str, dim: int, dropout: float = 0.3):
        super().__init__()
        b_key = b_type.lower()
        if b_key not in self.BACKBONES:
            b_key = 'densenet121'
        self.backbone = self.BACKBONES[b_key]()
        in_features = self.BACKBONE_DIMS.get(b_key, 1024)
        self.proj = nn.Linear(in_features, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        if f.dim() == 4:
            if f.shape[-1] == self.proj.in_features:
                f = f.permute(0, 3, 1, 2)
            f = F.adaptive_avg_pool2d(f, (14, 14)).flatten(2).transpose(1, 2)
        elif f.dim() == 3:
            f = F.adaptive_avg_pool1d(f.transpose(1, 2), 196).transpose(1, 2)
        return self.dropout(self.proj(f))


class GatedCrossAttentionFusion(nn.Module):
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)

    def forward(self, local_f: torch.Tensor, global_f: torch.Tensor) -> torch.Tensor:
        att, _ = self.attn(local_f, global_f, global_f)
        g = self.gate(torch.cat([local_f, att], dim=-1))
        return self.norm(g * att + (1 - g) * local_f)


class SimpleConcatFusion(nn.Module):
    """Ablation fallback: Linear concatenation without cross-attention gating."""
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, local_f: torch.Tensor, global_f: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(torch.cat([local_f, global_f], dim=-1)))


def associative_scan_ssm(delta: torch.Tensor, A: torch.Tensor, Bx: torch.Tensor) -> torch.Tensor:
    B, L, N = delta.shape
    F_mat = torch.exp(delta * A.view(1, 1, -1))
    b_mat = delta * Bx
    h = torch.zeros((B, N), device=delta.device, dtype=delta.dtype)
    h_seq = []
    for f_t, b_t in zip(F_mat.unbind(1), b_mat.unbind(1)):
        h = f_t * h + b_t
        h_seq.append(h)
    return torch.stack(h_seq, dim=1)


class LinearBioSSSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, drop_path: float = 0.0):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)
        self.dt_proj = nn.Linear(d_model, d_state)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.ls = LayerScale(d_model)
        self.drop_path = drop_path
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        res = x
        xi, z = self.in_proj(x).chunk(2, dim=-1)
        B_p, C_p = self.x_proj(xi).chunk(2, dim=-1)
        delta = F.softplus(self.dt_proj(x))
        A = -torch.exp(self.A_log)
        h_fwd = associative_scan_ssm(delta, A, B_p)
        h_bak = associative_scan_ssm(delta.flip(1), A, B_p.flip(1)).flip(1)
        y = ((h_fwd + h_bak) * C_p).sum(-1, keepdim=True).expand(-1, -1, self.d_inner) + z
        out = self.ls(self.out_proj(y))
        return self.norm(stochastic_depth(out, self.drop_path, self.training) + (skip if skip is not None else res))


class StandardTransformerBlock(nn.Module):
    """Ablation fallback: Standard Multi-Head Self-Attention in place of SSSM."""
    def __init__(self, d_model: int, heads: int = 8, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )
        self.drop_path = drop_path

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + stochastic_depth(attn_out, self.drop_path, self.training)
        x = x + stochastic_depth(self.mlp(self.norm2(x)), self.drop_path, self.training)
        return x if skip is None else (x + skip)


class ModularBioModel(nn.Module):
    """
    Configurable proposed architecture with complete component ablation toggles:
    - use_sssm: If False, falls back to StandardTransformerBlock
    - use_gcaf: If False, falls back to SimpleConcatFusion
    - use_hierarchical_cnn: If False, falls back to SimpleConvBlock
    - use_peg: If False, disables PEG
    - use_aux_head: If False, aux logits are None
    """
    def __init__(
        self,
        num_classes: int = 5,
        dim: int = 256,
        b_type: str = 'densenet121',
        num_blocks: int = 4,
        d_state: int = 16,
        dropout: float = 0.3,
        use_sssm: bool = True,
        use_gcaf: bool = True,
        use_hierarchical_cnn: bool = True,
        use_peg: bool = True,
        use_aux_head: bool = True
    ):
        super().__init__()
        self.use_sssm = use_sssm
        self.use_peg = use_peg
        self.use_aux_head = use_aux_head

        # 1. Feature Extraction Branches
        if use_hierarchical_cnn:
            self.cnn = HierarchicalConvBlock(3, dim)
        else:
            self.cnn = SimpleConvBlock(3, dim)

        self.bb = BackboneExtractor(b_type, dim, dropout=dropout)

        # 2. Fusion
        if use_gcaf:
            self.fusion = GatedCrossAttentionFusion(dim)
        else:
            self.fusion = SimpleConcatFusion(dim)

        if use_peg:
            self.peg = PEG(dim)

        self.extractor_drop = nn.Dropout(0.2)

        # 3. Sequential Sequence Modeling Blocks
        dp_rates = [0.1 * i / max(1, num_blocks) for i in range(num_blocks)]
        if use_sssm:
            self.blocks = nn.ModuleList([
                LinearBioSSSMBlock(dim, d_state=d_state + i * 4, drop_path=dp_rates[i])
                for i in range(num_blocks)
            ])
        else:
            self.blocks = nn.ModuleList([
                StandardTransformerBlock(dim, heads=8, drop_path=dp_rates[i])
                for i in range(num_blocks)
            ])

        # 4. Classification Heads
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(dim, num_classes)

        if use_aux_head:
            self.aux_head = nn.Linear(dim, num_classes)
        else:
            self.aux_head = None

    def forward(self, x: torch.Tensor):
        # Dual-branch extraction & fusion
        f = self.fusion(self.bb(x), self.cnn(x))
        if self.use_peg:
            f = f + self.peg(f)
        x = self.extractor_drop(f)

        # Sequential blocks with residual skip connections
        outs = [x]
        mid_idx = len(self.blocks) // 2
        aux_logits = None

        for i, b in enumerate(self.blocks):
            x = b(x, skip=outs[max(0, i - 1)])
            outs.append(x)
            if self.use_aux_head and i == mid_idx and self.aux_head is not None:
                aux_feat = self.drop(self.norm(x.mean(dim=1)))
                aux_logits = self.aux_head(aux_feat)

        feat = self.drop(self.norm(x.mean(dim=1)))
        logits = self.head(feat)
        return logits, feat, x, aux_logits


def create_model_optimizer(model: nn.Module, lr_backbone: float = 2e-5, lr_head: float = 1e-4, weight_decay: float = 1e-2):
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'bb.backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    return optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone, 'weight_decay': weight_decay},
        {'params': head_params, 'lr': lr_head, 'weight_decay': weight_decay}
    ])


# ==============================================================================
# 5. EVALUATION & TTA
# ==============================================================================

@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, use_tta: bool = True) -> Dict[str, Any]:
    model.eval()

    def scale_crop(x, scale_factor):
        h, w = x.shape[-2:]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        x_scaled = TF.resize(x, [new_h, new_w], antialias=True)
        return TF.center_crop(x_scaled, [h, w])

    tta_tfs = [
        lambda x: x,
        lambda x: scale_crop(x, 1.05),
        lambda x: TF.adjust_contrast(x, 1.1)
    ] if use_tta else [lambda x: x]

    all_preds, all_labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(DEVICE)
        p_sum = None
        for tf in tta_tfs:
            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                logits, _, _, _ = model(tf(imgs))
            p = torch.softmax(logits, dim=1)
            p_sum = p if p_sum is None else p_sum + p

        probs = p_sum / len(tta_tfs)
        preds = probs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(lbls.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    wf1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)
    class_f1s = f1_score(all_labels, all_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0).tolist()

    return {
        'acc': float(acc),
        'macro_f1': float(f1),
        'weighted_f1': float(wf1),
        'kappa': float(kappa),
        'class_f1s': class_f1s,
        'preds': all_preds,
        'labels': all_labels
    }


# ==============================================================================
# 6. DEFINITION OF 20 SYSTEMATIC ABLATION & HP TRIALS (DENSENET-FOCUSED)
# ==============================================================================

TRIALS_CONFIG = [
    # 01. Base Proposed Model (Dual-Chain DenseNet121 + SSSM + GCAF + Distill)
    {
        "id": 1,
        "name": "baseline_dual_densenet",
        "description": "Full proposed architecture (Dual-Chain DenseNet121 + SSSM + GCAF + KD)",
        "backbone_type": "densenet121", "dim": 256, "num_blocks": 4, "d_state": 16,
        "use_sssm": True, "use_gcaf": True, "use_hierarchical_cnn": True,
        "use_peg": True, "use_aux_head": True, "use_distillation": True,
        "use_clahe": True, "use_tta": True, "loss_type": "focal",
        "lr_backbone": 2e-5, "lr_head": 1e-4, "weight_decay": 1e-2, "dropout": 0.3
    },
    # 02. Ablation: Without Knowledge Distillation
    {
        "id": 2,
        "name": "ablation_no_distillation",
        "description": "Student model trained directly without teacher supervision",
        "backbone_type": "densenet121", "dim": 256, "num_blocks": 4, "d_state": 16,
        "use_sssm": True, "use_gcaf": True, "use_hierarchical_cnn": True,
        "use_peg": True, "use_aux_head": True, "use_distillation": False,
        "use_clahe": True, "use_tta": True, "loss_type": "focal",
        "lr_backbone": 2e-5, "lr_head": 1e-4, "weight_decay": 1e-2, "dropout": 0.3
    },
    # 03. Ablation: Without SSSM (Standard Transformer Multi-Head Self-Attention)
    {
        "id": 3,
        "name": "ablation_no_sssm_mha",
        "description": "SSSM replaced with standard Transformer MHA attention blocks",
        "backbone_type": "densenet121", "dim": 256, "num_blocks": 4, "d_state": 16,
        "use_sssm": False, "use_gcaf": True, "use_hierarchical_cnn": True,
        "use_peg": True, "use_aux_head": True, "use_distillation": True,
        "use_clahe": True, "use_tta": True, "loss_type": "focal",
        "lr_backbone": 2e-5, "lr_head": 1e-4, "weight_decay": 1e-2, "dropout": 0.3
    },
    # 04. Ablation: Without Gated Cross-Attention Fusion (Simple Linear Concat)
    {
        "id": 4,
        "name": "ablation_no_gcaf",
        "description": "Gated cross-attention replaced with simple linear concatenation",
        "backbone_type": "densenet121", "dim": 256, "num_blocks": 4, "d_state": 16,
        "use_sssm": True, "use_gcaf": False, "use_hierarchical_cnn": True,
        "use_peg": True, "use_aux_head": True, "use_distillation": True,
        "use_clahe": True, "use_tta": True, "loss_type": "focal",
        "lr_backbone": 2e-5, "lr_head": 1e-4, "weight_decay": 1e-2, "dropout": 0.3
    },
    # 05. Ablation: Without Hierarchical Conv / SPPF (Single-scale Conv branch)
    {
        "id": 5,
        "name": "ablation_no_hierarchical_cnn",
        "description": "Hierarchical multi-scale CNN replaced with single Conv2d branch",
        "backbone_type": "densenet121", "dim": 256, "num_blocks": 4, "d_state": 16,
        "use_sssm": True, "use_gcaf": True, "use_hierarchical_cnn": False,
        "use_peg": True, "use_aux_head": True, "use_distillation": True,
        "use_clahe": True, "use_tta": True, "loss_type": "focal",
        "lr_backbone": 2e-5, "lr_head": 1e-4, "weight_decay": 1e-2, "dropout": 0.3
    },
    # 06. Ablation: Without Positional Encoding Generator (PEG)
    {
        "id": 6,
        "name": "ablation_no_peg",
        "description": "Positional Encoding Generator (PEG) depthwise conv removed",
        "backbone_type": "densenet121", "dim": 256, "num_blocks": 4, "d_state": 16,
        "use_sssm": True, "use_gcaf": True, "use_hierarchical_cnn": True,
        "use_peg": False, "use_aux_head": True, "use_distillation": True,
        "use_clahe": True, "use_tta": True, "loss_type": "focal",
        "lr_backbone": 2e-5, "lr_head": 1e-4, "weight_decay": 1e-2, "dropout": 0.3
    },
    # 07. Ablation: Without Auxiliary Deep Supervision Head
    {
        "id": 7,
        "name": "ablation_no_aux_head",
        "description": "Auxiliary mid-layer loss supervision head disabled",
        "backbone_type": "densenet121", "dim": 256, "num_blocks": 4, "d_state": 16,
        "use_sssm": True, "use_gcaf": True, "use_hierarchical_cnn": True,
        "use_peg": True, "use_aux_head": False, "use_distillation": True,
        "use_clahe": True, "use_tta": True, "loss_type": "focal",
        "lr_backbone": 2e-5, "lr_head": 1e-4, "weight_decay": 1e-2, "dropout": 0.3
    },
    # 08. Ablation: Without CLAHE Histopathology Enhancement
    {
        "id": 8,
        "name": "ablation_no_clahe",
        "description": "CLAHE contrast enhancement disabled in input pipeline",
        "backbone_type": "densenet121", "dim": 256, "num_blocks": 4, "d_state": 16,
        "use_sssm": True, "use_gcaf": True, "use_hierarchical_cnn": True,
        "use_peg": True, "use_aux_head": True, "use_distillation": True,
        "use_clahe": False, "use_tta": True, "loss_type": "focal",
        "lr_backbone": 2e-5, "lr_head": 1e-4, "weight_decay": 1e-2, "dropout": 0.3
    },
    # 09. Backbone Exploration: DenseNet-169
    {
        "id": 9,
        "name": "backbone_densenet169",
        "description": "DenseNet-169 backbone (1664-dim feature extraction)",
        "backbone_type": "densenet169", "dim": 256, "num_blocks": 4, "d_state": 16,
        "use_sssm": True, "use_gcaf": True, "use_hierarchical_cnn": True,
        "use_peg": True, "use_aux_head": True, "use_distillation": True,
        "use_clahe": True, "use_tta": True, "loss_type": "focal",
        "lr_backbone": 2e-5, "lr_head": 1e-4, "weight_decay": 1e-2, "dropout": 0.3
    },
    # 10. Backbone Exploration: DenseNet-201
    {
        "id": 10,
        "name": "backbone_densenet201",
        "description": "DenseNet-201 backbone (1920-dim feature extraction)",
        "backbone_type": "densenet201", "dim": 256, "num_blocks": 4, "d_state": 16,
        "use_sssm": True, "use_gcaf": True, "use_hierarchical_cnn": True,
        "use_peg": True, "use_aux_head": True, "use_distillation": True,
        "use_clahe": True, "use_tta": True, "loss_type": "focal",
        "lr_backbone": 2e-5, "lr_head": 1e-4, "weight_decay": 1e-2, "dropout": 0.3
    },
    # 11. Hyperparameter: Compact Latent Dimension (dim=128)
    {
        "id": 11,
        "name": "hp_dim_128",
        "description": "Compact latent dimension dim=128",
        "backbone_type": "densenet121", "dim": 128, "num_blocks": 4, "d_state": 16,
        "use_sssm": True, "use_gcaf": True, "use_hierarchical_cnn": True,
        "use_peg": True, "use_aux_head": True, "use_distillation": True,
        "use_clahe": True, "use_tta": True, "loss_type": "focal",
        "lr_backbone": 2e-5, "lr_head": 1e-4, "weight_decay": 1e-2, "dropout": 0.3
    },
    # 12. Hyperparameter: High Latent Dimension (dim=384)
    {
        "id": 12,
        "name": "hp_dim_384",
        "description": "High-capacity latent dimension dim=384",
        "backbone_type": "densenet121", "dim": 384, "num_blocks": 4, "d_state": 16,
        "use_sssm": True, "use_gcaf": True, "use_hierarchical_cnn": True,
        "use_peg": True, "use_aux_head": True, "use_distillation": True,
        "use_clahe": True, "use_tta": True, "loss_type": "focal",
        "lr_backbone": 2e-5, "lr_head": 1e-4, "weight_decay": 1e-2, "dropout": 0.3
    },
    # 13. Hyperparameter: SSSM Depth = 2 Blocks
    {
        "id": 13,
        "name": "hp_sssm_blocks_2",
        "description": "Shallow sequence modeling depth (2 SSSM blocks)",
        "backbone_type": "densenet121", "dim": 256, "num_blocks": 2, "d_state": 16,
        "use_sssm": True, "use_gcaf": True, "use_hierarchical_cnn": True,
        "use_peg": True, "use_aux_head": True, "use_distillation": True,
        "use_clahe": True, "use_tta": True, "loss_type": "focal",
        "lr_backbone": 2e-5, "lr_head": 1e-4, "weight_decay": 1e-2, "dropout": 0.3
    },
    # 14. Hyperparameter: SSSM Depth = 6 Blocks
    {
        "id": 14,
        "name": "hp_sssm_blocks_6",
        "description": "Deep sequence modeling depth (6 SSSM blocks)",
        "backbone_type": "densenet121", "dim": 256, "num_blocks": 6, "d_state": 16,
        "use_sssm": True, "use_gcaf": True, "use_hierarchical_cnn": True,
        "use_peg": True, "use_aux_head": True, "use_distillation": True,
        "use_clahe": True, "use_tta": True, "loss_type": "focal",
        "lr_backbone": 2e-5, "lr_head": 1e-4, "weight_decay": 1e-2, "dropout": 0.3
    },
    # 15. Hyperparameter: Higher Learning Rates (Backbone 5e-5, Head 3e-4)
    {
        "id": 15,
        "name": "hp_lr_high",
        "description": "Aggressive learning rate schedule (Backbone 5e-5, Head 3e-4)",
        "backbone_type": "densenet121", "dim": 256, "num_blocks": 4, "d_state": 16,
        "use_sssm": True, "use_gcaf": True, "use_hierarchical_cnn": True,
        "use_peg": True, "use_aux_head": True, "use_distillation": True,
        "use_clahe": True, "use_tta": True, "loss_type": "focal",
        "lr_backbone": 5e-5, "lr_head": 3e-4, "weight_decay": 1e-2, "dropout": 0.3
    },
    # 16. Hyperparameter: Conservative Learning Rates (Backbone 1e-5, Head 5e-5)
    {
        "id": 16,
        "name": "hp_lr_low",
        "description": "Conservative learning rate schedule (Backbone 1e-5, Head 5e-5)",
        "backbone_type": "densenet121", "dim": 256, "num_blocks": 4, "d_state": 16,
        "use_sssm": True, "use_gcaf": True, "use_hierarchical_cnn": True,
        "use_peg": True, "use_aux_head": True, "use_distillation": True,
        "use_clahe": True, "use_tta": True, "loss_type": "focal",
        "lr_backbone": 1e-5, "lr_head": 5e-5, "weight_decay": 1e-2, "dropout": 0.3
    },
    # 17. Hyperparameter: Heavy Weight Decay Regularization (0.05)
    {
        "id": 17,
        "name": "hp_weight_decay_high",
        "description": "High weight decay regularization (wd=0.05)",
        "backbone_type": "densenet121", "dim": 256, "num_blocks": 4, "d_state": 16,
        "use_sssm": True, "use_gcaf": True, "use_hierarchical_cnn": True,
        "use_peg": True, "use_aux_head": True, "use_distillation": True,
        "use_clahe": True, "use_tta": True, "loss_type": "focal",
        "lr_backbone": 2e-5, "lr_head": 1e-4, "weight_decay": 5e-2, "dropout": 0.3
    },
    # 18. Hyperparameter: Loss Function (Label Smoothed Cross Entropy)
    {
        "id": 18,
        "name": "hp_loss_ce_smoothed",
        "description": "Standard label-smoothed cross-entropy (eps=0.1) instead of Focal",
        "backbone_type": "densenet121", "dim": 256, "num_blocks": 4, "d_state": 16,
        "use_sssm": True, "use_gcaf": True, "use_hierarchical_cnn": True,
        "use_peg": True, "use_aux_head": True, "use_distillation": True,
        "use_clahe": True, "use_tta": True, "loss_type": "ce_smoothed",
        "lr_backbone": 2e-5, "lr_head": 1e-4, "weight_decay": 1e-2, "dropout": 0.3
    },
    # 19. Hyperparameter: Heavy Knowledge Distillation Ratio
    {
        "id": 19,
        "name": "hp_heavy_kd_ratio",
        "description": "Distillation-heavy loss: 0.3 Focal + 0.5 KD + 0.2 SP",
        "backbone_type": "densenet121", "dim": 256, "num_blocks": 4, "d_state": 16,
        "use_sssm": True, "use_gcaf": True, "use_hierarchical_cnn": True,
        "use_peg": True, "use_aux_head": True, "use_distillation": True,
        "use_clahe": True, "use_tta": True, "loss_type": "focal",
        "distill_weights": (0.3, 0.5, 0.2),
        "lr_backbone": 2e-5, "lr_head": 1e-4, "weight_decay": 1e-2, "dropout": 0.3
    },
    # 20. Ablation: Without Test-Time Augmentation (TTA Disabled)
    {
        "id": 20,
        "name": "ablation_no_tta",
        "description": "Direct single-pass evaluation (TTA disabled during validation/test)",
        "backbone_type": "densenet121", "dim": 256, "num_blocks": 4, "d_state": 16,
        "use_sssm": True, "use_gcaf": True, "use_hierarchical_cnn": True,
        "use_peg": True, "use_aux_head": True, "use_distillation": True,
        "use_clahe": True, "use_tta": False, "loss_type": "focal",
        "lr_backbone": 2e-5, "lr_head": 1e-4, "weight_decay": 1e-2, "dropout": 0.3
    }
]


# ==============================================================================
# 7. TRAINING & FAULT-TOLERANT RESUME ENGINE
# ==============================================================================

def train_ablation_fold(
    trial_cfg: Dict[str, Any],
    fold: int,
    train_samples: List[Tuple[str, int]],
    val_samples: List[Tuple[str, int]],
    num_epochs: int = 30,
    batch_size: int = 16,
    num_workers: int = 4
) -> Dict[str, Any]:
    trial_id = trial_cfg["id"]
    trial_name = trial_cfg["name"]

    print(f"\n{'='*70}")
    print(f"  [TRIAL {trial_id:02d} / 20] {trial_name.upper()} | FOLD {fold+1}/5")
    print(f"  Description: {trial_cfg['description']}")
    print(f"{'='*70}")

    # Build DataLoaders
    train_tf = get_transforms(224, mode='train', use_clahe=trial_cfg.get('use_clahe', True))
    val_tf = get_transforms(224, mode='val', use_clahe=trial_cfg.get('use_clahe', True))

    train_ds = SubsetWithTransform(train_samples, train_tf)
    val_ds = SubsetWithTransform(val_samples, val_tf)

    # Class-weighted balanced sampler
    train_labels = [s[1] for s in train_samples]
    counts = Counter(train_labels)
    total_samples = len(train_labels)
    num_classes = len(counts)
    class_weights = {l: total_samples / (num_classes * (counts[l] + 1e-6)) for l in range(num_classes)}
    sample_weights = [class_weights.get(l, 1.0) for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Checkpoint paths for power-cut safety
    latest_ckpt = os.path.join(CHECKPOINT_DIR, f"trial_{trial_id:02d}_fold{fold+1}_latest.pth")
    best_ckpt = os.path.join(CHECKPOINT_DIR, f"trial_{trial_id:02d}_fold{fold+1}_best.pth")
    teacher_ckpt = os.path.join(CHECKPOINT_DIR, f"trial_{trial_id:02d}_teacher_fold{fold+1}.pth")

    # Build Model
    use_distill = trial_cfg.get('use_distillation', True)
    model = ModularBioModel(
        num_classes=NUM_CLASSES,
        dim=trial_cfg.get('dim', 256),
        b_type=trial_cfg.get('backbone_type', 'densenet121'),
        num_blocks=trial_cfg.get('num_blocks', 4),
        d_state=trial_cfg.get('d_state', 16),
        dropout=trial_cfg.get('dropout', 0.3),
        use_sssm=trial_cfg.get('use_sssm', True),
        use_gcaf=trial_cfg.get('use_gcaf', True),
        use_hierarchical_cnn=trial_cfg.get('use_hierarchical_cnn', True),
        use_peg=trial_cfg.get('use_peg', True),
        use_aux_head=trial_cfg.get('use_aux_head', True)
    ).to(DEVICE)

    # Build Loss & Optimizer
    loss_type = trial_cfg.get('loss_type', 'focal')
    if loss_type == 'ce_smoothed':
        criterion = LabelSmoothedCrossEntropy(smoothing=0.1)
    else:
        criterion = FocalLoss(gamma=2.0, num_classes=NUM_CLASSES).to(DEVICE)

    optimizer = create_model_optimizer(
        model,
        lr_backbone=trial_cfg.get('lr_backbone', 2e-5),
        lr_head=trial_cfg.get('lr_head', 1e-4),
        weight_decay=trial_cfg.get('weight_decay', 1e-2)
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(DEVICE.type == "cuda"))

    # Teacher setup if knowledge distillation enabled
    teacher = None
    if use_distill:
        teacher = ModularBioModel(
            num_classes=NUM_CLASSES,
            dim=trial_cfg.get('dim', 256),
            b_type=trial_cfg.get('backbone_type', 'densenet121'),
            num_blocks=trial_cfg.get('num_blocks', 4) + 1,  # Teacher has 1 extra block
            d_state=trial_cfg.get('d_state', 16),
            use_sssm=trial_cfg.get('use_sssm', True),
            use_gcaf=trial_cfg.get('use_gcaf', True),
            use_hierarchical_cnn=trial_cfg.get('use_hierarchical_cnn', True),
            use_peg=trial_cfg.get('use_peg', True),
            use_aux_head=trial_cfg.get('use_aux_head', True)
        ).to(DEVICE)

        if os.path.exists(teacher_ckpt):
            teacher.load_state_dict(torch.load(teacher_ckpt, map_location=DEVICE, weights_only=True))
            teacher.eval()
        else:
            # Quick Teacher Pre-training (5 epochs)
            print(f"  [Teacher] Initializing and training 5-epoch teacher model...")
            t_opt = create_model_optimizer(teacher, lr_backbone=2e-5, lr_head=1e-4)
            for t_ep in range(1, 6):
                teacher.train()
                for imgs, lbls in train_loader:
                    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                    t_opt.zero_grad()
                    with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                        t_logits, _, _, t_aux = teacher(imgs)
                        t_loss = criterion(t_logits, lbls)
                        if t_aux is not None:
                            t_loss = t_loss + 0.4 * criterion(t_aux, lbls)
                    scaler.scale(t_loss).backward()
                    scaler.step(t_opt)
                    scaler.update()
            atomic_torch_save(teacher.state_dict(), teacher_ckpt)
            teacher.eval()

    start_epoch = 1
    best_score = -1.0
    best_metrics = {}

    # --------------------------------------------------------------------------
    # RESUME CHECKPOINT DETECTION (Crash / Power-Cut Safety)
    # --------------------------------------------------------------------------
    if os.path.exists(latest_ckpt):
        try:
            ckpt = torch.load(latest_ckpt, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['opt_state'])
            if 'sched_state' in ckpt:
                scheduler.load_state_dict(ckpt['sched_state'])
            if 'scaler_state' in ckpt:
                scaler.load_state_dict(ckpt['scaler_state'])
            start_epoch = ckpt['epoch'] + 1
            best_score = ckpt.get('best_score', -1.0)
            best_metrics = ckpt.get('best_metrics', {})
            print(f"  [RESUME] Found saved checkpoint! Resuming Trial {trial_id} Fold {fold+1} at Epoch {start_epoch}/{num_epochs}")
        except Exception as e:
            print(f"  [Checkpoint Warning] Error loading {latest_ckpt}: {e}. Starting fold from epoch 1.")

    # --------------------------------------------------------------------------
    # EPOCH TRAINING LOOP
    # --------------------------------------------------------------------------
    w_hard, w_kd, w_sp = trial_cfg.get('distill_weights', (0.5, 0.3, 0.2))

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        running_loss, total_correct, total_count = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"T{trial_id:02d}-F{fold+1} E{epoch:02d}/{num_epochs}", leave=False)

        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", enabled=(DEVICE.type == "cuda")):
                logits, _, s_feat, aux_logits = model(imgs)

                if use_distill and teacher is not None:
                    with torch.no_grad():
                        t_logits, _, t_feat, _ = teacher(imgs)
                    loss_hard = criterion(logits, lbls)
                    loss_kd = F.kl_div(
                        F.log_softmax(logits / 4.0, dim=1),
                        F.softmax(t_logits / 4.0, dim=1),
                        reduction='batchmean'
                    ) * 16.0
                    loss_sp = F.mse_loss(
                        F.normalize(s_feat.mean(dim=-1), dim=-1),
                        F.normalize(t_feat.detach().mean(dim=-1), dim=-1)
                    )
                    loss = w_hard * loss_hard + w_kd * loss_kd + w_sp * loss_sp
                else:
                    loss = criterion(logits, lbls)

                if aux_logits is not None:
                    loss = loss + 0.2 * criterion(aux_logits, lbls)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == lbls).sum().item()
            total_count += lbls.size(0)

            pbar.set_postfix({"Loss": f"{loss.item():.3f}", "Acc": f"{100*total_correct/total_count:.1f}%"})

        scheduler.step()

        # Validation at end of epoch
        eval_metrics = evaluate_model(model, val_loader, use_tta=trial_cfg.get('use_tta', True))
        if hasattr(criterion, 'update_alpha'):
            criterion.update_alpha(eval_metrics.get('class_f1s', []))

        current_score = (eval_metrics['acc'] + eval_metrics['macro_f1']) / 2.0
        is_best = current_score > best_score
        if is_best:
            best_score = current_score
            best_metrics = eval_metrics
            # Save Best Model Weights
            atomic_torch_save({
                'trial_id': trial_id,
                'fold': fold,
                'epoch': epoch,
                'model_state': model.state_dict(),
                'metrics': best_metrics,
                'config': trial_cfg
            }, best_ckpt)

        # Save Latest Checkpoint for crash recovery
        atomic_torch_save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'opt_state': optimizer.state_dict(),
            'sched_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'best_score': best_score,
            'best_metrics': best_metrics
        }, latest_ckpt)

        print(f"  E{epoch:02d}/{num_epochs:02d} | TrainLoss: {running_loss/len(train_loader):.4f} | ValAcc: {eval_metrics['acc']*100:.2f}% | MacroF1: {eval_metrics['macro_f1']:.4f} {'[BEST]' if is_best else ''}")

    # Load and return the best checkpoint evaluated metrics
    if os.path.exists(best_ckpt):
        best_data = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
        best_metrics = best_data['metrics']

    print(f"  >>> FOLD {fold+1} COMPLETED | Best Acc: {best_metrics.get('acc', 0.0)*100:.2f}% | Best Macro F1: {best_metrics.get('macro_f1', 0.0):.4f}")
    return best_metrics


# ==============================================================================
# 8. MASTER CROSS-VALIDATION ABLATION RUNNER
# ==============================================================================

def run_ablation_study(
    trial_ids: Optional[List[int]] = None,
    num_epochs: int = 30,
    batch_size: int = 16,
    num_folds: int = 5,
    data_dir: str = DEFAULT_DATA_DIR,
    num_workers: int = 4
):
    seed_everything(42)
    tracker = ExperimentTracker()

    print("\n" + "="*70)
    print("  5-FOLD CROSS-VALIDATION ABLATION & HYPERPARAMETER OPTIMIZATION")
    print(f"  Device:         {DEVICE}")
    print(f"  Total Trials:   {len(TRIALS_CONFIG)}")
    print(f"  Target Folds:   {num_folds}")
    print(f"  Epochs / Fold:  {num_epochs}")
    print(f"  Summary File:   {MASTER_SUMMARY_CSV}")
    print("="*70 + "\n")

    # Load dataset
    resolved_dir = find_dataset_dir(data_dir)
    print(f"Loading histopathology dataset from: {resolved_dir}")
    dataset = OvarianDataset(resolved_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No valid images found in {resolved_dir} under classes {CLASS_NAMES}")

    samples = dataset.samples
    labels = [s[1] for s in samples]
    print(f"Total Dataset Size: {len(samples)} images across {NUM_CLASSES} classes: {CLASS_NAMES}\n")

    # Generate deterministic 5-fold splits
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_splits = list(skf.split(samples, labels))

    # Filter trials if specific ones requested
    trials_to_run = TRIALS_CONFIG
    if trial_ids:
        trials_to_run = [t for t in TRIALS_CONFIG if t["id"] in trial_ids]

    for trial in trials_to_run:
        t_id = trial["id"]
        t_name = trial["name"]

        # Level 1 Guard: Check if trial is fully completed across all 5 folds
        if tracker.is_trial_fully_completed(t_id, num_folds=num_folds):
            print(f"\n{'*'*70}")
            print(f"  [CHECKPOINT NOTICE] WOW! Trial {t_id:02d} ({t_name}) has ALREADY been trained across all {num_folds} folds!")
            print(f"  -> Skipping Trial {t_id:02d} automatically.")
            print(f"{'*'*70}\n")
            continue

        fold_results = []
        for fold in range(num_folds):
            # Level 2 Guard: Check if this specific fold is already completed
            completed_info = tracker.is_fold_completed(t_id, fold)
            best_ckpt = os.path.join(CHECKPOINT_DIR, f"trial_{t_id:02d}_fold{fold+1}_best.pth")

            if completed_info and os.path.exists(best_ckpt):
                m = completed_info.get("metrics", {})
                print(f"  [CHECKPOINT NOTICE] Trial {t_id:02d} Fold {fold+1} already completed! (Acc: {m.get('acc',0)*100:.2f}%, F1: {m.get('macro_f1',0):.4f}) -> Skipping.")
                fold_results.append(m)
                continue

            # Execute Fold Training
            train_idx, val_idx = fold_splits[fold]
            train_samples_fold = [samples[i] for i in train_idx]
            val_samples_fold = [samples[i] for i in val_idx]

            fold_metric = train_ablation_fold(
                trial_cfg=trial,
                fold=fold,
                train_samples=train_samples_fold,
                val_samples=val_samples_fold,
                num_epochs=num_epochs,
                batch_size=batch_size,
                num_workers=num_workers
            )

            tracker.record_fold_completion(t_id, t_name, fold, fold_metric)
            fold_results.append(fold_metric)

            # Cleanup memory between folds
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Compute 5-Fold Aggregated Metrics
        accs = [m.get('acc', 0.0) for m in fold_results]
        f1s = [m.get('macro_f1', 0.0) for m in fold_results]
        wf1s = [m.get('weighted_f1', 0.0) for m in fold_results]
        kappas = [m.get('kappa', 0.0) for m in fold_results]

        summary = {
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "wf1_mean": float(np.mean(wf1s)),
            "kappa_mean": float(np.mean(kappas))
        }

        tracker.record_trial_summary(t_id, t_name, summary, trial)

        print("\n" + "#"*70)
        print(f"  TRIAL {t_id:02d} ({t_name}) 5-FOLD AGGREGATE RESULTS:")
        print(f"    Mean Accuracy:    {summary['acc_mean']*100:.2f}% ± {summary['acc_std']*100:.2f}%")
        print(f"    Mean Macro F1:    {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
        print(f"    Mean Weighted F1: {summary['wf1_mean']:.4f}")
        print(f"    Mean Cohen Kappa: {summary['kappa_mean']:.4f}")
        print("#"*70 + "\n")

    print_master_summary()


def print_trials_list():
    """Prints list of all 20 configured trials with descriptions."""
    print("\n" + "="*85)
    print("                      CONFIGURED ABLATION & HP OPTIMIZATION TRIALS (20)")
    print("="*85)
    print(f"{'ID':<4} | {'Trial Name':<28} | {'Description'}")
    print("-" * 85)
    for t in TRIALS_CONFIG:
        print(f"{t['id']:<4} | {t['name']:<28} | {t['description']}")
    print("="*85 + "\n")


def print_master_summary():
    """Prints a clean, formatted table of all ablation experiments recorded in CSV."""
    if not os.path.exists(MASTER_SUMMARY_CSV):
        print("No master summary CSV found yet.")
        return

    print("\n" + "="*95)
    print("                      MASTER ABLATION & HYPERPARAMETER OPTIMIZATION SUMMARY")
    print("="*95)
    print(f"{'ID':<4} | {'Trial Name':<28} | {'Mean Acc ± Std':<18} | {'Macro F1 ± Std':<18} | {'Weighted F1':<12}")
    print("-" * 95)

    with open(MASTER_SUMMARY_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_id = row.get('trial_id', '')
            name = row.get('trial_name', '')[:28]
            acc = f"{row.get('mean_acc', '')} ± {row.get('std_acc', '')}"
            f1 = f"{row.get('mean_macro_f1', '')} ± {row.get('std_macro_f1', '')}"
            wf1 = row.get('mean_weighted_f1', '')
            print(f"{t_id:<4} | {name:<28} | {acc:<18} | {f1:<18} | {wf1:<12}")

    print("="*95 + "\n")


# ==============================================================================
# 9. CLI INTERFACE
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Ovarian Histopathology 5-Fold Cross-Validation Ablation Study")
    parser.add_argument('--trials', type=str, default='all',
                        help="Comma-separated trial IDs to run (e.g. '1,2,5' or '1-5' or 'all'). Default: 'all'")
    parser.add_argument('--epochs', type=int, default=30,
                        help="Number of epochs per fold (default: 30)")
    parser.add_argument('--batch-size', type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument('--folds', type=int, default=5,
                        help="Number of cross-validation folds (default: 5)")
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR,
                        help="Path to ovarian dataset directory")
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1),
                        help="DataLoader worker count")
    parser.add_argument('--list-trials', action='store_true',
                        help="List all 20 configured ablation trials and exit")
    parser.add_argument('--summary-only', action='store_true',
                        help="Display the master summary table of completed trials and exit")
    return parser.parse_args()


def parse_trial_ids(trials_str: str) -> Optional[List[int]]:
    if trials_str.lower() == 'all':
        return None
    ids = []
    for part in trials_str.split(','):
        part = part.strip()
        if '-' in part:
            s, e = part.split('-')
            ids.extend(list(range(int(s), int(e) + 1)))
        elif part.isdigit():
            ids.append(int(part))
    return sorted(list(set(ids)))


if __name__ == "__main__":
    args = parse_args()
    if args.list_trials:
        print_trials_list()
    elif args.summary_only:
        print_master_summary()
    else:
        selected_ids = parse_trial_ids(args.trials)
        run_ablation_study(
            trial_ids=selected_ids,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            num_folds=args.folds,
            data_dir=args.data_dir,
            num_workers=args.workers
        )
