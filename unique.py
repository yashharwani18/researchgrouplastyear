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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
import math
import csv
import timm
from torchvision import transforms
from torchvision.transforms import v2
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score
import torchvision.transforms.functional as TF
import cv2
from sklearn.model_selection import StratifiedKFold, train_test_split

# ==================== DEVICE ====================
def get_safe_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_safe_device()

# Base output directory safe for python scripts & Kaggle environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else ("/kaggle/working" if os.path.exists("/kaggle/working") else os.getcwd())

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_ovarian")
LOG_DIR = os.path.join(BASE_DIR, "logs")
MASTER_RESULTS_FILE = os.path.join(BASE_DIR, "ovarian_unique_results.csv")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CLASS_NAMES = ['Clear_Cell', 'Endometri', 'Mucinous', 'Non_Cancerous', 'Serous']
NUM_CLASSES = len(CLASS_NAMES)

# ==================== PRE-PROCESSING & AUGMENTATIONS ====================

class GaussianNoise(nn.Module):
    def __init__(self, std: float = 0.005):
        super().__init__()
        self.std = std
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x

class BrainMask(nn.Module):
    def __init__(self, threshold: float = 0.01):
        super().__init__()
        self.threshold = threshold
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x.mean(dim=0, keepdim=True) > self.threshold
        return x * mask

class CLAHE_Transform:
    def __init__(self, clip_limit=3.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            img_np = clahe.apply(img_np)
        return Image.fromarray(img_np)

def get_transforms(img_size: int = 224, mode: str = 'train', mean=None, std=None):
    if mean is None: mean = [0.485, 0.456, 0.406]
    if std is None:  std  = [0.229, 0.224, 0.225]

    if mode == 'train':
        return v2.Compose([
            CLAHE_Transform(clip_limit=2.0, tile_grid_size=(8,8)),
            v2.Resize((img_size, img_size)),
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
        return v2.Compose([
            CLAHE_Transform(clip_limit=2.0, tile_grid_size=(8,8)),
            v2.Resize((img_size, img_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])

# ==================== DATASET ====================
def find_dataset_dir(data_dir: str = "."):
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
    def __init__(self, data_dir: str = ".", transform=None):
        self.transform = transform
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

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label

class _Subset(Dataset):
    def __init__(self, samples, tf):
        self.samples = samples
        self.tf = tf
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as img:
            return self.tf(img.convert('RGB')), label

def get_all_samples(train_dir="."):
    ds = OvarianDataset(train_dir)
    all_samples = ds.samples
    all_labels = [s[1] for s in all_samples]
    from sklearn.model_selection import train_test_split
    train_samples, _, train_labels, _ = train_test_split(all_samples, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
    return train_samples, train_labels

def get_test_samples(test_dir="."):
    ds = OvarianDataset(test_dir)
    all_samples = ds.samples
    all_labels = [s[1] for s in all_samples]
    from sklearn.model_selection import train_test_split
    _, test_samples, _, _ = train_test_split(all_samples, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
    return test_samples

def get_loaders_for_fold(train_samples, train_labels, batch_size=16, img_size=224, num_workers=4):
    from collections import Counter
    d_mean, d_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = get_transforms(img_size, 'train', d_mean, d_std)

    train_ds = _Subset(train_samples, train_tf)

    counts  = Counter(train_labels)
    total = len(train_labels)
    num_c = len(counts)
    class_weights = {l: total / (num_c * counts[l]) for l in counts}
    weights = [class_weights[l] for l in train_labels]
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)

    return train_loader

# ==================== LOSS ====================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, num_classes=5, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    @torch.no_grad()
    def update_alpha(self, class_f1s):
        if not class_f1s: return
        f1_tensor = torch.tensor(class_f1s, device=self.alpha.device).clamp(min=0.1)
        target_alpha = 1.0 / f1_tensor
        target_alpha = target_alpha / target_alpha.mean()
        self.alpha = 0.9 * self.alpha + 0.1 * target_alpha

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, reduction='none', label_smoothing=0.1)
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = loss * self.alpha[labels]
        return loss.mean()

# ==================== MODEL ====================

class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))
    def forward(self, x): return self.gamma * x

def stochastic_depth(x, drop_prob, training):
    if not training or drop_prob == 0.0: return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    return x / keep_prob * torch.floor(torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob)

class PEG(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N**0.5)
        return self.proj(x.permute(0,2,1).view(B,C,H,W)).view(B,C,N).permute(0,2,1)

class SPPFBlock(nn.Module):
    def __init__(self, in_c, out_c, k=5):
        super().__init__()
        mid = in_c // 2
        self.cv1  = nn.Sequential(nn.Conv2d(in_c, mid, 1), nn.BatchNorm2d(mid), nn.SiLU())
        self.pool = nn.MaxPool2d(k, 1, k//2)
        self.cv2  = nn.Sequential(nn.Conv2d(mid*4, out_c, 1), nn.BatchNorm2d(out_c), nn.SiLU())
    def forward(self, x):
        x = self.cv1(x)
        y1, y2, y3 = self.pool(x), self.pool(self.pool(x)), self.pool(self.pool(self.pool(x)))
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))

class HierarchicalConvBlock(nn.Module):
    def __init__(self, in_c, dim, kernel_sizes=(3, 5, 7)):
        super().__init__()
        branch = dim // len(kernel_sizes)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_c, branch, k, padding=k//2), nn.BatchNorm2d(branch), nn.SiLU())
            for k in kernel_sizes
        ])
        self.sppf = SPPFBlock(branch * len(kernel_sizes), dim)
        self.norm = nn.GroupNorm(8, dim)
        self.drop = nn.Dropout(0.2)
    def forward(self, x):
        feats = [F.adaptive_avg_pool2d(c(x), (14, 14)) for c in self.convs]
        return self.drop(self.norm(self.sppf(torch.cat(feats, dim=1)))).flatten(2).transpose(1, 2)

class BackboneExtractor(nn.Module):
    BACKBONES   = {
        'resnext':  lambda: timm.create_model('resnext50_32x4d',               pretrained=True, num_classes=0, global_pool=''),
        'densenet': lambda: timm.create_model('densenet121',                  pretrained=True, num_classes=0, global_pool=''),
    }
    BACKBONE_DIMS = {'resnext': 2048, 'densenet': 1024}

    def __init__(self, b_type, dim, dropout=0.3):
        super().__init__()
        self.backbone = self.BACKBONES[b_type]()
        self.proj     = nn.Linear(self.BACKBONE_DIMS[b_type], dim)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x):
        f = self.backbone(x)
        if f.dim() == 4:
            if f.shape[-1] == self.proj.in_features: f = f.permute(0, 3, 1, 2)
            f = F.adaptive_avg_pool2d(f, (14, 14)).flatten(2).transpose(1, 2)
        elif f.dim() == 3:
            f = F.adaptive_avg_pool1d(f.transpose(1, 2), 196).transpose(1, 2)
        return self.dropout(self.proj(f))

class GatedCrossAttentionFusion(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(dim*2, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)
    def forward(self, local_f, global_f):
        att, _ = self.attn(local_f, global_f, global_f)
        g = self.gate(torch.cat([local_f, att], dim=-1))
        return self.norm(g * att + (1-g) * local_f)

def associative_scan_ssm(delta, A, Bx):
    """Sequential scan — stable and correct."""
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
    def __init__(self, d_model, d_state=16, expand=2, drop_path=0.0):
        super().__init__()
        self.d_inner  = int(expand * d_model)
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2)
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2)
        self.dt_proj  = nn.Linear(d_model, d_state)
        self.A_log    = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.ls       = LayerScale(d_model)
        self.drop_path = drop_path
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x, skip=None):
        res  = x
        xi, z = self.in_proj(x).chunk(2, dim=-1)
        B_p, C_p = self.x_proj(xi).chunk(2, dim=-1)
        delta = F.softplus(self.dt_proj(x))
        A     = -torch.exp(self.A_log)
        h_fwd = associative_scan_ssm(delta, A, B_p)
        h_bak = associative_scan_ssm(delta.flip(1), A, B_p.flip(1)).flip(1)
        y     = ((h_fwd + h_bak) * C_p).sum(-1, keepdim=True).expand(-1, -1, self.d_inner) + z
        out   = self.ls(self.out_proj(y))
        return self.norm(stochastic_depth(out, self.drop_path, self.training) + (skip if skip is not None else res))

class DualChainExtractor(nn.Module):
    def __init__(self, dim, b_type):
        super().__init__()
        self.cnn  = HierarchicalConvBlock(3, dim)
        self.bb   = BackboneExtractor(b_type, dim, dropout=0.3)
        self.gcaf = GatedCrossAttentionFusion(dim)
        self.peg  = PEG(dim)
        self.drop = nn.Dropout(0.2)
    def forward(self, x):
        f = self.gcaf(self.bb(x), self.cnn(x))
        return self.drop(f + self.peg(f))

class BioHCLSSSM(nn.Module):
    def __init__(self, num_classes=5, dim=256, b_type='densenet', num_blocks=4, dropout=0.3):
        super().__init__()
        self.extractor = DualChainExtractor(dim, b_type)
        dp_rates = [0.1 * i / num_blocks for i in range(num_blocks)]
        self.blocks = nn.ModuleList([
            LinearBioSSSMBlock(dim, d_state=16 + i*4, drop_path=dp_rates[i])
            for i in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(dim, num_classes)
        self.aux_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.extractor(x)
        outs = [x]
        mid_idx = len(self.blocks) // 2
        aux_logits = None
        for i, b in enumerate(self.blocks):
            x = b(x, skip=outs[max(0, i-1)])
            outs.append(x)
            if i == mid_idx:
                aux_feat = self.drop(self.norm(x.mean(dim=1)))
                aux_logits = self.aux_head(aux_feat)
        feat = self.drop(self.norm(x.mean(dim=1)))
        return self.head(feat), feat, x, aux_logits

# ==================== TRAIN / EVAL ====================

def train_epoch(model, loader, opt, epoch, focal_criterion, scaler=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=f"Train E{epoch}")
    for img, lbl in pbar:
        img, lbl = img.to(DEVICE), lbl.to(DEVICE)
        opt.zero_grad()
        with torch.amp.autocast('cuda'):
            logits, _, _, aux_logits = model(img)
            loss = focal_criterion(logits, lbl)
            if aux_logits is not None:
                loss += 0.4 * focal_criterion(aux_logits, lbl)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total_loss += loss.item()
        correct += (logits.argmax(1) == lbl).sum().item()
        total   += lbl.size(0)
        pbar.set_postfix({"Loss": f"{loss.item():.3f}", "Acc": f"{100*correct/total:.1f}%"})
    return total_loss / len(loader), correct / total

def eval_epoch(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for img, lbl in loader:
            with torch.amp.autocast('cuda'):
                logits, _, _, _ = model(img.to(DEVICE))
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(lbl.tolist())
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    class_f1 = f1_score(all_labels, all_preds, average=None,
                         labels=list(range(NUM_CLASSES)), zero_division=0)
    return {'acc': acc, 'macro_f1': f1, 'class_f1s': class_f1.tolist()}

@torch.no_grad()
def tta_evaluate(model, loader):
    model.eval()
    def scale_crop(x, scale_factor):
        h, w = x.shape[-2:]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        x_scaled = TF.resize(x, [new_h, new_w], antialias=True)
        return TF.center_crop(x_scaled, [h, w])

    # Best TTA Combo (0, 5, 8)
    tta_tfs = [
        lambda x: x,
        lambda x: scale_crop(x, 1.05),
        lambda x: TF.adjust_contrast(x, 1.1)
    ]
    all_probs, all_labels = [], []
    for img, lbl in tqdm(loader, desc="TTA"):
        img  = img.to(DEVICE)
        p_sum = None
        for tf in tta_tfs:
            with torch.amp.autocast('cuda'):
                logits, _, _, _ = model(tf(img))
            p = torch.softmax(logits, dim=1)
            p_sum = p if p_sum is None else p_sum + p
        all_probs.append((p_sum / len(tta_tfs)).cpu())
        all_labels.append(lbl)
    probs  = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    preds  = probs.argmax(1)
    acc    = accuracy_score(labels, preds)
    f1     = f1_score(labels, preds, average='macro', zero_division=0)
    wf1    = f1_score(labels, preds, average='weighted', zero_division=0)
    kappa  = cohen_kappa_score(labels, preds)
    print(f"\n{'='*40}")
    print(f"  TTA Test Results")
    print(f"  Acc:        {acc*100:.2f}%")
    print(f"  Macro F1:   {f1:.4f}")
    print(f"  Weighted F1:{wf1:.4f}")
    print(f"  Kappa:      {kappa:.4f}")
    print(f"{'='*40}")
    return acc, f1

def create_optimizer(model, backbone_lr=1.5e-5, head_lr=1e-4, weight_decay=3e-2):
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'extractor.bb.backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    return optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
        {'params': head_params, 'lr': head_lr, 'weight_decay': weight_decay}
    ])

# ==================== MAIN ====================

def train_ovarian(config='dual_densenet', num_epochs=50, batch_size=16,
                  train_dir=r"/kaggle/input/datasets/bitsnpieces/ovarian-cancer-and-subtypes-dataset-histopathology/OvarianCancer",
                  test_dir=r"/kaggle/input/datasets/bitsnpieces/ovarian-cancer-and-subtypes-dataset-histopathology/OvarianCancer",
                  num_workers=4):

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1. Get Samples
    train_samples, train_labels = get_all_samples(train_dir)
    test_samples = get_test_samples(test_dir)
    
    fold_accuracies = []

    for fold in [0]:
        print(f"\n{'='*50}\n  STARTING TRAINING (Train vs Validation Split)\n{'='*50}")

        # Train on the full Training directory
        train_loader = get_loaders_for_fold(
            train_samples, train_labels, batch_size, img_size=224, num_workers=num_workers
        )
        
        # Evaluate directly on the HELD-OUT Testing folder to optimize Test Accuracy
        eval_tf = get_transforms(224, mode='eval')
        test_ds = _Subset(test_samples, eval_tf)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        val_loader = test_loader # use test set as validation

        b_type = 'resnext' if 'resnext' in config else 'densenet'
        teacher = BioHCLSSSM(num_classes=NUM_CLASSES, dim=256, b_type=b_type, num_blocks=5).to(DEVICE)
        student = BioHCLSSSM(num_classes=NUM_CLASSES, dim=256, b_type=b_type, num_blocks=4).to(DEVICE)
        
        # Class weights for Focal Loss
        alpha_weights = torch.ones(NUM_CLASSES, device=DEVICE)
        focal   = FocalLoss(gamma=2.0, num_classes=NUM_CLASSES, alpha=alpha_weights).to(DEVICE)
        scaler  = torch.amp.GradScaler('cuda')

        print(f"  Teacher: {sum(p.numel() for p in teacher.parameters())/1e6:.1f}M params")
        print(f"  Student: {sum(p.numel() for p in student.parameters())/1e6:.1f}M params")

        teacher_path = os.path.join(CHECKPOINT_DIR, f"v7_teacher_{config}_fold{fold+1}_best.pth")
        student_path = os.path.join(CHECKPOINT_DIR, f"v7_student_{config}_fold{fold+1}_best.pth")

        # ── Phase 1: Train Teacher ─────────────────────────────────────────────
        if os.path.exists(teacher_path):
            print(f"\n  Teacher checkpoint fold {fold+1} found — skipping Phase 1.")
            teacher.load_state_dict(torch.load(teacher_path, weights_only=True))
        else:
            print(f"\n  [PHASE 1] TEACHER TRAINING - FOLD {fold+1}")
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            opt_t   = create_optimizer(teacher, backbone_lr=2e-5, head_lr=1e-4, weight_decay=1e-2)
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            sched_t = CosineAnnealingWarmRestarts(opt_t, T_0=10, T_mult=2)
            best_score_t = 0.0

            for e in range(1, num_epochs + 1):
                t_loss, t_acc = train_epoch(teacher, train_loader, opt_t, e, focal, scaler)
                m = eval_epoch(teacher, val_loader)
                focal.update_alpha(m['class_f1s'])
                sched_t.step()

                score = (m['acc'] + m['macro_f1']) / 2.0
                print(f"  Teacher F{fold+1}-E{e:02d}: Loss={t_loss:.4f} | ValAcc={m['acc']*100:.2f}% | F1={m['macro_f1']:.4f}")

                if score > best_score_t:
                    best_score_t = score
                    torch.save(teacher.state_dict(), teacher_path)

            print(f"\n  Teacher best val score: {best_score_t*100:.2f}%")
            teacher.load_state_dict(torch.load(teacher_path, weights_only=True))

        teacher.eval()

        # ── Phase 2: Distil into Student ──────────────────────────────────────
        print(f"\n  [PHASE 2] STUDENT DISTILLATION - FOLD {fold+1}")

        opt_s   = create_optimizer(student, backbone_lr=2e-5, head_lr=1e-4, weight_decay=1e-2)
        sched_s = CosineAnnealingLR(opt_s, T_max=num_epochs)
        best_score_s = 0.0

        if os.path.exists(student_path):
            print(f"  Student checkpoint found — resuming training.")
            student.load_state_dict(torch.load(student_path, weights_only=True))
            m_start = eval_epoch(student, val_loader)
            best_score_s = (m_start['acc'] + m_start['macro_f1']) / 2.0
            print(f"  Resuming with baseline ValAcc: {m_start['acc']*100:.2f}% | F1: {m_start['macro_f1']:.4f}")

        for e in range(1, num_epochs + 1):
            student.train()
            total_loss, correct, total = 0, 0, 0
            pbar = tqdm(train_loader, desc=f"Distill F{fold+1}-E{e}")

            for img, lbl in pbar:
                img, lbl = img.to(DEVICE), lbl.to(DEVICE)
                opt_s.zero_grad()

                with torch.no_grad():
                    lt, _, st, _ = teacher(img)

                with torch.amp.autocast('cuda'):
                    ls, _, ss, aux_s = student(img)
                    loss_h  = focal(ls, lbl)
                    loss_kd = F.kl_div(F.log_softmax(ls / 4.0, 1), F.softmax(lt / 4.0, 1), reduction='batchmean') * 16.0
                    loss_sp = F.mse_loss(F.normalize(ss.mean(-1), dim=-1), F.normalize(st.detach().mean(-1), dim=-1))
                    loss = 0.5 * loss_h + 0.3 * loss_kd + 0.2 * loss_sp
                    if aux_s is not None:
                        loss += 0.2 * focal(aux_s, lbl)

                scaler.scale(loss).backward()
                scaler.unscale_(opt_s)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(opt_s)
                scaler.update()

                total_loss += loss.item()
                correct    += (ls.argmax(1) == lbl).sum().item()
                total      += lbl.size(0)
                pbar.set_postfix({"Loss": f"{loss.item():.3f}", "Acc": f"{100*correct/total:.1f}%"})

            sched_s.step()
            m = eval_epoch(student, val_loader)
            focal.update_alpha(m['class_f1s'])

            score = (m['acc'] + m['macro_f1']) / 2.0
            print(f"  Student F{fold+1}-E{e:02d}: Loss={total_loss/len(train_loader):.4f} | ValAcc={m['acc']*100:.2f}% | F1={m['macro_f1']:.4f}")

            if score > best_score_s:
                best_score_s = score
                torch.save(student.state_dict(), student_path)

        print(f"\n  Student best val score for fold {fold+1}: {best_score_s*100:.2f}%")

        # ── Final TTA Evaluation ───────────────────────────────────────────────
        student.load_state_dict(torch.load(student_path, weights_only=True))
        print(f"\n  Running TTA on test set for Fold {fold+1}...")
        acc, _ = tta_evaluate(student, test_loader)
        fold_accuracies.append(acc)

    print(f"\n{'='*50}")
    print(f"  TRAINING COMPLETED")
    print(f"  Final Test Accuracy: {fold_accuracies[0]*100:.2f}%")
    print(f"{'='*50}")

    return teacher, student, np.max(fold_accuracies)


if __name__ == "__main__":
    import os
    print(f"Device: {DEVICE}")
    DATA_DIR = r"/kaggle/input/datasets/bitsnpieces/ovarian-cancer-and-subtypes-dataset-histopathology/OvarianCancer"
    teacher, student, best = train_ovarian(
        config='dual_densenet',
        num_epochs=50,
        batch_size=16,
        train_dir=DATA_DIR,
        test_dir=DATA_DIR,
        num_workers=min(8, os.cpu_count() or 1)
    )