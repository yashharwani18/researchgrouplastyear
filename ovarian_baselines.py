import os
import gc
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import timm
from tqdm import tqdm
from PIL import Image
from torchvision import datasets
from torchvision.transforms import v2
import csv
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# ==============================================================================
# 0. GLOBAL CONFIG & REPRODUCIBILITY
# ==============================================================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Base output directory safe for both python scripts and Kaggle/Jupyter notebooks
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else ("/kaggle/working" if os.path.exists("/kaggle/working") else os.getcwd())

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_ovarian")
MASTER_RESULTS_FILE = os.path.join(BASE_DIR, "ovarian_baseline_results.csv")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class EpochLogger:
    """Helper to save per-epoch metrics into a CSV file."""
    def __init__(self, filename):
        self.filename = filename

    def log(self, metrics):
        file_exists = os.path.isfile(self.filename)
        with open(self.filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

# Ovarian Cancer & Subtypes Histopathology Dataset classes
# Classes: Clear_Cell, Endometrioid, Mucinous, Non_Cancerous, Serous
CLASS_NAMES = ['Clear_Cell', 'Endometrioid', 'Mucinous', 'Non_Cancerous', 'Serous']
NUM_CLASSES = len(CLASS_NAMES)

# ------------------------------------------------------------------------------
# Self-contained Histopathology Augmentations & Loss Functions
# ------------------------------------------------------------------------------
class CLAHE_Transform:
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) for histopathology image enhancement."""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        try:
            import cv2
            np_img = np.array(img)
            if len(np_img.shape) == 3 and np_img.shape[2] == 3:
                lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
                cl = clahe.apply(l)
                limg = cv2.merge((cl, a, b))
                final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
                return Image.fromarray(final)
        except Exception:
            pass
        return img


class FocalLoss(nn.Module):
    """Focal Loss with dynamic alpha weighting for class imbalance."""
    def __init__(self, gamma=2.0, num_classes=5, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        if alpha is None:
            self.alpha = torch.ones(num_classes, device=DEVICE)
        else:
            self.alpha = alpha.to(DEVICE)

    def update_alpha(self, class_f1s):
        if class_f1s and len(class_f1s) == self.num_classes:
            f1s = torch.tensor(class_f1s, device=DEVICE)
            weights = 1.0 - f1s
            weights = torch.clamp(weights, min=0.1, max=2.0)
            self.alpha = weights / (weights.sum() + 1e-6) * self.num_classes

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()

# ------------------------------------------------------------------------------
# 1. PER-MODEL RESOLUTION LOOKUP
# ------------------------------------------------------------------------------
# Native input resolutions per architecture.
# Priority: official paper/training resolution -> timm default.
MODEL_IMG_SIZES = {
    # Inception family
    "inception_v3":              299,
    "inception_resnet_v2":       299,
    "xception":                  299,
    # EfficientNet family
    "efficientnet_b0":           224,
    "tf_efficientnet_b3":        300,
    "tf_efficientnet_b7":        600,
    # Swin family
    "swin_tiny_patch4_window7_224":        224,
    "swin_base_patch4_window7_224":        224,
    "swinv2_base_window12to16_192to256":   256,
    # ViT family
    "vit_base_patch16_224":      224,
    "vit_large_patch16_224":     224,
    "vit_base_patch16_224_miil": 224,
    # DeiT family
    "deit_small_distilled_patch16_224": 224,
    "deit_base_distilled_patch16_224":  224,
    # BEiT / T2T
    "beit_base_patch16_224":     224,
    "t2t_vit_14":                224,
    # ConvNeXt / ConvMixer
    "convnext_base":             224,
    "convmixer_768_32":          224,
    # Classic CNN
    "resnet18":                  224,
    "resnet50":                  224,
    "resnet101":                 224,
    "vgg16":                     224,
    "vgg19":                     224,
    "seresnet50":                224,
    "resnext101_32x8d":          224,
    "densenet121":               224,
    "densenet201":               224,
    "mobilenetv2_100":           224,
    "nasnet_mobile":             224,
    "squeezenet1_0":             224,
    "mobilevit_s":               256,
    # Modern / experimental
    "regnety_008":               224,
    "resnetv2_50x1_bit":         224,
    "hrnet_w32":                 224,
    "pvt_v2_b2":                 224,
    "cvt_13":                    224,
    "coat_lite_small":           224,
    "twins_svt_small":           224,
    "poolformer_s12":            224,
    "efficientformer_l1":        224,
}


def get_img_size(model_name: str) -> int:
    """
    Returns the correct input resolution for a given timm model name.
    1. Checks the explicit MODEL_IMG_SIZES table (fastest, no model load).
    2. Falls back to timm's pretrained_cfg (loads config only, no weights).
    """
    if model_name in MODEL_IMG_SIZES:
        return MODEL_IMG_SIZES[model_name]

    print(f"  [get_img_size] '{model_name}' not in table - querying timm config...")
    try:
        cfg = timm.get_pretrained_cfg(model_name)
        size = cfg.input_size[1]
        print(f"  [get_img_size] timm reports {size}px for '{model_name}'")
        return size
    except Exception:
        print(f"  [get_img_size] Could not resolve size for '{model_name}', defaulting to 224")
        return 224


# ------------------------------------------------------------------------------
# 2. DATA UTILS
# ------------------------------------------------------------------------------
def load_flat_dataset(data_dir: str):
    """
    Loads dataset from directory structure.
    Checks candidate Kaggle input paths if data_dir is default or not found.
    Subdirectories are treated as target class names.
    """
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    
    candidate_dirs = [
        data_dir,
        "/kaggle/input/datasets/bitsnpieces/ovarian-cancer-and-subtypes-dataset-histopathology/OvarianCancer",
        "/kaggle/input/ovarian-cancer-and-subtypes-dataset-histopathology/OvarianCancer",
        "/kaggle/input/datasets/bitsnpieces/ovarian-cancer-and-subtypes-dataset-histopathology",
        "/kaggle/input/ovarian-cancer-and-subtypes-dataset-histopathology",
        "./OvarianCancer",
        "."
    ]
    
    target_dir = None
    for c_dir in candidate_dirs:
        if c_dir and os.path.exists(c_dir) and os.path.isdir(c_dir):
            subdirs = [d for d in os.listdir(c_dir) if os.path.isdir(os.path.join(c_dir, d)) and not d.startswith('.')]
            if subdirs:
                target_dir = c_dir
                break

    if target_dir is None:
        target_dir = data_dir

    print(f"Loading dataset from: {target_dir}")
    
    detected_classes = sorted([
        d for d in os.listdir(target_dir)
        if os.path.isdir(os.path.join(target_dir, d)) and not d.startswith('.')
    ])
    
    global CLASS_NAMES, NUM_CLASSES
    if detected_classes:
        CLASS_NAMES = detected_classes
        NUM_CLASSES = len(CLASS_NAMES)

    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    all_samples, all_targets = [], []

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(target_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        label = class_to_idx[class_name]
        for root, _, files in os.walk(class_dir):
            for fname in files:
                if fname.lower().endswith(valid_extensions):
                    img_path = os.path.join(root, fname)
                    all_samples.append((img_path, label))
                    all_targets.append(label)

    print(f"Successfully loaded {len(all_samples)} samples across {NUM_CLASSES} classes: {CLASS_NAMES}")
    return all_samples, all_targets, class_to_idx


def stratified_train_test_split(all_samples, all_targets,
                                test_size=0.2, seed=42):
    """
    Stratified 80/20 split performed ONCE in __main__ and shared across all
    models - guarantees every model is evaluated on the identical test set.
    """
    from sklearn.model_selection import train_test_split as _tts
    idx = list(range(len(all_samples)))
    train_idx, test_idx = _tts(
        idx, test_size=test_size, stratify=all_targets, random_state=seed
    )
    train_samples = [all_samples[i] for i in train_idx]
    train_targets = [all_targets[i] for i in train_idx]
    test_samples  = [all_samples[i] for i in test_idx]
    test_targets  = [all_targets[i] for i in test_idx]

    print(f"  Train: {len(train_samples)} | Test: {len(test_samples)} "
          f"(stratified {int((1-test_size)*100)}/{int(test_size*100)} split, seed={seed})")
    return train_samples, train_targets, test_samples, test_targets


class SafeDiskDataset(Dataset):
    def __init__(self, samples, targets, transform=None):
        self.samples  = samples
        self.targets  = targets
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ------------------------------------------------------------------------------
# 3. TRANSFORMS - tailored for histopathology images
# ------------------------------------------------------------------------------
def build_transforms(img_size, mean, std, mode='train'):
    if mode == 'train':
        return v2.Compose([
            CLAHE_Transform(clip_limit=2.0, tile_grid_size=(8,8)),
            v2.Resize((img_size, img_size)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.15, contrast=0.15),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
            v2.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3))
        ])
    else:  # val / test
        return v2.Compose([
            CLAHE_Transform(clip_limit=2.0, tile_grid_size=(8,8)),
            v2.Resize((img_size, img_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std)
        ])


# ------------------------------------------------------------------------------
# 4. TTA - Test-Time Augmentation for microscopy & histopathology
# ------------------------------------------------------------------------------
def tta_predict(model, imgs):
    import torchvision.transforms.functional as TF
    def scale_crop(x, scale_factor):
        h, w = x.shape[-2:]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        x_scaled = TF.resize(x, [new_h, new_w], antialias=True)
        return TF.center_crop(x_scaled, [h, w])
    
    # TTA transformations
    tta_tfs = [
        lambda x: x,
        lambda x: scale_crop(x, 1.05),
        lambda x: TF.adjust_contrast(x, 1.1)
    ]
    p_sum = None
    for tf in tta_tfs:
        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
            logits = model(tf(imgs))
        p = torch.softmax(logits, dim=1)
        p_sum = p if p_sum is None else p_sum + p
    return p_sum / len(tta_tfs)


# ------------------------------------------------------------------------------
# 5. EVALUATION
# ------------------------------------------------------------------------------
def evaluate_set(model, loader, use_tta=True):
    """Evaluates with optional TTA. Returns acc, macro-F1, recall, precision."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            if use_tta:
                probs = tta_predict(model, imgs)
                preds = probs.argmax(dim=1).cpu()
            else:
                with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                    preds = model(imgs).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(lbls)

    pv = torch.cat(all_preds).numpy()
    lv = torch.cat(all_labels).numpy()

    class_recalls = recall_score(lv, pv, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    class_f1s = f1_score(lv, pv, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)

    return {
        'acc':           accuracy_score(lv, pv),
        'f1':            f1_score(lv, pv, average='macro', zero_division=0),
        'recall':        recall_score(lv, pv, average='macro', zero_division=0),
        'precision':     precision_score(lv, pv, average='macro', zero_division=0),
        'class_recalls': class_recalls.tolist(),
        'class_f1s':     class_f1s.tolist(),
    }


def print_confusion_matrix(model, loader):
    """Prints a labeled confusion matrix to console for final diagnostics."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            preds = tta_predict(model, imgs).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(lbls)
    pv = torch.cat(all_preds).numpy()
    lv = torch.cat(all_labels).numpy()
    cm = confusion_matrix(lv, pv)
    print("\n  Confusion Matrix (rows=True, cols=Predicted):")
    header = "          " + " ".join(f"{n[:6]:>7}" for n in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i][:10]:<10}", " ".join(f"{v:>7}" for v in row))


# ==============================================================================
# 6. MODEL MAP
# ==============================================================================
MODELS_MAP = {
    # Classic Architectures
    "RESNET18":           "resnet18",
    "RESNET50":           "resnet50",
    "RESNET101":          "resnet101",
    "VGG16":              "vgg16",
    "VGG19":              "vgg19",
    "SE_RESNET50":        "seresnet50",
    "RESNEXT101":         "resnext101_32x8d",
    # Dense & Inception
    "DENSENET121":        "densenet121",
    "DENSENET201":        "densenet201",
    "INCEPTIONV3":        "inception_v3",
    "INCEPTIONRESNETV2":  "inception_resnet_v2",
    "XCEPTION":           "xception",
    # Efficient & Mobile
    "EFFICIENTNETB0":     "efficientnet_b0",
    "EFFICIENTNETB3":     "tf_efficientnet_b3",
    "EFFICIENTNETB7":     "tf_efficientnet_b7",
    "MOBILENETV2":        "mobilenetv2_100",
    "NASNETMOBILE":       "nasnet_mobile",
    "SQUEEZENET":         "squeezenet1_0",
    "MOBILEVIT":          "mobilevit_s",
    # Transformers & Hybrids
    "PROPOSED_GCSWIN":    "swin_tiny_patch4_window7_224",
    "SWIN":               "swin_base_patch4_window7_224",
    "SWIN_V2":            "swinv2_base_window12to16_192to256",
    "VIT_BASE":           "vit_base_patch16_224",
    "VIT_LARGE":          "vit_large_patch16_224",
    "DEIT_SMALL":         "deit_small_distilled_patch16_224",
    "DEIT_BASE":          "deit_base_distilled_patch16_224",
    "BEIT":               "beit_base_patch16_224",
    "T2T_VIT":            "t2t_vit_14",
    # Modern Conv & Experimental
    "CONVNEXT":           "convnext_base",
    "CONVMIXER":          "convmixer_768_32",
    "REGNETY":            "regnety_008",
    "BIT_R50":            "resnetv2_50x1_bit",
    "HRNET":              "hrnet_w32",
    "PVT":                "pvt_v2_b2",
    "CVT":                "cvt_13",
    "COAT_LITE":          "coat_lite_small",
    "TWINS":              "twins_svt_small",
    "POOLFORMER":         "poolformer_s12",
    "EFFICIENTFORMER":    "efficientformer_l1",
    "VITAEV2":            "vit_base_patch16_224_miil",
    "MAXVIT":             "maxvit_tiny_tf_224.in1k",
    "EFFICIENTNETV2_S":   "tf_efficientnetv2_s.in21k_ft_in1k",
    "FASTVIT":            "fastvit_t8.apple_in1k",
}


def get_model_kwargs(model_name):
    kwargs = {}
    if "inception" in model_name:
        kwargs['aux_logits'] = False
    if "swinv2" in model_name:
        kwargs['img_size'] = 256
    return kwargs


def build_classifier(model_name, kwargs):
    """
    Standard linear head for multi-class ovarian histopathology classification.
    """
    backbone = timm.create_model(
        model_name, pretrained=True, num_classes=0, **kwargs
    )
    head = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(backbone.num_features, NUM_CLASSES)
    )
    return nn.Sequential(backbone, head)


# ==============================================================================
# 7. EXPERIMENT ENGINE
# ==============================================================================
def run_experiment(model_name, train_samples, train_targets,
                   test_samples, test_targets,
                   epochs=20, folds=5, patience=5):
    """
    train_samples / test_samples : list of (abs_path, label_int)
    train_targets / test_targets : list of label_int
    The train/test split is done ONCE in __main__ and passed in here so
    every model is evaluated on the identical held-out test set.
    """
    # Resume guard
    if os.path.exists(MASTER_RESULTS_FILE):
        if model_name in pd.read_csv(MASTER_RESULTS_FILE)['model_name'].values:
            print(f"--- Skipping {model_name} (already completed) ---")
            return

    img_size = get_img_size(model_name)
    stats    = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    print(f"  -> img_size={img_size}px (Using ImageNet Stats)")

    fold_metrics = []
    safe_name    = model_name.replace("/", "_").replace(":", "_")
    kwargs       = get_model_kwargs(model_name)

    # -----------------------------------------------------------------------
    # FOLD SPLITS — StratifiedKFold on training split only
    # folds=1 -> single pass: entire train set for training, no CV val loop
    # folds=5 -> full 5-fold CV, change FOLDS in __main__ to enable
    # -----------------------------------------------------------------------
    if folds == 1:
        from sklearn.model_selection import train_test_split as _tts
        all_idx = list(range(len(train_samples)))
        tr_idx, val_idx = _tts(
            all_idx, test_size=0.2, stratify=train_targets, random_state=42
        )
        fold_splits = [(tr_idx, val_idx)]
    else:
        skf         = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        fold_splits = list(skf.split(train_samples, train_targets))

    # -----------------------------------------------------------------------
    # FOLD LOOP
    # -----------------------------------------------------------------------
    for fold, (tr_idx, val_idx) in enumerate(fold_splits):
        model = None
        try:
            print(f"\n--- {model_name} | Fold {fold+1}/{folds} ---")

            fold_train_samples = [train_samples[i] for i in tr_idx]
            fold_train_targets = [train_targets[i] for i in tr_idx]
            fold_val_samples   = [train_samples[i] for i in val_idx]
            fold_val_targets   = [train_targets[i] for i in val_idx]

            fold_class_counts = np.bincount(fold_train_targets, minlength=NUM_CLASSES)
            total = len(fold_train_targets)
            num_c = len(fold_class_counts)
            class_weights = {l: total / (num_c * (fold_class_counts[l] + 1e-6)) for l in range(num_c)}
            sample_weights = torch.tensor([class_weights[l] for l in fold_train_targets], dtype=torch.float)
            sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(sample_weights), replacement=True
            )

            # Model
            model = build_classifier(model_name, kwargs).to(DEVICE)

            # Transforms
            train_trans = build_transforms(img_size, stats[0], stats[1], mode='train')
            val_trans   = build_transforms(img_size, stats[0], stats[1], mode='val')

            train_loader = DataLoader(
                SafeDiskDataset(fold_train_samples, fold_train_targets, train_trans),
                batch_size=32, sampler=sampler, num_workers=4, pin_memory=True
            )
            val_loader = DataLoader(
                SafeDiskDataset(fold_val_samples, fold_val_targets, val_trans),
                batch_size=32, shuffle=False, num_workers=4, pin_memory=True
            )

            # --- Per-epoch Logger ---
            history_file = os.path.join(LOG_DIR, f"{safe_name}_f{fold+1}_history.csv")
            if os.path.exists(history_file):
                os.remove(history_file)
            epoch_logger = EpochLogger(history_file)

            # Loss & Optimizer
            criterion = FocalLoss(gamma=2.0, num_classes=NUM_CLASSES, alpha=torch.ones(NUM_CLASSES, device=DEVICE)).to(DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            scaler    = torch.amp.GradScaler(device="cuda", enabled=(DEVICE == "cuda"))

            best_f1    = -1.0
            best_m     = {}
            no_improve = 0

            for epoch in range(epochs):
                # Training pass
                model.train()
                running_loss = 0.0
                for imgs, lbls in tqdm(train_loader, desc=f"F{fold+1} E{epoch+1}", leave=False):
                    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                    optimizer.zero_grad()
                    with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
                        loss = criterion(model(imgs), lbls)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    running_loss += loss.item()

                scheduler.step()

                # Validation
                m = evaluate_set(model, val_loader, use_tta=False)
                criterion.update_alpha(m.get('class_f1s', []))
                
                avg_loss = running_loss / len(train_loader)
                print(f"  E{epoch+1} | Loss: {avg_loss:.4f} | "
                      f"Acc: {m['acc']:.4f} | F1: {m['f1']:.4f} | "
                      f"Recall: {m['recall']:.4f}")

                # Save to CSV
                log_data = {
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "val_acc": m['acc'],
                    "val_f1": m['f1'],
                    "val_recall": m['recall'],
                    "val_precision": m['precision']
                }
                if 'class_recalls' in m:
                    for i, r in enumerate(m['class_recalls']):
                        log_data[f"recall_{CLASS_NAMES[i]}"] = r
                epoch_logger.log(log_data)

                # Best checkpoint tracked by macro-F1 (primary metric for imbalanced subtypes)
                if m['f1'] > best_f1:
                    best_f1, best_m = m['f1'], m
                    no_improve = 0
                    torch.save(
                        model.state_dict(),
                        f"{CHECKPOINT_DIR}/{safe_name}_f{fold+1}.pth"
                    )
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                        break

            fold_metrics.append(best_m)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"--- OOM on Fold {fold+1} for {model_name} — skipping fold ---")
                continue
            raise
        finally:
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()

    if not fold_metrics:
        print(f"--- All folds failed for {model_name} — skipping ---")
        return

    # -----------------------------------------------------------------------
    # TEST SET: ENSEMBLE SOFTMAX AVERAGING ACROSS ALL FOLD CHECKPOINTS
    # -----------------------------------------------------------------------
    try:
        test_trans = build_transforms(img_size, stats[0], stats[1], mode='test')
        test_loader = DataLoader(
            SafeDiskDataset(test_samples, test_targets, test_trans),
            batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )

        ensemble_probs = None
        loaded_folds   = 0

        for fold_idx in range(len(fold_metrics)):
            ckpt_path = f"{CHECKPOINT_DIR}/{safe_name}_f{fold_idx+1}.pth"
            if not os.path.exists(ckpt_path):
                print(f"  Warning: checkpoint missing for fold {fold_idx+1}, skipping")
                continue

            fold_model = build_classifier(model_name, kwargs).to(DEVICE)
            fold_model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=DEVICE))
            fold_model.eval()

            fold_probs = []
            with torch.no_grad():
                for imgs, _ in tqdm(test_loader, desc=f"Ensemble fold {fold_idx+1}", leave=False):
                    imgs = imgs.to(DEVICE)
                    fold_probs.append(tta_predict(fold_model, imgs).cpu())

            fold_probs_cat = torch.cat(fold_probs)
            ensemble_probs = fold_probs_cat if ensemble_probs is None \
                             else ensemble_probs + fold_probs_cat
            loaded_folds += 1

            del fold_model
            gc.collect()
            torch.cuda.empty_cache()

        ensemble_probs /= loaded_folds
        pred_ens = ensemble_probs.argmax(dim=1).numpy()
        lv       = np.array(test_targets)

        test_results = {
            'acc':       accuracy_score(lv, pred_ens),
            'f1':        f1_score(lv, pred_ens, average='macro', zero_division=0),
            'recall':    recall_score(lv, pred_ens, average='macro', zero_division=0),
            'precision': precision_score(lv, pred_ens, average='macro', zero_division=0),
        }

        # Per-class F1 breakdown
        per_class_f1 = f1_score(lv, pred_ens, average=None, zero_division=0)
        for i, cls in enumerate(CLASS_NAMES):
            test_results[f'f1_{cls.replace("-", "_")}'] = per_class_f1[i]

        # Save results
        fold_df   = pd.DataFrame(fold_metrics)
        final_row = {'model_name': model_name, 'img_size': img_size}
        numeric_cols = [c for c in fold_df.columns if isinstance(fold_df[c].iloc[0], (int, float, np.number))]
        for col in numeric_cols:
            final_row[f'cv_{col}_mean'] = fold_df[col].mean()
            final_row[f'cv_{col}_std']  = fold_df[col].std()
        for k, v in test_results.items():
            final_row[f'test_{k}'] = v

        pd.DataFrame([final_row]).to_csv(
            MASTER_RESULTS_FILE, mode='a',
            header=not os.path.exists(MASTER_RESULTS_FILE), index=False
        )

        print(f"\nDone: {model_name}")
        print(f"   Test -> Acc: {test_results['acc']:.4f} | F1: {test_results['f1']:.4f} | "
              f"Recall: {test_results['recall']:.4f}")
        print(f"   CV   -> F1: {fold_df['f1'].mean():.4f} ± {fold_df['f1'].std():.4f}")
        print(f"   Per-class F1: { {c: f'{per_class_f1[i]:.3f}' for i, c in enumerate(CLASS_NAMES)} }")

    except Exception as e:
        print(f"--- Ensemble Eval Error for {model_name}: {e} ---")


# ==============================================================================
# 8. MAIN
# ==============================================================================
if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- Point to your Ovarian Cancer Histopathology dataset folder ----------
    # Expected Kaggle structure:
    #   DATA_DIR/ (or /kaggle/input/datasets/bitsnpieces/ovarian-cancer-and-subtypes-dataset-histopathology/OvarianCancer)
    #     Clear_Cell/
    #     Endometrioid/
    #     Mucinous/
    #     Non_Cancerous/
    #     Serous/
    # ------------------------------------------------------------------------
    DATA_DIR = r"/kaggle/input/datasets/bitsnpieces/ovarian-cancer-and-subtypes-dataset-histopathology/OvarianCancer"

    print(f"\nRunning on: {DEVICE}")

    # --- Load ovarian dataset & split ONCE — shared across all models ------------
    # Stratified 80/20 split with fixed seed ensures every model sees the same
    # training and test samples. Stats are computed per model inside run_experiment
    # (cached per resolution so 224px models only scan the dataset once total).
    all_samples, all_targets, class_to_idx = load_flat_dataset(DATA_DIR)
    train_samples, train_targets, test_samples, test_targets = \
        stratified_train_test_split(all_samples, all_targets, test_size=0.2, seed=42)

    # --- Top 20 models for ovarian cancer histopathology research (SOTA Transformers + Efficient CNNs) ---
    TOP20_KEYS = [
        # --- Transformers & Hybrids ---
        "PVT",               # Pyramid Vision Transformer
        "CVT",               # Convolutional Vision Transformer
        "SWIN",              # Hierarchical Transformer Baseline
        "VIT_LARGE",         # Large Vision Transformer Baseline
        
        # --- Modern ConvNets ---
        "CONVNEXT",          # Modern overhaul of CNN architecture
        "HRNET",             # Excellent for detail (maintains high resolution)
        "BIT_R50",           # Big Transfer: strong inductive bias
        "REGNETY",           # Highly optimized for speed/accuracy trade-offs
        
        # --- Efficiency & Classic Strong Baselines ---
        "EFFICIENTNETB7",    # High-capacity scaling contender
        "MOBILEVIT",         # Mobile-friendly Hybrid
        "DENSENET201",       # DenseNet: strong gradient flow
        "DENSENET121",       # DenseNet: Standard benchmark
        "RESNEXT101",        # Cardinality-based residual variant
        "RESNET50",          # Standard Residual Benchmark
        "INCEPTIONV3",       # Classic Inception baseline
        "MOBILENETV2",       # Efficiency baseline
        "PROPOSED_GCSWIN",   # Gated-Linear Swin variant
        
        # --- New High-Performance Baselines ---
        "MAXVIT",            # Multi-Axis Vision Transformer
        "EFFICIENTNETV2_S",  # EfficientNetV2 (Small, but newer/faster)
        "FASTVIT",           # Fast and robust mobile-focused architecture
    ]
    target_models = [MODELS_MAP[k] for k in TOP20_KEYS]
    print(f"Total target models for research: {len(target_models)}\n")

    # --- Research settings ---------------------------------------------------
    EPOCHS   = 30   # Enough for convergence in medical imaging
    FOLDS    = 1    # Single fold for lightning-fast baseline benchmarking
    PATIENCE = 7    # Moderate patience for diverse architectures

    for m in target_models:
        try:
            run_experiment(
                m,
                train_samples, train_targets,
                test_samples,  test_targets,
                epochs=EPOCHS, folds=FOLDS, patience=PATIENCE
            )
        except Exception as e:
            print(f"Critical failure on {m}: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            continue

    print("\n✅ Master experiment complete. Results in:", MASTER_RESULTS_FILE)
