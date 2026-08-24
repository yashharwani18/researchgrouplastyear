import os
import torch
import numpy as np
from PIL import Image
from collections import Counter
import unique
from torchvision import transforms

def run_diagnostics():
    print("="*60)
    print("  OVARIAN CANCER DATASET DIAGNOSTIC REPORT")
    print("="*60)
    
    # 1. Dataset Directory & Class Discovery
    data_dir = unique.find_dataset_dir()
    print(f"\n1. DATASET RESOLUTION:")
    print(f"   Target Dataset Path: {os.path.abspath(data_dir)}")
    
    dataset = unique.OvarianDataset(data_dir)
    print(f"   Total Images Loaded: {len(dataset.samples)}")
    print(f"   Detected Classes: {dataset.class_names}")
    
    # Class distribution across total dataset
    all_labels = [s[1] for s in dataset.samples]
    total_counts = Counter(all_labels)
    print("   Per-Class Sample Count (Total):")
    for idx, name in enumerate(dataset.class_names):
        print(f"     - Class {idx} ({name:<15}): {total_counts[idx]} images")
        
    # 2. Train / Val Split Analysis
    train_samples, train_labels = unique.get_all_samples(data_dir)
    test_samples = unique.get_test_samples(data_dir)
    
    print(f"\n2. STRATIFIED TRAIN / TEST SPLIT (80/20):")
    print(f"   Train Set Count: {len(train_samples)}")
    print(f"   Test Set Count:  {len(test_samples)}")
    
    train_counts = Counter(train_labels)
    test_counts = Counter([s[1] for s in test_samples])
    
    print("\n   Class Breakdown (Train vs Test):")
    print(f"   {'Class Name':<18} | {'Train Count':<12} | {'Test Count':<12} | {'Train Ratio':<12}")
    print("   " + "-"*60)
    for idx, name in enumerate(dataset.class_names):
        tr_c = train_counts[idx]
        te_c = test_counts[idx]
        tot = tr_c + te_c
        ratio = tr_c / tot if tot > 0 else 0
        print(f"   {name:<18} | {tr_c:<12} | {te_c:<12} | {ratio*100:.1f}%")
        
    # 3. Check Image Properties (Dimensions, Channels, Min/Max intensities)
    print("\n3. IMAGE PROPERTY & PREPROCESSING DIAGNOSTIC:")
    train_tf = unique.get_transforms(224, mode='train')
    eval_tf  = unique.get_transforms(224, mode='eval')
    
    sample_img_path, sample_lbl = dataset.samples[0]
    with Image.open(sample_img_path) as img:
        img_rgb = img.convert('RGB')
        print(f"   Sample File: {os.path.basename(sample_img_path)}")
        print(f"   Native Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
        
        t_tensor = train_tf(img_rgb)
        e_tensor = eval_tf(img_rgb)
        
        print(f"   Train Tensor Shape: {t_tensor.shape}, Min: {t_tensor.min():.3f}, Max: {t_tensor.max():.3f}, Mean: {t_tensor.mean():.3f}")
        print(f"   Eval Tensor Shape:  {e_tensor.shape}, Min: {e_tensor.min():.3f}, Max: {e_tensor.max():.3f}, Mean: {e_tensor.mean():.3f}")
        
    # 4. Standard ResNet18 / DenseNet121 Baseline Diagnostic Run
    print("\n4. BASELINE BACKBONE PERFORMANCE CHECK (5 Epochs):")
    device = unique.DEVICE
    import timm
    baseline_model = timm.create_model('densenet121', pretrained=True, num_classes=len(dataset.class_names)).to(device)
    
    train_loader = unique.get_loaders_for_fold(train_samples, train_labels, batch_size=16, img_size=224, num_workers=2)
    eval_ds = unique._Subset(test_samples, eval_tf)
    eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=16, shuffle=False)
    
    opt = torch.optim.AdamW(baseline_model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = torch.nn.CrossEntropyLoss()
    
    for ep in range(1, 6):
        baseline_model.train()
        t_correct, t_total, t_loss = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = baseline_model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            t_loss += loss.item()
            t_correct += (out.argmax(1) == y).sum().item()
            t_total += y.size(0)
            
        baseline_model.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for x, y in eval_loader:
                out = baseline_model(x.to(device))
                v_preds.extend(out.argmax(1).cpu().tolist())
                v_targets.extend(y.tolist())
                
        from sklearn.metrics import accuracy_score, f1_score
        val_acc = accuracy_score(v_targets, v_preds)
        val_f1  = f1_score(v_targets, v_preds, average='macro', zero_division=0)
        
        print(f"   DenseNet121 Epoch {ep}: Train Acc: {t_correct/t_total*100:.1f}% | Val Acc: {val_acc*100:.1f}% | Val F1: {val_f1:.4f}")
        print(f"   Val Predictions Distribution: {Counter(v_preds)}")

if __name__ == "__main__":
    run_diagnostics()
