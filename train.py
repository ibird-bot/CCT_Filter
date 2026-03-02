"""
Photogrammetry Target Classifier — Training Script
===================================================
Prerequisites (run these in terminal first):

1. Check your CUDA version:
   nvidia-smi

2. Install PyTorch with CUDA (pick the right one from https://pytorch.org):
   For CUDA 12.x:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

   For CUDA 11.8:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

3. Install timm (model library):
   pip install timm

4. Verify GPU is detected:
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

Usage:
    python train.py --data_dir ./data/prepared --epochs 30 --batch_size 32
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import timm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import time
import json

# ── config ────────────────────────────────────────────────────────────────────

CLASSES     = ['Coded', 'not_coded', 'not_target']
IMG_SIZE    = 128
VAL_SPLIT   = 0.2      # 20% of data for validation
SEED        = 42

# ── transforms ────────────────────────────────────────────────────────────────

# Training: some extra augmentation on top of what prepare_data.py already did
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.ToTensor(),
    # ImageNet mean/std — required for pretrained EfficientNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Validation: no augmentation, just normalize
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes=3):
    """
    EfficientNet-B0 pretrained on ImageNet.
    We freeze the backbone and only train the classifier head first.
    After a few epochs we unfreeze everything for fine-tuning.
    This is called transfer learning — standard practice.
    """
    model = timm.create_model(
        'efficientnet_b0',
        pretrained=True,
        num_classes=num_classes
    )
    return model


def freeze_backbone(model):
    """Freeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params (head only): {trainable:,}")


def unfreeze_all(model):
    """Unfreeze everything for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params (full model): {trainable:,}")


# ── training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels      = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            loss           = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_history(history, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#1a1a2e')

    for ax in [ax1, ax2]:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='white')
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], color='#64B5F6', label='Train')
    ax1.plot(epochs, history['val_loss'],   color='#EF9A9A', label='Val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend(facecolor='#2a2a3e', labelcolor='white')

    ax2.plot(epochs, history['train_acc'], color='#64B5F6', label='Train')
    ax2.plot(epochs, history['val_acc'],   color='#EF9A9A', label='Val')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylim(0, 1)
    ax2.legend(facecolor='#2a2a3e', labelcolor='white')

    # mark where backbone was unfrozen
    if 'unfreeze_epoch' in history:
        for ax in [ax1, ax2]:
            ax.axvline(history['unfreeze_epoch'], color='#FFD54F',
                       linestyle='--', linewidth=1, alpha=0.8)
            ax.text(history['unfreeze_epoch'] + 0.2, ax.get_ylim()[0] + 0.01,
                    'unfrozen', color='#FFD54F', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / 'training_history.png', dpi=130,
                bbox_inches='tight', facecolor='#1a1a2e')
    print(f"  Saved training_history.png")


def plot_confusion_matrix(cm, class_names, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, color='white', rotation=15)
    ax.set_yticklabels(class_names, color='white')
    ax.set_xlabel('Predicted', color='white')
    ax.set_ylabel('True', color='white')
    ax.set_title('Confusion Matrix', color='white')

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] < cm.max() * 0.6 else 'black',
                    fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_dir / 'confusion_matrix.png', dpi=130,
                bbox_inches='tight', facecolor='#1a1a2e')
    print(f"  Saved confusion_matrix.png")


# ── main ──────────────────────────────────────────────────────────────────────

def train(data_dir, epochs, batch_size):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data_dir = Path(data_dir)
    out_dir  = data_dir.parent / 'results'
    out_dir.mkdir(exist_ok=True)

    # ── device ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # ── dataset ──
    # Load full dataset with train transforms first to get class mapping
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
    print(f"\nClasses found : {full_dataset.classes}")
    print(f"Total patches : {len(full_dataset)}")

    # train/val split
    val_size   = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # apply val transforms to val split
    val_ds.dataset = datasets.ImageFolder(data_dir, transform=val_transforms)

    print(f"Train patches : {train_size}")
    print(f"Val patches   : {val_size}")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── model ──
    print(f"\nBuilding EfficientNet-B0 (pretrained)...")
    model = build_model(num_classes=len(full_dataset.classes)).to(device)

    criterion = nn.CrossEntropyLoss()

    # ── training strategy ──
    # Phase 1: freeze backbone, train head only (fast, low risk of overfitting)
    # Phase 2: unfreeze all, fine-tune with low LR
    PHASE1_EPOCHS = min(10, epochs // 3)
    PHASE2_EPOCHS = epochs - PHASE1_EPOCHS

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  [],
        'unfreeze_epoch': PHASE1_EPOCHS + 1
    }

    best_val_acc = 0.0
    best_model_path = out_dir / 'best_model.pth'

    # ── Phase 1: head only ──
    print(f"\n{'='*50}")
    print(f"Phase 1: Training head only ({PHASE1_EPOCHS} epochs)")
    print(f"{'='*50}")
    freeze_backbone(model)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, PHASE1_EPOCHS)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc           = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, vp, vl  = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), best_model_path)
            saved = " ← best"
        else:
            saved = ""

        print(f"  Ep {epoch:02d}/{PHASE1_EPOCHS} | "
              f"loss {tr_loss:.3f}/{vl_loss:.3f} | "
              f"acc {tr_acc:.3f}/{vl_acc:.3f} | "
              f"{time.time()-t0:.1f}s{saved}")

    # ── Phase 2: fine-tune all ──
    print(f"\n{'='*50}")
    print(f"Phase 2: Fine-tuning full model ({PHASE2_EPOCHS} epochs)")
    print(f"{'='*50}")
    unfreeze_all(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # lower LR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, PHASE2_EPOCHS)

    for epoch in range(1, PHASE2_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc           = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, vp, vl  = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), best_model_path)
            saved = " ← best"
        else:
            saved = ""

        print(f"  Ep {epoch:02d}/{PHASE2_EPOCHS} | "
              f"loss {tr_loss:.3f}/{vl_loss:.3f} | "
              f"acc {tr_acc:.3f}/{vl_acc:.3f} | "
              f"{time.time()-t0:.1f}s{saved}")

    # ── final evaluation ──
    print(f"\n{'='*50}")
    print(f"Final Evaluation (best model, val set)")
    print(f"{'='*50}")
    model.load_state_dict(torch.load(best_model_path))
    _, final_acc, all_preds, all_labels = val_epoch(
        model, val_loader, criterion, device
    )

    class_names = full_dataset.classes
    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    print(f"Confusion Matrix:")
    print(cm)

    # ── save outputs ──
    plot_history(history, out_dir)
    plot_confusion_matrix(cm, class_names, out_dir)

    # save class mapping for inference
    class_map = {i: name for i, name in enumerate(class_names)}
    with open(out_dir / 'class_map.json', 'w') as f:
        json.dump(class_map, f, indent=2)

    print(f"\nAll outputs saved to: {out_dir}/")
    print(f"  best_model.pth       ← use this for inference")
    print(f"  training_history.png ← loss and accuracy curves")
    print(f"  confusion_matrix.png ← where the model makes mistakes")
    print(f"  class_map.json       ← class index to name mapping")
    print(f"\nNext step: run predict.py to test on individual images")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   default='./data/prepared',
                        help='Prepared data folder (output of prepare_data.py)')
    parser.add_argument('--epochs',     default=30, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    args = parser.parse_args()

    # check dependencies
    try:
        import timm
        from sklearn.metrics import confusion_matrix
    except ImportError:
        print("Missing dependencies. Run:")
        print("  pip install timm scikit-learn")
        exit(1)

    train(args.data_dir, args.epochs, args.batch_size)