"""
Test script: Ellipse detection + CCT Filter model
==================================================
1. Runs ellipse fitting (from image_processing_saas) on images in test_data/
2. Keeps only (x, y) coords from detections
3. Crops a patch around each (x, y), feeds to the trained classifier
4. Keeps detections classified as "Coded" OR "not_coded"
5. Discards detections classified as "not_target"
6. Plots results:
     GREEN  = Coded target      (kept)
     BLUE   = Uncoded target    (kept)
     RED x  = False positive    (deleted)

Requires:
- image_processing_saas repo sibling to CCT_Filter (same parent folder)
- Trained model at data/results/best_model.pth (run train.py first)
- test_data/ folder with images

Usage:
    python test.py
    python test.py --test_dir ./test_data --model_path ./data/results/best_model.pth
    python test.py --save   (saves result images to test_data/)
"""

import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image

# ── path setup ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
SAAS_ROOT    = PROJECT_ROOT.parent / "image_processing_saas"
if not SAAS_ROOT.is_dir():
    raise FileNotFoundError(
        f"image_processing_saas not found at {SAAS_ROOT}. "
        "Clone it next to CCT_Filter (same parent folder)."
    )
sys.path.insert(0, str(SAAS_ROOT))

from src.fit_ellipsis              import EllipsisDetector
from src.optimize_fit_ellipsis_params import EllipseFittingOptimizer

# ── config ────────────────────────────────────────────────────────────────────

IMG_SIZE = 128
VAL_MEAN = [0.485, 0.456, 0.406]
VAL_STD  = [0.229, 0.224, 0.225]
CONFIDENCE_THRESHOLD = 0.85  # only keep if model is sure

# Visual style for each class
CLASS_STYLE = {
    'Coded':      {'color': 'lime',    'edge': 'darkgreen', 'marker': 'o', 'label': 'Coded target (kept)'},
    'not_coded':  {'color': '#4FC3F7', 'edge': '#0277BD',   'marker': 'o', 'label': 'Uncoded target (kept)'},
    'not_target': {'color': 'red',     'edge': 'darkred',   'marker': 'x', 'label': 'False positive (deleted)'},
}

# ── model ─────────────────────────────────────────────────────────────────────

def load_filter_model(model_path, device):
    """Load trained EfficientNet-B0 classifier and class map."""
    import timm

    model_path     = Path(model_path)
    class_map_path = model_path.parent / "class_map.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run train.py first."
        )
    if not class_map_path.exists():
        raise FileNotFoundError(
            f"class_map.json not found at {class_map_path}."
        )

    with open(class_map_path) as f:
        class_map = json.load(f)  # {str_index: class_name}

    # build reverse map: class_name -> int index
    name_to_idx = {v: int(k) for k, v in class_map.items()}

    model = timm.create_model(
        "efficientnet_b0", pretrained=False,
        num_classes=len(class_map)
    )
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()

    return model, class_map, name_to_idx


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=VAL_MEAN, std=VAL_STD),
    ])

# ── ellipse detection ─────────────────────────────────────────────────────────

def detect_ellipses(image_bgr, accuracy_level=2, black_on_white=False):
    """
    Run ellipse detection using image_processing_saas.
    Returns centers (N, 2) and ellipse_array_1 (N, 5).
    """
    optimizer   = EllipseFittingOptimizer(image_bgr)
    params      = optimizer.optimize_parameters(accuracy_level, black_on_white)
    detector    = EllipsisDetector(params)
    ellipse_set = detector.detect_and_refine(image_bgr)
    centers     = ellipse_set.get_center_coordinates()   # (N, 2) x, y
    ellipse_arr = ellipse_set.get_ellipse_array_1()       # (N, 5) x,y,A,B,psi
    return centers, ellipse_arr


# ── patch extraction ──────────────────────────────────────────────────────────

def crop_patch(img_bgr, x, y, half_size):
    """Crop a square patch around (x, y) and resize to IMG_SIZE."""
    h, w = img_bgr.shape[:2]
    x0   = max(0, int(x) - half_size)
    y0   = max(0, int(y) - half_size)
    x1   = min(w, int(x) + half_size)
    y1   = min(h, int(y) + half_size)
    patch = img_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    return cv2.resize(patch, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)


# ── classifier ────────────────────────────────────────────────────────────────

def run_filter(model, patches_rgb, transform, device):
    """
    Run classifier on a list of RGB patches.
    Returns predicted class indices and confidence scores.
    """
    if len(patches_rgb) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    tensors = []
    for patch in patches_rgb:
        pil = Image.fromarray(patch)
        tensors.append(transform(pil))

    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        logits      = model(batch)
        probs       = torch.softmax(logits, dim=1)
        preds       = probs.argmax(dim=1)
        confidences = probs.max(dim=1).values

    return preds.cpu().numpy(), confidences.cpu().numpy()


# ── main processing ───────────────────────────────────────────────────────────

def process_image(img_path, model, transform, device,
                  name_to_idx, accuracy_level=2, black_on_white=False):
    """
    Full pipeline: load → detect ellipses → crop → classify → sort results.
    Returns img_rgb and a dict of {class_name: [(x, y, confidence), ...]}
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Could not load image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    centers, ellipse_arr = detect_ellipses(img_bgr, accuracy_level, black_on_white)

    if centers.shape[0] == 0:
        print("  No ellipses detected.")
        return img_rgb, {cls: [] for cls in CLASS_STYLE}

    xs = centers[:, 0]
    ys = centers[:, 1]
    A  = ellipse_arr[:, 2]
    B  = ellipse_arr[:, 3]

    # crop size based on ellipse radii
    half_sizes = np.ceil(np.maximum(A, B) * 2).astype(int)
    half_sizes = np.clip(
        half_sizes, 32,
        min(img_bgr.shape[0], img_bgr.shape[1]) // 2
    )

    patches_rgb = []
    for i in range(len(centers)):
        patch = crop_patch(img_bgr, xs[i], ys[i], int(half_sizes[i]))
        if patch is not None:
            patches_rgb.append(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        else:
            patches_rgb.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

    preds, confs = run_filter(model, patches_rgb, transform, device)

    # Keep all detections that are not "not_target"; delete only false positives
    not_target_idx = name_to_idx['not_target']
    kept_indices   = (preds != not_target_idx).nonzero()[0]
    deleted_indices = (preds == not_target_idx).nonzero()[0]
    assert len(kept_indices) + len(deleted_indices) == len(preds), "kept + deleted should equal total"

    # Sort into classes for plotting (Coded / not_coded / not_target)
    results = {cls: [] for cls in CLASS_STYLE}
    idx_to_name = {v: k for k, v in name_to_idx.items()}

    for i, (pred, conf) in enumerate(zip(preds, confs)):
        cls_name = idx_to_name.get(int(pred), 'not_target')
        # if confidence is low, treat as not_target regardless
        if conf < CONFIDENCE_THRESHOLD:
            cls_name = 'not_target'
        results[cls_name].append((float(xs[i]), float(ys[i]), float(conf)))

    return img_rgb, results


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_result(img_rgb, results, title="", save_path=None):
    """
    Overlay classified detections on image.
    Green  = Coded      (kept)
    Blue   = Uncoded    (kept)
    Red x  = not_target (deleted / false positive)
    """
    _, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.imshow(img_rgb)

    # Three groups for clarity: coded, uncoded, false positive
    coded_xy   = [(p[0], p[1]) for p in results.get('Coded', [])]
    uncoded_xy = [(p[0], p[1]) for p in results.get('not_coded', [])]
    deleted_xy = [(p[0], p[1]) for p in results.get('not_target', [])]

    total_kept    = len(coded_xy) + len(uncoded_xy)
    total_deleted = len(deleted_xy)

    if coded_xy:
        ax.scatter(
            [p[0] for p in coded_xy], [p[1] for p in coded_xy],
            c=CLASS_STYLE['Coded']['color'],
            s=90, marker=CLASS_STYLE['Coded']['marker'],
            edgecolors=CLASS_STYLE['Coded']['edge'],
            linewidths=2,
            label=f"{CLASS_STYLE['Coded']['label']}  [{len(coded_xy)}]",
            zorder=3, alpha=0.85
        )
    if uncoded_xy:
        ax.scatter(
            [p[0] for p in uncoded_xy], [p[1] for p in uncoded_xy],
            c=CLASS_STYLE['not_coded']['color'],
            s=90, marker=CLASS_STYLE['not_coded']['marker'],
            edgecolors=CLASS_STYLE['not_coded']['edge'],
            linewidths=2,
            label=f"{CLASS_STYLE['not_coded']['label']}  [{len(uncoded_xy)}]",
            zorder=3, alpha=0.85
        )
    if deleted_xy:
        ax.scatter(
            [p[0] for p in deleted_xy], [p[1] for p in deleted_xy],
            c=CLASS_STYLE['not_target']['color'],
            s=90, marker=CLASS_STYLE['not_target']['marker'],
            linewidths=2,
            label=f"{CLASS_STYLE['not_target']['label']}  [{len(deleted_xy)}]",
            zorder=2, alpha=0.85
        )

    total = total_kept + total_deleted
    ax.set_title(
        f"{title}\n"
        f"Total detections: {total}  |  "
        f"Kept: {total_kept}  |  "
        f"Deleted: {total_deleted}",
        color='white', fontsize=11, pad=10
    )

    ax.legend(
        loc='upper right',
        facecolor='#2a2a3e',
        labelcolor='white',
        fontsize=10,
        framealpha=0.85
    )

    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor='#1a1a2e')
        print(f"    Saved → {save_path}")

    plt.show()
    plt.close()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ellipse detection + CCT Filter — three-class output"
    )
    parser.add_argument(
        '--test_dir',
        default=str(PROJECT_ROOT / 'test_data'),
        help='Folder with test images'
    )
    parser.add_argument(
        '--model_path',
        default=str(PROJECT_ROOT / 'data' / 'results' / 'best_model.pth'),
        help='Path to best_model.pth'
    )
    parser.add_argument(
        '--accuracy', type=int, default=2, choices=[1, 2, 3],
        help='Ellipse detection accuracy level (1=fast, 3=slow)'
    )
    parser.add_argument(
        '--black_on_white', action='store_true',
        help='Use if targets are black on white background'
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Save result plots alongside test images'
    )
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice : {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    print("Loading model...")
    model, class_map, name_to_idx = load_filter_model(args.model_path, device)
    transform = get_val_transform()
    print(f"Classes: {class_map}\n")

    exts   = {'.png', '.jpg', '.jpeg', '.bmp'}
    images = sorted([f for f in test_dir.iterdir()
                     if f.suffix.lower() in exts
                     and '_filtered' not in f.stem])

    if not images:
        print(f"No images found in {test_dir}")
        return

    print(f"Processing {len(images)} image(s) from {test_dir}\n")

    for img_path in images:
        print(f"  {img_path.name} ...", end=' ', flush=True)
        try:
            img_rgb, results = process_image(
                img_path, model, transform, device, name_to_idx,
                accuracy_level=args.accuracy,
                black_on_white=args.black_on_white,
            )

            n_coded    = len(results['Coded'])
            n_uncoded  = len(results['not_coded'])
            n_fp       = len(results['not_target'])
            n_total    = n_coded + n_uncoded + n_fp

            print(
                f"{n_total} detections → "
                f"coded: {n_coded}  "
                f"uncoded: {n_uncoded}  "
                f"deleted: {n_fp}"
            )

            save_path = (
                test_dir / f"{img_path.stem}_filtered.png"
                if args.save else None
            )

            plot_result(
                img_rgb, results,
                title=img_path.name,
                save_path=save_path
            )

        except Exception as e:
            print(f"ERROR: {e}")
            raise

    print("\nDone.")


if __name__ == '__main__':
    main()