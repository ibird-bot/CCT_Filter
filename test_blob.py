"""
Test script: Blob-based detection + CCT Filter model
====================================================
This version avoids the private ellipse fitting code and uses only
OpenCV primitives (SimpleBlobDetector) to propose candidate targets.

Pipeline:
1. Runs a blob detector on images in test_data/ to find bright/dark spots.
2. Uses blob centers (x, y) and size as approximate ellipse radius.
3. Crops a patch around each (x, y), feeds it to the trained classifier.
4. Keeps detections classified as "Coded" OR "not_coded"
   (deleted = "not_target" or low-confidence predictions).
5. Plots results:
     GREEN  = Coded target      (kept)
     BLUE   = Uncoded target    (kept)
     RED x  = False positive    (deleted)

Usage:
    python test_blob.py
    python test_blob.py --test_dir ./test_data --model_path ./data/results/best_model.pth
    python test_blob.py --save   (saves result images to test_data/)
"""

import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image


# ── config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent

IMG_SIZE = 128
VAL_MEAN = [0.485, 0.456, 0.406]
VAL_STD  = [0.229, 0.224, 0.225]
BATCH_SIZE = 64  # run inference in smaller chunks to avoid GPU OOM
CONFIDENCE_THRESHOLD = 0.9  # only keep if model is sure

# Visual style for each class
CLASS_STYLE = {
    "Coded": {
        "color": "lime",
        "edge": "darkgreen",
        "marker": "o",
        "label": "Coded target (kept)",
    },
    "not_coded": {
        "color": "#4FC3F7",
        "edge": "#0277BD",
        "marker": "o",
        "label": "Uncoded target (kept)",
    },
    "not_target": {
        "color": "red",
        "edge": "darkred",
        "marker": "x",
        "label": "False positive (deleted)",
    },
}


# ── model ─────────────────────────────────────────────────────────────────────

def load_filter_model(model_path, device):
    """Load trained EfficientNet-B0 classifier and class map."""
    import timm

    model_path = Path(model_path)
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
        "efficientnet_b0",
        pretrained=False,
        num_classes=len(class_map),
    )
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()

    return model, class_map, name_to_idx


def get_val_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=VAL_MEAN, std=VAL_STD),
        ]
    )


# ── blob / ellipse proposal (public OpenCV only) ──────────────────────────────

def create_blob_detector(min_area=50, max_area=200, min_circularity=0.8):
    """
    Construct a SimpleBlobDetector tuned for bright or dark round-ish blobs.

    This is intentionally conservative: we would rather get more candidates and
    let the classifier filter them, than miss real targets.
    """
    params = cv2.SimpleBlobDetector_Params()

    # Thresholds for binarization inside the detector
    params.minThreshold = 10
    params.maxThreshold = 220

    # Filter by area
    params.filterByArea = True
    params.minArea = float(min_area)
    params.maxArea = float(max_area)

    # Filter by circularity (targets are roughly circular/elliptical)
    params.filterByCircularity = True
    params.minCircularity = float(min_circularity)

    # Filter by inertia ratio (elongated ellipses allowed)
    params.filterByInertia = True
    params.minInertiaRatio = 0.1

    # Filter by convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5

    # Detect both bright-on-dark and dark-on-bright
    params.filterByColor = False

    return cv2.SimpleBlobDetector_create(params)


def detect_candidates_with_blobs(image_bgr, use_adaptive=True):
    """
    Detect candidate ellipses using a public blob detector.

    Returns:
        centers: (N, 2) array of x, y coordinates.
        radii:   (N,)   array of approximate radii in pixels.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if use_adaptive:
        # Improve contrast and local structure
        gray = cv2.equalizeHist(gray)

    detector = create_blob_detector()
    keypoints = detector.detect(gray)

    if not keypoints:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    centers = []
    radii = []
    for kp in keypoints:
        centers.append((kp.pt[0], kp.pt[1]))
        # keypoint.size is diameter; approximate radius = size / 2
        radii.append(max(kp.size / 2.0, 4.0))

    return np.array(centers, dtype=np.float32), np.array(radii, dtype=np.float32)


# ── patch extraction ──────────────────────────────────────────────────────────

def crop_patch(img_bgr, x, y, half_size):
    """Crop a square patch around (x, y) and resize to IMG_SIZE."""
    h, w = img_bgr.shape[:2]
    x0 = max(0, int(x) - half_size)
    y0 = max(0, int(y) - half_size)
    x1 = min(w, int(x) + half_size)
    y1 = min(h, int(y) + half_size)
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

    all_preds = []
    all_confs = []

    with torch.no_grad():
        for i in range(0, len(tensors), BATCH_SIZE):
            batch = torch.stack(tensors[i : i + BATCH_SIZE]).to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            confidences = probs.max(dim=1).values
            all_preds.append(preds)
            all_confs.append(confidences)

    preds = torch.cat(all_preds, dim=0).cpu().numpy()
    confidences = torch.cat(all_confs, dim=0).cpu().numpy()
    return preds, confidences


# ── main processing ───────────────────────────────────────────────────────────

def process_image(img_path, model, transform, device, name_to_idx):
    """
    Full pipeline: load → blob detection → crop → classify → sort results.
    Returns img_rgb and a dict of {class_name: [(x, y, confidence), ...]}.
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Could not load image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    centers, radii = detect_candidates_with_blobs(img_bgr)

    if centers.shape[0] == 0:
        print("  No blobs detected.")
        return img_rgb, {cls: [] for cls in CLASS_STYLE}

    xs = centers[:, 0]
    ys = centers[:, 1]

    # Crop radius: a bit larger than blob radius to capture full target
    half_sizes = np.ceil(radii * 2.0).astype(int)
    half_sizes = np.clip(
        half_sizes,
        24,
        min(img_bgr.shape[0], img_bgr.shape[1]) // 2,
    )

    patches_rgb = []
    for i in range(len(centers)):
        patch = crop_patch(img_bgr, xs[i], ys[i], int(half_sizes[i]))
        if patch is not None:
            patches_rgb.append(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        else:
            patches_rgb.append(
                np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            )

    preds, confs = run_filter(model, patches_rgb, transform, device)

    # Keep all detections that are not "not_target"; delete only false positives.
    not_target_idx = name_to_idx["not_target"]
    kept_indices = (preds != not_target_idx).nonzero()[0]
    deleted_indices = (preds == not_target_idx).nonzero()[0]
    assert (
        len(kept_indices) + len(deleted_indices) == len(preds)
    ), "kept + deleted should equal total"

    # Sort into classes for plotting (Coded / not_coded / not_target).
    results = {cls: [] for cls in CLASS_STYLE}
    idx_to_name = {v: k for k, v in name_to_idx.items()}

    for i, (pred, conf) in enumerate(zip(preds, confs)):
        cls_name = idx_to_name.get(int(pred), "not_target")
        # If confidence is low, treat as not_target regardless.
        if conf < CONFIDENCE_THRESHOLD:
            cls_name = "not_target"
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
    ax.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.imshow(img_rgb)

    # Three groups for clarity: coded, uncoded, false positive
    coded_xy = [(p[0], p[1]) for p in results.get("Coded", [])]
    uncoded_xy = [(p[0], p[1]) for p in results.get("not_coded", [])]
    deleted_xy = [(p[0], p[1]) for p in results.get("not_target", [])]

    total_kept = len(coded_xy) + len(uncoded_xy)
    total_deleted = len(deleted_xy)

    if coded_xy:
        ax.scatter(
            [p[0] for p in coded_xy],
            [p[1] for p in coded_xy],
            c=CLASS_STYLE["Coded"]["color"],
            s=90,
            marker=CLASS_STYLE["Coded"]["marker"],
            edgecolors=CLASS_STYLE["Coded"]["edge"],
            linewidths=2,
            label=f"{CLASS_STYLE['Coded']['label']}  [{len(coded_xy)}]",
            zorder=3,
            alpha=0.85,
        )
    if uncoded_xy:
        ax.scatter(
            [p[0] for p in uncoded_xy],
            [p[1] for p in uncoded_xy],
            c=CLASS_STYLE["not_coded"]["color"],
            s=90,
            marker=CLASS_STYLE["not_coded"]["marker"],
            edgecolors=CLASS_STYLE["not_coded"]["edge"],
            linewidths=2,
            label=f"{CLASS_STYLE['not_coded']['label']}  [{len(uncoded_xy)}]",
            zorder=3,
            alpha=0.85,
        )
    if deleted_xy:
        ax.scatter(
            [p[0] for p in deleted_xy],
            [p[1] for p in deleted_xy],
            c=CLASS_STYLE["not_target"]["color"],
            s=90,
            marker=CLASS_STYLE["not_target"]["marker"],
            linewidths=2,
            label=f"{CLASS_STYLE['not_target']['label']}  [{len(deleted_xy)}]",
            zorder=2,
            alpha=0.85,
        )

    total = total_kept + total_deleted
    ax.set_title(
        f"{title}\n"
        f"Total detections: {total}  |  "
        f"Kept: {total_kept}  |  "
        f"Deleted: {total_deleted}",
        color="white",
        fontsize=11,
        pad=10,
    )

    ax.legend(
        loc="upper right",
        facecolor="#2a2a3e",
        labelcolor="white",
        fontsize=10,
        framealpha=0.85,
    )

    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path, dpi=130, bbox_inches="tight", facecolor="#1a1a2e"
        )
        print(f"    Saved → {save_path}")

    plt.show()
    plt.close()


# ── entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Blob-based detection + CCT Filter — three-class output"
    )
    parser.add_argument(
        "--test_dir",
        default=str(PROJECT_ROOT / "test_data"),
        help="Folder with test images",
    )
    parser.add_argument(
        "--model_path",
        default=str(PROJECT_ROOT / "data" / "results" / "best_model.pth"),
        help="Path to best_model.pth",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save result plots alongside test images",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available (use if GPU runs out of memory)",
    )
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    print("Loading model...")
    model, class_map, name_to_idx = load_filter_model(args.model_path, device)
    transform = get_val_transform()
    print(f"Classes: {class_map}\n")

    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    images = sorted(
        [
            f
            for f in test_dir.iterdir()
            if f.suffix.lower() in exts and "_filtered" not in f.stem
        ]
    )

    if not images:
        print(f"No images found in {test_dir}")
        return

    print(f"Processing {len(images)} image(s) from {test_dir}\n")

    for img_path in images:
        print(f"  {img_path.name} ...", end=" ", flush=True)
        try:
            img_rgb, results = process_image(
                img_path,
                model,
                transform,
                device,
                name_to_idx,
            )

            n_coded = len(results["Coded"])
            n_uncoded = len(results["not_coded"])
            n_fp = len(results["not_target"])
            n_total = n_coded + n_uncoded + n_fp

            print(
                f"{n_total} detections → "
                f"coded: {n_coded}  "
                f"uncoded: {n_uncoded}  "
                f"deleted: {n_fp}"
            )

            save_path = (
                test_dir / f"{img_path.stem}_filtered_blob.png"
                if args.save
                else None
            )

            plot_result(
                img_rgb,
                results,
                title=img_path.name,
                save_path=save_path,
            )

        except Exception as e:
            print(f"ERROR: {e}")
            raise

    print("\nDone.")


if __name__ == "__main__":
    main()

