"""
Photogrammetry Target Classifier — Data Preparation
=====================================================
Input:  3 folders with your raw patches
        data/
          Coded/
          not_coded/
          not_target/

Output: data/prepared/
          Coded/          ← augmented patches, 128x128
          not_coded/
          not_target/
        data/mosaic.png   ← visual sanity check

Usage:
    python prepare_data.py --data_dir ./data --target_size 128 --augment_factor 20
"""

import cv2
import numpy as np
import os
import argparse
import random
import math
from pathlib import Path

# ── augmentations ─────────────────────────────────────────────────────────────

def augment(img, size):
    """
    Full augmentation pipeline.
    Every transform simulates something real that happens in the field.
    """
    # 1. Resize to target size first
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    # 2. Random rotation — targets appear at any angle
    angle = random.uniform(0, 360)
    M = cv2.getRotationMatrix2D((size // 2, size // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (size, size), borderMode=cv2.BORDER_REFLECT)

    # 3. Horizontal / vertical flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    if random.random() < 0.5:
        img = cv2.flip(img, 0)

    # 4. Brightness and contrast — lighting varies a lot in field
    alpha = random.uniform(0.6, 1.5)
    beta  = random.randint(-40, 40)
    img   = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # 5. Perspective warp — targets on curved/angled surfaces
    if random.random() < 0.5:
        margin = int(size * 0.08)
        pts1 = np.float32([[0,0],[size,0],[0,size],[size,size]])
        pts2 = np.float32([
            [random.randint(0, margin),      random.randint(0, margin)],
            [size-random.randint(0,margin),  random.randint(0, margin)],
            [random.randint(0, margin),      size-random.randint(0,margin)],
            [size-random.randint(0,margin),  size-random.randint(0,margin)],
        ])
        M2  = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M2, (size, size),
                                  borderMode=cv2.BORDER_REFLECT)

    # 6. Gaussian blur — focus variation, motion
    if random.random() < 0.4:
        k   = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # 7. Sensor noise
    if random.random() < 0.5:
        noise = np.random.normal(0, random.uniform(3, 15), img.shape).astype(np.int16)
        img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 8. JPEG compression artifact — common when extracting from video
    if random.random() < 0.3:
        quality    = random.randint(45, 80)
        _, enc     = cv2.imencode('.jpg', img,
                                  [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        img        = cv2.imdecode(enc, 1)

    # 9. Slight scale jitter — target occupies different fraction of patch
    if random.random() < 0.4:
        scale  = random.uniform(0.75, 1.0)
        new_s  = int(size * scale)
        pad    = (size - new_s) // 2
        small  = cv2.resize(img, (new_s, new_s))
        canvas = np.zeros_like(img)
        canvas[pad:pad+new_s, pad:pad+new_s] = small
        img    = canvas

    return img


def load_image(path, size):
    """Load, convert to BGR if needed, resize."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img


# ── mosaic ────────────────────────────────────────────────────────────────────

def make_mosaic(prepared_dir, size, n_per_class=8):
    """
    Visual sanity check — shows a grid of samples per class.
    Run this BEFORE training. If something looks wrong here, fix it now.
    """
    classes    = ['Coded', 'not_coded', 'not_target']
    labels     = ['Coded Target', 'Uncoded Target', 'False Positive']
    colors_bgr = [(100, 200, 100), (100, 160, 255), (80, 80, 220)]

    cell    = size
    padding = 6
    label_h = 28
    rows    = []

    for cls, label, color in zip(classes, labels, colors_bgr):
        cls_dir = Path(prepared_dir) / cls
        files   = sorted(cls_dir.glob('*.png'))
        if not files:
            print(f"  WARNING: no files found in {cls_dir}")
            continue

        sample_files = random.sample(files, min(n_per_class, len(files)))
        row_imgs     = []

        for f in sample_files:
            img = cv2.imread(str(f))
            if img is not None:
                row_imgs.append(img)

        if not row_imgs:
            continue

        # pad to n_per_class if fewer
        while len(row_imgs) < n_per_class:
            row_imgs.append(np.zeros((cell, cell, 3), dtype=np.uint8))

        row_strip = np.hstack(row_imgs[:n_per_class])

        # add label bar on left
        bar_w   = 130
        bar     = np.zeros((cell, bar_w, 3), dtype=np.uint8)
        bar[:]  = (40, 40, 40)
        # vertical text via rotation
        text_img = np.zeros((bar_w, cell, 3), dtype=np.uint8)
        text_img[:] = (40, 40, 40)
        cv2.putText(text_img, label, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        bar = cv2.rotate(text_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        row_with_label = np.hstack([bar, row_strip])
        rows.append(row_with_label)

    if not rows:
        print("No images to display.")
        return

    mosaic = np.vstack(rows)

    # title bar
    title_bar      = np.zeros((50, mosaic.shape[1], 3), dtype=np.uint8)
    title_bar[:]   = (25, 25, 40)
    cv2.putText(title_bar,
                "Data Preparation Sanity Check — verify before training",
                (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    mosaic = np.vstack([title_bar, mosaic])

    out_path = Path(prepared_dir).parent / 'mosaic.png'
    cv2.imwrite(str(out_path), mosaic)
    print(f"\n  Mosaic saved → {out_path}")
    print("  Open it and check: does every patch look correct for its class?")
    return mosaic


# ── main ──────────────────────────────────────────────────────────────────────

def prepare(data_dir, target_size, augment_factor):
    data_dir    = Path(data_dir)
    out_dir     = data_dir / 'prepared'
    classes     = ['Coded', 'not_coded', 'not_target']

    print(f"\nData dir   : {data_dir}")
    print(f"Output dir : {out_dir}")
    print(f"Patch size : {target_size}x{target_size}")
    print(f"Augment x  : {augment_factor}")
    print(f"Expected output per class: ~{augment_factor * 29} patches\n")

    for cls in classes:
        src_dir = data_dir / cls
        dst_dir = out_dir  / cls
        dst_dir.mkdir(parents=True, exist_ok=True)

        images = list(src_dir.glob('*'))
        images = [f for f in images
                  if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']]

        if not images:
            print(f"  WARNING: no images found in {src_dir} — skipping")
            continue

        print(f"  {cls}: {len(images)} source images → ", end='', flush=True)

        count = 0
        for img_path in images:
            img = load_image(img_path, target_size)
            if img is None:
                print(f"\n  Could not load {img_path}, skipping")
                continue

            # save original (resized, cleaned)
            out_name = dst_dir / f"{img_path.stem}_orig.png"
            cv2.imwrite(str(out_name), img)
            count += 1

            # save augmented versions
            for i in range(augment_factor - 1):
                aug      = augment(img.copy(), target_size)
                out_name = dst_dir / f"{img_path.stem}_aug{i:03d}.png"
                cv2.imwrite(str(out_name), aug)
                count += 1

        print(f"{count} patches saved")

    print(f"\nBuilding mosaic for visual check...")
    make_mosaic(out_dir, target_size, n_per_class=8)

    # print final counts
    print("\nFinal dataset counts:")
    total = 0
    for cls in classes:
        cls_dir = out_dir / cls
        n       = len(list(cls_dir.glob('*.png')))
        total  += n
        print(f"  {cls:15s}: {n} patches")
    print(f"  {'TOTAL':15s}: {total} patches")
    print("\nDone. Check mosaic.png then run train.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',       default='./data',
                        help='Root folder containing Coded/, not_coded/, not_target/')
    parser.add_argument('--target_size',    default=128, type=int,
                        help='Output patch size in pixels (default: 128)')
    parser.add_argument('--augment_factor', default=20,  type=int,
                        help='How many patches to generate per source image (default: 20)')
    args = parser.parse_args()

    prepare(args.data_dir, args.target_size, args.augment_factor)