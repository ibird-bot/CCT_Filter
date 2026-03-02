## Photogrammetry Target Classifier (CCT_Filter)

This project trains a classifier to distinguish between **coded targets**, **uncoded targets**, and **non‑targets** from image patches. It also includes a **public-only test script** (`test_blob.py`) that uses OpenCV’s blob detector to propose candidate detections on arbitrary images.

---

### 1. Setup

- **Python version**: 3.10+ recommended
- From the repo root:

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1   # PowerShell on Windows
pip install -r requirements.txt
```

The repo is configured to ignore large / private assets:

- `data/` and `test_data/` (your images)
- `venv/` (virtual environment)
- `test.py` (private ellipse-fitting harness)

You are expected to provide your **own images** locally; they are never committed.

---

### 2. Training the classifier (optional)

If you want to retrain the model:

1. Prepare your data as three folders under `data/`:
   - `data/Coded/`
   - `data/not_coded/`
   - `data/not_target/`

2. Run the augmentation / preparation script:

```bash
python augment.py      # or python prepare_data.py if you rename it
```

3. Train the EfficientNet classifier:

```bash
python train.py --data_dir ./data/prepared --epochs 30 --batch_size 32
```

This will create:

- `data/results/best_model.pth`
- `data/results/class_map.json`

These are used by `test_blob.py` for inference.

---

### 3. Using `test_blob.py` with your own images

`test_blob.py` is the **public test harness**. It:

- uses **OpenCV SimpleBlobDetector** to find bright/dark round-ish blobs,
- crops patches around each detection,
- runs the trained classifier on each patch,
- keeps predictions labeled **Coded** or **not_coded**,
- treats **not_target** or **low-confidence** detections as false positives,
- overlays results on the original image:
  - **green circles** = Coded (kept)
  - **blue circles**  = not_coded (kept)
  - **red X**         = not_target / low-confidence (deleted)

#### 3.1. Place your images

Create a folder (default is `test_data/`) and drop your test images there:

```text
CCT_Filter/
  test_blob.py
  data/
    results/
      best_model.pth
      class_map.json
  test_data/
    your_image_01.png
    your_image_02.jpg
    ...
```

Images can be `.png`, `.jpg`, `.jpeg`, or `.bmp`.

#### 3.2. Run the script

From the repo root with the venv activated:

```bash
python test_blob.py
```

By default this will:

- read images from `./test_data`,
- load the model from `./data/results/best_model.pth`,
- open a Matplotlib window per image with the overlayed detections.

You can customize paths and saving:

```bash
python test_blob.py \
  --test_dir ./my_images \
  --model_path ./data/results/best_model.pth \
  --save
```

- `--test_dir` – folder containing your own images.
- `--model_path` – path to a compatible `best_model.pth` and its `class_map.json`.
- `--save` – additionally writes `{image_stem}_filtered_blob.png` next to each input.

#### 3.3. Confidence threshold

Inside `test_blob.py`:

```python
CONFIDENCE_THRESHOLD = 0.85  # only keep if model is sure
```

Any prediction with confidence below this threshold is treated as **not_target**, so it will be drawn as a red X and counted as “deleted”. You can tune this value (e.g. `0.9` for stricter filtering, `0.7` for more permissive).

---

### 4. Notes on privacy and data

- **Your images are never committed**: `data/` and `test_data/` are in `.gitignore`.
- `test_blob.py` uses only **public OpenCV APIs** and your local model; it does **not** depend on any private ellipse-fitting implementation.

