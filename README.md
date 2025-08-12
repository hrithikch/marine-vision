# Marine Vision — Quick Start & Guide

This repo is a clean, “batteries-included” setup for maritime object detection and tracking using Ultralytics YOLOv8. It supports:

* **Images + labels** in YOLO format (e.g., MODD2/MODS)
* **Masked videos** (boats visible, background blacked out)
* **One-liner** CLI inference and tracking
* A small **Python script** for batch inference with more control

> If you’re on Windows, use the **Anaconda Prompt** for all commands.

---

## 1) Environment Setup

### Option A — Create the `marine` env (recommended)

```powershell
conda create -n marine python=3.10 -y
conda activate marine
conda install -c pytorch pytorch torchvision cpuonly -y
conda install -c conda-forge opencv scipy -y
pip install ultralytics
```

> GPU machine later? Replace `cpuonly` with an appropriate CUDA build from the PyTorch site and re-install.

### Option B — Use `environment.yml` (if present)

```powershell
conda env create -f environment.yml
conda activate marine
```

---

## 2) Project Layout

```
marine-vision/
├─ data/
│  ├─ images/
│  │  ├─ train/
│  │  │  └─ modd2/                # images (.jpg)
│  │  └─ val/                     # optional validation images
│  ├─ labels/
│  │  ├─ train/
│  │  │  └─ modd2/                # YOLO labels (.txt) matching image basenames
│  │  └─ val/
│  └─ raw/
│     └─ seg_vid/
│        └─ crop11.mp4            # example segmented video (boats visible)
├─ src/
│  └─ utils/
│     ├─ convert_modd2.py         # MODD2 .mat → YOLO labels converter
│     └─ convert_mastr.py         # (mask → box) demo; see caveats below
├─ sample_and_infer.py            # random sample inference for images
├─ data
│  └─ modd2.yaml                  # dataset config (paths, classes) for training/val
└─ README.md
```

**YOLO label format:** for each image `0001.jpg`, there is `0001.txt` with lines:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are **normalized** to `[0,1]`. For single-class “boat,” class\_id is `0`.

---

## 3) Getting Data Into the Right Place

### A) If you already have YOLO-ready data (images + labels)

* Put images under `data/images/train/<any_folder>` and labels under `data/labels/train/<same_folder>`.
* For validation, mirror the structure under `val/`.

### B) If you’re converting **MODD2** annotations

`src/utils/convert_modd2.py` reads MODD2 **rectified** `.mat` files and copies frames into YOLO layout while writing labels.

Edit paths at the top if needed, then:

```powershell
conda activate marine
python src/utils/convert_modd2.py
```

It will print how many frames were converted and populate:

```
data/images/train/modd2/*.jpg
data/labels/train/modd2/*.txt
```

> **Sanity check:** For every `*.jpg` there should be a matching `*.txt` in `labels/train/...` with the same base name.

### C) About MaSTr1325 (segmentation → boxes)

`convert_mastr.py` shows a simple mask→box approach, but **MaSTr labels “object=0” (boats + land)**, so naïve conversion can produce giant or incorrect boxes. Prefer MODD2/MODS (with real boxes) for detection, or add connected components and class filtering if you adapt this.

---

## 4) Inference Options

### Option 1 — **Python** batch inference on images

Use this when you want more control (e.g., filtering tiny boxes). It randomly samples images and writes annotated results.

```powershell
conda activate marine
python sample_and_infer.py
```

* Results go to `runs/sample_predict/<run_name>/`.
* Tweak thresholds or filters inside the script as needed.

### Option 2 — **CLI** video inference (annotate an MP4)

This is the fastest path to “boxed video out”:

```powershell
conda activate marine
yolo predict model=yolov8s.pt source=data\raw\seg_vid\crop11.mp4 save=True conf=0.1 project=runs\predict name=boats
```

* **model=** can be `yolov8s.pt` (COCO) or your fine-tuned weights (e.g., `runs\fine_tune\modd2_ft\weights\best.pt`).
* **save=True** writes an annotated video.
* **save\_txt=True** adds per-frame detections (YOLO format) next to outputs.
* You can also set `imgsz=640` (or smaller for speed), `iou=0.5`, `classes=0` (if your model has multiple classes).

### Option 3 — **CLI tracking** (ByteTrack / BoTSORT)

To get **persistent IDs across frames**:

```powershell
conda activate marine
# (optional) copy tracker yaml locally once:
copy %CONDA_PREFIX%\Lib\site-packages\ultralytics\tracker\botsort.yaml .

yolo track model=yolov8s.pt `
  source=data\raw\seg_vid\crop11.mp4 `
  tracker=botsort.yaml `
  save=True save_txt=True `
  conf=0.1 project=runs\track name=boats
```

Outputs:

* Annotated video with IDs: `runs\track\boats\...mp4`
* Per-frame label files in `runs\track\boats\labels\` (IDs included)

---

## 5) Training / Fine-Tuning (Optional)

If you want a **boat-only** model:

`data/modd2.yaml`

```yaml
path: data
train: images/train/modd2
val:   images/train/modd2   # or your own val split
nc:    1
names: ['boat']
```

Train (CPU demo with YOLOv8 small):

```powershell
yolo train data=data/modd2.yaml model=yolov8s.pt epochs=10 batch=8 device=cpu project=runs\fine_tune name=modd2_ft
```

Then infer with the new weights:

```powershell
yolo predict model=runs\fine_tune\modd2_ft\weights\best.pt source=data\raw\seg_vid\crop11.mp4 save=True conf=0.1
```

Evaluate (mAP on val set):

```powershell
yolo val data=data/modd2.yaml model=runs\fine_tune\modd2_ft\weights\best.pt
```

---

## 6) Performance Tips

* **imgsz=** smaller is faster (e.g., `imgsz=512` or `imgsz=384`)
* **batch=** process frames in batches when predicting on image folders
* **device=0** use GPU when available; add `half=True` for FP16
* Export to **TensorRT/ONNX** for deployment:

  ```powershell
  yolo export model=best.pt format=onnx
  yolo export model=best.pt format=engine  # TensorRT (on supported machines)
  ```

---

## 7) Troubleshooting

* **No detections** on tough frames → lower `conf` (e.g., `conf=0.1`), try a larger backbone (`yolov8m.pt`), or fine-tune on MODD2.
* **Corner/edge false positives** on masked video → try `conf`↑, `iou`↑, or a simple area filter in Python.
* **Ultralytics settings** live in `%APPDATA%\Ultralytics\settings.json`.
* **Windows line breaks**: on PowerShell you can split lines with backticks (`` ` ``), not backslashes.

---

## 8) What’s What (Repo Components)

* **`src/utils/convert_modd2.py`**: turns MODD2 rectified `.mat` annotations into YOLO labels + copies frames.
* **`sample_and_infer.py`**: random sample of images → runs inference → saves annotated outputs.
* **`data/*.yaml`**: dataset configs for YOLO training/validation (not used during `predict`/`track`).
* **`runs/…`**: all outputs (predictions, training runs, tracking) are versioned here by Ultralytics.
* **`botsort.yaml` / `bytetrack.yaml`**: tracker configs (use either, both are included with Ultralytics).

---

## 9) Roadmap (Optional)

* Integrate segmentation → crop/mask before detection → fewer false positives.
* Use tracker outputs (IDs + boxes) to drive stereo depth on **only** ROIs → speed + accuracy win.
* Compute trajectories, headings, speeds from tracked positions + depth.

---

That’s it! If your data is in the right place (see Section 3) you can immediately:

```powershell
# Quick video annotation
yolo predict model=yolov8s.pt source=data\raw\seg_vid\crop11.mp4 save=True conf=0.1

# Tracking with IDs
yolo track model=yolov8s.pt source=data\raw\seg_vid\crop11.mp4 tracker=botsort.yaml save=True save_txt=True conf=0.1
```
