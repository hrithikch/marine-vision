import os
import glob
import scipy.io
import cv2

# =========== Configuration ===========
# Base folder where the rectified annotations live:
ANNOT_ROOT = "data/raw/modd/raw/MODD2_annotations_v2_rectified/annotationsV2_rectified"

# Base folder where the rectified video frames live:
IMG_ROOT = "data/raw/modd/raw/MODD2_video_data_rectified/video_data"

# Where to write YOLO images & labels:
YOLO_IMG_DIR   = "data/images/train/modd2"
YOLO_LABEL_DIR = "data/labels/train/modd2"
# =====================================

# Create output dirs if missing
os.makedirs(YOLO_IMG_DIR, exist_ok=True)
os.makedirs(YOLO_LABEL_DIR, exist_ok=True)

print(f"Looking for annotation root at: {ANNOT_ROOT}")
if not os.path.isdir(ANNOT_ROOT):
    raise FileNotFoundError(f"Annotation folder not found: {ANNOT_ROOT}")

# List the sequence folders directly under ANNOT_ROOT
seq_dirs = [d for d in os.listdir(ANNOT_ROOT) if os.path.isdir(os.path.join(ANNOT_ROOT, d))]
print(f"Found {len(seq_dirs)} sequences: {seq_dirs[:5]}{'...' if len(seq_dirs)>5 else ''}")

# Glob for all .mat files: each sequence has a ground_truth subfolder
pattern = os.path.join(ANNOT_ROOT, "*", "ground_truth", "*.mat")
mat_files = glob.glob(pattern)
print(f"Found {len(mat_files)} annotation files with pattern {pattern}")

count = 0
for mat_path in mat_files:
    # E.g. mat_path = ".../kope67-00-00004500-00005050/ground_truth/00004701L.mat"
    seq_name = os.path.basename(os.path.dirname(os.path.dirname(mat_path)))
    base = os.path.splitext(os.path.basename(mat_path))[0]  # e.g. "00004701L"

    # Load the .mat and inspect
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if 'annotations' not in mat:
        print(f"  ⚠️  No 'annotations' key in {mat_path}, skipping")
        continue
    ann = mat['annotations']

    # Extract obstacles array
    obs = ann.obstacles
    # Debug: print type and shape of obstacles
    print(f"Frame {base}: obstacles type = {type(obs)}, shape = {getattr(obs, 'shape', None)}")

    # Skip if no obstacles
    if not hasattr(obs, "shape") or obs.size == 0:
        continue

    # Build image path
    img_path = os.path.join(IMG_ROOT, seq_name, "framesRectified", base + ".jpg")
    if not os.path.exists(img_path):
        print(f"  ⚠️  Missing image for {base}: expected at {img_path}")
        continue

    # Read image to get its size
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Prepare YOLO label file
    label_path = os.path.join(YOLO_LABEL_DIR, base + ".txt")
    with open(label_path, "w") as f:
        # obs is an N×4 array: [x_center_px, y_center_px, width_px, height_px]
        for row in obs.reshape(-1, 4):
            x_c_px, y_c_px, bw_px, bh_px = row
            # Normalize to [0,1]
            x_c = float(x_c_px) / w
            y_c = float(y_c_px) / h
            bw  = float(bw_px)  / w
            bh  = float(bh_px)  / h
            # Write class 0 (obstacle) + normalized box
            f.write(f"0 {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

    # Copy the image into the YOLO folder (if not present)
    dst_img = os.path.join(YOLO_IMG_DIR, base + ".jpg")
    if not os.path.exists(dst_img):
        cv2.imwrite(dst_img, img)

    count += 1

print(f"✅  Converted {count} frames with obstacles into YOLO format")
