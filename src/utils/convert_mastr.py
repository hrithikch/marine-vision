import os
import glob
import cv2

# All paths are relative to the repo root. Use forward-slashes so Python handles them cross-platform.
RAW_IMG_DIR    = "data/raw/mastr/MaSTr1325_images_512x384"
RAW_MASK_DIR   = "data/raw/mastr/MaSTr1325_masks_512x384"
YOLO_IMG_DIR   = "data/images/train"
YOLO_LABEL_DIR = "data/labels/train"

# Ensure outputs dirs exist
os.makedirs(YOLO_IMG_DIR, exist_ok=True)
os.makedirs(YOLO_LABEL_DIR, exist_ok=True)

mask_paths = glob.glob(os.path.join(RAW_MASK_DIR, "*.png"))
print(f"Found {len(mask_paths)} masks in {RAW_MASK_DIR}")

count = 0
for mask_path in mask_paths:
    mask_name = os.path.basename(mask_path)
    name, _ = os.path.splitext(mask_name)    # e.g. '0001m'

    # If mask names end with an extra 'm' suffix, drop it to match '0001.jpg'
    if name.endswith("m"):
        img_base = name[:-1]
    else:
        img_base = name

    # Try to find the image file (JPEG)
    img_path = os.path.join(RAW_IMG_DIR, img_base + ".jpg")
    if not os.path.exists(img_path):
        print(f"  ⚠️  IMAGE MISSING for mask: {mask_name}")
        continue

    # Read image and mask
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]

    # Find all non-zero mask pixels
    ys, xs = mask.nonzero()
    if len(xs) == 0:
        print(f"  ⚠️  EMPTY MASK for {mask_name}, skipping")
        continue

    # Compute bounding box in pixel coords
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Convert to YOLO format (normalized center x, center y, width, height)
    x_center = (x_min + x_max) / 2 / w
    y_center = (y_min + y_max) / 2 / h
    bw = (x_max - x_min) / w
    bh = (y_max - y_min) / h

    # Write the label file next to the image in YOLO folder
    label_path = os.path.join(YOLO_LABEL_DIR, img_base + ".txt") #label_path = os.path.join(YOLO_LABEL_DIR, img_base + ".txt")
    with open(label_path, "w") as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

    # Move (or rename) the image into our YOLO folder
    dst_img = os.path.join(YOLO_IMG_DIR, os.path.basename(img_path)) #dst_img = os.path.join(YOLO_IMG_DIR, os.path.basename(img_path))
    if not os.path.exists(dst_img):
        os.replace(img_path, dst_img)

    count += 1

print(f"✅  Converted and moved {count} image+label pairs to {YOLO_IMG_DIR}")
