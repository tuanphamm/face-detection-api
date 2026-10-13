from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

# =========================
# Config
# =========================
DATA_DIR = Path("data")

IMG_DIR = DATA_DIR / "images"
LBL_DIR = DATA_DIR / "labels"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# =========================
# Collect image files
# =========================
image_files = []

for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
    image_files.extend(list(IMG_DIR.glob(ext)))

image_files = sorted(image_files)

print(f"Found {len(image_files)} images")

# =========================
# Split dataset
# =========================
train_files, temp_files = train_test_split(
    image_files,
    test_size=(VAL_RATIO + TEST_RATIO),
    random_state=RANDOM_SEED,
    shuffle=True
)

val_files, test_files = train_test_split(
    temp_files,
    test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    random_state=RANDOM_SEED,
    shuffle=True
)

print(f"Train: {len(train_files)}")
print(f"Val:   {len(val_files)}")
print(f"Test:  {len(test_files)}")

# =========================
# Create directories
# =========================
for split in ["train", "val", "test"]:
    (IMG_DIR / split).mkdir(parents=True, exist_ok=True)
    (LBL_DIR / split).mkdir(parents=True, exist_ok=True)

# =========================
# Copy image + label pairs
# =========================
def copy_files(file_list, split_name):
    copied = 0

    for img_path in file_list:

        label_path = LBL_DIR / f"{img_path.stem}.txt"

        if not label_path.exists():
            print(f"WARNING: Missing label for {img_path.name}")
            continue

        shutil.copy2(
            img_path,
            IMG_DIR / split_name / img_path.name
        )

        shutil.copy2(
            label_path,
            LBL_DIR / split_name / label_path.name
        )

        copied += 1

    print(f"{split_name}: copied {copied} samples")

copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print("\nDone. This is a random split")