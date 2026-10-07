from pathlib import Path
import shutil
import numpy as np

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# ==================================================
# CONFIG
# ==================================================

DATASET_DIR = Path("dataset_v2")

IMG_DIR = DATASET_DIR / "images"
LBL_DIR = DATASET_DIR / "labels"

N_CLASSES = 10

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# ==================================================
# LOAD IMAGES
# ==================================================

image_files = []

for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
    image_files.extend(list(IMG_DIR.glob(ext)))

image_files = sorted(image_files)

print(f"Found {len(image_files)} images")

# ==================================================
# BUILD MULTI-LABEL MATRIX
# ==================================================

X = []
Y = []

for img_path in image_files:

    label_path = LBL_DIR / f"{img_path.stem}.txt"

    if not label_path.exists():
        print(f"Missing label: {img_path.name}")
        continue

    vector = np.zeros(N_CLASSES, dtype=int)

    with open(label_path, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            cls = int(line.split()[0])

            vector[cls] = 1

    X.append(img_path)
    Y.append(vector)

X = np.array(X)
Y = np.array(Y)

print("Dataset matrix created")
print("Shape:", Y.shape)

# ==================================================
# TRAIN / TEMP
# ==================================================

msss = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=(VAL_RATIO + TEST_RATIO),
    random_state=RANDOM_SEED
)

train_idx, temp_idx = next(msss.split(X, Y))

train_files = X[train_idx]
temp_files = X[temp_idx]

Y_temp = Y[temp_idx]

# ==================================================
# VAL / TEST
# ==================================================

msss2 = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=0.5,
    random_state=RANDOM_SEED
)

val_idx, test_idx = next(
    msss2.split(temp_files, Y_temp)
)

val_files = temp_files[val_idx]
test_files = temp_files[test_idx]

print("\nSplit Sizes")
print("-----------")
print("Train:", len(train_files))
print("Val  :", len(val_files))
print("Test :", len(test_files))

# ==================================================
# CREATE FOLDERS
# ==================================================

for split in ["train", "val", "test"]:

    (IMG_DIR / split).mkdir(exist_ok=True)
    (LBL_DIR / split).mkdir(exist_ok=True)

# ==================================================
# COPY FILES
# ==================================================

def copy_split(files, split_name):

    count = 0

    for img_path in files:

        label_path = LBL_DIR / f"{img_path.stem}.txt"

        shutil.copy2(
            img_path,
            IMG_DIR / split_name / img_path.name
        )

        shutil.copy2(
            label_path,
            LBL_DIR / split_name / label_path.name
        )

        count += 1

    print(f"{split_name}: {count}")

copy_split(train_files, "train")
copy_split(val_files, "val")
copy_split(test_files, "test")

# ==================================================
# CLASS DISTRIBUTION REPORT
# ==================================================

CLASS_NAMES = [
    "eye_closed",
    "eye_occluded",
    "eye_open",
    "hat",
    "mask",
    "mouth_closed",
    "mouth_occluded",
    "mouth_open",
    "nose",
    "sunglasses"
]

def count_boxes(files):

    counts = np.zeros(N_CLASSES, dtype=int)

    for img_path in files:

        label_path = LBL_DIR / f"{img_path.stem}.txt"

        with open(label_path, "r") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                cls = int(line.split()[0])
                counts[cls] += 1

    return counts

train_counts = count_boxes(train_files)
val_counts = count_boxes(val_files)
test_counts = count_boxes(test_files)



print("\n")
print("=" * 80)
print("CLASS DISTRIBUTION")
print("=" * 80)

print(
    f"{'Class':<20}"
    f"{'Total':<8}"
    f"{'Train':<8}"
    f"{'Val':<8}"
    f"{'Test':<8}"
)

print("-" * 52)

for i, cls_name in enumerate(CLASS_NAMES):

    total = (
        train_counts[i]
        + val_counts[i]
        + test_counts[i]
    )

    print(
        f"{cls_name:<20}"
        f"{total:<8}"
        f"{train_counts[i]:<8}"
        f"{val_counts[i]:<8}"
        f"{test_counts[i]:<8}"
    )
