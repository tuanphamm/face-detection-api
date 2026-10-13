import shutil
from pathlib import Path

# Additional data
src_img = Path("add_on_data_v1/images")
src_lbl = Path("add_on_data_v1/labels")

# Existing training set
dst_img = Path("dataset_enriched/images/train")
dst_lbl = Path("dataset_enriched/labels/train")

# Copy images
for f in src_img.iterdir():
    if f.is_file():
        shutil.copy2(f, dst_img)

# Copy labels
for f in src_lbl.iterdir():
    if f.is_file():
        shutil.copy2(f, dst_lbl)

print("✅ Additional images and labels copied successfully.")