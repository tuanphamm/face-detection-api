from pathlib import Path
from collections import Counter, defaultdict

LABEL_ROOT = Path("/home/pmtuan/workspace/FQA/det_classi/det_3class/labels")

class_names = {
    0: "mouth_closed",
    1: "mouth_open",
    2: "mask"
}

overall_objects = Counter()
overall_images = defaultdict(set)

for split in ["train", "val", "test"]:
    split_dir = LABEL_ROOT / split

    object_counter = Counter()
    image_counter = defaultdict(set)

    for label_file in split_dir.glob("*.txt"):

        with open(label_file, "r") as f:
            classes_in_image = set()

            for line in f:
                line = line.strip()
                if not line:
                    continue

                class_id = int(line.split()[0])

                if class_id in class_names:
                    class_name = class_names[class_id]

                    object_counter[class_name] += 1
                    overall_objects[class_name] += 1

                    classes_in_image.add(class_name)

            # count image once per class
            for cls in classes_in_image:
                image_counter[cls].add(label_file.stem)
                overall_images[cls].add(label_file.stem)

    print(f"\n{'='*50}")
    print(f"SPLIT: {split.upper()}")
    print(f"{'='*50}")

    print("\nObject count:")
    for cls in class_names.values():
        print(f"{cls:<15}: {object_counter[cls]}")

    print("\nImages containing class:")
    for cls in class_names.values():
        print(f"{cls:<15}: {len(image_counter[cls])}")

print(f"\n{'='*50}")
print("OVERALL")
print(f"{'='*50}")

print("\nObject count:")
for cls in class_names.values():
    print(f"{cls:<15}: {overall_objects[cls]}")

total = sum(overall_objects.values())

print("\nPercentage:")
for cls in class_names.values():
    pct = overall_objects[cls] / total * 100 if total else 0
    print(f"{cls:<15}: {pct:.2f}%")

print("\nImages containing class:")
for cls in class_names.values():
    print(f"{cls:<15}: {len(overall_images[cls])}")