from ultralytics import YOLO
import torch

print("=" * 60)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("=" * 60)

model = YOLO("yolo11s.pt")

results = model.train(
    data="data/data.yaml",
    epochs=20,
    imgsz=640,
    batch=64,
    patience=20,
    device=0,
    workers=8,
    pretrained=True,
    project="runs",
    name="yolo11s_baseline",
)

print("Training completed.")

metrics = model.val(
    data="data/data.yaml",
    split="val"
)

print(metrics)