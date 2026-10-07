from ultralytics import YOLO


model = YOLO("yolo11s.pt")

results = model.train(
    data="dataset_enriched/data.yaml",
    epochs=20,
    imgsz=640,
    batch=64,
    patience=20,
    device=0,
    workers=8,
    pretrained=True,
    project="runs",
    name="yolo11s_enrichment",
)

print("Training completed.")

metrics = model.val(
    data="dataset_enriched/data.yaml",
    split="val"
)

print(metrics)