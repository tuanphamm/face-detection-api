from ultralytics import YOLO

#model config
model = YOLO("yolo11s.pt")


results = model.train(

    #data config
    data="dataset_v2/data.yaml",

    #training config
    epochs=100,
    batch=16,
    imgsz=800,
    patience=15,
    device=0,
    workers=8,
    pretrained=True,

    #folder setup
    project='runs',
    name="yolo11s_v2_img800",

    #save the result
    save=True,
    plots=True
)

print("\nTraining completed.")
