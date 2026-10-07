from ultralytics import YOLO

#model config
model = YOLO("yolo11m.pt")


results = model.train(

    #data config
    data="dataset_v2/data.yaml",

    #training config
    epochs=100,
    batch=16,
    imgsz=640,
    patience=15,
    device=0,
    workers=8,
    pretrained=True,

    #folder setup
    project='runs',
    name="yolo11m_v2",

    #save the result
    save=True,
    plots=True
)

print("\nTraining completed.")
