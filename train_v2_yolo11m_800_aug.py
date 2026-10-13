from ultralytics import YOLO

#model config
model = YOLO("yolo11m.pt")


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

    #augmentation measures
    hsv_h=.02,
    hsv_s=.8,
    hsv_v=.5,

    translate=.15,
    scale=.6,

    fliplr=.5,

    mosaic=1.0,
    mixup=.1,

    #folder setup
    project='runs',
    name="yolo11m_v2_img800_aug",

    #save the result
    save=True,
    plots=True
)

print("\nTraining completed.")
