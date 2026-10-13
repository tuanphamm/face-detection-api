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

    # Augmentation
    fliplr=.5,      #default=0.5, range 0 to 1 (50-50 chance)
    flipud=0,       #default=0, range 0 to 1 (50-50 chance)
    degrees=10,     #default=0, range 0 to 180 (rotation)
    translate=.15,  #default=0.1, range 0 to 1
    scale=.5,       #default=0.5, range 0 to 1
    shear=10,       #default=0, range -180 to 180
    perspective=0,  #default=0, range 0 to .001 (shear and perspective kinda similar, set same time x2 effect)
    mosaic=.8,      #default=1, range 0 to 1 
    mixup=0.1,      #default=0, range 0 to 1 
    hsv_h=.1,       #default=0.015, range 0 to 1
    hsv_s=.5,       #default=0.7, range 0 to 1
    hsv_v=.6,       #default=0.4, range 0 to 1
    copy_paste=.1,  #default=0, range 0 to 1(by chance, 0.5=50%)
   

    #folder setup
    project='runs',
    name="yolo11s_v2_img800_deep_aug",

    #save the result
    save=True,
    plots=True
)

print("\nTraining completed.")
