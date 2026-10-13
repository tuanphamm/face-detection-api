from ultralytics import YOLO

model_path = '/home/pmtuan/workspace/10_class_det/runs/detect/runs/yolo11s_baseline/weights/best.pt'

model = YOLO(model_path)

metrics = model.val(
    data = 'dataset_v2/data.yaml',
    split = 'val',
    save_json = True,
    name = 'eval_v2',
    exist_ok = True,
    plots = True,
    verbose = True,

)