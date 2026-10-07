from ultralytics import YOLO


class InferenceEngine:

    def __init__(self, model_path):

        print("Loading model...")

        self.model = YOLO(model_path)

        print("Model loaded")

    def predict_raw(self, image_path):
        results = self.model(image_path)
        return results

    def predict(self, image_path, conf_threshold=0.5):

        results = self.model(image_path)

        result = results[0]

        detections = []

        boxes = result.boxes

        for cls_id, conf, bbox in zip(
            boxes.cls,
            boxes.conf,
            boxes.xyxy,
        ):

            confidence = float(conf)

            if confidence < conf_threshold:
                continue

            class_id = int(cls_id)

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": round(confidence, 4),
                    "bbox": [
                        round(float(x), 2)
                        for x in bbox.tolist()
                    ]
                }
            )

        return {
            "image_path": image_path,
            "num_detections": len(detections),
            "detections": detections
        }


if __name__ == "__main__":


    engine = InferenceEngine(
        "/home/pmtuan/workspace/10_class_det/runs/detect/runs/yolo11s_baseline/weights/best.pt"
    )

    prediction = engine.predict(
        "test3.jpg"
    )

    print(prediction)