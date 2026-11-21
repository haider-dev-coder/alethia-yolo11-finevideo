from ultralytics import YOLO

TARGET_CLASSES = {0: "person", 2: "car", 16: "dog", 63: "laptop", 67: "cell phone"}


class ObjectDetector:
    def __init__(self, model_path="yolo11l.pt", device=0, half=False):
        self.model = YOLO(model_path)
        self.device = device
        self.half = half

    def detect_batch(self, frames):
        results = self.model.predict(
            source=frames, imgsz=640, conf=0.4, iou=0.7,
            device=self.device, half=self.half, verbose=False,
        )
        out = []
        for r in results:
            objs, people = set(), 0
            if r.boxes is not None and len(r.boxes):
                for c in r.boxes.cls.tolist():
                    c = int(c)
                    name = TARGET_CLASSES.get(c)
                    if name is None:
                        continue
                    objs.add(name)
                    if c == 0:
                        people += 1
            out.append({"objects": objs, "people": people})
        return out