from ultralytics import YOLO
import os
from config import CONF_THRESHOLD

class Detector:

    def __init__(self, model_path="yolov8n.pt", vehicle_classes=None):
        self.model = YOLO(model_path)
        self.vehicle_classes = vehicle_classes if vehicle_classes else [2, 3, 5, 7]

    def run_sequence(self, sequence_path):
        frame_detections = []
        image_files = sorted(
            f for f in os.listdir(sequence_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        for frame_id, image in enumerate(image_files, start=1):
            image_path = sequence_path + "/" + image
            detections = self.detect_frame(image_path)
            frame_detections.append({
                "frame_id": frame_id,
                "detections": detections
            })
        return frame_detections

    def detect_frame(self, image_path):
        """Run detection on an image file path."""
        return self._parse_results(self.model(image_path, verbose=False))

    def detect_frame_bgr(self, frame):
        """Run detection on an already-loaded BGR numpy array."""
        return self._parse_results(self.model(frame, verbose=False))

    def _parse_results(self, results):
        detections = []
        for result in results:
            boxes   = result.boxes.xyxy
            confs   = result.boxes.conf
            classes = result.boxes.cls

            for box, conf, cls in zip(boxes, confs, classes):
                class_id = int(cls)
                if class_id in self.vehicle_classes and float(conf) >= CONF_THRESHOLD:
                    detections.append({
                        "bbox":       box.cpu().tolist(),
                        "conf":       float(conf.cpu()),
                        "class_id":   class_id,
                        "class_name": self.model.names[class_id],
                    })
        return detections
