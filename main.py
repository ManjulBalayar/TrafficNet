from detection.detector import Detector

def main_pipeline(sequence_path: str):
    detector = Detector()
    frame_detections = detector.run_sequence(sequence_path)
    return frame_detections

if __name__ == "__main__":
    detections = main_pipeline("data/frames1/")
    print(detections)
