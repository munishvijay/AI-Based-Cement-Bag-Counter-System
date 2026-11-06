import cv2
from ultralytics import YOLO

def run_video(video_path, model_path):
    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video source")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = model.predict(frame)[0]
            cv2.imshow("YOLO video test", result.plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_video("path/to/input_video.mp4", "path/to/best.pt")
