import cv2
from ultralytics import YOLO

def predict_image(image_path, model_path):
    model = YOLO(str(model_path))
    result = model.predict(str(image_path))[0]
    cv2.imshow("Prediction", result.plot())
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_image("path/to/image.jpg", "path/to/best.pt")
