from ultralytics import YOLO

def train_model(data_yaml, epochs=1000, image_size=640):
    model = YOLO("yolov8n.pt")
    model.train(data=str(data_yaml), epochs=epochs, imgsz=image_size)

if __name__ == "__main__":
    train_model("path/to/data.yaml")
