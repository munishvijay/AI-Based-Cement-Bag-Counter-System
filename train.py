"""Train a YOLO cement-bag detector with a licensed YOLO-format dataset."""
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    p=argparse.ArgumentParser(description="Train a YOLO cement-bag detector.")
    p.add_argument("--data",required=True,help="Dataset YAML path")
    p.add_argument("--model",default="yolo11n.pt",help="Starting model")
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--image-size",type=int,default=640)
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--project",default="runs/cement_bag")
    p.add_argument("--name",default="detector")
    a=p.parse_args()
    if not Path(a.data).is_file(): p.error(f"Dataset YAML not found: {a.data}")
    if a.epochs<1 or a.image_size<32 or a.batch<1: p.error("Invalid training parameters")
    YOLO(a.model).train(data=a.data,epochs=a.epochs,imgsz=a.image_size,batch=a.batch,
                        project=a.project,name=a.name)

if __name__=="__main__": main()
