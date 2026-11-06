"""AI cement-bag detection and line-crossing counter. Requires custom-trained YOLO weights."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import cv2
from ultralytics import YOLO


class CrossingCounter:
    """Count each tracker ID once when its centre crosses a horizontal line."""
    def __init__(self, line_y=0.5):
        if not 0 <= line_y <= 1:
            raise ValueError("line_y must be between 0 and 1")
        self.line_y = line_y
        self.previous_side = {}
        self.counted_ids = set()

    def update(self, track_id, center_y, frame_height):
        side = -1 if center_y < self.line_y * frame_height else 1
        previous = self.previous_side.get(track_id)
        self.previous_side[track_id] = side
        if previous is not None and previous != side and track_id not in self.counted_ids:
            self.counted_ids.add(track_id)
            return True
        return False


def parse_source(value):
    return int(value) if value.isdigit() else value


def build_parser():
    p = argparse.ArgumentParser(description="Detect, track and count cement bags crossing a virtual line.")
    p.add_argument("--source", required=True, help="Video path or webcam index, e.g. 0")
    p.add_argument("--model", required=True, help="Custom-trained YOLO .pt weights for cement bags")
    p.add_argument("--class-id", type=int, default=0, help="Bag class index in your model")
    p.add_argument("--line-y", type=float, default=0.5, help="Horizontal line position, normalized 0-1")
    p.add_argument("--confidence", type=float, default=0.35)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--output", help="Optional annotated MP4 output")
    p.add_argument("--csv", help="Optional CSV count-event report")
    p.add_argument("--display", action="store_true", help="Show preview; press q to quit")
    return p


def run(args):
    if not Path(args.model).is_file():
        raise FileNotFoundError(f"Model not found: {args.model}. Train a cement-bag detector; see README.")
    counter = CrossingCounter(args.line_y)
    model = YOLO(args.model)
    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not 0 < fps <= 240:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    video = None
    csv_file = None
    csv_writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        video = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not video.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create output video: {args.output}")
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(args.csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp_utc", "frame", "track_id", "total_count"])
    total = frame_no = 0
    line_px = int(args.line_y * height)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            result = model.track(
                frame, persist=True, tracker=args.tracker, conf=args.confidence,
                imgsz=args.image_size,
                classes=[args.class_id] if args.class_id is not None else None,
                verbose=False,
            )[0]
            cv2.line(frame, (0, line_px), (width, line_px), (0, 255, 255), 2)
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
                coords = boxes.xyxy.cpu().tolist()
                scores = boxes.conf.cpu().tolist()
                for track_id, box, score in zip(ids, coords, scores):
                    x1, y1, x2, y2 = map(int, box)
                    if counter.update(int(track_id), (y1 + y2) / 2, height):
                        total += 1
                        if csv_writer:
                            csv_writer.writerow([datetime.now(timezone.utc).isoformat(),
                                                 frame_no, track_id, total])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 200, 40), 2)
                    cv2.putText(frame, f"Bag {track_id} {score:.2f}", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 200, 40), 2)
            cv2.rectangle(frame, (8, 8), (230, 52), (25, 25, 25), -1)
            cv2.putText(frame, f"Total bags: {total}", (18, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if video:
                video.write(frame)
            if args.display:
                cv2.imshow("Cement Bag Counter - press q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if video:
            video.release()
        if csv_file:
            csv_file.close()
        if args.display:
            cv2.destroyAllWindows()
    print(f"Finished. Counted bags: {total}")
    return total


if __name__ == "__main__":
    run(build_parser().parse_args())
