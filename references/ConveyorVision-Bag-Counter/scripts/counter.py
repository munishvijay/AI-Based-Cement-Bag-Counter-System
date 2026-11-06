import cv2
import numpy as np
from typing import List
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker, STrack
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from onemetric.cv.utils.iou import box_iou_batch

class BYTETrackerArgs:
    track_thresh = 0.50
    track_buffer = 1000
    match_thresh = 0.9
    aspect_ratio_thresh = 3.0
    min_box_area = 1.0
    mot20 = False

class VideoCounter:
    def __init__(self, video_path, model_path):
        self.cap = cv2.VideoCapture(str(video_path))
        self.model = YOLO(str(model_path))
        self.byte_tracker = BYTETracker(BYTETrackerArgs())
        self.line_counter = LineCounter(start=Point(561, 517), end=Point(906, 510))

    def detections2boxes(self, detections: Detections) -> np.ndarray:
        return np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))

    def tracks2boxes(self, tracks: List[STrack]) -> np.ndarray:
        return np.array([track.tlbr for track in tracks], dtype=float)

    def match_detections_with_tracks(self, detections, tracks):
        if not np.any(detections.xyxy) or not tracks:
            return np.empty((0,))
        iou = box_iou_batch(self.tracks2boxes(tracks), detections.xyxy)
        track_to_detection = np.argmax(iou, axis=1)
        ids = [None] * len(detections)
        for track_index, detection_index in enumerate(track_to_detection):
            if iou[track_index, detection_index] != 0:
                ids[detection_index] = tracks[track_index].track_id
        return ids

    def deploy_model(self, frame):
        results = self.model.predict(frame)
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int))
        tracks = self.byte_tracker.update(
            output_results=self.detections2boxes(detections),
            img_info=frame.shape, img_size=frame.shape)
        detections.tracker_id = np.array(self.match_detections_with_tracks(detections, tracks))
        labels = [f"#{track_id} {self.model.names[class_id]} {confidence:.2f}"
                  for _, confidence, class_id, track_id in detections]
        self.line_counter.update(detections=detections)
        annotated = BoxAnnotator(color=ColorPalette(), thickness=4,
                                 text_thickness=4, text_scale=2).annotate(
                                     frame=frame, detections=detections, labels=labels)
        LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2).annotate(
            frame=annotated, line_counter=self.line_counter)
        return annotated

    def run(self):
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video source")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            cv2.imshow("ConveyorVision", self.deploy_model(frame))
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    VideoCounter("path/to/input_video.mp4", "path/to/best.pt").run()
