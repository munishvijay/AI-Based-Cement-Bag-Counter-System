import cv2
import os

def extract_frames(video_path, output_dir, frame_interval=50):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    saved_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_interval == 0:
                saved_count += 1
                image_name = f"{saved_count:05d}.jpg"
                cv2.imwrite(os.path.join(output_dir, image_name), frame)
    finally:
        cap.release()

if __name__ == "__main__":
    extract_frames("path/to/input_video.mp4", "path/to/output_frames")
