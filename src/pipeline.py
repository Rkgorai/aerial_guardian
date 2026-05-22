"""
Main inference pipeline: Detection -> Custom ByteTrack -> Visualization
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import sys
from pathlib import Path

# Add src and parent directories to system path for flexible loading
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
root_dir = src_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from trackers import create_tracker


COLORS = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)


class AerialGuardianPipeline:
    """End-to-end detection and tracking pipeline for aerial imagery."""

    def __init__(
        self,
        model_path,
        conf_thresh=0.25,
        iou_thresh=0.5,
        img_size=640,
        tracker_cfg="bytetrack.yaml",
    ):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        
        # Dynamically create tracker based on config name/path (e.g. 'bytetrack.yaml' or 'botsort.yaml')
        self.tracker = create_tracker(tracker_cfg, device="cpu")
        self.track_history = {}

    def detect(self, frame):
        """Run detection on a single frame."""
        results = self.model(
            frame,
            imgsz=self.img_size,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False,
        )

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            w = x2 - x1
            h = y2 - y1
            x_center = x1 + w / 2
            y_center = y1 + h / 2
            detections.append([x_center, y_center, w, h, conf])

        return detections

    def visualize(self, frame, tracks, fps):
        """Draw bounding boxes, IDs, and trajectory tails."""
        output = frame.copy()

        for track in tracks:
            x, y, w, h, track_id, conf = track
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            cx = int(x)
            cy = int(y)

            color = COLORS[int(track_id) % len(COLORS)].tolist()

            # Bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Trajectory tail
            if int(track_id) not in self.track_history:
                self.track_history[int(track_id)] = []
            self.track_history[int(track_id)].append((cx, cy))
            if len(self.track_history[int(track_id)]) > 30:
                self.track_history[int(track_id)] = self.track_history[int(track_id)][-30:]

            points = self.track_history[int(track_id)]
            for i in range(1, len(points)):
                alpha = i / len(points)
                thickness = max(1, int(3 * alpha))
                cv2.line(output, points[i - 1], points[i], color, thickness)

            # ID label
            label = f"ID:{int(track_id)} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                output,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                output, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
            )

        # FPS and track count
        cv2.putText(
            output, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
        )
        cv2.putText(
            output, f"Tracks: {len(tracks)}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
        )

        return output

    def process_video(self, input_path, output_path):
        """Process entire video through the pipeline."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            detections = self.detect(frame)
            tracks = self.tracker.update(detections, frame)

            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0

            output_frame = self.visualize(frame, tracks, current_fps)
            out.write(output_frame)

            if frame_count % 50 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({current_fps:.1f} FPS)")

        cap.release()
        out.release()

        total_time = time.time() - start_time
        avg_fps = frame_count / total_time

        print(f"\nProcessing complete!")
        print(f"Total frames: {frame_count}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Output saved to: {output_path}")

        return avg_fps


def main():
    parser = argparse.ArgumentParser(description="Aerial Guardian Pipeline")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker config name or path")
    args = parser.parse_args()

    pipeline = AerialGuardianPipeline(
        model_path=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        img_size=args.imgsz,
        tracker_cfg=args.tracker,
    )

    pipeline.process_video(args.input, args.output)


if __name__ == "__main__":
    main()
