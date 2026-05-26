"""
Main inference pipeline: Detection -> Custom ByteTrack -> Visualization
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
import torch

from aerial_guardian.tracking.algorithms import create_tracker
from aerial_guardian.tracking.video_writer import get_video_writer, NullVideoWriter

COLORS = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)

# Automatically choose GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


class AerialGuardianPipeline:
    """End-to-end detection and tracking pipeline for aerial imagery."""

    def __init__(
        self,
        model_path,
        conf_thresh=0.25,
        iou_thresh=0.5,
        img_size=640,
        tracker_cfg="bytetrack.yaml",
        device=device,
    ):
        self.model = YOLO(model_path, task="detect")

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.img_size = img_size
        self.device = device
        
        # Dynamically create tracker based on config name/path (e.g. 'bytetrack.yaml' or 'botsort.yaml')
        self.tracker = create_tracker(tracker_cfg, device=device)
        self.track_history = {}

    def detect(self, frame):
        """Run detection on a single frame."""
        results = self.model(
            frame,
            imgsz=self.img_size,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False,
            device=self.device,
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
        """Draw bounding boxes, IDs, and trajectory tails directly in-place."""
        # Draw directly on the frame in-place to avoid expensive deep copies
        output = frame

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

    def process_video(self, input_path, output_path=None, encoder="opencv"):
        """Process entire video through the pipeline.

        Parameters
        ----------
        input_path: str
            Path to input video.
        output_path: str | None
            Destination file path; if None or empty, output is disabled.
        encoder: str
            Video encoder to use ("opencv", "ffmpeg", or "none").
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")

        # Initialize appropriate video writer (or Null writer for benchmark)
        writer = get_video_writer(output_path, fps, (width, height), encoder=encoder)

        frame_count = 0
        
        # High-precision profiling variables
        total_read_time = 0.0
        total_detect_time = 0.0
        total_tracker_time = 0.0
        total_visualize_time = 0.0
        total_write_time = 0.0
        
        start_time = time.time()

        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            t1 = time.perf_counter()
            if not ret:
                break
            total_read_time += (t1 - t0)

            frame_count += 1

            t_d0 = time.perf_counter()
            detections = self.detect(frame)
            t_d1 = time.perf_counter()
            total_detect_time += (t_d1 - t_d0)

            t_tr0 = time.perf_counter()
            tracks = self.tracker.update(detections, frame)
            t_tr1 = time.perf_counter()
            total_tracker_time += (t_tr1 - t_tr0)

            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0

            # Visualization and writing are optional based on writer type
            if not isinstance(writer, NullVideoWriter):
                t_v0 = time.perf_counter()
                output_frame = self.visualize(frame, tracks, current_fps)
                t_v1 = time.perf_counter()
                total_visualize_time += (t_v1 - t_v0)

                t_w0 = time.perf_counter()
                writer.write(output_frame)
                t_w1 = time.perf_counter()
                total_write_time += (t_w1 - t_w0)

            if frame_count % 50 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({current_fps:.1f} FPS)")

        cap.release()
        # Release the writer (noop for NullVideoWriter)
        writer.release()

        total_time = time.time() - start_time
        avg_fps = frame_count / total_time

        print(f"\nProcessing complete!")
        print(f"Total frames: {frame_count}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        if output_path:
            print(f"Output saved to: {output_path}")
            
        # Print breakdown results
        print("\n" + "=" * 50)
        print(" pipeline latency breakdown ".upper().center(50, "="))
        print("=" * 50)
        print(f"1. Video Read Time:   {total_read_time:.3f}s (Avg: {total_read_time*1000/frame_count:.1f}ms/frame)")
        print(f"2. Model Detect Time: {total_detect_time:.3f}s (Avg: {total_detect_time*1000/frame_count:.1f}ms/frame)")
        print(f"3. Tracker Update:    {total_tracker_time:.3f}s (Avg: {total_tracker_time*1000/frame_count:.1f}ms/frame)")
        if not isinstance(writer, NullVideoWriter):
            print(f"4. Visualization:     {total_visualize_time:.3f}s (Avg: {total_visualize_time*1000/frame_count:.1f}ms/frame)")
            print(f"5. Video Write Time:  {total_write_time:.3f}s (Avg: {total_write_time*1000/frame_count:.1f}ms/frame)")
        else:
            print("4. Visualization:     skipped (benchmark mode)")
            print("5. Video Write Time:  skipped (benchmark mode)")
        print("=" * 50 + "\n")

        return avg_fps


def main():
    parser = argparse.ArgumentParser(description="Aerial Guardian Pipeline")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path (set to '' or 'none' to disable visualization & writing for raw tracking speed)")
    parser.add_argument("--video_encoder", type=str, default="opencv", choices=["opencv", "ffmpeg", "none"], help="Video encoder to use: opencv (default), ffmpeg (NVENC), or none (benchmark mode)")
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

    output_path = args.output
    if not output_path or output_path.strip().lower() in ["none", ""]:
        output_path = None

    pipeline.process_video(args.input, output_path, encoder=args.video_encoder)


if __name__ == "__main__":
    main()
