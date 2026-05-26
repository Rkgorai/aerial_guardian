"""
Benchmark script for comparing multiple YOLO models on VisDrone validation set.

Usage:
    python benchmark.py --model1 path/to/model1.pt --model2 path/to/model2.pt

Or edit MODEL_PATHS list below for quick testing.
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


# ============================================================
# Configuration - Edit these paths for your models
# ============================================================
MODEL_PATHS = [
    # "path/to/model1.pt",
    # "path/to/model2.pt",
]

DATASET_YAML = "yolo_dataset/dataset.yaml"
IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
FPS_FRAMES = 200  # Number of frames for FPS benchmark


def validate_model(model_path, data_yaml, imgsz, conf, iou):
    """Run YOLO validation and return metrics."""
    print(f"\n{'='*60}")
    print(f"Validating: {model_path}")
    print(f"{'='*60}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(model_path)

    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=True,
        split="val",
        device=device,
    )

    metrics = {
        "model": model_path,
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "parameters": model.info()["parameters"],
    }

    print(f"\nResults for {model_path}:")
    print(f"  mAP50:     {metrics['mAP50']:.4f}")
    print(f"  mAP50-95:  {metrics['mAP50_95']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Params:    {metrics['parameters']:,}")

    return metrics


def benchmark_fps(model_path, imgsz, num_frames=FPS_FRAMES):
    """Benchmark inference speed."""
    print(f"\n{'='*60}")
    print(f"FPS Benchmark: {model_path}")
    print(f"{'='*60}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(model_path)

    # Find a sample image for benchmarking
    images_dir = Path("yolo_dataset/images/val")
    sample_images = list(images_dir.glob("**/*.jpg"))

    if not sample_images:
        print("No images found for FPS benchmark!")
        return None

    # Use first image, cycle through if needed
    img = cv2.imread(str(sample_images[0]))
    if img is None:
        print("Failed to load sample image!")
        return None

    print(f"Benchmarking with {num_frames} frames (device: {device})...")

    # Warmup
    for _ in range(10):
        model(img, imgsz=imgsz, verbose=False, conf=CONF_THRESHOLD, device=device)

    # Timed inference
    times = []
    for _ in range(num_frames):
        start = time.perf_counter()
        model(img, imgsz=imgsz, verbose=False, conf=CONF_THRESHOLD, device=device)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1.0 / avg_time

    # Calculate percentiles
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    p99 = np.percentile(times, 99)

    fps_metrics = {
        "model": model_path,
        "avg_time_ms": avg_time * 1000,
        "std_time_ms": std_time * 1000,
        "fps": fps,
        "p50_ms": p50 * 1000,
        "p95_ms": p95 * 1000,
        "p99_ms": p99 * 1000,
    }

    print(f"\nFPS Results for {model_path}:")
    print(f"  Avg FPS:     {fps:.2f}")
    print(f"  Avg Time:    {avg_time*1000:.2f} ms")
    print(f"  Std Dev:     {std_time*1000:.2f} ms")
    print(f"  P50:         {p50*1000:.2f} ms")
    print(f"  P95:         {p95*1000:.2f} ms")
    print(f"  P99:         {p99*1000:.2f} ms")

    return fps_metrics


def generate_report(all_metrics, output_path="benchmark_results.json"):
    """Generate comparison report."""
    print(f"\n{'='*60}")
    print("BENCHMARK COMPARISON REPORT")
    print(f"{'='*60}\n")

    # Print validation metrics table
    print(f"{'Model':<40} {'mAP50':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 80)

    for m in all_metrics["validation"]:
        name = Path(m["model"]).name
        print(
            f"{name:<40} "
            f"{m['mAP50']:<10.4f} "
            f"{m['mAP50_95']:<10.4f} "
            f"{m['precision']:<10.4f} "
            f"{m['recall']:<10.4f}"
        )

    print("\n")

    # Print FPS table
    print(f"{'Model':<40} {'FPS':<10} {'Avg ms':<10} {'P95 ms':<10} {'P99 ms':<10}")
    print("-" * 80)

    for m in all_metrics["fps"]:
        name = Path(m["model"]).name
        print(
            f"{name:<40} "
            f"{m['fps']:<10.2f} "
            f"{m['avg_time_ms']:<10.2f} "
            f"{m['p95_ms']:<10.2f} "
            f"{m['p99_ms']:<10.2f}"
        )

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLO models on VisDrone dataset")
    parser.add_argument("--model1", type=str, help="Path to first model")
    parser.add_argument("--model2", type=str, help="Path to second model")
    parser.add_argument("--data", type=str, default=DATASET_YAML, help="Path to dataset YAML")
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE, help="Image size")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="IoU threshold")
    parser.add_argument("--fps_frames", type=int, default=FPS_FRAMES, help="Frames for FPS test")
    args = parser.parse_args()

    # Collect model paths
    models = []
    if args.model1:
        models.append(args.model1)
    if args.model2:
        models.append(args.model2)
    if not models and MODEL_PATHS:
        models = MODEL_PATHS

    if not models:
        print("Error: No models specified!")
        print("Usage: python benchmark.py --model1 path/to/model1.pt --model2 path/to/model2.pt")
        print("Or edit MODEL_PATHS list in the script.")
        return

    # Verify dataset exists
    if not Path(args.data).exists():
        print(f"Error: Dataset YAML not found at {args.data}")
        print("Run convert_to_yolo.py first.")
        return

    all_metrics = {"validation": [], "fps": []}

    for model_path in models:
        if not Path(model_path).exists():
            print(f"Warning: Model not found at {model_path}, skipping...")
            continue

        # Validation
        val_metrics = validate_model(model_path, args.data, args.imgsz, args.conf, args.iou)
        all_metrics["validation"].append(val_metrics)

        # FPS Benchmark
        fps_metrics = benchmark_fps(model_path, args.imgsz, args.fps_frames)
        if fps_metrics:
            all_metrics["fps"].append(fps_metrics)

    # Generate comparison report
    if all_metrics["validation"]:
        generate_report(all_metrics)


if __name__ == "__main__":
    main()
