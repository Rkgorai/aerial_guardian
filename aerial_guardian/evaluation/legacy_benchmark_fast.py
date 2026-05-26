"""
Fast benchmark script for comparing multiple YOLO models on VisDrone validation set.
Uses a subset of images for quick evaluation.

Usage:
    python benchmark_fast.py --model1 path/to/model1.pt --model2 path/to/model2.pt
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================
MODEL_PATHS = [
    # "path/to/model1.pt",
    # "path/to/model2.pt",
]

DATASET_YAML = "yolo_dataset/dataset.yaml"
IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
SUBSET_FRAMES = 100  # Number of frames for quick validation
FPS_FRAMES = 50  # Number of frames for FPS benchmark


def get_subset_images(images_dir, num_frames=SUBSET_FRAMES):
    """Get a representative subset of images."""
    all_images = sorted(Path(images_dir).glob("*.jpg"))
    if not all_images:
        return []

    # Take evenly spaced frames
    step = max(1, len(all_images) // num_frames)
    subset = all_images[::step][:num_frames]

    np.random.seed(42)
    np.random.shuffle(subset)
    return subset


def validate_model_fast(model_path, images, imgsz, conf):
    """Run fast validation on subset of images."""
    print(f"\n{'='*60}")
    print(f"Validating: {model_path}")
    print(f"{'='*60}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(model_path, task="detect")

    # Load ground truth labels for these images
    gt_boxes = []
    pred_boxes = []

    total_gt = 0
    total_pred = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Get predictions
        results = model(img, imgsz=imgsz, conf=conf, verbose=False, device=device)
        preds = results[0].boxes

        # Load ground truth
        label_path = Path(str(img_path).replace("/images/val/", "/labels/val/").replace(".jpg", ".txt"))
        gt = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, xc, yc, w, h = map(float, parts)
                        gt.append([xc, yc, w, h])

        total_gt += len(gt)
        total_pred += len(preds)

        # Simple IoU matching
        if len(preds) > 0 and len(gt) > 0:
            img_h, img_w = img.shape[:2]

            # Convert GT to absolute coordinates
            gt_abs = []
            for xc, yc, w, h in gt:
                x1 = (xc - w / 2) * img_w
                y1 = (yc - h / 2) * img_h
                x2 = (xc + w / 2) * img_w
                y2 = (yc + h / 2) * img_h
                gt_abs.append([x1, y1, x2, y2])

            # Convert predictions to absolute coordinates
            pred_abs = []
            for box in preds:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                pred_abs.append([x1, y1, x2, y2])

            # Calculate IoU and match
            matched_gt = set()
            for p_idx, p_box in enumerate(pred_abs):
                best_iou = 0
                best_gt_idx = -1

                for g_idx, g_box in enumerate(gt_abs):
                    if g_idx in matched_gt:
                        continue

                    # IoU calculation
                    x1_i = max(p_box[0], g_box[0])
                    y1_i = max(p_box[1], g_box[1])
                    x2_i = min(p_box[2], g_box[2])
                    y2_i = min(p_box[3], g_box[3])

                    inter_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
                    p_area = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
                    g_area = (g_box[2] - g_box[0]) * (g_box[3] - g_box[1])
                    union_area = p_area + g_area - inter_area

                    iou = inter_area / union_area if union_area > 0 else 0

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

                if best_iou >= IOU_THRESHOLD:
                    true_positives += 1
                    matched_gt.add(best_gt_idx)
                else:
                    false_positives += 1

            false_negatives += len(gt) - len(matched_gt)
        else:
            false_positives += len(preds)
            false_negatives += len(gt)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "model": model_path,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "total_gt": total_gt,
        "total_pred": total_pred,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "parameters": model.info()[1],  # info() returns (layers, params, grads, GFLOPs)
    }

    print(f"\nResults for {model_path}:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  GT boxes:  {total_gt}")
    print(f"  Pred boxes: {total_pred}")
    print(f"  TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}")
    print(f"  Params:    {metrics['parameters']:,}")

    return metrics


def benchmark_fps(model_path, imgsz, num_frames=FPS_FRAMES):
    """Benchmark inference speed."""
    print(f"\n{'='*60}")
    print(f"FPS Benchmark: {model_path}")
    print(f"{'='*60}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(model_path, task="detect")

    # Find a sample image for benchmarking
    images_dir = Path("yolo_dataset/images/val")
    sample_images = list(images_dir.glob("**/*.jpg"))

    if not sample_images:
        print("No images found for FPS benchmark!")
        return None

    img = cv2.imread(str(sample_images[0]))
    if img is None:
        print("Failed to load sample image!")
        return None

    print(f"Benchmarking with {num_frames} frames (device: {device})...")

    # Warmup
    for _ in range(5):
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
    print(f"{'Model':<40} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Params':<12}")
    print("-" * 88)

    for m in all_metrics["validation"]:
        name = Path(m["model"]).name
        print(
            f"{name:<40} "
            f"{m['precision']:<12.4f} "
            f"{m['recall']:<12.4f} "
            f"{m['f1_score']:<12.4f} "
            f"{m['parameters']:<12,}"
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
    parser = argparse.ArgumentParser(description="Fast benchmark YOLO models on VisDrone dataset")
    parser.add_argument("--model1", type=str, help="Path to first model")
    parser.add_argument("--model2", type=str, help="Path to second model")
    parser.add_argument("--data", type=str, default=DATASET_YAML, help="Path to dataset YAML")
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE, help="Image size")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="IoU threshold")
    parser.add_argument("--subset", type=int, default=SUBSET_FRAMES, help="Frames for validation")
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
        print("Usage: python benchmark_fast.py --model1 path/to/model1.pt --model2 path/to/model2.pt")
        return

    # Get subset images
    images_dir = Path("yolo_dataset/images/val")
    subset_images = get_subset_images(images_dir, args.subset)
    print(f"Using {len(subset_images)} images for validation")

    all_metrics = {"validation": [], "fps": []}

    for model_path in models:
        if not Path(model_path).exists():
            print(f"Warning: Model not found at {model_path}, skipping...")
            continue

        # Fast Validation
        val_metrics = validate_model_fast(model_path, subset_images, args.imgsz, args.conf)
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
