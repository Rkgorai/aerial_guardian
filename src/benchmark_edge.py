#!/usr/bin/env python3
"""
YOLOv8 Edge Model Benchmarking Engine.
Compares PyTorch, ONNX, TFLite, and other exported formats on file size, latency, FPS, and stability.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


def discover_models(model_dir):
    """Discover all benchmarkable models in a directory."""
    path = Path(model_dir)
    if not path.exists() or not path.is_dir():
        print(f"Error: Directory '{model_dir}' does not exist or is not a directory.", file=sys.stderr)
        return []
    
    discovered = []
    
    # 1. Look for direct model files
    exts = [".pt", ".onnx", ".tflite"]
    try:
        import torch
        if torch.cuda.is_available():
            exts.append(".engine")
        else:
            print("CUDA is not available. Skipping TensorRT (.engine) models from automatic discovery.")
    except Exception:
        pass
        
    for ext in exts:
        discovered.extend(list(path.glob(f"*{ext}")))
        
    # 2. Look for model directories
    for subdir in path.iterdir():
        if not subdir.is_dir():
            continue
            
        # Check for OpenVINO model
        if subdir.name.endswith("_openvino_model") or (list(subdir.glob("*.xml")) and list(subdir.glob("*.bin"))):
            discovered.append(subdir)
            continue
            
        # Check for NCNN model
        if subdir.name.endswith("_ncnn_model") or (list(subdir.glob("model.ncnn.bin")) and list(subdir.glob("model.ncnn.param"))):
            discovered.append(subdir)
            continue
            
        # Check for TFLite / SavedModel folder
        if subdir.name.endswith("_saved_model") or (subdir / "saved_model.pb").exists():
            # Find individual TFLite files inside
            tflites = list(subdir.glob("*.tflite"))
            if tflites:
                discovered.extend(tflites)
            else:
                discovered.append(subdir) # Fallback to saved_model directory itself
                
    return [str(p) for p in discovered]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark edge-optimized YOLOv8 models")
    parser.add_argument(
        "--models",
        type=str,
        required=False,
        default=None,
        help="Comma-separated list of model files to benchmark (e.g. yolov8n.pt,yolov8n.onnx)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing multiple edge-optimized models to benchmark automatically",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (imgsz)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames to benchmark per model",
    )
    args = parser.parse_args()
    if not args.models and not args.model_dir:
        parser.error("At least one of --models or --model-dir must be specified.")
    return args



def get_model_size_mb(model_path):
    """Calculate the size of a model file or directory in Megabytes."""
    path = Path(model_path)
    if not path.exists():
        return 0.0

    if path.is_file():
        return path.stat().st_size / (1024 * 1024)

    # If it is a folder (like TFLite or OpenVINO folders)
    total_size = sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())
    return total_size / (1024 * 1024)


def benchmark_model(model_path, imgsz, num_frames=100):
    """Measure file size, latency, FPS, and percentiles for a single model format."""
    print(f"\n{'='*60}")
    print(f"Benchmarking Model: {model_path}")
    print(f"{'='*60}")

    size_mb = get_model_size_mb(model_path)
    print(f"Model File Size: {size_mb:.2f} MB")

    # Find sample images for benchmark
    images_dir = Path("yolo_dataset/images/val")
    if not images_dir.exists():
        # Fallback to visual validation sequence or random array
        print("Warning: yolo_dataset/images/val not found. Generating dummy image...")
        img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    else:
        sample_images = list(images_dir.glob("*.jpg"))
        if not sample_images:
            print("Warning: No JPG images in validation folder. Generating dummy image...")
            img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        else:
            img = cv2.imread(str(sample_images[0]))

    # Resize to benchmark size
    img = cv2.resize(img, (imgsz, imgsz))

    # Load model and run warmup via Ultralytics (automatically routes based on extension)
    try:
        model = YOLO(str(model_path))
        print("Running warmup frames...")
        for _ in range(5):
            _ = model(img, imgsz=imgsz, verbose=False)
    except Exception as e:
        print(f"Error loading or warming up model: {e}", file=sys.stderr)
        return None

    # Benchmarking
    print(f"Running {num_frames} timed inference runs...")
    latencies = []
    
    for _ in range(num_frames):
        start = time.perf_counter()
        _ = model(img, imgsz=imgsz, verbose=False)
        end = time.perf_counter()
        latencies.append(end - start)

    latencies = np.array(latencies) * 1000  # convert to milliseconds

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    fps = 1000.0 / avg_latency

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    print(f"Results:")
    print(f"  Avg Latency : {avg_latency:.2f} ms")
    print(f"  Avg FPS     : {fps:.2f}")
    print(f"  P50 Latency : {p50:.2f} ms")
    print(f"  P95 Latency : {p95:.2f} ms")
    print(f"  P99 Latency : {p99:.2f} ms")

    return {
        "model_name": Path(model_path).name,
        "format": Path(model_path).suffix or "folder",
        "size_mb": size_mb,
        "avg_latency_ms": avg_latency,
        "std_latency_ms": std_latency,
        "fps": fps,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
    }


def generate_report(results):
    """Display a clean comparative Markdown table."""
    print(f"\n{'='*60}")
    print("COMPARATIVE EDGE PERFORMANCE REPORT")
    print(f"{'='*60}\n")

    # Table Header
    header = f"| {'Model Format':<22} | {'File Size (MB)':<14} | {'Avg Latency (ms)':<16} | {'Avg FPS':<10} | {'P95 Latency (ms)':<16} |"
    divider = "| " + "-"*22 + " | " + "-"*14 + " | " + "-"*16 + " | " + "-"*10 + " | " + "-"*16 + " |"
    
    print(header)
    print(divider)

    for r in results:
        print(
            f"| {r['model_name']:<22} | "
            f"{r['size_mb']:<14.2f} | "
            f"{r['avg_latency_ms']:<16.2f} | "
            f"{r['fps']:<10.2f} | "
            f"{r['p95_ms']:<16.2f} |"
        )
    print()


def main():
    args = parse_args()
    model_paths = []
    
    if args.models:
        model_paths.extend([m.strip() for m in args.models.split(",")])
        
    if args.model_dir:
        dir_models = discover_models(args.model_dir)
        if dir_models:
            print(f"Discovered {len(dir_models)} models in '{args.model_dir}':")
            for m in sorted(dir_models):
                print(f"  - {m}")
            model_paths.extend(sorted(dir_models))
        else:
            print(f"Warning: No models discovered in directory '{args.model_dir}'")

    results = []
    for path in model_paths:
        if not Path(path).exists():
            print(f"Warning: Model path '{path}' does not exist. Skipping...", file=sys.stderr)
            continue
        
        res = benchmark_model(path, args.imgsz, args.frames)
        if res:
            results.append(res)

    if results:
        generate_report(results)
    else:
        print("Error: No models benchmarked successfully.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
