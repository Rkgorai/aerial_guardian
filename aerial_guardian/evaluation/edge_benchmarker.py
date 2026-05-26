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
    """Discover all benchmarkable models in a directory recursively, ignoring virtualenvs and hidden folders."""
    path = Path(model_dir)
    if not path.exists() or not path.is_dir():
        print(f"Error: Directory '{model_dir}' does not exist or is not a directory.", file=sys.stderr)
        return []
    
    discovered = []
    
    # 1. Supported direct file extensions
    exts = [".pt", ".onnx", ".tflite"]
    try:
        import torch
        if torch.cuda.is_available():
            exts.append(".engine")
        else:
            print("CUDA is not available. Skipping TensorRT (.engine) models from automatic discovery.")
    except Exception:
        pass
        
    # Standard directories to completely ignore during recursive traversal
    ignore_dirs = {"env", "venv", ".git", ".gemini", ".system_generated", "__pycache__", "node_modules", "runs"}
        
    # We will search the entire directory tree recursively
    for p in path.rglob("*"):
        # Check if this path contains any ignored directory in its parts
        if any(part in ignore_dirs or part.startswith(".") for part in p.parts):
            continue
            
        # 1. Check for files with supported extensions
        if p.is_file() and p.suffix in exts:
            discovered.append(p)
            continue
            
        # 2. Check for OpenVINO model directory
        if p.is_dir():
            if p.name.endswith("_openvino_model") or (list(p.glob("*.xml")) and list(p.glob("*.bin"))):
                discovered.append(p)
                continue
                
            # Check for NCNN model directory
            if p.name.endswith("_ncnn_model") or (list(p.glob("model.ncnn.bin")) and list(p.glob("model.ncnn.param"))):
                discovered.append(p)
                continue
                
            # Check for TFLite / SavedModel folder
            if p.name.endswith("_saved_model") or (p / "saved_model.pb").exists():
                tflites = list(p.glob("*.tflite"))
                if tflites:
                    discovered.extend(tflites)
                else:
                    discovered.append(p)
                continue
                
    # Deduplicate in case files/folders are discovered multiple times
    unique_discovered = []
    seen = set()
    for p in discovered:
        abs_p = p.resolve()
        if abs_p not in seen:
            seen.add(abs_p)
            unique_discovered.append(p)
            
    # Filter out files inside OpenVINO/NCNN/SavedModel directories that we already discovered as directories
    final_discovered = []
    for p in unique_discovered:
        is_sub_file = False
        for parent in p.parents:
            if parent in unique_discovered:
                is_sub_file = True
                break
        if not is_sub_file:
            final_discovered.append(p)
            
    return [str(p) for p in final_discovered]


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
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Path to save benchmark results in JSON format (e.g. output/results.json)",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Path to save benchmark results in CSV format (e.g. output/results.csv)",
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


def extract_precision(model_path):
    """Extract precision type (fp32, fp16, int8) from path structure or filename."""
    p = Path(model_path)
    
    # 1. Check parent folder name first (our structured format)
    parent_name = p.parent.name.lower()
    if parent_name in ["fp32", "fp16", "int8"]:
        return parent_name.upper()
        
    # 2. Check filename clues
    filename = p.name.lower()
    if "int8" in filename or "quant" in filename:
        return "INT8"
    if "fp16" in filename or "float16" in filename or "half" in filename:
        return "FP16"
    if "fp32" in filename or "float32" in filename:
        return "FP32"
        
    # 3. Default fallback based on standard formats
    return "FP32"


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
        model = YOLO(str(model_path), task="detect")
        
        # Determine device dynamically: default to GPU if available for standard models
        device = None
        if Path(model_path).suffix in [".pt", ".onnx", ".engine"]:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        print(f"Running warmup frames (device: {device or 'auto'})...")
        for _ in range(5):
            if device:
                _ = model(img, imgsz=imgsz, verbose=False, device=device)
            else:
                _ = model(img, imgsz=imgsz, verbose=False)
    except Exception as e:
        print(f"Error loading or warming up model: {e}", file=sys.stderr)
        return None

    # Benchmarking
    print(f"Running {num_frames} timed inference runs...")
    latencies = []
    
    for _ in range(num_frames):
        start = time.perf_counter()
        if device:
            _ = model(img, imgsz=imgsz, verbose=False, device=device)
        else:
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

    precision = extract_precision(model_path)

    return {
        "model_name": Path(model_path).name,
        "format": Path(model_path).suffix or "folder",
        "precision": precision,
        "size_mb": size_mb,
        "avg_latency_ms": avg_latency,
        "std_latency_ms": std_latency,
        "fps": fps,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
    }


def save_json_results(results, save_path):
    """Save benchmark results to a JSON file."""
    try:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"Benchmark results successfully saved to JSON: {save_path}")
    except Exception as e:
        print(f"Error saving JSON results: {e}", file=sys.stderr)


def save_csv_results(results, save_path):
    """Save benchmark results to a CSV file."""
    try:
        import csv
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        if not results:
            return
        headers = list(results[0].keys())
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)
        print(f"Benchmark results successfully saved to CSV: {save_path}")
    except Exception as e:
        print(f"Error saving CSV results: {e}", file=sys.stderr)


def generate_report(results):
    """Display a clean comparative Markdown table."""
    print(f"\n{'='*80}")
    print("COMPARATIVE EDGE PERFORMANCE REPORT")
    print(f"{'='*80}\n")

    # Table Header
    header = f"| {'Model Format':<22} | {'Precision':<10} | {'File Size (MB)':<14} | {'Avg Latency (ms)':<16} | {'Avg FPS':<10} | {'P95 Latency (ms)':<16} |"
    divider = "| " + "-"*22 + " | " + "-"*10 + " | " + "-"*14 + " | " + "-"*16 + " | " + "-"*10 + " | " + "-"*16 + " |"
    
    print(header)
    print(divider)

    for r in results:
        print(
            f"| {r['model_name']:<22} | "
            f"{r['precision']:<10} | "
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
        
        # Save results if requested
        if args.save_json:
            save_json_results(results, args.save_json)
        if args.save_csv:
            save_csv_results(results, args.save_csv)
    else:
        print("Error: No models benchmarked successfully.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
