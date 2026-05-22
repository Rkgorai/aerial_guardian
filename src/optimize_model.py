#!/usr/bin/env python3
"""
YOLOv8 Model Optimizer for Edge Devices.
Supports exporting PyTorch models (.pt) to ONNX, TensorRT, OpenVINO, NCNN, and TFLite formats.
"""

import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize YOLOv8 models for edge devices")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to input PyTorch model (.pt), e.g. yolov8n.pt",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="onnx,tflite",
        help="Comma-separated formats to export (onnx, engine, openvino, tflite, ncnn)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Target input image size (imgsz)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 half precision (recommended for TensorRT/GPU deployment)",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Use INT8 integer quantization (requires calibration data for high accuracy)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"Error: Model file not found at '{model_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Initializing YOLOv8 Model Optimizer")
    print(f"  Input Model: {model_path}")
    print(f"  Image Size:  {args.imgsz}")
    print(f"  FP16 (Half): {args.half}")
    print(f"  INT8 Quant:  {args.int8}")
    print(f"{'='*60}\n")

    # Load model
    print(f"Loading PyTorch model weights...")
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    # Process formats
    export_formats = [fmt.strip().lower() for fmt in args.formats.split(",")]
    
    # Valid formats registry mapping to Ultralytics formats
    valid_formats = {
        "onnx": "onnx",
        "engine": "engine",
        "tensorrt": "engine",
        "openvino": "openvino",
        "tflite": "tflite",
        "ncnn": "ncnn",
    }

    results = {}

    for fmt in export_formats:
        if fmt not in valid_formats:
            print(f"Warning: Format '{fmt}' is not recognized. Skipping...")
            continue
        
        target_fmt = valid_formats[fmt]
        print(f"\n--- Exporting to '{target_fmt.upper()}' format ---")
        
        try:
            # Run Ultralytics export
            exported_path = model.export(
                format=target_fmt,
                imgsz=args.imgsz,
                half=args.half,
                int8=args.int8,
                dynamic=False,
            )
            print(f"Successfully exported to '{target_fmt.upper()}'. Output path: {exported_path}")
            results[target_fmt] = exported_path
        except Exception as e:
            print(f"Error exporting to '{target_fmt.upper()}': {e}", file=sys.stderr)
            results[target_fmt] = f"FAILED: {e}"

    print(f"\n{'='*60}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    for fmt, path in results.items():
        print(f"  {fmt.upper():<12} : {path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
