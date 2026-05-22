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
    parser.add_argument(
        "--no-dynamic",
        action="store_true",
        help="Disable dynamic batching / dynamic input axes",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable model structure simplification (onnxslim / onnxsim)",
    )
    return parser.parse_args()


def get_export_kwargs(fmt, args):
    """
    Build safe and optimized export arguments for each specific model format.
    Ultralytics YOLO.export() throws errors if unsupported arguments (like dynamic or int8)
    are passed to certain formats (like tflite, ncnn, or onnx).
    """
    kwargs = {
        "format": fmt,
        "imgsz": args.imgsz,
    }
    
    # 1. ONNX supports standard features (but NOT int8)
    if fmt == "onnx":
        kwargs["half"] = args.half
        kwargs["dynamic"] = not args.no_dynamic
        kwargs["simplify"] = not args.no_simplify
        
    # 2. TensorRT Engine supports all, but needs GPU/device configuration
    elif fmt == "engine":
        kwargs["half"] = args.half
        kwargs["int8"] = args.int8
        kwargs["dynamic"] = not args.no_dynamic
        kwargs["simplify"] = not args.no_simplify
        
        # Only assign device=0 if GPU is actually available to avoid CUDA crash
        try:
            import torch
            if torch.cuda.is_available():
                kwargs["device"] = 0
            else:
                kwargs["device"] = "cpu"
        except Exception:
            kwargs["device"] = "cpu"
            
    # 3. OpenVINO supports half, int8 and dynamic
    elif fmt == "openvino":
        kwargs["half"] = args.half
        kwargs["int8"] = args.int8
        kwargs["dynamic"] = not args.no_dynamic
        
    # 4. TFLite does NOT support dynamic, simplify or half in standard Ultralytics CLI
    elif fmt == "tflite":
        kwargs["int8"] = args.int8
        
    # 5. NCNN does NOT support dynamic, simplify or int8 in standard Ultralytics CLI
    elif fmt == "ncnn":
        kwargs["half"] = args.half
        
    return kwargs


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
    print(f"  Dynamic:     {not args.no_dynamic}")
    print(f"  Simplify:    {not args.no_simplify}")
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
            # Run Ultralytics export with dynamically built, pruned arguments
            export_kwargs = get_export_kwargs(target_fmt, args)
            print(f"  Export parameters: {export_kwargs}")
            exported_path = model.export(**export_kwargs)
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
