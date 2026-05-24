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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-export and overwrite existing optimized models",
    )
    return parser.parse_args()


def get_export_kwargs(fmt, half, int8, imgsz, no_dynamic, no_simplify):
    """
    Build safe and optimized export arguments for each specific model format.
    Ultralytics YOLO.export() throws errors if unsupported arguments (like dynamic or int8)
    are passed to certain formats (like tflite, ncnn, or onnx).
    """
    kwargs = {
        "format": fmt,
        "imgsz": imgsz,
    }
    
    # 1. ONNX supports standard features (but NOT int8)
    if fmt == "onnx":
        kwargs["half"] = half
        kwargs["dynamic"] = not no_dynamic
        kwargs["simplify"] = not no_simplify
        
    # 2. TensorRT Engine supports all, but needs GPU/device configuration
    elif fmt == "engine":
        kwargs["half"] = half
        kwargs["int8"] = int8
        kwargs["dynamic"] = not no_dynamic
        kwargs["simplify"] = not no_simplify
        
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
        kwargs["half"] = half
        kwargs["int8"] = int8
        kwargs["dynamic"] = not no_dynamic
        
    # 4. TFLite does NOT support dynamic, simplify or half in standard Ultralytics CLI
    elif fmt == "tflite":
        kwargs["int8"] = int8
        
    # 5. NCNN does NOT support dynamic, simplify or int8 in standard Ultralytics CLI
    elif fmt == "ncnn":
        kwargs["half"] = half
        
    return kwargs


def main():
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"Error: Model file not found at '{model_path}'", file=sys.stderr)
        sys.exit(1)

    # Process formats
    export_formats = [fmt.strip().lower() for fmt in args.formats.split(",")]

    # Pre-defined matrix of all standard format + precision combinations
    # (format_key, target_format_str, half_flag, int8_flag, precision_label)
    EXPORT_MATRIX = [
        ("onnx", "onnx", False, False, "fp32"),
        ("onnx", "onnx", True, False, "fp16"),
        
        ("engine", "engine", False, False, "fp32"),
        ("engine", "engine", True, False, "fp16"),
        ("engine", "engine", False, True, "int8"),
        
        ("openvino", "openvino", False, False, "fp32"),
        ("openvino", "openvino", True, False, "fp16"),
        ("openvino", "openvino", False, True, "int8"),
        
        ("tflite", "tflite", False, False, "fp32"),
        ("tflite", "tflite", False, True, "int8"),
        
        ("ncnn", "ncnn", False, False, "fp32"),
        ("ncnn", "ncnn", True, False, "fp16"),
    ]

    # Valid formats registry mapping to target strings
    valid_formats = {
        "onnx": "onnx",
        "engine": "engine",
        "tensorrt": "engine",
        "openvino": "openvino",
        "tflite": "tflite",
        "ncnn": "ncnn",
    }

    # Resolve active runs to execute
    runs = []
    if "all" in export_formats:
        runs = EXPORT_MATRIX
    else:
        for fmt in export_formats:
            if fmt not in valid_formats:
                print(f"Warning: Format '{fmt}' is not recognized. Skipping...")
                continue
            target_fmt = valid_formats[fmt]
            
            # Map standard flags to specific runs
            if args.int8:
                runs.append((fmt, target_fmt, args.half, args.int8, "int8"))
            elif args.half:
                runs.append((fmt, target_fmt, args.half, args.int8, "fp16"))
            else:
                runs.append((fmt, target_fmt, args.half, args.int8, "fp32"))

    print(f"\n{'='*60}")
    print(f"Initializing YOLOv8 Model Optimizer")
    print(f"  Input Model: {model_path}")
    print(f"  Image Size:  {args.imgsz}")
    print(f"  Dynamic:     {not args.no_dynamic}")
    print(f"  Simplify:    {not args.no_simplify}")
    print(f"  Force Flag:  {args.force}")
    print(f"  Total Runs Scheduled: {len(runs)}")
    print(f"{'='*60}\n")

    # Load model
    print(f"Loading PyTorch model weights...")
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    results = {}

    for fmt_key, target_fmt, half, int8, precision in runs:
        # Map format to structured folder name
        folder_mapping = {
            "onnx": "onnx",
            "engine": "tensorrt",
            "openvino": "openvino",
            "tflite": "tflite",
            "ncnn": "ncnn"
        }
        format_folder = folder_mapping.get(target_fmt, target_fmt)
        
        # Predict source name
        if target_fmt == "onnx":
            src_name = f"{model_path.stem}.onnx"
        elif target_fmt == "engine":
            src_name = f"{model_path.stem}.engine"
        elif target_fmt == "openvino":
            src_name = f"{model_path.stem}_openvino_model"
        elif target_fmt == "tflite":
            src_name = f"{model_path.stem}_saved_model"
        elif target_fmt == "ncnn":
            src_name = f"{model_path.stem}_ncnn_model"
        else:
            src_name = model_path.name
            
        dest_path = model_path.parent / format_folder / precision / src_name
        
        # Check if the restructured path already exists and bypass if --force is False
        if dest_path.exists() and not args.force:
            print(f"\n--- Skipping '{target_fmt.upper()}' ({precision}) ---")
            print(f"  Model already exists at: {dest_path}")
            print(f"  Skipping export. Use --force to overwrite.")
            results[f"{target_fmt}_{precision}"] = str(dest_path)
            continue

        print(f"\n--- Exporting to '{target_fmt.upper()}' ({precision}) format ---")
        
        try:
            # Run Ultralytics export with dynamically built, pruned arguments
            export_kwargs = get_export_kwargs(
                target_fmt, half, int8, args.imgsz, args.no_dynamic, args.no_simplify
            )
            print(f"  Export parameters: {export_kwargs}")
            exported_path = model.export(**export_kwargs)
            print(f"Successfully exported to '{target_fmt.upper()}'. Raw output path: {exported_path}")
            
            # Reorganize output folder structure by format and precision
            if exported_path:
                src_path = Path(exported_path)
                if src_path.exists():
                    # Target directory: [model_dir]/[format]/[precision]/
                    target_dir = model_path.parent / format_folder / precision
                    target_dir.mkdir(parents=True, exist_ok=True)
                    
                    dest_path = target_dir / src_path.name
                    
                    # If target already exists, remove it first to avoid collision
                    import shutil
                    if dest_path.exists():
                        if dest_path.is_dir():
                            shutil.rmtree(dest_path)
                        else:
                            dest_path.unlink()
                            
                    if src_path.resolve() != dest_path.resolve():
                        shutil.move(str(src_path), str(dest_path))
                        
                    print(f"  Model successfully restructured to: {dest_path}")
                    results[f"{target_fmt}_{precision}"] = str(dest_path)
                else:
                    print(f"  Warning: Exported file/folder '{src_path}' not found on disk.", file=sys.stderr)
                    results[f"{target_fmt}_{precision}"] = exported_path
            else:
                print(f"  Warning: Export succeeded but no path was returned by Ultralytics.", file=sys.stderr)
                results[f"{target_fmt}_{precision}"] = "SUCCESS (no path returned)"
        except Exception as e:
            print(f"Error exporting to '{target_fmt.upper()}' ({precision}): {e}", file=sys.stderr)
            results[f"{target_fmt}_{precision}"] = f"FAILED: {e}"

    print(f"\n{'='*60}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    for run_key, path in results.items():
        print(f"  {run_key.upper():<20} : {path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
