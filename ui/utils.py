"""
Utility functions for the Aerial Guardian Web UI.
Handles model discovery, optimization, inference, and video downloading.
"""

import os
import subprocess
import sys
from pathlib import Path

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Format metadata – single source of truth for labels, descriptions, icons
# ---------------------------------------------------------------------------
FORMAT_INFO = {
    "pt": {
        "label": "PyTorch (.pt)",
        "icon": "🔥",
        "desc": "Original training format. Works everywhere, no conversion needed.",
        "precisions": ["default"],
    },
    "onnx": {
        "label": "ONNX",
        "icon": "🔷",
        "desc": "Cross-platform format. Good for CPU inference and broad compatibility.",
        "precisions": ["fp32", "fp16"],
    },
    "tensorrt": {
        "label": "TensorRT",
        "icon": "⚡",
        "desc": "NVIDIA GPU-optimised. Fastest inference on CUDA hardware.",
        "precisions": ["fp32", "fp16", "int8"],
    },
    "openvino": {
        "label": "OpenVINO",
        "icon": "🟦",
        "desc": "Intel-optimised. Best for Intel CPUs, iGPUs, and VPUs.",
        "precisions": ["fp32", "fp16", "int8"],
    },
    "tflite": {
        "label": "TensorFlow Lite",
        "icon": "📱",
        "desc": "Mobile & edge-friendly. Ideal for ARM devices and Coral TPUs.",
        "precisions": ["fp32", "int8"],
    },
    "ncnn": {
        "label": "NCNN",
        "icon": "🪶",
        "desc": "Ultra-lightweight. Perfect for ARM CPUs and low-resource boards.",
        "precisions": ["fp32", "fp16"],
    },
}


def get_format_choices():
    """Return a list of human-readable format labels for the dropdown."""
    return [f"{v['icon']}  {v['label']}" for v in FORMAT_INFO.values()]


def label_to_key(label: str) -> str:
    """Convert a dropdown label like '🔥  PyTorch (.pt)' back to its key 'pt'."""
    for key, val in FORMAT_INFO.items():
        full = f"{val['icon']}  {val['label']}"
        if label == full:
            return key
    return "pt"


def get_precisions_for_format(format_label: str):
    """Return the list of available precisions for a given format label."""
    key = label_to_key(format_label)
    info = FORMAT_INFO.get(key, {})
    return info.get("precisions", ["fp32"])


def get_format_description(format_label: str) -> str:
    """Return a user-friendly markdown description for the selected format."""
    key = label_to_key(format_label)
    info = FORMAT_INFO.get(key)
    if not info:
        return ""
    return f"> {info['icon']} **{info['label']}** — {info['desc']}"


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------
def scan_available_models() -> dict:
    """
    Scans the weights/ directory and returns a dict of what's present on disk.
    Example: {'onnx': ['fp32', 'fp16'], 'tensorrt': ['fp16'], 'pt': ['default']}
    """
    weights_dir = Path("weights")
    available = {}

    # Check root-level PyTorch model
    if (weights_dir / "best.pt").exists():
        available["pt"] = ["default"]

    for fmt_key in ["onnx", "tensorrt", "openvino", "tflite", "ncnn"]:
        fmt_path = weights_dir / fmt_key
        if not fmt_path.exists():
            continue
        precisions = []
        for prec_dir in fmt_path.iterdir():
            if prec_dir.is_dir() and any(prec_dir.iterdir()):
                precisions.append(prec_dir.name)
        if precisions:
            available[fmt_key] = sorted(precisions)

    return available


def is_model_present(format_key: str, precision: str) -> bool:
    """Check if a specific model format + precision exists on disk."""
    weights_dir = Path("weights")
    if format_key == "pt":
        return (weights_dir / "best.pt").exists()
    model_dir = weights_dir / format_key / precision
    if not model_dir.exists():
        return False
    return any(model_dir.iterdir())


def get_model_path(format_key: str, precision: str) -> str | None:
    """Return the path to a compiled model, or None if not found."""
    weights_dir = Path("weights")
    if format_key == "pt":
        pt_path = weights_dir / "best.pt"
        return str(pt_path) if pt_path.exists() else None
    model_dir = weights_dir / format_key / precision
    if model_dir.exists():
        files = list(model_dir.iterdir())
        if files:
            return str(files[0])
    return None


def get_availability_summary(format_label: str) -> str:
    """Return a formatted markdown summary of what's available for a format."""
    key = label_to_key(format_label)
    info = FORMAT_INFO.get(key, {})
    precisions = info.get("precisions", [])

    lines = [f"**Available precisions for {info.get('label', key)}:**\n"]
    for p in precisions:
        present = is_model_present(key, p)
        icon = "✅" if present else "⬜"
        status = "Ready" if present else "Will be built on first run"
        lines.append(f"  {icon}  `{p}` — {status}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model optimisation (subprocess with streamed logs)
# ---------------------------------------------------------------------------
def trigger_model_optimizer(model_path: str, target_format: str, half=False, int8=False):
    """
    Runs model_optimizer.py in a subprocess and yields log lines for streaming.
    """
    optimizer_script = Path("aerial_guardian/export/model_optimizer.py")
    cmd = [
        sys.executable,
        str(optimizer_script),
        "--model", str(model_path),
        "--formats", target_format,
    ]
    if half:
        cmd.append("--half")
    if int8:
        cmd.append("--int8")

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in iter(process.stdout.readline, ""):
        yield line.rstrip()
    process.stdout.close()
    rc = process.wait()
    if rc != 0:
        yield f"⚠️  Optimisation finished with exit code {rc}"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def infer_video(
    model_path: str,
    source: str,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    tracker: str = "botsort.yaml",
    device: str = "cpu",
):
    """
    Generator that yields (frame, fps, detection_count, track_count) using
    the custom AerialGuardianPipeline from the project's tracking package.
    """
    import cv2
    import time
    from aerial_guardian.tracking.pipeline import AerialGuardianPipeline
    
    # Initialize the custom pipeline
    pipeline = AerialGuardianPipeline(
        model_path=model_path,
        conf_thresh=conf,
        iou_thresh=iou,
        img_size=imgsz,
        tracker_cfg=tracker,
        device=device,
    )
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {source}")
        
    processing_times = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            t_start = time.time()
            
            # 1. Run detection
            detections = pipeline.detect(frame)
            
            # 2. Run custom tracker
            tracks = pipeline.tracker.update(detections, frame)
            
            # 3. Calculate sliding average active inference FPS
            t_end = time.time()
            dt = t_end - t_start
            processing_times.append(dt)
            if len(processing_times) > 30:
                processing_times.pop(0)
            avg_dt = sum(processing_times) / len(processing_times)
            fps = 1.0 / avg_dt if avg_dt > 0 else 0.0
            
            # 4. Generate visual frame with custom visualize helper using correct FPS
            output_frame = pipeline.visualize(frame, tracks, fps)
            
            # Downsample preview frame to eliminate websocket transmission bottlenecks
            h, w = output_frame.shape[:2]
            max_preview_width = 800
            if w > max_preview_width:
                scale = max_preview_width / w
                new_h = int(h * scale)
                output_frame = cv2.resize(output_frame, (max_preview_width, new_h), interpolation=cv2.INTER_LINEAR)
                
            frame_rgb = output_frame[..., ::-1]  # BGR to RGB for Gradio
            
            # Count detections & unique tracks
            detection_count = len(detections)
            track_count = len(tracks)
            
            yield frame_rgb, fps, detection_count, track_count
            
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# YouTube download
# ---------------------------------------------------------------------------
def download_youtube_video(url: str, output_path: str = "downloads/youtube_video.mp4") -> str:
    """Download a YouTube video using yt-dlp and return the local file path."""
    import yt_dlp

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return output_path
