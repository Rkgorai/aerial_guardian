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
    if list(weights_dir.glob("*.pt")):
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


def _get_active_pt_path() -> Path | None:
    """Return the most recently modified .pt model in weights/."""
    weights_dir = Path("weights")
    pt_files = list(weights_dir.glob("*.pt"))
    if not pt_files:
        return None
    # Sort by modification time (newest first)
    pt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return pt_files[0]

def get_active_model_name() -> str:
    active_pt = _get_active_pt_path()
    return active_pt.name if active_pt else "None"

def _get_compiled_name(base_stem: str, target_fmt: str) -> str:
    if target_fmt == "onnx": return f"{base_stem}.onnx"
    elif target_fmt == "tensorrt": return f"{base_stem}.engine"
    elif target_fmt == "openvino": return f"{base_stem}_openvino_model"
    elif target_fmt == "tflite": return f"{base_stem}_saved_model"
    elif target_fmt == "ncnn": return f"{base_stem}_ncnn_model"
    return base_stem


def is_model_present(format_key: str, precision: str) -> bool:
    """Check if a specific model format + precision exists on disk for the active model."""
    weights_dir = Path("weights")
    if format_key == "pt":
        return _get_active_pt_path() is not None
        
    active_pt = _get_active_pt_path()
    if not active_pt:
        return False
        
    target_fmt = "tensorrt" if format_key == "engine" else format_key
    model_dir = weights_dir / target_fmt / precision
    if not model_dir.exists():
        return False
        
    expected_name = _get_compiled_name(active_pt.stem, target_fmt)
    return (model_dir / expected_name).exists()


def get_model_path(format_key: str, precision: str) -> str | None:
    """Return the path to the active compiled model, or None if not found."""
    weights_dir = Path("weights")
    active_pt = _get_active_pt_path()
    if not active_pt:
        return None
        
    if format_key == "pt":
        return str(active_pt)
        
    target_fmt = "tensorrt" if format_key == "engine" else format_key
    model_dir = weights_dir / target_fmt / precision
    if model_dir.exists():
        expected_name = _get_compiled_name(active_pt.stem, target_fmt)
        expected_path = model_dir / expected_name
        if expected_path.exists():
            return str(expected_path)
    return None

def download_yolo_model_generator():
    import requests
    import sys
    from pathlib import Path
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    out_path = weights_dir / "mot_visdrone_finetuned.pt"
    
    url = "https://drive.google.com/uc?id=1GuKD-B_mH8sCiQMK25qeLnRdp-9CJnde"
    r = requests.get(url, stream=True)
    r.raise_for_status()
    
    total_size = int(r.headers.get('content-length', 0))
    chunk_size = 1024 * 1024 # 1 MB
    downloaded = 0
    
    yield f"⏳ Starting download of {total_size / (1024*1024):.1f} MB..."
    
    with open(out_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    perc = downloaded / total_size
                    msg = f"⏳ **Downloading:** {mb_downloaded:.1f} MB / {mb_total:.1f} MB ({(perc*100):.1f}%)"
                    yield msg
    yield "✅ Download complete!"


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
    output_path: str,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    tracker: str = "botsort.yaml",
    device: str = "cpu",
):
    """
    Generator that yields (frame, fps, detection_count, track_count) using
    the custom AerialGuardianPipeline from the project's tracking package.
    Saves the full-resolution annotated video to output_path.
    """
    import cv2
    import time
    from aerial_guardian.tracking.pipeline import AerialGuardianPipeline
    from aerial_guardian.tracking.video_writer import get_video_writer
    
    # Initialize the custom pipeline (defaults to custom tracker)
    pipeline = AerialGuardianPipeline(
        model_path=model_path,
        conf_thresh=conf,
        iou_thresh=iou,
        img_size=imgsz,
        tracker_cfg=tracker,
        device=device,
        # app.py currently doesn't pass tracker_type, so it defaults to "custom"
        # but we use pipeline.tracker_type to dynamically support it if added later
    )
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {source}")
        
    fps_cap = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    import numpy as np
    if fps_cap <= 0 or np.isnan(fps_cap):
        fps_cap = 25.0
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Ensure parent output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize VideoWriter using our robust abstraction
    # Use "ffmpeg" encoder natively, which automatically falls back: NVENC -> libx264 -> opencv
    try:
        out = get_video_writer(output_path, fps_cap, (width, height), encoder="ffmpeg")
    except Exception as e:
        print(f"Failed to init ffmpeg writer, falling back to opencv: {e}")
        out = get_video_writer(output_path, fps_cap, (width, height), encoder="opencv")
        
    processing_times = []
    
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            t_start = time.time()
            
            # 1 & 2. Run detection and tracking
            if pipeline.tracker_type == "custom":
                detections = pipeline.detect(frame)
                tracks = pipeline.tracker.update(detections, frame)
            else:
                results = pipeline.model.track(
                    frame,
                    persist=True,
                    tracker=pipeline.tracker_cfg,
                    conf=pipeline.conf_thresh,
                    iou=pipeline.iou_thresh,
                    imgsz=pipeline.img_size,
                    verbose=False,
                    device=pipeline.device,
                )
                tracks = []
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    for box, track_id, det_conf in zip(boxes, track_ids, confs):
                        x, y, w, h = box
                        tracks.append([x, y, w, h, track_id, det_conf])
            
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
            
            # Write full-resolution annotated BGR frame to output video file
            out.write(output_frame)
                
            # Downsample preview frame to eliminate websocket transmission bottlenecks
            h, w = output_frame.shape[:2]
            max_preview_width = 800
            if w > max_preview_width:
                scale = max_preview_width / w
                new_h = int(h * scale)
                output_frame = cv2.resize(output_frame, (max_preview_width, new_h), interpolation=cv2.INTER_LINEAR)
                
            frame_rgb = output_frame[..., ::-1]  # BGR to RGB for Gradio
            
            # Count detections & unique tracks
            detection_count = len(detections) if pipeline.tracker_type == "custom" else len(tracks)
            track_count = len(tracks)
            
            yield frame_rgb, fps, detection_count, track_count, frame_idx, total_frames
            
    finally:
        cap.release()
        out.release()


# ---------------------------------------------------------------------------
# YouTube download
# ---------------------------------------------------------------------------
def download_youtube_video(url: str, output_path: str = "downloads/youtube_video.mp4") -> str:
    """Download a YouTube video using yt-dlp and return the local file path."""
    import yt_dlp

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "best[ext=mp4][vcodec^=avc1][height<=720]/best[ext=mp4][vcodec^=avc1]/best[ext=mp4]/best",
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return output_path
