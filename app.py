"""
Aerial Guardian — Web UI

A clean, modern SaaS-style dashboard for running multi-object tracking across
different model formats and precisions.

Usage:
    source env/bin/activate
    python app.py
"""

import threading
import psutil
import torch
import time
import gradio as gr
from pathlib import Path

# Global stop flag — checked every frame during inference
_stop_event = threading.Event()

from ui.utils import (
    FORMAT_INFO,
    get_format_choices,
    get_format_description,
    get_precisions_for_format,
    get_availability_summary,
    label_to_key,
    is_model_present,
    get_model_path,
    trigger_model_optimizer,
    infer_video,
    download_youtube_video,
)


# ═══════════════════════════════════════════════════════════════════════════
# CSS — premium dark theme overrides globally forcing Gradio variables
# ═══════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Global Theme & Variable Overrides ───────────────────────────────────── */
:root {
    --bg-primary:    #0c0e14;
    --bg-card:       #14171f;
    --bg-card-hover: #181c26;
    --border:        #1e2330;
    --accent:        #6366f1;
    --accent-glow:   rgba(99, 102, 241, 0.25);
    --text-primary:  #e8eaed;
    --text-muted:    #8b8fa3;
    --success:       #34d399;
    --warning:       #fbbf24;
    --error:         #f87171;
    --radius:        12px;

    /* Force Gradio themes into premium dark mode variables globally */
    --body-background-fill: #0c0e14 !important;
    --body-background-fill-dark: #0c0e14 !important;
    --block-background-fill: #14171f !important;
    --block-background-fill-dark: #14171f !important;
    --block-border-color: #1e2330 !important;
    --block-border-color-dark: #1e2330 !important;
    --block-title-text-color: #e8eaed !important;
    --block-title-text-color-dark: #e8eaed !important;
    --block-label-text-color: #e8eaed !important;
    --block-label-text-color-dark: #e8eaed !important;
    --body-text-color: #e8eaed !important;
    --body-text-color-dark: #e8eaed !important;
    --body-text-color-subdued: #8b8fa3 !important;
    --body-text-color-subdued-dark: #8b8fa3 !important;
    --input-background-fill: #1a1e29 !important;
    --input-background-fill-dark: #1a1e29 !important;
    --input-border-color: #272e3f !important;
    --input-border-color-dark: #272e3f !important;
    --input-text-color: #e8eaed !important;
    --input-text-color-dark: #e8eaed !important;
    --input-placeholder-color: #5d637c !important;
    --input-placeholder-color-dark: #5d637c !important;
    --button-primary-background-fill: #6366f1 !important;
    --button-primary-background-fill-dark: #6366f1 !important;
    --button-primary-text-color: #ffffff !important;
    --button-primary-text-color-dark: #ffffff !important;
    --button-secondary-background-fill: #1a1e29 !important;
    --button-secondary-background-fill-dark: #1a1e29 !important;
    --button-secondary-text-color: #e8eaed !important;
    --button-secondary-text-color-dark: #e8eaed !important;
    --checkbox-label-background-fill: #1a1e29 !important;
    --checkbox-label-background-fill-dark: #1a1e29 !important;
    --radius-md: 8px !important;
    --radius-lg: 12px !important;

    /* Extra theme overrides for lists and dropdowns */
    --background-fill-secondary: #1a1e29 !important;
    --background-fill-secondary-dark: #1a1e29 !important;
    --dropdown-background-fill: #1a1e29 !important;
    --dropdown-background-fill-dark: #1a1e29 !important;
    --dropdown-border-color: #272e3f !important;
    --dropdown-border-color-dark: #272e3f !important;
    --dropdown-text-color: #ffffff !important;
    --dropdown-text-color-dark: #ffffff !important;
    --dropdown-option-background-fill-hover: #202636 !important;
    --dropdown-option-background-fill-hover-dark: #202636 !important;
    --dropdown-option-text-color-hover: #ffffff !important;
    --dropdown-option-text-color-hover-dark: #ffffff !important;

    /* Extra overrides for file previews */
    --file-background-fill: #1a1e29 !important;
    --file-background-fill-dark: #1a1e29 !important;
    --file-border-color: #272e3f !important;
    --file-border-color-dark: #272e3f !important;
    --file-text-color: #ffffff !important;
    --file-text-color-dark: #ffffff !important;
}

body, .gradio-container {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
}

/* Fix radio button layout background white box issue */
.gradio-container .gr-group, 
.gradio-container .gr-box,
.gradio-container .gr-panel,
.gradio-container .form {
    background-color: var(--bg-card) !important;
    border-color: var(--border) !important;
}

/* Force dark styling on Gradio check box and radio labels */
.gradio-container label.selected, 
.gradio-container label:hover {
    background-color: #202636 !important;
    border-color: var(--accent) !important;
    color: #ffffff !important;
}
.gradio-container label {
    background-color: #1a1e29 !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}

/* ── Bulletproof Dropdown & Option List Contrast Overrides ─────────────────── */
.gradio-container select,
.gradio-container select *,
.gradio-container .dropdown,
.gradio-container .dropdown *,
.gradio-container .dropdown-container,
.gradio-container .dropdown-container *,
.gradio-container .dropdown-select,
.gradio-container .dropdown-select *,
.gradio-container .dropdown-menu,
.gradio-container .dropdown-menu *,
.gradio-container .dropdown-options,
.gradio-container .dropdown-options *,
.gradio-container .options,
.gradio-container .options *,
.gradio-container .option,
.gradio-container .option *,
.gradio-container li,
.gradio-container li *,
.gradio-container .selected-item,
.gradio-container .selected-item *,
.gradio-container .dropdown-button,
.gradio-container .dropdown-button * {
    background-color: #1a1e29 !important;
    background: #1a1e29 !important;
    color: #ffffff !important;
    border-color: #272e3f !important;
}

/* Dropdown option items hover */
.gradio-container .option:hover,
.gradio-container .option *:hover,
.gradio-container li:hover,
.gradio-container li *:hover,
.gradio-container .dropdown-options div:hover,
.gradio-container .dropdown-options div *:hover {
    background-color: #202636 !important;
    color: #ffffff !important;
}

/* Dropdown filter search box */
.gradio-container .dropdown-input,
.gradio-container .dropdown-input *,
.gradio-container input[type="text"] {
    background-color: #1a1e29 !important;
    color: #ffffff !important;
    border-color: #272e3f !important;
}

/* ── Bulletproof File Upload & Video Name Contrast Overrides ────────────────── */
.gradio-container .file-preview,
.gradio-container .file-preview *,
.gradio-container .file-preview-card,
.gradio-container .file-preview-card *,
.gradio-container .file-preview-row,
.gradio-container .file-preview-row *,
.gradio-container .file-card,
.gradio-container .file-card *,
.gradio-container .upload-container,
.gradio-container .upload-container *,
.gradio-container .file-preview-container,
.gradio-container .file-preview-container *,
.gradio-container .file-name,
.gradio-container .file-name *,
.gradio-container .file-size,
.gradio-container .file-size *,
.gradio-container .file-size-info,
.gradio-container .file-size-info * {
    background-color: #1a1e29 !important;
    background: #1a1e29 !important;
    color: #ffffff !important;
    border-color: #272e3f !important;
}

/* Muted labels inside file cards */
.gradio-container .file-size,
.gradio-container .file-size *,
.gradio-container .file-size-info,
.gradio-container .file-size-info * {
    color: #8b8fa3 !important;
}

/* ── Bulletproof Available Precision Markdown Overrides ───────────────────── */
.gradio-container .info-box,
.gradio-container .info-box *,
.gradio-container .prose,
.gradio-container .prose * {
    background-color: rgba(99, 102, 241, 0.05) !important;
    color: #ffffff !important;
}

/* Standard inline code blocks inside info displays */
.gradio-container .info-box code,
.gradio-container .prose code {
    background-color: #1a1e29 !important;
    color: #34d399 !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    border: 1px solid #272e3f !important;
}

/* ── Accordion Parameter Panel Override ───────────────────────────────────── */
.gradio-container .accordion,
.gradio-container .accordion *,
.gradio-container .accordion-trigger,
.gradio-container .accordion-trigger * {
    background-color: #14171f !important;
    color: #ffffff !important;
    border-color: #1e2330 !important;
}

/* ── Card Overrides ────────────────────────────────────────────────────── */
.card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 24px !important;
    transition: border-color 0.2s ease, background 0.2s ease;
}
.card:hover {
    background: var(--bg-card-hover) !important;
    border-color: rgba(99, 102, 241, 0.3) !important;
}

/* ── Section Headers (Step Badge Replacement) ───────────────────────────── */
.card-header {
    display: flex !important;
    align-items: center !important;
    gap: 14px !important;
    margin-bottom: 20px !important;
}
.card-icon {
    font-size: 20px !important;
    width: 40px !important;
    height: 40px !important;
    border-radius: 8px !important;
    background: rgba(99, 102, 241, 0.1) !important;
    color: var(--accent) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.card-title {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #ffffff !important;
    margin: 0 !important;
}
.card-subtitle {
    font-size: 12px !important;
    color: var(--text-muted) !important;
    margin: 2px 0 0 0 !important;
}

/* ── Premium Launch/Stop Buttons ───────────────────────────────────────── */
.launch-btn {
    background: var(--accent) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    border-radius: 10px !important;
    padding: 14px 0 !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2) !important;
}
.launch-btn:hover {
    box-shadow: 0 0 24px var(--accent-glow) !important;
    transform: translateY(-1px) !important;
}
.stop-btn {
    background: #dc2626 !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    border-radius: 10px !important;
    padding: 14px 0 !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
    box-shadow: 0 4px 12px rgba(220, 38, 38, 0.2) !important;
}
.stop-btn:hover {
    box-shadow: 0 0 20px rgba(220, 38, 38, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Status Bar ────────────────────────────────────────────────────────── */
.status-bar {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 14px 18px !important;
    font-size: 14px !important;
}

/* ── Premium Metric Cards ──────────────────────────────────────────────── */
.metric-row {
    margin-bottom: 20px !important;
}
.metric-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 12px 16px !important;
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.metric-card:hover {
    border-color: rgba(99, 102, 241, 0.4) !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
}
.metric-icon {
    font-size: 20px !important;
    width: 40px !important;
    height: 40px !important;
    border-radius: 8px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex-shrink: 0 !important;
}
.metric-indigo .metric-icon { background: rgba(99, 102, 241, 0.1) !important; color: #6366f1 !important; }
.metric-emerald .metric-icon { background: rgba(16, 185, 129, 0.1) !important; color: #10b981 !important; }
.metric-amber .metric-icon { background: rgba(245, 158, 11, 0.1) !important; color: #f59e0b !important; }
.metric-rose .metric-icon { background: rgba(244, 63, 94, 0.1) !important; color: #f43f5e !important; }
.metric-sky .metric-icon { background: rgba(14, 165, 233, 0.1) !important; color: #0ea5e9 !important; }

.metric-info {
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
}
.metric-label {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
.metric-val {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    line-height: 1.1 !important;
}
.metric-unit {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    margin-left: 3px !important;
}

/* ── App Header ────────────────────────────────────────────────────────── */
.app-header h1 {
    background: linear-gradient(135deg, #818cf8, #6366f1, #4f46e5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 32px !important;
    font-weight: 800 !important;
    margin: 0 !important;
    letter-spacing: -0.75px;
}
.app-header p {
    color: var(--text-muted) !important;
    font-size: 15px !important;
    margin: 6px 0 0 !important;
}

/* ── Feed Area ─────────────────────────────────────────────────────────── */
.feed-area {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--bg-primary);
}
::-webkit-scrollbar-thumb {
    background: #272e3f;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--accent);
}

footer { display: none !important; }
"""


# ═══════════════════════════════════════════════════════════════════════════
# GPU Telemetry Helper
# ═══════════════════════════════════════════════════════════════════════════
def get_gpu_telemetry():
    """Query nvidia-smi programmatically for GPU load % and memory usage %."""
    try:
        import subprocess
        # Query utilization and memory for the first GPU
        cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
        parts = [p.strip() for p in output.split(",")]
        if len(parts) >= 3:
            gpu_load = float(parts[0])
            gpu_mem = (float(parts[1]) / float(parts[2])) * 100 if float(parts[2]) > 0 else 0.0
            return gpu_load, gpu_mem
    except Exception:
        pass
    return 0.0, 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Metrics HTML builders
# ═══════════════════════════════════════════════════════════════════════════
def make_metric_card(icon, label, value, unit="", color="indigo"):
    return f"""
    <div class="metric-card metric-{color}">
        <div class="metric-icon">{icon}</div>
        <div class="metric-info">
            <div class="metric-label">{label}</div>
            <div class="metric-val">{value}<span class="metric-unit">{unit}</span></div>
        </div>
    </div>
    """


def get_metric_updates(fps=0.0, det=0, tracks=0, cpu=0.0, ram=0.0, device="cpu", gpu_load=0.0, gpu_vram=0.0):
    if device == "cuda":
        return (
            make_metric_card("⚡", "Processing Speed", f"{fps:.1f}", "FPS", "indigo"),
            make_metric_card("🔍", "Detections", f"{det}", "", "emerald"),
            make_metric_card("🛸", "Active Tracks", f"{tracks}", "", "amber"),
            make_metric_card("🚀", "GPU Load", f"{gpu_load:.1f}", "%", "rose"),
            make_metric_card("📼", "GPU VRAM", f"{gpu_vram:.1f}", "%", "sky"),
        )
    else:
        return (
            make_metric_card("⚡", "Processing Speed", f"{fps:.1f}", "FPS", "indigo"),
            make_metric_card("🔍", "Detections", f"{det}", "", "emerald"),
            make_metric_card("🛸", "Active Tracks", f"{tracks}", "", "amber"),
            make_metric_card("💻", "CPU Usage", f"{cpu:.1f}", "%", "rose"),
            make_metric_card("💾", "RAM Usage", f"{ram:.1f}", "%", "sky"),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Callback helpers
# ═══════════════════════════════════════════════════════════════════════════
def on_format_change(format_label):
    """When the user picks a format, update precision dropdown + description."""
    precisions = get_precisions_for_format(format_label)
    desc = get_format_description(format_label)
    avail = get_availability_summary(format_label)
    return (
        gr.update(choices=precisions, value=precisions[0]),
        desc,
        avail,
    )


def on_input_type_change(choice):
    """Show/hide the correct input widget."""
    return (
        gr.update(visible=(choice == "📁 Upload a video file")),
        gr.update(visible=(choice == "🔗 Paste a YouTube link")),
        gr.update(visible=(choice == "📡 Enter a stream URL")),
    )


def toggle_feed(show):
    """Show or hide the live tracking image."""
    return gr.update(visible=show)


def on_start_click():
    """Swap button visibility: hide Start, show Stop."""
    return gr.update(visible=False), gr.update(visible=True)


def on_stop_click(device="cpu"):
    """Signal stop and swap buttons back: hide Stop, show Start, reset metrics, hide video outputs."""
    _stop_event.set()
    resets = get_metric_updates(0.0, 0, 0, 0.0, 0.0, device=device)
    return (
        "⏹️  **Stopped** — tracking was cancelled.",
        gr.update(visible=True),
        gr.update(visible=False),
        *resets,
        gr.update(visible=False),
        gr.update(visible=False),
    )


def on_device_change(device_choice):
    """Dynamically swap fourth and fifth cards between CPU/RAM and GPU Load/VRAM when the device selection changes."""
    if device_choice == "cuda":
        gpu_load, gpu_vram = get_gpu_telemetry()
        return (
            make_metric_card("🚀", "GPU Load", f"{gpu_load:.1f}", "%", "rose"),
            make_metric_card("📼", "GPU VRAM", f"{gpu_vram:.1f}", "%", "sky"),
        )
    else:
        cpu_val = psutil.cpu_percent()
        ram_val = psutil.virtual_memory().percent
        return (
            make_metric_card("💻", "CPU Usage", f"{cpu_val:.1f}", "%", "rose"),
            make_metric_card("💾", "RAM Usage", f"{ram_val:.1f}", "%", "sky"),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Inference Pipeline Orchestrator
# ═══════════════════════════════════════════════════════════════════════════
def run_pipeline(
    input_type,
    video_file,
    yt_url,
    stream_url,
    format_label,
    precision,
    device,
    imgsz,
    conf,
    iou,
    tracker,
):
    """
    Main generator — yields (status_text, frame, fps_html, det_html, track_html, cpu_html, ram_html, download_update, replay_update) tuples.
    Because it's a generator, Gradio streams the outputs live.
    """
    _stop_event.clear()  # reset stop flag at the start of each run

    cpu = lambda: psutil.cpu_percent()
    ram = lambda: psutil.virtual_memory().percent
    fmt_key = label_to_key(format_label)

    # Telemetry values
    cpu_val = cpu()
    ram_val = ram()
    gpu_load_val = 0.0
    gpu_vram_val = 0.0

    if device == "cuda":
        gpu_load_val, gpu_vram_val = get_gpu_telemetry()

    metrics = lambda f, d, t: get_metric_updates(
        fps=f, det=d, tracks=t,
        cpu=cpu_val, ram=ram_val,
        device=device, gpu_load=gpu_load_val, gpu_vram=gpu_vram_val
    )

    # Helper function to reset file & video displays during processing
    reset_outputs = (gr.update(visible=False), gr.update(visible=False))

    yield "⏳  Preparing…", None, *metrics(0.0, 0, 0), *reset_outputs

    # ── Step 1: resolve source ────────────────────────────────────────
    source = None
    if "Upload" in input_type and video_file:
        source = video_file
    elif "YouTube" in input_type and yt_url:
        yield "⬇️  Downloading from YouTube…", None, *metrics(0.0, 0, 0), *reset_outputs
        try:
            source = download_youtube_video(yt_url.strip())
        except Exception as e:
            yield f"❌  YouTube download failed: {e}", None, *metrics(0.0, 0, 0), *reset_outputs
            return
    elif "stream" in input_type.lower() and stream_url:
        source = stream_url.strip()

    if not source:
        yield "❌  No input provided. Please upload a video, paste a link, or enter a stream URL above.", None, *metrics(0.0, 0, 0), *reset_outputs
        return

    # Create a unique output path using the input filename and a timestamp
    source_stem = Path(source).stem if isinstance(source, str) else "stream"
    out_video_path = f"output/tracked_{source_stem}_{int(time.time())}.mp4"

    # ── Step 2: ensure model exists ───────────────────────────────────
    actual_precision = precision if fmt_key != "pt" else "default"

    if not is_model_present(fmt_key, actual_precision):
        base_model = "weights/best.pt"
        if not Path(base_model).exists():
            yield "❌  Base model `weights/best.pt` not found. Cannot build the requested format.", None, *metrics(0.0, 0, 0), *reset_outputs
            return

        yield f"🔨  Building **{format_label}** ({precision}). This may take a while…", None, *metrics(0.0, 0, 0), *reset_outputs
        half = precision == "fp16"
        int8 = precision == "int8"
        log_lines = []
        for line in trigger_model_optimizer(base_model, fmt_key, half=half, int8=int8):
            if _stop_event.is_set():
                yield "⏹️  **Stopped** — model build was cancelled.", None, *metrics(0.0, 0, 0), *reset_outputs
                return
            log_lines.append(line)
            recent = "\n".join(log_lines[-6:])
            yield f"🔨  Building model…\n```\n{recent}\n```", None, *metrics(0.0, 0, 0), *reset_outputs

    # ── Step 3: load model path ───────────────────────────────────────
    model_path = get_model_path(fmt_key, actual_precision)
    if not model_path:
        yield "❌  Could not locate the model after building. Check the weights/ directory.", None, *metrics(0.0, 0, 0), *reset_outputs
        return

    yield f"✅  Model loaded: `{Path(model_path).name}`. Starting tracking…", None, *metrics(0.0, 0, 0), *reset_outputs

    # ── Step 4: stream inference frames ───────────────────────────────
    try:
        for frame, fps, detection_count, track_count, curr_frame_idx, total_frames in infer_video(
            model_path,
            source,
            output_path=out_video_path,
            imgsz=int(imgsz),
            conf=float(conf),
            iou=float(iou),
            tracker=tracker,
            device=device,
        ):
            if _stop_event.is_set():
                yield "⏹️  **Stopped** — tracking was cancelled by user.", frame, *metrics(0.0, 0, 0), *reset_outputs
                return
            
            if total_frames > 0:
                pct = (curr_frame_idx / total_frames) * 100
                status_text = f"🟢  Tracking in progress… [Frame {curr_frame_idx}/{total_frames} — {pct:.1f}%]"
            else:
                status_text = f"🟢  Tracking in progress… [Frame {curr_frame_idx} (Live stream)]"
            
            # Throttle telemetry checks to every 10 frames to maximize inference FPS
            if curr_frame_idx % 10 == 0:
                cpu_val = cpu()
                ram_val = ram()
                if device == "cuda":
                    gpu_load_val, gpu_vram_val = get_gpu_telemetry()
                
            yield status_text, frame, *metrics(fps, detection_count, track_count), *reset_outputs
        
        # When successfully finished, expose the download file and replay video!
        yield "🏁  **Done!** Tracking finished successfully.", None, *metrics(0.0, 0, 0), gr.update(visible=True, value=out_video_path), gr.update(visible=True, value=out_video_path)
    except Exception as e:
        yield f"❌  Inference error: {e}", None, *metrics(0.0, 0, 0), *reset_outputs


# ═══════════════════════════════════════════════════════════════════════════
# Build the Gradio app
# ═══════════════════════════════════════════════════════════════════════════
def build_app():
    theme = gr.themes.Base(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(title="Aerial Guardian") as app:

        # ── Header ────────────────────────────────────────────────────
        gr.HTML("""
        <div class="app-header" style="padding:24px 0 8px">
            <h1>🛸 Aerial Guardian</h1>
            <p>Sleek, high-performance drone object tracking and deep optimization dashboard.</p>
        </div>
        """)

        # ══════════════ TOP ROW — Input + Model side by side ══════════
        with gr.Row(equal_height=True):

            # ── Left: Choose input ────────────────────────────────────
            with gr.Column(scale=1, min_width=340):
                with gr.Group(elem_classes=["card"]):
                    gr.HTML("""
                    <div class="card-header">
                        <div class="card-icon">📥</div>
                        <div>
                            <p class="card-title">Inference Source</p>
                            <p class="card-subtitle">Upload drone footage, link a YouTube video, or stream live RTSP.</p>
                        </div>
                    </div>
                    """)

                    input_type = gr.Radio(
                        choices=[
                            "📁 Upload a video file",
                            "🔗 Paste a YouTube link",
                            "📡 Enter a stream URL",
                        ],
                        value="📁 Upload a video file",
                        label="",
                        show_label=False,
                    )
                    video_upload = gr.File(
                        label="Video file",
                        visible=True,
                        file_types=[".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".ts"],
                    )
                    youtube_url = gr.Textbox(
                        label="YouTube URL",
                        placeholder="https://www.youtube.com/watch?v=…",
                        visible=False,
                    )
                    stream_url = gr.Textbox(
                        label="Stream URL",
                        placeholder="rtsp://camera-ip:554/stream  or  http://…",
                        visible=False,
                    )

            # ── Right: Pick model ─────────────────────────────────────
            with gr.Column(scale=1, min_width=340):
                with gr.Group(elem_classes=["card"]):
                    gr.HTML("""
                    <div class="card-header">
                        <div class="card-icon">🧠</div>
                        <div>
                            <p class="card-title">Model & Runtime Config</p>
                            <p class="card-subtitle">Select a hardware-accelerated runtime and precision.</p>
                        </div>
                    </div>
                    """)

                    format_choices = get_format_choices()
                    
                    with gr.Row():
                        model_format = gr.Dropdown(
                            choices=format_choices,
                            value=format_choices[0],
                            label="Format",
                            interactive=True,
                        )
                        first_precisions = get_precisions_for_format(format_choices[0])
                        precision = gr.Dropdown(
                            choices=first_precisions,
                            value=first_precisions[0],
                            label="Precision",
                            interactive=True,
                        )
                        
                        # Dynamically detect hardware acceleration (CUDA)
                        device_choices = ["cpu"]
                        if torch.cuda.is_available():
                            device_choices.append("cuda")
                            
                        default_device = "cuda" if torch.cuda.is_available() else "cpu"
                        device = gr.Dropdown(
                            choices=device_choices,
                            value=default_device,
                            label="Device",
                            interactive=True,
                        )

                    # Collapsible Advanced Parameters menu
                    with gr.Accordion("⚙️ Additional Inference Parameters", open=False):
                        with gr.Row():
                            imgsz = gr.Slider(
                                minimum=320,
                                maximum=1280,
                                step=32,
                                value=640,
                                label="Image Size (imgsz)",
                            )
                            conf = gr.Slider(
                                minimum=0.01,
                                maximum=1.0,
                                step=0.01,
                                value=0.25,
                                label="Confidence Threshold",
                            )
                        with gr.Row():
                            iou = gr.Slider(
                                minimum=0.01,
                                maximum=1.0,
                                step=0.01,
                                value=0.70,
                                label="IoU Threshold (iou)",
                            )
                            tracker = gr.Dropdown(
                                choices=["bytetrack.yaml", "botsort.yaml"],
                                value="bytetrack.yaml",
                                label="Tracker Config",
                                interactive=True,
                            )

                    format_desc = gr.Markdown(
                        get_format_description(format_choices[0]),
                        elem_classes=["info-box"],
                    )

                    availability = gr.Markdown(
                        get_availability_summary(format_choices[0]),
                        elem_classes=["info-box"],
                    )

        # ══════════════ ACTION BAR — Start / Stop (mutually exclusive) ═
        with gr.Row():
            with gr.Column(scale=1):
                launch_btn = gr.Button(
                    "▶  Start Tracking",
                    elem_classes=["launch-btn"],
                    size="lg",
                    visible=True,
                )
                stop_btn = gr.Button(
                    "⏹  Stop Tracking",
                    elem_classes=["stop-btn"],
                    size="lg",
                    visible=False,
                )

        # ══════════════ RESULTS AREA — Status + Feed + Metrics ════════
        with gr.Row(equal_height=False):

            with gr.Column(scale=1):
                status_bar = gr.Markdown(
                    "Status: **Idle** — waiting for you to start.",
                    elem_classes=["status-bar"],
                )

                # Unified Performance Metrics Row!
                with gr.Row(elem_classes=["metric-row"]):
                    fps_card = gr.HTML(make_metric_card("⚡", "Processing Speed", "0.0", "FPS", "indigo"))
                    det_card = gr.HTML(make_metric_card("🔍", "Detections", "0", "", "emerald"))
                    track_card = gr.HTML(make_metric_card("🛸", "Active Tracks", "0", "", "amber"))
                    if torch.cuda.is_available():
                        gpu_load, gpu_vram = get_gpu_telemetry()
                        cpu_card = gr.HTML(make_metric_card("🚀", "GPU Load", f"{gpu_load:.1f}", "%", "rose"))
                        ram_card = gr.HTML(make_metric_card("📼", "GPU VRAM", f"{gpu_vram:.1f}", "%", "sky"))
                    else:
                        cpu_card = gr.HTML(make_metric_card("💻", "CPU Usage", "0.0", "%", "rose"))
                        ram_card = gr.HTML(make_metric_card("💾", "RAM Usage", "0.0", "%", "sky"))

                show_feed = gr.Checkbox(
                    label="Show live tracking feed",
                    value=False,
                )

                output_image = gr.Image(
                    label="Live Tracking Feed",
                    interactive=False,
                    height=520,
                    elem_classes=["feed-area"],
                    visible=False,
                )

                # New download and replay components!
                download_file = gr.File(
                    label="📥 Download Tracked Video File",
                    visible=False,
                    interactive=False,
                )
                replay_video = gr.Video(
                    label="▶️ Replay Tracked Video",
                    visible=False,
                    interactive=False,
                    autoplay=False,
                )

        # ══════════════ EVENT WIRING ══════════════════════════════════
        input_type.change(
            fn=on_input_type_change,
            inputs=input_type,
            outputs=[video_upload, youtube_url, stream_url],
        )

        model_format.change(
            fn=on_format_change,
            inputs=model_format,
            outputs=[precision, format_desc, availability],
        )

        device.change(
            fn=on_device_change,
            inputs=device,
            outputs=[cpu_card, ram_card],
        )

        # Start click: swap buttons, then run pipeline
        run_event = launch_btn.click(
            fn=on_start_click,
            inputs=[],
            outputs=[launch_btn, stop_btn],
        ).then(
            fn=run_pipeline,
            inputs=[
                input_type,
                video_upload,
                youtube_url,
                stream_url,
                model_format,
                precision,
                device,
                imgsz,
                conf,
                iou,
                tracker,
            ],
            outputs=[
                status_bar,
                output_image,
                fps_card,
                det_card,
                track_card,
                cpu_card,
                ram_card,
                download_file,
                replay_video,
            ],
        ).then(
            # When pipeline finishes (naturally or error), restore Start button
            fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
            inputs=[],
            outputs=[launch_btn, stop_btn],
        )

        # Stop click: signal stop + swap buttons back + reset metrics + hide download/replay
        stop_btn.click(
            fn=on_stop_click,
            inputs=[device],
            outputs=[
                status_bar,
                launch_btn,
                stop_btn,
                fps_card,
                det_card,
                track_card,
                cpu_card,
                ram_card,
                download_file,
                replay_video,
            ],
            cancels=[run_event],
        )

        show_feed.change(
            fn=toggle_feed,
            inputs=show_feed,
            outputs=output_image,
        )

    return app


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=CUSTOM_CSS,
    )
