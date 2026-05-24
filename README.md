# 🛡️ Aerial Guardian: Edge-Optimized Aerial Object Tracking

Lightweight, high-throughput multi-object detection and tracking pipeline optimized for aerial imagery. Designed specifically for tracking "Persons" from moving drone platforms under ego-motion, occlusions, and scale changes, with an end-to-end edge deployment optimization suite.

> **Target Benchmark**: Developed for the VisDrone MOT challenge, achieving real-time execution speeds (up to **240+ FPS** on edge GPUs and **7+ FPS** on standard host CPUs) under a strict **300MB footprint budget**.

---

## ✨ Key Capabilities

* **YOLO26s Target Detection** — Fine-tuned specifically for person detection in high-altitude aerial imagery (~10MB footprint, 9.5M params).
* **Stabilized BoT-SORT & BYTE Tracking** — Dual association tracker pipeline with coordinate-space stabilized Kalman XYWH filtering.
* **Camera Ego-Motion Correction** — ORB keypoint extraction and RANSAC homography estimation (GMC) to prevent tracking drift during rapid drone movement.
* **Appearance Re-identification** — ResNet18 Cosine distance embeddings with HSV color histogram fallbacks for tracking continuity through occlusions.
* **Unified Telemetry HUD Visualization** — Overlays format tags, input resolution, inference latency (ms), and frame-by-frame FPS badges natively.
* **Automated Multi-Format Edge Exporter** — Systematically compiles weights into **12 standard edge formats** (ONNX, TensorRT, OpenVINO, TFLite, NCNN) at all precision ranges (`FP32`, `FP16`, `INT8`).
* **Recursive & Protected Benchmarking** — Multi-format performance analyzer with recursive scanning, package directory exclusions, and raw JSON/CSV data sheets exports.
* **Flexible Native Setup** — Optimized for universal execution natively across `uv`, standard `pip`, or `conda` environments.

---

## 📁 Repository Directory Structure

```
aerial_guardian/
├── src/
│   ├── tracking_pipeline.py     # Main end-to-end tracking pipeline
│   ├── model_optimizer.py       # Automated multi-format export compiler
│   ├── edge_benchmarker.py      # Recursive benchmark scanner & reporter
│   ├── trackers/                # Stabilized BoT-SORT & BYTE trackers
│   │   ├── __init__.py          # Tracker creation factory
│   │   ├── basetrack.py         # Base track representations
│   │   ├── byte_tracker.py      # BYTETracker: Fast IoU tracker
│   │   ├── bot_sort.py          # BOTSORT: BYTE + ReID + GMC homography
│   │   ├── gmc.py               # Global Motion Compensation (ORB + RANSAC)
│   │   ├── reid.py              # Appearance Re-ID CNN + HSV histograms
│   │   └── cfg/                 # Ultralytics-compatible YAML configs
│   └── utils/                   # Clean repository helpers module
│       ├── __init__.py
│       ├── format_converter.py  # VisDrone MOT to YOLO converter
│       ├── video_generator.py   # Frames image sequence to MP4 video builder
│       ├── legacy_benchmark.py  # Base validation benchmark script
│       └── legacy_benchmark_fast.py # Base fast validator script
├── results/                     # Benchmark CSV/JSON data sheets exports
│   ├── cpu_benchmark_results.json
│   └── cpu_benchmark_results.csv
├── requirements.txt             # CPU/GPU python package requirements
└── README.md                    # Core project documentation
```

---

## 🚀 Native Installation & Setup

Since pre-release CPython versions (like Python 3.14) do not yet have stable compiled wheels for deep learning backends (like TensorFlow, ONNX Runtime, OpenVINO, or NCNN), you **must** set up a local virtual environment using a stable Python version (like **Python 3.10** or **Python 3.12**).

Choose one of the three industry-standard ways below to set up your environment natively:

### Option A: Using `uv` (Fastest, Recommended ⚡)
`uv` is extremely fast and can automatically manage Python versions. Even if your host system only has Python 3.14, `uv` will automatically download and install Python 3.10/3.12 for you:
```bash
# 1. Create a Python 3.10 virtual environment
uv venv env --python 3.10

# 2. Activate the environment
source env/bin/activate

# 3. Install all edge dependencies in seconds
uv pip install -r requirements.txt
```

### Option B: Using standard Python `venv`
```bash
# 1. Create environment (ensure python 3.10 or 3.12 is installed on your system)
python3.10 -m venv env

# 2. Activate the environment
source env/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Option C: Using `conda`
```bash
# 1. Create a Conda environment with Python 3.10
conda create -n aerial_guardian python=3.10 -y

# 2. Activate the environment
conda activate aerial_guardian

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## ⚙️ Edge Model Optimization Matrix (`src/model_optimizer.py`)

The exporter supports a fully automated matrix mode to compile all quantizations without manual scripting, including **smart duplicate checks** to skip existing files and a **`--force` overwrite bypass**.

```bash
# Automated compilation: Builds all 12 formats & precision combinations (onnx, tensorrt, openvino, tflite, ncnn)
python src/model_optimizer.py --model model/best.pt --formats all

# Force re-compile and overwrite existing formats
python src/model_optimizer.py --model model/best.pt --formats all --force

# Custom export targeting specific formats
python src/model_optimizer.py --model model/best.pt --formats onnx,tflite --half
```

### Structured Output Directories:
Models are automatically reorganized into structured folders relative to your base weights checkpoint:
* **ONNX FP32** ➔ `model/onnx/fp32/best.onnx`
* **TensorRT FP16** ➔ `model/tensorrt/fp16/best.engine`
* **OpenVINO INT8** ➔ `model/openvino/int8/best_int8_openvino_model/`
* **TFLite INT8** ➔ `model/tflite/int8/best_saved_model/`

---

## 📊 Comparative Performance Reports (`src/edge_benchmarker.py`)

The recursive benchmark scanner recursively crawls your structured directories (automatically ignoring library dependencies, git caches, and system folders) to generate a Markdown table and export data sheets.

```bash
# Run recursive discovery benchmark and export results
python src/edge_benchmarker.py \
    --model-dir model/ \
    --save-json results/cpu_benchmark_results.json \
    --save-csv results/cpu_benchmark_results.csv
```

### Benchmark Summary (Google Colab GPU vs. CPU)

| Model Format | Precision | File Size (MB) | Avg Latency (ms) | Avg FPS | P95 Latency (ms) | Target Hardware |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`best.engine` (TensorRT)** | **INT8** | 13.52 MB | **4.80 ms** | **208.27** | **5.57 ms** | NVIDIA GPU (Edge/Jetson) |
| **`best.engine` (TensorRT)** | **FP16** | 22.82 MB | **6.49 ms** | **154.08** | **7.83 ms** | NVIDIA GPU (Edge/Jetson) |
| **`best.pt` (PyTorch Baseline)** | **FP32** | 21.46 MB | 10.70 ms | 93.49 | 11.09 ms | GPU / Server Baseline |
| **`best.onnx` (ONNX GPU)** | **FP16** | 21.65 MB | 9.70 ms | 103.09 | 13.85 ms | Universal GPU (CUDA) |
| **`best_int8_openvino`** | **INT8** | 11.17 MB | 222.76 ms | 4.49 | 291.12 ms | Intel CPU / iGPU |
| **`best_float16.tflite`** | **FP16** | 21.38 MB | 469.87 ms | 2.13 | 602.06 ms | Mobile / Embedded (ARM) |

---

## 🎯 Running the Tracking Pipeline (`src/tracking_pipeline.py`)

Process raw aerial video feeds frame-by-frame, execute the tracker, draw bounding boxes, IDs, and trajectory tails, and output an annotated video.

```bash
# Default: runs ByteTrack, imgsz=640, conf=0.15
python src/tracking_pipeline.py \
    --model model/onnx/fp32/best.onnx \
    --input test_videos/uav0000086_00000_v.mp4 \
    --output output/result_onnx.mp4

# Run BoT-SORT (GMC homography ego-motion compensation + appearance Re-ID)
python src/tracking_pipeline.py \
    --model model/tensorrt/fp16/best.engine \
    --input test_videos/uav0000086_00000_v.mp4 \
    --output output/result_botsort.mp4 \
    --tracker botsort.yaml
```

### CLI Arguments:

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `--model` | *(required)* | Path to detection weights (`.pt`, `.onnx`, `.engine`, etc.) |
| `--input` | *(required)* | Path to input aerial video file or image sequence |
| `--output` | `output.mp4` | Path to save the annotated video result |
| `--conf` | `0.15` | Target confidence threshold (set lower for high recall) |
| `--iou` | `0.5` | Non-Maximum Suppression (NMS) IoU threshold |
| `--imgsz` | `640` | Resolution for inference scaling |
| `--tracker` | `bytetrack.yaml` | Tracker configuration file name (`bytetrack.yaml` or `botsort.yaml`) |

---

## 📚 References

* **ByteTrack**: Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box* — https://github.com/ifzhang/ByteTrack
* **BoT-SORT**: Aharon et al., *BoT-SORT: Robust Associations Multi-Pedestrian Tracking* — https://github.com/NirAharon/BoT-SORT
* **VisDrone MOT**: https://github.com/VisDrone/VisDrone-Dataset
