# 🛡️ Aerial Guardian: Edge-Optimized Aerial Object Tracking

Lightweight, high-throughput multi-object detection and tracking pipeline optimized for aerial imagery. Designed specifically for tracking "Persons" from moving drone platforms under ego-motion, occlusions, and scale changes, with an end-to-end edge deployment optimization suite.

> **Target Benchmark**: Developed for the VisDrone MOT challenge to accurately track persons from a moving drone while achieving real-time execution speeds on lightweight computing hardware.

---

## 📥 Pre-trained Models & UI Management

The final fine-tuned PyTorch model (`mot_visdrone_finetuned.pt`), highly optimized for person detection on the VisDrone dataset, can be easily managed directly from the Web UI:

* **Automatic Download**: If no models are detected on startup, the UI provides a one-click button to automatically download the fine-tuned PyTorch weights from Google Drive directly into your `weights/` directory.
* **Custom Model Upload**: You can also manually upload your own `.pt` YOLO model files directly via the UI's inline upload button, which will automatically activate your custom model for inference.

Alternatively, you can download the default model manually here:
**[Download Fine-tuned YOLO Model](https://drive.google.com/file/d/1GuKD-B_mH8sCiQMK25qeLnRdp-9CJnde/view?usp=sharing)**

---

## 🚀 Native Installation & Setup

Because compiling advanced edge formats (TensorRT, OpenVINO, NCNN) often requires specific Python versions and native build headers, we highly recommend using a local virtual environment with **Python 3.10** or **Python 3.12**.

Choose one of the three industry-standard ways below to set up your environment:

### Option A: Using `uv` (Fastest, Recommended ⚡)
`uv` is extremely fast and can automatically manage Python versions.
```bash
# 1. Create a Python 3.10 virtual environment
uv venv env --python 3.10

# 2. Activate the environment
source env/bin/activate

# 3. Install all dependencies and the local package
uv pip install -r requirements.txt
uv pip install -e .
```

### Option B: Using standard Python `venv`
```bash
# 1. Create environment (ensure python 3.10 or 3.12 is installed natively)
python3.10 -m venv env

# 2. Activate the environment
source env/bin/activate

# 3. Upgrade pip and install dependencies + package
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Option C: Using `conda`
```bash
# 1. Create a Conda environment with Python 3.10
conda create -n aerial_guardian python=3.10 -y

# 2. Activate the environment
conda activate aerial_guardian

# 3. Install all dependencies and the local package
pip install -r requirements.txt
pip install -e .
```

---

## 📊 Summary Report: Tackling Drone Challenges

Based on the core challenges of aerial object tracking, here is the architectural and engineering approach utilized by Aerial Guardian:

### 1. Choice of Architecture & Small Object Detection
Drones capture subjects at extremely high altitudes, resulting in tiny pixel footprints for targets like persons. We utilized the **YOLOv8** architecture as a base, fine-tuning it heavily on the VisDrone dataset (`mot_visdrone_finetuned.pt`). YOLOv8 features an anchor-free detection head and multi-scale feature pyramids (FPN+PAN), which allows it to fuse high-resolution spatial details with deep semantic features. This maximizes small object recall while maintaining the minimal parameter count required for drone payloads.

### 2. Addressing "ID Switching" & Ego-Motion
Significant camera motion (ego-motion) is the primary cause of ID switching in drone footage. We mitigate this using a dual-layered tracking approach:
* **Global Motion Compensation (GMC)**: Utilizing BoT-SORT, the pipeline extracts ORB keypoints from the background to estimate camera homography, mathematically warping the previous frame's bounding box tracks to the current frame before Kalman filtering. This neutralizes drone drift.
* **Low-Confidence Association**: BYTE association logic is heavily utilized to recover temporarily occluded targets (e.g., a person walking under a tree) by retaining low-confidence detection boxes that traditional trackers normally discard.

### 3. Edge Hardware Adaptation
For physical deployment on edge devices like the NVIDIA Jetson Nano/Orin:
* **Automated Format Compilation**: The PyTorch model is systematically exported to TensorRT (`.engine`), utilizing FP16 or INT8 quantization to maximize hardware Tensor Core utilization while keeping the model footprint under 300MB.
* **Vectorized Overhead**: Tracker matching logic (IoU matrices) is heavily vectorized using NumPy broadcasting to prevent CPU bottlenecking on low-power ARM CPUs.
* **Hardware Video Encoding**: The tracking loop offloads video writing to FFmpeg NVENC (hardware-accelerated h264), freeing up the CPU for Kalman filtering.

---

## 📈 VisDrone Validation Benchmark Report

> ⚠️ **Test Environment**: All benchmarks listed below were executed on a **Google Colab Tesla T4 GPU**.

Performance metrics evaluated directly on the VisDrone Validation Dataset. The targeted fine-tuning yielded massive gains in both mAP accuracy and execution speed.

### Accuracy Metrics
| Model | mAP50 | mAP50-95 | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| `yolo26s.pt` | 0.2994 | 0.1388 | 0.5232 | 0.4372 |
| `yolov8s.pt` | 0.3606 | 0.1518 | 0.5164 | 0.4841 |
| `mot_finetuned.pt` | 0.3468 | 0.1478 | 0.4591 | 0.4247 |
| **`mot_visdrone_finetuned.pt`** | **0.4685** | **0.2031** | **0.4882** | **0.5200** |

### Latency & Throughput Metrics
| Model | FPS | Avg ms | P95 ms | P99 ms |
| :--- | :---: | :---: | :---: | :---: |
| `yolo26s.pt` | 86.29 | 11.59 | 12.27 | 14.25 |
| `yolov8s.pt` | 112.55 | 8.88 | 10.39 | 14.06 |
| `mot_finetuned.pt` | 83.58 | 11.96 | 13.10 | 13.78 |
| **`mot_visdrone_finetuned.pt`**| **120.86**| **8.27** | **9.04** | **10.64** |

---

## ⚡ Edge Format Evaluation Data

The pipeline natively converts models into numerous edge formats. Below is the automated benchmark extracted directly from `aerial_guardian/evaluation/benchmark_results/results.csv`, showcasing execution speeds across formats natively deployed on a **Google Colab Tesla T4 GPU**:

| Model Format | Precision | File Size (MB) | Avg Latency (ms) | Avg FPS | P95 Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `best.engine` (TensorRT) | INT8 | 13.52 MB | 3.89 ms | 257.00 | 4.06 ms |
| `best.engine` (TensorRT) | FP16 | 22.81 MB | 6.77 ms | 147.50 | 8.81 ms |
| `best.onnx` (ONNX GPU) | FP16 | 21.65 MB | 10.44 ms | 95.70 | 14.33 ms |
| `best.pt` (PyTorch) | FP32 | 21.45 MB | 10.51 ms | 95.14 | 11.42 ms |
| `best_int8_openvino` | INT8 | 11.16 MB | 224.01 ms | 4.46 | 287.52 ms |
| `best_ncnn_model` | FP16 | 21.40 MB | 340.00 ms | 2.94 | 419.51 ms |
| `best_full_integer_quant` (TFLite) | INT8 | 10.98 MB | 507.16 ms | 1.97 | 670.80 ms |

---

## 🧩 Supported Model Formats Explained

The Aerial Guardian project is designed to run inference not just in standard PyTorch, but optimized heavily for target environments. 

1. **PyTorch (`.pt`)**: The standard training and development format. Highly flexible but typically runs heavier inference payloads than compiled formats.
2. **ONNX (`.onnx`)**: Open Neural Network Exchange. A universal interoperability format. Highly recommended for standard server GPU deployments (`onnxruntime-gpu`) as it offers excellent plug-and-play speed with minimal environment setup.
3. **TensorRT (`.engine`)**: NVIDIA's ultimate optimization engine. Extremely fast natively on NVIDIA Edge devices (like Jetson platforms) and datacenter GPUs (like Tesla T4). Compiling to INT8 or FP16 activates NVIDIA Tensor Cores for maximum throughput. Engine files are hardware-specific (they must be compiled on the target GPU architecture).
4. **OpenVINO**: Intel's optimization toolkit. This is the absolute best format for edge devices without a dedicated GPU that must rely solely on an Intel CPU or integrated GPU (iGPU).
5. **NCNN**: Tencent's highly optimized neural network inference framework tailored for mobile platforms. Excellent for Android or ARM-based architectures without heavy proprietary libraries.
6. **TFLite (`.tflite`)**: TensorFlow Lite. Ideal for embedded environments and mobile edge devices. We support INT8 quantization which massively shrinks the model to ~10MB for deeply constrained environments.

---

## 🛠️ Project Usage & Commands

### 1. Converting Models to Edge Formats
The project includes a unified model optimizer to cross-compile PyTorch `.pt` files into all edge formats automatically.

```bash
# Automated compilation: Builds all 12 combinations (onnx, tensorrt, openvino, tflite, ncnn)
python -m aerial_guardian.export.model_optimizer --model weights/mot_visdrone_finetuned.pt --formats all

# Custom export targeting specific formats and precisions
python -m aerial_guardian.export.model_optimizer --model weights/mot_visdrone_finetuned.pt --formats onnx,tensorrt --half
```

### 2. CLI Video Tracking
Process raw aerial videos frame-by-frame and export annotated outputs directly from the command line.

```bash
# Basic tracking using the custom pipeline and PyTorch model
python -m aerial_guardian.tracking.pipeline \
    --model weights/mot_visdrone_finetuned.pt \
    --input data/test_videos/drone_clip.mp4 \
    --output output/result.mp4

# Run with Ultralytics built-in tracker and hardware FFmpeg video encoding
python -m aerial_guardian.tracking.pipeline \
    --model weights/onnx/fp16/best.onnx \
    --input data/test_videos/drone_clip.mp4 \
    --output output/result_fast.mp4 \
    --video_encoder ffmpeg \
    --tracker_type ultralytics
```

#### Complete CLI Flag Explanations:
| Parameter | Default | Description |
| :--- | :---: | :--- |
| `--model` | *(required)* | Path to your detection weights (`.pt`, `.onnx`, `.engine`, etc.) |
| `--input` | *(required)* | Path to your input video file to be tracked. |
| `--output` | `output.mp4` | Path to save the annotated tracking video. Setting this to `none` or `""` will bypass video writing entirely, allowing you to benchmark raw tracking speed without disk I/O bottlenecking. |
| `--video_encoder` | `opencv` | Options are `opencv` or `ffmpeg`. Using `ffmpeg` will attempt to use NVENC hardware acceleration (highly recommended if running on Colab/CUDA). |
| `--tracker_type` | `custom` | Options are `custom` or `ultralytics`. `custom` utilizes Aerial Guardian's independently decoupled tracker components. `ultralytics` utilizes YOLO's bundled internal tracker. |
| `--conf` | `0.15` | Confidence threshold for bounding box detection. Decrease this if you want higher recall on extremely small, blurry subjects. |
| `--iou` | `0.5` | Intersection over Union (IoU) threshold for Non-Maximum Suppression (NMS). |
| `--imgsz` | `640` | Input resolution inference scale. |
| `--tracker` | `bytetrack.yaml` | Tracker algorithm configuration file. Acceptable options are `bytetrack.yaml` (faster) or `botsort.yaml` (includes camera motion compensation and appearance re-ID, slightly slower). |

### 3. Launching the Gradio Web UI
The project features a premium, responsive Web UI to visually track targets, switch between precision formats, and analyze telemetry metrics live.

```bash
# Start the Gradio Web Dashboard
python app.py
```

**How to use the UI**:
1. Open the provided Local URL in your browser (e.g., `http://localhost:7860`).
2. **Missing Model?**: If no model is found, use the prominent Download button to fetch the default MOT+VisDrone finetuned model, or click the 📤 Upload icon next to the Format selector to upload your own.
3. **Upload** your drone video file, paste a YouTube link, or provide an RTSP stream.
4. Select your desired **Model Format** (e.g., PyTorch, TensorRT, ONNX). If the edge format isn't compiled yet, the system will automatically build it for you!
5. Tweak the tracking thresholds (Confidence, IoU) and select the tracker algorithm (`botsort` or `bytetrack`).
6. Click **Start Tracking**. 
7. A real-time progress bar will show the completion status. Once finished, you can replay the annotated tracked video directly in the browser and download the `.mp4` file!
