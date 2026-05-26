# 🛡️ Aerial Guardian: Edge-Optimized Aerial Object Tracking

Lightweight, high-throughput multi-object detection and tracking pipeline optimized for aerial imagery. Designed specifically for tracking "Persons" from moving drone platforms under ego-motion, occlusions, and scale changes, with an end-to-end edge deployment optimization suite.

> **Target Benchmark**: Developed for the VisDrone MOT challenge to accurately track persons from a moving drone while achieving real-time execution speeds on lightweight computing hardware.

---

## 📥 Pre-trained Models

The final fine-tuned PyTorch model (`mot_visdrone_finetuned.pt`), highly optimized for person detection on the VisDrone dataset, can be downloaded directly here:
**[Download Fine-tuned YOLO Model](https://drive.google.com/file/d/1GuKD-B_mH8sCiQMK25qeLnRdp-9CJnde/view?usp=sharing)**

---

## 📊 Summary Report: Tackling Drone Challenges

Based on the core challenges of aerial object tracking, here is the architectural and engineering approach utilized by Aerial Guardian:

### 1. Choice of Architecture & Small Object Detection
Drones capture subjects at extremely high altitudes, resulting in tiny pixel footprints for targets like persons. We utilized the **YOLOv8** architecture as a base, fine-tuning it heavily on the VisDrone dataset (`mot_visdrone_finetuned.pt`). YOLOv8 features an anchor-free detection head and multi-scale feature pyramids (FPN+PAN), which allows it to fuse high-resolution spatial details with deep semantic features. This maximizes small object recall while maintaining the minimal parameter count required for drone payloads.

### 2. Addressing "ID Switching" & Ego-Motion
Significant camera motion (ego-motion) is the primary cause of ID switching in drone footage. We mitigate this using a dual-layered tracking approach:
* **Global Motion Compensation (GMC)**: Utilizing BoT-SORT, the pipeline extracts ORB keypoints from the background to estimate camera homography, mathematically warping the previous frame's bounding box tracks to the current frame before Kalman filtering. This neutralizes drone drift.
* **Low-Confidence Association**: BYTE association logic is heavily utilized to recover temporarily occluded targets (e.g., a person walking under a tree) by retaining low-confidence detection boxes that traditional trackers normally discard.

### 3. Edge Hardware Adaptation (NVIDIA Jetson)
For physical deployment on edge devices like the NVIDIA Jetson Nano or Orin, the pipeline balances inference speed and precision:
* **Automated Format Compilation**: The PyTorch model is systematically exported to TensorRT (`.engine`), utilizing FP16 or INT8 quantization to maximize hardware Tensor Core utilization while keeping the model footprint well under 300MB.
* **Vectorized Overhead**: Tracker matching logic (IoU matrices) is heavily vectorized using NumPy broadcasting to prevent CPU bottlenecking on low-power ARM CPUs.
* **Hardware Video Encoding**: The tracking loop offloads video writing to FFmpeg NVENC (hardware-accelerated h264), freeing up the CPU for Kalman filtering.

---

## 📈 VisDrone Validation Benchmark Report

Performance metrics evaluated on the VisDrone Validation Dataset using various models. The targeted fine-tuning yielded significant gains.

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

The pipeline natively converts models into numerous edge formats. Below is the automated benchmark extracted directly from `aerial_guardian/evaluation/benchmark_results/results.csv`, showcasing execution speeds across formats:

| Model Format | Precision | File Size (MB) | Avg Latency (ms) | Avg FPS | P95 Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `best.engine` (TensorRT) | INT8 | 13.52 MB | 3.89 ms | 257.00 | 4.06 ms |
| `best.engine` (TensorRT) | FP16 | 22.81 MB | 6.77 ms | 147.50 | 8.81 ms |
| `best.onnx` (ONNX GPU) | FP16 | 21.65 MB | 10.44 ms | 95.70 | 14.33 ms |
| `best.pt` (PyTorch) | FP32 | 21.45 MB | 10.51 ms | 95.14 | 11.42 ms |
| `best_int8_openvino` | INT8 | 11.16 MB | 224.01 ms | 4.46 | 287.52 ms |
| `best_ncnn_model` | FP16 | 21.40 MB | 340.00 ms | 2.94 | 419.51 ms |
| `best_full_integer_quant` (TFLite) | INT8 | 10.98 MB | 507.16 ms | 1.97 | 670.80 ms |

*(Note: CPU-based formats like OpenVINO, TFLite, and NCNN execute significantly slower than GPU-accelerated formats like TensorRT and ONNX on the same machine).*

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

### 3. Launching the Gradio Web UI
The project features a premium, responsive Web UI to visually track targets, switch between precision formats, and analyze telemetry metrics live.

```bash
# Start the Gradio Web Dashboard
python app.py
```

**How to use the UI**:
1. Open the provided Local URL in your browser (e.g., `http://localhost:7860`).
2. **Upload** your drone video file, paste a YouTube link, or provide an RTSP stream.
3. Select your desired **Model Format** (e.g., PyTorch, TensorRT, ONNX). If the edge format isn't compiled yet, the system will automatically build it for you!
4. Tweak the tracking thresholds (Confidence, IoU) and select the tracker algorithm (`botsort` or `bytetrack`).
5. Click **Start Tracking**. 
6. A real-time progress bar will show the completion status. Once finished, you can replay the annotated tracked video directly in the browser and download the `.mp4` file!
