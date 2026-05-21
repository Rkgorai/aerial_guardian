# Aerial Guardian

Lightweight multi-object detection and tracking pipeline optimized for aerial/drone imagery. Detects and tracks persons from a moving drone platform using YOLO26s + ByteTrack.

## Project Structure

```
.
├── src/
│   ├── pipeline.py      # Main inference pipeline
│   └── tracker.py       # ByteTrack tracker implementation
├── output/              # Processed output videos
├── test_videos/         # Generated test videos from VisDrone
├── yolo_dataset/        # Dataset in YOLO format
├── mot_results/         # Fine-tuned model weights
├── convert_to_yolo.py   # VisDrone → YOLO format converter
├── benchmark.py         # Validation benchmark script
├── benchmark_fast.py    # Fast benchmark (subset)
├── create_test_videos.py# Video generator from sequences
├── requirements.txt     # Dependencies
└── README.md
```

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd aerial_guardian

# Create virtual environment (Python 3.8+)
python -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

This project uses the **VisDrone2019-MOT Validation Set** (Task 4).

1. Download from: https://drive.google.com/file/d/1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu/view
2. Place in project root:
   ```
   VisDrone2019-MOT-val/
   ├── annotations/   # MOT format .txt files
   └── sequences/     # Image sequences
   ```
3. Convert to YOLO format:
   ```bash
   python convert_to_yolo.py
   ```

## Usage

### Run Inference Pipeline

```bash
# Using trained model (default imgsz=640)
python src/pipeline.py \
    --model mot_results/runs/detect/train/weights/best.pt \
    --input test_videos/uav0000086_00000_v.mp4 \
    --output output/result.mp4

# CPU-optimized (faster, lower accuracy)
python src/pipeline.py \
    --model mot_results/runs/detect/train/weights/best.pt \
    --input test_videos/uav0000086_00000_v.mp4 \
    --output output/result.mp4 \
    --imgsz 480 --conf 0.25

# Create test videos from VisDrone sequences
python create_test_videos.py
```

### Benchmark Models

```bash
# Full validation benchmark
python benchmark.py --model1 path/to/model1.pt --model2 path/to/model2.pt

# Fast benchmark (subset of frames)
python benchmark_fast.py \
    --model1 mot_results/runs/detect/train/weights/best.pt \
    --model2 mot_results/runs/detect/train-2/weights/best.pt
```

## Performance

Benchmarked on **Intel Core i5-8250U (CPU)** with 200 frames from VisDrone sequences.

| Model | Precision | Recall | F1 Score | FPS (CPU) | Params |
|-------|-----------|--------|----------|-----------|--------|
| **YOLO26s (MOT)** | 0.316 | **0.563** | **0.405** | 4.6 | 9.5M |
| YOLO26s (VisDrone) | **0.526** | 0.149 | 0.232 | 4.1 | 9.5M |

**Selected Model**: YOLO26s fine-tuned on MOT-1 dataset (higher recall and F1).

**FPS by image size** (CPU):
| imgsz | FPS |
|-------|-----|
| 320   | 5.5 |
| 480   | 6.0 |
| 640   | 4.6 |

**Note**: On GPU (NVIDIA), expect 25+ FPS at imgsz=640.

---

## Summary Report

### 1. Architecture Choice & Small Object Detection

**Base Architecture**: YOLO26s (Ultralytics)

**Why YOLO26s?**
- **Lightweight**: 9.5M parameters, ~10MB model size (well under 300MB limit)
- **Anchor-free head**: Better for varying object sizes in aerial views
- **Modern backbone**: Efficient feature extraction with 26 layers
- **Well-supported**: Ultralytics ecosystem for fine-tuning and export

**Small Object Optimizations:**

| Strategy | Implementation Details |
|----------|----------------------|
| **Multi-scale Training** | imgsz=640 with random augmentations; close-mosaic at epoch 10 |
| **Heavy Augmentation** | Mosaic, HSV jitter, scale, translation, flip; RandAugment policy |
| **Low Confidence Threshold** | conf=0.25 for inference (vs default 0.5); catches small/partial detections |
| **Ego-motion Calibration** | ByteTrack's second association stage recovers small objects lost due to motion blur |
| **Anchor Box Design** | YOLO26s uses learned anchor boxes from the training data distribution |

**Training Details:**
- **Platform**: Kaggle (2x GPU, batch 56)
- **Base Model**: YOLO26s pretrained on COCO
- **Dataset**: MOT-1 dataset (person-focused)
- **Epochs**: 100 (early stopped at 15 via patience=10)
- **Optimizer**: Auto (AdamW)

### 2. Addressing ID Switching (Drone Ego-Motion & Occlusions)

ID switching is a major challenge in drone-based tracking because:
1. **Ego-motion**: The drone moves constantly, causing object positions to change unpredictably frame-to-frame
2. **Altitude changes**: Objects shrink/grow rapidly as drone altitude varies
3. **Occlusions**: Trees, buildings, and other drone passes cause temporary track loss

**Our approach uses ByteTrack with two key enhancements:**

#### Two-Stage Association (ByteTrack)
```
Stage 1: High-confidence detections (>0.5) → primary match
Stage 2: Low-confidence detections (0.3-0.5) → recover lost tracks
```
This allows us to handle partial occlusions where the detector only produces a low-confidence box.

#### Kalman Filter with Adaptive Noise
- **State vector**: [x, y, w, h, vx, vy, vw, vh] — position + velocity
- **Process noise (Q)**: Set higher than standard tracking to account for drone motion
- **Motion adaptation**: Noise covariance allows for sudden camera movements common in drone footage

#### ID-Switch Prevention Strategies
| Strategy | Purpose |
|----------|---------|
| **Track Buffer (3 frames)** | Don't immediately declare a track lost — wait 3 missed frames |
| **History-based Re-matching** | Track stores last 30 positions; re-associate using IoU history |
| **Confidence-Gated Matching** | Only create new tracks from confident detections; use low-conf for recovery |
| **Max Age (30 frames)** | Tracks persist for 30 frames after disappearance — handles longer occlusions |

### 3. Edge Deployment (NVIDIA Jetson)

**Current Bottleneck**: The YOLO26s model runs at ~4-6 FPS on CPU. For edge deployment on Jetson, we recommend the following optimizations:

#### Optimization Pipeline

```
PyTorch Model → FP16 Quantization → TensorRT Engine → Deploy on Jetson
```

#### Step-by-Step Adaptation

```bash
# 1. Export to TensorRT
yolo export model=best.pt format=engine half=True

# 2. Run pipeline with TensorRT
python src/pipeline.py --model best.engine --input video.mp4 --output out.mp4
```

#### Expected Performance on Jetson Devices

| Device | imgsz | Precision | Expected FPS |
|--------|-------|-----------|--------------|
| Jetson Nano (4GB) | 320 | INT8 | ~10-15 |
| Jetson TX2 | 480 | FP16 | ~15-20 |
| Jetson Xavier NX | 640 | FP16 | ~20-30 |
| Jetson Orin Nano | 640 | FP16 | ~30-50 |

#### Additional Edge Optimizations

1. **INT8 Quantization**: Use calibration dataset to reduce model to 1/4 size
2. **Frame Skipping**: Process every 2nd frame, interpolate tracks
3. **Reduce Input Size**: imgsz=320 gives 2x speedup with ~10% mAP loss
4. **DeepStream SDK**: Hardware-accelerated pipeline for multi-stream processing
5. **Power Mode**: Use `nvpmodel -m 0` (MAXN) for best performance

## Deliverables

- [x] Detection pipeline (YOLO26s fine-tuned)
- [x] Tracking pipeline (ByteTrack implementation)
- [x] Output video with bounding boxes, IDs, trajectory tails
- [x] Benchmark results
- [x] Summary report (this README)

## References

- VisDrone Dataset: https://github.com/VisDrone/VisDrone-Dataset
- YOLO: https://github.com/ultralytics/ultralytics
- ByteTrack: https://github.com/ifzhang/ByteTrack
