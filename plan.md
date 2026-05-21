# The Aerial Guardian - Detailed Implementation Plan

## Project Overview
Build an end-to-end drone video processing pipeline that detects and tracks "Persons" from aerial imagery, optimized for small objects on moving platforms.

---

## 1. Dataset & Data Preparation

### 1.1 Source
- **Dataset**: VisDrone MOT Validation Set (Task 4)
- **Target Class**: Persons only
- **Download**: https://drive.google.com/file/d/1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu/view

### 1.2 Local Preparation
- Extract video frames to image sequence
- Convert VisDrone annotations to YOLO format (txt files with class, x_center, y_center, width, height)
- Structure:
  ```
  dataset/
  ├── images/
  │   └── val/
  ├── labels/
  │   └── val/
  └── videos/
  ```

---

## 2. Model Architecture

### 2.1 Detection: YOLOv8n (Nano)

| Attribute | Value |
|-----------|-------|
| Model Size | ~6 MB |
| Parameters | 3.2M |
| mAP on VisDrone | ~30-35% (base) |
| Inference Speed | ~1ms on GPU, ~10ms on CPU |

**Why YOLOv8n?**
- Lightest YOLOv8 variant - fits easily under 300MB constraint
- Anchor-free detection head - better for varying object sizes
- Built-in data augmentation for small objects
- Easy fine-tuning on custom dataset

### 2.2 Small Object Optimizations
1. **Multi-scale training**: Train on 640x640, 480x480, 320x320
2. **Image augmentation**: Mosaic, MixUp, Copy-paste for small objects
3. **Enhanced FPN**: Use P2/P3/P4 feature pyramids for multi-scale detection
4. **Lower confidence threshold**: 0.1 instead of default 0.25 (for small/partial objects)

---

## 3. Tracking: ByteTrack + Enhanced Kalman Filter

### 3.1 ByteTrack Implementation
- **First association**: High confidence detections → tracks (IoU matching)
- **Second association**: Low confidence detections → recover lost tracks
- **Benefits**: Handles occlusions better than DeepSORT

### 3.2 Drone Ego-Motion Compensation
1. **Adaptive Kalman Filter**:
   - Process noise (Q) increases when frame-to-frame motion is high
   - Accounts for camera shake/rapid movement

2. **Motion-Aware Matching**:
   - Larger IoU threshold for high-motion frames
   - Shorter trajectory matching in static scenes

3. **ID Switch Mitigation**:
   - Buffer of 3 frames before declaring track lost
   - Re-detection using appearance features if IoU fails
   - Track history for re-matching after occlusion

---

## 4. Training Pipeline (Colab/Kaggle)

### 4.1 Configuration
```yaml
# training_config.yaml
model: yolov8n
data: visdrone_person.yaml
epochs: 50
imgsz: 640
batch: 16
optimizer: AdamW
lr0: 0.001
augmentation: true
mosaic: 1.0
mixup: 0.1
copy_paste: 0.1
patience: 10
save_period: 10
```

### 4.2 Training Strategy
- **Epochs**: 50 (with early stopping)
- **Image Size**: Start 480, increase to 640
- **Augmentation**: Heavy mosaic/mixup for small object robustness
- **Output**: `best.pt` (~6-10MB after fine-tuning)

---

## 5. Inference Pipeline (Local)

### 5.1 Pipeline Architecture
```
Input Video → Frame Extraction → YOLOv8 Detection → ByteTrack → Visualize → Output Video
```

### 5.2 Visualization
- Bounding boxes with class label
- Unique ID overlay per track
- Trajectory tail (last 30 frames)
- FPS counter overlay

### 5.3 Code Structure
```
src/
├── detect.py          # YOLO inference wrapper
├── track.py           # ByteTrack integration
├── visualize.py       # Drawing boxes + trajectories
├── pipeline.py        # Main end-to-end pipeline
└── utils.py           # Helpers
```

---

## 6. Optimization & Benchmarking

### 6.1 Performance Targets

| Metric | Target |
|--------|--------|
| Model Size | < 300 MB |
| FPS (GPU) | > 25 |
| FPS (CPU) | > 5 |
| MOTA | > 0.3 |

### 6.2 Optimization Techniques
1. **FP16 inference** (if GPU supports)
2. **Batch processing** for videos
3. **NMS optimization** for overlapping detections

---

## 7. Edge Deployment (NVIDIA Jetson)

### 7.1 Jetson Optimization Steps
1. **TensorRT conversion**:
   ```bash
   yolo export model=best.pt format=engine
   ```
2. **INT8 quantization**: Use calibration dataset
3. **Power mode**: Enable MAXN for best performance
4. **DeepStream**: Optional integration for production

### 7.2 Expected Jetson Performance

| Device | Expected FPS |
|--------|---------------|
| Jetson Nano | ~5-10 |
| Jetson Xavier NX | ~20-30 |

---

## 8. Deliverables

### 8.1 GitHub Repository
- `README.md` with setup instructions
- Training code and config
- Inference pipeline
- Trained weights (optional - may be large)

### 8.2 Output Video
- Processed clip showing:
  - Bounding boxes
  - Unique ID labels
  - Trajectory tail lines
  - FPS overlay

### 8.3 Summary Report (in README)
1. **Architecture Choice**: Why YOLOv8n, small object handling
2. **ID Switching**: How we addressed drone ego-motion and occlusions
3. **Edge Deployment**: Jetson adaptation strategy

---

## 9. Timeline

| Day | Task |
|-----|------|
| 1 | Dataset prep, environment setup, test YOLOv8 base |
| 2 | Upload data to Colab, start fine-tuning |
| 3 | Complete fine-tuning, download weights |
| 4 | Implement ByteTrack pipeline + visualization |
| 5 | Optimize, benchmark FPS, generate output video |
| 6 | Write documentation, prepare deliverables |

---

## 10. Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| YOLOv8n over YOLOv5/others | Better small-object performance, easier fine-tuning |
| ByteTrack over DeepSORT | Handles low-confidence detections better (critical for small objects) |
| Anchor-free head | More flexible for varying object scales in aerial view |
| Custom Kalman noise | Essential for moving camera (drone) scenarios |
| 300MB budget | Kept by using nano variant (6MB) + lightweight tracker |

---

## 11. Next Steps

1. **Prepare dataset** - Extract VisDrone validation videos and convert to YOLO format
2. **Create Colab notebook** - For model fine-tuning
3. **Set up local environment** - Install dependencies
4. **Run baseline test** - Test YOLOv8n on sample frames