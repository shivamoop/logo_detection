# Nike & Adidas Logo Detection (YOLOv8)

## Problem Statement
Build a computer vision system to **detect and localize Nike and Adidas logos** in images and long videos.  
The solution must handle **small logos**, scale to **long videos**, and produce **structured outputs** for analysis.

---

## Approach Summary
- Formulated as an **object detection** problem
- Performed logo annotation using **Roboflow** by extracting and labeling **random frames** from videos.
- Augmented annotations with an **existing logo dataset** to improve class coverage and robustness.
- Merged all labeled data into a unified YOLOv8-format dataset for training.
- Trained a **YOLOv8-m** model (accuracy-first) on merged Nike & Adidas datasets
- Applied **inference-time filtering** (confidence + size) to reduce false positives
- Used **chunk-based video processing** to handle long videos safely
- Generated **CSV outputs only when a logo is detected**

---

## Pipeline

### 1. Dataset Preparation
- Two datasets (Nike, Adidas) in YOLO format
- Merged with explicit class mapping:
  - `0 → Nike`
  - `1 → Adidas`

### 2. Model Training
- YOLOv8-m
- Image size: `768 × 768`
- Transfer learning from COCO
- Training on **Google Colab (Tesla T4 GPU)**

### 3. Inference Optimization
- Confidence thresholding
- Bounding-box **size filtering** (logo-aware)
- Video chunking:
  - `1000` frames per chunk
  - `200` frame overlap

### 4. Outputs
- Annotated images
- Annotated videos
- CSV files containing **only logo detections**

---

## Requirements

```txt
Python 3.9+
ultralytics
opencv-python
numpy
pandas
yt-dlp
```
---


## Environment
- Google Colab
- GPU: **Tesla T4**

---

## Training Configuration

| Parameter        | Value |
|------------------|-------|
| Model            | YOLOv8-m |
| Image Size       | 768 |
| Epochs           | 100 |
| Batch Size       | 8 |
| Optimizer        | AdamW |
| Learning Rate    | 3e-4 |
| Early Stopping   | Patience = 15 |

---

## Results (Baseline)

- **Precision:** ~0.71
- **Recall:** ~0.65
- **mAP@0.5:** ~0.67
- **mAP@0.5:0.95:** ~0.40

**Observation**
- Adidas logos were detected more reliably
- Nike logos were more challenging due to small size and labeling noise

---

## Inference-Time Filtering (Key Design)

To reduce false positives without retraining:

```python
CONF_THRES = 0.15
MIN_AREA   = 0.0003   # allows very small logos
MAX_AREA   = 0.08     # blocks person / jersey boxes
```
## Key Engineering Decisions

- Used **inference-time constraints** instead of overfitting the model
- Processed long videos via **overlapping chunks**
- Generated **CSV only when detections occur**
- Kept the GitHub repository **lightweight and reproducible**

## 🐞 Debugging

During development, multiple debugging steps were applied to ensure correctness and stability:

- Verified YOLO label format and normalized coordinates
- Handled empty detections safely during inference
- Tuned confidence and size thresholds to balance precision vs recall
- Resolved long-video crashes using chunk-based processing
- Debugged Git issues (ignored files, large files, submodules)

All debugging was done incrementally with visual validation before automation.

## Outputs & Datasets (Important)

Due to **GitHub file size limits**, large artifacts are **not stored in this repository**.

### External Artifacts ([Google Drive Link](https://drive.google.com/drive/folders/1ibaDVzev2u3XautEjVtu60vnr93YnULV?usp=drive_link))

- Annotated video (final merged)
- Chunked annotated videos
- Final detections CSV
- Merged dataset
- Trained model weights
