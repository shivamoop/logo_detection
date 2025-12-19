# Adidas & Puma Logo Detection (YOLOv8 – Google Colab)

## 📌 Problem Statement
Detect **Adidas** and **Puma** logos in a video, draw bounding boxes with brand labels, save an annotated video, and generate a CSV file containing detection details.

---

## 🧠 Approach Summary
- Problem formulated as **object detection**
- Used **YOLOv8-s** with transfer learning
- Merged two Roboflow datasets with **explicit class ID remapping**
- Trained and evaluated model on **Google Colab (T4 GPU)**
- Optimized video inference using **temporal frame sampling**
- Generated both **annotated video** and **CSV output**

---

## 🏗️ Workflow (Colab Notebook)
1. Environment setup (Colab + GPU)
2. Dataset inspection (Adidas & Puma)
3. Dataset merge with label remapping  
   - `0 → Adidas`
   - `1 → Puma`
4. Model training (YOLOv8-s)
5. Model evaluation (mAP, precision, recall)
6. Video inference (FPS sampling to handle long video)
7. CSV generation (frame-wise detections)

---

## ⚙️ Tech Stack
- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Pandas
- yt-dlp
- **Google Colab (Tesla T4 GPU)**

---

## 📊 Evaluation Results
- mAP@0.5 ≈ **0.61**
- Precision ≈ **0.74**
- Recall ≈ **0.55**

These metrics are reasonable for **small logo detection in videos**.

---

## 🎥 Video Processing Strategy
- Original video contained ~32k frames
- Full-frame inference caused Colab memory/time issues
- Used **temporal sampling (1–2 FPS)**:
  - Preserves logo detections
  - Reduces redundant computation
  - Industry-standard optimization

---

## 📁 Outputs
- `output_annotated.mp4` – video with bounding boxes and labels
- `detections.csv` – frame-wise detections with:
  - Frame ID
  - Timestamp
  - Class name
  - Confidence
  - Bounding box coordinates

---

## ▶️ How to Run
1. Open `Logo_Detection_Colab.ipynb` in Google Colab
2. Enable GPU: `Runtime → Change runtime type → GPU`
3. Run cells sequentially

---

## ⚠️ Notes
- Datasets are stored on Google Drive and not committed
- Model weights not committed due to size
- Designed for clarity, correctness, and reproducibility in Colab

---

## 👤 Author
**Shivam Yadav**  
AI / Computer Vision Engineer
