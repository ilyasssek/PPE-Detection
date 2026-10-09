# 🦺 PPE Detection using YOLOv8

## 📌 Overview

This project aims to improve industrial workplace safety through automatic detection of Personal Protective Equipment (PPE) using Artificial Intelligence and Computer Vision.

The system detects:

* 🪖 Safety Helmets (`helmet`)
* 👷 Safety Vests (`vest`)
* 🧑 Unprotected Heads (`head`)

and automatically classifies workers as:

* ✅ **SAFE**
* ❌ **NOT SAFE**

The model is based on **YOLOv8m** and is capable of performing real-time object detection on images and video streams.

---

## 🎯 Objectives

* Detect PPE in industrial environments.
* Improve workplace safety monitoring.
* Automate compliance verification.
* Reduce risks caused by missing safety equipment.
* Provide real-time SAFE / NOT SAFE classification.

---

## 🏗️ System Architecture

```text
Camera / Image / Video
            │
            ▼
      YOLOv8m Model
            │
            ▼
 Object Detection
(helmet / vest / head)
            │
            ▼
 SAFE / NOT SAFE Logic
            │
            ▼
   Visualized Results
```

---

## 🛠️ Technologies Used

* Python
* YOLOv8 (Ultralytics)
* PyTorch
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Computer Vision
* Deep Learning

---

## 📊 Dataset

Dataset: **Hardhat & Vest Detection Dataset**

* 1,964 annotated images
* 18,500 labeled instances
* 3 target classes:

  * helmet
  * vest
  * head

Data augmentation techniques:

* Mosaic Augmentation
* Horizontal Flip
* HSV Augmentation
* Random Crop
* Translation
* Perspective Transform

---

## ⚙️ Training Configuration

| Parameter            | Value     |
| -------------------- | --------- |
| Model                | YOLOv8m   |
| Epochs               | 30        |
| Image Size           | 640 × 640 |
| Batch Size           | 16        |
| Optimizer            | SGD       |
| Learning Rate        | 0.01      |
| Confidence Threshold | 0.314     |
| IoU Threshold        | 0.45      |

---

## 📈 Results

### Overall Performance

| Metric    | Value |
| --------- | ----- |
| Precision | 86.9% |
| Recall    | 86.5% |
| mAP@0.5   | 84.6% |
| F1-Score  | 87%   |

### Per-Class Performance

| Class  | Precision | Recall | mAP@0.5 |
| ------ | --------- | ------ | ------- |
| Helmet | 91.5%     | 86.3%  | 86.1%   |
| Vest   | 79.8%     | 86.4%  | 82.7%   |
| Head   | 89.2%     | 86.7%  | 85.1%   |

---

## 🔮 Future Improvements

* Real-time dashboard (Streamlit)
* Email/SMS alerts
* IP camera integration
* Detection of additional PPE:

  * Gloves
  * Safety Goggles
  * Safety Harnesses
* Cloud deployment
* Industrial-scale monitoring

---

## 👨‍💻 Author

* Ettayeb Benchekroune

EMSI Marrakech — 4IADATA

Academic Year 2025/2026

---

## 📄 License

This project was developed for educational and research purposes as part of a final-year academic project.
