# 🩺 Skin Cancer Detector

> A deep learning-powered skin lesion classification system built using **PyTorch**, **EfficientNet-B0**, and **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b)

---

## 📌 Overview

Skin Cancer Detector is an AI-powered web application that classifies dermoscopic skin lesion images into one of seven lesion categories using transfer learning with EfficientNet-B0.

The application provides an intuitive web interface where users can upload an image and receive:

- ✅ Predicted lesion type
- ✅ Confidence score
- ✅ Top-3 predictions
- ✅ Interactive medical dashboard

> **Disclaimer:** This project is intended for educational and research purposes only. It is **not** a replacement for professional medical diagnosis.

---

## ✨ Features

- 🧠 EfficientNet-B0 Transfer Learning
- 🩺 7-Class Skin Lesion Classification
- 📤 Upload Image for Prediction
- 📊 Confidence Score
- 📈 Top-3 Prediction Probabilities
- 🎨 Modern Streamlit Dashboard
- ⚡ Fast Inference
- 📱 Responsive User Interface

---

## 🖼️ Application Preview

### Dashboard

> <img width="959" height="475" alt="image" src="https://github.com/user-attachments/assets/19a16218-4c45-4ff4-b70b-eb904eb44c6d" />


### Prediction

<img width="665" height="317" alt="image" src="https://github.com/user-attachments/assets/0c63fa7b-0b67-4365-bd96-13fcf2489e79" />



## 📂 Dataset

This project uses the **HAM10000 (Human Against Machine with 10,000 Training Images)** dataset.

### Classes

| Code | Disease |
|------|----------------------------|
| akiec | Actinic Keratosis |
| bcc | Basal Cell Carcinoma |
| bkl | Benign Keratosis |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic Nevus |
| vasc | Vascular Lesion |

---

## 🏗️ Model Architecture

The project uses **EfficientNet-B0** pretrained on ImageNet.

```
Input Image
      │
Resize (224 × 224)
      │
Image Transform
      │
EfficientNet-B0
      │
Dropout
      │
Linear Layer
      │
7 Output Classes
```

---

## 🧠 Training Pipeline

```
Dataset
      │
DataLoader
      │
Data Augmentation
      │
EfficientNet-B0
      │
CrossEntropy Loss
      │
Adam Optimizer
      │
Backpropagation
```

---

## 📊 Model Performance

| Metric | Score |
|---------|-------|
| Validation Accuracy | **87%** |
| Weighted Precision | **87%** |
| Weighted Recall | **87%** |
| Weighted F1 Score | **87%** |

### Class-wise Performance

| Class | F1 Score |
|--------|----------|
| Actinic Keratosis | 0.72 |
| Basal Cell Carcinoma | 0.81 |
| Benign Keratosis | 0.75 |
| Dermatofibroma | 0.73 |
| Melanoma | 0.65 |
| Melanocytic Nevus | 0.94 |
| Vascular Lesion | 0.88 |

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Torchvision
- Streamlit
- Pandas
- NumPy
- Pillow
- Matplotlib
- Scikit-learn

---

## 📁 Project Structure

```
skin-cancer-detector/

│── app.py
│── predict.py
│── train.py
│── requirements.txt
│── README.md
│── skin_cancer_model.pth
│
├── assets/
│
├── images/
│
└── notebook/
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/skin-cancer-detector.git
```

Move into the project

```bash
cd skin-cancer-detector
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📸 Using the App

1. Launch the Streamlit application.
2. Upload a dermoscopic skin lesion image.
3. Click **Predict**.
4. View:
   - Predicted lesion type
   - Confidence score
   - Top-3 predictions

---

## 🎯 Future Improvements

- Explainable AI using Grad-CAM
- Docker deployment
- Hugging Face deployment
- Prediction history
- User authentication
- REST API using FastAPI
- Mobile application
- ONNX model export

---

## 📖 What I Learned

During this project I gained hands-on experience with:

- Custom PyTorch Datasets
- DataLoaders
- Image preprocessing
- Transfer Learning
- EfficientNet-B0
- Model evaluation
- Classification metrics
- Streamlit deployment
- Deep learning inference pipelines


---

## 👨‍💻 Author

**Darshno**

---

⭐ If you found this project interesting, consider giving it a star!
