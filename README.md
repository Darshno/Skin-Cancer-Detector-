# 🩺 Skin Cancer Detector

A deep learning-powered skin lesion classification system built using **PyTorch**, **EfficientNet-B0**, **DenseNet121**, and **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b)


---

# 📌 Overview

**Skin Cancer Detector** is an end-to-end deep learning system for automated skin lesion classification using dermoscopic images from the **HAM10000** dataset.

Unlike a standard image classification project, this project explores multiple deep learning strategies including transfer learning, dataset engineering, ensemble learning, hierarchical classification, class imbalance handling, and explainable AI.

After extensive experimentation, the final system uses a **soft-voting ensemble of EfficientNet-B0 and DenseNet121**, providing the best overall performance while maintaining strong generalization across seven lesion classes.

The project also includes an interactive **Streamlit dashboard** that allows users to upload lesion images, visualize predictions, inspect confidence scores, and view Grad-CAM attention maps.

> **Disclaimer**
>
> This project is intended solely for educational and research purposes.
> It must **not** be used as a substitute for professional medical diagnosis.

---

# ✨ Features

- Deep Learning Skin Lesion Classification
- EfficientNet-B0 Transfer Learning
- DenseNet121 Transfer Learning
- Soft-Voting Ensemble
- Seven-Class Classification
- Lesion-Level Data Splitting
- Class Imbalance Handling
- Image Upload Interface
- Confidence Score
- Top-3 Predictions
- Grad-CAM Explainability
- Interactive Streamlit Dashboard
- Classification Reports
- Confusion Matrix
- Model Checkpointing
- Fast GPU Inference

---

---

# 📂 Dataset

The project uses the **HAM10000 (Human Against Machine with 10,000 Training Images)** dataset.

The dataset contains over **10,000 dermoscopic skin lesion images** belonging to seven disease categories.

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

# 🔬 Dataset Engineering

One major focus of this project was building a reliable and unbiased data pipeline.

### Lesion-Level Train/Validation Split

Instead of randomly splitting individual images, the dataset is split using **lesion_id**, preventing duplicate images of the same lesion from appearing in both training and validation sets.

### Image Path Mapping

Automatically maps every `image_id` to the correct image across both HAM10000 image directories.

### Class Imbalance Handling

The HAM10000 dataset is heavily imbalanced.

To address this:

- Undersampled the dominant **nv** class
- Oversampled minority classes
  - akiec
  - bcc
  - df
  - vasc

Additionally, a **class-weighted loss function** was explored as an alternative balancing strategy.

---

# 🏗️ Model Architecture

The final system uses an ensemble of two pretrained CNNs.

```text
                 Input Image
                      │
             Resize (224 × 224)
                      │
           Image Normalization
                      │
      ┌───────────────┴───────────────┐
      │                               │
 EfficientNet-B0               DenseNet121
      │                               │
   Softmax                        Softmax
      └───────────────┬───────────────┘
                      │
          Soft Voting (Average)
                      │
              Final Prediction
```

---

# 🧠 Models Explored

## Model 1

**EfficientNet-B0**

- ImageNet pretrained
- Fine-tuned on HAM10000

---

## Model 2

**DenseNet121**

- ImageNet pretrained
- Fine-tuned on HAM10000

---

## Final Model

A **soft-voting ensemble** combining EfficientNet-B0 and DenseNet121.

The prediction probabilities from both models are averaged to produce the final classification.

---

# 🚀 Training Pipeline

```text
HAM10000 Dataset
        │
Lesion-Level Split
        │
Class Balancing
        │
Data Augmentation
        │
PyTorch Dataset
        │
DataLoader
        │
Transfer Learning
        │
CrossEntropy Loss
        │
Adam Optimizer
        │
Validation
        │
Checkpoint Saving
```

---

# 🧪 Research Experiments

During development, multiple approaches were implemented and evaluated.

| Experiment | Result |
|------------|---------|
| CNN from Scratch | Poor performance |
| EfficientNet-B0 | Strong baseline |
| DenseNet121 | Comparable performance |
| Soft Voting Ensemble | Best overall performance |
| Weighted Loss | Moderate improvement |
| Class Balancing | Improved minority recall |
| Hierarchical Classification | Lower than ensemble |
| Grad-CAM | Added explainability |

---

# 🌳 Hierarchical Classification

After analyzing the confusion matrix, it became evident that most prediction errors occurred between:

- Melanoma
- Benign Keratosis
- Melanocytic Nevus

A hierarchical classification pipeline was designed to address this challenge.

```text
Input Image
      │
Binary Classifier
      │
 ┌───────────────┐
 │               │
3-Class CNN   4-Class CNN
 │               │
 └───────┬───────┘
         │
 Final Prediction
```

Although theoretically promising, this approach achieved **75% validation accuracy**, which was lower than the ensemble model. It is therefore included as a comparative experiment rather than the final solution.

---

# 🔥 Explainable AI

The project integrates **Grad-CAM** to visualize model attention.

Grad-CAM highlights the regions of the input image that contribute most strongly to the model's prediction, improving interpretability and helping users better understand model decisions.

---

# 📊 Model Performance

## Final Model

**Soft Voting Ensemble**

| Metric | Score |
|---------|-------|
| Validation Accuracy | **83%** |
| Weighted Precision | **84%** |
| Weighted Recall | **83%** |
| Weighted F1 Score | **84%** |

---

# 📈 Model Comparison

| Model | Accuracy | Weighted F1 |
|--------|----------|-------------|
| CNN From Scratch | 72% | Low |
| EfficientNet-B0 | 82% | 0.83 |
| DenseNet121 | 81% | 0.82 |
| Ensemble | **83%** | **0.84** |

---

# 📉 Error Analysis

Confusion matrix analysis revealed that most misclassifications occurred among:

- Melanoma
- Benign Keratosis
- Melanocytic Nevus

These lesion types share similar dermoscopic features, making them difficult to distinguish even for deep neural networks.

This finding motivated the hierarchical classification experiment.

---

# 📁 Project Structure

```text
skin-cancer-detector/

├── app.py
├── predict.py
├── train.py
├── dataset.py
├── models.py
├── utils.py
├── requirements.txt
├── README.md
├── checkpoints/
├── assets/
├── notebook/
├── images/
└── gradcam/
```

---

# 🛠️ Tech Stack

- Python
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Pandas
- Pillow
- OpenCV
- Matplotlib
- Scikit-learn

---

# 🚀 Installation

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

# ▶️ Run the Application

```bash
streamlit run app.py
```

---

# 📸 Using the Application

1. Launch the Streamlit application.
2. Upload a dermoscopic skin lesion image.
3. Click **Predict**.

The application displays:

- Predicted lesion type
- Confidence score
- Top-3 predictions
- Grad-CAM visualization

---

# ⚙️ Engineering Highlights

- Modular PyTorch training pipeline
- Automatic checkpoint saving
- Validation-loss based model selection
- GPU training support
- Kaggle persistence workflow
- Reproducible experiments
- Ensemble inference pipeline
- Explainability integration

---

# 🚀 Future Improvements

- Vision Transformers (ViT)
- ConvNeXt
- EfficientNetV2
- Test-Time Augmentation
- Confidence Calibration
- Docker Deployment
- Hugging Face Deployment
- FastAPI REST API
- Prediction History
- User Authentication
- Mobile Application
- ONNX Export
- TensorRT Optimization

---

# 📚 What I Learned

This project provided hands-on experience with:

- Medical image preprocessing
- Dataset engineering
- Handling class imbalance
- Transfer learning
- Ensemble learning
- Hierarchical classification
- PyTorch training pipelines
- Model checkpointing
- Classification metrics
- Error analysis
- Confusion matrix interpretation
- Grad-CAM explainability
- Deep learning deployment
- Scientific experimentation
- Comparative model evaluation

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 🙏 Acknowledgements

- HAM10000 Dataset
- PyTorch
- Torchvision
- Streamlit
- Scikit-learn
- EfficientNet
- DenseNet

---

## ⭐ Support

If you found this project useful, consider giving it a **star** on GitHub. It helps others discover the project and supports future development.
