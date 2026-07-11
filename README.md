# 🩺 Skin Cancer Detector

A deep learning-powered skin lesion classification system built using **PyTorch**, **EfficientNet-B0**, **EfficientNet-B3**, **DenseNet121**, **ResNet50**, **ConvNeXt-Tiny**, and **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b)

---

# 📌 Overview

**Skin Cancer Detector** is an end-to-end deep learning system for automated skin lesion classification using dermoscopic images from the **HAM10000** dataset.

Unlike a standard image classification project, this project explores a wide range of deep learning strategies including transfer learning, dataset engineering, multi-resolution modeling, ensemble learning (2-model through 5-model), hierarchical classification, class imbalance handling, external dataset augmentation, test-time augmentation, class-set reduction, weighted ensemble optimization, and explainable AI.

After extensive experimentation, the best-performing system is a **5-model weighted soft-voting ensemble** (EfficientNet-B0, DenseNet121, EfficientNet-B3, ResNet50, and ConvNeXt-Tiny), reaching **85% accuracy and 0.86 weighted F1-score** across all seven lesion classes — the strongest result found across every method attempted.

The project also includes an interactive **Streamlit dashboard** that allows users to upload lesion images, visualize predictions, inspect confidence scores, and view Grad-CAM attention maps.

> **Disclaimer**
>
> This project is intended solely for educational and research purposes.
> It must **not** be used as a substitute for professional medical diagnosis.

---

# ✨ Features

- Deep Learning Skin Lesion Classification
- EfficientNet-B0 Transfer Learning
- EfficientNet-B3 Transfer Learning (300×300 resolution)
- DenseNet121 Transfer Learning
- ResNet50 Transfer Learning
- ConvNeXt-Tiny Transfer Learning
- 2-Model, 3-Model, and 5-Model Soft-Voting Ensembles
- Weighted Ensemble Optimization (random search over model weights)
- Test-Time Augmentation (flip + rotation variants)
- Seven-Class and Five-Class Classification Modes
- Lesion-Level Data Splitting (leakage prevention)
- Class Imbalance Handling (undersampling + oversampling)
- External Dataset Augmentation (ISIC 2019)
- Hierarchical Multi-Stage Classification Pipeline
- Image Upload Interface
- Confidence Score
- Top-3 Predictions
- Grad-CAM Explainability
- Interactive Streamlit Dashboard
- Classification Reports
- Confusion Matrix Analysis
- Model Checkpointing (best validation loss)
- Fast GPU Inference

---

# 📂 Dataset

The project primarily uses the **HAM10000 (Human Against Machine with 10,000 Training Images)** dataset, with a supplementary experiment using the **ISIC 2019 archive**.

The core dataset contains over **10,000 dermoscopic skin lesion images** belonging to seven disease categories.

| Code | Disease |
|------|----------------------------|
| akiec | Actinic Keratosis |
| bcc | Basal Cell Carcinoma |
| bkl | Benign Keratosis |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic Nevus |
| vasc | Vascular Lesion |

### Supplementary Dataset: ISIC 2019

The ISIC 2019 archive (of which HAM10000 is a subset) was used in one experiment to source additional **real, non-overlapping** images for the three most data-scarce classes (akiec, df, vasc). Images already present in HAM10000 were explicitly filtered out to prevent leakage.

---

# 🔬 Dataset Engineering

One major focus of this project was building a reliable and unbiased data pipeline.

### Lesion-Level Train/Validation Split

Instead of randomly splitting individual images, the dataset is split using **lesion_id**, preventing duplicate images of the same lesion from appearing in both training and validation sets.

### Image Path Mapping

Automatically maps every `image_id` to the correct image across both HAM10000 image directories (and the ISIC 2019 directory, for the external-data experiment).

### Class Imbalance Handling

The HAM10000 dataset is heavily imbalanced (nv outnumbers df by roughly 67:1).

To address this:

- Undersampled the dominant **nv** class (5,403 → 900)
- Oversampled minority classes via duplication + stochastic augmentation
  - akiec, bcc, df, vasc

Additionally, a **class-weighted loss function** was explored as an alternative balancing strategy (tested, but under-sampling/over-sampling gave more stable results).

### External Data Augmentation (ISIC 2019)

Real, non-duplicate ISIC 2019 samples were merged into training for akiec, df, and vasc, boosting akiec from 259→996 real images, df from 81→205, and vasc from 106→217. This was tested as an alternative to synthetic oversampling.

### Class-Set Reduction Experiment

The 7-class problem was also re-run as a **5-class problem** (dropping df and vasc, the two lowest-support/lowest-urgency classes), to test whether removing the weakest classes improved overall performance.

---

# 🏗️ Model Architecture (Final 5-Model Ensemble)

```text
                            Input Image
                                 │
                  ┌──────────────┼───────────────┬───────────────┬───────────────┐
                  │              │                │               │               │
           EfficientNet-B0  DenseNet121   EfficientNet-B3    ResNet50      ConvNeXt-Tiny
             (224×224)       (224×224)      (300×300)        (224×224)      (224×224)
                  │              │                │               │               │
               Softmax        Softmax          Softmax          Softmax        Softmax
                  └──────────────┴────────┬───────┴───────────────┴───────────────┘
                                           │
                         Weighted Soft Voting (optimized via search)
                                           │
                                  Final Prediction
```

---

# 🧠 Models Explored

## Model 1 — EfficientNet-B0
- ImageNet pretrained, fine-tuned on HAM10000 at 224×224

## Model 2 — DenseNet121
- ImageNet pretrained, fine-tuned on HAM10000 at 224×224

## Model 3 — EfficientNet-B3
- ImageNet pretrained, fine-tuned at higher resolution (300×300) for finer detail capture

## Model 4 — ResNet50
- ImageNet pretrained, fine-tuned on HAM10000 at 224×224

## Model 5 — ConvNeXt-Tiny
- ImageNet pretrained, fine-tuned on HAM10000 at 224×224
- Strongest individual model of all five (82.7% standalone validation accuracy)

## Final Model — Weighted 5-Model Soft-Voting Ensemble
The prediction probabilities from all five models are combined using weights optimized via random search (Dirichlet-sampled weight combinations, selected by best macro F1 on the validation set):

```
EfficientNet-B0:  0.054
DenseNet121:      0.079
EfficientNet-B3:  0.276
ResNet50:         0.211
ConvNeXt-Tiny:    0.380
```

---

# 🚀 Training Pipeline

```text
HAM10000 Dataset (+ optional ISIC 2019 samples)
        │
Lesion-Level Split
        │
Class Balancing (under-sample + over-sample)
        │
Data Augmentation
        │
PyTorch Dataset
        │
DataLoader
        │
Transfer Learning (5 architectures)
        │
CrossEntropy Loss
        │
Adam Optimizer
        │
Validation + Early Stopping
        │
Checkpoint Saving (best val loss)
        │
Weighted Ensemble Search
        │
Final Evaluation
```

---

# 🧪 Research Experiments

Every method below was implemented and empirically evaluated — including approaches that did **not** improve performance, documented here for transparency.

| # | Experiment | Result |
|---|------------|---------|
| 1 | CNN from Scratch | Poor performance (~72% accuracy) |
| 2 | EfficientNet-B0 (single) | Strong baseline (~78–82%) |
| 3 | DenseNet121 (single) | Comparable to EfficientNet-B0 |
| 4 | Class-Weighted Loss | Rebalanced predictions but hurt overall accuracy on its own |
| 5 | Under/Over-Sampling | Improved minority-class recall substantially |
| 6 | 2-Model Soft-Voting Ensemble (EffNet-B0 + DenseNet121) | 83% accuracy, 0.84 weighted F1 |
| 7 | Hierarchical 3-Stage Classification | 75% accuracy — **underperformed** the flat ensemble due to compounding stage errors |
| 8 | Grad-CAM Explainability | Confirmed model attention aligns with lesion regions |
| 9 | EfficientNet-B3 @ 300×300 (3rd ensemble member) | Accuracy unchanged (83%), but macro F1 improved (0.73→0.76) |
| 10 | Test-Time Augmentation (flip + rotation) | No further net improvement |
| 11 | External Data Merge (ISIC 2019) | Accuracy **dropped** to 81% — attributed to cross-dataset domain shift |
| 12 | Class-Set Reduction (5-class: dropped df, vasc) | 84% accuracy, 0.85 weighted F1 — best result until ensemble expansion |
| 13 | 5-Model Ensemble (equal weights) | 85% accuracy, 0.85 weighted F1, 0.77 macro F1 |
| 14 | 5-Model Weighted Ensemble (optimized weights) | **85% accuracy, 0.86 weighted F1, 0.79 macro F1 — best overall result** |

---

# 🌳 Hierarchical Classification (Explored, Not Used in Final Model)

After analyzing the confusion matrix, it became evident that most prediction errors occurred between:

- Melanoma
- Benign Keratosis
- Melanocytic Nevus

A hierarchical classification pipeline was designed to address this:

```text
Input Image
      │
Binary Classifier (confused cluster vs. rest)
      │
 ┌───────────────┐
 │               │
3-Class CNN   4-Class CNN
(nv/mel/bkl)  (akiec/bcc/df/vasc)
 │               │
 └───────┬───────┘
         │
 Final Prediction
```

Although theoretically promising — and despite each individual stage achieving high standalone accuracy (Stage 1 binary router: 94.6%) — this approach achieved only **75% validation accuracy** overall, lower than the flat ensemble. This is because pipeline accuracy is bounded by the product of stage-wise accuracies: a Stage 1 misrouting can never be corrected downstream. It is included here as a documented comparative experiment rather than the final solution.

---

# 🔥 Explainable AI

The project integrates **Grad-CAM** to visualize model attention.

Grad-CAM highlights the regions of the input image that contribute most strongly to the model's prediction, improving interpretability and confirming that predictions are driven by lesion regions rather than background artifacts or imaging noise.

---

# 📊 Model Performance

## Final Model — 5-Model Weighted Ensemble

| Metric | Score |
|---------|-------|
| Validation Accuracy | **85%** |
| Weighted Precision | **87%** |
| Weighted Recall | **85%** |
| Weighted F1 Score | **86%** |
| Macro F1 Score | **79%** |

### Per-Class Results

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Actinic Keratosis (akiec) | 0.74 | 0.63 | 0.68 |
| Basal Cell Carcinoma (bcc) | 0.83 | 0.93 | 0.88 |
| Benign Keratosis (bkl) | 0.74 | 0.81 | 0.77 |
| Dermatofibroma (df) | 0.81 | 0.62 | 0.70 |
| Melanoma (mel) | 0.60 | 0.77 | 0.67 |
| Melanocytic Nevus (nv) | 0.95 | 0.88 | 0.92 |
| Vascular Lesion (vasc) | 0.87 | 0.94 | 0.91 |

---

# 📈 Full Model Comparison

| Approach | Accuracy | Weighted F1 |
|--------|----------|-------------|
| CNN From Scratch | 72% | 0.68 |
| EfficientNet-B0 (single) | 78% | 0.80 |
| DenseNet121 (single) | 79% | 0.80 |
| EfficientNet-B3 (single, 300×300) | 80% | 0.81 |
| ResNet50 (single) | 77% | 0.78 |
| ConvNeXt-Tiny (single) | 83% | 0.83 |
| Hierarchical (3-stage) | 75% | 0.77 |
| 2-Model Ensemble | 83% | 0.84 |
| 3-Model Ensemble (+EffNet-B3) | 83% | 0.84 |
| 3-Model Ensemble + TTA | 83% | 0.84 |
| 2-Model Ensemble + ISIC 2019 Data | 81% | 0.82 |
| 5-Class Ensemble (df, vasc removed) | 84% | 0.85 |
| **5-Model Ensemble (equal weights)** | **85%** | **0.85** |
| **5-Model Ensemble (optimized weights)** | **85%** | **0.86** |

---

# 📉 Error Analysis

Confusion matrix analysis revealed that most misclassifications occurred among:

- Melanoma
- Benign Keratosis
- Melanocytic Nevus

These lesion types share similar dermoscopic features, making them difficult to distinguish even for deep neural networks and, at times, expert dermatologists. This finding motivated both the hierarchical classification experiment and the class-set reduction experiment — notably, removing the two lowest-support classes (df, vasc) only modestly improved results, confirming that the nv–mel–bkl overlap (not data volume for any single class) is the primary factor limiting accuracy on this dataset.

A separate finding: merging external ISIC 2019 data reduced accuracy, most likely due to domain shift between imaging sources (different acquisition equipment, color calibration, and artifact profiles) — indicating that naively adding "more real data" is not guaranteed to help without domain adaptation.

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
│   ├── best_model.pth              (EfficientNet-B0)
│   ├── best_model_densenet.pth     (DenseNet121)
│   ├── best_model_b3.pth           (EfficientNet-B3)
│   ├── best_model_r50.pth          (ResNet50)
│   └── best_model_convnext.pth     (ConvNeXt-Tiny)
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

- Modular PyTorch training pipeline across 5 architectures
- Automatic checkpoint saving with early stopping
- Validation-loss based model selection
- GPU training support
- Kaggle persistence workflow (dataset uploads, working directory checkpointing)
- Reproducible experiments (fixed random seeds throughout)
- Multi-model weighted ensemble inference pipeline
- Random-search weight optimization for ensembling
- Explainability integration (Grad-CAM)
- Cross-dataset augmentation pipeline (ISIC 2019 integration + leakage filtering)

---

# 🚀 Future Improvements

- Domain adaptation techniques for external dataset integration (e.g., color/stain normalization before merging ISIC data)
- Malignant-vs-benign binary classification framing
- GAN-based synthetic oversampling for extremely low-support classes
- Vision Transformers (ViT)
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
- Dataset engineering and leakage prevention
- Handling class imbalance (under-sampling, over-sampling, weighted loss)
- Transfer learning across multiple architectures
- Ensemble learning (2 through 5 models) and weighted ensemble optimization
- Hierarchical classification design and its failure modes
- Cross-dataset augmentation and domain shift
- Test-time augmentation
- PyTorch training pipelines
- Model checkpointing and early stopping
- Classification metrics (precision, recall, F1, macro vs. weighted averaging)
- Error analysis and confusion matrix interpretation
- Grad-CAM explainability
- Deep learning deployment
- Scientific experimentation and honest reporting of negative results
- Comparative model evaluation

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 🙏 Acknowledgements

- HAM10000 Dataset
- ISIC 2019 Archive
- PyTorch
- Torchvision
- Streamlit
- Scikit-learn
- EfficientNet
- DenseNet
- ResNet
- ConvNeXt

---

## ⭐ Support

If you found this project useful, consider giving it a **star** on GitHub. It helps others discover the project and supports future development.
