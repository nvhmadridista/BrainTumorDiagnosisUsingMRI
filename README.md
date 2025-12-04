# Brain Tumor Diagnosis Using MRI

A deep-learning system for **multi-class brain tumor classification** using MRI T1-weighted images. This repository presents a reproducible, research-grade pipeline integrating **ConvNeXt-Tiny** and **Transformer Encoder** modules for robust global–local feature modeling, accompanied by a full deployment stack and monitoring workflow.

---

## 1. Introduction

Brain tumors are among the most critical neurological disorders requiring timely and accurate diagnosis. MRI remains the modality of choice due to its high soft-tissue contrast. This project develops an end‑to‑end automated diagnostic pipeline capable of classifying MRI images into four categories:

- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor**

**GitHub Repository:** [https://github.com/nvhmadridista/BrainTumorDiagnosisUsingMRI](https://github.com/nvhmadridista/BrainTumorDiagnosisUsingMRI)

---

## 2. System Architecture

A detailed pipeline diagram is available at: [https://byvn.net/FIXG](https://byvn.net/FIXG)

### High-Level Workflow

1. **Data Acquisition & Splitting**
2. **MRI Preprocessing & Augmentation**
3. **Feature Extraction (ConvNeXt-Tiny + Transformer Encoder)**
4. **Training & Hyperparameter Optimization**
5. **Model Evaluation (Multimetric)**
6. **Deployment (FastAPI + Streamlit)**
7. **Monitoring & Drift Detection**

---

## 3. Dataset Description

Dataset used: [**Brain Tumor MRI Dataset (Kaggle)**](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

### Dataset Statistics

- **~7,000** T1-weighted MRI images
- **~5,712** samples for training
- **4 balanced classes**, minimizing bias
- **Non-uniform image sizes** across samples

### Class Characteristics

- **Glioma:** Irregular, infiltrative morphology
- **Meningioma:** Smooth, well-defined margins
- **Pituitary Tumor:** Located near sella turcica
- **No Tumor:** Normal anatomical structures

---

## 4. Methodology

### 4.1 Data Acquisition

- Dataset downloaded via Kaggle API
- Split into **70% train**, **15% validation**, **15% test**
- Verified class distribution to ensure balance

### 4.2 MRI Preprocessing

- Resize to **224×224** or **256×256**
- Pixel normalization:

  - `[0,1]` or ImageNet statistics

- Noise reduction (optional)
- **Augmentation strategies:**

  - Rotation ±15°
  - Translation / Zoom
  - Brightness adjustment
  - Horizontal flipping

### 4.3 Model Architecture

This project adopts a **hybrid CNN–Transformer approach**, leveraging:

- **ConvNeXt-Tiny** for hierarchical spatial representation
- **Transformer Encoder** for long-range global context

#### Architectural Flow

```
MRI Image (224×224)
        ↓
ConvNeXt-Tiny
        ↓
Feature Map (7×7×768)
        ↓
Flatten → 49 tokens × 768-dim
        ↓
Transformer Encoder (1–3 layers)
        ↓
Global Pooling / CLS Token
        ↓
Fully Connected Layer
        ↓
Softmax (4 classes)
```

### 4.4 Training Strategy

- **Loss:** CrossEntropy
- **Optimizer:** AdamW
- **LR Scheduler:** Cosine Annealing / ReduceLROnPlateau
- **Epochs:** 20–40
- **Batch Size:** 16–32
- **Regularization:** weight decay, dropout
- **Early Stopping:** patience = 5

### 4.5 Evaluation Protocol

The model is evaluated using a multi-metric approach:

- **Accuracy**
- **Macro F1-score** (class-balanced)
- **Macro AUC**
- **Confusion Matrix**
- **Per-class ROC Curves**

Results are visualized and logged for reproducibility.

### 4.6 Deployment

#### API Layer

- Implemented in **FastAPI**
- Model exported using **ONNX** or **TorchScript**

#### Web UI

Built with **Streamlit**/**Gradio**, offering:

- MRI image upload
- Predicted tumor class
- Confidence scores
- Grad-CAM visualization for interpretability

### 4.7 Monitoring & Drift Detection

- Logs prediction distribution over time
- Drift detection monitors shifts in input distribution or output predictions
- Optional automated retraining mechanism

---

## 5. Citation

If you use this project in academic work, please cite:

```
@misc{BrainTumorDiagnosisMRI2025,
  title  = {Brain Tumor Diagnosis Using MRI with ConvNeXt-Tiny and Transformer Encoder},
  author = {Nguyen, Van-Huong and Contributors},
  year   = {2025},
  url    = {https://github.com/nvhmadridista/BrainTumorDiagnosisUsingMRI}
}
```
