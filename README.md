# 🔬 Skin Lesion Classification System

An AI-powered deep learning system for classifying 8 different skin conditions using EfficientNetB0, achieving **81.75% test accuracy** on a balanced dataset of 14,711 dermatoscopic images.

## 🎯 Features

- **8-Class Skin Condition Detection**
  - 🔴 Acne
  - ⚠️ Basal Cell Carcinoma (Cancer)
  - ✅ Benign Keratosis
  - 🟡 Eczema
  - 🟠 Fungal Infection
  - ✅ Melanocytic Nevi (Moles)
  - 🚨 Melanoma (Cancer)
  - ✅ Normal Skin

- **Grad-CAM Visualization** - See where the AI focuses
- **Confidence Scores** - Understand prediction reliability
- **Web Interface** - User-friendly Streamlit app
- **Medical Recommendations** - Actionable insights

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | **81.75%** |
| Validation Accuracy | 83.60% |
| Weighted F1-Score | 0.82 |

### Per-Class Accuracy
| Class | Accuracy |
|-------|----------|
| Normal Skin | 100.00% |
| Acne | 93.82% |
| Basal Cell Carcinoma | 86.00% |
| Melanoma | 84.67% |
| Eczema | 82.94% |
| Melanocytic Nevi | 75.00% |
| Fungal Infection | 69.14% |
| Benign Keratosis | 66.33% |

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow, Keras
- **Model**: EfficientNetB0 (Transfer Learning)
- **Web Framework**: Streamlit
- **Image Processing**: OpenCV, Pillow
- **Visualization**: Matplotlib, Seaborn
- **Explainability**: Grad-CAM

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/mahnoor-2722/skin-lesion-classification.git
cd skin-lesion-classification