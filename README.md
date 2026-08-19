# 🔬 Skin Lesion Classification System

[![Model on Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/mahnoor-2722/skin-lesion-classifier)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)

An AI-powered deep learning system for classifying **8 different skin conditions** using EfficientNetB0, achieving **81.75% test accuracy** on a balanced dataset of **14,711 dermatoscopic images**.

---

## 📋 Table of Contents

- [Features](#-features)
- [Demo Screenshots](#-demo-screenshots)
- [Model Performance](#-model-performance)
- [Technology Stack](#️-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Hosting](#-model-hosting)
- [Project Structure](#-project-structure)
- [Training Details](#-training-details)
- [Grad-CAM Explainability](#-grad-cam-explainability)
- [Medical Disclaimer](#️-medical-disclaimer)
- [License](#-license)
- [Author](#-author)

---

## 🎯 Features

### 8-Class Skin Condition Detection

| Icon | Condition | Type |
|------|-----------|------|
| 🔴 | Acne | Inflammatory |
| ⚠️ | Basal Cell Carcinoma | **Cancer** |
| ✅ | Benign Keratosis | Non-cancerous |
| 🟡 | Eczema | Inflammatory |
| 🟠 | Fungal Infection | Infection |
| ✅ | Melanocytic Nevi | Common Moles |
| 🚨 | Melanoma | **Cancer** |
| ✅ | Normal Skin | Healthy |

### Key Capabilities

- 🔥 **Grad-CAM Visualization** - See where the AI focuses to make decisions
- 📊 **Confidence Scores** - Understand prediction reliability with detailed probabilities
- 🌐 **Interactive Web Interface** - User-friendly Streamlit application
- 💡 **Medical Recommendations** - Actionable insights for each condition
- 🤗 **Auto Model Download** - Automatically fetches model from Hugging Face Hub
- ⚡ **Smart Caching** - Fast subsequent loads with local caching

---

## 🎨 Demo Screenshots

### 🏠 Application Home
![App Home](screenshots/app_home.png)

### 🔬 Acne Detection (91.11% Confidence)
![Acne Prediction](screenshots/prediction_acne.png)

### 🔥 Grad-CAM Visualization
Visual explanation showing where the AI focused to make its decision.

---

## 📊 Model Performance

### Overall Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **81.75%** |
| Validation Accuracy | 83.60% |
| Weighted F1-Score | 0.82 |
| Total Training Images | 14,711 |
| Model Size | 25.63 MB |

### Per-Class Accuracy

| Class | Accuracy |
|-------|----------|
| ✅ Normal Skin | **100.00%** |
| 🔴 Acne | 93.82% |
| ⚠️ Basal Cell Carcinoma | 86.00% |
| 🚨 Melanoma | 84.67% |
| 🟡 Eczema | 82.94% |
| ✅ Melanocytic Nevi | 75.00% |
| 🟠 Fungal Infection | 69.14% |
| ✅ Benign Keratosis | 66.33% |

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | TensorFlow 2.x, Keras |
| **Base Model** | EfficientNetB0 (Transfer Learning) |
| **Web Framework** | Streamlit |
| **Image Processing** | OpenCV, Pillow |
| **Visualization** | Matplotlib, Seaborn |
| **Explainability** | Grad-CAM |
| **Model Hosting** | Hugging Face Hub |
| **Version Control** | Git, GitHub |

---

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- 500 MB free disk space (for dependencies + model)

### Step 1: Clone the Repository

```bash
git clone https://github.com/mahnoor-2722/skin-lesion-classification.git
cd skin-lesion-classification
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

> **📝 Note:** On first run, the trained model (25 MB) will be automatically downloaded from [Hugging Face Hub](https://huggingface.co/mahnoor-2722/skin-lesion-classifier) and cached locally for subsequent runs.

---

## 🚀 Usage

### Web Interface

1. **Launch the app** using `streamlit run app/streamlit_app.py`
2. **Upload an image** of a skin condition (JPG, PNG, JPEG)
3. **View predictions** with confidence scores
4. **Explore Grad-CAM** visualization to see AI focus areas
5. **Read recommendations** for the detected condition

### Programmatic Usage

```python
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Download and load model from Hugging Face
model_path = hf_hub_download(
    repo_id="mahnoor-2722/skin-lesion-classifier",
    filename="efficientnetb0_8class_best.h5"
)
model = load_model(model_path)

# Class labels
CLASS_NAMES = [
    'acne', 'basal_cell_carcinoma', 'benign_keratosis', 
    'eczema', 'fungal_infection', 'melanocytic_nevi',
    'melanoma', 'normal_skin'
]

# Load and preprocess image
img = image.load_img('skin_image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
predictions = model.predict(img_array)[0]
predicted_class = CLASS_NAMES[np.argmax(predictions)]
confidence = np.max(predictions) * 100

print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
```

---

## 🤗 Model Hosting

The trained model is hosted on **Hugging Face Hub** for easy access and version control.

**🔗 Model Repository:** [mahnoor-2722/skin-lesion-classifier](https://huggingface.co/mahnoor-2722/skin-lesion-classifier)

### Why Hugging Face?
- ✅ Free public model hosting
- ✅ Version control for ML models
- ✅ Easy programmatic access
- ✅ Community discoverability
- ✅ Professional ML deployment standard

### Manual Download (Alternative)

If auto-download fails, you can manually download the model:

1. Visit: https://huggingface.co/mahnoor-2722/skin-lesion-classifier/tree/main
2. Download `efficientnetb0_8class_best.h5`
3. Place it in the `models/` directory:
   ```
   skin-lesion-classification/
   └── models/
       └── efficientnetb0_8class_best.h5
   ```

---

## 📁 Project Structure

```
skin-lesion-classification/
│
├── app/
│   └── streamlit_app.py          # Main Streamlit application
│
├── models/                        # Local model storage (auto-created)
│   └── efficientnetb0_8class_best.h5
│
├── scripts/
│   └── upload_to_huggingface.py  # Script to upload model to HF
│
├── screenshots/                   # App screenshots for documentation
│   ├── app_home.png
│   ├── prediction_acne.png
│   └── ...
│
├── model_cache/                   # HF downloaded models (gitignored)
├── venv/                          # Virtual environment (gitignored)
│
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation (this file)
└── LICENSE                        # MIT License
```

---

## 🎓 Training Details

| Parameter | Value |
|-----------|-------|
| **Dataset Size** | 14,711 dermatoscopic images |
| **Train/Val/Test Split** | 70% / 15% / 15% |
| **Base Model** | EfficientNetB0 (ImageNet pretrained) |
| **Input Resolution** | 224 × 224 × 3 |
| **Optimizer** | Adam (learning rate: 1e-4) |
| **Loss Function** | Categorical Crossentropy |
| **Batch Size** | 32 |
| **Epochs** | 50 (with early stopping) |
| **Data Augmentation** | Rotation, flip, zoom, brightness |

### Training Strategy

1. **Transfer Learning:** Started with ImageNet pretrained EfficientNetB0
2. **Fine-tuning:** Unfroze last 20 layers for domain adaptation
3. **Class Balancing:** Balanced dataset across all 8 classes
4. **Regularization:** Dropout, batch normalization, early stopping
5. **Augmentation:** Extensive data augmentation to improve generalization

---

## 🔥 Grad-CAM Explainability

Our system uses **Gradient-weighted Class Activation Mapping (Grad-CAM)** to provide visual explanations for predictions.

### How It Works:
1. **Forward Pass:** Image passes through the model
2. **Gradient Calculation:** Compute gradients of predicted class w.r.t. last conv layer
3. **Weight Pooling:** Global average pooling of gradients
4. **Heatmap Generation:** Weighted combination of feature maps
5. **Overlay:** Superimpose heatmap on original image

### Interpretation:
- 🔴 **Red areas** = Most important for the decision
- 🟡 **Yellow areas** = Moderately important
- 🔵 **Blue areas** = Less important for classification

This helps users **trust and verify** the AI's decisions.

---

## ⚠️ Medical Disclaimer

> **This AI tool is developed for educational and research purposes only.**
>
> It is **NOT a substitute for professional medical diagnosis, advice, or treatment.**
> 
> - Always consult a qualified dermatologist for skin condition concerns
> - Do not rely solely on this tool for medical decisions
> - Early detection by professionals saves lives, especially for skin cancers
> - This tool should complement, not replace, medical expertise

---

## 🔮 Future Improvements

- [ ] Add more skin conditions (expand to 15+ classes)
- [ ] Improve accuracy for lower-performing classes (Benign Keratosis, Fungal)
- [ ] Deploy on Hugging Face Spaces for public access
- [ ] Mobile app version (Android/iOS)
- [ ] Multi-language support
- [ ] Ensemble model for higher accuracy
- [ ] Integration with electronic health records

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

You are free to:
- ✅ Use commercially
- ✅ Modify
- ✅ Distribute
- ✅ Use privately

With attribution to the original author.

---

## 👤 Author

**Mahnoor**

- 💻 GitHub: [@mahnoor-2722](https://github.com/mahnoor-2722)
- 🤗 Hugging Face: [@mahnoor-2722](https://huggingface.co/mahnoor-2722)
- 🎓 UET Taxila

---

## 🙏 Acknowledgments

- **TensorFlow & Keras** teams for the amazing deep learning framework
- **EfficientNet** authors for the powerful base architecture
- **Streamlit** for the intuitive web app framework
- **Hugging Face** for free model hosting infrastructure
- **Dermatology community** for open datasets that enable research

---

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@misc{mahnoor2026skinlesion,
  author = {Mahnoor},
  title = {Skin Lesion Classification System using EfficientNetB0},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/mahnoor-2722/skin-lesion-classification}}
}
```

---

<div align="center">

### ⭐ If you found this project helpful, please give it a star!

**Built with ❤️ for advancing medical AI research**

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?style=flat&logo=python)](https://www.python.org/)
[![Powered by TensorFlow](https://img.shields.io/badge/Powered%20by-TensorFlow-orange?style=flat&logo=tensorflow)](https://www.tensorflow.org/)

</div>