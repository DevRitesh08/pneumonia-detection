# Pneumonia Detection from Chest X-Rays

A production-grade medical imaging system designed to detect pneumonia from chest X-ray scans using transfer learning with a fine-tuned ResNet18 backbone. The system handles severe class imbalance and is deployed as a live interactive Streamlit dashboard.

**[Live Interactive Dashboard](https://pneumonia-detection-devritesh08.streamlit.app)**

## Technical Overview

Pneumonia is a life-threatening lung infection where early and accurate diagnosis from chest X-rays is critical. Manual diagnosis requires expert radiologists and is error-prone under high workload — especially in resource-constrained healthcare settings. This project automates the binary classification of chest X-rays into **Normal** vs **Pneumonia** using a deep learning pipeline built on ImageNet pretrained features.

The Kaggle chest X-ray dataset ([Mooney, 2018](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)) contains 5,856 images across train/val/test splits with a ~3:1 Pneumonia-to-Normal imbalance — a distribution that would cause naive models to collapse into predicting the majority class.

### 1. Transfer Learning with ResNet18

Instead of training a CNN from scratch on a small medical dataset, we leverage a ResNet18 backbone pretrained on ImageNet (1.4M images, 1,000 classes). The convolutional layers act as a universal feature extractor — by **freezing the backbone**, we eliminate 11M+ parameters from optimization and only train a custom classification head:

```
ResNet18 Backbone (frozen) → Dropout(0.3) → Linear(512, 2)
```

This approach achieves high accuracy with minimal training data, reduces compute cost, and significantly mitigates overfitting — a critical concern in medical imaging with small datasets.

### 2. Class Imbalance Handling via Weighted Sampling

The dataset is structurally imbalanced: ~3× more Pneumonia images than Normal. A naive model optimizing for accuracy would learn to always predict Pneumonia. We address this using PyTorch's `WeightedRandomSampler`:

- Each class is assigned an inverse-frequency weight
- The sampler oversamples the minority class (Normal) during each epoch
- **Result:** Balanced gradient updates regardless of raw class distribution — the model learns both classes rigorously

### 3. Data Augmentation Pipeline

Training images are processed through a stochastic augmentation pipeline to artificially increase dataset diversity and suppress overfitting:

| Augmentation | Purpose |
|---|---|
| `RandomResizedCrop(224)` | Scale & position invariance |
| `RandomHorizontalFlip` | Mirror symmetry in X-rays |
| `RandomRotation(10°)` | Slight angular variation |
| `ColorJitter` | Brightness/contrast robustness |
| `ImageNet Normalization` | Aligned with pretrained backbone statistics |

Test/val images use only `Resize → CenterCrop → Normalize` (no augmentation) to ensure deterministic evaluation.

### 4. Training Configuration

| Hyperparameter | Value |
|---|---|
| Backbone | ResNet18 (ImageNet pretrained) |
| Classifier Head | Dropout(0.3) + Linear |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Weight Decay | 1e-4 |
| Epochs | 10 |
| Batch Size | 32 |
| Loss Function | CrossEntropyLoss |

## Project Structure

```
pneumonia-detection/
├── app.py                  # Streamlit frontend for live X-ray inference
├── main.py                 # Central training orchestrator
├── requirements.txt        # Python dependencies
├── pneumonia_model.pth     # Trained model weights (ready for inference)
├── src/
│   ├── config.py           # All hyperparameters and constants
│   ├── data.py             # DataLoaders, augmentation, and balanced sampling
│   ├── model.py            # ResNet18 architecture and training loop
│   └── evaluate.py         # Metrics, confusion matrix, and training curves
└── showcase/               # Research notebook with full EDA and experimentation
```

## Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/DevRitesh08/pneumonia-detection.git
   cd pneumonia-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Dataset**
   Download from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and extract to `data/`:
   ```
   data/
     train/
       NORMAL/
       PNEUMONIA/
     val/
       NORMAL/
       PNEUMONIA/
     test/
       NORMAL/
       PNEUMONIA/
   ```

4. **Train the Model**
   *(The trained `pneumonia_model.pth` is already included for immediate inference — skip this step to go straight to the dashboard.)*
   ```bash
   python main.py
   ```

5. **Launch the Dashboard**
   ```bash
   python -m streamlit run app.py
   ```

## Disclaimer

> ⚠️ This tool is for **educational and research purposes only**. It is not a certified medical device and must not be used for clinical diagnosis. Always consult a qualified healthcare professional for medical advice.
