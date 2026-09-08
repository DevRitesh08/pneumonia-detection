# Pneumonia Detection from Chest X-Rays

A deep learning system for detecting pneumonia from chest X-ray images using transfer learning with ResNet18. The system handles severe class imbalance and is deployed as an interactive Streamlit dashboard.

## Technical Overview

Pneumonia is a life-threatening lung infection where early diagnosis from chest X-rays is critical. Manual diagnosis requires expert radiologists and is error-prone under high workload. This project automates the classification of chest X-rays into **Normal** vs **Pneumonia** using a fine-tuned ResNet18 backbone.

### 1. Transfer Learning with ResNet18

Instead of training a CNN from scratch on a small medical dataset, we leverage a ResNet18 model pre-trained on ImageNet (1.4M images, 1000 classes). The backbone layers act as a powerful feature extractor — we freeze them and only train a custom classification head (Dropout + Linear), reducing trainable parameters from 11M to ~1K while achieving high accuracy.

### 2. Class Imbalance Handling

The Kaggle chest X-ray dataset is imbalanced (~3x more Pneumonia than Normal images). A naive model would bias toward predicting Pneumonia. We address this using `WeightedRandomSampler` which oversamples the minority class during training, ensuring balanced gradient updates.

### 3. Data Augmentation

Training images are augmented with random crops, horizontal flips, rotations, and color jitter to artificially increase dataset diversity and reduce overfitting — critical when working with small medical datasets.

## Project Structure

- `showcase/` — Research notebook with full EDA and experimentation
- `main.py` — Central training orchestrator
- `app.py` — Streamlit frontend for live X-ray inference
- `src/config.py` — All hyperparameters and constants
- `src/data.py` — Data loading, augmentation, and balanced sampling
- `src/model.py` — ResNet18 architecture and training loop
- `src/evaluate.py` — Metrics, confusion matrix, and training curves

## Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/DevRitesh08/pneumonia-detection.git
   cd pneumonia-detection
   ```

2. **Download Dataset**
   Download from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and extract to `data/`:
   ```
   data/
     train/
       NORMAL/
       PNEUMONIA/
     test/
       NORMAL/
       PNEUMONIA/
   ```

3. **Train the Model**
   *(The trained `pneumonia_model.pth` is included for immediate inference.)*
   ```bash
   python main.py
   ```

4. **Launch the Dashboard**
   ```bash
   python -m streamlit run app.py
   ```
