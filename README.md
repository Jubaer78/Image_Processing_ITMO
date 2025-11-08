# CIFAR-10 Image Classification using EfficientNetB0 + TensorBoard

## 🧩 1. Task Description
This project solves the **image classification** problem using a **deep convolutional neural network (EfficientNetB0)**.  
The main goals:
- Train a model to classify 10 object categories from the CIFAR-10 dataset.
- Visualize training performance (loss, accuracy) in **TensorBoard**.
- Evaluate precision, recall, F1-score for each class.
- Fine-tune the pretrained EfficientNet model for better accuracy.

**Tasks implemented:**
- Data loading, visualization, and preprocessing (normalization, one-hot encoding).
- Model creation using transfer learning.
- Model training in two stages:  
  1️⃣ Frozen base model training  
  2️⃣ Fine-tuning selected layers  
- Experiment tracking with TensorBoard (loss & accuracy curves).
- Model evaluation and metrics visualization (confusion matrix, class metrics).

---

## 🗂️ 2. Dataset Description

**Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

<!-- Image in the same directory -->
![Dataset](./dataset.png)

**Composition:**
- **Total images:** 60,000  
  - 50,000 for training  
  - 10,000 for testing
- **Classes (10 categories):**
  - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Image size:** 32x32 pixels, RGB (3 channels)

---

## 🧠 3. Learning Results & Experiments

**Model:** EfficientNetB0 (pretrained on ImageNet, fine-tuned on CIFAR-10)

### Training Setup
- Optimizer: `Adam`
- Loss: `categorical_crossentropy`
- Epochs: 10
- Batch sizes: 64 (stage 1), 32 (stage 2)
- Callbacks:
  - EarlyStopping (patience=10)
  - ReduceLROnPlateau
  - TensorBoard (for logging)

### Experiment Results
| Metric | Value |
|--------:|:------|
| Final Test Accuracy | 0.3620 |
| Final Test Loss | 1.7113 |
| Precision  | 0.4692 |
| Recall  | 0.3620 |
| F1-Score | 0.3135 |

![Value of Accuracy](./accuracy.png)
![Value of Loss](./loss.png)
## 📊 Classification Report

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| airplane | 0.6792 | 0.0720 | 0.1302 |
| automobile | 0.5975 | 0.3400 | 0.4334 | 
| bird | 0.4654 | 0.0740 | 0.1277 | 
| cat | 0.2450 | 0.1470 | 0.1837 | 
| deer | 0.5116 | 0.0440 | 0.0810 | 
| dog | 0.3737 | 0.4630 | 0.4136 |
| frog | 0.2527 | 0.8870 | 0.3933 |
| horse | 0.7254 | 0.2510 | 0.3730 | 
| ship | 0.3547 | 0.8200 | 0.4952 |
| truck | 0.4865 | 0.5220 | 0.5036 |


## 📊 4. Visualizations
- Training/validation loss and accuracy curves (via TensorBoard)
- Confusion matrix
- Class-wise precision, recall, and F1 comparison
- Sample predictions with confidence scores

![Sample Predictions](./sample.png)
---

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
