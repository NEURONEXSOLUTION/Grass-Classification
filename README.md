# Grass Classification using Deep Learning

This project is a **Deep Learning-based image classification system** that identifies whether a given image contains **Dry Grass** or **Green Grass**.  
It is built using **PyTorch** for model training and **Flask** for serving a simple web interface to upload and classify grass images.

---

## Features

- Classifies grass images into two categories:
  - Dry Grass  
  - Green Grass
- Data augmentation for better generalization.
- Balanced data loading using weighted sampling.
- Flask-based web app for real-time image prediction.
- Deployed on a free hosting platform (e.g., Render / Hugging Face Spaces / GitHub Pages backend).

---

## Model Details

- **Framework:** PyTorch  
- **Architecture:** CNN-based custom model (or transfer learning backbone like ResNet18, if used)
- **Input Size:** `224 x 224`
- **Optimizer:** Adam
- **Loss Function:** Cross Entropy Loss
- **Performance Metrics:** Accuracy, Precision, Recall

---

## Dataset Distribution

**Training Set**
| Class | Images |
|--------|--------|
| Dry_Grass | 337 |
| Green_Grass | 120 |

**Validation Set**
| Class | Images |
|--------|--------|
| Dry_Grass | 113 |
| Green_Grass | 40 |

To handle imbalance, a **WeightedRandomSampler** was applied during training.

---

## Data Transformations

### Training Transformations
```python
transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.Lambda(repeat_channels)
])


**Create Virtual Environment**
python -m venv venv
venv\Scripts\activate  # (Windows)
# or
source venv/bin/activate  # (Linux/Mac)


**Install Dependencies**
pip install -r requirements.txt


**Run Flask App**
python app.py
