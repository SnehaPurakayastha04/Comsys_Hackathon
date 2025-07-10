# Task A – Gender Classification

This task involves classifying face images as **Male** or **Female** using a Convolutional Neural Network (CNN).

## This repository contains solutions both Task A of the Comsys Hackathon. All required model weights, training/validation results, and instructions are included inside this file.

##  Dataset Structure

The dataset is divided into `train` and `val` (validation) sets with images grouped into two folders: `Male/` and `Female/`.

dataset/
├── train/
│ ├── Male/
│ └── Female/
└── val/
├── Male/
└── Female/

## Approach

- Used a custom CNN architecture with convolutional, batch normalization, and dense layers.
- Applied data augmentation to improve generalization.
- Trained using **cross-entropy loss**.
- Evaluated using standard classification metrics.

## Model Architecture (Summary)

- Conv2D → ReLU → MaxPool (x3)
- Flatten
- Fully Connected → ReLU → Dropout
- Final Layer: Output = 2 (Male / Female)


## Evaluation Metrics

### Evaluation on Train Set:
- **Accuracy**: 0.9990
- **Precision**: 0.9990
- **Recall**   : 0.9990
- **F1-Score** : 0.9990

![PHOTO-2025-07-05-17-06-54](https://github.com/user-attachments/assets/6bd7d517-7155-46a1-bcd9-45c2609e67ab)


### Evaluation on Validation Set:
- **Accuracy**: 0.9763
- **Precision**: 0.9766
- **Recall**   : 0.9763
- **F1-Score** : 0.9760
  
  ![final_cropped_confusion_matrix](https://github.com/user-attachments/assets/6c70340c-e0bb-4590-8d01-9d3d8745724a)
    

##  Best Gender Model

Download best gender model here.
[best_gender_model.pth(Google Drive)](https://drive.google.com/file/d/1U5ym2yO7IDumm9TrDusCn8xZpxnTvuet/view?usp=sharing)

## Contributors

- Sneha Purakayastha(https://github.com/SnehaPurakayastha04)
- Bhumika Hazra(https://github.com/Bhumika0305)











