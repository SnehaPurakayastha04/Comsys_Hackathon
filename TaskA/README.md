# Task A – Gender Classification

This task involves classifying face images as **Male** or **Female** using a Convolutional Neural Network (CNN).

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

## How to Run

> This notebook was developed and tested on **Google Colab**.

1. Open the notebook: `New_Task_A.ipynb`
2. Mount your Google Drive if needed
3. Set `train_dir` and `val_dir` to the correct dataset paths
4. Run all cells


## Evaluation Metrics

After training and validation, the following metrics are computed:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**
  
    ![Screenshot 2025-06-27 120955](https://github.com/user-attachments/assets/32150966-2380-458a-9917-d14de576e606)

##  Model File

- Pretrained weights: `taskA_model.pth` 

## Contributors

- Sneha Purakayastha(https://github.com/SnehaPurakayastha04)
- Bhumika Hazra(https://github.com/Bhumika0305)











