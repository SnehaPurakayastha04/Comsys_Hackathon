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
  
    ![Screenshot 2025-07-02 235553](https://github.com/user-attachments/assets/87f096fc-7b2d-40ac-867c-81b9c14bc7da)


##  Model File

Download the pre-trained model here.
 [model.pth(Google Drive)](https://drive.google.com/file/d/11EJ9J02rdOzUovpGogrnzkcnZJsfIX7b/view?usp=sharing)

##  Best Gender Model

Download best gender model here.
[best_gender_model.pth(Google Drive)](https://drive.google.com/file/d/1Gu5I39dwx--IGKLM_QtQ6MByF4mBKqnX/view?usp=sharing)

## Contributors

- Sneha Purakayastha(https://github.com/SnehaPurakayastha04)
- Bhumika Hazra(https://github.com/Bhumika0305)











