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

## How to Run

> This notebook was developed and tested on **Google Colab**.

1. Open the notebook: `New_Task_A.ipynb`
2. Mount your Google Drive if needed
3. Set `train_dir` and `val_dir` to the correct dataset paths
4. Run all cells


## Evaluation Metrics

### Evaluation on Train Set:
- **Accuracy**: 0.9990
- **Precision**: 0.9990
- **Recall**   : 0.9990
- **F1-Score** : 0.9990

![final_cropped_train_confusion_matrix](https://github.com/user-attachments/assets/2379d43e-28ed-4946-b5cc-8f95e4a75122)



### Evaluation on Validation Set:
- **Accuracy**: 0.9763
- **Precision**: 0.9766
- **Recall**   : 0.9763
- **F1-Score** : 0.9760
  
  ![final_cropped_confusion_matrix](https://github.com/user-attachments/assets/6c70340c-e0bb-4590-8d01-9d3d8745724a)
    
## Model Weights
Download the pre-trained model here.
[taskA_model.pth(Google Drive)](https://drive.google.com/file/d/1XieLM15TYgOYpAZZd2u0-45m3P8KVb_u/view?usp=sharing)

##  Best Gender Model

Download best gender model here.
[best_gender_model.pth(Google Drive)](https://drive.google.com/file/d/1U5ym2yO7IDumm9TrDusCn8xZpxnTvuet/view?usp=sharing)

## Contributors

- Sneha Purakayastha(https://github.com/SnehaPurakayastha04)
- Bhumika Hazra(https://github.com/Bhumika0305)











