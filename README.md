# Comsys_Hackathon
Face recognition algorithms often demonstrate degraded performance when confronted with images captured in non-ideal environments such as in blur, fog, rain, low-light, or overexposed scenes. To bridge this gap, we developed models that maintain consistent performance despite these challenges. Additionally, a concurrent task of gender classification is included to evaluate the robustness of feature representations across different semantic attributes.


This repository contains the complete implementation for both tasks of the COMSYS Hackathon-5 challenge:

-  **Task A:** Gender Classification using CNN
-  **Task B:** Face Verification under Adverse Visual Conditions using FaceNet and Metric Learning


## Repository Structure

Comsys_Hackathon5/
├── test.py
├── README.md
├── train_verification_results.csv
├── val_verification_results.csv
├── TaskA/
│ ├── test_code(taska).py
│ ├── README.md
│ ├── train_code(taska).py
│ └── Task_A(Model Diagram).png
│
├── TaskB/
│ ├── test_code(taskb).py
│ ├── train_code(taskb).py
| ├──README.md
│ └── Task_B(Model Diagram).png


# Setup Instructions

Clone the repository:

git clone https://github.com/SnehaPurakayastha04/Comsys_Hackathon.git

cd Comsys_Hackathon

Install required libraries:

Install all dependencies using pip:

pip install torch torchvision facenet-pytorch scikit-learn pandas matplotlib seaborn

Download Model Weights:

Task A: [TaskA_best_gender_model(Google Drive)](https://drive.google.com/file/d/1U5ym2yO7IDumm9TrDusCn8xZpxnTvuet/view?usp=drive_link)

Task B: [TaskB_model(Google Drive)](https://drive.google.com/file/d/1lrgSgV2Bado7IATlQJEHL5o3ENDaSL6m/view?usp=drive_link)

## Important: Download best_gender_model.pth from the above link and place it in the root directory (same location as test.py) before running Task A.

# How to Run

Run the final test script from the root directory:

python test.py --task A --test_folder /path/to/test/folder
python test.py --task B --test_folder /path/to/test/folder
Ensure /test_folder/Task_A/val/ and /test_folder/Task_B/val/ follow the same structure used during training.


# Output Format

The script prints performance metrics separately for Task A and Task B, for both train and validation datasets:

Task A:
Train Results: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
Validation Results: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

Task B:
Train Set: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
Validation Set: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## Task A: Gender Classification

### Description:
A convolutional neural network (CNN) was trained to classify gender from face images into **Male** and **Female** classes using cross-entropy loss.

### Dataset Format:
train/
├── Male/
├── Female/
val/
├── Male/
├── Female/

## Approach
Base model: ResNet50 (pretrained on ImageNet)
Final layers replaced with:
Linear(in_features, 256)
ReLU activation
Dropout(0.5)
Linear(256, 2)
Label smoothing with CrossEntropyLoss(label_smoothing=0.1)
Optimizer: AdamW with learning rate decay via StepLR
Regularization via Dropout + Data augmentation
Early Stopping based on validation accuracy


## Task B: Face Verification

###  Description:
This task involves verifying if a distorted face image matches a clean reference image using **FaceNet embeddings** and **cosine similarity**.  
The system handles:
- Positive matches (same identity)
- Negative matches (different identities)

### Dataset Format:
train/
├── ID_001/
│ ├── ID_001.jpg
│ └── distortion/
│ ├── distorted1.jpg
│ ├── ...
val/
├── ID_002/
│ ├── ID_002.jpg
│ └── distortion/
│ ├── distorted1.jpg
│ ├── ...


### Approach:
- Used **InceptionResNetV1** (FaceNet pretrained on VGGFace2)
- Generated embeddings with test-time data augmentation (brightness, contrast, flipping)
- Calculated **cosine similarity**
- Tuned threshold based on validation F1-score
- Included **negative matches** for generalizability


##  Results Summary

### Task A (Gender Classification)

## Train Set
| Metric     | Value   |                             
|------------|---------|              
| Accuracy   | 0.9990  |             
| Precision  | 0.9990  |              
| Recall     | 0.9990  |              
| F1-Score   | 0.9990  | 

## Validation Set
| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.9763  |
| Precision  | 0.9766  |
| Recall     | 0.9763  |
| F1-Score   | 0.9760  |

### Task B (Face Verification)

## Train Set                          
| Metric     | Value   |                             
|------------|---------|              
| Accuracy   | 0.9888  |             
| Precision  | 0.9964  |              
| Recall     | 0.9893  |              
| F1-Score   | 0.9929  |          
| Best Threshold | 0.44 |             

## Validation Set
| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.9880  |
| Precision  | 0.9945  |
| Recall     | 0.9890  |
| F1-Score   | 0.9917  |
| Best Threshold | 0.39 |

## Contributors

- Sneha Purakayastha(https://github.com/SnehaPurakayastha04)
- Bhumika Hazra(https://github.com/Bhumika0305)













