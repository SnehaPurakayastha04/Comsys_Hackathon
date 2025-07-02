# Comsys_Hackathon
Face recognition algorithms often demonstrate degraded performance when confronted with images captured in non-ideal environments such as in blur, fog, rain, low-light, or overexposed scenes. To bridge this gap, we developed models that maintain consistent performance despite these challenges. Additionally, a concurrent task of gender classification is included to evaluate the robustness of feature representations across different semantic attributes.


This repository contains the complete implementation for both tasks of the COMSYS Hackathon-5 challenge:

- 🔹 **Task A:** Gender Classification using CNN
- 🔹 **Task B:** Face Verification under Adverse Visual Conditions using FaceNet and Metric Learning


## Repository Structure

Comsys_Hackathon5/
├── TaskA_Gender_Classification/
│ ├── TaskA_Gender_Classification.ipynb
│ ├── taskA_model.pth
│ └── README.md
│
├── TaskB_Face_Verification/
│ ├── TaskB_Face_Verification.ipynb
│ ├── taskB_model.pth
│ └── README.md




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


###  Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-score

### Files:
- `TaskA_Gender_Classification.ipynb` – Colab notebook for training & evaluation
- `taskA_model.pth` – Pretrained weights
- `README.md` – Task A overview and instructions

---

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

### Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-score

### Files:
- `TaskB_Face_Verification.ipynb` – End-to-end training, testing, and metric computation
- `taskB_model.pth` – FaceNet model weights (optional, FaceNet can be loaded in Colab too)
- `README.md` – Task B overview and instructions

---

##  How to Run

> You can open and run each task's notebook directly in **Google Colab**.

### Run in Colab
1. Open the notebook (e.g., `TaskA_Gender_Classification.ipynb`)
2. Mount your Google Drive if needed
3. Update dataset paths (`train_dir`, `val_dir`)
4. Run all cells
5. Metrics will be printed at the end

---

##  Results Summary

### Task A (Gender Classification)
| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.9763  |
| Precision  | 0.9850  |
| Recall     | 0.9500  |
| F1-Score   | 0.9650  |

### Task B (Face Verification)
| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.9910  |
| Precision  | 1.0000  |
| Recall     | 0.9890  |
| F1-Score   | 0.9945  |
| Best Threshold | 0.40 |

## Contributors

- Sneha Purakayastha(https://github.com/SnehaPurakayastha04)
- Bhumika Hazra(https://github.com/Bhumika0305)













