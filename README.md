# Comsys_Hackathon

This repository contains the complete implementation for both tasks of the COMSYS Hackathon-5 challenge:

- 🔹 **Task A:** Gender Classification using CNN
- 🔹 **Task B:** Face Verification under Adverse Visual Conditions using FaceNet and Metric Learning

---

## 📁 Repository Structure

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




## 🔍 Task A: Gender Classification

### 📌 Description:
A convolutional neural network (CNN) was trained to classify gender from face images into **Male** and **Female** classes using cross-entropy loss.

### ✅ Dataset Format:
train/
├── Male/
├── Female/
val/
├── Male/
├── Female/

markdown
Copy
Edit

### 📊 Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-score

### 📎 Files:
- `TaskA_Gender_Classification.ipynb` – Colab notebook for training & evaluation
- `taskA_model.pth` – Pretrained weights
- `README.md` – Task A overview and instructions

---

## 🔐 Task B: Face Verification

### 📌 Description:
This task involves verifying if a distorted face image matches a clean reference image using **FaceNet embeddings** and **cosine similarity**.  
The system handles:
- Positive matches (same identity)
- Negative matches (different identities)

### ✅ Dataset Format:
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

yaml
Copy
Edit

### 💡 Approach:
- Used **InceptionResNetV1** (FaceNet pretrained on VGGFace2)
- Generated embeddings with test-time data augmentation (brightness, contrast, flipping)
- Calculated **cosine similarity**
- Tuned threshold based on validation F1-score
- Included **negative matches** for generalizability

### 📊 Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-score

### 📎 Files:
- `TaskB_Face_Verification.ipynb` – End-to-end training, testing, and metric computation
- `taskB_model.pth` – FaceNet model weights (optional, FaceNet can be loaded in Colab too)
- `README.md` – Task B overview and instructions

---

## 🛠 How to Run

> You can open and run each task's notebook directly in **Google Colab**.

### 🔗 Run in Colab
1. Open the notebook (e.g., `TaskA_Gender_Classification.ipynb`)
2. Mount your Google Drive if needed
3. Update dataset paths (`train_dir`, `val_dir`)
4. Run all cells
5. Metrics will be printed at the end

---

## 📈 Results Summary

### ✅ Task A (Gender Classification)
| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.XXXX  |
| Precision  | 0.XXXX  |
| Recall     | 0.XXXX  |
| F1-Score   | 0.XXXX  |

### ✅ Task B (Face Verification)
| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.XXXX  |
| Precision  | 0.XXXX  |
| Recall     | 0.XXXX  |
| F1-Score   | 0.XXXX  |
| Best Threshold | 0.XX |

*(Replace with your actual values.)*

---

## 👥 Contributors

- [Your Name](https://github.com/yourusername)
- [Your Friend’s Name](https://github.com/theirusername)

---

## 📩 Submission Instructions
This GitHub repository contains:
- ✅ Well-documented code
- ✅ Pretrained model weights
- ✅ Notebooks for both tasks
- ✅ Evaluation metrics printed inside notebooks

Ready for submission via Google Form once released.

---

Let me know if you'd like to auto-generate the Task A or Task B README.md inside each subfolder too — I can help with that instantly.










