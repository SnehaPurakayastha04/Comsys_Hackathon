# Task B – Face Verification under Adverse Visual Conditions

This task addresses face verification, determining whether a distorted face image belongs to the same person as a clean reference image. The approach is based on *FaceNet embeddings* and *cosine similarity*, using metric learning concepts rather than classification.


## Dataset Structure

The dataset has a specific format where each identity folder contains:
- One clean reference image: person_id.jpg
- A folder named distortion/ with distorted variants

Example:
train/
├── ID_001/
│ ├── ID_001.jpg
│ └── distortion/
│ ├── distorted1.jpg
│ ├── distorted2.jpg
│ └── ...


## Approach Overview

- *FaceNet (InceptionResnetV1)* model pretrained on VGGFace2 is used to extract 512-dimensional embeddings
- *MTCNN* is used for face alignment before embedding extraction
- Distorted images are compared to reference images using *cosine similarity*
- Multiple *test-time augmentations* (brightness, contrast, flip) are applied and averaged to get robust embeddings
- Both *positive* and *negative pairs* are generated for validation
- A *similarity threshold* is tuned on validation data to determine match

## Evaluation Metrics

- Accuracy  : 0.9910
- Precision : 1.0000
- Recall    : 0.9890
- F1-Score  : 0.9945

These metrics are computed on the validation set using the threshold that gives the *best F1-score*.


## How to Run

> The code is written and tested in *Google Colab*

1. Open the notebook: TaskB_Face_Verification.ipynb

2. Install required libraries (automated in the notebook):

!pip install torch torchvision facenet-pytorch scikit-learn matplotlib pandas --quiet
Set the dataset paths:

python
Copy code
train_dir = '/content/drive/MyDrive/Comys_Hackathon52/Task_B/train'
val_dir   = '/content/drive/MyDrive/Comys_Hackathon52/Task_B/val'
Run all cells:

The notebook will extract embeddings, compute cosine similarity, generate negative matches, and evaluate metrics.

The final results will be printed as:

yaml
Copy code
Evaluation on Validation Set:
Accuracy : 0.XXXX
Precision: 0.XXXX
Recall   : 0.XXXX
F1-Score : 0.XXXX
💾 Model Files
FaceNet model is loaded from facenet-pytorch (no need to upload separately)

If any .pth file is used for custom layers, it may be linked from Google Drive:

markdown
Copy code
🔗 [Download taskB_model.pth](https://drive.google.com/your_link_here)
📁 Output Files
The script saves results as CSVs:

train_verification_results.csv

val_verification_results.csv

Each contains:

Person ID

Distorted image name

Similarity score

Reference ID

Label (1: match, 0: non-match)

👥 Contributors
Your Name

Friend’s Name
