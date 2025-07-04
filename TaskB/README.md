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

- Accuracy  : 0.9836
- Precision : 0.9890
- Recall    : 0.9890
- F1-Score  : 0.9890

These metrics are computed on the validation set using the threshold that gives the *best F1-score*.
![Screenshot 2025-07-02 235628](https://github.com/user-attachments/assets/bee81729-a1dd-4836-94af-3ef0afae4fc1)



## How to Run

- Open the notebook: Train_Code(TaskB).ipynb and Test_Code(TaskB).ipynb
- Mount your Google Drive if needed
- Set train_dir and val_dir to the correct dataset paths
- Run all cells

## Model Weights
Download the pre-trained model here.
[model.pth(Google Drive)](https://drive.google.com/file/d/1AqHqQsepm7iyT45-cI4DnhqnL_0J_lBi/view?usp=sharing)

## Contributors

- Sneha Purakayastha(https://github.com/SnehaPurakayastha04)
- Bhumika Hazra(https://github.com/Bhumika0305)
