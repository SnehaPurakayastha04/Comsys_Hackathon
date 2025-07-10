# Task B – Face Verification under Adverse Visual Conditions

This task addresses face verification, determining whether a distorted face image belongs to the same person as a clean reference image. The approach is based on *FaceNet embeddings* and *cosine similarity*, using metric learning concepts rather than classification.

## This repository contains solutions for Task B of the Comsys Hackathon. All training/validation results, and instructions are included inside this file.


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
## Evaluation on Train Set:
Best Threshold: 0.44
- Accuracy  : 0.9888
- Precision : 0.9964
- Recall    : 0.9893
- F1-Score  : 0.9929

![Screenshot 2025-07-05 090938](https://github.com/user-attachments/assets/882b4b06-f035-488c-8402-8ef48712be2a)


 ## Evaluation on Validation Set:
 Best Threshold: 0.39
 - Accuracy  : 0.9880
 - Precision : 0.9945
 - Recall    : 0.9890
 - F1-Score  : 0.9917

![Screenshot 2025-07-05 091010](https://github.com/user-attachments/assets/48e0e47a-39c7-41e1-8c50-a82cd3e2b0e3)


## Contributors

- Sneha Purakayastha(https://github.com/SnehaPurakayastha04)
- Bhumika Hazra(https://github.com/Bhumika0305)
