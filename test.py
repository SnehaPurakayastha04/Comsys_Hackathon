# ✅ Combined test.py script for Colab (Task A & B)

import sys
import argparse
import os
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.models import ResNet50_Weights

# ✅ Simulate argparse for Colab
sys.argv = ['test.py', '--task', 'A', '--test_folder', '/content/drive/MyDrive/Comys_Hackathon5/Comys_Hackathon5']

# ✅ Evaluation for Classification (Task A)
def evaluate_classification(model, loader, class_names, device, name="Validation"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted')
    rec = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"\n\U0001F4CA {name} Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# ✅ Run Task A

def run_task_a(test_folder):
    print("\n\u25B6\ufe0f Running Task A: Gender Classification")
    model_path = os.path.join("/content/drive/MyDrive/best_gender_model.pth")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_path = os.path.join(test_folder, 'Task_A', 'train')
    val_path = os.path.join(test_folder, 'Task_A', 'val')

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    val_dataset = datasets.ImageFolder(val_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    class_names = train_dataset.classes

    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    evaluate_classification(model, train_loader, class_names, device, name="Train")
    evaluate_classification(model, val_loader, class_names, device, name="Validation")

# ✅ Evaluation for Verification (Task B)

def evaluate_verification(df, dataset_name='Dataset'):
    y_true = df['label'].tolist()
    best_thresh = 0.5
    best_f1 = 0
    for thresh in np.arange(0.3, 0.8, 0.01):
        preds = [1 if s >= thresh else 0 for s in df['similarity']]
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_thresh = thresh
            best_f1 = f1

    y_pred = [1 if s >= best_thresh else 0 for s in df['similarity']]
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n\U0001F4CA Evaluation on {dataset_name}:")
    print(f"\U0001F50D Best Threshold: {best_thresh:.2f}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    labels = ['Negative (0)', 'Positive (1)']
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'{dataset_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# ✅ Run Task B

def run_task_b(test_folder):
    print("\n\u25B6\ufe0f Running Task B: Face Verification")
    train_csv = os.path.join(test_folder, 'train_verification_results.csv')
    val_csv = os.path.join(test_folder, 'val_verification_results.csv')
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    evaluate_verification(df_train, dataset_name='Train Set')
    evaluate_verification(df_val, dataset_name='Validation Set')

# ✅ Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['A', 'B'], help='Task A or B')
    parser.add_argument('--test_folder', type=str, required=True, help='Path to test folder')
    args = parser.parse_args()

    if args.task == 'A':
        run_task_a(args.test_folder)
    elif args.task == 'B':
        run_task_b(args.test_folder)

if __name__ == '__main__':
    main()