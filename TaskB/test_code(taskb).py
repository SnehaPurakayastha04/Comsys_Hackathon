# STEP 1: Install & Import
!pip install scikit-learn pandas matplotlib seaborn --quiet

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# STEP 2: Load CSVs
df_train = pd.read_csv('/content/train_verification_results.csv')
df_val = pd.read_csv('/content/val_verification_results.csv')

# STEP 3: Metric Function
def evaluate(df, dataset_name='Dataset'):
    y_true = df['label'].tolist()
    best_thresh = 0.5
    best_f1 = 0

    # Threshold Tuning
    for thresh in np.arange(0.3, 0.8, 0.01):
        preds = [1 if s >= thresh else 0 for s in df['similarity']]
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    # Final Evaluation
    y_pred = [1 if s >= best_thresh else 0 for s in df['similarity']]
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Print Metrics
    print(f"\n Evaluation on {dataset_name}:")
    print(f" Best Threshold: {best_thresh:.2f}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Negative (0)', 'Positive (1)']
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'{dataset_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# STEP 4: Run for Train and Val
evaluate(df_train, dataset_name='Train Set')
evaluate(df_val, dataset_name='Validation Set')

torch.save(model.state_dict(), 'facenet_vggface2.pth')
