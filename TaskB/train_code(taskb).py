import os
import random
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import InceptionResnetV1, MTCNN

# Setup device and models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(image_size=160, margin=10, device=device)

@torch.no_grad()
def get_augmented_embeddings(img_path, n_augments=3):
    img = Image.open(img_path).convert('RGB')
    aligned = mtcnn(img)
    if aligned is None:
        return None

    variants = [aligned.cpu()]
    for _ in range(n_augments):
        aug_img = img.copy()
        aug_img = ImageEnhance.Brightness(aug_img).enhance(np.random.uniform(0.8, 1.2))
        aug_img = ImageEnhance.Contrast(aug_img).enhance(np.random.uniform(0.8, 1.2))
        if np.random.rand() > 0.5:
            aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
        aligned_aug = mtcnn(aug_img)
        if aligned_aug is not None:
            variants.append(aligned_aug.cpu())

    embeddings = []
    for v in variants:
        v = v.unsqueeze(0).to(device)
        emb = model(v).cpu().numpy().flatten()
        embeddings.append(emb)

    return np.mean(embeddings, axis=0)

def verify_from_folder(folder_path, save_csv_path, use_negatives=True, negatives_per_sample=3):
    results = []
    person_list = os.listdir(folder_path)

    for person_id in tqdm(person_list, desc=f"Processing {os.path.basename(folder_path)}"):
        person_dir = os.path.join(folder_path, person_id)
        ref_img_path = os.path.join(person_dir, f"{person_id}.jpg")
        distortion_folder = os.path.join(person_dir, "distortion")

        if not os.path.exists(ref_img_path) or not os.path.isdir(distortion_folder):
            continue

        ref_embedding = get_augmented_embeddings(ref_img_path)
        if ref_embedding is None:
            continue

        for distorted_img_name in os.listdir(distortion_folder):
            distorted_img_path = os.path.join(distortion_folder, distorted_img_name)
            distorted_embedding = get_augmented_embeddings(distorted_img_path)
            if distorted_embedding is None:
                continue

            similarity = cosine_similarity([ref_embedding], [distorted_embedding])[0][0]
            results.append({
                'person_id': person_id,
                'distorted_img': distorted_img_name,
                'reference_id': person_id,
                'similarity': similarity,
                'label': 1
            })

            if use_negatives:
                other_people = [p for p in person_list if p != person_id]
                random.shuffle(other_people)
                for neg_id in other_people[:negatives_per_sample]:
                    neg_ref_path = os.path.join(folder_path, neg_id, f"{neg_id}.jpg")
                    if os.path.exists(neg_ref_path):
                        neg_embedding = get_augmented_embeddings(neg_ref_path)
                        if neg_embedding is not None:
                            sim_neg = cosine_similarity([neg_embedding], [distorted_embedding])[0][0]
                            results.append({
                                'person_id': person_id,
                                'distorted_img': distorted_img_name,
                                'reference_id': neg_id,
                                'similarity': sim_neg,
                                'label': 0
                            })

    df = pd.DataFrame(results)
    df.to_csv(save_csv_path, index=False)
    print(f"Saved results to {save_csv_path}")
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_folder', type=str, required=True, help='Path to Task_B train folder')
    parser.add_argument('--val_folder', type=str, required=True, help='Path to Task_B val folder')
    parser.add_argument('--output_dir', type=str, default='.', help='Where to save output CSVs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_csv = os.path.join(args.output_dir, 'train_verification_results.csv')
    val_csv = os.path.join(args.output_dir, 'val_verification_results.csv')

    verify_from_folder(args.train_folder, train_csv)
    verify_from_folder(args.val_folder, val_csv)

if __name__ == '__main__':
    main()
