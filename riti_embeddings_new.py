import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    CLIPModel, CLIPTokenizer, CLIPProcessor, CLIPImageProcessor,
    BlipProcessor, BlipForConditionalGeneration, BlipForImageTextRetrieval
)

# Constants
IMAGE_MAIN_DIRECTORY = "small/"
SHOE_PRODUCT_TYPES = ["SHOES", "SANDAL", "BOOT", "TECHNICAL_SPORT_SHOE"]

# Load CLIP models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
clip_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# Load BLIP models
blip_caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
blip_caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_caption_model.eval()

blip_retrieval_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco", use_fast=True)
blip_retrieval_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
blip_retrieval_model.eval()

def embed_with_clip(text=None, image_path=None):
    if text:
        inputs = clip_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=77
        )
        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.squeeze().cpu()

    if image_path:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.squeeze().cpu()

def construct_metadata_text(row):
    parts = [
        row['title'],
        row.get('style', ''),
        row.get('color', ''),
        " ".join(row['bulletPoints']) if row.get('bulletPoints') else '',
        " ".join(row['keywords']) if row.get('keywords') else '',
        row.get('productType', '')
    ]
    return ". ".join(filter(None, parts))

def query_clip(text=None, image_path=None, df=None, top_k=6):
    components = []
    if text:
        components.append(embed_with_clip(text=text))
    if image_path:
        components.append(embed_with_clip(image_path=image_path))
    if not components:
        raise ValueError("Provide at least one of text or image.")

    query_vector = torch.stack(components).mean(dim=0)
    query_vector = query_vector / query_vector.norm()

    item_vectors = torch.stack([
        (i + t) / 2 for i, t in zip(df['clipImageEmbedding'], df['clipTextEmbedding'])
    ])
    item_vectors = item_vectors / item_vectors.norm(dim=1, keepdim=True)

    sims = cosine_similarity(query_vector.unsqueeze(0).cpu(), item_vectors.cpu())[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return df.iloc[top_indices]

def query_blip(text, df, top_k=6):
    scores = []
    for idx, row in df.iterrows():
        image_path = IMAGE_MAIN_DIRECTORY + row["path"]
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = blip_retrieval_processor(images=image, text=text, return_tensors="pt")
            with torch.no_grad():
                outputs = blip_retrieval_model(**inputs)
            score = outputs.itm_score.squeeze()[0].item() if outputs.itm_score.ndim == 2 else outputs.itm_score.item()
            scores.append((score, idx))
        except:
            continue
    top = sorted(scores, key=lambda x: x[0], reverse=True)[:top_k]
    return df.iloc[[idx for (_, idx) in top]]

def show_image_results(results_df):
    fig, axes = plt.subplots(1, len(results_df), figsize=(4 * len(results_df), 5))
    if len(results_df) == 1:
        axes = [axes]
    for i, (_, row) in enumerate(results_df.iterrows()):
        img = Image.open(IMAGE_MAIN_DIRECTORY + row['path'])
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"{row.get('title', '')[:30]}...")
    plt.tight_layout()
    plt.show()

def generate_caption(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = blip_caption_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        out = blip_caption_model.generate(**inputs)
        return blip_caption_processor.decode(out[0], skip_special_tokens=True)
    
def get_blip_similarity_score(image_path, text_query):
    image = Image.open(image_path).convert('RGB')
    inputs = blip_retrieval_processor(images=image, text=text_query, return_tensors="pt")
    with torch.no_grad():
        outputs = blip_retrieval_model(**inputs)
    scores = outputs.itm_score

    # Handle [1, 2] shape → return first score
    if scores.ndim == 2 and scores.shape == (1, 2):
        return scores[0, 0].item()
    elif scores.ndim == 1:
        return scores[0].item()
    elif scores.ndim == 0:
        return scores.item()
    else:
        raise ValueError(f"Unexpected itm_score shape: {scores.shape}")

def query_blip_with_image_and_text(image_path, text_query, df, top_k=6):
    """
    Uses the image to generate a caption, combines it with the user text,
    and scores all dataset images using BLIP.
    """
    try:
        image_caption = generate_caption(image_path)
        full_query = f"{text_query.strip()}. {image_caption.strip()}"
    except Exception as e:
        print(f"Failed to generate caption from query image: {e}")
        full_query = text_query

    scores = []
    for idx, row in df.iterrows():
        try:
            candidate_path = IMAGE_MAIN_DIRECTORY + row["path"]
            score = get_blip_similarity_score(candidate_path, full_query)
            scores.append((score, idx))
        except Exception as e:
            print(f"Error scoring {row['path']}: {e}")
            continue

    top = sorted(scores, key=lambda x: x[0], reverse=True)[:top_k]
    return df.iloc[[idx for (_, idx) in top]]

def query_blip_with_image_and_text_batch(image_path, text_query, df, top_k=6, batch_size=32):
    try:
        image_caption = generate_caption(image_path)
        full_query = f"{text_query.strip()}. {image_caption.strip()}"
    except Exception as e:
        print(f"Failed to generate caption from query image: {e}")
        full_query = text_query

    image_paths = [IMAGE_MAIN_DIRECTORY + row['path'] for _, row in df.iterrows()]
    all_images, all_indices = [], []

    for idx, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert('RGB')
            all_images.append(img)
            all_indices.append(idx)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

    scores = []
    for i in range(0, len(all_images), batch_size):
        batch_imgs = all_images[i:i+batch_size]
        batch_idxs = all_indices[i:i+batch_size]
        inputs = blip_retrieval_processor(images=batch_imgs, text=[full_query]*len(batch_imgs), return_tensors='pt', padding=True)
        with torch.no_grad():
            outputs = blip_retrieval_model(**inputs)
            batch_scores = outputs.itm_score.squeeze()

        for s, idx in zip(batch_scores, batch_idxs):
            if s.ndim == 1 and s.shape[0] == 2:
                scores.append((s[0].item(), idx))
            else:
                scores.append((s.item(), idx))


    top = sorted(scores, key=lambda x: x[0], reverse=True)[:top_k]
    return df.iloc[[idx for (_, idx) in top]]


shoeImages = pd.read_pickle("shoe_clip_embeddings.pkl")

image_path = "small/13/133d2255.jpg"
query = "a similar shoe but not this color"

results = query_blip_with_image_and_text_batch(image_path, query, shoeImages[:10], top_k=6)
show_image_results(results)



