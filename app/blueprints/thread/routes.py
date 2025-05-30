from flask import request, jsonify
from . import thread_bp  # Import the blueprint
import uuid
from collections import defaultdict

# Import your necessary modules
import base64
import torch
from PIL import Image
import io
import matplotlib.pyplot as plt
import pandas as pd
from transformers import CLIPTokenizer, CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForImageTextRetrieval, BlipForConditionalGeneration
import openai
from pinecone import Pinecone
import os

# Load env vars
from dotenv import load_dotenv
load_dotenv()

# --- Setup ---
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("clip-shoe-index")  

# Load models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
clip_image_processor = clip_processor.feature_extractor
clip_model.eval()

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco", use_fast=True)
blip_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
blip_model.eval()

caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model.eval()

shoe_images = pd.read_pickle("shoe_clip_embeddings.pkl")

def image_to_base64(path):
    with open(path, "rb") as f:
        string = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{string}"
    
# Convert image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def get_full_path(image_id):
    image_id = str(image_id).strip()

    # 1. Match exact in mainImageId
    match = shoe_images[shoe_images["mainImageId"].astype(str).str.strip() == image_id]
    if not match.empty:
        return match["fullImagePath"].iloc[0]

    # 2. Fallback: check if fullImagePath contains image_id as substring
    match = shoe_images[shoe_images["fullImagePath"].str.contains(image_id, na=False)]
    if not match.empty:
        return match["fullImagePath"].iloc[0]

    print(f"[WARN] Image ID '{image_id}' not found in mainImageId or fullImagePath")
    return None

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(
        torch.tensor(a).unsqueeze(0), torch.tensor(b).unsqueeze(0)
    ).item()

def embed_with_clip(text=None, image_path=None):
    if text:
        inputs = clip_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=77
        )
        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.squeeze().cpu().numpy()

    if image_path:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.squeeze().cpu().numpy()
    

def get_embeddings_for_ids(id_list):
    embeddings = []
    for i in id_list:
        row = shoe_images[shoe_images["image_id"] == i]
        if not row.empty:
            try:
                emb = row["clipImageEmbedding"].iloc[0]
                embeddings.append(emb)
            except Exception as e:
                print(f"Missing embedding for {i}: {e}")
    return embeddings



# In-memory storage
threads = defaultdict(list)

@thread_bp.route('/create', methods=['POST'])
def create_thread():
    thread_id = str(uuid.uuid4())
    print(f"New thread created: {thread_id}")
    threads[thread_id] = []
    return jsonify({"threadId": thread_id})


# @thread_bp.route('/sendMessage', methods=['POST'])
# def send_message():
#     data = request.json
#     thread_id = data.get("threadId")
#     messages = data.get("messages", [])

#     if not thread_id or thread_id not in threads:
#         return jsonify({"error": "Invalid or missing threadId"}), 400

#     threads[thread_id].extend(messages)
#     full_thread = threads[thread_id]

#     # Get latest user message (with or without image)
#     latest_user_msg = None
#     for msg in reversed(full_thread):
#         if msg["role"] == "USER":
#             latest_user_msg = msg
#             break

#     if not latest_user_msg:
#         return jsonify({"error": "No valid user message found"}), 400

#     user_query = latest_user_msg["text"]
#     image_id = latest_user_msg.get("imageId")
#     image_id = str(image_id).strip()
#     print(f"Incoming image_id: {repr(image_id)} (type={type(image_id)})")
#     image_path = get_full_path(image_id)
#     print(f"image_path is {image_path}")

#     # if image_path:
#     #     base64_image = encode_image(image_path)

#     #     # --- GPT-4o rephrasing ---
#     #     system_instruction = {
#     #         "role": "system",
#     #         "content": f"This is the user’s latest query and a reference image: {user_query}. Use the image to guide your answer. I want you to create a description of a product given a users query and an image relating to the query. Make sure you extract the semantic meaning - negatives should be converted to positives. The query has no referance to the image so add all the image info into the query. State the description of what the user might be searching for. Provide no formatting, just a very short 10 word description of the target shoe."
#     #     }

#     #     gpt_messages = [system_instruction]

#     #     for msg in full_thread[:-1]:
#     #         role = msg["role"].lower()
#     #         if role == "assistant":
#     #             gpt_messages.append({ "role": "assistant", "content": msg["text"] })
#     #         elif role == "user":
#     #             image_id = msg.get("imageId")
#     #             if image_id:
#     #                 full_path = get_full_path(image_id)
#     #                 if full_path:
#     #                     try:
#     #                         image_b64 = encode_image(full_path)
#     #                         gpt_messages.append({
#     #                             "role": "user",
#     #                             "content": [
#     #                                 { "type": "text", "text": msg["text"] },
#     #                                 { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{image_b64}" } }
#     #                             ]
#     #                         })
#     #                     except Exception as e:
#     #                         print(f"Could not load image {image_id} → {full_path}: {e}")
#     #                         gpt_messages.append({ "role": "user", "content": msg["text"] })
#     #                 else:
#     #                     print(f"Image ID {image_id} not found in shoe_images.")
#     #                     gpt_messages.append({ "role": "user", "content": msg["text"] })
#     #             else:
#     #                 gpt_messages.append({ "role": "user", "content": msg["text"] })
        
#     #     gpt_messages.append({
#     #         "role": "user",
#     #         "content": [
#     #             {
#     #                 "type": "text",
#     #                 "text": f"This is the user’s latest query and a reference image: {user_query}. Use the image to guide your answer. I want you to create a description of a product given a users query and an image relating to the query. Make sure you extract the semantic meaning - negatives should be converted to positives. The query has no referance to the image so add all the image info into the query. State the description of what the user might be searching for. Provide no formatting, just a very short 10 word description of the target shoe."
#     #             },
#     #             {
#     #                 "type": "image_url",
#     #                 "image_url": {
#     #                     "url": f"data:image/jpeg;base64,{base64_image}"
#     #                 }
#     #             }
#     #         ]
#     #     })

#     #     response = client.chat.completions.create(
#     #         model="gpt-4o",
#     #         messages=gpt_messages,
#     #         temperature=0.3,
#     #         max_tokens=50,
#     #     )
#     #     text_query = response.choices[0].message.content.strip()
#     # else:
#     #     # No image: use latest user query directly
#     #     text_query = user_query.strip()

#     base64_image = encode_image(image_path) if image_path else None

#     # --- GPT-4o rephrasing ---
#     system_instruction = {
#         "role": "system",
#         "content": "You are a helpful assistant that rewrites vague product queries into concise product descriptions. Use any image provided to infer semantic meaning, and always turn negatives into positives. Provide only a 10-word product description."
#     }

#     gpt_messages = [system_instruction]

#     for msg in full_thread[:-1]:
#         role = msg["role"].lower()
#         if role == "assistant":
#             gpt_messages.append({ "role": "assistant", "content": msg["text"] })
#         elif role == "user":
#             image_id = msg.get("imageId")
#             if image_id:
#                 full_path = get_full_path(image_id)
#                 if full_path:
#                     try:
#                         image_b64 = encode_image(full_path)
#                         gpt_messages.append({
#                             "role": "user",
#                             "content": [
#                                 { "type": "text", "text": msg["text"] },
#                                 { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{image_b64}" } }
#                             ]
#                         })
#                         continue
#                     except Exception as e:
#                         print(f"Could not load image {image_id} → {full_path}: {e}")
#             gpt_messages.append({ "role": "user", "content": msg["text"] })

#     # Append the latest user message
#     if image_path and base64_image:
#         gpt_messages.append({
#             "role": "user",
#             "content": [
#                 { "type": "text", "text": user_query },
#                 { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{base64_image}" } }
#             ]
#         })
#     else:
#         gpt_messages.append({
#             "role": "user",
#             "content": user_query
#         })

#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=gpt_messages,
#         temperature=0.3,
#         max_tokens=50,
#     )
#     text_query = response.choices[0].message.content.strip()

#     # --- Embed query using CLIP ---
#     clip_inputs = clip_tokenizer(text_query, return_tensors="pt", padding=True, truncation=True, max_length=77)
#     with torch.no_grad():
#         text_features = clip_model.get_text_features(**clip_inputs)
#         text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
#     query_embedding = text_features.squeeze().cpu().numpy()

#     if image_path:
#         # --- Dual Pinecone search if image exists ---
#         results_text = index.query(vector=query_embedding.tolist(), top_k=10, namespace="clip-text", include_metadata=True)
#         results_image = index.query(vector=query_embedding.tolist(), top_k=10, namespace="clip-image", include_metadata=True)
#         matches = results_text["matches"] + results_image["matches"]
#     else:
#         # --- Only image search if no image in query ---
#         results_image = index.query(vector=query_embedding.tolist(), top_k=20, namespace="clip-image", include_metadata=True)
#         matches = results_image["matches"]

#     image_ids = {r["metadata"]["imageId"] for r in matches}

#     # Map image IDs to paths
#     id_to_path = {}
#     for image_id in image_ids:
#         match = shoe_images[shoe_images["image_id"] == image_id]
#         if not match.empty:
#             id_to_path[image_id] = match["fullImagePath"].iloc[0]

#     if image_path:
#         # --- Generate image caption ---
#         image = Image.open(image_path).convert("RGB")
#         inputs = caption_processor(images=image, return_tensors="pt")
#         with torch.no_grad():
#             caption_output = caption_model.generate(**inputs)
#         image_caption = caption_processor.decode(caption_output[0], skip_special_tokens=True)

#         full_query = f"{user_query}. {image_caption.strip()}"

#         # --- BLIP Reranking ---
#         scored_images = []
#         for img_id, path in id_to_path.items():
#             try:
#                 image = Image.open(path).convert("RGB")
#                 inputs = blip_processor(images=image, text=full_query, return_tensors="pt")
#                 with torch.no_grad():
#                     score = blip_model(**inputs).itm_score
#                 final_score = score.item() if score.ndim == 0 else score[0, 0].item()
#                 scored_images.append((final_score, img_id, path))
#             except Exception as e:
#                 print(f"Skipping {img_id}: {e}")

#         top_images = sorted(scored_images, key=lambda x: x[0], reverse=True)[:10]
#     else:
#         # --- No reranking needed, just top 20 from Pinecone ---
#         top_images = [(1.0, img_id, path) for img_id, path in id_to_path.items()]

#     response_messages = {"images": [
#         {
#             "imageId": img_id,
#             "imagePath": path,
#             "image": image_to_base64(path),
#             "score": score
#         }
#         for score, img_id, path in top_images
#     ], "text": f"Found {len(top_images)} results for '{text_query}'"}

#     return jsonify(response_messages)


@thread_bp.route('/sendMessage', methods=['POST'])
def send_message():
    data = request.json
    thread_id = data.get("threadId")
    messages = data.get("messages", [])
    historic_ids = data.get("historic_purchaces", [])
    trend_ids = data.get("trends", [])

    if not thread_id or thread_id not in threads:
        return jsonify({"error": "Invalid or missing threadId"}), 400

    threads[thread_id].extend(messages)
    full_thread = threads[thread_id]

    latest_user_msg = next((msg for msg in reversed(full_thread) if msg["role"] == "USER"), None)
    if not latest_user_msg:
        return jsonify({"error": "No valid user message found"}), 400

    user_query = latest_user_msg["text"]
    image_id = str(latest_user_msg.get("imageId", "")).strip()
    image_path = get_full_path(image_id)
    base64_image = encode_image(image_path) if image_path else None

    # --- Prepare messages for GPT ---
    system_instruction = {
        "role": "system",
        "content": "You are a helpful assistant that rewrites vague product queries into concise product descriptions. Use any image provided to infer semantic meaning, and always turn negatives into positives. Provide only a 10-word product description."
    }
    gpt_messages = [system_instruction]

    for msg in full_thread[:-1]:
        role = msg["role"].lower()
        if role == "assistant":
            gpt_messages.append({ "role": "assistant", "content": msg["text"] })
        elif role == "user":
            mid = msg.get("imageId")
            if mid:
                full_path = get_full_path(mid)
                if full_path:
                    try:
                        image_b64 = encode_image(full_path)
                        gpt_messages.append({
                            "role": "user",
                            "content": [
                                { "type": "text", "text": msg["text"] },
                                { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{image_b64}" } }
                            ]
                        })
                        continue
                    except Exception as e:
                        print(f"Could not load image {mid}: {e}")
            gpt_messages.append({ "role": "user", "content": msg["text"] })

    if image_path and base64_image:
        gpt_messages.append({
            "role": "user",
            "content": [
                { "type": "text", "text": user_query },
                { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{base64_image}" } }
            ]
        })
    else:
        gpt_messages.append({ "role": "user", "content": user_query })

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=gpt_messages,
        temperature=0.3,
        max_tokens=50,
    )
    text_query = response.choices[0].message.content.strip()

    # --- Embed query using CLIP ---
    clip_inputs = clip_tokenizer(text_query, return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**clip_inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    query_embedding = text_features.squeeze().cpu().numpy()

    # --- Pinecone query ---
    results_text = index.query(vector=query_embedding.tolist(), top_k=10, namespace="clip-text", include_metadata=True)
    results_image = index.query(vector=query_embedding.tolist(), top_k=10, namespace="clip-image", include_metadata=True)
    matches = results_text["matches"] + results_image["matches"]

    image_ids = {r["metadata"]["imageId"] for r in matches}
    id_to_path = {}
    for img_id in image_ids:
        match = shoe_images[shoe_images["image_id"] == img_id]
        if not match.empty:
            id_to_path[img_id] = match["fullImagePath"].iloc[0]

    # --- Embed Pinecone results ---
    clip_emb_map = {}
    for img_id, path in id_to_path.items():
        try:
            clip_emb_map[img_id] = embed_with_clip(image_path=path)
        except Exception as e:
            print(f"Could not embed image {img_id}: {e}")

    # --- Embed historic and trend image IDs ---
    historic_embs = get_embeddings_for_ids(historic_ids)
    trend_embs = get_embeddings_for_ids(trend_ids)

    # --- Optional BLIP query ---
    blip_score_map = {}
    full_query = user_query
    if image_path:
        image = Image.open(image_path).convert("RGB")
        inputs = caption_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            caption_output = caption_model.generate(**inputs)
        image_caption = caption_processor.decode(caption_output[0], skip_special_tokens=True)
        full_query = f"{user_query}. {image_caption.strip()}"

        for img_id, path in id_to_path.items():
            try:
                img = Image.open(path).convert("RGB")
                inputs = blip_processor(images=img, text=full_query, return_tensors="pt")
                with torch.no_grad():
                    score = blip_model(**inputs).itm_score
                blip_score_map[img_id] = score.item() if score.ndim == 0 else score[0, 0].item()
            except Exception as e:
                print(f"BLIP failed for {img_id}: {e}")
                blip_score_map[img_id] = 0.0

    # --- Final scoring ---
    final_scores = []
    for img_id, path in id_to_path.items():
        clip_emb = clip_emb_map.get(img_id)
        if clip_emb is None:
            continue

        sim_scores = []
        for emb_list in [historic_embs, trend_embs]:
            for emb in emb_list:
                sim_scores.append(cosine_similarity(clip_emb, emb))

        norm_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0
        blip_score = blip_score_map.get(img_id, 0.0)
        final_score = norm_sim + blip_score

        final_scores.append((final_score, img_id, path))

    top_images = sorted(final_scores, key=lambda x: x[0], reverse=True)[:10]

    response_messages = {"images": [
        {
            "imageId": img_id,
            "imagePath": path,
            "image": image_to_base64(path),
            "score": score
        }
        for score, img_id, path in top_images
    ], "text": f"Found {len(top_images)} results for '{text_query}'"}

    return jsonify(response_messages)



@thread_bp.route('/randomImages', methods=['GET'])
def get_random_images():
    try:
        # Sample 20 random rows
        sampled = shoe_images.sample(n=20)

        images_data = []
        for _, row in sampled.iterrows():
            img_id = row["image_id"]
            path = row["fullImagePath"]
            try:
                image_data = image_to_base64(path)
                images_data.append({
                    "imageId": img_id,
                    "imagePath": path,
                    "image": image_data
                })
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                continue

        return jsonify({"images": images_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# @thread_bp.route('/')
# def index():
#     return jsonify("Hello World")

