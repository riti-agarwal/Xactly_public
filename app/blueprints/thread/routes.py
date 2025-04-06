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
#     # TODO: Make this work with if the last message does not have an image - what do we do. 
#     data = request.json
#     thread_id = data.get("threadId")
#     messages = data.get("messages", [])

#     if not thread_id or thread_id not in threads:
#         return jsonify({"error": "Invalid or missing threadId"}), 400

#     # Append new messages to thread history
#     threads[thread_id].extend(messages)
#     full_thread = threads[thread_id]

#     # Get latest USER message with image
#     latest_user_msg = None
#     for msg in reversed(full_thread):
#         if msg["role"] == "USER" and msg.get("imageId"):
#             latest_user_msg = msg
#             break

#     if not latest_user_msg:
#         return jsonify({"error": "No valid user message with image found"}), 400

#     user_query = latest_user_msg["text"]
#     image_path = latest_user_msg["imageId"]

#     base64_image = encode_image(image_path)

#     # --- GPT-4o Rephrasing using full history ---
#     system_instruction = {
#         "role": "system",
#         "content": "You are a shopping assistant helping rephrase user queries. Use all previous conversation context but emphasize the latest user message and the image provided. I want you to create a description of a product given a users query and an image relating to the query. Make sure you extract the semantic meaning - negatives should be converted to positives. The query has no referance to the image so add all the image info into the query. State the description of what the user might be searching for. Provide no formatting, just a very short 10 word description of the target shoe."
#     }

#     gpt_messages = [system_instruction]

#     # Add all previous messages (text-only)
#     # TODO: Map image path to image ID, and then pass it as part of the content history
#     # TODO: loop through everything expecpt the last message in full_thread
#     for msg in full_thread:
#         if msg["role"] == "USER":
#             gpt_messages.append({
#                 "role": "user",
#                 "content": msg["text"]
#             })
#         elif msg["role"] == "ASSISTANT":
#             gpt_messages.append({
#                 "role": "assistant",
#                 "content": msg["text"]
#             })

#     # Add the image + emphasized final query
#     gpt_messages.append({
#         "role": "user",
#         "content": [
#             {
#                 "type": "text",
#                 "text": f"This is the user’s latest query and a reference image: {user_query}. Use the image to guide your answer. I want you to create a description of a product given a users query and an image relating to the query. Make sure you extract the semantic meaning - negatives should be converted to positives. The query has no referance to the image so add all the image info into the query. State the description of what the user might be searching for. Provide no formatting, just a very short 10 word description of the target shoe."
#             },
#             {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": f"data:image/jpeg;base64,{base64_image}"
#                 }
#             }
#         ]
#     })

#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=gpt_messages,
#         temperature=0.3,
#         max_tokens=50,
#     )

#     text_query = response.choices[0].message.content.strip()

#     # --- Embed query with CLIP ---
#     clip_inputs = clip_tokenizer(text_query, return_tensors="pt", padding=True, truncation=True, max_length=77)
#     with torch.no_grad():
#         text_features = clip_model.get_text_features(**clip_inputs)
#         text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
#     query_embedding = text_features.squeeze().cpu().numpy()

#     # --- Pinecone search ---
#     results_text = index.query(vector=query_embedding.tolist(), top_k=10, namespace="clip-text", include_metadata=True)
#     results_image = index.query(vector=query_embedding.tolist(), top_k=10, namespace="clip-image", include_metadata=True)

#     image_ids = {r["metadata"]["imageId"] for r in results_text["matches"] + results_image["matches"]}

#     # Map image IDs to full paths
#     id_to_path = {}
#     for image_id in image_ids:
#         match = shoe_images[shoe_images["image_id"] == image_id]
#         if not match.empty:
#             id_to_path[image_id] = match["fullImagePath"].iloc[0]

#     # --- Generate caption from reference image ---
#     image = Image.open(image_path).convert("RGB")
#     inputs = caption_processor(images=image, return_tensors="pt")
#     with torch.no_grad():
#         caption_output = caption_model.generate(**inputs)
#     image_caption = caption_processor.decode(caption_output[0], skip_special_tokens=True)

#     # --- Construct BLIP query ---
#     full_query = f"{user_query}. {image_caption.strip()}"

#     # --- BLIP reranking ---
#     scored_images = []
#     for img_id, path in id_to_path.items():
#         try:
#             image = Image.open(path).convert("RGB")
#             inputs = blip_processor(images=image, text=full_query, return_tensors="pt")
#             with torch.no_grad():
#                 score = blip_model(**inputs).itm_score
#             final_score = score.item() if score.ndim == 0 else score[0, 0].item()
#             scored_images.append((final_score, img_id, path))
#         except Exception as e:
#             print(f"Skipping {img_id}: {e}")

#     top_images = sorted(scored_images, key=lambda x: x[0], reverse=True)[:5]

#     # --- Prepare response ---
#     response_messages = {"images": [
#         {
#             "imageId": img_id,
#             "imagePath": path,
#             "image": image_to_base64(path),
#             "score": score
#         }
#         for score, img_id, path in top_images
#     ], "text": f"Found {len(top_images)} results for {text_query}"}

#     return jsonify(response_messages)

@thread_bp.route('/sendMessage', methods=['POST'])
def send_message():
    data = request.json
    thread_id = data.get("threadId")
    messages = data.get("messages", [])

    if not thread_id or thread_id not in threads:
        return jsonify({"error": "Invalid or missing threadId"}), 400

    threads[thread_id].extend(messages)
    full_thread = threads[thread_id]

    # Get latest user message (with or without image)
    latest_user_msg = None
    for msg in reversed(full_thread):
        if msg["role"] == "USER":
            latest_user_msg = msg
            break

    if not latest_user_msg:
        return jsonify({"error": "No valid user message found"}), 400

    user_query = latest_user_msg["text"]
    image_path = latest_user_msg.get("imageId")

    if image_path:
        base64_image = encode_image(image_path)

        # --- GPT-4o rephrasing ---
        system_instruction = {
            "role": "system",
            "content": f"This is the user’s latest query and a reference image: {user_query}. Use the image to guide your answer. I want you to create a description of a product given a users query and an image relating to the query. Make sure you extract the semantic meaning - negatives should be converted to positives. The query has no referance to the image so add all the image info into the query. State the description of what the user might be searching for. Provide no formatting, just a very short 10 word description of the target shoe."
        }

        gpt_messages = [system_instruction]
        for msg in full_thread:
            if msg["role"] == "USER":
                gpt_messages.append({"role": "user", "content": msg["text"]})
            elif msg["role"] == "ASSISTANT":
                gpt_messages.append({"role": "assistant", "content": msg["text"]})
        
        gpt_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"This is the user’s latest query and a reference image: {user_query}..."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        })

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=gpt_messages,
            temperature=0.3,
            max_tokens=50,
        )
        text_query = response.choices[0].message.content.strip()
    else:
        # No image: use latest user query directly
        text_query = user_query.strip()

    # --- Embed query using CLIP ---
    clip_inputs = clip_tokenizer(text_query, return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**clip_inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    query_embedding = text_features.squeeze().cpu().numpy()

    if image_path:
        # --- Dual Pinecone search if image exists ---
        results_text = index.query(vector=query_embedding.tolist(), top_k=10, namespace="clip-text", include_metadata=True)
        results_image = index.query(vector=query_embedding.tolist(), top_k=10, namespace="clip-image", include_metadata=True)
        matches = results_text["matches"] + results_image["matches"]
    else:
        # --- Only image search if no image in query ---
        results_image = index.query(vector=query_embedding.tolist(), top_k=20, namespace="clip-image", include_metadata=True)
        matches = results_image["matches"]

    image_ids = {r["metadata"]["imageId"] for r in matches}

    # Map image IDs to paths
    id_to_path = {}
    for image_id in image_ids:
        match = shoe_images[shoe_images["image_id"] == image_id]
        if not match.empty:
            id_to_path[image_id] = match["fullImagePath"].iloc[0]

    if image_path:
        # --- Generate image caption ---
        image = Image.open(image_path).convert("RGB")
        inputs = caption_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            caption_output = caption_model.generate(**inputs)
        image_caption = caption_processor.decode(caption_output[0], skip_special_tokens=True)

        full_query = f"{user_query}. {image_caption.strip()}"

        # --- BLIP Reranking ---
        scored_images = []
        for img_id, path in id_to_path.items():
            try:
                image = Image.open(path).convert("RGB")
                inputs = blip_processor(images=image, text=full_query, return_tensors="pt")
                with torch.no_grad():
                    score = blip_model(**inputs).itm_score
                final_score = score.item() if score.ndim == 0 else score[0, 0].item()
                scored_images.append((final_score, img_id, path))
            except Exception as e:
                print(f"Skipping {img_id}: {e}")

        top_images = sorted(scored_images, key=lambda x: x[0], reverse=True)[:5]
    else:
        # --- No reranking needed, just top 20 from Pinecone ---
        top_images = [(1.0, img_id, path) for img_id, path in id_to_path.items()]

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

