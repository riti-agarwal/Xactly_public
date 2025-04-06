import base64
import matplotlib.pyplot as plt
import pandas as pd
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import openai

# Blip model imports
from transformers import BlipProcessor, BlipForImageTextRetrieval
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPImageProcessor
from PIL import Image
import torch

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Load HF CLIP model + processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
clip_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# --------- Embedding Functions ---------
def embed_with_clip(text=None, image_path=None):
    if text:
        inputs = clip_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=77
        )
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.squeeze().cpu().numpy()

    if image_path:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_image_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.squeeze().cpu().numpy()

# --------- Load your data ---------
shoeImages = pd.read_pickle("shoe_clip_embeddings.pkl")

# Step 1: Choose sample image and user query
imagePath = 'small/13/133d2255.jpg'
query = "Find me a similar shoe but in a different color"

# Step 2: Display the reference image
plt.figure(figsize=(2,2))
plt.imshow(plt.imread(imagePath))
plt.axis('off')
plt.show()

# Step 3: Convert image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64Image = encode_image(imagePath)

# Step 4: GPT-4o query for target shoe description
response = client.responses.create(
    model="gpt-4o",
    input=[
        {
            "role": "developer",
            "content": [
                { "type": "input_text", "text": "I want you to create a description of a shoe given a users query and an image relating to the query. Make sure you extract the semantic meaning - negatives should be converted to positives. The query has no referance to the image so add all the image info into the query. State the description of what the user might be searching for. Provide no formatting, just a very short 10 word description of the target shoe." },],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text", 
                    "text": f"This is the users query: {query}. And this is the reference image:",
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64Image}",
                     "detail" : "low"
                },

            ]
        }
    ],
)

# Step 5: Embed GPT response using HuggingFace CLIP
textQuery = response.output_text.strip()
print(f"GPT reprhase: {textQuery}")
queryEmbedding = embed_with_clip(text=textQuery)

index = pc.Index("clip-shoe-index")


# Retrieve top 100 image names from clip-text and clip-image results
results_text_top_100 = index.query(
    vector=queryEmbedding.tolist(),
    top_k=10,
    namespace="clip-text",
    include_metadata=True
)

results_image_top_100 = index.query(
    vector=queryEmbedding.tolist(),
    top_k=10,
    namespace="clip-image",
    include_metadata=True
)

# Extract image names
text_image_names = {match['metadata']['imageId'] for match in results_text_top_100['matches']}
image_image_names = {match['metadata']['imageId'] for match in results_image_top_100['matches']}

# Combine and remove duplicates
unique_image_names = text_image_names.union(image_image_names)

# Map unique image IDs to their full image paths
unique_image_paths = {}
for image_id in unique_image_names:
    matching_image = shoeImages[shoeImages['image_id'] == image_id]
    if not matching_image.empty:
        unique_image_paths[image_id] = matching_image['fullImagePath'].iloc[0]

# Display the full image paths
for image_id, path in unique_image_paths.items():
    print(f"Image ID: {image_id}, Path: {path}")

# Pass full image paths to the blip model - have it choose the best 7. 

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco", use_fast=True)
blip_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
blip_model.eval()

# --- Generate caption from the reference image ---
def generate_caption(image_path):
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model.eval()

    image = Image.open(image_path).convert("RGB")
    inputs = caption_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        out = caption_model.generate(**inputs)
    return caption_processor.decode(out[0], skip_special_tokens=True)

def get_blip_similarity_score(image_path, text_query):
    image = Image.open(image_path).convert('RGB')
    inputs = blip_processor(images=image, text=text_query, return_tensors="pt")
    with torch.no_grad():
        outputs = blip_model(**inputs)
    scores = outputs.itm_score
    if scores.ndim == 2 and scores.shape == (1, 2):
        return scores[0, 0].item()
    elif scores.ndim == 1:
        return scores[0].item()
    elif scores.ndim == 0:
        return scores.item()
    else:
        raise ValueError(f"Unexpected itm_score shape: {scores.shape}")
    

# Construct blip query
image_caption = generate_caption(imagePath)
full_query = f"{query}. {image_caption.strip()}"
print(f"blip query is: {full_query}")

# --- Score each image with BLIP ---
blip_scores = []
for image_id, path in unique_image_paths.items():
    try:
        score = get_blip_similarity_score(path, full_query)
        blip_scores.append((score, image_id, path))
    except Exception as e:
        print(f"Error with image {image_id}: {e}")

# --- Sort by BLIP score and select top 20 ---
blip_scores = sorted(blip_scores, key=lambda x: x[0], reverse=True)[:10]


# TODO: Rank resulting blip images by the historic purchases and trending images. 
# Add images for historic, add images for trends. 
# Compute similarity between each historic/tranding image and the blip output images
# keep a score for each blip image - and add the normalised similarity score to the score of each blip image every time you compute the similarity

# --- Display top 20 matching shoes ---
fig, axes = plt.subplots(1, len(blip_scores), figsize=(4 * len(blip_scores), 5))
if len(blip_scores) == 1:
    axes = [axes]

for idx, (score, image_id, path) in enumerate(blip_scores):
    img = Image.open(path)
    axes[idx].imshow(img)
    axes[idx].set_title(f"Score: {score:.2f}")
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
