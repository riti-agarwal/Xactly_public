import base64
import matplotlib.pyplot as plt
import pandas as pd
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import openai

from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPImageProcessor
from PIL import Image
import torch


# TODO: Add a flow where we first just give the user something that matches their desired text query
# TODO: after that - they can select an image and ask for something more - which is where we will pass in the text and the user chat history
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
queryEmbedding = embed_with_clip(text=textQuery)

# --------- Query 1: Text Embedding Index ---------
index = pc.Index("clip-shoe-index")

results_text = index.query(
    vector=queryEmbedding.tolist(),
    top_k=5,
    namespace="clip-text",
    include_metadata=True
)

# Display text-based results
print(f"\nText-based query: {textQuery}")
fig, axes = plt.subplots(1, len(results_text['matches']), figsize=(15, 3))
for idx, match in enumerate(results_text['matches']):
    matchingImage = shoeImages[shoeImages['image_id'] == match['metadata']['imageId']]
    if not matchingImage.empty:
        img = plt.imread(matchingImage['fullImagePath'].iloc[0])
        axes[idx].imshow(img)
        axes[idx].set_title(f"Score: {match['score']:.3f}")
        axes[idx].axis('off')
plt.tight_layout()
plt.show()

# --------- Query 2: Image Embedding Index ---------
results_image = index.query(
    vector=queryEmbedding.tolist(),
    top_k=5,
    namespace="clip-image",
    include_metadata=True
)

# Display image-based results
print(f"\nImage-based query: {textQuery}")
fig, axes = plt.subplots(1, len(results_image['matches']), figsize=(15, 3))
for idx, match in enumerate(results_image['matches']):
    matchingImage = shoeImages[shoeImages['image_id'] == match['metadata']['imageId']]
    if not matchingImage.empty:
        img = plt.imread(matchingImage['fullImagePath'].iloc[0])
        axes[idx].imshow(img)
        axes[idx].set_title(f"Score: {match['score']:.3f}")
        axes[idx].axis('off')
plt.tight_layout()
plt.show()