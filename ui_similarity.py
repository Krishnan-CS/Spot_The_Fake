import torch
from torch.nn.functional import cosine_similarity
from PIL import Image
import os
import numpy as np

# Load CLIP once
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ----- CLIP Embedding -----
def get_clip_embedding(image: Image.Image, clip_model, clip_processor):
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)  # normalize
    return emb.squeeze(0).cpu()

webpage_folder = "UI_Similarity/Webpages"
webpage_embeddings = {}

def create_webpage_embeddings(clip_model, clip_processor):
    for file in os.listdir(webpage_folder):
        path = os.path.join(webpage_folder, file)
        label = file.split("-")[0]   # All files have been named: "brand_name-xxx.png"
        image = Image.open(path)
        emb = get_clip_embedding(image, clip_model, clip_processor)
        webpage_embeddings[label] = emb


def check_UI_Similarity(file, clip_model, clip_processor):
    # Read and preprocess
    image = Image.open(file).convert("RGB")
    img_emb = get_clip_embedding(image, clip_model, clip_processor)

    create_webpage_embeddings(clip_model, clip_processor)

    # Compare with references
    sims = {
        name: torch.cosine_similarity(img_emb.unsqueeze(0), ref_emb.unsqueeze(0)).item()
        for name, ref_emb in webpage_embeddings.items()
    }

    # out_dict = {}
    best_match = max(sims, key=sims.get)
    # out_dict[file.split('/')[-1]] = best_match

    # Sort sims in descending order
    # sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    
    # Pick top 5 matches
    # top_5_matches = sorted_sims[:5]
    # for name, score in top_5_matches:
        # out_dict[name] = score

    return best_match if sims[best_match] > 0.7 else ''