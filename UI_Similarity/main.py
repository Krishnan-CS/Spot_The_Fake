# Run: uvicorn main:app --reload (Optional: --host 0.0.0.0 --port 8000)
# Open http://127.0.0.1:8000/docs in browser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import numpy as np
import os
from transformers import CLIPProcessor, CLIPModel
import math

def clean_float(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return 0.0  # or None, depending on what makes sense
    return float(x)

# Load CLIP once
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

app = FastAPI()

# ----- Model -----
class SiameseNetwork(nn.Module):
    def __init__(self, base_model):
        super(SiameseNetwork, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(1000, 256)  # compress ResNet output to 256-dim embedding

    def forward_once(self, x):
        output = self.base_model(x)
        output = self.fc(output)
        return F.normalize(output, p=2, dim=1)

    def forward(self, input1, input2=None):
        emb1 = self.forward_once(input1)
        if input2 is not None:
            emb2 = self.forward_once(input2)
            return emb1, emb2
        return emb1

# Load ResNet backbone
resnet18 = models.resnet18(pretrained=True)
model = SiameseNetwork(resnet18)
model.eval()

# ----- Image Preprocessing -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file)).convert("RGB")
    return image

def get_embedding(image):
    image = image.convert("RGB")
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        emb = model(img_t)
		
	# Normalize to unit length (L2 norm)
    emb = emb / emb.norm(p=2, dim=1, keepdim=True)
    return emb.squeeze(0).numpy()

# ----- CLIP Embedding -----
def get_clip_embedding(image: Image.Image):
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)  # normalize
    return emb.squeeze(0).cpu()


'''
Stretch goal: Check if a logo appears in a webpage
'''
# Folder with your logos
logo_folder = "Logos/"
known_logos = {}

for file in os.listdir(logo_folder):
    path = os.path.join(logo_folder, file)
    label = os.path.splitext(file)[0]   # e.g. nike.jpg → "nike"
    image = Image.open(path)
    emb = get_embedding(image)
    known_logos[label] = emb

def safe_cosine_similarity(embedding, logo_emb):
    # Convert to tensors if they are numpy arrays
    if not isinstance(embedding, torch.Tensor):
        embedding = torch.from_numpy(embedding).float()
    if not isinstance(logo_emb, torch.Tensor):
        logo_emb = torch.from_numpy(logo_emb).float()

    # Avoid division by zero
    if torch.norm(embedding) == 0 or torch.norm(logo_emb) == 0:
        return 0.0

    sim = cosine_similarity(
        embedding.unsqueeze(0),
        logo_emb.unsqueeze(0)
    ).item()

    # Replace invalid values with 0.0
    if math.isnan(sim) or math.isinf(sim):
        return 0.0

    return sim

def find_best_match(embedding, threshold=0.8):
    best_match = None
    best_score = -1.0
    for name, logo_emb in known_logos.items():
        score = safe_cosine_similarity(embedding, logo_emb)

        if score > best_score:
            best_score = score
            best_match = name

    # Prepare clean output for FastAPI
    result = {
        "best_match": best_match,
        "similarity": float(best_score),  # guaranteed finite
    }

    if best_score >= threshold:
        result["message"] = (
            "Logos are very similar. Analyze website metadata for authenticity."
        )
    else:
        result["message"] = "Logos are not highly similar."

    return result

# ----- API Endpoints -----
@app.post("/predict")
async def predict_logo(file: UploadFile = File(...)):
    image = read_imagefile(await file.read())
    embedding = get_embedding(image)
    result = find_best_match(embedding)
    return result


# --------Webpage Embeddings--------

webpage_folder = "Webpages"
webpage_embeddings = {}

for file in os.listdir(webpage_folder):
    path = os.path.join(webpage_folder, file)
    label = file.split("-")[0]   # All files have been named: "brand_name-xxx.png"
    image = Image.open(path)
    emb = get_clip_embedding(image)
    webpage_embeddings[label] = emb


@app.post("/ui-similarity")
async def ui_similarity(file: UploadFile = File(...)):
    # Read and preprocess
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_emb = get_clip_embedding(image)

    # Compare with references
    sims = {
        name: torch.cosine_similarity(img_emb.unsqueeze(0), ref_emb.unsqueeze(0)).item()
        for name, ref_emb in webpage_embeddings.items()
    }

    # Sort sims in descending order
    sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    out_dict = {}

    # Pick top 5 matches
    top_5_matches = sorted_sims[:5]
    for name, score in top_5_matches:
        out_dict[name] = score

    # best_match = max(sims, key=sims.get)
    # out_dict = {"best_match": best_match, "similarity_score": sims[best_match]}

    return out_dict
