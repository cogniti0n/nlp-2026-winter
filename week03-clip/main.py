import torch

from PIL import Image
import open_clip

import os
from tqdm import tqdm

import streamlit as st

def preprocess_images(image_dir, model, preprocess, device, cache=True):
    image_paths = []
    image_features = []

    for fname in tqdm(os.listdir(image_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        path = os.path.join(image_dir, fname)
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            image_feature = model.encode_images(image)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
        
        image_paths.append(path)
        image_features.append(image_feature.cpu())
    
    image_features = torch.cat(image_features, dim=0)

    if cache:
        torch.save(
            {
                "paths": image_paths,
                "image_features": image_features,
            },
            "image_index.pt"
        )

    return image_paths, image_features

def text_to_image(text_query, image_features, image_paths, model, tokenizer, device, top_k=5):
    text_tok = tokenizer(text_query)
    with torch.no_grad():
        text_features = model.encode_text(text_tok)
        text_features /= text_features.norm(dim=-1, keepdims=True)
    
    sims = (text_features.cpu() @ image_features.T).squeeze(0)
    values, indices = sims.topk(top_k)

    results = [(image_paths[i], values[j].item()) for j, i in enumerate(indices)]
    return results

def image_to_image(image_query, image_features, image_paths, model, preprocess, device, top_k=5):
    image = preprocess(image_query).unsqueeze(0).to(device)
    with torch.no_grad():
        image_query_features = model.encode_image(image)
        image_query_features /= image_query_features.norm(dim=-1, keepdims=True)
    
    sims = (image_query_features.cpu() @ image_features.cpu().T).squeeze(0)
    values, indices = sims.topk(top_k)

    results = [(image_paths[i], values[j].item()) for j, i in enumerate(indices)]
    return results

@st.cache_resource
def load_index():
    data = torch.load("image_index.pt")
    return data["paths"], data["image_features"]

def main(args):

    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained)
    model = model.to(args.device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model_name)

    image_paths, image_features = preprocess_images(args.image_dir, model, preprocess, device=args.device, cache=False)

    st.title("CLIP Semantic Search")

    mode = st.radio("Search mode: ", ["Text to Image", "Image to Image"])

    if mode == "Text to Image":
        query = st.text_input("Enter text query")
        if query:
            results = text_to_image(query, image_features, image_paths, model, tokenizer, args.device)
            for path, score in results:
                st.image(path, caption=f"Score: {score:.3f}", width=250)
    else:
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if uploaded:
            image_query = Image.open(uploaded).convert("RGB")
            st.image(image_query, caption="Query Image", width=250)
            results = image_to_image(image_query, image_features, image_paths, model, preprocess, args.device)
            for path, score in results:
                st.image(path, caption=f"Score: {score:.3f}", width=250)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--device", type=str, default="cuda")