import torch
from datasets import load_dataset

from PIL import Image
import open_clip

from tqdm import tqdm

import streamlit as st

def preprocess_images(dataset, model, preprocess, device, cache=True):
    image_items = []
    image_features = []

    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        if isinstance(sample, dict):
            image = sample.get("image") or sample.get("img")
        elif isinstance(sample, (tuple, list)):
            image = sample[0]
        else:
            image = sample
        if image is None:
            continue

        image_items.append(image)
        image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_feature = model.encode_image(image)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
        
        image_features.append(image_feature.cpu())
    
    image_features = torch.cat(image_features, dim=0)

    if cache:
        torch.save(
            {
                "indices": list(range(len(image_items))),
                "image_features": image_features,
            },
            "image_index.pt"
        )

    return image_items, image_features

def text_to_image(text_query, image_features, image_paths, model, tokenizer, device, top_k=5):
    text_tok = tokenizer(text_query).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tok)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    sims = (text_features.cpu() @ image_features.T).squeeze(0)
    values, indices = sims.topk(top_k)

    results = [(image_paths[i], values[j].item()) for j, i in enumerate(indices)]
    return results

def image_to_image(image_query, image_features, image_paths, model, preprocess, device, top_k=5):
    image = preprocess(image_query).unsqueeze(0).to(device)
    with torch.no_grad():
        image_query_features = model.encode_image(image)
        image_query_features /= image_query_features.norm(dim=-1, keepdim=True)
    
    sims = (image_query_features.cpu() @ image_features.cpu().T).squeeze(0)
    values, indices = sims.topk(top_k)

    results = [(image_paths[i], values[j].item()) for j, i in enumerate(indices)]
    return results

@st.cache_resource
def load_index():
    data = torch.load("image_index.pt")
    return data["indices"], data["image_features"]

def main(args):

    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained)
    model = model.to(args.device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model_name)

    dataset = load_dataset("cifar10", split="test[:300]")

    image_paths, image_features = preprocess_images(dataset, model, preprocess, device=args.device, cache=False)

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
    args = parser.parse_args()
    main(args)
