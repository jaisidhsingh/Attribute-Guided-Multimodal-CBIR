import json
from PIL import Image
from tqdm import tqdm
import os
import warnings
import torch
from torchvision.datasets import CocoCaptions
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import cv2
import matplotlib.pyplot as plt
from captum.attr import visualization
from GRADCAMCLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import GRADCAMCLIP.clip as clip
from attention_map import interpret, show_heatmap_on_text, show_image_relevance, color
clip.clip._MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}
warnings.simplefilter('ignore')
parser = argparse.ArgumentParser()
parser.add_argument(
    "--features-path",
    type=str,
    required=True
)
parser.add_argument(
    "--output-dir",
    type=str,
    required=True
)
parser.add_argument(
    "--dataset-path",
    type=str,
    required=True
)
parser.add_argument(
    "--k-value",
    type=int,
    default=3
)
parser.add_argument(
    "--device",
    type=str,
    default='cuda'
)
parser.add_argument(
    "--heatmap-dir",
    type=str,
)
args = parser.parse_args()
device = args.device
_tokenizer = _Tokenizer()
model, preprocess = clip.load("ViT-B/32", device='cuda', jit=False)
start_layer = 11
start_layer_text = 11
text_inputs = input("Enter retrieval query: ")
texts = [text_inputs]
text = torch.cat([clip.tokenize([text_inputs])]).to(device)
text_features = model.encode_text(text)

with open(args.features_path, 'rb') as f:
    features = pickle.load(f)

def get_topk_similarity(text_features, features, k):
    features = torch.from_numpy(features).squeeze(1).to(device)
    features /= features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * text_features @ features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(k)
    return indices

def get_retrieved_images(dataset, indices, output_dir):
    images_retrieved = []
    for idx in indices:
        # image = dataset[idx][0]
        image = os.listdir(dataset)[idx]
        image = Image.open(os.path.join(dataset, image))
        images_retrieved.append(image)

    for idx, image in enumerate(images_retrieved):
        image.save(
            os.path.join(output_dir, f'retrieved_{idx+1}.png')
        )
        img = preprocess(image).unsqueeze(0).to(device)
        R_text, R_image = interpret(model=model, image=img, texts=text, device=device)
        show_heatmap_on_text(_tokenizer, texts[0], text[0], R_text[0])
        show_image_relevance(R_image[0], img, orig_image=image)
        plt.savefig(os.path.join(args.heatmap_dir, f"heatmap_{idx+1}.png"))


indices = get_topk_similarity(text_features, features, args.k_value)
get_retrieved_images(args.dataset_path, indices, args.output_dir)
