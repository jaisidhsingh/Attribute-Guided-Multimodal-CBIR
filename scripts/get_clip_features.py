from tqdm import tqdm
import numpy as np
import torch
import os
import clip
import argparse
from PIL import Image
import pickle


parser = argparse.ArgumentParser()

parser.add_argument(
    "--input-dir",
    type=str,
    required=True
)

parser.add_argument(
    "--output-name",
    type=str,
    required=True
)

parser.add_argument(
    "--output-dir",
    type=str,
    required=True
)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

features = []
for image_name in tqdm(os.listdir(args.input_dir)):
    image_path = os.path.join(args.input_dir, image_name)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    image_features = model.encode_image(image).detach().cpu().numpy()
    features.append(image_features)

features = np.concatenate([np.array(features)], axis=0)
db_pickle_file = args.output_name + "_" + "features.pickle"
db_pickle_file = os.path.join(args.output_dir, db_pickle_file)
with open(db_pickle_file, "wb") as f:
    pickle.dump(features, f)


