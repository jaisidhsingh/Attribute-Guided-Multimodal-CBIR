import os
import json
import torch
import clip
import imagecorruptions
from PIL import Image
from torchvision.datasets import CocoCaptions, VOCDetection
import numpy as np
from imagecorruptions import corrupt, get_corruption_names
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    required=True
)
parser.add_argument(
    "--dir2save",
    type=str,
    required=True
)
parser.add_argument(
    "--limit",
    type=int,
    default=1000
)
args = parser.parse_args()

root_path = "/media/mediadrive2/intern/emno/code/clip-experiments/data/coco/train2014/"
annFile_path = "/media/mediadrive2/intern/emno/code/clip-experiments/data/coco/annotations/captions_train2014.json"
corruption_names = ['snow', 'fog', 'defocus_blur', 'gaussian_noise']
coco = CocoCaptions(root=root_path, annFile=annFile_path)

pascal_root = "/media/mediadrive2/intern/emno/code/clip-experiments/data/pascalvoc/"
pascalvoc = VOCDetection(root=pascal_root, year="2012", download=True)
dataset_dict = {"coco": coco, "pascalvoc": pascalvoc}

def make_corrupted_dataset(dataset_name, limit, corruption_names, dir2save):
    loop = tqdm(range(limit))
    dataset = dataset_dict[dataset_name]
    for i in loop:
        image = dataset[i][0]
        image.save(
            os.path.join(dir2save, f'{dataset_name}_{i}_none.jpg')
        )
        image = np.array(image)

        for corruption_name in corruption_names:
            corrupted_image = corrupt(
                image, 
                corruption_name=corruption_name, 
                severity=1
            )
            corrupted_image = Image.fromarray(corrupted_image)
            corrupted_image.save(
                os.path.join(dir2save, f'{dataset_name}_{i}_{corruption_name}.jpg')
            )
        loop.set_postfix({'image_index': i})

make_corrupted_dataset(args.dataset, 1000, corruption_names, args.dir2save)
