#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
from PIL import Image
import requests
from io import BytesIO
from datasets import load_dataset
import json
import numpy as np
import torch

from training.dr.transforms import compose_from_config


if __name__ == '__main__':
    rconfig_aug = {
        "normalize": {
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711]
        },
        "rand_augment": {"enable": True, "p": 1.0},
        "random_resized_crop": {"interpolation": "bicubic", "size": 224},
        "to_rgb": {"enable": True},
        "to_tensor": {"enable": True}
    }
    dr_transforms = compose_from_config(rconfig_aug)

    dataset = load_dataset("apple/DataCompDR-12M", split="train", streaming=True)
    sample = next(iter(dataset))

    # Load image from URL
    url = sample['url.txt']
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    sample["image"] = img

    # Preprocess image
    # Sample an image augmentation
    param_augs = json.loads(sample["paug.json"]["param_aug"])
    aug_idx = np.random.randint(0, len(param_augs))
    params = param_augs[aug_idx]
    params = dr_transforms.decompress(params)
    image = sample["image"].convert('RGB')
    image, _ = dr_transforms.reapply(image, params)

    # Preprocess synthetic text
    scapi = np.random.randint(0, len(sample["syn.json"]["syn_text"]))
    syn_text = sample["syn.json"]["syn_text"][scapi]

    # Preprocess embeddings
    if "npz" in sample:
        image_emb = sample["npz"]["image_emb"][aug_idx]
        text_emb_all = sample["npz"]["text_emb"]
    elif "pth.gz" in sample:
        image_emb = sample["pth.gz"]["image_emb"][aug_idx]
        text_emb_all = sample["pth.gz"]["text_emb"]
    capi = 0
    text_emb = text_emb_all[capi]
    syn_text_emb = text_emb_all[1+scapi]
    if not isinstance(image_emb, torch.Tensor):
        image_emb = torch.tensor(image_emb)
        text_emb = torch.tensor(text_emb)
        syn_text_emb = torch.tensor(syn_text_emb)
    image_emb = image_emb.type(torch.float32)
    text_emb = text_emb.type(torch.float32)
    syn_text_emb = syn_text_emb.type(torch.float32)

    print(
        {
            'image': image.shape,
            'image_emb': image_emb.shape,
            'text_emb': text_emb.shape,
            "syn_text": syn_text,
            'syn_text_emb': syn_text_emb.shape,
        }
    )
