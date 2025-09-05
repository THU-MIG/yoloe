# 官方提供的负样本挖掘脚本是针对.cache文件，而官方并没有提供这一后缀的文件，而生成.cache文件需要用到原始的数据集，对小存储空间的设备不友好。
# 这个文件的功能是直接从json文件中提取类别名称，并统计频次，生成负样本类别列表和对应的嵌入文件。


import numpy as np
from pathlib import Path
from collections import defaultdict
import os
import json
from tqdm import tqdm
from generate_label_embedding import generate_label_embedding
import torch
from ultralytics.utils import yaml_load

def obtain_cat_freq(cache_path, cat_name_freq):
    labels = np.load(cache_path, allow_pickle=True)
    
    for label in labels:
        for text in label["texts"]:
            for t in text:
                t = t.strip()
                assert(t)
                cat_name_freq[t] += 1

def obtain_cat_freq_from_json(json_path, cat_name_freq):
    with open(json_path) as f:
        annotations = json.load(f)
    
    images = {str(x["id"]): x for x in annotations["images"]}
    img_to_anns = {str(k): [] for k in images.keys()}

    for ann in annotations["annotations"]:
        img_id_str = str(ann["image_id"])
        if img_id_str in img_to_anns:
            img_to_anns[img_id_str].append(ann)
        
    for img_id_str, anns in tqdm(img_to_anns.items(), desc=f"Processing {json_path.name}"):
        img = images[img_id_str]
        for ann in anns:
            if ann["iscrowd"]:
                continue
            
            cat_name = ""
            if "tokens_positive" in ann and ann["tokens_positive"]:
                cat_name = " ".join([img["caption"][t[0] : t[1]] for t in ann["tokens_positive"]]).lower().strip()
            elif "category_name" in ann:
                cat_name = ann["category_name"].lower().strip()
            elif "category_id" in ann and "categories" in annotations:
                cat_id = ann["category_id"]
                for cat in annotations["categories"]:
                    if cat["id"] == cat_id:
                        cat_name = cat["name"].lower().strip()
                        break
            
            if not cat_name:
                continue
            
            cat_name_freq[cat_name] += 1

if __name__ == '__main__':
    os.environ["PYTHONHASHSEED"] = "0"
    cat_name_freq = defaultdict(int)
    
    flickr_cache_path = Path('../datasets/yoloe_annotations/final_flickr_separateGT_train_segm.json')
    obtain_cat_freq_from_json(flickr_cache_path, cat_name_freq)

    mixed_grounding_cache_path = Path('../datasets/yoloe_annotations/final_mixed_train_no_coco_segm.json')
    obtain_cat_freq_from_json(mixed_grounding_cache_path, cat_name_freq)

    global_neg_cat = []
    for k, v in cat_name_freq.items():
        if v >= 100:
            global_neg_cat.append(k)

    print(len(global_neg_cat))

    with open('tools/global_grounding_neg_cat.json', 'w') as f:
        json.dump(global_neg_cat, f, indent=2)
    
    model = yaml_load('ultralytics/cfg/default.yaml')['text_model']
    global_neg_embeddings = generate_label_embedding(model, global_neg_cat)
    os.makedirs(f'tools/{model}', exist_ok=True)
    torch.save(global_neg_embeddings, f'tools/{model}/global_grounding_neg_embeddings.pt')
        
