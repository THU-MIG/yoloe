


import numpy as np
from ultralytics.utils import yaml_load
from ultralytics.utils.torch_utils import smart_inference_mode
import torch
from tqdm import tqdm
import os
import json
from ultralytics.nn.text_model import build_text_model

@smart_inference_mode()
def generate_label_embedding(model, texts, batch=512):
    model = build_text_model(model, device='cuda')
    assert(not model.training)
    
    text_tokens = model.tokenize(texts)
    txt_feats = []
    for text_token in tqdm(text_tokens.split(batch)):
        txt_feats.append(model.encode_text(text_token))
    txt_feats = torch.cat(txt_feats, dim=0)
    return txt_feats.cpu()


def collect_grounding_labels(cache_path):
    labels = np.load(cache_path, allow_pickle=True)
    cat_names = set()
    
    for label in labels:
        for text in label["texts"]:
            for t in text:
                t = t.strip()
                assert(t)
                cat_names.add(t)
    
    return cat_names

def collect_grounding_labels_from_json(json_path):
    with open(json_path) as f:
        annotations = json.load(f)
    
    # 处理混合类型的ID：将所有ID转换为字符串作为键
    images = {str(x["id"]): x for x in annotations["images"]}
    img_to_anns = {str(k): [] for k in images.keys()}

    for ann in annotations["annotations"]:
        img_id_str = str(ann["image_id"])
        if img_id_str in img_to_anns:
            img_to_anns[img_id_str].append(ann)
    
    cat_names = set()
    for img_id_str, anns in tqdm(img_to_anns.items(), desc=f"Processing {json_path}"):
        img = images[img_id_str]
        for ann in anns:
            if ann["iscrowd"]:
                continue
            
            # 处理不同数据集格式
            cat_name = ""
            if "tokens_positive" in ann and ann["tokens_positive"]:
                # Grounding数据集格式 (flickr, mixed_grounding等)
                cat_name = " ".join([img["caption"][t[0] : t[1]] for t in ann["tokens_positive"]]).lower().strip()
            elif "category_name" in ann:
                # 直接包含category_name字段的数据集
                cat_name = ann["category_name"].lower().strip()
            elif "category_id" in ann and "categories" in annotations:
                # 通过category_id查找类别名称
                cat_id = ann["category_id"]
                for cat in annotations["categories"]:
                    if cat["id"] == cat_id:
                        cat_name = cat["name"].lower().strip()
                        break
            
            if not cat_name:
                continue
            
            cat_names.add(cat_name)
    
    return cat_names

def collect_detection_labels(yaml_path):
    cat_names = set()
    
    data = yaml_load(yaml_path, append_filename=True)
    names = [name.split("/") for name in data["names"].values()]
    for name in names:
        for n in name:
            n = n.strip()
            assert(n)
            cat_names.add(n)
    
    return cat_names

if __name__ == '__main__':
    os.environ["PYTHONHASHSEED"] = "0"
    
    flickr_cache = '../datasets/yoloe_annotations/final_flickr_separateGT_train_segm.json'
    mixed_grounding_cache = '../datasets/yoloe_annotations/final_mixed_train_no_coco_segm.json'
    objects365v1_yaml = 'ultralytics/cfg/datasets/Objects365v1.yaml'
    custom_cache = '../datasets/VisDrone/annotations/train_annotations_segm.json'
    
    all_cat_names = set()
    all_cat_names |= collect_detection_labels(objects365v1_yaml)
    all_cat_names |= collect_grounding_labels_from_json(flickr_cache)
    all_cat_names |= collect_grounding_labels_from_json(mixed_grounding_cache)
    all_cat_names |= collect_grounding_labels_from_json(custom_cache)
    
    all_cat_names = list(all_cat_names)
    
    model = yaml_load('ultralytics/cfg/default.yaml')['text_model']
    all_cat_feats = generate_label_embedding(model, all_cat_names)
    
    cat_name_feat_map = {}
    for name, feat in zip(all_cat_names, all_cat_feats):
        cat_name_feat_map[name] = feat
    
    os.makedirs(f'tools/{model}', exist_ok=True)
    torch.save(cat_name_feat_map, f'tools/{model}/train_label_embeddings.pt')
