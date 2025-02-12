from ultralytics import YOLOWorld
from ultralytics.models.yolo.world.train_seg_vp import WorldSegVPTrainer
import os
from ultralytics.nn.tasks import guess_model_scale
from ultralytics.utils import yaml_load, LOGGER

os.environ["PYTHONHASHSEED"] = "0"

data = dict(
    train=dict(
        yolo_data=["Objects365v1.yaml"],
        grounding_data=[
            dict(
                img_path="../datasets/flickr/full_images/",
                json_file="../datasets/flickr/annotations/final_flickr_separateGT_train_segm.json",
            ),
            dict(
                img_path="../datasets/mixed_grounding/gqa/images",
                json_file="../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
            ),
        ],
    ),
    val=dict(yolo_data=["lvis.yaml"]),
)

model_path = "yolov8l-worldv2-vl.yaml"

scale = guess_model_scale(model_path)
cfg_dir = "ultralytics/cfg"
default_cfg_path = f"{cfg_dir}/default.yaml"
extend_cfg_path = f"{cfg_dir}/{scale}_train.yaml"
defaults = yaml_load(default_cfg_path)
extends = yaml_load(extend_cfg_path)
assert(all(k in defaults for k in extends))
LOGGER.info(f"Extends: {extends}")

model = YOLOWorld("yolov8l-worldv2-vlhead-mobileclip-ladapterglu-imgsz800-alpha1-segm.pt")

freeze = list(range(0, 22))
for name, child in model.model.model[-1].named_children():
    if 'vpe' not in name:
        freeze.append(f"22.{name}")

model.train(data=data, batch=128, epochs=5, **extends, close_mosaic=2, \
    optimizer='AdamW', lr0=2e-3, warmup_bias_lr=0.0, \
        weight_decay=0.025, momentum=0.9, workers=4, \
        trainer=WorldSegVPTrainer, device='0,1,2,3,4,5,6,7', freeze=freeze, load_vp=True)