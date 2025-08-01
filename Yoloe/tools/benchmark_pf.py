import argparse
import cv2
import time
from tqdm import tqdm

import torch
from ultralytics import YOLOE
from ultralytics.utils import ops
from ultralytics.data.utils import yaml_load
from ultralytics.data.augment import LetterBox, ToTensor
from ultralytics.nn.modules.head import YOLOEDetect

@torch.inference_mode()
def measure_inference_time(model, dataset, device="cuda"):
    model.eval()
    model.to(device)
    # warmup
    imgsz=(1, 3, 640, 640)
    im = torch.empty(*imgsz, device=device)
    for _ in range(20):
        model(im)
    
    # measure speed
    timings = []
    for index, image in tqdm(enumerate(dataset), total=len(dataset)):
        image = image.unsqueeze(0).to(device)
        torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        preds = model(image)
        torch.cuda.synchronize(device)
        elapsed_time = time.perf_counter() - start_time  # 计算耗时
        
        timings.append(elapsed_time)

        # if index >= 100:
        #     break

    latency = sum(timings) / len(timings) * 1000
    fps = round(1000 / (latency + 1e-3), 2) 
    print("Latency:", round(latency, 2))
    print("FPS:", fps)

@torch.inference_mode()
def get_vl_model(model_prefix):
    model = YOLOE(f'{model_prefix}.yaml')
    model.load(f'runs/final5/{model_prefix}-seg/weights/best.pt')
    model.eval()
    model.cuda()

    with open('tools/ram_tag_list.txt', 'r') as f:
        names = [x.strip() for x in f.readlines()]

    model.set_classes(names, model.get_text_pe(names))
    model.fuse()
    return model.model


@torch.inference_mode()
def get_vl_pf_model(model_prefix):
    
    unfused_model = YOLOE(f"{model_prefix}.yaml")
    unfused_model.load(f"pretrain/{model_prefix}-seg/weights/best.pt")
    unfused_model.eval()
    unfused_model.cuda()

    with open('tools/ram_tag_list.txt', 'r') as f:
        names = [x.strip() for x in f.readlines()]
        
    vocab = unfused_model.get_vocab(names)

    torch.cuda.empty_cache()
    model = YOLOE(f"runs/final5/{model_prefix}-vl-seg-pf/weights/best.pt").cuda()
    model.eval()
    model.set_vocab(vocab, names=names)
    model.model.model[-1].is_fused = True
    model.model.model[-1].conf = 0.001
    model.model.model[-1].max_det = 1000
    # For detection evaluation
    model.model.model[-1].__class__ = YOLOEDetect
    model.fuse()
    return model.model

class SimpleDataset:
    def __init__(self, data, half=False):
        with open(data) as f:
            self.data = ["../datasets/lvis" + x[1:].strip() for x in f.readlines() if len(x.strip()) > 0]
        self.letterbox = LetterBox(new_shape=(640, 640), scaleup=False)
        self.to_tensor = ToTensor(half=half)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        filename = self.data[idx]
        image = cv2.imread(str(filename))
        image = self.letterbox(image=image)
        image = self.to_tensor(image)
        return image

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='yolov8l')
args = parser.parse_args()

model_prefix = args.name
print(f"Measuring {model_prefix} speed")
model = get_vl_pf_model(model_prefix)

torch.cuda.empty_cache()
dataset = SimpleDataset(data="../datasets/lvis/minival.txt")
measure_inference_time(model, dataset)