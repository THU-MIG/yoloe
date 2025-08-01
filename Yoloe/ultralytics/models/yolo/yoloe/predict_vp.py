from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.data.augment import LetterBox, LoadVisualPrompt
from ultralytics.utils.instance import Instances
import numpy as np
import torch
from copy import deepcopy

from ultralytics.utils.torch_utils import select_device

class YOLOEVPPredictorMixin:
    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        device = select_device(self.args.device, verbose=verbose)
        self.model = model.to(device)

        self.device = device  # update device
        self.model.fp16 = False
        self.args.half = False
        self.model.eval()
        
        self.done_warmup = True
        self.return_vpe = False
        
    def set_return_vpe(self, return_vpe):
        self.return_vpe = return_vpe
    
    def set_prompts(self, prompts):
        self.prompts = deepcopy(prompts)
    
    def load_vp(self, label):
        label["img"] = label["img"].transpose(2, 0, 1)
        load_vp = LoadVisualPrompt(nc=len(label["cls"]), augment=False)
        label = load_vp(label)
        label["img"] = label["img"].transpose(1, 2, 0)
        return label
    
    def process_box_label(self, img, bboxes, cls, letterbox):
        label = dict(
            img=img,
            instances=Instances(bboxes=bboxes.astype(np.float32), 
                                segments=np.zeros((0, 1000, 2), dtype=np.float32), 
                                bbox_format="xyxy", normalized=False),
            cls=torch.tensor(cls).unsqueeze(-1)
        )
        label = letterbox(label)
        instances = label.pop("instances")
        h, w = label["img"].shape[:2]
        instances.normalize(w, h)
        instances.convert_bbox(format="xywh")
        label["bboxes"] = torch.from_numpy(instances.bboxes)
        return self.load_vp(label)
    
    def process_mask_label(self, img, masks, cls, letterbox):
        img = letterbox(image=img)
        masks = np.stack([letterbox(image=mask) for mask in masks])
        masks[masks == 114] = 0
        label = dict(
            img=img,
            masks=masks,
            cls=torch.tensor(cls).unsqueeze(-1)
        )
        return self.load_vp(label)
        
    def pre_transform(self, im):
        letterbox = LetterBox(
            self.imgsz,
            auto=False,
            stride=int(self.model.stride[-1].item()),
        )

        cls = self.prompts["cls"]
        cls = [cls] if not isinstance(cls, list) else cls
            
        if "bboxes" in self.prompts:
            bboxes = self.prompts["bboxes"]
            bboxes = [bboxes] if not isinstance(bboxes, list) else bboxes
            labels = [self.process_box_label(im[i], bboxes[i], cls[i], letterbox) for i in range(len(im))]
        elif "masks" in self.prompts:
            masks = self.prompts["masks"]
            masks = [masks] if not isinstance(masks, list) else masks
            labels = [self.process_mask_label(im[i], masks[i], cls[i], letterbox) for i in range(len(im))]
        else:
            raise ValueError("Please provide valid bboxes or masks")

        self.prompts = torch.nn.utils.rnn.pad_sequence([label["visuals"] for label in labels], batch_first=True).to(self.device)
        
        self.model.model[-1].nc = self.prompts.shape[1]
        self.model.names = [f"object{i}" for i in range(self.prompts.shape[1])]
        
        return [label["img"] for label in labels]

    def inference(self, im, *args, **kwargs):
        if self.return_vpe:
            self.vpe = self.model.get_visual_pe(im, visual=self.prompts)
        return super().inference(im, vpe=self.prompts, *args, **kwargs)


class YOLOEVPDetectPredictor(YOLOEVPPredictorMixin, DetectionPredictor):
    pass

class YOLOEVPSegPredictor(YOLOEVPPredictorMixin, SegmentationPredictor):
    pass