#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
import json
from typing import Optional, Union, Tuple, Any

import torch
import torch.nn as nn
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Resize,
    ToTensor,
)

from mobileclip.clip import CLIP
from mobileclip.modules.text.tokenizer import (
    ClipTokenizer,
)
from mobileclip.modules.common.mobileone import reparameterize_model


def create_model_and_transforms(
    model_name: str,
    pretrained: Optional[str] = None,
    reparameterize: Optional[bool] = True,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[nn.Module, Any, Any]:
    """
    Method to instantiate model and pre-processing transforms necessary for inference.

    Args:
        model_name: Model name. Choose from ['mobileclip_s0', 'mobileclip_s1', 'mobileclip_s2', 'mobileclip_b']
        pretrained: Location of pretrained checkpoint.
        reparameterize: When set to True, re-parameterizable branches get folded for faster inference.
        device: Device identifier for model placement.

    Returns:
        Tuple of instantiated model, and preprocessing transforms for inference.
    """
    # Config files
    root_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(root_dir, "configs")
    model_cfg_file = os.path.join(configs_dir, model_name + ".json")

    # Get config from yaml file
    if not os.path.exists(model_cfg_file):
        raise ValueError(f"Unsupported model name: {model_name}")
    model_cfg = json.load(open(model_cfg_file, "r"))

    # Build preprocessing transforms for inference
    resolution = model_cfg["image_cfg"]["image_size"]
    resize_size = resolution
    centercrop_size = resolution
    aug_list = [
        Resize(
            resize_size,
            interpolation=InterpolationMode.BILINEAR,
        ),
        CenterCrop(centercrop_size),
        ToTensor(),
    ]
    preprocess = Compose(aug_list)

    # Build model
    model = CLIP(cfg=model_cfg)
    model.to(device)
    model.eval()

    # Load checkpoint
    if pretrained is not None:
        chkpt = torch.load(pretrained)
        model.load_state_dict(chkpt)

    # Reparameterize model for inference (if specified)
    if reparameterize:
        model = reparameterize_model(model)

    return model, None, preprocess


def get_tokenizer(model_name: str) -> nn.Module:
    # Config files
    root_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(root_dir, "configs")
    model_cfg_file = os.path.join(configs_dir, model_name + ".json")

    # Get config from yaml file
    model_cfg = json.load(open(model_cfg_file, "r"))

    # Build tokenizer
    text_tokenizer = ClipTokenizer(model_cfg)
    return text_tokenizer
