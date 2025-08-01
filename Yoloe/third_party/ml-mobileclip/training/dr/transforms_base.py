#
# Copyright (C) 2023 Apple Inc. All rights reserved.
#
# https://github.com/apple/ml-dr/blob/main/LICENSE
# Reinforce Data, Multiply Impact: Improved Model Accuracy and Robustness with
# Dataset Reinforcement. , Faghri, F., Pouransari, H., Mehta, S., Farajtabar,
# M., Farhadi, A., Rastegari, M., & Tuzel, O., Proceedings of the IEEE/CVF
# International Conference on Computer Vision (ICCV), 2023.
#

"""Simplified composition of PyTorch transformations from a configuration dictionary."""

import math
import random
from typing import Any, Dict, Optional, OrderedDict, Tuple
import numpy as np

import timm
from timm.data.transforms import str_to_interp_mode
import torch
from torch import Tensor
import torchvision.transforms as T
from torch.nn import functional as F


INTERPOLATION_MODE_MAP = {
    "nearest": T.InterpolationMode.NEAREST,
    "bilinear": T.InterpolationMode.BILINEAR,
    "bicubic": T.InterpolationMode.BICUBIC,
    "cubic": T.InterpolationMode.BICUBIC,
    "box": T.InterpolationMode.BOX,
    "hamming": T.InterpolationMode.HAMMING,
    "lanczos": T.InterpolationMode.LANCZOS,
}


class AutoAugment(T.AutoAugment):
    """Extend PyTorch's AutoAugment to init from a policy and an interpolation name."""

    def __init__(
        self, policy: str = "imagenet", interpolation: str = "bilinear", *args, **kwargs
    ) -> None:
        """Init from an policy and interpolation name."""
        if "cifar" in policy.lower():
            policy = T.AutoAugmentPolicy.CIFAR10
        elif "svhn" in policy.lower():
            policy = T.AutoAugmentPolicy.SVHN
        else:
            policy = T.AutoAugmentPolicy.IMAGENET
        interpolation = INTERPOLATION_MODE_MAP[interpolation]
        super().__init__(*args, policy=policy, interpolation=interpolation, **kwargs)


class RandAugment(T.RandAugment):
    """Extend PyTorch's RandAugment to init from an interpolation name."""

    def __init__(self, interpolation: str = "bilinear", *args, **kwargs) -> None:
        """Init from an interpolation name."""
        interpolation = INTERPOLATION_MODE_MAP[interpolation]
        super().__init__(*args, interpolation=interpolation, **kwargs)


class TrivialAugmentWide(T.TrivialAugmentWide):
    """Extend PyTorch's TrivialAugmentWide to init from an interpolation name."""

    def __init__(self, interpolation: str = "bilinear", *args, **kwargs) -> None:
        """Init from an interpolation name."""
        interpolation = INTERPOLATION_MODE_MAP[interpolation]
        super().__init__(*args, interpolation=interpolation, **kwargs)


# Transformations are composed according to the order in this dict, not the order in
# yaml config
TRANSFORMATION_TO_NAME = OrderedDict(
    [
        ("resize", T.Resize),
        ("center_crop", T.CenterCrop),
        ("random_crop", T.RandomCrop),
        ("random_resized_crop", T.RandomResizedCrop),
        ("random_horizontal_flip", T.RandomHorizontalFlip),
        ("rand_augment", RandAugment),
        ("auto_augment", AutoAugment),
        ("trivial_augment_wide", TrivialAugmentWide),
        ("to_tensor", T.ToTensor),
        ("random_erase", T.RandomErasing),
        ("normalize", T.Normalize),
    ]
)


def timm_resize_crop_norm(config: Dict[str, Any]) -> torch.nn.Module:
    """Set Resize/RandomCrop/Normalization parameters from configs of a Timm teacher."""
    teacher_name = config["timm_resize_crop_norm"]["name"]
    cfg = timm.models.get_pretrained_cfg(teacher_name).to_dict()
    if "test_input_size" in cfg:
        img_size = list(cfg["test_input_size"])[-1]
    else:
        img_size = list(cfg["input_size"])[-1]
    # Crop ratio and image size for optimal performance of a Timm model
    crop_pct = cfg["crop_pct"]
    scale_size = int(math.floor(img_size / crop_pct))
    interpolation = cfg["interpolation"]
    config["resize"] = {
        "size": scale_size,
        "interpolation": str_to_interp_mode(interpolation),
    }
    config["random_crop"] = {
        "size": img_size,
        "pad_if_needed": True,
    }
    config["normalize"] = {"mean": cfg["mean"], "std": cfg["std"]}
    return config


def clean_config(config: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return a clone of configs and remove unnecessary keys from configurations."""
    new_config = {}
    for k, v in config.items():
        vv = dict(v)
        if vv.pop("enable", True):
            new_config[k] = vv
    return new_config


def compose_from_config(config_tr: Dict[str, Any]) -> torch.nn.Module:
    """Initialize transformations given the dataset name and configurations.

    Args:
        config_tr: A dictionary of transformation parameters.

    Returns a composition of transformations.
    """
    config_tr = clean_config(config_tr)
    if "timm_resize_crop_norm" in config_tr:
        config_tr = timm_resize_crop_norm(config_tr)
    transforms = []
    for t_name, t_class in TRANSFORMATION_TO_NAME.items():
        if t_name in config_tr:
            # TODO: warn for every key in config_tr that was not used
            transforms += [t_class(**config_tr[t_name])]
    return T.Compose(transforms)


class MixUp(torch.nn.Module):
    r"""MixUp image transformation.

    For an input x the
    output is :math:`\lambda x + (1-\lambda) x_p` , where :math:`x_p` is a
    random permutation of `x` along the batch dimension, and lam is a random
    number between 0 and 1.
    See https://arxiv.org/abs/1710.09412 for more details.
    """

    def __init__(
        self, alpha: float = 1.0, p: float = 1.0, div_by: float = 1.0, *args, **kwargs
    ) -> None:
        """Initialize MixUp transformation.

        Args:
            alpha: A positive real number that determines the sampling
                distribution. Each mixed sample is a convex combination of two
                examples from the batch with mixing coefficient lambda.
                lambda is sampled from a symmetric Beta distribution with
                parameter alpha. When alpha=0 no mixing happens. Defaults to 1.0.
            p: Mixing is applied with probability `p`. Defaults to 1.0.
            div_by: Divide the lambda by a constant. Set to 2.0 to make sure mixing is
                biased towards the first input. Defaults to 1.0.
        """
        super().__init__(*args, **kwargs)
        assert alpha >= 0
        assert p >= 0 and p <= 1.0
        assert div_by >= 1.0
        self.alpha = alpha
        self.p = p
        self.div_by = div_by

    def get_params(self, alpha: float, div_by: float) -> float:
        """Return MixUp random parameters."""
        # Skip mixing by probability 1-self.p
        if alpha == 0 or torch.rand(1) > self.p:
            return None

        lam = np.random.beta(alpha, alpha) / div_by
        return lam

    def forward(
        self,
        x: Tensor,
        x2: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        y2: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""Apply pixel-space mixing to a batch of examples.

        Args:
            x: A tensor with a batch of samples. Shape: [batch_size, ...].
            x2: A tensor with exactly one matching sample for any input in `x`. Shape:
                [batch_size, ...].
            y: A tensor of target labels. Shape: [batch_size, ...].
            y2: A tensor of target labels for paired samples. Shape: [batch_size, ...].

        Returns:
            Mixed x tensor, y labels, and dictionary of mixing parameter {'lam': lam}.
        """
        alpha = self.alpha
        # Randomly sample lambda if not provided
        params = self.get_params(alpha, self.div_by)
        if params is None:
            return x, y
        lam = params

        # Randomly sample second input from the same mini-batch if not provided
        if x2 is None:
            batch_size = int(x.size()[0])
            index = torch.randperm(batch_size, device=x.device)
            x2 = x[index, :]
            y2 = y[index, :] if y is not None else None

        # Mix inputs and labels
        mixed_x = lam * x + (1 - lam) * x2
        mixed_y = y
        if y is not None:
            mixed_y = lam * y + (1 - lam) * y2

        return mixed_x, mixed_y


class CutMix(torch.nn.Module):
    r"""CutMix image transformation.

    Please see the full paper for more details:
    https://arxiv.org/pdf/1905.04899.pdf
    """

    def __init__(self, alpha: float = 1.0, p: float = 1.0, *args, **kwargs) -> None:
        """Initialize CutMix transformation.

        Args:
            alpha: The alpha parameter to the Beta for producing a mixing lambda.
        """
        super().__init__(*args, **kwargs)
        assert alpha >= 0
        assert p >= 0 and p <= 1.0
        self.alpha = alpha
        self.p = p

    @staticmethod
    def rand_bbox(size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
        """Return a random bbox coordinates.

        Args:
            size: model input tensor shape in this format: (...,H,W)
            lam: lambda sampling parameter in CutMix method. See equation 1
                in the original paper: https://arxiv.org/pdf/1905.04899.pdf

        Returns:
            The output bbox format is a tuple: (x1, y1, x2, y2), where (x1,
            y1) and (x2,y2) are the coordinates of the top-left and bottom-right
            corners of the bbox in the pixel-space.
        """
        assert lam >= 0 and lam <= 1.0
        h = size[-2]
        w = size[-1]
        cut_rat = np.sqrt(1.0 - lam)
        cut_h = int(h * cut_rat)
        cut_w = int(w * cut_rat)

        # uniform
        cx = np.random.randint(h)
        cy = np.random.randint(w)

        bbx1 = np.clip(cx - cut_h // 2, 0, h)
        bby1 = np.clip(cy - cut_w // 2, 0, w)
        bbx2 = np.clip(cx + cut_h // 2, 0, h)
        bby2 = np.clip(cy + cut_w // 2, 0, w)

        return (bbx1, bby1, bbx2, bby2)

    def get_params(
        self, size: torch.Size, alpha: float
    ) -> Tuple[float, Tuple[int, int, int, int]]:
        """Return CutMix random parameters."""
        # Skip mixing by probability 1-self.p
        if alpha == 0 or torch.rand(1) > self.p:
            return None

        lam = np.random.beta(alpha, alpha)
        # Compute mask
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(size, lam)
        return lam, (bbx1, bby1, bbx2, bby2)

    def forward(
        self,
        x: Tensor,
        x2: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        y2: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Mix images by replacing random patches from one to the other.

        Args:
            x: A tensor with a batch of samples. Shape: [batch_size, ...].
            x2: A tensor with exactly one matching sample for any input in `x`. Shape:
                [batch_size, ...].
            y: A tensor of target labels. Shape: [batch_size, ...].
            y2: A tensor of target labels for paired samples. Shape: [batch_size, ...].
            params: Dictionary of {'lam': lam_val} to reproduce a mixing.

        """
        alpha = self.alpha

        # Randomly sample lambda and bbox coordinates if not provided
        params = self.get_params(x.shape, alpha)
        if params is None:
            return x, y
        lam, (bbx1, bby1, bbx2, bby2) = params

        # Randomly sample second input from the same mini-batch if not provided
        if x2 is None:
            batch_size = int(x.size()[0])
            index = torch.randperm(batch_size, device=x.device)
            x2 = x[index, :]
            y2 = y[index, :] if y is not None else None

        # Mix inputs and labels
        mixed_x = x.detach().clone()
        mixed_x[:, bbx1:bbx2, bby1:bby2] = x2[:, bbx1:bbx2, bby1:bby2]
        mixed_y = y
        if y is not None:
            # Adjust lambda
            lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
            mixed_y = lam * y + (1 - lam) * y2

        return mixed_x, mixed_y


class MixingTransforms:
    """Randomly apply only one of MixUp or CutMix. Used for standard training."""

    def __init__(self, config_tr: Dict[str, Any], num_classes: int) -> None:
        """Initialize mixup and/or cutmix."""
        config_tr = clean_config(config_tr)
        self.mixing_transforms = []
        if "mixup" in config_tr:
            self.mixing_transforms += [MixUp(**config_tr["mixup"])]
        if "cutmix" in config_tr:
            self.mixing_transforms += [CutMix(**config_tr["cutmix"])]
        self.num_classes = num_classes

    def __call__(self, images: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply only one of MixUp or CutMix."""
        if len(self.mixing_transforms) > 0:
            one_hot_label = F.one_hot(target, num_classes=self.num_classes)
            mix_f = random.choice(self.mixing_transforms)
            images, target = mix_f(x=images, y=one_hot_label)
        return images, target
