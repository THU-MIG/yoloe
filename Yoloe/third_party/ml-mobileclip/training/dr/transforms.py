#
# Copyright (C) 2023 Apple Inc. All rights reserved.
#
# https://github.com/apple/ml-dr/blob/main/LICENSE
# Reinforce Data, Multiply Impact: Improved Model Accuracy and Robustness with
# Dataset Reinforcement. , Faghri, F., Pouransari, H., Mehta, S., Farajtabar,
# M., Farhadi, A., Rastegari, M., & Tuzel, O., Proceedings of the IEEE/CVF
# International Conference on Computer Vision (ICCV), 2023.
#

"""Extending transformations from torchvision to be reproducible."""

from collections import defaultdict
from typing import List, OrderedDict, Union, Tuple, Optional, Any, Dict

import torch
from torch import Tensor
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms.autoaugment import _apply_op

import training.dr.transforms_base as transforms
from training.dr.transforms_base import clean_config


NO_PARAM = ()
NO_PARAM_TYPE = Tuple


class Compressible:
    """Base class for reproducible transformations with compressible parameters."""

    @staticmethod
    def compress_params(params: Any) -> Any:
        """Return compressed parameters."""
        return params

    @staticmethod
    def decompress_params(params: Any) -> Any:
        """Return decompressed parameters."""
        return params


class Resize(T.Resize, Compressible):
    """Extending PyTorch's Resize to reapply a given transformation."""

    def forward(
        self, img: Tensor, params: Optional[torch.Size] = None
    ) -> Tuple[Tensor, torch.Size]:
        """Transform an image randomly or reapply based on given parameters."""
        img = super().forward(img)
        return img, self.size


class CenterCrop(T.CenterCrop, Compressible):
    """Extending PyTorch's CenterCrop to reapply a given transformation."""

    def forward(
        self, img: Tensor, params: Optional[NO_PARAM_TYPE] = None
    ) -> Tuple[Tensor, NO_PARAM_TYPE]:
        """Transform an image randomly or reapply based on given parameters."""
        img = super().forward(img)
        # TODO: can we remove contiguous?
        img = img.contiguous()
        return img, NO_PARAM


class RandomCrop(T.RandomCrop, Compressible):
    """Extending PyTorch's RandomCrop to reapply a given transformation."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize super and set last parameters to None."""
        super().__init__(*args, **kwargs)
        self.params = None

    def get_params(self, *args, **kwargs) -> Tuple[int, int, int, int]:
        """Return self.params or new transformation params if self.params not set."""
        if self.params is None:
            self.params = super().get_params(*args, **kwargs)
        return self.params

    def forward(
        self, img: Tensor, params: Optional[Tuple[int, int]] = None
    ) -> Tuple[Tensor, Tuple[int, int]]:
        """Transform an image randomly or reapply based on given parameters."""
        self.params = None
        if params is not None:
            # Add the constant value of size
            self.params = (params[0], params[1], self.size[0], self.size[1])
        img = super().forward(img)
        params = self.params
        return img, params[:2]  # Return only [top, left], ie random parameters


class RandomResizedCrop(T.RandomResizedCrop, Compressible):
    """Extending PyTorch's RandomResizedCrop to reapply a given transformation."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize super and set last parameters to None."""
        if "interpolation" in kwargs and isinstance(kwargs['interpolation'], str):
            kwargs["interpolation"] = T.InterpolationMode(kwargs["interpolation"])
        super().__init__(*args, **kwargs)
        self.params = None

    def get_params(self, *args, **kwargs) -> Tuple[int, int, int, int]:
        """Return self.params or new transformation params if self.params not set."""
        if self.params is None:
            self.params = super().get_params(*args, **kwargs)
        return self.params

    def forward(
        self, img: Tensor, params: Optional[Tuple[int, int, int, int]] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tensor, Tuple[int, int, int, int]]:
        """Transform an image randomly or reapply based on given parameters."""
        self.params = None
        if params is not None:
            self.params = params
        size_old = self.size
        if size is not None:
            # Support for variable batch size
            self.size = size
        img = super().forward(img)
        self.size = size_old
        params = self.params
        return img, params


class RandomHorizontalFlip(T.RandomHorizontalFlip, Compressible):
    """Extending PyTorch's RandomHorizontalFlip to reapply a given transformation."""

    def forward(
        self, img: Tensor, params: Optional[bool] = None
    ) -> Tuple[Tensor, bool]:
        """Transform an image randomly or reapply based on given parameters.

        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if params is None:
            # Randomly skip only if params=None
            params = torch.rand(1).item() < self.p
        if params:
            img = F.hflip(img)
        return img, params


class RandAugment(T.RandAugment, Compressible):
    """Extending PyTorch's RandAugment to reapply a given transformation."""

    op_names = [
        "Identity",
        "ShearX",
        "ShearY",
        "TranslateX",
        "TranslateY",
        "Rotate",
        "Brightness",
        "Color",
        "Contrast",
        "Sharpness",
        "Posterize",
        "Solarize",
        "AutoContrast",
        "Equalize",
    ]

    def __init__(self, p: float = 1.0, *args, **kwargs) -> None:
        """Initialize RandAugment with probability p of augmentation.

        Args:
            p: The probability of applying transformation. A float in [0, 1.0].
        """
        super().__init__(*args, **kwargs)
        self.p = p

    def forward(
        self, img: Tensor, params: Optional[List[Tuple[str, float]]] = None, **kwargs
    ) -> Tuple[Tensor, List[Tuple[str, float]]]:
        """Transform an image randomly or reapply based on given parameters.

        Args:
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins, (height, width))
        if params is None:
            # Randomly skip only if params=None
            if torch.rand(1) > self.p:
                return img, None

            params = []
            for _ in range(self.num_ops):
                op_index = int(torch.randint(len(op_meta), (1,)).item())
                op_name = list(op_meta.keys())[op_index]
                magnitudes, signed = op_meta[op_name]
                magnitude = (
                    float(magnitudes[self.magnitude].item())
                    if magnitudes.ndim > 0
                    else 0.0
                )
                if signed and torch.randint(2, (1,)):
                    magnitude *= -1.0
                params += [(op_name, magnitude)]

        for i in range(self.num_ops):
            op_name, magnitude = params[i]
            img = _apply_op(
                img, op_name, magnitude, interpolation=self.interpolation, fill=fill
            )

        return img, params

    @staticmethod
    def compress_params(params: List[Tuple[str, float]]) -> List[Tuple[int, float]]:
        """Return compressed parameters."""
        if params is None:
            return None
        pc = []
        for p in params:
            pc += [(RandAugment.op_names.index(p[0]), p[1])]
        return pc

    @staticmethod
    def decompress_params(params: List[Tuple[int, float]]) -> List[Tuple[str, float]]:
        """Return decompressed parameters."""
        if params is None:
            return None
        pc = []
        for p in params:
            pc += [(RandAugment.op_names[p[0]], p[1])]
        return pc


class RandomErasing(T.RandomErasing, Compressible):
    """Extending PyTorch's RandomErasing to reapply a given transformation."""

    def forward(
        self, img: Tensor, params: Optional[Tuple] = None, **kwargs
    ) -> Tuple[Tensor, Tuple]:
        """Transform an image randomly or reapply based on given parameters.

        Args:
            img (Tensor): Tensor image to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        if params is None:
            # Randomly skip only if params=None
            if torch.rand(1) > self.p:
                return img, None
            x, y, h, w, _ = self.get_params(img, scale=self.scale, ratio=self.ratio)
        else:
            x, y, h, w = params
        # In early experiments F.erase used in pytorch's RE was very slow
        # TODO: verify that F.erase is still slower than assigning zeros
        if x != -1:
            img[:, x : x + h, y : y + w] = 0
        return img, (x, y, h, w)


class Normalize(T.Normalize, Compressible):
    """PyTorch's Normalize with an extra dummy transformation parameter."""

    def forward(
        self, *args, params: Optional[NO_PARAM_TYPE] = None, **kwargs
    ) -> Tuple[Tensor, Tuple]:
        """Return normalized input and NO_PARAM as parameters."""
        x = super().forward(*args, **kwargs)
        return x, NO_PARAM


class MixUp(transforms.MixUp, Compressible):
    """Extending MixUp to reapply a given transformation."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize super and set last parameters to None."""
        super().__init__(*args, **kwargs)
        self.params = None

    def get_params(self, *args, **kwargs) -> float:
        """Return self.params or new transformation params if self.params not set."""
        if self.params is None:
            self.params = super().get_params(*args, **kwargs)
        return self.params

    def forward(
        self,
        x: Tensor,
        x2: Tensor,
        y: Optional[Tensor] = None,
        y2: Optional[Tensor] = None,
        params: Dict[str, float] = None,
    ) -> Tuple[Tuple[Tensor, Tensor], Dict[str, float]]:
        """Transform an image randomly or reapply based on given parameters."""
        self.params = None
        if params is not None:
            self.params = params
        x, y = super().forward(x, x2, y, y2)
        params = self.params
        return (x, y), params


class CutMix(transforms.CutMix, Compressible):
    """Extending CutMix to reapply a given transformation."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize super and set last parameters to None."""
        super().__init__(*args, **kwargs)
        self.params = None

    def get_params(self, *args, **kwargs) -> Tuple[float, Tuple[int, int, int, int]]:
        """Return self.params or new transformation params if self.params not set."""
        if self.params is None:
            self.params = super().get_params(*args, **kwargs)
        return self.params

    def forward(
        self,
        x: Tensor,
        x2: Tensor,
        y: Optional[Tensor] = None,
        y2: Optional[Tensor] = None,
        params: Dict[str, float] = None,
    ) -> Tuple[
        Tuple[Tensor, Tensor], Dict[str, Union[float, Tuple[int, int, int, int]]]
    ]:
        """Transform an image randomly or reapply based on given parameters."""
        self.params = None
        if params is not None:
            self.params = params
        x, y = super().forward(x, x2, y, y2)
        params = self.params
        return (x, y), params

    @staticmethod
    def compress_params(params: Any) -> Any:
        """Return compressed parameters."""
        if params is None:
            return None
        return [params[0]]+list(params[1])

    @staticmethod
    def decompress_params(params: Any) -> Any:
        """Return decompressed parameters."""
        if params is None:
            return None
        return params[0], tuple(params[1:])


class ToUint8(torch.nn.Module, Compressible):
    """Convert float32 Tensor in range [0, 1] to uint8 [0, 255]."""

    def forward(self, img: Tensor, **kwargs) -> Tuple[Tensor, NO_PARAM_TYPE]:
        """Return uint8(img) and NO_PARAM as parameters."""
        if not isinstance(img, torch.Tensor):
            return img, NO_PARAM
        return (img * 255).to(torch.uint8), NO_PARAM


class ToTensor(torch.nn.Module, Compressible):
    """Convert PIL to torch.Tensor or if Tensor uint8 [0, 255] to float32 [0, 1]."""

    def forward(self, img: Tensor, **kwargs) -> Tuple[Tensor, NO_PARAM_TYPE]:
        """Return tensor(img) and NO_PARAM as parameters."""
        if isinstance(img, torch.Tensor):
            """Return float32(img) and NO_PARAM as parameters."""
            return (img / 255.0).to(torch.float32), NO_PARAM
        return F.to_tensor(img), NO_PARAM


# Transformations are composed according to the order below, not the order in config
TRANSFORMATION_TO_NAME = OrderedDict(
    [
        ("uint8", ToUint8),
        ("resize", Resize),
        ("center_crop", CenterCrop),
        ("random_crop", RandomCrop),
        ("random_resized_crop", RandomResizedCrop),
        ("random_horizontal_flip", RandomHorizontalFlip),
        ("rand_augment", RandAugment),
        ("to_tensor", ToTensor),
        ("random_erase", RandomErasing),  # TODO: fix the order of RE with transforms
        ("normalize", Normalize),
        ("mixup", MixUp),
        ("cutmix", CutMix),
    ]
)
# Only in datagen
BEFORE_COLLATE_TRANSFORMS = [
    "uint8",
    "resize",
    "center_crop",
    "random_crop",
    "random_resized_crop",
    "to_tensor",
]
NO_PARAM_TRANSFORMS = [
    "uint8",
    "center_crop",
    "to_tensor",
    "normalize",
]


class Compose:
    """Compose a list of reproducible data transformations."""

    def __init__(self, transforms: List[Tuple[str, Compressible]]) -> None:
        """Initialize transformations."""
        self.transforms = transforms

    def has_random_resized_crop(self) -> bool:
        """Return True if RandomResizedCrop is one of the transformations."""
        return any([t.__class__ == RandomResizedCrop for _, t in self.transforms])

    def __call__(
        self,
        img: Tensor,
        img2: Tensor = None,
        after_collate: Optional[bool] = False,
        size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Apply a transformation to two images and return augmentation parameters.

        Args:
            img: A tensor to be transformed.
            params: Transformation parameters to be reapplied.
            img2: Second tensor to be used for mixing transformations.

        The value of `params` can be None or empty in 3 cases:
            1) `params=None` in apply(): The value should be generated randomly,
            2) `params=None` in reapply(): Transformation was randomly skipped during
            generation time,
            3) `params=()`: Trasformation has no random parameters.

        Returns:
            A Tuple of a transformed image and a dictionary with transformation
            parameters.
        """
        params = dict()
        for t_name, t in self.transforms:
            if after_collate and (
                t_name in BEFORE_COLLATE_TRANSFORMS
                and t_name != "uint8"
                and t_name != "to_tensor"
            ):
                # Skip transformations applied in data loader
                pass
            elif t_name == "cutmix" or t_name == "mixup":
                # Mix images
                if img2 is not None:
                    (img, _), p = t(img, img2)
                    params[t_name] = p
            else:
                # Apply an augmentation to both images, skip img2 if no mixing
                img, p = t(img)
                if img2 is not None:
                    if t_name == 'random_resized_crop':
                        img2, p2 = t(img2, size=size)
                    else:
                        img2, p2 = t(img2)
                    p = (p, p2)
                params[t_name] = p
        return img, params

    def reapply(
        self, img: Tensor, params: Dict[str, Any], img2: Tensor = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Reapply transformations to an image given augmentation parameters.

        Args:
            img: A tensor to be transformed.
            params: Transformation parameters to be reapplied.
            img2: Second tensor to be used for mixing transformations.

        The value of `params` can be None or empty in 3 cases:
            1) `params=None` in apply(): The value should be generated randomly,
            2) `params=None` in reapply(): Transformation was randomly skipped during
            generation time,
            3) `params=()`: Trasformation has no random parameters.

        Returns:
            A Tuple of a transformed image and a dictionary with transformation
            parameters.
        """
        for t_name, t in self.transforms:
            if t_name in params:
                if t_name == "cutmix" or t_name == "mixup":
                    # Remix images
                    if params[t_name] is not None:
                        (img, _), _ = t(img, img2, params=params[t_name])
                else:
                    # Reapply an augmentation to both images, skip img2 if no
                    # mixing
                    if params[t_name][0] is not None:
                        if t_name == 'random_resized_crop':
                            img, _ = t(img, params=params[t_name][0], size=size)
                        else:
                            img, _ = t(img, params=params[t_name][0])
                    if img2 is not None and params[t_name][1] is not None:
                        img2, _ = t(img2, params=params[t_name][1])
        return img, params

    def compress(self, params: Dict[str, Any]) -> List[Any]:
        """Compress augmentation parameters."""
        params_compressed = []

        no_pair = True
        # Save second pair id if mixup or cutmix enabled
        t_names = [t[0] for t in self.transforms]
        if "mixup" in t_names or "cutmix" in t_names:
            if (
                params.get("mixup", None) is not None
                or params.get("cutmix", None) is not None
            ):
                params_compressed += [params["id2"]]
                no_pair = False
            else:
                params_compressed += [None]

        # Save transformation parameters
        for t_name, t in self.transforms:
            p = params[t_name]
            if t_name in NO_PARAM_TRANSFORMS:
                pass
            elif t_name == "mixup" or t_name == "cutmix":
                params_compressed += [t.compress_params(p)]
            else:
                if no_pair:
                    params_compressed += [t.compress_params(p)]
                else:
                    params_compressed += [
                        [t.compress_params(p[0]), t.compress_params(p[1])]
                    ]
        return params_compressed

    def decompress(self, params_compressed: List[Any]) -> Dict[str, Any]:
        """Decompress augmentation parameters."""
        params = {}

        # Read second pair id if mixup or cutmix enabled
        t_names = [t[0] for t in self.transforms]
        no_pair = None
        if "mixup" in t_names or "cutmix" in t_names:
            no_pair = params_compressed[0]
            if no_pair is not None:
                params["id2"] = no_pair
            params_compressed = params_compressed[1:]

        # Read parameters for transformations with random parameters
        with_param_transforms = [(t_name, t)
                                 for t_name, t in self.transforms
                                 if t_name not in NO_PARAM_TRANSFORMS]
        for p, (t_name, t) in zip(params_compressed, with_param_transforms):
            if p is None:
                pass
            elif t_name == "mixup" or t_name == "cutmix":
                params[t_name] = t.decompress_params(p)
            else:
                if no_pair is not None and len(p) > 1:
                    params[t_name] = (
                        t.decompress_params(p[0]),
                        t.decompress_params(p[1]),
                    )
                else:
                    params[t_name] = (t.decompress_params(p),)

        # Fill non-random transformations
        for t_name, t in self.transforms:
            if t_name in NO_PARAM_TRANSFORMS:
                params[t_name] = (NO_PARAM, NO_PARAM)
        return params


def compose_from_config(config: Dict[str, Any]) -> Compose:
    """Initialize transformations given the dataset name and configurations.

    Args:
        config: A dictionary of augmentation parameters.
    """
    config = clean_config(config)
    transforms = []
    for t_name, t_class in TRANSFORMATION_TO_NAME.items():
        if t_name in config:
            # TODO: warn for every key in config_tr that was not used
            transforms += [(t_name, t_class(**config[t_name]))]
    return Compose(transforms)


def before_collate_config(
    config: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Return configs with resize/crop transformations to pass to data loader.

    Only transformations that cannot be applied after data collate are
    composed. For example, RandomResizedCrop has to be applied before collate
    To create tensors of similar shapes.

    Args:
        config: A dictionary of augmentation parameters.
    """
    return {k: v for k, v in config.items() if k in BEFORE_COLLATE_TRANSFORMS}


def after_collate_config(
    config: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Return configs after excluding transformations from `befor_collate_config`."""
    return {k: v for k, v in config.items() if k not in BEFORE_COLLATE_TRANSFORMS}


def before_collate_apply(
    sample: Tensor, transform: Compose, num_samples: int
) -> Tuple[Tensor, Tensor]:
    """Return multiple samples applying the transformations.

    Args:
        sample: A single sample to be randomly transformed.
        transform: A list of transformations to be applied.
        num_samples: The number of random transformations to be generated.

    Returns:
        Random transformations of the input. Shape: [num_samples,]+sample.shape
    """
    sample_all = []
    params_all = defaultdict(list)
    for _ in range(num_samples):
        # [height, width, channels]
        # -> ([height_new, width_new, channels], Dict(str, Tuple))
        sample_new, params = transform(sample)
        sample_all.append(sample_new)
        for k, v in params.items():
            params_all[k].append(v)

    sample_all = torch.stack(sample_all, axis=0)
    for k in params_all.keys():
        params_all[k] = torch.tensor(params_all[k])
    return sample_all, params_all
