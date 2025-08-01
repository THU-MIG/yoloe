#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
""" Model schema in open_clip format for inference only. """
import math
from typing import Any, Optional, Dict

import torch
import torch.nn.functional as F
from torch import nn

from mobileclip.text_encoder import (
    TextTransformer,
)

from .image_encoder import MCi


class CLIP(nn.Module):
    """Base class for multi-modal image-text data"""

    def __init__(self, cfg: Dict, output_dict: bool = False, *args, **kwargs) -> None:
        super().__init__()
        self.output_dict = output_dict
        self.projection_dim = cfg["embed_dim"]
        if self.projection_dim is None:
            raise ValueError("Please specify `embed_dim` in model config.")

        self.image_encoder = MCi(
            model_name=cfg["image_cfg"]["model_name"],
            projection_dim=self.projection_dim,
        )
        self.text_encoder = TextTransformer(
            cfg=cfg["text_cfg"], projection_dim=self.projection_dim
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def _exponentiate_and_clip_logits(self, max_scale: float = 100.0):
        scale = self.logit_scale.exp()
        scale = torch.clamp(scale, 0, max_scale)
        return scale

    def encode_image(self, image: torch.Tensor, normalize: bool = False):
        image_encoder_out = self.image_encoder(image)
        if isinstance(image_encoder_out, dict):
            features = image_encoder_out["logits"]
        else:
            features = image_encoder_out
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text: torch.Tensor, normalize: bool = False):
        text_features = self.text_encoder(text_tokens=text, key_padding_mask=None)
        return F.normalize(text_features, dim=-1) if normalize else text_features

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> Any:

        image_embeddings = (
            self.encode_image(image, normalize=True) if image is not None else None
        )
        text_embeddings = (
            self.encode_text(text, normalize=True) if text is not None else None
        )

        if self.output_dict:
            return {
                "image_features": image_embeddings,
                "text_features": text_embeddings,
                "logit_scale": self._exponentiate_and_clip_logits(),
            }
        return image_embeddings, text_embeddings, self._exponentiate_and_clip_logits()
