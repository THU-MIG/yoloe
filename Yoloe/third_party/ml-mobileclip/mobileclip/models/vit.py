#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""
Implementation of the following modules is borrowed from ml-cvnets repo:
https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/vit.py

Please see ACKNOWLEDGEMENTS for license details.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from timm.models import register_model
from mobileclip.modules.common.transformer import (
    PositionalEmbedding,
    TransformerEncoder,
    get_normalization_layer,
)
from mobileclip.modules.image.image_projection import SimpleImageProjectionHead
from mobileclip import logger


class ConvNormAct(nn.Module):
    """
    Applies an N-dimensional convolution over an input.

    Args:
        cfg: Model configuration.
        in_channels: :math:`C_{out}` from an expected output of size
            :math:`(bs, C_{in}, X_{1}, ..., X_{N})`.
        out_channels: :math:`C_{out}` from an expected output of size
            :math:`(bs, C_{out}, Y_{1}, ..., Y_{N})`.
        kernel_size: Kernel size for convolution. An integer, or tuple of length ``N``.
        stride: Stride for convolution. An integer, or tuple of length ``N``. Default: 1.
        dilation: Dilation rate for convolution. An integer, or tuple of length ``N``.
            Default: ``1``.
        padding: Padding for convolution. An integer, or tuple of length ``N``.
            If not specified, padding is automatically computed based on kernel size and
            dilation range. Default : ``None`` (equivalent to ``[
            int((kernel_size[i] - 1) / 2) * dilation[i] for i in range(N)]``).
        groups: Number of groups in convolution. Default: ``1``.
        bias: Use bias. Default: ``False``.
        padding_mode: Padding mode ('zeros', 'reflect', 'replicate' or 'circular').
            Default: ``zeros``.
        use_norm: Use normalization layer after convolution. Default: ``True``.
        use_act: Use activation layer after convolution (or convolution and normalization).
            Default: ``True``.
        norm_layer: If not None, the provided normalization layer object will be used.
            Otherwise, a normalization object will be created based on config
            ``model.normalization.*`` opts.
        act_layer: If not None, the provided activation function will be used.
            Otherwise, an activation function will be created based on config
            ``model.activation.*`` opts.

    Shape:
        - Input: :math:`(bs, C_{in}, X_{1}, ..., X_{N})`.
        - Output: :math:`(bs, C_{out}, Y_{1}, ..., Y_{N})`.

    .. note::
        For depth-wise convolution, `groups=C_{in}=C_{out}`.
    """

    def __init__(
        self,
        cfg: Dict,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        dilation: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...]]] = None,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
        use_norm: bool = True,
        use_act: bool = True,
        norm_layer: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.ndim = 2

        if norm_layer is None and use_norm:
            norm_type = cfg.get("normalization", "batch_norm")
            if norm_type == "batch_norm":
                norm_layer = nn.BatchNorm2d(
                    num_features=out_channels,
                    momentum=cfg.get("momentum", 0.1),
                )
            else:
                norm_layer = get_normalization_layer(
                    num_features=out_channels, norm_type=norm_type
                )
        elif norm_layer is not None and use_norm:
            logger.error(
                f"When use_norm is False, norm_layer should be None, but norm_layer={norm_layer} is provided."
            )

        if act_layer is None and use_act:
            act_layer = nn.GELU()  # Default to GELU
        elif act_layer is not None and use_act:
            logger.error(
                f"When use_act is False, act_layer should be None, but act_layer={act_layer} is provided."
            )

        if (
            use_norm
            and any(param[0] == "bias" for param in norm_layer.named_parameters())
            and bias
        ):
            assert (
                not bias
            ), "Do not use bias when using normalization layers with bias."

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * self.ndim

        if isinstance(stride, int):
            stride = (stride,) * self.ndim

        if isinstance(dilation, int):
            dilation = (dilation,) * self.ndim

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)
        assert isinstance(dilation, Tuple)

        if padding is None:
            padding = (
                int((kernel_size[i] - 1) / 2) * dilation[i] for i in range(self.ndim)
            )

        if in_channels % groups != 0:
            logger.error(
                "Input channels are not divisible by groups. {}%{} != 0 ".format(
                    in_channels, groups
                )
            )
        if out_channels % groups != 0:
            logger.error(
                "Output channels are not divisible by groups. {}%{} != 0 ".format(
                    out_channels, groups
                )
            )

        block = nn.Sequential()

        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,  # type: ignore
            stride=stride,  # type: ignore
            padding=padding,
            dilation=dilation,  # type: ignore
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        block.add_module(name="conv", module=conv_layer)

        self.norm_name = None
        if use_norm:
            block.add_module(name="norm", module=norm_layer)
            self.norm_name = norm_layer.__class__.__name__

        self.act_name = None
        if use_act:
            block.add_module(name="act", module=act_layer)
            self.act_name = act_layer.__class__.__name__

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.kernel_size = conv_layer.kernel_size
        self.bias = bias
        self.dilation = dilation

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class VisionTransformer(nn.Module):
    """
    This class defines the `Vision Transformer architecture <https://arxiv.org/abs/2010.11929>`_. Our model implementation
    is inspired from `Early Convolutions Help Transformers See Better <https://arxiv.org/abs/2106.14881>`_

    .. note::
        Our implementation is different from the original implementation in two ways:
        1. Kernel size is odd.
        2. Our positional encoding implementation allows us to use ViT with any multiple input scales
        3. We do not use StochasticDepth
        4. We do not add positional encoding to class token (if enabled), as suggested in `DeiT-3 paper <https://arxiv.org/abs/2204.07118>`_
    """

    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__()
        image_channels = 3
        num_classes = cfg.get("n_classes", 1000)

        self.projection_dim = None
        if "projection_dim" in kwargs:
            self.projection_dim = kwargs.get("projection_dim")

        kernel_sizes_conv_stem = [4, 2, 2]
        strides_conv_stem = [4, 2, 2]

        # Typically, in the ImageNet dataset, we use 224x224 as a resolution.
        # For out ViT implementation, patch size is 16 (16 = 4 * 2 * 2)
        # Therefore, total number of embeddings along width and height are (224 / 16)^2
        num_embeddings = (224 // 16) ** 2

        embed_dim = cfg["embed_dim"]
        ffn_dim = cfg["embed_dim"] * 4
        pos_emb_drop_p = cfg.get("pos_emb_drop_p", 0.0)
        n_transformer_layers = cfg["n_transformer_layers"]
        num_heads = cfg["n_attn_heads"]
        attn_dropout = cfg.get("attn_dropout", 0.0)
        dropout = cfg.get("dropout", 0.0)
        ffn_dropout = cfg.get("ffn_dropout", 0.0)
        norm_layer = cfg.get("norm_layer", "layer_norm")

        conv_stem_proj_dim = max(32, embed_dim // 4)
        patch_emb = [
            ConvNormAct(
                cfg=cfg,
                in_channels=image_channels,
                out_channels=conv_stem_proj_dim,
                kernel_size=kernel_sizes_conv_stem[0],
                stride=strides_conv_stem[0],
                bias=False,
                use_norm=True,
                use_act=True,
            ),
            ConvNormAct(
                cfg=cfg,
                in_channels=conv_stem_proj_dim,
                out_channels=conv_stem_proj_dim,
                kernel_size=kernel_sizes_conv_stem[1],
                stride=strides_conv_stem[1],
                bias=False,
                use_norm=True,
                use_act=True,
            ),
            ConvNormAct(
                cfg=cfg,
                in_channels=conv_stem_proj_dim,
                out_channels=embed_dim,
                kernel_size=kernel_sizes_conv_stem[2],
                stride=strides_conv_stem[2],
                bias=True,
                use_norm=False,
                use_act=False,
            ),
        ]

        self.patch_emb = nn.Sequential(*patch_emb)

        use_cls_token = not cfg.get("no_cls_token", False)
        stochastic_dropout = cfg.get("stochastic_dropout", 0.0)
        per_layer_stochastic_drop_rate = [
            round(x, 3)
            for x in np.linspace(0, stochastic_dropout, n_transformer_layers)
        ]
        transformer_blocks = [
            TransformerEncoder(
                embed_dim=embed_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                transformer_norm_layer=norm_layer,
                stochastic_dropout=per_layer_stochastic_drop_rate[layer_idx],
            )
            for layer_idx in range(n_transformer_layers)
        ]

        self.post_transformer_norm = get_normalization_layer(
            num_features=embed_dim, norm_type=norm_layer
        )

        self.transformer = nn.Sequential(*transformer_blocks)

        if self.projection_dim is None:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = SimpleImageProjectionHead(embed_dim, self.projection_dim)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(size=(1, 1, embed_dim)))
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        self.pos_embed = PositionalEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embed_dim,
            padding_idx=None,
            interpolation_mode="bilinear",
        )
        self.emb_dropout = nn.Dropout(p=pos_emb_drop_p)

    def extract_patch_embeddings(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        # input is of shape [Batch, in_channels, height, width]. in_channels is mostly 3 (for RGB images)
        batch_size = x.shape[0]

        # [Batch, in_channels, height, width] --> [Batch, emb_dim, num_patches_height, num_patches_width]
        patch_emb = self.patch_emb(x)
        n_h, n_w = patch_emb.shape[-2:]

        # [Batch, emb_dim, num_patches_height, num_patches_width] --> [Batch, emb_dim, num_patches]
        patch_emb = patch_emb.flatten(2)
        # [Batch, emb_dim, num_patches] --> [Batch, num_patches, emb_dim]
        patch_emb = patch_emb.transpose(1, 2).contiguous()

        n_patches = patch_emb.shape[1]
        # we resize the positional encodings dynamically.
        pos_emb = self.pos_embed(n_patches).to(patch_emb.dtype)

        # add positional encodings
        patch_emb = pos_emb + patch_emb

        # add classification token
        if self.cls_token is not None:
            # [1, 1, emb_dim] --> [Batch, 1, emb_dim]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            # Concat([Batch, 1, emb_dim], [Batch, num_patches, emb_dim]) --> [Batch, num_patches + 1, emb_dim]
            patch_emb = torch.cat((cls_tokens, patch_emb), dim=1)

        # dropout
        patch_emb = self.emb_dropout(patch_emb)
        return patch_emb, (n_h, n_w)

    def _features_from_transformer(
        self, x: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, Tuple[int, int]]:
        # this function extract patch embeddings and then apply transformer module to learn
        # inter-patch representations

        # [B, N, C] --> [N, B, embed_dim], where B is batch size, N is number of tokens,
        # and embed_dim is feature dim
        x, (n_h, n_w) = self.extract_patch_embeddings(x)

        for layer in self.transformer:
            x = layer(x)
        x = self.post_transformer_norm(x)

        return x, (n_h, n_w)

    def extract_features(
        self, x: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # The extract_features function for ViT returns two outputs: (1) embedding corresponding to CLS token
        # and (2) image embeddings of the shape [B, C, h//o, w//o], where the value of o is typically 16.
        return_image_embeddings = kwargs.get("return_image_embeddings", False)

        # [B, C, H, W] --> [B, N + 1, embed_dim] or [B, N, embed_dim]
        # here, B is batch size, C is input channels
        # H and W are input height and width
        # N is the number of pixels (or tokens) after processing input with conv stem and reshaping
        # We add +1 for cls token (if applicable)
        # embed_dim --> embedding dimension
        x, (n_h, n_w) = self._features_from_transformer(x, *args, **kwargs)

        if self.cls_token is not None:
            # [B, N + 1, embed_dim] --> [B, embed_dim], [B, N, embed_dim]
            cls_embedding, image_embedding = torch.split(
                x, split_size_or_sections=[1, x.shape[1] - 1], dim=1
            )
            cls_embedding = cls_embedding.squeeze(1)
        else:
            # [B, N, embed_dim] -> [B, embed_dim]
            cls_embedding = torch.mean(x, dim=1)
            # [B, N, embed_dim]
            image_embedding = x

        if return_image_embeddings:
            # reshape image embedding to 4-D tensor
            # [B, N, C] --> [B, C, N]
            image_embedding = image_embedding.transpose(1, 2).contiguous()
            image_embedding = image_embedding.reshape(
                image_embedding.shape[0], -1, n_h, n_w
            )

            return cls_embedding, image_embedding
        else:
            return cls_embedding, None

    def forward_classifier(self, x: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        cls_embedding, image_embedding = self.extract_features(x, *args, **kwargs)
        # classify based on CLS token
        cls_embedding = self.classifier(cls_embedding)
        return cls_embedding, image_embedding

    def forward(self, x: Tensor, *args, **kwargs) -> Union[Tensor, Dict[str, Tensor]]:
        # In ViT model, we can return either classifier embeddings (logits) or image embeddings or both.
        # To return the image embeddings, we need to set keyword argument (return_image_embeddings) as True.
        if kwargs.get("return_image_embeddings", False):
            out_dict = dict()
            prediction, image_embedding = self.forward_classifier(x, *args, **kwargs)
            out_dict.update({"logits": prediction})
            if image_embedding is not None:
                out_dict.update({"image_embeddings": image_embedding})
            return out_dict
        else:
            prediction, _ = self.forward_classifier(x, *args, **kwargs)
            return prediction


@register_model
def vit_b16(pretrained=False, **kwargs):
    # Vision transformer config
    cfg = {
        "norm_layer": "layer_norm_fp32",
        "act_layer": "gelu",
        "embed_dim": 768,
        "n_transformer_layers": 12,
        "n_attn_heads": 12,
    }
    model = VisionTransformer(cfg=cfg, **kwargs)
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model
