# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# <claudes_code_comments>
# ** Function List **
# CLIPVisionTower.__init__(...) - Initialize vision encoder with SigLIP or SAM or HuggingFace CLIP
# CLIPVisionTower.build_vision_tower(params) - Construct appropriate vision tower based on model name
# CLIPVisionTower.feature_select(outputs) - Extract desired layer features from vision model output
# CLIPVisionTower.forward(images) - Encode images to patch embeddings
#
# ** Technical Review **
# This module provides a unified interface for loading and using various vision encoders
# (SigLIP, SAM, CLIP) in Janus's understanding pathway. It abstracts away model-specific
# details, presenting a consistent API regardless of underlying architecture.
#
# **Supported Vision Models**:
# 1. **SigLIP** (Sigmoid Loss for Language-Image Pre-training):
#    - Default choice for Janus
#    - Models: siglip_large_patch16_384, siglip_so400m_patch14_384
#    - Trained with sigmoid loss instead of softmax (better scaling)
#    - No CLS token, uses attention pooling ("map" global_pool)
#    - Output: [batch, num_patches, hidden_dim]
#
# 2. **SAM** (Segment Anything Model):
#    - Vision encoder from Meta's SAM model
#    - Primarily designed for segmentation, repurposed for understanding
#    - Strong spatial feature extraction capabilities
#
# 3. **HuggingFace CLIP**:
#    - Standard OpenAI CLIP models
#    - Includes CLS token at position 0
#    - Feature selection options: "patch" (no CLS), "cls_patch" (with CLS), "same" (all)
#
# **Feature Selection Strategy**:
# - `select_layer`: Which transformer layer to extract features from (default: -2, second-to-last)
# - `select_feature`: What tokens to use ("patch", "cls_patch", "same")
# - Intermediate layer features often generalize better than final layer
#
# **Architecture Integration**:
# In Janus understanding pathway:
#   images [b, 3, H, W] → CLIPVisionTower → features [b, num_patches, hidden_dim] → MlpProjector → LLM embeddings
#
# **Image Normalization**:
# - Optional pixel normalization with configurable mean/std
# - If pixel_mean/std provided, applies before vision model forward pass
# - Enables consistent preprocessing across different vision encoders
#
# **Design Rationale**:
# - Abstraction layer decouples model choice from downstream components
# - Easy to swap vision encoders by changing model_name parameter
# - Consistent output format [b, num_patches, d] regardless of encoder type
# - Supports frozen encoders (typical) or fine-tunable encoders
#
# **Typical Configuration**:
# - model_name: "siglip_large_patch16_384" (Janus default)
# - image_size: 384 (higher res = more patches = better detail)
# - select_layer: -2 (penultimate layer, best balance)
# - select_feature: "same" (all output tokens for SigLIP)
#
# The vision tower is the "eyes" of Janus, converting pixels into semantic features
# that the language model can process. Choice of encoder significantly impacts
# understanding quality, with SigLIP providing state-of-the-art vision-language alignment.
# </claudes_code_comments>

from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms
from einops import rearrange

from janus.models.siglip_vit import create_siglip_vit


class CLIPVisionTower(nn.Module):
    def __init__(
        self,
        model_name: str = "siglip_large_patch16_384",
        image_size: Union[Tuple[int, int], int] = 336,
        select_feature: str = "patch",
        select_layer: int = -2,
        select_layers: list = None,
        ckpt_path: str = "",
        pixel_mean: Optional[List[float]] = None,
        pixel_std: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__()

        self.model_name = model_name
        self.select_feature = select_feature
        self.select_layer = select_layer
        self.select_layers = select_layers

        vision_tower_params = {
            "model_name": model_name,
            "image_size": image_size,
            "ckpt_path": ckpt_path,
            "select_layer": select_layer,
        }
        vision_tower_params.update(kwargs)
        self.vision_tower, self.forward_kwargs = self.build_vision_tower(
            vision_tower_params
        )

        if pixel_mean is not None and pixel_std is not None:
            image_norm = torchvision.transforms.Normalize(
                mean=pixel_mean, std=pixel_std
            )
        else:
            image_norm = None

        self.image_norm = image_norm

    def build_vision_tower(self, vision_tower_params):
        if self.model_name.startswith("siglip"):
            self.select_feature = "same"
            vision_tower = create_siglip_vit(**vision_tower_params)
            forward_kwargs = dict()

        elif self.model_name.startswith("sam"):
            vision_tower = create_sam_vit(**vision_tower_params)
            forward_kwargs = dict()

        else:  # huggingface
            from transformers import CLIPVisionModel

            vision_tower = CLIPVisionModel.from_pretrained(**vision_tower_params)
            forward_kwargs = dict(output_hidden_states=True)

        return vision_tower, forward_kwargs

    def feature_select(self, image_forward_outs):
        if isinstance(image_forward_outs, torch.Tensor):
            # the output has been the self.select_layer"s features
            image_features = image_forward_outs
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]

        if self.select_feature == "patch":
            # if the output has cls_token
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        elif self.select_feature == "same":
            image_features = image_features

        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def forward(self, images):
        """

        Args:
            images (torch.Tensor): [b, 3, H, W]

        Returns:
            image_features (torch.Tensor): [b, n_patch, d]
        """

        if self.image_norm is not None:
            images = self.image_norm(images)

        image_forward_outs = self.vision_tower(images, **self.forward_kwargs)
        image_features = self.feature_select(image_forward_outs)
        return image_features
