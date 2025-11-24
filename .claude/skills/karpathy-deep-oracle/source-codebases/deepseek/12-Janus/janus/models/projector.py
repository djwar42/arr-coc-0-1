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
# MlpProjector.__init__(cfg) - Initialize projection layers based on config type
# MlpProjector.forward(x_or_tuple) - Project vision features to language model embedding space
#
# ** Technical Review **
# This module implements the vision-to-language alignment layer ("aligner") in Janus's understanding pathway.
# After the vision encoder (CLIP/SigLIP) extracts image features, the projector transforms them into the
# language model's embedding space, enabling seamless fusion with text tokens.
#
# **Supported Projector Types**:
# 1. **identity**: Pass-through (no transformation) - used when vision and language dims match
# 2. **linear**: Single linear layer - simplest alignment, input_dim → n_embed
# 3. **mlp_gelu**: Multi-layer perceptron with GELU activations - adds expressiveness
#    - Depth configurable (typically 1-2 layers)
#    - Each layer: Linear(n_embed, n_embed) + GELU
#    - Enables non-linear feature transformation
# 4. **low_high_hybrid_split_mlp_gelu**: Dual-stream for hybrid vision encoders
#    - Processes high-resolution and low-resolution features separately
#    - high_up_proj: projects high-res features → n_embed/2
#    - low_up_proj: projects low-res features → n_embed/2
#    - Concatenates both streams → full n_embed dimension
#    - Then applies shared MLP layers for joint refinement
#
# **Architecture Integration**:
# In Janus understanding pathway: CLIP vision encoder → MlpProjector → language model
# - Vision encoder outputs: [batch, num_patches, vision_dim]
# - Projector transforms to: [batch, num_patches, n_embed]
# - These visual embeddings replace <image_placeholder> tokens in text sequence
#
# **Design Rationale**:
# - MLPs learn optimal alignment between vision and language spaces during training
# - GELU activations provide smooth gradients (better than ReLU for transformers)
# - Hybrid split projector exploits multi-resolution visual information
# - Depth parameter allows trading off complexity vs parameter count
#
# **Typical Configuration**:
# - input_dim: 1024 (SigLIP output), 768 (CLIP ViT-Base), or 1536 (CLIP ViT-Large)
# - n_embed: 2048 (Llama-2-7B), 4096 (Llama-2-13B)
# - depth: 1-2 layers (deeper doesn't always help)
# - projector_type: mlp_gelu (most common, good balance)
#
# The projector is a learned bridge that aligns frozen or fine-tuned vision encoders with the
# language model's semantic space, critical for multimodal understanding quality.
# </claudes_code_comments>

from typing import Tuple, Union

import torch
import torch.nn as nn
from attrdict import AttrDict


class MlpProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        if cfg.projector_type == "identity":
            modules = nn.Identity()

        elif cfg.projector_type == "linear":
            modules = nn.Linear(cfg.input_dim, cfg.n_embed)

        elif cfg.projector_type == "mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            modules = [nn.Linear(cfg.input_dim, cfg.n_embed)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "low_high_hybrid_split_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            self.high_up_proj = nn.Linear(cfg.input_dim, cfg.n_embed // 2)
            self.low_up_proj = nn.Linear(cfg.input_dim, cfg.n_embed // 2)

            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        else:
            raise ValueError(f"Unknown projector type: {cfg.projector_type}")

        self.layers = modules

    def forward(
        self, x_or_tuple: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
    ):
        """

        Args:
            x_or_tuple (Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:  if it is a tuple of torch.Tensor,
                then it comes from the hybrid vision encoder, and x = high_res_x, low_res_x);
                otherwise it is the feature from the single vision encoder.

        Returns:
            x (torch.Tensor): [b, s, c]
        """

        if isinstance(x_or_tuple, tuple):
            # self.cfg.projector_type == "low_high_hybrid_split_mlp_gelu":
            high_x, low_x = x_or_tuple
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = torch.concat([high_x, low_x], dim=-1)
        else:
            x = x_or_tuple

        return self.layers(x)


if __name__ == "__main__":
    cfg = AttrDict(
        input_dim=1024,
        n_embed=2048,
        depth=2,
        projector_type="low_high_hybrid_split_mlp_gelu",
    )
    inputs = (torch.rand(4, 576, 1024), torch.rand(4, 576, 1024))

    m = MlpProjector(cfg)
    out = m(inputs)
    print(out.shape)
