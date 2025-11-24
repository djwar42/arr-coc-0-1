# <claudes_code_comments>
# ** Function List **
# MlpProjector.__init__(cfg) - initializes projection layer based on config projector_type
# MlpProjector.forward(x) - projects vision features to language model dimension
# MlpProjector.get_flops_per_sample(cfg) - calculates FLOPs for projector layer
#
# ** Technical Review **
# Multi-Layer Perceptron projector that maps vision encoder outputs to language model input dimension.
# Bridges the 2048-dim hybrid vision features (SAM + CLIP) to 1280-dim language model space.
#
# Supported Projector Types:
# 1. identity: No-op passthrough (input_dim must equal n_embed)
# 2. linear: Single linear layer (2048→1280)
# 3. mlp_gelu: Multi-layer MLP with GELU activation (configurable depth)
# 4. downsample_mlp_gelu: Spatial downsampling via unfold before MLP
# 5. normlayer_downsample_mlp_gelu: LayerNorm + downsampling + MLP
# 6. low_high_hybrid_split_mlp_gelu: Separate projections for low/high-res features
# 7. hybrid_split_feature_mlp_gelu: Split features by channel dimension
# 8. low_high_split_mlp_gelu: Independent MLP paths for low/high features
#
# DeepSeek-OCR Configuration:
# Uses projector_type="linear" with input_dim=2048, n_embed=1280
# - Input: 2048-dim hybrid features [CLIP_1024 + SAM_1024]
# - Output: 1280-dim embeddings for DeepSeek-V2/V3 language model
# - Simple linear projection chosen for efficiency (inference-optimized)
#
# Downsampling Projectors (types 4-5):
# Apply spatial token reduction before projection to compress vision sequence:
# - Uses F.unfold(kernel_size=downsample_ratio, stride=downsample_ratio)
# - Concatenates neighboring patches: 4 tokens → 1 token with 4× channels
# - Example: (B, 256, 2048) → (B, 64, 8192) after 2× downsample
# - Then project: 8192 → 1280 to match language model
# - Reduces sequence length for memory efficiency in long documents
#
# Hybrid Split Projectors (types 6-8):
# Process low-resolution (global) and high-resolution (crop) features separately:
# - Useful when global context needs different transformation than local details
# - low_high_hybrid_split: concatenates after separate up-projections
# - hybrid_split_feature: splits input by channel dimension before projection
# - low_high_split: completely independent MLP paths that concatenate outputs
#
# Token Pooling (optional):
# When cfg.token_pooling=True, applies 2×2 spatial pooling before projection:
# - Reshapes features: (B, H×W, C) → (B, H, W, C)
# - Unfolds into 2×2 patches: (B, C, H/2, W/2, 4)
# - Concatenates: (B, H×W/4, 4C)
# - Linear: 4C → C to merge pooled features
# - Reduces sequence length 4× while preserving information
#
# FLOPs Calculation (get_flops_per_sample):
# Estimates computational cost for projector forward pass:
# - Linear: 2 × input_dim × n_embed (matrix multiply FLOPs)
# - MLP: 2 × input_dim × n_embed + (depth-1) × 2 × n_embed²
# - Returns 3× for forward+backward+weight_grad during training
# - Used for model profiling and optimization analysis
#
# Design Rationale:
# Simple linear projection chosen for DeepSeek-OCR because:
# - Vision encoders (SAM + CLIP) already highly capable
# - 2048→1280 linear layer sufficient to align feature spaces
# - Minimal parameters (~2.6M) keeps model efficient
# - Inference speed critical for high-throughput OCR
# More complex projectors (MLP, downsampling) available for different
# vision-language architectures or when additional capacity needed
# </claudes_code_comments>

import torch.nn as nn
import torch
import torch.nn.functional as F
import copy


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
        
        elif cfg.projector_type == "normlayer_downsample_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            mlp_ratio = cfg.get("mlp_ratio", 1)
            modules = [
                nn.LayerNorm(cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio),
                nn.Linear(cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio, cfg.n_embed * mlp_ratio)
            ]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed * mlp_ratio))
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed))
            modules = nn.Sequential(*modules)
        
        elif cfg.projector_type == "downsample_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            mlp_ratio = cfg.get("mlp_ratio", 1)
            modules = [nn.Linear(cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio, cfg.n_embed * mlp_ratio)]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed * mlp_ratio))
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed))
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

        elif cfg.projector_type == "hybrid_split_feature_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            channel_div = cfg.get("channel_div", 0.5)
            self.high_up_proj = nn.Linear(cfg.input_dim[0], int(cfg.n_embed * channel_div))
            self.low_up_proj = nn.Linear(cfg.input_dim[1], cfg.n_embed - int(cfg.n_embed * channel_div))

            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "low_high_split_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed // 2, cfg.n_embed // 2))
            modules = nn.Sequential(*modules)
            self.high_layers = nn.Sequential(*modules)
            self.low_layers = copy.deepcopy(modules)

        else:
            raise ValueError(f"Unknown projector type: {cfg.projector_type}")

        if cfg.get("token_pooling", False):
            self.token_pooling_layer = nn.Linear(cfg.input_dim * 4, cfg.input_dim)

        if cfg.get("conv_fusion_high_low_features", False):
            self.fusion_layer = nn.Linear(cfg.input_dim, cfg.input_dim)
        self.layers = modules

    def forward(self, x):
        if self.cfg.get("token_pooling", False):
            batch_size, wxh, channels = x.shape
            w = h = int(wxh**0.5)
            x = x.view(batch_size, w, h, channels)
            x = x.permute(0, 3, 1, 2)
            # import ipdb; ipdb.set_trace()
            patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
            batch_size, channels, h_patches, w_patches, _, _ = patches.size()
            # 在通道维度上拼接
            patches = patches.contiguous().view(batch_size, channels, h_patches * w_patches, -1)

            # 通过线性层
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(batch_size, h_patches * w_patches, channels * 4)

            x = self.token_pooling_layer(patches)
        
        if self.cfg.get("conv_fusion_high_low_features", False):
            x = self.fusion_layer(x[:, 0]) + x[:, 1]

        if self.cfg.projector_type == 'low_high_hybrid_split_mlp_gelu':
            high_x, low_x = x[0], x[1]
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = torch.concat([high_x, low_x], dim=-1)
        
        if self.cfg.projector_type == 'hybrid_split_feature_mlp_gelu':
            high_x = x[...,:self.cfg.input_dim[0]]
            low_x = x[...,self.cfg.input_dim[0]:]
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = torch.concat([high_x, low_x], dim=-1)
        
        if self.cfg.projector_type == 'low_high_split_mlp_gelu':
            high_x, low_x = x[0], x[1]
            high_x = self.high_layers(high_x)
            low_x = self.low_layers(low_x)
            x = torch.concat([high_x, low_x], dim=-1)
            return x
        
        if self.cfg.projector_type == 'downsample_mlp_gelu' or self.cfg.projector_type == 'normlayer_downsample_mlp_gelu':
            bs, hw, input_dim = x.shape
            h = w = int((hw) ** 0.5)

            """compute padding"""
            if h % self.cfg.downsample_ratio:
                pad = self.cfg.downsample_ratio - h % self.cfg.downsample_ratio
            else:
                pad = 0
            x = x.reshape(bs, h, w, input_dim)
            if pad > 0:
                x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)

            """4 to 1 concat"""
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            x = F.unfold(x, kernel_size=self.cfg.downsample_ratio, stride=self.cfg.downsample_ratio, padding=0) # B, C*4, HW // 4
            x = x.permute(0, 2, 1)
            
        return self.layers(x)

    @staticmethod
    def get_flops_per_sample(cfg):
        if cfg.projector_type == "linear":
            fwd = 2 * cfg.input_dim * cfg.n_embed

        elif "mlp_gelu" in cfg.projector_type :
            mlp_depth = cfg.get("depth", 1)
            downsample_ratio = cfg.get("downsample_ratio", 1)
            input_dim = sum(cfg.input_dim) if isinstance(cfg.input_dim, list) else cfg.input_dim
            input_dim = input_dim * downsample_ratio * downsample_ratio
            fwd = 2 * input_dim * cfg.n_embed + (mlp_depth - 1) * 2 * cfg.n_embed * cfg.n_embed
        else:
            fwd = 0

        return fwd * 3


