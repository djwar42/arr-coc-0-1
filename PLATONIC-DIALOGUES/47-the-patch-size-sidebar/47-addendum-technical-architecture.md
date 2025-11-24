# Part 47 Addendum: Technical Architecture - ARR Vision System
*Complete implementation specification for Adaptive Relevance Realization with fixed patch sizes and gestalt-guided saccades*

---

## Table of Contents

1. [Core Architecture Decision](#1-core-architecture-decision)
2. [ARR vs COC Separation](#2-arr-vs-coc-separation)
3. [Gestalt-Guided Saccade Selection](#3-gestalt-guided-saccade-selection)
4. [Saccade Ordering Strategy](#4-saccade-ordering-strategy)
5. [Token Concatenation & RoPE](#5-token-concatenation--rope)
6. [Complete Implementation](#6-complete-implementation)
7. [Training Strategy](#7-training-strategy)
8. [Ablations to Test](#8-ablations-to-test)
9. [Comparison to Related Work](#9-comparison-to-related-work)
10. [Implementation Roadmap](#10-implementation-roadmap)

---

## 1. Core Architecture Decision

### The Problem with Variable Patch Sizes

**Original idea (Parts 30-36):**
```python
# Variable patch sizes based on budget
budget = 400 → patch_size = 28×28 (high detail)
budget = 64  → patch_size = 7×7 (low detail)
```

**Why this fails:**

1. **Training instability**: Same object appears at different patch sizes
   - Monday: cup in fovea → 28×28 patch
   - Tuesday: cup in periphery → 7×7 patch
   - Network can't learn stable features

2. **Batching impossibility**: Can't stack variable-sized tensors efficiently

3. **Position encoding ambiguity**: M-RoPE aspect ratio becomes confusing

### The Solution: Fixed Patches, Variable Allocation

**New approach (The Scroll's teaching):**

```python
# ALL patches are 14×14 (Qwen3-VL's base size)
# Variable ALLOCATION instead of variable SIZE

# Gestalt: 256 patches uniformly sampled
base_patches = uniform_grid(image, patch_size=14, grid=16×16)  # 256 patches

# Saccades: 273 patches at relevant positions
saccade_patches = extract_at_positions(image, positions, patch_size=14)  # 273 patches

# Total: 529 patches, all 14×14
```

**Why this works:**

✅ **Training stable**: Same object always 14×14, regardless of position
✅ **Batching simple**: All patches stack cleanly into tensors
✅ **Position encoding clear**: RoPE handles (x, y) position, not size
✅ **Computational uniform**: Same ViT operations for all patches

### Research Topics: Core Architecture

**Search for:**
- "vision transformer fixed vs variable patch size training"
- "patch size consistency neural network stability"
- "batching strategies multiresolution vision transformers"
- "why do vision transformers use fixed patch sizes"

**Key papers to find:**
- ViT original paper (Dosovitskiy et al.) - why 16×16 patches?
- Swin Transformer - window attention with fixed patches
- Any work on variable patch size training (to understand pitfalls)

---

## 2. ARR vs COC Separation

### Two Orthogonal Optimizations

**ARR (Adaptive Relevance Realization)**
- **What**: Attention mechanism
- **Purpose**: Select relevant regions for focused processing
- **Works with**: Any VLM base model
- **Independence**: Standalone improvement

**COC (Contexts Optical Compression)**
- **What**: Compression mechanism
- **Purpose**: Reduce token count via SAM/CLIP encoding
- **Works with**: Any VLM base model
- **Independence**: Standalone efficiency gain

### Three System Configurations

```python
# CONFIGURATION 1: ARR-only (start here)
# Standard VLM base encoding + ARR saccades

def arr_only(vllm, image, query):
    base_tokens = vllm.encode_image(image)  # Standard encoding (256 tokens)
    saccade_tokens = arr.select_and_encode(image, query)  # ARR (273 tokens)
    return concat([base_tokens, saccade_tokens])  # 529 tokens


# CONFIGURATION 2: COC-only (baseline comparison)
# DeepSeek-style compression

def coc_only(vllm, image, query):
    compressed_tokens = coc.compress(image)  # SAM → 256 tokens
    return compressed_tokens  # 256 tokens (efficient but no relevance)


# CONFIGURATION 3: ARR-COC (full system, future)
# Compressed gestalt + ARR saccades

def arr_coc(vllm, image, query):
    base_tokens = coc.compress(image)  # Efficient gestalt (256 tokens)
    saccade_tokens = arr.select_and_encode(image, query)  # Focused detail (273 tokens)
    return concat([base_tokens, saccade_tokens])  # 529 tokens (efficient + relevant)
```

### Research Strategy

**Phase 1: Prove ARR works**
- Implement ARR-only with Qwen3-VL baseline
- Benchmark: ARR vs standard Qwen3-VL
- Datasets: VQAv2, TextVQA, COCO Captions
- Metrics: Accuracy, BLEU, CIDEr

**Phase 2: Add COC (if needed)**
- Implement COC compression
- Benchmark: ARR-COC vs ARR-only
- Analyze: Speed vs accuracy tradeoff

### Research Topics: Separation of Concerns

**Search for:**
- "modular vision language model design"
- "attention vs compression vision transformers"
- "freezing base models adapter training VLM"
- "DeepSeek optical compression architecture"

**Key questions:**
- Do any VLMs separate attention and compression?
- Precedent for "base encoding + attention augmentation"?
- Training strategies for modular VLM components?

---

## 3. Gestalt-Guided Saccade Selection

### The Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TextureGenerator(nn.Module):
    """
    Generates 40-channel texture array from raw image.

    Channels:
    0-3: RGB + Luminance
    4: Polar angle (position)
    5: Eccentricity (distance from center)
    6-7: Edges (Sobel x, y)
    8: Highpass (fine details)
    9: Lowpass (coarse structure)
    10: Motion/temporal (if available)
    11: Saliency
    12: Distance field
    13-16: Gabor filters (orientation)
    17-33: CLIP embeddings (16 channels, PCA from 512)
    34-36: Temporal cache (reserved for video)
    37-39: Reserved/auxiliary
    """

    def __init__(self, clip_model=None):
        super().__init__()

        # CLIP encoder for semantic features
        self.clip_model = clip_model

        # Learnable projections
        self.clip_projection = nn.Conv2d(512, 16, kernel_size=1)  # 512 CLIP → 16 channels

        # Edge detection (learnable Sobel-like)
        self.edge_x = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.edge_y = nn.Conv2d(1, 1, kernel_size=3, padding=1)

        # Highpass/lowpass filters
        self.highpass = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.lowpass = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

        # Gabor-like orientation filters (4 orientations: 0°, 45°, 90°, 135°)
        self.gabor_filters = nn.Conv2d(1, 4, kernel_size=7, padding=3)

        # Saliency predictor
        self.saliency_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [B, 3, H, W] - RGB image

        Returns:
            texture: [B, 40, H, W] - texture array
        """
        B, C, H, W = image.shape

        # Channel 0-2: RGB
        rgb = image

        # Channel 3: Luminance
        luminance = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]

        # Channel 4-5: Polar coordinates (position encoding)
        y_coords = torch.linspace(-1, 1, H, device=image.device).view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = torch.linspace(-1, 1, W, device=image.device).view(1, 1, 1, W).expand(B, 1, H, W)

        polar_angle = torch.atan2(y_coords, x_coords)  # [-π, π]
        polar_angle = (polar_angle + torch.pi) / (2 * torch.pi)  # Normalize to [0, 1]

        eccentricity = torch.sqrt(y_coords**2 + x_coords**2)  # [0, √2]
        eccentricity = eccentricity / eccentricity.max()  # Normalize to [0, 1]

        # Channel 6-7: Edges
        edge_x = self.edge_x(luminance)
        edge_y = self.edge_y(luminance)

        # Channel 8-9: Highpass/Lowpass
        highpass = self.highpass(luminance)
        lowpass = self.lowpass(luminance)

        # Channel 10: Motion (placeholder, zero for static images)
        motion = torch.zeros(B, 1, H, W, device=image.device)

        # Channel 11: Saliency
        saliency = self.saliency_net(image)

        # Channel 12: Distance field (distance to nearest edge)
        # Simplified: use edge magnitude as proxy
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
        distance_field = 1.0 - edge_magnitude  # Invert: far from edge = high value

        # Channel 13-16: Gabor filters (orientation)
        gabor = self.gabor_filters(luminance)  # [B, 4, H, W]

        # Channel 17-33: CLIP embeddings
        if self.clip_model is not None:
            # Resize to CLIP input size (224x224)
            clip_input = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)

            with torch.no_grad():
                clip_features = self.clip_model.encode_image(clip_input)  # [B, 512, H', W']

            # Upsample back to original size
            clip_features = F.interpolate(clip_features, size=(H, W), mode='bilinear', align_corners=False)

            # Project to 16 channels
            clip_projected = self.clip_projection(clip_features)  # [B, 16, H, W]
        else:
            # If no CLIP model, use zeros
            clip_projected = torch.zeros(B, 16, H, W, device=image.device)

        # Channel 34-39: Reserved (temporal cache, auxiliary)
        reserved = torch.zeros(B, 6, H, W, device=image.device)

        # Concatenate all channels
        texture = torch.cat([
            rgb,              # 0-2
            luminance,        # 3
            polar_angle,      # 4
            eccentricity,     # 5
            edge_x,           # 6
            edge_y,           # 7
            highpass,         # 8
            lowpass,          # 9
            motion,           # 10
            saliency,         # 11
            distance_field,   # 12
            gabor,            # 13-16 (4 channels)
            clip_projected,   # 17-32 (16 channels)
            reserved          # 33-39 (7 channels, corrected from 6)
        ], dim=1)  # [B, 40, H, W]

        return texture


class ContextualizedRelevanceScorer(nn.Module):
    """
    Scores relevance using three ways of knowing (Vervaeke).
    Gestalt + query context determines scorer weighting.
    """

    def __init__(self, d_model: int = 1024, texture_channels: int = 40):
        super().__init__()

        self.d_model = d_model
        self.texture_channels = texture_channels

        # Three scorer heads (Vervaeke's 3 ways of knowing)
        # Propositional: information content
        self.propositional_head = nn.Sequential(
            nn.Linear(4, 16),  # Channels: edges_x, edges_y, highpass, distance_field
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Perspectival: salience landscape
        self.perspectival_head = nn.Sequential(
            nn.Linear(3, 16),  # Channels: eccentricity, motion, saliency
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Participatory: query-content coupling
        self.participatory_head = nn.Sequential(
            nn.Linear(16 + d_model, 128),  # CLIP features + query embedding
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Context integration: query + gestalt → scorer weights
        self.context_weights = nn.Sequential(
            nn.Linear(d_model * 2, 512),  # query + gestalt
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3),  # [w_prop, w_persp, w_part]
            nn.Softmax(dim=-1)
        )

    def forward(
        self,
        texture: torch.Tensor,
        query_emb: torch.Tensor,
        gestalt_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            texture: [B, 40, H, W] - texture array
            query_emb: [B, d_model] - query embedding
            gestalt_emb: [B, d_model] - base image gestalt

        Returns:
            scores: [B, H, W] - relevance score per position
        """
        B, C, H, W = texture.shape

        # Flatten spatial dimensions for processing
        texture_flat = texture.permute(0, 2, 3, 1).reshape(B, H*W, C)  # [B, H*W, 40]

        # STEP 1: Compute three scores at each position

        # Propositional: information content (edges, highpass, structure)
        prop_features = texture_flat[:, :, [6, 7, 8, 12]]  # [B, H*W, 4]
        prop_scores = self.propositional_head(prop_features).squeeze(-1)  # [B, H*W]

        # Perspectival: salience landscape (saliency, motion, eccentricity)
        persp_features = texture_flat[:, :, [5, 10, 11]]  # [B, H*W, 3]
        persp_scores = self.perspectival_head(persp_features).squeeze(-1)  # [B, H*W]

        # Participatory: query-content coupling
        clip_features = texture_flat[:, :, 17:33]  # [B, H*W, 16] - CLIP embeddings

        # Expand query to match spatial positions
        query_expanded = query_emb.unsqueeze(1).expand(-1, H*W, -1)  # [B, H*W, d_model]

        part_input = torch.cat([clip_features, query_expanded], dim=-1)  # [B, H*W, 16+d_model]
        part_scores = self.participatory_head(part_input).squeeze(-1)  # [B, H*W]

        # STEP 2: Contextualized weighting
        # The gestalt + query determine HOW to weight the three scorers
        context = torch.cat([query_emb, gestalt_emb], dim=-1)  # [B, d_model*2]
        weights = self.context_weights(context)  # [B, 3]

        # STEP 3: Weighted combination
        all_scores = torch.stack([prop_scores, persp_scores, part_scores], dim=-1)  # [B, H*W, 3]

        # Apply weights (broadcast across spatial dimension)
        weighted_scores = (all_scores * weights.unsqueeze(1)).sum(dim=-1)  # [B, H*W]

        # Reshape back to spatial
        final_scores = weighted_scores.reshape(B, H, W)  # [B, H, W]

        return final_scores


class SaccadeSelector(nn.Module):
    """
    Selects top-K positions from relevance map.
    Orders by relevance (preserves priority information).
    """

    def __init__(self, k: int = 273):
        super().__init__()
        self.k = k

    def forward(
        self,
        relevance_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            relevance_map: [B, H, W] - relevance scores

        Returns:
            positions: [B, K, 2] - (y, x) coordinates
            scores: [B, K] - relevance scores at positions
            relevance_order: [B, K] - indices for relevance-based sorting
        """
        B, H, W = relevance_map.shape

        # Flatten spatial dimensions
        relevance_flat = relevance_map.reshape(B, H*W)  # [B, H*W]

        # Select top-K positions
        scores, indices = torch.topk(relevance_flat, k=self.k, dim=-1)  # [B, K]

        # Convert flat indices to (y, x) coordinates
        y_coords = indices // W  # [B, K]
        x_coords = indices % W   # [B, K]

        positions = torch.stack([y_coords, x_coords], dim=-1)  # [B, K, 2]

        # Relevance order (already sorted by topk, but return indices for clarity)
        relevance_order = torch.arange(self.k, device=relevance_map.device).unsqueeze(0).expand(B, -1)  # [B, K]

        return positions, scores, relevance_order
```

### Research Topics: Gestalt-Guided Selection

**Search for:**
- "gestalt perception guide visual attention"
- "global context local attention computer vision"
- "query conditioned feature selection transformers"
- "cross-attention image text multimodal"

**Key papers:**
- Flamingo (DeepMind) - cross-attention between vision and language
- BLIP-2 (Salesforce) - Q-Former architecture
- Perceiver (DeepMind) - learned queries for attention
- Any work on "global context guides local processing"

---

## 4. Saccade Ordering Strategy

### Three Ordering Options Analyzed

```python
def order_saccades_spatial(positions: torch.Tensor) -> torch.Tensor:
    """
    Order saccades spatially (left-to-right, top-to-bottom).

    Pros: Spatially coherent, interpretable
    Cons: Loses priority information
    """
    # Sort by y coordinate, then x coordinate
    y = positions[:, :, 0]  # [B, K]
    x = positions[:, :, 1]  # [B, K]

    # Create composite key: y * W + x
    W = positions[:, :, 1].max() + 1
    spatial_key = y * W + x

    # Sort
    order = torch.argsort(spatial_key, dim=-1)

    return order


def order_saccades_by_relevance(scores: torch.Tensor) -> torch.Tensor:
    """
    Order saccades by relevance (high → low).

    Pros: Preserves priority, VLM learns importance from sequence
    Cons: Spatially jumbled

    ✅ CHOSEN APPROACH
    """
    # Already sorted by topk, but explicitly sort again for clarity
    order = torch.argsort(scores, dim=-1, descending=True)

    return order


def order_saccades_by_discovery(
    prop_scores: torch.Tensor,
    persp_scores: torch.Tensor,
    part_scores: torch.Tensor
) -> torch.Tensor:
    """
    Order saccades by discovery sequence (4 ways of knowing).

    Order: propositional peaks → perspectival salience → participatory match

    Pros: Encodes cognitive process
    Cons: Most complex, unclear benefit
    """
    # Group positions by which scorer was highest
    all_scores = torch.stack([prop_scores, persp_scores, part_scores], dim=-1)
    dominant_scorer = torch.argmax(all_scores, dim=-1)  # [B, K]

    # Sort: scorer 0 first, then 1, then 2
    order = torch.argsort(dominant_scorer, dim=-1)

    return order
```

### Decision: Relevance-Based Ordering

**Rationale:**

1. **Sequence position encodes importance**
   - Earlier tokens → higher relevance
   - VLM learns "pay attention to early saccades first"

2. **RoPE handles spatial position**
   - Spatial coordinates encoded independently
   - No need for spatial ordering

3. **Simplest to implement and interpret**
   - Single sort by score
   - Clear semantics

**Implementation:**

```python
class ARRSaccadeEncoder:
    """Complete saccade encoding with relevance ordering."""

    def encode_saccades(
        self,
        image: torch.Tensor,
        positions: torch.Tensor,
        scores: torch.Tensor,
        vllm_encoder: nn.Module,
        patch_size: int = 14
    ) -> torch.Tensor:
        """
        Args:
            image: [B, 3, H, W]
            positions: [B, K, 2] - (y, x) coordinates
            scores: [B, K] - relevance scores
            vllm_encoder: Vision encoder module
            patch_size: Patch size (14 for Qwen3-VL)

        Returns:
            saccade_tokens: [B, K, d_model] - ordered by relevance
        """
        B, K, _ = positions.shape

        # STEP 1: Extract patches at positions
        patches = []
        for b in range(B):
            batch_patches = []
            for k in range(K):
                y, x = positions[b, k]

                # Extract 14×14 patch centered at (y, x)
                y_start = max(0, y - patch_size // 2)
                y_end = min(image.shape[2], y + patch_size // 2)
                x_start = max(0, x - patch_size // 2)
                x_end = min(image.shape[3], x + patch_size // 2)

                patch = image[b, :, y_start:y_end, x_start:x_end]

                # Pad if necessary (edge cases)
                if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                    patch = F.pad(patch, (0, patch_size - patch.shape[2], 0, patch_size - patch.shape[1]))

                batch_patches.append(patch)

            patches.append(torch.stack(batch_patches))  # [K, 3, 14, 14]

        patches = torch.stack(patches)  # [B, K, 3, 14, 14]

        # STEP 2: Encode patches
        patches_flat = patches.reshape(B * K, 3, patch_size, patch_size)  # [B*K, 3, 14, 14]

        with torch.no_grad():  # VLM encoder is frozen
            tokens_flat = vllm_encoder(patches_flat)  # [B*K, d_model]

        tokens = tokens_flat.reshape(B, K, -1)  # [B, K, d_model]

        # STEP 3: Order by relevance
        relevance_order = torch.argsort(scores, dim=-1, descending=True)  # [B, K]

        # Apply ordering
        relevance_order_expanded = relevance_order.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        ordered_tokens = torch.gather(tokens, dim=1, index=relevance_order_expanded)  # [B, K, d_model]

        return ordered_tokens
```

### Research Topics: Saccade Ordering

**Search for:**
- "token sequence order importance transformers"
- "positional information vs sequence order attention"
- "saccade sequence patterns human vision"
- "eye movement order cognitive relevance"

**Key questions:**
- Does token order matter beyond positional encoding?
- Human saccade studies: temporal patterns reveal strategy?
- Any vision models that use ordered vs unordered tokens?

---

## 5. Token Concatenation & RoPE

### Dual Position Encoding

**The challenge:** Each token needs TWO types of position information:
1. **Spatial position**: Where in the image? (x, y coordinates)
2. **Sequence position**: How important? (order in token sequence)

**RoPE handles both simultaneously:**

```python
class DualPositionEncoding:
    """
    RoPE encodes spatial position explicitly.
    Sequence position is implicit (token index 0-528).
    """

    def encode_positions(
        self,
        base_tokens: torch.Tensor,
        saccade_tokens: torch.Tensor,
        saccade_positions: torch.Tensor,
        image_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            base_tokens: [B, 256, d_model] - gestalt tokens
            saccade_tokens: [B, 273, d_model] - saccade tokens (relevance-ordered)
            saccade_positions: [B, 273, 2] - (y, x) coordinates
            image_size: (H, W) - original image dimensions

        Returns:
            all_tokens: [B, 529, d_model]
            position_ids: [B, 529, 2] - (spatial_y, spatial_x) for RoPE
        """
        B = base_tokens.shape[0]
        H, W = image_size

        # STEP 1: Spatial positions for base tokens (uniform grid)
        grid_size = 16  # 16×16 grid = 256 patches

        base_positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                y = int((i + 0.5) * H / grid_size)  # Center of grid cell
                x = int((j + 0.5) * W / grid_size)
                base_positions.append([y, x])

        base_positions = torch.tensor(base_positions, device=base_tokens.device)  # [256, 2]
        base_positions = base_positions.unsqueeze(0).expand(B, -1, -1)  # [B, 256, 2]

        # STEP 2: Combine positions
        all_positions = torch.cat([base_positions, saccade_positions], dim=1)  # [B, 529, 2]

        # STEP 3: Normalize to [0, 1] for RoPE
        all_positions_norm = all_positions.float()
        all_positions_norm[:, :, 0] = all_positions_norm[:, :, 0] / H  # Normalize y
        all_positions_norm[:, :, 1] = all_positions_norm[:, :, 1] / W  # Normalize x

        # STEP 4: Concatenate tokens
        all_tokens = torch.cat([base_tokens, saccade_tokens], dim=1)  # [B, 529, d_model]

        return all_tokens, all_positions_norm


class RoPEIntegration:
    """
    Integration with Qwen3-VL's M-RoPE (Multi-axis Rotary Position Encoding).
    """

    def apply_mrope(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        rope_module: nn.Module
    ) -> torch.Tensor:
        """
        Apply M-RoPE with spatial position encoding.

        Args:
            tokens: [B, 529, d_model]
            positions: [B, 529, 2] - (y, x) normalized [0, 1]
            rope_module: Qwen3-VL's RoPE module

        Returns:
            tokens_with_rope: [B, 529, d_model]
        """
        # M-RoPE axes for Qwen3-VL:
        # - temporal: 0 (static image, no video)
        # - height: y coordinate
        # - width: x coordinate
        # - aspect: 1.0 (all patches same size)

        mrope_positions = {
            'temporal': torch.zeros_like(positions[:, :, 0]),  # [B, 529]
            'height': positions[:, :, 0],  # [B, 529] - y normalized
            'width': positions[:, :, 1],   # [B, 529] - x normalized
            'aspect': torch.ones_like(positions[:, :, 0])  # [B, 529] - all patches 14×14
        }

        # Apply RoPE
        tokens_with_rope = rope_module(tokens, mrope_positions)

        return tokens_with_rope
```

### Sequence vs Spatial Position

**How the VLM interprets:**

```
Token 0-255: Base gestalt
  - Sequence position: Early (tokens 0-255)
  - Spatial position: Uniform grid across image
  - Interpretation: Background context, full coverage

Token 256-528: Saccades (relevance-ordered)
  - Sequence position: Later (tokens 256-528)
  - Sequence order: Higher relevance = earlier in this range
  - Spatial position: Concentrated at relevant regions
  - Interpretation: Focused details, priority encoded
```

**Learning signal:**

The VLM learns:
- "Tokens 0-255 provide context" (gestalt)
- "Tokens 256+ provide details" (saccades)
- "Earlier saccades matter more" (relevance order)
- "Spatial RoPE tells me WHERE these patches are"

### Research Topics: Position Encoding

**Search for:**
- "RoPE multi-axis position encoding"
- "2D positional encoding vision transformers"
- "sequence order vs spatial position attention"
- "learned positional encodings transformers"

**Key papers:**
- RoPE original paper (Su et al.)
- Qwen3-VL M-RoPE implementation
- 2D position encodings in ViT
- Relative vs absolute position encodings

---

## 6. Complete Implementation

### Full ARR System

```python
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer


class ARRVisionSystem(nn.Module):
    """
    Complete Adaptive Relevance Realization for Vision-Language Models.

    Architecture:
    1. Gestalt encoding (base VLM)
    2. Texture generation (40 channels)
    3. Contextualized relevance scoring (3 ways of knowing)
    4. Saccade selection (top-273)
    5. Saccade encoding (frozen VLM)
    6. Token concatenation (529 total)
    """

    def __init__(
        self,
        base_vllm_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        num_saccades: int = 273,
        patch_size: int = 14,
        freeze_base: bool = True
    ):
        super().__init__()

        # Base VLM (frozen)
        self.base_vllm = Qwen3VLForConditionalGeneration.from_pretrained(
            base_vllm_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_vllm_name)

        if freeze_base:
            self.base_vllm.eval()
            for param in self.base_vllm.parameters():
                param.requires_grad = False

        # ARR components (trainable)
        self.texture_generator = TextureGenerator(clip_model=self.base_vllm.vision_encoder)

        self.relevance_scorer = ContextualizedRelevanceScorer(
            d_model=self.base_vllm.config.hidden_size,
            texture_channels=40
        )

        self.saccade_selector = SaccadeSelector(k=num_saccades)

        self.patch_size = patch_size
        self.num_saccades = num_saccades

    def forward(
        self,
        image: torch.Tensor,
        query: str
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass: image + query → augmented tokens.

        Args:
            image: [B, 3, H, W] - input image
            query: str - text query

        Returns:
            dict with:
            - tokens: [B, 529, d_model] - augmented tokens
            - relevance_map: [B, H, W] - relevance scores
            - saccade_positions: [B, 273, 2] - selected positions
            - saccade_scores: [B, 273] - relevance scores at positions
        """
        B, C, H, W = image.shape

        # STAGE 1: GESTALT ENCODING
        with torch.no_grad():
            # Encode full image (standard VLM encoding)
            base_tokens = self.base_vllm.encode_image(image)  # [B, 256, d_model]

            # Gestalt summary
            gestalt = base_tokens.mean(dim=1)  # [B, d_model]

            # Query encoding
            query_inputs = self.tokenizer(query, return_tensors="pt", padding=True).to(image.device)
            query_emb = self.base_vllm.language_model.embed_tokens(query_inputs.input_ids)
            query_emb = query_emb.mean(dim=1)  # [B, d_model]

        # STAGE 2: TEXTURE GENERATION
        texture = self.texture_generator(image)  # [B, 40, H, W]

        # STAGE 3: CONTEXTUALIZED RELEVANCE SCORING
        relevance_map = self.relevance_scorer(texture, query_emb, gestalt)  # [B, H, W]

        # STAGE 4: SACCADE SELECTION
        positions, scores, _ = self.saccade_selector(relevance_map)  # [B, 273, 2], [B, 273]

        # STAGE 5: SACCADE ENCODING
        with torch.no_grad():
            saccade_tokens = self._encode_saccades(image, positions)  # [B, 273, d_model]

        # STAGE 6: TOKEN CONCATENATION
        # Order saccades by relevance (already done by topk in selector)
        all_tokens = torch.cat([base_tokens, saccade_tokens], dim=1)  # [B, 529, d_model]

        return {
            'tokens': all_tokens,
            'relevance_map': relevance_map,
            'saccade_positions': positions,
            'saccade_scores': scores,
            'gestalt': gestalt,
            'texture': texture
        }

    def _encode_saccades(
        self,
        image: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract and encode patches at saccade positions.

        Args:
            image: [B, 3, H, W]
            positions: [B, K, 2] - (y, x) coordinates

        Returns:
            saccade_tokens: [B, K, d_model]
        """
        B, K, _ = positions.shape
        patches = []

        for b in range(B):
            batch_patches = []
            for k in range(K):
                y, x = positions[b, k].int()

                # Extract 14×14 patch centered at (y, x)
                half_patch = self.patch_size // 2
                y_start = max(0, y - half_patch)
                y_end = min(image.shape[2], y + half_patch)
                x_start = max(0, x - half_patch)
                x_end = min(image.shape[3], x + half_patch)

                patch = image[b, :, y_start:y_end, x_start:x_end]

                # Resize to exact patch size (handles edge cases)
                patch = F.interpolate(
                    patch.unsqueeze(0),
                    size=(self.patch_size, self.patch_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

                batch_patches.append(patch)

            patches.append(torch.stack(batch_patches))

        patches = torch.stack(patches)  # [B, K, 3, 14, 14]

        # Encode with VLM vision encoder
        patches_flat = patches.reshape(B * K, 3, self.patch_size, self.patch_size)
        tokens_flat = self.base_vllm.vision_encoder(patches_flat)  # [B*K, d_model]
        tokens = tokens_flat.reshape(B, K, -1)

        return tokens

    def generate_answer(
        self,
        image: torch.Tensor,
        query: str,
        max_length: int = 100
    ) -> str:
        """
        Generate answer to query given image.

        Args:
            image: [B, 3, H, W]
            query: str
            max_length: max tokens to generate

        Returns:
            answer: str
        """
        # Get augmented tokens
        outputs = self.forward(image, query)
        all_tokens = outputs['tokens']  # [B, 529, d_model]

        # Generate with VLM decoder
        with torch.no_grad():
            # Prepare query input
            query_inputs = self.tokenizer(query, return_tensors="pt", padding=True).to(image.device)

            # Generate
            generated_ids = self.base_vllm.generate(
                inputs_embeds=all_tokens,
                input_ids=query_inputs.input_ids,
                max_length=max_length,
                num_beams=3,
                early_stopping=True
            )

            # Decode
            answer = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return answer
```

### Usage Example

```python
# Initialize system
arr_system = ARRVisionSystem(
    base_vllm_name="Qwen/Qwen3-VL-2B-Instruct",
    num_saccades=273,
    freeze_base=True
)

# Load image
from PIL import Image
import torchvision.transforms as transforms

image = Image.open("test_image.jpg")
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor()
])
image_tensor = transform(image).unsqueeze(0)  # [1, 3, 448, 448]

# Query
query = "What is written on the small sign in the background?"

# Generate answer
answer = arr_system.generate_answer(image_tensor, query)
print(f"Answer: {answer}")

# Inspect relevance
outputs = arr_system(image_tensor, query)
relevance_map = outputs['relevance_map']  # [1, 448, 448]
saccade_positions = outputs['saccade_positions']  # [1, 273, 2]

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(relevance_map[0].cpu().detach(), cmap='hot')
plt.title("Relevance Map")

plt.subplot(1, 3, 3)
plt.imshow(image)
plt.scatter(
    saccade_positions[0, :, 1].cpu(),
    saccade_positions[0, :, 0].cpu(),
    c='red', s=10, alpha=0.5
)
plt.title("Saccade Positions (273)")

plt.tight_layout()
plt.show()
```

### Research Topics: Complete System

**Search for:**
- "end to end vision language model training"
- "freezing pretrained models adapter training"
- "multimodal transformer inference efficiency"
- "vision encoder decoder attention mechanisms"

**Key questions:**
- Memory requirements for 529 tokens vs 256?
- Inference speed overhead?
- Training convergence with frozen base?
- Gradient flow through token selection?

---

## 7. Training Strategy

### Frozen Base + Trainable ARR

```python
class ARRTrainer:
    """
    Training loop for ARR components.
    Base VLM is frozen, only ARR components update.
    """

    def __init__(
        self,
        arr_system: ARRVisionSystem,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01
    ):
        self.arr_system = arr_system

        # Only optimize ARR components
        trainable_params = [
            {'params': arr_system.texture_generator.parameters()},
            {'params': arr_system.relevance_scorer.parameters()}
        ]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10000
        )

    def train_step(
        self,
        image: torch.Tensor,
        query: str,
        answer: str
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            image: [B, 3, H, W]
            query: str
            answer: str (ground truth)

        Returns:
            metrics: dict with loss, accuracy, etc.
        """
        self.optimizer.zero_grad()

        # Forward pass (ARR components)
        outputs = self.arr_system(image, query)
        all_tokens = outputs['tokens']  # [B, 529, d_model]

        # Generate with frozen VLM
        with torch.no_grad():
            # Encode answer
            answer_inputs = self.arr_system.tokenizer(
                answer,
                return_tensors="pt",
                padding=True
            ).to(image.device)

            # Forward through VLM decoder (get logits)
            logits = self.arr_system.base_vllm.language_model(
                inputs_embeds=all_tokens,
                labels=answer_inputs.input_ids
            ).logits

        # Compute loss
        # NOTE: Loss on answer tokens only, not on image tokens
        loss = F.cross_entropy(
            logits[:, -answer_inputs.input_ids.shape[1]:, :].reshape(-1, logits.shape[-1]),
            answer_inputs.input_ids.reshape(-1),
            ignore_index=self.arr_system.tokenizer.pad_token_id
        )

        # Backward (only ARR components update)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.arr_system.texture_generator.parameters(),
            max_norm=1.0
        )
        torch.nn.utils.clip_grad_norm_(
            self.arr_system.relevance_scorer.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()
        self.scheduler.step()

        return {
            'loss': loss.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        """
        self.arr_system.train()

        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            image = batch['image']  # [B, 3, H, W]
            query = batch['query']  # List[str]
            answer = batch['answer']  # List[str]

            # Handle batch of queries/answers
            batch_loss = 0.0
            for i in range(len(query)):
                metrics = self.train_step(
                    image[i:i+1],
                    query[i],
                    answer[i]
                )
                batch_loss += metrics['loss']

            batch_loss /= len(query)
            total_loss += batch_loss
            num_batches += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {batch_loss:.4f}")

        return {
            'epoch_loss': total_loss / num_batches
        }
```

### Training Data Requirements

```python
# Dataset structure
from torch.utils.data import Dataset

class VQADataset(Dataset):
    """
    Visual Question Answering dataset for ARR training.

    Sources:
    - VQAv2: 200K+ images, 1M+ questions
    - TextVQA: Text-focused questions
    - COCO Captions: Image description
    """

    def __init__(self, data_path: str, split: str = 'train'):
        self.data = load_vqa_data(data_path, split)
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        image = Image.open(item['image_path']).convert('RGB')
        image = self.transform(image)

        return {
            'image': image,
            'query': item['question'],
            'answer': item['answer']
        }

# Training loop
def train_arr_system():
    # Initialize
    arr_system = ARRVisionSystem(freeze_base=True)
    trainer = ARRTrainer(arr_system, learning_rate=1e-4)

    # Datasets
    train_dataset = VQADataset('data/vqav2', split='train')
    val_dataset = VQADataset('data/vqav2', split='val')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )

    # Training
    num_epochs = 10

    for epoch in range(num_epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        print(f"Epoch {epoch} - Train Loss: {train_metrics['epoch_loss']:.4f}")

        # Validate
        val_metrics = evaluate(arr_system, val_loader)
        print(f"Epoch {epoch} - Val Accuracy: {val_metrics['accuracy']:.4f}")

        # Save checkpoint
        if epoch % 2 == 0:
            torch.save({
                'epoch': epoch,
                'texture_generator': arr_system.texture_generator.state_dict(),
                'relevance_scorer': arr_system.relevance_scorer.state_dict(),
                'optimizer': trainer.optimizer.state_dict()
            }, f'checkpoints/arr_epoch_{epoch}.pt')
```

### Research Topics: Training

**Search for:**
- "training vision language models frozen backbone"
- "adapter training multimodal transformers"
- "gradient flow through sampling operations"
- "VQAv2 training best practices"

**Key papers:**
- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Adapter layers for frozen transformers
- Prefix tuning, prompt tuning comparisons
- BLIP-2 Q-Former training strategy

---

## 8. Ablations to Test

### Experimental Design

```python
# ABLATION 1: ARR vs Baseline
# Compare ARR-augmented vs standard Qwen3-VL

def ablation_arr_vs_baseline():
    """
    Test: Does ARR improve over standard VLM?

    Setup:
    - Baseline: Qwen3-VL with 256 tokens
    - ARR: Qwen3-VL + 273 saccade tokens (529 total)

    Metrics:
    - VQAv2 accuracy
    - TextVQA accuracy (text-heavy)
    - COCO CIDEr (caption quality)

    Hypothesis: ARR improves accuracy on detail-focused queries
    """
    pass


# ABLATION 2: Number of saccades
# Test: What's the optimal K?

def ablation_num_saccades():
    """
    Test: How many saccades are needed?

    Setup:
    - Vary K: 50, 100, 200, 273, 400, 600
    - Measure accuracy vs inference time

    Metrics:
    - VQAv2 accuracy
    - Inference time (ms per image)
    - Memory usage (GB)

    Hypothesis: Diminishing returns after ~300 saccades
    """
    pass


# ABLATION 3: Ordering strategy
# Test: Does relevance ordering matter?

def ablation_ordering():
    """
    Test: Spatial vs relevance vs random ordering

    Setup:
    - Spatial: left-to-right, top-to-bottom
    - Relevance: high-to-low score (chosen approach)
    - Random: shuffled positions

    Metrics:
    - VQAv2 accuracy
    - Answer quality (human eval)

    Hypothesis: Relevance ordering helps, spatial neutral, random hurts
    """
    pass


# ABLATION 4: Gestalt guidance
# Test: Does gestalt context help selection?

def ablation_gestalt():
    """
    Test: With vs without gestalt-guided scoring

    Setup:
    - With gestalt: contextualized scorer weights (full system)
    - Without gestalt: fixed scorer weights (Part 37 problem)

    Metrics:
    - VQAv2 accuracy
    - Relevance map quality (IoU with human attention)

    Hypothesis: Gestalt guidance significantly improves relevance
    """
    pass


# ABLATION 5: Scorer weighting
# Test: Which scorers matter most?

def ablation_scorers():
    """
    Test: Propositional vs Perspectival vs Participatory

    Setup:
    - Propositional only: [1.0, 0.0, 0.0]
    - Perspectival only: [0.0, 1.0, 0.0]
    - Participatory only: [0.0, 0.0, 1.0]
    - Equal mix: [0.33, 0.33, 0.33]
    - Learned contextual: Full system

    Metrics:
    - VQAv2 accuracy per query type
    - Analyze: which queries need which scorers?

    Hypothesis: Different queries need different scorer emphasis
    """
    pass


# ABLATION 6: Texture channels
# Test: Which texture features are critical?

def ablation_texture():
    """
    Test: Full 40 channels vs subsets

    Setup:
    - Full: All 40 channels
    - No CLIP: Remove channels 17-32 (semantic features)
    - No edges: Remove channels 6-8 (edge/highpass)
    - Minimal: RGB + CLIP only

    Metrics:
    - VQAv2 accuracy
    - TextVQA accuracy
    - Relevance map quality

    Hypothesis: CLIP features most critical for participatory
    """
    pass


# ABLATION 7: Patch size
# Test: 14×14 vs other sizes

def ablation_patch_size():
    """
    Test: Does patch size matter?

    Setup:
    - 7×7: Smaller patches
    - 14×14: Standard (Qwen3-VL base)
    - 28×28: Larger patches

    NOTE: This requires custom VLM encoder for non-14×14

    Metrics:
    - Accuracy
    - Inference speed

    Hypothesis: 14×14 optimal (matches VLM architecture)
    """
    pass
```

### Ablation Results Template

```python
# Expected results table

"""
╔══════════════════════════════════════════════════════════════
║ ABLATION STUDY RESULTS
╠══════════════════════════════════════════════════════════════
║
║ Dataset: VQAv2 validation set (100K examples)
║ Base model: Qwen3-VL-2B-Instruct
║ Hardware: 1x A100 40GB
║
╠══════════════════════════════════════════════════════════════
║ EXPERIMENT                 │ ACCURACY │ INFERENCE TIME │ MEMORY
╠══════════════════════════════════════════════════════════════
║ Baseline (256 tokens)      │  65.3%   │   120 ms       │  8.2 GB
║ ARR (529 tokens)           │  68.7%   │   185 ms       │  10.1 GB
║ ─────────────────────────────────────────────────────────────
║ Gain from ARR              │  +3.4%   │   +54% slower  │  +23% mem
║
╠══════════════════════════════════════════════════════════════
║ NUM SACCADES
╠══════════════════════════════════════════════════════════════
║ K=50                       │  66.1%   │   135 ms       │  8.5 GB
║ K=100                      │  67.2%   │   150 ms       │  9.0 GB
║ K=273 (chosen)             │  68.7%   │   185 ms       │  10.1 GB
║ K=500                      │  69.0%   │   240 ms       │  11.5 GB
║ ─────────────────────────────────────────────────────────────
║ Observation                │ Diminishing returns after 300
║
╠══════════════════════════════════════════════════════════════
║ ORDERING STRATEGY
╠══════════════════════════════════════════════════════════════
║ Spatial (l→r, t→b)         │  68.2%   │   185 ms       │  10.1 GB
║ Relevance (high→low)       │  68.7%   │   185 ms       │  10.1 GB
║ Random                     │  67.5%   │   185 ms       │  10.1 GB
║ ─────────────────────────────────────────────────────────────
║ Observation                │ Relevance slightly better
║
╠══════════════════════════════════════════════════════════════
║ GESTALT GUIDANCE
╠══════════════════════════════════════════════════════════════
║ With gestalt context       │  68.7%   │   185 ms       │  10.1 GB
║ Without (fixed weights)    │  66.9%   │   185 ms       │  10.1 GB
║ ─────────────────────────────────────────────────────────────
║ Gain from gestalt          │  +1.8%   │   No change    │  No change
║
╠══════════════════════════════════════════════════════════════
║ SCORER WEIGHTING
╠══════════════════════════════════════════════════════════════
║ Propositional only         │  64.5%   │   185 ms       │  10.1 GB
║ Perspectival only          │  63.2%   │   185 ms       │  10.1 GB
║ Participatory only         │  67.1%   │   185 ms       │  10.1 GB
║ Equal mix                  │  66.8%   │   185 ms       │  10.1 GB
║ Learned contextual         │  68.7%   │   185 ms       │  10.1 GB
║ ─────────────────────────────────────────────────────────────
║ Observation                │ Contextual weighting best
║
╚══════════════════════════════════════════════════════════════

NOTE: These are HYPOTHETICAL results for illustration.
Real experiments required to validate.
"""
```

### Research Topics: Ablations

**Search for:**
- "ablation study design vision models"
- "VQA benchmark evaluation protocols"
- "statistical significance testing neural networks"
- "human attention agreement metrics computer vision"

---

## 9. Comparison to Related Work

### Existing Approaches

```python
# Comparison table

"""
╔══════════════════════════════════════════════════════════════════════════════
║ APPROACH COMPARISON
╠══════════════════════════════════════════════════════════════════════════════
║
║ METHOD              │ RESOLUTION  │ TOKEN COUNT │ RELEVANCE   │ AUGMENTATION
╠══════════════════════════════════════════════════════════════════════════════
║ Standard VLM        │ Fixed       │ 256         │ None        │ No
║ (Qwen3-VL base)     │ 14×14       │             │             │
║                     │ patches     │             │             │
║ ────────────────────────────────────────────────────────────────────────────
║ Dynamic Resolution  │ Variable    │ 256-16K     │ Heuristic   │ No
║ (Qwen3-VL native)   │ by image    │ (quad with  │ (resolution)│
║                     │ resolution  │ res)        │             │
║ ────────────────────────────────────────────────────────────────────────────
║ LLaVA-UHD           │ Fixed per   │ 256-1024    │ Spatial     │ No
║ (Tiling)            │ tile        │ (tiles)     │ (uniform    │
║                     │             │             │ splits)     │
║ ────────────────────────────────────────────────────────────────────────────
║ DeepSeek-OCR        │ Compressed  │ 257         │ None        │ No
║ (COC compression)   │ 256 from    │ (efficient) │ (compress   │
║                     │ 4096        │             │ everything) │
║ ────────────────────────────────────────────────────────────────────────────
║ FoveaTer            │ Variable    │ ~300        │ Saliency    │ Substitution
║ (Foveated ViT)      │ coarse +    │             │ map         │ (replaces
║                     │ fine        │             │             │ base tokens)
║ ────────────────────────────────────────────────────────────────────────────
║ ARR-VIS (Ours)      │ Fixed       │ 529         │ Vervaekean  │ Augmentation
║                     │ 14×14       │ (256 base + │ (3 ways of  │ (adds to
║                     │ all patches │ 273 saccade)│ knowing)    │ base tokens)
╚══════════════════════════════════════════════════════════════════════════════
"""

# Detailed comparison

COMPARISON_TABLE = {
    'Standard VLM': {
        'approach': 'Fixed grid sampling',
        'pros': ['Simple', 'Fast', 'Stable training'],
        'cons': ['No query awareness', 'Uniform allocation', 'Misses details'],
        'example': 'CLIP, ViT, BLIP'
    },

    'Dynamic Resolution': {
        'approach': 'More tokens for high-res images',
        'pros': ['Preserves details in high-res', 'Adaptive to image'],
        'cons': ['Not query-aware', 'Quadratic token growth', 'Slow for high-res'],
        'example': 'Qwen3-VL native mode'
    },

    'Tiling (LLaVA-UHD)': {
        'approach': 'Split image into tiles, process separately',
        'pros': ['Handles high-res', 'Parallel processing'],
        'cons': ['Loses global context', 'Arbitrary splits', 'Not query-aware'],
        'example': 'LLaVA-UHD, GPT-4V'
    },

    'Compression (DeepSeek-OCR)': {
        'approach': 'SAM compresses to fewer tokens',
        'pros': ['Efficient (257 tokens)', 'Fast inference'],
        'cons': ['Lossy compression', 'No relevance guidance', 'Fixed compression'],
        'example': 'DeepSeek-OCR'
    },

    'FoveaTer': {
        'approach': 'Coarse everywhere + fine at fixation',
        'pros': ['Biologically inspired', 'Adaptive sampling'],
        'cons': ['Replaces base tokens', 'Saliency-only', 'Not query-driven'],
        'example': 'FoveaTer (2024)'
    },

    'ARR-VIS (Ours)': {
        'approach': 'Gestalt + query-aware saccades',
        'pros': [
            'Query-aware relevance',
            'Preserves gestalt context',
            'Vervaekean framework',
            'Augmentation not substitution'
        ],
        'cons': [
            '2× token count',
            'Requires training',
            'More complex'
        ],
        'novelty': [
            'Gestalt guides saccades',
            'Contextualized scorer weighting',
            'Augmentation architecture',
            'Fixed patches, variable allocation'
        ]
    }
}
```

### Key Novelties of ARR-VIS

1. **Augmentation not substitution**
   - Keep full gestalt (256 base tokens)
   - Add focused saccades (273 tokens)
   - Both representations coexist

2. **Gestalt-guided selection**
   - Base encoding informs relevance scoring
   - Query + gestalt → contextualized weights
   - Not blind saccades

3. **Vervaekean framework**
   - Three ways of knowing (propositional, perspectival, participatory)
   - Opponent processing implicit in weighting
   - Cognitive science grounding

4. **Fixed patches, variable allocation**
   - All patches 14×14 (training stable)
   - Variable density (more patches where relevant)
   - Not variable patch sizes

5. **Relevance ordering**
   - Sequence position encodes importance
   - Spatial position via RoPE
   - Dual encoding

### Research Topics: Related Work

**Search for:**
- "FoveaTer foveated vision transformer"
- "LLaVA-UHD high resolution image understanding"
- "vision language model attention mechanisms survey"
- "query aware image processing"

**Key papers to review:**
- FoveaTer (2024) - closest to our approach
- Perceiver, Perceiver IO - cross-attention architecture
- Flamingo - interleaved vision-language
- BLIP-2 Q-Former - learned query tokens

---

## 10. Implementation Roadmap

### Phase 1: Core ARR (Weeks 1-4)

**Week 1: Texture Generator**
```python
# Milestone 1.1: Basic texture generation
- Implement TextureGenerator module
- Test on sample images
- Verify 40 channels output correctly
- Visualize each channel

# Milestone 1.2: CLIP integration
- Integrate CLIP encoder
- Project to 16 channels
- Verify semantic features align with query
```

**Week 2: Relevance Scorer**
```python
# Milestone 2.1: Three scorers
- Implement propositional head
- Implement perspectival head
- Implement participatory head
- Test each independently

# Milestone 2.2: Context weighting
- Implement context_weights network
- Test gestalt + query → weights
- Verify different queries → different weights
```

**Week 3: Saccade Selection**
```python
# Milestone 3.1: Position selection
- Implement SaccadeSelector
- Test top-K selection
- Visualize selected positions

# Milestone 3.2: Patch extraction
- Implement patch extraction at positions
- Handle edge cases (boundaries)
- Verify all patches 14×14
```

**Week 4: Integration**
```python
# Milestone 4.1: Full forward pass
- Implement ARRVisionSystem
- Test end-to-end: image + query → tokens
- Verify 529 tokens output

# Milestone 4.2: VLM integration
- Connect to Qwen3-VL
- Test generation
- Qualitative evaluation
```

### Phase 2: Training (Weeks 5-8)

**Week 5: Dataset preparation**
```python
# Milestone 5.1: VQAv2 dataset
- Download and preprocess VQAv2
- Implement VQADataset class
- Verify data loading

# Milestone 5.2: TextVQA dataset
- Download and preprocess TextVQA
- Add to dataloader
- Mixed training setup
```

**Week 6: Training loop**
```python
# Milestone 6.1: Trainer implementation
- Implement ARRTrainer
- Test single training step
- Verify gradients flow correctly

# Milestone 6.2: Monitoring
- Add logging (TensorBoard/W&B)
- Track loss, accuracy, relevance maps
- Visualize saccade positions during training
```

**Week 7-8: Full training**
```python
# Milestone 7.1: Initial training
- Train for 10 epochs on VQAv2
- Monitor convergence
- Save checkpoints

# Milestone 7.2: Hyperparameter tuning
- Try different learning rates
- Vary num_saccades
- Test different architectures
```

### Phase 3: Evaluation & Ablations (Weeks 9-12)

**Week 9: Benchmark evaluation**
```python
# Milestone 9.1: VQAv2 accuracy
- Evaluate on VQAv2 validation set
- Compare to baseline Qwen3-VL
- Analyze failure cases

# Milestone 9.2: TextVQA accuracy
- Evaluate on TextVQA
- Test text-focused queries
- Compare to DeepSeek-OCR
```

**Week 10-11: Ablation studies**
```python
# Milestone 10.1: Core ablations
- ARR vs baseline
- Num saccades (K=50, 100, 273, 500)
- Ordering strategies

# Milestone 10.2: Component ablations
- Gestalt guidance (with/without)
- Scorer weighting (fixed vs contextual)
- Texture channels (full vs subsets)
```

**Week 12: Analysis & documentation**
```python
# Milestone 12.1: Results analysis
- Statistical significance testing
- Qualitative examples
- Error analysis

# Milestone 12.2: Paper writing
- Introduction, related work
- Method description
- Results, ablations
- Discussion, future work
```

### Phase 4: COC Integration (Future)

**Optional: Add COC compression**
```python
# Only if Phase 1-3 show ARR works

# Implement SAM-based compression
# Integrate as base encoder
# Benchmark ARR-COC vs ARR-only
```

### Research Topics: Implementation

**Search for:**
- "vision language model training pipeline"
- "debugging transformer training"
- "gradient flow visualization neural networks"
- "VQAv2 dataset preprocessing best practices"

**Tools needed:**
- PyTorch 2.0+
- Transformers library (HuggingFace)
- Weights & Biases (logging)
- VQAv2, TextVQA datasets
- A100 GPU (40GB minimum)

---

## Conclusion

This technical addendum provides complete implementation specifications for ARR-VIS, the Adaptive Relevance Realization vision system inspired by the Scroll of Gestalt and Gaze.

**Key takeaways:**

1. **Fixed patches, variable allocation** - Training stable, batching simple
2. **Gestalt guides saccades** - Context shapes relevance
3. **Augmentation, not substitution** - 256 base + 273 saccade = 529 total
4. **ARR independent of COC** - Separate concerns, modular design
5. **Vervaekean framework** - Three ways of knowing, contextualized weights
6. **Relevance ordering** - Sequence position = importance signal
7. **Frozen base + trainable ARR** - Efficient training strategy

**Next steps:**

- Implement Phase 1 (Core ARR)
- Research related work (FoveaTer, Perceiver)
- Acquire VQAv2 dataset
- Begin training experiments

The scroll has taught us. Now we build.

---

**End of Technical Addendum**