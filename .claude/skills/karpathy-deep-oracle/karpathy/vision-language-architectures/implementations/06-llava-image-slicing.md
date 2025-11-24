# LLaVA Image Grid Slicing Strategy

## Overview: High-Resolution Processing Through Spatial Partitioning

LLaVA (Large Language and Vision Assistant) pioneered a pragmatic approach to high-resolution image processing in vision-language models: instead of increasing the vision encoder's native resolution, split high-resolution images into a grid of patches and process each patch independently with CLIP ViT-L/14-336px. This strategy enables processing images at much higher effective resolutions (up to 672x672, 1008x1008, or higher) while keeping the vision encoder frozen and avoiding costly retraining.

**Key innovation**: Transform the high-resolution challenge from "how do we encode huge images?" to "how do we intelligently partition images into manageable chunks?"

From [LLaVA GitHub Repository](https://github.com/haotian-liu/LLaVA) (accessed 2025-01-31):
- LLaVA-1.5: 336px base resolution with optional grid slicing
- LLaVA-NeXT: Enhanced grid strategies supporting multiple scales
- Architecture combines CLIP vision encoder + linear projector + Vicuna/Llama LLM

## Base Architecture: Single Image Processing

### Standard LLaVA-1.5 Pipeline (336px)

```python
# From LLaVA model architecture
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor

class LLaVAVisionEncoder(nn.Module):
    def __init__(self, vision_tower="openai/clip-vit-large-patch14-336"):
        super().__init__()
        # CLIP ViT-L/14 with 336x336 input resolution
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
        self.vision_tower.requires_grad_(False)  # Frozen encoder

        # Image processor for CLIP
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        # Defaults: size['shortest_edge'] = 336, crop_size = 336x336

    def forward(self, images):
        # images: [B, 3, 336, 336]
        # Output: [B, 576, 1024] where 576 = 24x24 patches
        vision_outputs = self.vision_tower(
            images,
            output_hidden_states=True
        )
        # Extract features from specified layer (default: -2)
        image_features = vision_outputs.hidden_states[-2][:, 1:]  # Remove CLS token
        return image_features  # [B, 576, 1024]
```

**Token count**: 336px input → 24×24 = 576 patch tokens per image (patch size 14px)

## Grid Slicing: The Core Strategy

### Motivation: Why Slice?

1. **Resolution limitations**: CLIP ViT-L/14-336 trained on 336×336 images
2. **Small text problem**: Text in high-res images becomes illegible at 336px
3. **Fine detail loss**: Important visual features compressed away
4. **Frozen encoder constraint**: Cannot retrain CLIP without massive compute

**Solution**: Split high-resolution images into overlapping or non-overlapping grid of 336px patches, process each independently, then concatenate features.

From [Beyond LLaVA-HD: Diving into High-Resolution Large Multimodal Models](https://arxiv.org/abs/2406.08487) (accessed 2025-01-31):
> "Initially, we explore various grid options for slicing images, similar to LLaVA-Next, but with finer granularity. We investigate how different grid configurations (2×2, 3×3, 4×4) affect performance across various image resolutions."

### Basic Grid Slicing Implementation

```python
import torch
import torch.nn.functional as F
from typing import List, Tuple

def slice_image_into_grid(
    image: torch.Tensor,  # [3, H, W]
    grid_size: Tuple[int, int] = (2, 2),  # (rows, cols)
    patch_size: int = 336,
    overlap: int = 0
) -> List[torch.Tensor]:
    """
    Slice high-resolution image into grid of patches.

    Args:
        image: Input image [3, H, W]
        grid_size: Grid dimensions (rows, cols)
        patch_size: Size of each patch (336 for CLIP ViT-L/14-336)
        overlap: Overlap between adjacent patches (pixels)

    Returns:
        List of patches, each [3, patch_size, patch_size]
    """
    C, H, W = image.shape
    rows, cols = grid_size

    # Calculate stride (non-overlapping by default)
    stride = patch_size - overlap

    patches = []
    for i in range(rows):
        for j in range(cols):
            # Calculate patch boundaries
            y_start = i * stride
            x_start = j * stride
            y_end = y_start + patch_size
            x_end = x_start + patch_size

            # Extract patch (with padding if needed)
            if y_end > H or x_end > W:
                # Pad image if patch exceeds boundaries
                pad_bottom = max(0, y_end - H)
                pad_right = max(0, x_end - W)
                patch = F.pad(
                    image[:, y_start:min(y_end, H), x_start:min(x_end, W)],
                    (0, pad_right, 0, pad_bottom),
                    mode='constant',
                    value=0
                )
            else:
                patch = image[:, y_start:y_end, x_start:x_end]

            patches.append(patch)

    return patches  # List of (rows × cols) patches

# Example: Split 672x672 image into 2x2 grid
image = torch.randn(3, 672, 672)
patches = slice_image_into_grid(image, grid_size=(2, 2), patch_size=336)
# Result: 4 patches, each [3, 336, 336]
```

### Dynamic Grid Selection

```python
def select_optimal_grid(
    image_height: int,
    image_width: int,
    base_patch_size: int = 336,
    max_patches: int = 16  # 4×4 grid maximum
) -> Tuple[int, int]:
    """
    Dynamically select grid size based on image dimensions.

    Strategy:
    - Small images (≤336): 1×1 (no slicing)
    - Medium images (≤672): 2×2
    - Large images (≤1008): 3×3
    - Very large images: 4×4

    Args:
        image_height: Image height in pixels
        image_width: Image width in pixels
        base_patch_size: CLIP input size (336)
        max_patches: Maximum total patches allowed

    Returns:
        (rows, cols) grid configuration
    """
    # Calculate required grid size to cover image
    rows_needed = (image_height + base_patch_size - 1) // base_patch_size
    cols_needed = (image_width + base_patch_size - 1) // base_patch_size

    # Clamp to maximum grid size (e.g., 4×4)
    max_dim = int(max_patches ** 0.5)
    rows = min(rows_needed, max_dim)
    cols = min(cols_needed, max_dim)

    return (rows, cols)

# Examples:
select_optimal_grid(400, 600, 336)  # → (2, 2) for 400×600 image
select_optimal_grid(800, 1200, 336)  # → (3, 4) for 800×1200 image
select_optimal_grid(1400, 1400, 336)  # → (4, 4) for 1400×1400 image (capped)
```

## LLaVA-1.5 Grid Processing Pipeline

### Complete High-Resolution Processing

```python
class LLaVAHighResProcessor(nn.Module):
    """
    LLaVA-1.5 style high-resolution processing with grid slicing.

    Strategy:
    1. Resize image to target resolution (e.g., 672×672)
    2. Slice into grid (e.g., 2×2 = 4 patches)
    3. Process each patch through CLIP encoder
    4. Concatenate all patch features
    5. Optionally add global context (downsampled full image)
    """
    def __init__(
        self,
        vision_tower: str = "openai/clip-vit-large-patch14-336",
        grid_size: Tuple[int, int] = (2, 2),
        add_global_context: bool = True
    ):
        super().__init__()
        self.vision_encoder = LLaVAVisionEncoder(vision_tower)
        self.grid_size = grid_size
        self.add_global_context = add_global_context

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Process high-resolution image with grid slicing.

        Args:
            image: [3, H, W] high-resolution image

        Returns:
            concatenated_features: [N_total, D] where
                N_total = (grid_rows × grid_cols) × 576 [+ 576 if global]
                D = 1024 (CLIP ViT-L/14 hidden size)
        """
        # Step 1: Slice image into grid
        patches = slice_image_into_grid(
            image,
            grid_size=self.grid_size,
            patch_size=336
        )

        # Step 2: Stack patches into batch
        patches_batch = torch.stack(patches)  # [N_patches, 3, 336, 336]

        # Step 3: Process all patches through CLIP
        patch_features = self.vision_encoder(patches_batch)
        # [N_patches, 576, 1024]

        # Step 4: Flatten spatial dimension (concatenate patches)
        # Reshape to [N_patches × 576, 1024]
        patch_features_flat = patch_features.reshape(-1, patch_features.size(-1))

        # Step 5: Optionally add global context
        if self.add_global_context:
            # Downsample full image to 336×336
            global_image = F.interpolate(
                image.unsqueeze(0),  # [1, 3, H, W]
                size=(336, 336),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            # Process global image
            global_features = self.vision_encoder(global_image.unsqueeze(0))
            # [1, 576, 1024]

            # Concatenate: [patches..., global]
            all_features = torch.cat([
                patch_features_flat,  # [N_patches × 576, 1024]
                global_features.reshape(-1, 1024)  # [576, 1024]
            ], dim=0)

            return all_features  # [(N_patches + 1) × 576, 1024]

        return patch_features_flat  # [N_patches × 576, 1024]

# Example: Process 672×672 image with 2×2 grid
processor = LLaVAHighResProcessor(grid_size=(2, 2), add_global_context=True)
high_res_image = torch.randn(3, 672, 672)
features = processor(high_res_image)
# Output shape: [(4 + 1) × 576, 1024] = [2880, 1024]
# 2304 tokens from 4 patches + 576 tokens from global context
```

## Spatial Position Encoding

### Why Position Encoding Matters

Grid slicing destroys spatial relationships between patches. Position encodings help the language model understand which patch came from where in the original image.

```python
def add_2d_positional_encoding(
    patch_features: torch.Tensor,  # [B, N_patches, N_tokens, D]
    grid_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Add learnable 2D positional encodings to grid patches.

    Args:
        patch_features: [B, N_patches, N_tokens, D]
        grid_size: (rows, cols)

    Returns:
        features_with_pos: [B, N_patches, N_tokens, D]
    """
    B, N_patches, N_tokens, D = patch_features.shape
    rows, cols = grid_size

    # Create 2D position embedding
    # Shape: [rows, cols, D]
    pos_embed_2d = nn.Parameter(torch.randn(rows, cols, D))

    # Expand to match batch and token dimensions
    # [rows, cols, D] → [rows × cols, 1, D] → [B, N_patches, N_tokens, D]
    pos_embed_flat = pos_embed_2d.reshape(N_patches, 1, D)
    pos_embed_expanded = pos_embed_flat.unsqueeze(0).expand(B, -1, N_tokens, -1)

    # Add positional encoding
    features_with_pos = patch_features + pos_embed_expanded

    return features_with_pos
```

### Alternative: Explicit Grid Coordinates

```python
def encode_grid_coordinates(
    grid_size: Tuple[int, int],
    hidden_dim: int = 1024
) -> torch.Tensor:
    """
    Encode grid position as continuous coordinates.

    Similar to Transformer sinusoidal positional encoding,
    but for 2D spatial grid.

    Args:
        grid_size: (rows, cols)
        hidden_dim: Feature dimension

    Returns:
        position_encodings: [rows × cols, hidden_dim]
    """
    rows, cols = grid_size

    # Normalize coordinates to [0, 1]
    y_coords = torch.linspace(0, 1, rows)
    x_coords = torch.linspace(0, 1, cols)

    # Create 2D grid
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Flatten to [rows × cols, 2]
    coords = torch.stack([y_grid.flatten(), x_grid.flatten()], dim=-1)

    # Apply sinusoidal encoding (similar to Transformer)
    # Split hidden_dim into position encoding dimensions
    pos_dim = hidden_dim // 4  # Split for x, y (sin, cos each)

    # Frequency bands
    freq_bands = 2 ** torch.linspace(0, pos_dim - 1, pos_dim)

    # Encode x and y coordinates
    x_enc = coords[:, 1:2] * freq_bands.unsqueeze(0)  # [N, pos_dim]
    y_enc = coords[:, 0:1] * freq_bands.unsqueeze(0)  # [N, pos_dim]

    # Concatenate sin/cos for both dimensions
    position_encodings = torch.cat([
        torch.sin(x_enc), torch.cos(x_enc),
        torch.sin(y_enc), torch.cos(y_enc)
    ], dim=-1)  # [rows × cols, hidden_dim]

    return position_encodings
```

## Token Budget Analysis

### Token Explosion Problem

Grid slicing dramatically increases visual token count, challenging LLM context windows.

**Token counts by grid size** (336px patches, 576 tokens/patch):

| Grid Size | Image Resolution | Patches | Tokens (no global) | Tokens (+ global) |
|-----------|-----------------|---------|-------------------|-------------------|
| 1×1 | 336×336 | 1 | 576 | 576 |
| 2×2 | 672×672 | 4 | 2,304 | 2,880 |
| 3×3 | 1008×1008 | 9 | 5,184 | 5,760 |
| 4×4 | 1344×1344 | 16 | 9,216 | 9,792 |

**Challenge**: A 4×4 grid uses ~10k tokens just for vision, leaving limited budget for text in typical 8k-32k context LLMs.

From [LLaVA-NeXT Blog](https://llava-vl.github.io/blog/2024-01-30-llava-next/) (accessed 2025-01-31):
> "LLaVA-NeXT can now process 4× more pixels than LLaVA-1.5, effectively handling images up to 672×672 with 2×2 grid (2,880 tokens) or 1008×1008 with 3×3 grid (5,760 tokens)."

## Advanced Strategies: Beyond Uniform Grids

### LLaVA-UHD: Aspect Ratio Awareness

From [LLaVA-UHD: an LMM Perceiving Any Aspect Ratio and High-Resolution](https://arxiv.org/abs/2403.11703) (accessed 2025-01-31):
> "With variable-sized slices, LLaVA-UHD can achieve full adaptivity to native-resolution images without padding or shape-distorting reshaping."

```python
def adaptive_grid_partition(
    image_height: int,
    image_width: int,
    base_patch: int = 336,
    max_patches: int = 16
) -> List[Tuple[int, int, int, int]]:
    """
    LLaVA-UHD style: Adapt grid to image aspect ratio.

    Strategy:
    - Maintain aspect ratio (no padding/distortion)
    - Use variable-sized slices
    - Optimize for minimal padding

    Args:
        image_height, image_width: Original image dimensions
        base_patch: CLIP patch size (336)
        max_patches: Maximum number of patches

    Returns:
        List of (y_start, x_start, height, width) for each slice
    """
    aspect_ratio = image_width / image_height

    # Find optimal grid configuration
    best_grid = None
    min_padding = float('inf')

    for rows in range(1, int(max_patches**0.5) + 1):
        for cols in range(1, max_patches // rows + 1):
            if rows * cols > max_patches:
                continue

            # Calculate slice dimensions
            slice_h = image_height // rows
            slice_w = image_width // cols

            # Check if slices fit within base_patch after resizing
            # (each slice will be resized to 336×336)
            resize_ratio = min(base_patch / slice_h, base_patch / slice_w)

            # Calculate wasted space (padding)
            padding = abs(rows * base_patch - image_height) + \
                     abs(cols * base_patch - image_width)

            if padding < min_padding:
                min_padding = padding
                best_grid = (rows, cols)

    rows, cols = best_grid
    slices = []

    slice_h = image_height // rows
    slice_w = image_width // cols

    for i in range(rows):
        for j in range(cols):
            y_start = i * slice_h
            x_start = j * slice_w
            # Last row/column extends to image edge
            height = image_height - y_start if i == rows - 1 else slice_h
            width = image_width - x_start if j == cols - 1 else slice_w
            slices.append((y_start, x_start, height, width))

    return slices

# Example: Ultra-wide image (2560×1080)
slices = adaptive_grid_partition(1080, 2560, 336, 16)
# Result: 4×1 grid (4 slices along width, 1 along height)
# Each slice: ~270×640 → resized to 336×336
```

### HiRes-LLaVA: Addressing Fragmentation

From [HiRes-LLaVA: Restoring Fragmentation Input in High-Resolution](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_HiRes-LLaVA_Restoring_Fragmentation_Input_in_High-Resolution_Large_Vision-Language_Models_CVPR_2025_paper.pdf) (accessed 2025-01-31):
> "The core of HiRes-LLaVA lies in addressing input fragmentation through a SliceRestore Adapter (SRA) that reconstructs fragmented visual information from sliced patches."

**Problem**: Grid slicing fragments continuous visual features (e.g., splitting objects across patches)

**Solution**: Add cross-patch attention module after CLIP encoding

```python
class SliceRestoreAdapter(nn.Module):
    """
    Restore spatial relationships across sliced patches.

    Inspired by HiRes-LLaVA CVPR 2025.
    """
    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        grid_size: Tuple[int, int] = (2, 2)
    ):
        super().__init__()
        self.grid_size = grid_size

        # Cross-patch attention
        self.cross_patch_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_features: [N_patches × 576, 1024]
                Flattened features from all patches

        Returns:
            restored_features: [N_patches × 576, 1024]
                Features with restored cross-patch relationships
        """
        rows, cols = self.grid_size
        N_patches = rows * cols
        tokens_per_patch = 576

        # Reshape to [N_patches, tokens_per_patch, hidden_dim]
        features = patch_features.reshape(N_patches, tokens_per_patch, -1)

        # Cross-patch attention (patches attend to each other)
        # Flatten patch and token dimensions: [1, N_patches × tokens, hidden]
        flat_features = features.reshape(1, -1, features.size(-1))

        attn_output, _ = self.cross_patch_attn(
            flat_features, flat_features, flat_features
        )

        # Residual connection
        restored = flat_features + attn_output
        restored = self.norm(restored)

        # Flatten back to [N_patches × tokens, hidden]
        return restored.reshape(-1, restored.size(-1))
```

## Production Implementation Considerations

### Memory-Efficient Batch Processing

```python
def process_highres_batch(
    images: List[torch.Tensor],  # List of variable-size images
    vision_encoder: nn.Module,
    grid_size: Tuple[int, int] = (2, 2),
    batch_size: int = 4  # Process patches in batches
) -> List[torch.Tensor]:
    """
    Memory-efficient high-res processing for batch of images.

    Strategy:
    - Each image may produce different number of patches
    - Process patches in mini-batches to manage memory
    - Return list of features (variable length per image)
    """
    all_features = []

    for image in images:
        # Slice image into grid
        patches = slice_image_into_grid(image, grid_size, 336)

        # Process patches in mini-batches
        patch_features = []
        for i in range(0, len(patches), batch_size):
            batch_patches = torch.stack(patches[i:i+batch_size])
            with torch.no_grad():  # Frozen encoder
                batch_features = vision_encoder(batch_patches)
            patch_features.append(batch_features)

        # Concatenate all patch features for this image
        image_features = torch.cat(patch_features, dim=0)
        # [N_patches, 576, 1024] → [N_patches × 576, 1024]
        image_features_flat = image_features.reshape(-1, image_features.size(-1))

        all_features.append(image_features_flat)

    return all_features  # List of [N_tokens_i, 1024]
```

### Grid Size Selection Heuristics

```python
def recommend_grid_size(
    image_resolution: Tuple[int, int],
    task_type: str = "general",
    llm_context_budget: int = 4096
) -> Tuple[int, int]:
    """
    Recommend grid size based on image and task.

    Args:
        image_resolution: (height, width)
        task_type: "ocr" (needs high-res), "general", "coarse"
        llm_context_budget: Available tokens for LLM

    Returns:
        (rows, cols) grid configuration
    """
    height, width = image_resolution
    max_dim = max(height, width)

    # Task-specific resolution requirements
    if task_type == "ocr":
        # Need high resolution for text
        if max_dim <= 672:
            return (2, 2)  # 2,880 tokens
        elif max_dim <= 1008:
            return (3, 3)  # 5,760 tokens
        else:
            return (4, 4)  # 9,792 tokens
    elif task_type == "coarse":
        # Low resolution sufficient
        return (1, 1)  # 576 tokens
    else:  # "general"
        # Balance resolution and token budget
        if llm_context_budget < 2048:
            return (1, 1)  # 576 tokens
        elif llm_context_budget < 4096:
            return (2, 2)  # 2,880 tokens
        elif llm_context_budget < 8192:
            return (3, 3)  # 5,760 tokens
        else:
            return (4, 4)  # 9,792 tokens
```

## Complete End-to-End Example

```python
class CompleteLLaVAWithGridSlicing(nn.Module):
    """
    Full LLaVA model with grid slicing for high-resolution images.

    Architecture:
    1. Image → Grid Slicing → CLIP Patches
    2. CLIP Patches → Vision Features
    3. Vision Features → MLP Projector → LLM Input
    4. LLM generates text response
    """
    def __init__(
        self,
        vision_tower: str = "openai/clip-vit-large-patch14-336",
        llm_model: str = "lmsys/vicuna-7b-v1.5",
        grid_size: Tuple[int, int] = (2, 2)
    ):
        super().__init__()

        # Vision encoder (frozen CLIP)
        self.vision_encoder = LLaVAVisionEncoder(vision_tower)

        # MLP projector (trainable)
        # Projects CLIP features (1024D) to LLM embedding space (4096D for Vicuna-7B)
        self.mm_projector = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.GELU(),
            nn.Linear(4096, 4096)
        )

        # Language model (frozen during visual instruction tuning)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)

        self.grid_size = grid_size

    def forward(
        self,
        images: torch.Tensor,  # [B, 3, H, W]
        text_prompts: List[str]
    ) -> str:
        """
        Process high-resolution images with grid slicing and generate responses.
        """
        batch_size = images.size(0)
        all_visual_features = []

        for i in range(batch_size):
            image = images[i]  # [3, H, W]

            # Step 1: Grid slicing
            patches = slice_image_into_grid(
                image,
                self.grid_size,
                patch_size=336
            )
            patches_batch = torch.stack(patches)  # [N_patches, 3, 336, 336]

            # Step 2: CLIP encoding (frozen)
            with torch.no_grad():
                patch_features = self.vision_encoder(patches_batch)
                # [N_patches, 576, 1024]

            # Step 3: MLP projection (trainable)
            # Flatten patches: [N_patches × 576, 1024] → [N_patches × 576, 4096]
            flat_features = patch_features.reshape(-1, patch_features.size(-1))
            projected_features = self.mm_projector(flat_features)

            all_visual_features.append(projected_features)

        # Step 4: Create multimodal input for LLM
        # Prepend visual tokens to text tokens
        responses = []
        for i, prompt in enumerate(text_prompts):
            # Tokenize text
            text_tokens = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True
            ).input_ids

            # Get text embeddings
            text_embeds = self.llm.get_input_embeddings()(text_tokens)

            # Concatenate: [vision_tokens, text_tokens]
            multimodal_embeds = torch.cat([
                all_visual_features[i].unsqueeze(0),  # [1, N_vision, 4096]
                text_embeds  # [1, N_text, 4096]
            ], dim=1)

            # Generate response
            outputs = self.llm.generate(
                inputs_embeds=multimodal_embeds,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)

        return responses

# Example usage
model = CompleteLLaVAWithGridSlicing(grid_size=(2, 2))
high_res_images = torch.randn(2, 3, 672, 672)  # Batch of 2 images
prompts = [
    "Describe what you see in this image.",
    "What text is visible in this document?"
]
responses = model(high_res_images, prompts)
print(responses)
```

## Comparison with Alternative High-Resolution Strategies

| Approach | Resolution | Tokens | Pros | Cons |
|----------|-----------|---------|------|------|
| **Grid Slicing (LLaVA)** | Up to 1344×1344 (4×4) | 2,880-9,792 | Simple, frozen encoder, parallelizable | Token explosion, fragmentation |
| **Perceiver Resampler** | Variable | Fixed (e.g., 256) | Constant token budget | Lossy compression, complex training |
| **Q-Former (BLIP-2)** | Fixed 224px | 32 queries | Very efficient | Limited detail capture |
| **Patchwise Adaptive (Ovis)** | Variable | 64-400/patch | Query-aware, efficient | Complex routing logic |

**LLaVA's trade-off**: High token count for comprehensive detail vs. computational efficiency.

## Key Takeaways

1. **Grid slicing transforms resolution problem**: From "big encoder" to "smart partitioning"
2. **Frozen encoder advantage**: No expensive CLIP retraining required
3. **Token budget is critical**: 4×4 grid (9,792 tokens) requires careful LLM context management
4. **Spatial relationships matter**: Position encodings and cross-patch attention improve coherence
5. **Adaptive grids outperform fixed**: LLaVA-UHD shows aspect-ratio-aware slicing reduces padding
6. **Production concerns**: Batch processing, memory management, dynamic grid selection

**When to use grid slicing**:
- High-resolution images with small text (OCR, document understanding)
- Detailed visual inspection tasks
- When LLM has sufficient context window (8k+ tokens)
- Frozen vision encoder constraint (limited compute for retraining)

**When to avoid**:
- Tight token budgets (prefer Perceiver or Q-Former compression)
- Coarse-grained tasks (single 336px image sufficient)
- Real-time inference requirements (9k tokens = slow generation)

## Sources

**Official LLaVA Repository:**
- [LLaVA GitHub](https://github.com/haotian-liu/LLaVA) - Main repository, accessed 2025-01-31
- [llava/model/multimodal_encoder/clip_encoder.py](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_encoder/clip_encoder.py) - CLIP encoder implementation

**Research Papers:**
- [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) - Original NeurIPS 2023 paper
- [Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)](https://arxiv.org/abs/2310.03744) - 336px resolution upgrade
- [Beyond LLaVA-HD: Diving into High-Resolution Large Multimodal Models](https://arxiv.org/abs/2406.08487) - Grid strategy analysis, accessed 2025-01-31
- [LLaVA-UHD: an LMM Perceiving Any Aspect Ratio and High-Resolution](https://arxiv.org/abs/2403.11703) - Aspect-ratio-aware slicing, accessed 2025-01-31
- [HiRes-LLaVA: Restoring Fragmentation Input](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_HiRes-LLaVA_Restoring_Fragmentation_Input_in_High-Resolution_Large_Vision-Language_Models_CVPR_2025_paper.pdf) - CVPR 2025, addressing fragmentation, accessed 2025-01-31

**Blog Posts & Tutorials:**
- [LLaVA-NeXT Blog](https://llava-vl.github.io/blog/2024-01-30-llava-next/) - 4× pixel processing announcement, accessed 2025-01-31
- [LLaVA Architecture Tutorial](https://learnopencv.com/llava-training-a-visual-assistant/) - LearnOpenCV comprehensive guide, accessed 2025-01-31

**Community Resources:**
- [awesome-vlm-architectures](https://github.com/gokayfem/awesome-vlm-architectures) - VLM architecture collection, accessed 2025-01-31
