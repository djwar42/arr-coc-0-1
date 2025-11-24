# Variable Rate Shading for Foveated VLM Token Allocation

## Overview

Variable Rate Shading (VRS) is a hardware-accelerated rendering technique that enables per-region control over shading rates, making it a natural fit for foveated vision-language models (VLMs). VRS allows different regions of the frame to be shaded at different frequencies—from supersampling (multiple shading operations per pixel) to coarse shading (one shading operation for multiple pixels). When applied to VLM token allocation, VRS concepts provide a hardware-inspired framework for dynamic level-of-detail (LOD) allocation based on relevance realization.

From [NVIDIA VRWorks - Variable Rate Shading](https://developer.nvidia.com/vrworks/graphics/variablerateshading) (accessed 2025-01-31):
- VRS is a Turing architecture feature that varies shading rate for different regions
- Provides seven shading rate options per 16×16 pixel region: 1×1 (full rate), 1×2, 2×1, 2×2, 2×4, 4×2, 4×4
- Enables fine-grained control: single triangle can be shaded at multiple rates
- Decouples shading rate from visibility rate—critical for maintaining quality

The parallel to VLM token allocation is direct: just as VRS allocates compute density based on visual importance, foveated VLMs allocate token budgets based on query-aware relevance. Both systems face the same fundamental tradeoff between quality and efficiency.

## Variable Rate Shading Hardware Architecture

### VRS Tier System

VRS hardware support comes in two tiers with progressively more sophisticated control:

**Tier 1 - Per-Draw Shading Rate:**
- Entire draw call uses uniform shading rate
- Coarsest level of control
- Simple API: set rate once, applies to all primitives
- Minimal overhead

**Tier 2 - Image-Based Shading Rate:**
- Shading rate controlled via special texture (shading rate image)
- Each texel in shading rate image controls 16×16 screen-space tile
- Enables arbitrary spatial patterns of shading density
- Foundation for foveated rendering

From [Eye Gaze Tells You Where to Compute (arXiv:2509.16476v1)](https://arxiv.org/html/2509.16476v1) (accessed 2025-01-31):
- GazeVLM uses gaze-driven ROI extraction for token allocation
- Reduces visual tokens by up to 93.1% (from 64 to 4.4 tokens)
- Total FLOPs reduced by 50% while maintaining quality
- Two-scale input (global + foveated) mimics fovea-periphery perception

The shading rate image concept directly maps to VLM relevance maps: a low-resolution "attention map" that determines token density across image patches.

### Coarse Pixel Shading Mechanics

VRS operates by grouping pixels into coarse pixel quads:

**1×1 shading (baseline):**
- One fragment shader invocation per pixel
- Full resolution, maximum quality
- Equivalent: 576 tokens for 24×24 patch grid (standard ViT)

**2×2 coarse shading:**
- One fragment shader invocation per 2×2 pixel block
- 4× reduction in shading cost
- Equivalent: 144 tokens for same area (75% reduction)

**4×4 coarse shading:**
- One fragment shader invocation per 4×4 pixel block
- 16× reduction in shading cost
- Equivalent: 36 tokens (93.75% reduction)

The visibility samples (pixel coverage) remain unchanged—only shading frequency varies. This is critical: VRS maintains edge antialiasing while reducing interior shading work, analogous to preserving patch boundaries while reducing internal token counts.

From [NVIDIA VRWorks VRS](https://developer.nvidia.com/vrworks/graphics/variablerateshading):
> "VRS allows the developer to control the shading rate without changing the visibility rate. The ability to decouple shading rate and visibility rate makes VRS more broadly applicable than techniques such as MRS and LMS."

For VLMs, this means: token sparsification doesn't compromise spatial resolution of the visual encoder's receptive fields, only the semantic resolution of processing depth.

### Shading Rate Image Construction

The shading rate image is the control surface for VRS:

**Structure:**
- Low-resolution texture (e.g., 1920×1080 screen → 120×68 rate image)
- Each texel encodes desired shading rate for 16×16 screen tile
- Updated per-frame or per-draw based on attention/gaze data
- Accessed by hardware during rasterization

**Encoding:**
- 2-bit per axis: 4 rates × 2 axes = 16 combinations
- Common format: R8_UINT with packed encoding
- Hardware applies combiner function (min, max, sum) when multiple rate sources conflict

**Construction pipeline for foveated rendering:**
1. Obtain gaze/attention heatmap (e.g., eye tracker, attention scores)
2. Compute distance from focal point for each tile
3. Map distance to shading rate via eccentricity function
4. Write shading rate image
5. Bind as special resource during rendering

From [GazeVLM - Eye Gaze for Efficient VLMs](https://arxiv.org/html/2509.16476v1):
> "We compute a gaze heatmap and extract a compact ROI that preserves attended areas... optionally combined with a downsampled global view... emulating human fovea-periphery perception."

The parallel to VLM token allocation is explicit: the shading rate image is analogous to a "token density map" that guides patch sampling or token pruning decisions.

## Foveated Rendering Principles

### Biological Foveal Vision

Human vision exhibits extreme non-uniformity:

**Foveal characteristics:**
- Central 2° of visual field
- Cone density: ~150,000/mm² (fovea centralis)
- Acuity: ~60 cycles per degree
- Color sensitivity: maximum
- Object recognition: high detail

**Peripheral characteristics:**
- Eccentricity > 10°
- Cone density: ~5,000/mm² (10° eccentricity)
- Acuity: ~10 cycles per degree
- Motion sensitivity: maximum
- Scene awareness: coarse layout

**Mapping to VLM token allocation:**
- Foveal region → high token density (64-400 tokens per patch)
- Peripheral region → low token density (4-16 tokens per patch)
- Eccentricity-based falloff determines allocation curve

From [NVIDIA VRWorks - VRSS](https://developer.nvidia.com/vrworks/graphics/variablerateshading):
> "Variable Rate Supersampling (VRSS) leverages VRS to apply supersampling in the region of interest... Foveated rendering is a technique where a region of the HMD screen is sampled at a higher shading rate."

### Gaze-Contingent Rendering

Gaze-contingent rendering dynamically repositions the high-quality region based on eye tracking:

**Fixed foveated rendering:**
- High-quality region at screen center
- No eye tracking required
- Assumption: user gazes at center (reasonable for VR/AR, poor for VLMs)
- Shading rate pattern: concentric rings of decreasing quality

**Dynamic foveated rendering:**
- High-quality region follows gaze point
- Requires eye tracking (Tobii, Varjo, Apple Vision Pro)
- Adapts to actual user attention
- Shading rate pattern: moves with gaze

From [GazeVLM findings](https://arxiv.org/html/2509.16476v1):
> "By extracting gaze-driven regions of interest (ROIs) and optionally combining them with a low-resolution global view, GazeVLM mimics fovea–periphery perception to cut redundant visual tokens while preserving task-relevant details."

For VLMs, this translates to:
- Fixed attention: center-crop or uniform token allocation (current standard)
- Query-driven attention: relevance realization determines token density
- Human-in-the-loop: gaze signals guide token allocation directly

### Eccentricity-Based Quality Reduction

The quality falloff function determines shading rate as a function of distance from gaze point:

**Polynomial falloff (typical):**
```
shading_rate(r) = base_rate × (1 + k × r²)

where:
- r = distance from gaze point (normalized)
- k = falloff coefficient (0.5 - 2.0 typical)
- base_rate = center shading rate
```

**VRS rate selection:**
```
if shading_rate < 1.5:  rate = 1×1  # Full quality
elif shading_rate < 3:  rate = 2×2  # Slight reduction
elif shading_rate < 6:  rate = 2×4  # Peripheral
else:                   rate = 4×4  # Far peripheral
```

**VLM token allocation equivalent:**
```
token_density(r) = max_tokens × exp(-λ × r)

where:
- max_tokens = 400 (foveal region)
- λ = decay rate (attention profile steepness)
- r = distance from relevance center

Discretized to token budgets:
if token_density > 200: allocate 400 tokens
elif token_density > 100: allocate 200 tokens
elif token_density > 50: allocate 100 tokens
else: allocate 64 tokens
```

The exponential decay matches visual psychophysics better than polynomial for large eccentricities, but polynomial is computationally cheaper for real-time VRS.

## VRS Application to VLM Token Allocation

### Attention-Driven Shading Rates

VRS concepts translate directly to VLM token budgets:

**Relevance realization → shading rate mapping:**

| Relevance Score | VRS Equivalent | Token Budget | Use Case |
|----------------|----------------|--------------|----------|
| > 0.8 | 1×1 supersampled | 400 tokens | Text, faces, query-critical objects |
| 0.6 - 0.8 | 1×1 full rate | 200 tokens | Salient objects, context objects |
| 0.4 - 0.6 | 2×2 coarse | 100 tokens | Background with structure |
| 0.2 - 0.4 | 2×4 coarse | 64 tokens | Peripheral context |
| < 0.2 | 4×4 coarse | 32 tokens | Far peripheral, redundant regions |

**Computation:**
Relevance scores come from:
1. Query-image cross-attention (Participatory knowing)
2. Visual saliency (Perspectival knowing)
3. Information density (Propositional knowing)
4. Opponent processing balances tensions (Compress ↔ Particularize)

From [GazeVLM architecture](https://arxiv.org/html/2509.16476v1):
> "GazeVLM uses the user's eye gaze to form an adaptive foveated input... cuts visual tokens from 64 to 4.4 by 93.1%... while achieving pairwise win rates of 53.4% against full-resolution baselines."

### Dynamic LOD with VRS Concepts

Level-of-detail allocation using VRS tile structure:

**Patch-based allocation (standard ViT: 16×16 patches):**
```python
# Image: 384×384 → 24×24 patches
# VRS tile: 16×16 pixels → maps to 1×1 patch exactly

def allocate_tokens_vrs_style(image, relevance_map):
    """
    VRS-inspired token allocation for VLM.

    Args:
        image: [B, 3, 384, 384]
        relevance_map: [B, 24, 24] # One score per patch

    Returns:
        token_budget: [B, 24, 24] # Tokens per patch
    """
    # Normalize relevance to [0, 1]
    relevance = (relevance_map - relevance_map.min()) / \
                (relevance_map.max() - relevance_map.min())

    # VRS-style rate selection
    token_budget = torch.zeros_like(relevance)

    # Foveal region: supersampling
    token_budget[relevance > 0.8] = 400

    # Perifoveal: full rate
    token_budget[(relevance > 0.6) & (relevance <= 0.8)] = 200

    # Near peripheral: 2×2 coarse
    token_budget[(relevance > 0.4) & (relevance <= 0.6)] = 100

    # Peripheral: 2×4 coarse
    token_budget[(relevance > 0.2) & (relevance <= 0.4)] = 64

    # Far peripheral: 4×4 coarse
    token_budget[relevance <= 0.2] = 32

    return token_budget
```

**Adaptive resampling:**
```python
def resample_patches_vrs(patches, token_budget):
    """
    Resample patches based on VRS-style token budget.

    Args:
        patches: [B, N, D] where N = 576 (24×24)
        token_budget: [B, 24, 24]

    Returns:
        resampled: [B, M, D] where M < N
    """
    resampled = []

    for i, budget in enumerate(token_budget.flatten()):
        patch = patches[:, i, :]

        if budget == 400:
            # Supersample: expand to 4 tokens
            resampled.extend([patch] * 4)
        elif budget == 200:
            # Full rate: 2 tokens
            resampled.extend([patch] * 2)
        elif budget == 100:
            # Keep 1 token
            resampled.append(patch)
        elif budget == 64:
            # Subsample: keep every other patch
            if i % 2 == 0:
                resampled.append(patch)
        else:  # budget == 32
            # Coarse subsample: keep every 4th patch
            if i % 4 == 0:
                resampled.append(patch)

    return torch.stack(resampled, dim=1)
```

This VRS-style approach provides hardware-inspired structure for token allocation decisions.

### VRS for Query-Aware Compression

Query-aware token allocation using VRS principles:

**Multi-pass VRS (inspired by Turing variable rate):**

Pass 1 - Coarse global understanding:
- Entire image at 4×4 coarse rate (36 tokens)
- Fast global scene understanding
- Identifies candidate regions for refinement

Pass 2 - Refined regions:
- High-relevance patches at 1×1 rate (200 tokens)
- Medium-relevance at 2×2 rate (100 tokens)
- Low-relevance remains at 4×4 (already processed)

Pass 3 - Foveal detail (query-critical only):
- Query-critical regions supersampled to 1×1×4 (400 tokens)
- Text, faces, small objects requiring fine detail

**Total token budget example:**
- Image: 384×384 = 576 patches at full resolution
- Coarse pass: 36 tokens (global context)
- Refined pass: 150 tokens (20 high-relevance patches × 2, 25 medium × 1)
- Foveal pass: 100 tokens (5 critical patches × 4)
- Total: 286 tokens vs 576 baseline (50.3% reduction)

From [GazeVLM results](https://arxiv.org/html/2509.16476v1):
> "GazeVLM reduces visual tokens by up to 93.1%, total tokens by up to 59.6%, and FLOPs by 50%, while keeping better answer quality relative to full-resolution baselines."

The VRS multi-pass structure naturally implements a coarse-to-fine processing strategy that aligns with Vervaekean relevance realization.

## Implementation Patterns

### DirectX 12 VRS API

DirectX 12 provides explicit VRS control:

**Tier 2 VRS setup:**
```cpp
// Create shading rate image
D3D12_RESOURCE_DESC rateImageDesc = {};
rateImageDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
rateImageDesc.Width = screenWidth / D3D12_SHADING_RATE_TILE_WIDTH;  // 16
rateImageDesc.Height = screenHeight / D3D12_SHADING_RATE_TILE_HEIGHT; // 16
rateImageDesc.Format = DXGI_FORMAT_R8_UINT;
rateImageDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

device->CreateCommittedResource(
    &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
    D3D12_HEAP_FLAG_NONE,
    &rateImageDesc,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    nullptr,
    IID_PPV_ARGS(&shadingRateImage)
);

// Populate shading rate image (compute shader)
// Each texel encodes one of:
// D3D12_SHADING_RATE_1X1 = 0x0
// D3D12_SHADING_RATE_1X2 = 0x1
// D3D12_SHADING_RATE_2X1 = 0x4
// D3D12_SHADING_RATE_2X2 = 0x5
// D3D12_SHADING_RATE_2X4 = 0x6
// D3D12_SHADING_RATE_4X2 = 0x9
// D3D12_SHADING_RATE_4X4 = 0xa

// Compute shader to generate foveated pattern
ComputeGazeFoveatedRateImage(gazeCenterX, gazeCenterY, shadingRateImage);

// Bind during rendering
commandList->RSSetShadingRateImage(shadingRateImage);

// Optional: set combiner to control interaction with per-draw rates
D3D12_SHADING_RATE_COMBINER combiners[2] = {
    D3D12_SHADING_RATE_COMBINER_MAX,  // Screen-space vs per-draw
    D3D12_SHADING_RATE_COMBINER_MIN   // Screen-space vs per-primitive
};
commandList->RSSetShadingRate(D3D12_SHADING_RATE_1X1, combiners);
```

**Gaze-driven rate image generation:**
```cpp
// Compute shader: Generate foveated VRS pattern
[numthreads(8, 8, 1)]
void GenerateFoveatedVRS(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint2 gazeFocus : CB_GAZE_CENTER,  // Normalized [0,1]
    RWTexture2D<uint> shadingRateImage : register(u0)
) {
    // Convert tile coordinate to screen space
    float2 tileCenter = (float2(dispatchThreadID.xy) + 0.5) /
                        float2(screenWidth / 16, screenHeight / 16);

    // Distance from gaze point
    float2 delta = tileCenter - gazeFocus;
    float dist = length(delta);

    // Eccentricity-based rate selection
    uint shadingRate;
    if (dist < 0.1) {
        shadingRate = D3D12_SHADING_RATE_1X1;  // Foveal: full detail
    } else if (dist < 0.25) {
        shadingRate = D3D12_SHADING_RATE_2X2;  // Perifoveal
    } else if (dist < 0.4) {
        shadingRate = D3D12_SHADING_RATE_2X4;  // Near peripheral
    } else {
        shadingRate = D3D12_SHADING_RATE_4X4;  // Far peripheral
    }

    shadingRateImage[dispatchThreadID.xy] = shadingRate;
}
```

### Vulkan VRS API

Vulkan provides equivalent VRS capabilities through extensions:

**VK_KHR_fragment_shading_rate setup:**
```cpp
// Check for VRS support
VkPhysicalDeviceFragmentShadingRatePropertiesKHR shadingRateProps = {};
shadingRateProps.sType =
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_PROPERTIES_KHR;

VkPhysicalDeviceProperties2 deviceProps2 = {};
deviceProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
deviceProps2.pNext = &shadingRateProps;
vkGetPhysicalDeviceProperties2(physicalDevice, &deviceProps2);

// Tile size typically 8×8 or 16×16
uint32_t tileWidth = shadingRateProps.minFragmentShadingRateAttachmentTexelSize.width;
uint32_t tileHeight = shadingRateProps.minFragmentShadingRateAttachmentTexelSize.height;

// Create shading rate attachment
VkImageCreateInfo rateImageInfo = {};
rateImageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
rateImageInfo.imageType = VK_IMAGE_TYPE_2D;
rateImageInfo.format = VK_FORMAT_R8_UINT;  // Typical format
rateImageInfo.extent = {
    screenWidth / tileWidth,
    screenHeight / tileHeight,
    1
};
rateImageInfo.usage = VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR |
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT;

vkCreateImage(device, &rateImageInfo, nullptr, &shadingRateImage);

// Update rate image via compute shader (similar to DX12)
UpdateFoveatedVRSPattern(gazePosition, shadingRateImage);

// Attach during rendering
VkRenderingFragmentShadingRateAttachmentInfoKHR shadingRateAttachment = {};
shadingRateAttachment.sType =
    VK_STRUCTURE_TYPE_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_INFO_KHR;
shadingRateAttachment.imageView = shadingRateImageView;
shadingRateAttachment.imageLayout =
    VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR;
shadingRateAttachment.shadingRateAttachmentTexelSize = {tileWidth, tileHeight};

VkRenderingInfo renderingInfo = {};
renderingInfo.pNext = &shadingRateAttachment;
// ... other rendering info

vkCmdBeginRendering(commandBuffer, &renderingInfo);
```

### PyTorch VLM Integration

Translating VRS concepts to PyTorch-based VLM inference:

**VRS-style token allocator:**
```python
import torch
import torch.nn.functional as F

class VRSTokenAllocator:
    """
    VRS-inspired token allocation for vision-language models.

    Implements hardware VRS concepts (shading rate image, tile-based
    allocation) for dynamic token budgets in VLM inference.
    """

    def __init__(
        self,
        tile_size: int = 16,  # Pixels per tile (like VRS)
        rates: dict = None    # Token budget per rate
    ):
        self.tile_size = tile_size
        self.rates = rates or {
            'supersample': 400,  # 1×1 supersampled
            'full': 200,         # 1×1 full rate
            'coarse_2x2': 100,   # 2×2 coarse
            'coarse_2x4': 64,    # 2×4 coarse
            'coarse_4x4': 32     # 4×4 coarse
        }

    def compute_relevance_map(
        self,
        image_features: torch.Tensor,
        query_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-patch relevance (analogous to generating VRS rate image).

        Args:
            image_features: [B, N, D] patch features
            query_embedding: [B, D] query features

        Returns:
            relevance: [B, H, W] relevance map
        """
        # Compute attention scores (Participatory knowing)
        scores = torch.matmul(
            image_features,
            query_embedding.unsqueeze(-1)
        ).squeeze(-1)  # [B, N]

        # Reshape to spatial grid
        H = W = int(image_features.shape[1] ** 0.5)
        relevance = scores.reshape(-1, H, W)

        # Normalize to [0, 1]
        relevance = (relevance - relevance.min()) / \
                    (relevance.max() - relevance.min() + 1e-8)

        return relevance

    def generate_rate_image(
        self,
        relevance: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate VRS-style shading rate image from relevance map.

        Args:
            relevance: [B, H, W] normalized relevance scores

        Returns:
            rate_image: [B, H, W] integer rate codes
        """
        rate_image = torch.zeros_like(relevance, dtype=torch.long)

        # Threshold-based rate assignment (like VRS)
        rate_image[relevance > 0.8] = 0  # supersample
        rate_image[(relevance > 0.6) & (relevance <= 0.8)] = 1  # full
        rate_image[(relevance > 0.4) & (relevance <= 0.6)] = 2  # 2×2
        rate_image[(relevance > 0.2) & (relevance <= 0.4)] = 3  # 2×4
        rate_image[relevance <= 0.2] = 4  # 4×4

        return rate_image

    def allocate_tokens(
        self,
        rate_image: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert rate image to token budgets.

        Args:
            rate_image: [B, H, W] rate codes

        Returns:
            token_budget: [B, H, W] tokens per patch
        """
        rate_to_tokens = torch.tensor([
            self.rates['supersample'],
            self.rates['full'],
            self.rates['coarse_2x2'],
            self.rates['coarse_2x4'],
            self.rates['coarse_4x4']
        ], device=rate_image.device)

        return rate_to_tokens[rate_image]

    def apply_foveated_allocation(
        self,
        patches: torch.Tensor,
        relevance: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full VRS-style pipeline: relevance → rate image → token allocation.

        Args:
            patches: [B, N, D] patch embeddings
            relevance: [B, H, W] relevance scores

        Returns:
            processed_patches: [B, M, D] with M < N
            token_budget: [B, H, W] allocation map
        """
        # Generate rate image (like VRS shading rate image)
        rate_image = self.generate_rate_image(relevance)

        # Allocate tokens
        token_budget = self.allocate_tokens(rate_image)

        # Resample patches based on budget
        processed = self._resample_patches(patches, token_budget)

        return processed, token_budget

    def _resample_patches(
        self,
        patches: torch.Tensor,
        token_budget: torch.Tensor
    ) -> torch.Tensor:
        """
        Resample patches according to token budget (VRS-style coarse shading).
        """
        B, N, D = patches.shape
        H = W = int(N ** 0.5)
        patches_2d = patches.reshape(B, H, W, D)

        resampled_list = []

        for b in range(B):
            patch_list = []
            for i in range(H):
                for j in range(W):
                    budget = token_budget[b, i, j].item()
                    patch = patches_2d[b, i, j]

                    if budget == self.rates['supersample']:
                        # Supersample: replicate 4×
                        patch_list.extend([patch] * 4)
                    elif budget == self.rates['full']:
                        # Full rate: replicate 2×
                        patch_list.extend([patch] * 2)
                    elif budget == self.rates['coarse_2x2']:
                        # Keep 1
                        patch_list.append(patch)
                    elif budget == self.rates['coarse_2x4']:
                        # Subsample: keep every other
                        if j % 2 == 0:
                            patch_list.append(patch)
                    else:  # coarse_4x4
                        # Coarse subsample: keep every 4th
                        if i % 2 == 0 and j % 2 == 0:
                            patch_list.append(patch)

            resampled_list.append(torch.stack(patch_list))

        return torch.stack(resampled_list)
```

**Usage example:**
```python
# Initialize VRS allocator
vrs_allocator = VRSTokenAllocator(tile_size=16)

# Process image with query
image = torch.randn(1, 3, 384, 384)  # Input image
patches = vision_encoder(image)      # [1, 576, 768]
query = text_encoder("What's in the image?")  # [1, 768]

# Compute relevance (like generating VRS rate image)
relevance = vrs_allocator.compute_relevance_map(patches, query)

# Apply foveated allocation
processed_patches, token_budget = vrs_allocator.apply_foveated_allocation(
    patches, relevance
)

# processed_patches now has variable token density
# Feed to language model
output = language_model(processed_patches, query)
```

## Performance Analysis

### Token Reduction Benchmarks

From [GazeVLM experimental results](https://arxiv.org/html/2509.16476v1):

**Qwen2.5-VL-3B model:**
| ρ (gaze mass) | Visual tokens | Total tokens | Win rate | GPT-4o score | FLOPs (G) |
|---------------|---------------|--------------|----------|--------------|-----------|
| Baseline | 64 | 100 | — | 3.98 | 267.6 |
| 0.05 | 4.4 (-93.1%) | 40.4 (-59.6%) | 53.4% | 4.22 | 132.8 (-50.4%) |
| 0.30 | 15.6 (-75.6%) | 51.8 (-48.2%) | 61.2% | 4.45 | 158.6 (-40.7%) |
| 0.50 | 23.4 (-63.4%) | 59.3 (-40.7%) | 68.2% | 4.83 | 176.1 (-34.2%) |

**Qwen2.5-VL-7B model:**
| ρ (gaze mass) | Visual tokens | Total tokens | Win rate | GPT-4o score | FLOPs (G) |
|---------------|---------------|--------------|----------|--------------|-----------|
| Baseline | 64 | 100 | — | 5.73 | 631.0 |
| 0.30 | 15.6 (-75.6%) | 51.8 (-48.2%) | 50.8% | 5.84 | 374.0 (-40.7%) |
| 0.60 | 27.8 (-56.6%) | 63.8 (-36.2%) | 53.1% | 5.95 | 438.6 (-30.5%) |

Key findings:
- Even aggressive pruning (93% token reduction) maintains quality
- Sweet spot: 50-75% token reduction with quality improvements
- Larger models (7B) require more context, peak at higher ρ
- FLOPs reduction roughly tracks token reduction (compute bound)

### VRS Performance Impact in VR/AR

From [NVIDIA VRWorks VRSS](https://developer.nvidia.com/vrworks/graphics/variablerateshading):

**Variable Rate Supersampling (VRSS):**
- Fixed foveated region at screen center
- Up to 8× supersampling in central region
- Adaptive mode: scales region based on GPU headroom
- Typical frame rate impact: +5-15% (quality improvement at similar performance)

**Dynamic foveated VRS with eye tracking:**
- Gaze-driven high-quality region
- 20-40% frame rate improvement vs uniform resolution
- Quality perception: indistinguishable from full resolution (studies show < 5% detection rate of artifacts)
- Latency: < 20ms gaze-to-update (Tobii, Varjo trackers)

**VLM inference parallels:**
- Fixed attention: center-crop (current practice)
- Query-driven: relevance realization (proposed)
- Gaze-driven: human-in-the-loop (GazeVLM)
- Token reduction: 50-95% depending on sparsity tolerance

### Quality vs Efficiency Tradeoffs

**VRS quality impact factors:**

1. **Shading rate granularity:**
   - Tier 1 (per-draw): coarse control, minimal overhead
   - Tier 2 (image-based): fine control, 1-2% overhead for rate image generation
   - Combiner functions: 0.5% overhead

2. **Peripheral quality degradation:**
   - 2×2 coarse: barely perceptible in motion
   - 4×4 coarse: noticeable but acceptable for far periphery
   - Temporal stability: important for avoiding flicker (filter rate image across frames)

3. **Eccentricity function tuning:**
   - Steep falloff: maximum savings, risk of visible boundary
   - Gradual falloff: smoother quality gradient, less savings
   - Optimal: 2-ring model (foveal, perifoveal, peripheral)

**VLM token allocation tradeoffs:**

From [GazeVLM analysis](https://arxiv.org/html/2509.16476v1):
> "For Qwen2.5-VL-3B... the win fraction is comparatively steady, the tie fraction rises, and the loss fraction falls. The rising tie indicates that enlarging the ROI primarily converts losses into ties."

Translation:
- Small token budgets: high risk, high reward (93% reduction, 53% win rate)
- Medium budgets: balanced (75% reduction, 61% win rate)
- Large budgets: diminishing returns (63% reduction, 68% win rate)
- Optimal point: depends on model capacity (3B peaks lower than 7B)

## Sources

**Web Research:**

- [NVIDIA VRWorks - Variable Rate Shading (VRS)](https://developer.nvidia.com/vrworks/graphics/variablerateshading) - NVIDIA Developer documentation on VRS hardware, VRSS, and foveated rendering (accessed 2025-01-31)

- [Eye Gaze Tells You Where to Compute: Gaze-Driven Efficient VLMs](https://arxiv.org/html/2509.16476v1) - arXiv:2509.16476v1 by Qinyu Chen and Jiawen Qi (accessed 2025-01-31) - GazeVLM architecture, experimental results on VOILA-COCO benchmark

**Additional References:**

- [Godot Engine - Variable Rate Shading Documentation](https://docs.godotengine.org/en/stable/tutorials/3d/variable_rate_shading.html) - Implementation guide for VRS in game engines (referenced search results, accessed 2025-01-31)

- DirectX 12 and Vulkan VRS API specifications - Implementation patterns for hardware-accelerated variable rate shading (referenced search results, accessed 2025-01-31)

- AMD FidelityFX Variable Shading - Open source header implementation for cross-platform VRS (referenced search results, accessed 2025-01-31)
