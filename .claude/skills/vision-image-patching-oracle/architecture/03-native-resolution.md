# Native Resolution Processing

**Processing images at their original aspect ratio and resolution without shape distortion**

## Overview

Native resolution processing preserves the original image dimensions and aspect ratio throughout the visual encoding pipeline, avoiding the information loss and distortion caused by forced resizing or padding to fixed shapes.

## The Problem with Fixed Resolution

### Traditional Approach (LLaVA 1.5, Early GPT-4V)

**Standard pipeline**:
1. Receive image (e.g., 1920×1080, 16:9 aspect ratio)
2. Resize to fixed square (e.g., 336×336, 1:1)
3. Process through ViT encoder
4. Generate visual tokens

**Problems**:
- **Shape distortion**: 16:9 → 1:1 squashes or stretches content
- **Information loss**: Downsampling loses fine details
- **Blur**: Small text becomes unreadable
- **Hallucinations**: Model "guesses" distorted content

**From [source-documents/11_LLaVA-UHD](../source-documents/11_LLaVA-UHD_ an LMM Perceiving any Aspect Ratio and High-Resolution Images.md)**:

> "Most existing LMMs perceive images in a fixed aspect ratio (i.e., 1:1) and a low resolution (i.e., 224×224). The compromise to this simplified setting typically leads to shape distortion and blur of image contents... The issue also exacerbates hallucination problems, since models can only learn to make best guesses to blurred images."

## LLaVA-UHD: Image Modularization Strategy

### Core Innovation

**Image modularization**: Divide native-resolution images into variable-sized slices that preserve original aspect ratio and resolution.

### Three Key Components

#### 1. Variable-Sized Slicing

**Principle**: Adaptive slice generation based on image dimensions

**Process**:
```python
# Conceptual implementation
def modularize_image(image, base_size=336, max_slices=9):
    """
    Divide image into variable-sized slices preserving aspect ratio

    Args:
        image: Original image (H×W×3)
        base_size: Base resolution for each slice
        max_slices: Maximum number of slices

    Returns:
        slices: List of image slices
        spatial_schema: Position information for each slice
    """
    h, w = image.shape[:2]
    aspect_ratio = w / h

    # Determine optimal slicing strategy
    if aspect_ratio > 1.5:  # Wide image
        num_h_slices = 1
        num_w_slices = min(ceil(w / base_size), max_slices)
    elif aspect_ratio < 0.67:  # Tall image
        num_h_slices = min(ceil(h / base_size), max_slices)
        num_w_slices = 1
    else:  # Balanced
        num_h_slices = min(ceil(sqrt(aspect_ratio * max_slices)), max_slices)
        num_w_slices = min(ceil(max_slices / num_h_slices), max_slices)

    # Create slices
    slice_h = h // num_h_slices
    slice_w = w // num_w_slices

    slices = []
    spatial_schema = []
    for i in range(num_h_slices):
        for j in range(num_w_slices):
            slice_img = image[i*slice_h:(i+1)*slice_h,
                              j*slice_w:(j+1)*slice_w]
            slices.append(slice_img)
            spatial_schema.append((i, j))  # Position in grid

    return slices, spatial_schema
```

**Key properties**:
- **No padding**: Slices match content boundaries
- **No distortion**: Preserve original aspect ratio
- **Flexible sizing**: Adapt to any resolution
- **Minimal deviation**: Stay close to pretraining distribution

**Analogy** (from LLaVA-UHD paper):
> "Water drops vs ice cubes in filling variable-sized glasses"
- **Ice cubes** (fixed slices): Force-fit, gaps, overflow
- **Water drops** (variable slices): Perfect filling, no waste

#### 2. Compression Module

**Purpose**: Reduce token count from each slice

**From [source-documents/11_LLaVA-UHD](../source-documents/11_LLaVA-UHD_ an LMM Perceiving any Aspect Ratio and High-Resolution Images.md)**:

**Architecture**:
- Input: Visual tokens from ViT encoder (e.g., 576 tokens for 336×336 slice)
- Module: Transformer layers with learned compression
- Output: Compressed tokens (e.g., 64-144 tokens)

**Compression ratio**: 4×-9× reduction

**Benefits**:
- Reduces LLM computational cost (quadratic in token count)
- Maintains critical visual information
- Enables higher resolution support

#### 3. Spatial Schema

**Purpose**: Inform LLM about slice positions in original image

**Implementation**:
```python
# Spatial schema encoding
def encode_spatial_schema(slices, spatial_schema):
    """
    Add position information to slice tokens

    Args:
        slices: List of visual token sequences
        spatial_schema: List of (row, col) positions

    Returns:
        organized_tokens: Tokens with spatial encoding
    """
    # Method 1: Position embeddings
    for tokens, (row, col) in zip(slices, spatial_schema):
        pos_embed = learned_position_embedding(row, col)
        tokens = tokens + pos_embed

    # Method 2: Special position tokens
    for tokens, (row, col) in zip(slices, spatial_schema):
        pos_token = create_position_token(row, col)
        tokens = concatenate([pos_token, tokens])

    # Method 3: Spatial attention bias
    # Add bias to attention scores based on relative positions

    return organized_tokens
```

**Importance**: Without spatial schema, LLM cannot understand:
- Relative positions of slices
- Reading order (left-to-right, top-to-bottom)
- Spatial relationships across slices
- Global image structure

## Benefits of Native Resolution

### 1. No Shape Distortion

**Example**: 16:9 image (1920×1080)
- **Traditional**: Resize to 336×336 → severe horizontal squashing
- **Native resolution**: Split into 3 slices (640×1080 each) → preserves aspect ratio

### 2. High-Resolution Support

**From [source-documents/11_LLaVA-UHD](../source-documents/11_LLaVA-UHD_ an LMM Perceiving any Aspect Ratio and High-Resolution Images.md)**:

> "LLaVA-UHD built on LLaVA-1.5 336×336 supports 6× larger (i.e., 672×1008) resolution images, and achieves 5.7 accuracy improvement on TextVQA."

**Resolution scaling**:
- Base model: 336×336 (1 slice)
- Native resolution: 672×1008 (6 slices)
- Effective resolution: 6× increase
- Token increase: ~2-3× (due to compression)

### 3. Reduced Hallucinations

**Mechanism**: Clear, undistorted visual input → less guessing

**Evidence**: 3.0 accuracy improvement on POPE benchmark (measuring hallucination)

### 4. Better OCR and Fine-Grained Understanding

**Tasks benefiting most**:
- Text recognition (TextVQA)
- Small object detection
- Detailed spatial reasoning
- Multi-panel images (comics, diagrams)

## Challenges and Solutions

### Challenge 1: Computational Cost

**Problem**: More slices → more tokens → higher cost

**Solutions**:
- **Compression module**: 4-9× token reduction per slice
- **Selective processing**: Process only relevant slices for query
- **Hierarchical approach**: Low-res overview + high-res details

### Challenge 2: Encoder Out-of-Distribution

**Problem**: Pretrained ViT expects 224×224 or 336×336, not variable slices

**Solution** (from LLaVA-UHD):
- Keep slice sizes close to pretraining distribution
- Use slices of ~336×336 (within training range)
- Minimal position embedding interpolation

### Challenge 3: Spatial Reasoning Across Slices

**Problem**: LLM must understand relationships between slices

**Solution**:
- **Spatial schema**: Explicit position encoding
- **Global token**: Add overview image token
- **Cross-slice attention**: Enable attention across slice boundaries

## Implementation Approaches

### Approach 1: Fixed-Size Slices (GPT-4V)

**Strategy**: Divide into fixed 512×512 slices

**Pros**: Simple, predictable token count
**Cons**: Padding needed, aspect ratio not preserved within slices

### Approach 2: Variable-Sized Slices (LLaVA-UHD)

**Strategy**: Adaptive slicing based on aspect ratio

**Pros**: No padding, aspect ratio preserved, efficient
**Cons**: Variable token count, more complex batching

### Approach 3: Hybrid (Emerging)

**Strategy**: Combine fixed encoder with variable slicing

**Pros**: Balance between simplicity and flexibility
**Cons**: Requires careful engineering

## Performance Comparison

**From [source-documents/11_LLaVA-UHD](../source-documents/11_LLaVA-UHD_ an LMM Perceiving any Aspect Ratio and High-Resolution Images.md)**:

| Benchmark | LLaVA-1.5 336×336 | LLaVA-UHD (native resolution) | Improvement |
|-----------|-------------------|-------------------------------|-------------|
| TextVQA | 58.2 | 63.9 | +5.7 |
| POPE | 85.9 | 88.9 | +3.0 |
| GQA | 62.0 | 63.2 | +1.2 |
| VQA-v2 | 78.5 | 79.3 | +0.8 |

**Key insight**: Larger improvements on fine-grained tasks (TextVQA, POPE)

## Design Choices

### Slice Size Selection

**Tradeoff**: Smaller slices → more tokens → higher cost

**Recommendations**:
- **336×336**: Sweet spot for CLIP-ViT pretrained models
- **256×256**: More efficient, slight quality loss
- **512×512**: Higher quality, 4× more tokens

### Compression Ratio

**Tradeoff**: Higher compression → fewer tokens → information loss

**Recommendations**:
- **4× compression**: Conservative, minimal loss
- **6× compression**: Balanced (LLaVA-UHD default)
- **9× compression**: Aggressive, for efficiency priority

### Maximum Slices

**Tradeoff**: More slices → higher resolution → more cost

**Recommendations**:
- **4-6 slices**: Typical use cases
- **9-12 slices**: High-resolution priority (documents, detailed scenes)
- **1-2 slices**: Efficiency priority

## Primary Sources

- [11_LLaVA-UHD](../source-documents/11_LLaVA-UHD_ an LMM Perceiving any Aspect Ratio and High-Resolution Images.md) - Complete native resolution strategy
- [08_Design Choices for Context Length](../source-documents/08_Design Choices for Extending the Context Length of Visual Language Models - OpenReview.md) - Context extension strategies
- [04_Breaking Resolution Curse](../source-documents/04_Breaking resolution curse of vision-language models - Hugging Face.md) - Resolution challenges

## Related Documents

- [02-adaptive-patching.md](02-adaptive-patching.md) - Adaptive patch sizing
- [04-compression-modules.md](04-compression-modules.md) - Token compression details
- [05-spatial-encoding.md](05-spatial-encoding.md) - Spatial schema implementation
- [../models/02-llava-uhd.md](../models/02-llava-uhd.md) - Complete LLaVA-UHD system
