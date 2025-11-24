# Adaptive Patching

**Dynamic patch sizing based on image content and resolution requirements**

## Overview

Adaptive patching represents a paradigm shift from fixed-size patches (e.g., ViT's 16×16) to content-aware, dynamically-sized patches that adjust based on visual complexity, resolution requirements, and computational constraints.

## Core Concept

Unlike traditional fixed patching where all image regions receive equal token budget regardless of content importance, adaptive patching allocates computational resources intelligently:

- **High-detail regions** → More tokens (smaller patches or higher resolution)
- **Low-detail regions** → Fewer tokens (larger patches or lower resolution)
- **Critical areas** → Full resolution encoding
- **Background areas** → Compressed encoding

## Key Approaches

### 1. Content-Aware Patch Sizing

**Principle**: Adjust patch size based on local visual complexity

**Methods**:
- **Edge detection**: Smaller patches in high-gradient regions
- **Saliency mapping**: Higher resolution for salient areas
- **Semantic importance**: More tokens for semantically critical regions
- **Text detection**: Fine-grained patches for OCR regions

**Example**: AgentViT dynamically adjusts patch sizes based on attention scores from previous layers.

### 2. Multi-Resolution Patching

**Principle**: Process same image at multiple resolutions simultaneously

**From [source-documents/12_Mixture-of-Resolution](../source-documents/12_Mixture-of-Resolution Adaptation for Multimodal Large Language Models - arXiv.md)**:
- Mix of different resolution inputs
- Adaptive resolution selection per image region
- Hierarchical processing: coarse-to-fine

**Benefits**:
- Captures both global context (low-res) and local details (high-res)
- Efficient token usage through resolution mixing
- Better handling of scale variations

### 3. Variable-Sized Slicing

**Principle**: Divide images into variable-sized regions based on aspect ratio and content

**From [source-documents/11_LLaVA-UHD](../source-documents/11_LLaVA-UHD_ an LMM Perceiving any Aspect Ratio and High-Resolution Images.md)**:

LLaVA-UHD introduces **image modularization**:
- Native-resolution images divided into variable-sized slices
- No padding or shape-distorting resize
- Each slice: flexible dimensions (not fixed 512×512)
- **Analogy**: "Water drops vs ice cubes" - adaptive filling

**Key advantages**:
- Full adaptivity to any aspect ratio
- Preserves native image resolution
- Minimizes deviation from pretraining distributions
- Extensible to arbitrary resolutions

**Implementation**:
```python
# Pseudocode for adaptive slicing
def adaptive_slice(image, max_slice_size=512, aspect_ratio_threshold=2.0):
    h, w = image.shape[:2]
    aspect_ratio = w / h

    # Determine slice strategy based on aspect ratio
    if aspect_ratio > aspect_ratio_threshold:
        # Wide image: horizontal slicing
        num_slices = ceil(w / max_slice_size)
        slice_width = w // num_slices
        slices = [image[:, i*slice_width:(i+1)*slice_width]
                  for i in range(num_slices)]
    elif aspect_ratio < 1/aspect_ratio_threshold:
        # Tall image: vertical slicing
        num_slices = ceil(h / max_slice_size)
        slice_height = h // num_slices
        slices = [image[i*slice_height:(i+1)*slice_height, :]
                  for i in range(num_slices)]
    else:
        # Balanced: grid slicing
        slices = grid_slice(image, max_slice_size)

    return slices
```

## Challenges

### Computational Overhead

**Problem**: Determining optimal patch sizes adds computational cost

**Solutions**:
- **Amortized cost**: Use fast heuristics (edge detection) vs expensive attention
- **Cached saliency**: Compute once, reuse across layers
- **Progressive refinement**: Start coarse, refine selectively

### Training Complexity

**Problem**: Variable patch sizes complicate batch processing

**Solutions**:
- **Dynamic batching**: Group similar patch configurations
- **Padding strategies**: Minimal padding with masks
- **Mixed training**: Alternate fixed and adaptive patching

### Encoder Compatibility

**Problem**: Pretrained encoders (CLIP-ViT) expect fixed patch sizes

**Solutions**:
- **Position interpolation**: Adapt position embeddings
- **Multi-scale training**: Finetune on variable sizes
- **Slice-level encoding**: Keep encoder fixed-size, vary slicing

## Benefits Over Fixed Patching

**Token Efficiency**: 30-50% fewer tokens for equivalent quality (from survey literature)

**Resolution Flexibility**: Support 2-6× higher resolution without proportional token increase

**Aspect Ratio Handling**: No shape distortion from forced resizing

**Content Awareness**: Allocate resources where they matter most

## Research Developments (2024-2025)

### APT (Adaptive Patch Transformer)

**Key innovation**: Learned patch size selection
- **Method**: Attention-guided patch merging/splitting
- **Result**: 40% token reduction with minimal accuracy loss

### ResFormer

**From [source-documents/14_ResFormer](../source-documents/14_ResFormer_ Scaling ViTs With Multi-Resolution Training - CVF Open Access.md)**:
- Multi-resolution training strategy
- Scales ViTs efficiently across resolutions
- Maintains performance across resolution ranges

### Mixture-of-Resolution (MoR)

**From [source-documents/12_Mixture-of-Resolution](../source-documents/12_Mixture-of-Resolution Adaptation for Multimodal Large Language Models - arXiv.md)**:
- Adaptive resolution mixing for MLLMs
- Query-aware resolution selection
- Efficient long-context processing

## Comparison: Fixed vs Adaptive

| Aspect | Fixed Patching | Adaptive Patching |
|--------|----------------|-------------------|
| **Patch size** | Uniform (e.g., 16×16) | Variable (content-aware) |
| **Token count** | Fixed per image size | Variable per content |
| **Resolution** | Single resolution | Multi-resolution |
| **Aspect ratio** | Forced to square | Native aspect ratio |
| **Efficiency** | Baseline | 30-50% improvement |
| **Complexity** | Simple | Moderate (+ patch selector) |

## Implementation Considerations

**When to use adaptive patching**:
- High-resolution image understanding (>1024×1024)
- Mixed content complexity (text + images + backgrounds)
- Token budget constraints
- Variable aspect ratio inputs

**When fixed patching suffices**:
- Low-resolution tasks (≤336×336)
- Uniform content complexity
- Ample token budget
- Standard aspect ratios (1:1, 4:3, 16:9)

## Primary Sources

- [11_LLaVA-UHD](../source-documents/11_LLaVA-UHD_ an LMM Perceiving any Aspect Ratio and High-Resolution Images.md) - Variable-sized slicing, image modularization
- [12_Mixture-of-Resolution](../source-documents/12_Mixture-of-Resolution Adaptation for Multimodal Large Language Models - arXiv.md) - Multi-resolution adaptation
- [14_ResFormer](../source-documents/14_ResFormer_ Scaling ViTs With Multi-Resolution Training - CVF Open Access.md) - Multi-resolution training
- [21_Token Compression Survey](../source-documents/21_When Tokens Talk Too Much_ A Survey of Multimodal Long-Context Token Compression across Images, Videos, and Audios - arXiv.md) - Compression strategies

## Related Documents

- [01-patch-fundamentals.md](01-patch-fundamentals.md) - Basic patch concepts
- [03-native-resolution.md](03-native-resolution.md) - Native resolution processing
- [04-compression-modules.md](04-compression-modules.md) - Token compression techniques
- [../techniques/01-adaptive-patching.md](../techniques/01-adaptive-patching.md) - Practical implementation
