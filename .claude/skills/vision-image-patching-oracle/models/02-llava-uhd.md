# LLaVA-UHD: Variable-Sized Slice Modularization

**Native-resolution processing through adaptive image slicing**

## Overview

**LLaVA-UHD** (Ultra-High Definition) enables VLMs to process images in any aspect ratio and high resolution without shape distortion. Key innovation: **variable-sized slices** that adapt to native image dimensions.

**Paper**: *LLaVA-UHD: an LMM Perceiving any Aspect Ratio and High-Resolution Images* (ECCV 2024)
**Source**: `RESEARCH/Vision-Language Models Image Patching/NotebookLM_Sources/LLaVA-UHD.md`

---

## The Problem with Fixed Aspect Ratios

### Standard VLM Approach (LLaVA-1.5)

```
Input: 800×600 image (4:3 aspect ratio)
↓
Resize to 336×336 (force 1:1)  ← Shape distortion!
↓
Divide into 24×24 patches
↓
576 tokens
```

**Problems:**
1. **Shape distortion**: Stretching/squashing degrades fine-grained tasks
2. **Information loss**: Resizing blurs details
3. **Fixed resolution**: Cannot exceed 336×336 without massive token count

**Impact:**
- OCR accuracy drops (text becomes blurry/distorted)
- Object counting fails (shapes change)
- Small object detection degrades

---

## LLaVA-UHD Solution: Image Modularization

### Core Innovation

**Variable-sized slices** that preserve native aspect ratio

```
Input: 672×1008 image (2:3 aspect ratio)
↓
Divide into 336×336 slices (variable count, no distortion!)
↓
Slice 1 [0:336, 0:336]    │ Slice 2 [336:672, 0:336]
Slice 3 [0:336, 336:672]  │ Slice 4 [336:672, 336:672]
Slice 5 [0:336, 672:1008] │ Slice 6 [336:672, 672:1008]
↓
6 slices × 576 tokens/slice = 3456 tokens (before compression)
```

**Key Principle**: Adapt number of slices to image dimensions, not force-resize

---

## Three-Component Architecture

### 1. Image Modularization Strategy

**Divide native-resolution images into variable-sized 336×336 slices**

```python
def modularize_image(image, slice_size=336):
    """
    Divide image into variable number of slices
    """
    H, W = image.shape[:2]

    # Calculate number of slices needed
    n_h = math.ceil(H / slice_size)  # Vertical slices
    n_w = math.ceil(W / slice_size)  # Horizontal slices

    slices = []
    for i in range(n_h):
        for j in range(n_w):
            # Extract slice (may be smaller at edges)
            slice_h = min(slice_size, H - i * slice_size)
            slice_w = min(slice_size, W - j * slice_size)

            slice = image[i*slice_size:i*slice_size+slice_h,
                         j*slice_size:j*slice_size+slice_w]

            # Pad if necessary to 336×336
            slice = pad_to_size(slice, slice_size)
            slices.append(slice)

    return slices, (n_h, n_w)
```

**Analogy from Paper:**
> "Using water drops vs ice cubes to fill variable-sized glasses"
> - **Ice cubes** (fixed aspect ratios): Must force-fit, leaves gaps
> - **Water drops** (variable slices): Perfectly adapts to any container

**Advantages:**
- **Full adaptivity**: Works for any aspect ratio
- **No padding waste**: Minimal dead space
- **Preserves resolution**: No downsampling needed
- **Guaranteed minimal deviation**: Stays close to ViT pretraining distribution

---

### 2. Compression Module

**Reduce tokens per slice from 576 → 144 (4× compression)**

```python
class CompressionLayer(nn.Module):
    """Condense visual tokens"""

    def __init__(self, input_dim=1024, output_dim=1024, compression=4):
        super().__init__()
        # Spatial pooling or learned compression
        self.pool = nn.AdaptiveAvgPool1d(576 // compression)

    def forward(self, slice_tokens):
        # slice_tokens: [B, 576, 1024]
        # Pool spatially: 576 → 144 tokens
        compressed = self.pool(slice_tokens.transpose(1,2)).transpose(1,2)
        return compressed  # [B, 144, 1024]
```

**Why Compression?**
- **Memory**: 6 slices × 576 tokens = 3456 tokens (too many for LLM)
- **Efficiency**: 6 slices × 144 tokens = 864 tokens (manageable)
- **Quality**: 4× reduction preserves most information

**Methods:**
- **Spatial pooling**: Average or max over patches
- **Learned compression**: Trainable module
- **Cross-attention**: Query-based selection

---

### 3. Spatial Schema

**Organize slice tokens to inform LLM of spatial layout**

```python
def create_spatial_schema(slices, grid_shape):
    """
    Add position markers to indicate slice locations
    """
    n_h, n_w = grid_shape
    organized_tokens = []

    for i in range(n_h):
        for j in range(n_w):
            slice_idx = i * n_w + j

            # Add spatial markers
            row_marker = f"<row_{i}>"
            col_marker = f"<col_{j}>"

            # Organize: [row marker] [col marker] [slice tokens]
            organized_tokens.extend([
                row_marker,
                col_marker,
                slices[slice_idx]  # 144 compressed tokens
            ])

    return organized_tokens
```

**Example Output:**
```
<row_0><col_0> [slice_0_tokens] <row_0><col_1> [slice_1_tokens]
<row_1><col_0> [slice_2_tokens] <row_1><col_1> [slice_3_tokens]
<row_2><col_0> [slice_4_tokens] <row_2><col_1> [slice_5_tokens]
```

**Why Spatial Schema?**
- **Positional awareness**: LLM knows which slice is where
- **Spatial reasoning**: Enables "top-left corner" or "bottom row" queries
- **Slice coordination**: Connect information across slices

---

## Complete Forward Pass

```python
class LLaVAUHD(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = CLIPViT(patch_size=14, embed_dim=1024)
        self.compression = CompressionLayer(compression=4)
        self.llm = Llama3_8B()

    def forward(self, image, text_prompt):
        # 1. Image Modularization
        slices, grid_shape = self.modularize_image(image, slice_size=336)
        # slices: List of [336, 336, 3], length = n_h × n_w

        # 2. Encode each slice
        slice_tokens = []
        for slice in slices:
            tokens = self.vision_encoder(slice)  # [1, 576, 1024]
            compressed = self.compression(tokens)  # [1, 144, 1024]
            slice_tokens.append(compressed)

        # 3. Spatial schema organization
        organized_tokens = self.create_spatial_schema(slice_tokens, grid_shape)

        # 4. Concatenate with text
        text_tokens = self.tokenize(text_prompt)
        combined = torch.cat([organized_tokens, text_tokens], dim=1)

        # 5. LLM processing
        output = self.llm(combined)
        return output
```

---

## Resolution Support

### Flexible Scaling

**LLaVA-1.5 (Fixed):**
```
336×336 only → 576 tokens
Anything higher requires downsampling
```

**LLaVA-UHD (Variable):**
```
336×336   → 1 slice  → 144 tokens
672×336   → 2 slices → 288 tokens
672×1008  → 6 slices → 864 tokens
1008×1344 → 12 slices → 1728 tokens
```

**Max Resolution**: Limited only by LLM context length
```
If LLM supports 8K tokens, and each slice = 144 tokens:
Max slices = 8000 / 144 ≈ 55 slices
Supports up to ~2500×2500 images!
```

---

## Performance Gains

### Benchmark Results (from paper)

**TextVQA (OCR-heavy task):**
- LLaVA-1.5 (336×336): 58.2% accuracy
- LLaVA-UHD (672×1008): **63.9% accuracy** (+5.7 points)

**POPE (Object hallucination):**
- LLaVA-1.5 (336×336): 85.9% F1
- LLaVA-UHD (672×1008): **88.9% F1** (+3.0 points)

**Key Finding**: Performance improves with extreme aspect ratios
- Standard 1:1 images: Moderate improvement
- Extreme 3:1 or 1:3 images: Large improvement (shape distortion eliminated)

---

## Comparison to Other Approaches

### vs Standard LLaVA

| Aspect | LLaVA-1.5 | LLaVA-UHD |
|--------|-----------|-----------|
| **Aspect ratio** | Force 1:1 | Native (any) |
| **Max resolution** | 336×336 | Flexible (e.g., 672×1008) |
| **Token count** | 576 fixed | Variable (144-1728+) |
| **Fine-grained** | Limited | Strong (no blur) |

### vs GPT-4V

**GPT-4V approach** (inferred from behavior):
```
High-res mode: Divide into 512×512 slices
Issue: Fixed slice boundaries cause systematic flaws
```

**LLaVA-UHD improvement**:
- Flexible slice sizes
- Optimal alignment with ViT training resolution
- Compression module for efficiency

**Paper Result**: LLaVA-UHD outperforms models trained with 2-3 orders of magnitude more data

---

## Implementation Considerations

### Training Strategy

**Stage 1**: Train on standard 336×336 images
- Learn basic vision-language alignment
- Efficient, fast convergence

**Stage 2**: Fine-tune with multi-slice images
- Introduce variable resolutions
- Train compression module
- Learn spatial schema understanding

### Slice Size Selection

**Why 336×336?**
1. **ViT compatibility**: Close to 224×224 pretraining resolution
2. **GPU memory**: Fits in memory with batch size > 1
3. **Token efficiency**: 576 → 144 tokens is reasonable

**Alternatives:**
- Smaller (224×224): More slices, but closer to ViT pretraining
- Larger (448×448): Fewer slices, but OOD for ViT

### Compression Ratio

**Trade-off:**
- **4× compression** (576→144): Good balance, used in paper
- **8× compression** (576→72): More efficient, some quality loss
- **No compression** (576→576): Best quality, impractical for multi-slice

---

## Systematic Investigation of Visual Encoding

### Paper Experiment: GPT-4V Flaws

**Finding**: GPT-4V struggles with basic counting tasks
**Hypothesis**: Flawed visual encoding strategy

**Controlled Test:**
```
Synthetic images: Varying number of circles
Position circles systematically across image
Query: "How many circles?"
```

**Result**: Counting accuracy depends on circle positions
- Some positions: 100% accurate
- Other positions: Systematic undercounting

**Root Cause**: Fixed 512×512 slice boundaries
- Objects straddling boundaries get split
- LLM double-counts or misses them

**LLaVA-UHD Solution**: Flexible slicing + compression avoids hard boundaries

---

## Code Example

### Basic Usage

```python
from llava_uhd import LLaVAUHD

# Initialize model
model = LLaVAUHD.from_pretrained("llava-uhd-v1.5-7b")

# Load high-resolution image (any aspect ratio!)
image = load_image("photo_672x1008.jpg")  # 2:3 aspect ratio

# Query
prompt = "What text appears in the image?"

# Inference
response = model.generate(image, prompt)
print(response)
# "The image contains text saying 'Welcome to the Conference'..."
```

### Custom Slice Configuration

```python
# Adjust slice size for different GPU memory
model = LLaVAUHD(slice_size=448, compression_ratio=8)

# Very high resolution
large_image = load_image("diagram_2000x3000.png")
response = model.generate(large_image, prompt)
```

---

## Key Innovations Summary

### 1. Variable-Sized Slices
**Principle**: Adapt slice count to image, not force aspect ratio
**Benefit**: No shape distortion, preserves fine details

### 2. Compression Module
**Principle**: Reduce tokens per slice for efficiency
**Benefit**: Supports multi-slice images without context explosion

### 3. Spatial Schema
**Principle**: Explicit position markers for slice organization
**Benefit**: Enables spatial reasoning across slices

---

## Limitations & Future Work

### Current Limitations

**Token Budget**: Still grows with resolution
```
1008×1344 image = 12 slices × 144 tokens = 1728 tokens
Still expensive for very large images
```

**Slice Coordination**: LLM must integrate information across slices
- Requires good spatial schema understanding
- May struggle with objects spanning many slices

**Compression Loss**: 4× reduction loses some fine-grained detail
- Acceptable for most tasks
- May impact tasks requiring pixel-level precision

### Future Directions

**Adaptive Slice Sizing**: Variable slice sizes based on content
- Large slices for simple regions
- Small slices for detailed regions

**Hierarchical Slicing**: Multi-resolution pyramid
- Coarse slices for global context
- Fine slices for detailed regions

**Query-Aware Slicing**: Different slicing for different questions
- "What color?" → Coarse slices
- "Read the text" → Fine slices on text regions

---

## Related Documentation

- **[architecture/03-native-resolution.md](../architecture/03-native-resolution.md)** - Native resolution principles
- **[architecture/04-compression-modules.md](../architecture/04-compression-modules.md)** - Token compression
- **[architecture/05-spatial-encoding.md](../architecture/05-spatial-encoding.md)** - Position encoding
- **[techniques/02-variable-sized-slices.md](../techniques/02-variable-sized-slices.md)** - Slicing methods
- **[comparisons/02-resolution-strategies.md](../comparisons/02-resolution-strategies.md)** - Resolution handling

---

**Key Takeaway**: LLaVA-UHD demonstrates that flexible image modularization enables native-resolution processing without shape distortion, significantly improving fine-grained visual understanding.
