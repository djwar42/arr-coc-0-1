# Patching Approaches Compared

**A comprehensive comparison of image patching strategies in VLMs**

## The Three Paradigms

### Fixed Patching
**Uniform grid division, predictable token count**

### Adaptive Patching
**Content-aware sizing, variable token allocation**

### Native Resolution
**Flexible slicing, no shape distortion**

---

## Quick Comparison Table

| Approach | Token Count | Aspect Ratio | Complexity | Best For |
|----------|-------------|--------------|------------|----------|
| **Fixed (ViT)** | Constant | Force 1:1 | Low | Standard images, efficiency |
| **Adaptive (APT)** | Variable | Force 1:1 | High | Complex scenes, speed |
| **Native (LLaVA-UHD)** | Variable | Preserved | Medium | Fine-grained, OCR, varied ratios |

---

## Detailed Comparison

### 1. Fixed Patching (ViT Standard)

**Models**: Original ViT, CLIP, LLaVA-1.5, most VLMs

**Mechanism:**
```
Input: Any image
↓
Resize to 336×336 (force square)
↓
Divide into 24×24 grid
↓
Result: 576 tokens (always)
```

**Characteristics:**

✅ **Advantages:**
- Simple, predictable implementation
- Constant token count (easy batching)
- Well-understood training dynamics
- Fast, efficient processing
- Proven effective across many tasks

❌ **Disadvantages:**
- Shape distortion on non-square images
- Inefficient token allocation (sky = face = text)
- Limited resolution support (downsampling required)
- Blurs fine-grained details

**Best Use Cases:**
- General-purpose vision-language tasks
- Batch processing with consistent token counts
- Applications where speed matters more than fine detail
- Standard datasets (ImageNet, COCO: mostly ~square images)

**Token Count Examples:**
```
224×224 image (patch=16) → 196 tokens
336×336 image (patch=14) → 576 tokens
448×448 image (patch=16) → 784 tokens
```

**Reference**: `Vision transformer: To discover the "four secrets" of image patches`

---

### 2. Adaptive Patching (Content-Aware)

**Models**: APT, AgentViT, AdaViT, FastViT

**Mechanism:**
```
Input: Image
↓
Analyze content complexity (saliency map)
↓
Allocate patches:
  - Simple regions (sky) → Large patches (32×32)
  - Complex regions (face) → Small patches (8×8)
↓
Result: Variable tokens (150-500 typical)
```

**Characteristics:**

✅ **Advantages:**
- Efficient token allocation (more tokens where needed)
- 2-4× speedup with minimal accuracy loss
- Preserves detail in important regions
- Reduces redundancy in uniform regions
- Computationally efficient

❌ **Disadvantages:**
- Complex implementation (saliency scoring, RL policies)
- Irregular structure (complicates position encoding)
- Training overhead (must learn allocation policy)
- Variable token counts (batching challenges)
- Still requires aspect ratio forcing

**Best Use Cases:**
- Speed-critical applications (real-time, edge devices)
- Images with high spatial redundancy (large sky/background regions)
- Resource-constrained environments
- Applications where 90-95% of full accuracy is acceptable

**Token Count Examples:**
```
Simple image (mostly sky) → 150 tokens (4× fewer)
Complex image (crowded scene) → 500 tokens (similar to fixed)
Average across dataset → 300 tokens (2× reduction)
```

**Key Innovation from APT (2024):**
- Multiple patch sizes per image (e.g., 8×8, 16×16, 32×32)
- Saliency-guided allocation
- Maintains accuracy while reducing compute

**Reference**: `Accelerating Vision Transformers with Adaptive Patch Sizes` (arXiv 2510.18091)

---

### 3. Native-Resolution Processing

**Models**: LLaVA-UHD, Ovis 2.5, DeepSeek-OCR (with Gundam tiling)

**Mechanism:**
```
Input: 672×1008 image (native aspect ratio)
↓
Divide into variable-sized 336×336 slices
↓
Encode each slice → Compress → Organize spatially
↓
Result: 6 slices × 144 tokens = 864 tokens
```

**Characteristics:**

✅ **Advantages:**
- Preserves aspect ratio (no shape distortion)
- Handles any resolution flexibly
- Excellent for fine-grained tasks (OCR, counting, small objects)
- No information loss from resizing
- Scalable to very high resolutions

❌ **Disadvantages:**
- Variable token counts (batching complexity)
- Slice coordination overhead
- Higher token counts for large images
- Requires spatial schema understanding
- More complex training

**Best Use Cases:**
- OCR and document understanding
- Counting objects (no shape distortion)
- Extreme aspect ratios (panoramas, vertical documents)
- Fine-grained visual details
- High-resolution image analysis

**Token Count Examples:**
```
336×336 image   → 144 tokens (1 slice, compressed)
672×1008 image  → 864 tokens (6 slices, compressed)
1008×1344 image → 1728 tokens (12 slices, compressed)
```

**Key Innovations:**
- **LLaVA-UHD**: Variable-sized slices + compression + spatial schema
- **Ovis 2.5**: Native resolution processing + Visual Embedding Table
- **DeepSeek-OCR**: 16× optical compression + Gundam tiling

**Reference**: `LLaVA-UHD: an LMM Perceiving any Aspect Ratio and High-Resolution Images`

---

## Feature Matrix

### Resolution Handling

| Approach | 224×224 | 336×336 | 672×1008 | 2000×3000 |
|----------|---------|---------|----------|-----------|
| **Fixed** | ✅ Native | ✅ Native | ⚠️ Downsample | ❌ Too large |
| **Adaptive** | ✅ Efficient | ✅ Efficient | ⚠️ Downsample | ❌ Too large |
| **Native** | ✅ 1 slice | ✅ 1 slice | ✅ 6 slices | ✅ ~36 slices |

### Aspect Ratio Support

| Approach | 1:1 (Square) | 2:1 (Wide) | 3:4 (Tall) |
|----------|--------------|------------|------------|
| **Fixed** | ✅ Perfect | ⚠️ Distortion | ⚠️ Distortion |
| **Adaptive** | ✅ Perfect | ⚠️ Distortion | ⚠️ Distortion |
| **Native** | ✅ Perfect | ✅ Perfect | ✅ Perfect |

### Task Performance

| Approach | Classification | OCR | Counting | General VQA |
|----------|----------------|-----|----------|-------------|
| **Fixed** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Adaptive** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Native** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

### Computational Efficiency

| Approach | Encoding Speed | Memory Usage | LLM Processing |
|----------|----------------|--------------|----------------|
| **Fixed** | ⭐⭐⭐ Fast | ⭐⭐⭐ Low | ⭐⭐⭐ Fast |
| **Adaptive** | ⭐⭐⭐ Fast (2-4× vs fixed) | ⭐⭐⭐ Lower | ⭐⭐⭐ Faster |
| **Native** | ⭐⭐ Moderate | ⭐⭐ Moderate | ⭐⭐ Moderate |

---

## Real-World Scenarios

### Scenario 1: Document Understanding (OCR-heavy)

**Task**: Extract text from a document image (aspect ratio 2:3, 600×900)

**Fixed Patching:**
```
Resize 600×900 → 336×336 (shape distortion!)
→ Text becomes blurry, aspect ratio wrong
→ OCR accuracy: 70%
```

**Adaptive Patching:**
```
Resize 600×900 → 336×336 (still distorted)
Allocate more tokens to text regions
→ Helps, but still blurry from resize
→ OCR accuracy: 75%
```

**Native Resolution:**
```
Slice into 2×3 grid of 300×300 slices
Encode each slice → Organize with spatial schema
→ No distortion, preserves text clarity
→ OCR accuracy: 90%
```

**Winner**: Native Resolution (LLaVA-UHD style)

---

### Scenario 2: General Image Captioning (1:1, 512×512)

**Task**: Generate caption for square photo

**Fixed Patching:**
```
Resize 512×512 → 336×336 (slight downsample)
→ 576 tokens → Caption: "A dog playing in a park"
→ Fast, accurate
```

**Adaptive Patching:**
```
Resize 512×512 → 336×336
Allocate: 250 tokens (reduce redundancy in grass)
→ Faster, slightly less detail
→ Caption: "A dog in a park"
```

**Native Resolution:**
```
Slice into 4 slices (512÷336 ≈ 2×2)
→ 4×144 = 576 tokens (same as fixed, but more overhead)
→ Caption: "A dog playing in a park"
```

**Winner**: Fixed or Adaptive (native adds unnecessary complexity)

---

### Scenario 3: Panorama Scene Understanding (3:1, 1200×400)

**Task**: Understand wide panoramic landscape

**Fixed Patching:**
```
Resize 1200×400 → 336×336 (severe width compression!)
→ Loses horizontal detail, distorts objects
→ Can't identify distinct regions properly
```

**Adaptive Patching:**
```
Resize 1200×400 → 336×336 (still distorted)
Allocate more tokens to buildings vs sky
→ Helps efficiency, but shape still wrong
```

**Native Resolution:**
```
Slice into 1×3 grid (or 4×2 with overlap)
Preserve 3:1 aspect ratio
→ Clear horizontal relationships preserved
→ Accurate region understanding
```

**Winner**: Native Resolution

---

### Scenario 4: Real-Time Object Detection (640×480, 30 FPS)

**Task**: Fast detection on video stream

**Fixed Patching:**
```
Resize 640×480 → 336×336
→ 576 tokens → Process time: 50ms
→ FPS: 20 (acceptable)
```

**Adaptive Patching:**
```
Resize 640×480 → 336×336
Allocate: ~300 tokens (2× reduction)
→ Process time: 25ms
→ FPS: 40 (great!)
```

**Native Resolution:**
```
Slice into 6 slices
→ 864 tokens → Process time: 75ms
→ FPS: 13 (too slow)
```

**Winner**: Adaptive Patching

---

## Trade-off Analysis

### The Fundamental Trilemma

```
        Fine-Grained Detail
              ▲
             / \
            /   \
           /     \
          /       \
         /_________\
    Efficiency    Native Resolution
```

**Cannot maximize all three simultaneously:**
- **Native + Detail** → High token count (inefficient)
- **Native + Efficiency** → Compression loses detail
- **Efficiency + Detail** → Must sacrifice native aspect ratio

### Strategy Selection Guide

```python
def choose_patching_strategy(task, image_shape, constraints):
    """Decision tree for patching approach"""

    aspect_ratio = max(image_shape) / min(image_shape)

    if constraints.realtime_required:
        return "Adaptive Patching"  # 2-4× speedup

    if task in ["OCR", "counting", "fine-grained"]:
        return "Native Resolution"  # Preserves detail

    if aspect_ratio < 1.5:  # Roughly square
        return "Fixed Patching"  # Simple, effective

    if aspect_ratio > 2.0:  # Extreme ratio
        return "Native Resolution"  # Avoid distortion

    # Default: balanced
    return "Fixed Patching" or "Native Resolution"
```

---

## Evolution Timeline

### 2020-2021: Fixed Era
- ViT establishes 16×16 patching
- CLIP extends to vision-language
- Standard: Force aspect ratio, fixed resolution

### 2022-2023: Efficiency Focus
- Token compression emerges (ToMe, FastV)
- Initial adaptive approaches (AdaViT)
- Recognition: Fixed patching is wasteful

### 2024: Adaptive & Native Era
- **APT**: Content-aware adaptive sizing (speed)
- **LLaVA-UHD**: Variable-sized slices (quality)
- **DeepSeek-OCR**: Extreme compression (16×)
- Paradigm shift: One size does NOT fit all

### 2025: Future Directions
- Query-aware patching (ARR-COC-VIS approach)
- Learned vocabularies (discrete visual tokens)
- Multi-granularity hierarchies
- Integration of all three paradigms

---

## Hybrid Approaches

### Combining Strengths

**Adaptive + Native:**
```
Step 1: Slice image natively (preserve aspect ratio)
Step 2: Apply adaptive patching within each slice
Result: Native resolution + efficient allocation
```

**Fixed + Compression:**
```
Step 1: Fixed patching (simple, fast)
Step 2: Aggressive token compression (8-16×)
Result: ViT simplicity + efficiency
```
**Example**: DeepSeek-OCR uses fixed patching + optical compression

---

## Key Takeaways

### Fixed Patching (ViT)
**Use when**: Standard images, speed matters, simple implementation
**Avoid when**: Extreme aspect ratios, fine-grained details critical

### Adaptive Patching (APT)
**Use when**: Speed is critical, images have high redundancy
**Avoid when**: Need full accuracy, irregular structure is problematic

### Native Resolution (LLaVA-UHD)
**Use when**: Aspect ratio matters, OCR/counting, high-resolution
**Avoid when**: Real-time required, standard square images

---

## Related Documentation

- **[architecture/00-overview.md](../architecture/00-overview.md)** - Patching landscape
- **[architecture/02-adaptive-patching.md](../architecture/02-adaptive-patching.md)** - Adaptive details
- **[architecture/03-native-resolution.md](../architecture/03-native-resolution.md)** - Native processing
- **[comparisons/01-token-budgets.md](01-token-budgets.md)** - Token counts
- **[comparisons/02-resolution-strategies.md](02-resolution-strategies.md)** - Resolution handling

---

**Next**: See [token budgets](01-token-budgets.md) for detailed token count analysis across models
