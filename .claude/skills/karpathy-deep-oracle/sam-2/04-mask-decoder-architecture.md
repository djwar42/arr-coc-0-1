# SAM 2 Mask Decoder: Promptable Segmentation Architecture

**"How SAM 2 Generates Masks from Prompts + Image Features"**
**Innovation**: Lightweight decoder for real-time mask generation
**Performance**: 5ms per mask @ H100 (200 masks/sec per frame)
**Key Insight**: Prompts + attention → precise boundaries

---

## Overview

The **mask decoder** is the final stage of SAM 2's pipeline. It takes:
- **Image features** from Hiera encoder
- **Memory features** from memory encoder
- **Prompt embeddings** (clicks, boxes, masks)

And outputs:
- **Segmentation mask** (pixel-level binary mask)
- **IoU score** (confidence/quality prediction)

### Design Philosophy

**Lightweight and fast:**
- Only 4M parameters (vs 256M in Hiera encoder)
- Most compute in encoder (amortized across multiple prompts)
- Decoder runs once per prompt (cheap)

**Promptable:**
- Works with clicks, boxes, masks, text (flexible prompting)
- Multiple prompts → refined segmentation
- Interactive refinement (add clicks to improve)

---

## Architecture Overview

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    SAM 2 FULL PIPELINE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Image → Hiera Encoder → Image Features [256 channels]     │
│     ↓                           ↓                           │
│  Memory Bank → Memory Attention → Memory Features          │
│     ↓                           ↓                           │
│  Prompts → Prompt Encoder → Prompt Embeddings              │
│                                 ↓                           │
│           ┌─────────────────────────────────┐              │
│           │      MASK DECODER               │              │
│           │  (This File's Focus)            │              │
│           ├─────────────────────────────────┤              │
│           │  1. Fuse image + memory         │              │
│           │  2. Cross-attend to prompts     │              │
│           │  3. Transformer blocks          │              │
│           │  4. MLP head → mask logits      │              │
│           │  5. IoU prediction head         │              │
│           └─────────────────────────────────┘              │
│                       ↓                                     │
│           Output: Mask [H×W] + IoU Score                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Decoder Components

### 1. Feature Fusion

**Combine image features + memory features:**

```
Image Features:   [H/16×W/16×256]  (from Hiera)
Memory Features:  [H/16×W/16×256]  (from memory attention)

Fusion:
  features = image_features + α·memory_features
  where α = learned weight (temporal importance)

Output: [H/16×W/16×256]
```

**Why fusion?**
- Image features: Current frame appearance
- Memory features: Temporal context (past frames)
- Combined: Object identity + appearance

### 2. Prompt Embedding Injection

**Add prompt information to features:**

```
Prompt Embeddings: [N_prompts×256]
  - Clicks: Positional encoding + learned embedding
  - Boxes: Corner coordinates → embedding
  - Masks: Mask → CNN → embedding

Injection:
  for each spatial location (i,j):
    features[i,j] += Σ prompt_embed[k] · attention(i,j,k)
```

**Cross-attention to prompts:**
- Each spatial location attends to all prompts
- Prompts "guide" where to segment

### 3. Transformer Decoder Blocks

**2-layer transformer for refinement:**

```
Block 1:
  1. Self-attention (within features)
  2. Cross-attention (to prompts)
  3. FFN (feedforward)
  4. Residual + LayerNorm

Block 2:
  (same structure, different weights)

Output: Refined features [H/16×W/16×256]
```

**Why 2 layers?**
- More layers = better quality
- But slower inference
- 2 layers = sweet spot (quality vs speed)

### 4. Mask Prediction Heads

**Two parallel heads:**

**a) Mask Head:**
```
Refined Features → MLP (3 layers) → Logits [H/16×W/16×1]
                                        ↓
                      Upsample 4× → Logits [H/4×W/4×1]
                                        ↓
                      Upsample 4× → Logits [H×W×1]
                                        ↓
                                  Sigmoid → Mask [H×W]
```

**b) IoU Prediction Head:**
```
Refined Features → Global Average Pool → [256]
                                           ↓
                      MLP (2 layers) → IoU Score [1]
```

**IoU score**: Predicts mask quality (0-1)
- Used to rank multiple predictions
- Helps select best mask when ambiguous

---

## Mask Upsampling Strategy

### Why Progressive Upsampling?

**Encoder outputs**: H/16×W/16 resolution (downsampled 16×)
**Target**: H×W resolution (full image)

**Two-stage upsampling:**
1. **16→4**: Upsample to H/4×W/4 (4× reduction)
2. **4→1**: Upsample to H×W (full resolution)

**Why not direct 16→1?**
- Progressive upsampling = smoother boundaries
- Each stage refines details incrementally
- More stable training

### Upsampling Implementation

**Transposed convolutions:**
```
Stage 1: 16×16 → 64×64
  ConvTranspose2d(256, 256, kernel=4, stride=4)

Stage 2: 64×64 → 256×256
  ConvTranspose2d(256, 1, kernel=4, stride=4)
```

**Alternatives:**
- Bilinear upsampling + Conv (simpler)
- Pixel shuffle (efficient)
- SAM 2 uses transposed conv (better boundaries)

---

## Promptable Segmentation

### Supported Prompt Types

**1. Point Prompts (Clicks)**
```
Positive click: "Segment THIS object"
Negative click: "DON'T segment this region"

Encoding:
  position (x,y) → Positional encoding (sin/cos)
  label (+1/-1) → Learned embedding
  Combined: [256] vector per click
```

**2. Box Prompts**
```
Bounding box: (x1, y1, x2, y2)

Encoding:
  4 corner points → 4 positional encodings
  Aggregate → [256] vector
```

**3. Mask Prompts**
```
Prior mask: [H×W×1] (from previous frame or rough annotation)

Encoding:
  Mask → Lightweight CNN → [H/16×W/16×256]
  Aggregate → [256] vector
```

**4. Text Prompts** (SAM 2.1)
```
Text: "person wearing red shirt"

Encoding:
  Text → CLIP text encoder → [512]
  Project → [256] vector
```

### Multi-Prompt Fusion

**Combining multiple prompts:**

```
User: Click 1 + Click 2 + Box

Encoding:
  prompt_embeds = [
    encode(click_1),   # [256]
    encode(click_2),   # [256]
    encode(box)        # [256]
  ]  # Total: [3×256]

Decoder cross-attention:
  for each spatial location:
    attend to all 3 prompts
    weighted combination based on relevance
```

**Benefit**: Ambiguous prompts → clarified by multiple prompts

---

## Interactive Refinement

### How Users Refine Masks

**Iterative prompting:**

```
Frame 1:
  User: Click object center → Mask 1 (rough)
  User: Add positive click on missed region → Mask 2 (better)
  User: Add negative click on over-segmented region → Mask 3 (refined)

Memory Bank:
  Store Mask 3 (best quality)

Frame 2:
  Propagate Mask 3 via memory attention → Mask 4
  User can refine Mask 4 if needed
```

**Key insight**: Decoder is fast (5ms), so interactive refinement is real-time!

### Prompt Accumulation

**SAM 2 accumulates prompts across refinement:**

```
Iteration 1: Click 1 → Mask 1
Iteration 2: Click 1 + Click 2 → Mask 2  (doesn't forget Click 1)
Iteration 3: Click 1 + Click 2 + Box → Mask 3
```

**Why accumulate?**
- User intent accumulates (more info = better mask)
- Prior prompts still relevant
- Avoids regression (mask quality monotonically improves)

---

## Loss Functions

### Training Objectives

**1. Mask Loss (Focal + Dice):**

**Focal Loss** (class imbalance):
```
FL = -α·(1-p_t)^γ·log(p_t)

Where:
  p_t = predicted probability
  α = class weight (0.25 for foreground)
  γ = focusing parameter (2.0)
```

**Why focal?**
- Image segmentation is imbalanced (99% background, 1% object)
- Focal loss focuses on hard pixels (uncertain predictions)

**Dice Loss** (boundary accuracy):
```
Dice = 1 - (2·|A∩B|) / (|A|+|B|)

Where:
  A = predicted mask
  B = ground truth mask
```

**Why dice?**
- Focal loss: Pixel-level accuracy
- Dice loss: Mask-level overlap (emphasizes boundaries)

**2. IoU Prediction Loss:**
```
IoU_pred = decoder.iou_head(features)
IoU_true = compute_iou(pred_mask, gt_mask)

Loss = MSE(IoU_pred, IoU_true)
```

**Why predict IoU?**
- Helps model know when it's uncertain
- Used to rank multiple predictions
- Confidence calibration

### Training Strategy

**Multi-scale supervision** (optional):
```
Loss = λ₁·Loss(16×16) + λ₂·Loss(4×4) + λ₃·Loss(1×1)
       ↑                ↑               ↑
       coarse           mid             fine
```

**Why multi-scale?**
- Supervise intermediate upsampling stages
- Smoother gradient flow
- Better boundary quality

---

## Performance Characteristics

### Inference Speed Breakdown

**Per-frame cost (H100):**
- Hiera encoder: 15ms (amortized across prompts)
- Memory attention: 3ms
- **Mask decoder: 5ms per prompt**
- Total (1 prompt): 23ms → 43 FPS

**Multiple prompts:**
- 1 prompt: 5ms
- 2 prompts: 6ms (shared encoder features)
- 5 prompts: 10ms

**Amortization**: Encoder cost shared across all prompts!

### Memory Footprint

**Decoder parameters:**
- Transformer blocks: 3M params
- Mask head: 0.5M params
- IoU head: 0.5M params
- **Total: 4M params (~16MB)**

**Activations (per prompt):**
- Features: H/16×W/16×256 (~4MB for 1024×1024)
- Mask logits: H×W×1 (~1MB for 1024×1024)

**Total VRAM (decoder only)**: ~20MB per prompt

---

## Comparison with SAM 1 Decoder

### Architectural Differences

| Component | SAM 1 | SAM 2 |
|-----------|-------|-------|
| **Memory attention** | ❌ No | ✅ Yes (temporal) |
| **Transformer layers** | 2 layers | 2 layers (same) |
| **Mask head** | MLP (3 layers) | MLP (3 layers) (same) |
| **IoU head** | ✅ Yes | ✅ Yes (same) |
| **Upsampling** | 16×→4×→1× | 16×→4×→1× (same) |

**Key difference**: Memory attention integration!

**SAM 1**: Image-only
**SAM 2**: Image + temporal memory

### Performance Comparison

| Metric | SAM 1 | SAM 2 |
|--------|-------|-------|
| **Decoder speed** | 5ms | 5ms (same) |
| **Image segmentation** | 91.1% (COCO) | 91.5% (slightly better) |
| **Video segmentation** | N/A | **82.5% J&F** (new!) |

**Insight**: SAM 2 maintains image quality while adding video capability!

---

## Advanced Features

### 1. Multi-Mask Output

**Ambiguous prompts → multiple masks:**

```
User clicks object center (ambiguous):
  → Could be: entire object, part, surrounding region

SAM 2:
  Mask 1: Whole object (IoU = 0.92)
  Mask 2: Object part (IoU = 0.85)
  Mask 3: Background region (IoU = 0.60)

User selects best mask (highest IoU typically)
```

**Implementation:**
- Decoder runs 3 times with slight variations
- Each run produces different mask
- IoU head ranks them

### 2. Mask Quality Assessment

**IoU prediction head learns mask quality:**

```
Pred_mask IoU=0.95 → High quality (trust it!)
Pred_mask IoU=0.60 → Low quality (needs refinement)
```

**Applications:**
- Automatic prompt selection (choose high-IoU prompts)
- Active learning (request labels for low-IoU cases)
- Confidence calibration (reject unreliable masks)

### 3. Hierarchical Mask Decoding

**Optional: Multi-resolution masks**

```
Coarse mask (H/4×W/4):  Fast, rough boundaries
Mid mask (H/2×W/2):     Balanced
Fine mask (H×W):        Slow, precise boundaries

User selects resolution based on speed/quality trade-off
```

---

## Key Innovations

### 1. Lightweight Design

**4M params (decoder) vs 256M params (encoder):**
- Most compute in encoder (amortized)
- Decoder is cheap → interactive refinement possible

### 2. Memory Integration

**Temporal context from memory attention:**
- Prior video methods: Optical flow (fails on occlusions)
- SAM 2: Memory attention (robust to occlusions)

### 3. Promptable Interface

**Flexible prompting (clicks, boxes, masks, text):**
- Adapts to user preference
- Multi-prompt fusion for ambiguous cases

---

## Limitations & Future Work

### Current Limitations

**1. Small objects:**
- H/16 downsampling loses small object details
- Potential fix: Higher-resolution encoder or multi-scale decoding

**2. Boundary precision:**
- Transposed conv can create artifacts
- Potential fix: Boundary refinement network (post-processing)

**3. Semantic understanding:**
- SAM 2 is class-agnostic (doesn't know object categories)
- Potential fix: Add semantic branch (object classification)

### Future Directions

**1. 3D mask prediction:**
- Predict depth + mask → 3D segmentation
- Applications: AR/VR, robotics

**2. Multi-object decoding:**
- Currently: 1 prompt → 1 mask
- Future: 1 prompt → N masks (all instances)

**3. Text-conditional decoding:**
- "Segment all red cars"
- Requires language-vision alignment

---

## Key Takeaways

1. **Lightweight**: Only 4M params (vs 256M in encoder)
2. **Promptable**: Works with clicks, boxes, masks, text
3. **Real-time**: 5ms per mask @ H100 (200 masks/sec)
4. **Memory-aware**: Integrates temporal context from memory attention
5. **Interactive**: Fast enough for real-time refinement (add prompts → better mask)

**The mask decoder is what makes SAM 2 practical for real-world interactive video segmentation.**

---

## References

- SAM 2 Paper: "SAM 2: Segment Anything in Images and Videos" (arXiv 2024)
- Mask decoder: Section 3.3 of paper
- Promptable interface: Section 3.4 of paper
- Meta AI Blog: https://ai.meta.com/sam2/
- GitHub: https://github.com/facebookresearch/sam2
