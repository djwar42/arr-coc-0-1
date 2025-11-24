# DeepSeek-OCR Architecture Overview

## System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT: Image (any resolution)                              │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Image Preprocessing (process/image_process.py)             │
│ - Dynamic tiling (best aspect ratio match)                 │
│ - Normalization                                             │
│ - Token sequence creation with vision/text masks           │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ DeepEncoder (380M params)                                   │
│ ┌─────────────────────────────────────────┐                │
│ │ SAM-base (80M params)                   │                │
│ │ Window Attention Dominated              │                │
│ │ Patch Size: 16×16                       │                │
│ │ 1024×1024 → 4096 tokens (64×64 patches) │                │
│ │ ↓                                        │                │
│ │ 12 Transformer Blocks (window attn)     │                │
│ │ ↓                                        │                │
│ │ Neck: Conv 768→256, LayerNorm           │                │
│ │ ↓                                        │                │
│ │ Conv stride=2: 256→512 (spatial /2)     │                │
│ │ ↓                                        │                │
│ │ Conv stride=2: 512→1024 (spatial /2)    │                │
│ │ ↓                                        │                │
│ │ Output: [B, 1024, 16, 16] (256 patches) │                │
│ │ Total: 16× spatial compression          │                │
│ └─────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ CLIP-large (300M params)                                    │
│ ┌─────────────────────────────────────────┐                │
│ │ Uses SAM output as patch embeddings     │                │
│ │ Flatten: [B, 1024, 16, 16] → [B, 256, 1024] │           │
│ │ Add CLS token → [B, 257, 1024]          │                │
│ │ ↓                                        │                │
│ │ 24 Transformer Blocks (global attn)     │                │
│ │ Dense Global Attention                  │                │
│ │ Rich semantic understanding             │                │
│ │ ↓                                        │                │
│ │ Output: [B, 257, 1024]                  │                │
│ └─────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Feature Fusion & Projection                                 │
│ ┌─────────────────────────────────────────┐                │
│ │ Concatenate:                            │                │
│ │ - CLIP [B, 256, 1024] (drop CLS)        │                │
│ │ - SAM [B, 256, 1024] (flattened)        │                │
│ │ → [B, 256, 2048]                        │                │
│ │ ↓                                        │                │
│ │ MLP Projector: Linear 2048 → 1280       │                │
│ │ → [B, 256, 1280]                        │                │
│ └─────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ DeepSeek-3B-MoE (570M active / 3B total)                   │
│ - 12 transformer layers                                     │
│ - 64 experts (6 active per token)                          │
│ - Shared/routed experts design                             │
│ - Merges vision + text embeddings                          │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT: Text (OCR, markdown, structured data)              │
└─────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. Image Preprocessing
**File**: `process/image_process.py`

- **Dynamic tiling**: Finds best aspect ratio match for input image
- **Normalization**: Standard ImageNet stats
- **Token masking**: Creates vision/text masks for model

**Key**: Not uniform grid - content-aware tiling!

### 2. DeepEncoder (380M params)
**Files**:
- `deepseek_ocr.py:321-407` (forward pass)
- `deepencoder/sam_vary_sdpa.py` (SAM model)
- `deepencoder/clip_sdpa.py` (CLIP model)

**Design**: Serial SAM → CLIP architecture

**SAM Stage** (80M params):
- Window attention (cheap)
- Processes high-resolution (4096 tokens)
- 16× compression INSIDE SAM (neck + conv layers)
- Output: [B, 1024, 16, 16] = 256 spatial patches

**CLIP Stage** (300M params):
- Global attention (expensive)
- Uses SAM's compressed output as patch embeddings
- Adds semantic knowledge
- Output: [B, 257, 1024] (includes CLS)

**Why Serial?**
1. SAM processes cheap (window attention)
2. Compression happens BEFORE expensive CLIP
3. CLIP builds on SAM's compressed features
4. No activation memory explosion!

### 3. Feature Fusion & Projection
**File**: `deepencoder/build_linear.py`

**Steps**:
1. Drop CLIP CLS token
2. Flatten SAM output: [B, 1024, 16, 16] → [B, 256, 1024]
3. Concatenate: CLIP + SAM → [B, 256, 2048]
4. Project: MLP 2048 → 1280 (language space)

**Result**: [B, 256, 1280] vision tokens ready for language model

### 4. DeepSeek-3B-MoE
**File**: Proprietary (not in OSS release)

**Architecture**:
- 12 transformer layers
- 64 experts total
- 6 experts active per token
- Shared experts (always active) + routed experts
- Total: 3B params, 570M active

**Function**: Decodes vision tokens + text into text output

## Key Design Decisions

**1. Serial Architecture (not parallel)**
- SAM first (cheap window attention)
- Compression inside SAM
- CLIP second (expensive global attention on compressed)
- Result: Memory efficient, computationally optimal

**2. 16× Compression**
- Happens INSIDE SAM (neck + 2 conv layers)
- 4096 tokens → 256 patches spatially
- Prevents activation memory explosion
- Enables expensive CLIP processing

**3. Feature Fusion**
- CLIP provides semantic knowledge
- SAM provides fine-grained details
- Concatenation preserves both
- MLP projects to language space

**4. Multi-Resolution Support**
- Single model handles 73-421 tokens
- Positional encoding interpolation
- Trained on all resolutions simultaneously

## Performance Characteristics

**Computational Complexity**:
- SAM (window): O(N) where N = 4096
- Compression: O(N) convolutions
- CLIP (global): O(M²) where M = 256
- Total: Dominated by CLIP, but on compressed input

**Memory**:
- Peak: During CLIP forward pass
- Optimized: Flash Attention 2
- Activations: Gradient checkpointing

**Throughput**:
- Single A100: 20k+ pages/day (base mode)
- Batch size: Limited by sequence length, not vision tokens
- vLLM: 10-20× faster than HF

## Resolution Modes

| Mode | Resolution | Tokens* | Allocation |
|------|-----------|--------|------------|
| Tiny | 512×512 | 73 | Simple content |
| Small | 640×640 | 111 | Books, reports |
| Base | 1024×1024 | 273 | Standard docs |
| Large | 1280×1280 | 421 | High detail |
| Gundam | Dynamic | Variable | Ultra-high-res (tiled) |

*Actual implementation includes newlines (~13-15% more than paper reports)

## Token Budget Formula

**Base modes**: `(tokens_per_row + 1) × num_rows + 1`

Example (Base 1024×1024):
- Grid: 16×16 patches
- Per row: 16 visual + 1 newline = 17
- Total: 17 × 16 + 1 = 273 tokens

**Gundam**: `(q*w + 1) × (q*h) + global_tokens`

Example (2×3 tiling at 640×640):
- Local: (10*2 + 1) × (10*3) = 21 × 30 = 630
- Global: 273 (1024×1024 base)
- Total: 903 tokens

## File References

**Core Implementation**:
- `deepseek_ocr.py` - Main model class
- `deepseek_ocr.py:321-407` - Forward pass
- `deepseek_ocr.py:61-106` - Token calculation
- `process/image_process.py` - Preprocessing
- `deepencoder/sam_vary_sdpa.py` - SAM implementation
- `deepencoder/clip_sdpa.py` - CLIP implementation
- `deepencoder/build_linear.py` - Projector

**See Also**:
- [deepencoder.md](deepencoder.md) - Detailed SAM + CLIP design
- [compression.md](compression.md) - How 16× compression works
- [resolution-modes.md](resolution-modes.md) - Multi-resolution support
- [../code-reference/inference-flow.md](../code-reference/inference-flow.md) - Complete execution trace
