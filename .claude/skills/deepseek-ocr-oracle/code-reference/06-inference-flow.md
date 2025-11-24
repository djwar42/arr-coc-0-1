# Complete Inference Flow

## Overview

**Full documentation**: See `RESEARCH/DeepSeekOCR/CODE_FLOW.md` (1526 lines)

This file provides a quick reference. For detailed code analysis, see CODE_FLOW.md.

## 9-Step Pipeline

```
1. Entry Point → run_dpsk_ocr_image.py or model.infer()
2. Image Preprocessing → process/image_process.py
3. SAM Processing → deepencoder/sam_vary_sdpa.py
4. 16× Compression → SAM neck + conv layers
5. CLIP Processing → deepencoder/clip_sdpa.py
6. Feature Fusion → deepencoder/build_linear.py
7. LLM Integration → deepseek_ocr.py:forward()
8. Token Generation → DeepSeek-3B-MoE decoder
9. Post-processing → Bounding boxes, markdown
```

## Key Code Locations

### Entry Point
`run_dpsk_ocr_image.py:260-304` OR `deepseek_ocr.py:409-503` (infer method)

### Image Preprocessing
`process/image_process.py`:
- Lines 45-120: Dynamic tiling
- Lines 182-250: Vision/text masking
- Lines 252-340: Padding & normalization

### Vision Processing
`deepseek_ocr.py:394-407`:
```python
# SAM
sam_features = self.sam_model(patches)              # [B, 4096, 256]

# CLIP
clip_features = self.vision_model(patches, sam_features) # [B, 256, 1792]

# Fusion
vision_tokens = self.projector(sam_features, clip_features) # [B, 256, 1280]
```

### Token Calculation
`deepseek_ocr.py:61-106`:
```python
def get_num_image_tokens(self, image_width, image_height, cropping):
    # Returns actual token count including newlines
    return (tokens_per_row + 1) * num_rows + 1
```

## Tensor Shapes at Each Step

| Step | Output Shape | Description |
|------|-------------|-------------|
| 1. Input | [3, 1024, 1024] | RGB image |
| 2. Patches | [B, 4096, 768] | 64×64 patches |
| 3. SAM | [B, 4096, 768] | Window attention features |
| 4. Compress | [B, 1024, 16, 16] | 16× spatial compression |
| 5. CLIP | [B, 257, 1024] | Global semantic features |
| 6. Fusion | [B, 256, 2048] | SAM+CLIP concatenated |
| 7. Project | [B, 256, 1280] | Language space |
| 8. Generate | [B, T_out] | Output tokens |

## Performance Metrics

**Base mode (1024×1024)**:
- Preprocessing: ~5ms
- SAM: ~15ms
- CLIP: ~20ms
- LLM: ~10ms/token
- **Total**: ~50ms + generation time

**See CODE_FLOW.md** for complete step-by-step trace with code snippets!

**File Reference**: `RESEARCH/DeepSeekOCR/CODE_FLOW.md`
