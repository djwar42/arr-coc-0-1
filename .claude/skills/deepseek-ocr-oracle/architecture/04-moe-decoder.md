# DeepSeek-3B-MoE Decoder

## Overview

Language model decoder: **DeepSeek-3B-MoE** (Mixture of Experts)
- Total params: 3B
- Active params: 570M per token
- 12 transformer layers
- 64 experts (6 active per token)

## Architecture

```
Vision Tokens [B, 256, 1280] + Text Tokens [B, T, 1280]
    ↓
Merged Sequence [B, 256+T, 1280]
    ↓
12 Transformer Layers (MoE)
    ├── Shared Experts (always active)
    └── Routed Experts (6 of 64 selected)
    ↓
Language Model Head
    ↓
Output Tokens [B, T_out]
```

## MoE Design

**Shared + Routed Experts**:
- 2 shared experts (always active)
- 64 routed experts (top-6 selected per token)
- Router network selects based on token embedding

**Benefits**:
- 3B total params, but only 570M active
- Fast inference (19% of full params)
- Specialized experts for different content types

## Integration with Vision

**File**: `deepseek_ocr.py:321-407`

```python
# Merge vision + text
vision_embeds = projector(sam_feat, clip_feat)  # [B, 256, 1280]
text_embeds = embed_tokens(input_ids)            # [B, T, 1280]
merged = merge_tokens(vision_embeds, text_embeds)

# LLM forward
output = language_model(merged)
```

## Performance

- **Throughput**: 20k+ tokens/sec on A100
- **Memory**: ~6GB for inference
- **Quality**: Competitive with larger models

**Note**: Full implementation is proprietary, not in OSS release

**See Also**:
- [overview.md](overview.md) - Complete system architecture
- [projector.md](projector.md) - Vision→Language bridge
