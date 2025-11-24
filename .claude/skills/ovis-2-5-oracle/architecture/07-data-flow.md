# Complete Data Flow

**Category**: Architecture
**Related**: [00-overview.md](00-overview.md)

## Inference Flow

```
User Input
    image: PIL Image (1920×1080)
    query: "Describe this image"
    ↓
┌─────────────────────────────────────┐
│ STEP 1: Preprocessing               │
├─────────────────────────────────────┤
│ smart_resize(1920×1080, 448²-1792²) │
│ → 1792×896 (preserve aspect ratio)  │
│ → [3, 1792, 896]                    │
│                                     │
│ grid_thw = [1, 128, 64]            │
│ (128 = 1792/14, 64 = 896/14)      │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ STEP 2: NaViT Encoding              │
├─────────────────────────────────────┤
│ Patch embedding: → [B, 8192, 1152]  │
│ (8192 = 128×64 patches)            │
│                                     │
│ 27 transformer blocks with RoPE    │
│ → [B, 8192, 1152]                  │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ STEP 3: Visual Tokenizer            │
├─────────────────────────────────────┤
│ Visual head projection:             │
│ [B, 8192, 1152] → [B, 8192, 16384] │
│                                     │
│ Softmax → probabilities             │
│ [B, 8192, 16384]                   │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ STEP 4: VET Lookup                  │
├─────────────────────────────────────┤
│ probabilities @ VET                 │
│ [B, 8192, 16384] @ [16384, 3584]   │
│ → [B, 8192, 3584]                  │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ STEP 5: Multimodal Merging          │
├─────────────────────────────────────┤
│ Text: "Describe this image"         │
│ Tokenized: [5791, 420, 2217]      │
│ Embedded: [3, 3584]                │
│                                     │
│ Merged: [8192 vision + 3 text]     │
│ → [8195, 3584]                     │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ STEP 6: LLM Generation              │
├─────────────────────────────────────┤
│ Qwen3 forward pass                  │
│ [8195, 3584] → [8195, 3584]        │
│                                     │
│ LM head: → [8195, 151936]          │
│ Sample next token                   │
│ Repeat autoregressively            │
└─────────────────┬───────────────────┘
                  ↓
Text Output: "The image shows..."
```

## Training Flow

```
Batch: [(image, query, answer), ...]
    ↓
Same preprocessing + encoding (Steps 1-5)
    ↓
┌─────────────────────────────────────┐
│ Forward Pass                        │
├─────────────────────────────────────┤
│ merged_embeds: [B, L, D]           │
│ labels: [B, L] with IGNORE_ID      │
│                                     │
│ LLM forward → logits [B, L, V]     │
│ CrossEntropy(logits, labels)       │
│ (ignore vision tokens)             │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ Backward Pass                       │
├─────────────────────────────────────┤
│ loss.backward()                     │
│ Update trainable modules:           │
│ - Phase P1: VT, VET only           │
│ - Phase P2-P4: All modules         │
│ - Phase P5: LLM only               │
└─────────────────────────────────────┘
```

## Tensor Shapes Summary

| Stage | Shape | Description |
|-------|-------|-------------|
| **Input Image** | [3, 1792, 896] | RGB image |
| **Patches** | [8192, 1152] | After patch embed |
| **NaViT Output** | [8192, 1152] | Visual features |
| **Probabilities** | [8192, 16384] | Over vocabulary |
| **Vision Embeds** | [8192, 3584] | After VET |
| **Text Embeds** | [3, 3584] | Query tokens |
| **Merged** | [8195, 3584] | Vision + text |
| **LLM Output** | [8195, 3584] | Hidden states |
| **Logits** | [8195, 151936] | Next token probs |

## Related Topics

- [00-overview.md](00-overview.md) - Architecture overview
- All architecture files - Details for each stage
