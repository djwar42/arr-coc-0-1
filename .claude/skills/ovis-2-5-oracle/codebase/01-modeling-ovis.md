# modeling_ovis.py

**Category**: Codebase
**File**: `ovis/model/modeling_ovis.py`

## Classes

### VisualEmbedding (lines 25-34)

```python
class VisualEmbedding(nn.Embedding):
    """
    Visual Embedding Table (VET)

    Probabilistic discrete embedding lookup
    """
    def forward(self, x):
        if x.dtype in [torch.int32, torch.int64]:
            return F.embedding(x, self.weight)
        # Probabilistic lookup
        return torch.matmul(x, self.weight)
```

### VisualTokenizer (lines 36-189)

```python
class VisualTokenizer(nn.Module):
    """
    Visual feature extraction and tokenization

    Components:
    - ViT encoder (SigLIP 2 NaViT)
    - Visual head (Linear + LayerNorm)
    - Smart resize algorithm
    """
```

**Key Methods**:
- `smart_resize()`: Aspect-ratio preserving resize
- `preprocess()`: Image preprocessing
- `forward()`: Full tokenization pipeline

### Ovis (lines 196-end)

```python
class Ovis(PreTrainedModel):
    """
    Main Ovis model class

    Integrates:
    - Visual tokenizer
    - Visual embedding table
    - LLM (Qwen3)
    - Multimodal merging
    """
```

**Key Methods**:
- `chat()`: High-level inference
- `generate()`: Generation with thinking mode
- `merge_multimodal()`: Combine vision + text
