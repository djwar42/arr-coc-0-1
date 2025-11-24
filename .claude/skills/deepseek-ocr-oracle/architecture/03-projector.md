# Feature Fusion & Projection

## Purpose

Bridge vision features to language model embedding space:
- Fuse SAM (fine-grained) + CLIP (semantic)
- Project 2048-dim → 1280-dim (DeepSeek LLM embedding dim)

## Implementation

**File**: `deepencoder/build_linear.py`

```python
class MlpProjector(nn.Module):
    def __init__(self, in_features=2048, out_features=1280):
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, sam_features, clip_features):
        # Inputs:
        # sam_features: [B, 1024, 16, 16] from SAM compression
        # clip_features: [B, 257, 1024] from CLIP (includes CLS)

        # 1. Drop CLIP CLS token
        clip_no_cls = clip_features[:, 1:, :]  # [B, 256, 1024]

        # 2. Flatten SAM spatial
        sam_flat = sam_features.flatten(2).permute(0, 2, 1)  # [B, 256, 1024]

        # 3. Concatenate along feature dimension
        fused = torch.cat([clip_no_cls, sam_flat], dim=-1)  # [B, 256, 2048]

        # 4. Project to language space
        vision_tokens = self.linear(fused)  # [B, 256, 1280]

        return vision_tokens
```

## Why Concatenate (not add/multiply)?

**Concatenation** preserves both:
- SAM: Local fine-grained details
- CLIP: Global semantic understanding

**Addition/multiplication** would mix features, losing distinct information.

## Why MLP (not complex projection)?

Simple linear projection is sufficient:
- SAM+CLIP already extract good features
- Language model handles complexity
- Fewer params = faster, less overfitting

## Output

**Result**: [B, 256, 1280] vision tokens ready for LLM
- 256 tokens (base mode)
- 1280-dim (matches DeepSeek-3B embedding dim)

**See Also**:
- [deepencoder.md](deepencoder.md) - SAM+CLIP fusion rationale
- [moe-decoder.md](moe-decoder.md) - How LLM uses these tokens
