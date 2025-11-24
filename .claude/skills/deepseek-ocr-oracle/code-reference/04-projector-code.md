# Projector Code

**File**: `deepencoder/build_linear.py:40-80`

```python
class MlpProjector(nn.Module):
    def __init__(self, in_features=2048, out_features=1280):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, sam_features, clip_features):
        # sam: [B, 1024, 16, 16]
        # clip: [B, 257, 1024]

        # Drop CLS
        clip = clip_features[:, 1:, :]  # [B, 256, 1024]

        # Flatten SAM
        sam = sam_features.flatten(2).permute(0, 2, 1)  # [B, 256, 1024]

        # Concatenate
        fused = torch.cat([clip, sam], dim=-1)  # [B, 256, 2048]

        # Project
        return self.linear(fused)  # [B, 256, 1280]
```

**See Also**:
- [../architecture/projector.md](../architecture/projector.md) - Design rationale
