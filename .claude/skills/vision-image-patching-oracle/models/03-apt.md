# Adaptive Patch Transformer (APT)

**Content-aware adaptive patch sizing based on learned importance**

## Overview

APT dynamically adjusts patch sizes based on image content, allocating more tokens to informative regions and fewer to homogeneous areas.

## Key Innovation

**Learned patch selection**: Neural network determines optimal patch size per region based on attention scores from previous layers.

## Architecture

```python
class AdaptivePatchSelector(nn.Module):
    def __init__(self):
        self.importance_network = ImportanceNet()
    
    def forward(self, features, attention_scores):
        # Compute region importance
        importance = self.importance_network(features, attention_scores)
        
        # Assign patch sizes based on importance
        patch_sizes = assign_patch_sizes(importance)
        # High importance → small patches (8×8)
        # Low importance → large patches (32×32)
        
        return adaptive_patching(features, patch_sizes)
```

## Benefits

- 30-40% token reduction
- Minimal accuracy loss (<1%)
- Content-adaptive allocation

## Primary Sources

Research papers on adaptive patching in `../source-documents/`

## Related Documents

- [../architecture/02-adaptive-patching.md](../architecture/02-adaptive-patching.md)
- [../techniques/01-adaptive-patching.md](../techniques/01-adaptive-patching.md)
