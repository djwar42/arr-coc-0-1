# Ovis Native Resolution Processing

**Ovis 2.5's approach to native resolution image processing without slicing**

## Overview

Ovis processes images at their native resolution without dividing into slices, using the Visual Embedding Table (VET) for dynamic token allocation.

## Key Features

**From Ovis 2.5 documentation**:

### Visual Embedding Table (VET)
- Maps visual features to variable-length token sequences
- 64-400 tokens per image based on complexity
- No fixed slicing strategy

### Native Resolution
- Processes full image without slicing
- Preserves spatial relationships
- No boundary artifacts

## Implementation

```python
class OvisVET(nn.Module):
    def __init__(self, min_tokens=64, max_tokens=400):
        self.vet = VisualEmbeddingTable()
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
    
    def forward(self, image):
        # Process full image
        features = self.visual_encoder(image)
        
        # Dynamic token allocation via VET
        tokens = self.vet(features)
        # Automatically determines token count (64-400)
        
        return tokens
```

## Benefits

- No slicing artifacts
- Dynamic token allocation
- Preserves global context
- Adaptive to content complexity

## Primary Sources

- Ovis 2.5 documentation
- Related papers in `../source-documents/`

## Related Documents

- [../architecture/03-native-resolution.md](../architecture/03-native-resolution.md)
- [02-llava-uhd.md](02-llava-uhd.md) - Alternative native resolution approach
