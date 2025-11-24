# DeepSeek-OCR

**Extreme 16× optical compression for document understanding**

## Overview

DeepSeek-OCR achieves unprecedented 16× compression ratio specifically for document images through specialized optical compression architecture.

## Architecture

**From [source-documents/07_DeepSeek-OCR](../source-documents/07_DeepSeek-OCR_ Contexts Optical Compression - arXiv.md)**:

### Serial Design

```
Image → SAM (Segment Anything) → Visual Extraction 
  → CLIP Encoding → 16× Compression → 36 Tokens
```

### Key Components

1. **SAM Integration**: Extract visual content regions
2. **CLIP Encoding**: Semantic visual representation
3. **Serial Processing**: Sequential compression for efficiency
4. **Optical Compression**: Specialized for text/document layouts

## Compression Pipeline

```python
class DeepSeekOCRCompression(nn.Module):
    def __init__(self):
        self.sam = SAMVisualExtractor()
        self.clip = CLIPEncoder()
        self.compressor = OpticalCompressor(ratio=16)
    
    def forward(self, document_image):
        # Extract visual content with SAM
        visual_regions = self.sam(document_image)
        
        # Encode with CLIP
        features = self.clip(visual_regions)
        
        # Compress 576 → 36 tokens (16×)
        compressed = self.compressor(features)
        
        return compressed  # 36 tokens for 336×336 image
```

## Performance

**Compression**: 576 tokens → 36 tokens (16×)
**Quality**: Maintains OCR accuracy despite extreme compression
**Speed**: 16× faster LLM processing

## Use Case

Optimized for:
- Long documents
- Text-heavy images
- Multi-page PDFs
- Document understanding tasks

## Primary Sources

- [07_DeepSeek-OCR](../source-documents/07_DeepSeek-OCR_ Contexts Optical Compression - arXiv.md)
- [19_Vision-Driven OCR](../source-documents/19_Vision-Driven OCR for Long Documents_ How Images Compress Text for LLMs.md)

## Related Documents

- [../architecture/04-compression-modules.md](../architecture/04-compression-modules.md)
- [../techniques/03-compression-strategies.md](../techniques/03-compression-strategies.md)
