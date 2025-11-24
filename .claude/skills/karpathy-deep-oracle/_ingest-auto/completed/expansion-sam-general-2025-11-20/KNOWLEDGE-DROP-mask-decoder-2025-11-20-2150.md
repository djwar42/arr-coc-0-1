# Knowledge Drop: SAM Mask Decoder Architecture

**Date:** 2025-11-20 21:50
**Part:** 10
**File Created:** `sam-general/09-mask-decoder.md`
**Lines:** 960

## Summary

Created comprehensive documentation of SAM's mask decoder architecture - the lightweight transformer-based module that efficiently generates segmentation masks from image and prompt embeddings.

## Key Concepts Documented

### Core Architecture
- **Lightweight Design:** ~4M parameters vs 636M for image encoder
- **2-layer transformer:** Modified decoder with bidirectional cross-attention
- **Multi-mask output:** 3 candidates with IoU confidence scores
- **Real-time inference:** ~10ms per prompt after image encoding

### Technical Components

1. **Two-Way Transformer:**
   - Self-attention on tokens
   - Token-to-image cross-attention
   - Image-to-token cross-attention (novel)
   - Downsampled attention for efficiency

2. **Token Organization:**
   - IoU prediction token (learned)
   - Mask output tokens (3-4, learned)
   - Sparse prompt tokens (points, boxes)
   - Dense prompts added to image embedding

3. **Upsampling Pipeline:**
   - Two transposed convolutions (4x total)
   - 64x64 -> 128x128 -> 256x256
   - Channel reduction: 256 -> 64 -> 32

4. **Output Projection:**
   - Hypernetwork approach (MLP outputs as filters)
   - Point-wise product for mask generation
   - IoU head for confidence prediction

## Sources Used

**Web Research:**
- [How SAM's Decoder Works](https://medium.com/data-science/how-does-the-segment-anything-models-sam-s-decoder-work-0e4ab4732c37) - Wei Yi, Medium
- [Segment Anything Explained](https://storrs.io/segment-anything-explained/) - Erik Storrs
- [Keras Hub SAMMaskDecoder](https://keras.io/keras_hub/api/models/sam/sam_mask_decoder/) - Keras Documentation

**Source Documents:**
- SAM_STUDY_GENERAL.md (lines 636-694) - Mask decoder architecture section

## ARR-COC Integration

- ONNX export for edge deployment
- Fine-tuning strategy (decoder only)
- Multi-scale extension opportunities
- Loss configuration (focal + dice + IoU MSE)

## Quality Notes

- Exceeds target (960 lines vs ~700 requested)
- Complete code examples throughout
- Detailed attention mechanism explanations
- Practical implementation guidance
- Well-cited sources with access dates
