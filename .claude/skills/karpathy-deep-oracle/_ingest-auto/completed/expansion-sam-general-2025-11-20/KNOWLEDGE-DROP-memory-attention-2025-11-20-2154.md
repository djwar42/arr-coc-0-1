# Knowledge Drop: SAM 2 Memory Attention Module

**Date:** 2025-11-20 21:54
**Part:** 16
**File Created:** `sam-general/15-memory-attention-module.md`
**Lines:** 698

---

## Summary

Created comprehensive documentation of SAM 2's Memory Attention module - the key architectural innovation enabling video segmentation by conditioning current frame features on temporal context from the memory bank.

## Key Concepts Documented

### Memory Attention Overview
- Core functionality bridging spatial and temporal understanding
- Position in SAM 2 pipeline (between image encoder and mask decoder)
- Streaming architecture for real-time processing (44 FPS on A100)

### Cross-Attention Mechanism
- Standard cross-attention formula and implementation
- Three memory types: recent frames, prompted frames, object pointers
- 2D Rotary Positional Embeddings (RoPE) for spatial relationships
- FlashAttention-2 compatibility for efficiency

### Memory-to-Frame Attention
- Query generation from current frame
- Key-Value from memory bank
- Temporal position encoding (only for recent frames)
- Information flow patterns

### Frame-to-Memory Update
- Memory encoder architecture (downsampling + fusion)
- FIFO queue strategy for memory bank
- Object pointer extraction from mask decoder tokens
- Memory compression (256 → 64 dimensions)

### Implementation Details
- Complete PyTorch implementation (~200 lines)
- MemoryAttentionBlock with self-attention, cross-attention, MLP
- Integration with SAM 2 video predictor
- Memory bank management

## Technical Specifications

**Default Configuration:**
- Hidden dimension: 256
- Number of heads: 8
- Number of layers: 4
- Memory dimension: 64 (compressed)
- Max recent memories: 6
- Max prompted memories: 8

**Performance Impact:**
- ~15% overhead vs no-memory baseline
- Linear scaling with memory size
- 4x storage reduction from compression

## ARR-COC Integration Opportunities

1. **Cross-Modal Memory Bank**: Apply memory attention for multi-modal feature alignment
2. **Temporal Consistency Module**: Maintain consistency in autoregressive generation
3. **Key Transferable Patterns**:
   - Streaming architecture (constant memory)
   - Object pointers (semantic identity tokens)
   - Separate memory types (recent + important + identity)

## Sources Used

**Primary:**
- SAM 2 Paper (arXiv:2408.00714)
- SAM 2 GitHub (facebookresearch/sam2)
- Towards Data Science technical analysis

**Related Work:**
- RMem (CVPR 2024) - Restricted memory banks
- FlashAttention-2 - Efficient attention
- RoPE - Rotary position embeddings
- Hiera - Hierarchical vision transformer

---

**Status:** PART 16 Complete
**Next:** Continue with remaining SAM General expansion parts
