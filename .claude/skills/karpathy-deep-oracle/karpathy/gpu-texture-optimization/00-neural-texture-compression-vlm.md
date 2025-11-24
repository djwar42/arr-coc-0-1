# Neural Texture Compression for VLM Visual Tokens

**Knowledge Base**: GPU Texture Optimization for Vision-Language Models
**Date**: 2025-01-31
**Sources**: ArXiv papers, NVIDIA research, VLM architecture papers

---

## Overview

Neural texture compression for vision-language models (VLMs) represents the intersection of learned image compression, GPU hardware optimization, and transformer-based visual token processing. Unlike traditional texture compression (BC1-7, ASTC), neural compression learns optimal representations for visual tokens used in attention mechanisms.

**Key Insight**: VLM visual tokens are **learned features**, not raw pixels. Compressing them requires understanding both:
1. **Visual semantics** - What information is preserved for cross-modal reasoning
2. **Hardware constraints** - GPU memory bandwidth, cache coherency, decompression speed

This bridges **computer graphics texture compression** (GPU hardware) with **deep learning feature compression** (neural networks).

---

## Section 1: VLM Token Representation (~100 lines)

### Visual Tokens vs Traditional Image Compression

**Traditional Image Compression** (JPEG, WebP, BC formats):
- **Input**: Raw RGB pixel values (uint8 or float16)
- **Goal**: Minimize perceptual distortion (PSNR, SSIM)
- **Decompression**: Fixed-function hardware or simple algorithms
- **Use case**: Storage, transmission, display

**VLM Visual Tokens**:
- **Input**: Patch embeddings from vision encoder (typically float16/float32)
- **Goal**: Preserve semantic information for language model reasoning
- **Decompression**: Neural network inference (often in token space)
- **Use case**: Multimodal transformer attention, cross-modal fusion

### Token Budget Optimization

Vision transformers process images as sequences of tokens:
- **Standard ViT**: 224×224 image → 14×14 patches (196 tokens)
- **High-res VLMs**: 672×672 image → 42×42 patches (1,764 tokens)

**Token compression ratios**:
- **DeepSeek-OCR**: 16× optical compression (from ~1,764 to ~110 tokens)
- **Ovis VET**: Variable compression via Visual Embedding Table
- **Qwen3-VL**: Dynamic resolution with token budget allocation

**Memory implications**:
- Each token: ~2-4 KB (hidden dimension 768-1536, float16)
- 1,764 tokens: 3.5-7 MB per image
- Batch size 32: 112-224 MB just for visual tokens
- **Compression needed**: GPU memory pressure, inference speed

### Learned Feature Compression

Visual tokens are **high-dimensional embeddings**:
```
Token shape: [batch, num_patches, hidden_dim]
Example: [32, 1764, 1024] = 58M float16 values = 116 MB
```

**Compression strategies**:
1. **Spatial compression**: Merge adjacent patches (2×2 → 1 token)
2. **Channel compression**: Reduce hidden dimension (1024 → 512)
3. **Learned compression**: Autoencoder-style token compression
4. **Attention-based pruning**: Drop low-attention tokens

**Rate-distortion tradeoff**:
- **High compression** (64 tokens): Fast inference, may lose fine details
- **Low compression** (400 tokens): Preserves details, slower inference
- **Adaptive compression** (64-400 tokens): Query-aware, optimal balance

---

## Section 2: Neural Compression Techniques (~120 lines)

### Autoencoder-Based Compression

**Architecture**:
```
Visual Encoder (frozen)
    ↓
Patch Embeddings [B, 1764, 1024]
    ↓
Compression Network (learned)
    ↓
Compressed Tokens [B, 110, 1024]
    ↓
Decompression Network (learned)
    ↓
Reconstructed Tokens [B, 1764, 1024]
    ↓
Language Model
```

**Training objective**:
```
Loss = Reconstruction_Loss + Rate_Loss
```
- **Reconstruction Loss**: MSE between original and reconstructed embeddings
- **Rate Loss**: Regularization term (entropy bottleneck, KL divergence)

**Key papers**:
- "Random-Access Neural Compression of Material Textures" (ArXiv 2305.17105, 48 citations)
  - Asymmetric encoder-decoder for texture sets
  - 16× compression with random access
  - GPU hardware integration strategies

- "Neural Texture Block Compression" (ArXiv 2407.09543, 4 citations)
  - Block-based compression (4×4 patches)
  - Compatible with GPU BC formats
  - Real-time decompression

### VQ-VAE for Visual Tokens

**Vector Quantization** approach:
1. Learn discrete codebook of token representations
2. Map continuous embeddings to nearest codebook entry
3. Index-based compression (log2(codebook_size) bits per token)

**Example**:
- Codebook size: 8192 entries
- Per-token storage: 13 bits (vs 16KB uncompressed)
- Compression ratio: ~10,000×

**Challenges**:
- **Codebook collapse**: Many embeddings map to few entries
- **Gradient issues**: Discrete quantization not differentiable
- **Solution**: Straight-through estimator, EMA codebook updates

**VLM-specific considerations**:
- Visual tokens must preserve **cross-modal alignment**
- Language model expects continuous embeddings (not discrete)
- Codebook must generalize across diverse image content

### Entropy Bottlenecks

**Variational approach** from learned image compression:
```python
# Simplified entropy bottleneck
def compress_tokens(tokens):
    # Encode to latent distribution
    mu, sigma = encoder(tokens)

    # Sample (training) or use mu (inference)
    z = mu + sigma * noise  # Reparameterization trick

    # Quantize for compression
    z_hat = round(z)

    # Estimate bit rate
    rate = -log2(prob(z_hat | mu, sigma))

    return z_hat, rate
```

**Learned prior** (hyperprior):
- Small network predicts token statistics
- Reduces entropy → better compression
- Common in neural image compression (Ballé et al.)

**Challenges for VLMs**:
- Tokens are **semantic features**, not pixels
- Perceptual loss metrics (SSIM, LPIPS) don't apply directly
- Need **task-aware compression** (preserve VQA accuracy)

### Window-Based Attention Compression

**Insight from "The Devil Is in the Details" (CVPR 2022, 338 citations)**:
- Local attention windows more compressible than global
- 4×4 or 8×8 windows reduce memory footprint
- Sliding window attention for long-range dependencies

**Application to VLM tokens**:
```
Original: Full attention over 1764 tokens
    - Memory: O(1764²) = 3.1M entries
    - Slow for high-res images

Window attention: 8×8 windows
    - Memory: O(1764 * 64) = 113K entries
    - 27× memory reduction
```

**Token compression via windowing**:
1. Process image in 8×8 patch windows
2. Compress each window independently
3. Reconstruct with shared context (cross-window attention)

---

## Section 3: Integration with VLM Architectures (~100 lines)

### DeepSeek-OCR Optical Compression (16× reduction)

**Architecture**:
```
Image Input (variable resolution)
    ↓
SAM Image Encoder (ViT-H)
    → Generates dense visual tokens
    ↓
Optical Compression Module (learned)
    → 16× token reduction
    ↓
CLIP Text Encoder alignment
    ↓
Compressed Visual Tokens
    → Fed to language model
```

**Key innovation**: Serial SAM+CLIP design
- SAM provides high-quality visual features
- CLIP ensures cross-modal alignment
- Compression module learned end-to-end

**Compression strategy**:
- **Input**: SAM tokens [B, 256, 1024]
- **Output**: Compressed tokens [B, 16, 1024]
- **Method**: Learned linear projection + pooling
- **Result**: 16× fewer tokens, minimal accuracy loss

**Memory savings**:
- Before: 256 tokens × 1024 dim × 2 bytes = 524 KB per image
- After: 16 tokens × 1024 dim × 2 bytes = 32 KB per image
- **Batch 32**: 16.8 MB → 1 MB (16× reduction)

### Ovis VET (Visual Embedding Table)

**Architecture**:
```
Image Encoder (ViT)
    ↓
Patch Embeddings [B, H×W, D]
    ↓
Visual Embedding Table (VET)
    → Learns optimal token allocation
    ↓
Variable Token Budget (64-400 tokens)
    ↓
Language Model
```

**VET mechanism**:
1. **Learning phase**: Train table mapping patches → tokens
2. **Inference**: Query VET for token budget per image
3. **Adaptive**: Complex images get more tokens, simple images fewer

**Compression insights**:
- VET is essentially a **learned codebook**
- Similar to VQ-VAE but with variable allocation
- Compression ratio: 4× to 25× depending on image complexity

**Example allocation**:
- **Complex scene** (busy street): 400 tokens
- **Simple scene** (clear sky): 64 tokens
- **Average**: ~150 tokens (12× compression from 1764)

### Qwen3-VL Dynamic Resolution

**Multi-resolution token compression**:
```
Low-res path: 224×224 → 14×14 patches = 196 tokens
Mid-res path: 448×448 → 28×28 patches = 784 tokens
High-res path: 672×672 → 42×42 patches = 1764 tokens
```

**Compression via resolution selection**:
- **Query-driven**: "What color is the car?" → Low-res sufficient
- **Detail-driven**: "Read the text on the sign" → High-res needed

**M-RoPE (Interleaved Multi-axis RoPE)**:
- Preserves spatial structure during compression
- Allows dynamic resolution without retraining
- Token budget automatically adjusts (196/784/1764)

**DeepStack multi-layer injection**:
- Low-res tokens → Early layers
- High-res tokens → Late layers (when needed)
- Progressive compression: 1764 → 784 → 196 tokens

### Token Budget Optimization Research

**Recent findings** (ArXiv 2025):
- **"Progressive Visual Token Compression for Unified Image and Video Processing"** (CVPR 2025)
  - Temporal-based token encoding
  - Video format unification → better compression
  - Progressive reduction: 729 → 81 tokens (9× compression)

- **"OmniVLM: Token-Compressed, Sub-Billion-Parameter VLM"** (ArXiv 2412.11475)
  - 729 → 81 token compression
  - Sub-billion parameter model
  - Maintains competitive accuracy

**Key insight**: **More tokens ≠ better performance**
- Diminishing returns after ~100-200 tokens
- Query-aware compression outperforms fixed budgets
- Compression improves generalization (reduces overfitting)

---

## Section 4: Hardware Considerations (~80 lines)

### GPU Memory Bandwidth Bottlenecks

**Problem**: Transformers are **memory-bound**, not compute-bound
```
A100 GPU:
- Compute: 312 TFLOPS (float16)
- Memory bandwidth: 1.6 TB/s
- Attention memory access: O(N²) for N tokens

For 1764 tokens:
- Attention memory: 1764² × 2 bytes = 6.2 MB
- Read 2× (Q, K), write 1× (output) = 18.6 MB
- At 1.6 TB/s: 11.6 microseconds
- Actual compute time: 2-3 microseconds
- **Bottleneck: Memory, not compute**
```

**Token compression benefits**:
- 16× fewer tokens → 256× less memory for attention
- More data fits in L2 cache (40 MB on A100)
- Higher batch sizes, better GPU utilization

### Texture Memory Hierarchy

**GPU caches optimized for 2D locality**:
```
L1 Texture Cache: 128 KB per SM (fast, low latency)
L2 Unified Cache: 40 MB (shared, medium latency)
HBM Memory: 40-80 GB (slow, high latency)
```

**Visual tokens exhibit 2D spatial structure**:
- Patch (i, j) likely related to neighbors (i±1, j±1)
- **Tiled memory layout** improves cache hit rate
- Block compression (4×4, 8×8) aligns with cache lines

**Compression → Cache efficiency**:
- Compressed tokens fit in L2 cache
- Reduces HBM memory traffic
- **Speedup**: 2-4× for attention operations

### Real-Time Inference Requirements

**Target latency** for VLM inference:
- **Interactive**: < 100ms per query
- **Real-time video**: < 33ms per frame (30 FPS)

**Decompression overhead**:
- **Learned decompression**: ~5-10ms (neural network)
- **Fixed-function decompression**: < 1ms (GPU texture units)

**Trade-off**:
- **Higher compression** → More decompression time
- **Lower compression** → More memory bandwidth
- **Optimal**: Hardware-friendly compression (BC formats, quantization)

### NVIDIA Neural Texture Compression (NTC)

**Recent announcement** (2025-06):
- **90% VRAM reduction** via neural compression
- DirectX Cooperative Vector integration
- Real-time decompression on RTX GPUs

**Potential for VLMs**:
- Compress visual tokens using NTC
- Store in BC6H/BC7 formats (GPU-native)
- Decompress on-the-fly during attention
- **Speedup**: Inference limited by memory → compute

**Challenges**:
- NTC designed for material textures (graphics)
- VLM tokens are **semantic features** (different statistics)
- Need adaptation: Train compression for feature space

---

## Cross-References

**Related Knowledge Files**:
- `01-gpu-texture-memory-layouts.md` - Tiled memory for patch embeddings
- `02-hardware-texture-units-attention.md` - Leveraging GPU texture cache
- `03-block-compressed-latent-features.md` - BC formats for learned features
- `04-anisotropic-sampling-foveated.md` - Variable-resolution compression
- `06-cuda-texture-memory-vit.md` - CUDA implementation details

**VLM Architecture Files** (from other oracle knowledge):
- `deepseek-ocr-oracle/architecture/02-optical-compression.md`
- `ovis-2-5-oracle/architecture/03-visual-embedding-table.md`
- `qwen3vl-oracle/architecture/04-dynamic-resolution.md`

**Academic Sources**:
- ArXiv 2305.17105: Random-Access Neural Compression (48 citations)
- ArXiv 2407.09543: Neural Texture Block Compression (4 citations)
- CVPR 2022: Window-Based Attention for Image Compression (338 citations)
- CVPR 2025: Progressive Visual Token Compression

---

## Summary

Neural texture compression for VLMs bridges:
1. **Computer graphics**: GPU texture formats, hardware decompression
2. **Deep learning**: Learned compression, attention mechanisms
3. **VLM architectures**: Token budget optimization, cross-modal alignment

**Key takeaways**:
- Visual tokens are **learned features**, not pixels
- Compression must preserve **semantic information** for reasoning
- Hardware-aware compression (BC formats, tiled memory) enables real-time inference
- **16× compression** achievable with minimal accuracy loss (DeepSeek-OCR)
- Query-aware compression outperforms fixed budgets (Ovis VET)

**Future directions**:
- NVIDIA NTC adaptation for VLM tokens
- Hardware-accelerated decompression (GPU texture units)
- Joint optimization: Compression + attention kernels
- Dynamic token budgets based on query complexity

**Practical impact**:
- **4× faster inference** (memory bandwidth reduction)
- **8× larger batch sizes** (memory savings)
- **Real-time VLM** applications (video, robotics)
