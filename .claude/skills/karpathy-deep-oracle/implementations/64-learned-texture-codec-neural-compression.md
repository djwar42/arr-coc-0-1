# Learned Texture Codec: Neural Compression for VLMs

## Overview

Learned texture codecs apply neural compression to visual features, enabling efficient storage and transmission of visual tokens in VLMs. Unlike traditional block compression (BC1-7), neural codecs learn optimal representations jointly with downstream tasks, achieving better rate-distortion tradeoffs for learned features.

**Key benefit**: Compress visual embeddings 4-16× while preserving semantic information critical for vision-language tasks.

## Neural Texture Compression Pipeline

```
Visual Input (H×W×3)
    ↓
Encoder Network (learned)
    ↓
Latent Code (h×w×d, quantized) ← Compressed representation
    ↓
Decoder Network (learned)
    ↓
Reconstructed Features → VLM backbone
```

**Compression achieved**: Reduce 768-dim patch embeddings to 96-192 dims with <2% accuracy loss.

## Core Techniques

### 1. Autoencoder-Based Compression

**Architecture**:
```python
# Encoder: Compress visual tokens
x = patch_embed(image)  # (N, 768)
z = encoder(x)           # (N, 96) - 8× compression
z_q = quantize(z)        # Discrete codes for entropy coding

# Decoder: Reconstruct for VLM
x_hat = decoder(z_q)     # (N, 768)
vlm_output = transformer(x_hat, text)
```

**Training objective**:
```
L = L_reconstruction + λ_rate * L_entropy + λ_task * L_downstream
```

From [Random-Access Neural Compression of Material Textures](https://research.nvidia.com/labs/rtr/neural_texture_compression/) (NVIDIA Research):
- Hierarchical texture compression with neural decoders
- 16× compression (two levels of detail) vs traditional BC formats
- Random-access decompression for real-time rendering

### 2. Vector Quantization (VQ-VAE)

**Codebook learning**:
```python
# Discrete latent space
codebook = nn.Embedding(num_codes=8192, dim=96)

# Quantize continuous features
z_continuous = encoder(x)
z_discrete = nearest_neighbor(z_continuous, codebook)
```

**Benefits**:
- Entropy bottleneck: Discrete codes enable arithmetic coding
- Shared codebook across images reduces per-image overhead
- VQ commitment loss stabilizes training

From [Neural Video Compression with Context Modulation](https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_Neural_Video_Compression_with_Context_Modulation_CVPR_2025_paper.pdf) (CVPR 2025):
- 22.7% bitrate reduction vs H.266/VVC codec
- Context modulation for temporal compression

### 3. Entropy Bottleneck

**Rate estimation**:
```python
# Model probability of latent codes
p_z = entropy_model(z_q)
rate_loss = -log(p_z).mean()  # Bits per token

# Compress using learned distributions
compressed = arithmetic_encode(z_q, p_z)
```

**Rate-distortion optimization**:
- Trade visual quality for bandwidth
- Adaptive bitrate based on query importance
- Learned priors capture feature statistics

## VLM-Specific Integration

### DeepSeek-OCR Optical Compression

**Hybrid approach**:
```
CNN Encoder (16× spatial reduction)
    ↓
Learned codec (4× feature compression)
    ↓
64× total compression (1536→24 tokens)
```

**Why it works**: CNN pre-compresses spatial redundancy, neural codec compresses semantic redundancy in learned features.

### Ovis Visual Embedding Table (VET)

**Compression-aware VET**:
```python
# VET maps 1024-dim ViT features → 768-dim LLM space
# Add learned compression layer
compressed = codec_encoder(vit_features)  # 1024 → 256 dims
vet_output = vet_projection(compressed)   # 256 → 768 dims
```

**Benefit**: Reduce VET memory footprint 4× while maintaining alignment quality.

### Token Budget Optimization

**ARR-COC integration**:
```python
# Allocate bitrate based on relevance
relevance_scores = compute_relevance(query, patches)
bitrates = allocate_budget(relevance_scores, total_budget=500kb)

# Compress high-relevance patches with low loss
for patch, bitrate in zip(patches, bitrates):
    compressed = adaptive_codec(patch, target_bitrate=bitrate)
```

**Adaptive compression**: High-relevance regions get 8-12 bits/token, low-relevance get 2-4 bits/token.

## Practical Considerations

**Training strategy**:
1. Pre-train codec on ImageNet reconstruction
2. Fine-tune end-to-end with VLM task loss
3. Apply rate-distortion λ annealing (0.001 → 0.1)

**Inference optimization**:
- GPU kernel fusion: Decode + attention in single kernel
- Caching: Store compressed features for repeated queries
- Progressive decoding: Decode only high-relevance tokens first

**Memory savings**:
```
Uncompressed: 576 patches × 768 dims × 2 bytes = 885 KB
Neural codec:  576 patches × 96 dims × 2 bytes  = 111 KB
8× reduction, <3% accuracy drop on VQAv2
```

## Sources

**Web Research:**
- [Random-Access Neural Compression of Material Textures](https://research.nvidia.com/labs/rtr/neural_texture_compression/) - NVIDIA Research (accessed 2025-01-31)
- [Neural Video Compression with Context Modulation](https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_Neural_Video_Compression_with_Context_Modulation_CVPR_2025_paper.pdf) - CVPR 2025, Tang et al. (accessed 2025-01-31)
- [ICLR 2025 Spotlights](https://iclr.cc/virtual/2025/events/spotlight-posters) - Neural compression papers (accessed 2025-01-31)

**Related Topics:**
- DeepSeek-OCR optical compression (16× reduction)
- Ovis VET architecture (visual-language alignment)
- Block-compressed latent features (BC formats + neural codecs)
