# GPU Texture Compression and Mipmap Storage

## Overview

Block-based texture compression reduces memory bandwidth and storage requirements while enabling fast random access to individual texels. Modern GPUs support hardware-accelerated decompression of formats like BC7, ASTC, and ETC2, with compression ratios of 4:1 to 8:1 compared to uncompressed RGBA8. Each format uses fixed-size blocks (typically 4×4 pixels) that compress to either 64 bits (4bpp) or 128 bits (8bpp), allowing O(1) texture lookups without full image decompression.

**Why compress mipmaps?** Each mip level stores increasingly coarse versions of a texture. While lower mip levels are already smaller, compressing them provides:
- **Reduced memory footprint**: 75% memory savings across the entire mipmap pyramid
- **Lower bandwidth usage**: Coarse mip sampling accesses fewer bytes from VRAM
- **Faster texture streaming**: Compressed data loads faster from disk/network
- **Better cache utilization**: More texture data fits in GPU L2/L3 caches

**VLM application**: Vision transformers using hierarchical patch processing benefit enormously from compressed mipmap pyramids - query-driven LOD selection can sample appropriate detail levels with minimal memory traffic.

From [Texture Compression in 2020](https://aras-p.info/blog/2020/12/08/Texture-Compression-in-2020/) (accessed 2025-01-31):
- BC7 achieves 42+ dB PSNR at 8bpp with 10-20 Mpix/s compression speed
- ASTC 4×4 matches BC7 quality but at 2-8 Mpix/s compression speed
- DXTC (BC1) provides 35-40 dB at 4-8bpp with 100-650 Mpix/s compression

## Compression Formats

### BC (Block Compression) Family - Desktop GPUs

**BC1 (DXT1)** - 64 bits per 4×4 block (4bpp):
- Two RGB565 endpoints (32 bits) + 16 2-bit indices (32 bits)
- 4-color palette interpolated between endpoints: [0, 1/3, 2/3, 1]
- Optional 1-bit alpha via endpoint symmetry exploitation
- Quality: 35-40 dB PSNR for typical color maps
- Compression speed: 100-650 Mpix/s (ISPC compressor)
- **Best for**: Albedo textures without alpha, simple color patterns

**BC3 (DXT5)** - 128 bits per 4×4 block (8bpp):
- BC1 block for RGB (64 bits) + BC4 block for alpha (64 bits)
- Alpha: 2 8-bit endpoints + 16 3-bit indices
- 8-color alpha palette: [0,1,2,3,4,5,6,7]/7 or [0,1,2,3,4,5]/5 + {0.0, 1.0}
- Quality: 38-42 dB for RGBA content
- **Best for**: Textures with full alpha channels, packing color+gloss

**BC4 (Single-channel)** - 64 bits per 4×4 block (4bpp):
- 2 8-bit endpoints + 16 3-bit indices
- 8-value palette with precise gradients
- Quality: 45+ dB for grayscale (much better than BC1)
- **Best for**: Height maps, gloss maps, AO maps, font atlases

**BC5 (Two-channel)** - 128 bits per 4×4 block (8bpp):
- Two BC4 blocks side-by-side (uncorrelated channels)
- Quality: 40-45 dB for normal maps (XY components)
- **Best for**: Tangent-space normal maps (X,Y), metallic-roughness

**BC7 (Modern high-quality)** - 128 bits per 4×4 block (8bpp):
- 8 modes with variable endpoint precision (4-8 bits per component)
- 4-16 color palettes per mode
- 1-3 partitions per block (spatial regions with own endpoints)
- P-bits for precise endpoint positioning
- Quality: 42-48 dB PSNR (near-lossless for most content)
- Compression: 10-20 Mpix/s (bc7enc, high quality)
- **Best for**: High-fidelity albedo, UI textures, smooth gradients

**BC6H (HDR)** - 128 bits per 4×4 block (8bpp):
- FP16 RGB endpoints with delta compression
- 8-16 color palettes
- Signed/unsigned modes for HDR content
- Quality: Preserves HDR range while compressing to 8bpp
- **Best for**: HDR environment maps, emissive textures

From [Understanding BCn Texture Compression Formats](https://www.reedbeta.com/blog/understanding-bcn-texture-compression-formats/) (accessed 2025-01-31):
> "BC7 cleverly exploits degeneracy by breaking it: defines alternative modes triggered by endpoint order, allowing per-block format changes for optimal quality"

### ASTC Family - Mobile GPUs

**ASTC (Adaptive Scalable Texture Compression)**:
- Variable block sizes: 4×4, 5×5, 6×6, 8×8, 10×10, 12×12
- Compression ratios: 8bpp (4×4), 3.6bpp (6×6), 2bpp (8×8), 0.89bpp (12×12)
- Modes with 1-4 partitions, 2-4 color endpoints per partition
- LDR and HDR profiles (HDR similar to BC6H)
- Quality varies by block size:
  - 4×4: 42-45 dB (matches BC7)
  - 6×6: 35-38 dB (similar to BC1 but lower bitrate)
  - 8×8: 30-35 dB (visible artifacts)

**ASTC 4×4** (8bpp - High quality):
- 128 bits per 4×4 block
- ARM astcenc: 2-8 Mpix/s compression, 42+ dB quality
- Modes: endpoints 4-8 bits, 4-16 color palettes
- **Best for**: High-quality mobile textures matching BC7

**ASTC 6×6** (3.6bpp - Medium quality):
- 128 bits per 6×6 block (36 pixels)
- Quality: 35-40 dB (lower than BC1 but better compression)
- Compression speed: 0.5-5 Mpix/s
- **Best for**: Mobile textures where memory is constrained

**ASTC 8×8** (2bpp - Low quality):
- 128 bits per 8×8 block (64 pixels)
- Quality: 30-35 dB (visible banding/artifacts)
- **Best for**: Background textures, distant LODs

From [Compressed GPU texture formats review](https://themaister.net/blog/2020/08/12/compressed-gpu-texture-formats-a-review-and-compute-shader-decoders-part-1/) (accessed 2025-01-31):
> "ASTC is the final boss of texture compression. Its complexity is staggering and aims to dominate the world with 128 bits."

### ETC2 Family - Mobile GPUs (Legacy)

**ETC2 RGB** - 64 bits per 4×4 block (4bpp):
- 2×4 or 4×2 sub-blocks with own endpoints
- Modes: Differential (correlated), Individual (uncorrelated)
- T/H modes: 4-color partition schemes
- Planar mode: Bilinear interpolation across block (unique!)
- Quality: 35-38 dB
- Compression: 0.1-5 Mpix/s (Etc2Comp)
- **Note**: Obsoleted by ASTC on modern mobile GPUs

**ETC2 RGBA** - 128 bits per 4×4 block (8bpp):
- ETC2 RGB block + EAC alpha block
- EAC: 11-bit precision alpha with table-based modulation
- **Best for**: Legacy mobile RGBA textures (prefer ASTC on modern GPUs)

## Per-Mip Compression Strategy

### Independent Block Compression

Each mip level is compressed independently:

```
Mip 0 (2048×2048): 512×512 blocks → 16MB compressed (BC7 @ 8bpp)
Mip 1 (1024×1024): 256×256 blocks → 4MB compressed
Mip 2 (512×512):   128×128 blocks → 1MB compressed
Mip 3 (256×256):   64×64 blocks   → 256KB compressed
...
Mip 11 (1×1):      1×1 block      → 16 bytes compressed
```

**Benefits**:
- Random access to any mip level without decompressing others
- GPU can stream in coarse mips first for progressive loading
- Compression artifacts don't propagate between levels

**Quality considerations per mip**:
- **Mip 0-2**: High quality needed (BC7/ASTC 4×4) - close-up viewing
- **Mip 3-6**: Medium quality (BC1/ASTC 6×6) - mid-distance
- **Mip 7+**: Low quality acceptable (ASTC 8×8) - distant/tiny on screen

### Mipmap Tail Optimization

**Sparse texture residency** (DirectX 12 Tiled Resources, Vulkan Sparse):
- Last 3-5 mip levels (16×16 down to 1×1) often packed into single 64KB page
- "Mipmap tail" can be loaded as unit for small textures
- Compressed formats reduce tail size:
  - Uncompressed RGBA8 tail (16×16 to 1×1): ~1.4KB
  - BC7 compressed tail: ~350 bytes (4× smaller)
  - ASTC 8×8 tail: ~175 bytes (8× smaller)

**VLM application**: Attention-driven LOD selection can skip streaming high-res mips for patches with low relevance scores, loading only coarse mip tail.

## Hardware Decompression

### Fixed-Function Decompression Units

Modern GPUs include dedicated texture decompression hardware:

**NVIDIA (Turing/Ampere/Ada)**:
- Texture units (TMUs): 4-16 per SM (Streaming Multiprocessor)
- Decompression: BC1-BC7, ASTC LDR (4×4 to 12×12)
- Throughput: ~4 texels/cycle per TMU at full speed
- Bandwidth savings: 4-8× reduction for compressed vs uncompressed

**AMD (RDNA2/RDNA3)**:
- Texture addressing units + L1 texture cache per CU
- Decompression: BC1-BC7, ASTC LDR
- Delta Color Compression (DCC): Additional lossless layer
- Throughput: ~4 texels/cycle per TMU

**ARM Mali (Valhall/5th gen)**:
- Texture units: 1-4 per core
- Native ASTC support (all block sizes, LDR + HDR)
- Decompression: 8-16 texels/cycle
- Also supports ETC2/EAC for legacy content

**Apple GPU (M1/M2/A-series)**:
- Fast ASTC decompression (all standard block sizes)
- BC formats via software fallback (slower)
- Hardware-optimized for ASTC LDR/HDR

### Decompression Pipeline

```
1. Address Calculation (tex_coord → block_address)
   ↓
2. Cache Lookup (L1 texture cache)
   ↓ (miss)
3. Memory Fetch (64 or 128 bits per block)
   ↓
4. Decompression (fixed-function or compute shader)
   ↓
5. Cache Store (decompressed 4×4 block → L1 cache)
   ↓
6. Filter (bilinear/trilinear/anisotropic on decompressed texels)
```

**Performance**: Decompression is effectively "free" when memory-bound:
- BC7 fetch: 128 bits → 512 bits (4× bandwidth savings)
- Decompression latency: 2-4 cycles (hidden by fetch latency)
- Overall: ~3-4× texture sampling speedup in bandwidth-limited scenarios

### Compression Artifacts at Different Mip Levels

**Mip 0 (original resolution)**:
- BC1: Visible banding in smooth gradients (5:6:5 endpoint precision)
- BC7: Near-imperceptible artifacts (42+ dB)
- ASTC 4×4: Similar to BC7 quality

**Mip 2-4 (medium resolution)**:
- Block boundaries may become visible with extreme compression
- BC3: Alpha banding if gradient spans multiple blocks
- ASTC 6×6: Slight color shift in high-frequency detail

**Mip 7+ (very coarse)**:
- Compression artifacts largely irrelevant (texels are tiny on screen)
- Even aggressive compression (ASTC 8×8) acceptable
- Memory savings more valuable than quality

From [Texture Compression in 2020](https://aras-p.info/blog/2020/12/08/Texture-Compression-in-2020/):
> "BC7 produces >45dB quality at 10-20Mpix/s, whereas ASTC 4×4 achieves same quality several times slower at 2-8Mpix/s. But there's a 50,000× compression speed difference between slowest (ETC2/ETCPACK: 0.013Mpix/s) and fastest (DXTC/ISPC: 654Mpix/s) compressors!"

## VLM Feature Map Compression

### Compressing Neural Network Activations

Vision transformers produce multi-scale feature maps that exhibit similar spatial redundancy to texture images:

**Feature map characteristics**:
- Spatial dimensions: 64×64, 32×32, 16×16 (typical ViT feature resolutions)
- Channels: 768, 1024, 1536 (embedding dimensions)
- Value range: Typically [-5, 5] after normalization

**Compression approaches**:

**1. Block-based compression (BC/ASTC)**:
```
Input: Feature map [H, W, C]
Reshape: [H, W, C] → [H, W, C/4, 4] (group 4 channels as RGBA)
Compress: BC7 encode each [H/4, W/4, 4] block
Storage: (H × W × C / 16) bytes (8bpp)
```

Benefits:
- 8× compression vs FP32 features
- Hardware decompression during inference
- Random access to spatial locations

Challenges:
- Requires quantization to 8-bit UNORM
- May lose precision for outlier activations
- Channel grouping assumes local correlation

**2. Neural compression (learned codecs)**:
- Train small MLP to compress/decompress feature maps
- 16-32× compression with learned representation
- Inference cost: Run decompressor network before attention

From [Random-Access Neural Compression of Material Textures](https://research.nvidia.com/labs/rtr/neural_texture_compression/) (NVIDIA Research):
> "Key idea: compress multiple material textures and mipmap chains together using small neural network optimized jointly with texture data"

**3. Quantization + entropy coding**:
- Quantize features to INT8 or INT4
- Apply lossless compression (zlib, Zstd, Brotli-G)
- 4-16× compression depending on feature statistics
- Decompression via CPU/GPU before inference

### Mipmap Pyramids for VLM Patch Hierarchies

**Analogy**: GPU texture mipmaps ↔ Vision transformer patch hierarchies

```
Texture mipmaps:              VLM patch pyramid:
┌─────────────┐              ┌─────────────┐
│ Mip 0: 1024 │              │ Patch: 16×16│ (High detail)
├─────────────┤              ├─────────────┤
│ Mip 1: 512  │              │ Patch: 32×32│ (Medium detail)
├─────────────┤              ├─────────────┤
│ Mip 2: 256  │              │ Patch: 64×64│ (Coarse detail)
└─────────────┘              └─────────────┘
```

**Query-driven LOD selection**:
1. Attention scores determine relevance of image regions
2. High attention → sample fine mip levels (high token budget)
3. Low attention → sample coarse mip levels (low token budget)
4. Compressed mipmaps enable efficient multi-resolution sampling

**Memory savings**:
- Uncompressed pyramid: 4/3 × base resolution size
- BC7 compressed: 4/3 × (base resolution / 8) = 1/6 of uncompressed
- Enables larger batch sizes and higher resolution VLM inference

**Bandwidth optimization**:
- Accessing coarse mips (low relevance regions): 64 bits/block
- Accessing fine mips (high relevance regions): 128 bits/block
- Overall: 4-8× bandwidth reduction vs uncompressed features

## Sources

**Web Research:**
- [Texture Compression in 2020](https://aras-p.info/blog/2020/12/08/Texture-Compression-in-2020/) - Aras Pranckevičius (accessed 2025-01-31)
- [Understanding BCn Texture Compression Formats](https://www.reedbeta.com/blog/understanding-bcn-texture-compression-formats/) - Nathan Reed (accessed 2025-01-31)
- [Compressed GPU texture formats review](https://themaister.net/blog/2020/08/12/compressed-gpu-texture-formats-a-review-and-compute-shader-decoders-part-1/) - Hans-Kristian Arntzen (accessed 2025-01-31)
- [Random-Access Neural Compression of Material Textures](https://research.nvidia.com/labs/rtr/neural_texture_compression/) - NVIDIA Research (accessed 2025-01-31)

**Additional References:**
- Khronos Data Format Specification - Block compression standards
- DirectX 12 block compression documentation - Microsoft Learn
- AMD GPUOpen: Neural Texture Block Compression (2024)
- ARM ASTC encoder documentation
