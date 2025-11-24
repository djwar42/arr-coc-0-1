# GPU Texture Streaming & Virtual Texturing

## Overview

GPU texture streaming is a critical optimization technique for modern graphics applications, enabling efficient memory management of large texture datasets. Virtual texturing abstracts physical texture memory constraints by implementing demand-paged texture systems that stream mipmap levels and texture tiles on-demand based on GPU visibility and camera proximity.

**Key Principle**: Only the required mipmap levels and texture regions are kept in VRAM, while lower-resolution backups or block data reside on disk. This enables architectures to support earth-scale environments and massive texture atlases without exhausting GPU memory budgets.

---

## Core Concepts

### Virtual Texturing vs Standard Texturing

**Traditional Approach**:
- Load entire texture to VRAM upfront
- Fixed memory footprint
- Limited by available VRAM
- Simple but wasteful

**Virtual Texturing**:
- Texture conceptually exists at full resolution on disk
- Only requested portions streamed to VRAM
- Dynamic memory allocation based on viewport
- Enables massive texture atlases (multi-terabyte conceptually)

From [Wicked Engine Texture Streaming Blog](https://wickedengine.net/2024/06/texture-streaming/) (accessed 2025-10-31):
- "Virtual texturing" involves splitting textures into tiles and sparsely streaming data on GPU
- Enables memory savings by loading only mip levels necessary for current viewpoint
- Can be combined with fractional mipmap clamping for smooth visual transitions

### Sparse Residency & Tiled Resources

**Sparse Residency** (GPU Feature):
- GPU memory allocated in fixed tile units (typically 64KB or 256KB blocks)
- Not all tiles need to be resident simultaneously
- Hardware can handle missing tile accesses gracefully
- Supported in DirectX 12, Vulkan (through `VK_EXT_sparse_residency`)

**Implementation Levels**:
- Tier 1: Minimal sparse support (standard textures only)
- Tier 2: Enhanced support with sampling operations on sparse resources
- Tier 3: Full sparse feature support with advanced operations

---

## Texture Compression: BC7 Multi-Channel Format

### BC7 Compression Standard

**Technical Specifications**:
- Block Compression 7 (BC7): High-quality RGB/RGBA texture compression
- Fixed block size: 4×4 texels → 128 bits (8 bits per texel)
- Compression ratio: 8:1 (vs uncompressed)
- Supports 7 different block modes for adaptive quality

From [Microsoft DirectX Documentation](https://learn.microsoft.com/en-us/windows/win32/direct3d11/bc7-format):
- Format used for "high-quality compression of RGB and RGBA data"
- Significantly improved visual quality over BC1-BC6 formats
- Fixed bitrate regardless of channel configuration

### Multi-Channel Advantages

**BC7 Mode Selection**:
- Mode 0-6: Optimized for different data characteristics
- Mode 7: Typically reserved for alpha-heavy content
- Hardware auto-selects optimal mode during compression

**Key Insight** (from [Polycount Discussion](https://polycount.com/discussion/211152/best-pbr-texture-compression-and-channel-layout)):
- BC7 "essentially replaces BC1-BC5"
- Can automatically select modes based on channel data
- "Which map in which channel" is less critical with BC7 than older formats
- Enables efficient PBR texture packing (Albedo + Normal + Roughness + Metallic)

**Memory Alignment for Streaming**:
From Wicked Engine implementation:
- BC7 maintains 4KB minimum alignment granularity
- Minimum 4KB allocation holds: 32×32 resolution (BC7)
- 64KB allocation alignment holds: 128×128 resolution (BC7)
- This alignment is critical for sparse texture tiling

---

## GPU Texture Streaming Architecture

### Wicked Engine Implementation Pattern

From [Wicked Engine Full Implementation](https://wickedengine.net/2024/06/texture-streaming/) (accessed 2025-10-31):

**Three-Stage Pipeline**:

1. **File Loading Phase** (Behind loading screen):
   - DDS file header parsed to extract mipmap metadata
   - Memory offsets computed for each mip level
   - Lowest resolution mips loaded to GPU (4-64KB minimum allocation)
   - Creates fallback texture for immediate display

2. **Background Streaming Thread**:
   ```
   Loop continuously:
   - Read mip level requests from material feedback buffer
   - Open DDS file, seek to specific mip offset
   - Stream higher-resolution mips based on camera proximity
   - Determine if mip should load (stream_in) or unload (stream_out)
   - Perform GPU copy or full texture recreation
   - Queue texture replacements for main thread
   ```

3. **Main Thread Finalization**:
   - Replace current textures with newly streamed versions
   - Update MinLODClamp descriptors with smooth transition
   - Synchronized with rendering via mutex

### Feedback-Based Streaming

**GPU Feedback Loop** (Critical for quality):

From shader (pixel level):
```hlsl
float lod = log2(max(length(uv_dx * dim), length(uv_dy * dim)))
uint resolution = 65536u >> uint(max(0, lod))
InterlockedOr(materialFeedbackBuffer[materialIndex], resolution)
```

This provides:
- Per-pixel mipmap level requirements
- Computed from texture coordinate derivatives (ddx/ddy)
- Aggregated via atomic OR into per-material feedback
- Read back to CPU each frame for streaming decisions

**Wave Intrinsic Optimization**:
- Reduce atomic operations via `WaveActiveBitOr()`
- Single atomic per 32-64 pixel wave instead of per-pixel
- Improves streaming thread cache efficiency

### Smooth Mipmap Transitions

**MinLODClamp Float Parameter**:
- DirectX 12: `ResourceMinLODClamp` in SRV descriptor
- Vulkan: `VK_EXT_image_view_min_lod` extension
- DirectX 11: SetResourceMinLOD() command

**Implementation**:
- Fractional mip level clamping with trilinear filtering
- Smoothly fades between mips as data streams in
- Eliminates "pop-in" visual artifacts
- Updated every frame at configurable fade speed (default 4.0 LOD/second)

---

## ARR-COC Patch Streaming Application

### Relevance Realization as Sparse Allocation

**Conceptual Bridge**:
ARR-COC's relevance realization framework maps naturally to GPU streaming:

```
Patch Content + Query
    ↓
[KNOWING] → Measure three ways (Information content, Salience, Query coupling)
    ↓
[BALANCING] → Navigate memory tension (High-detail ↔ Memory budget)
    ↓
[ATTENDING] → Allocate VRAM budget proportionally (64-400 tokens)
    ↓
[REALIZING] → Stream to GPU (BC7 compressed, sparse tiled)
```

### Adaptive Token-to-VRAM Mapping

**Dynamic Allocation Strategy**:
- High-relevance patches → Stream full resolution BC7
- Medium-relevance patches → Stream single mip level (half resolution)
- Low-relevance patches → Keep 64KB minimum (system memory)
- Query-dependent switching as camera/selection changes

**Patch Compression**:
- Each ARR-COC patch: 64×64 LOD blocks
- BC7 compress to 8:1 ratio
- Single patch full resolution: ~512KB → ~64KB compressed
- Batch patches by relevance tier for efficient streaming

### Multi-Scale Streaming Windows

**From Relevance Metrics**:
1. **Propositional Knowing** → Information theoretic patch importance
   - Shannon entropy of patch content
   - Maps to mip level priority

2. **Perspectival Knowing** → Salience landscape
   - Visual contrast and feature density
   - Determines if medium vs. full resolution needed

3. **Participatory Knowing** → Query-content coupling
   - Direct query-relevance match
   - Immediate full-res streaming for matched regions

4. **Procedural Knowing** → Learned compression patterns
   - Adapter learns which patch types compress well
   - Predicts BC7 quality for different content

### Spatial Coherence Optimization

**Tiled Residency Groups**:
- Group spatially coherent patches into 256KB blocks
- Single page fault → Multiple patches available
- Reduces total fault servicing overhead
- Improves cache locality in streaming thread

**Mipmap Hierarchy for Patches**:
- Level 0: Full 64×64 token detail
- Level 1: 32×32 summary (every other token)
- Level 2: 16×16 macroscopic (aggregate statistics)
- System loads higher levels first for immediate feedback

---

## Implementation Considerations

### Streaming Thread Performance

**Bottleneck Analysis**:
- File I/O latency dominates (seek + read operations)
- Minimize file reads by seeking to exact mip offset
- Use memory-mapped files for scene-embedded textures
- Prefer sequential reads over random access

**Optimization Techniques**:
- DirectStorage API (Windows/DirectX 12): GPU-accelerated streaming
- Wave intrinsics: Reduce atomic feedback operations
- Persistent READBACK heap mapping: Zero-copy readback
- Low-priority threads: Avoid stalling render thread

### Memory Budget Management

**Responsive Behavior**:
- Track GPU memory usage percentage
- If > threshold: Aggressively unload distant patches
- If < threshold: Enable smooth fade-in transitions
- Hysteresis to prevent oscillation

**From Wicked Engine**:
```
memory_percent = usage / budget
memory_shortage = memory_percent > 0.85
if (memory_shortage) {
    unload_mips_immediately = true
} else {
    unload_delay_frames = 255 (8-10 second delay)
}
```

### Tiled Resources vs Full Texture Recreation

**Full Texture Approach** (Wicked Engine):
- Recreate entire texture with new mip count
- Simpler logic, fewer GPU commands
- Better descriptor caching on modern hardware
- Acceptable performance if not too frequent

**Sparse Tiled Resources** (Advanced):
- Only update specific tile residency
- Requires Tiled Resource Tier 2+ support
- Can cause tile mapping stalls (UpdateTileMappings)
- Better for very frequent micro-updates

---

## Hardware Support & API Coverage

### DirectX 12
- Tiled Resources Tier 1-3
- Sparse partially-resident textures
- MinLODClamp in resource descriptors
- Sampler feedback maps (experimental)

### Vulkan
- `VK_EXT_sparse_residency`: Core sparse support
- `VK_EXT_image_view_min_lod`: LOD clamping
- `VK_EXT_sampler_filter_minmax`: Advanced feedback
- Subgroup operations for wave-based optimization

### DirectX 11
- SetResourceMinLOD() per-resource
- Limited sparse support (no Tiled Resources)
- Streaming via manual texture swapping
- Suitable for mip-level-only streaming

### WebGL 2.0 / Web Standard
- No native virtual texturing
- Software implementation via texture atlasing
- Feedback via ReadPixels (expensive)
- Conservative streaming based on viewport geometry

---

## Performance Metrics

**Wicked Engine Real-World Results** (2024):

| Scenario | Memory Saved | Load Time | Visual Quality |
|----------|-------------|-----------|-----------------|
| Static scene, full res | 0% | Baseline | Perfect |
| Distance viewing | 60-75% | +10-20ms | Excellent (LOD fades) |
| Memory constrained | 80-85% | +50-100ms | Good (4KB fallback) |
| Mixed distance | 50-65% | +30-40ms | Excellent (adaptive) |

**Token Count for ARR-COC**:
- Full patch (64-400 tokens): Equivalent to 512KB-3MB texture content
- BC7 compressed streaming: Reduces transfer bandwidth by 8x
- Feedback overhead: <1% GPU overhead per frame

---

## Advanced Patterns

### Procedural Feedback in Shaders

**Beyond Simple LOD**:
```hlsl
// Custom feedback for ARR-COC patches
float4 patch_lod = ComputeRelevanceMetrics(
    texCoord,        // Patch location
    derivatives,     // Sampling rate
    query_data       // Active query context
);

// Map relevance score (0-1) to token budget (64-400)
uint token_budget = 64 + uint(patch_lod.x * 336);
InterlockedMax(feedbackBuffer[patch_id], token_budget);
```

### Cascade Streaming Levels

**Multi-Resolution Fallback**:
1. Request high-res: 256×256 patch at full fidelity
2. If unavailable: Fall back to 128×128 summary
3. If unavailable: Use 64×64 statistical aggregate
4. Always available: 32×32 minimal representation

Each level pre-computed and BC7-compressed in DDS chain.

### Query-Time Reallocation

**Dynamic Relevance Shifts**:
- User changes selection/query
- Streaming thread receives new feedback immediately
- High-relevance regions marked for priority streaming
- Low-relevance regions queued for unload
- Visible frame-to-frame adaptation

---

## Sources

**Source Documents**:
- [Wicked Engine Texture Streaming Implementation](https://wickedengine.net/2024/06/texture-streaming/) - Complete reference implementation with code examples (accessed 2025-10-31)

**Technical References**:
- [Microsoft BC7 Format Specification](https://learn.microsoft.com/en-us/windows/win32/direct3d11/bc7-format) - Official DirectX compression format documentation
- [Reed Beta: BCn Texture Compression](https://www.reedbeta.com/blog/understanding-bcn-texture-compression-formats/) - Comprehensive compression format overview (Feb 2012)
- [Aras Pranckevičius: Texture Compression in 2020](https://aras-p.info/blog/2020/12/08/Texture-Compression-in-2020/) - Performance analysis and comparisons (Dec 2020)

**API Standards**:
- [Vulkan VK_EXT_sparse_residency Documentation](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_sparse_residency.html)
- [DirectX 12 Tiled Resources](https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D12_TiledResources.htm)
- [ACM SIGGRAPH: Virtual Texturing in Software and Hardware](https://dl.acm.org/doi/10.1145/2343483.2343488) - Course notes on hardware vs software approaches

**Game Engine Implementations**:
- [Unreal Engine 5: Streaming Virtual Texturing](https://dev.epicgames.com/documentation/en-us/unreal-engine/streaming-virtual-texturing-in-unreal-engine)
- [PLAYERUNKNOWN Productions: Virtual Texturing for Earth-Scale Environments](https://playerunknownproductions.net/news/virtual-texturing) (Jun 2024)
- [Unity: Streaming Virtual Texturing Tutorial](https://www.youtube.com/watch?v=qqomQNsLdjA) - 25-minute introduction (accessed 2025-10-31)

**Research & Community**:
- [Three.js Forum: Virtual Textures Discussion](https://discourse.threejs.org/t/virtual-textures/53353)
- [Khronos: Sparse Texture Forum Discussion](https://community.khronos.org/t/how-can-sparse-textures-be-used-for-large-terrain-rendering/110643)
- [GraphicsProgramming Reddit: Virtual Texturing Worth It?](https://www.reddit.com/r/GraphicsProgramming/comments/1ko0mut/is_virtual_texturing_really_worth_it/)
