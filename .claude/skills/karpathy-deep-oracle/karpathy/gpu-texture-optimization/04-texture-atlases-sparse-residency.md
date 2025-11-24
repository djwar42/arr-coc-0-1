# GPU Texture Atlases and Sparse Residency

## Overview

**Sparse texture residency**, also known as **tiled resources** (DirectX), **sparse resources** (Vulkan), or **virtual texturing**, is a GPU memory management technique that allows textures to appear larger than physical memory by only loading visible tiles on-demand. This approach decouples texture size from GPU memory, enabling massive texture pyramids (megatextures) that can be streamed hierarchically based on view requirements.

Key benefit: A 16,384×16,384 texture with mipmaps normally requires ~360MB of VRAM, but with sparse residency, only the visible tiles (potentially <50MB) need to be resident at any time.

**Virtual texturing** is the application-level implementation of sparse residency, managing tile visibility, streaming, and cache updates. The technique was pioneered by id Software's MegaTexture system (id Tech 4/5) and has become standard in modern game engines (Unreal Engine 5's Virtual Textures, Unity's Texture Streaming).

## Sparse Texture Architecture

### Vulkan Sparse Resources

From [Vulkan Sparse Resources Documentation](https://docs.vulkan.org/spec/latest/chapters/sparsemem.html) (accessed 2025-01-31):

Vulkan provides three sparse resource capabilities:

1. **Sparse Binding** — Basic virtual memory for resources
   - Resources created with `VK_IMAGE_CREATE_SPARSE_BINDING_BIT`
   - Manual memory binding via `vkQueueBindSparse()`
   - Applications control which memory ranges are bound

2. **Sparse Residency** — Tile-level granularity
   - Requires `VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT`
   - Texture divided into fixed-size tiles (typically 64KB)
   - Each tile can be individually bound/unbound
   - Supports "mipmap tail" — coarse levels packed together

3. **Sparse Aliasing** — Multiple bindings to same memory
   - Used with `VK_IMAGE_CREATE_SPARSE_ALIASED_BIT`
   - Allows format casting between views

**Tile sizes (Vulkan):**
```
BC7/ASTC compressed: 128×128 at 64KB
RGBA8 uncompressed: 64×64 at 64KB
```

### DirectX 12 Tiled Resources

From [Microsoft DirectX 12 Tiled Resources](https://learn.microsoft.com/en-us/windows/win32/direct3d12/volume-tiled-resources) (accessed 2025-01-31):

DirectX 12 provides **tiled resources** with 4 tiers of support:

**Tier 1:** Basic tiled resource support
- 64KB tiles (D3D12_TILED_RESOURCES_TIER_1)
- No guarantees on NULL tile reads (undefined behavior)

**Tier 2:** Expanded support
- `D3D12_TILED_RESOURCES_TIER_2`
- NULL tiles read as zero
- Mipmap clamping support via shader instructions

**Tier 3:** Volume texture tiling
- 3D textures as tiled resources
- Critical for volumetric datasets

**Tier 4:** 64KB-aligned dimensions (latest)
- From [D3D12 Tiled Resource Tier 4](https://microsoft.github.io/DirectX-Specs/d3d/D3D12TiledResourceTier4.html) (accessed 2025-01-31)
- Removes restriction on array slices with small mipmaps
- More flexible mipmap tail handling

**DirectX tile properties:**
- Fixed 64KB tile size
- API: `ID3D12Device::CreateReservedResource()`
- Mapping: `ID3D12CommandQueue::UpdateTileMappings()`

### Memory Layout and Mipmap Tails

Both APIs pack coarse mipmap levels into a **mipmap tail**:

```
Level 0: 4096×4096  →  16 tiles per side (256 tiles total)
Level 1: 2048×2048  →  4 tiles per side (16 tiles total)
Level 2: 1024×1024  →  1 tile (packed into tail)
Level 3: 512×512    →  (packed into tail)
...
Level N: 4×4        →  (packed into tail)
```

The tail contains all mips that fit within one tile (64KB). For BC7 compressed textures, this means levels 6 and below (128×128 down to 4×4) share a single tile.

**ASCII Memory Diagram:**

```
Virtual Texture Address Space (16K×16K with mipmaps)
╔═══════════════════════════════════════════════════
║ LEVEL 0 (4096×4096 - 256 tiles)
║ ┌───┬───┬───┬───┐
║ │ T │ T │ T │...│  ← Each T = 64KB tile
║ ├───┼───┼───┼───┤
║ │ T │ T │ T │...│
║ └───┴───┴───┴───┘
║
║ LEVEL 1 (2048×2048 - 64 tiles)
║ ┌───┬───┐
║ │ T │ T │
║ └───┴───┘
║
║ LEVEL 2-N (Mipmap Tail - 1 tile)
║ ┌───────────
║ │ [All coarse mips packed: 1024→512→256→...→4×4]
║ └───────────

Physical GPU Memory (Cache - 12MB example)
╔═══════════════════════════════════════════════════
║ Resident Tile Pool (12×12 tiles at 64KB = 9MB)
║ ┌─────┬─────┬─────┬─────┬─────┬─────
║ │T0,0 │T0,1 │T1,0 │T1,1 │Tail │ ...
║ │4K L0│4K L0│4K L0│4K L0│Coarse│
║ └─────┴─────┴─────┴─────┴─────┴─────
║
║ Indirection Table (Lookup texture - 64×64×6 mips)
║ Maps virtual tile (x,y,level) → physical cache (x,y)
```

## Streaming Strategies

### Tile Visibility Detection

From [Wicked Engine Texture Streaming](https://wickedengine.net/2024/06/texture-streaming/) (accessed 2025-01-31):

**Method 1: Render to tile detection buffer**
```glsl
// Fragment shader writes required tile info
uniform vec2 u_svtDimensionInTiles;
uniform float u_svtDepth;

void main() {
    // Compute mip level from UV gradients
    vec2 dxVtc = dFdx(texCoords * textureSize);
    vec2 dyVtc = dFdy(texCoords * textureSize);
    float deltaMaxSqr = max(dot(dxVtc, dxVtc), dot(dyVtc, dyVtc));
    float mipLevel = 0.5 * log2(deltaMaxSqr);

    // Convert to tile coordinates
    float nTilesLevel = pow(2.0, svtDepth - mipLevel);
    vec2 tileXY = floor(texCoords * nTilesLevel);

    // Output: (tileX, tileY, textureID, mipLevel)
    fragColor = vec4(tileXY, textureID, mipLevel);
}
```

**Optimization**: Render at 1/4 resolution (4× downsample) to reduce pixel shader cost. Adjust mip level calculation:
```glsl
float scaleFactor = -log2(4.0); // Correct for downsampled buffer
float mipLevel = 0.5 * log2(deltaMaxSqr) + scaleFactor;
```

**Method 2: Hardware sampler feedback (DirectX 12)**

From [Sampler Feedback Spec](https://microsoft.github.io/DirectX-Specs/d3d/SamplerFeedback.html) (accessed 2025-01-31):

DirectX 12 provides `SamplerFeedbackTexture2D` for hardware-accelerated tile tracking:
```hlsl
SamplerFeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> feedbackTex;
Texture2D baseTexture;

float4 color = baseTexture.Sample(sampler, uv);
feedbackTex.WriteSamplerFeedback(baseTexture, sampler, uv);
```

GPU writes which tiles were accessed to feedback texture. CPU reads back to determine streaming priorities.

### Streaming Pipeline

**4-stage pipeline** (adapted from Wicked Engine implementation):

**Stage 1: Tile Request Collection**
```cpp
// Read tile detection buffer (asynchronous GPU→CPU transfer)
glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, tileRequests);

// Parse requests into priority queue
for (auto& pixel : tileRequests) {
    uint16_t tileX = pixel.r;
    uint16_t tileY = pixel.g;
    uint8_t textureID = pixel.b;
    uint8_t mipLevel = pixel.a;

    streamQueue.push({textureID, mipLevel, tileX, tileY, priority});
}
```

**Stage 2: Tile Loading (Background Thread)**
```cpp
// Load tiles from disk using memory-mapped files
void loadTileAsync(TileRequest req) {
    // DDS file structure: header + mip0 + mip1 + ... + mipN
    // Pre-computed offsets allow direct seeking
    size_t offset = tileInfo.mipOffsets[req.mipLevel]
                  + (req.tileY * tilesPerRow + req.tileX) * tileSize;

    std::ifstream file(texturePath, std::ios::binary);
    file.seekg(offset);
    file.read(tileData, tileSize);

    // Add to cache with LRU eviction
    tileCache.insert(req, tileData);
}
```

**Stage 3: GPU Upload (Main Thread)**
```cpp
// Update tile cache texture
glBindTexture(GL_TEXTURE_2D, tileCacheTexture);
glTexSubImage2D(
    GL_TEXTURE_2D,
    0,                          // mip level in cache
    cacheX, cacheY,            // position in cache
    tileSize, tileSize,        // tile dimensions
    GL_RGBA, GL_UNSIGNED_BYTE,
    tileData
);

// Update indirection table
indirectionTable[virtualTileIndex] = vec4(
    cacheX / cacheSizeInTiles,
    cacheY / cacheSizeInTiles,
    mipLevel,
    1.0  // validity flag
);
```

**Stage 4: Smooth LOD Transition**
```glsl
// Fragment shader with fractional mip clamping
uniform float minLODClamp; // Animated from coarse→fine

vec4 queryIndirection(vec2 uv, float baseLOD) {
    float lod = clamp(baseLOD + minLODClamp, 0.0, maxMipLevel);
    vec4 indirection = textureLod(indirectionTable, uv, lod);

    // Loop up mip levels if tile not resident
    while (indirection.a != 1.0 && lod < maxMipLevel) {
        lod += 1.0;
        indirection = textureLod(indirectionTable, uv, lod);
    }
    return indirection;
}
```

Animate `minLODClamp` from previous level to new level over ~0.25 seconds for smooth pop-in.

### Prefetching and Prediction

**Camera velocity-based prefetching:**
```cpp
// Predict tile requests 2 frames ahead
vec3 predictedPos = cameraPos + cameraVelocity * (2.0 * deltaTime);
vec3 predictedDir = cameraDir + cameraAngularVel * (2.0 * deltaTime);

// Add predicted visible tiles to high-priority queue
for (auto& tile : predictVisibleTiles(predictedPos, predictedDir)) {
    streamQueue.pushHighPriority(tile);
}
```

**Hierarchical prefetching:**
When loading tile (x, y, L), preload adjacent tiles and parent:
- Spatial neighbors: (x±1, y±1, L)
- Parent tile: (x/2, y/2, L+1) — always keep in cache

## Memory Savings

### Quantified Benefits

From [Sparse Virtual Textures - Gaia Sky](https://tonisagrista.com/blog/2023/sparse-virtual-textures/) (accessed 2025-01-31):

**Example: Earth surface texture (16K × 8K with 10 mip levels)**

**Traditional approach:**
```
Full pyramid memory = base_size * (4/3)
                    = (16384 * 8192 * 4 bytes) * 1.333
                    = 716 MB
```

**Sparse residency with 1024×1024 tiles:**
```
Total tiles in base level = (16 × 8) = 128 tiles
Visible tiles (60° FOV, Earth) ≈ 32 tiles
Plus parent levels (always resident) = 6 tiles

Active memory = (32 + 6) * (1024 * 1024 * 4 bytes)
              = 38 * 4 MB = 152 MB

Memory reduction: 79% savings (716 MB → 152 MB)
```

**Extreme case: 131K × 65K Earth texture (id Tech 5 scale)**
```
Full pyramid = 68 GB
Visible tiles (sparse) = ~500 MB active
Reduction: 99.3% savings
```

### Memory Access Patterns

**Cache hit rates** (measured in Wicked Engine):

| Scenario | Cache Size | Hit Rate | Frame Time |
|----------|-----------|----------|------------|
| Static view | 9 MB | 98% | 0.8 ms |
| Slow pan | 9 MB | 92% | 1.2 ms |
| Fast rotation | 9 MB | 76% | 2.4 ms |
| Rapid zoom | 18 MB | 88% | 1.8 ms |

**Bandwidth savings:**
- Traditional: 716 MB/frame (worst case, if fully streamed)
- Sparse: ~50 MB/frame (only tile updates)
- Reduction: 93% bandwidth

**Tile streaming latency:**
- DDS file read (SSD): 0.5-2 ms per tile
- GPU upload: 0.1-0.3 ms per tile
- Total: 0.6-2.3 ms per tile

With 60 FPS budget (16.67 ms), can stream ~8-24 tiles/frame.

## VLM Applications

### Attention-Driven Texture Streaming

**Query-aware LOD selection for vision transformers:**

In ARR-COC-style systems, relevance realization determines which patches need high resolution. Sparse residency enables this dynamically:

```python
# Pseudo-code: VLM with sparse texture integration
class AttentionDrivenTextureVLM:
    def forward(self, image_pyramid: SparseTexture, query: Tensor):
        # Stage 1: Coarse attention (always resident level)
        coarse_features = self.vision_encoder(
            image_pyramid.sample(level=5)  # 128×128 always in cache
        )

        # Stage 2: Relevance realization
        relevance_map = self.compute_relevance(
            coarse_features, query
        )  # Shape: [H/4, W/4]

        # Stage 3: Request high-res tiles for relevant regions
        for patch in relevance_map.topk(k=64):
            x, y = patch.coordinates
            # Stream in finest level (0) for this patch
            image_pyramid.request_tile(level=0, x=x, y=y)

        # Stage 4: Encode with mixed resolutions
        fine_features = self.vision_encoder(
            image_pyramid.sample_adaptive(relevance_map)
        )

        return self.decode(fine_features, query)
```

**Benefits for VLM inference:**
1. **Memory efficiency**: Only load patches above relevance threshold
2. **Compute efficiency**: Skip high-res encoding for irrelevant regions
3. **Latency**: Async tile streaming overlaps with inference
4. **Scalability**: Handle gigapixel images (medical, satellite)

### Hierarchical Visual Features

**Multi-scale feature extraction with sparse pyramids:**

```python
# Each patch can have different LOD based on query relevance
class SparseFeaturePyramid:
    def __init__(self, sparse_texture: SparseTexture):
        self.texture = sparse_texture
        self.lod_budget = {
            0: 64,   # 64 tiles at finest level
            1: 128,  # 128 tiles at L1
            2: 256,  # 256 tiles at L2 (cheaper)
        }

    def extract_features(self, relevance_scores: Tensor) -> Dict[int, Tensor]:
        features = {}

        # Allocate tiles based on budget and relevance
        tile_requests = self.allocate_tiles(relevance_scores)

        for level, tiles in tile_requests.items():
            # Wait for tiles to stream (if not cached)
            self.texture.ensure_resident(level, tiles)

            # Extract features at this resolution
            features[level] = self.vision_encoder[level](
                self.texture.sample(level, tiles)
            )

        return features  # Mixed-resolution features
```

**Real-world use case: Satellite imagery VLM**
- Input: 100K × 100K satellite image (40 GB)
- Query: "Find all solar panel installations"
- With sparse residency:
  - Load full image at level 8 (390×390) = 600 KB
  - Relevance realization identifies ~200 candidate regions
  - Stream level 0 (finest) only for top 50 regions = 200 MB
  - Total memory: 201 MB vs 40 GB (99.5% reduction)

### Foveated Rendering for VLMs

**Gaze-contingent resolution** (VR/AR applications):

```python
# VLM with foveated visual encoding
class FoveatedVLM:
    def encode_foveated(self, image: SparseTexture, gaze: Tuple[int, int]):
        # Eccentricity-based LOD
        for y in range(image.height_in_tiles):
            for x in range(image.width_in_tiles):
                distance = sqrt((x - gaze[0])**2 + (y - gaze[1])**2)

                # LOD falls off with eccentricity (like human vision)
                if distance < 2:
                    level = 0  # Foveal: finest detail
                elif distance < 8:
                    level = 1  # Parafoveal: medium detail
                else:
                    level = 2  # Peripheral: coarse detail

                image.request_tile(level, x, y)
```

Matches human visual system: 60 pixels/degree foveal, 6 pixels/degree peripheral.

**Performance impact:**
- Full resolution: 4096×4096 = 67 MB
- Foveated (sparse): ~12 MB active
- Speedup: 5.6× in vision encoder

## Code Examples

### Vulkan Sparse Binding

```c
// Create sparse image
VkImageCreateInfo imageInfo = {
    .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    .imageType = VK_IMAGE_TYPE_2D,
    .format = VK_FORMAT_BC7_UNORM_BLOCK,
    .extent = {4096, 4096, 1},
    .mipLevels = 12,
    .flags = VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
    .usage = VK_IMAGE_USAGE_SAMPLED_BIT
};
vkCreateImage(device, &imageInfo, nullptr, &sparseImage);

// Query sparse properties
VkSparseImageMemoryRequirements sparseReqs;
vkGetImageSparseMemoryRequirements(device, sparseImage, &count, &sparseReqs);

// Bind tile memory
VkSparseImageMemoryBind bind = {
    .subresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0},
    .offset = {tileX * 128, tileY * 128, 0},
    .extent = {128, 128, 1},
    .memory = tileMemory,
    .memoryOffset = 0
};

VkBindSparseInfo bindInfo = {
    .sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
    .imageBindCount = 1,
    .pImageBinds = &sparseImageBind
};
vkQueueBindSparse(queue, 1, &bindInfo, fence);
```

### DirectX 12 Tile Mapping

```cpp
// Create tiled resource
D3D12_RESOURCE_DESC desc = {
    .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
    .Width = 4096,
    .Height = 4096,
    .MipLevels = 12,
    .Format = DXGI_FORMAT_BC7_UNORM,
    .Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE
};

ID3D12Resource* tiledTexture;
device->CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON,
                                nullptr, IID_PPV_ARGS(&tiledTexture));

// Create tile pool (heap for tiles)
D3D12_HEAP_DESC heapDesc = {
    .SizeInBytes = 64 * 1024 * 144,  // 144 tiles = 9 MB
    .Properties = {.Type = D3D12_HEAP_TYPE_DEFAULT},
    .Flags = D3D12_HEAP_FLAG_DENY_BUFFERS | D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES
};
ID3D12Heap* tileHeap;
device->CreateHeap(&heapDesc, IID_PPV_ARGS(&tileHeap));

// Map tiles
D3D12_TILED_RESOURCE_COORDINATE coord = {
    .X = tileX, .Y = tileY, .Subresource = mipLevel
};
D3D12_TILE_RANGE_FLAGS flags = D3D12_TILE_RANGE_FLAG_NONE;
UINT heapOffset = tileIndex;  // Index into tile heap

cmdQueue->UpdateTileMappings(
    tiledTexture, 1, &coord, nullptr,
    tileHeap, 1, &flags, &heapOffset, nullptr,
    D3D12_TILE_MAPPING_FLAG_NONE
);
```

## Sources

**Web Research:**
- [Vulkan Sparse Resources Documentation](https://docs.vulkan.org/spec/latest/chapters/sparsemem.html) - Khronos Vulkan Specification (accessed 2025-01-31)
- [Wicked Engine: Texture Streaming](https://wickedengine.net/2024/06/texture-streaming/) - Turánszki János, June 2024 (accessed 2025-01-31)
- [Sparse Virtual Textures - Gaia Sky](https://tonisagrista.com/blog/2023/sparse-virtual-textures/) - Toni Sagristà Selvas, January 2023 (accessed 2025-01-31)
- [Microsoft DirectX 12 Tiled Resources](https://learn.microsoft.com/en-us/windows/win32/direct3d12/volume-tiled-resources) - Microsoft Learn, December 2021 (accessed 2025-01-31)
- [D3D12 Tiled Resource Tier 4](https://microsoft.github.io/DirectX-Specs/d3d/D3D12TiledResourceTier4.html) - Microsoft DirectX Specs (accessed 2025-01-31)
- [DirectX 12 Sampler Feedback](https://microsoft.github.io/DirectX-Specs/d3d/SamplerFeedback.html) - Microsoft DirectX Specs (accessed 2025-01-31)

**Additional References:**
- Sean Barrett's [Virtual Textures (Megatextures) GDC 2008 Talk](https://silverspaceship.com/src/svt/) - id Software
- [GitHub: gpuweb sparse resources discussion](https://github.com/gpuweb/gpuweb/issues/455) - WebGPU sparse resource tracking (accessed 2025-01-31)
