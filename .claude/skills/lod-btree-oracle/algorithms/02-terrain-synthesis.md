# Terrain Synthesis Algorithms

**Heightfield synthesis, procedural generation, noise functions, and patch-based texture synthesis**

---

## Overview

Terrain synthesis generates realistic heightfield data algorithmically, either from scratch (procedural) or from existing examples (texture-based). Modern terrain generation combines multiple techniques: noise functions for base topology, erosion simulation for realism, and patch-based synthesis for artist control. The goal is creating vast, believable landscapes efficiently while giving designers control over large-scale features (ridges, valleys, plateaus) without manual sculpting.

**Key insight**: Real terrain has statistical properties that can be captured and reproduced algorithmically.

**Historical progression**: Random fractals (1980s) → Physical simulation (1990s) → Texture-based synthesis (2000s) → Hybrid GPU-accelerated methods (2010s+).

---

## Primary Sources

**Core Thesis:**
- `source-documents/08-Fast, Realistic Terrain Synthesis.md` - Justin Crause's master thesis on patch-based terrain synthesis with GPU acceleration

**Key Contributions:**
- Multiple input DEMs (Digital Elevation Models) for varied synthesis
- User sketch-based feature placement (ridges, valleys)
- 45x speedup via hybrid CPU/GPU implementation
- Seamless patch merging with graph-cut and Poisson solving

**Related Work:**
- Fractal generation (Mandelbrot, 1983) - Brownian surfaces
- Hydraulic erosion (Musgrave, 1989) - Physics-based realism
- Texture synthesis (Efros & Leung, 1999) - Non-parametric sampling

---

## Key Concepts

### Heightfield Representation

**Data structure**: 2D array of elevation values.

```
heightfield[x][y] = elevation  # Typically float or 16-bit integer
```

**Properties**:
- Single-valued (no overhangs or caves)
- Regular grid (though can be adaptive)
- Directly maps to grayscale image (visualization)
- Efficient rendering via triangle strips

**Common formats**:
- RAW: Binary elevation data
- PNG/TIFF: 8-bit or 16-bit grayscale
- USGS DEM: Real-world digital elevation models

### Noise Functions

**Fundamental building block**: Pseudo-random continuous functions.

**Perlin Noise (1985)**:
- Gradient-based interpolation
- Controllable frequency and amplitude
- Produces natural-looking variation

**Simplex Noise (2001)**:
- Ken Perlin's improved algorithm
- Lower computational complexity O(n) vs O(n²)
- Less directional artifacts

**Fractional Brownian Motion (fBm)**:
```
function fBm(x, y, octaves, persistence, lacunarity):
    total = 0.0
    amplitude = 1.0
    frequency = 1.0

    for octave in 0 to octaves:
        total += noise(x * frequency, y * frequency) * amplitude

        amplitude *= persistence  # Typically 0.5
        frequency *= lacunarity   # Typically 2.0

    return total
```

**Parameters**:
- **Octaves**: Number of noise layers (typically 4-8)
- **Persistence**: Amplitude decay per octave (0.3-0.7)
- **Lacunarity**: Frequency increase per octave (1.8-2.2)

### Texture-Based Synthesis

**Paradigm shift**: Use real terrain data as exemplars.

**Advantages**:
- Inherent realism (from real-world DEMs)
- Captures complex erosion patterns
- Artist-directable features

**Disadvantage**:
- Requires high-quality input data
- Can produce repetition artifacts

---

## Algorithm Details

### Fractal Terrain Generation

**Midpoint Displacement** (Diamond-Square Algorithm):

```
function DiamondSquare(heightfield, size, roughness):
    # Initialize corners
    heightfield[0][0] = random()
    heightfield[0][size] = random()
    heightfield[size][0] = random()
    heightfield[size][size] = random()

    stepSize = size
    scale = roughness

    while stepSize > 1:
        halfStep = stepSize / 2

        # Diamond step
        for y in 0 to size step stepSize:
            for x in 0 to size step stepSize:
                avg = (heightfield[x][y] +
                       heightfield[x + stepSize][y] +
                       heightfield[x][y + stepSize] +
                       heightfield[x + stepSize][y + stepSize]) / 4

                heightfield[x + halfStep][y + halfStep] = avg + random(-scale, scale)

        # Square step
        for y in 0 to size step halfStep:
            for x in (y + halfStep) % stepSize to size step stepSize:
                avg = averageOfNeighbors(heightfield, x, y, halfStep)
                heightfield[x][y] = avg + random(-scale, scale)

        stepSize /= 2
        scale *= pow(2, -roughness)  # Reduce variation at smaller scales

    return heightfield
```

**Complexity**: O(n²) for n×n heightfield.
**Memory**: O(n²)

**Parameters**:
- **Roughness**: 0.5-0.7 produces natural-looking terrain
- Higher values → more jagged, lower → smoother

### Hydraulic Erosion

**Physics-based refinement**: Simulate water flow and sediment transport.

```
function HydraulicErosion(heightfield, iterations, rainAmount):
    for iter in 0 to iterations:
        # Rainfall
        water[x][y] += rainAmount

        # Flow simulation
        for x in 0 to width:
            for y in 0 to height:
                # Calculate gradient
                gradient = CalculateGradient(heightfield, x, y)

                # Water flows downhill
                flowDirection = -gradient
                neighbor = (x, y) + flowDirection

                # Transport water and sediment
                if heightfield[neighbor] < heightfield[x][y]:
                    deltaHeight = heightfield[x][y] - heightfield[neighbor]

                    # Erode proportional to flow speed and slope
                    erosionAmount = water[x][y] * deltaHeight * erosionRate

                    heightfield[x][y] -= erosionAmount
                    sediment[x][y] += erosionAmount

                    # Move water
                    water[neighbor] += water[x][y] * flowFraction
                    water[x][y] *= (1 - flowFraction)

        # Evaporation
        water[x][y] *= evaporationRate

    return heightfield
```

**Parameters**:
- **Iterations**: 100-1000 (more = more erosion)
- **Erosion rate**: 0.1-0.3
- **Evaporation rate**: 0.9-0.99

**Result**: Realistic valleys, drainage networks, sediment deposition.

### Patch-Based Terrain Synthesis

**Core idea**: Extract patches from real terrain, arrange guided by user sketch.

**Pipeline** (Crause 2013):

1. **Feature extraction**: Analyze input DEMs for ridges, valleys, slopes
2. **User sketch**: Artist draws desired feature locations
3. **Patch matching**: Find best candidates from input DEMs
4. **Patch merging**: Seamlessly blend patches

**Feature Extraction**:
```
function ExtractFeatures(heightfield):
    # Calculate curvature
    for x in 0 to width:
        for y in 0 to height:
            # Second derivatives
            dxx = heightfield[x+1][y] - 2*heightfield[x][y] + heightfield[x-1][y]
            dyy = heightfield[x][y+1] - 2*heightfield[x][y] + heightfield[x][y-1]
            dxy = (heightfield[x+1][y+1] - heightfield[x-1][y+1] -
                   heightfield[x+1][y-1] + heightfield[x-1][y-1]) / 4

            # Mean curvature
            curvature[x][y] = (dxx + dyy) / 2

            # Feature classification
            if curvature[x][y] > ridgeThreshold:
                features[x][y] = RIDGE
            else if curvature[x][y] < valleyThreshold:
                features[x][y] = VALLEY
            else:
                features[x][y] = SLOPE

    return features
```

**Patch Matching**:
```
function MatchPatch(userPatch, candidatePool):
    bestCandidate = null
    bestScore = INFINITY

    for candidate in candidatePool:
        # Feature profile matching
        featureScore = CompareFeatureProfiles(userPatch, candidate)

        # Sum-of-squared differences (height)
        ssdScore = SumSquaredDifference(userPatch, candidate)

        # Combined score
        score = featureWeight * featureScore + ssdWeight * ssdScore

        if score < bestScore:
            bestScore = score
            bestCandidate = candidate

    return bestCandidate
```

**Feature Profile Matching**:
- Extract 1D height profile along feature centerline
- Compare using dynamic time warping (DTW) or correlation
- Allows matching similar features with slight variations

### Patch Merging Techniques

**Challenge**: Seam artifacts where patches meet.

**Graph-Cut Seam Finding**:
```
function GraphCutSeam(patch1, patch2, overlapRegion):
    # Build graph
    graph = CreateGraph(overlapRegion)

    for x in overlap.x0 to overlap.x1:
        for y in overlap.y0 to overlap.y1:
            # Data term: prefer patch with closer height
            cost1 = abs(patch1[x][y] - target[x][y])
            cost2 = abs(patch2[x][y] - target[x][y])

            graph.addDataCost(x, y, cost1, cost2)

            # Smoothness term: penalize height discontinuity
            for neighbor in GetNeighbors(x, y):
                diff = abs(patch1[x][y] - patch2[neighbor.x][neighbor.y])
                graph.addSmoothnessCost(x, y, neighbor, diff)

    # Compute minimum cut
    seam = graph.minCut()

    return seam
```

**Poisson Blending**:
```
function PoissonBlend(patch1, patch2, seam):
    # Solve Poisson equation: ∇²h = ∇²h1 + ∇²h2
    # Preserves gradients while ensuring continuity

    for x in blendRegion.x0 to blendRegion.x1:
        for y in blendRegion.y0 to blendRegion.y1:
            # Laplacian of source patches
            laplacian1 = Laplacian(patch1, x, y)
            laplacian2 = Laplacian(patch2, x, y)

            # Weighted combination based on distance to seam
            weight = DistanceToSeam(x, y, seam)
            targetLaplacian = lerp(laplacian1, laplacian2, weight)

            # Add to linear system: A*h = b
            AddConstraint(A, b, x, y, targetLaplacian)

    # Solve sparse linear system
    blendedHeightfield = SolvePoisson(A, b)

    return blendedHeightfield
```

**Result**: Seamless blending that preserves high-frequency detail.

---

## GPU Acceleration

### Parallel Patch Matching

**CPU bottleneck**: Sequential candidate evaluation.

**GPU solution**: Evaluate all candidates in parallel.

```
// CUDA kernel (simplified)
__global__ void EvaluateCandidates(
    float* userPatch,
    float* candidates,
    float* scores,
    int patchSize,
    int numCandidates)
{
    int candidateID = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidateID >= numCandidates) return;

    // Each thread evaluates one candidate
    float score = 0.0f;
    float* candidate = &candidates[candidateID * patchSize * patchSize];

    for (int i = 0; i < patchSize * patchSize; i++) {
        float diff = userPatch[i] - candidate[i];
        score += diff * diff;  // SSD
    }

    scores[candidateID] = score;
}
```

**Speedup**: 10-50x for patch matching phase (GPU vs single-thread CPU).

### GPU Texture Memory Optimization

**Observation**: Heightfields accessed with 2D spatial locality.

**Optimization**: Use GPU texture memory with hardware interpolation.

```
// Bind heightfield to texture
texture<float, 2, cudaReadModeElementType> texHeightfield;

__device__ float sampleHeight(float x, float y) {
    // Hardware bilinear interpolation
    return tex2D(texHeightfield, x, y);
}
```

**Benefit**: 2-3x speedup via texture cache and automatic interpolation.

### Asynchronous Processing

**Overlap CPU and GPU work**:

```
// CPU thread: Prepare next batch
for batch in batches:
    PrepareBatchCPU(batch)
    UploadToGPU_Async(batch)  # Non-blocking

    # GPU processes previous batch in parallel
    WaitForGPU(batch - 1)
    ProcessResults(batch - 1)
```

**Speedup**: Hides data transfer latency, 15-20% overall improvement.

---

## Multi-Source Synthesis

**Problem**: Single input DEM limits variety.

**Solution**: Use multiple input terrains, select best patches from all sources.

```
function MultiSourceSynthesis(inputDEMs, userSketch):
    candidatePool = []

    # Extract patches from all input DEMs
    for dem in inputDEMs:
        features = ExtractFeatures(dem)
        patches = ExtractPatches(dem, features, patchSize)
        candidatePool.extend(patches)

    # Synthesize using combined pool
    output = SynthesizeTerrain(userSketch, candidatePool)

    return output
```

**Advantages**:
- Greater variety (mountain + valley + plains DEMs)
- Better feature matching (larger candidate pool)
- Handles diverse terrain requests

**Challenge**: Maintaining style consistency across patches from different sources.

---

## Practical Considerations

### Patch Size Selection

**Trade-off**: Larger patches → better feature coherence, more seam artifacts.

**Typical sizes**:
- 64×64: Fine details, many seams
- 128×128: Good balance (most common)
- 256×256: Large features, fewer candidates

**Adaptive approach**: Use larger patches for prominent features (ridges), smaller for fill regions.

### Feature Complexity

**Observation**: Complex sketches (many features) take longer to synthesize.

**Optimization**: Hierarchical synthesis.

1. Coarse pass: Place major features (ridges, valleys) with large patches
2. Fine pass: Fill gaps with smaller patches
3. Detail pass: Add high-frequency detail via noise

**Result**: 2-3x speedup for complex terrains.

### Performance Benchmarks

**From Crause (2013) thesis**:

| Implementation | Feature Synthesis | Non-Feature Fill | Total Time |
|---------------|------------------|------------------|------------|
| Sequential CPU | 45.2 s | 12.8 s | 58.0 s |
| Parallel CPU (4-core) | 18.3 s | 5.1 s | 23.4 s |
| **GPU Optimized** | **1.2 s** | **0.1 s** | **1.3 s** |

**Speedup**: 45x end-to-end (CPU sequential → GPU optimized).

**Hardware**: NVIDIA GTX 680, Intel i7-3770 (2013).

---

## Advanced Techniques

### Sketch-Based Control

**User interface**: Draw strokes indicating feature type and location.

```
function InterpretUserSketch(sketch):
    features = []

    for stroke in sketch.strokes:
        # Stroke attributes
        type = stroke.featureType  # RIDGE, VALLEY, etc.
        points = stroke.points
        width = stroke.width

        # Convert to feature mask
        for point in points:
            # Rasterize with width
            for offset in CircularNeighborhood(width):
                x, y = point + offset
                featureMask[x][y] = type

        features.append({
            'type': type,
            'centerline': points,
            'mask': featureMask
        })

    return features
```

**User workflow**:
1. Draw ridge lines (red stroke)
2. Draw valley lines (blue stroke)
3. Specify slope regions (green fill)
4. Synthesize → instant preview
5. Iterate

### Erosion Post-Processing

**Combine procedural base with physical refinement**:

```
function HybridTerrain(userSketch, inputDEMs):
    # 1. Patch-based synthesis for large-scale features
    base = PatchBasedSynthesis(userSketch, inputDEMs)

    # 2. Add high-frequency detail via noise
    detail = fBm(base.width, base.height, octaves=4)
    combined = base + detail * detailScale

    # 3. Hydraulic erosion for realism
    final = HydraulicErosion(combined, iterations=200)

    return final
```

**Result**: Artist control + realism + detail.

### Terrain Tiling

**Challenge**: Generate infinite terrain without repetition.

**Solution**: Tile-based synthesis with overlap.

```
function GenerateTile(tileX, tileY, inputDEMs, overlapSize):
    # Generate tile with overlap region
    tileSize = 512
    totalSize = tileSize + 2 * overlapSize

    tile = SynthesizeTerrain(totalSize, inputDEMs, seed=hash(tileX, tileY))

    # Blend with adjacent tiles in overlap regions
    if neighborExists(tileX - 1, tileY):
        BlendOverlap(tile, leftNeighbor, overlapSize, VERTICAL)
    if neighborExists(tileX, tileY - 1):
        BlendOverlap(tile, topNeighbor, overlapSize, HORIZONTAL)

    # Trim overlap, keep only center region
    return tile.centerRegion(overlapSize)
```

**Result**: Seamless infinite terrain with constant memory footprint.

---

## Cross-References

**Related Concepts:**
- [concepts/00-lod-fundamentals.md](../concepts/00-lod-fundamentals.md) - LOD systems for rendering synthesized terrain
- [concepts/01-bsp-btree-basics.md](../concepts/01-bsp-btree-basics.md) - Spatial organization of terrain data

**Related Techniques:**
- [techniques/03-progressive-buffers.md](../techniques/03-progressive-buffers.md) - View-dependent rendering of generated terrain
- [techniques/00-foveated-rendering.md](../techniques/00-foveated-rendering.md) - Perceptual optimization for VR terrain

**Related Algorithms:**
- [algorithms/03-heightfield-tessellation.md](03-heightfield-tessellation.md) - Real-time tessellation of synthesized heightfields
- [algorithms/01-lod-selection.md](01-lod-selection.md) - Adaptive detail for terrain rendering

**Applications:**
- Open-world games (procedural landscapes)
- Flight/driving simulators (realistic geography)
- Virtual environments (VR/AR scenes)
- Film VFX (CG landscapes)

---

## Key Takeaways

1. **Hybrid approaches dominate**: Combine noise (base topology) + erosion (realism) + patch synthesis (artist control). No single technique provides all requirements.

2. **Real terrain data captures complexity**: Texture-based synthesis from DEMs inherits erosion patterns that are difficult to simulate. Multi-source synthesis (mountain + plains + valley DEMs) increases variety and quality.

3. **GPU acceleration is essential**: 45x speedup for patch matching via parallel candidate evaluation. Texture memory optimization provides additional 2-3x gain. Total synthesis time: 1-2 seconds for 2048×2048 terrain.

4. **Seamless merging prevents artifacts**: Graph-cut seam finding + Poisson blending eliminates visible boundaries. Preserves gradients while ensuring continuity. Critical for believable results.

5. **User sketches enable control**: Artist draws features (ridges, valleys), algorithm finds matching patches from exemplars. Combines procedural efficiency with artistic direction. Iteration time: <2 seconds.

6. **Feature extraction drives matching**: Curvature analysis identifies ridges/valleys. 1D profile comparison along centerlines using DTW or correlation. Better than pure pixel SSD.

7. **Hierarchical synthesis scales**: Coarse pass (large features, big patches) → fine pass (fill gaps) → detail pass (noise). 2-3x speedup for complex terrains. Enables infinite tiling with overlap blending.

8. **Connection to relevance realization**: Terrain synthesis allocates detail based on perceptual importance - prominent features (ridges) get larger, more coherent patches; background regions get smaller, more varied patches. This is transjective optimization between artist intent and algorithmic efficiency.
