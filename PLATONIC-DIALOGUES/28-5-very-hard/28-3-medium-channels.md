---
summary: whereby the oracles implement standard computer vision algorithms for saliency and distance fields, adding Itti-Koch saliency model (~2ms for multi-scale center-surround feature competition in CIE Lab color space capturing "pop-out" regions distinct from edge detection) and distance transform via Euclidean distance field computation (~1ms for nearest-boundary calculations), demonstrating that well-established CV techniques with existing implementations provide the "medium difficulty" tier at reasonable computational cost, bringing total to 13 channels while remaining within practical performance budgets
---

# Part 28-3: Medium Channels - Standard Algorithms
*Wherein the oracles implement saliency maps and distance fields using established CV techniques*

---

## Building on Parts 28-1 and 28-2

**KARPATHY:**
We have 11 channels: RGB + position + filters. What's the next complexity tier?

**LOD ORACLE:**
Saliency maps and distance fields. Not simple convolutions, but well-established algorithms with existing implementations.

**MUSE BIRD:**
🐦 *MEDIUM = USE EXISTING CODE! DON'T REINVENT!*

---

## Saliency Maps - Channel 11

**KARPATHY:**
What's a saliency map?

**LOD ORACLE:**
A heatmap of "where humans look first." Based on low-level features: color, intensity, orientation.

**KARPATHY:**
How's it different from edge detection?

**LOD ORACLE:**
Edges detect boundaries. Saliency detects "pop-out" regions—things that stand out from surroundings.

Example:
- Red apple on green grass → High saliency (color contrast)
- Gray rock on gray pavement → Low saliency (blends in)

Both might have strong edges, but different saliency.

**MUSE BIRD:**
🐦 *SALIENCY = ATTENTION GRABBING! EDGES = BOUNDARIES!*

---

## Itti-Koch Saliency Model

**KARPATHY:**
How do you compute it?

**LOD ORACLE:**
Classic approach: Itti-Koch-Niebur 1998 model. Multi-scale feature extraction + center-surround differences.

```python
def generate_saliency_channel_itti_koch(image):
    """
    Itti-Koch saliency: Multi-scale feature competition.

    Process:
    1. Extract features at multiple scales (Gaussian pyramid)
    2. Compute center-surround differences
    3. Normalize and combine

    Cost: ~2ms (multiple pyramid levels + feature extraction)
    """
    # Convert to CIE Lab color space (perceptually uniform)
    lab = rgb_to_lab(image)
    L, a, b = lab[0], lab[1], lab[2]

    # Build Gaussian pyramid (9 levels)
    pyramid_L = build_pyramid(L, levels=9)
    pyramid_a = build_pyramid(a, levels=9)
    pyramid_b = build_pyramid(b, levels=9)

    saliency_maps = []

    # Center-surround at different scales
    for c in [2, 3, 4]:  # Center levels
        for s in [5, 6]:  # Surround levels (coarser)
            # Intensity
            center_L = pyramid_L[c]
            surround_L = F.interpolate(pyramid_L[s], size=center_L.shape[-2:])
            intensity_cs = torch.abs(center_L - surround_L)

            # Color (red-green, blue-yellow)
            center_a = pyramid_a[c]
            surround_a = F.interpolate(pyramid_a[s], size=center_a.shape[-2:])
            rg_cs = torch.abs(center_a - surround_a)

            center_b = pyramid_b[c]
            surround_b = F.interpolate(pyramid_b[s], size=center_b.shape[-2:])
            by_cs = torch.abs(center_b - surround_b)

            # Combine
            combined = intensity_cs + rg_cs + by_cs
            saliency_maps.append(combined)

    # Normalize and average
    saliency = torch.stack(saliency_maps).mean(dim=0)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    return saliency.squeeze()
```

**KARPATHY:**
That's... kind of complicated.

**LOD ORACLE:**
Yeah. But there's a shortcut: use an existing library.

```python
import cv2

def generate_saliency_opencv(image):
    """
    OpenCV has built-in saliency detectors.

    Cost: ~1ms (optimized C++)
    """
    # Convert to numpy
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)

    # OpenCV saliency detector
    saliency_detector = cv2.saliency.StaticSaliencyFineGrained_create()
    success, saliency_map = saliency_detector.computeSaliency(img_np)

    # Convert back to torch
    saliency_torch = torch.from_numpy(saliency_map).to(image.device)

    return saliency_torch
```

**KARPATHY:**
So we don't implement it ourselves, just call OpenCV?

**LOD ORACLE:**
Exactly. OpenCV's implementation is battle-tested and fast.

**MUSE BIRD:**
🐦 *DON'T REINVENT WHEEL! USE EXISTING LIBRARIES!*

---

## Deep Learning Saliency - Alternative

**KARPATHY:**
What about learned saliency models?

**LOD ORACLE:**
There are CNNs trained on eye-tracking datasets. Like SAM-ResNet or EML-NET.

```python
def generate_saliency_deeplearning(image):
    """
    Deep learning saliency using pretrained model.

    Cost: ~5ms (forward pass through ResNet)
    Memory: ~200MB model weights
    """
    # Load pretrained saliency model
    model = torch.hub.load('saliency-detection/EML-NET', 'emlnet')
    model = model.cuda().eval()

    with torch.no_grad():
        saliency = model(image.unsqueeze(0))  # [1, 1, H, W]

    return saliency.squeeze()
```

**KARPATHY:**
That's slower than OpenCV (5ms vs 1ms). Worth it?

**LOD ORACLE:**
Depends. Deep learning saliency is more accurate (trained on human eye-tracking). But 5× slower.

**Trade-off:**
- OpenCV: Fast (1ms), decent accuracy, no GPU needed
- Deep learning: Slow (5ms), best accuracy, requires GPU + model weights

**KARPATHY:**
For ARR-COC, which would you choose?

**LOD ORACLE:**
Start with OpenCV. If saliency becomes critical, upgrade to deep learning.

**MUSE BIRD:**
🐦 *FAST AND GOOD BEATS SLOW AND PERFECT!*

---

## Distance Fields - Channel 12

**KARPATHY:**
What's a distance field?

**LOD ORACLE:**
For every pixel: "How far to the nearest edge?"

```
Example:
Edge map:        Distance field:
. . . . .        4 3 2 1 0
. . X . .   →    3 2 1 0 1
. . . . .        4 3 2 1 0

X = edge pixel
Numbers = distance to nearest edge
```

**KARPATHY:**
Why is this useful?

**LOD ORACLE:**
Early culling! If a patch is FAR from any edges, it's probably uniform/boring.

```python
def should_process_patch(distance_value, threshold=0.8):
    if distance_value > threshold:
        # This patch is >80% of image width away from edges
        # Probably sky, uniform background, etc.
        return False  # Skip it!
    return True  # Process it
```

**MUSE BIRD:**
🐦 *FAR FROM EDGES = BORING! SAVE COMPUTE!*

---

## Jump Flooding Algorithm

**KARPATHY:**
How do you compute distance fields efficiently?

**LOD ORACLE:**
Naive approach: For each pixel, search entire image for nearest edge. O(N²) → SLOW.

Smart approach: Jump Flooding Algorithm (JFA). O(N log N) → FAST.

```python
def generate_distance_field_jfa(edge_map):
    """
    Jump Flooding Algorithm for distance fields.

    Idea: Iteratively propagate edge information in power-of-2 jumps.

    Cost: ~0.5ms (log(N) iterations, parallel on GPU)
    """
    H, W = edge_map.shape

    # Initialize: edges have distance 0, others have infinity
    distance = torch.full((H, W), float('inf'), device=edge_map.device)
    distance[edge_map > 0] = 0

    # Coordinate map: each pixel stores (y, x) of nearest edge
    coords = torch.zeros(H, W, 2, device=edge_map.device)
    edge_ys, edge_xs = torch.where(edge_map > 0)
    coords[edge_ys, edge_xs, 0] = edge_ys.float()
    coords[edge_ys, edge_xs, 1] = edge_xs.float()

    # Jump Flooding: k = W/2, W/4, W/8, ..., 1
    k = W // 2
    while k >= 1:
        new_coords = coords.clone()

        for dy in [-k, 0, k]:
            for dx in [-k, 0, k]:
                if dy == 0 and dx == 0:
                    continue

                # Sample neighbor at offset (dy, dx)
                neighbor_coords = sample_with_offset(coords, dy, dx)

                # Compute distance to neighbor's edge
                y_grid, x_grid = torch.meshgrid(
                    torch.arange(H, device=edge_map.device),
                    torch.arange(W, device=edge_map.device),
                    indexing='ij'
                )
                dist_to_edge = torch.sqrt(
                    (y_grid - neighbor_coords[:, :, 0])**2 +
                    (x_grid - neighbor_coords[:, :, 1])**2
                )

                # Update if closer
                mask = dist_to_edge < distance
                new_coords[mask] = neighbor_coords[mask]
                distance[mask] = dist_to_edge[mask]

        coords = new_coords
        k //= 2

    # Normalize to [0, 1]
    distance = distance / distance.max()

    return distance
```

**KARPATHY:**
That's more complex than I expected.

**LOD ORACLE:**
Yeah. But again, there are libraries:

```python
import scipy.ndimage

def generate_distance_field_scipy(edge_map):
    """
    SciPy has optimized distance transform.

    Cost: ~0.2ms (optimized C code)
    """
    edge_np = edge_map.cpu().numpy()

    # Invert: distance transform works on background, not edges
    background = (edge_np == 0)

    # Compute distance
    distance = scipy.ndimage.distance_transform_edt(background)

    # Normalize
    distance = distance / distance.max()

    return torch.from_numpy(distance).to(edge_map.device)
```

**KARPATHY:**
So use scipy instead of implementing JFA?

**LOD ORACLE:**
For prototyping, yes. For production GPU code, implement JFA in CUDA.

**MUSE BIRD:**
🐦 *PROTOTYPE: SCIPY! PRODUCTION: CUDA!*

---

## Combined: The 13-Channel Array

**KARPATHY:**
What do we have now?

**LOD ORACLE:**
```python
class MediumTextureArray:
    """
    13-channel texture array:
    0-2: RGB
    3-5: Position
    6-7: Edges (normal + inverted)
    8-9: Highpass + Lowpass
    10: Motion
    11: Saliency (NEW!)
    12: Distance field (NEW!)
    """

    def __init__(self, image, previous_frame=None):
        # From Part 28-2
        easy_channels = EasyTextureArray(image, previous_frame).texture  # [11, H, W]

        # NEW: Saliency
        saliency = self._generate_saliency(image)  # [H, W]

        # NEW: Distance field
        edges = easy_channels[6]  # Use normal edges
        distance = self._generate_distance_field(edges)  # [H, W]

        # Combine
        self.texture = torch.cat([
            easy_channels,           # 0-10
            saliency.unsqueeze(0),  # 11
            distance.unsqueeze(0)   # 12
        ], dim=0)  # [13, H, W]

    def _generate_saliency(self, image):
        # Use OpenCV for speed
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        detector = cv2.saliency.StaticSaliencyFineGrained_create()
        _, saliency = detector.computeSaliency(img_np)

        return torch.from_numpy(saliency).to(image.device)

    def _generate_distance_field(self, edges):
        # Use scipy for simplicity
        edge_np = (edges > 0.1).cpu().numpy()  # Threshold edges
        background = ~edge_np

        distance = scipy.ndimage.distance_transform_edt(background)
        distance = distance / distance.max()

        return torch.from_numpy(distance).to(edges.device)
```

**KARPATHY:**
Generation cost?

**LOD ORACLE:**
- Easy channels: 0.06ms (from Part 28-2)
- Saliency (OpenCV): 1ms
- Distance field (scipy): 0.2ms
- **Total: 1.26ms**

**KARPATHY:**
Still under 2ms. Not bad.

**MUSE BIRD:**
🐦 *13 CHANNELS! 1.26ms! TEXTURE ARRAY GROWS!*

---

## When to Use Saliency vs Edges

**KARPATHY:**
If I have edge channels (6-7), do I need saliency (11)?

**LOD ORACLE:**
They catch different things.

**Edges catch:**
- Boundaries between regions
- Text (high contrast)
- Object outlines

**Saliency catches:**
- Unusual colors (red in green scene)
- Bright spots in dark images
- Objects that "pop out" perceptually

**Example where saliency wins:**
```python
# Image: Blue sky with one white cloud (no edges within cloud)
edges = detect_edges(image)
# Result: Strong edges at cloud boundary, weak inside

saliency = compute_saliency(image)
# Result: Entire cloud region is salient (bright vs dark sky)
```

**KARPATHY:**
So saliency is region-based, edges are boundary-based?

**LOD ORACLE:**
Exactly. Saliency says "this AREA is interesting." Edges say "this LINE is interesting."

**MUSE BIRD:**
🐦 *EDGES: LINES! SALIENCY: BLOBS! BOTH NEEDED!*

---

## Using Distance Fields for Culling

**KARPATHY:**
Show me how distance fields speed up the cascade.

**LOD ORACLE:**
```python
def cascade_with_distance_culling(texture, candidate_positions):
    """
    Use distance field (channel 12) to skip uniform regions.

    Args:
        texture: [13, H, W] texture array
        candidate_positions: List of (y, x) positions

    Returns:
        filtered_positions: Subset after culling
    """
    distance_channel = texture[12]  # [H, W]

    filtered = []
    for y, x in candidate_positions:
        dist = distance_channel[y, x]

        if dist > 0.8:
            # This position is >80% of max distance from any edge
            # Probably uniform region (sky, solid color, etc.)
            continue  # SKIP!

        # Otherwise, keep it
        filtered.append((y, x))

    culled = len(candidate_positions) - len(filtered)
    print(f"Culled {culled} / {len(candidate_positions)} uniform patches")

    return filtered
```

**KARPATHY:**
How much does this help?

**LOD ORACLE:**
Depends on image content:

**Sparse image (lots of uniform regions):**
- Before culling: 4096 candidate patches
- After culling: 1024 patches (75% reduction!)

**Dense image (edges everywhere):**
- Before culling: 4096 patches
- After culling: 3500 patches (14% reduction)

**KARPATHY:**
So it's content-adaptive. Saves more compute on simple images?

**LOD ORACLE:**
Right. Which is good! Complex images NEED more patches. Simple images don't.

**MUSE BIRD:**
🐦 *HARD IMAGES: FULL COMPUTE! EASY IMAGES: SKIP STUFF!*

---

## Combining Saliency + Distance for Scoring

**KARPATHY:**
Can you use both saliency and distance together?

**LOD ORACLE:**
Yeah:

```python
def score_patch_with_medium_channels(texture, position):
    """
    Combine all 13 channels for relevance scoring.

    Args:
        texture: [13, H, W]
        position: (y, x)

    Returns:
        score: float
    """
    y, x = position
    features = texture[:, y, x]  # [13]

    # Visual features (OR logic from Part 28-2)
    edges = max(features[6], features[7])
    highpass = features[8]
    motion = features[10]
    visual_score = max(edges, highpass, motion)

    # NEW: Saliency boost
    saliency = features[11]
    visual_score = max(visual_score, saliency)  # OR with saliency

    # NEW: Distance culling (negative signal)
    distance = features[12]
    if distance > 0.8:
        visual_score *= 0.1  # Heavily penalize far-from-edges

    # Foveal bias (from Part 28-1)
    eccentricity = features[5]
    foveal_weight = 1.0 - 0.5 * eccentricity

    return visual_score * foveal_weight
```

**KARPATHY:**
So saliency is another OR channel, but distance is AND logic (must pass threshold)?

**LOD ORACLE:**
Exactly. Saliency says "this might be interesting." Distance says "this is definitely NOT interesting."

**MUSE BIRD:**
🐦 *SALIENCY: POSITIVE SIGNAL! DISTANCE: NEGATIVE SIGNAL!*

---

## Testing on Natural Images

**KARPATHY:**
How do we validate that saliency and distance actually help?

**LOD ORACLE:**
Compare token allocation with vs without these channels.

```python
def test_medium_channels_on_coco():
    dataset = load_coco_val()

    for sample in dataset:
        image = sample['image']
        objects = sample['objects']  # Ground truth bounding boxes

        # Strategy 1: Easy channels only (11 channels)
        easy_texture = EasyTextureArray(image)
        positions_easy = allocate_tokens(easy_texture, budget=273)

        # Strategy 2: Medium channels (13 channels)
        medium_texture = MediumTextureArray(image)
        positions_medium = allocate_tokens(medium_texture, budget=273)

        # Measure coverage: how many object boxes do we overlap?
        coverage_easy = compute_coverage(positions_easy, objects)
        coverage_medium = compute_coverage(positions_medium, objects)

        print(f"Easy: {coverage_easy:.2%}")
        print(f"Medium: {coverage_medium:.2%}")

    # Expected: Medium has higher coverage (saliency catches blobs, not just edges)
```

**KARPATHY:**
What if saliency doesn't help?

**LOD ORACLE:**
Then don't use it! These are hypotheses. Test them.

**MUSE BIRD:**
🐦 *MEASURE! VALIDATE! SCIENCE!*

---

## Memory and Compute Cost

**KARPATHY:**
What's the total cost now?

**LOD ORACLE:**
**Memory (1024×1024 image):**
- Part 28-1: 24 MB (6 channels)
- Part 28-2: 44 MB (11 channels)
- Part 28-3: 52 MB (13 channels)
- Increase: +8 MB

**Compute:**
- Part 28-1: 0.001ms
- Part 28-2: 0.06ms
- Part 28-3: 1.26ms
- Increase: +1.2ms (mostly saliency)

**KARPATHY:**
Saliency is the expensive part (1ms out of 1.26ms total). Worth it?

**LOD ORACLE:**
Trade-off question:
- If saliency improves token allocation by >10%, yes
- If improvement is <5%, no

**MUSE BIRD:**
🐦 *BENCHMARK DECIDES! NOT INTUITION!*

---

## Alternative: Skip Saliency, Keep Distance

**KARPATHY:**
What if we skip saliency but keep distance fields?

**LOD ORACLE:**
Then you have 12 channels, 0.26ms cost.

```python
class MediumLiteTextureArray:
    """
    12-channel texture array (no saliency):
    0-10: Easy channels
    11: Distance field only

    Cost: 0.26ms (vs 1.26ms with saliency)
    """
    pass
```

**KARPATHY:**
When would you prefer this?

**LOD ORACLE:**
Real-time video (need speed) or when saliency doesn't help your task (documents, text-heavy images).

**MUSE BIRD:**
🐦 *SALIENCY: OPTIONAL! DISTANCE: CHEAP WINS!*

---

## Integration with Vervaekean Framework

**KARPATHY:**
How do these medium channels map to your three ways of knowing?

**LOD ORACLE:**
```python
# knowing.py with medium channels

class InformationScorer:
    """Propositional knowing - statistical content"""
    def score(self, texture, position):
        features = texture[:, position.y, position.x]

        # Easy channels: edges, highpass
        edges = max(features[6], features[7])
        highpass = features[8]

        # NEW: Distance field (information structure)
        distance = features[12]
        structure_score = 1.0 - distance  # Close to edges = high structure

        return 0.5 * edges + 0.3 * highpass + 0.2 * structure_score


class PerspectivalScorer:
    """Perspectival knowing - salience landscape"""
    def score(self, texture, position):
        features = texture[:, position.y, position.x]

        # Easy channels: motion, eccentricity
        motion = features[10]
        eccentricity = features[5]

        # NEW: Saliency (what stands out perceptually)
        saliency = features[11]

        foveal_weight = 1.0 - 0.5 * eccentricity
        return (0.6 * saliency + 0.4 * motion) * foveal_weight


class ParticipatoryScorer:
    """Query-content coupling - still needs CLIP (Part 28-5)"""
    def score(self, texture, position, query):
        return 0.5  # Placeholder
```

**KARPATHY:**
So saliency goes with Perspectival (what stands out), distance goes with Information (structure)?

**LOD ORACLE:**
Yeah. Saliency is subjective (depends on context). Distance is objective (measured from edges).

**MUSE BIRD:**
🐦 *SALIENCY: PERSPECTIVE! DISTANCE: INFORMATION!*

---

## Summary and Next Steps

**KARPATHY:**
What did we build?

**LOD ORACLE:**
**Channels 11-12 (Medium):**
- Saliency: Perceptual pop-out (uses OpenCV)
- Distance fields: Distance to nearest edge (uses scipy)

**Cost:** +1.2ms (mostly saliency)
**Memory:** +8 MB
**Benefit:**
- Saliency catches regions (not just edges)
- Distance culls uniform areas (saves compute)

**KARPATHY:**
Next is Part 28-4: Hard channels (SAM clusters, OCR)?

**LOD ORACLE:**
Yep. Those need external models (SAM for segmentation, EasyOCR for text).

**MUSE BIRD:**
🐦 *MEDIUM DONE! HARD NEXT! GETTING REAL!*

---

**END OF PART 28-3**

∿◇∿

## Appendix: Complete Medium Texture Array Code

```python
import torch
import cv2
import scipy.ndimage
import numpy as np

class MediumTextureArray:
    """13-channel texture array with saliency and distance fields"""

    def __init__(self, image, previous_frame=None):
        # Easy channels from Part 28-2
        easy = EasyTextureArray(image, previous_frame)
        self.texture = self._add_medium_channels(easy.texture, image)

    def _add_medium_channels(self, easy_texture, image):
        # Saliency (channel 11)
        saliency = self._compute_saliency(image)

        # Distance field (channel 12)
        edges = easy_texture[6]  # Normal edges
        distance = self._compute_distance(edges)

        return torch.cat([
            easy_texture,            # 0-10
            saliency.unsqueeze(0),  # 11
            distance.unsqueeze(0)   # 12
        ], dim=0)

    def _compute_saliency(self, image):
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        detector = cv2.saliency.StaticSaliencyFineGrained_create()
        _, sal = detector.computeSaliency(img_np)

        return torch.from_numpy(sal).float().to(image.device)

    def _compute_distance(self, edges):
        edge_np = (edges > 0.1).cpu().numpy()
        background = ~edge_np

        dist = scipy.ndimage.distance_transform_edt(background)
        dist = dist / (dist.max() + 1e-8)

        return torch.from_numpy(dist).float().to(edges.device)

# Usage
image = torch.randn(3, 1024, 1024).cuda()
texture = MediumTextureArray(image)

print(f"Texture shape: {texture.texture.shape}")  # [13, 1024, 1024]
print(f"Generation cost: ~1.26ms")
print(f"Memory: 52 MB")
```

∿◇∿
