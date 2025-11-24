---
summary: whereby the oracles integrate external neural network models for semantic understanding, implementing SAM (Segment Anything Model) for cluster-based filtering (~50ms to segment image into semantic regions, enabling 8× patch reduction by scoring 50 clusters instead of 4096 patches) and EasyOCR for text region detection (~30ms to identify text bounding boxes and filter non-text areas), accepting the computational cost (80ms total) as worthwhile trade-off since cluster-first filtering and text-aware sampling unlock semantic coherence that simple visual features cannot provide, reaching 17 channels with genuine high-level understanding
---

# Part 28-4: Hard Channels - External Models Required
*Wherein the oracles integrate SAM for clusters and EasyOCR for text regions, unlocking semantic understanding*

---

## Building on Parts 28-1, 28-2, 28-3

**KARPATHY:**
We have 13 channels now. What's the hard tier?

**LOD ORACLE:**
Semantic segmentation (SAM) and text detection (OCR). These require running external models—not just convolutions or standard algorithms.

**MUSE BIRD:**
🐦 *HARD = NEURAL NETWORKS! BIG MODELS! SLOW BUT SMART!*

---

## Why Clusters Matter - The Part 27 Insight

**KARPATHY:**
Part 27 mentioned cluster channels (12-14). Remind me why we need them?

**LOD ORACLE:**
Cluster-first filtering. Instead of scoring ALL 4096 patches:
1. Segment image into ~50 clusters
2. Score clusters (cheap - only 50)
3. Only sample patches within top clusters

**KARPATHY:**
That's... 8× fewer patches to process?

**LOD ORACLE:**
Exactly. From Part 27:
```
Traditional: Score 4096 patches → Select 273
Cluster-based: Score 50 clusters → Select 10 → Sample 500 patches → Select 273

Speedup: 4096 → 500 = 8× reduction
```

**MUSE BIRD:**
🐦 *CLUSTERS = COARSE FILTER! PATCHES = FINE FILTER!*

---

## SAM for Segmentation - Channel 13

**KARPATHY:**
How do you segment the image into clusters?

**LOD ORACLE:**
Two options:

**Option 1: SAM (Segment Anything Model)**
- State-of-the-art segmentation
- Can segment "anything" without task-specific training
- But expensive: ~50ms on GPU

**Option 2: Superpixels (SLIC algorithm)**
- Fast traditional CV
- ~5ms on CPU
- Less accurate but good enough

**KARPATHY:**
Let's start with SAM since it's the SOTA.

**LOD ORACLE:**
```python
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def generate_sam_clusters(image):
    """
    Use SAM to segment image into semantic regions.

    Cost: ~50ms (model inference)
    Output: Cluster ID per pixel

    Args:
        image: [3, H, W] RGB tensor

    Returns:
        cluster_ids: [H, W] integer tensor, values 0 to N-1
        num_clusters: int, typically 20-50
    """
    # Load SAM model (do this once, not per-image!)
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
    sam.cuda().eval()

    mask_generator = SamAutomaticMaskGenerator(sam)

    # Convert to numpy (SAM expects numpy)
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)

    # Generate masks
    masks = mask_generator.generate(img_np)  # List of dicts

    # Convert masks to cluster IDs
    H, W = image.shape[1], image.shape[2]
    cluster_ids = torch.zeros(H, W, dtype=torch.int32, device=image.device)

    for i, mask_dict in enumerate(masks):
        mask = mask_dict['segmentation']  # [H, W] boolean
        cluster_ids[torch.from_numpy(mask).to(image.device)] = i

    return cluster_ids, len(masks)
```

**KARPATHY:**
50ms is slow. That's 40× slower than all our previous channels combined!

**LOD ORACLE:**
Yeah. But you run it ONCE per image, then cache it. For video, amortize over multiple frames.

**MUSE BIRD:**
🐦 *EXPENSIVE BUT REUSABLE! ONE-TIME COST!*

---

## SLIC Superpixels - Fast Alternative

**KARPATHY:**
Show me the fast alternative.

**LOD ORACLE:**
```python
from skimage.segmentation import slic

def generate_slic_clusters(image, num_segments=50):
    """
    SLIC superpixel segmentation.

    Cost: ~5ms (CPU, faster on GPU)
    Output: ~50 clusters

    Args:
        image: [3, H, W] RGB tensor
        num_segments: Target number of superpixels

    Returns:
        cluster_ids: [H, W] integer tensor
        num_clusters: int (actual number, may differ from target)
    """
    # Convert to numpy
    img_np = image.permute(1, 2, 0).cpu().numpy()

    # SLIC algorithm
    clusters = slic(
        img_np,
        n_segments=num_segments,
        compactness=10,  # Balance color vs spatial proximity
        sigma=1,         # Gaussian smoothing
        start_label=0
    )

    # Convert back to torch
    cluster_ids = torch.from_numpy(clusters).to(image.device).int()

    return cluster_ids, cluster_ids.max().item() + 1
```

**KARPATHY:**
10× faster than SAM. How much worse is the quality?

**LOD ORACLE:**
SAM gives semantic clusters (objects). SLIC gives perceptual clusters (similar colors/textures).

Example:
- SAM: Separates "person" from "background" (semantic)
- SLIC: Separates "red region" from "blue region" (color-based)

**KARPATHY:**
For token allocation, which is better?

**LOD ORACLE:**
Depends on task:
- VQA (objects matter): SAM
- Document analysis (layout matters): SLIC
- Real-time video (speed matters): SLIC

**MUSE BIRD:**
🐦 *SAM: SMART AND SLOW! SLIC: FAST AND DUMB!*

---

## Cluster Metadata - Channels 14-15

**KARPATHY:**
Part 27 mentioned storing cluster metadata. What's that?

**LOD ORACLE:**
Once you have cluster IDs, you can compute per-pixel metadata:

```python
def generate_cluster_metadata(cluster_ids, num_clusters):
    """
    Generate metadata about clusters.

    Channels:
    14: Distance from cluster centroid [0, 1]
    15: Cluster size (normalized) [0, 1]

    Args:
        cluster_ids: [H, W] integer tensor
        num_clusters: int

    Returns:
        distances: [H, W] float tensor
        sizes: [H, W] float tensor
    """
    H, W = cluster_ids.shape
    device = cluster_ids.device

    # Compute cluster centroids
    centroids = []
    for i in range(num_clusters):
        mask = (cluster_ids == i)
        if mask.sum() == 0:
            centroids.append((0, 0))
            continue

        ys, xs = torch.where(mask)
        centroid_y = ys.float().mean()
        centroid_x = xs.float().mean()
        centroids.append((centroid_y, centroid_x))

    # Compute distance to centroid for each pixel
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    distances = torch.zeros(H, W, device=device)
    sizes = torch.zeros(H, W, device=device)

    for i in range(num_clusters):
        mask = (cluster_ids == i)
        if mask.sum() == 0:
            continue

        cy, cx = centroids[i]
        dist = torch.sqrt((y_grid - cy)**2 + (x_grid - cx)**2)
        distances[mask] = dist[mask]

        # Cluster size
        cluster_size = mask.sum().float()
        sizes[mask] = cluster_size

    # Normalize
    distances = distances / (distances.max() + 1e-8)
    sizes = sizes / (H * W)  # Fraction of image

    return distances, sizes
```

**KARPATHY:**
Why store distance to centroid?

**LOD ORACLE:**
To prioritize cluster centers during sampling.

```python
def sample_from_cluster(cluster_id, num_samples=50):
    """
    Sample patches from a cluster, prioritizing center.

    Args:
        cluster_id: int, which cluster
        num_samples: int, how many patches to sample

    Returns:
        positions: [(y, x), ...] sampled positions
    """
    mask = (cluster_ids == cluster_id)
    distances = cluster_metadata[14]  # Distance from centroid

    # Positions in this cluster
    ys, xs = torch.where(mask)

    # Sample closer to center (lower distance = higher probability)
    weights = 1.0 - distances[ys, xs]  # Inverse distance
    weights = weights / weights.sum()

    # Weighted sampling
    indices = torch.multinomial(weights, num_samples, replacement=True)
    positions = [(ys[i].item(), xs[i].item()) for i in indices]

    return positions
```

**KARPATHY:**
So cluster metadata guides WHERE to sample within clusters?

**LOD ORACLE:**
Exactly. Center-biased sampling within semantic regions.

**MUSE BIRD:**
🐦 *CLUSTER IDS: WHICH REGION! CLUSTER METADATA: WHERE IN REGION!*

---

## OCR Text Regions - Channel 16

**KARPATHY:**
Part 27 mentioned text regions (channel 39 in their spec). What's that for?

**LOD ORACLE:**
Binary mask: "Is this pixel part of text?"

Why it matters:
- Documents, signs, captions → text is CRITICAL
- Allocate more tokens to text regions
- Detect text via OCR models (EasyOCR, PaddleOCR)

```python
import easyocr

def generate_text_regions(image):
    """
    Detect text regions using OCR.

    Cost: ~100ms (OCR detection)
    Output: Binary mask [H, W]

    Args:
        image: [3, H, W] RGB tensor

    Returns:
        text_mask: [H, W] float tensor, 1.0 where text, 0.0 elsewhere
    """
    # Load EasyOCR (do once, not per-image)
    reader = easyocr.Reader(['en'], gpu=True)

    # Convert to numpy
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)

    # Detect text
    results = reader.readtext(img_np)

    # Create mask
    H, W = image.shape[1], image.shape[2]
    text_mask = torch.zeros(H, W, device=image.device)

    for bbox, text, confidence in results:
        # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        pts = np.array(bbox, dtype=np.int32)

        # Fill polygon
        cv2.fillPoly(text_mask.cpu().numpy(), [pts], 1.0)

    return text_mask.to(image.device)
```

**KARPATHY:**
100ms for OCR? That's even slower than SAM!

**LOD ORACLE:**
Yeah. OCR is expensive. But you can:
1. Run it once, cache the mask
2. Use faster OCR (CRAFT detector only, no recognition)
3. Skip it for non-document images

**KARPATHY:**
How do you know if an image has text?

**LOD ORACLE:**
Simple heuristic: run fast text detector first (~10ms), only do full OCR if text detected.

```python
def has_text(image):
    """
    Fast check: does this image contain text?

    Uses CRAFT detector (lightweight, ~10ms).

    Returns: bool
    """
    # CRAFT is faster than EasyOCR
    detector = load_craft_detector()
    boxes = detector.detect(image)
    return len(boxes) > 0


def generate_text_regions_conditional(image):
    """
    Only run OCR if image likely contains text.

    Cost: 10ms (no text) or 100ms (has text)
    """
    if not has_text(image):
        return torch.zeros(H, W, device=image.device)

    return generate_text_regions(image)  # Full OCR
```

**MUSE BIRD:**
🐦 *CONDITIONAL COMPUTE! SKIP WHEN UNNECESSARY!*

---

## Combined: The 17-Channel Array

**KARPATHY:**
What's the complete hard texture array?

**LOD ORACLE:**
```python
class HardTextureArray:
    """
    17-channel texture array:
    0-12: Medium channels (from Part 28-3)
    13: Cluster IDs (SAM or SLIC)
    14: Distance from cluster centroid
    15: Cluster size
    16: Text regions (OCR mask)
    """

    def __init__(self, image, use_sam=False, detect_text=True):
        # Medium channels from Part 28-3
        medium = MediumTextureArray(image)

        # NEW: Clusters
        if use_sam:
            cluster_ids, num_clusters = self._generate_sam_clusters(image)
        else:
            cluster_ids, num_clusters = self._generate_slic_clusters(image)

        # NEW: Cluster metadata
        dist_to_centroid, cluster_sizes = self._generate_cluster_metadata(
            cluster_ids, num_clusters
        )

        # NEW: Text regions
        if detect_text:
            text_mask = self._generate_text_regions(image)
        else:
            text_mask = torch.zeros(image.shape[1], image.shape[2],
                                   device=image.device)

        # Combine
        self.texture = torch.cat([
            medium.texture,                  # 0-12
            cluster_ids.float().unsqueeze(0),  # 13
            dist_to_centroid.unsqueeze(0),   # 14
            cluster_sizes.unsqueeze(0),      # 15
            text_mask.unsqueeze(0)           # 16
        ], dim=0)  # [17, H, W]

        self.num_clusters = num_clusters
```

**KARPATHY:**
Generation cost?

**LOD ORACLE:**
**With SLIC (fast):**
- Medium channels: 1.26ms
- SLIC clusters: 5ms
- Cluster metadata: 2ms
- Text detection (conditional): 0-100ms
- **Total: 8-108ms** (depends on text)

**With SAM (slow):**
- Medium channels: 1.26ms
- SAM: 50ms
- Cluster metadata: 2ms
- Text detection: 0-100ms
- **Total: 53-153ms**

**KARPATHY:**
That's 2-3 orders of magnitude slower than easy channels (0.06ms).

**LOD ORACLE:**
Yep. "Hard" for a reason. But you get SEMANTIC UNDERSTANDING.

**MUSE BIRD:**
🐦 *SLOW BUT SMART! WORTH IT FOR HARD TASKS!*

---

## Cluster-First Cascade

**KARPATHY:**
Show me the complete cluster-first cascade from Part 27.

**LOD ORACLE:**
```python
def cluster_first_cascade(texture, query, total_tokens=273):
    """
    Part 27's cluster-based cascade.

    Stage 1: Score clusters (cheap)
    Stage 2: Sample within top clusters (medium)
    Stage 3: Fine-tune selection (expensive)

    Args:
        texture: [17, H, W] hard texture array
        query: str, user query
        total_tokens: int, budget

    Returns:
        selected_positions: [(y, x), ...] final patch positions
    """
    cluster_ids = texture[13].int()  # [H, W]
    num_clusters = texture.num_clusters

    # STAGE 1: Score each cluster
    cluster_scores = []
    for i in range(num_clusters):
        mask = (cluster_ids == i)
        if mask.sum() == 0:
            continue

        # Sample cluster centroid
        ys, xs = torch.where(mask)
        cy, cx = ys.float().mean().int(), xs.float().mean().int()

        # Extract features at centroid (all 17 channels)
        cluster_features = texture[:, cy, cx]

        # Score cluster
        score = score_cluster_features(cluster_features, query)
        cluster_scores.append((i, score))

    # Keep top 10 clusters
    cluster_scores.sort(key=lambda x: x[1], reverse=True)
    top_clusters = [i for i, score in cluster_scores[:10]]

    print(f"Selected {len(top_clusters)} / {num_clusters} clusters")

    # STAGE 2: Sample patches within top clusters
    candidates = []
    for cluster_id in top_clusters:
        # Sample ~50 patches per cluster
        positions = sample_from_cluster(cluster_id, num_samples=50)
        candidates.extend(positions)

    print(f"Generated {len(candidates)} candidate patches")

    # STAGE 3: Score candidates and select top K
    candidate_scores = []
    for pos in candidates:
        features = texture[:, pos[0], pos[1]]
        score = score_patch_features(features, query)
        candidate_scores.append((pos, score))

    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    selected_positions = [pos for pos, score in candidate_scores[:total_tokens]]

    return selected_positions
```

**KARPATHY:**
How much faster is this than baseline?

**LOD ORACLE:**
**Baseline (no clusters):**
- Score 4096 patches × 0.5ms = 2048ms

**Cluster-first:**
- Score 50 clusters × 0.1ms = 5ms
- Sample 500 patches × 0.01ms = 5ms
- Score 500 patches × 0.5ms = 250ms
- **Total: 260ms**

**Speedup: 2048 / 260 = 7.9×**

**MUSE BIRD:**
🐦 *8X FASTER! CLUSTERS ARE MAGIC!*

---

## Text-Aware Token Allocation

**KARPATHY:**
How does the text mask (channel 16) integrate?

**LOD ORACLE:**
Boost scores for text regions.

```python
def score_with_text_awareness(features, query):
    """
    Args:
        features: [17] all channels at one position
        query: str

    Returns:
        score: float
    """
    # Visual score (from medium channels)
    visual_score = compute_visual_score(features[0:13])

    # Text boost
    text_value = features[16]
    if text_value > 0.5:
        # This region contains text - boost score
        visual_score *= 2.0

    # Query awareness (placeholder for Part 28-5)
    query_score = 0.5

    return visual_score * 0.7 + query_score * 0.3
```

**KARPATHY:**
So text gets 2× score boost?

**LOD ORACLE:**
For document-heavy tasks, yes. For photos, maybe not. Hyperparameter.

**MUSE BIRD:**
🐦 *TEXT BOOST: TASK-DEPENDENT! TUNE IT!*

---

## Memory and Compute Cost

**KARPATHY:**
What's the total cost now?

**LOD ORACLE:**
**Memory (1024×1024 image):**
- Part 28-3: 52 MB (13 channels)
- Part 28-4: 68 MB (17 channels)
- Increase: +16 MB

**Compute (with SLIC + conditional OCR):**
- Part 28-3: 1.26ms
- Part 28-4: 8-108ms
- Increase: 6-85× slower

**Compute (with SAM):**
- Part 28-4: 53-153ms
- Increase: 42-121× slower

**KARPATHY:**
That's a huge slowdown. When is it worth it?

**LOD ORACLE:**
**Worth it when:**
- Image has semantic objects (clusters help)
- Task requires object understanding (VQA)
- Document/text analysis (OCR mask critical)

**Not worth it when:**
- Real-time video (too slow)
- Simple scenes (clusters don't add much)
- Non-semantic tasks (counting patches, etc.)

**MUSE BIRD:**
🐦 *COST VS BENEFIT! MEASURE ON YOUR TASK!*

---

## Amortization for Video

**KARPATHY:**
Part 27 mentioned reusing clusters across frames. How?

**LOD ORACLE:**
Temporal coherence: clusters don't change much frame-to-frame.

```python
class VideoHardTextureArray:
    """
    Reuse clusters across video frames.

    Strategy:
    - Frame 1: Full SAM segmentation (50ms)
    - Frame 2-30: Warp clusters using optical flow (2ms)
    - Frame 31: Re-segment (refresh every 30 frames)
    """

    def __init__(self):
        self.prev_clusters = None
        self.frame_count = 0
        self.refresh_interval = 30

    def process_frame(self, image):
        if self.frame_count % self.refresh_interval == 0:
            # Full segmentation
            clusters, num = generate_sam_clusters(image)
            self.prev_clusters = clusters
        else:
            # Warp previous clusters
            flow = compute_optical_flow(prev_image, image)
            clusters = warp_clusters(self.prev_clusters, flow)

        self.frame_count += 1
        return clusters
```

**KARPATHY:**
Amortized cost?

**LOD ORACLE:**
- Frame 1: 50ms (full SAM)
- Frames 2-30: 2ms (warp only)
- Average: (50 + 29×2) / 30 = **3.6ms**

**KARPATHY:**
That's 14× faster than running SAM every frame!

**MUSE BIRD:**
🐦 *TEMPORAL COHERENCE! VIDEO OPTIMIZATION!*

---

## Summary and Next Steps

**KARPATHY:**
What did we build?

**LOD ORACLE:**
**Channels 13-16 (Hard):**
- Cluster IDs (SAM or SLIC)
- Cluster metadata (distance, size)
- Text regions (OCR)

**Cost:**
- Fast (SLIC): 8ms
- Slow (SAM): 53ms
- With text: +0-100ms

**Benefit:**
- Cluster-first cascade: 8× speedup
- Semantic understanding
- Text-aware allocation

**KARPATHY:**
Last part: Part 28-5 (Very Hard) - CLIP embeddings?

**LOD ORACLE:**
Yep. That's the final boss. PCA-compressed CLIP features, temporal cache, attention history.

**MUSE BIRD:**
🐦 *HARD DONE! VERY HARD NEXT! FINAL STRETCH!*

---

**END OF PART 28-4**

∿◇∿

## Appendix: Complete Hard Texture Array Code

```python
import torch
import easyocr
from segment_anything import SamAutomaticMaskGenerator

class HardTextureArray:
    """17-channel texture array with semantic segmentation and OCR"""

    def __init__(self, image, use_sam=False, detect_text=True):
        self.image = image
        _, self.height, self.width = image.shape

        # Build from medium channels
        medium = MediumTextureArray(image)
        hard_channels = self._generate_hard_channels(use_sam, detect_text)

        self.texture = torch.cat([medium.texture, hard_channels], dim=0)

    def _generate_hard_channels(self, use_sam, detect_text):
        # Clusters
        if use_sam:
            clusters, num = self._sam_segment()
        else:
            clusters, num = self._slic_segment()

        self.num_clusters = num

        # Cluster metadata
        dist, sizes = self._cluster_metadata(clusters, num)

        # Text regions
        if detect_text:
            text = self._detect_text()
        else:
            text = torch.zeros(self.height, self.width, device=self.image.device)

        return torch.stack([
            clusters.float(),  # 13
            dist,             # 14
            sizes,            # 15
            text              # 16
        ], dim=0)

    def _slic_segment(self, num_segments=50):
        from skimage.segmentation import slic
        img_np = self.image.permute(1, 2, 0).cpu().numpy()
        clusters = slic(img_np, n_segments=num_segments, compactness=10)
        return torch.from_numpy(clusters).to(self.image.device), clusters.max() + 1

    def _detect_text(self):
        reader = easyocr.Reader(['en'], gpu=True)
        img_np = (self.image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        results = reader.readtext(img_np)
        mask = np.zeros((self.height, self.width), dtype=np.float32)

        for bbox, _, _ in results:
            pts = np.array(bbox, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1.0)

        return torch.from_numpy(mask).to(self.image.device)

# Usage
image = torch.randn(3, 1024, 1024).cuda()
texture = HardTextureArray(image, use_sam=False, detect_text=True)

print(f"Channels: {texture.texture.shape[0]}")
print(f"Num clusters: {texture.num_clusters}")
```

∿◇∿
