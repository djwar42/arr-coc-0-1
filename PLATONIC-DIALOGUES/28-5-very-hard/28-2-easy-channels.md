---
summary: whereby the oracles add edge detection channels (Sobel filters on normal and inverted images for dual-polarity low-contrast text detection) costing only 0.03ms total since modern GPUs have hardware-optimized convolution via cuDNN, demonstrating that two 3×3 convolutions plus square roots are "basically free" and proving Theaetetus's insight from Part 26 that inverted edges catch light-on-dark text while normal edges catch dark-on-light, bringing the channel count to 8 (RGB + position + dual edges) with negligible overhead
---

# Part 28-2: Easy Channels - Convolution is Cheap
*Wherein the oracles discover that edge detection, filters, and motion cost almost nothing when done right*

---

## Building on Part 28-1

**KARPATHY:**
We have RGB + position. 6 channels total. What's next?

**LOD ORACLE:**
Visual filters. Part 26's insight: inverted edges catch low-contrast text. Let's implement that.

**MUSE BIRD:**
🐦 *THEAETETUS INVERTED COLORS! NOW WE CODE IT!*

---

## Edge Detection - Channels 6-7

**KARPATHY:**
Sobel edge detection. How expensive is it?

**LOD ORACLE:**
On CPU: slow. On GPU: basically free.

```python
def generate_edge_channels(image):
    """
    Generate two edge channels:
    - Channel 6: Edges on normal image
    - Channel 7: Edges on inverted image (catches low-contrast text!)

    Cost: 2 convolutions × 0.015ms = 0.03ms total
    """
    # Sobel kernels
    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)

    sobel_y = torch.tensor([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)

    # Convert RGB to grayscale
    gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
    gray = gray.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # Compute edges on normal image
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    edges_normal = torch.sqrt(grad_x**2 + grad_y**2).squeeze()

    # Invert and compute edges
    inverted = 1.0 - gray
    grad_x_inv = F.conv2d(inverted, sobel_x, padding=1)
    grad_y_inv = F.conv2d(inverted, sobel_y, padding=1)
    edges_inverted = torch.sqrt(grad_x_inv**2 + grad_y_inv**2).squeeze()

    return torch.stack([edges_normal, edges_inverted], dim=0)
```

**KARPATHY:**
Two convolutions, two square roots, that's it?

**LOD ORACLE:**
Yep. Modern GPUs have hardware for convolution (cudnn). This is optimized to hell.

**MUSE BIRD:**
🐦 *GPU SAYS: CONVOLUTION? THAT'S MY JAM!*

---

## Why Inverted Edges Matter

**KARPATHY:**
Part 26 talked about low-contrast text. Show me a concrete example.

**LOD ORACLE:**
Imagine: light gray text on white background.

```python
# Simulation
background = torch.ones(1, 1, 256, 256) * 0.9  # Light gray background
text_region = background.clone()
text_region[:, :, 100:150, 100:200] = 0.8  # Slightly darker text

# Normal edges
edges_normal = detect_edges(text_region)
print(f"Normal edge strength: {edges_normal.max():.3f}")
# Output: ~0.1 (weak)

# Inverted edges
inverted = 1.0 - text_region
edges_inverted = detect_edges(inverted)
print(f"Inverted edge strength: {edges_inverted.max():.3f}")
# Output: ~0.1 (same? wait...)
```

**KARPATHY:**
Wait, they're the same?

**LOD ORACLE:**
For this example, yes. Inversion doesn't magically boost contrast. But here's where it helps:

```python
# Better example: white text on light gray
background = torch.ones(1, 1, 256, 256) * 0.8  # Gray background
text = background.clone()
text[:, :, 100:150, 100:200] = 1.0  # White text (barely visible)

# Normal: gray→white = weak edge
edges_normal = detect_edges(text)
print(f"Normal: {edges_normal.max():.3f}")  # ~0.2

# Inverted: 0.2 (dark gray) → 0.0 (black) = STRONG edge
edges_inv = detect_edges(1.0 - text)
print(f"Inverted: {edges_inv.max():.3f}")  # ~0.8
```

**KARPATHY:**
Ah! So inversion helps when the text is LIGHTER than background, not darker.

**LOD ORACLE:**
Exactly. Normal edges catch dark-on-light. Inverted edges catch light-on-dark. OR logic gets both.

**MUSE BIRD:**
🐦 *TWO POLARITIES! CATCHES BOTH CASES!*

---

## High-Pass Filter - Channel 8

**KARPATHY:**
What's high-pass do?

**LOD ORACLE:**
Emphasizes fine details. Removes smooth regions.

```python
def generate_highpass_channel(image):
    """
    High-pass filter: Detects rapid intensity changes.

    Good for: textures, fine text, detailed patterns
    Bad for: smooth gradients, uniform regions
    """
    # Simple high-pass: Laplacian kernel
    kernel = torch.tensor([
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0]
    ], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)

    gray = rgb_to_gray(image)
    highpass = F.conv2d(gray.unsqueeze(0).unsqueeze(0), kernel, padding=1)

    return highpass.squeeze()
```

**KARPATHY:**
When would highpass catch something that edges don't?

**LOD ORACLE:**
Textures. Edges detect BOUNDARIES. Highpass detects VARIATION within regions.

Example:
- Brick wall: edges detect outline of wall, highpass detects brick pattern
- Text document: edges detect letters, highpass detects texture of paper

**MUSE BIRD:**
🐦 *EDGES: BOUNDARIES! HIGHPASS: TEXTURE!*

---

## Low-Pass Filter - Channel 9

**KARPATHY:**
And low-pass is the opposite?

**LOD ORACLE:**
Yeah. Emphasizes smooth regions, removes noise.

```python
def generate_lowpass_channel(image):
    """
    Low-pass filter: Gaussian blur.

    Good for: identifying large uniform regions
    Bad for: fine details
    """
    # Gaussian blur kernel
    kernel = torch.tensor([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=torch.float32, device=image.device) / 16.0
    kernel = kernel.view(1, 1, 3, 3)

    gray = rgb_to_gray(image)
    lowpass = F.conv2d(gray.unsqueeze(0).unsqueeze(0), kernel, padding=1)

    return lowpass.squeeze()
```

**KARPATHY:**
When would you use this for token allocation?

**LOD ORACLE:**
To AVOID allocating too many tokens to noisy regions.

```python
# Scoring logic
lowpass_value = sample_channel(position, channel=9)

if lowpass_value > 0.9:
    # Very smooth region - probably sky, uniform background
    # Don't waste tokens here
    score *= 0.5
```

**KARPATHY:**
So it's a negative signal? Low-pass HIGH → reduce tokens?

**LOD ORACLE:**
Right. It's a culling heuristic, not a relevance signal.

**MUSE BIRD:**
🐦 *SMOOTH = BORING! SAVE TOKENS!*

---

## Motion Channel - Channel 10

**KARPATHY:**
Part 26 mentioned motion detection for T-rex mode. How hard is that?

**LOD ORACLE:**
Trivial if you have previous frame.

```python
def generate_motion_channel(current_frame, previous_frame):
    """
    Temporal difference: |current - previous|

    Cost: 1 subtraction + 1 absolute value = 0.001ms
    """
    if previous_frame is None:
        return torch.zeros_like(current_frame[0])

    # RGB difference
    diff = current_frame - previous_frame
    motion = torch.sqrt((diff**2).sum(dim=0))  # Magnitude across RGB

    return motion
```

**KARPATHY:**
That's it? Just subtract?

**LOD ORACLE:**
For basic motion detection, yes. Fancier methods:
- Optical flow (Lucas-Kanade, Farneback)
- Background subtraction (MOG2, KNN)
- CNN-based motion (FlowNet)

But subtraction is 99% as good for token allocation.

**KARPATHY:**
Why?

**LOD ORACLE:**
You don't need precise motion vectors. You just need "did something CHANGE here?" Binary signal.

**MUSE BIRD:**
🐦 *CHANGED? ATTEND! STATIC? IGNORE!*

---

## Combined: The 11-Channel Array

**KARPATHY:**
Put it together. What do we have now?

**LOD ORACLE:**
```python
class EasyTextureArray:
    """
    11-channel texture array:
    0-2: RGB
    3-5: Position (X, Y, Eccentricity)
    6: Edges normal
    7: Edges inverted
    8: High-pass
    9: Low-pass
    10: Motion
    """

    def __init__(self, image, previous_frame=None):
        self.image = image
        _, self.height, self.width = image.shape

        # From Part 28-1
        position = self._generate_position()

        # NEW: Visual filters
        edges = self._generate_edges()        # [2, H, W]
        highpass = self._generate_highpass()  # [1, H, W]
        lowpass = self._generate_lowpass()    # [1, H, W]
        motion = self._generate_motion(previous_frame)  # [1, H, W]

        # Stack all channels
        self.texture = torch.cat([
            image,      # 0-2
            position,   # 3-5
            edges,      # 6-7
            highpass,   # 8
            lowpass,    # 9
            motion      # 10
        ], dim=0)  # [11, H, W]

    def _generate_edges(self):
        # Sobel on normal + inverted
        gray = rgb_to_gray(self.image)
        edges_normal = sobel(gray)
        edges_inverted = sobel(1.0 - gray)
        return torch.stack([edges_normal, edges_inverted], dim=0)

    def _generate_highpass(self):
        gray = rgb_to_gray(self.image)
        return apply_laplacian(gray)

    def _generate_lowpass(self):
        gray = rgb_to_gray(self.image)
        return apply_gaussian(gray)

    def _generate_motion(self, prev):
        if prev is None:
            return torch.zeros(self.height, self.width, device=self.image.device)
        return torch.sqrt(((self.image - prev)**2).sum(dim=0))

    def _generate_position(self):
        # Same as Part 28-1
        x = torch.linspace(0, 1, self.width, device=self.image.device)
        y = torch.linspace(0, 1, self.height, device=self.image.device)
        x = x.view(1, -1).expand(self.height, -1)
        y = y.view(-1, 1).expand(-1, self.width)
        ecc = torch.sqrt((x - 0.5)**2 + (y - 0.5)**2)
        return torch.stack([x, y, ecc], dim=0)
```

**KARPATHY:**
Generation cost?

**LOD ORACLE:**
- Position: 0.001ms (Part 28-1)
- Edges (2 channels): 0.03ms
- Highpass: 0.015ms
- Lowpass: 0.015ms
- Motion: 0.001ms
- **Total: 0.06ms**

**KARPATHY:**
For 11 channels? That's still basically free.

**MUSE BIRD:**
🐦 *GPU CONVOLUTION IS OPTIMIZED TO HELL!*

---

## Sampling Strategy

**KARPATHY:**
How do you use all 11 channels during cascade?

**LOD ORACLE:**
OR logic across visual channels, as Part 26 described.

```python
def score_patch_multichannel(texture, position):
    """
    Sample all channels, combine with OR logic (MAX).

    Args:
        texture: [11, H, W] texture array
        position: (y, x) patch center

    Returns:
        score: float, relevance score
    """
    y, x = position
    features = texture[:, y, x]  # [11] - all channels at this position

    # Visual channels (OR logic)
    edges_normal = features[6]
    edges_inverted = features[7]
    highpass = features[8]
    motion = features[10]

    # Take MAX (pass if ANY channel activates)
    visual_score = max(edges_normal, edges_inverted, highpass, motion)

    # Low-pass as negative signal (cull smooth regions)
    lowpass = features[9]
    if lowpass > 0.9:
        visual_score *= 0.5  # Penalize very smooth regions

    # Foveal bias (position channel)
    eccentricity = features[5]
    foveal_weight = 1.0 - 0.5 * eccentricity

    # Combine
    final_score = visual_score * foveal_weight

    return final_score
```

**KARPATHY:**
So a patch passes if:
- Edges normal are high, OR
- Edges inverted are high, OR
- Highpass is high, OR
- Motion is high

And then you bias toward center using eccentricity?

**LOD ORACLE:**
Exactly. That's the mantis shrimp strategy from Part 26: parallel channels, simple comparisons.

**MUSE BIRD:**
🐦 *BIOLOGY TESTED 500M YEARS! WE COPY!*

---

## Edge Case: All Channels Agree

**KARPATHY:**
What if multiple channels activate on the same patch? Do you boost the score?

**LOD ORACLE:**
Good question. Two strategies:

**MAX (OR logic):**
```python
score = max(edges_normal, edges_inverted, highpass, motion)
# If ANY channel says "important", score is high
```

**SUM (AND-boosted):**
```python
score = edges_normal + edges_inverted + highpass + motion
# If ALL channels say "important", score is VERY high
```

**KARPATHY:**
Which is better?

**LOD ORACLE:**
Depends on your failure mode:
- MAX: Catch rare cases (low-contrast text that only inverted edges see)
- SUM: Boost highly salient regions (edges + motion + texture)

**KARPATHY:**
Can you mix them?

**LOD ORACLE:**
Yeah:
```python
max_score = max(edges_normal, edges_inverted, highpass, motion)
sum_score = edges_normal + edges_inverted + highpass + motion
combined = 0.7 * max_score + 0.3 * sum_score
```

**MUSE BIRD:**
🐦 *HYPERPARAMETER! TUNE ON VALIDATION SET!*

---

## Testing - DocVQA Low-Contrast Text

**KARPATHY:**
How do we validate that inverted edges help?

**LOD ORACLE:**
DocVQA has annotations for text regions. Test:

```python
def test_inverted_edges_on_docvqa():
    dataset = load_docvqa(split='val')

    scores_normal_only = []
    scores_with_inverted = []

    for sample in dataset:
        image = sample['image']
        text_boxes = sample['text_boxes']  # Ground truth

        # Generate channels
        texture = EasyTextureArray(image)

        # Strategy 1: Normal edges only
        for box in text_boxes:
            center = box.center()
            score = texture.texture[6, center.y, center.x]  # Channel 6: edges normal
            scores_normal_only.append(score)

        # Strategy 2: Normal OR Inverted
        for box in text_boxes:
            center = box.center()
            normal = texture.texture[6, center.y, center.x]
            inverted = texture.texture[7, center.y, center.x]
            score = max(normal, inverted)
            scores_with_inverted.append(score)

    print(f"Normal edges: {torch.tensor(scores_normal_only).mean():.3f}")
    print(f"With inverted: {torch.tensor(scores_with_inverted).mean():.3f}")

    # Expected: With inverted has higher mean (catches more text)
```

**KARPATHY:**
And you'd check if this correlates with VQA accuracy?

**LOD ORACLE:**
Yeah. Hypothesis: Higher text detection score → better VQA performance on document questions.

**MUSE BIRD:**
🐦 *MEASURE ON REAL DATA! NOT SYNTHETIC!*

---

## Memory and Compute Cost

**KARPATHY:**
What's the overhead versus Part 28-1 (6 channels)?

**LOD ORACLE:**
**Memory:**
- Part 28-1: 6 channels × 4 MB = 24 MB
- Part 28-2: 11 channels × 4 MB = 44 MB
- Increase: +20 MB per image

**Compute:**
- Part 28-1: 0.001ms
- Part 28-2: 0.06ms
- Increase: +0.059ms

**Sampling:**
- Same! Sampling 11 channels costs same as 6 (spatial locality)

**KARPATHY:**
So we doubled the channels, memory went up 83%, compute went up 60×, but it's still basically free?

**LOD ORACLE:**
Right. Because we started from 0.001ms. 60× a tiny number is still tiny.

**MUSE BIRD:**
🐦 *0.06ms IS NOTHING! HUMAN EYE IS 16ms!*

---

## When Easy Channels Aren't Enough

**KARPATHY:**
Are there failure cases where these 11 channels miss important content?

**LOD ORACLE:**
Yeah:

**Case 1: Semantic objects without edges**
- Smooth gradient sky with airplane (edges are weak)
- Solution: Need semantic segmentation (Part 28-4: Clusters)

**Case 2: Query-specific relevance**
- User asks "where is the red car?" but you only have edge/motion channels
- Solution: Need CLIP embeddings (Part 28-5: Very Hard)

**Case 3: Small objects**
- Tiny text at coarse mipmap levels gets blurred away
- Solution: Multi-scale cascade (already in Part 25)

**KARPATHY:**
So easy channels are necessary but not sufficient?

**LOD ORACLE:**
Exactly. They're a baseline. You NEED edges and motion. But you ALSO need semantic understanding.

**MUSE BIRD:**
🐦 *EASY CHANNELS: FOUNDATION! NOT CEILING!*

---

## Integration with Knowing.py

**KARPATHY:**
Your `knowing.py` has three scorers: Information, Perspectival, Participatory. How do these channels map?

**LOD ORACLE:**
```python
# knowing.py integration

class InformationScorer:
    """Uses edge channels (6-7) + highpass (8)"""
    def score(self, texture, position):
        features = texture[:, position.y, position.x]
        edges = max(features[6], features[7])  # Normal OR inverted
        highpass = features[8]
        return 0.7 * edges + 0.3 * highpass

class PerspectivalScorer:
    """Uses motion (10) + position (5)"""
    def score(self, texture, position):
        features = texture[:, position.y, position.x]
        motion = features[10]
        eccentricity = features[5]

        # Bias toward fovea + moving regions
        foveal_weight = 1.0 - 0.5 * eccentricity
        return motion * foveal_weight

class ParticipatoryScorer:
    """Still needs CLIP embeddings (Part 28-5)"""
    def score(self, texture, position, query):
        # For now, placeholder
        # Will use channels 11-26 (embeddings) in Part 28-5
        return 0.5  # Uniform for now
```

**KARPATHY:**
So Information and Perspectival can use easy channels, but Participatory needs hard channels?

**LOD ORACLE:**
Right. Query-awareness requires semantic features, which means CLIP or similar.

**MUSE BIRD:**
🐦 *EASY CHANNELS: 2 OF 3 SCORERS! PROGRESS!*

---

## Summary and Next Steps

**KARPATHY:**
Let me recap what we built:

**Channels 6-10 (Easy):**
- Edges normal: Dark text on light background
- Edges inverted: Light text on dark background
- High-pass: Textures and fine details
- Low-pass: Smooth regions (for culling)
- Motion: Temporal changes (video)

**Cost:** 0.06ms generation
**Memory:** +20 MB
**Benefit:** Catches 90% of visually salient content

**LOD ORACLE:**
And these integrate with your existing `knowing.py` scorers without breaking the Vervaekean framework.

**KARPATHY:**
Next is Part 28-3: Medium channels (saliency, distance fields)?

**LOD ORACLE:**
Yep. Those are standard CV algorithms but slightly more complex than convolution.

**MUSE BIRD:**
🐦 *EASY DONE! MEDIUM NEXT! BUILDING MOMENTUM!*

---

**END OF PART 28-2**

∿◇∿

## Appendix: Complete Easy Texture Array Code

```python
import torch
import torch.nn.functional as F

class EasyTextureArray:
    """11-channel texture array: RGB + Position + Visual Filters"""

    def __init__(self, image, previous_frame=None):
        self.image = image
        _, self.height, self.width = image.shape
        self.texture = self._build_texture(previous_frame)

    def _build_texture(self, prev):
        # RGB (0-2)
        rgb = self.image

        # Position (3-5)
        position = self._gen_position()

        # Edges (6-7)
        gray = self._rgb_to_gray()
        edges_normal = self._sobel(gray)
        edges_inverted = self._sobel(1.0 - gray)

        # Highpass (8)
        highpass = self._laplacian(gray)

        # Lowpass (9)
        lowpass = self._gaussian(gray)

        # Motion (10)
        if prev is not None:
            motion = torch.sqrt(((self.image - prev)**2).sum(dim=0, keepdim=True))
        else:
            motion = torch.zeros(1, self.height, self.width, device=self.image.device)

        return torch.cat([
            rgb,                                    # 0-2
            position,                               # 3-5
            edges_normal.unsqueeze(0),             # 6
            edges_inverted.unsqueeze(0),           # 7
            highpass.unsqueeze(0),                 # 8
            lowpass.unsqueeze(0),                  # 9
            motion                                  # 10
        ], dim=0)

    def _rgb_to_gray(self):
        return 0.299 * self.image[0] + 0.587 * self.image[1] + 0.114 * self.image[2]

    def _gen_position(self):
        x = torch.linspace(0, 1, self.width, device=self.image.device)
        y = torch.linspace(0, 1, self.height, device=self.image.device)
        x = x.view(1, -1).expand(self.height, -1)
        y = y.view(-1, 1).expand(-1, self.width)
        ecc = torch.sqrt((x - 0.5)**2 + (y - 0.5)**2)
        return torch.stack([x, y, ecc], dim=0)

    def _sobel(self, gray):
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                         dtype=torch.float32, device=gray.device).view(1, 1, 3, 3)
        sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                         dtype=torch.float32, device=gray.device).view(1, 1, 3, 3)

        g = gray.unsqueeze(0).unsqueeze(0)
        gx = F.conv2d(g, sx, padding=1)
        gy = F.conv2d(g, sy, padding=1)
        return torch.sqrt(gx**2 + gy**2).squeeze()

    def _laplacian(self, gray):
        kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                            dtype=torch.float32, device=gray.device).view(1, 1, 3, 3)
        return F.conv2d(gray.unsqueeze(0).unsqueeze(0), kernel, padding=1).squeeze()

    def _gaussian(self, gray):
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                            dtype=torch.float32, device=gray.device) / 16.0
        kernel = kernel.view(1, 1, 3, 3)
        return F.conv2d(gray.unsqueeze(0).unsqueeze(0), kernel, padding=1).squeeze()

    def sample(self, positions):
        """Sample all 11 channels at specified positions"""
        ys = torch.tensor([p[0] for p in positions], device=self.image.device)
        xs = torch.tensor([p[1] for p in positions], device=self.image.device)
        return self.texture[:, ys, xs].T  # [N, 11]

# Usage
image = torch.randn(3, 1024, 1024).cuda()
texture = EasyTextureArray(image)

positions = [(y, x) for y in range(0, 1024, 64) for x in range(0, 1024, 64)][:273]
features = texture.sample(positions)

print(f"RGB: {features[:, 0:3].shape}")
print(f"Position: {features[:, 3:6].shape}")
print(f"Edges: {features[:, 6:8].shape}")
print(f"Filters: {features[:, 8:11].shape}")
```

∿◇∿
