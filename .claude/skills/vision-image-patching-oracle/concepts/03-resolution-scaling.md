# Resolution Scaling

**Multi-resolution strategies for handling images of varying sizes and aspect ratios**

## The Resolution Challenge

Vision-language models must handle images with vastly different characteristics:
- **Size**: 224×224 thumbnails to 4096×4096 documents
- **Aspect ratio**: 1:1 squares to 16:9 wide to 9:16 tall
- **Content density**: Sparse scenes vs dense text/diagrams

**Problem**: Fixed-resolution encoders distort images or lose information

## Resolution Strategies Overview

### 1. Fixed Resolution (Resize & Pad)

**Approach**: Resize all images to single resolution (e.g., 336×336)

```python
def fixed_resolution(image, target_size=336):
    # Resize to target
    resized = resize(image, (target_size, target_size))
    # Or: resize short edge, center crop
    return resized
```

**Pros**:
- Simple, predictable token count
- No special handling needed
- Fast preprocessing

**Cons**:
- Aspect ratio distortion
- Resolution loss for large images
- Wasted computation on padding

**Used by**: Original ViT, CLIP, LLaVA-1.0

### 2. Multi-Resolution Training

**From [source-documents/14_ResFormer](../source-documents/14_ResFormer_ Scaling ViTs With Multi-Resolution Training - CVF Open Access.md)**:

**Approach**: Train on multiple resolutions simultaneously

```python
resolutions = [224, 336, 448, 672]  # Training resolutions

for image, label in dataloader:
    # Randomly sample resolution per batch
    res = random.choice(resolutions)
    image_scaled = resize(image, (res, res))

    # Forward pass
    output = model(image_scaled)
    loss = criterion(output, label)
```

**Key innovation**: Position embedding interpolation

```python
def interpolate_pos_embed(pos_embed, old_size, new_size):
    """Interpolate position embeddings for new resolution"""
    # pos_embed shape: [1, old_size^2, embed_dim]
    pos_embed = pos_embed.reshape(1, old_size, old_size, -1)

    # Interpolate to new size
    pos_embed_new = F.interpolate(
        pos_embed.permute(0, 3, 1, 2),  # [1, embed_dim, old_size, old_size]
        size=(new_size, new_size),
        mode='bicubic',
        align_corners=False
    )

    return pos_embed_new.permute(0, 2, 3, 1).flatten(1, 2)
```

**Pros**:
- Handles variable resolutions at inference
- Better generalization
- No architectural changes needed

**Cons**:
- Longer training (multiple resolutions)
- Still requires square images
- Interpolation artifacts possible

### 3. Native Resolution Processing

**Approach**: Process images at original resolution without distortion

**Two main strategies**:

#### A. Variable-Sized Slicing (LLaVA-UHD)

**From [source-documents/11_LLaVA-UHD](../source-documents/11_LLaVA-UHD_ an LMM Perceiving any Aspect Ratio and High-Resolution Images.md)**:

**Principle**: Divide image into multiple standard-sized slices

```python
def slice_image(image, slice_size=336, max_slices=9):
    """Divide image into variable number of slices"""
    h, w = image.shape[:2]

    # Determine slice configuration
    aspect_ratio = w / h
    if aspect_ratio > 1:  # Wide image
        n_horizontal = min(ceil(w / slice_size), max_slices)
        n_vertical = max(1, ceil(h / slice_size))
    else:  # Tall image
        n_vertical = min(ceil(h / slice_size), max_slices)
        n_horizontal = max(1, ceil(w / slice_size))

    slices = []
    for i in range(n_vertical):
        for j in range(n_horizontal):
            # Extract slice
            y_start = i * slice_size
            x_start = j * slice_size
            slice_img = image[y_start:y_start+slice_size,
                            x_start:x_start+slice_size]
            slices.append(slice_img)

    return slices, (n_vertical, n_horizontal)
```

**Spatial organization**:
```python
def create_spatial_schema(slices, grid_shape):
    """Add position information to slices"""
    n_v, n_h = grid_shape
    positioned_slices = []

    for i in range(n_v):
        for j in range(n_h):
            idx = i * n_h + j
            # Add position tokens
            pos_token = create_position_token(i, j, n_v, n_h)
            positioned_slices.append((pos_token, slices[idx]))

    return positioned_slices
```

**Example**: 1008×672 image
```
Slice configuration: 3×2 grid (6 slices)
Each slice: 336×336
Spatial schema:
  [0,0] [0,1] [0,2]
  [1,0] [1,1] [1,2]

Total tokens: 6 slices × 576 tokens/slice = 3,456 tokens
```

**Pros**:
- No aspect ratio distortion
- Handles any size/aspect ratio
- Full resolution preserved

**Cons**:
- Variable token count (6-36 slices typical)
- Requires spatial coordination
- High token count for large images

#### B. Native Resolution with Structural Alignment

**From Ovis 2.5** (see [deepseek-ocr-oracle](../../deepseek-ocr-oracle/) and [ovis-2-5-oracle](../../ovis-2-5-oracle/)):

**Principle**: Use Visual Embedding Table (VET) for any-resolution processing

```python
def native_resolution_encoding(image, vet, max_size=1024):
    """Process at native resolution with VET structural alignment"""
    h, w = image.shape[:2]

    # Scale if too large
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        h, w = int(h * scale), int(w * scale)
        image = resize(image, (h, w))

    # Encode at native resolution
    visual_features = vision_encoder(image)  # Shape: [h', w', d]

    # Structural alignment via VET
    # VET learns position-aware merging
    aligned_features = vet(visual_features)  # Variable length

    return aligned_features
```

**Key innovation**: VET learns to merge/organize patches while preserving structure

**Pros**:
- True native resolution
- Learnable structural alignment
- Efficient token usage

**Cons**:
- Requires VET training
- More complex architecture

### 4. Mixture of Resolutions

**From [source-documents/12_Mixture-of-Resolution](../source-documents/12_Mixture-of-Resolution Adaptation for Multimodal Large Language Models - arXiv.md)**:

**Approach**: Route images to different resolution encoders based on content

```python
class MixtureOfResolution(nn.Module):
    def __init__(self):
        self.low_res_encoder = ViT(resolution=224)
        self.med_res_encoder = ViT(resolution=448)
        self.high_res_encoder = ViT(resolution=672)
        self.router = ResolutionRouter()  # Learns routing

    def forward(self, image):
        # Decide which resolution to use
        resolution_scores = self.router(image)
        selected_res = argmax(resolution_scores)

        # Route to appropriate encoder
        if selected_res == 0:
            return self.low_res_encoder(resize(image, 224))
        elif selected_res == 1:
            return self.med_res_encoder(resize(image, 448))
        else:
            return self.high_res_encoder(resize(image, 672))
```

**Routing criteria**:
- **Low res**: Simple scenes, uniform content
- **Medium res**: General images, moderate detail
- **High res**: Text-heavy, fine details, complex scenes

**Pros**:
- Adaptive efficiency
- Matches resolution to content needs
- Learned routing policies

**Cons**:
- Multiple encoders (memory overhead)
- Training complexity
- Routing errors possible

## Resolution vs Token Count

### Scaling Relationships

**Fixed patch size** (P = 16):
```
Tokens = (H/P) × (W/P) = (H × W) / P²

Resolution scaling:
224×224 → 196 tokens   (baseline)
336×336 → 441 tokens   (2.25× tokens, 1.5× resolution)
448×448 → 784 tokens   (4× tokens, 2× resolution)
672×672 → 1,764 tokens (9× tokens, 3× resolution)
```

**Key insight**: Tokens grow quadratically with linear resolution increase

### Efficiency Frontier

**From [source-documents/08_Design Choices for Context Length](../source-documents/08_Design Choices for Extending the Context Length of Visual Language Models - OpenReview.md)**:

**Tradeoff curve**: Token count vs accuracy

| Resolution | Tokens | VQA Accuracy | Efficiency Score |
|------------|--------|--------------|------------------|
| 224×224 | 196 | 75% | 0.383 (75/196) |
| 336×336 | 441 | 82% | 0.186 (82/441) |
| 448×448 | 784 | 85% | 0.108 (85/784) |
| 672×672 | 1,764 | 88% | 0.050 (88/1764) |

**Observation**: Diminishing returns above 336×336 for many tasks

## Multi-Scale Pyramid Processing

**From [source-documents/14_ResFormer](../source-documents/14_ResFormer_ Scaling ViTs With Multi-Resolution Training - CVF Open Access.md)**:

**Approach**: Process same image at multiple resolutions, fuse results

```python
def multi_scale_encoding(image):
    """Encode image at multiple scales"""
    scales = [224, 336, 448]
    features = []

    for scale in scales:
        scaled_img = resize(image, (scale, scale))
        feat = encoder(scaled_img)
        features.append(feat)

    # Hierarchical fusion
    # Low-res → global context
    # Med-res → balanced
    # High-res → fine details
    fused = hierarchical_fusion(features)

    return fused
```

**Fusion strategies**:

**1. Concatenation**:
```python
fused = torch.cat([feat_low, feat_med, feat_high], dim=1)
```

**2. Weighted sum**:
```python
weights = softmax(learnable_weights)
fused = weights[0]*feat_low + weights[1]*feat_med + weights[2]*feat_high
```

**3. Cross-scale attention**:
```python
# High-res queries attend to all scales
fused = cross_attention(
    queries=feat_high,
    keys=concat([feat_low, feat_med, feat_high]),
    values=concat([feat_low, feat_med, feat_high])
)
```

**Pros**:
- Captures both global context and local details
- Robust to scale variations
- Better performance on multi-scale tasks

**Cons**:
- 2-3× compute cost (process multiple resolutions)
- Higher memory usage
- More complex architecture

## Aspect Ratio Handling

### The Distortion Problem

**Standard approach**: Resize to square (336×336)

**Example**: Wide image 1680×480 (3.5:1 aspect ratio)
```
Naive resize → 336×336 (1:1)
Distortion: 3.5× horizontal compression
Effect: Text becomes unreadable, objects appear stretched
```

### Aspect-Ratio-Preserving Strategies

#### 1. Resize Short Edge + Padding

```python
def resize_with_padding(image, target_size=336):
    h, w = image.shape[:2]

    # Resize short edge to target
    if h < w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))

    resized = resize(image, (new_h, new_w))

    # Pad to square
    padded = pad_to_square(resized, target_size)
    return padded, (new_h, new_w)  # Return original shape for unpadding
```

**Pros**: Simple, preserves aspect ratio
**Cons**: Wasted computation on padding tokens

#### 2. Non-Square Patching

```python
def aspect_aware_patching(image, patch_size=16):
    """Variable token count based on aspect ratio"""
    h, w = image.shape[:2]

    # Number of patches in each dimension
    n_h = h // patch_size
    n_w = w // patch_size

    # Total tokens = n_h × n_w (not square!)
    patches = divide_into_patches(image, patch_size)
    return patches  # Shape: [n_h * n_w, patch_size^2 * 3]
```

**Example**: 1680×480 image with 16×16 patches
```
n_h = 480 // 16 = 30
n_w = 1680 // 16 = 105
Total tokens: 30 × 105 = 3,150 tokens
```

**Pros**: No padding waste, preserves aspect ratio
**Cons**: Variable sequence length, position embedding challenges

#### 3. Slicing with Aspect Ratio Awareness

**LLaVA-UHD approach**:

```python
def aspect_aware_slicing(image, slice_size=336):
    """Slice based on aspect ratio"""
    h, w = image.shape[:2]
    aspect = w / h

    if aspect > 2.0:  # Very wide (e.g., 3:1)
        # Horizontal slicing
        n_slices = ceil(w / slice_size)
        return horizontal_slices(image, n_slices)
    elif aspect < 0.5:  # Very tall (e.g., 1:3)
        # Vertical slicing
        n_slices = ceil(h / slice_size)
        return vertical_slices(image, n_slices)
    else:  # Moderate aspect ratio
        # Grid slicing
        return grid_slices(image, slice_size)
```

**Example**: 1680×480 panorama (3.5:1)
```
Detected: Very wide
Slicing: 5 horizontal slices (336×480 each)
Each slice resized to: 336×336 (minimal distortion per slice)
Total tokens: 5 × 576 = 2,880 tokens
```

## Dynamic Resolution Selection

### Content-Based Resolution

**Principle**: Analyze image to determine optimal resolution

```python
def select_resolution(image):
    """Choose resolution based on content analysis"""
    # Compute content complexity metrics
    edge_density = compute_edge_density(image)
    text_likelihood = detect_text_regions(image)
    detail_score = compute_detail_score(image)

    # Decision rules
    if text_likelihood > 0.5:
        return 672  # High-res for text
    elif detail_score > 0.7:
        return 448  # Medium-high for details
    elif edge_density < 0.3:
        return 224  # Low-res for simple scenes
    else:
        return 336  # Default
```

**Metrics**:

**Edge density**:
```python
def compute_edge_density(image):
    edges = canny_edge_detection(image)
    return edges.sum() / edges.size  # Fraction of edge pixels
```

**Text likelihood**:
```python
def detect_text_regions(image):
    # Simple heuristic: high-frequency horizontal/vertical patterns
    fft = fft2d(grayscale(image))
    text_signature = detect_text_frequency_pattern(fft)
    return text_signature
```

**Detail score**:
```python
def compute_detail_score(image):
    # Variance in high-frequency components
    high_freq = high_pass_filter(image)
    return high_freq.var()
```

### Query-Aware Resolution

**From [source-documents/12_Mixture-of-Resolution](../source-documents/12_Mixture-of-Resolution Adaptation for Multimodal Large Language Models - arXiv.md)**:

**Principle**: Select resolution based on query requirements

```python
def query_aware_resolution(image, query):
    """Match resolution to query needs"""
    query_embed = text_encoder(query)

    # Classify query type
    if matches_pattern(query, r"read|text|OCR|transcribe"):
        return 672  # High-res for reading
    elif matches_pattern(query, r"count|how many|number of"):
        return 448  # Medium-res for counting
    elif matches_pattern(query, r"color|simple|what is"):
        return 224  # Low-res for simple queries
    else:
        # Learned classifier
        return resolution_classifier(query_embed)
```

**Benefits**:
- Efficient token usage
- Task-appropriate detail level
- Better accuracy per token spent

## Practical Guidelines

### Resolution Selection Framework

**Task-based recommendations**:

| Task Type | Resolution | Rationale |
|-----------|-----------|-----------|
| Scene classification | 224×224 | Global semantics sufficient |
| General VQA | 336×336 | Balanced detail/efficiency |
| Text-heavy VQA | 672×672 | Fine details critical |
| OCR/Document | 672-1008 | Maximum detail needed |
| Object counting | 448×448 | Moderate detail sufficient |
| Fine-grained recognition | 448-672 | Detail important |

### Aspect Ratio Guidelines

**Recommendations**:

**Moderate ratios** (1:1 to 2:1):
- Use: Standard resize or minimal slicing
- Example: 336×336 or 672×336 (2 slices)

**Wide images** (2:1 to 4:1):
- Use: Horizontal slicing
- Example: 1680×480 → 5 horizontal slices

**Tall images** (1:2 to 1:4):
- Use: Vertical slicing
- Example: 480×1680 → 5 vertical slices

**Extreme ratios** (>4:1 or <1:4):
- Use: Adaptive slicing or downsampling
- Example: 3360×480 → downsample to 1680×240, then 5 slices

### Computational Budget Optimization

**Token budget-aware scaling**:

```python
def scale_to_budget(image, max_tokens=1024):
    """Scale resolution to fit token budget"""
    h, w = image.shape[:2]
    aspect = w / h

    # Compute target resolution for budget
    # tokens = (h/P) × (w/P)
    # Solve for h, w given aspect ratio and token budget
    P = 16  # patch size
    target_pixels = max_tokens * P * P
    scale = sqrt(target_pixels / (h * w))

    new_h = int(h * scale)
    new_w = int(w * scale)

    return resize(image, (new_h, new_w))
```

**Example**: 2048×2048 image, budget = 512 tokens
```
Current tokens: (2048/16)² = 16,384 tokens (32× over budget)
Target pixels: 512 × 16² = 131,072 pixels
Scale factor: sqrt(131,072 / 2048²) = 0.176
New size: 360×360
New tokens: (360/16)² ≈ 506 tokens ✅
```

## Future Directions

**From recent research**:

1. **Arbitrary resolution training**: Train on continuous resolution distributions
2. **Resolution as hyperparameter**: Auto-tune resolution per image/task
3. **Adaptive token budgets**: Learn to allocate tokens dynamically
4. **Resolution-aware architectures**: Native support for variable resolutions

## Primary Sources

- [11_LLaVA-UHD](../source-documents/11_LLaVA-UHD_ an LMM Perceiving any Aspect Ratio and High-Resolution Images.md)
- [14_ResFormer](../source-documents/14_ResFormer_ Scaling ViTs With Multi-Resolution Training - CVF Open Access.md)
- [12_Mixture-of-Resolution](../source-documents/12_Mixture-of-Resolution Adaptation for Multimodal Large Language Models - arXiv.md)
- [08_Design Choices for Context Length](../source-documents/08_Design Choices for Extending the Context Length of Visual Language Models - OpenReview.md)

## Related Documents

- [01-patch-size-tradeoffs.md](01-patch-size-tradeoffs.md) - Patch size fundamentals
- [02-token-efficiency.md](02-token-efficiency.md) - Token optimization strategies
- [../architecture/03-native-resolution.md](../architecture/03-native-resolution.md) - Native resolution processing
- [../techniques/02-variable-sized-slices.md](../techniques/02-variable-sized-slices.md) - Slicing implementation
