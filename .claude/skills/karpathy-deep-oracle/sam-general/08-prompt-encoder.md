# SAM Prompt Encoder: Sparse and Dense Embeddings

## Overview

The **Prompt Encoder** is a critical component of SAM's architecture that transforms user inputs (prompts) into embedding vectors that can be combined with image features. It enables SAM's remarkable flexibility by supporting multiple prompt modalities through a unified encoding scheme.

**Key Characteristics:**
- Handles both **sparse prompts** (points, boxes, text) and **dense prompts** (masks)
- Lightweight design (~4M parameters combined with mask decoder)
- Real-time performance (~50ms in web browser)
- Enables ambiguity-aware multi-mask output

From [SAM_STUDY_GENERAL.md](../../../PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md) lines 600-634:
- Sparse prompts use positional encoding + learned type embeddings
- Dense prompts use convolutional embedding
- Text prompts leverage CLIP encoder

**Why Prompt Encoding Matters:**

The prompt encoder bridges user intent and visual understanding:
1. **Spatial grounding**: Points/boxes specify WHERE to segment
2. **Semantic guidance**: Text describes WHAT to segment
3. **Refinement input**: Masks provide PRIOR segmentation to improve
4. **Ambiguity resolution**: Multiple prompts resolve uncertainty

---

## Section 1: Prompt Encoder Architecture Overview

### 1.1 Two-Branch Design

The prompt encoder splits into two processing branches:

```
User Prompts
    |
    +--> Sparse Branch: Points, Boxes, Text
    |         |
    |         +--> Positional Encoding
    |         +--> Type Embeddings (learned)
    |         +--> Sum --> Sparse Embeddings
    |
    +--> Dense Branch: Masks
              |
              +--> Convolutional Embedding
              +--> Element-wise Sum with Image Embedding
              +--> Dense Embeddings
```

### 1.2 Output Dimensions

From [Encord SAM Guide](https://encord.com/blog/segment-anything-model-explained/) (accessed 2025-11-20):

**Sparse Embeddings:**
- Shape: `(N_prompts, 256)`
- N_prompts varies based on input (points, box corners, etc.)
- 256-dimensional embedding space matches image encoder output

**Dense Embeddings:**
- Shape: `(1, 256, 64, 64)`
- Spatial resolution matches downsampled image embedding
- Added element-wise to image features

### 1.3 Component Breakdown

**Learned Components:**
- Point type embeddings (foreground/background)
- Box corner embeddings (top-left/bottom-right)
- No-mask embedding (when mask prompt absent)
- IoU token embedding
- Mask tokens (for multi-mask output)

**Fixed Components:**
- Positional encoding (sine-cosine based)
- CLIP text encoder (frozen)

### 1.4 Design Philosophy

From [Towards AI SAM Analysis](https://towardsai.net/p/generative-ai/sam-a-image-segmentation-foundation-model) (accessed 2025-11-20):

The prompt encoder embodies several design principles:
- **Efficiency**: Lightweight to enable real-time interaction
- **Flexibility**: Unified treatment of diverse prompt types
- **Composability**: Prompts can be combined freely
- **Ambiguity awareness**: Supports multiple valid interpretations

---

## Section 2: Sparse Prompts - Points and Boxes

### 2.1 Point Prompt Encoding

Points are the most intuitive prompt type - click to segment.

**Encoding Process:**

```python
def encode_point(coord, label):
    """
    Encode a single point prompt.

    Args:
        coord: (x, y) pixel coordinates
        label: 1 for foreground, 0 for background

    Returns:
        point_embedding: (1, 256) tensor
    """
    # Step 1: Positional encoding of coordinates
    pos_enc = positional_encoding_2d(coord)  # (256,)

    # Step 2: Learned type embedding
    if label == 1:
        type_enc = foreground_embedding  # Learned (256,)
    else:
        type_enc = background_embedding  # Learned (256,)

    # Step 3: Combine
    point_embedding = pos_enc + type_enc

    return point_embedding
```

**Point Labels:**
- `1`: Foreground (include this region)
- `0`: Background (exclude this region)

**Multi-Point Prompts:**

Users can provide multiple points for refinement:
```python
# Example: Foreground point + background points
input_points = np.array([
    [500, 375],  # Foreground
    [200, 100],  # Background
    [800, 600]   # Background
])
input_labels = np.array([1, 0, 0])

# Each point encoded separately, then stacked
embeddings = [encode_point(p, l) for p, l in zip(input_points, input_labels)]
sparse_embeddings = torch.stack(embeddings)  # (3, 256)
```

### 2.2 Box Prompt Encoding

Bounding boxes provide region-level guidance.

**Encoding Process:**

```python
def encode_box(box):
    """
    Encode a bounding box as two corner points.

    Args:
        box: [x1, y1, x2, y2] coordinates

    Returns:
        box_embedding: (2, 256) tensor
    """
    # Extract corners
    top_left = box[:2]      # (x1, y1)
    bottom_right = box[2:]  # (x2, y2)

    # Encode as special point types
    tl_enc = positional_encoding_2d(top_left) + top_left_embedding
    br_enc = positional_encoding_2d(bottom_right) + bottom_right_embedding

    return torch.stack([tl_enc, br_enc])  # (2, 256)
```

**Box Type Embeddings:**
- Top-left corner: Learned embedding (label=2)
- Bottom-right corner: Learned embedding (label=3)

**Robustness to Loose Boxes:**

SAM handles imprecise bounding boxes gracefully:
- Boxes 10-20% larger than object still work well
- Shifted boxes produce reasonable masks
- This robustness comes from training with perturbed boxes

### 2.3 Text Prompt Encoding

Text prompts enable natural language segmentation (via CLIP).

**Encoding Process:**

```python
def encode_text(text_prompt):
    """
    Encode text using CLIP text encoder.

    Args:
        text_prompt: String describing object to segment

    Returns:
        text_embedding: (1, 256) tensor
    """
    # Use CLIP text encoder
    clip_embedding = clip_text_encoder(text_prompt)  # (512,)

    # Project to SAM embedding dimension
    text_embedding = linear_projection(clip_embedding)  # (256,)

    return text_embedding
```

**Text Prompt Capabilities:**
- Object categories: "dog", "car", "building"
- Descriptive phrases: "red apple on the table"
- Relational: "person wearing a hat"

**Limitations:**
- Text prompts less precise than spatial prompts
- Best used with SAM 3 which has native text support
- Original SAM treats text as experimental feature

### 2.4 Combining Multiple Sparse Prompts

Prompts can be combined for better results:

```python
# Example: Box + refinement points
box = np.array([100, 100, 400, 300])
points = np.array([[250, 200], [150, 250]])  # fg, bg
labels = np.array([1, 0])

# Encode all
box_emb = encode_box(box)           # (2, 256)
point_embs = encode_points(points, labels)  # (2, 256)

# Concatenate
sparse_embeddings = torch.cat([box_emb, point_embs], dim=0)  # (4, 256)
```

**Combination Strategies:**
- Box + foreground point: Refine within region
- Box + background points: Exclude specific areas
- Multiple foreground points: Indicate object extent
- Points only: Quick interactive selection

---

## Section 3: Dense Prompts - Mask Encoding

### 3.1 Mask Prompt Purpose

Dense mask prompts enable iterative refinement workflows:
- Provide rough initial mask for refinement
- Use previous model output as new input
- Enable interactive editing of segmentation

### 3.2 Mask Embedding Architecture

From [SAM_STUDY_GENERAL.md](../../../PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md) lines 607-610:

**Convolutional Embedding Network:**

```python
class MaskEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        # Convolutional layers to embed mask
        self.conv1 = nn.Conv2d(1, embed_dim // 4, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=2, stride=2)

        # Layer normalization
        self.ln1 = nn.LayerNorm(embed_dim // 4)
        self.ln2 = nn.LayerNorm(embed_dim)

        # No-mask embedding (when mask not provided)
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def forward(self, mask):
        """
        Encode mask into dense embedding.

        Args:
            mask: (1, 1, H, W) binary or soft mask

        Returns:
            dense_embedding: (1, 256, 64, 64)
        """
        if mask is None:
            # Return no-mask embedding broadcast to spatial dims
            return self.no_mask_embed.weight.view(1, -1, 1, 1).expand(
                1, 256, 64, 64
            )

        # Downsample mask to match image embedding resolution
        # Input: (1, 1, 256, 256) -> Output: (1, 256, 64, 64)
        x = self.conv1(mask)  # (1, 64, 128, 128)
        x = self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = F.gelu(x)

        x = self.conv2(x)     # (1, 256, 64, 64)
        x = self.ln2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return x
```

### 3.3 Mask-Image Fusion

The dense mask embedding combines with image embedding:

```python
def fuse_mask_with_image(image_embedding, mask_embedding):
    """
    Combine mask embedding with image embedding.

    Args:
        image_embedding: (1, 256, 64, 64) from image encoder
        mask_embedding: (1, 256, 64, 64) from mask encoder

    Returns:
        fused_embedding: (1, 256, 64, 64)
    """
    # Element-wise addition
    fused = image_embedding + mask_embedding

    return fused
```

**Why Element-wise Addition:**
- Preserves spatial structure
- Allows mask to modulate image features
- Computationally efficient
- Enables soft/probabilistic masks

### 3.4 Iterative Refinement Workflow

Dense prompts enable multi-step refinement:

```python
# Step 1: Initial segmentation with point
masks1, scores1, logits1 = predictor.predict(
    point_coords=np.array([[500, 375]]),
    point_labels=np.array([1]),
    multimask_output=True
)

# Step 2: Refine with previous mask as prompt
best_mask = masks1[np.argmax(scores1)]
masks2, scores2, logits2 = predictor.predict(
    point_coords=np.array([[500, 375]]),
    point_labels=np.array([1]),
    mask_input=logits1[np.argmax(scores1)][None, :, :],  # Use logits!
    multimask_output=False  # Single refined output
)
```

**Key Insight:** Use logits (pre-sigmoid) as mask input, not binary masks.

### 3.5 No-Mask Embedding

When no mask prompt is provided:

```python
# Learned embedding for "no mask" condition
no_mask_embed = nn.Embedding(1, 256)

# Broadcast to spatial dimensions
dense_embedding = no_mask_embed.weight.view(1, 256, 1, 1)
dense_embedding = dense_embedding.expand(1, 256, 64, 64)
```

This ensures consistent tensor shapes regardless of mask presence.

---

## Section 4: Positional Encoding System

### 4.1 Fourier Feature Encoding

SAM uses sine-cosine positional encoding for spatial locations.

**Mathematical Foundation:**

```python
def positional_encoding_2d(coord, num_pos_feats=128, temperature=10000):
    """
    Generate 2D positional encoding for a coordinate.

    Args:
        coord: (x, y) normalized to [0, 1]
        num_pos_feats: Features per dimension (total = 2 * num_pos_feats)
        temperature: Scaling factor for frequencies

    Returns:
        pos_enc: (256,) positional encoding
    """
    x, y = coord

    # Frequency bands
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    # Encode x coordinate
    pos_x = x / dim_t
    pos_x = torch.stack([pos_x[0::2].sin(), pos_x[1::2].cos()], dim=-1)
    pos_x = pos_x.flatten()  # (128,)

    # Encode y coordinate
    pos_y = y / dim_t
    pos_y = torch.stack([pos_y[0::2].sin(), pos_y[1::2].cos()], dim=-1)
    pos_y = pos_y.flatten()  # (128,)

    # Concatenate
    pos_enc = torch.cat([pos_x, pos_y], dim=0)  # (256,)

    return pos_enc
```

### 4.2 Why Fourier Features?

From neural radiance field (NeRF) research:
- Enables learning high-frequency functions
- Prevents "spectral bias" toward low frequencies
- Different frequencies capture different spatial scales

**Frequency Spectrum:**
- Low frequencies: Coarse spatial location
- High frequencies: Fine-grained position discrimination

### 4.3 Coordinate Normalization

Coordinates are normalized before encoding:

```python
def normalize_coords(coords, image_size):
    """
    Normalize pixel coordinates to [0, 1] range.

    Args:
        coords: (N, 2) pixel coordinates
        image_size: (H, W) image dimensions

    Returns:
        normalized: (N, 2) normalized coordinates
    """
    H, W = image_size
    normalized = coords.clone().float()
    normalized[:, 0] /= W  # x
    normalized[:, 1] /= H  # y

    return normalized
```

### 4.4 Dense Positional Encoding

For the image embedding, SAM pre-computes dense positional encodings:

```python
def get_dense_pe(self, image_size=(64, 64)):
    """
    Get positional encoding for entire image grid.

    Returns:
        dense_pe: (1, 256, 64, 64)
    """
    H, W = image_size

    # Create coordinate grid
    y_coords = torch.arange(H).float() / H
    x_coords = torch.arange(W).float() / W

    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords = torch.stack([xx, yy], dim=-1)  # (64, 64, 2)

    # Encode each position
    pe = positional_encoding_2d_grid(coords)  # (64, 64, 256)
    pe = pe.permute(2, 0, 1).unsqueeze(0)     # (1, 256, 64, 64)

    return pe
```

This dense PE is added to image embeddings before mask decoding.

---

## Section 5: Learned Embeddings

### 5.1 Type Embeddings Overview

SAM learns distinct embeddings for each prompt type:

```python
class PromptEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_points=4):
        super().__init__()

        # Point type embeddings
        self.point_embeddings = nn.ModuleList([
            nn.Embedding(1, embed_dim) for _ in range(num_points)
        ])
        # Index 0: Background point
        # Index 1: Foreground point
        # Index 2: Top-left box corner
        # Index 3: Bottom-right box corner

        # Not-a-point embedding (padding)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
```

### 5.2 Embedding Semantics

**Point Type Meanings:**

| Index | Type | Semantic |
|-------|------|----------|
| 0 | Background | "Exclude this region" |
| 1 | Foreground | "Include this region" |
| 2 | Box Top-Left | "Object starts here" |
| 3 | Box Bottom-Right | "Object ends here" |

### 5.3 Mask Decoder Tokens

Additional learned embeddings for the mask decoder:

```python
class MaskDecoder(nn.Module):
    def __init__(self, embed_dim=256, num_mask_tokens=4):
        super().__init__()

        # IoU prediction token
        self.iou_token = nn.Embedding(1, embed_dim)

        # Mask output tokens (for multi-mask output)
        self.mask_tokens = nn.Embedding(num_mask_tokens, embed_dim)
        # Token 0: Whole object
        # Token 1: Part
        # Token 2: Subpart
        # Token 3: (reserved for single-mask mode)
```

### 5.4 Learning Prompt Semantics

The embeddings are learned end-to-end during training:

**Training Signal:**
- Points: IoU between predicted and ground-truth masks
- Boxes: Same IoU objective with box-derived prompts
- Text: Contrastive alignment with CLIP features

**What the Model Learns:**
- Foreground embedding: "Attend to this location"
- Background embedding: "Suppress this region"
- Box embeddings: "Object bounded by these corners"

### 5.5 Embedding Visualization

Researchers have visualized learned embeddings:
- Foreground/background embeddings are nearly opposite
- Box corner embeddings capture spatial relationships
- Embeddings cluster by semantic function

---

## Section 6: Multi-Prompt Handling

### 6.1 Prompt Combination Rules

SAM handles multiple prompts through concatenation:

```python
def combine_prompts(point_embeddings, box_embeddings, text_embedding):
    """
    Combine all sparse prompt embeddings.

    Args:
        point_embeddings: (N_points, 256) or None
        box_embeddings: (2 * N_boxes, 256) or None
        text_embedding: (1, 256) or None

    Returns:
        sparse_embeddings: (N_total, 256)
    """
    embeddings = []

    if point_embeddings is not None:
        embeddings.append(point_embeddings)

    if box_embeddings is not None:
        embeddings.append(box_embeddings)

    if text_embedding is not None:
        embeddings.append(text_embedding)

    if len(embeddings) == 0:
        # No prompts - return empty
        return torch.zeros(0, 256)

    return torch.cat(embeddings, dim=0)
```

### 6.2 Prompt Attention in Decoder

The mask decoder processes prompts through self-attention:

```python
# In mask decoder
def forward(self, image_embedding, sparse_embeddings, dense_embeddings):
    # Combine with output tokens
    output_tokens = torch.cat([
        self.iou_token.weight,      # (1, 256)
        self.mask_tokens.weight,    # (4, 256)
    ], dim=0)

    tokens = torch.cat([output_tokens, sparse_embeddings], dim=0)

    # Self-attention among all tokens
    # Prompts can attend to each other
    tokens = self.transformer_block(
        query=tokens,
        key=image_embedding,
        value=image_embedding
    )
```

### 6.3 Resolving Conflicting Prompts

When prompts conflict (e.g., overlapping boxes), SAM:
1. Generates multiple mask candidates
2. Scores each with predicted IoU
3. Returns all with confidence scores

```python
# Multiple boxes for same image
boxes = np.array([
    [100, 100, 300, 300],  # Box 1
    [200, 200, 400, 400],  # Box 2 (overlapping)
])

# Process each separately
for box in boxes:
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    # Each box produces 3 mask candidates
```

### 6.4 Batch Processing

For efficiency, multiple prompts can be batched:

```python
def batch_predict(predictor, points_batch, labels_batch):
    """
    Process multiple prompt sets in batch.

    Args:
        points_batch: List of point arrays
        labels_batch: List of label arrays

    Returns:
        all_masks: List of mask arrays
    """
    # Transform all prompts
    transformed_points = []
    transformed_labels = []

    for points, labels in zip(points_batch, labels_batch):
        tp, tl = predictor.transform.apply_coords(points, labels)
        transformed_points.append(tp)
        transformed_labels.append(tl)

    # Batch inference
    masks_batch = predictor.predict_batch(
        transformed_points, transformed_labels
    )

    return masks_batch
```

### 6.5 Prompt Engineering Best Practices

**For best results:**
1. Start with single point, add more for refinement
2. Use box when object boundaries are known
3. Combine box + points for fine control
4. Add background points to exclude unwanted regions
5. Use mask prompt for iterative refinement

---

## Section 7: Implementation Details

### 7.1 Complete Prompt Encoder Code

```python
class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        image_embedding_size: tuple = (64, 64),
        input_image_size: tuple = (1024, 1024),
        mask_in_chans: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size

        # Positional encoding
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # Point embeddings: bg, fg, top-left, bottom-right
        self.num_point_embeddings = 4
        self.point_embeddings = nn.Embedding(
            self.num_point_embeddings, embed_dim
        )

        # Not-a-point embedding
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        # Mask encoder
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )

        # No-mask embedding
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self):
        """Get positional encoding for image embedding grid."""
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, points, labels, pad):
        """Embed point prompts."""
        points = points + 0.5  # Shift to center of pixel

        if pad:
            # Pad with not-a-point
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)

        # Positional encoding
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )

        # Add type embeddings
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings.weight[0]
        point_embedding[labels == 1] += self.point_embeddings.weight[1]

        return point_embedding

    def _embed_boxes(self, boxes):
        """Embed box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel

        # Reshape to corner points
        coords = boxes.reshape(-1, 2, 2)

        # Encode corners
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size
        )

        # Add type embeddings
        corner_embedding[:, 0, :] += self.point_embeddings.weight[2]  # Top-left
        corner_embedding[:, 1, :] += self.point_embeddings.weight[3]  # Bottom-right

        return corner_embedding

    def _embed_masks(self, masks):
        """Embed mask prompts."""
        return self.mask_downscaling(masks)

    def forward(self, points, boxes, masks):
        """
        Encode prompts into sparse and dense embeddings.

        Returns:
            sparse_embeddings: (B, N, embed_dim)
            dense_embeddings: (B, embed_dim, H, W)
        """
        sparse_embeddings = []

        # Encode points
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings.append(point_embeddings)

        # Encode boxes
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings.append(box_embeddings)

        # Combine sparse
        if len(sparse_embeddings) > 0:
            sparse_embeddings = torch.cat(sparse_embeddings, dim=1)
        else:
            sparse_embeddings = torch.empty(
                (1, 0, self.embed_dim), device=self.point_embeddings.weight.device
            )

        # Encode masks
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                1, -1, *self.image_embedding_size
            )

        return sparse_embeddings, dense_embeddings
```

### 7.2 Performance Characteristics

**Encoding Speed:**
- Point encoding: <1ms
- Box encoding: <1ms
- Mask encoding: ~5ms (convolutions)
- Total prompt encoding: ~5-10ms

**Memory Footprint:**
- Prompt encoder parameters: ~0.5M
- Per-prompt memory: Negligible
- Mask prompt: ~1MB (256x256 input)

### 7.3 Inference Pipeline

```python
# Complete inference with prompts
def segment_with_prompts(predictor, image, points=None, boxes=None, masks=None):
    # 1. Set image (computes image embedding once)
    predictor.set_image(image)

    # 2. Prepare prompts
    point_coords = points[:, :2] if points is not None else None
    point_labels = points[:, 2] if points is not None else None

    # 3. Predict masks
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=boxes,
        mask_input=masks,
        multimask_output=True
    )

    return masks, scores, logits
```

---

## Section 8: ARR-COC Integration

### 8.1 Prompts as Relevance Allocation

In the ARR-COC framework, SAM's prompts map directly to relevance realization:

**Propositional Knowing (What):**
- Text prompts: "Segment the cat" = explicit propositional statement
- Labels: Foreground/background = binary propositions about inclusion

**Perspectival Knowing (How):**
- Point prompts: Spatial perspective on where attention should focus
- Box prompts: Bounded perspective defining region of interest
- Positional encoding: How location relates to semantic content

**Participatory Knowing (Engagement):**
- Multi-point refinement: Interactive loop of click-observe-refine
- Mask prompts: Prior segmentation guiding new understanding
- Iterative improvement: Participatory engagement with visual content

### 8.2 Sparse vs Dense as Attention Modes

**Sparse Prompts = Focal Attention:**
- Points/boxes specify discrete loci of attention
- Like eye fixations in visual search
- High precision, low coverage

**Dense Prompts = Distributed Attention:**
- Masks provide field-wide relevance weighting
- Like spatial attention maps in visual cortex
- Lower precision, broader coverage

### 8.3 ARR-COC Code Example

```python
class RelevanceGuidedSegmenter:
    """
    SAM segmentation guided by ARR relevance principles.
    """

    def __init__(self, sam_predictor):
        self.predictor = sam_predictor
        self.relevance_history = []

    def segment_with_relevance(self, image, relevance_map):
        """
        Use relevance map to guide prompt generation.

        Args:
            image: Input image
            relevance_map: (H, W) attention/relevance weights

        Returns:
            masks: Segmentation masks
            relevance_alignment: How well masks align with relevance
        """
        self.predictor.set_image(image)

        # Extract prompts from relevance map
        # High relevance -> foreground points
        # Low relevance -> background points
        fg_points = self._extract_peaks(relevance_map, threshold=0.8)
        bg_points = self._extract_valleys(relevance_map, threshold=0.2)

        # Combine prompts
        all_points = np.vstack([fg_points, bg_points])
        all_labels = np.array(
            [1] * len(fg_points) + [0] * len(bg_points)
        )

        # Predict with relevance-guided prompts
        masks, scores, logits = self.predictor.predict(
            point_coords=all_points,
            point_labels=all_labels,
            multimask_output=True
        )

        # Select mask most aligned with relevance
        best_idx = self._compute_relevance_alignment(
            masks, relevance_map
        )

        return masks[best_idx], scores[best_idx]

    def _extract_peaks(self, relevance_map, threshold):
        """Extract local maxima above threshold as foreground points."""
        from scipy.ndimage import maximum_filter

        local_max = maximum_filter(relevance_map, size=20)
        peaks = (relevance_map == local_max) & (relevance_map > threshold)

        coords = np.argwhere(peaks)
        return coords[:, ::-1]  # (x, y) format

    def _extract_valleys(self, relevance_map, threshold):
        """Extract local minima below threshold as background points."""
        from scipy.ndimage import minimum_filter

        local_min = minimum_filter(relevance_map, size=20)
        valleys = (relevance_map == local_min) & (relevance_map < threshold)

        coords = np.argwhere(valleys)
        return coords[:, ::-1]

    def _compute_relevance_alignment(self, masks, relevance_map):
        """Find mask most aligned with relevance distribution."""
        alignments = []

        for mask in masks:
            # Resize mask to relevance map size
            mask_resized = cv2.resize(
                mask.astype(float),
                relevance_map.shape[::-1]
            )

            # Compute correlation
            alignment = np.corrcoef(
                mask_resized.flatten(),
                relevance_map.flatten()
            )[0, 1]
            alignments.append(alignment)

        return np.argmax(alignments)
```

### 8.4 Prompt Encoding as Relevance Transformation

The prompt encoder transforms user-specified relevance into a format compatible with visual features:

```
User Intent (Relevance)
    |
    v
Prompt Specification (Points, Boxes, Masks, Text)
    |
    v
Prompt Encoder (Positional + Type Embeddings)
    |
    v
Embedding Space (Compatible with Image Features)
    |
    v
Mask Decoder (Relevance-Guided Feature Selection)
    |
    v
Segmentation Mask (Realized Relevance)
```

This pipeline embodies the ARR principle that relevance must be transformed through multiple representational stages to bridge intention and perception.

---

## Sources

**Source Documents:**
- [SAM_STUDY_GENERAL.md](../../../PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md) - Lines 600-634 (Prompt Encoder architecture)

**Web Research:**
- [Encord - Segment Anything Model Explained](https://encord.com/blog/segment-anything-model-explained/) - Comprehensive architecture guide (accessed 2025-11-20)
- [Towards AI - SAM Foundation Model](https://towardsai.net/p/generative-ai/sam-a-image-segmentation-foundation-model) - Prompt encoder overview (accessed 2025-11-20)
- [V7 Labs - SAM Guide](https://www.v7labs.com/blog/segment-anything-model-sam) - Positional encoding details (accessed 2025-11-20)
- [Medium - SAM Encoder Deep Dive](https://medium.com/data-science/how-does-the-segment-anything-models-sam-s-encoder-work-003a8a6e3f8b) - Technical implementation (accessed 2025-11-20)

**Code References:**
- [SAM GitHub Repository](https://github.com/facebookresearch/segment-anything) - Official implementation
- [SAM Paper](https://arxiv.org/abs/2304.02643) - arXiv:2304.02643 (Kirillov et al., 2023)

**Additional Resources:**
- [CLIP Paper](https://arxiv.org/abs/2103.00020) - Text encoder foundation
- [NeRF Fourier Features](https://arxiv.org/abs/2006.10739) - Positional encoding theory
