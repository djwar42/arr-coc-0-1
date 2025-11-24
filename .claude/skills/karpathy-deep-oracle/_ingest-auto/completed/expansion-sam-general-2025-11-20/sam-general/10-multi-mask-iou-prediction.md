# Multi-Mask Output and IoU Prediction in SAM

## Overview

The Segment Anything Model (SAM) introduces a novel approach to handling prompt ambiguity through **multi-mask output prediction** combined with **learned IoU scoring**. When a user provides a single point prompt, it could legitimately correspond to multiple valid interpretations - a point on a shirt could mean "shirt", "person", or "torso". Rather than forcing the model to choose one interpretation, SAM predicts multiple mask candidates and ranks them by estimated quality.

This design represents a fundamental shift from traditional segmentation models that output a single mask. SAM outputs **three mask candidates per prompt**, each representing a different level of segmentation granularity (whole object, part, subpart), along with confidence scores that enable automatic selection of the most appropriate mask.

**Key Components:**
- **Multi-mask prediction**: Three simultaneous mask outputs per prompt
- **IoU prediction head**: Learned quality estimation for each mask
- **Minimum-loss training**: Backpropagation only through best-matching mask
- **Hierarchical granularity**: Whole/part/subpart coverage

**Sources:**
- [SAM Paper](https://arxiv.org/abs/2304.02643) - Kirillov et al., 2023
- [SAM: Segment Anything with Prompts](https://medium.com/@kdk199604/sam-segment-anything-with-prompts-not-labels-7a85e6ec4d09) - Dong-Keon Kim (accessed 2025-11-20)

---

## Section 1: Multi-Mask Output Overview

### Why Three Masks?

SAM predicts **three masks per prompt** to capture the typical granularity hierarchy found in natural images:

```
Granularity Hierarchy Example (Person Image):
├── Mask 1 (Whole): Complete person
├── Mask 2 (Part): Upper body / torso
└── Mask 3 (Subpart): Face / head
```

**The Three Masks Represent:**

1. **Whole Object**: The largest semantically coherent region
2. **Part**: A meaningful sub-component of the whole
3. **Subpart**: The most localized interpretation

### Architecture for Multi-Mask Output

The mask decoder generates three masks through **learnable mask tokens**:

```python
class MaskDecoder(nn.Module):
    def __init__(self, transformer_dim=256):
        super().__init__()

        # Three learnable mask tokens
        self.num_mask_tokens = 3
        self.mask_tokens = nn.Embedding(
            self.num_mask_tokens,
            transformer_dim
        )

        # IoU prediction token
        self.iou_token = nn.Embedding(1, transformer_dim)

        # MLP heads for each mask
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for _ in range(self.num_mask_tokens)
        ])

        # IoU prediction head
        self.iou_prediction_head = MLP(
            transformer_dim, 256, self.num_mask_tokens, 3
        )
```

### Mask Generation Process

Each mask token produces its own mask through a dynamic linear classifier:

```python
def predict_masks(self, image_embeddings, output_tokens):
    """Generate three masks from output tokens."""

    masks = []
    for i in range(self.num_mask_tokens):
        # Get hypernetwork weights from token
        hyper_weights = self.output_hypernetworks_mlps[i](
            output_tokens[:, i, :]
        )

        # Dynamic per-pixel classification
        # Shape: (B, 1, H, W)
        mask_logits = torch.einsum(
            'bchw,bc->bhw',
            image_embeddings,
            hyper_weights
        )
        masks.append(mask_logits)

    # Stack: (B, 3, H, W)
    return torch.stack(masks, dim=1)
```

### Single vs Multi-Mask Mode

SAM supports both modes through the `multimask_output` parameter:

```python
# Multi-mask mode (default for ambiguous prompts)
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True  # Returns 3 masks
)

# Single-mask mode (for unambiguous prompts)
masks, scores, logits = predictor.predict(
    point_coords=input_points,  # Multiple points
    point_labels=input_labels,
    multimask_output=False  # Returns 1 mask
)
```

**When to Use Each Mode:**
- **Multi-mask (True)**: Single point, single box, ambiguous prompts
- **Single-mask (False)**: Multiple points, refined prompts, iterative refinement

**Reference:** [SAM Study](../source-documents/) and [Segment Anything Explained](https://storrs.io/segment-anything-explained/) (accessed 2025-11-20)

---

## Section 2: Ambiguity Handling

### The Ambiguity Problem

Traditional segmentation models face a fundamental challenge with ambiguous prompts:

```
Problem: Single point on a car wheel
├── Valid Interpretation 1: Just the wheel
├── Valid Interpretation 2: The entire car
└── Valid Interpretation 3: The wheel hub/rim

Single-output model → Forces averaging → Blurry, poor mask
Multi-output model → Proposes all → User/system selects best
```

### SAM's Ambiguity-Aware Design

SAM explicitly tolerates ambiguity by requiring the model to produce **at least one valid mask** for any reasonable interpretation:

```python
# Training objective: minimum loss over candidates
def ambiguity_aware_loss(pred_masks, gt_mask, pred_ious):
    """
    Compute loss only for best-matching mask.

    This operationalizes the 'any valid mask' requirement
    without penalizing plausible alternatives.
    """
    batch_size, num_masks = pred_masks.shape[:2]

    losses = []
    for i in range(num_masks):
        # Compute loss for each mask candidate
        mask_loss = focal_loss(pred_masks[:, i], gt_mask) + \
                   dice_loss(pred_masks[:, i], gt_mask)
        losses.append(mask_loss)

    losses = torch.stack(losses, dim=1)  # (B, 3)

    # Backpropagate only through minimum loss
    min_loss, min_idx = losses.min(dim=1)

    return min_loss.mean()
```

### Hierarchical Granularity Coverage

The three masks naturally capture different semantic levels:

```
Example: Point on person's eye

Mask 1 (Whole):     ████████████████████
                    █   Full Person    █
                    ████████████████████
                    IoU Score: 0.75

Mask 2 (Part):      ████████████
                    █   Face   █
                    ████████████
                    IoU Score: 0.92  ← Best match

Mask 3 (Subpart):   ████
                    █Eye█
                    ████
                    IoU Score: 0.85
```

### Training for Ambiguity

SAM's training protocol explicitly addresses ambiguity:

1. **11 Rounds Per Mask**: Simulates interactive refinement
2. **Valid at Every Step**: Must produce valid mask from first click
3. **Minimum-Loss Selection**: Only best match receives gradients

```python
def train_step(model, image, gt_masks):
    """Training with simulated prompts over 11 rounds."""

    for mask_idx, gt_mask in enumerate(gt_masks):
        for round_num in range(11):
            # Simulate prompts of increasing specificity
            if round_num == 0:
                prompt = sample_center_point(gt_mask)
            else:
                # Add error-correcting points
                prompt = add_correction_point(
                    prev_pred, gt_mask, prompt
                )

            # Model must be valid at EVERY round
            pred_masks, pred_ious = model(image, prompt)

            # Minimum loss across 3 candidates
            loss = ambiguity_aware_loss(pred_masks, gt_mask, pred_ious)
            loss.backward()
```

### Confidence-Based Automatic Selection

When automatic mask selection is needed, SAM uses predicted IoU scores:

```python
def select_best_mask(masks, iou_scores):
    """
    Automatic mask selection based on predicted IoU.

    In practice, this achieves near-oracle performance
    for most images.
    """
    best_idx = iou_scores.argmax(dim=-1)

    # Gather best mask per batch element
    batch_size = masks.shape[0]
    best_masks = masks[
        torch.arange(batch_size),
        best_idx
    ]

    return best_masks, iou_scores.max(dim=-1).values
```

**Key Insight**: The minimum-loss training combined with IoU prediction creates a self-consistent system where the model learns both to propose diverse valid masks AND to accurately rank them.

**Reference:** [SAM Paper](https://arxiv.org/abs/2304.02643) Section 3

---

## Section 3: IoU Prediction Head

### Purpose and Architecture

The IoU prediction head estimates the **Intersection over Union** between each predicted mask and the (unknown) ground truth. This enables:

1. **Automatic mask ranking** without ground truth
2. **Confidence filtering** for downstream tasks
3. **Quality-aware selection** in automatic pipelines

```python
class IoUPredictionHead(nn.Module):
    """
    Predicts IoU scores for each mask candidate.

    Architecture: 3-layer MLP
    Input: IoU output token (256-dim)
    Output: 3 IoU scores (one per mask)
    """

    def __init__(self, hidden_dim=256, num_masks=3):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_masks),
            nn.Sigmoid()  # IoU in [0, 1]
        )

    def forward(self, iou_token):
        """
        Args:
            iou_token: (B, 256) from mask decoder
        Returns:
            iou_scores: (B, 3) predicted IoU per mask
        """
        return self.mlp(iou_token)
```

### Training the IoU Head

The IoU prediction is trained with Mean Squared Error (MSE) against actual IoU:

```python
def iou_prediction_loss(pred_iou, pred_mask, gt_mask):
    """
    Supervise IoU prediction with actual mask IoU.

    This teaches the model to estimate its own quality.
    """
    # Compute actual IoU
    intersection = (pred_mask * gt_mask).sum(dim=(-2, -1))
    union = pred_mask.sum(dim=(-2, -1)) + gt_mask.sum(dim=(-2, -1)) - intersection
    actual_iou = intersection / (union + 1e-6)

    # MSE loss
    return F.mse_loss(pred_iou, actual_iou)
```

### IoU Token Flow

The IoU token flows through the same transformer as mask tokens:

```
Input Tokens:
├── Mask Token 1 (256-dim)
├── Mask Token 2 (256-dim)
├── Mask Token 3 (256-dim)
├── IoU Token (256-dim)  ← Dedicated token
└── Prompt Tokens...

        ↓ Transformer Blocks ↓

Output:
├── Updated Mask Token 1 → MLP → Mask 1
├── Updated Mask Token 2 → MLP → Mask 2
├── Updated Mask Token 3 → MLP → Mask 3
└── Updated IoU Token → IoU Head → [0.85, 0.92, 0.78]
```

### Practical IoU Prediction Performance

From empirical studies, the IoU prediction correlates well with actual quality:

```python
# Typical correlation between predicted and actual IoU
# Based on SA-1B validation

def evaluate_iou_prediction(model, val_loader):
    """Measure IoU prediction accuracy."""

    predicted_ious = []
    actual_ious = []

    for images, gt_masks, prompts in val_loader:
        pred_masks, pred_iou = model(images, prompts)

        # Select best mask by predicted IoU
        best_idx = pred_iou.argmax(dim=-1)
        best_mask = pred_masks[range(len(images)), best_idx]

        # Compute actual IoU
        actual = compute_iou(best_mask, gt_masks)

        predicted_ious.append(pred_iou.max(dim=-1).values)
        actual_ious.append(actual)

    # Pearson correlation typically > 0.8
    correlation = pearsonr(predicted_ious, actual_ious)
    return correlation

# Empirical results:
# - Correlation: 0.82-0.88
# - MAE: 0.05-0.08 IoU
# - Best-mask selection accuracy: ~85%
```

### IoU Thresholds in Practice

Different applications use different IoU thresholds:

```python
# Automatic mask generation thresholds
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    pred_iou_thresh=0.88,        # Quality threshold
    stability_score_thresh=0.95,  # Stability threshold
    box_nms_thresh=0.7,          # NMS threshold
    min_mask_region_area=100     # Size threshold
)
```

**Reference:** [SAM GitHub Issues](https://github.com/facebookresearch/segment-anything/issues/495) and [Segment Anything Explained](https://storrs.io/segment-anything-explained/) (accessed 2025-11-20)

---

## Section 4: Mask Selection Strategies

### Automatic Selection (Default)

The simplest strategy uses predicted IoU scores directly:

```python
def automatic_selection(masks, iou_scores):
    """
    Select mask with highest predicted IoU.

    This is the default behavior in SAM.
    """
    best_idx = iou_scores.argmax(dim=-1)
    return masks[best_idx], iou_scores[best_idx]

# Usage
masks, scores, _ = predictor.predict(
    point_coords=point,
    point_labels=label,
    multimask_output=True
)

# Automatic: use argmax
best_mask = masks[scores.argmax()]
```

### Oracle Selection (Analysis)

For benchmarking, oracle selection picks the best actual match:

```python
def oracle_selection(masks, gt_mask):
    """
    Select mask with highest actual IoU (oracle).

    Used for analysis to understand ambiguity effects.
    """
    ious = []
    for mask in masks:
        iou = compute_iou(mask, gt_mask)
        ious.append(iou)

    best_idx = np.argmax(ious)
    return masks[best_idx], ious[best_idx]

# Empirical finding:
# - Oracle outperforms automatic by 2-5% IoU on average
# - Gap indicates room for better IoU prediction
```

### Application-Specific Selection

Different applications require different selection strategies:

```python
class TaskSpecificSelector:
    """Select masks based on task requirements."""

    def select_for_detection(self, masks, scores, detection_box):
        """Select mask with best box overlap."""
        best_idx = 0
        best_overlap = 0

        for i, mask in enumerate(masks):
            mask_box = get_bounding_box(mask)
            overlap = compute_iou_boxes(mask_box, detection_box)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i

        return masks[best_idx]

    def select_largest(self, masks, scores):
        """Select largest mask (whole object)."""
        areas = [mask.sum() for mask in masks]
        return masks[np.argmax(areas)]

    def select_smallest(self, masks, scores):
        """Select smallest mask (fine detail)."""
        areas = [mask.sum() for mask in masks]
        return masks[np.argmin(areas)]

    def select_by_area_ratio(self, masks, scores, target_ratio):
        """Select mask closest to target area ratio."""
        total_pixels = masks[0].numel()
        ratios = [mask.sum() / total_pixels for mask in masks]

        diffs = [abs(r - target_ratio) for r in ratios]
        return masks[np.argmin(diffs)]
```

### Hierarchical Selection

Leverage the granularity hierarchy explicitly:

```python
def hierarchical_selection(masks, scores, granularity='medium'):
    """
    Select based on expected granularity level.

    Masks are typically ordered by size (largest first).
    """
    # Sort masks by area
    areas = [mask.sum().item() for mask in masks]
    sorted_indices = np.argsort(areas)[::-1]  # Descending

    if granularity == 'whole':
        return masks[sorted_indices[0]]
    elif granularity == 'part':
        return masks[sorted_indices[1]]
    elif granularity == 'subpart':
        return masks[sorted_indices[2]]
    else:  # 'best'
        return masks[scores.argmax()]
```

### Ensemble Selection

Combine multiple masks for robustness:

```python
def ensemble_selection(masks, scores, method='weighted'):
    """
    Combine masks into a single prediction.

    Useful when all interpretations are partially correct.
    """
    if method == 'union':
        # Take union of all masks
        return (masks.sum(dim=0) > 0).float()

    elif method == 'intersection':
        # Take intersection
        return (masks.prod(dim=0) > 0).float()

    elif method == 'weighted':
        # IoU-weighted average
        weights = F.softmax(scores, dim=0)
        weighted = (masks * weights.view(-1, 1, 1)).sum(dim=0)
        return (weighted > 0.5).float()

    elif method == 'voting':
        # Majority voting
        votes = masks.sum(dim=0)
        return (votes >= 2).float()  # At least 2 of 3
```

---

## Section 5: Training Objectives

### Combined Loss Function

SAM uses focal loss and dice loss for mask supervision:

```python
def compute_mask_loss(pred_masks, gt_mask, pred_ious):
    """
    Combined loss for multi-mask prediction.

    Components:
    1. Focal loss (handles class imbalance)
    2. Dice loss (overlap-based)
    3. IoU prediction loss (quality estimation)
    """

    num_masks = pred_masks.shape[1]

    # Compute per-mask losses
    mask_losses = []
    for i in range(num_masks):
        pred = pred_masks[:, i]

        # Focal loss
        focal = sigmoid_focal_loss(pred, gt_mask, reduction='mean')

        # Dice loss
        dice = dice_loss(pred.sigmoid(), gt_mask)

        # Combined
        mask_losses.append(focal + dice)

    # Stack and take minimum (ambiguity-aware)
    mask_losses = torch.stack(mask_losses, dim=1)
    min_loss, min_idx = mask_losses.min(dim=1)

    # IoU prediction loss for selected mask
    selected_pred = pred_masks[range(len(pred_masks)), min_idx]
    actual_iou = compute_iou(selected_pred.sigmoid(), gt_mask)
    iou_loss = F.mse_loss(pred_ious[range(len(pred_ious)), min_idx], actual_iou)

    return min_loss.mean() + iou_loss
```

### Focal Loss for Class Imbalance

Masks are highly imbalanced (mostly background):

```python
def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    """
    Focal loss for dense prediction.

    Down-weights easy negatives (background pixels).
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction='none'
    )

    # Focal weight
    p_t = prob * targets + (1 - prob) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma

    # Alpha weighting
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    return (alpha_t * focal_weight * ce_loss).mean()
```

### Dice Loss for Overlap

Dice loss directly optimizes IoU-like overlap:

```python
def dice_loss(pred, target, smooth=1e-6):
    """
    Dice loss for better overlap optimization.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)

    intersection = (pred_flat * target_flat).sum(1)
    union = pred_flat.sum(1) + target_flat.sum(1)

    dice = (2 * intersection + smooth) / (union + smooth)

    return 1 - dice.mean()
```

### Stability Score (Inference)

During inference, a stability score filters unreliable masks:

```python
def compute_stability_score(mask_logits, threshold_offset=1.0):
    """
    Stability score measures mask consistency across thresholds.

    High stability = mask shape doesn't change much with threshold
    Low stability = mask is sensitive to threshold choice
    """
    # Binary masks at different thresholds
    high_thresh_mask = mask_logits > threshold_offset
    low_thresh_mask = mask_logits > -threshold_offset

    # IoU between them
    intersection = (high_thresh_mask & low_thresh_mask).sum()
    union = (high_thresh_mask | low_thresh_mask).sum()

    stability = intersection / (union + 1e-6)

    return stability.item()
```

---

## Section 6: Inference Modes

### Interactive Mode (Single Prompt)

For interactive use, multi-mask output handles ambiguity:

```python
def interactive_segment(predictor, image, point):
    """
    Interactive segmentation with multi-mask output.

    User can refine by adding more points.
    """
    predictor.set_image(image)

    # First click: multi-mask for ambiguity
    masks, scores, logits = predictor.predict(
        point_coords=point,
        point_labels=np.array([1]),
        multimask_output=True
    )

    # Return all 3 for user selection
    return masks, scores

def refine_with_point(predictor, prev_logits, new_point, new_label):
    """
    Refine previous prediction with additional point.

    After refinement, single mask is usually sufficient.
    """
    masks, scores, logits = predictor.predict(
        point_coords=new_point,
        point_labels=new_label,
        mask_input=prev_logits,  # Previous prediction
        multimask_output=False   # Single refined mask
    )

    return masks[0], scores[0], logits
```

### Automatic Mode (Everything)

For automatic segmentation, IoU thresholds filter quality:

```python
def automatic_everything(sam, image,
                         pred_iou_thresh=0.88,
                         stability_score_thresh=0.95):
    """
    Segment everything in image automatically.

    Uses predicted IoU and stability for quality filtering.
    """
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,             # Grid density
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100
    )

    masks = mask_generator.generate(image)

    # Each mask dict contains:
    # - 'segmentation': binary mask
    # - 'predicted_iou': IoU score
    # - 'stability_score': stability
    # - 'area': pixel count
    # - 'bbox': bounding box

    return masks
```

### Batch Mode (Multiple Prompts)

Process multiple prompts efficiently:

```python
def batch_segment(predictor, image, points_list, labels_list):
    """
    Batch processing for multiple prompts.

    Image embedding computed once, reused for all prompts.
    """
    predictor.set_image(image)  # Compute once

    results = []
    for points, labels in zip(points_list, labels_list):
        masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=len(points) == 1
        )

        # Select best mask
        if len(masks) > 1:
            best_mask = masks[scores.argmax()]
        else:
            best_mask = masks[0]

        results.append(best_mask)

    return results
```

### SAM 2 Video Mode

SAM 2 extends multi-mask to video with occlusion scores:

```python
def video_segment_with_occlusion(predictor, video_path, init_point):
    """
    Video segmentation with occlusion handling.

    SAM 2 generates occlusion score for each mask candidate.
    """
    state = predictor.init_state(video_path=video_path)

    # Initialize on first frame
    _, obj_ids, mask_logits = predictor.add_new_points(
        inference_state=state,
        frame_idx=0,
        obj_id=1,
        points=[init_point],
        labels=[1]
    )

    # Propagate through video
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        # SAM 2 provides occlusion scores
        # Use to handle temporary disappearance

        for i, obj_id in enumerate(obj_ids):
            mask = masks[i]
            # Low confidence might indicate occlusion
            # Track through occlusion using memory

        yield frame_idx, masks
```

---

## Section 7: ARR-COC Integration

### Multi-Mask for Attention Refinement

ARR-COC can leverage multi-mask output for attention analysis:

```python
class AttentionRefinementWithSAM:
    """
    Use SAM multi-mask for attention granularity analysis.

    Different mask levels correspond to different attention scales.
    """

    def __init__(self, sam_model, arr_model):
        self.sam = sam_model
        self.arr = arr_model
        self.predictor = SamPredictor(sam_model)

    def analyze_attention_granularity(self, image, attention_point):
        """
        Analyze what granularity level attention focuses on.
        """
        self.predictor.set_image(image)

        # Get 3 granularity levels
        masks, scores, _ = self.predictor.predict(
            point_coords=np.array([attention_point]),
            point_labels=np.array([1]),
            multimask_output=True
        )

        # Compare with ARR attention
        arr_attention = self.arr.get_attention_map(image)

        # Find which granularity matches best
        correlations = []
        for i, mask in enumerate(masks):
            corr = self.compute_correlation(
                mask.astype(float),
                arr_attention
            )
            correlations.append(corr)

        best_granularity = ['whole', 'part', 'subpart'][np.argmax(correlations)]

        return {
            'masks': masks,
            'scores': scores,
            'correlations': correlations,
            'best_match': best_granularity
        }
```

### IoU Scoring for Mask Quality

Use SAM's IoU prediction for ARR mask evaluation:

```python
class ARRMaskQualityEstimator:
    """
    Estimate ARR-generated mask quality using SAM's IoU predictor.
    """

    def estimate_mask_quality(self, arr_mask, image_embedding, sam_decoder):
        """
        Score an externally-generated mask using SAM.

        Useful for evaluating ARR attention-derived masks.
        """
        # Encode ARR mask as dense prompt
        mask_prompt = self.encode_mask_prompt(arr_mask)

        # Run through SAM decoder
        pred_masks, pred_ious = sam_decoder(
            image_embeddings=image_embedding,
            sparse_prompt_embeddings=None,
            dense_prompt_embeddings=mask_prompt,
            multimask_output=True
        )

        # Find closest SAM mask
        ious_with_input = []
        for sam_mask in pred_masks:
            iou = self.compute_iou(sam_mask, arr_mask)
            ious_with_input.append(iou)

        best_idx = np.argmax(ious_with_input)
        quality_estimate = pred_ious[best_idx]

        return {
            'quality': quality_estimate,
            'best_sam_mask': pred_masks[best_idx],
            'alignment_iou': ious_with_input[best_idx]
        }
```

### Hierarchical Loss for ARR Training

Leverage multi-mask hierarchy in ARR training:

```python
class HierarchicalAttentionLoss:
    """
    Train ARR with hierarchical mask supervision from SAM.
    """

    def compute_loss(self, arr_outputs, sam_masks, sam_scores):
        """
        Multi-scale loss using SAM granularity hierarchy.
        """
        # arr_outputs: multi-scale attention from ARR
        # sam_masks: [whole, part, subpart] from SAM

        losses = []

        for scale_idx, (arr_attention, sam_mask) in enumerate(
            zip(arr_outputs['multi_scale'], sam_masks)
        ):
            # Weight by SAM confidence
            weight = sam_scores[scale_idx]

            # Scale-specific loss
            scale_loss = self.focal_dice_loss(arr_attention, sam_mask)
            losses.append(weight * scale_loss)

        return sum(losses)
```

### Implementation Notes

1. **Multi-mask for data augmentation**: Generate 3 masks per annotation
2. **IoU prediction transfer**: Fine-tune SAM's IoU head on ARR masks
3. **Granularity control**: Use SAM levels to train multi-scale ARR
4. **Quality filtering**: Use SAM IoU to filter training samples

---

## Sources

**Primary Paper:**
- [Segment Anything](https://arxiv.org/abs/2304.02643) - Kirillov et al., ICCV 2023

**Web Research (accessed 2025-11-20):**
- [SAM: Segment Anything with Prompts, Not Labels](https://medium.com/@kdk199604/sam-segment-anything-with-prompts-not-labels-7a85e6ec4d09) - Dong-Keon Kim, Medium
- [Segment Anything: Explained](https://storrs.io/segment-anything-explained/) - Erik Storrs
- [SAM GitHub Issues - IoU Score](https://github.com/facebookresearch/segment-anything/issues/495)
- [Ultralytics SAM 2 Docs](https://docs.ultralytics.com/models/sam-2/)

**Additional References:**
- [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) - Ravi et al., 2024
- [Hugging Face SAM Documentation](https://huggingface.co/docs/transformers/en/model_doc/sam)
