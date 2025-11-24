# Loss Functions for Segmentation Training

**KNOWLEDGE-DROP**: SA-1B Dataset Mastery - Loss Functions for Training Segmentation Models
**Date**: 2025-11-20
**Source**: Web research on segmentation losses, SAM paper analysis
**Focus**: Loss function design for spatial grounding quality in ARR-COC

---

## 1. Overview: Why Loss Functions Matter for Segmentation

Loss functions are critical for training segmentation models like SAM on SA-1B. They guide the optimization process, handle class imbalance (foreground vs background pixels), and determine mask quality. SAM uses a **linear combination of Focal Loss and Dice Loss** with a 20:1 ratio, plus an auxiliary IoU prediction loss.

**Key challenges addressed:**
- Extreme class imbalance (foreground pixels << background pixels)
- Fine boundary precision
- Multi-scale mask quality
- Predicted IoU confidence calibration

From [SoftwareMill Instance Segmentation Loss Functions](https://softwaremill.com/instance-segmentation-loss-functions/) (accessed 2025-11-20):
> "Instance segmentation aims to generate a binary mask for each single object detected in the scene."

---

## 2. Binary Cross Entropy (BCE) Loss

**Foundation loss for pixel-wise classification.**

### Mathematical Formulation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def binary_cross_entropy_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Standard BCE loss for segmentation.

    Args:
        pred: Predicted logits [B, 1, H, W]
        target: Ground truth masks [B, 1, H, W]

    Returns:
        Scalar loss value
    """
    return F.binary_cross_entropy_with_logits(pred, target, reduction='mean')


# With class weighting for imbalance
def weighted_bce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = None
) -> torch.Tensor:
    """
    Weighted BCE to handle class imbalance.

    The pos_weight parameter increases loss for positive (foreground) pixels.
    For SA-1B where foreground << background, pos_weight > 1.

    Args:
        pred: Predicted logits [B, 1, H, W]
        target: Ground truth masks [B, 1, H, W]
        pos_weight: Weight for positive class (foreground)

    Returns:
        Scalar loss value
    """
    if pos_weight is None:
        # Compute from batch statistics
        num_pos = target.sum()
        num_neg = target.numel() - num_pos
        pos_weight = num_neg / (num_pos + 1e-8)

    pos_weight_tensor = torch.tensor([pos_weight], device=pred.device)

    return F.binary_cross_entropy_with_logits(
        pred, target,
        pos_weight=pos_weight_tensor,
        reduction='mean'
    )
```

### Characteristics

**Advantages:**
- Simple and well-understood
- Pixel-wise independence
- Smooth gradients

**Limitations:**
- Sensitive to class imbalance
- Treats all pixels equally (hard vs easy)
- No spatial awareness

From [Nature - Lumbar Spine Segmentation](https://www.nature.com/articles/s41598-025-20721-3) (accessed 2025-11-20):
> "BCE loss ensures pixel-wise classification, Dice loss emphasizes class imbalance handling by maximizing the overlap between ground truth and prediction."

---

## 3. Dice Loss

**Region-based loss that directly optimizes segmentation overlap.**

### Mathematical Formulation

```python
def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0
) -> torch.Tensor:
    """
    Dice loss for segmentation.

    Dice coefficient = 2 * |A ∩ B| / (|A| + |B|)
    Dice loss = 1 - Dice coefficient

    Args:
        pred: Predicted logits [B, 1, H, W]
        target: Ground truth masks [B, 1, H, W]
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Scalar loss value
    """
    pred_probs = torch.sigmoid(pred)

    # Flatten predictions and targets
    pred_flat = pred_probs.view(-1)
    target_flat = target.view(-1)

    # Compute Dice coefficient
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return 1.0 - dice


def soft_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    square_denominator: bool = False
) -> torch.Tensor:
    """
    Soft Dice loss with optional squared denominator.

    The squared denominator version reduces contribution of
    large regions, balancing between small and large objects.

    Args:
        pred: Predicted logits [B, 1, H, W]
        target: Ground truth masks [B, 1, H, W]
        smooth: Smoothing factor
        square_denominator: Use squared values in denominator

    Returns:
        Scalar loss value
    """
    pred_probs = torch.sigmoid(pred)

    # Per-sample computation
    batch_size = pred.shape[0]
    pred_flat = pred_probs.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)

    intersection = (pred_flat * target_flat).sum(dim=1)

    if square_denominator:
        denominator = (pred_flat ** 2).sum(dim=1) + (target_flat ** 2).sum(dim=1)
    else:
        denominator = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    return 1.0 - dice.mean()


def generalized_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0
) -> torch.Tensor:
    """
    Generalized Dice Loss for multi-class segmentation.

    Weights each class by inverse of its area, giving equal
    importance to small and large regions.

    Args:
        pred: Predicted logits [B, C, H, W] for C classes
        target: One-hot encoded masks [B, C, H, W]
        smooth: Smoothing factor

    Returns:
        Scalar loss value
    """
    pred_probs = torch.softmax(pred, dim=1)

    # Flatten spatial dimensions
    pred_flat = pred_probs.view(pred.shape[0], pred.shape[1], -1)
    target_flat = target.view(target.shape[0], target.shape[1], -1)

    # Compute class weights (inverse of area)
    class_weights = 1.0 / (target_flat.sum(dim=2) ** 2 + smooth)

    # Weighted intersection and union
    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

    weighted_intersection = (class_weights * intersection).sum(dim=1)
    weighted_union = (class_weights * union).sum(dim=1)

    dice = (2.0 * weighted_intersection + smooth) / (weighted_union + smooth)

    return 1.0 - dice.mean()
```

### Characteristics

**Advantages:**
- Handles class imbalance naturally
- Directly optimizes overlap metric
- Less sensitive to small object sizes

**Limitations:**
- Not pixel-wise (regional)
- Gradient can be unstable for very small objects
- Same loss for distant vs nearby misses

From [SoftwareMill](https://softwaremill.com/instance-segmentation-loss-functions/) (accessed 2025-11-20):
> "Dice loss is widely used in medical image segmentation tasks. It tackles the problem of class imbalance."

---

## 4. Focal Loss

**Addresses class imbalance AND hard example mining in one loss.**

### Mathematical Formulation

```python
def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0
) -> torch.Tensor:
    """
    Focal Loss for segmentation.

    Focal Loss = -alpha * (1 - pt)^gamma * log(pt)

    Where pt is the probability of the correct class.
    - alpha: Class balancing factor
    - gamma: Focusing parameter (higher = more focus on hard examples)

    Args:
        pred: Predicted logits [B, 1, H, W]
        target: Ground truth masks [B, 1, H, W]
        alpha: Weighting factor for positive class
        gamma: Focusing parameter

    Returns:
        Scalar loss value
    """
    pred_probs = torch.sigmoid(pred)

    # Compute pt (probability of correct class)
    pt = torch.where(target == 1, pred_probs, 1 - pred_probs)

    # Compute alpha_t
    alpha_t = torch.where(target == 1, alpha, 1 - alpha)

    # Focal loss
    focal_weight = (1 - pt) ** gamma
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

    loss = alpha_t * focal_weight * ce_loss

    return loss.mean()


def sigmoid_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Sigmoid Focal Loss (as used in RetinaNet and SAM).

    More numerically stable implementation using log-sum-exp.

    Args:
        pred: Predicted logits [B, 1, H, W]
        target: Ground truth masks [B, 1, H, W]
        alpha: Weighting factor
        gamma: Focusing parameter
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Loss tensor
    """
    p = torch.sigmoid(pred)
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    p_t = p * target + (1 - p) * (1 - target)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss
```

### Why SAM Uses Focal Loss

From [arXiv SAM 2++ Paper](https://arxiv.org/html/2510.18822v1) (accessed 2025-11-20):
> "For mask tracking, we combine focal and dice losses for mask prediction, L1 loss for IoU prediction."

**Focal Loss benefits for SA-1B:**
1. **Class imbalance**: Background >> foreground pixels
2. **Hard examples**: Edge pixels, ambiguous boundaries
3. **Multi-granularity**: Works across fine to coarse masks

---

## 5. IoU Loss and Generalized IoU (GIoU)

**Directly optimize Intersection over Union metric.**

### Mathematical Formulation

```python
def iou_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0
) -> torch.Tensor:
    """
    IoU Loss for segmentation.

    IoU = |A ∩ B| / |A ∪ B|
    Loss = 1 - IoU

    Args:
        pred: Predicted logits [B, 1, H, W]
        target: Ground truth masks [B, 1, H, W]
        smooth: Smoothing factor

    Returns:
        Scalar loss value
    """
    pred_probs = torch.sigmoid(pred)

    # Flatten
    pred_flat = pred_probs.view(-1)
    target_flat = target.view(-1)

    # Intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return 1.0 - iou


def lovasz_hinge_loss(
    pred: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Lovász-Softmax loss for IoU optimization.

    Direct optimization of IoU using Lovász extension.
    Better gradients than standard IoU loss.

    Args:
        pred: Predicted logits [B, 1, H, W]
        target: Ground truth masks [B, 1, H, W]

    Returns:
        Scalar loss value
    """
    def lovasz_grad(gt_sorted):
        """Compute Lovász gradient."""
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if len(googles) > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard

    # Per-sample computation
    losses = []
    for pred_i, target_i in zip(pred, target):
        pred_flat = pred_i.view(-1)
        target_flat = target_i.view(-1)

        signs = 2 * target_flat - 1
        errors = 1 - pred_flat * signs
        errors_sorted, perm = torch.sort(errors, descending=True)
        target_sorted = target_flat[perm]
        grad = lovasz_grad(target_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        losses.append(loss)

    return torch.stack(losses).mean()
```

### Difference Between IoU and Dice

| Metric | Formula | Denominator |
|--------|---------|-------------|
| IoU | 2*intersection / union | A ∪ B |
| Dice | 2*intersection / (A + B) | A + B |

Dice >= IoU always, with equality when A = B.

---

## 6. SAM Combined Loss: Focal + Dice

**SAM's actual training loss with 20:1 weighting.**

### Implementation

```python
class SAMSegmentationLoss(nn.Module):
    """
    Combined loss function used by SAM for mask prediction.

    Loss = focal_weight * focal_loss + dice_weight * dice_loss

    SAM uses focal_weight=20, dice_weight=1
    """

    def __init__(
        self,
        focal_weight: float = 20.0,
        dice_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> dict:
        """
        Compute combined SAM loss.

        Args:
            pred: Predicted mask logits [B, 1, H, W]
            target: Ground truth masks [B, 1, H, W]

        Returns:
            Dictionary with total loss and components
        """
        # Focal loss
        focal = sigmoid_focal_loss(
            pred, target,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma
        )

        # Dice loss
        dice = dice_loss(pred, target)

        # Combined loss
        total = self.focal_weight * focal + self.dice_weight * dice

        return {
            'loss': total,
            'focal_loss': focal,
            'dice_loss': dice
        }


class SAMFullLoss(nn.Module):
    """
    Complete SAM loss including IoU prediction head.

    Total Loss = mask_loss + iou_loss

    Where:
    - mask_loss = 20 * focal + 1 * dice
    - iou_loss = MSE(predicted_iou, actual_iou)
    """

    def __init__(
        self,
        focal_weight: float = 20.0,
        dice_weight: float = 1.0,
        iou_weight: float = 1.0
    ):
        super().__init__()
        self.mask_loss = SAMSegmentationLoss(focal_weight, dice_weight)
        self.iou_weight = iou_weight

    def forward(
        self,
        pred_masks: torch.Tensor,
        pred_iou: torch.Tensor,
        target_masks: torch.Tensor
    ) -> dict:
        """
        Compute full SAM loss.

        Args:
            pred_masks: Predicted mask logits [B, N, H, W]
            pred_iou: Predicted IoU scores [B, N]
            target_masks: Ground truth masks [B, N, H, W]

        Returns:
            Dictionary with all loss components
        """
        # Mask loss
        mask_dict = self.mask_loss(pred_masks, target_masks)

        # Compute actual IoU for supervision
        pred_probs = torch.sigmoid(pred_masks)
        intersection = (pred_probs * target_masks).sum(dim=(-2, -1))
        union = pred_probs.sum(dim=(-2, -1)) + target_masks.sum(dim=(-2, -1)) - intersection
        actual_iou = intersection / (union + 1e-8)

        # IoU prediction loss (MSE)
        iou_loss = F.mse_loss(pred_iou, actual_iou)

        # Total loss
        total = mask_dict['loss'] + self.iou_weight * iou_loss

        return {
            'loss': total,
            'mask_loss': mask_dict['loss'],
            'focal_loss': mask_dict['focal_loss'],
            'dice_loss': mask_dict['dice_loss'],
            'iou_loss': iou_loss
        }
```

### Why 20:1 Ratio?

From [Notion - SAM Analysis](https://modulabs.notion.site/SAM-ce92ac6c771d44ddb1b09eb0a8651731) (accessed 2025-11-20):
> "We supervise mask prediction with the linear combination of focal loss and dice loss. Focal Loss is a variant of BCE, it enables the model to focus on learning hard examples."

**Reasoning:**
- Focal loss values are typically smaller (range ~0.01-0.1)
- Dice loss values are larger (range ~0.1-0.5)
- 20:1 ratio balances their contributions
- Focal handles pixel-wise learning, Dice handles regional overlap

---

## 7. Predicted IoU Auxiliary Loss

**Trains SAM to predict its own mask quality.**

### Purpose and Implementation

```python
class IoUPredictionLoss(nn.Module):
    """
    Auxiliary loss for IoU prediction head.

    SAM predicts an IoU score for each mask, which is used
    to rank masks when multiple are generated. This loss
    trains the model to accurately estimate its own performance.
    """

    def __init__(self, loss_type: str = 'mse'):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        pred_iou: torch.Tensor,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute IoU prediction loss.

        Args:
            pred_iou: Model's predicted IoU [B, N]
            pred_masks: Predicted mask logits [B, N, H, W]
            target_masks: Ground truth masks [B, N, H, W]

        Returns:
            Scalar loss value
        """
        # Compute actual IoU
        with torch.no_grad():
            pred_probs = torch.sigmoid(pred_masks)

            # Threshold for binary mask
            pred_binary = (pred_probs > 0.5).float()

            intersection = (pred_binary * target_masks).sum(dim=(-2, -1))
            union = pred_binary.sum(dim=(-2, -1)) + target_masks.sum(dim=(-2, -1)) - intersection
            actual_iou = intersection / (union + 1e-8)

        # Loss between predicted and actual IoU
        if self.loss_type == 'mse':
            return F.mse_loss(pred_iou, actual_iou)
        elif self.loss_type == 'l1':
            return F.l1_loss(pred_iou, actual_iou)
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss(pred_iou, actual_iou)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


def compute_mask_stability_score(
    mask_logits: torch.Tensor,
    threshold_offset: float = 1.0
) -> torch.Tensor:
    """
    Compute stability score for mask quality assessment.

    Measures how stable the mask is under threshold perturbation.
    Used alongside predicted IoU for mask ranking.

    Args:
        mask_logits: Predicted mask logits [B, 1, H, W]
        threshold_offset: Amount to shift threshold

    Returns:
        Stability scores [B]
    """
    # Masks at different thresholds
    mask_high = (mask_logits > threshold_offset).float()
    mask_low = (mask_logits > -threshold_offset).float()

    # Compute IoU between high and low threshold masks
    intersection = (mask_high * mask_low).sum(dim=(-2, -1))
    union = mask_high.sum(dim=(-2, -1)) + mask_low.sum(dim=(-2, -1)) - intersection

    stability = intersection / (union + 1e-8)

    return stability.squeeze()
```

### Why Predicted IoU Matters

1. **Mask ranking**: When SAM generates multiple masks, predicted IoU ranks them
2. **Confidence calibration**: Users know which masks to trust
3. **Automatic selection**: System can auto-select best mask
4. **Quality estimation**: No need for ground truth at inference

---

## 8. Loss Weighting Strategies

**How to balance multiple loss components.**

### Adaptive Loss Weighting

```python
class AdaptiveLossWeighting(nn.Module):
    """
    Learnable loss weighting using uncertainty.

    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses"
    Each task gets a learnable weight based on homoscedastic uncertainty.
    """

    def __init__(self, num_losses: int):
        super().__init__()
        # Log variance for numerical stability
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses: list) -> torch.Tensor:
        """
        Compute weighted sum of losses.

        Args:
            losses: List of individual loss tensors

        Returns:
            Weighted total loss
        """
        total = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total += precision * loss + self.log_vars[i]

        return total


class GradientNormalizationWeighting:
    """
    Gradient normalization for loss balancing.

    Adjusts weights so all losses contribute equally
    to the gradient magnitude.
    """

    def __init__(self, num_losses: int, alpha: float = 0.16):
        self.num_losses = num_losses
        self.alpha = alpha
        self.initial_losses = None
        self.weights = [1.0] * num_losses

    def update_weights(self, losses: list):
        """
        Update loss weights based on gradient ratios.

        Args:
            losses: List of current loss values
        """
        if self.initial_losses is None:
            self.initial_losses = [l.item() for l in losses]
            return

        # Compute relative inverse training rates
        loss_ratios = []
        for i, (current, initial) in enumerate(zip(losses, self.initial_losses)):
            ratio = current.item() / (initial + 1e-8)
            loss_ratios.append(ratio)

        # Compute weights
        mean_ratio = sum(loss_ratios) / len(loss_ratios)
        for i in range(self.num_losses):
            self.weights[i] = (loss_ratios[i] / mean_ratio) ** self.alpha

        # Normalize
        weight_sum = sum(self.weights)
        self.weights = [w * self.num_losses / weight_sum for w in self.weights]
```

### Dynamic Weighting Schedule

```python
class DynamicLossScheduler:
    """
    Schedule loss weights during training.

    Common patterns:
    - Linear warmup for auxiliary losses
    - Curriculum: start with easy loss, add hard later
    - Cosine annealing for smooth transitions
    """

    def __init__(
        self,
        total_steps: int,
        focal_range: tuple = (10.0, 20.0),
        dice_range: tuple = (1.0, 1.0),
        iou_range: tuple = (0.0, 1.0),
        warmup_steps: int = 1000
    ):
        self.total_steps = total_steps
        self.focal_range = focal_range
        self.dice_range = dice_range
        self.iou_range = iou_range
        self.warmup_steps = warmup_steps

    def get_weights(self, step: int) -> dict:
        """
        Get loss weights for current step.

        Args:
            step: Current training step

        Returns:
            Dictionary of loss weights
        """
        # Warmup factor
        warmup = min(1.0, step / self.warmup_steps)

        # Progress factor (after warmup)
        progress = min(1.0, (step - self.warmup_steps) /
                       (self.total_steps - self.warmup_steps))
        progress = max(0.0, progress)

        # Interpolate weights
        focal = self.focal_range[0] + progress * (self.focal_range[1] - self.focal_range[0])
        dice = self.dice_range[0] + progress * (self.dice_range[1] - self.dice_range[0])
        iou = warmup * (self.iou_range[0] + progress * (self.iou_range[1] - self.iou_range[0]))

        return {
            'focal_weight': focal,
            'dice_weight': dice,
            'iou_weight': iou
        }
```

---

## 9. Gradient Analysis and Optimization

**Understanding how gradients flow through different losses.**

### Gradient Characteristics

```python
def analyze_loss_gradients(
    model: nn.Module,
    loss_fn: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> dict:
    """
    Analyze gradient statistics for different loss components.

    Args:
        model: Segmentation model
        loss_fn: Loss function returning dict of losses
        dataloader: Data loader
        device: Device for computation

    Returns:
        Dictionary of gradient statistics
    """
    grad_stats = {
        'focal': {'mean': [], 'std': [], 'max': []},
        'dice': {'mean': [], 'std': [], 'max': []},
        'iou': {'mean': [], 'std': [], 'max': []}
    }

    model.train()

    for batch in dataloader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        # Compute losses
        pred_masks, pred_iou = model(images)
        losses = loss_fn(pred_masks, pred_iou, masks)

        # Analyze gradients for each loss
        for loss_name in ['focal', 'dice', 'iou']:
            model.zero_grad()
            losses[f'{loss_name}_loss'].backward(retain_graph=True)

            # Collect gradient statistics
            grads = []
            for param in model.parameters():
                if param.grad is not None:
                    grads.append(param.grad.view(-1))

            all_grads = torch.cat(grads)
            grad_stats[loss_name]['mean'].append(all_grads.mean().item())
            grad_stats[loss_name]['std'].append(all_grads.std().item())
            grad_stats[loss_name]['max'].append(all_grads.abs().max().item())

    # Average statistics
    for loss_name in grad_stats:
        for stat in grad_stats[loss_name]:
            grad_stats[loss_name][stat] = sum(grad_stats[loss_name][stat]) / len(grad_stats[loss_name][stat])

    return grad_stats


# Gradient clipping for stability
def train_step_with_gradient_analysis(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    batch: dict,
    max_grad_norm: float = 1.0
) -> dict:
    """
    Training step with gradient clipping and analysis.

    Args:
        model: Segmentation model
        optimizer: Optimizer
        loss_fn: Loss function
        batch: Data batch
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Dictionary with loss and gradient info
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass
    pred_masks, pred_iou = model(batch['image'])
    losses = loss_fn(pred_masks, pred_iou, batch['mask'])

    # Backward pass
    losses['loss'].backward()

    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_grad_norm
    )

    # Optimizer step
    optimizer.step()

    return {
        **{k: v.item() for k, v in losses.items()},
        'grad_norm': grad_norm.item()
    }
```

---

## 10. Boundary Loss (Advanced)

**Distance-based loss for precise boundary learning.**

### Implementation

```python
from scipy.ndimage import distance_transform_edt

def compute_distance_map(mask: np.ndarray) -> np.ndarray:
    """
    Compute signed distance transform for boundary loss.

    Positive inside mask, negative outside.

    Args:
        mask: Binary mask [H, W]

    Returns:
        Signed distance map [H, W]
    """
    # Distance transform for foreground
    dist_inside = distance_transform_edt(mask)

    # Distance transform for background
    dist_outside = distance_transform_edt(1 - mask)

    # Signed distance (negative inside, positive outside)
    signed_dist = dist_outside - dist_inside

    return signed_dist


class BoundaryLoss(nn.Module):
    """
    Boundary loss using distance transform.

    Minimized when predicted mask aligns with GT boundary.
    Better for fine boundary learning than region-based losses.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        dist_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute boundary loss.

        Args:
            pred: Predicted mask probabilities [B, 1, H, W]
            dist_map: Pre-computed distance maps [B, 1, H, W]

        Returns:
            Scalar loss value
        """
        # Element-wise product
        boundary_loss = pred * dist_map

        return boundary_loss.mean()


class CombinedRegionBoundaryLoss(nn.Module):
    """
    Combined region (Dice) and boundary loss.

    Boundary loss helps early training when masks may not overlap.
    Dice loss takes over as training progresses.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        boundary_weight: float = 1.0,
        schedule_boundary: bool = True
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.schedule_boundary = schedule_boundary
        self.current_epoch = 0

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        dist_map: torch.Tensor
    ) -> dict:
        """
        Compute combined loss.

        Args:
            pred: Predicted logits [B, 1, H, W]
            target: Ground truth masks [B, 1, H, W]
            dist_map: Distance maps [B, 1, H, W]

        Returns:
            Dictionary of losses
        """
        pred_probs = torch.sigmoid(pred)

        # Dice loss
        dice = dice_loss(pred, target)

        # Boundary loss
        boundary = (pred_probs * dist_map).mean()

        # Schedule boundary weight (decrease over epochs)
        if self.schedule_boundary:
            boundary_w = self.boundary_weight * (1 - self.current_epoch / 100)
            boundary_w = max(0.0, boundary_w)
        else:
            boundary_w = self.boundary_weight

        total = self.dice_weight * dice + boundary_w * boundary

        return {
            'loss': total,
            'dice_loss': dice,
            'boundary_loss': boundary
        }
```

From [SoftwareMill](https://softwaremill.com/instance-segmentation-loss-functions/) (accessed 2025-11-20):
> "Dice or cross-entropy are based on integrals over the segmentation regions. Unfortunately, for highly imbalanced segmentations, such regional summations have values that differ by several orders of magnitude across classes."

---

## 11. ARR-COC-0-1 Integration (10%): Loss Functions for Spatial Relevance Quality

### Application to Vision-Language Grounding

Loss function design principles from SA-1B segmentation training directly apply to ARR-COC's spatial relevance realization:

**1. Combined Loss Strategy**
```python
class SpatialRelevanceLoss(nn.Module):
    """
    Loss for training spatial grounding in VLM.

    Combines:
    - Segmentation loss (focal + dice) for mask quality
    - Language alignment loss for text-region matching
    - IoU prediction for confidence calibration
    """

    def __init__(self):
        super().__init__()
        self.mask_loss = SAMSegmentationLoss(focal_weight=20.0, dice_weight=1.0)
        self.iou_loss = IoUPredictionLoss()
        self.alignment_weight = 1.0

    def forward(
        self,
        pred_masks: torch.Tensor,
        pred_iou: torch.Tensor,
        target_masks: torch.Tensor,
        text_region_alignment: torch.Tensor,
        text_region_target: torch.Tensor
    ) -> dict:
        """
        Compute spatial relevance loss.

        Ensures VLM learns to:
        1. Generate accurate spatial masks
        2. Align text descriptions with regions
        3. Predict reliable confidence scores
        """
        # Mask quality loss
        mask_dict = self.mask_loss(pred_masks, target_masks)

        # IoU prediction loss
        iou = self.iou_loss(pred_iou, pred_masks, target_masks)

        # Text-region alignment (contrastive or BCE)
        alignment = F.binary_cross_entropy_with_logits(
            text_region_alignment, text_region_target
        )

        total = mask_dict['loss'] + iou + self.alignment_weight * alignment

        return {
            'loss': total,
            'mask_loss': mask_dict['loss'],
            'iou_loss': iou,
            'alignment_loss': alignment
        }
```

**2. Multi-Granularity Support**
```python
def multi_granularity_loss(
    pred_masks: list,  # [fine, medium, coarse]
    target_masks: list,
    granularity_weights: list = [1.0, 1.0, 1.0]
) -> torch.Tensor:
    """
    Loss supporting multiple mask granularities.

    SA-1B contains masks at all granularities (door handles to buildings).
    ARR-COC needs to ground at appropriate granularity for each query.
    """
    total = 0
    for pred, target, weight in zip(pred_masks, target_masks, granularity_weights):
        mask_loss = SAMSegmentationLoss()(pred, target)['loss']
        total += weight * mask_loss

    return total / len(pred_masks)
```

**3. Key Design Principles for ARR-COC**

| SA-1B Lesson | ARR-COC Application |
|--------------|---------------------|
| 20:1 focal:dice ratio | Balance pixel-wise and regional losses |
| Predicted IoU | Confidence scoring for grounding quality |
| Boundary loss | Precise spatial boundaries for relevance |
| Adaptive weighting | Balance segmentation vs language alignment |

---

## Sources

**Web Research:**
- [SoftwareMill - Instance Segmentation Loss Functions](https://softwaremill.com/instance-segmentation-loss-functions/) (accessed 2025-11-20)
- [Nature - Lumbar Spine Segmentation](https://www.nature.com/articles/s41598-025-20721-3) (accessed 2025-11-20)
- [NIH PMC - Unified Focal Loss](https://pmc.ncbi.nlm.nih.gov/articles/PMC8785124/) (accessed 2025-11-20)
- [arXiv - SAM 2++ Tracking](https://arxiv.org/html/2510.18822v1) (accessed 2025-11-20)
- [Notion - SAM Analysis](https://modulabs.notion.site/SAM-ce92ac6c771d44ddb1b09eb0a8651731) (accessed 2025-11-20)
- [Medium - SAM 2 Papers Explained](https://ritvik19.medium.com/papers-explained-239-sam-2-6ffb7f187281) (accessed 2025-11-20)
- [SpringerOpen - Crack SAM](https://jipr.springeropen.com/articles/10.1186/s43065-024-00103-1) (accessed 2025-11-20)

**Original Papers:**
- Lin et al., "Focal Loss for Dense Object Detection" (RetinaNet, 2017)
- Milletari et al., "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation" (Dice Loss, 2016)
- Kervadec et al., "Boundary loss for highly unbalanced segmentation" (2019)
- Kirillov et al., "Segment Anything" (SAM Loss Design, 2023)

**Source Document:**
- SA-1B Dataset Mastery ingestion plan
