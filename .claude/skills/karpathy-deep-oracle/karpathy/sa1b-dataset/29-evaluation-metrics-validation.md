# Evaluation Metrics & Validation for SA-1B Segmentation

## Overview

Evaluating segmentation models trained on SA-1B requires comprehensive metrics that capture both region overlap and boundary accuracy. This knowledge drop covers IoU, mIoU, Dice coefficient, pixel accuracy, boundary F-measure, and SA-1B-specific metrics like predicted_iou correlation and stability score validation.

**Key Challenge**: SA-1B's class-agnostic, multi-granular masks require evaluation metrics that work across varying object sizes (door handles to buildings) without semantic class labels.

---

## Section 1: Intersection over Union (IoU / Jaccard Index)

### Definition

IoU measures the overlap between predicted and ground truth masks:

```
IoU = |Prediction ∩ Ground Truth| / |Prediction ∪ Ground Truth|
    = TP / (TP + FP + FN)
```

Where:
- TP = True Positives (correct foreground)
- FP = False Positives (predicted foreground, actually background)
- FN = False Negatives (predicted background, actually foreground)

From [Medium - Medical Image Segmentation](https://medium.com/mastering-data-science/understanding-evaluation-metrics-in-medical-image-segmentation-d289a373a3f):
> "Dice coefficient and IoU are the most commonly used metrics for semantic segmentation because both metrics penalize false positives."

### PyTorch Implementation

```python
import torch
import numpy as np

def iou_score(pred: torch.Tensor, target: torch.Tensor,
              smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate IoU (Jaccard Index) for binary segmentation.

    Args:
        pred: Predicted mask (N, H, W) or (N, 1, H, W)
        target: Ground truth mask (N, H, W) or (N, 1, H, W)
        smooth: Smoothing factor to avoid division by zero

    Returns:
        IoU score per sample
    """
    # Ensure binary predictions
    pred = (pred > 0.5).float()
    target = target.float()

    # Flatten spatial dimensions
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

    # IoU
    iou = (intersection + smooth) / (union + smooth)

    return iou

def iou_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    """NumPy implementation of IoU."""
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union
```

### IoU Thresholds

Common IoU thresholds for evaluation:

| Threshold | Use Case |
|-----------|----------|
| IoU > 0.5 | Standard detection (COCO) |
| IoU > 0.75 | Strict evaluation |
| mAP@[0.5:0.95] | COCO-style averaging |

```python
def compute_iou_at_thresholds(pred, target, thresholds=[0.5, 0.75, 0.9]):
    """Compute IoU matches at multiple thresholds."""
    iou = iou_numpy(pred, target)

    results = {}
    for thresh in thresholds:
        results[f'IoU@{thresh}'] = 1.0 if iou >= thresh else 0.0
        results['IoU'] = iou

    return results
```

---

## Section 2: Mean IoU (mIoU)

### Definition

mIoU averages IoU across all classes (or masks for SA-1B):

```
mIoU = (1/C) * Σ IoU_c
```

For SA-1B's class-agnostic setting, mIoU averages across all masks in a sample.

From [Tencent Cloud TechPedia](https://www.tencentcloud.com/techpedia/112106):
> "In autonomous driving, mIoU assesses road, pedestrian, and vehicle segmentation performance."

### Implementation

```python
def mean_iou(pred_masks: list, target_masks: list) -> float:
    """
    Calculate mean IoU across multiple masks.

    Args:
        pred_masks: List of predicted binary masks
        target_masks: List of ground truth binary masks

    Returns:
        Mean IoU score
    """
    if len(pred_masks) != len(target_masks):
        raise ValueError("Number of masks must match")

    if len(pred_masks) == 0:
        return 0.0

    ious = []
    for pred, target in zip(pred_masks, target_masks):
        iou = iou_numpy(pred, target)
        ious.append(iou)

    return np.mean(ious)

class MeanIoUMeter:
    """Track mIoU across batches."""

    def __init__(self, num_classes=None):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        if self.num_classes:
            self.ious = {c: [] for c in range(self.num_classes)}
        else:
            self.ious = []

    def update(self, pred, target, class_id=None):
        iou = iou_numpy(pred, target)

        if self.num_classes and class_id is not None:
            self.ious[class_id].append(iou)
        else:
            self.ious.append(iou)

    def compute(self):
        if self.num_classes:
            class_ious = []
            for c in range(self.num_classes):
                if self.ious[c]:
                    class_ious.append(np.mean(self.ious[c]))
            return np.mean(class_ious) if class_ious else 0.0
        else:
            return np.mean(self.ious) if self.ious else 0.0
```

### Weighted mIoU

For datasets with class imbalance:

```python
def weighted_miou(ious: list, weights: list) -> float:
    """
    Weighted mIoU for imbalanced datasets.

    Args:
        ious: IoU scores per class
        weights: Class weights (e.g., by frequency)

    Returns:
        Weighted mIoU
    """
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize

    return np.sum(np.array(ious) * weights)
```

---

## Section 3: Dice Coefficient (F1 Score)

### Definition

The Dice coefficient (also known as Dice-Sorensen coefficient or F1 score for segmentation):

```
Dice = 2 * |Prediction ∩ Ground Truth| / (|Prediction| + |Ground Truth|)
     = 2 * TP / (2 * TP + FP + FN)
```

From [VeRADP AI](https://veradp-ai.com/iou-vs-dice/):
> "Dice and IoU are essential metrics for evaluating segmentation performance, each with its strengths and limitations."

### Relationship to IoU

```
Dice = 2 * IoU / (1 + IoU)
IoU = Dice / (2 - Dice)
```

### Implementation

```python
def dice_coefficient(pred: torch.Tensor, target: torch.Tensor,
                     smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate Dice coefficient for binary segmentation.

    Args:
        pred: Predicted mask
        target: Ground truth mask
        smooth: Smoothing factor

    Returns:
        Dice score
    """
    pred = (pred > 0.5).float()
    target = target.float()

    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()

    dice = (2 * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )

    return dice

def dice_loss(pred: torch.Tensor, target: torch.Tensor,
              smooth: float = 1e-6) -> torch.Tensor:
    """Dice loss for training (1 - Dice)."""
    return 1 - dice_coefficient(pred, target, smooth)

class SoftDiceLoss(torch.nn.Module):
    """Soft Dice loss for end-to-end training."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Apply sigmoid for soft predictions
        pred = torch.sigmoid(pred)

        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )

        return 1 - dice
```

### Dice vs IoU Comparison

| Aspect | Dice | IoU |
|--------|------|-----|
| Range | [0, 1] | [0, 1] |
| Sensitivity | Less sensitive to small regions | More sensitive |
| Loss function | Smoother gradients | Sharper penalties |
| Interpretability | F1 score analogy | Overlap intuition |

---

## Section 4: Pixel Accuracy

### Definition

Pixel accuracy measures the percentage of correctly classified pixels:

```
Pixel Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

From [GeeksforGeeks](https://www.geeksforgeeks.org/computer-vision/what-are-different-evaluation-metrics-used-to-evaluate-image-segmentation-models/):
> "Various metrics exist to measure performance, each focusing on different aspects like overlap and boundary accuracy."

### Implementation

```python
def pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate pixel accuracy.

    Args:
        pred: Predicted mask (H, W)
        target: Ground truth mask (H, W)

    Returns:
        Pixel accuracy score
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    correct = (pred == target).sum()
    total = pred.size

    return correct / total

def mean_pixel_accuracy(pred: np.ndarray, target: np.ndarray,
                        num_classes: int = 2) -> float:
    """
    Mean pixel accuracy across classes.

    Args:
        pred: Predicted class labels (H, W)
        target: Ground truth class labels (H, W)
        num_classes: Number of classes

    Returns:
        Mean pixel accuracy
    """
    class_accuracies = []

    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c

        if target_c.sum() > 0:
            acc = (pred_c & target_c).sum() / target_c.sum()
            class_accuracies.append(acc)

    return np.mean(class_accuracies) if class_accuracies else 0.0
```

### Limitations

Pixel accuracy can be misleading for imbalanced segmentation:

```python
def demonstrate_pixel_accuracy_limitation():
    """Show why pixel accuracy alone is insufficient."""
    # Example: 1% of image is foreground
    h, w = 100, 100
    target = np.zeros((h, w), dtype=bool)
    target[45:55, 45:55] = True  # Small 10x10 foreground

    # Prediction 1: All background (terrible segmentation)
    pred_all_bg = np.zeros((h, w), dtype=bool)

    # Prediction 2: Perfect segmentation
    pred_perfect = target.copy()

    # Calculate metrics
    acc_all_bg = pixel_accuracy(pred_all_bg, target)
    acc_perfect = pixel_accuracy(pred_perfect, target)

    iou_all_bg = iou_numpy(pred_all_bg, target)
    iou_perfect = iou_numpy(pred_perfect, target)

    print(f"All background - Pixel Acc: {acc_all_bg:.2f}, IoU: {iou_all_bg:.2f}")
    # Output: Pixel Acc: 0.99, IoU: 0.00

    print(f"Perfect pred   - Pixel Acc: {acc_perfect:.2f}, IoU: {iou_perfect:.2f}")
    # Output: Pixel Acc: 1.00, IoU: 1.00
```

---

## Section 5: Boundary F-Measure

### Definition

The Boundary F-measure evaluates how well predicted boundaries align with ground truth:

From [CVPR 2021 Paper - Boundary IoU](https://openaccess.thecvf.com/content/CVPR2021/papers/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.pdf):
> "Trimap and F-measure are often used to evaluate boundary quality for semantic segmentation tasks."

### Implementation

```python
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial.distance import cdist

def get_boundary(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    """Extract boundary from binary mask."""
    eroded = binary_erosion(mask, iterations=thickness)
    boundary = mask ^ eroded  # XOR to get boundary
    return boundary

def boundary_f_measure(pred: np.ndarray, target: np.ndarray,
                       threshold: float = 2.0) -> float:
    """
    Calculate boundary F-measure.

    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        threshold: Distance threshold in pixels

    Returns:
        Boundary F-measure score
    """
    # Get boundaries
    pred_boundary = get_boundary(pred)
    target_boundary = get_boundary(target)

    # Get boundary pixel coordinates
    pred_coords = np.argwhere(pred_boundary)
    target_coords = np.argwhere(target_boundary)

    if len(pred_coords) == 0 and len(target_coords) == 0:
        return 1.0
    if len(pred_coords) == 0 or len(target_coords) == 0:
        return 0.0

    # Calculate distances
    dist_pred_to_target = cdist(pred_coords, target_coords, 'euclidean')
    dist_target_to_pred = cdist(target_coords, pred_coords, 'euclidean')

    # Precision: predicted boundary pixels close to ground truth
    min_dist_pred = dist_pred_to_target.min(axis=1)
    precision = (min_dist_pred <= threshold).sum() / len(pred_coords)

    # Recall: ground truth boundary pixels close to prediction
    min_dist_target = dist_target_to_pred.min(axis=1)
    recall = (min_dist_target <= threshold).sum() / len(target_coords)

    # F-measure
    if precision + recall == 0:
        return 0.0

    f_measure = 2 * precision * recall / (precision + recall)
    return f_measure

def boundary_iou(pred: np.ndarray, target: np.ndarray,
                 dilation: int = 1) -> float:
    """
    Calculate Boundary IoU (trimap-based).

    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        dilation: Dilation for boundary region

    Returns:
        Boundary IoU score
    """
    # Get boundary regions
    pred_boundary = get_boundary(pred, dilation)
    target_boundary = get_boundary(target, dilation)

    # Expand to trimap region
    pred_trimap = binary_dilation(pred_boundary, iterations=dilation)
    target_trimap = binary_dilation(target_boundary, iterations=dilation)

    # Combined boundary region
    boundary_region = pred_trimap | target_trimap

    # Calculate IoU within boundary region
    pred_in_boundary = pred & boundary_region
    target_in_boundary = target & boundary_region

    intersection = (pred_in_boundary & target_in_boundary).sum()
    union = (pred_in_boundary | target_in_boundary).sum()

    if union == 0:
        return 1.0

    return intersection / union
```

### Multi-Scale Boundary Evaluation

```python
def multiscale_boundary_f_measure(pred: np.ndarray, target: np.ndarray,
                                   thresholds: list = [1, 2, 5]) -> dict:
    """
    Evaluate boundary accuracy at multiple scales.

    Args:
        pred: Predicted mask
        target: Ground truth mask
        thresholds: Distance thresholds in pixels

    Returns:
        F-measures at each threshold
    """
    results = {}
    for thresh in thresholds:
        f_score = boundary_f_measure(pred, target, threshold=thresh)
        results[f'BF@{thresh}px'] = f_score

    return results
```

---

## Section 6: SA-1B Specific Metrics

### Predicted IoU Correlation

SA-1B includes `predicted_iou` for each mask - the model's confidence in mask quality. Evaluating correlation with actual IoU measures model calibration:

```python
from scipy.stats import pearsonr, spearmanr

def evaluate_predicted_iou(masks: list, predictions: list) -> dict:
    """
    Evaluate correlation between predicted_iou and actual IoU.

    Args:
        masks: List of dicts with 'predicted_iou' and 'segmentation'
        predictions: List of predicted masks

    Returns:
        Correlation metrics
    """
    predicted_ious = []
    actual_ious = []

    for mask_info, pred in zip(masks, predictions):
        predicted_iou = mask_info['predicted_iou']
        target = decode_rle(mask_info['segmentation'])
        actual_iou = iou_numpy(pred, target)

        predicted_ious.append(predicted_iou)
        actual_ious.append(actual_iou)

    predicted_ious = np.array(predicted_ious)
    actual_ious = np.array(actual_ious)

    # Calculate correlations
    pearson_r, pearson_p = pearsonr(predicted_ious, actual_ious)
    spearman_r, spearman_p = spearmanr(predicted_ious, actual_ious)

    # Mean absolute error
    mae = np.mean(np.abs(predicted_ious - actual_ious))

    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'mae': mae,
        'predicted_mean': predicted_ious.mean(),
        'actual_mean': actual_ious.mean(),
    }

def calibration_plot(predicted_ious: np.ndarray, actual_ious: np.ndarray):
    """Plot predicted vs actual IoU for calibration analysis."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.scatter(predicted_ious, actual_ious, alpha=0.3, s=10)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')

    plt.xlabel('Predicted IoU')
    plt.ylabel('Actual IoU')
    plt.title('Model Calibration')
    plt.legend()

    return plt.gcf()
```

### Stability Score Validation

SA-1B's `stability_score` measures mask consistency under input perturbations:

```python
def evaluate_stability_score(model, images: list, masks: list,
                             perturbation_range: float = 0.1) -> dict:
    """
    Validate stability scores by measuring mask consistency.

    Args:
        model: Segmentation model
        images: Input images
        masks: Mask annotations with 'stability_score'
        perturbation_range: Range for point perturbation

    Returns:
        Stability validation metrics
    """
    reported_stabilities = []
    measured_stabilities = []

    for image, mask_info in zip(images, masks):
        # Get original mask
        original_mask = model.predict(
            image,
            point_coords=mask_info['point_coords'],
            point_labels=mask_info.get('point_labels', [1])
        )

        # Perturb input and measure consistency
        stabilities = []
        for _ in range(10):  # Multiple perturbations
            # Perturb point coordinates
            perturbed_coords = mask_info['point_coords'] + \
                np.random.uniform(-perturbation_range, perturbation_range,
                                  mask_info['point_coords'].shape)

            perturbed_mask = model.predict(
                image,
                point_coords=perturbed_coords,
                point_labels=mask_info.get('point_labels', [1])
            )

            # Measure IoU between original and perturbed
            stability = iou_numpy(original_mask, perturbed_mask)
            stabilities.append(stability)

        # Measured stability is mean IoU across perturbations
        measured_stability = np.mean(stabilities)

        reported_stabilities.append(mask_info['stability_score'])
        measured_stabilities.append(measured_stability)

    # Correlation analysis
    correlation, p_value = pearsonr(reported_stabilities, measured_stabilities)

    return {
        'correlation': correlation,
        'p_value': p_value,
        'mean_reported': np.mean(reported_stabilities),
        'mean_measured': np.mean(measured_stabilities),
    }
```

---

## Section 7: Validation Set Creation

### Hold-Out Tar Strategy

For SA-1B's 1,000 tar files, hold out entire tar files for validation:

```python
import random
from pathlib import Path

def create_sa1b_splits(sa1b_dir: str, val_ratio: float = 0.1,
                       test_ratio: float = 0.05, seed: int = 42) -> dict:
    """
    Create train/val/test splits for SA-1B using tar file hold-out.

    Args:
        sa1b_dir: Path to SA-1B directory with tar files
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility

    Returns:
        Dictionary with train/val/test tar file lists
    """
    random.seed(seed)

    # Get all tar files
    tar_files = sorted(Path(sa1b_dir).glob('sa_*.tar'))
    num_tars = len(tar_files)  # Should be 1000

    # Calculate split sizes
    num_test = int(num_tars * test_ratio)
    num_val = int(num_tars * val_ratio)
    num_train = num_tars - num_val - num_test

    # Shuffle tar file indices
    indices = list(range(num_tars))
    random.shuffle(indices)

    # Split indices
    test_indices = indices[:num_test]
    val_indices = indices[num_test:num_test + num_val]
    train_indices = indices[num_test + num_val:]

    splits = {
        'train': [tar_files[i].name for i in sorted(train_indices)],
        'val': [tar_files[i].name for i in sorted(val_indices)],
        'test': [tar_files[i].name for i in sorted(test_indices)],
    }

    # Statistics
    print(f"Total tar files: {num_tars}")
    print(f"Train: {len(splits['train'])} tars (~{len(splits['train']) * 11000:,} images)")
    print(f"Val:   {len(splits['val'])} tars (~{len(splits['val']) * 11000:,} images)")
    print(f"Test:  {len(splits['test'])} tars (~{len(splits['test']) * 11000:,} images)")

    return splits

def save_splits(splits: dict, output_path: str):
    """Save splits to JSON file."""
    import json

    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)

# Example usage:
# splits = create_sa1b_splits('/data/sa1b', val_ratio=0.1, test_ratio=0.05)
# Train: 850 tars (~9,350,000 images)
# Val:   100 tars (~1,100,000 images)
# Test:  50 tars (~550,000 images)
```

### Stratified Sampling by Mask Count

```python
def stratified_split_by_mask_count(sa1b_metadata: list,
                                    val_ratio: float = 0.1) -> tuple:
    """
    Stratified split ensuring similar mask count distributions.

    Args:
        sa1b_metadata: List of image metadata with mask counts
        val_ratio: Fraction for validation

    Returns:
        Train and validation indices
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    # Bin mask counts for stratification
    mask_counts = [m['num_masks'] for m in sa1b_metadata]
    bins = [0, 20, 50, 100, 200, float('inf')]
    labels = list(range(len(bins) - 1))
    binned_counts = np.digitize(mask_counts, bins) - 1

    # Stratified split
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=42
    )

    indices = np.arange(len(sa1b_metadata))
    train_idx, val_idx = next(splitter.split(indices, binned_counts))

    return train_idx, val_idx
```

---

## Section 8: Benchmark Protocols

### Comprehensive Evaluation Pipeline

```python
class SA1BEvaluator:
    """Complete evaluation suite for SA-1B segmentation models."""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.metrics = {}

    def evaluate(self, dataloader) -> dict:
        """Run full evaluation on dataset."""
        self.model.eval()

        # Initialize metric accumulators
        all_ious = []
        all_dice = []
        all_boundary_f = []
        all_pixel_acc = []
        predicted_ious = []
        actual_ious = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                target_masks = batch['masks']
                mask_info = batch['mask_info']

                # Get predictions
                pred_masks = self.model(images)
                pred_masks = (pred_masks > 0.5).cpu().numpy()
                target_masks = target_masks.cpu().numpy()

                # Compute metrics for each sample
                for pred, target, info in zip(pred_masks, target_masks, mask_info):
                    # IoU
                    iou = iou_numpy(pred, target)
                    all_ious.append(iou)

                    # Dice
                    dice = 2 * iou / (1 + iou)  # Convert from IoU
                    all_dice.append(dice)

                    # Boundary F-measure
                    bf = boundary_f_measure(pred, target)
                    all_boundary_f.append(bf)

                    # Pixel accuracy
                    pa = pixel_accuracy(pred, target)
                    all_pixel_acc.append(pa)

                    # Predicted IoU correlation
                    if 'predicted_iou' in info:
                        predicted_ious.append(info['predicted_iou'])
                        actual_ious.append(iou)

        # Aggregate metrics
        results = {
            'mIoU': np.mean(all_ious),
            'mDice': np.mean(all_dice),
            'mBF': np.mean(all_boundary_f),
            'mPA': np.mean(all_pixel_acc),
            'IoU_std': np.std(all_ious),
        }

        # IoU correlation if available
        if predicted_ious:
            corr, _ = pearsonr(predicted_ious, actual_ious)
            results['predicted_iou_correlation'] = corr

        # Per-threshold IoU
        for thresh in [0.5, 0.75, 0.9]:
            results[f'IoU@{thresh}'] = np.mean([
                1.0 if iou >= thresh else 0.0 for iou in all_ious
            ])

        return results

    def evaluate_by_size(self, dataloader) -> dict:
        """Evaluate separately by object size (small/medium/large)."""
        size_results = {
            'small': [],    # < 32^2 pixels
            'medium': [],   # 32^2 to 96^2 pixels
            'large': [],    # > 96^2 pixels
        }

        # ... similar evaluation loop with size binning

        return {
            size: np.mean(ious) for size, ious in size_results.items()
        }

def benchmark_model(model, test_loader, output_path: str):
    """Run standardized benchmark and save results."""
    evaluator = SA1BEvaluator(model)

    # Full evaluation
    results = evaluator.evaluate(test_loader)

    # Size-stratified evaluation
    size_results = evaluator.evaluate_by_size(test_loader)
    results['by_size'] = size_results

    # Save results
    import json
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\nBenchmark Results:")
    print(f"  mIoU:  {results['mIoU']:.4f} +/- {results['IoU_std']:.4f}")
    print(f"  mDice: {results['mDice']:.4f}")
    print(f"  mBF:   {results['mBF']:.4f}")
    print(f"  mPA:   {results['mPA']:.4f}")
    print(f"  IoU@0.5:  {results['IoU@0.5']:.4f}")
    print(f"  IoU@0.75: {results['IoU@0.75']:.4f}")

    return results
```

### Cross-Validation for Small Experiments

```python
def k_fold_cross_validation(dataset, model_fn, k=5, seed=42):
    """
    K-fold cross-validation for SA-1B experiments.

    Args:
        dataset: Full SA-1B dataset
        model_fn: Function that returns initialized model
        k: Number of folds
        seed: Random seed

    Returns:
        Cross-validation results
    """
    from sklearn.model_selection import KFold

    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(dataset)))):
        print(f"\nFold {fold + 1}/{k}")

        # Create fold datasets
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # Train model
        model = model_fn()
        train_model(model, train_subset)

        # Evaluate
        evaluator = SA1BEvaluator(model)
        val_loader = DataLoader(val_subset, batch_size=8)
        results = evaluator.evaluate(val_loader)

        fold_results.append(results)

    # Aggregate across folds
    aggregated = {}
    for metric in fold_results[0].keys():
        if isinstance(fold_results[0][metric], (int, float)):
            values = [r[metric] for r in fold_results]
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
            }

    return aggregated
```

---

## Section 9: ARR-COC-0-1 Integration - Spatial Grounding Evaluation

### Evaluation Metrics for VLM Spatial Grounding

ARR-COC's relevance realization requires metrics that capture both visual segmentation and text-region alignment:

```python
class ARRCOCSpatialGroundingEvaluator:
    """Evaluate spatial grounding quality for ARR-COC VLM."""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def evaluate(self, dataloader) -> dict:
        """Evaluate spatial grounding on SA-1B with text."""
        self.model.eval()

        # Metrics
        all_seg_iou = []
        all_grounding_acc = []
        all_text_alignment = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                masks = batch['masks'].to(self.device)
                text = batch['text']

                # Forward pass
                outputs = self.model(images, text)

                # 1. Segmentation quality (standard IoU)
                pred_masks = (outputs['masks'] > 0.5).cpu().numpy()
                target_masks = masks.cpu().numpy()

                for pred, target in zip(pred_masks, target_masks):
                    seg_iou = iou_numpy(pred, target)
                    all_seg_iou.append(seg_iou)

                # 2. Grounding accuracy (text refers to correct region)
                grounding_acc = self.compute_grounding_accuracy(
                    outputs['region_features'],
                    outputs['text_features'],
                    batch['region_text_mapping']
                )
                all_grounding_acc.extend(grounding_acc)

                # 3. Text-region alignment score
                alignment = self.compute_alignment_score(
                    outputs['region_features'],
                    outputs['text_features']
                )
                all_text_alignment.append(alignment)

        return {
            'segmentation_mIoU': np.mean(all_seg_iou),
            'grounding_accuracy': np.mean(all_grounding_acc),
            'text_alignment': np.mean(all_text_alignment),
        }

    def compute_grounding_accuracy(self, region_features, text_features,
                                    mappings) -> list:
        """Compute accuracy of text-to-region grounding."""
        accuracies = []

        for region_feat, text_feat, mapping in zip(
            region_features, text_features, mappings
        ):
            # Compute similarity matrix
            similarity = torch.mm(
                F.normalize(text_feat, dim=1),
                F.normalize(region_feat, dim=1).T
            )

            # Get predictions
            predicted_regions = similarity.argmax(dim=1)
            correct = (predicted_regions == mapping).float().mean()
            accuracies.append(correct.item())

        return accuracies

    def compute_alignment_score(self, region_features, text_features) -> float:
        """Compute overall text-region alignment quality."""
        # Use contrastive loss as alignment score
        # Lower loss = better alignment
        batch_size = len(region_features)

        alignment_scores = []
        for region_feat, text_feat in zip(region_features, text_features):
            # Cosine similarity
            similarity = F.cosine_similarity(
                region_feat.mean(dim=0, keepdim=True),
                text_feat.mean(dim=0, keepdim=True)
            )
            alignment_scores.append(similarity.item())

        return np.mean(alignment_scores)

def evaluate_arrcoc_spatial_grounding(model, sa1b_loader):
    """Main evaluation function for ARR-COC spatial grounding."""
    evaluator = ARRCOCSpatialGroundingEvaluator(model)

    results = evaluator.evaluate(sa1b_loader)

    print("\nARR-COC Spatial Grounding Evaluation:")
    print(f"  Segmentation mIoU:    {results['segmentation_mIoU']:.4f}")
    print(f"  Grounding Accuracy:   {results['grounding_accuracy']:.4f}")
    print(f"  Text Alignment:       {results['text_alignment']:.4f}")

    # Compute composite score for relevance realization
    relevance_score = (
        0.4 * results['segmentation_mIoU'] +
        0.4 * results['grounding_accuracy'] +
        0.2 * results['text_alignment']
    )
    results['relevance_realization_score'] = relevance_score
    print(f"  Relevance Score:      {relevance_score:.4f}")

    return results
```

### Validation Strategy for ARR-COC

```python
def create_arrcoc_validation_splits(sa1b_dir: str,
                                     text_annotations: str) -> dict:
    """
    Create validation splits for ARR-COC spatial grounding.

    Args:
        sa1b_dir: Path to SA-1B data
        text_annotations: Path to text annotations for grounding

    Returns:
        Train/val/test splits with balanced text coverage
    """
    # Load text annotation statistics
    with open(text_annotations) as f:
        annotations = json.load(f)

    # Stratify by:
    # 1. Number of masks per image (granularity)
    # 2. Text complexity (word count)
    # 3. Spatial relationships mentioned

    # Create balanced splits
    splits = create_sa1b_splits(sa1b_dir, val_ratio=0.1, test_ratio=0.05)

    # Verify text coverage in each split
    for split_name in ['train', 'val', 'test']:
        split_tars = splits[split_name]
        # Ensure diverse text annotations in each split
        # ...

    return splits
```

---

## Complete Evaluation Script

```python
#!/usr/bin/env python3
"""
Comprehensive evaluation script for SA-1B segmentation models.

Usage:
    python evaluate_sa1b.py --model_path checkpoints/best.pt \
                            --data_path /data/sa1b \
                            --output_dir ./eval_results
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--splits_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    parser.add_argument('--batch_size', type=int, default=8)
    return parser.parse_args()

def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.model_path)
    model.eval()

    # Load splits
    with open(args.splits_path) as f:
        splits = json.load(f)

    # Create test dataset
    test_dataset = SA1BDataset(
        args.data_path,
        tar_files=splits['test']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )

    # Run evaluation
    evaluator = SA1BEvaluator(model)

    print("Running evaluation...")
    results = evaluator.evaluate(test_loader)

    # Size-stratified results
    print("Evaluating by object size...")
    size_results = evaluator.evaluate_by_size(test_loader)
    results['by_size'] = size_results

    # Save results
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"mIoU:       {results['mIoU']:.4f}")
    print(f"mDice:      {results['mDice']:.4f}")
    print(f"mBF:        {results['mBF']:.4f}")
    print(f"mPA:        {results['mPA']:.4f}")
    print(f"IoU@0.5:    {results['IoU@0.5']:.4f}")
    print(f"IoU@0.75:   {results['IoU@0.75']:.4f}")
    print(f"IoU@0.9:    {results['IoU@0.9']:.4f}")

    if 'by_size' in results:
        print("\nBy Object Size:")
        for size, iou in results['by_size'].items():
            print(f"  {size}: {iou:.4f}")

    print(f"\nResults saved to: {results_path}")

if __name__ == '__main__':
    main()
```

---

## Sources

**Web Research (accessed 2025-11-20):**
- [Medium - Medical Image Segmentation Metrics](https://medium.com/mastering-data-science/understanding-evaluation-metrics-in-medical-image-segmentation-d289a373a3f)
- [ApX Machine Learning - Segmentation Metrics](https://apxml.com/courses/cnns-for-computer-vision/chapter-4-image-segmentation-techniques/segmentation-evaluation-metrics)
- [VeRADP AI - Dice vs IoU](https://veradp-ai.com/iou-vs-dice/)
- [GeeksforGeeks - Segmentation Evaluation Metrics](https://www.geeksforgeeks.org/computer-vision/what-are-different-evaluation-metrics-used-to-evaluate-image-segmentation-models/)
- [Tencent Cloud - Segmentation Metrics](https://www.tencentcloud.com/techpedia/112106)
- [arXiv - Medical Image Segmentation Metrics](https://arxiv.org/pdf/2202.05273)
- [Boundary IoU Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.pdf)
- [Jeremy Jordan - Evaluating Segmentation](https://www.jeremyjordan.me/evaluating-image-segmentation-models/)

**Source Document:**
- PLAN-MD-FILES/november/20th/SAM_DATASET_SA1B.md

**ARR-COC Integration:**
- Spatial grounding quality metrics for VLM
- Text-region alignment evaluation
- Relevance realization composite scoring
- Validation strategies for multimodal training
