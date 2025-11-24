# Batch Sampling Strategies for SA-1B Training

**KNOWLEDGE-DROP**: SA-1B Dataset Mastery - Efficient Sampling Strategies
**Date**: 2025-11-20
**Source**: Web research on sampling methods, OHEM, curriculum learning
**Focus**: Sampling strategies for relevance diversity in ARR-COC training

---

## 1. Overview: Why Sampling Matters for SA-1B

SA-1B contains **11M images with 1.1B masks** across 1,000 tar files. Efficient sampling is critical for:

- **Training efficiency**: Cannot load all data at once
- **Diversity**: Balance across mask granularities, domains, geographic regions
- **Hard example mining**: Focus on challenging samples
- **Memory constraints**: ~350MB per fully decoded sample
- **Convergence speed**: Smart sampling accelerates learning

From [ACM - Efficient Training of Large-Scale Models](https://dl.acm.org/doi/10.1145/3700439) (accessed 2025-11-20):
> "During training, updates are performed on batches of samples, and all samples within a batch are treated equally. However, this results in inefficient training."

---

## 2. Random Sampling Baseline

**Simple but often suboptimal for large-scale training.**

### Implementation

```python
import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List, Iterator
import numpy as np


class RandomSA1BSampler(Sampler):
    """
    Basic random sampler for SA-1B dataset.

    Samples uniformly at random from all available samples.
    Simple baseline but may underrepresent rare categories.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool = True,
        seed: int = 42
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[List[int]]:
        # Set seed for reproducibility across epochs
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Generate random permutation
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        # Create batches
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)

        # Shuffle batches
        random.seed(self.seed + self.epoch)
        random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int):
        """Set epoch for different random seed each epoch."""
        self.epoch = epoch


class TarAwareRandomSampler(Sampler):
    """
    Random sampler that respects tar file boundaries.

    SA-1B is organized in 1,000 tar files (~11k images each).
    This sampler ensures we sample from tar files that are loaded,
    avoiding repeated tar loading overhead.
    """

    def __init__(
        self,
        tar_indices: dict,  # {tar_id: [sample_indices]}
        batch_size: int,
        samples_per_tar: int = 1000,
        seed: int = 42
    ):
        self.tar_indices = tar_indices
        self.batch_size = batch_size
        self.samples_per_tar = samples_per_tar
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Shuffle tar order
        tar_ids = list(self.tar_indices.keys())
        tar_order = torch.randperm(len(tar_ids), generator=g).tolist()

        all_indices = []

        for tar_idx in tar_order:
            tar_id = tar_ids[tar_idx]
            indices = self.tar_indices[tar_id]

            # Sample from this tar
            if len(indices) > self.samples_per_tar:
                perm = torch.randperm(len(indices), generator=g)[:self.samples_per_tar]
                sampled = [indices[i] for i in perm.tolist()]
            else:
                sampled = indices

            all_indices.extend(sampled)

        # Create batches
        for i in range(0, len(all_indices), self.batch_size):
            batch = all_indices[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                yield batch

    def set_epoch(self, epoch: int):
        self.epoch = epoch
```

### Limitations of Random Sampling

1. **No quality consideration**: Easy and hard samples treated equally
2. **Granularity imbalance**: May oversample common granularities
3. **Geographic bias**: May not cover all 63 countries equally
4. **Inefficient learning**: Wastes compute on already-learned patterns

---

## 3. Stratified Sampling by Mask Granularity

**Balance representation across fine to coarse masks.**

### Implementation

```python
class GranularityStratifiedSampler(Sampler):
    """
    Stratified sampler based on mask granularity.

    SA-1B contains masks from fine (door handles) to coarse (buildings).
    This sampler ensures balanced representation across granularity levels.
    """

    def __init__(
        self,
        dataset: Dataset,
        granularity_labels: List[int],  # 0=fine, 1=medium, 2=coarse
        batch_size: int,
        samples_per_granularity: int = None,
        seed: int = 42
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

        # Group indices by granularity
        self.granularity_indices = {}
        for idx, granularity in enumerate(granularity_labels):
            if granularity not in self.granularity_indices:
                self.granularity_indices[granularity] = []
            self.granularity_indices[granularity].append(idx)

        self.num_granularities = len(self.granularity_indices)

        # Samples per granularity per epoch
        if samples_per_granularity is None:
            min_size = min(len(v) for v in self.granularity_indices.values())
            self.samples_per_granularity = min_size
        else:
            self.samples_per_granularity = samples_per_granularity

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Sample equally from each granularity
        all_indices = []

        for granularity, indices in self.granularity_indices.items():
            # Random sample from this granularity
            perm = torch.randperm(len(indices), generator=g)
            sampled = [indices[i] for i in perm[:self.samples_per_granularity].tolist()]
            all_indices.extend(sampled)

        # Shuffle all samples
        perm = torch.randperm(len(all_indices), generator=g)
        all_indices = [all_indices[i] for i in perm.tolist()]

        # Create batches
        for i in range(0, len(all_indices), self.batch_size):
            batch = all_indices[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                yield batch

    def set_epoch(self, epoch: int):
        self.epoch = epoch


def compute_mask_granularity(mask_area: float, image_area: float) -> int:
    """
    Classify mask into granularity level based on relative size.

    Args:
        mask_area: Number of pixels in mask
        image_area: Total image pixels

    Returns:
        Granularity level (0=fine, 1=medium, 2=coarse)
    """
    relative_size = mask_area / image_area

    if relative_size < 0.01:  # < 1% of image
        return 0  # Fine (e.g., door handles, buttons)
    elif relative_size < 0.1:  # 1-10% of image
        return 1  # Medium (e.g., chairs, people)
    else:  # > 10% of image
        return 2  # Coarse (e.g., buildings, sky)


class MultiLevelGranularitySampler(Sampler):
    """
    Sample with granularity-aware batch composition.

    Each batch contains samples from multiple granularity levels
    to help the model learn hierarchical segmentation.
    """

    def __init__(
        self,
        dataset: Dataset,
        granularity_labels: List[int],
        batch_size: int,
        granularity_ratio: tuple = (1, 1, 1),  # fine:medium:coarse
        seed: int = 42
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

        # Group by granularity
        self.granularity_indices = {0: [], 1: [], 2: []}
        for idx, g in enumerate(granularity_labels):
            self.granularity_indices[g].append(idx)

        # Compute samples per granularity per batch
        total_ratio = sum(granularity_ratio)
        self.samples_per_batch = {
            0: int(batch_size * granularity_ratio[0] / total_ratio),
            1: int(batch_size * granularity_ratio[1] / total_ratio),
            2: batch_size - int(batch_size * granularity_ratio[0] / total_ratio) - \
               int(batch_size * granularity_ratio[1] / total_ratio)
        }

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Shuffle indices within each granularity
        shuffled = {}
        for gran, indices in self.granularity_indices.items():
            perm = torch.randperm(len(indices), generator=g)
            shuffled[gran] = [indices[i] for i in perm.tolist()]

        # Pointers for each granularity
        pointers = {0: 0, 1: 0, 2: 0}

        # Generate batches
        max_batches = min(
            len(shuffled[g]) // self.samples_per_batch[g]
            for g in range(3)
            if self.samples_per_batch[g] > 0
        )

        for _ in range(max_batches):
            batch = []
            for gran in range(3):
                n = self.samples_per_batch[gran]
                batch.extend(shuffled[gran][pointers[gran]:pointers[gran] + n])
                pointers[gran] += n

            # Shuffle within batch
            random.shuffle(batch)
            yield batch

    def set_epoch(self, epoch: int):
        self.epoch = epoch
```

---

## 4. Hard Example Mining (OHEM)

**Focus on samples the model struggles with.**

### Online Hard Example Mining

From [Papers With Code - OHEM](https://paperswithcode.com/method/ohem) (accessed 2025-11-20):
> "Online Hard Example Mining automatically selects hard examples for training, enabling efficient and effective training."

```python
class OnlineHardExampleMiner:
    """
    Online Hard Example Mining (OHEM) for segmentation.

    Selects the hardest examples (highest loss) in each batch
    for gradient computation, focusing learning on difficult cases.
    """

    def __init__(
        self,
        ratio: float = 0.25,  # Keep top 25% hardest
        min_samples: int = 256
    ):
        self.ratio = ratio
        self.min_samples = min_samples

    def mine(
        self,
        losses: torch.Tensor,  # Per-sample losses [B]
        batch_size: int
    ) -> torch.Tensor:
        """
        Select hard examples based on loss.

        Args:
            losses: Loss values for each sample
            batch_size: Original batch size

        Returns:
            Boolean mask of selected samples
        """
        num_hard = max(
            self.min_samples,
            int(batch_size * self.ratio)
        )
        num_hard = min(num_hard, batch_size)

        # Get indices of top-k losses
        _, hard_indices = torch.topk(losses, num_hard)

        # Create selection mask
        mask = torch.zeros(batch_size, dtype=torch.bool, device=losses.device)
        mask[hard_indices] = True

        return mask


class OHEMSegmentationLoss(torch.nn.Module):
    """
    Segmentation loss with Online Hard Example Mining.

    Computes loss on all samples, then backprops only on hardest.
    """

    def __init__(
        self,
        base_loss: torch.nn.Module,
        ohem_ratio: float = 0.25,
        min_kept: int = 256
    ):
        super().__init__()
        self.base_loss = base_loss
        self.miner = OnlineHardExampleMiner(ohem_ratio, min_kept)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute OHEM loss.

        Args:
            pred: Predicted masks [B, 1, H, W]
            target: Ground truth masks [B, 1, H, W]

        Returns:
            Scalar loss value (only from hard examples)
        """
        batch_size = pred.shape[0]

        # Compute per-sample loss
        per_sample_losses = []
        for i in range(batch_size):
            loss_i = self.base_loss(pred[i:i+1], target[i:i+1])
            per_sample_losses.append(loss_i)

        per_sample_losses = torch.stack(per_sample_losses)

        # Mine hard examples
        hard_mask = self.miner.mine(per_sample_losses, batch_size)

        # Return mean loss of hard examples only
        return per_sample_losses[hard_mask].mean()


class PixelWiseOHEM(torch.nn.Module):
    """
    Pixel-wise OHEM for dense prediction.

    Instead of selecting hard samples, selects hard pixels
    within each sample.
    """

    def __init__(
        self,
        ratio: float = 0.25,
        min_kept: int = 100000
    ):
        super().__init__()
        self.ratio = ratio
        self.min_kept = min_kept

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pixel-wise OHEM loss.

        Args:
            pred: Predicted logits [B, 1, H, W]
            target: Ground truth masks [B, 1, H, W]

        Returns:
            Scalar loss from hard pixels
        """
        # Compute per-pixel loss
        pixel_losses = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )

        # Flatten
        pixel_losses_flat = pixel_losses.view(-1)
        num_pixels = pixel_losses_flat.numel()

        # Number of hard pixels to keep
        num_hard = max(
            self.min_kept,
            int(num_pixels * self.ratio)
        )

        # Select hardest pixels
        hard_losses, _ = torch.topk(pixel_losses_flat, num_hard)

        return hard_losses.mean()
```

### Stratified OHEM

From [arXiv - S-OHEM](https://arxiv.org/abs/1705.02233) (accessed 2025-11-20):
> "Stratified Online Hard Example Mining (S-OHEM) algorithm for training higher efficiency and accuracy detectors."

```python
class StratifiedOHEM:
    """
    Stratified OHEM: Mine hard examples within each stratum.

    Ensures hard examples are selected from all categories,
    not just the most difficult overall category.
    """

    def __init__(
        self,
        strata_labels: torch.Tensor,  # Stratum for each sample
        ratio: float = 0.25
    ):
        self.strata_labels = strata_labels
        self.ratio = ratio
        self.num_strata = len(torch.unique(strata_labels))

    def mine(
        self,
        losses: torch.Tensor,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Select hard examples from each stratum.

        Args:
            losses: Loss values for batch samples
            indices: Dataset indices for batch samples

        Returns:
            Selected indices
        """
        selected = []

        for stratum in range(self.num_strata):
            # Find samples in this stratum
            stratum_mask = self.strata_labels[indices] == stratum
            stratum_losses = losses[stratum_mask]
            stratum_indices = indices[stratum_mask]

            # Mine hard examples from this stratum
            num_hard = max(1, int(len(stratum_losses) * self.ratio))
            _, hard_idx = torch.topk(stratum_losses, num_hard)

            selected.extend(stratum_indices[hard_idx].tolist())

        return torch.tensor(selected)
```

---

## 5. Curriculum Learning

**Start easy, progressively increase difficulty.**

### Implementation

```python
class CurriculumSampler(Sampler):
    """
    Curriculum learning sampler: easy to hard progression.

    Training starts with easy samples (large, clear masks)
    and progressively introduces harder samples (small, ambiguous).
    """

    def __init__(
        self,
        dataset: Dataset,
        difficulty_scores: List[float],  # Higher = harder
        batch_size: int,
        total_epochs: int,
        warmup_epochs: int = 5,
        seed: int = 42
    ):
        self.dataset = dataset
        self.difficulty_scores = np.array(difficulty_scores)
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.seed = seed
        self.epoch = 0

        # Sort indices by difficulty
        self.sorted_indices = np.argsort(self.difficulty_scores)

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Curriculum progress (0 = only easy, 1 = all samples)
        if self.epoch < self.warmup_epochs:
            progress = 0.3 + 0.7 * (self.epoch / self.warmup_epochs)
        else:
            progress = min(1.0, 0.3 + 0.7 * (self.epoch / self.total_epochs))

        # Select samples up to current difficulty threshold
        num_available = int(len(self.sorted_indices) * progress)
        available_indices = self.sorted_indices[:num_available].tolist()

        # Shuffle available samples
        perm = torch.randperm(len(available_indices), generator=g)
        shuffled = [available_indices[i] for i in perm.tolist()]

        # Create batches
        for i in range(0, len(shuffled), self.batch_size):
            batch = shuffled[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                yield batch

    def set_epoch(self, epoch: int):
        self.epoch = epoch


def compute_sample_difficulty(
    mask_area: float,
    image_area: float,
    stability_score: float,
    predicted_iou: float,
    num_masks_in_image: int
) -> float:
    """
    Compute difficulty score for curriculum learning.

    Considers multiple factors that make segmentation harder.

    Args:
        mask_area: Area of the mask
        image_area: Total image area
        stability_score: How stable the mask is
        predicted_iou: Model's confidence in the mask
        num_masks_in_image: Number of masks (more = harder)

    Returns:
        Difficulty score (higher = harder)
    """
    # Smaller relative area = harder
    size_difficulty = 1.0 - (mask_area / image_area)

    # Lower stability = harder
    stability_difficulty = 1.0 - stability_score

    # Lower predicted IoU = harder
    iou_difficulty = 1.0 - predicted_iou

    # More masks = more complex scene
    complexity = min(1.0, num_masks_in_image / 200)  # Normalize

    # Weighted combination
    difficulty = (
        0.3 * size_difficulty +
        0.3 * stability_difficulty +
        0.2 * iou_difficulty +
        0.2 * complexity
    )

    return difficulty


class AdaptiveCurriculumSampler(Sampler):
    """
    Adaptive curriculum that adjusts based on model performance.

    Instead of fixed schedule, monitors model loss to decide
    when to introduce harder samples.
    """

    def __init__(
        self,
        dataset: Dataset,
        difficulty_scores: List[float],
        batch_size: int,
        initial_percentile: float = 0.3,
        growth_rate: float = 0.05,
        loss_threshold: float = 0.1,
        seed: int = 42
    ):
        self.dataset = dataset
        self.difficulty_scores = np.array(difficulty_scores)
        self.batch_size = batch_size
        self.current_percentile = initial_percentile
        self.growth_rate = growth_rate
        self.loss_threshold = loss_threshold
        self.seed = seed
        self.epoch = 0

        self.sorted_indices = np.argsort(self.difficulty_scores)
        self.recent_losses = []

    def update_curriculum(self, epoch_loss: float):
        """
        Update curriculum based on training loss.

        If loss is below threshold, expand to harder samples.
        """
        self.recent_losses.append(epoch_loss)

        if len(self.recent_losses) >= 3:
            # Check if loss has stabilized
            recent_mean = np.mean(self.recent_losses[-3:])
            if recent_mean < self.loss_threshold:
                self.current_percentile = min(
                    1.0,
                    self.current_percentile + self.growth_rate
                )
                self.recent_losses = []  # Reset after expansion

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Select samples based on current curriculum
        num_available = int(len(self.sorted_indices) * self.current_percentile)
        available_indices = self.sorted_indices[:num_available].tolist()

        # Shuffle
        perm = torch.randperm(len(available_indices), generator=g)
        shuffled = [available_indices[i] for i in perm.tolist()]

        for i in range(0, len(shuffled), self.batch_size):
            batch = shuffled[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                yield batch

    def set_epoch(self, epoch: int):
        self.epoch = epoch
```

---

## 6. Balanced Sampling Across Tar Files

**Ensure uniform coverage across SA-1B's 1,000 tar files.**

### Implementation

```python
class BalancedTarSampler(Sampler):
    """
    Sample uniformly across tar files.

    SA-1B is split into 1,000 tar files. This sampler ensures
    we don't oversample from any particular tar file.
    """

    def __init__(
        self,
        tar_to_indices: dict,  # {tar_id: [sample_indices]}
        batch_size: int,
        samples_per_tar_per_epoch: int = 100,
        seed: int = 42
    ):
        self.tar_to_indices = tar_to_indices
        self.batch_size = batch_size
        self.samples_per_tar = samples_per_tar_per_epoch
        self.seed = seed
        self.epoch = 0

        self.tar_ids = list(tar_to_indices.keys())

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Sample from each tar file
        all_samples = []

        for tar_id in self.tar_ids:
            indices = self.tar_to_indices[tar_id]

            # Random sample from this tar
            n_samples = min(len(indices), self.samples_per_tar)
            perm = torch.randperm(len(indices), generator=g)[:n_samples]
            sampled = [indices[i] for i in perm.tolist()]

            all_samples.extend(sampled)

        # Shuffle all samples
        perm = torch.randperm(len(all_samples), generator=g)
        all_samples = [all_samples[i] for i in perm.tolist()]

        # Create batches
        for i in range(0, len(all_samples), self.batch_size):
            batch = all_samples[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                yield batch

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class StreamingTarSampler(Sampler):
    """
    Streaming sampler that loads tar files sequentially.

    For memory-constrained environments, processes one tar
    at a time instead of random access across all tars.
    """

    def __init__(
        self,
        tar_to_indices: dict,
        batch_size: int,
        tars_per_epoch: int = 100,  # Process subset of tars each epoch
        seed: int = 42
    ):
        self.tar_to_indices = tar_to_indices
        self.batch_size = batch_size
        self.tars_per_epoch = tars_per_epoch
        self.seed = seed
        self.epoch = 0

        self.tar_ids = list(tar_to_indices.keys())

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Select tars for this epoch
        tar_perm = torch.randperm(len(self.tar_ids), generator=g)
        selected_tars = [self.tar_ids[i] for i in tar_perm[:self.tars_per_epoch].tolist()]

        # Process each tar
        for tar_id in selected_tars:
            indices = self.tar_to_indices[tar_id]

            # Shuffle within tar
            perm = torch.randperm(len(indices), generator=g)
            shuffled = [indices[i] for i in perm.tolist()]

            # Yield batches from this tar
            for i in range(0, len(shuffled), self.batch_size):
                batch = shuffled[i:i + self.batch_size]
                if len(batch) == self.batch_size:
                    yield batch

    def set_epoch(self, epoch: int):
        self.epoch = epoch
```

---

## 7. Dynamic Sampling Based on Loss

**Adjust sampling probabilities based on recent losses.**

### Implementation

```python
class LossWeightedSampler(Sampler):
    """
    Sample proportionally to recent loss values.

    Samples with higher recent loss get sampled more frequently.
    Implements importance sampling for faster convergence.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        temperature: float = 1.0,
        ema_decay: float = 0.99,
        seed: int = 42
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.temperature = temperature
        self.ema_decay = ema_decay
        self.seed = seed
        self.epoch = 0

        # Initialize uniform losses
        self.sample_losses = np.ones(len(dataset))

    def update_losses(self, indices: List[int], losses: List[float]):
        """
        Update loss estimates with EMA.

        Args:
            indices: Sample indices in batch
            losses: Loss values for those samples
        """
        for idx, loss in zip(indices, losses):
            self.sample_losses[idx] = (
                self.ema_decay * self.sample_losses[idx] +
                (1 - self.ema_decay) * loss
            )

    def __iter__(self) -> Iterator[List[int]]:
        # Compute sampling probabilities
        scaled_losses = self.sample_losses / self.temperature
        probs = scaled_losses / scaled_losses.sum()

        # Sample with replacement based on loss
        rng = np.random.RandomState(self.seed + self.epoch)
        indices = rng.choice(
            len(self.dataset),
            size=len(self.dataset),
            replace=True,
            p=probs
        )

        # Create batches
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size].tolist()
            if len(batch) == self.batch_size:
                yield batch

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class ReplayBufferSampler(Sampler):
    """
    Maintain a replay buffer of hard examples.

    Keeps track of recent hard examples and replays them
    periodically during training.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        buffer_size: int = 10000,
        replay_ratio: float = 0.2,
        loss_percentile: float = 0.9,
        seed: int = 42
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio
        self.loss_percentile = loss_percentile
        self.seed = seed
        self.epoch = 0

        # Buffer: (index, loss) pairs
        self.buffer = []

    def update_buffer(self, indices: List[int], losses: List[float]):
        """
        Add hard examples to buffer.

        Args:
            indices: Sample indices
            losses: Loss values
        """
        threshold = np.percentile(losses, self.loss_percentile * 100)

        for idx, loss in zip(indices, losses):
            if loss > threshold:
                self.buffer.append((idx, loss))

        # Trim buffer if too large
        if len(self.buffer) > self.buffer_size:
            # Keep highest loss samples
            self.buffer.sort(key=lambda x: x[1], reverse=True)
            self.buffer = self.buffer[:self.buffer_size]

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Normal sampling
        all_indices = torch.randperm(len(self.dataset), generator=g).tolist()

        # Determine replay samples per batch
        replay_per_batch = int(self.batch_size * self.replay_ratio)
        normal_per_batch = self.batch_size - replay_per_batch

        # Create batches
        buffer_indices = [x[0] for x in self.buffer]

        for i in range(0, len(all_indices), normal_per_batch):
            batch = all_indices[i:i + normal_per_batch]

            # Add replay samples
            if len(buffer_indices) >= replay_per_batch:
                replay = random.sample(buffer_indices, replay_per_batch)
                batch.extend(replay)

            if len(batch) == self.batch_size:
                random.shuffle(batch)
                yield batch

    def set_epoch(self, epoch: int):
        self.epoch = epoch
```

---

## 8. Memory-Efficient Sampling

**Handle SA-1B's ~10TB scale with limited memory.**

### Implementation

```python
class ChunkedSampler(Sampler):
    """
    Memory-efficient sampler that processes data in chunks.

    Loads subset of dataset indices at a time,
    suitable for distributed filesystems.
    """

    def __init__(
        self,
        total_samples: int,
        batch_size: int,
        chunk_size: int = 100000,
        seed: int = 42
    ):
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Generate random permutation in chunks to save memory
        all_indices = []

        for chunk_start in range(0, self.total_samples, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, self.total_samples)
            chunk_indices = list(range(chunk_start, chunk_end))

            # Shuffle chunk
            perm = torch.randperm(len(chunk_indices), generator=g)
            shuffled_chunk = [chunk_indices[i] for i in perm.tolist()]

            all_indices.extend(shuffled_chunk)

        # Shuffle chunks
        num_chunks = len(all_indices) // self.chunk_size
        chunk_perm = torch.randperm(num_chunks, generator=g)

        final_indices = []
        for i in chunk_perm.tolist():
            start = i * self.chunk_size
            end = start + self.chunk_size
            final_indices.extend(all_indices[start:end])

        # Add remaining samples
        final_indices.extend(all_indices[num_chunks * self.chunk_size:])

        # Create batches
        for i in range(0, len(final_indices), self.batch_size):
            batch = final_indices[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                yield batch

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class LazyIndexSampler(Sampler):
    """
    Lazy index generation for very large datasets.

    Generates indices on-the-fly instead of storing full permutation.
    Uses consistent hashing for reproducibility.
    """

    def __init__(
        self,
        total_samples: int,
        batch_size: int,
        seed: int = 42
    ):
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def _hash_index(self, idx: int) -> int:
        """
        Consistent hash for index permutation.

        Args:
            idx: Original index

        Returns:
            Permuted index
        """
        # Simple multiplicative hash
        h = (idx * 2654435761 + self.seed + self.epoch) % (2**32)
        return h % self.total_samples

    def __iter__(self) -> Iterator[List[int]]:
        # Generate indices lazily
        batch = []

        for i in range(self.total_samples):
            permuted_idx = self._hash_index(i)
            batch.append(permuted_idx)

            if len(batch) == self.batch_size:
                yield batch
                batch = []

    def __len__(self) -> int:
        return self.total_samples // self.batch_size

    def set_epoch(self, epoch: int):
        self.epoch = epoch
```

---

## 9. Multi-Attribute Balanced Sampling

**Balance across multiple factors simultaneously.**

### Implementation

```python
class MultiAttributeSampler(Sampler):
    """
    Balance sampling across multiple attributes.

    For SA-1B: balance granularity, domain, geographic region, etc.
    """

    def __init__(
        self,
        dataset: Dataset,
        attributes: dict,  # {attr_name: List[attr_value]}
        batch_size: int,
        primary_attribute: str = 'granularity',
        seed: int = 42
    ):
        self.dataset = dataset
        self.attributes = attributes
        self.batch_size = batch_size
        self.primary_attribute = primary_attribute
        self.seed = seed
        self.epoch = 0

        # Build multi-attribute index
        self._build_index()

    def _build_index(self):
        """Build hierarchical index for multi-attribute sampling."""
        self.attr_to_indices = {}

        primary = self.attributes[self.primary_attribute]
        unique_primary = list(set(primary))

        for pval in unique_primary:
            # Indices with this primary attribute value
            indices = [i for i, v in enumerate(primary) if v == pval]
            self.attr_to_indices[pval] = indices

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Sample equally from each primary attribute value
        all_samples = []
        primary_values = list(self.attr_to_indices.keys())

        # Determine samples per primary value
        min_count = min(len(v) for v in self.attr_to_indices.values())
        samples_per_value = min_count

        for pval in primary_values:
            indices = self.attr_to_indices[pval]
            perm = torch.randperm(len(indices), generator=g)[:samples_per_value]
            sampled = [indices[i] for i in perm.tolist()]
            all_samples.extend(sampled)

        # Shuffle
        perm = torch.randperm(len(all_samples), generator=g)
        all_samples = [all_samples[i] for i in perm.tolist()]

        # Create batches
        for i in range(0, len(all_samples), self.batch_size):
            batch = all_samples[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                yield batch

    def set_epoch(self, epoch: int):
        self.epoch = epoch


def create_balanced_sampler_for_sa1b(
    dataset,
    mask_areas: List[float],
    image_areas: List[float],
    stability_scores: List[float],
    tar_ids: List[int],
    batch_size: int = 64,
    strategy: str = 'multi_attribute'
) -> Sampler:
    """
    Factory function to create appropriate sampler for SA-1B.

    Args:
        dataset: SA-1B dataset
        mask_areas: Area of each mask
        image_areas: Area of each image
        stability_scores: Stability score of each mask
        tar_ids: Tar file ID for each sample
        batch_size: Batch size
        strategy: Sampling strategy

    Returns:
        Configured sampler
    """
    # Compute granularity levels
    granularities = [
        compute_mask_granularity(ma, ia)
        for ma, ia in zip(mask_areas, image_areas)
    ]

    if strategy == 'random':
        return RandomSA1BSampler(dataset, batch_size)

    elif strategy == 'stratified_granularity':
        return GranularityStratifiedSampler(
            dataset, granularities, batch_size
        )

    elif strategy == 'curriculum':
        difficulties = [
            compute_sample_difficulty(
                ma, ia, ss, 0.8, 100
            )
            for ma, ia, ss in zip(mask_areas, image_areas, stability_scores)
        ]
        return CurriculumSampler(
            dataset, difficulties, batch_size,
            total_epochs=100, warmup_epochs=5
        )

    elif strategy == 'balanced_tar':
        tar_to_indices = {}
        for i, tid in enumerate(tar_ids):
            if tid not in tar_to_indices:
                tar_to_indices[tid] = []
            tar_to_indices[tid].append(i)
        return BalancedTarSampler(
            tar_to_indices, batch_size
        )

    elif strategy == 'multi_attribute':
        return MultiAttributeSampler(
            dataset,
            {'granularity': granularities, 'tar_id': tar_ids},
            batch_size
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
```

---

## 10. Distributed Sampling Strategies

**Sampling for multi-GPU/multi-node training.**

### Implementation

```python
from torch.utils.data.distributed import DistributedSampler


class DistributedBalancedSampler(DistributedSampler):
    """
    Distributed sampler with balanced sampling.

    Ensures each GPU gets balanced samples across attributes.
    """

    def __init__(
        self,
        dataset: Dataset,
        granularity_labels: List[int],
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = True,
        seed: int = 42
    ):
        super().__init__(dataset, num_replicas, rank, shuffle, seed)

        self.granularity_labels = granularity_labels

        # Group by granularity
        self.granularity_indices = {}
        for idx, g in enumerate(granularity_labels):
            if g not in self.granularity_indices:
                self.granularity_indices[g] = []
            self.granularity_indices[g].append(idx)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Balanced sampling
        all_indices = []
        min_count = min(len(v) for v in self.granularity_indices.values())

        for gran, indices in self.granularity_indices.items():
            perm = torch.randperm(len(indices), generator=g)[:min_count]
            sampled = [indices[i] for i in perm.tolist()]
            all_indices.extend(sampled)

        # Shuffle
        perm = torch.randperm(len(all_indices), generator=g)
        all_indices = [all_indices[i] for i in perm.tolist()]

        # Distribute across replicas
        num_samples = len(all_indices) // self.num_replicas
        start = self.rank * num_samples
        end = start + num_samples

        return iter(all_indices[start:end])


class ShardedSampler(Sampler):
    """
    Sampler for sharded data across nodes.

    Each node has subset of tar files. Sampler only uses local shards.
    """

    def __init__(
        self,
        local_indices: List[int],  # Indices available on this node
        batch_size: int,
        seed: int = 42
    ):
        self.local_indices = local_indices
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Shuffle local indices
        perm = torch.randperm(len(self.local_indices), generator=g)
        shuffled = [self.local_indices[i] for i in perm.tolist()]

        # Create batches
        for i in range(0, len(shuffled), self.batch_size):
            batch = shuffled[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                yield batch

    def set_epoch(self, epoch: int):
        self.epoch = epoch
```

---

## 11. ARR-COC-0-1 Integration (10%): Sampling for Relevance Diversity

### Application to Vision-Language Training

Sampling strategies from SA-1B training directly apply to ARR-COC's relevance realization:

**1. Multi-Attribute Balanced Sampling**
```python
class RelevanceDiversitySampler(Sampler):
    """
    Sample for diverse relevance patterns in VLM training.

    Balance across:
    - Spatial granularity (fine/medium/coarse grounding)
    - Query complexity (simple/compound)
    - Domain (natural/synthetic/document)
    """

    def __init__(
        self,
        dataset: Dataset,
        granularities: List[int],
        query_complexities: List[int],
        domains: List[int],
        batch_size: int
    ):
        self.attributes = {
            'granularity': granularities,
            'complexity': query_complexities,
            'domain': domains
        }
        # ... balanced sampling across all attributes


class RelevanceHardExampleMiner:
    """
    Mine hard examples for relevance grounding.

    Focus on:
    - Ambiguous spatial references
    - Fine-grained distinctions
    - Multi-object scenes
    """

    def mine(
        self,
        grounding_losses: torch.Tensor,
        alignment_losses: torch.Tensor
    ) -> torch.Tensor:
        """
        Select hard examples based on grounding quality.

        High grounding loss = poor spatial localization
        High alignment loss = poor text-region matching
        """
        combined = grounding_losses + alignment_losses
        # ... select top-k hardest
```

**2. Curriculum for Spatial Grounding**
```python
def compute_grounding_difficulty(
    mask_area: float,
    image_area: float,
    query_length: int,
    num_objects: int,
    spatial_relations: int  # Number of spatial terms in query
) -> float:
    """
    Compute difficulty for spatial grounding learning.

    Curriculum progression:
    1. Large objects, simple queries
    2. Medium objects, compound queries
    3. Small objects, complex spatial relations
    """
    size_diff = 1.0 - (mask_area / image_area)
    query_diff = min(1.0, query_length / 50)
    scene_diff = min(1.0, num_objects / 20)
    relation_diff = min(1.0, spatial_relations / 5)

    return 0.3 * size_diff + 0.2 * query_diff + 0.3 * scene_diff + 0.2 * relation_diff
```

**3. Key Sampling Principles for ARR-COC**

| SA-1B Strategy | ARR-COC Application |
|----------------|---------------------|
| Granularity stratification | Balance fine/medium/coarse grounding |
| OHEM | Focus on ambiguous spatial references |
| Curriculum learning | Start with clear groundings, add ambiguity |
| Tar-balanced sampling | Balance across document types/domains |
| Loss-weighted sampling | More samples for poor alignment scores |

---

## Sources

**Web Research:**
- [Papers With Code - OHEM](https://paperswithcode.com/method/ohem) (accessed 2025-11-20)
- [arXiv - S-OHEM Stratified Mining](https://arxiv.org/abs/1705.02233) (accessed 2025-11-20)
- [ResearchGate - Hard Sample Mining Survey](https://www.researchgate.net/publication/395497039) (accessed 2025-11-20)
- [ACM - Efficient Training of Large-Scale Models](https://dl.acm.org/doi/10.1145/3700439) (accessed 2025-11-20)
- [arXiv - Curriculum Learning Survey](https://arxiv.org/pdf/2101.10382) (accessed 2025-11-20)
- [IEEE - Federated Learning Sampling](https://ieeexplore.ieee.org/iel8/8782661/10362961/10677499.pdf) (accessed 2025-11-20)

**Original Papers:**
- Shrivastava et al., "Training Region-based Object Detectors with Online Hard Example Mining" (OHEM, CVPR 2016)
- Bengio et al., "Curriculum Learning" (ICML 2009)
- Li et al., "S-OHEM: Stratified Online Hard Example Mining" (2017)

**Source Document:**
- SA-1B Dataset Mastery ingestion plan
