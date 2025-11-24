# Training SAM from Scratch on SA-1B

## Overview

Training the Segment Anything Model (SAM) from scratch requires massive computational resources and careful orchestration of the training pipeline. This document covers the complete training recipe used by Meta AI to train SAM on the SA-1B dataset, including hardware requirements, hyperparameters, and optimization strategies.

**Key Training Facts:**
- 256 A100 GPUs for training
- ~68 hours training time for ViT-H model
- 11M images with 1.1B masks
- MAE pre-trained ViT as image encoder backbone
- AdamW optimizer with learning rate warmup

## Sources

**Primary Sources:**
- [Segment Anything (arXiv:2304.02643)](https://arxiv.org/abs/2304.02643) - Original SAM paper by Kirillov et al.
- [SAM: Segment Anything with Prompts, Not Labels](https://medium.com/@kdk199604/sam-segment-anything-with-prompts-not-labels-7a85e6ec4d09) - Detailed review by Dong-Keon Kim

**Code References:**
- [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) - Official SAM repository
- [mazurowski-lab/finetune-SAM](https://github.com/mazurowski-lab/finetune-SAM) - Fine-tuning reference implementation

---

## Training Recipe Overview

### Hardware Requirements

```python
# SAM Training Hardware Configuration
training_config = {
    # GPU Setup
    "num_gpus": 256,
    "gpu_type": "NVIDIA A100",
    "gpu_memory": "80GB",  # A100-80GB recommended

    # Training Duration
    "training_time_hours": 68,  # For ViT-H
    "total_gpu_hours": 256 * 68,  # ~17,408 GPU-hours

    # Memory Requirements
    "per_gpu_batch_size": 2,  # Images per GPU
    "effective_batch_size": 256 * 2,  # 512 total

    # Storage
    "dataset_size_tb": 2.5,  # SA-1B compressed
    "checkpoint_size_gb": 2.4,  # ViT-H checkpoint
}

# Estimated cloud costs (approximate)
cloud_costs = {
    "aws_p4d_24xlarge": "$32/hour per instance",
    "total_instances": 32,  # 8 GPUs each
    "estimated_cost": "$70,000-100,000",  # Full training run
}
```

### Model Architecture Variants

SAM comes in three sizes, each with different training requirements:

```python
# SAM Model Variants
model_variants = {
    "vit_h": {
        "name": "ViT-Huge",
        "encoder_params": "632M",
        "total_params": "641M",
        "image_encoder": {
            "embed_dim": 1280,
            "depth": 32,
            "num_heads": 16,
            "global_attn_indexes": [7, 15, 23, 31],
        },
        "training_time": "68 hours",
        "checkpoint_url": "sam_vit_h_4b8939.pth",
    },
    "vit_l": {
        "name": "ViT-Large",
        "encoder_params": "308M",
        "total_params": "312M",
        "image_encoder": {
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "global_attn_indexes": [5, 11, 17, 23],
        },
        "training_time": "~40 hours",
        "checkpoint_url": "sam_vit_l_0b3195.pth",
    },
    "vit_b": {
        "name": "ViT-Base",
        "encoder_params": "91M",
        "total_params": "93M",
        "image_encoder": {
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "global_attn_indexes": [2, 5, 8, 11],
        },
        "training_time": "~20 hours",
        "checkpoint_url": "sam_vit_b_01ec64.pth",
    },
}
```

---

## MAE Pre-training for Image Encoder

### What is MAE?

Masked Autoencoder (MAE) pre-training provides the foundation for SAM's image encoder. The ViT is first pre-trained on ImageNet using self-supervised learning before being adapted for SAM.

```python
# MAE Pre-training Configuration
mae_pretraining = {
    "method": "Masked Autoencoder",
    "dataset": "ImageNet-1K",
    "num_images": 1_281_167,

    # Masking Strategy
    "mask_ratio": 0.75,  # 75% of patches masked
    "patch_size": 16,

    # Training
    "epochs": 1600,
    "base_lr": 1.5e-4,
    "warmup_epochs": 40,
    "batch_size": 4096,

    # Benefits for SAM
    "benefits": [
        "Rich visual representations",
        "Efficient training convergence",
        "Better generalization",
        "Reduced labeled data requirements",
    ],
}
```

### Adapting MAE ViT for High-Resolution

SAM adapts the MAE-pretrained ViT to handle high-resolution inputs (1024x1024):

```python
import torch
import torch.nn as nn
from functools import partial

class ImageEncoderViT(nn.Module):
    """
    SAM Image Encoder based on MAE-pretrained ViT.
    Adapted for high-resolution inputs with windowed attention.
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1280,  # ViT-H
        depth: int = 32,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        act_layer = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = True,
        rel_pos_zero_init: bool = True,
        window_size: int = 14,
        global_attn_indexes: tuple = (7, 15, 23, 31),
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Positional embeddings
        self.pos_embed = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size,
                           img_size // patch_size, embed_dim)
            )

        # Transformer blocks with windowed attention
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        # Neck to reduce channel dimension
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))
        return x
```

---

## Optimizer and Training Configuration

### AdamW Optimizer Setup

```python
import torch.optim as optim

def configure_optimizer(model, config):
    """
    Configure AdamW optimizer for SAM training.

    From SAM paper appendix:
    - AdamW optimizer
    - Weight decay: 0.1
    - Learning rate with warmup and decay
    """

    # Separate parameters for different weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No weight decay for biases and LayerNorm
        if 'bias' in name or 'norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = optim.AdamW([
        {'params': decay_params, 'weight_decay': 0.1},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=config['base_lr'], betas=(0.9, 0.999))

    return optimizer

# Training hyperparameters
training_hyperparams = {
    # Optimizer
    "optimizer": "AdamW",
    "base_lr": 8e-4,
    "weight_decay": 0.1,
    "betas": (0.9, 0.999),

    # Learning Rate Schedule
    "lr_schedule": "step_decay",
    "warmup_iterations": 250,
    "lr_decay_factor": 10,
    "lr_decay_iterations": [60000, 86666],

    # Batch Size
    "batch_size_per_gpu": 2,
    "num_gpus": 256,
    "effective_batch_size": 512,

    # Training Duration
    "total_iterations": 90000,  # ~68 hours on 256 A100s

    # Data Augmentation
    "augmentation": {
        "horizontal_flip": True,
        "scale_jitter": [0.1, 2.0],
        "crop_size": 1024,
    },
}
```

### Learning Rate Schedule

```python
import math

class SAMScheduler:
    """
    Learning rate scheduler for SAM training.
    Warmup followed by step decay.
    """

    def __init__(
        self,
        optimizer,
        base_lr: float = 8e-4,
        warmup_iters: int = 250,
        decay_iters: list = [60000, 86666],
        decay_factor: float = 10,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_iters = warmup_iters
        self.decay_iters = decay_iters
        self.decay_factor = decay_factor
        self.current_iter = 0

    def step(self):
        self.current_iter += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self):
        # Warmup phase
        if self.current_iter < self.warmup_iters:
            return self.base_lr * self.current_iter / self.warmup_iters

        # Step decay
        lr = self.base_lr
        for decay_iter in self.decay_iters:
            if self.current_iter >= decay_iter:
                lr /= self.decay_factor

        return lr

# Usage example
scheduler = SAMScheduler(
    optimizer,
    base_lr=8e-4,
    warmup_iters=250,
    decay_iters=[60000, 86666],
    decay_factor=10
)
```

---

## Loss Functions

### Combined Focal and Dice Loss

SAM uses a combination of focal loss and dice loss for mask prediction:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMLoss(nn.Module):
    """
    Combined loss for SAM training.

    Loss = focal_loss + dice_loss + iou_loss

    Focal loss handles class imbalance (background vs foreground).
    Dice loss ensures good overlap with ground truth masks.
    IoU loss supervises the IoU prediction head.
    """

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_weight: float = 1.0,
        focal_weight: float = 20.0,
        iou_weight: float = 1.0,
    ):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.iou_weight = iou_weight

    def forward(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        pred_ious: torch.Tensor,
        num_masks: int = 3,
    ):
        """
        Compute loss with multi-mask ambiguity handling.

        Args:
            pred_masks: (B, num_masks, H, W) predicted mask logits
            gt_masks: (B, 1, H, W) ground truth masks
            pred_ious: (B, num_masks) predicted IoU scores
            num_masks: number of predicted masks per prompt

        Returns:
            Total loss (backprop through best-matching mask only)
        """
        batch_size = pred_masks.shape[0]

        # Compute loss for each predicted mask
        losses = []
        for i in range(num_masks):
            mask_pred = pred_masks[:, i:i+1]

            focal = self._focal_loss(mask_pred, gt_masks)
            dice = self._dice_loss(mask_pred, gt_masks)

            mask_loss = self.focal_weight * focal + self.dice_weight * dice
            losses.append(mask_loss)

        # Stack losses: (B, num_masks)
        losses = torch.stack(losses, dim=1)

        # Select minimum loss mask for each sample (ambiguity handling)
        min_losses, min_indices = losses.min(dim=1)

        # IoU loss for the selected masks
        gt_ious = self._compute_iou(pred_masks, gt_masks)
        selected_pred_ious = pred_ious.gather(1, min_indices.unsqueeze(1))
        selected_gt_ious = gt_ious.gather(1, min_indices.unsqueeze(1))
        iou_loss = F.mse_loss(selected_pred_ious, selected_gt_ious)

        total_loss = min_losses.mean() + self.iou_weight * iou_loss

        return total_loss

    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor):
        """Sigmoid focal loss for binary segmentation."""
        pred_sigmoid = torch.sigmoid(pred)

        # Binary cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )

        # Focal weighting
        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        focal_weight = (1 - p_t) ** self.focal_gamma

        # Alpha weighting
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)

        focal_loss = alpha_t * focal_weight * ce_loss

        return focal_loss.mean()

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor):
        """Dice loss for mask overlap."""
        pred_sigmoid = torch.sigmoid(pred)

        # Flatten
        pred_flat = pred_sigmoid.flatten(1)
        target_flat = target.flatten(1)

        # Dice coefficient
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2 * intersection + 1) / (union + 1)

        return 1 - dice.mean()

    def _compute_iou(self, pred: torch.Tensor, target: torch.Tensor):
        """Compute IoU between predictions and targets."""
        pred_binary = (torch.sigmoid(pred) > 0.5).float()
        target = target.expand_as(pred_binary)

        intersection = (pred_binary * target).sum(dim=(2, 3))
        union = pred_binary.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection

        iou = (intersection + 1) / (union + 1)
        return iou
```

---

## Prompt Encoder Training

### Simulated Prompt Training

SAM is trained with simulated prompts over 11 rounds per mask:

```python
import numpy as np
import torch

class PromptSimulator:
    """
    Simulate interactive prompting during training.

    SAM training simulates 11 rounds of prompts per mask:
    - First prompt: random foreground point
    - Subsequent prompts: error-correcting points

    This teaches SAM to handle ambiguous prompts and
    iterative refinement.
    """

    def __init__(self, num_rounds: int = 11):
        self.num_rounds = num_rounds

    def generate_point_prompts(
        self,
        gt_mask: torch.Tensor,
        pred_mask: torch.Tensor = None,
    ):
        """
        Generate point prompts for training.

        Args:
            gt_mask: Ground truth binary mask (H, W)
            pred_mask: Previous prediction for error correction

        Returns:
            point_coords: (N, 2) point coordinates
            point_labels: (N,) 1 for foreground, 0 for background
        """
        if pred_mask is None:
            # First round: sample random foreground point
            fg_points = torch.nonzero(gt_mask)
            if len(fg_points) > 0:
                idx = np.random.randint(len(fg_points))
                point = fg_points[idx]
                return point.unsqueeze(0), torch.tensor([1])
            else:
                # No foreground, sample background
                h, w = gt_mask.shape
                point = torch.tensor([h // 2, w // 2])
                return point.unsqueeze(0), torch.tensor([0])

        # Subsequent rounds: sample from error regions
        error_mask = (pred_mask > 0.5) != gt_mask

        # False negatives (should be foreground but predicted background)
        fn_mask = error_mask & gt_mask
        # False positives (should be background but predicted foreground)
        fp_mask = error_mask & ~gt_mask

        points = []
        labels = []

        # Sample from false negatives (add foreground point)
        if fn_mask.any():
            fn_points = torch.nonzero(fn_mask)
            idx = np.random.randint(len(fn_points))
            points.append(fn_points[idx])
            labels.append(1)

        # Sample from false positives (add background point)
        if fp_mask.any():
            fp_points = torch.nonzero(fp_mask)
            idx = np.random.randint(len(fp_points))
            points.append(fp_points[idx])
            labels.append(0)

        if points:
            return torch.stack(points), torch.tensor(labels)
        else:
            return None, None

    def generate_box_prompt(self, gt_mask: torch.Tensor):
        """
        Generate bounding box prompt from mask.

        Returns:
            box: (4,) tensor [x1, y1, x2, y2]
        """
        fg_points = torch.nonzero(gt_mask)
        if len(fg_points) == 0:
            return None

        y_min, x_min = fg_points.min(dim=0).values
        y_max, x_max = fg_points.max(dim=0).values

        # Add small random jitter for robustness
        jitter = np.random.randint(-5, 6)
        box = torch.tensor([
            max(0, x_min + jitter),
            max(0, y_min + jitter),
            min(gt_mask.shape[1], x_max - jitter),
            min(gt_mask.shape[0], y_max - jitter),
        ])

        return box

def training_step(model, batch, loss_fn, prompt_simulator):
    """
    Single training step with simulated prompts.
    """
    images = batch['images']  # (B, 3, 1024, 1024)
    gt_masks = batch['masks']  # (B, 1, H, W)

    # Encode images once
    image_embeddings = model.image_encoder(images)

    total_loss = 0

    for round_idx in range(prompt_simulator.num_rounds):
        # Generate prompts
        if round_idx == 0:
            # First round: random point or box
            points, labels = prompt_simulator.generate_point_prompts(gt_masks)
            boxes = None
        else:
            # Subsequent rounds: error-correcting points
            with torch.no_grad():
                prev_masks = torch.sigmoid(prev_low_res_masks)
            points, labels = prompt_simulator.generate_point_prompts(
                gt_masks, prev_masks
            )
            boxes = None

        # Encode prompts
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=(points, labels) if points is not None else None,
            boxes=boxes,
            masks=None,
        )

        # Decode masks
        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )

        # Compute loss
        loss = loss_fn(low_res_masks, gt_masks, iou_predictions)
        total_loss += loss

        prev_low_res_masks = low_res_masks

    return total_loss / prompt_simulator.num_rounds
```

---

## Mask Decoder Architecture

### Lightweight Transformer Decoder

The mask decoder is intentionally lightweight (~4M parameters) for fast inference:

```python
import torch
import torch.nn as nn

class MaskDecoder(nn.Module):
    """
    SAM Mask Decoder - predicts masks from image and prompt embeddings.

    Architecture:
    - 2 transformer decoder blocks
    - Bidirectional cross-attention (prompt <-> image)
    - Dynamic mask prediction via MLP

    This is the component to fine-tune for downstream tasks.
    """

    def __init__(
        self,
        transformer_dim: int = 256,
        num_multimask_outputs: int = 3,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_multimask_outputs = num_multimask_outputs

        # Transformer decoder
        self.transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=transformer_dim,
            mlp_dim=2048,
            num_heads=8,
        )

        # Output tokens
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1  # +1 for single mask
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # Upscaling network
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4,
                              kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8,
                              kernel_size=2, stride=2),
            nn.GELU(),
        )

        # Mask prediction heads (one per output mask)
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for _ in range(self.num_mask_tokens)
        ])

        # IoU prediction head
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim,
            self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ):
        """
        Forward pass for mask prediction.

        Args:
            image_embeddings: (B, C, H, W) from image encoder
            image_pe: (B, C, H, W) positional encoding
            sparse_prompt_embeddings: (B, N, C) from prompt encoder
            dense_prompt_embeddings: (B, C, H, W) from prompt encoder
            multimask_output: if True, return 3 masks; else return 1

        Returns:
            masks: (B, num_masks, H*4, W*4) mask logits
            iou_pred: (B, num_masks) predicted IoU scores
        """
        # Concatenate output tokens with prompt tokens
        output_tokens = torch.cat([
            self.iou_token.weight,
            self.mask_tokens.weight
        ], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.shape[0], -1, -1
        )
        tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)

        # Add dense embeddings to image embeddings
        src = image_embeddings + dense_prompt_embeddings
        pos_src = image_pe
        b, c, h, w = src.shape

        # Run transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:1+self.num_mask_tokens, :]

        # Upscale image embeddings
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        # Predict masks
        hyper_in_list = []
        for i, mlp in enumerate(self.output_hypernetworks_mlps):
            hyper_in_list.append(mlp(mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(
            b, -1, h, w
        )

        # Predict IoU
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)

        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred
```

---

## Batch Size Scaling

### Multi-GPU Training with DDP

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed_training(rank, world_size):
    """
    Initialize distributed training.

    SAM uses 256 GPUs with DDP.
    """
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def create_distributed_dataloader(dataset, batch_size, world_size, rank):
    """
    Create distributed dataloader for SAM training.
    """
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader

# Gradient accumulation for larger effective batch sizes
class GradientAccumulator:
    """
    Accumulate gradients across multiple steps.

    Useful when GPU memory is limited but larger batch
    sizes are desired.
    """

    def __init__(self, accumulation_steps: int = 4):
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def should_step(self):
        self.step_count += 1
        return self.step_count % self.accumulation_steps == 0

    def scale_loss(self, loss):
        return loss / self.accumulation_steps
```

---

## Checkpoint Strategy

### Saving and Loading Checkpoints

```python
import torch
import os

class CheckpointManager:
    """
    Manage training checkpoints for SAM.
    """

    def __init__(self, save_dir: str, max_checkpoints: int = 5):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(save_dir, exist_ok=True)
        self.checkpoints = []

    def save(
        self,
        model,
        optimizer,
        scheduler,
        iteration: int,
        loss: float,
    ):
        """Save checkpoint with all training state."""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
        }

        path = os.path.join(
            self.save_dir,
            f'checkpoint_iter_{iteration}.pth'
        )
        torch.save(checkpoint, path)

        self.checkpoints.append(path)

        # Remove old checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            old_path = self.checkpoints.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)

        print(f"Saved checkpoint: {path}")

    def load(self, path: str, model, optimizer=None, scheduler=None):
        """Load checkpoint and restore training state."""
        checkpoint = torch.load(path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint['iteration'], checkpoint.get('loss', 0)

# Official checkpoint URLs
CHECKPOINT_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}
```

---

## Complete Training Script

### End-to-End Training Pipeline

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def train_sam(
    rank: int,
    world_size: int,
    config: dict,
):
    """
    Complete SAM training pipeline.

    Requires:
    - 256 A100 GPUs (or equivalent)
    - SA-1B dataset
    - ~68 hours for ViT-H
    """
    # Setup
    setup_distributed_training(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # Create model
    model = build_sam_vit_h().to(device)
    model = DDP(model, device_ids=[rank])

    # Load MAE pretrained weights for image encoder
    if config['mae_checkpoint']:
        load_mae_weights(model.module.image_encoder, config['mae_checkpoint'])

    # Create dataset and dataloader
    dataset = SA1BDataset(config['data_dir'], config['transforms'])
    dataloader = create_distributed_dataloader(
        dataset,
        config['batch_size_per_gpu'],
        world_size,
        rank
    )

    # Optimizer and scheduler
    optimizer = configure_optimizer(model, config)
    scheduler = SAMScheduler(
        optimizer,
        base_lr=config['base_lr'],
        warmup_iters=config['warmup_iters'],
        decay_iters=config['decay_iters'],
    )

    # Loss and prompt simulator
    loss_fn = SAMLoss()
    prompt_simulator = PromptSimulator(num_rounds=11)

    # Checkpoint manager
    ckpt_manager = CheckpointManager(config['checkpoint_dir'])

    # Resume from checkpoint if available
    start_iter = 0
    if config['resume_checkpoint']:
        start_iter, _ = ckpt_manager.load(
            config['resume_checkpoint'],
            model.module, optimizer, scheduler
        )

    # Training loop
    model.train()
    data_iter = iter(dataloader)

    for iteration in range(start_iter, config['total_iterations']):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        loss = training_step(model, batch, loss_fn, prompt_simulator)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        # Logging
        if rank == 0 and iteration % 100 == 0:
            print(f"Iter {iteration}: loss={loss.item():.4f}, "
                  f"lr={scheduler._get_lr():.6f}")

        # Save checkpoint
        if rank == 0 and iteration % 10000 == 0:
            ckpt_manager.save(
                model.module, optimizer, scheduler,
                iteration, loss.item()
            )

    # Final checkpoint
    if rank == 0:
        ckpt_manager.save(
            model.module, optimizer, scheduler,
            config['total_iterations'], loss.item()
        )

    dist.destroy_process_group()

# Training configuration
TRAINING_CONFIG = {
    'data_dir': '/path/to/sa1b',
    'checkpoint_dir': '/path/to/checkpoints',
    'mae_checkpoint': '/path/to/mae_pretrain_vit_huge.pth',
    'resume_checkpoint': None,

    # Hyperparameters
    'base_lr': 8e-4,
    'weight_decay': 0.1,
    'batch_size_per_gpu': 2,
    'total_iterations': 90000,
    'warmup_iters': 250,
    'decay_iters': [60000, 86666],

    # Data augmentation
    'transforms': {
        'horizontal_flip': True,
        'scale_jitter': [0.1, 2.0],
        'crop_size': 1024,
    },
}
```

---

## ARR-COC Relevance: Foundation Model Training Patterns

### Training Patterns for VLM Development

The SAM training recipe provides several patterns relevant to training vision-language models:

```python
# Key patterns from SAM training applicable to VLM:

vlm_training_patterns = {
    # 1. Encoder-Decoder Architecture
    "architecture": {
        "heavy_encoder": "Pre-trained vision encoder (frozen or fine-tuned)",
        "lightweight_decoder": "Task-specific decoder for fast inference",
        "benefit": "Amortized computation across multiple queries",
    },

    # 2. Multi-task Training
    "multi_task": {
        "prompting_simulation": "Train with diverse prompt types",
        "ambiguity_handling": "Multiple outputs for ambiguous inputs",
        "benefit": "Robust generalization to new tasks",
    },

    # 3. Large-scale Pre-training
    "pretraining": {
        "mae_style": "Self-supervised pre-training on large data",
        "transfer": "Fine-tune for downstream tasks",
        "benefit": "Rich representations without labeled data",
    },

    # 4. Efficient Training
    "efficiency": {
        "gradient_checkpointing": "Trade compute for memory",
        "mixed_precision": "FP16/BF16 for faster training",
        "distributed": "Multi-GPU with DDP",
    },
}

# Application to ARR-COC relevance scoring
relevance_training_insights = {
    "encoder_reuse": """
        Like SAM reuses image embeddings across prompts,
        ARR-COC can cache document embeddings for multiple queries
    """,

    "lightweight_heads": """
        Relevance scoring heads should be lightweight like SAM's
        mask decoder for fast inference on many documents
    """,

    "multi_output": """
        Similar to SAM's 3-mask output for ambiguity,
        relevance models can provide multiple interpretations
        with confidence scores
    """,
}
```

---

## Summary

Training SAM from scratch requires:

1. **Massive compute**: 256 A100 GPUs, ~68 hours for ViT-H
2. **MAE pre-training**: Self-supervised learning on ImageNet
3. **Careful optimization**: AdamW with warmup and step decay
4. **Multi-round training**: 11 rounds of simulated prompts per mask
5. **Ambiguity-aware loss**: Backprop through best-matching mask only
6. **Efficient architecture**: Heavy encoder + lightweight decoder

The training recipe demonstrates several patterns applicable to foundation model development for vision-language tasks, particularly the encoder-decoder decomposition and multi-task prompt simulation.

---

## Additional Resources

- [SAM Paper (arXiv:2304.02643)](https://arxiv.org/abs/2304.02643)
- [Official SAM Repository](https://github.com/facebookresearch/segment-anything)
- [SA-1B Dataset](https://segment-anything.com/dataset/index.html)
- [MAE Paper](https://arxiv.org/abs/2111.06377)
- [Hugging Face SAM](https://huggingface.co/facebook/sam-vit-huge)
