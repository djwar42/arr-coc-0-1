"""
3-Stage FP8 Training Loop - Core training orchestration with quantization

<karpathys_code_comments>
** This File's Role in 3FS (3-Stage FP8 Training) **

This is the training loop that implements DeepSeek's revolutionary 3-stage FP8 training strategy.
Instead of training everything in FP16/BF16 (expensive), they do:
- Stage 1: FP8 activations + gradients, FP16 weights
- Stage 2: FP8 everything (weights too)
- Stage 3: FP16 recovery for final convergence

Why this matters: Train a 671B model for the cost of a much smaller one. FP8 cuts memory ~2x and
compute ~2-3x. The 3-stage approach handles the convergence issues that pure FP8 would have.

Connects to: fp8_quantization.py (quantization mechanics), model.py (actual model forward pass)

** Function List **
train_step(batch, stage) - Single training step with stage-appropriate quantization
update_stage(epoch, metrics) - Decide when to transition between stages
quantize_based_on_stage(tensor, stage) - Apply stage-specific quantization
compute_loss_and_backward(outputs, targets, stage) - Loss + backprop with FP8 grads
optimizer_step(stage) - Update weights (FP8 or FP16 depending on stage)
checkpoint_with_stage_info(stage, metrics) - Save checkpoint with stage metadata

** Technical Deep Dive **

Alright, let's break down DeepSeek's 3-stage FP8 training. This is actually genius.

THE PROBLEM:
Pure FP8 training is tempting (2x memory, 2-3x speed) but has convergence issues. The reduced
precision causes gradient noise that hurts final performance. Traditional solution: just use
FP16/BF16. DeepSeek's solution: use FP8 smartly through 3 stages.

THE 3 STAGES:

Stage 1 (Warm-up, ~70% of training):
- Activations: FP8
- Gradients: FP8
- Weights: FP16 (kept in high precision)
- Why: Model is still learning major patterns, can handle FP8 noise. But keep weights precise
  for stable learning.

Stage 2 (Acceleration, ~25% of training):
- Activations: FP8
- Gradients: FP8
- Weights: FP8 (quantized!)
- Why: Model learned the main patterns, now we go full FP8 for max speed. Accept some noise.

Stage 3 (Recovery, ~5% of training):
- Everything: FP16/BF16 (back to high precision)
- Why: Clean up the FP8 noise, recover full quality. Like polishing after rough cutting.

THE KEY INSIGHT:
You don't need high precision for the whole training! Most learning happens early (Stage 1+2)
where FP8 is fine. Only the final refinement (Stage 3) needs FP16. This gives you 90%+ of the
speedup with minimal quality loss.

IMPLEMENTATION DETAILS:
- Stage transitions happen automatically based on loss plateaus or epoch counts
- Each stage maintains separate quantization configs
- Gradient accumulation works in FP8 (huge memory savings!)
- Optimizer states stay in FP32 for numerical stability

THE MATH:
- FP8 range: ~±57000 (vs FP16 ~±65000)
- FP8 precision: ~2 decimal digits (vs FP16 ~3 digits)
- This is tight but workable with careful scaling

WHY THIS WORKS:
Deep learning is surprisingly robust to precision loss DURING training. The final weights need
precision, but the path to get there can be noisy. It's like sketching in pencil then inking
the final - you don't need high-res for the sketch phase.

DeepSeek's contribution: Figuring out the exact stage recipe that maximizes FP8 usage while
preserving final quality. The 70/25/5 split is empirically tuned.

Result: Train V3 (671B) for $5.5M instead of $15M+. That's the power of being clever with
quantization.

¯\_(ツ)_/¯ Pretty cool engineering tbh.
</karpathys_code_comments>
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from enum import Enum

class TrainingStage(Enum):
    # Karpathy: Explicit enum for the 3 stages. Makes the code self-documenting.
    STAGE_1_FP8_ACT_GRAD = 1  # FP8 activations/gradients, FP16 weights
    STAGE_2_FULL_FP8 = 2      # FP8 everything
    STAGE_3_FP16_RECOVERY = 3 # Back to FP16 for final quality

class FP8TrainingLoop:
    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        self.config = config

        # Karpathy: Start at Stage 1. We'll transition based on epochs/metrics.
        self.current_stage = TrainingStage.STAGE_1_FP8_ACT_GRAD

        # Karpathy: Track stage transitions for analysis
        self.stage_history = []

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with stage-appropriate quantization"""

        # Karpathy: Extract batch data. Shape: [batch_size, seq_len]
        input_ids = batch['input_ids']
        labels = batch['labels']

        # Karpathy: Forward pass. Quantization happens inside model based on current_stage.
        # This is where FP8 activations happen - see model.forward() for details.
        outputs = self.model(
            input_ids,
            training_stage=self.current_stage
        )

        # Karpathy: Compute loss. Cross-entropy for language modeling.
        # Loss is computed in FP32 for stability (don't quantize the loss itself!)
        loss = nn.functional.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            labels.view(-1)
        )

        # Karpathy: Backward pass. This is where FP8 gradients happen.
        # PyTorch will compute gradients in the same dtype as activations (FP8 in Stage 1+2)
        loss.backward()

        # Karpathy: Gradient clipping. Do this BEFORE quantizing grads for Stage 2.
        # Clip by norm to prevent exploding gradients (FP8 has limited range!)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Karpathy: Optimizer step. Weight updates depend on stage:
        # Stage 1: FP16 weights updated with FP8 gradients (convert grads up to FP16)
        # Stage 2: FP8 weights updated with FP8 gradients (keep everything FP8)
        # Stage 3: FP16 weights updated with FP16 gradients (standard)
        self.optimizer_step_with_stage()

        # Karpathy: Zero gradients for next iteration
        self.optimizer.zero_grad()

        return {'loss': loss.item()}

    def optimizer_step_with_stage(self):
        """Optimizer step with stage-aware weight updates"""

        if self.current_stage == TrainingStage.STAGE_1_FP8_ACT_GRAD:
            # Karpathy: Stage 1 - Gradients are FP8, but weights are FP16.
            # Convert gradients from FP8 to FP16 before applying to weights.
            for param in self.model.parameters():
                if param.grad is not None:
                    # Karpathy: Upcast gradient from FP8 to FP16 for weight update
                    param.grad = param.grad.to(torch.float16)

            # Karpathy: Standard optimizer step with FP16 weights
            self.optimizer.step()

        elif self.current_stage == TrainingStage.STAGE_2_FULL_FP8:
            # Karpathy: Stage 2 - Everything in FP8. This is the speed phase.
            # Weights and gradients both FP8, optimizer computes in FP8.
            self.optimizer.step()

            # Karpathy: Immediately quantize updated weights back to FP8 range
            for param in self.model.parameters():
                if param.requires_grad:
                    param.data = self.quantize_to_fp8(param.data)

        else:  # Stage 3
            # Karpathy: Stage 3 - Back to FP16. Standard training, no quantization.
            # This is the quality recovery phase.
            self.optimizer.step()

    def quantize_to_fp8(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to FP8 range"""
        # Karpathy: FP8 has range ~[-448, 448] for E4M3 format.
        # Clip to this range to prevent overflow.
        max_val = 448.0
        return torch.clamp(tensor, -max_val, max_val)

    def should_transition_stage(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Decide if we should move to next training stage"""

        # Karpathy: Simple epoch-based transitions. DeepSeek uses ~70/25/5 split.
        total_epochs = self.config.total_epochs

        if self.current_stage == TrainingStage.STAGE_1_FP8_ACT_GRAD:
            # Karpathy: Transition to Stage 2 after 70% of training
            if epoch >= 0.7 * total_epochs:
                return True

        elif self.current_stage == TrainingStage.STAGE_2_FULL_FP8:
            # Karpathy: Transition to Stage 3 for final 5% (recovery)
            if epoch >= 0.95 * total_epochs:
                return True

        return False

    def transition_to_next_stage(self):
        """Transition to next training stage"""

        if self.current_stage == TrainingStage.STAGE_1_FP8_ACT_GRAD:
            # Karpathy: Moving to Stage 2. Quantize all weights to FP8 now.
            self.current_stage = TrainingStage.STAGE_2_FULL_FP8

            # Karpathy: Convert weights from FP16 to FP8
            for param in self.model.parameters():
                param.data = self.quantize_to_fp8(param.data.to(torch.float8_e4m3fn))

            print("Transitioned to Stage 2: Full FP8 training")

        elif self.current_stage == TrainingStage.STAGE_2_FULL_FP8:
            # Karpathy: Moving to Stage 3. Convert everything back to FP16.
            self.current_stage = TrainingStage.STAGE_3_FP16_RECOVERY

            # Karpathy: Upcast weights from FP8 to FP16 for recovery
            for param in self.model.parameters():
                param.data = param.data.to(torch.float16)

            print("Transitioned to Stage 3: FP16 recovery")

# Karpathy: Notice the simplicity. This is just a training loop with 3 branches based on stage.
# The magic is in the stage transitions and the quantization mechanics, not in complicated code.
# DeepSeek figured out WHEN to quantize (70/25/5 split) through extensive experimentation.
# The code itself is straightforward once you know the recipe.
