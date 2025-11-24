"""
Multi-GPU Training Script for ARR-COC.

Uses Accelerate for automatic distributed training across all available GPUs.
Logs to Weights & Biases for real-time monitoring
Auto-uploads checkpoints to HuggingFace Hub.

Usage:
    # Configure once (interactive)
    accelerate config

    # Train on all available GPUs
    accelerate launch Training/train.py --config configs/vqav2.yaml

    # Or single GPU for testing
    python Training/train.py --config configs/vqav2.yaml
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_linear_schedule_with_warmup

import wandb

# ARR-COC components
from ARR_COC import ARRCOCQwen


class ARRCOCConfig:
    """Configuration for ARR-COC training"""

    def __init__(
        self,
        # Model
        base_model: str = "Qwen/Qwen3-VL-2B-Instruct",
        num_visual_tokens: int = 200,
        # Training
        learning_rate: float = 1e-5,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        # Dataset
        dataset_name: str = "HuggingFaceM4/VQAv2",
        max_train_samples: int = None,  # None = use all
        # Checkpointing
        save_every_n_steps: int = 500,
        output_dir: str = "./checkpoints",
        hub_repo_id: str = None,  # e.g., "username/arr-coc-0-1"
        # W&B
        wandb_project: str = "arr-coc-0-1",
        wandb_run_name: str = None,
        # Misc
        seed: int = 42,
    ):
        # Allow overriding from environment (Launch injects these)
        self.base_model = os.getenv("BASE_MODEL", base_model)
        self.num_visual_tokens = int(os.getenv("NUM_VISUAL_TOKENS", num_visual_tokens))
        self.learning_rate = float(os.getenv("LEARNING_RATE", learning_rate))
        self.batch_size = int(os.getenv("BATCH_SIZE", batch_size))
        self.gradient_accumulation_steps = int(
            os.getenv("GRADIENT_ACCUMULATION_STEPS", gradient_accumulation_steps)
        )
        self.num_epochs = int(os.getenv("NUM_EPOCHS", num_epochs))
        self.max_train_samples = (
            int(os.getenv("MAX_TRAIN_SAMPLES", max_train_samples or 0)) or None
        )
        self.output_dir = os.getenv("OUTPUT_DIR", output_dir)
        self.hub_repo_id = os.getenv("HUB_REPO_ID", hub_repo_id)
        self.wandb_project = os.getenv("WANDB_PROJECT", wandb_project)
        # Generate W&B-style cool names with timestamp (ethereal-snowflake-1234567890)
        if wandb_run_name is None and "WANDB_RUN_NAME" not in os.environ:
            import wandb

            # Let W&B generate the cool name, then append timestamp
            temp_name = (
                wandb.util.generate_id()
            )  # Generates things like "ethereal-snowflake"
            self.wandb_run_name = f"{temp_name}-{int(time.time())}"
        else:
            self.wandb_run_name = os.getenv("WANDB_RUN_NAME", wandb_run_name)
        self.seed = int(os.getenv("SEED", seed))

        # Save checkpoint config
        self.save_every_n_steps = int(
            os.getenv("SAVE_EVERY_N_STEPS", save_every_n_steps)
        )

        # Warmup and grad clipping
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.dataset_name = dataset_name


def load_vqav2_dataset(config: ARRCOCConfig, processor):
    """Load and prepare VQAv2 dataset"""
    print(f"Loading dataset: {config.dataset_name}")

    # SECURITY: Pin to specific verified commit to prevent supply-chain attacks
    # VQAv2 commit 4b98c86 verified 2021-04-14 (clean, no malicious code)
    VQAV2_COMMIT = "4b98c864262e9db184eb039e85e97e6630825b6a"

    # trust_remote_code required (VQAv2 has no Parquet alternative)
    # Mitigated by: pinned revision + isolated cache + container sandboxing
    dataset = load_dataset(
        config.dataset_name,
        split="train",
        trust_remote_code=True,  # Required - no alternative exists
        revision=VQAV2_COMMIT,  # Security: Pinned to verified commit
        cache_dir="/tmp/hf_cache",  # Isolated cache directory
    )

    if config.max_train_samples:
        dataset = dataset.select(range(config.max_train_samples))
        print(f"Using {config.max_train_samples} training examples")

    def preprocess_function(examples):
        """Preprocess VQAv2 examples"""
        # TODO: Implement proper preprocessing for your model
        # This is a placeholder
        processed = processor(
            images=examples["image"],
            text=examples["question"],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Add labels (for language modeling loss)
        processed["labels"] = processed["input_ids"].clone()

        return processed

    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    dataset.set_format(type="torch")

    return dataset


class ARRCOCTrainer:
    """Trainer for ARR-COC with multi-GPU support"""

    def __init__(self, config: ARRCOCConfig):
        self.config = config

        # Initialize Accelerator (auto-detects multi-GPU)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="wandb" if config.wandb_project else None,
        )

        # Set seed for reproducibility
        set_seed(config.seed)

        # Initialize W&B (only on main process)
        if self.accelerator.is_main_process and config.wandb_project:
            # IMPORTANT: Force W&B dir to Training/wandb/ (not project root!)
            import os

            wandb_dir = os.path.join(os.path.dirname(__file__), "wandb")
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                job_type="train",  # Launch uses this
                tags=["vertex-ai", "arr-coc", "v0.1", "spot-instance"],  # For filtering
                dir=wandb_dir,
                config={
                    "base_model": config.base_model,
                    "num_visual_tokens": config.num_visual_tokens,
                    "learning_rate": config.learning_rate,
                    "batch_size": config.batch_size,
                    "gradient_accumulation": config.gradient_accumulation_steps,
                    "num_epochs": config.num_epochs,
                    "dataset": config.dataset_name,
                    "seed": config.seed,
                    "output_dir": config.output_dir,  # Log where checkpoints go
                },
            )

        # Load ARR-COC model
        print(f"Loading ARR-COC model: {config.base_model}")
        self.model = ARRCOCQwen(
            base_model=config.base_model,
            num_visual_tokens=config.num_visual_tokens,
            freeze_base=True,  # Only train ARR-COC components for MVP
        )

        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(config.base_model)

        # Load dataset
        self.train_dataset = load_vqav2_dataset(config, self.processor)

        # Create dataloader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # Create learning rate scheduler
        num_training_steps = len(self.train_dataloader) * config.num_epochs
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Prepare everything for distributed training
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.scheduler,
        )

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize HuggingFace Hub API (for checkpoint uploads)
        self.hf_api = HfApi() if config.hub_repo_id else None

        print(f"✓ Trainer initialized")
        print(f"  Devices: {self.accelerator.num_processes}")
        print(f"  Training samples: {len(self.train_dataset)}")
        print(f"  Steps per epoch: {len(self.train_dataloader)}")
        print(f"  Total steps: {num_training_steps}")

    def train(self):
        """Main training loop"""
        config = self.config
        global_step = 0

        for epoch in range(config.num_epochs):
            self.model.train()
            epoch_loss = 0

            for step, batch in enumerate(self.train_dataloader):
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss

                # Backward pass (handles gradient accumulation)
                self.accelerator.backward(loss)

                # Gradient clipping
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), config.max_grad_norm
                    )

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # Track metrics
                epoch_loss += loss.item()
                global_step += 1

                # Log to W&B (only on main process)
                if self.accelerator.is_main_process and global_step % 10 == 0:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/perplexity": torch.exp(loss).item(),
                            "train/learning_rate": self.scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                            "train/step": global_step,
                            "system/gpu_memory_gb": torch.cuda.max_memory_allocated()
                            / 1e9,
                        }
                    )

                # Print progress
                if step % 50 == 0:
                    print(
                        f"Epoch {epoch} | Step {step}/{len(self.train_dataloader)} | "
                        f"Loss: {loss.item():.4f} | LR: {self.scheduler.get_last_lr()[0]:.2e}"
                    )

                # Save checkpoint
                if global_step % config.save_every_n_steps == 0:
                    self.save_checkpoint(f"step-{global_step}")

            # End of epoch
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            print(f"\n✓ Epoch {epoch} complete | Avg Loss: {avg_epoch_loss:.4f}\n")

            # Save epoch checkpoint
            self.save_checkpoint(f"epoch-{epoch}")

        # Training complete
        print("\n🎉 Training complete!")
        self.save_checkpoint("final")

        if self.accelerator.is_main_process:
            wandb.finish()

    def save_checkpoint(self, checkpoint_name: str):
        """Save checkpoint and upload to HuggingFace Hub"""
        if not self.accelerator.is_main_process:
            return

        checkpoint_dir = Path(self.config.output_dir) / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"💾 Saving checkpoint: {checkpoint_name}")

        # Unwrap model (remove distributed wrapper)
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Save model
        unwrapped_model.save_pretrained(checkpoint_dir)
        self.processor.save_pretrained(checkpoint_dir)

        # Upload to HuggingFace Hub (if configured)
        if self.config.hub_repo_id and self.hf_api:
            try:
                print(f"  Uploading to Hub: {self.config.hub_repo_id}")
                self.hf_api.upload_folder(
                    folder_path=str(checkpoint_dir),
                    repo_id=self.config.hub_repo_id,
                    path_in_repo=f"checkpoints/{checkpoint_name}",
                    commit_message=f"Checkpoint: {checkpoint_name}",
                )
                print(f"  ✓ Uploaded to Hub")
            except Exception as e:
                print(f"  ⚠️  Hub upload failed: {e}")

        print(f"✓ Checkpoint saved: {checkpoint_dir}")


def main():
    # =====================================================================
    # GPU VALIDATION - SAFETY NET (Layer 3 of 3)
    # =====================================================================
    # This is the FINAL defense layer that catches silent CPU fallback!
    #
    # Defense Architecture:
    #   Layer 1: GPU string validation in launch/core.py (FUTURE)
    #            → Validates GPU name against known-good list
    #            → Fails immediately at launch time (< 10s)
    #            → Prevents wasted Cloud Run execution time
    #
    #   Layer 2: GPU quota check in launch/core.py (CURRENT)
    #            → Checks GCP quota for requested GPU type
    #            → Catches quota exhaustion before job submission
    #            → NOTE: Currently has fallback that allows invalid names!
    #
    #   Layer 3: Runtime GPU validation HERE (CURRENT - SAFETY NET!)
    #            → Detects GCP silent CPU fallback at container startup
    #            → Catches invalid GPU names that bypassed Layer 1+2
    #            → Prevents 60-minute CPU training on wrong hardware
    #            → Duration: ~5 min bailout instead of 60 min timeout
    #
    # Why This Layer is Critical:
    #   - GCP doesn't always error on invalid GPU names
    #   - Sometimes it just silently uses CPU instead
    #   - Layer 1+2 might have bugs or be bypassed
    #   - This is the last chance to catch the issue!
    #
    # TODO: When Layer 1 (GPU string validation) is implemented:
    #       - This layer becomes pure safety net (shouldn't trigger often)
    #       - Keep it anyway! Defense in depth prevents silent failures
    #       - Update comments to reflect it's now backup validation
    # =====================================================================

    # GPU validation: Check if GPU was requested but we're running on CPU
    requested_gpu = os.getenv("TRAINING_GPU")
    if requested_gpu and requested_gpu != "NONE":
        print(f"🔍 GPU Validation: Requested {requested_gpu}")
        if not torch.cuda.is_available():
            print("")
            print("=" * 80)
            print("🚨 FATAL ERROR: GPU REQUESTED BUT RUNNING ON CPU!")
            print(f"   Requested GPU: {requested_gpu}")
            print(f"   CUDA Available: {torch.cuda.is_available()}")
            print(f"   Visible Devices: {torch.cuda.device_count()}")
            print("")
            print("This indicates GCP silently fell back to CPU.")
            print("Possible causes:")
            print("  - Invalid GPU type name (not recognized by GCP)")
            print("  - GPU quota exhausted in this region")
            print("  - Validation bypassed (machine type auto-selection failed)")
            print("=" * 80)
            print("")
            raise RuntimeError(
                f"GPU validation failed: {requested_gpu} requested but torch.cuda.is_available()={torch.cuda.is_available()}"
            )
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            print(f"✅ GPU Validation Passed: {gpu_count}x {gpu_name}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config YAML (TODO)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--hub_repo_id", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="arr-coc-0-1")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    # Create config
    config = ARRCOCConfig(
        base_model=args.base_model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_epochs=args.num_epochs,
        max_train_samples=args.max_train_samples,
        output_dir=args.output_dir,
        hub_repo_id=args.hub_repo_id,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    # Create trainer and train
    trainer = ARRCOCTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
