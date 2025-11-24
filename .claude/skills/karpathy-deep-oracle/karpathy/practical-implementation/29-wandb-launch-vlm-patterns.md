# W&B Launch: VLM-Specific Training Automation & ARR-COC Integration

**Complete guide to automating Vision-Language Model training with W&B Launch, featuring ARR-COC production patterns.**

---

## Overview

This guide covers VLM-specific automation patterns for W&B Launch, focusing on the unique challenges of multimodal training: high GPU memory requirements, multi-stage pipelines, dataset management, and evaluation orchestration. Includes complete ARR-COC training automation examples.

**Why VLMs Need Specialized Launch Patterns:**
- **GPU Memory**: VLMs require 16-80GB per model (vision encoder + LLM + projection layers)
- **Multi-Stage Training**: Pre-training → SFT → Alignment (each stage different resource needs)
- **Multimodal Data**: Images + text require specialized data loading and preprocessing
- **Long Training Times**: Days to weeks, requiring robust checkpointing and resumption
- **Complex Evaluation**: VQA, captioning, multimodal reasoning require automated evaluation jobs

**This Guide:**
- Section 1: VLM Training Automation (resource management, checkpoint strategies)
- Section 2: Multi-Stage Training Pipelines (pre-training → SFT → alignment)
- Section 3: ARR-COC Production Automation (complete relevance realization pipeline)

---

## Section 1: VLM Training Automation (~150 lines)

### 1.1 VLM Resource Requirements

**Typical VLM Memory Breakdown:**
```
Vision Encoder (e.g., SigLIP): ~1-2GB
Language Model (7B): ~14GB (bf16)
Projection Layers: ~100MB
Optimizer States (AdamW): 2× model size
Gradients: 1× model size
Activations: 4-8GB (batch size dependent)

Total for 7B VLM: 40-50GB minimum
```

**GPU Selection for Launch Jobs:**
```yaml
# launch-config-vlm.yaml
resources:
  gpu_type: "a100-40gb"  # Minimum for 7B VLMs
  num_gpus: 1            # Single GPU for fine-tuning
  memory: "80GB"         # System RAM for data loading
  cpu: 16                # Cores for preprocessing
```

**Multi-GPU Training for Larger Models:**
```yaml
# launch-config-vlm-distributed.yaml
resources:
  gpu_type: "a100-80gb"
  num_gpus: 4            # For 13B+ VLMs or larger batches
  distributed_strategy: "fsdp"  # Fully Sharded Data Parallel
```

**Recommended GPU Types by VLM Size:**
- **<3B VLMs**: T4 (16GB), A10G (24GB) - fine-tuning only
- **3-7B VLMs**: A100-40GB, A10G-24GB (with gradient checkpointing)
- **7-13B VLMs**: A100-80GB, H100-80GB
- **13B+ VLMs**: Multi-GPU with FSDP (4-8× A100 or H100)

From [Lightly.ai VLM Training Guide](https://www.lightly.ai/blog/efficient-vlm-training) (accessed 2025-01-31):
> "Scaling the vision encoder can yield stronger performance increases than scaling the language model, even with fewer additional parameters."

### 1.2 VLM Training Job Template

**Basic VLM Fine-Tuning Job:**
```yaml
# wandb-launch-vlm-finetune.yaml
job_type: training
runtime: python
uri: https://github.com/your-org/vlm-training
entry_point: train_vlm.py

build:
  type: docker
  base_image: nvcr.io/nvidia/pytorch:24.01-py3
  requirements:
    - transformers>=4.36.0
    - accelerate>=0.25.0
    - datasets>=2.16.0
    - wandb>=0.16.0
    - pillow>=10.0.0
    - torchvision>=0.16.0

resource:
  gpu: a100-40gb
  gpu_count: 1
  memory: 80GB

parameters:
  model_name: "HuggingFaceM4/idefics2-8b"
  dataset: "your-org/vqa-dataset"
  batch_size: 4
  gradient_accumulation: 4  # Effective batch size = 16
  learning_rate: 2e-5
  epochs: 3
  output_dir: "checkpoints/"
  eval_steps: 500
  save_steps: 1000
  bf16: true
  gradient_checkpointing: true
```

**Training Script (train_vlm.py):**
```python
import torch
import wandb
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

def train_vlm(
    model_name: str,
    dataset: str,
    batch_size: int,
    gradient_accumulation: int,
    learning_rate: float,
    epochs: int,
    output_dir: str,
    eval_steps: int,
    save_steps: int,
    bf16: bool = True,
    gradient_checkpointing: bool = True
):
    # Initialize W&B
    wandb.init(
        project="vlm-training",
        config={
            "model": model_name,
            "dataset": dataset,
            "batch_size": batch_size,
            "lr": learning_rate,
            "epochs": epochs
        }
    )

    # Load model and processor
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map="auto"
    )

    # Enable gradient checkpointing for memory efficiency
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Load dataset (multimodal)
    ds = load_dataset(dataset)

    # Preprocess function for VQA
    def preprocess_vqa(examples):
        images = examples["image"]
        questions = examples["question"]
        answers = examples["answer"]

        # Format prompt
        prompts = [f"Question: {q}\nAnswer:" for q in questions]

        # Process with image + text
        inputs = processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Add labels (answer tokens)
        answer_tokens = processor(
            text=answers,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs["labels"] = answer_tokens.input_ids

        return inputs

    # Preprocess dataset
    train_ds = ds["train"].map(
        preprocess_vqa,
        batched=True,
        remove_columns=ds["train"].column_names
    )
    eval_ds = ds["validation"].map(
        preprocess_vqa,
        batched=True,
        remove_columns=ds["validation"].column_names
    )

    # Training arguments
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        bf16=bf16,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        logging_steps=10,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=gradient_checkpointing
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor  # For saving
    )

    # Train
    trainer.train()

    # Save final checkpoint as artifact
    artifact = wandb.Artifact(
        name=f"vlm-checkpoint-{wandb.run.id}",
        type="model",
        metadata={
            "model_name": model_name,
            "final_loss": trainer.state.log_history[-1]["eval_loss"]
        }
    )
    artifact.add_dir(output_dir)
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="checkpoints/")
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    args = parser.parse_args()

    train_vlm(**vars(args))
```

**Launch the Job:**
```bash
# Create queue for VLM training
wandb launch-queue create vlm-training-queue

# Start agent on GPU machine
wandb launch-agent --queue vlm-training-queue --resource a100-40gb

# Queue training job
wandb launch --queue vlm-training-queue \
  --config wandb-launch-vlm-finetune.yaml \
  --parameters '{"dataset": "my-org/custom-vqa"}'
```

### 1.3 Checkpoint Management for Long VLM Training

**Problem**: VLM training can take days. Need robust checkpointing and resumption.

**Solution**: W&B Artifacts + Automatic Resumption

**Enhanced Training Script with Checkpoint Resumption:**
```python
def train_vlm_with_resumption(
    model_name: str,
    resume_from_artifact: str = None,
    **training_args
):
    wandb.init(project="vlm-training", resume="allow")

    # Check for existing checkpoint
    if resume_from_artifact:
        artifact = wandb.use_artifact(resume_from_artifact)
        checkpoint_dir = artifact.download()
        print(f"Resuming from {checkpoint_dir}")

        # Load model from checkpoint
        model = AutoModelForVision2Seq.from_pretrained(checkpoint_dir)
        processor = AutoProcessor.from_pretrained(checkpoint_dir)
    else:
        # Start fresh
        model = AutoModelForVision2Seq.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)

    # ... rest of training code ...

    # Save checkpoint every N steps with artifact versioning
    class CheckpointCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            artifact = wandb.Artifact(
                name="vlm-checkpoint",
                type="model",
                metadata={
                    "step": state.global_step,
                    "loss": state.log_history[-1].get("loss", None)
                }
            )
            artifact.add_dir(args.output_dir)
            wandb.log_artifact(artifact)

    trainer = Trainer(
        model=model,
        args=args,
        callbacks=[CheckpointCallback()]
    )

    trainer.train()
```

**Launch Config with Auto-Resumption:**
```yaml
# wandb-launch-vlm-resumable.yaml
job_type: training
runtime: python
uri: https://github.com/your-org/vlm-training
entry_point: train_vlm.py

parameters:
  model_name: "HuggingFaceM4/idefics2-8b"
  resume_from_artifact: "latest"  # Auto-resume from latest checkpoint

# Preemption handling
spot_instance: true
max_retries: 3
checkpoint_artifact: "vlm-checkpoint:latest"
```

From [Lightly.ai VLM Training](https://www.lightly.ai/blog/efficient-vlm-training):
> "Multi-stage training occurs due to limited availability of high-quality data, memory constraints, and stability issues. During these stages, progressively higher-quality data is introduced, the maximum image resolution is gradually increased, and more model parts are unfrozen."

---

## Section 2: Multi-Stage VLM Training Pipelines (~150 lines)

### 2.1 Three-Stage VLM Training Pipeline

**Stage 1: Vision-Language Pre-training** (Low resolution, frozen backbones)
**Stage 2: Supervised Fine-Tuning** (High resolution, task-specific)
**Stage 3: Alignment** (RLHF or DPO for safety/quality)

**Pipeline Orchestration with W&B Launch:**
```yaml
# pipeline-vlm-training.yaml
stages:
  - name: pretrain
    job_config: configs/stage1-pretrain.yaml
    resources:
      gpu_type: a100-80gb
      num_gpus: 4
    outputs:
      - pretrain-checkpoint

  - name: sft
    job_config: configs/stage2-sft.yaml
    depends_on: pretrain
    inputs:
      - pretrain-checkpoint
    resources:
      gpu_type: a100-40gb
      num_gpus: 1
    outputs:
      - sft-checkpoint

  - name: alignment
    job_config: configs/stage3-alignment.yaml
    depends_on: sft
    inputs:
      - sft-checkpoint
    resources:
      gpu_type: a100-80gb
      num_gpus: 2
    outputs:
      - final-model

  - name: evaluation
    job_config: configs/evaluate-vlm.yaml
    depends_on: alignment
    inputs:
      - final-model
    resources:
      gpu_type: a100-40gb
      num_gpus: 1
```

### 2.2 Stage 1: Vision-Language Pre-training

**Goal**: Align vision encoder and LLM, train projection layers

**Launch Config (stage1-pretrain.yaml):**
```yaml
job_type: training
entry_point: scripts/pretrain_vlm.py

parameters:
  vision_encoder: "google/siglip-so400m-patch14-384"
  llm: "meta-llama/Llama-3.2-7B"
  dataset: "multimodal/cc12m"  # Web-scale image-text pairs

  # Training config
  batch_size: 128  # Per GPU
  gradient_accumulation: 2
  learning_rate: 1e-4
  warmup_steps: 2000
  total_steps: 100000

  # Image resolution (start low)
  image_size: 224

  # Freezing strategy
  freeze_vision: true
  freeze_llm: true
  train_projection: true  # Only train new layers

  # Efficiency
  bf16: true
  fsdp: true  # Fully Sharded Data Parallel
  gradient_checkpointing: true

resources:
  gpu_type: a100-80gb
  num_gpus: 4
  distributed_strategy: fsdp
```

**Pre-training Script (pretrain_vlm.py):**
```python
import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import wandb

class VLMPretrainer:
    def __init__(
        self,
        vision_encoder: str,
        llm: str,
        freeze_vision: bool = True,
        freeze_llm: bool = True
    ):
        # Load vision encoder
        self.vision_model = AutoModel.from_pretrained(vision_encoder)
        if freeze_vision:
            for param in self.vision_model.parameters():
                param.requires_grad = False

        # Load LLM
        self.llm = AutoModel.from_pretrained(llm, torch_dtype=torch.bfloat16)
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

        # Create projection layers (trainable)
        vision_dim = self.vision_model.config.hidden_size
        llm_dim = self.llm.config.hidden_size

        self.projection = torch.nn.Sequential(
            torch.nn.Linear(vision_dim, llm_dim),
            torch.nn.GELU(),
            torch.nn.Linear(llm_dim, llm_dim)
        ).to(torch.bfloat16)

    def forward(self, images, text):
        # Encode images
        with torch.no_grad():
            vision_features = self.vision_model(images).last_hidden_state

        # Project to LLM space
        projected_features = self.projection(vision_features)

        # Encode text
        with torch.no_grad():
            text_features = self.llm(text).last_hidden_state

        # Contrastive loss (align image-text pairs)
        return self.contrastive_loss(projected_features, text_features)

    def contrastive_loss(self, image_embeds, text_embeds):
        # CLIP-style contrastive learning
        # ... implementation ...
        pass

def pretrain(
    vision_encoder: str,
    llm: str,
    dataset: str,
    batch_size: int,
    learning_rate: float,
    total_steps: int,
    **kwargs
):
    wandb.init(project="vlm-pretrain")

    model = VLMPretrainer(vision_encoder, llm)

    # Load multimodal dataset
    train_loader = DataLoader(
        load_dataset(dataset)["train"],
        batch_size=batch_size,
        num_workers=8
    )

    optimizer = torch.optim.AdamW(
        model.projection.parameters(),
        lr=learning_rate
    )

    for step, batch in enumerate(train_loader):
        if step >= total_steps:
            break

        loss = model(batch["image"], batch["text"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        wandb.log({"pretrain_loss": loss.item(), "step": step})

        # Save checkpoint every 10k steps
        if step % 10000 == 0:
            artifact = wandb.Artifact("pretrain-checkpoint", type="model")
            torch.save(model.state_dict(), f"checkpoint-{step}.pt")
            artifact.add_file(f"checkpoint-{step}.pt")
            wandb.log_artifact(artifact)

    wandb.finish()
```

### 2.3 Stage 2: Supervised Fine-Tuning (SFT)

**Goal**: Adapt to specific tasks (VQA, captioning, etc.)

**Launch Config (stage2-sft.yaml):**
```yaml
job_type: training
entry_point: scripts/sft_vlm.py

parameters:
  pretrain_checkpoint: "pretrain-checkpoint:latest"
  task: "vqa"  # vqa, captioning, reasoning
  dataset: "your-org/vqa-dataset"

  # Higher resolution now
  image_size: 384

  # Unfreeze more parameters
  freeze_vision: false
  freeze_llm: false
  use_lora: true  # LoRA for efficient fine-tuning
  lora_r: 16
  lora_alpha: 32

  # Training
  batch_size: 16
  learning_rate: 2e-5
  epochs: 5

resources:
  gpu_type: a100-40gb
  num_gpus: 1
```

**SFT with LoRA (sft_vlm.py):**
```python
from peft import LoraConfig, get_peft_model

def sft_with_lora(
    pretrain_checkpoint: str,
    task: str,
    use_lora: bool = True,
    lora_r: int = 16,
    **kwargs
):
    # Load pretrained model
    artifact = wandb.use_artifact(pretrain_checkpoint)
    checkpoint_path = artifact.download()

    model = load_pretrained_vlm(checkpoint_path)

    # Apply LoRA for efficient fine-tuning
    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        print(f"Trainable params: {model.print_trainable_parameters()}")

    # Task-specific training
    # ... (similar to Stage 1 but with task data)
```

From [Lightly.ai](https://www.lightly.ai/blog/efficient-vlm-training) on LoRA for VLMs (2025-01-31):
> "LoRA adaptation can be done at a fraction of the GPU cost of pre-training and can be merged back at no additional inference cost."

### 2.4 Stage 3: Alignment (Safety & Quality)

**Goal**: Reduce hallucinations, improve safety, align with human preferences

**Launch Config (stage3-alignment.yaml):**
```yaml
job_type: training
entry_point: scripts/align_vlm.py

parameters:
  sft_checkpoint: "sft-checkpoint:latest"
  alignment_method: "dpo"  # dpo or rlhf
  preference_dataset: "your-org/vlm-preferences"

  # DPO hyperparameters
  beta: 0.1  # KL penalty
  learning_rate: 5e-7
  epochs: 1

resources:
  gpu_type: a100-80gb
  num_gpus: 2  # Reference model + policy model
```

**Automated Evaluation After Each Stage:**
```yaml
# evaluate-vlm.yaml
job_type: evaluation
entry_point: scripts/evaluate_vlm.py

parameters:
  checkpoint: "{depends_on_artifact}"
  benchmarks:
    - "vqav2"
    - "gqa"
    - "nocaps"
    - "ok-vqa"

  metrics:
    - accuracy
    - cider  # For captioning
    - bleu
    - hallucination_rate

resources:
  gpu_type: a100-40gb
  num_gpus: 1
```

---

## Section 3: ARR-COC Production Automation (~150 lines)

### 3.1 ARR-COC Training Job Template

**ARR-COC Architecture Reminder:**
- Vision encoder (produces patches)
- Knowing module (3 ways: Propositional, Perspectival, Participatory)
- Balancing module (opponent processing)
- Attending module (relevance → LOD budgets)
- Realizing module (compression pipeline)

**Complete ARR-COC Training Automation:**
```yaml
# wandb-launch-arr-coc.yaml
job_type: training
runtime: python
uri: https://github.com/your-org/arr-coc
entry_point: train_arr_coc.py

build:
  type: docker
  base_image: nvcr.io/nvidia/pytorch:24.01-py3
  requirements:
    - transformers>=4.36.0
    - accelerate>=0.25.0
    - datasets>=2.16.0
    - wandb>=0.16.0
    - torch>=2.1.0
    - einops>=0.7.0
    - scipy>=1.11.0

resource:
  gpu: a100-40gb
  gpu_count: 1
  memory: 80GB

parameters:
  # Base model
  vision_encoder: "google/siglip-so400m-patch14-384"
  llm: "meta-llama/Llama-3.2-7B"

  # ARR-COC specific
  num_patches: 576  # 24x24 grid for 384px
  min_tokens_per_patch: 64
  max_tokens_per_patch: 400

  # Knowing module weights
  propositional_weight: 0.33
  perspectival_weight: 0.33
  participatory_weight: 0.34

  # Training
  dataset: "your-org/vqa-dataset"
  batch_size: 4
  gradient_accumulation: 8
  learning_rate: 2e-5
  epochs: 3
  warmup_ratio: 0.1

  # Evaluation
  eval_strategy: "steps"
  eval_steps: 500
  save_steps: 1000

  # W&B
  wandb_project: "arr-coc-production"
  wandb_run_name: "arr-coc-{timestamp}"
```

**ARR-COC Training Script (train_arr_coc.py):**
```python
import torch
import wandb
from transformers import Trainer, TrainingArguments
from arr_coc import ARRCOCModel, ARRCOCConfig

def train_arr_coc(
    vision_encoder: str,
    llm: str,
    dataset: str,
    num_patches: int,
    min_tokens_per_patch: int,
    max_tokens_per_patch: int,
    propositional_weight: float,
    perspectival_weight: float,
    participatory_weight: float,
    **training_kwargs
):
    # Initialize W&B with ARR-COC specific tracking
    wandb.init(
        project=training_kwargs.get("wandb_project", "arr-coc"),
        config={
            "vision_encoder": vision_encoder,
            "llm": llm,
            "num_patches": num_patches,
            "token_range": f"{min_tokens_per_patch}-{max_tokens_per_patch}",
            "knowing_weights": {
                "propositional": propositional_weight,
                "perspectival": perspectival_weight,
                "participatory": participatory_weight
            }
        }
    )

    # Create ARR-COC model
    config = ARRCOCConfig(
        vision_encoder_name=vision_encoder,
        llm_name=llm,
        num_patches=num_patches,
        min_tokens_per_patch=min_tokens_per_patch,
        max_tokens_per_patch=max_tokens_per_patch,
        knowing_weights=[
            propositional_weight,
            perspectival_weight,
            participatory_weight
        ]
    )

    model = ARRCOCModel(config)

    # Load dataset
    from datasets import load_dataset
    ds = load_dataset(dataset)

    # Custom data collator for ARR-COC
    class ARRCOCDataCollator:
        def __call__(self, features):
            images = [f["image"] for f in features]
            queries = [f["question"] for f in features]
            answers = [f["answer"] for f in features]

            return {
                "images": images,
                "queries": queries,
                "answers": answers
            }

    # Training arguments
    args = TrainingArguments(
        output_dir="checkpoints/arr-coc",
        per_device_train_batch_size=training_kwargs.get("batch_size", 4),
        gradient_accumulation_steps=training_kwargs.get("gradient_accumulation", 8),
        learning_rate=training_kwargs.get("learning_rate", 2e-5),
        num_train_epochs=training_kwargs.get("epochs", 3),
        bf16=True,
        evaluation_strategy=training_kwargs.get("eval_strategy", "steps"),
        eval_steps=training_kwargs.get("eval_steps", 500),
        save_strategy="steps",
        save_steps=training_kwargs.get("save_steps", 1000),
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="relevance_accuracy",
        report_to="wandb",
        logging_steps=10,
        warmup_ratio=training_kwargs.get("warmup_ratio", 0.1),
        gradient_checkpointing=True
    )

    # Custom metrics for ARR-COC
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # Standard accuracy
        accuracy = (predictions.argmax(-1) == labels).mean()

        # ARR-COC specific: Relevance realization metrics
        relevance_scores = predictions["relevance_scores"]  # From model output
        token_budgets = predictions["token_budgets"]

        # Average tokens allocated per patch
        avg_tokens_per_patch = token_budgets.mean()

        # Compression ratio
        max_possible_tokens = num_patches * max_tokens_per_patch
        actual_tokens = token_budgets.sum()
        compression_ratio = actual_tokens / max_possible_tokens

        return {
            "accuracy": accuracy,
            "relevance_accuracy": accuracy,  # Alias for best model selection
            "avg_tokens_per_patch": avg_tokens_per_patch,
            "compression_ratio": compression_ratio,
            "knowing_propositional": relevance_scores[:, 0].mean(),
            "knowing_perspectival": relevance_scores[:, 1].mean(),
            "knowing_participatory": relevance_scores[:, 2].mean()
        }

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=ARRCOCDataCollator(),
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Save final model as W&B artifact
    artifact = wandb.Artifact(
        name=f"arr-coc-checkpoint-{wandb.run.id}",
        type="model",
        metadata={
            "final_accuracy": trainer.state.log_history[-1]["eval_accuracy"],
            "compression_ratio": trainer.state.log_history[-1]["eval_compression_ratio"]
        }
    )
    artifact.add_dir("checkpoints/arr-coc")
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_encoder", type=str, required=True)
    parser.add_argument("--llm", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_patches", type=int, default=576)
    parser.add_argument("--min_tokens_per_patch", type=int, default=64)
    parser.add_argument("--max_tokens_per_patch", type=int, default=400)
    parser.add_argument("--propositional_weight", type=float, default=0.33)
    parser.add_argument("--perspectival_weight", type=float, default=0.33)
    parser.add_argument("--participatory_weight", type=float, default=0.34)
    # ... add remaining args ...
    args = parser.parse_args()

    train_arr_coc(**vars(args))
```

### 3.2 ARR-COC Ablation Study Automation

**Automated Ablation Studies with W&B Sweeps + Launch:**
```yaml
# arr-coc-ablation-sweep.yaml
program: train_arr_coc.py
method: grid
metric:
  name: relevance_accuracy
  goal: maximize

parameters:
  # Fixed
  vision_encoder:
    value: "google/siglip-so400m-patch14-384"
  llm:
    value: "meta-llama/Llama-3.2-7B"
  dataset:
    value: "your-org/vqa-dataset"

  # Ablation: Test each way of knowing independently
  propositional_weight:
    values: [1.0, 0.33, 0.0]
  perspectival_weight:
    values: [1.0, 0.33, 0.0]
  participatory_weight:
    values: [1.0, 0.33, 0.0]

  # Token budget ranges
  min_tokens_per_patch:
    values: [32, 64, 128]
  max_tokens_per_patch:
    values: [200, 400, 600]

# Launch integration
launch:
  queue: arr-coc-ablation-queue
  resource:
    gpu: a100-40gb
    num_gpus: 1
```

**Run Sweep:**
```bash
# Create sweep
wandb sweep arr-coc-ablation-sweep.yaml

# Launch agents (can run multiple in parallel)
wandb launch-agent --queue arr-coc-ablation-queue --count 3
```

### 3.3 ARR-COC Evaluation Job

**Automated VQA Evaluation:**
```yaml
# evaluate-arr-coc.yaml
job_type: evaluation
entry_point: evaluate_arr_coc.py

parameters:
  checkpoint: "arr-coc-checkpoint:latest"
  benchmarks:
    - vqav2
    - gqa
    - ok-vqa

  # ARR-COC specific metrics
  measure_relevance: true
  measure_compression: true
  measure_patch_distribution: true

resources:
  gpu_type: a100-40gb
  num_gpus: 1
```

**Evaluation Script (evaluate_arr_coc.py):**
```python
import torch
import wandb
from arr_coc import ARRCOCModel
from datasets import load_dataset

def evaluate_arr_coc(
    checkpoint: str,
    benchmarks: list,
    measure_relevance: bool = True,
    measure_compression: bool = True,
    measure_patch_distribution: bool = True
):
    wandb.init(project="arr-coc-eval", job_type="evaluation")

    # Load checkpoint
    artifact = wandb.use_artifact(checkpoint)
    checkpoint_path = artifact.download()
    model = ARRCOCModel.from_pretrained(checkpoint_path)
    model.eval()

    results = {}

    for benchmark in benchmarks:
        print(f"Evaluating on {benchmark}...")
        ds = load_dataset(f"benchmark/{benchmark}")

        correct = 0
        total = 0

        # ARR-COC specific metrics
        relevance_scores_all = []
        token_budgets_all = []
        patch_selections = []

        for sample in ds["test"]:
            with torch.no_grad():
                output = model(
                    images=sample["image"],
                    query=sample["question"],
                    return_relevance=measure_relevance
                )

            prediction = output["answer"]
            correct += (prediction == sample["answer"])
            total += 1

            if measure_relevance:
                relevance_scores_all.append(output["relevance_scores"])
                token_budgets_all.append(output["token_budgets"])
                patch_selections.append(output["selected_patches"])

        # Calculate metrics
        accuracy = correct / total

        results[benchmark] = {
            "accuracy": accuracy,
            "num_samples": total
        }

        if measure_compression:
            avg_tokens = torch.cat(token_budgets_all).mean()
            max_tokens = model.config.num_patches * model.config.max_tokens_per_patch
            compression_ratio = avg_tokens / max_tokens

            results[benchmark]["avg_tokens_per_image"] = avg_tokens.item()
            results[benchmark]["compression_ratio"] = compression_ratio.item()

        if measure_relevance:
            relevance_matrix = torch.cat(relevance_scores_all)
            results[benchmark]["knowing_propositional"] = relevance_matrix[:, 0].mean().item()
            results[benchmark]["knowing_perspectival"] = relevance_matrix[:, 1].mean().item()
            results[benchmark]["knowing_participatory"] = relevance_matrix[:, 2].mean().item()

        if measure_patch_distribution:
            # Heatmap of which patches get selected most
            patch_heatmap = torch.zeros(24, 24)  # 24x24 grid
            for patches in patch_selections:
                for patch_idx in patches:
                    row, col = patch_idx // 24, patch_idx % 24
                    patch_heatmap[row, col] += 1

            # Log as W&B image
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(patch_heatmap.numpy(), cmap="hot")
            ax.set_title(f"Patch Selection Heatmap - {benchmark}")
            plt.colorbar(im)

            wandb.log({f"{benchmark}_patch_heatmap": wandb.Image(fig)})
            plt.close()

    # Log all results
    wandb.log(results)

    # Create summary table
    table = wandb.Table(
        columns=["Benchmark", "Accuracy", "Compression Ratio", "Avg Tokens"],
        data=[
            [
                bench,
                results[bench]["accuracy"],
                results[bench].get("compression_ratio", None),
                results[bench].get("avg_tokens_per_image", None)
            ]
            for bench in benchmarks
        ]
    )
    wandb.log({"evaluation_summary": table})

    wandb.finish()
```

### 3.4 Complete ARR-COC Production Pipeline

**Full CI/CD Pipeline with W&B Launch:**
```yaml
# arr-coc-production-pipeline.yaml
stages:
  - name: data-preparation
    job_config: configs/prepare-data.yaml
    outputs:
      - vqa-dataset-artifact

  - name: train-arr-coc
    job_config: configs/train-arr-coc.yaml
    depends_on: data-preparation
    inputs:
      - vqa-dataset-artifact
    outputs:
      - arr-coc-checkpoint

  - name: ablation-studies
    job_config: configs/arr-coc-ablation-sweep.yaml
    depends_on: train-arr-coc
    parallel: 9  # Run 9 ablation experiments in parallel

  - name: evaluation
    job_config: configs/evaluate-arr-coc.yaml
    depends_on: train-arr-coc
    inputs:
      - arr-coc-checkpoint
    outputs:
      - evaluation-results

  - name: deploy-gradio
    job_config: configs/deploy-gradio.yaml
    depends_on: evaluation
    inputs:
      - arr-coc-checkpoint
      - evaluation-results
    resources:
      platform: huggingface-spaces
      gpu_type: a10g-small
```

**Trigger Pipeline:**
```bash
# Create production queue
wandb launch-queue create arr-coc-production

# Start agents for different stages
wandb launch-agent --queue arr-coc-production --resource a100-40gb &
wandb launch-agent --queue arr-coc-production --resource a100-40gb &

# Queue pipeline
wandb launch --queue arr-coc-production \
  --config arr-coc-production-pipeline.yaml \
  --parameters '{"experiment_name": "arr-coc-v1.0"}'
```

---

## Sources

**Web Research (accessed 2025-01-31):**
- [Lightly.ai: Efficient Training for Multimodal Vision Models](https://www.lightly.ai/blog/efficient-vlm-training) - VLM training techniques, multi-stage pipelines, LoRA efficiency
- W&B Launch documentation (official docs)
- VLM distributed training best practices (Google Scholar research)
- Multimodal model training pipelines automation patterns

**Existing Oracle Knowledge:**
- `practical-implementation/15-wandb-quick-validation.md` - W&B fundamentals
- `practical-implementation/16-wandb-hyperparameter-sweeps.md` - Sweep patterns
- `practical-implementation/17-wandb-production-monitoring.md` - Production tracking
- `practical-implementation/19-wandb-vlm-evaluation.md` - VLM evaluation metrics
- `practical-implementation/21-wandb-integration-cookbook.md` - Integration patterns
- `training-llms/07-mixed-precision-2025-best-practices-2025-01-30.md` - bfloat16, GPU optimization
- `practical-implementation/08-gpu-memory-debugging-vlm-2025-01-30.md` - GPU OOM debugging, checkpoint management

**ARR-COC Context:**
- ARR-COC project architecture (README.md, CLAUDE.md from main repo)
- Vervaeke's relevance realization framework (3 ways of knowing, opponent processing)

---

**Last Updated**: 2025-01-31
**Part of**: W&B Launch & Automation Expansion (Files 22-29)
**Oracle Version**: 1.4
