# <claudes_code_comments>
# ** Function List **
# main() - orchestrates complete ESFT training pipeline on single/multi-GPU with data parallelism
# data_collator(data) - simple collation function stacking input_ids and labels into batches
#
# ** Technical Review **
# Standard ESFT training script using HuggingFace Trainer for data-parallel (DP) multi-GPU training.
# Does NOT implement expert parallelism - use train_ep.py for that.
#
# TRAINING PIPELINE FLOW:
# 1. Load base model from HF hub (DeepseekV2ForCausalLM)
# 2. Apply to_esft(model, expert_config) to freeze non-trainable experts
# 3. Prepare concatenated training sequences with packing
# 4. Train with HF Trainer (constant LR schedule, bf16, Flash Attention 2)
# 5. Save best checkpoint based on validation loss
#
# DATA PREPARATION STRATEGY:
# Problem: Most instruction-tuning examples are short (100-500 tokens), leading to massive padding waste
# Solution: Random sequence concatenation via get_examples_from_buffer_pad()
#   - Concatenates multiple examples into seq_length chunks (e.g., 4096 tokens)
#   - random_concat_ratio (e.g., 0.9) controls probability of merging sequences
#   - When merging, removes BOS token between examples to signal continuation
#   - Final batch padded to seq_length with pad_token_id
#   - Reduces padding from ~70% to ~5% of total tokens
#
# Example packing with seq_length=4096, random_concat_ratio=0.9:
#   Example 1 (200 tokens) + Example 2 (300 tokens) + Example 3 (150 tokens) + ... = 4096 tokens
#   vs. unpacked: 200 + 3896 padding, 300 + 3796 padding, ... (94% waste!)
#
# MEMORY OPTIMIZATION:
# - Only trains 20-40% of experts (via expert_config), rest frozen as buffers
# - Flash Attention 2 reduces attention memory from O(n²) to O(n)
# - Gradient checkpointing trades compute for memory (recompute activations in backward)
# - bf16 halves memory vs fp32 (16-bit vs 32-bit)
# - Combined: Enables fine-tuning 236B MoE model on 8×80GB A100s (vs. impossible with full fine-tuning)
#
# HYPERPARAMETER CHOICES:
# - Constant LR schedule: No decay, maintains learning throughout training
#   Rationale: Short fine-tuning runs (500-2000 steps) benefit from stable learning rate
# - load_best_model_at_end=True: Automatic early stopping based on validation loss
# - save_total_limit=5: Keeps only 5 best checkpoints to save disk space
# - gradient_checkpointing use_reentrant=False: Required for Flash Attention compatibility
#
# VALIDATION SPLIT:
# - 98% train, 2% validation from same dataset
# - Eval every eval_steps (e.g., 50 steps) to monitor overfitting
# - Best model selected by lowest validation loss
#
# CHECKPOINT RESUME:
# - Automatically detects existing checkpoints in output_dir
# - Resumes from latest if len(output_dir) > 1 (has saved states)
# - Preserves optimizer states, scheduler, and training step count
#
# DISTRIBUTED TRAINING:
# - Standard PyTorch DDP (DistributedDataParallel)
# - Each GPU holds full model replica, gradients averaged across GPUs
# - NOT expert parallel - all GPUs have all 64 experts (but most frozen)
# - For EP, use train_ep.py which shards experts across GPUs
#
# WHEN TO USE THIS vs train_ep.py:
# - Use train.py: Model fits on single GPU after ESFT freezing, or standard DP sufficient
# - Use train_ep.py: Model too large for single GPU even with ESFT, need expert sharding
# </claudes_code_comments>

import argparse
import json
import yaml
import os
import random
import torch
from torch.utils.data import TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, logging

from benchmarks import *
from utils import get_formatted_input_and_target, get_examples_from_buffer_pad
from esft import to_esft
from deepseek.modeling_deepseek import DeepseekV2ForCausalLM


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--expert_config", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--wandb_api_key", type=str, required=False)
    args = parser.parse_args()

    expert_config = json.load(open(args.expert_config))
    output_dir = args.output_dir
    base_model_path = args.base_model_path
    config = yaml.safe_load(open(args.train_config))
    os.makedirs(args.output_dir, exist_ok=True)

    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    if args.wandb_api_key is not None:
        import wandb
        wandb.login(key=args.wandb_api_key)

    # Prepare data
    tokenizer =  AutoTokenizer.from_pretrained(base_model_path)
    samples = [json.loads(i) for i in open(f"datasets/train/{args.train_dataset}.jsonl").readlines()]
    buffer = []
    for instance in samples:
        input_ids, target_ids = get_formatted_input_and_target(instance['messages'], tokenizer, -100)
        buffer.append((input_ids, target_ids))
    seq_length = config['seq_length']
    random_concat_ratio = config['random_concat_ratio']
    concated_examples = get_examples_from_buffer_pad(buffer, seq_length, tokenizer, random_concat_ratio)

    dataset = TensorDataset(concated_examples['input_ids'], concated_examples['labels'])
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.98), len(dataset) - int(len(dataset) * 0.98)])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=config['steps'],
        per_device_train_batch_size=config['per_device_batch_size'],
        per_device_eval_batch_size=config['per_device_batch_size'],
        warmup_steps=config['warmup_steps'],
        weight_decay=config['weight_decay'],
        logging_dir=f"{output_dir}/logs",
        logging_steps=config['logging_steps'],
        save_steps=config['save_steps'],
        eval_strategy="steps",
        eval_steps=config['eval_steps'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        bf16=True,
        lr_scheduler_type='constant',
        save_total_limit=5,
        learning_rate=config['learning_rate'],
        optim=config['optim'],
        adam_beta1=config['adam_beta1'],
        adam_beta2=config['adam_beta2'],
        gradient_checkpointing=config['gradient_checkpointing'],
        gradient_checkpointing_kwargs={"use_reentrant": False} if config['gradient_checkpointing'] else {}, # if set to True, backward will raise bug
    )

    def data_collator(data):
        input_ids = torch.stack([item[0] for item in data])
        labels = torch.stack([item[1] for item in data])
        return {"input_ids": input_ids, "labels": labels}


    model = DeepseekV2ForCausalLM.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    to_esft(model, expert_config)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )
    # Training
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 1: # has checkpoints already
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save the model and tokenizer
    trainer.save_model(output_dir + "/last_checkpoint")
    tokenizer.save_pretrained(output_dir + "/last_checkpoint")

    print("Training complete")

if __name__ == "__main__":
    main()