# <claudes_code_comments>
# ** Function List **
# infer_auto_device_map(model, pp_splits, visible_devices) - creates pipeline parallel device map from layer splits
# eval_expert(rank, args, dataset) - runs inference on dataset subset, logging expert activations for profiling
#
# ** Technical Review **
# Expert usage profiling script - measures which experts activate on task-specific data.
# Critical first step in 3-stage ESFT pipeline: Profile → Select → Train
#
# WHY PROFILING IS NECESSARY:
# MoE models route different tokens to different experts based on learned gating function.
# For task-specific fine-tuning, only a subset of experts activate frequently on task data.
# Example: Translation task might heavily use experts 8,9,15 in layer 1, barely touch others.
# Training all experts is wasteful - ESFT trains only the task-relevant ones.
#
# PROFILING STRATEGY:
# 1. Enable expert logging: model.config.log_expert_weights = True
#    This hooks into DeepSeekV2MoE.forward() to record gating outputs
# 2. Run inference on n_sample_tokens (typically 131k) from training data
# 3. For each token, log (expert_ids, expert_weights) for each MoE layer
# 4. Aggregate statistics across all tokens to identify important experts
#
# MULTI-PROCESS ARCHITECTURE:
# Uses torch.multiprocessing.spawn() to parallelize profiling across world_size processes
# Each process (rank) handles 1/world_size of dataset: rank processes dataset[rank::world_size]
# Example: world_size=4, dataset=[0,1,2,3,4,5,6,7,8,9,10,11]
#   Rank 0: [0,4,8,...]
#   Rank 1: [1,5,9,...]
#   Rank 2: [2,6,10,...]
#   Rank 3: [3,7,11,...]
#
# PIPELINE PARALLELISM (infer_auto_device_map):
# Model is large (~236B params), doesn't fit on single GPU for inference
# Solution: Split layers across gpus_per_rank GPUs using pipeline parallelism
#
# pp_splits=[14,13] with visible_devices=[GPU0, GPU1]:
#   GPU0: embed_tokens + layers 0-13 (14 layers)
#   GPU1: layers 14-26 + norm + lm_head (13 layers)
#
# Device map dictionary format:
#   {"model.embed_tokens": 0, "model.layers.0": 0, ..., "model.layers.13": 0,
#    "model.layers.14": 1, ..., "model.layers.26": 1, "model.norm": 1, "lm_head": 1}
#
# accelerate.dispatch_model() places modules on GPUs according to device_map
# Activations automatically move between GPUs during forward pass (layer 13→14 transfers GPU0→GPU1)
#
# TOKEN SAMPLING BUDGET:
# n_sample_tokens (e.g., 131072) split across world_size processes
# Each rank processes n_sample_tokens // world_size tokens
# Why 131k? Empirically sufficient to capture expert usage distribution
# Too few (10k): Noisy statistics, might miss important experts
# Too many (1M): Diminishing returns, profiling takes hours
#
# LOGGING OUTPUT FORMAT:
# Files: output_dir/rank_{rank}/layer_{layer_id}.txt
# Each line: "{expert_id1}\t{expert_id2}\t...\t{expert_id6}\t\t{weight1}\t{weight2}\t...\t{weight6}"
# Example line for layer 1: "8\t9\t15\t23\t47\t51\t\t0.25\t0.20\t0.18\t0.15\t0.12\t0.10"
#   Interpretation: For this token, top-6 experts were [8,9,15,23,47,51] with gating weights [0.25,0.20,...]
#
# DeepSeek-V2 uses Top-K=6 routing: Each token routed to 6 experts
# Gating weights sum to 1.0 (softmax over selected experts)
# Double tab (\t\t) separates expert IDs from weights
#
# AGGREGATION (next step):
# generate_expert_config.py reads all rank_N/layer_X.txt files, computes:
#   - gate_scores: Sum of weights per expert (measures activation strength)
#   - token_scores: Count of tokens routed to each expert (measures frequency)
# Then selects top-p experts per layer (e.g., experts covering 20% of cumulative score)
#
# MEMORY AND PERFORMANCE:
# - torch_dtype=torch.bfloat16: Reduces memory by 50% vs fp32
# - No gradients computed (inference only): Saves memory for activations
# - tqdm progress bar: Tracks tokens processed per rank
# - gpus_per_rank=2 typical: Balances memory (more GPUs) vs overhead (GPU-to-GPU transfers)
#
# ERROR HANDLING:
# try-except in eval_expert(): Catches CUDA OOM or other errors, prints rank-specific error
# Multiprocessing makes debugging harder - rank-specific error messages crucial
#
# TYPICAL USAGE:
# python scripts/expert/get_expert_scores.py \
#   --eval_dataset=translation \
#   --base_model_path=deepseek-ai/deepseek-moe-16b-base \
#   --output_dir=results/expert_scores/translation \
#   --n_sample_tokens=131072 \
#   --world_size=4 \
#   --gpus_per_rank=2
#
# Output: 4 ranks × 27 layers = 108 log files with expert activation data
# Next step: Run generate_expert_config.py to aggregate and select experts
# </claudes_code_comments>

import json
import os
import torch
import argparse
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_formatted_input_and_target
import torch.multiprocessing as mp
from itertools import accumulate
from accelerate import dispatch_model
from tqdm import tqdm

def infer_auto_device_map(model, pp_splits, visible_devices):
    assert len(pp_splits) == len(visible_devices)
    device_map = {
        "model.embed_tokens": 0,
        "model.norm": len(pp_splits) - 1,
        "lm_head": len(pp_splits) - 1
    }
    assert len(model.model.layers) == sum(pp_splits)
    pp_splits = [0, *list(accumulate(pp_splits))]
    for idx, (start, end) in enumerate(zip(pp_splits[:-1], pp_splits[1:])):
        for i in range(start, end):
            device_map.update({f"model.layers.{i}": idx})
    for k, v in device_map.items():
        device_map[k] = visible_devices[v]
    return device_map


def eval_expert(rank, args, dataset):
    try:
        model = AutoModelForCausalLM.from_pretrained(args.base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16) # not using tokenizer here to aviod deadlock
        model.config.log_expert_weights = True
        print(f"Rank {rank} starting expert evaluation...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
        visible_devices = list(range(rank * args.gpus_per_rank, (rank + 1) * args.gpus_per_rank))
        device_map = infer_auto_device_map(model, [14, 13], visible_devices)
        model = dispatch_model(model, device_map)
        model.config.expert_log_dir = os.path.join(args.output_dir, f"rank_{rank}")
        n_sample_tokens = args.n_sample_tokens // args.world_size
        os.makedirs(os.path.join(args.output_dir, f"rank_{rank}"), exist_ok=True)
        done_tokens = 0
        cur_dataset = dataset[rank::args.world_size]
        pbar = tqdm(total=n_sample_tokens, desc=f"Rank {rank} processing tokens", position=rank)
        for instance in cur_dataset:
            input_ids, target_ids = get_formatted_input_and_target(instance['messages'], tokenizer, -100)
            model(input_ids=torch.tensor(input_ids).unsqueeze(0), labels=torch.tensor(target_ids).unsqueeze(0))
            done_tokens += len(input_ids)
            pbar.update(len(input_ids))
            if done_tokens >= n_sample_tokens:
                break
        pbar.close()


    except Exception as e:
        print(f"Error in process {rank}: {e}", flush=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model with adapters on a specified dataset.")
    parser.add_argument("--eval_dataset", type=str, required=True, help="Name of the evaluation dataset")
    parser.add_argument("--base_model_path", type=str,  required=True, help="Path to the base model")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the evaluation results")
    parser.add_argument("--world_size", type=int, default=4, help="Number of processes to use for evaluation")
    parser.add_argument("--gpus_per_rank", type=int, default=2, help="Number of GPUs per process")
    parser.add_argument("--n_sample_tokens", type=int, required=True, help="Token to sample for expert evaluation")
    args = parser.parse_args()
    random.seed(5934875)


    print(f"Running expert evaluation on {args.eval_dataset}...")
    dataset = [json.loads(i) for i in open(f"datasets/train/{args.eval_dataset}.jsonl").readlines()]
    random.shuffle(dataset)


    print("Start Evaluating...")
    mp.spawn(eval_expert, args=(args, dataset), nprocs=args.world_size, join=True)
