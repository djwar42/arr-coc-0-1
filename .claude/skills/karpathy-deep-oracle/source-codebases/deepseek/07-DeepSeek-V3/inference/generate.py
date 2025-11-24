# <claudes_code_comments>
# ** Function List **
# sample(logits, temperature) - Temperature-scaled sampling from logits using Gumbel trick
# generate(model, prompt_tokens, max_new_tokens, eos_id, temperature) - Autoregressive generation with KV caching
# main(ckpt_path, config, input_file, interactive, max_new_tokens, temperature) - Entry point for inference
#
# ** Technical Review **
# This script implements efficient autoregressive text generation for DeepSeek-V3 using KV caching
# and distributed inference across multiple GPUs. Supports both interactive chat and batch inference.
#
# **Sampling Strategy** (lines 14-27):
# Uses the Gumbel-max trick for efficient sampling:
# - Temperature scaling: logits / temperature (higher = more random, lower = more deterministic)
# - Gumbel noise: probs.div_(exponential_noise) is equivalent to adding Gumbel(-log(-log(U))) noise
# - Argmax over noisy probabilities = sampling from categorical distribution
# - More efficient than torch.multinomial for single-token sampling
#
# **Autoregressive Generation** (lines 30-78):
# Implements standard transformer generation with two-phase decoding:
#
# Phase 1 - Prompt Processing (parallel):
# - Process entire prompt in one forward pass (lines 60-66)
# - Build KV cache for all prompt tokens simultaneously
# - More efficient than sequential processing since all positions known
#
# Phase 2 - Token Generation (sequential):
# - Generate one token at a time autoregressively (lines 60-71)
# - Reuse KV cache from previous tokens (start_pos parameter)
# - Only process newly generated token in each step
# - Cache update: model stores K/V for position start_pos:end_pos
# - Early stopping: terminates when all sequences hit EOS token
#
# **KV Cache Efficiency**:
# Critical for long-context generation (up to 128K tokens):
# - Without cache: O(n²) computation for n tokens (re-compute all positions)
# - With cache: O(n) computation (only new token, reuse cached K/V)
# - MLA further compresses cache: stores latent representation instead of full K/V
# - For 128K context, MLA cache ~10x smaller than standard attention cache
#
# **Distributed Inference** (lines 100-149):
# Supports multi-GPU/multi-node inference via PyTorch distributed:
# - Model sharded across world_size GPUs using tensor/expert parallelism
# - Rank 0 handles user I/O, broadcasts prompts to other ranks
# - All ranks run forward pass in parallel on their model shard
# - Final logits gathered via all_gather (model.py line 796) before sampling
# - Vocabulary parallelism: each rank holds vocab_size//world_size embeddings
#
# **Interactive Mode** (lines 121-144):
# Multi-turn chat interface with conversation history:
# - Maintains messages list with role/content format
# - Applies chat template (model-specific formatting for user/assistant turns)
# - Commands: /exit to quit, /clear to reset conversation
# - Generation_prompt flag ensures model continues in assistant role
#
# **Batch Mode** (lines 145-150):
# Processes multiple prompts from file in single forward pass:
# - Reads prompts from input_file (one per line)
# - Pads to same length for batched processing
# - Max batch_size limited by memory (configured in ModelArgs)
# - More efficient than sequential processing (amortizes model overhead)
#
# **Model Loading**:
# - Config from JSON: ModelArgs(**json.load(config_file))
# - Weights from SafeTensors: load_model(model, "model{rank}-mp{world_size}.safetensors")
# - Each rank loads only its shard of the model weights
# - Distributed init (line 104): NCCL backend for GPU communication
#
# **Precision and Performance**:
# - Default dtype: torch.bfloat16 for memory efficiency (line 109)
# - FP8 weights automatically handled by model.py's linear() dispatcher
# - Inference mode: torch.inference_mode() disables gradient tracking (line 30)
# - Seed (line 111): for reproducible generation in testing
#
# **Torchrun Launch**:
# Typical usage: torchrun --nnodes 2 --nproc-per-node 8 generate.py ...
# - nnodes: number of machines (e.g., 2 for 16 GPU deployment)
# - nproc-per-node: GPUs per machine (e.g., 8 for H100/H800 nodes)
# - Environment vars: WORLD_SIZE, RANK, LOCAL_RANK set automatically
# - MASTER_ADDR/MASTER_PORT: for multi-node communication coordination
#
# This script demonstrates efficient large-scale inference, with KV caching and MLA compression
# enabling interactive generation at 671B scale with reasonable latency and memory usage.
# </claudes_code_comments>

import os
import json
from argparse import ArgumentParser
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs


def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.

    Args:
        model (Transformer): The transformer model used for token generation.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
    """
    Main function to load the model and perform interactive or batch text generation.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        input_file (str, optional): Path to a file containing input prompts. Defaults to "".
        interactive (bool, optional): Whether to run in interactive mode. Defaults to True.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 1.0.
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer.decode(generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.)[0])
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))

    if interactive:
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            print(completion)
            messages.append({"role": "assistant", "content": completion})
    else:
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines()]
        assert len(prompts) <= args.max_batch_size, f"Number of prompts exceeds maximum batch size ({args.max_batch_size})"
        prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    """
    Command-line interface for distributed text generation.

    Arguments:
        --ckpt-path (str): Path to the model checkpoint directory.
        --config (str): Path to the model configuration file.
        --input-file (str, optional): File containing prompts for batch processing.
        --interactive (bool, optional): Enable interactive mode for generating text.
        --max-new-tokens (int, optional): Maximum number of new tokens to generate. Defaults to 200.
        --temperature (float, optional): Temperature for sampling. Defaults to 0.2.

    Raises:
        AssertionError: If neither input-file nor interactive mode is specified.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature)
