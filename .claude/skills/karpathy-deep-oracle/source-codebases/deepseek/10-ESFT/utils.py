# <claudes_code_comments>
# ** Function List **
# get_formatted_question(line) - formats user input into "User: X\n\nAssistant:" template
# get_formatted_answer(line) - formats assistant response into " Y" template (note leading space)
# get_formatted_input_and_target(messages, tokenizer, IGNORE_TOKEN_ID, mask_prompt) - converts multi-turn chat to tokenized (input_ids, target_ids) pairs
# get_examples_from_buffer_pad(buffer, seq_length, tokenizer, random_concat_ratio, IGNORE_TOKEN_ID) - packs multiple examples into fixed-length sequences with random concatenation
# init_parallel_groups(ep_size) - initializes 2D process groups (EP and EDP) for expert parallelism
#
# ** Technical Review **
# Core utilities for data preparation and distributed training infrastructure in ESFT.
#
# CHAT FORMAT TOKENIZATION:
# DeepSeek chat format: "User: {input}\n\nAssistant: {response}"
# Template constants enforce strict formatting:
#   PROMPT_USER = 'User: {input}\n\n'  (note double newline)
#   PROMPT_ASSISTANT = 'Assistant:'  (no trailing space - space comes from response)
#   ASSISTANT_RESPONSE = ' {input}'  (note leading space before response)
#
# Why the spacing matters: Tokenizer treats "Assistant:Hello" vs "Assistant: Hello" differently
# Leading space in response ensures consistent tokenization across all examples
#
# get_formatted_input_and_target() CONVERSATION FLOW:
# Input: messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
# Processing:
#   1. Add BOS token at start (idx==0): input_ids=[BOS], target_ids=[BOS]
#   2. User message: tokenize "User: Hi\n\nAssistant:"
#      - Append to input_ids
#      - Append IGNORE_TOKEN_ID (-100) to target_ids if mask_prompt=True (loss won't compute on prompts)
#   3. Assistant message: tokenize " Hello" + EOS
#      - Append to input_ids
#      - Append to target_ids (loss will compute here)
#      - If message.get('mask', 0)==1, use IGNORE_TOKEN_ID instead (allows masking specific responses)
#
# Multi-turn example:
#   User: "What is 2+2?" → tokenized, masked in loss
#   Assistant: " 4" → tokenized, included in loss
#   User: "And 3+3?" → tokenized, masked in loss
#   Assistant: " 6" → tokenized, included in loss
#
# Result: input_ids contains full conversation, target_ids has -100 for prompts and actual tokens for responses
# Cross-entropy loss automatically ignores target_ids==-100
#
# SEQUENCE PACKING (get_examples_from_buffer_pad):
# Problem: Instruction examples vary widely in length (50-2000 tokens). Padding to seq_length=4096 wastes memory.
# Solution: Pack multiple short examples into single seq_length sequence
#
# Algorithm:
#   1. Initialize all_input_ids=[], all_target_ids=[] as accumulators
#   2. For each (input_ids, target_ids) in buffer:
#      a. If current example doesn't fit in remaining space, truncate it (keeps last seq_length-len(all_input_ids) tokens)
#      b. With probability random_concat_ratio (e.g., 0.9), remove first token (BOS) to merge sequences
#         Why remove BOS? Signals to model this is continuation of context, not new conversation
#      c. Extend all_input_ids and all_target_ids with (possibly truncated) example
#      d. If len(all_input_ids) >= seq_length, save sequence and reset accumulators
#   3. Final incomplete sequence: Pad with pad_token_id (input) and IGNORE_TOKEN_ID (target) to reach seq_length
#
# Packing efficiency example (seq_length=4096, random_concat_ratio=0.9):
#   Without packing: [200 tokens + 3896 padding], [300 tokens + 3796 padding], ... → 95% waste
#   With packing: [200 + 300 + 150 + ... + 3896 tokens] → ~5% waste in final sequence only
#   Effective throughput: 20x improvement in tokens/batch
#
# Edge case handling:
#   - Truncation: If single example > seq_length, keeps LAST (seq_length - current_len) tokens
#     Rationale: For long examples, ending (conclusion/answer) often more important than beginning
#   - Random concatenation: Probabilistic merge prevents model from learning "BOS always means new task"
#     Some sequences have BOS between examples (10%), some don't (90%)
#
# EXPERT PARALLELISM INITIALIZATION (init_parallel_groups):
# Creates two orthogonal process groups for 2D parallelism (EP × EDP)
#
# Example: 8 GPUs total, ep_size=4
#   world_size=8, ep_size=4 → edp_size = world_size // ep_size = 2
#
# EP groups (expert parallel - different expert shards, same data):
#   Group 0: [GPU0, GPU1, GPU2, GPU3] - each holds different experts (0-15, 16-31, 32-47, 48-63)
#   Group 1: [GPU4, GPU5, GPU6, GPU7] - same expert split as Group 0
#
# EDP groups (expert data parallel - same expert shards, different data):
#   Group 0: [GPU0, GPU4] - both hold experts 0-15, process different batches
#   Group 1: [GPU1, GPU5] - both hold experts 16-31, process different batches
#   Group 2: [GPU2, GPU6] - both hold experts 32-47, process different batches
#   Group 3: [GPU3, GPU7] - both hold experts 48-63, process different batches
#
# Process group creation:
#   1. EP groups: For i in range(0, world_size, ep_size): create group [i, i+1, ..., i+ep_size-1]
#   2. EDP groups: For i in range(ep_size): create group [i, i+ep_size, i+2*ep_size, ...]
#   3. Return groups that current rank belongs to (ep_group, edp_group)
#
# All-reduce validation: Runs test all-reduce on each group to ensure correct initialization
# Critical: Groups must be created in same order on all ranks, otherwise collective ops deadlock
#
# DISTRIBUTED TRAINING RANK TERMINOLOGY:
# - world_size: Total number of GPUs (e.g., 8)
# - local_rank: GPU ID on current machine (0-7 if single 8-GPU node)
# - ep_rank: Rank within EP group (0 to ep_size-1, determines which expert shard)
# - edp_rank: Rank within EDP group (0 to edp_size-1, determines which data shard)
# </claudes_code_comments>

import os
import random
import torch
import torch.distributed as dist
# given a message object, convert to prompt and response

PROMPT_USER: str = 'User: {input}\n\n'
PROMPT_ASSISTANT: str = 'Assistant:'  # should not have a space at the end
ASSISTANT_RESPONSE: str = ' {input}'

def get_formatted_question(line):
    return PROMPT_USER.format(input=str(line).strip()) + PROMPT_ASSISTANT

def get_formatted_answer(line):
    return ASSISTANT_RESPONSE.format(input=str(line).strip())

def get_formatted_input_and_target(messages, tokenizer, IGNORE_TOKEN_ID=-100, mask_prompt=True):
    input_ids = []
    target_ids = []
    for idx, message in enumerate(messages):
        if idx == 0:
            input_ids.extend([tokenizer.bos_token_id])
            target_ids.extend([tokenizer.bos_token_id])

        if message['role'] == "user":
            formatted_question = get_formatted_question(message['content'])
            tokenized_line = tokenizer.encode(formatted_question, add_special_tokens=False)
            input_ids.extend(tokenized_line)
            if mask_prompt:
                target_ids.extend([IGNORE_TOKEN_ID] * len(tokenized_line))
            else:
                target_ids.extend(tokenized_line)
        elif message['role'] == "assistant":
            formatted_answer = get_formatted_answer(message['content'])
            tokenized_line = tokenizer.encode(formatted_answer, add_special_tokens=False) + [tokenizer.eos_token_id]
            input_ids.extend(tokenized_line)
            if message.get('mask', 0) == 1:
                target_ids.extend([IGNORE_TOKEN_ID] * len(tokenized_line))
            else:
                target_ids.extend(tokenized_line)
        else:
            assert False, f"Unknown role: {message['role']}"

    return [input_ids, target_ids]


def get_examples_from_buffer_pad(buffer, seq_length, tokenizer, random_concat_ratio, IGNORE_TOKEN_ID=-100):
    all_input_ids_list, all_target_ids_list = [], []
    all_input_ids, all_target_ids = [], []

    for input_ids, target_ids in buffer:
        if len(input_ids) > seq_length - len(all_input_ids):
            input_ids = input_ids[-(seq_length - len(all_input_ids)):]
            target_ids = target_ids[-(seq_length - len(all_target_ids)):]
        if len(all_input_ids) > 0 and random.random() < random_concat_ratio:
            input_ids = input_ids[1:]
            target_ids = target_ids[1:]
        all_input_ids.extend(input_ids)
        all_target_ids.extend(target_ids)
        if len(all_input_ids) >= seq_length:
            assert len(all_input_ids) == seq_length, f"{len(all_input_ids)=}, {seq_length=}, {len(buffer)=}"
            all_input_ids_list.append(all_input_ids)
            all_target_ids_list.append(all_target_ids)
            all_input_ids, all_target_ids = [], []

    all_input_ids = all_input_ids + [tokenizer.pad_token_id for i in range(seq_length - len(all_input_ids))]
    all_target_ids = all_target_ids + [IGNORE_TOKEN_ID for i in range(seq_length - len(all_target_ids))]
    all_input_ids_list.append(all_input_ids)
    all_target_ids_list.append(all_target_ids)

    if len(all_input_ids) <= 0:
        return None
    return {
        "input_ids": torch.tensor(all_input_ids_list, dtype=torch.long),
        "labels": torch.tensor(all_target_ids_list, dtype=torch.long)
    }


def init_parallel_groups(ep_size=1):
    dist.init_process_group("nccl")
    world_size = int(os.getenv("WORLD_SIZE", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    ep_group = edp_group = None
    for i in range(0, world_size, ep_size):
        ranks = list(range(i, i + ep_size))
        group = dist.new_group(ranks)
        if local_rank in ranks:
            ep_group = group
    edp_group = None
    for i in range(ep_size):
        ranks = list(range(i, world_size, ep_size))
        group = dist.new_group(ranks)
        if local_rank in ranks:
            edp_group = group
    dist.all_reduce(torch.zeros(1, device="cuda"), group=ep_group)
    dist.all_reduce(torch.zeros(1, device="cuda"), group=edp_group)
    return world_size, local_rank, ep_group, edp_group
