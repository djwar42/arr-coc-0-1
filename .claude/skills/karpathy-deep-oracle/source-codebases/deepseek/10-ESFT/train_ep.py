# <claudes_code_comments>
# ** Function List **
# main() - orchestrates ESFT training with expert parallelism (EP) and expert data parallelism (EDP)
# data_collator(data) - simple batching function for input_ids and labels
# custom_save_model(self, output_dir, _internal_call) - coordinator for distributed checkpoint saving
# custom_save(self, output_dir, state_dict) - handles sharded expert state gathering and merging
# custom_forward(self, hidden_states) - ensures gradient flow through MoE layers in EP mode
# custom_backward(self, loss, **kwargs) - manages expert gradient all-reduce across EDP group
#
# ** Technical Review **
# Expert-parallel (EP) training for ESFT when models don't fit on single GPU even after freezing.
# Implements 2D parallelism: Expert Parallel (EP) × Expert Data Parallel (EDP).
#
# EXPERT PARALLELISM (EP) ARCHITECTURE:
# Problem: DeepSeek-V2 has 64 experts/layer. Even with ESFT freezing 60%, 25 trainable experts × 27 layers
# = 675 experts still don't fit on single 80GB GPU.
#
# Solution: Shard experts across ep_size GPUs. Example with ep_size=4:
#   GPU 0: experts 0-15 (16 experts)
#   GPU 1: experts 16-31 (16 experts)
#   GPU 2: experts 32-47 (16 experts)
#   GPU 3: experts 48-63 (16 experts)
#
# During forward/backward, MoE layer performs All-to-All communication to route tokens to correct GPU.
# Each GPU only holds 1/ep_size of experts, reducing memory by ep_size factor.
#
# 2D PARALLELISM: EP × EDP:
# EP group (ep_group): GPUs holding different expert shards (expert parallel)
# EDP group (edp_group): GPUs holding same expert shards but different data (data parallel)
#
# Example with 8 GPUs, ep_size=4:
#   EP groups: [0,1,2,3], [4,5,6,7]  (experts 0-15 on GPU0 and GPU4, experts 16-31 on GPU1 and GPU5, ...)
#   EDP groups: [0,4], [1,5], [2,6], [3,7]  (GPU0 and GPU4 have same experts, process different batches)
#
# Gradient synchronization:
# - Non-expert params (attention, embeddings): Standard DDP all-reduce across all GPUs
# - Expert params: Manual all-reduce ONLY within EDP group (custom_backward)
#   Why manual? DDP can't handle experts split across EP group - would average gradients incorrectly
#
# CRITICAL IMPLEMENTATION DETAILS:
#
# 1. DDP EXPERT EXCLUSION (line 105):
#    model._ddp_params_and_buffers_to_ignore = [n for n, _ in model.named_parameters() if ".expert" in n]
#    Tells PyTorch DDP: "Don't manage expert params, I'll handle them manually"
#    Otherwise DDP would try to all-reduce expert grads across all 8 GPUs (wrong - should only reduce within EDP group)
#
# 2. DUMMY PARAMETER (line 107):
#    model.dummy = torch.nn.Parameter(torch.zeros(1, dtype=model.dtype))
#    When non_expert_modules=False (only training experts), DDP would fail with "no parameters to sync"
#    Dummy param ensures DDP has something to manage, preventing initialization errors
#
# 3. GRADIENT FLOW FORCING (lines 115-121):
#    Custom forward hook: hidden_states.requires_grad_(torch.is_grad_enabled())
#    Why needed: When all experts in first few layers are frozen, backward pass might not reach them
#    This forces gradients to flow through All-to-All operations even when experts don't train
#    Ensures backward All-to-All runs same number of times as forward (required for collective ops)
#
# 4. CUSTOM CHECKPOINT SAVING (lines 132-247):
#    Challenge: Each EP rank holds different expert shards, but checkpoint needs all experts
#    Solution: Coordinated 3-phase save:
#      Phase 1 (all EP ranks): Save local expert state to expert_state_{rank}.bin
#      Phase 2 (EP rank 0 only): Load all expert_state_{rank}.bin files, merge into full state_dict
#      Phase 3 (EP rank 0 only): Save merged checkpoint with original_save()
#      dist.barrier() between phases ensures synchronization
#
#    Optimizer state handling (if --save_opt_states):
#      Each EP rank saves expert optimizer states (momentum, variance)
#      EP rank 0 loads and merges all optimizer states into global optimizer.pt
#      Uses parameter name → optimizer index mapping to correctly merge states
#
#    Retry logic (lines 207-228): File I/O race conditions handled with 3 retries + 1s sleep
#
# 5. EXPERT GRADIENT REDUCTION (lines 251-260):
#    custom_backward() replaces accelerator.backward
#    After standard backward pass, manually all-reduces expert gradients within EDP group
#    Uses dist.ReduceOp.AVG to average gradients across edp_size replicas
#    Only runs when sync_gradients=True (respects gradient accumulation schedule)
#
# MEMORY CALCULATION:
# Without EP: Model (236B param) + Optimizer (2x) = ~708GB (doesn't fit any GPU)
# With ESFT: 20% experts trainable = ~142GB model + 28GB optimizer = 170GB (still too big)
# With ESFT + EP (ep_size=4): 170GB / 4 = ~42.5GB per GPU (fits 80GB A100!)
#
# ALL-TO-ALL COMMUNICATION:
# Forward: Each GPU sends tokens to GPU with assigned experts, receives tokens for its experts
# Backward: Reverse All-to-All to return gradients to original GPU
# Bandwidth critical: 8 GPUs × 50GB/s = 400GB/s aggregate All-to-All bandwidth required
#
# HYPERPARAMETER CONSIDERATIONS:
# - ep_size: Higher reduces memory but increases communication overhead. Optimal: 2-8 for most models
# - gradient_accumulation_steps: Can increase to compensate for smaller per-GPU batch size
# - NCCL_AVOID_RECORD_STREAMS=1: Reduces memory fragmentation in collective ops
#
# WHEN TO USE train_ep.py vs train.py:
# - Use train.py: Model fits on GPU after ESFT (most cases)
# - Use train_ep.py: Model still too large after ESFT, need expert sharding (very large models)
# </claudes_code_comments>

import argparse
import json
import yaml
import os
import random
import torch
import torch.distributed as dist
from types import MethodType
from torch.utils.data import TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, logging

from benchmarks import *
from utils import get_formatted_input_and_target, get_examples_from_buffer_pad, init_parallel_groups
from esft import to_esft
from deepseek.modeling_deepseek import DeepseekV2ForCausalLM
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_AVOID_RECORD_STREAMS"] = "1"
logging.set_verbosity_error()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--expert_config", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--save_opt_states", action="store_true", help="Whether to save optimizer states")
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

    ep_size = config.get("ep_size", 1)
    world_size, local_rank, ep_group, edp_group = init_parallel_groups(ep_size)
    edp_size = world_size // ep_size

    # Prepare data
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
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
        disable_tqdm=False,
        gradient_checkpointing=config['gradient_checkpointing'],
        gradient_checkpointing_kwargs={"use_reentrant": False} if config['gradient_checkpointing'] else {}, # if set to True, backward will raise bug
    )

    def data_collator(data):
        input_ids = torch.stack([item[0] for item in data])
        labels = torch.stack([item[1] for item in data])
        return {"input_ids": input_ids, "labels": labels}


    model = DeepseekV2ForCausalLM.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                  ep_size=ep_size, attn_implementation="flash_attention_2")
    model._ddp_params_and_buffers_to_ignore = [n for n, _ in model.named_parameters() if ".expert" in n]    # we manage grad synchronization of expert parameters
    to_esft(model, expert_config)
    model.dummy = torch.nn.Parameter(torch.zeros(1, dtype=model.dtype))    # prevent DDP from having no trainable parameters
    model._keys_to_ignore_on_save = ["dummy"]
    expert_params = [p for n, p in model.named_parameters() if p.requires_grad and ".expert" in n]
    for layer in model.model.layers:
        if type(layer.mlp).__name__ != "DeepseekV2MoE":
            continue
        layer.mlp.ep_group = ep_group
    # Force all2all backward the same number of times
    if ep_size > 1 and not expert_config["non_expert_modules"]:
        min_layer_id = min(int(k) for k, v in expert_config["experts"].items() if v)
        mlp = model.model.layers[min_layer_id].mlp
        forward = mlp.forward
        def custom_forward(self, hidden_states: torch.Tensor):
            return forward(hidden_states.requires_grad_(torch.is_grad_enabled()))
        mlp.forward = MethodType(custom_forward, mlp)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )

    original_save_model = trainer.save_model
    def custom_save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
            
        # Ensure all ranks participate in saving
        self._save(output_dir)
        dist.barrier()
        
    trainer.save_model = MethodType(custom_save_model, trainer)

    original_save = trainer._save
    def custom_save(self, output_dir=None, state_dict=None):
        ep_rank = ep_group.rank()
        edp_rank = edp_group.rank()        
        os.makedirs(output_dir, exist_ok=True)
        
        if ep_rank > 0 and edp_rank == 0:
            # Save expert model state
            expert_state = {k: v for k, v in self.model.state_dict().items() if ".expert" in k}
            expert_save_path = os.path.join(output_dir, f"expert_state_{ep_rank}.bin")
            
            # Save expert optimizer state only if save_opt_states is True
            if args.save_opt_states:
                optimizer = self.optimizer
                opt_state_dict = optimizer.state_dict()
                
                # Create a mapping from parameter id to parameter name
                id_to_name = {}
                for name, param in self.model.named_parameters():
                    if ".expert" in name:
                        id_to_name[id(param)] = name
                
                # Get the mapping from optimizer state index to parameter
                param_to_idx = {param: idx for idx, param in enumerate(optimizer.param_groups[0]['params'], 1)}
                
                # Save optimizer state using parameter names as keys
                expert_opt_state = {'state': {}, 'param_groups': opt_state_dict['param_groups']}
                for param, idx in param_to_idx.items():
                    if id(param) in id_to_name:
                        param_name = id_to_name[id(param)]
                        if idx in opt_state_dict['state']:
                            expert_opt_state['state'][param_name] = opt_state_dict['state'][idx]
                
                expert_opt_path = os.path.join(output_dir, f"expert_optimizer_{ep_rank}.bin")
                # Save optimizer state
                temp_opt_path = expert_opt_path + ".tmp"
                torch.save(expert_opt_state, temp_opt_path)
                os.replace(temp_opt_path, expert_opt_path)
            
            # Save model state
            temp_expert_path = expert_save_path + ".tmp"
            torch.save(expert_state, temp_expert_path)
            os.sync()
            os.replace(temp_expert_path, expert_save_path)
        
        dist.barrier()
            
        if local_rank == 0:
            original_state = self.model.state_dict()
            
            if args.save_opt_states:
                optimizer_state = self.optimizer.state_dict()
                # Create a mapping from parameter name to optimizer index for the current session
                name_to_idx = {}
                for name, param in self.model.named_parameters():
                    if ".expert" in name:
                        idx = next((i for i, p in enumerate(self.optimizer.param_groups[0]['params'], 1) if id(p) == id(param)), None)
                        if idx is not None:
                            name_to_idx[name] = idx
            
            time.sleep(1)
            
            # load expert state and optimizer state from all ranks
            for rank in range(1, ep_size):
                expert_path = os.path.join(output_dir, f"expert_state_{rank}.bin")
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        expert_state = torch.load(expert_path)
                        if args.save_opt_states:
                            opt_path = os.path.join(output_dir, f"expert_optimizer_{rank}.bin")
                            expert_opt_state = torch.load(opt_path)
                            
                            # Convert parameter names back to indices for the optimizer state
                            for param_name, state in expert_opt_state['state'].items():
                                if param_name in name_to_idx:
                                    idx = name_to_idx[param_name]
                                    optimizer_state['state'][idx] = state
                        
                        # Update model state
                        original_state.update(expert_state)
                        break
                    except Exception as e:
                        if retry == max_retries - 1:
                            raise
                        time.sleep(1)
            
            original_save(output_dir, state_dict=original_state)
            
            # Save complete optimizer state if enabled
            if args.save_opt_states:
                opt_save_path = os.path.join(output_dir, "optimizer.pt")
                torch.save(optimizer_state, opt_save_path)
            
            # remove those intermediate .bin files
            for rank in range(1, ep_size):
                os.remove(os.path.join(output_dir, f"expert_state_{rank}.bin"))
                if args.save_opt_states:
                    os.remove(os.path.join(output_dir, f"expert_optimizer_{rank}.bin"))
        
        dist.barrier()
        tokenizer.save_pretrained(output_dir)

    
    trainer._save = MethodType(custom_save, trainer)

    accelerator = trainer.accelerator
    backward = accelerator.backward
    def custom_backward(self, loss, **kwargs):
        backward(loss, **kwargs)
        if not self.sync_gradients or edp_size == 1:
            return
        for p in expert_params:
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            dist.all_reduce(g, op=dist.ReduceOp.AVG, group=edp_group)
            if p.grad is not g:
                p.grad = g
    accelerator.backward = MethodType(custom_backward, accelerator)

    trainer.train()

    print("Training complete")

if __name__ == "__main__":
    main()
