# <claudes_code_comments>
# ** Function List **
# main(hf_ckpt_path, save_path, n_experts, mp) - Convert HuggingFace weights to model-parallel format
#
# ** Technical Review **
# Converts DeepSeek-V3 weights from HuggingFace format to the custom format expected by model.py,
# with support for model parallelism (tensor and expert parallelism). This enables distributing
# the 671B model across multiple GPUs for efficient inference.
#
# **Conversion Process**:
# 1. Loads HuggingFace SafeTensors weights from hf_ckpt_path
# 2. Renames parameters using mapping dict (HF names → custom names)
# 3. Shards parameters across mp (model parallel) ranks for distributed inference
# 4. Saves mp separate files: model0-mp{mp}.safetensors, model1-mp{mp}.safetensors, ...
#
# **Parameter Sharding Strategy**:
# - Tensor parallelism (dim=0 or dim=1): splits weight matrices across ranks
#   - dim=0: splits output dimension (e.g., wq, w1, w3)
#   - dim=1: splits input dimension (e.g., wo, w2)
# - Expert parallelism: assigns different experts to different ranks
#   - Each rank gets n_experts//mp consecutive experts
#   - Expert idx must be in range [rank*n_local_experts, (rank+1)*n_local_experts]
# - No sharding (dim=None): replicated across all ranks (e.g., layer norms, gate)
#
# **Name Mapping** (lines 11-30):
# Translates HuggingFace parameter names to custom implementation names:
# - "embed_tokens" → "embed" (embedding layer)
# - "q_proj"/"q_a_proj"/"q_b_proj" → "wq"/"wq_a"/"wq_b" (MLA query projections)
# - "kv_a_proj_with_mqa"/"kv_b_proj" → "wkv_a"/"wkv_b" (MLA KV projections)
# - "o_proj" → "wo" (attention output)
# - "gate_proj"/"up_proj"/"down_proj" → "w1"/"w3"/"w2" (MLP/Expert layers)
# - "lm_head" → "head" (output projection)
#
# **Special Handling**:
# - Skips "model.layers.61": Multi-Token Prediction (MTP) module (line 53-54)
#   MTP is separate functionality not needed for standard inference
# - Copies tokenizer files: *token* files copied unchanged (line 83-85)
# - "weight_scale_inv" → "scale": FP8 quantization scales (line 60)
# - "e_score_correction_bias" → "bias": MoE gate bias for load balancing (line 61)
#
# **Usage Example**:
# python convert.py --hf-ckpt-path /path/to/DeepSeek-V3 --save-path /path/to/output
#                   --n-experts 256 --model-parallel 16
# Creates 16 sharded weight files for 16-GPU deployment (each GPU handles 16 experts)
#
# This conversion is required before using generate.py, as it prepares weights for the custom
# model parallelism and expert parallelism implementations in model.py.
# </claudes_code_comments>

import os
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file


mapping = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate": ("gate", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", 0),
    "scale": ("scale", None),
}


def main(hf_ckpt_path, save_path, n_experts, mp):
    """
    Converts and saves model checkpoint files into a specified format.

    Args:
        hf_ckpt_path (str): Path to the directory containing the input checkpoint files.
        save_path (str): Path to the directory where the converted checkpoint files will be saved.
        n_experts (int): Total number of experts in the model.
        mp (int): Model parallelism factor.
        
    Returns:
        None
    """
    torch.set_num_threads(8)
    n_local_experts = n_experts // mp
    state_dicts = [{} for _ in range(mp)]

    for file_path in tqdm(glob(os.path.join(hf_ckpt_path, "*.safetensors"))):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if "model.layers.61" in name:
                    continue
                param: torch.Tensor = f.get_tensor(name)
                if name.startswith("model."):
                    name = name[len("model."):]
                name = name.replace("self_attn", "attn")
                name = name.replace("mlp", "ffn")
                name = name.replace("weight_scale_inv", "scale")
                name = name.replace("e_score_correction_bias", "bias")
                key = name.split(".")[-2]
                assert key in mapping, f"Key {key} not found in mapping"
                new_key, dim = mapping[key]
                name = name.replace(key, new_key)
                for i in range(mp):
                    new_param = param
                    if "experts" in name and "shared_experts" not in name:
                        idx = int(name.split(".")[-3])
                        if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts:
                            continue
                    elif dim is not None:
                        assert param.size(dim) % mp == 0, f"Dimension {dim} must be divisible by {mp}"
                        shard_size = param.size(dim) // mp
                        new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()
                    state_dicts[i][name] = new_param

    os.makedirs(save_path, exist_ok=True)

    for i in trange(mp):
        save_file(state_dicts[i], os.path.join(save_path, f"model{i}-mp{mp}.safetensors"))

    for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--model-parallel", type=int, required=True)
    args = parser.parse_args()
    assert args.n_experts % args.model_parallel == 0, "Number of experts must be divisible by model parallelism"
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)
