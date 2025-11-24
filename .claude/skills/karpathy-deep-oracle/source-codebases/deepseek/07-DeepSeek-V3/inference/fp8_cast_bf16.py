# <claudes_code_comments>
# ** Function List **
# get_tensor(tensor_name) - Retrieve tensor from cached safetensor files
# main(fp8_path, bf16_path) - Convert FP8 weights to BF16 format
#
# ** Technical Review **
# Utility script to convert DeepSeek-V3 weights from FP8 (8-bit floating point) to BF16
# (16-bit brain float) format. Required for inference frameworks that don't support FP8,
# or when BF16 precision is preferred over FP8's memory efficiency.
#
# **Conversion Process**:
# DeepSeek-V3 natively uses FP8 training, so official weights are distributed in FP8 format.
# This script converts them to BF16 for compatibility:
# 1. Load FP8 weights and their scale_inv (inverse scale) tensors from input directory
# 2. Dequantize: weight_bf16 = weight_fp8 * scale_inv (using kernel.weight_dequant)
# 3. Save BF16 weights to output directory
# 4. Update model.safetensors.index.json to remove scale_inv references
#
# **FP8 Weight Format**:
# FP8 weights stored with companion scale_inv tensors:
# - weight: FP8 quantized values (element_size() == 1)
# - weight_scale_inv: FP32 inverse scales for dequantization (1/scale)
# - Block-wise scales: each (128×128) block has its own scale_inv value
# - Dequantization: multiply each FP8 value by its block's scale_inv
#
# **Memory Management** (lines 90-94):
# Processes large model (671B parameters) with limited GPU memory:
# - Loads one safetensor file at a time
# - Caches at most 2 recent files to handle cross-file scale_inv lookups
# - Explicitly calls torch.cuda.empty_cache() after discarding old files
# - Prevents OOM errors when converting full 685B parameter model
#
# **Cross-file Scale Lookups** (lines 44-61):
# Some weights and their scale_inv tensors may be in different safetensor files:
# - get_tensor() helper caches loaded files and retrieves from correct file
# - weight_map (from model.safetensors.index.json) maps tensor_name → file_name
# - Ensures scale_inv can be found regardless of file organization
#
# **Index File Update** (lines 96-103):
# After conversion, BF16 weights no longer need scale_inv tensors:
# - Removes all "*_scale_inv" entries from weight_map
# - Creates new model.safetensors.index.json in output directory
# - Ensures loading code doesn't expect missing scale_inv tensors
#
# **Usage**:
# python fp8_cast_bf16.py --input-fp8-hf-path /path/to/fp8 --output-bf16-hf-path /path/to/bf16
# - Typical runtime: ~10-20 minutes for full 671B model on GPU
# - Output size: ~1.34TB (2x FP8 size due to BF16 being 2 bytes vs 1 byte)
#
# **When to Use**:
# - Inference framework lacks FP8 support (e.g., older vLLM versions)
# - Need full BF16 precision for research/debugging
# - Hardware doesn't efficiently support FP8 operations
#
# **Tradeoffs**:
# - Doubles weight storage: 1 byte → 2 bytes per parameter
# - Increases memory bandwidth: 2x more data movement during inference
# - May improve accuracy slightly (BF16 > FP8 precision)
# - Slower inference on FP8-optimized hardware (wastes specialized units)
#
# Most production deployments should use native FP8 for 2x memory savings without
# significant quality loss. BF16 conversion is primarily for compatibility.
# </claudes_code_comments>

import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file

from kernel import weight_dequant

def main(fp8_path, bf16_path):
    """
    Converts FP8 weights to BF16 and saves the converted weights.

    This function reads FP8 weights from the specified directory, converts them to BF16,
    and saves the converted weights to another specified directory. It also updates the
    model index file to reflect the changes.

    Args:
    fp8_path (str): The path to the directory containing the FP8 weights and model index file.
    bf16_path (str): The path to the directory where the converted BF16 weights will be saved.

    Raises:
    KeyError: If a required scale_inv tensor is missing for a weight.

    Notes:
    - The function assumes that the FP8 weights are stored in safetensor files.
    - The function caches loaded safetensor files to optimize memory usage.
    - The function updates the model index file to remove references to scale_inv tensors.
    """
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    
    # Cache for loaded safetensor files
    loaded_files = {}
    fp8_weight_names = []

    # Helper function to get tensor from the correct file
    def get_tensor(tensor_name):
        """
        Retrieves a tensor from the cached safetensor files or loads it from disk if not cached.

        Args:
            tensor_name (str): The name of the tensor to retrieve.

        Returns:
            torch.Tensor: The retrieved tensor.

        Raises:
            KeyError: If the tensor does not exist in the safetensor file.
        """
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cuda")
        return loaded_files[file_name][tensor_name]

    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = current_state_dict
        
        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            if weight_name.endswith("_scale_inv"):
                continue
            elif weight.element_size() == 1:  # FP8 weight
                scale_inv_name = f"{weight_name}_scale_inv"
                try:
                    # Get scale_inv from the correct file
                    scale_inv = get_tensor(scale_inv_name)
                    fp8_weight_names.append(weight_name)
                    new_state_dict[weight_name] = weight_dequant(weight, scale_inv)
                except KeyError:
                    print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                    new_state_dict[weight_name] = weight
            else:
                new_state_dict[weight_name] = weight
                
        new_safetensor_file = os.path.join(bf16_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
        
        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            torch.cuda.empty_cache()
    
    # Update model index
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    for weight_name in fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map:
            weight_map.pop(scale_inv_name)
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True)
    parser.add_argument("--output-bf16-hf-path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)
    
