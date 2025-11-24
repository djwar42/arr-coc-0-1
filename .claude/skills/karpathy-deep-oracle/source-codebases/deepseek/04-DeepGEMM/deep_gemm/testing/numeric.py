# <claudes_code_comments>
# ** Function List **
# calc_diff(x, y) - Calculate normalized difference between tensors using cosine-based similarity
# count_bytes(*tensors) - Recursively count total memory bytes across tensors/tuples/lists
#
# ** Technical Review **
# Numeric testing utilities for DeepGEMM kernel validation.
#
# calc_diff: Measures tensor difference via 1 - cosine_similarity. Returns normalized error in [0, 2] range.
# Formula: 1 - (2 * dot(x, y) / (||x||² + ||y||²)). Perfect match = 0, opposite = 2.
# Uses FP64 for numerical stability during accumulation. Denominator = sum of squared norms prevents
# division by zero for zero tensors. Symmetric measure: calc_diff(x, y) == calc_diff(y, x).
#
# count_bytes: Recursively counts memory footprint for validation. Handles nested tuples/lists
# (common for FP8 quantized tensors: (data, scaling_factors)). Uses numel() * element_size()
# for accurate byte count including different dtypes. Skips None tensors. Used for bandwidth
# calculations: GB/s = count_bytes(inputs, outputs) / execution_time / 1e9.
# </claudes_code_comments>

import torch
from typing import Iterable


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def count_bytes(*tensors):
    total = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            total += count_bytes(*t)
        elif t is not None:
            total += t.numel() * t.element_size()
    return total
