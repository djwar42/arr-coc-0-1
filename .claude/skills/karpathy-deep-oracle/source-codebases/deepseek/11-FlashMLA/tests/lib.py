"""Test Utility Functions for FlashMLA"""

# <claudes_code_comments>
# ** Function List **
# cdiv(x, y) - Integer ceiling division (rounds up)
# check_is_allclose(name, ans, ref, abs_tol, rel_tol, cos_diff_tol) - Comprehensive tensor equality check
# get_cos_diff(x, y) - Computes cosine difference between tensors (1 - cosine_similarity)
# deal_with_anomalies(val) - Handles inf/nan matching between tensors
# get_pos_in_tensor(t, pos) - Converts flat index to multi-dimensional position
#
# ** Technical Review **
# This module provides utility functions for FlashMLA test suite, focusing on robust
# numerical comparison between kernel outputs and reference implementations.
#
# Ceiling Division (cdiv):
# - Computes ⌈x/y⌉ without floating-point arithmetic
# - Formula: (x + y - 1) // y
# - Use cases: Block count calculation, padding computation
# - Example: cdiv(100, 64) = 2 (need 2 blocks of size 64 to fit 100 elements)
#
# Tensor Comparison (check_is_allclose):
# Implements multi-criteria validation strategy combining:
# 1. **Absolute tolerance**: |ans - ref| < abs_tol (default 1e-5)
# 2. **Relative tolerance**: |ans - ref| / |ref| < rel_tol (default 1e-2)
# 3. **Cosine difference**: 1 - cosine_similarity(ans, ref) < cos_diff_tol (default 1e-7)
#
# Why Three Criteria?
# - Absolute: Catches errors in near-zero values where relative fails
# - Relative: Scales to large values where absolute fails
# - Cosine: Global shape similarity, robust to scaling issues
# - Element passes if: (abs_err < abs_tol) OR (rel_err < rel_tol) AND (global cos_diff < cos_diff_tol)
#
# Anomaly Handling:
# - Checks inf, -inf, nan values separately
# - Requires ans and ref to have anomalies at same positions
# - Zeros out anomalies after checking to avoid interference with error metrics
# - Returns False if anomaly positions mismatch (e.g., ans has inf where ref is finite)
#
# Error Reporting:
# - Identifies position and value of maximum absolute error
# - Identifies position and value of maximum relative error
# - Reports percentage of passing elements
# - Provides cosine difference metric
# - Includes actual vs expected values at error positions
#
# Cosine Difference Formula:
# - cos_sim = 2 * sum(x * y) / (sum(x^2) + sum(y^2))
# - cos_diff = 1 - cos_sim
# - Range: [0, 2] where 0 = identical, 1 = orthogonal, 2 = opposite
# - Computed in float64 for numerical stability
#
# Usage Pattern:
# ```python
# out_kernel, lse_kernel = flash_mla_kernel(q, k, v, ...)
# out_ref, lse_ref = reference_torch(q, k, v, ...)
# assert check_is_allclose("output", out_kernel, out_ref)
# assert check_is_allclose("lse", lse_kernel, lse_ref, abs_tol=1e-3, rel_tol=5e-2)
# ```
#
# Tolerance Guidelines:
# - FP32 kernels: abs_tol=1e-5, rel_tol=1e-2
# - BF16 kernels: abs_tol=1e-3, rel_tol=5e-2
# - FP8 kernels: abs_tol=1e-2, rel_tol=1e-1
# - LSE (log-sum-exp): Often needs looser tolerances due to exp instability
#
# Implementation Notes:
# - Converts to float32 before comparison for consistency
# - Masks small errors: abs_err ignored if rel_err < rel_tol, and vice versa
# - This prevents false positives from noise in high/low magnitude regions
# </claudes_code_comments>

from typing import List

import torch

def cdiv(x: int, y: int):
    return (x+y-1) // y

def check_is_allclose(name: str, ans: torch.Tensor, ref: torch.Tensor, abs_tol: float = 1e-5, rel_tol: float = 1e-2, cos_diff_tol: float = 1e-7):
    """
    Check if two tensors are close enough
    """
    def get_cos_diff(x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Calculate the cosine diff between two tensors
        """
        x, y = x.double(), y.double()
        denominator = (x * x + y * y).sum().item()
        if denominator == 0:
            return 0
        sim = 2 * (x * y).sum().item() / denominator
        return 1 - sim
    assert ans.shape == ref.shape, f"`{name}` Shape mismatch: {ans.shape} vs {ref.shape}"
    
    ans = ans.clone().to(torch.float)
    ref = ref.clone().to(torch.float)

    # Deal with anomalies
    def deal_with_anomalies(val: float):
        ref_mask = (ref == val) if (val == val) else (ref != ref)
        ans_mask = (ans == val) if (val == val) else (ans != ans)
        ref[ref_mask] = 0.0
        ans[ans_mask] = 0.0
        if not torch.equal(ref_mask, ans_mask):
            print(f"`{name}` Anomaly number `{val}` mismatch: {ans_mask.sum().item()} in ans but {ref_mask.sum().item()} in ref")
            return False
        return True
    
    anomalies_check_passed = True
    anomalies_check_passed &= deal_with_anomalies(float("inf"))
    anomalies_check_passed &= deal_with_anomalies(float("-inf"))
    anomalies_check_passed &= deal_with_anomalies(float("nan"))

    if not anomalies_check_passed:
        return False

    cos_diff = get_cos_diff(ans, ref)
    raw_abs_err = torch.abs(ans-ref)
    raw_rel_err = raw_abs_err / (torch.abs(ref)+(1e-6))
    rel_err = raw_rel_err.masked_fill(raw_abs_err<abs_tol, 0)
    abs_err = raw_abs_err.masked_fill(raw_rel_err<rel_tol, 0)
    pass_mask = (abs_err < abs_tol) | (rel_err < rel_tol)

    if not pass_mask.all():
        print(f"`{name}` mismatch")
        max_abs_err_pos: int = torch.argmax(abs_err, keepdim=True).item()   # type: ignore
        max_rel_err_pos: int = torch.argmax(rel_err, keepdim=True).item()   # type: ignore
        def get_pos_in_tensor(t: torch.Tensor, pos: int) -> List[int]:
            result = []
            for size in t.shape[::-1]:
                result.append(pos % size)
                pos = pos // size
            assert pos == 0
            return result[::-1]
        print(f"max abs err: {torch.max(abs_err).item()}: pos {get_pos_in_tensor(ans, max_abs_err_pos)}, {ans.reshape(-1)[max_abs_err_pos].item()} vs {ref.reshape(-1)[max_abs_err_pos].item()}")
        print(f"max rel err: {torch.max(rel_err).item()}: pos {get_pos_in_tensor(ans, max_rel_err_pos)}, {ans.reshape(-1)[max_rel_err_pos].item()} vs {ref.reshape(-1)[max_rel_err_pos].item()}")
        print(f"{pass_mask.sum()} out of {pass_mask.numel()} passed ({pass_mask.sum()/pass_mask.numel()*100.0:.2f}%)")
        print(f"Cosine diff: {cos_diff} (threshold: {cos_diff_tol})")
        return False
    else:
        if abs(cos_diff) > cos_diff_tol:
            print(f"`{name}` mismatch: Cosine diff too large: {cos_diff} vs {cos_diff_tol})")
            return False
        return True