# CUTLASS - NVIDIA GEMM Library - Study

**Source**: NVIDIA Docs (Welcome to CUTLASS - NVIDIA Docs Hub.md)
**Date Processed**: 2025-10-29
**Category**: LOW - Hardware Library / Low-Level Implementation

---

## 📝 TL;DR

CUTLASS = CUDA C++ templates + Python DSLs for high-performance GEMM (matrix multiply). Targets Tensor Cores. Supports mixed precision (FP64→FP4, INT4, BF16, etc). Used by cuBLAS/cuDNN internally. This is the low-level machinery that makes DeepSeek's FP8 training fast on NVIDIA hardware.

---

## 🎯 Key Concepts

### What It Is
- Collection of CUDA templates for matrix multiplication
- Hierarchical decomposition (like cuBLAS but customizable)
- C++ templates + Python DSL (CuTe) for kernel development

### Supported Data Types
- **Floating-point**: FP64, FP32, TF32, FP16, BF16
- **8-bit FP**: e5m2, e4m3
- **Block scaled**: NVFP4, MXFP4/6/8
- **Integer**: INT4, INT8
- **Binary**: 1-bit

### Target Hardware
- NVIDIA Tensor Cores (Ampere → Hopper → Blackwell)
- Peak performance GEMM on modern GPUs

---

## 💡 Why This Matters

**Foundation for FP8**: DeepSeek's FP8-LM training uses kernels like this under the hood. You need efficient low-precision GEMM to make FP8 practical.

**Customization**: Unlike cuBLAS (black box), CUTLASS lets you customize tiling, data movement, and algorithmic policies. Useful for research.

---

## 🔧 Karpathy-Style Implementation Notes

```python
# You typically don't call CUTLASS directly
# It's used by PyTorch/JAX/TF under the hood

# But if you're writing custom CUDA kernels:
# 1. Include CUTLASS templates
# 2. Configure tiling sizes, data types
# 3. Let templates generate optimized GEMM
# 4. Get near-cuBLAS performance with custom behavior
```

Most ML practitioners never touch this - PyTorch abstracts it away. But if you're debugging why your FP8 training is slow, CUTLASS is where the magic (or lack thereof) happens.

---

## 🔗 Connections

- **05-fp8-lm-paper**: Uses FP8 GEMM kernels (likely CUTLASS-based)
- **10-fine-grained-fp8**: FP8 quantization that needs efficient kernels
- **19-vllm-mla-fp8-optimization**: vLLM uses optimized GEMM for MLA

---

## 💭 Karpathy Take

This is plumbing. Important plumbing, but plumbing. If you're training models, you care that FP8 is fast. CUTLASS is WHY it's fast - optimized Tensor Core kernels.

The Python DSL (CuTe) is interesting - lets you prototype CUDA kernels without writing raw C++. Still niche though, most people just use torch.compile or JAX's XLA and let the compiler figure it out.

If you're reading this to understand DeepSeek: just know that their FP8 training runs on GPU kernels like this. The actual kernel code is way down in the stack - PyTorch → cuBLAS/CUTLASS → Tensor Cores. ¯\_(ツ)_/¯

Not directly relevant unless you're optimizing kernel-level performance.
