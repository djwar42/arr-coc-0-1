# DeepGEMM: GEMM Kernel Optimizations

**Status**: Preliminary structure - awaiting deep dive
**Codebase**: DeepSeek GEMM (General Matrix Multiply) Optimizations

---

## 🎯 What This Codebase Does

**DeepGEMM** provides highly optimized GEMM kernels for Tensor Core acceleration.

**Key Innovation**: Custom kernels achieving near-theoretical peak performance on NVIDIA GPUs

**Optimization Techniques**:
- Tensor Core utilization (FP16, BF16, FP8, INT8)
- Memory hierarchy optimization (shared memory, registers)
- Warp-level programming
- Instruction-level parallelism

---

## 📁 Expected Codebase Structure

```
02-DeepGEMM/
├── 00-OVERVIEW.md           # This file
├── 01-architecture.md       # Kernel design principles (TO BE CREATED)
├── 02-tensor-cores.md       # Tensor Core programming (TO BE CREATED)
├── 03-memory-hierarchy.md   # Cache optimization (TO BE CREATED)
├── 04-fp8-kernels.md        # FP8-specific kernels (TO BE CREATED)
├── 05-benchmarks.md         # Performance analysis (TO BE CREATED)
├── code-snippets/           # CUDA code with line numbers (TO BE CREATED)
└── examples/                # Usage examples (TO BE CREATED)
```

---

## 🔍 What Needs to Be Done

### Phase 1: Architecture Analysis
- [ ] Document kernel hierarchy
- [ ] Explain tile size selection
- [ ] Map memory access patterns
- [ ] Show instruction scheduling

### Phase 2: Code Deep Dive
- [ ] Extract CUDA kernels with line numbers
- [ ] Document Tensor Core APIs
- [ ] Explain shared memory usage
- [ ] Show performance optimizations

### Phase 3: Usage Documentation
- [ ] Kernel invocation examples
- [ ] Performance tuning guide
- [ ] Profiling methodology
- [ ] Comparison with cuBLAS

---

## 🔗 Related Knowledge

**Will connect to**:
- Knowledge category: `training-efficiency/02-memory-optimization.md`
- Cross-reference: 3FS (FP8 training that uses these kernels)
- Hardware: NVIDIA Tensor Core architecture

---

## 📝 Next Steps

1. Locate CUDA source files
2. Identify key kernel implementations
3. Document optimization techniques
4. Extract code snippets with line numbers
5. Create performance benchmarks
6. Compare with standard libraries

---

**Last Updated**: 2025-10-28
**Status**: Awaiting Phase 4 deep dive
