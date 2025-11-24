# Oracle Knowledge Expansion: PyTorch Compilation & CUDA Expertise (2025-02-03)

**Topic**: Become experts at CUDA and PyTorch compiling, speedups, GPU architectures
**Date**: 2025-02-03
**Runners**: 6 (one per knowledge area)
**Strategy**: Deep technical expertise for production VLM optimization
**Context**: arr-coc-0-1 project recompiles PyTorch from source on Cloud Build (~2-4 hours)

---

## PART 1: PyTorch Build System & Compilation from Source

- [✓] PART 1: Create cuda/02-pytorch-build-system-compilation.md (Completed 2025-11-13 19:21)

**Goal**: Master PyTorch compilation from source with custom CUDA flags

**Step 0: Check Existing Knowledge**
- [ ] Read cuda/00-streams-concurrency-async.md (mentions PyTorch CUDA API)
- [ ] Read cuda/01-memory-management-unified.md (mentions PyTorch memory)
- [ ] Read vertex-ai-production/01-gpu-optimization-deep.md (mentions compilation flags)
- [ ] Identify gap: No detailed PyTorch build system knowledge

**Step 1: Web Research - PyTorch Build System**
Search queries:
- "PyTorch compile from source CUDA CMake 2024 2025"
- "PyTorch setup.py build CUDA_ARCH_LIST CMAKE flags"
- "PyTorch build system torch/csrc C++ extensions"

Target sources:
- PyTorch GitHub (pytorch/pytorch) build documentation
- PyTorch official docs on building from source
- CMakeLists.txt configuration options
- CUDA architecture flag reference

**Step 2: Extract Key Topics**
- CMake configuration (USE_CUDA, CUDA_ARCH_LIST, CMAKE_BUILD_TYPE=Release)
- setup.py build process (python setup.py build, python setup.py install)
- Compilation flags verification (check .so files for sm_XX)
- Multi-architecture builds (fatbin for sm_75;8.0;8.6;9.0)
- Debugging compilation errors (missing CUDA headers, ABI mismatches)
- Build time optimization (ccache, ninja vs make, parallel jobs)

**Step 3: Write Knowledge File** (~700 lines)
```markdown
# PyTorch Build System & Compilation from Source

## Section 1: Build System Overview (~100 lines)
- PyTorch build architecture (CMake → ninja → setup.py)
- Dependencies (CUDA toolkit, cuDNN, NCCL, MKL)
- Build modes (Release, Debug, RelWithDebInfo)

## Section 2: CMake Configuration (~150 lines)
- CUDA flags (USE_CUDA, CUDA_ARCH_LIST, CUDA_NVCC_FLAGS)
- Optional features (USE_TENSORRT, USE_CUDNN, USE_NCCL)
- Performance flags (CMAKE_BUILD_TYPE, USE_STATIC_MKL)
- Example CMake invocations

## Section 3: Compilation Process (~150 lines)
- setup.py build workflow
- torch/csrc/ C++ compilation
- CUDA kernel compilation (.cu → .ptx → .cubin)
- Linking phase and shared libraries
- Installation and verification

## Section 4: Multi-Architecture Builds (~100 lines)
- CUDA_ARCH_LIST for multiple GPUs (sm_75, sm_80, sm_86, sm_90)
- Fatbin vs single-arch builds (size vs performance)
- Forward compatibility (PTX fallback)

## Section 5: Debugging & Optimization (~150 lines)
- Compilation error patterns (CUDA not found, ABI mismatch)
- Build time optimization (ccache, ninja, -j flag)
- Verifying compiled features (torch.cuda.get_arch_list())
- Custom build for arr-coc-0-1 (Cloud Build example)

## Section 6: ARR-COC Connection (~50 lines)
- Compiling for A100 (sm_80) with Tensor Cores
- Enabling FlashAttention-2 at compile time
- Custom CUDA extensions for texture processing
```

**Step 4: Create KNOWLEDGE DROP**
Individual summary file: `KNOWLEDGE-DROP-pytorch-compilation-2025-02-03-[TIME].md`

---

## PART 2: CUDA Compute Capabilities & GPU Architectures

- [✓] PART 2: Create cuda/03-compute-capabilities-gpu-architectures.md (Completed 2025-11-13 19:45)

**Goal**: Understand GPU architecture evolution and capability differences

**Step 0: Check Existing Knowledge**
- [ ] Read vertex-ai-production/01-gpu-optimization-deep.md (mentions A100/H100)
- [ ] Read karpathy/practical-implementation/32-vertex-ai-gpu-tpu.md (GPU types)
- [ ] Identify gap: No compute capability deep dive (sm_XX features)

**Step 1: Web Research - CUDA Compute Capabilities**
Search queries:
- "CUDA compute capability sm_75 sm_80 sm_90 feature comparison 2024"
- "NVIDIA GPU architecture Turing Ampere Hopper differences"
- "Tensor Cores generation 2 3 4 comparison"

Target sources:
- NVIDIA CUDA C Programming Guide (Compute Capabilities section)
- NVIDIA architecture whitepapers (Ampere, Hopper)
- CUDA compatibility matrix
- PTX ISA reference

**Step 2: Extract Key Topics**
- Compute capability matrix (sm_35 → sm_90+)
- Turing (sm_75): T4, RTX 20XX, Tensor Cores gen 2
- Ampere (sm_80, sm_86): A100, A10, RTX 30XX, Tensor Cores gen 3
- Ada (sm_89): L4, RTX 40XX, Tensor Cores gen 4
- Hopper (sm_90): H100, Tensor Cores gen 4, TMA (Thread Memory Access)
- Feature differences: Async copy, mma.sync, cp.async, TMA
- When to compile for specific architecture vs multi-arch

**Step 3: Write Knowledge File** (~750 lines)
```markdown
# CUDA Compute Capabilities & GPU Architectures

## Section 1: Compute Capability Overview (~100 lines)
- What is compute capability (sm_XX)
- Major vs minor version (sm_8.0 vs sm_8.6)
- Feature matrix across architectures

## Section 2: Turing Architecture (sm_75) (~100 lines)
- T4 specifications (16GB, 8.1 TFLOPs FP32, 65 TFLOPs FP16)
- Tensor Cores gen 2 (FP16/FP32 only)
- Memory hierarchy (6MB L2 cache)
- Use cases (inference, small training jobs)

## Section 3: Ampere Architecture (sm_80, sm_86) (~150 lines)
- A100 specifications (40/80GB, 9.7 TFLOPs TF32, 312 TFLOPs FP16)
- Tensor Cores gen 3 (TF32, BF16, FP64)
- Async copy (cp.async for overlapping memcpy)
- Multi-Instance GPU (MIG) partitioning
- sm_80 vs sm_86 differences (A100 vs A10/RTX 3090)

## Section 4: Ada Architecture (sm_89) (~100 lines)
- L4 specifications (24GB, 30.3 TFLOPs TF32, 121 TFLOPs FP16)
- Tensor Cores gen 4 (FP8 support)
- NVENC/NVDEC for video (120× AI video performance)
- Cost-effective inference

## Section 5: Hopper Architecture (sm_90) (~150 lines)
- H100 specifications (80GB, 60 TFLOPs FP32, 3958 TFLOPs FP8)
- Tensor Cores gen 4 (FP8, INT8, FP16, BF16, TF32)
- TMA (Thread Memory Access) - hardware-accelerated data movement
- Transformer Engine integration
- DPX instructions for dynamic programming

## Section 6: Compilation Strategy (~100 lines)
- Single-arch builds (smaller binary, optimal for one GPU)
- Multi-arch builds (CUDA_ARCH_LIST="7.5;8.0;8.6;9.0")
- PTX fallback (forward compatibility)
- When to target specific architecture

## Section 7: ARR-COC Connection (~50 lines)
- arr-coc-0-1 targets A100 (sm_80) for Vertex AI training
- Enabling TF32 for relevance scoring (10× speedup)
- Future: H100 with FP8 (4× faster training)
```

**Step 4: Create KNOWLEDGE DROP**
Individual summary file: `KNOWLEDGE-DROP-compute-capabilities-2025-02-03-[TIME].md`

---

## PART 3: PyTorch Custom CUDA Extensions

- [✓] PART 3: Create cuda/04-pytorch-custom-cuda-extensions.md (Completed 2025-02-03)

**Goal**: Write and integrate custom CUDA kernels with PyTorch

**Step 0: Check Existing Knowledge**
- [ ] Read karpathy/practical-implementation/73-cuda-cooperative-groups.md (mentions custom kernels)
- [ ] Identify gap: No PyTorch extension system knowledge

**Step 1: Web Research - Custom CUDA Extensions**
Search queries:
- "PyTorch custom CUDA extension cpp_extension tutorial 2024"
- "PyTorch CUDAExtension autograd Function backward"
- "PyTorch JIT compile load_inline custom kernel"

Target sources:
- PyTorch official docs (Custom C++ and CUDA Extensions)
- torch.utils.cpp_extension API reference
- Example custom kernels (LLTM, PointNet++)
- Autograd integration guide

**Step 2: Extract Key Topics**
- JIT compilation (torch.utils.cpp_extension.load)
- Ahead-of-time compilation (setuptools.Extension, CUDAExtension)
- Writing CUDA kernels (forward + backward)
- PyTorch autograd integration (torch.autograd.Function)
- Pybind11 bindings (C++ ↔ Python)
- Debugging custom extensions (gdb, cuda-gdb)

**Step 3: Write Knowledge File** (~800 lines)
```markdown
# PyTorch Custom CUDA Extensions

## Section 1: Extension System Overview (~100 lines)
- Why custom extensions (performance, new ops)
- JIT vs AOT compilation
- Extension API (cpp_extension, CUDAExtension)

## Section 2: JIT Compilation (~150 lines)
- torch.utils.cpp_extension.load()
- Inline CUDA code
- Compilation caching
- Example: Simple element-wise kernel

## Section 3: Writing CUDA Kernels (~200 lines)
- Kernel structure (forward + backward)
- Thread indexing (blockIdx, threadIdx, gridDim)
- Memory access patterns (coalescing)
- Example: Fused RGB→LAB conversion for ARR-COC

## Section 4: PyTorch Autograd Integration (~150 lines)
- torch.autograd.Function class
- forward() and backward() methods
- Tensor metadata (shape, dtype, device)
- Gradient computation
- Example: Custom relevance scorer kernel

## Section 5: AOT Compilation (~100 lines)
- setup.py with CUDAExtension
- Building wheels
- Installation and import
- Debugging compilation errors

## Section 6: Optimization & Debugging (~100 lines)
- Profiling custom kernels (nvprof, ncu)
- Memory alignment
- Shared memory usage
- Common pitfalls (race conditions, deadlocks)

## Section 7: ARR-COC Applications (~100 lines)
- Fused texture extraction (RGB+LAB+Sobel in one kernel)
- Top-K patch selection (warp-level reduction)
- Batch relevance scoring (cooperative groups)
```

**Step 4: Create KNOWLEDGE DROP**
Individual summary file: `KNOWLEDGE-DROP-custom-extensions-2025-02-03-[TIME].md`

---

## PART 4: Tensor Core Programming (WMMA & MMA)

- [✓] PART 4: Create cuda/05-tensor-core-programming-wmma-mma.md (Completed 2025-11-13 19:22)

**Goal**: Program Tensor Cores directly for maximum performance

**Step 0: Check Existing Knowledge**
- [ ] Read vertex-ai-production/01-gpu-optimization-deep.md (mentions Tensor Cores)
- [ ] Identify gap: No WMMA API or PTX mma.sync programming

**Step 1: Web Research - Tensor Core Programming**
Search queries:
- "CUDA WMMA API warp matrix multiply accumulate tutorial"
- "PTX mma.sync.aligned Tensor Cores programming"
- "Tensor Cores generation 2 3 4 differences FP8"

Target sources:
- NVIDIA WMMA Programming Guide
- PTX ISA reference (mma.sync instructions)
- Tensor Core whitepapers (Ampere, Hopper)
- Example code (cutlass, FlashAttention)

**Step 2: Extract Key Topics**
- WMMA API (fragment, load_matrix_sync, store_matrix_sync, mma_sync)
- PTX mma.sync instructions
- Tensor Core generations (gen 2: FP16, gen 3: TF32/BF16, gen 4: FP8)
- Matrix shapes (16x8x16, 16x8x8, etc.)
- Verifying Tensor Core usage (ncu, occupancy)

**Step 3: Write Knowledge File** (~700 lines)
```markdown
# Tensor Core Programming (WMMA & MMA)

## Section 1: Tensor Core Architecture (~100 lines)
- What are Tensor Cores
- Matrix multiply-accumulate (MMA) operation
- Generations and supported precisions
- Performance benefits (10-20× vs CUDA cores)

## Section 2: WMMA API (~200 lines)
- fragment<> template types
- wmma::load_matrix_sync()
- wmma::mma_sync()
- wmma::store_matrix_sync()
- Example: Simple matrix multiplication

## Section 3: PTX mma.sync Programming (~150 lines)
- PTX inline assembly
- mma.sync.aligned.m16n8k16 instruction
- Register allocation
- Example: Low-level Tensor Core kernel

## Section 4: Precision Support (~150 lines)
- FP16 (Turing+, gen 2+)
- TF32 (Ampere+, gen 3+)
- BF16 (Ampere+, gen 3+)
- FP8 (Hopper+, gen 4)
- Accuracy vs speed tradeoffs

## Section 5: Verification & Profiling (~100 lines)
- ncu (NVIDIA Compute Profiler)
- Tensor Core utilization metrics
- Occupancy analysis
- Performance targets (TFLOPs achieved)

## Section 6: ARR-COC Integration (~100 lines)
- Relevance scorer matrix multiplications
- Query-content attention (Q×K^T)
- FP16/BF16 for texture processing
- TF32 for training (10× speedup)
```

**Step 4: Create KNOWLEDGE DROP**
Individual summary file: `KNOWLEDGE-DROP-tensor-cores-2025-02-03-[TIME].md`

---

## PART 5: PyTorch JIT & torch.compile

- [✓] PART 5: Create cuda/06-pytorch-jit-torch-compile.md (Completed 2025-02-03 10:15)

**Goal**: Optimize PyTorch models with JIT compilation and torch.compile

**Step 0: Check Existing Knowledge**
- [ ] Read karpathy/practical-implementation/71-cuda-graphs-kernel-optimization.md (mentions torch.cuda.graph)
- [ ] Identify gap: No TorchScript or torch.compile knowledge

**Step 1: Web Research - PyTorch JIT**
Search queries:
- "PyTorch torch.compile TorchDynamo TorchInductor tutorial 2024"
- "TorchScript torch.jit.script trace differences"
- "torch.compile CUDA graph integration speedup"

Target sources:
- PyTorch 2.0 torch.compile documentation
- TorchDynamo internals
- TorchScript guide
- AOTAutograd reference

**Step 2: Extract Key Topics**
- TorchScript (torch.jit.script, torch.jit.trace)
- torch.compile (TorchDynamo, TorchInductor, AOTAutograd)
- CUDA Graph integration
- Compilation modes (default, reduce-overhead, max-autotune)
- Debugging JIT (TORCH_LOGS, graph dumps)

**Step 3: Write Knowledge File** (~750 lines)
```markdown
# PyTorch JIT & torch.compile

## Section 1: JIT Compilation Overview (~100 lines)
- Why JIT compilation (remove Python overhead)
- TorchScript vs torch.compile
- When to use each

## Section 2: TorchScript (~150 lines)
- torch.jit.script (type annotations)
- torch.jit.trace (example inputs)
- Limitations (dynamic control flow)
- Saving and loading (.pt files)

## Section 3: torch.compile (PyTorch 2.0+) (~200 lines)
- TorchDynamo (bytecode analysis)
- TorchInductor (code generation)
- AOTAutograd (ahead-of-time autograd)
- Compilation modes (default, reduce-overhead, max-autotune)
- Backend selection (inductor, cudagraphs)

## Section 4: CUDA Graph Integration (~150 lines)
- torch.compile with cudagraphs backend
- Static shape requirements
- Memory address stability
- Performance benchmarks (2-3× speedup)

## Section 5: Debugging & Profiling (~100 lines)
- TORCH_LOGS environment variable
- Graph visualization (torch._dynamo.explain())
- Compilation cache management
- Common errors (dynamic shapes, unsupported ops)

## Section 6: ARR-COC Optimization (~150 lines)
- Compiling relevance scorers (torch.compile)
- CUDA Graphs for inference
- Benchmark: naive vs compiled vs CUDA Graphs
- Memory usage analysis
```

**Step 4: Create KNOWLEDGE DROP**
Individual summary file: `KNOWLEDGE-DROP-jit-compile-2025-02-03-[TIME].md`

---

## PART 6: Mixed Precision Training Internals

- [✓] PART 6: Create cuda/07-mixed-precision-training-internals.md (Completed 2025-02-03)

**Goal**: Understand torch.cuda.amp under the hood and FP8 training

**Step 0: Check Existing Knowledge**
- [ ] Read cuda/01-memory-management-unified.md (mentions AMP)
- [ ] Read vertex-ai-production/01-gpu-optimization-deep.md (mentions FP16/BF16/TF32)
- [ ] Identify gap: No deep dive on amp internals, GradScaler, FP8

**Step 1: Web Research - Mixed Precision Training**
Search queries:
- "PyTorch torch.cuda.amp autocast GradScaler internals"
- "FP16 BF16 TF32 differences mixed precision training"
- "FP8 training transformer_engine apex PyTorch 2.1"

Target sources:
- PyTorch AMP documentation
- NVIDIA Apex library
- Transformer Engine (FP8 training)
- Mixed precision training papers

**Step 2: Extract Key Topics**
- torch.cuda.amp.autocast (automatic casting)
- GradScaler (loss scaling, gradient overflow detection)
- FP16 vs BF16 (dynamic range, gradient stability)
- TF32 (automatic on Ampere+)
- FP8 training (H100, transformer_engine)

**Step 3: Write Knowledge File** (~800 lines)
```markdown
# Mixed Precision Training Internals

## Section 1: Mixed Precision Overview (~100 lines)
- Why mixed precision (speed + memory)
- Precision formats (FP32, FP16, BF16, TF32, FP8)
- Hardware support (Tensor Cores)

## Section 2: torch.cuda.amp (~200 lines)
- autocast context manager
- Automatic casting rules (matmul→FP16, softmax→FP32)
- GradScaler algorithm
- Loss scaling (dynamic vs static)
- Gradient overflow detection
- Example training loop

## Section 3: Precision Formats (~150 lines)
- FP16 (5 exponent, 10 mantissa, range 6e-5 to 65504)
- BF16 (8 exponent, 7 mantissa, same range as FP32)
- TF32 (8 exponent, 10 mantissa, Ampere+ automatic)
- FP8 (E4M3 vs E5M2 formats)
- When to use each

## Section 4: FP8 Training (~200 lines)
- Transformer Engine (NVIDIA)
- te.Linear layers with FP8 forward/backward
- Delayed scaling (per-tensor vs per-channel)
- H100 optimization (3958 TFLOPs FP8)
- PyTorch 2.1+ native FP8

## Section 5: Gradient Stability (~100 lines)
- Gradient underflow/overflow
- Loss scaling strategies
- BF16 advantage (no scaling needed)
- Debugging NaN gradients

## Section 6: ARR-COC Training (~150 lines)
- Mixed precision for relevance scorer training
- BF16 for stability (opponent processing needs stable gradients)
- TF32 automatic on A100 (10× speedup)
- Future: FP8 training on H100
```

**Step 4: Create KNOWLEDGE DROP**
Individual summary file: `KNOWLEDGE-DROP-mixed-precision-2025-02-03-[TIME].md`

---

## Finalization

After all 6 runners complete:

**Oracle Tasks**:
1. Review all KNOWLEDGE DROP files
2. Update INDEX.md (add 6 new files to cuda/ section)
3. Update SKILL.md (update cuda/ description: "8 files" → comprehensive CUDA expertise)
4. Move folder to `_ingest-auto/completed/expansion-pytorch-cuda-compilation-2025-02-03/`
5. Git commit: "Knowledge Expansion 8: PyTorch CUDA Compilation Expertise (6 files, ~4,500 lines)"

**Expected File Sizes**:
- PART 1: ~700 lines (PyTorch build system)
- PART 2: ~750 lines (Compute capabilities)
- PART 3: ~800 lines (Custom extensions)
- PART 4: ~700 lines (Tensor Cores)
- PART 5: ~750 lines (JIT/torch.compile)
- PART 6: ~800 lines (Mixed precision)
**Total**: ~4,500 lines of expert-level CUDA/PyTorch knowledge

---

**End of Ingestion Plan**
