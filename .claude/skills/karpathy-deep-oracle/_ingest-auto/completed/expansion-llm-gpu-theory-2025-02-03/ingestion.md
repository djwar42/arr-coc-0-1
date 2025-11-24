# Oracle Knowledge Expansion: LLM Theory + GPU Hardware Integration (2025-02-03)

**Topic**: How CUDA/GPU optimization connects to LLM theory and practice
**Date**: 2025-02-03
**Runners**: 4 (focused on LLM-GPU integration)
**Strategy**: Bridge CUDA expertise to transformer architecture, training dynamics, inference optimization
**Context**: Understanding why LLM architectures are designed for GPU hardware constraints

---

## PART 1: FlashAttention & Attention Kernel Optimization

- [✓] PART 1: Create karpathy/llm-gpu-integration/00-flashattention-internals.md (Completed 2025-11-13 19:36)

**Goal**: Understand FlashAttention algorithm and why it's 2-4× faster than standard attention

**Step 0: Check Existing Knowledge**
- [ ] Read cuda/01-memory-management-unified.md (shared memory, HBM)
- [ ] Read cuda/05-tensor-core-programming-wmma-mma.md (Tensor Cores)
- [ ] Read cuda/06-pytorch-jit-torch-compile.md (kernel fusion)
- [ ] Read vllm-knowledge/00-vllm-architecture-pagedattention.md (PagedAttention overview)
- [ ] Identify gap: No FlashAttention algorithm details, block-wise computation, tiling strategy

**Step 1: Web Research - FlashAttention**
Search queries:
- "FlashAttention algorithm block-wise computation shared memory 2024"
- "FlashAttention-2 sequence parallelism improvements"
- "FlashAttention-3 Hopper TMA asynchronous execution"

Target sources:
- FlashAttention paper (Dao et al., 2022, NeurIPS)
- FlashAttention-2 paper (Dao et al., 2023)
- FlashAttention-3 announcement (Hopper optimization)
- PyTorch FlashAttention integration
- HazyResearch GitHub (flash-attention repository)

**Step 2: Extract Key Topics**
- Standard attention memory bottleneck (O(N²) HBM accesses)
- Block-wise computation (tiling Q, K, V in SRAM)
- Online softmax algorithm (numerically stable, one-pass)
- IO complexity analysis (O(N²/M) HBM accesses, M = SRAM size)
- FlashAttention-2 improvements (sequence parallelism, work partitioning)
- FlashAttention-3 Hopper features (TMA, warpgroup specialization, FP8)
- Integration with PyTorch (torch.nn.functional.scaled_dot_product_attention)

**Step 3: Write Knowledge File** (~800 lines)
```markdown
# FlashAttention & Attention Kernel Optimization

## Section 1: Standard Attention Memory Bottleneck (~100 lines)
- Attention computation (Q×K^T → softmax → ×V)
- Memory hierarchy (SRAM vs HBM, 19.5TB/s vs 1.5TB/s on A100)
- O(N²) intermediate tensors (S = Q×K^T, P = softmax(S))
- Why it's slow: HBM bandwidth limited (reading/writing O(N²) tensors)

## Section 2: FlashAttention Algorithm (~200 lines)
- Block-wise computation (tile Q, K, V into SRAM blocks)
- Online softmax algorithm (incremental max, exp, sum)
- Forward pass (outer loop over KV blocks, inner loop over Q blocks)
- Backward pass (recomputation vs storing attention matrix)
- IO complexity: O(N²/M) HBM accesses (M = SRAM size)
- Performance: 2-4× faster, enables longer sequences (64K tokens)

## Section 3: FlashAttention-2 Improvements (~150 lines)
- Sequence parallelism (split sequence across thread blocks)
- Work partitioning (non-uniform splits for better load balancing)
- Reduced non-matmul FLOPs (optimized softmax rescaling)
- Benchmark results (2× faster than FA1 on A100, 225 TFLOPs)

## Section 4: FlashAttention-3 Hopper (~150 lines)
- TMA (Thread Memory Access) for asynchronous data movement
- Warpgroup specialization (producer-consumer pattern)
- FP8 low-precision format (Hopper H100, 3958 TFLOPs)
- Overlapping compute and memory (pipelining)
- Performance: 740 TFLOPs (75% of H100 peak)

## Section 5: PyTorch Integration (~100 lines)
- torch.nn.functional.scaled_dot_product_attention (SDPA)
- Automatic backend selection (FlashAttention vs math vs memory-efficient)
- Compilation with torch.compile (fuses SDPA into graph)
- Custom attention masks (causal, sliding window, block-sparse)

## Section 6: ARR-COC Connection (~100 lines)
- Query-aware attention for relevance scoring
- FlashAttention for 3 ways of knowing (propositional, perspectival, participatory)
- Variable sequence lengths (64-400 tokens per patch)
- Multi-query attention for efficient KV cache
```

**Step 4: Create KNOWLEDGE DROP**
Individual summary file: `KNOWLEDGE-DROP-flashattention-2025-02-03-[TIME].md`

---

## PART 2: Transformer Architecture & GPU Hardware Constraints

- [✓] PART 2: Create karpathy/llm-gpu-integration/01-architecture-gpu-constraints.md (Completed 2025-02-03 15:45)

**Goal**: Understand why LLM architectures are designed for GPU hardware (hidden dims, heads, vocab sizes)

**Step 0: Check Existing Knowledge**
- [ ] Read cuda/03-compute-capabilities-gpu-architectures.md (Tensor Core shapes)
- [ ] Read cuda/05-tensor-core-programming-wmma-mma.md (16×8×16 matmul)
- [ ] Read karpathy/gpt-architecture/ (transformer basics)
- [ ] Identify gap: No explanation of why hidden_dim = 4096, num_heads = 32, vocab = 32000

**Step 1: Web Research - Architecture Design**
Search queries:
- "transformer hidden dimension GPU optimization Tensor Cores"
- "why attention heads power of 2 warp size"
- "LLM vocab size design 32000 50000 matrix multiplication"

Target sources:
- Megatron-LM paper (tensor/pipeline parallelism, architecture choices)
- GPT-3 paper (architecture specifications)
- LLaMA paper (architecture decisions, tokenizer vocab)
- Chinchilla scaling laws (model size vs training tokens)
- GPU optimization blogs (NVIDIA, community)

**Step 2: Extract Key Topics**
- Hidden dimension choices (4096, 8192, 12288) → Tensor Core alignment
- Number of heads (32, 64, 128) → Warp size (32 threads), parallel reductions
- Head dimension (hidden_dim / num_heads = 128, 256) → Memory bank alignment
- Vocabulary size (32000, 50000, 128000) → Embedding matrix shapes
- MLP expansion ratio (4×) → Memory-compute balance
- Sequence length (2048, 4096, 8192) → Memory constraints, attention O(N²)
- Positional encoding (RoPE, ALiBi) → Memory access patterns

**Step 3: Write Knowledge File** (~750 lines)
```markdown
# Transformer Architecture & GPU Hardware Constraints

## Section 1: Hidden Dimension Design (~150 lines)
- Why 4096, 8192, 12288? (Multiples of 256 for Tensor Cores)
- GPT-3: 12,288 hidden_dim (48 × 256, optimal for A100 sm_80)
- LLaMA: 4096 (16 × 256, smaller but efficient)
- Memory bank alignment (128-byte cache lines)
- Matrix multiplication shapes (M×K × K×N, K = hidden_dim)

## Section 2: Attention Head Configuration (~150 lines)
- Why num_heads = 32, 64, 128? (Powers of 2)
- Warp size = 32 threads (NVIDIA GPUs)
- Head dimension = hidden_dim / num_heads (typically 64, 128, 256)
- Parallel reductions (softmax across heads)
- Multi-query attention (1 KV head, N query heads for efficiency)
- Grouped-query attention (MQA middle ground)

## Section 3: Vocabulary Size Trade-offs (~150 lines)
- GPT-2: 50,257 (BPE tokenizer)
- LLaMA: 32,000 (SentencePiece)
- GPT-4: ~100,000 (estimated, multimodal tokens)
- Embedding matrix: vocab_size × hidden_dim (memory cost)
- Output projection: hidden_dim × vocab_size (compute cost)
- Vocab size padding (round to multiples of 64 for Tensor Cores)

## Section 4: MLP Expansion Ratio (~100 lines)
- Standard: hidden_dim → 4×hidden_dim → hidden_dim
- Why 4×? Memory-compute balance (2/3 of model parameters)
- GLU variants (SwiGLU, GeGLU): 8/3× expansion instead of 4×
- Memory access patterns (column-major vs row-major)

## Section 5: Sequence Length Scaling (~150 lines)
- Attention memory: O(N²) for sequence length N
- Why 2048, 4096, 8192? (Powers of 2, memory alignment)
- Context length extensions (ALiBi, RoPE interpolation, YaRN)
- Sparse attention patterns (sliding window, block-sparse)
- Ring attention for extreme lengths (>1M tokens)

## Section 6: ARR-COC Token Budget (~100 lines)
- Why 64-400 tokens per patch? (64 = 4×16, 400 = 25×16)
- 8× average compression (200 tokens) = 12.5 × 16
- GPU-aware design for Tensor Core efficiency
- Variable LOD (64-400 range) maps to attention head dimensions
```

**Step 4: Create KNOWLEDGE DROP**
Individual summary file: `KNOWLEDGE-DROP-architecture-gpu-2025-02-03-[TIME].md`

---

## PART 3: LLM Training Dynamics & GPU Optimization

- [✓] PART 3: Create karpathy/llm-gpu-integration/02-training-dynamics-gpu.md (Completed 2025-02-03 16:45)

**Goal**: Understand gradient checkpointing, pipeline parallelism, ZeRO, mixed precision for LLM training

**Step 0: Check Existing Knowledge**
- [ ] Read cuda/07-mixed-precision-training-internals.md (AMP, GradScaler, FP8)
- [ ] Read cuda/01-memory-management-unified.md (memory optimization)
- [ ] Read karpathy/training-llms/ (pre-training, SFT, RLHF)
- [ ] Read vertex-ai-production/00-distributed-training-patterns.md (DDP, Horovod)
- [ ] Identify gap: No gradient checkpointing, pipeline parallelism, ZeRO, tensor parallelism

**Step 1: Web Research - Distributed Training**
Search queries:
- "gradient checkpointing activation recomputation LLM training"
- "ZeRO optimizer DeepSpeed stage 1 2 3"
- "Megatron-LM tensor parallelism pipeline parallelism"

Target sources:
- Megatron-LM paper (NVIDIA, tensor/pipeline parallelism)
- ZeRO paper (Microsoft DeepSpeed, optimizer state partitioning)
- PyTorch FSDP (Fully Sharded Data Parallel)
- Gradient checkpointing trade-offs
- Mixed precision training for 100B+ models

**Step 2: Extract Key Topics**
- Gradient checkpointing (trade memory for recomputation)
- Activation recomputation strategies (selective checkpointing)
- Pipeline parallelism (GPipe, PipeDream, micro-batches, bubble overhead)
- Tensor parallelism (Megatron column/row splitting, all-reduce communication)
- ZeRO stages (1: optimizer states, 2: gradients, 3: parameters)
- FSDP (PyTorch alternative to ZeRO)
- 3D parallelism (data + tensor + pipeline)
- Learning rate scaling laws (batch size scaling)

**Step 3: Write Knowledge File** (~850 lines)
```markdown
# LLM Training Dynamics & GPU Optimization

## Section 1: Gradient Checkpointing (~150 lines)
- Memory bottleneck (activations > parameters for large batch sizes)
- Activation recomputation (trade 33% compute for 75% memory savings)
- Selective checkpointing (which layers to checkpoint)
- PyTorch torch.utils.checkpoint.checkpoint()
- HuggingFace gradient_checkpointing_enable()

## Section 2: Pipeline Parallelism (~200 lines)
- GPipe (synchronous, micro-batches, bubble overhead)
- PipeDream (asynchronous, weight stashing)
- Megatron-LM pipeline (1F1B schedule)
- Bubble time (pipeline depth × micro-batch time)
- Communication patterns (P2P vs all-reduce)
- Optimal pipeline splits (balance compute across stages)

## Section 3: Tensor Parallelism (~200 lines)
- Megatron-LM column/row parallelism
- Attention: split heads across GPUs (embarrassingly parallel)
- MLP: column-parallel (scatter input) + row-parallel (gather output)
- All-reduce communication (NCCL, NVLink bandwidth)
- Memory vs communication trade-off

## Section 4: ZeRO Optimizer (~150 lines)
- Stage 1: Partition optimizer states (4× memory reduction)
- Stage 2: Partition gradients (8× memory reduction)
- Stage 3: Partition parameters (N× memory reduction, N = #GPUs)
- Communication overhead (all-gather parameters, reduce-scatter gradients)
- DeepSpeed ZeRO-Offload (CPU offloading)

## Section 5: 3D Parallelism (~100 lines)
- Combining data + tensor + pipeline parallelism
- Parallelism configuration (DP=8, TP=4, PP=2)
- Communication topology (NVLink within node, InfiniBand across nodes)
- Optimal parallelism strategy (model size, GPU count, interconnect)

## Section 6: ARR-COC Training (~100 lines)
- Multi-stage training (texture arrays → relevance scorers → quality adapter)
- Gradient checkpointing for 13-channel texture processing
- Mixed precision (BF16 for opponent processing stability)
- Future: Pipeline parallelism for 3 scorer stages
```

**Step 4: Create KNOWLEDGE DROP**
Individual summary file: `KNOWLEDGE-DROP-training-dynamics-2025-02-03-[TIME].md`

---

## PART 4: LLM Inference Optimization & KV Cache Management

- [✓] PART 4: Create karpathy/llm-gpu-integration/03-inference-kv-cache-optimization.md (Completed 2025-02-03 18:45)

**Goal**: Understand KV cache, continuous batching, PagedAttention, speculative decoding, quantized cache

**Step 0: Check Existing Knowledge**
- [ ] Read vllm-knowledge/00-vllm-architecture-pagedattention.md (PagedAttention overview)
- [ ] Read vllm-knowledge/02-vllm-prefix-caching.md (RadixAttention)
- [ ] Read vllm-knowledge/03-vllm-speculative-decoding.md (draft models)
- [ ] Read cuda/00-streams-concurrency-async.md (CUDA streams for multi-model)
- [ ] Identify gap: Deep CUDA implementation details, KV cache quantization, continuous batching internals

**Step 1: Web Research - Inference Optimization**
Search queries:
- "KV cache memory management continuous batching vLLM"
- "PagedAttention block table copy-on-write implementation"
- "speculative decoding CUDA streams draft target models"

Target sources:
- vLLM paper (Efficient Memory Management for LLM Serving)
- Orca paper (continuous batching)
- SpecInfer paper (speculative decoding)
- SGLang RadixAttention
- Quantized KV cache papers (INT8, FP8)

**Step 2: Extract Key Topics**
- KV cache fundamentals (cache K, V from previous tokens)
- Memory growth (N tokens × num_layers × 2 × hidden_dim × batch_size)
- Continuous batching (iteration-level scheduling, preemption)
- PagedAttention (block tables, virtual/physical memory, copy-on-write)
- Prefix caching (RadixAttention, trie structure, 98% cache hit)
- Speculative decoding (draft model generates k tokens, target verifies)
- Quantized KV cache (INT8, FP8, 2-4× memory savings)
- Multi-query attention for efficient cache (1 KV head, N query heads)

**Step 3: Write Knowledge File** (~800 lines)
```markdown
# LLM Inference Optimization & KV Cache Management

## Section 1: KV Cache Fundamentals (~100 lines)
- Why cache K, V? (Avoid recomputing from all previous tokens)
- Memory cost: seq_len × num_layers × 2 × hidden_dim × batch × bytes
- Example: LLaMA-7B, 2048 tokens, batch=8, FP16 → 4GB KV cache
- Autoregressive generation bottleneck (memory-bound, not compute-bound)

## Section 2: Continuous Batching (~150 lines)
- Iteration-level scheduling (vs request-level in naive serving)
- Dynamic batching (add new requests, remove finished requests each iteration)
- Preemption (pause low-priority requests when memory full)
- Swapping (move KV cache to CPU, swap back when needed)
- Orca paper (continuous batching, 2-3× throughput improvement)

## Section 3: PagedAttention Deep Dive (~200 lines)
- Virtual memory for KV cache (block tables, physical blocks)
- Block size (typically 16 tokens, trade-off: fragmentation vs overhead)
- Copy-on-write (shared prefixes, fork on first write)
- Memory allocation (buddy allocator, free block list)
- Attention computation with block tables (gather K, V from non-contiguous blocks)
- vLLM implementation (CUDA kernel for paged attention)

## Section 4: Prefix Caching (RadixAttention) (~150 lines)
- System prompts (same prefix for many requests)
- Trie structure (radix tree for shared prefixes)
- Cache hit rate (98% for common system prompts)
- Eviction policy (LRU, reference counting)
- SGLang implementation (automatic prefix detection)

## Section 5: Speculative Decoding (~150 lines)
- Draft model (small, fast, generates k tokens speculatively)
- Target model (large, accurate, verifies k tokens in parallel)
- Acceptance rate (typically 60-80%, language-dependent)
- CUDA streams (draft on stream 1, target on stream 2)
- Speedup: 1.5-2.8× (depends on draft model quality)
- SpecInfer, Medusa, EAGLE variants

## Section 6: Quantized KV Cache (~100 lines)
- INT8 quantization (per-tensor scaling, 2× memory reduction)
- FP8 quantization (E4M3 format, better accuracy)
- Accuracy impact (0.1-0.5 perplexity increase)
- CUDA kernels (dequantize on-the-fly during attention)
- H100 FP8 native support (zero overhead)

## Section 7: ARR-COC Inference (~100 lines)
- KV cache for 3 relevance scorers (propositional, perspectival, participatory)
- Prefix caching for texture features (shared across patches)
- Speculative decoding for token budget (draft allocation → verify with full model)
- Multi-query attention for cache efficiency
```

**Step 4: Create KNOWLEDGE DROP**
Individual summary file: `KNOWLEDGE-DROP-inference-kv-cache-2025-02-03-[TIME].md`

---

## Finalization

After all 4 runners complete:

**Oracle Tasks**:
1. Review all KNOWLEDGE DROP files
2. Create new folder: `karpathy/llm-gpu-integration/` (4 files)
3. Update INDEX.md (add new section: "LLM + GPU Integration")
4. Update SKILL.md (add llm-gpu-integration/ to karpathy/ tree)
5. Move folder to `_ingest-auto/completed/expansion-llm-gpu-theory-2025-02-03/`
6. Git commit: "Knowledge Expansion 9: LLM + GPU Integration (4 files, ~3,200 lines)"

**Expected File Sizes**:
- PART 1: ~800 lines (FlashAttention)
- PART 2: ~750 lines (Architecture constraints)
- PART 3: ~850 lines (Training dynamics)
- PART 4: ~800 lines (Inference optimization)
**Total**: ~3,200 lines of LLM-GPU integration knowledge

---

**End of Ingestion Plan**
