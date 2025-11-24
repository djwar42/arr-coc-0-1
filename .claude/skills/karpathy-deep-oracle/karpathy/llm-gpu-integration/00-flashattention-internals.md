# FlashAttention & Attention Kernel Optimization

## Overview

FlashAttention is an IO-aware attention algorithm that revolutionized transformer efficiency by minimizing memory reads/writes between GPU HBM (High Bandwidth Memory) and SRAM (on-chip cache). By using tiling and recomputation, FlashAttention achieves 2-4× speedup over standard attention implementations while using linear memory instead of quadratic, enabling much longer sequence lengths for LLMs.

**Key Innovation**: Block-wise computation that keeps intermediate results in fast SRAM rather than slow HBM, dramatically reducing memory bandwidth bottleneck.

**Performance Evolution**:
- **FlashAttention-1** (2022): 2-4× speedup, 25-40% of theoretical max FLOPs on A100
- **FlashAttention-2** (2023): 2× faster than FA-1, 50-73% of max FLOPs, 225 TFLOPs/s on A100
- **FlashAttention-3** (2024): 1.5-2× faster than FA-2, 75% utilization on H100, 740 TFLOPs (FP16), 1.2 PFLOPs (FP8)

From [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (Dao et al., NeurIPS 2022, accessed 2025-02-03):
> "We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and SRAM. We analyze the IO complexity of FlashAttention, showing that it requires fewer HBM accesses than standard attention, and is optimal for a range of SRAM sizes."

From [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://tridao.me/blog/2024/flash3/) (Tri Dao, 2024, accessed 2025-02-03):
> "FlashAttention-3 incorporates three main techniques to speed up attention on Hopper GPUs: exploiting asynchrony of the Tensor Cores and TMA to overlap overall computation and data movement via warp-specialization and interleave block-wise matmul and softmax operations, and incoherent processing that leverages hardware support for FP8 low-precision."

**Related Knowledge**:
- See [../cuda/01-memory-management-unified.md](../cuda/01-memory-management-unified.md) for GPU memory hierarchy (HBM vs SRAM)
- See [../cuda/05-tensor-core-programming-wmma-mma.md](../cuda/05-tensor-core-programming-wmma-mma.md) for Tensor Core programming
- See [../cuda/06-pytorch-jit-torch-compile.md](../cuda/06-pytorch-jit-torch-compile.md) for kernel fusion techniques
- See [../vllm-knowledge/00-vllm-architecture-pagedattention.md](../vllm-knowledge/00-vllm-architecture-pagedattention.md) for PagedAttention (builds on FlashAttention concepts)

---

## Section 1: Standard Attention Memory Bottleneck (~120 lines)

### The Attention Computation

Standard self-attention computes:

```
S = Q × K^T              # (N × N) matrix, N = sequence length
P = softmax(S)           # (N × N) attention probabilities
O = P × V                # (N × d) output
```

Where:
- Q, K, V: Query, Key, Value matrices (N × d, d = head dimension)
- S: Attention scores (N × N)
- P: Attention probabilities after softmax (N × N)
- O: Output (N × d)

**Memory Cost**: O(N²) for storing S and P matrices

### GPU Memory Hierarchy

From [../cuda/01-memory-management-unified.md](../cuda/01-memory-management-unified.md):

| Memory Type | Size | Bandwidth | Latency | Scope |
|------------|------|-----------|---------|-------|
| **HBM (Global Memory)** | 40-80GB | 1.6-3.35 TB/s | ~300 cycles | Device |
| **SRAM (Shared Memory)** | 48-164KB per SM | ~10 TB/s | ~4 cycles | Thread Block |
| **L2 Cache** | 40-60MB | ~5 TB/s | ~200 cycles | Device |
| **Registers** | 256KB per SM | ~20 TB/s | 1 cycle | Thread |

**A100 GPU Memory Bandwidth**:
- HBM2: 1.6 TB/s (40GB model) or 2.0 TB/s (80GB model)
- SRAM: ~19.5 TB/s (10× faster than HBM)
- Ratio: **SRAM is 10-12× faster than HBM**

**H100 GPU Memory Bandwidth**:
- HBM3: 3.35 TB/s
- SRAM: Still ~20 TB/s
- Ratio: **SRAM is 6× faster than HBM** (HBM improved more than SRAM)

### Standard Attention Implementation

**Naive PyTorch Implementation**:
```python
def standard_attention(Q, K, V):
    # Q, K, V: (batch, heads, seq_len, head_dim)
    # This is SLOW - writes large matrices to HBM

    # Step 1: Compute attention scores (writes N×N to HBM)
    S = Q @ K.transpose(-2, -1)  # (N × N) matrix - LARGE!
    S = S / math.sqrt(Q.size(-1))

    # Step 2: Softmax (reads N×N from HBM, writes N×N back)
    P = F.softmax(S, dim=-1)  # (N × N) matrix - LARGE!

    # Step 3: Weighted sum (reads N×N from HBM)
    O = P @ V  # Output (N × d)

    return O
```

**Memory Reads/Writes**:
1. **Write S**: N² elements to HBM
2. **Read S, Write P**: 2N² HBM accesses for softmax
3. **Read P**: N² elements for final matmul
4. **Total**: **4N² HBM accesses** (quadratic in sequence length)

**Why This Is Slow**:

From [cuda/01-memory-management-unified.md](../cuda/01-memory-management-unified.md):
- A100 HBM bandwidth: 1.6 TB/s
- For N=8192, d=128, FP16:
  - S matrix: 8192² × 2 bytes = 134 MB
  - P matrix: 8192² × 2 bytes = 134 MB
  - Total intermediate storage: 268 MB
  - Time to write/read at 1.6 TB/s: **0.67 ms just for memory**
  - Actual computation (matmuls): **~0.5 ms on Tensor Cores**
  - **Memory bound, not compute bound!**

### The IO Complexity Problem

**FLOPS vs Memory**:
- Matrix multiplication: O(N² × d) FLOPs
- Softmax: O(N²) FLOPs (much cheaper per element)
- **Total**: O(N²d) FLOPs, but O(N²) intermediate memory

**For head_dim=128**:
- Matmul FLOPs: 128× more than softmax
- But both require O(N²) memory reads/writes
- **Softmax becomes the bottleneck** due to memory bandwidth

**Bandwidth-Limited Performance**:

On A100 GPU (1.6 TB/s HBM):
- FP16 matmul peak: 312 TFLOPs
- Standard attention actual: **~80 TFLOPs** (25% utilization)
- Bottleneck: **4N² HBM accesses dominate runtime**

**Scaling to Long Sequences**:

| Sequence Length | S + P Memory | Time @ 1.6 TB/s | Problem |
|-----------------|--------------|-----------------|---------|
| 2K | 16 MB | 0.01 ms | Manageable |
| 8K | 256 MB | 0.16 ms | Noticeable |
| 32K | 4 GB | 2.5 ms | Severe |
| 128K | 64 GB | 40 ms | **OOM!** |

From [FlashAttention paper](https://arxiv.org/abs/2205.14135):
> "The attention layer is the main bottleneck in scaling to longer sequences, as its runtime and memory increase quadratically in the sequence length."

---

## Section 2: FlashAttention Algorithm (~200 lines)

### Core Idea: Tiling and Recomputation

**Key Insight**: Don't materialize the full N×N attention matrix. Compute attention block-by-block, keeping blocks in fast SRAM.

**Trade-off**:
- **Standard attention**: Write O(N²) to HBM (slow), compute once
- **FlashAttention**: Keep blocks in SRAM (fast), recompute during backward pass
- **Result**: Faster overall despite recomputation (memory bandwidth >> compute)

From [FlashAttention paper](https://arxiv.org/abs/2205.14135):
> "We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and SRAM."

### Block-Wise Computation

**Tiling Strategy**:

```
Divide Q, K, V into blocks that fit in SRAM:
- Q: Tile into blocks of size (Br × d)
- K, V: Tile into blocks of size (Bc × d)
- Where Br, Bc chosen so blocks fit in SRAM (~100-200 KB)

For each block of Q:
    For each block of K, V:
        1. Load Q_block, K_block, V_block to SRAM
        2. Compute S_block = Q_block × K_block^T (in SRAM)
        3. Compute P_block = softmax(S_block) (in SRAM)
        4. Compute partial output: O_partial += P_block × V_block
        5. Update running statistics for correct softmax
    Write final O_block to HBM
```

**Key Challenge**: Softmax requires global statistics (max, sum) across full sequence, but we're computing block-wise. Solution: **Online Softmax Algorithm**.

### Online Softmax Algorithm

Standard softmax requires two passes:
1. Find max across all elements
2. Compute exp and normalize

**Online Softmax** computes softmax incrementally with one pass:

```python
# Numerically stable online softmax
# Maintains running (max, sum) as we process blocks

def online_softmax_update(current_max, current_sum, current_out,
                         new_block_scores):
    # new_block_scores: current block's Q×K^T values

    # Find max of new block
    new_max = torch.max(new_block_scores)

    # Global max across all blocks seen so far
    global_max = torch.maximum(current_max, new_max)

    # Rescale previous sum (because max changed)
    rescale_factor = torch.exp(current_max - global_max)
    rescaled_sum = current_sum * rescale_factor

    # Add new block's contribution
    new_exp = torch.exp(new_block_scores - global_max)
    new_sum = rescaled_sum + torch.sum(new_exp, dim=-1)

    # Rescale previous output (numerator changed due to new max)
    rescaled_out = current_out * rescale_factor

    return global_max, new_sum, rescaled_out
```

**Why This Works**:

Softmax formula: `softmax(x_i) = exp(x_i) / sum(exp(x_j))`

Key insight: `exp(x - a) / sum(exp(x - a)) = exp(x - b) / sum(exp(x - b))` for any constants a, b

We can incrementally update the "correct" max and renormalize previous results.

From [FlashAttention-3 blog](https://tridao.me/blog/2024/flash3/):
> "FlashAttention uses tiling and softmax rescaling, we operate by blocks and avoid having to read/write from HBM, while obtaining the correct output with no approximation."

### FlashAttention Forward Pass Algorithm

**Pseudocode**:

```python
def flash_attention_forward(Q, K, V):
    # Q, K, V: (N × d) - already in HBM
    # Block sizes: Br (Q rows per block), Bc (K/V rows per block)

    # Choose block sizes to fit in SRAM
    # Typical: Br=Bc=128 for d=64, or Br=Bc=64 for d=128

    # Initialize output and statistics
    O = zeros(N, d)  # Final output (in HBM)
    l = zeros(N)     # Running sum for softmax (in HBM)
    m = -inf(N)      # Running max for softmax (in HBM)

    # Tile Q into Tr blocks of size Br
    Tr = ceil(N / Br)
    # Tile K, V into Tc blocks of size Bc
    Tc = ceil(N / Bc)

    # Outer loop: iterate over Q blocks
    for j in range(Tr):
        # Load Q block to SRAM (Br × d)
        Q_j = load_block(Q, j, Br)

        # Initialize block output and statistics (in SRAM)
        O_j = zeros(Br, d)
        l_j = zeros(Br)
        m_j = -inf(Br)

        # Inner loop: iterate over K, V blocks
        for i in range(Tc):
            # Load K, V blocks to SRAM (Bc × d)
            K_i = load_block(K, i, Bc)
            V_i = load_block(V, i, Bc)

            # Compute attention scores for this block (in SRAM)
            # S_ij: (Br × Bc)
            S_ij = Q_j @ K_i.T / sqrt(d)

            # Online softmax update (in SRAM)
            # Find block max
            m_block = rowmax(S_ij)  # (Br,)

            # Update global max
            m_new = maximum(m_j, m_block)

            # Compute rescaling for previous blocks
            alpha = exp(m_j - m_new)
            beta = exp(m_block - m_new)

            # Compute new exp and sum
            P_ij = exp(S_ij - m_new)  # (Br × Bc)
            l_new = alpha * l_j + rowsum(P_ij)

            # Update output (weighted average)
            O_j = (alpha * l_j / l_new) * O_j + (beta / l_new) * (P_ij @ V_i)

            # Update statistics
            l_j = l_new
            m_j = m_new

        # Write block output to HBM
        store_block(O, j, O_j)
        store_block(l, j, l_j)
        store_block(m, j, m_j)

    return O
```

**Key Operations (All in SRAM)**:
1. `S_ij = Q_j @ K_i.T`: Block-wise attention scores
2. `P_ij = exp(S_ij - m_new)`: Numerically stable exponential
3. `O_j += (P_ij @ V_i)`: Incremental output accumulation
4. Online statistics update: `m_new`, `l_new`

**No Intermediate HBM Writes**: S_ij and P_ij never written to HBM!

### FlashAttention Backward Pass

**Challenge**: Standard backward pass needs attention matrix P (N×N, stored in HBM). FlashAttention didn't save it!

**Solution**: Recompute P block-by-block during backward pass.

**Trade-off Analysis**:
- **Memory saved**: O(N²) by not storing P
- **Compute added**: Recompute S and P during backward (~30% more FLOPs)
- **Result**: Still faster (memory bandwidth bottleneck >> compute)

From [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691):
> "FlashAttention exploits the asymmetric GPU memory hierarchy to bring significant memory saving (linear instead of quadratic) and runtime speedup (2-4× compared to optimized baselines), with no approximation."

**Backward Pass Pseudocode**:

```python
def flash_attention_backward(Q, K, V, O, dO):
    # dO: gradient of output (N × d)
    # Recompute P block-by-block, never materialize full matrix

    dQ = zeros(N, d)
    dK = zeros(N, d)
    dV = zeros(N, d)

    for j in range(Tr):  # Q blocks
        Q_j = load_block(Q, j, Br)
        O_j = load_block(O, j, Br)
        dO_j = load_block(dO, j, Br)

        # Compute D = rowsum(dO * O) for rescaling
        D_j = rowsum(dO_j * O_j)

        for i in range(Tc):  # K, V blocks
            K_i = load_block(K, i, Bc)
            V_i = load_block(V, i, Bc)

            # Recompute S and P (same as forward)
            S_ij = Q_j @ K_i.T / sqrt(d)
            P_ij = softmax(S_ij)  # Recomputed!

            # Compute gradients using chain rule
            dV_i = P_ij.T @ dO_j
            dP_ij = dO_j @ V_i.T

            # Softmax backward (uses D_j for efficiency)
            dS_ij = P_ij * (dP_ij - D_j)

            dQ_j += dS_ij @ K_i / sqrt(d)
            dK_i = dS_ij.T @ Q_j / sqrt(d)

            # Accumulate gradients
            store_block(dV, i, dV_i)  # Atomic add
            store_block(dK, i, dK_i)  # Atomic add

        store_block(dQ, j, dQ_j)

    return dQ, dK, dV
```

**Key Insight**: Recomputing P block-by-block in SRAM is faster than reading full P from HBM.

### IO Complexity Analysis

From [FlashAttention paper](https://arxiv.org/abs/2205.14135):

**Standard Attention**:
- HBM accesses: **O(N²)** (reading/writing S and P matrices)

**FlashAttention**:
- HBM accesses: **O(N² / M)** where M = SRAM size
- Proof sketch:
  - Each of Tr × Tc block pairs requires loading Q, K, V blocks
  - Block size: Br × Bc ≈ M (SRAM size)
  - Number of blocks: Tr × Tc ≈ N²/ M
  - Total HBM accesses: O(N²d / M) = O(N² / M) for d < M

**Speedup Factor**:
- SRAM size M ≈ 100KB - 164KB per SM
- For N=4096, d=64: M/d ≈ 50-100
- **Theoretical speedup: 50-100×** in HBM accesses

**Actual Speedup**:
- A100 (FlashAttention-1): 2-4× wallclock time
- Why not 50×? Compute still takes time, and SRAM access isn't free
- Still **2-4× faster** than highly optimized baselines

### Block Size Selection

**Constraint**: Blocks must fit in SRAM per thread block.

From [cuda/01-memory-management-unified.md](../cuda/01-memory-management-unified.md):
- A100: 164KB shared memory per SM (configurable)
- H100: 228KB shared memory per SM

**Typical Block Sizes**:

| Head Dim | Br (Q block) | Bc (K/V block) | SRAM Usage |
|----------|--------------|----------------|------------|
| 64 | 128 | 128 | ~100KB |
| 128 | 64 | 64 | ~100KB |
| 256 | 32 | 32 | ~100KB |

**Formula**: `SRAM_usage ≈ (Br × d + Bc × d + Br × Bc) × sizeof(float16)`

**Trade-off**:
- Larger blocks: Fewer recomputations, more parallelism
- Smaller blocks: Better fit in SRAM, more thread blocks can run concurrently

### Performance Results

From [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691):

**A100 GPU Performance (FP16)**:
- Standard PyTorch attention: ~80 TFLOPs (25% utilization)
- FlashAttention-1: ~125 TFLOPs (40% utilization) - **2-3× faster**
- FlashAttention-2: ~225 TFLOPs (72% utilization) - **2× faster than FA-1**

**Sequence Length Scaling**:
- 2K tokens: 1.5× speedup
- 8K tokens: 2-3× speedup
- 32K tokens: 3-4× speedup
- **Longer sequences = bigger speedup** (more quadratic overhead in standard attention)

**Memory Usage**:
- Standard: O(N²) - 16GB for 32K sequence
- FlashAttention: O(N) - 2GB for 32K sequence
- **8× memory reduction** enables longer contexts

**Context Length Enabled**:
- Before FlashAttention: 2-4K tokens (GPT-3, OPT)
- With FlashAttention: 32-128K tokens (GPT-4, Claude)
- Latest: 1M+ tokens (Llama-3-Gradient-1048K)

---

## Section 3: FlashAttention-2 Improvements (~150 lines)

From [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (Dao, 2023):

FlashAttention-2 addresses three inefficiencies in FlashAttention-1:
1. Non-matmul FLOPs were not minimized
2. Work not parallelized across sequence length (within single head)
3. Suboptimal work distribution between warps

### Improvement 1: Reduce Non-Matmul FLOPs

**Problem in FA-1**: Softmax rescaling performed after every block, even when unnecessary.

**Solution**: Defer rescaling until after all K/V blocks processed for a Q block.

**Before (FA-1)**:
```python
for each K/V block:
    S_ij = Q_j @ K_i.T
    m_new = max(m_old, rowmax(S_ij))

    # Rescale immediately (expensive!)
    O_j = O_j * exp(m_old - m_new)  # Vector-matrix multiply
    P_ij = exp(S_ij - m_new)
    O_j = O_j + P_ij @ V_i
```

**After (FA-2)**:
```python
for each K/V block:
    S_ij = Q_j @ K_i.T
    m_new = max(m_old, rowmax(S_ij))
    P_ij = exp(S_ij - m_new)
    O_j = O_j + P_ij @ V_i  # Accumulate without rescaling

# Rescale once at the end
O_j = O_j / l_j  # Final normalization
```

**Savings**: Reduces rescaling from `O(N²)` to `O(N)` operations.

### Improvement 2: Sequence Parallelism

**Problem in FA-1**: Each attention head computed by single thread block. Low GPU occupancy for small batch or few heads.

**Solution**: Split sequence dimension across multiple thread blocks, even for single head.

**Parallelization Strategy**:

```python
# FA-1: One thread block per attention head
for head in range(num_heads):
    thread_block = assign_thread_block(head)
    compute_attention(Q[head], K[head], V[head])  # Full sequence

# FA-2: Multiple thread blocks per head
for head in range(num_heads):
    for seq_split in range(num_splits):  # NEW!
        thread_block = assign_thread_block(head, seq_split)
        # Each block processes subset of Q blocks
        compute_attention(Q[head, seq_split], K[head], V[head])
```

**Benefits**:
- **Higher occupancy**: More thread blocks active on GPU
- **Better parallelism**: Small batch sizes still saturate GPU
- **Load balancing**: Work distributed more evenly

From [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691):
> "The increased parallelism over sequence length helps improve occupancy (fraction of GPU resources being used) when the batch size and number of heads are small."

**Example**:
- Batch=1, Heads=8, SeqLen=8192
- FA-1: Only 8 thread blocks (one per head) - low occupancy
- FA-2: 8 × 4 = 32 thread blocks (4 splits per head) - much better occupancy

### Improvement 3: Work Partitioning Within Thread Block

**Problem in FA-1**: Work distributed evenly across warps, but some warps idle during synchronization.

**Solution**: Non-uniform work distribution - assign more work to some warps, less to others for better pipelining.

**Warp-Level Optimization**:

A thread block contains 4 warps (128 threads on NVIDIA GPUs).

**FA-1 Distribution** (even):
```
Warp 0: Process rows 0-31 of Q block
Warp 1: Process rows 32-63 of Q block
Warp 2: Process rows 64-95 of Q block
Warp 3: Process rows 96-127 of Q block
```
All warps do same work, synchronize frequently.

**FA-2 Distribution** (producer-consumer):
```
Warp 0-1 (Producers): Load Q, K, V from HBM to shared memory
Warp 2-3 (Consumers): Compute S = Q@K, P = softmax(S), O = P@V
```

Benefits:
- **Pipelining**: Producers load next block while consumers compute current block
- **Reduced synchronization**: Fewer barriers needed
- **Better memory utilization**: Hide memory latency with computation

From [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691):
> "Within each thread block, distribute the work between warps to reduce communication through shared memory."

### Algorithm Comparison

| Aspect | FlashAttention-1 | FlashAttention-2 |
|--------|------------------|------------------|
| **Parallelism** | Batch × Heads | Batch × Heads × SeqSplits |
| **Softmax Rescaling** | Every block | Once per Q block |
| **Warp Distribution** | Uniform | Producer-consumer |
| **Non-matmul FLOPs** | Higher | Minimized |
| **Occupancy** | Lower | Higher |

### Performance Improvements

From [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691):

**A100 GPU (FP16)**:
- FlashAttention-1: ~125 TFLOPs, 40% utilization
- FlashAttention-2: ~225 TFLOPs, **72% utilization**
- Speedup: **1.8-2.0×** faster than FA-1

**GPT-Style Model Training**:
- Model: GPT-3 architecture (175B parameters)
- Hardware: A100 80GB
- Training speed: **225 TFLOPs/s per GPU** (72% model FLOPs utilization)
- Previous (with FA-1): ~125 TFLOPs/s

**Benchmark Results** (Sequence length = 8192, Head dim = 128):

| Implementation | TFLOPs | % of Peak | Speedup vs FA-1 |
|----------------|--------|-----------|-----------------|
| PyTorch (eager) | 80 | 25% | 0.64× |
| Triton | 95 | 30% | 0.76× |
| FlashAttention-1 | 125 | 40% | 1.0× |
| FlashAttention-2 | 225 | 72% | **1.8×** |

**Memory Efficiency**: Same as FA-1 (linear in sequence length), but faster execution.

---

## Section 4: FlashAttention-3 Hopper Optimizations (~180 lines)

From [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://tridao.me/blog/2024/flash3/) (Tri Dao, 2024):

FlashAttention-3 leverages three new hardware features on NVIDIA Hopper (H100) GPUs:
1. **WGMMA** (Warpgroup Matrix Multiply-Accumulate) - Faster Tensor Cores
2. **TMA** (Tensor Memory Accelerator) - Asynchronous memory transfers
3. **FP8** - Low-precision computation with hardware support

### New Hopper Hardware Features

**1. WGMMA (Warpgroup Matrix Multiply-Accumulate)**

From [NVIDIA H100 Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper):
- **Warpgroup**: 4 warps (128 threads) that execute matrix operations cooperatively
- **Previous (Ampere)**: `mma.sync` instruction, operates on single warp (32 threads)
- **Hopper**: `wgmma` instruction, operates on warpgroup (128 threads)
- **Throughput increase**: ~1.5-2× higher than `mma.sync` at same clock speed

**Performance Comparison**:
- A100 (Ampere, `mma.sync`): 312 TFLOPs FP16
- H100 (Hopper, `wgmma`): 989 TFLOPs FP16 (**3.2× higher** - clock + architecture)
- Effective wgmma benefit: **~2/3 comes from wgmma vs mma.sync**

**2. TMA (Tensor Memory Accelerator)**

Dedicated hardware unit for memory transfers between HBM and shared memory.

**Benefits**:
- **Asynchronous**: TMA runs independently, doesn't block warps
- **Index calculation**: TMA handles address computation, saves registers
- **Out-of-bounds**: Hardware handles boundary checks
- **Register pressure**: Frees up 20-30% of registers for other uses

**Without TMA** (Ampere/FA-2):
```cuda
// Warp explicitly loads data
for (int i = threadIdx.x; i < size; i += blockDim.x) {
    smem[i] = gmem[i];  // Each thread loads elements
}
__syncthreads();  // Wait for all loads
```

**With TMA** (Hopper/FA-3):
```cuda
// TMA loads asynchronously
tma_load_async(smem, gmem, size);
// Warp continues other work while TMA loads!
tma_wait();  // Wait only when data needed
```

**3. FP8 Low-Precision**

From [cuda/07-mixed-precision-training-internals.md](../cuda/07-mixed-precision-training-internals.md):
- **FP16 format**: 1 sign bit, 5 exponent bits, 10 mantissa bits
- **FP8 format (E4M3)**: 1 sign bit, 4 exponent bits, 3 mantissa bits
- **Throughput**: **2× faster than FP16** on H100 Tensor Cores
- **Peak**: 1978 TFLOPs FP8 vs 989 TFLOPs FP16

**Trade-off**: Lower precision → faster compute, but higher quantization error

### Asynchrony: Overlapping Compute and Memory

**Problem**: Non-matmul operations (softmax, exp) are much slower than matmul on modern GPUs.

From [FlashAttention-3 blog](https://tridao.me/blog/2024/flash3/):
> "The H100 GPU SXM5 has 989 TFLOPS of FP16 matrix multiply, but only 3.9 TFLOPS (256× less throughput) for special functions. For head dimension 128, there are 512× more matmul FLOPS than exponential, which means that exponential can take 50% of the time compared to matmul."

**Calculation**:
- H100: 989 TFLOPs FP16 matmul, ~3.9 TFLOPs special functions (exp, log)
- Matmul/exp ratio: **256×** throughput difference
- For d=128: Matmul FLOPs / exp FLOPs = 512
- Time ratio: 512 / 256 = 2 (exp takes **50% as much time as matmul**)

**For FP8**:
- Matmul: 1978 TFLOPs (2× faster)
- Exp: Still 3.9 TFLOPs (same speed)
- **Exp now takes 100% as much time as matmul!**

**Solution**: Overlap matmul (on Tensor Cores) with softmax (on multi-function units).

### Inter-Warpgroup Overlapping (Pingpong Scheduling)

Use 2 warpgroups in producer-consumer pattern:

**Pingpong Schedule**:
```
Time    Warpgroup 1              Warpgroup 2
─────   ────────────────────────  ───────────────────────
t0      GEMM0 (Q0×K0)            [waiting]
t1      GEMM1 (Q0×V0)            GEMM0 (Q1×K1)
t2      Softmax0                 GEMM1 (Q1×V1)
t3      GEMM0 (Q1×K1)            Softmax1
t4      GEMM1 (Q1×V1)            GEMM0 (Q2×K2)
t5      Softmax1                 GEMM1 (Q2×V2)
...
```

**Key**: While WG1 does softmax, WG2 does GEMMs. Hide softmax latency!

From [FlashAttention-3 blog](https://tridao.me/blog/2024/flash3/):
> "Pingpong scheduling can improve FP16 attention forward pass from around 570 TFLOPS to 620 TFLOPS (head dim 128, seqlen 8K)."

**Performance**: +50 TFLOPs (**8% improvement**) from better scheduling.

### Intra-Warpgroup Overlapping (2-Stage Pipeline)

Even within one warpgroup, pipeline GEMM and softmax:

**2-Stage Pipeline**:
```
Producer Threads         Consumer Threads
────────────────        ────────────────
Load K0, V0             [waiting]
Load K1, V1             GEMM(Q, K0) + Softmax
Load K2, V2             GEMM(Q, K1) + Softmax
...
```

**Mechanism**:
- Split warpgroup into producer (load data) and consumer (compute) threads
- TMA enables asynchronous loading while compute happens
- Register pressure managed by careful allocation

From [FlashAttention-3 blog](https://tridao.me/blog/2024/flash3/):
> "This pipelining increases throughput from around 620 TFLOPS to around 640-660 TFLOPS for FP16 attention forward, at the cost of higher register pressure."

**Performance**: +20-40 TFLOPs (**3-6% improvement**), total **640-660 TFLOPs** on H100.

### Warp Specialization with TMA

From [NVIDIA CUTLASS Documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md#warp-specialization):

**Pattern**:
```cuda
// Warp specialization
if (warp_id < 2) {
    // Producer warps
    while (!done) {
        tma_load_async(smem_next, gmem_Q);
        tma_load_async(smem_next, gmem_K);
        tma_load_async(smem_next, gmem_V);
        swap(smem_current, smem_next);
    }
} else {
    // Consumer warps
    while (!done) {
        wgmma(accum, smem_current_Q, smem_current_K);  // Q×K
        softmax(accum);
        wgmma(output, accum, smem_current_V);  // P×V
    }
}
```

**Double Buffering**: Use two sets of shared memory buffers, swap between them.

**Benefits**:
- Producers always loading ahead
- Consumers always have data ready
- **Hide memory latency** completely

### Low-Precision: Incoherent Processing for FP8

**Problem**: LLM activations have outliers with large magnitude. Quantizing to FP8 produces large errors.

From [FlashAttention-3 blog](https://tridao.me/blog/2024/flash3/):
> "LLM activation can have outliers with much larger magnitude than the rest of the features. These outliers make it difficult to quantize, producing much larger quantization errors."

**Solution**: Incoherent processing with Hadamard transform.

**Technique** (from QuIP/QuIP# quantization papers):
1. Multiply Q, K by random orthogonal matrix (Hadamard)
2. "Spreads out" outliers across all dimensions
3. Reduces max quantization error by distributing magnitude

**Hadamard Transform**:
```python
def hadamard_transform(x, signs):
    # x: (batch, seq, d)
    # signs: random +1/-1 per dimension

    # Fast Hadamard Transform: O(d log d) instead of O(d²)
    x = x * signs  # Random sign flips

    # Recursive Hadamard (butterfly pattern)
    d = x.size(-1)
    while d > 1:
        half = d // 2
        x[..., :half], x[..., half:] = (
            x[..., :half] + x[..., half:],
            x[..., :half] - x[..., half:]
        )
        d = half

    return x / sqrt(d)  # Normalized
```

**Fusion**: Can fuse Hadamard with RoPE (rotary position embedding) since both are memory-bandwidth bound.

**Error Reduction**:

From [FlashAttention-3 blog](https://tridao.me/blog/2024/flash3/):

Experiment: Q, K, V from standard normal, but 0.1% entries have large magnitude (simulating outliers).

| Method | Mean Squared Error |
|--------|-------------------|
| FP8 (naive) | 0.026 |
| FP8 + Hadamard | **0.010** |
| Improvement | **2.6× lower error** |

**Performance with FP8**:
- H100 FP8: 1978 TFLOPs peak
- FlashAttention-3: **~1200 TFLOPs** (60% utilization)
- **1.6× faster than FP16 FA-3** while maintaining accuracy

### Performance Summary

From [FlashAttention-3 blog](https://tridao.me/blog/2024/flash3/):

**H100 GPU Performance (sequence length 8K, head dim 128)**:

| Implementation | FP16 TFLOPs | FP8 TFLOPs | % of Peak |
|----------------|-------------|------------|-----------|
| FlashAttention-2 | 350 | N/A | 35% |
| FA-3 (basic port) | 570 | 950 | 58% / 48% |
| FA-3 + pingpong | 620 | 1000 | 63% / 51% |
| FA-3 + intra-overlap | 660 | 1050 | 67% / 53% |
| FA-3 (full) | **740** | **1200** | **75% / 61%** |

**Speedup**:
- FA-3 vs FA-2: **2.1× (FP16)**, **3.4× (FP8)**
- FA-3 vs standard attention: **~6-8× (FP16)**, **~10-12× (FP8)**

**Context Lengths Enabled**:
- FA-2: Up to 128K tokens (with tight memory)
- FA-3: Up to 256K+ tokens (FP16), 512K+ tokens (FP8)

---

## Section 5: PyTorch Integration (~120 lines)

### torch.nn.functional.scaled_dot_product_attention

From [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) (accessed 2025-02-03):

PyTorch 2.0+ provides `scaled_dot_product_attention` (SDPA) as the recommended API for attention:

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None
):
    """
    Computes scaled dot-product attention with automatic backend selection.

    Backends:
    - FlashAttention (fastest, requires Ampere+ GPU)
    - Memory-efficient attention (Volta+)
    - Math implementation (fallback, all GPUs)
    """
    pass  # Implementation selects best backend automatically
```

**Automatic Backend Selection**:

PyTorch automatically chooses fastest available implementation:

1. **FlashAttention** (if available):
   - Requires: Ampere (A100, A6000) or newer GPU
   - Best for: Long sequences (>512 tokens)
   - Speed: 2-4× faster than alternatives

2. **Memory-efficient attention**:
   - Requires: Volta (V100) or newer
   - Uses kernel fusion, but not as optimized as FlashAttention
   - Speed: 1.5-2× faster than math

3. **Math implementation** (fallback):
   - Works on all GPUs
   - Standard PyTorch operations
   - Slowest, but most compatible

**Usage Example**:

```python
import torch
import torch.nn.functional as F

# Inputs
batch, num_heads, seq_len, head_dim = 2, 8, 4096, 64
query = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
key = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
value = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')

# Compute attention (automatically uses FlashAttention on A100+)
output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,  # Optional attention mask
    dropout_p=0.1,   # Dropout probability
    is_causal=True,  # Use causal mask for autoregressive
    scale=None       # Default: 1/sqrt(head_dim)
)

# output shape: (batch, num_heads, seq_len, head_dim)
```

### Backend Selection Control

```python
# Check which backend will be used
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    # Force FlashAttention (will error if not available)
    output = F.scaled_dot_product_attention(query, key, value)

# Or query backend availability
from torch.backends.cuda import sdp_kernel, SDPBackend

# Check what's available
flash_available = SDPBackend.FLASH_ATTENTION in sdp_kernel.available_backends()
mem_efficient_available = SDPBackend.EFFICIENT_ATTENTION in sdp_kernel.available_backends()

print(f"FlashAttention available: {flash_available}")
print(f"Memory-efficient available: {mem_efficient_available}")
```

### Integration with torch.compile

From [PyTorch Tutorial: Scaled Dot Product Attention](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html) (accessed 2025-02-03):

FlashAttention is **fully compatible with torch.compile**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.shape

        # QKV projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # FlashAttention via SDPA (automatically selected)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True  # Causal mask for GPT-style models
        )

        # Reshape and project back
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y

# Compile the module
model = CausalSelfAttention(n_embd=768, n_head=12).cuda()
compiled_model = torch.compile(model)

# Forward pass (FlashAttention fused into compiled graph)
x = torch.randn(2, 1024, 768, device='cuda')
output = compiled_model(x)
```

**Benefits of torch.compile + FlashAttention**:
- **Graph fusion**: SDPA fused with surrounding ops (LayerNorm, residuals)
- **Kernel specialization**: Compiled for specific sequence lengths
- **Further speedup**: 1.2-1.4× on top of FlashAttention alone

From [PyTorch 2.0 Blog](https://pytorch.org/blog/pytorch2-2/) (accessed 2025-02-03):
> "PyTorch 2.2 offers ~2x performance improvements to scaled_dot_product_attention via FlashAttention-v2 integration."

### Custom Attention Masks

FlashAttention supports various masking patterns:

```python
# 1. Causal mask (autoregressive, GPT-style)
output = F.scaled_dot_product_attention(
    q, k, v,
    is_causal=True  # Efficient causal implementation
)

# 2. Custom boolean mask
# mask shape: (batch, num_heads, seq_len, seq_len) or broadcastable
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
output = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=mask  # True = masked out
)

# 3. Additive mask (for attention bias)
# Common for relative position embeddings
bias = torch.randn(1, num_heads, seq_len, seq_len)
output = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=bias  # Added to attention scores before softmax
)

# 4. Sliding window attention (local context)
# Not directly supported, use custom mask:
window_size = 256
mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
for i in range(seq_len):
    start = max(0, i - window_size)
    end = min(seq_len, i + window_size + 1)
    mask[i, start:end] = False

output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

### PyTorch 2.2+ Improvements

From [PyTorch 2.2 Release Notes](https://pytorch.org/blog/pytorch2-2/):

**FlashAttention-2 Integration** (January 2024):
- Default backend upgraded from FA-1 to FA-2
- **2× performance improvement** over PyTorch 2.1
- Automatic on CUDA 11.8+, Ampere+ GPUs

**Performance Example** (A100, seq_len=8192, head_dim=128):
- PyTorch 2.1 (FA-1): ~125 TFLOPs
- PyTorch 2.2 (FA-2): ~225 TFLOPs
- **1.8× faster** with same API

**Backward Compatibility**:
- Old code using manual attention: Still works
- New code using SDPA: Automatically gets FlashAttention-2
- No code changes required for upgrade

### MultiheadAttention Module

```python
import torch.nn as nn

# PyTorch's built-in MultiheadAttention uses SDPA internally
attn = nn.MultiheadAttention(
    embed_dim=768,
    num_heads=12,
    dropout=0.1,
    batch_first=True  # (batch, seq, embed) format
).cuda()

# Automatically uses FlashAttention on compatible GPUs
x = torch.randn(2, 1024, 768, device='cuda')
output, attn_weights = attn(x, x, x)

# For self-attention without returning weights (faster):
output = attn(x, x, x, need_weights=False)[0]
```

---

## Section 6: ARR-COC Connection (~100 lines)

### Query-Aware Relevance Scoring with FlashAttention

ARR-COC uses three relevance scorers based on Vervaeke's "ways of knowing":

1. **Propositional** (knowing THAT): Information content via cross-attention
2. **Perspectival** (knowing WHAT IT'S LIKE): Salience via cross-attention
3. **Participatory** (knowing BY BEING): Query-content coupling via cross-attention

All three use **cross-attention** between query and visual patches:

```python
# Simplified ARR-COC relevance scoring
def participatory_scorer(query_features, patch_features):
    # query_features: (batch, query_tokens, dim)
    # patch_features: (batch, num_patches, dim)

    # Cross-attention: queries attend to patches
    relevance_scores = F.scaled_dot_product_attention(
        query=query_features,     # What we're looking for
        key=patch_features,        # Patch content
        value=patch_features,      # Patch features
        is_causal=False            # Bidirectional attention
    )

    # Aggregate to per-patch relevance
    patch_relevance = relevance_scores.mean(dim=1)  # Average over query tokens

    return patch_relevance  # Shape: (batch, num_patches, dim)
```

**Why FlashAttention Matters**:
- **Many patches**: 196-400 patches per image (14×14 to 20×20 grid)
- **Multiple scorers**: 3 scorers × 3 cross-attention operations = 9 attention calls
- **Per-patch resolution**: Each patch can have 64-400 tokens (variable LOD)
- **FlashAttention speedup**: 2-4× faster enables real-time relevance scoring

### Variable LOD Token Allocation

ARR-COC allocates 64-400 tokens per patch based on relevance:

```python
def allocate_tokens_by_relevance(patch_features, relevance_scores,
                                 total_budget=1600):
    # relevance_scores: (batch, num_patches)
    # total_budget: Total tokens across all patches (e.g., 8×200 = 1600)

    # Softmax to get token allocation distribution
    allocation_weights = F.softmax(relevance_scores, dim=-1)

    # Allocate tokens proportionally (64-400 range)
    tokens_per_patch = (allocation_weights * total_budget).round()
    tokens_per_patch = torch.clamp(tokens_per_patch, min=64, max=400)

    # Adjust to meet exact budget (distribute rounding error)
    total_allocated = tokens_per_patch.sum(dim=-1, keepdim=True)
    adjustment = total_budget - total_allocated
    tokens_per_patch = tokens_per_patch + (adjustment / num_patches)

    return tokens_per_patch.long()  # Integer token counts
```

**FlashAttention for Variable Lengths**:

Each patch becomes a "sequence" with variable length:

```python
# Process patches with different token counts
outputs = []
for patch_idx, num_tokens in enumerate(tokens_per_patch):
    # Extract patch tokens (variable length)
    patch_tokens = texture_features[patch_idx, :num_tokens]

    # Self-attention within patch (FlashAttention)
    patch_output = F.scaled_dot_product_attention(
        query=patch_tokens,
        key=patch_tokens,
        value=patch_tokens,
        is_causal=False
    )

    outputs.append(patch_output)

# Concatenate variable-length outputs
final_features = torch.cat(outputs, dim=1)  # Total: ~1600 tokens
```

### Multi-Query Attention for Efficient KV Cache

ARR-COC inference benefits from **Multi-Query Attention** (MQA):

From [vllm-knowledge/00-vllm-architecture-pagedattention.md](../vllm-knowledge/00-vllm-architecture-pagedattention.md):

**Standard Multi-Head Attention**:
```
Q: num_heads × seq_len × head_dim
K: num_heads × seq_len × head_dim
V: num_heads × seq_len × head_dim
KV cache: 2 × num_heads × seq_len × head_dim
```

**Multi-Query Attention** (used in ARR-COC):
```
Q: num_heads × seq_len × head_dim
K: 1 × seq_len × head_dim  # Shared across heads!
V: 1 × seq_len × head_dim  # Shared across heads!
KV cache: 2 × 1 × seq_len × head_dim  # Much smaller!
```

**Memory Savings**:
- Standard MHA: 32 heads → 64 KV tensors
- MQA: 32 heads → 2 KV tensors
- **32× smaller KV cache** (critical for long contexts)

**FlashAttention with MQA**:

```python
class MultiQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_query_heads, head_dim):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.head_dim = head_dim

        # Q projection: separate for each head
        self.q_proj = nn.Linear(hidden_dim, num_query_heads * head_dim)

        # K, V projections: shared across heads
        self.k_proj = nn.Linear(hidden_dim, head_dim)
        self.v_proj = nn.Linear(hidden_dim, head_dim)

    def forward(self, x):
        B, N, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, N, self.num_query_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, 1, self.head_dim)  # Single head
        v = self.v_proj(x).view(B, N, 1, self.head_dim)  # Single head

        # Broadcast K, V to all query heads
        k = k.expand(B, N, self.num_query_heads, self.head_dim)
        v = v.expand(B, N, self.num_query_heads, self.head_dim)

        # Transpose to (B, num_heads, N, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # FlashAttention (automatically selected)
        output = F.scaled_dot_product_attention(q, k, v)

        # Reshape back
        output = output.transpose(1, 2).reshape(B, N, -1)
        return output
```

### Attention Budget Design (64-400 tokens)

Why 64-400 tokens per patch? **GPU tensor core alignment**.

From [cuda/05-tensor-core-programming-wmma-mma.md](../cuda/05-tensor-core-programming-wmma-mma.md):

**Tensor Core Tile Sizes**:
- Ampere (A100): 16×8×16, 16×8×8 (m×n×k)
- Hopper (H100): 64×64×16, 64×128×16

**Optimal Dimensions**: Multiples of 16 (Ampere) or 64 (Hopper)

**ARR-COC Token Counts**:
- Min: 64 tokens = 4 × 16 (Ampere-aligned)
- Average: 200 tokens = 12.5 × 16 (close to 13 × 16 = 208)
- Max: 400 tokens = 25 × 16 (perfect Ampere alignment)

**Why Multiples of 16/64**:
- Tensor Cores process matrices in 16×16 tiles (Ampere) or 64×64 tiles (Hopper)
- Non-aligned dimensions waste GPU resources (padding required)
- **Aligned dimensions → full Tensor Core utilization**

### Texture Array Processing

ARR-COC uses 13-channel texture arrays:

```python
# Texture channels (per patch):
# RGB: 3 channels
# LAB: 3 channels
# Sobel: 2 channels (x, y gradients)
# Spatial: 2 channels (x, y coordinates)
# Eccentricity: 1 channel
# Local contrast: 1 channel
# Optical flow: 1 channel (optional)

# Total: 13 channels per patch

def process_texture_array(image_patches):
    # image_patches: (batch, num_patches, 13, patch_h, patch_w)

    # Flatten spatial dimensions
    batch, num_patches, channels, h, w = image_patches.shape
    texture_flat = image_patches.view(batch, num_patches, channels, h * w)

    # Project each channel to embedding dimension
    texture_embedded = self.texture_proj(texture_flat)
    # Shape: (batch, num_patches, channels * embed_dim)

    # Self-attention across texture channels (FlashAttention)
    texture_features = F.scaled_dot_product_attention(
        query=texture_embedded,
        key=texture_embedded,
        value=texture_embedded,
        is_causal=False
    )

    return texture_features
```

**FlashAttention Benefit**: 13 channels × 196 patches = 2548 tokens. FlashAttention enables processing this efficiently.

---

## Sources

**Original Papers**:
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) - Tri Dao et al., NeurIPS 2022 (accessed 2025-02-03)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) - Tri Dao, 2023 (accessed 2025-02-03)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) - Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao, 2024 (accessed 2025-02-03)

**Web Resources**:
- [FlashAttention-3 Blog Post](https://tridao.me/blog/2024/flash3/) - Tri Dao, July 2024 (accessed 2025-02-03)
- [PyTorch scaled_dot_product_attention Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) - PyTorch Team (accessed 2025-02-03)
- [PyTorch 2.2 Release: FlashAttention-v2 Integration](https://pytorch.org/blog/pytorch2-2/) - PyTorch Team, January 2024 (accessed 2025-02-03)
- [Scaled Dot Product Attention Tutorial](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html) - PyTorch Tutorials (accessed 2025-02-03)
- [NVIDIA Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper) - NVIDIA, 2022 (accessed 2025-02-03)
- [NVIDIA CUTLASS: Efficient GEMM Documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md) - NVIDIA (accessed 2025-02-03)

**Related Knowledge**:
- [../cuda/01-memory-management-unified.md](../cuda/01-memory-management-unified.md) - GPU memory hierarchy
- [../cuda/05-tensor-core-programming-wmma-mma.md](../cuda/05-tensor-core-programming-wmma-mma.md) - Tensor Core programming
- [../cuda/06-pytorch-jit-torch-compile.md](../cuda/06-pytorch-jit-torch-compile.md) - PyTorch compilation and kernel fusion
- [../cuda/07-mixed-precision-training-internals.md](../cuda/07-mixed-precision-training-internals.md) - FP8/FP16 mixed precision
- [../vllm-knowledge/00-vllm-architecture-pagedattention.md](../vllm-knowledge/00-vllm-architecture-pagedattention.md) - PagedAttention (builds on FlashAttention)

**GitHub Repositories**:
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) - Official FlashAttention implementation (accessed 2025-02-03)
