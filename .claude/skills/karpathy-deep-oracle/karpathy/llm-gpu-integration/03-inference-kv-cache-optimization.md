# LLM Inference Optimization: KV Cache Management and Advanced Techniques

## Overview

LLM inference is fundamentally memory-bound rather than compute-bound. The Key-Value (KV) cache—which stores attention keys and values from previously generated tokens—often exceeds the model's parameter memory, becoming the primary bottleneck in serving large language models at scale. This document explores KV cache management strategies, continuous batching, PagedAttention, speculative decoding, and quantization techniques that enable efficient high-throughput LLM inference.

**Why KV Cache Dominates Memory:**

From existing knowledge in [vllm-knowledge/00-vllm-architecture-pagedattention.md](../../vllm-knowledge/00-vllm-architecture-pagedattention.md):
- For LLaMA-13B, a single 2048-token sequence requires **1.7GB of KV cache**
- Traditional systems waste 60-80% of memory due to fragmentation and over-reservation
- KV cache size grows linearly: `N_tokens × N_layers × 2 (K+V) × hidden_dim × batch_size × bytes_per_param`

**Memory Calculation Example:**

```
Model: LLaMA-7B (32 layers, 32 heads, 128 head_dim, 4096 hidden_dim)
Sequence: 2048 tokens
Batch: 8 concurrent requests
Precision: FP16 (2 bytes)

KV cache per token = 2 × 32 layers × 4096 dim × 2 bytes = 524 KB
Total KV cache = 2048 tokens × 524 KB × 8 batch = 8.6 GB

Model weights (FP16) = 7B params × 2 bytes = 14 GB
KV cache ratio = 8.6 / 14 = 61% of model size!
```

From [TensorWave: Estimating LLM Inference Memory Requirements](https://tensorwave.com/blog/estimating-llm-inference-memory-requirements) (accessed 2025-02-03):
- Context length, batch size, and KV cache dominate memory requirements
- GPU parallelism (tensor/pipeline) affects memory distribution across devices

---

## Section 1: KV Cache Fundamentals (~120 lines)

### What is KV Cache?

During autoregressive LLM generation, each attention layer computes keys (K) and values (V) for all tokens. Without caching, generating token N+1 would require recomputing K,V for all N previous tokens—an O(N²) operation.

**KV Cache Solution:**
- Store computed K,V tensors from previous tokens
- New token only computes its own K,V
- Attention: `softmax(Q_new @ K_cached^T) @ V_cached`
- Transforms O(N²) to O(N) per new token

From [Medium: KV Cache - The Secret to Faster LLM Inference](https://medium.com/@sailakkshmiallada/kv-cache-the-secret-to-faster-llm-inference-f919839eae7a) (accessed 2025-02-03):
- KV cache is the "memory" of the conversation
- Enables incremental token generation without full recomputation
- Trade-off: faster inference vs. increased memory consumption

### Memory Growth Pattern

**Per-Layer KV Cache:**
```
K_cache: [batch, num_heads, seq_len, head_dim]
V_cache: [batch, num_heads, seq_len, head_dim]

Memory per layer = batch × num_heads × seq_len × head_dim × 2 (K+V) × bytes
```

**Growth with Sequence Length:**

| Sequence Length | KV Cache (LLaMA-7B, batch=1, FP16) |
|-----------------|-------------------------------------|
| 512 tokens      | 1.0 GB                              |
| 2048 tokens     | 4.0 GB                              |
| 8192 tokens     | 16.0 GB                             |
| 32768 tokens    | 64.0 GB (exceeds A100 80GB!)        |

From [arXiv: XKV - Personalized KV Cache Memory Reduction](https://arxiv.org/html/2412.05896v1) (accessed 2025-02-03):
- Parameters remain fixed at 16GB
- KV cache consumption increases linearly with sequence length
- At 32K context, KV cache can be 4× larger than model weights

### Why Inference is Memory-Bound

**Compute vs. Memory Bottleneck:**

From [MatX Research: Optimize for Inference Too](https://matx.com/research/lifetime_llm_cost) (accessed 2025-02-03):
- For LLaMA 3 405B, the ratio of KV cache load cost to compute cost is ~1.5
- Memory bandwidth (1.5 TB/s HBM on A100) limits token generation speed
- GPU compute (312 TFLOPs FP16 on A100) sits idle waiting for memory

**Autoregressive Generation Bottleneck:**
```
Token generation is memory-bound because:
1. Each token requires loading entire KV cache from HBM
2. Compute: O(seq_len × hidden_dim) - small matrix multiply
3. Memory: O(seq_len × num_layers × hidden_dim) - large KV cache read
4. Result: GPU spends more time moving data than computing
```

From [Databricks: LLM Inference Performance Engineering](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) (accessed 2025-02-03):
- Large batch size means larger KV cache, requiring more GPUs
- Trade-off: throughput (large batch) vs. latency (small batch) vs. cost (GPU count)

### Multi-Query and Grouped-Query Attention

**Reducing KV Cache with MQA/GQA:**

Standard Multi-Head Attention (MHA):
- Each of N heads has its own K,V projections
- KV cache: `N_heads × hidden_dim`

Multi-Query Attention (MQA):
- All N query heads share 1 KV head
- KV cache: `1 × hidden_dim` (N× reduction)
- Used in: Falcon, StarCoder, PaLM

Grouped-Query Attention (GQA):
- N query heads share G KV heads (1 < G < N)
- KV cache: `G × hidden_dim` (N/G× reduction)
- Used in: LLaMA 2 70B, Mistral

From [Karpathy's gpt-architecture knowledge](../gpt-architecture/) (in existing oracle):
- MQA reduces KV cache size by up to 8× with minimal quality loss
- GQA balances memory savings with model capacity

---

## Section 2: Continuous Batching and Iteration-Level Scheduling (~150 lines)

### The Problem with Naive Batching

**Request-Level Batching (Traditional):**
```python
# Wait for ALL requests in batch to finish before starting next batch
batch = [req1, req2, req3, req4]  # All start together
while not all_finished(batch):
    generate_next_token(batch)
# If req1 finishes at 10 tokens but req4 needs 200 tokens,
# req1 wastes GPU cycles waiting
```

**Inefficiency:**
- Sequences finish at different times (10-500 tokens)
- Batch held until longest sequence completes
- GPU underutilized as sequences finish
- Can't add new requests until entire batch done

### Continuous Batching: Iteration-Level Scheduling

From [Medium: Continuous Batching and Selective Batching (Orca)](https://medium.com/byte-sized-ai/inference-optimizations-1-continuous-batching-03408c673098) (accessed 2025-02-03):

**Orca's Innovation (OSDI 2022):**
- **Iteration-level scheduling**: Adjust batch composition at each token generation step
- Remove finished sequences immediately
- Add new sequences as slots free up
- Maximize GPU utilization at every iteration

**Continuous Batching Algorithm:**
```python
active_sequences = []
queue = RequestQueue()

while True:
    # Iteration-level scheduling
    for seq in active_sequences:
        if seq.is_finished():
            active_sequences.remove(seq)
            return_result(seq)

    # Add new sequences up to memory limit
    while has_memory_for_new() and not queue.empty():
        new_seq = queue.pop()
        active_sequences.append(new_seq)

    # Generate next token for current batch
    generate_tokens(active_sequences)
```

From [Anyscale: Continuous Batching for LLM Inference](https://www.anyscale.com/blog/continuous-batching-llm-inference) (accessed 2025-02-03):
- Achieves 23× higher throughput vs. naive batching
- Reduces average latency by allowing new requests to start sooner
- Enables 4× larger effective batch size with same memory

### Dynamic Batch Size Management

**Memory-Aware Batching:**

From existing knowledge in [vllm-knowledge/00-vllm-architecture-pagedattention.md](../../vllm-knowledge/00-vllm-architecture-pagedattention.md):
- Batch size adapts to available KV cache memory
- Higher memory efficiency → larger batches → better throughput
- Preemption and swapping for oversubscription

**Batch Size Formula:**
```
Max batch size = (Total GPU Memory - Model Weights) / KV Cache per Sequence

Example (A100 80GB, LLaMA-7B):
Available = 80 GB - 14 GB (weights) = 66 GB
KV per seq (2048 tokens) = 4 GB
Max batch = 66 / 4 = 16 concurrent sequences
```

### Preemption and Swapping

**Handling Memory Pressure:**

From [vllm-knowledge/00-vllm-architecture-pagedattention.md](../../vllm-knowledge/00-vllm-architecture-pagedattention.md):

**Preemption:**
- Pause low-priority requests when memory full
- Store KV cache state
- Resume when memory available
- Enables serving more requests than memory allows

**Swapping:**
- Move KV cache to CPU memory (slower but larger)
- Swap back when GPU memory frees up
- Trade latency for throughput
- Useful for long-context serving

**Priority Scheduling:**
```python
if gpu_memory_full():
    # Preempt lowest priority sequence
    victim = find_lowest_priority(active_sequences)
    save_kv_cache(victim, location='cpu')
    active_sequences.remove(victim)
    preempted_sequences.append(victim)

if gpu_memory_available() and preempted_sequences:
    # Resume highest priority preempted sequence
    seq = preempted_sequences.pop(0)
    load_kv_cache(seq, from_location='cpu')
    active_sequences.append(seq)
```

### Selective Batching (Orca)

From [Insujang GitHub: LLM Inference - Continuous Batching](https://insujang.github.io/2024-01-07/llm-inference-continuous-batching-and-pagedattention/) (accessed 2025-02-03):

**Orca's Two Key Techniques:**

1. **Continuous Batching:** Iteration-level scheduling (covered above)
2. **Selective Batching:** Different operators have different batching requirements

**Selective Batching Strategy:**
```
Attention operators:
- Benefit from batching (parallel Q×K computation)
- Batch all sequences together

Non-attention operators (LayerNorm, FFN):
- Different sequence lengths cause load imbalance
- Selectively batch similar-length sequences
- Pad minimally within sub-batches
```

From [arXiv: Disaggregated Inference Scheduling](https://arxiv.org/pdf/2403.02310) (accessed 2025-02-03):
- Orca and vLLM both use iteration-level batching with FCFS (First-Come-First-Served)
- Differ in batch composition policy
- vLLM focuses on PagedAttention memory management

---

## Section 3: PagedAttention: Virtual Memory for KV Cache (~220 lines)

### Core Concept: Blocks and Block Tables

From [vLLM Documentation: PagedAttention Design](https://docs.vllm.ai/en/latest/design/paged_attention/) (accessed 2025-02-03):

**Virtual Memory Analogy:**

| OS Virtual Memory | PagedAttention KV Cache |
|-------------------|-------------------------|
| Page              | Block (16 tokens)       |
| Byte              | Token                   |
| Process           | Sequence                |
| Page Table        | Block Table             |
| Physical Memory   | GPU HBM                 |

**Block Structure:**
```
Block size: 16 tokens (typical, configurable)
Each block contains:
  K: [block_size=16, num_heads, head_dim]
  V: [block_size=16, num_heads, head_dim]

Total block memory = 16 × num_heads × head_dim × 2 (K+V) × bytes
```

### Block Table Mapping

**Logical to Physical Mapping:**

```
Sequence: "The quick brown fox jumps over the lazy dog"
Tokens: [T0, T1, T2, T3, T4, ..., T43]

Logical blocks (contiguous view from model):
  Block 0: [T0...T15]    → Physical block 42
  Block 1: [T16...T31]   → Physical block 7
  Block 2: [T32...T43]   → Physical block 19 (partial, 12/16 tokens)

Block Table:
  [42, 7, 19]
```

**Benefits:**
- Physical blocks can be non-contiguous in GPU memory
- Eliminates fragmentation (only waste in last partial block)
- Enables memory sharing via copy-on-write

From existing [vllm-knowledge/00-vllm-architecture-pagedattention.md](../../vllm-knowledge/00-vllm-architecture-pagedattention.md):
- Memory waste < 4% (only last block per sequence)
- Enables up to 24× higher throughput vs. HuggingFace Transformers

### CUDA Kernel Implementation

From [vLLM Docs: PagedAttention CUDA Kernel](https://docs.vllm.ai/en/latest/design/paged_attention/) (accessed 2025-02-03):

**PagedAttention Kernel Steps:**

1. **Block Table Lookup:**
```cuda
// Thread gets query token position
int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
int logical_block = token_idx / BLOCK_SIZE;
int block_offset = token_idx % BLOCK_SIZE;

// Look up physical block
int physical_block = block_table[logical_block];
```

2. **Fetch K,V from Non-Contiguous Blocks:**
```cuda
// K,V not contiguous in memory!
// Gather from multiple physical blocks
for (int i = 0; i < num_blocks; i++) {
    int phys_block = block_table[i];
    K_block = kv_cache[phys_block];  // Fetch block i
    V_block = kv_cache[phys_block + offset];

    // Compute attention for this block
    attn_scores = Q @ K_block.T;  // [1, block_size]
    attn_output += softmax(attn_scores) @ V_block;
}
```

3. **Memory Coalescing:**
```cuda
// Despite non-contiguous blocks, kernel optimizes memory access:
// - Prefetch blocks to shared memory
// - Coalesce reads within block
// - Use warp-level primitives for efficient reduction
```

**Integration with FlashAttention:**

From [cuda/06-pytorch-jit-torch-compile.md](../../cuda/06-pytorch-jit-torch-compile.md) (in existing oracle):
- vLLM integrates FlashAttention with PagedAttention
- FlashAttention handles tiling for SRAM efficiency
- PagedAttention handles virtual memory mapping
- Combined: both memory bandwidth and capacity optimization

From [Hugging Face: PagedAttention Concept](https://huggingface.co/docs/text-generation-inference/en/conceptual/paged_attention) (accessed 2025-02-03):
- TGI's PagedAttention uses custom CUDA kernels from vLLM project
- Optimized memory access patterns despite non-contiguous storage

### Block Management and Allocation

**Block Allocator:**

```python
class BlockAllocator:
    def __init__(self, num_blocks):
        self.free_blocks = list(range(num_blocks))
        self.ref_counts = [0] * num_blocks

    def allocate_block(self):
        if not self.free_blocks:
            return None  # Out of memory
        block_id = self.free_blocks.pop()
        self.ref_counts[block_id] = 1
        return block_id

    def free_block(self, block_id):
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] == 0:
            self.free_blocks.append(block_id)

    def share_block(self, block_id):
        # Copy-on-write: increment reference count
        self.ref_counts[block_id] += 1
```

**Copy-on-Write (CoW) for Memory Sharing:**

From [vllm-knowledge/00-vllm-architecture-pagedattention.md](../../vllm-knowledge/00-vllm-architecture-pagedattention.md):

Use cases:
1. **Parallel Sampling:** Multiple outputs from same prompt share prefix blocks
2. **Beam Search:** Candidate sequences share common prefix
3. **System Prompts:** Many requests share same instruction prefix

```python
# Sequence 1: "System: You are helpful.\n\nUser: Hello"
# Sequence 2: "System: You are helpful.\n\nUser: Goodbye"

# Both share "System: You are helpful.\n\n" prefix
shared_blocks = [block_0, block_1]  # System prompt
seq1_blocks = shared_blocks + [block_42]  # "Hello"
seq2_blocks = shared_blocks + [block_99]  # "Goodbye"

# ref_count[block_0] = 2, ref_count[block_1] = 2
# Memory saved: 2 blocks (system prompt not duplicated)
```

**Benefits of CoW:**

From [vllm-knowledge/00-vllm-architecture-pagedattention.md](../../vllm-knowledge/00-vllm-architecture-pagedattention.md):
- 55% memory overhead reduction for complex sampling
- Up to 2.2× throughput improvement
- Makes beam search practical in production

### PagedAttention vs. Traditional Memory Management

**Traditional (Contiguous Allocation):**
```
Sequence 1: [████████████░░░░] 12 tokens, reserved 16
Sequence 2: [████████████████] 16 tokens, fully used
Sequence 3: [██░░░░░░░░░░░░░░] 2 tokens, reserved 16

Total: 30 tokens using 48 slots = 37.5% waste
Problem: Over-reservation + fragmentation
```

**PagedAttention (Block-Based):**
```
Sequence 1: [████████████████] 16 tokens (1 block full)
           [████░░░░░░░░░░░░] 4 tokens (1 partial)
Sequence 2: [████████████████] 16 tokens (1 block)
Sequence 3: [██░░░░░░░░░░░░░░] 2 tokens (1 partial)

Total: 30 tokens using 32 slots = 6.25% waste
Waste: Only in last partial blocks
```

### Performance Benchmarks

From [vLLM blog post](https://blog.vllm.ai/2023/06/20/vllm.html) (accessed 2025-02-03):

**LLaMA-7B on A10G (single output):**
- vLLM vs. HF Transformers: **24× higher throughput**
- vLLM vs. TGI: **2.2-2.5× higher throughput**

**LLaMA-13B on A100 40GB:**
- vLLM vs. HF Transformers: **14× higher throughput**
- vLLM vs. TGI: **2.2-2.5× higher throughput**

**Parallel Sampling (3 outputs per request):**
- vLLM vs. HF Transformers: **8.5-15× higher throughput**
- Memory sharing enables practical multi-output generation

**Production Deployment (LMSYS Chatbot Arena):**
- Serves 30K daily requests, 60K peak
- 30× higher throughput than initial HF backend
- 50% reduction in GPU count for same traffic

---

## Section 4: Prefix Caching (RadixAttention) (~150 lines)

### The Prefix Caching Problem

**Common Scenario:**
```
System prompt: "You are a helpful AI assistant..." (500 tokens)
User queries:
  - "What is machine learning?" (4 tokens)
  - "Explain neural networks" (3 tokens)
  - "How does backprop work?" (4 tokens)

Without caching: Recompute system prompt KV cache 3 times
With caching: Compute system prompt once, share across queries
```

From [vllm-knowledge/02-vllm-prefix-caching.md](../../vllm-knowledge/02-vllm-prefix-caching.md):

**Automatic Prefix Caching (APC):**
- Detects shared prefixes automatically (no manual configuration)
- Caches KV blocks for reusable prefixes
- Reduces redundant computation
- Particularly effective for chatbots with system prompts

### Block-Level Hashing (vLLM)

From [vllm-knowledge/02-vllm-prefix-caching.md](../../vllm-knowledge/02-vllm-prefix-caching.md):

**vLLM's Approach:**
```python
# Each block uniquely identified by hash of:
# 1. All tokens before this block (prefix)
# 2. Tokens in this block

block_hash = hash(prefix_tokens + block_tokens)

# Three-level mapping:
Request → Logical Block Table → Hash Table → Physical Blocks
```

**Example:**
```
Request 1: "The quick brown fox jumps over"
  Block 0 (tokens 0-15): hash("The quick brown fox")
  Physical block: 10

Request 2: "The quick brown fox sleeps under"
  Block 0 (tokens 0-15): hash("The quick brown fox")
  Physical block: 10 (SHARED! Cache hit)
  Block 1 (tokens 16-31): hash("...fox" + "sleeps under")
  Physical block: 42 (new)
```

**Block Metadata:**
- Block hash (unique identifier)
- Reference count (number of sequences using this block)
- Last accessed time (for LRU eviction)
- Total access count (usage statistics)
- Prefix length (position in token sequence)
- Completion status (full vs. partial block)

From [GitHub Issue #2614: RFC Automatic Prefix Caching](https://github.com/vllm-project/vllm/issues/2614) (accessed 2025-02-03):
- Only complete blocks are cacheable (16/16 tokens)
- Partial blocks kept out of hash table until full
- Indirection adds minimal overhead (~100-200ns per token)

### Cache Hit Rate and Performance

From [vllm-knowledge/02-vllm-prefix-caching.md](../../vllm-knowledge/02-vllm-prefix-caching.md):

**7k Context Benchmark (DeepSeek-R1 on 2× H100 SXM):**

| Test Type          | Duration | Speed (tok/s) | Speedup |
|--------------------|----------|---------------|---------|
| Fresh (no cache)   | 5.253s   | 28.6          | 1.0×    |
| Cache hit 1        | 4.572s   | 32.8          | 1.15×   |
| Cache hit 2        | 4.510s   | 33.3          | 1.16×   |
| Small context      | 4.124s   | 36.1          | -       |

**Key Insights:**
- ~15% latency reduction on cache hits for large contexts
- Cache performance approaches small prompt baseline
- Benefit scales with prefix length (longer prefix = larger speedup)

**Memory Savings Example:**

From [vllm-knowledge/02-vllm-prefix-caching.md](../../vllm-knowledge/02-vllm-prefix-caching.md):

```
Model: Llama-2-7B (32 layers, 32 heads, 128 head_dim)
Prefix: 2048 tokens (system prompt)
Concurrent requests: 10

KV cache per token = 2 × 32 × 32 × 128 × 2 bytes = 512 KB
Prefix cache = 2048 × 512 KB = 1 GB

Without caching: 10 requests × 1 GB = 10 GB
With caching: 1 GB (shared) + 10 × suffix_cache
Memory saved: 9 GB
```

### vLLM vs. SGLang RadixAttention

From [vllm-knowledge/02-vllm-prefix-caching.md](../../vllm-knowledge/02-vllm-prefix-caching.md):

**vLLM (Block-Level Hashing):**
- Hash table maps `hash(prefix + block) → physical block`
- Requires exact block-level match
- Fast hash table lookup O(1)
- Best for: Templated prompts, batch inference, predictable patterns

**SGLang (RadixAttention - Token-Level Radix Tree):**
- Radix tree for flexible prefix matching
- Token-level granularity
- Automatic tree pruning
- Best for: Dynamic conversations, varied flows, customer support

**Performance Comparison (7k context, 2× H100):**

From [RunPod: SGLang vs vLLM KV Cache](https://www.runpod.io/blog/sglang-vs-vllm-kv-cache) (accessed 2025-02-03):

| Engine  | Fresh    | Cache Hit 1 | Cache Hit 2 | Small Context |
|---------|----------|-------------|-------------|---------------|
| vLLM    | 5.253s   | 4.572s      | 4.510s      | 4.124s        |
| SGLang  | 5.093s   | 4.287s      | 4.295s      | 4.154s        |

**Insights:**
- Fresh: Similar (~2% difference)
- Cache hits: SGLang ~7% faster
- Both achieve significant speedup vs. no cache
- SGLang cache performance closer to small-context baseline

### Production Patterns

From [vllm-knowledge/02-vllm-prefix-caching.md](../../vllm-knowledge/02-vllm-prefix-caching.md):

**Pattern 1: Multi-Turn Chat**
```python
system_prompt = "You are a customer support agent..." # 500 tokens

# Turn 1: Full computation
response1 = llm.generate(system_prompt + "User: Reset password?")

# Turn 2: System prompt cached
response2 = llm.generate(system_prompt + conversation_history + "User: 2FA?")

# Turn 3: System + history cached
response3 = llm.generate(system_prompt + full_history + "User: Billing?")
```

**Pattern 2: RAG with Shared Context**
```python
retrieved_docs = "Context:\n" + doc1 + doc2 + doc3  # 3000 tokens

questions = [
    "What is gradient descent?",
    "Explain overfitting",
    "What are neural networks?"
]

# Context cached after first question
for q in questions:
    answer = llm.generate(retrieved_docs + f"\n\nQuestion: {q}")
    # Questions 2-3 reuse cached context (99% cache hit)
```

---

## Section 5: Speculative Decoding (~180 lines)

### The Core Idea: Draft and Verify

From [vllm-knowledge/03-vllm-speculative-decoding.md](../../vllm-knowledge/03-vllm-speculative-decoding.md):

**Problem:** Autoregressive generation is sequential—must wait for token N before generating N+1.

**Solution:** Use small draft model to speculate multiple tokens, large target model verifies in parallel.

**Draft-Target Paradigm:**

```
Draft Model (small, fast):
- 1-5% size of target model
- Quickly proposes K=3-12 candidate tokens
- Examples: Llama 68M drafts for Llama 2 70B

Target Model (large, authoritative):
- Production model whose output we want
- Verifies all K draft tokens in single forward pass
- Accepts correct predictions, rejects incorrect ones
```

From [NVIDIA Developer Blog: Introduction to Speculative Decoding](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/) (accessed 2025-02-03):

**Three-Step Process:**

1. **Draft Generation:**
```python
draft_tokens = []
for k in range(K):  # K = 5 typically
    token = draft_model.generate_next(context + draft_tokens)
    draft_tokens.append(token)
```

2. **Parallel Verification:**
```python
# Target processes ALL draft tokens in one pass
# Thanks to KV cache, only new tokens cost compute
logits = target_model.forward(context + draft_tokens)
target_probs = softmax(logits)  # [K+1, vocab_size]
```

3. **Rejection Sampling:**
```python
accepted = []
for i, draft_token in enumerate(draft_tokens):
    p_target = target_probs[i, draft_token]
    p_draft = draft_probs[i, draft_token]

    if p_target >= p_draft:
        accepted.append(draft_token)
    else:
        # Reject token i and all subsequent
        corrected_token = sample_from(target_probs[i])
        accepted.append(corrected_token)
        break

return accepted
```

### CUDA Streams for Parallel Execution

From existing [cuda/00-streams-concurrency-async.md](../../cuda/00-streams-concurrency-async.md):

**Multi-Stream Pattern for Speculative Decoding:**

```python
import torch

# Create separate streams for draft and target
draft_stream = torch.cuda.Stream()
target_stream = torch.cuda.Stream()

# Event for synchronization
draft_done = torch.cuda.Event()

# Step 1: Draft generation (small model, fast)
with torch.cuda.stream(draft_stream):
    draft_tokens = draft_model.generate(context, num_tokens=5)
    draft_done.record(draft_stream)

# Step 2: Target verification (waits for draft)
with torch.cuda.stream(target_stream):
    target_stream.wait_event(draft_done)
    logits = target_model(context + draft_tokens)
    verification = verify_tokens(draft_tokens, logits)

# Synchronize and return results
target_stream.synchronize()
return verification.accepted_tokens
```

**Overlapping Draft and Target (Advanced):**

From [vllm-knowledge/03-vllm-speculative-decoding.md](../../vllm-knowledge/03-vllm-speculative-decoding.md):

```python
# Pipeline: While target verifies batch N, draft generates batch N+1
for batch in batches:
    with torch.cuda.stream(draft_stream):
        draft_tokens_next = draft_model.generate(batch_next)

    with torch.cuda.stream(target_stream):
        # Verify current batch while draft works on next
        verified = target_model.verify(draft_tokens_current)

    draft_tokens_current = draft_tokens_next
```

### Performance Analysis

From [vllm-knowledge/03-vllm-speculative-decoding.md](../../vllm-knowledge/03-vllm-speculative-decoding.md):

**Key Metrics:**

1. **Acceptance Rate (α):**
   - Probability target accepts draft tokens
   - Typical: 40-80% depending on task
   - Higher = more tokens per target forward pass

2. **Block Efficiency (τ):**
   - Average tokens generated per target model pass
   - Maximum: γ + 1 (where γ = speculation length)
   - Example: γ=5, acceptance=60% → τ ≈ 3.5 tokens/pass

3. **Memory-Bound Speedup (MBSU):**
```
MBSU = (c × τ) / (c × γ + 1)

where:
  c = draft_params / target_params (relative latency)
  τ = block efficiency
  γ = speculation length

Example:
  c = 0.02 (68M draft, 7B target)
  γ = 5
  τ = 3.5 (60% acceptance)
  MBSU = (0.02 × 3.5) / (0.02 × 5 + 1) = 0.07 / 1.1 = 6.4% overhead
  Net speedup: 3.5 / 1.064 ≈ 3.3×
```

**Benchmark Results:**

From [vLLM Blog: Speculative Decoding Performance](https://blog.vllm.ai/2024/10/17/spec-decode.html) (accessed 2025-02-03):

Low QPS scenarios (QPS=1):
- ShareGPT with draft model: **1.5× speedup**
- CNN/DailyMail with n-gram: **2.8× speedup**
- Llama 3 70B: Block efficiency = 2.3 tokens/pass

From [Snowflake: Arctic Inference with vLLM](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/) (accessed 2025-02-03):

Arctic EAGLE-3 variant on 4× H100:
- **2.05-2.45× speedup** over non-speculative
- **1.69× faster** than vLLM n-gram speculator

### Draft Model Selection

From [arXiv: Direct Alignment of Draft Model](https://arxiv.org/html/2403.00858v4) (accessed 2025-02-03):

**Size Considerations:**
- Draft typically 1-5% of target size
- Example: 115M draft for 7B target (1.64% ratio)
- Too small: Low acceptance rate
- Too large: Draft overhead negates benefit

**Training Pipeline:**
1. Pretraining on large corpus (language modeling)
2. Distillation dataset from target model responses
3. Fine-tuning to align draft with target behavior

**Vocabulary Constraints:**
- Draft MUST use same tokenizer as target
- Shared vocabulary required for token verification
- Limits draft model selection for some LLM families

**Draft-Target Pairing Examples:**

From [vllm-knowledge/03-vllm-speculative-decoding.md](../../vllm-knowledge/03-vllm-speculative-decoding.md):

| Target Model | Draft Model | Size Ratio | Notes |
|--------------|-------------|------------|-------|
| Llama 2 7B | Llama 68M | 1% | Custom trained |
| Llama 3 70B | Llama 3 8B | 11% | Official small variant |
| OPT-6.7B | OPT-125M | 2% | Wide model family |
| OPT-30B | OPT-1.3B | 4% | Official pairing |

### Prompt Lookup Decoding (N-gram)

From [vllm-knowledge/03-vllm-speculative-decoding.md](../../vllm-knowledge/03-vllm-speculative-decoding.md):

**Draft-Model-Free Approach:**

```python
llm = LLM(
    model="facebook/opt-6.7b",
    speculative_model="[ngram]",  # Special keyword
    num_speculative_tokens=5,
    ngram_prompt_lookup_max=4,
)
```

**How It Works:**
1. Build n-gram lookup table from prompt
2. During generation, check if current n-gram in table
3. If match, propose following tokens from table
4. Target verifies as usual

**Best Use Cases:**
- Summarization (CNN/DailyMail: **2.8× speedup**)
- Q&A where context contains answer
- Document extraction/transformation
- Zero additional model overhead

---

## Section 6: Quantized KV Cache (~120 lines)

### Why Quantize KV Cache?

From [Medium: LLM Inference Series - KV Caching](https://medium.com/@plienhar/llm-inference-series-4-kv-caching-a-deeper-look-4ba9a77746c8) (accessed 2025-02-03):

**Memory Bottleneck:**
- KV cache grows linearly with sequence length
- FP16 KV cache: 2 bytes per value
- Large batches: KV cache > model weights

**Quantization Solution:**
- INT8: 1 byte per value (2× reduction)
- FP8: 1 byte per value (2× reduction, better accuracy)
- INT4: 0.5 bytes per value (4× reduction, more quality loss)

### FP8 Quantization (H100)

From [vLLM Docs: Quantized KV Cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache.html) (accessed 2025-02-03):

**H100 Native FP8 Support:**

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    kv_cache_dtype="fp8",  # Enable FP8 KV cache
    quantization="fp8",     # Also quantize weights
)
```

**FP8 Format (E4M3):**
- 1 sign bit
- 4 exponent bits
- 3 mantissa bits
- Range: ±448 (vs. FP16: ±65504)
- Sufficient for normalized KV values

From [NVIDIA Developer: Floating-Point 8 Introduction](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/) (accessed 2025-02-03):

**H100 FP8 Benefits:**
- Native Tensor Core support (zero overhead)
- 2× memory bandwidth reduction
- 2× capacity increase for same memory
- Minimal accuracy impact (0.1-0.5 perplexity increase)

### INT8 Quantization

**Per-Tensor Scaling:**

```python
def quantize_kv_cache_int8(kv_cache_fp16):
    # Compute scale per tensor
    max_val = torch.max(torch.abs(kv_cache_fp16))
    scale = max_val / 127.0

    # Quantize
    kv_cache_int8 = torch.clamp(
        kv_cache_fp16 / scale,
        -128, 127
    ).to(torch.int8)

    return kv_cache_int8, scale

def dequantize_kv_cache_int8(kv_cache_int8, scale):
    # Dequantize on-the-fly during attention
    return kv_cache_int8.to(torch.float16) * scale
```

**CUDA Kernel Integration:**

From [cuda/05-tensor-core-programming-wmma-mma.md](../../cuda/05-tensor-core-programming-wmma-mma.md) (in existing oracle):

```cuda
// INT8 KV cache dequantization in attention kernel
__global__ void paged_attention_int8_kv(
    float* Q,              // Query (FP16)
    int8_t* K_quantized,   // Key (INT8)
    int8_t* V_quantized,   // Value (INT8)
    float* K_scale,        // Per-tensor scale
    float* V_scale,
    int* block_table,      // PagedAttention
    float* output
) {
    // Fetch block using PagedAttention
    int block_id = block_table[logical_block];

    // Load quantized K,V
    int8_t K_int8 = K_quantized[block_id];
    int8_t V_int8 = V_quantized[block_id];

    // Dequantize on-the-fly
    float K = __int2float_rn(K_int8) * K_scale[0];
    float V = __int2float_rn(V_int8) * V_scale[0];

    // Compute attention (standard)
    float attn = Q * K;  // Dot product
    output += softmax(attn) * V;
}
```

### Accuracy vs. Memory Trade-off

From [Databricks: Serving Quantized LLMs on H100](https://www.databricks.com/blog/serving-quantized-llms-nvidia-h100-tensor-core-gpus) (accessed 2025-02-03):

**H100 FP8 vs A100 INT8:**

| GPU  | Mode      | Throughput | Speedup vs A100 INT8 |
|------|-----------|------------|----------------------|
| A100 | INT8 W+KV | Baseline   | 1.0×                 |
| H100 | FP8 W+KV  | 80% faster | 1.8×                 |

At large batch sizes, FP8 on H100 provides:
- 2× memory reduction
- 1.8× throughput improvement
- Near-zero quality degradation

**Quality Impact:**

From [vLLM Docs: Quantized KV Cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache.html) (accessed 2025-02-03):

- FP8: 0.1-0.5 perplexity increase (negligible)
- INT8: 0.3-1.0 perplexity increase (acceptable)
- INT4: 1-3 perplexity increase (task-dependent)

### Hierarchical KV Cache Quantization

From [arXiv: Self-Speculative Decoding with Hierarchical Quantized KV Cache](https://arxiv.org/html/2502.10424v1) (accessed 2025-02-03):

**Dynamic INT4/INT8 Switching:**

```python
class HierarchicalKVCache:
    def __init__(self):
        self.kv_int4 = None  # Low-precision draft
        self.kv_int8 = None  # High-precision verification

    def store(self, kv_fp16):
        # Store both INT4 and INT8 representations
        self.kv_int4 = quantize_int4(kv_fp16)
        self.kv_int8 = quantize_int8(kv_fp16)

    def get_draft(self):
        # Use INT4 for fast draft speculation
        return dequantize_int4(self.kv_int4)

    def get_verification(self):
        # Use INT8 for accurate target verification
        return dequantize_int8(self.kv_int8)
```

**Benefits:**
- Draft model uses INT4 (4× memory reduction)
- Target model uses INT8 (2× reduction, higher quality)
- No on-the-fly quantization overhead
- Dynamic switching based on speculative decoding phase

---

## Section 7: ARR-COC Multi-Stream Inference Pipeline (~100 lines)

### Relevance Scoring with KV Cache Optimization

**ARR-COC Three Ways of Knowing + KV Cache:**

From project context (arr-coc-ovis architecture):

```python
class ARRCOCKVCacheOptimized:
    """
    ARR-COC inference with PagedAttention and prefix caching.

    Three relevance scorers benefit from KV cache:
    - Propositional (Shannon entropy): Cache texture feature attention
    - Perspectival (Jungian salience): Cache salience map attention
    - Participatory (Query-aware): Cache query-content cross-attention
    """

    def __init__(self):
        # Separate streams for each scorer
        self.propositional_stream = torch.cuda.Stream()
        self.perspectival_stream = torch.cuda.Stream()
        self.participatory_stream = torch.cuda.Stream()

        # PagedAttention block allocator
        self.block_allocator = BlockAllocator(num_blocks=1024)

    def score_relevance(self, texture_array, query):
        """Score relevance with multi-stream execution."""

        # Texture features cached (shared across patches)
        texture_kv = self.cache_texture_features(texture_array)

        events = {}

        # Stream 1: Propositional (information content)
        with torch.cuda.stream(self.propositional_stream):
            prop_scores = self.propositional_scorer(texture_kv)
            events['prop_done'] = torch.cuda.Event()
            events['prop_done'].record(self.propositional_stream)

        # Stream 2: Perspectival (salience)
        with torch.cuda.stream(self.perspectival_stream):
            persp_scores = self.perspectival_scorer(texture_kv)
            events['persp_done'] = torch.cuda.Event()
            events['persp_done'].record(self.perspectival_stream)

        # Stream 3: Participatory (query-aware)
        with torch.cuda.stream(self.participatory_stream):
            # Cache query KV for multiple patches
            query_kv = self.cache_query_features(query)
            partic_scores = self.participatory_scorer(texture_kv, query_kv)
            events['partic_done'] = torch.cuda.Event()
            events['partic_done'].record(self.participatory_stream)

        # Synchronize all scorers
        for event in events.values():
            event.synchronize()

        return prop_scores, persp_scores, partic_scores

    def cache_texture_features(self, texture_array):
        """
        Cache 13-channel texture features (RGB, LAB, Sobel, etc.)
        with prefix caching for shared patches.
        """
        # Compute texture hash for prefix caching
        texture_hash = hash(texture_array.cpu().numpy().tobytes())

        if texture_hash in self.texture_cache:
            # Prefix cache HIT
            return self.texture_cache[texture_hash]

        # Prefix cache MISS: Compute and store
        with torch.no_grad():
            texture_kv = self.texture_encoder(texture_array)
            self.texture_cache[texture_hash] = texture_kv

        return texture_kv
```

### Multi-Query Attention for Efficient Cache

From existing knowledge on MQA/GQA:

```python
class ARRCOCAttention(nn.Module):
    """
    Multi-Query Attention for ARR-COC relevance scoring.
    Reduces KV cache size for 3 scorer heads.
    """

    def __init__(self, hidden_dim=4096, num_query_heads=32, num_kv_heads=4):
        super().__init__()
        self.num_query_heads = num_query_heads  # 32 for propositional
        self.num_kv_heads = num_kv_heads         # 4 shared KV heads
        self.head_dim = hidden_dim // num_query_heads

        # Query projections (32 heads)
        self.q_proj = nn.Linear(hidden_dim, num_query_heads * self.head_dim)

        # Key/Value projections (4 heads, shared across query heads)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim)

        # KV cache reduction: 32 → 4 heads = 8× smaller cache

    def forward(self, x, use_cache=True):
        batch, seq_len, hidden = x.shape

        # Compute Q, K, V
        Q = self.q_proj(x).view(batch, seq_len, self.num_query_heads, self.head_dim)
        K = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        V = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Cache KV (4 heads instead of 32)
        if use_cache:
            self.cached_k = K  # 4 heads, 8× smaller than 32 heads
            self.cached_v = V

        # Expand KV to match Q heads (repeat each KV head 8 times)
        K_expanded = K.repeat_interleave(self.num_query_heads // self.num_kv_heads, dim=2)
        V_expanded = V.repeat_interleave(self.num_query_heads // self.num_kv_heads, dim=2)

        # Standard attention
        attn_weights = torch.matmul(Q, K_expanded.transpose(-2, -1))
        attn_weights = torch.softmax(attn_weights / (self.head_dim ** 0.5), dim=-1)
        output = torch.matmul(attn_weights, V_expanded)

        return output
```

**Memory Savings:**
```
Standard MHA (32 heads):
  KV cache = 32 heads × 128 dim × seq_len × 2 (K+V) = 8192 × seq_len

MQA/GQA (4 KV heads):
  KV cache = 4 heads × 128 dim × seq_len × 2 (K+V) = 1024 × seq_len

Reduction: 8× smaller KV cache for ARR-COC relevance scoring
```

### Variable LOD Token Allocation

**KV Cache for 64-400 Token Range:**

```python
def allocate_tokens_with_cache_awareness(relevance_scores, total_budget=4096):
    """
    Allocate 64-400 tokens per patch based on relevance.
    Use quantized KV cache for low-relevance patches.
    """
    num_patches = len(relevance_scores)
    allocations = []

    for patch_idx, score in enumerate(relevance_scores):
        if score > 0.8:  # High relevance
            tokens = 400  # Maximum detail
            cache_dtype = "fp16"  # Full precision
        elif score > 0.5:  # Medium relevance
            tokens = 200  # Average detail
            cache_dtype = "fp8"   # 2× memory savings
        else:  # Low relevance
            tokens = 64   # Minimum detail
            cache_dtype = "int8"  # 2× memory savings

        allocations.append({
            'patch_idx': patch_idx,
            'tokens': tokens,
            'cache_dtype': cache_dtype,
            'relevance': score
        })

    return allocations
```

**GPU-Aware LOD:**

From existing knowledge in [karpathy/llm-gpu-integration/01-architecture-gpu-constraints.md](01-architecture-gpu-constraints.md):

```
ARR-COC token budgets aligned to Tensor Core shapes:
- 64 tokens = 4 × 16 (optimal for sm_80 A100)
- 200 tokens = 12.5 × 16 (average 8× compression)
- 400 tokens = 25 × 16 (maximum detail)

All multiples of 16 for efficient CUDA kernel execution.
```

---

## Sources

**Research Papers:**

- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - vLLM paper (SOSP 2023, accessed 2025-02-03)
- [XKV: Personalized KV Cache Memory Reduction](https://arxiv.org/html/2412.05896v1) (arXiv:2412.05896, accessed 2025-02-03)
- [InfiniGen: Efficient Generative Inference](https://www.usenix.org/system/files/osdi24-lee.pdf) (OSDI 2024, accessed 2025-02-03)
- [Direct Alignment of Draft Model for Speculative Decoding](https://arxiv.org/html/2403.00858v4) (arXiv:2403.00858, accessed 2025-02-03)
- [Self-Speculative Decoding with Hierarchical Quantized KV Cache](https://arxiv.org/html/2502.10424v1) (arXiv:2502.10424, accessed 2025-02-03)
- [Disaggregated Inference Scheduling](https://arxiv.org/pdf/2403.02310) (arXiv:2403.02310, accessed 2025-02-03)

**vLLM Documentation and Blog:**

- [vLLM Documentation: PagedAttention Design](https://docs.vllm.ai/en/latest/design/paged_attention/) (accessed 2025-02-03)
- [vLLM Documentation: Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching.html) (accessed 2025-02-03)
- [vLLM Blog: Easy, Fast, and Cheap LLM Serving](https://blog.vllm.ai/2023/06/20/vllm.html) (accessed 2025-02-03)
- [vLLM Blog: Speculative Decoding](https://blog.vllm.ai/2024/10/17/spec-decode.html) (accessed 2025-02-03)
- [vLLM Docs: Quantized KV Cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache.html) (accessed 2025-02-03)

**Industry Blogs and Guides:**

- [Medium: KV Cache - The Secret to Faster LLM Inference](https://medium.com/@sailakkshmiallada/kv-cache-the-secret-to-faster-llm-inference-f919839eae7a) by Saiii (accessed 2025-02-03)
- [Medium: LLM Inference Series - KV Caching Deep Dive](https://medium.com/@plienhar/llm-inference-series-4-kv-caching-a-deeper-look-4ba9a77746c8) by Pierre Lienhart (accessed 2025-02-03)
- [Medium: Continuous Batching (Orca)](https://medium.com/byte-sized-ai/inference-optimizations-1-continuous-batching-03408c673098) by Don Moon (accessed 2025-02-03)
- [Anyscale: Continuous Batching for LLM Inference](https://www.anyscale.com/blog/continuous-batching-llm-inference) (accessed 2025-02-03)
- [TensorWave: Estimating LLM Inference Memory](https://tensorwave.com/blog/estimating-llm-inference-memory-requirements) (accessed 2025-02-03)
- [MatX Research: Optimize for Inference Too](https://matx.com/research/lifetime_llm_cost) (accessed 2025-02-03)
- [Databricks: LLM Inference Performance Engineering](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) (accessed 2025-02-03)
- [Databricks: Serving Quantized LLMs on H100](https://www.databricks.com/blog/serving-quantized-llms-nvidia-h100-tensor-core-gpus) (accessed 2025-02-03)

**NVIDIA and Hardware:**

- [NVIDIA Developer: Introduction to Speculative Decoding](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/) (accessed 2025-02-03)
- [NVIDIA Developer: Floating-Point 8 Introduction](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/) (accessed 2025-02-03)
- [NVIDIA Developer: Blackwell with Llama 4 Maverick](https://developer.nvidia.com/blog/blackwell-breaks-the-1000-tps-user-barrier-with-metas-llama-4-maverick/) (accessed 2025-02-03)

**Benchmarks and Comparisons:**

- [RunPod: SGLang vs vLLM KV Cache](https://www.runpod.io/blog/sglang-vs-vllm-kv-cache) (accessed 2025-02-03)
- [Snowflake: Arctic Inference with vLLM](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/) (accessed 2025-02-03)
- [Hugging Face: PagedAttention Concept](https://huggingface.co/docs/text-generation-inference/en/conceptual/paged_attention) (accessed 2025-02-03)

**Community Resources:**

- [Insujang: LLM Inference - Continuous Batching](https://insujang.github.io/2024-01-07/llm-inference-continuous-batching-and-pagedattention/) (accessed 2025-02-03)
- [GitHub Issue #2614: RFC Automatic Prefix Caching](https://github.com/vllm-project/vllm/issues/2614) (accessed 2025-02-03)
- [BentoML: 3× Faster with Speculative Decoding](https://www.bentoml.com/blog/3x-faster-llm-inference-with-speculative-decoding) (accessed 2025-02-03)
- [Google Research: Looking Back at Speculative Decoding](https://research.google/blog/looking-back-at-speculative-decoding/) (accessed 2025-02-03)

**Related Knowledge Files:**

- [vllm-knowledge/00-vllm-architecture-pagedattention.md](../../vllm-knowledge/00-vllm-architecture-pagedattention.md) (local oracle knowledge)
- [vllm-knowledge/02-vllm-prefix-caching.md](../../vllm-knowledge/02-vllm-prefix-caching.md) (local oracle knowledge)
- [vllm-knowledge/03-vllm-speculative-decoding.md](../../vllm-knowledge/03-vllm-speculative-decoding.md) (local oracle knowledge)
- [cuda/00-streams-concurrency-async.md](../../cuda/00-streams-concurrency-async.md) (local oracle knowledge)
- [cuda/05-tensor-core-programming-wmma-mma.md](../../cuda/05-tensor-core-programming-wmma-mma.md) (local oracle knowledge)
- [cuda/07-mixed-precision-training-internals.md](../../cuda/07-mixed-precision-training-internals.md) (local oracle knowledge)

---

**Document Metadata:**

- **Created**: 2025-02-03
- **PART**: 4 of 4 (LLM + GPU Integration expansion)
- **Lines**: ~810
- **Topics**: KV cache, continuous batching, PagedAttention, prefix caching, speculative decoding, quantization
- **Target Audience**: ML engineers, LLM researchers, inference optimization practitioners
