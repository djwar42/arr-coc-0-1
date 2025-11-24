# KV Cache Management Internals

## Overview

KV cache management is the memory subsystem that enables efficient transformer inference by storing and reusing attention key-value pairs across autoregressive generation steps. This document provides a comprehensive technical analysis of KV cache structure, block-based allocation strategies, eviction policies, and memory optimization techniques used in production serving systems like vLLM.

**Key Insight**: KV cache grows linearly with sequence length and batch size, often consuming more GPU memory than model weights themselves. For a model with L layers, H attention heads, d hidden size per head, and sequence length S, the memory required is: **2 × S × L × H × d × 2 bytes** (for FP16).

From [PagedEviction: Structured Block-wise KV Cache Pruning](https://arxiv.org/html/2509.04377v1) (arXiv:2509.04377, accessed 2025-02-02):
- KV cache memory grows linearly with sequence length and can rival or exceed model weight memory
- For tens of thousands of tokens, the cache may consume more GPU memory than the entire model
- This severely limits throughput of LLM inference systems

## KV Cache Fundamentals

### What is KV Cache?

During autoregressive LLM inference, tokens are generated sequentially. Without caching, the model would recompute Key (K) and Value (V) states for all previously predicted tokens at every step. KV cache addresses this inefficiency by saving Key and Value activations for each token.

From [vLLM PagedAttention Documentation](https://docs.vllm.ai/en/v0.6.3.post1/automatic_prefix_caching/details.html) (accessed 2025-02-02):

**Without KV Cache** (naïve approach):
```
Step t: Compute Q_t, K_{1:t}, V_{1:t} from scratch
        O(t × d) computation for each step
```

**With KV Cache** (optimized):
```
Step 1: Compute K_1, V_1 → Store in cache
Step 2: Compute K_2, V_2 → Append to cache
        Retrieve K_{1:1}, V_{1:1} from cache
Step t: Compute K_t, V_t → Append to cache
        Retrieve K_{1:t-1}, V_{1:t-1} from cache

Attention(Q_t, [K_{1:t-1}; K_t], [V_{1:t-1}; V_t])
```

Where `[;]` denotes concatenation along the sequence dimension.

**Memory Trade-off**: KV caching significantly reduces computational costs by avoiding redundant calculations, but incurs substantial memory overhead that scales with:
- Sequence length (S)
- Batch size (B)
- Number of layers (L)
- Number of attention heads (H)
- Hidden dimension per head (d)

### Memory Layout in Standard Implementations

Traditional KV cache implementations use contiguous memory allocation:

```python
# Conceptual structure (not actual implementation)
class TraditionalKVCache:
    def __init__(self, num_layers, num_heads, max_seq_len, head_dim, dtype):
        self.cache = {
            'keys': torch.zeros(
                num_layers, max_seq_len, num_heads, head_dim, dtype=dtype
            ),
            'values': torch.zeros(
                num_layers, max_seq_len, num_heads, head_dim, dtype=dtype
            )
        }

    def update(self, layer_idx, new_keys, new_values, current_pos):
        # Append new keys/values to existing cache
        seq_len = new_keys.shape[0]
        self.cache['keys'][layer_idx, current_pos:current_pos+seq_len] = new_keys
        self.cache['values'][layer_idx, current_pos:current_pos+seq_len] = new_values
```

**Problems with Contiguous Allocation**:
1. **Pre-allocation waste**: Must allocate for `max_seq_len` upfront, even if actual sequence is shorter
2. **External fragmentation**: Different requests have variable sequence lengths, creating scattered gaps
3. **Inflexibility**: Cannot efficiently share KV cache across requests with common prefixes

## PagedAttention: Block-Based KV Cache Management

### Virtual Memory Inspiration

From [vLLM PagedAttention Documentation](https://docs.vllm.ai/en/v0.7.0/design/kernel/paged_attention.html) (accessed 2025-02-02):

vLLM's PagedAttention applies virtual memory principles from operating systems to manage KV cache. Instead of storing the entire KV cache in contiguous memory, it partitions the cache into fixed-size **blocks** (or **pages**).

**Core Concept**: Each KV block contains attention keys and values for a fixed number of tokens (typically 16). The PagedAttention algorithm allows these blocks to be stored in non-contiguous physical memory, eliminating memory fragmentation.

### Block Structure and Addressing

From [vLLM PagedAttention kernel implementation](https://docs.vllm.ai/en/v0.7.0/design/kernel/paged_attention.html):

**Memory Layout**:
```
K cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
V cache: [num_blocks, num_kv_heads, head_size, block_size]
```

Where:
- `num_blocks`: Total number of allocated KV blocks across all requests
- `num_kv_heads`: Number of key-value heads (may differ from query heads in GQA/MQA)
- `head_size`: Dimension of each attention head
- `block_size`: Number of tokens per block (typically 16)
- `x`: Thread group size for memory coalescing (typically 8 or 16)

**Block Addressing Example** (block_size=16, head_size=128):

```
Block 0: Tokens [0-15]   → 16 × 128 = 2048 elements per head
Block 1: Tokens [16-31]  → 16 × 128 = 2048 elements per head
Block 2: Tokens [32-47]  → 16 × 128 = 2048 elements per head
...

Physical memory layout for Keys (one head):
┌─────────────┬─────────────┬─────────────┐
│ Block 0     │ Block 1     │ Block 2     │
│ [head/x, 16, x] │ [head/x, 16, x] │ [head/x, 16, x] │
└─────────────┴─────────────┴─────────────┘
```

The `x` dimension enables memory coalescing - neighboring threads read neighboring memory locations, improving GPU memory bandwidth utilization.

### Block Table: Logical to Physical Mapping

Each sequence maintains a **block table** that maps logical KV block indices to physical block locations:

```python
# Conceptual block table structure
class BlockTable:
    def __init__(self):
        # Maps logical block index → physical block number
        self.mapping = {}  # {0: 42, 1: 17, 2: 91, ...}

    def get_physical_block(self, logical_idx):
        """Get physical block number for logical index"""
        return self.mapping[logical_idx]

    def allocate_new_block(self, physical_block_num):
        """Allocate new physical block for next logical position"""
        next_logical_idx = len(self.mapping)
        self.mapping[next_logical_idx] = physical_block_num
```

**Example Scenario**:

Request 1 (128 tokens, block_size=16):
```
Logical:  [Block 0] [Block 1] [Block 2] [Block 3] [Block 4] [Block 5] [Block 6] [Block 7]
Physical: [  42   ] [  17   ] [  91   ] [  3    ] [  55   ] [  28   ] [  64   ] [  10   ]
          ↑         ↑         ↑
          Non-contiguous physical memory!
```

Request 2 (64 tokens, block_size=16):
```
Logical:  [Block 0] [Block 1] [Block 2] [Block 3]
Physical: [  42   ] [  73   ] [  19   ] [  51   ]
          ↑
          Shared with Request 1 (same prefix)!
```

**Benefits**:
1. **Zero external fragmentation**: Blocks allocated on-demand
2. **Flexible sharing**: Multiple sequences can share physical blocks for common prefixes
3. **Efficient memory reuse**: Freed blocks immediately available for new requests

## Cache Allocation Strategies

### Dynamic Block Allocation

From [vLLM PagedAttention Documentation](https://docs.vllm.ai/en/v0.6.3.post1/automatic_prefix_caching/details.html):

vLLM uses on-demand block allocation rather than pre-allocating maximum sequence length:

```python
# Conceptual allocation flow
class BlockAllocator:
    def __init__(self, total_gpu_blocks):
        self.free_blocks = set(range(total_gpu_blocks))
        self.allocated_blocks = {}  # physical_block → refcount

    def allocate(self, num_blocks_needed):
        """Allocate requested number of blocks"""
        if len(self.free_blocks) < num_blocks_needed:
            raise OutOfMemoryError("Insufficient free blocks")

        allocated = []
        for _ in range(num_blocks_needed):
            block = self.free_blocks.pop()
            self.allocated_blocks[block] = 1  # refcount = 1
            allocated.append(block)

        return allocated

    def free(self, physical_blocks):
        """Free blocks when no longer needed"""
        for block in physical_blocks:
            refcount = self.allocated_blocks.get(block, 0)
            if refcount == 1:
                del self.allocated_blocks[block]
                self.free_blocks.add(block)
            else:
                self.allocated_blocks[block] = refcount - 1
```

**Allocation Scenarios**:

1. **New Request (no sharing)**:
   - Allocate new physical blocks as tokens are generated
   - Each block added when previous block becomes full

2. **Request with Prefix Sharing**:
   - Reuse existing physical blocks for shared prefix
   - Increment refcount on shared blocks
   - Allocate new blocks only for unique continuation

3. **Request Completion**:
   - Decrement refcount on all used blocks
   - Free blocks with refcount=0 to pool

### Prefix Caching and Automatic Sharing

From [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/v0.6.3.post1/automatic_prefix_caching/details.html):

**Key Observation**: Each KV block can be uniquely identified by the tokens within the block and the tokens in the prefix before the block.

```
Block 1: [A gentle breeze stirred]
Block 2: [the leaves as children]
Block 3: [laughed in the distance]

Block 1 identification: hash(["A", "gentle", "breeze", "stirred"])
Block 2 identification: hash(["A", "gentle", "breeze", "stirred"] +
                             ["the", "leaves", "as", "children"])
Block 3 identification: hash(["A", "gentle", "breeze", "stirred",
                             "the", "leaves", "as", "children"] +
                             ["laughed", "in", "the", "distance"])
```

**Hash-Based Mapping**:

```
hash(prefix tokens + block tokens) ↔ Physical KV Block
```

**Implementation Pattern**:

```python
class AutomaticPrefixCache:
    def __init__(self):
        # Maps content hash → physical block number
        self.hash_to_block = {}
        # Maps physical block → refcount
        self.block_refcount = {}

    def get_or_allocate_block(self, prefix_tokens, block_tokens):
        """Get existing block or allocate new one"""
        content_hash = self._compute_hash(prefix_tokens + block_tokens)

        if content_hash in self.hash_to_block:
            # Reuse existing block
            physical_block = self.hash_to_block[content_hash]
            self.block_refcount[physical_block] += 1
            return physical_block
        else:
            # Allocate new block
            physical_block = self._allocate_new_block()
            self.hash_to_block[content_hash] = physical_block
            self.block_refcount[physical_block] = 1
            return physical_block

    def _compute_hash(self, tokens):
        """Compute content-based hash for deduplication"""
        return hash(tuple(tokens))  # Simplified
```

**Automatic Sharing Benefits**:
- System prompts shared across requests (e.g., "You are a helpful assistant...")
- Few-shot examples reused for multiple queries
- RAG context shared when processing multiple questions
- No manual cache management required

From [Prefix Sharing and KV Cache Optimization](https://pub.towardsai.net/kv-cache-the-secret-weapon-making-your-llms-10x-faster-ec6953a76e77) (Towards AI, accessed 2025-02-02):

vLLM's prefix caching detects when queries share the same beginning (prefix) and reuses the cached K and V values from that shared portion. This is particularly effective for:
- **Batch serving**: Multiple users with same system prompt
- **RAG systems**: Shared context documents across queries
- **Few-shot learning**: Example demonstrations reused across requests

### Multi-LoRA and Multi-Modal Extensions

From [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/v0.6.3.post1/automatic_prefix_caching/details.html):

The hash-based KV cache management enables advanced serving scenarios:

**Multi-LoRA Serving**:
```python
# Include LoRA adapter ID in hash
hash_key = hash(prefix_tokens + block_tokens + lora_id)
```

This allows joint management of KV blocks for different LoRA adapters:
- Single global cache for all adapters
- Improves global cache hit rate
- Simplifies system implementation

**Multi-Modal Models**:
```python
# Different hashing for different modalities
def compute_hash(inputs):
    text_hash = hash(tuple(text_tokens))

    if image_input:
        # Perceptual hashing for images
        image_hash = perceptual_hash(image_input)
        return combine_hashes(text_hash, image_hash)

    return text_hash
```

Enables caching of:
- Image embeddings
- Audio representations
- Video frame features
- Cross-modal attention states

## Cache Eviction and Replacement Policies

### Eviction Policy in vLLM

From [vLLM Implementation Details](https://docs.vllm.ai/en/v0.6.3.post1/automatic_prefix_caching/details.html):

When free blocks are exhausted, vLLM implements the following eviction policy:

**Policy Components**:

1. **Reference Count Priority**:
   - Evict only blocks with refcount = 0
   - Never evict blocks actively used by running requests

2. **LRU (Least Recently Used)**:
   - Among refcount=0 blocks, evict least recently accessed
   - Tracks last access time per block

3. **Prefix Length Tie-Breaking**:
   - When multiple blocks have same last access time
   - Prioritize eviction of blocks at end of longest prefix
   - Preserves commonly shared prefixes

```python
class EvictionPolicy:
    def __init__(self):
        self.block_metadata = {}  # block_id → {refcount, last_access, prefix_length}

    def select_eviction_candidate(self):
        """Select block to evict based on policy"""
        # Step 1: Filter to refcount=0 blocks
        candidates = [
            (block_id, meta) for block_id, meta in self.block_metadata.items()
            if meta['refcount'] == 0
        ]

        if not candidates:
            raise OutOfMemoryError("No evictable blocks available")

        # Step 2: Sort by LRU
        candidates.sort(key=lambda x: x[1]['last_access'])

        # Step 3: Among equal LRU, prefer maximum prefix length
        min_access_time = candidates[0][1]['last_access']
        lru_candidates = [
            (bid, meta) for bid, meta in candidates
            if meta['last_access'] == min_access_time
        ]

        # Select block with maximum prefix length
        evict_block = max(lru_candidates, key=lambda x: x[1]['prefix_length'])

        return evict_block[0]
```

**Relationship to RadixAttention**:

From [vLLM documentation](https://docs.vllm.ai/en/v0.6.3.post1/automatic_prefix_caching/details.html):

> "This eviction policy effectively implements the exact policy as in RadixAttention when applied to models with full attention, which prioritizes to evict reference count zero and least recent used leaf nodes in the prefix tree."

### Block-Wise Eviction (PagedEviction)

From [PagedEviction: Structured Block-wise KV Cache Pruning](https://arxiv.org/html/2509.04377v1) (arXiv:2509.04377):

Traditional token-level eviction methods (H2O, StreamingLLM) face challenges in block-based systems. **PagedEviction** introduces structured block-wise eviction aligned with vLLM's architecture.

**Token Importance Metric**:

Instead of requiring attention scores (unavailable in FlashAttention), PagedEviction uses:

```
Token Importance = ||V_i||_2 / ||K_i||_2
```

**Rationale**:
- Value tensor (V) represents feature embeddings to extract
- Key L2-norm inversely correlates with attention weight (Devoto et al., 2024)
- No attention score storage required
- Computed on-demand from existing KV cache

**Prefill Phase** (token-level eviction before blocking):

```python
def prefill_eviction(K, V, cache_budget, block_size):
    """Evict tokens before dividing into blocks"""
    # Compute per-token importance
    importance = np.linalg.norm(V, axis=-1) / np.linalg.norm(K, axis=-1)

    # Evict least important tokens
    num_to_evict = len(K) - cache_budget
    evict_indices = np.argsort(importance)[:num_to_evict]

    K_retained = np.delete(K, evict_indices, axis=0)
    V_retained = np.delete(V, evict_indices, axis=0)

    # Divide into blocks
    num_blocks = len(K_retained) // block_size
    K_blocks = K_retained[:num_blocks * block_size].reshape(num_blocks, block_size, -1)
    V_blocks = V_retained[:num_blocks * block_size].reshape(num_blocks, block_size, -1)

    return K_blocks, V_blocks
```

**Decode Phase** (block-level eviction):

```python
def decode_eviction(K_blocks, V_blocks, block_size):
    """Evict one block when new block is full"""
    if current_block_full():
        # Compute block importance (mean of token importances)
        block_importance = []
        for k_block, v_block in zip(K_blocks, V_blocks):
            token_imp = np.linalg.norm(v_block, axis=-1) / np.linalg.norm(k_block, axis=-1)
            block_importance.append(np.mean(token_imp))

        # Evict block with lowest importance
        evict_idx = np.argmin(block_importance)
        K_blocks = np.delete(K_blocks, evict_idx, axis=0)
        V_blocks = np.delete(V_blocks, evict_idx, axis=0)

        # Update block table
        update_block_table(evict_idx)

    return K_blocks, V_blocks
```

**Performance Results** (from PagedEviction paper):

On LongBench with LLaMA-3.2-1B, cache budget=1024:
- **PagedEviction**: ROUGE score ~24.5
- **StreamingLLM**: ROUGE score ~21.0 (15% lower)
- **KeyDiff**: ROUGE score ~21.2 (15% lower)

Throughput improvements at cache budget=1024:
- **PagedEviction**: ~3020 tokens/sec (37% improvement over Full Cache)
- **Inverse Key L2-Norm**: ~2170 tokens/sec
- **Full Cache baseline**: ~2200 tokens/sec

### CPU Swapping and Memory Tiering

From [RFC: Enable Memory Tiering for vLLM](https://github.com/vllm-project/vllm/issues/7697) (GitHub, accessed 2025-02-02):

vLLM supports **KV cache swapping** to CPU memory when GPU memory is exhausted:

**Use Cases**:
1. Requests with multiple sequences (e.g., beam search, best_of > 1)
2. Temporary storage during context switching
3. Prefix cache overflow to CPU

**Swapping Mechanism**:

```python
class SwapManager:
    def __init__(self, gpu_blocks, cpu_blocks):
        self.gpu_allocator = BlockAllocator(gpu_blocks)
        self.cpu_allocator = BlockAllocator(cpu_blocks)
        self.swap_mapping = {}  # GPU block → CPU block

    def swap_out(self, gpu_block):
        """Move block from GPU to CPU memory"""
        cpu_block = self.cpu_allocator.allocate(1)[0]

        # DMA transfer GPU → CPU
        copy_block_async(
            src=gpu_block,
            dst=cpu_block,
            device='cuda:0 → cpu'
        )

        self.swap_mapping[gpu_block] = cpu_block
        self.gpu_allocator.free([gpu_block])

        return cpu_block

    def swap_in(self, cpu_block):
        """Move block from CPU to GPU memory"""
        gpu_block = self.gpu_allocator.allocate(1)[0]

        # DMA transfer CPU → GPU
        copy_block_async(
            src=cpu_block,
            dst=gpu_block,
            device='cpu → cuda:0'
        )

        original_gpu_block = self._find_original_gpu_block(cpu_block)
        del self.swap_mapping[original_gpu_block]
        self.cpu_allocator.free([cpu_block])

        return gpu_block
```

**When CPU Cache is Used**:

From [vLLM GitHub Issue #2853](https://github.com/vllm-project/vllm/issues/2853):

> "The CPU KV cache is only used in cases where a sequence group has multiple sequences running. An example of this would be in a generation request with beam_search enabled or best_of > 1."

**Swap Space Configuration**:

```python
# vLLM configuration
vllm_engine = LLMEngine(
    model="meta-llama/Llama-2-7b-hf",
    gpu_memory_utilization=0.9,  # Reserve 90% of GPU for KV cache
    swap_space=4,  # 4GB CPU swap space
    max_num_seqs=256  # Maximum concurrent sequences
)
```

From [vLLM Forums: Why vLLM uses a lot of CPU memory](https://discuss.vllm.ai/t/why-vllm-uses-alot-of-cpu-memory/511) (accessed 2025-02-02):

> "According to vLLM's configuration, vLLM reserves a portion of CPU memory as swap space to manage GPU memory more efficiently. This swap space serves as an overflow area."

## Prefix Sharing and Deduplication

### Hash-Based Deduplication

From [vLLM Prefix Caching Implementation](https://docs.vllm.ai/en/v0.6.3.post1/automatic_prefix_caching/details.html):

vLLM builds a one-to-one mapping between content and physical blocks:

```
hash(prefix tokens + block tokens) ↔ Physical KV Block
```

**Indirection Layer**:

```
Sequence → Logical Blocks → Content Hash → Physical Blocks
```

Traditional approach:
```
Sequence → Logical Blocks → Physical Blocks (no sharing)
```

**Implementation Pattern**:

```python
class PrefixCache:
    def __init__(self):
        self.hash_to_physical = {}  # content_hash → physical_block_id
        self.physical_refcount = {}  # physical_block_id → refcount
        self.global_block_pool = set(range(NUM_GPU_BLOCKS))

    def get_block_for_content(self, prefix_tokens, block_tokens):
        """Get or create physical block for content"""
        content_hash = self._hash_content(prefix_tokens, block_tokens)

        if content_hash in self.hash_to_physical:
            # Cache hit - reuse existing block
            physical_id = self.hash_to_physical[content_hash]
            self.physical_refcount[physical_id] += 1
            return physical_id, True  # (block_id, cache_hit)
        else:
            # Cache miss - allocate new block
            physical_id = self.global_block_pool.pop()
            self.hash_to_physical[content_hash] = physical_id
            self.physical_refcount[physical_id] = 1
            return physical_id, False  # (block_id, cache_hit)

    def release_block(self, content_hash):
        """Decrement refcount, free if zero"""
        if content_hash not in self.hash_to_physical:
            return

        physical_id = self.hash_to_physical[content_hash]
        self.physical_refcount[physical_id] -= 1

        if self.physical_refcount[physical_id] == 0:
            # No more references - free block
            del self.hash_to_physical[content_hash]
            del self.physical_refcount[physical_id]
            self.global_block_pool.add(physical_id)

    def _hash_content(self, prefix_tokens, block_tokens):
        """Compute deterministic hash of token content"""
        all_tokens = tuple(prefix_tokens) + tuple(block_tokens)
        return hash(all_tokens)
```

**Example Scenario**:

Three requests with shared prefix:

```
Request 1: "Translate to French: The cat sat on the mat."
Request 2: "Translate to French: The cat sat on the table."
Request 3: "Translate to French: The cat sat on the chair."

Shared prefix: "Translate to French: The cat sat on the"

Block layout (block_size=16):
┌───────────────────────────────────────────────────┐
│ Block 0: "Translate to French: The cat sat on the"│ ← Shared (refcount=3)
└───────────────────────────────────────────────────┘
┌──────────┐
│ "mat."   │ ← Request 1 unique (refcount=1)
└──────────┘
┌──────────┐
│ "table." │ ← Request 2 unique (refcount=1)
└──────────┘
┌──────────┐
│ "chair." │ ← Request 3 unique (refcount=1)
└──────────┘

Memory saved: 2 blocks (prefix stored once, not three times)
```

### Tree-Free Design

From [vLLM documentation](https://docs.vllm.ai/en/v0.6.3.post1/automatic_prefix_caching/details.html):

> "This design achieves automatic prefix caching without the need of maintaining a tree structure among the KV blocks. More specifically, all of the blocks are independent of each other and can be allocated and freed by itself, which enables us to manage the KV cache as ordinary caches in operating system."

**Benefits of Tree-Free Approach**:

1. **Simplicity**: No prefix tree traversal logic
2. **Flexibility**: Blocks allocated/freed independently
3. **Standard OS Cache Policies**: LRU, reference counting work naturally
4. **Lock-Free**: No tree rebalancing or restructuring

**Contrast with RadixAttention**:

RadixAttention (SGLang) uses explicit prefix tree:
```
         Root
          /|\
         / | \
    Req1 Req2 Req3
     |    |    |
   Unique tokens...
```

vLLM hash-based approach:
```
Hash Table:
  hash("prefix_A") → Block 42 (refcount=3)
  hash("prefix_A" + "unique_1") → Block 17 (refcount=1)
  hash("prefix_A" + "unique_2") → Block 91 (refcount=1)
  hash("prefix_A" + "unique_3") → Block 28 (refcount=1)
```

Both achieve similar results, but vLLM avoids explicit tree management.

### Prefix Sharing Performance

From [KV Cache: The Secret Weapon Making Your LLMs 10x Faster](https://pub.towardsai.net/kv-cache-the-secret-weapon-making-your-llms-10x-faster-ec6953a76e77) (Towards AI, accessed 2025-02-02):

**Typical Sharing Scenarios**:

1. **System Prompts** (100-500 tokens):
   - Shared across all requests in a deployment
   - Single physical copy regardless of batch size
   - Example: ChatGPT's "You are a helpful assistant..." preamble

2. **Few-Shot Examples** (500-2000 tokens):
   - Reused for all queries in same task
   - Example: 5-shot translation examples

3. **RAG Context** (2000-8000 tokens):
   - Shared document context for multiple questions
   - Example: Answering 10 questions about same paper

**Memory Savings Calculation**:

```
Without sharing (100 concurrent requests, 500-token system prompt):
  100 requests × 500 tokens × KV_size = 50,000 token-slots

With sharing (same scenario):
  1 shared copy × 500 tokens × KV_size = 500 token-slots

Savings: 99% reduction in system prompt memory
```

## Implementation Deep Dive: vLLM PagedAttention Kernel

### CUDA Kernel Structure

From [vLLM Paged Attention Kernel](https://docs.vllm.ai/en/v0.7.0/design/kernel/paged_attention.html) (accessed 2025-02-02):

vLLM implements a custom CUDA kernel (`csrc/attention/attention_kernels.cu`) compatible with paged KV caches.

**Template Parameters**:

```cpp
template<
    typename scalar_t,     // Data type (FP16, FP32, etc.)
    int HEAD_SIZE,         // Elements per attention head
    int BLOCK_SIZE,        // Tokens per KV block (typically 16)
    int NUM_THREADS,       // Threads per thread block
    int PARTITION_SIZE = 0 // Tensor parallel size (0 = disabled)
>
__device__ void paged_attention_kernel(
    const scalar_t* __restrict__ out,       // [num_seqs, num_heads, max_partitions, head_size]
    const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,   // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const scalar_t* __restrict__ v_cache,   // [num_blocks, num_kv_heads, head_size, block_size]
    const int* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens    // [num_seqs]
)
```

**Key Concepts**:

1. **Thread Group**: Small group of threads (THREAD_GROUP_SIZE, typically 2) that process one query token and one key token together

2. **Vec**: Group of elements fetched together for memory coalescing
   - `VEC_SIZE` for Q/K: Determined so thread group fetches 16 bytes at once
   - `V_VEC_SIZE` for V: Determined so single thread fetches 16 bytes at once

3. **Warp**: 32 threads that process one query token against all key tokens in one block

4. **Thread Block**: Group of warps (NUM_THREADS / 32) processing one full context

### Query Processing

From [PagedAttention kernel documentation](https://docs.vllm.ai/en/v0.7.0/design/kernel/paged_attention.html):

**Query Data Layout**:

```cpp
// Query pointer for current sequence and head
const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;

// Shared memory for query vectors
__shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];

// Load query data (coalesced access)
// Thread 0 loads vec 0, Thread 1 loads vec 1, etc.
for (int i = 0; i < NUM_VECS_PER_THREAD; i++) {
    int vec_idx = thread_group_id * NUM_VECS_PER_THREAD + i;
    q_vecs[thread_group_offset][i] = *reinterpret_cast<const Q_vec*>(
        q_ptr + vec_idx * VEC_SIZE
    );
}
```

**Memory Access Pattern**:

```
HEAD_SIZE = 128, VEC_SIZE = 4, THREAD_GROUP_SIZE = 2

Query tensor (128 elements):
┌─────┬─────┬─────┬─────┬─────┬─────┬ ─ ─ ┬─────┬─────┐
│ v0  │ v1  │ v2  │ v3  │ v4  │ v5  │ ... │ v30 │ v31 │  (32 vecs total)
└─────┴─────┴─────┴─────┴─────┴─────┴ ─ ─ ┴─────┴─────┘
  ↑     ↑
Thread 0│
  Thread 1

Shared memory layout:
q_vecs[0][...] ← Thread 0's vecs (even indices: 0, 2, 4, ...)
q_vecs[1][...] ← Thread 1's vecs (odd indices: 1, 3, 5, ...)
```

### Key Processing

**Key Pointer Addressing**:

```cpp
// Key pointer includes block number, head, and token offset
const scalar_t* k_ptr = k_cache
    + physical_block_number * kv_block_stride
    + kv_head_idx * kv_head_stride
    + physical_block_offset * x;

// Register memory for key vectors (not shared across threads)
K_vec k_vecs[NUM_VECS_PER_THREAD];

// Inner loop: load key vecs for current token
for (int i = 0; i < NUM_VECS_PER_THREAD; i++) {
    k_vecs[i] = *reinterpret_cast<const K_vec*>(
        k_ptr + i * VEC_SIZE
    );
}
```

**Block Iteration**:

```
Warp 0 processes blocks: 0, 4, 8, 12, ...
Warp 1 processes blocks: 1, 5, 9, 13, ...
Warp 2 processes blocks: 2, 6, 10, 14, ...
Warp 3 processes blocks: 3, 7, 11, 15, ...

For 6 total blocks with 4 warps:
Warp 0: Block 0, Block 4
Warp 1: Block 1, Block 5
Warp 2: Block 2
Warp 3: Block 3
```

### QK Computation and Softmax

**Dot Product with Reduction**:

```cpp
// Compute QK for one token (with cross-thread-group reduction)
float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(
    q_vecs[thread_group_offset],
    k_vecs
);

// Qk_dot performs:
// 1. Element-wise multiply Q and K vectors
// 2. Sum within thread
// 3. Reduce across thread group
// Result: full dot product Q · K
```

**Softmax Computation**:

```cpp
// Step 1: Compute qk_max across all tokens
float qk_max = -FLT_MAX;
for (each token) {
    qk_max = fmaxf(qk_max, qk);
}

// Reduce qk_max across warp
for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
}

// Reduce across warps via shared memory
__shared__ float red_smem[NUM_WARPS];
if (lane == 0) {
    red_smem[warp_idx] = qk_max;
}
__syncthreads();

// Final reduction
for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
}

// Step 2: Compute exp_sum and normalize
float exp_sum = 0.0f;
for (int i = 0; i < num_tokens; i++) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
}

// Reduce exp_sum across thread block
exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

// Normalize to get final softmax
const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
for (int i = 0; i < num_tokens; i++) {
    logits[i] *= inv_sum;
}
```

### Value Processing and Output

**Value Memory Layout**:

```cpp
// Value cache layout: [num_blocks, num_kv_heads, head_size, block_size]
// Different from Key: tokens are columns, head dims are rows

const scalar_t* v_ptr = v_cache
    + physical_block_number * kv_block_stride
    + kv_head_idx * kv_head_stride;

// Load V_VEC_SIZE elements from same tokens
V_vec v_vec = *reinterpret_cast<const V_vec*>(
    v_ptr + row_idx * BLOCK_SIZE + token_offset
);
```

**Value Accumulation**:

```cpp
// Register memory for accumulators (one per head position assigned to thread)
float accs[NUM_ROWS_PER_THREAD];

// Outer loop: iterate blocks
for (each block) {
    // Load logits_vec (V_VEC_SIZE softmax weights)
    V_vec logits_vec = ...;

    // Inner loop: iterate rows (head positions)
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        // Load V_VEC_SIZE value elements
        V_vec v_vec = ...;

        // Accumulate: accs[i] += dot(logits_vec, v_vec)
        accs[i] += dot(logits_vec, v_vec);
    }
}

// Reduce across warps
for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];

    // Intra-warp reduction
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
        acc += VLLM_SHFL_XOR_SYNC(acc, mask);
    }

    accs[i] = acc;
}

// Inter-warp reduction via shared memory
for (int num_warps = NUM_WARPS; num_warps > 1; num_warps /= 2) {
    // Upper warps write to shared memory
    // Lower warps read and accumulate
}
```

**Output Writing**:

```cpp
// Output pointer for current sequence and head
scalar_t* out_ptr = out
    + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
    + head_idx * max_num_partitions * HEAD_SIZE
    + partition_idx * HEAD_SIZE;

// Write accumulated results
for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
    if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
    }
}
```

## Performance Characteristics

### Memory Efficiency

From [PagedEviction paper](https://arxiv.org/html/2509.04377v1):

**Memory Fragmentation Reduction**:

Traditional contiguous allocation:
```
Request 1 (max_len=2048, actual=512):  Waste = 1536 tokens
Request 2 (max_len=2048, actual=1024): Waste = 1024 tokens
Request 3 (max_len=2048, actual=256):  Waste = 1792 tokens

Total allocated: 6144 tokens
Total used: 1792 tokens
Efficiency: 29.2%
```

PagedAttention (block_size=16):
```
Request 1 (512 tokens): 32 blocks allocated
Request 2 (1024 tokens): 64 blocks allocated
Request 3 (256 tokens): 16 blocks allocated

Total allocated: 1792 tokens
Total used: 1792 tokens
Efficiency: 100%
```

**Sharing Benefits**:

With prefix sharing (500-token common prefix, 3 requests):

Without sharing:
```
3 requests × 500 prefix tokens = 1500 tokens allocated
```

With sharing:
```
1 shared copy × 500 tokens = 500 tokens allocated
Savings: 1000 tokens (67% reduction)
```

### Throughput and Latency

From [PagedEviction paper results](https://arxiv.org/html/2509.04377v1):

**Throughput Improvements** (LLaMA-3.2-1B, cache_budget=1024):

| Method | Tokens/sec | vs Full Cache |
|--------|-----------|---------------|
| PagedEviction | 3,020 | +37% |
| StreamingLLM | 2,920 | +33% |
| Full Cache | 2,200 | baseline |
| Inverse Key L2 | 2,170 | -1.4% |

**Latency Reduction** (Time per Output Token):

| Model | Full Cache | PagedEviction | Reduction |
|-------|-----------|---------------|-----------|
| LLaMA-1B | 45.5ms | 40.0ms | 12% |
| LLaMA-3B | 68.2ms | 61.4ms | 10% |
| LLaMA-8B | 136.4ms | 121.3ms | 11% |

**Accuracy vs Compression** (LongBench GovReport, LLaMA-3.2-1B):

| Cache Budget | Full Cache ROUGE | PagedEviction ROUGE | StreamingLLM ROUGE |
|--------------|------------------|---------------------|-------------------|
| 256 | 30.0 | 18.5 | 16.2 |
| 512 | 30.0 | 21.8 | 18.9 |
| 1024 | 30.0 | 24.5 | 21.0 |
| 2048 | 30.0 | 27.3 | 25.1 |
| 4096 | 30.0 | 29.5 | 28.8 |

### Memory Bandwidth Optimization

From [vLLM kernel implementation](https://docs.vllm.ai/en/v0.7.0/design/kernel/paged_attention.html):

**Memory Coalescing**:

Query loading (coalesced):
```
Thread 0: Load vec[0] from address base + 0×16
Thread 1: Load vec[1] from address base + 1×16
Thread 2: Load vec[2] from address base + 2×16
...

All threads in warp access consecutive 16-byte chunks
→ Single coalesced memory transaction
```

Key loading (coalesced across x dimension):
```
K cache layout: [..., block_size, x]

Thread 0: Load from token_0, element_0
Thread 1: Load from token_0, element_1
...
Thread 7: Load from token_0, element_7
Thread 8: Load from token_1, element_0
...

Within each token, consecutive threads access consecutive elements
→ Coalesced memory access pattern
```

Value loading (coalesced across tokens):
```
V cache layout: [..., head_size, block_size]

Single thread loads V_VEC_SIZE elements from same row:
  [token_0_val, token_1_val, ..., token_7_val]

Consecutive elements in memory → vectorized load
```

**Shared Memory Usage**:

```cpp
// Query vectors: reused across all key tokens
__shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];

// Reduction scratch space
__shared__ float red_smem[NUM_WARPS];

// Output accumulation (for multi-warp reduction)
__shared__ float out_smem[NUM_WARPS * HEAD_SIZE];

// Logits (softmax weights)
__shared__ float logits[MAX_NUM_TOKENS];
```

Shared memory reduces global memory traffic by:
1. Loading query once per warp instead of per token
2. Storing intermediate softmax results
3. Coordinating reductions across warps

## Advanced Topics

### Context Switching and Priority Management

From [arXiv:2411.18424 - Context Switching in LLM Serving](https://arxiv.org/html/2411.18424v1) (accessed 2025-02-02):

**Context Switching Definition**:
> "In our work, context switching refers to requests and their KV cache being swapped between GPU and CPU memory based on their priorities."

**Priority-Based Swapping**:

```python
class PrioritySwapManager:
    def __init__(self):
        self.request_priorities = {}  # request_id → priority_score
        self.gpu_blocks = {}  # request_id → [gpu_block_ids]
        self.cpu_blocks = {}  # request_id → [cpu_block_ids]

    def prioritize_requests(self):
        """Compute priority scores for active requests"""
        for request_id in self.active_requests:
            # Priority factors:
            # - SLO deadline proximity
            # - Request age
            # - User tier (premium vs free)
            # - Partial completion

            priority = self._compute_priority(request_id)
            self.request_priorities[request_id] = priority

    def evict_low_priority(self, blocks_needed):
        """Swap out lowest priority requests to free GPU blocks"""
        # Sort requests by priority
        sorted_requests = sorted(
            self.request_priorities.items(),
            key=lambda x: x[1]  # priority score
        )

        freed_blocks = 0
        for request_id, priority in sorted_requests:
            if freed_blocks >= blocks_needed:
                break

            # Swap request to CPU
            gpu_blocks = self.gpu_blocks[request_id]
            cpu_blocks = self.swap_out_request(request_id, gpu_blocks)

            del self.gpu_blocks[request_id]
            self.cpu_blocks[request_id] = cpu_blocks
            freed_blocks += len(gpu_blocks)

        return freed_blocks

    def _compute_priority(self, request_id):
        """Compute multi-factor priority score"""
        request = self.get_request(request_id)

        # Higher score = higher priority (keep in GPU)
        priority = 0.0

        # SLO deadline urgency
        time_to_deadline = request.slo_deadline - current_time()
        priority += 100.0 / max(time_to_deadline, 1.0)

        # Completion percentage (avoid wasting partial work)
        completion = request.tokens_generated / request.target_tokens
        priority += 50.0 * completion

        # User tier
        if request.user_tier == "premium":
            priority += 200.0

        return priority
```

**Swap Granularity**:

Block-level swapping provides better control than request-level:

```python
def partial_swap(request_id, blocks_to_keep):
    """Swap out older blocks while keeping recent ones in GPU"""
    all_blocks = self.gpu_blocks[request_id]

    # Keep most recent blocks in GPU
    keep_blocks = all_blocks[-blocks_to_keep:]
    swap_blocks = all_blocks[:-blocks_to_keep]

    # Swap older blocks to CPU
    cpu_blocks = self.swap_out_blocks(swap_blocks)

    self.gpu_blocks[request_id] = keep_blocks
    self.cpu_blocks[request_id] = cpu_blocks

    return len(swap_blocks)  # freed GPU blocks
```

### KV Cache Quantization

From [A Survey on LLM Acceleration based on KV Cache Management](https://arxiv.org/html/2412.19442v3) (accessed 2025-02-02):

**Quantization Techniques**:

1. **Uniform Quantization** (INT8, INT4, FP8):
   ```python
   def quantize_kv_cache(K, V, bits=8):
       """Uniform quantization to lower precision"""
       K_min, K_max = K.min(), K.max()
       V_min, V_max = V.min(), V.max()

       # Compute scale and zero point
       K_scale = (K_max - K_min) / (2**bits - 1)
       V_scale = (V_max - V_min) / (2**bits - 1)

       K_zero = K_min
       V_zero = V_min

       # Quantize
       K_quant = ((K - K_zero) / K_scale).round().clamp(0, 2**bits - 1)
       V_quant = ((V - V_zero) / V_scale).round().clamp(0, 2**bits - 1)

       return K_quant, V_quant, (K_scale, K_zero), (V_scale, V_zero)

   def dequantize_kv_cache(K_quant, V_quant, K_params, V_params):
       """Restore to original precision for computation"""
       K_scale, K_zero = K_params
       V_scale, V_zero = V_params

       K = K_quant.float() * K_scale + K_zero
       V = V_quant.float() * V_scale + V_zero

       return K, V
   ```

2. **Mixed-Precision** (recent tokens full precision, old tokens quantized):
   ```python
   def mixed_precision_cache(K, V, recent_window=128):
       """Keep recent tokens in FP16, quantize older ones"""
       K_recent = K[-recent_window:]  # FP16
       K_old = K[:-recent_window]

       V_recent = V[-recent_window:]  # FP16
       V_old = V[:-recent_window]

       # Quantize old tokens to INT8
       K_old_quant, _, K_params, _ = quantize_kv_cache(K_old, V_old, bits=8)

       return {
           'K_recent': K_recent,
           'K_old': K_old_quant,
           'K_params': K_params,
           'V_recent': V_recent,
           'V_old': V_old_quant,
           'V_params': V_params
       }
   ```

**Memory Savings**:

| Precision | Bytes/element | vs FP16 | Quality |
|-----------|---------------|---------|---------|
| FP16 | 2 | 1.0× | Baseline |
| FP8 (E4M3) | 1 | 2.0× | ~99% quality |
| INT8 | 1 | 2.0× | ~97% quality |
| INT4 | 0.5 | 4.0× | ~90% quality |

### KV Cache Merging and Compression

From [Survey on KV Cache Optimization](https://arxiv.org/html/2412.19442v3):

**Token Merging Based on Similarity**:

```python
def merge_similar_tokens(K, V, similarity_threshold=0.95):
    """Merge tokens with high cosine similarity"""
    merged_K, merged_V = [], []
    skip_indices = set()

    for i in range(len(K)):
        if i in skip_indices:
            continue

        # Find similar subsequent tokens
        similar_group = [i]
        for j in range(i+1, min(i+16, len(K))):  # Look ahead window
            if j in skip_indices:
                continue

            # Compute cosine similarity
            sim = cosine_similarity(K[i], K[j])
            if sim > similarity_threshold:
                similar_group.append(j)
                skip_indices.add(j)

        # Merge group
        merged_K.append(K[similar_group].mean(dim=0))
        merged_V.append(V[similar_group].mean(dim=0))

    return torch.stack(merged_K), torch.stack(merged_V)

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    return (a @ b) / (torch.norm(a) * torch.norm(b) + 1e-8)
```

**Layer-wise Merging** (MiniCache approach):

```python
def layer_wise_merge(kv_caches, merge_layers=[8, 16, 24]):
    """Merge KV cache of specific layers into one representation"""
    merged_caches = {}

    for layer_idx in range(len(kv_caches)):
        if layer_idx in merge_layers:
            # Merge with adjacent layer
            next_layer = layer_idx + 1
            if next_layer < len(kv_caches):
                # Average keys and values
                K_merged = (kv_caches[layer_idx]['K'] +
                           kv_caches[next_layer]['K']) / 2
                V_merged = (kv_caches[layer_idx]['V'] +
                           kv_caches[next_layer]['V']) / 2

                merged_caches[layer_idx] = {'K': K_merged, 'V': V_merged}
                merged_caches[next_layer] = merged_caches[layer_idx]  # Share
            else:
                merged_caches[layer_idx] = kv_caches[layer_idx]
        else:
            merged_caches[layer_idx] = kv_caches[layer_idx]

    return merged_caches
```

## Best Practices and Recommendations

### Configuration Tuning

**Block Size Selection**:

From [vLLM documentation](https://docs.vllm.ai/en/v0.6.3.post1/automatic_prefix_caching/details.html):

Recommended block size: **16 tokens**

```python
# vLLM configuration
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    block_size=16,  # Optimal for most scenarios
    gpu_memory_utilization=0.9,
    enable_prefix_caching=True
)
```

Trade-offs:
- **Smaller blocks (8)**: More granular sharing, higher metadata overhead
- **Larger blocks (32)**: Less metadata, coarser sharing granularity
- **Sweet spot (16)**: Balances sharing efficiency and overhead

**Memory Allocation**:

```python
# Reserve 90% GPU memory for KV cache, 10% for model weights/activations
gpu_memory_utilization = 0.9

# Configure swap space for CPU offloading
swap_space = 4  # GB of CPU memory for swapped blocks

# Maximum concurrent sequences
max_num_seqs = 256  # Adjust based on average sequence length
```

### Monitoring and Debugging

**Cache Hit Rate Tracking**:

```python
class CacheMetrics:
    def __init__(self):
        self.total_blocks_requested = 0
        self.cache_hits = 0

    def record_request(self, was_cache_hit):
        self.total_blocks_requested += 1
        if was_cache_hit:
            self.cache_hits += 1

    def hit_rate(self):
        if self.total_blocks_requested == 0:
            return 0.0
        return self.cache_hits / self.total_blocks_requested

# Monitor in production
metrics = CacheMetrics()

# Log cache performance
print(f"KV Cache Hit Rate: {metrics.hit_rate():.2%}")
print(f"Total Blocks Requested: {metrics.total_blocks_requested}")
print(f"Cache Hits: {metrics.cache_hits}")
```

**Memory Utilization**:

```python
def log_memory_stats(block_allocator):
    """Log current memory utilization"""
    total_blocks = block_allocator.total_blocks
    free_blocks = len(block_allocator.free_blocks)
    allocated_blocks = len(block_allocator.allocated_blocks)

    utilization = allocated_blocks / total_blocks

    print(f"KV Cache Memory:")
    print(f"  Total Blocks: {total_blocks}")
    print(f"  Free Blocks: {free_blocks}")
    print(f"  Allocated Blocks: {allocated_blocks}")
    print(f"  Utilization: {utilization:.1%}")

    # Warn if near capacity
    if utilization > 0.95:
        print(f"  WARNING: Memory utilization above 95%")
```

### Common Pitfalls

1. **Insufficient Swap Space**:
   - Symptom: OOM errors with beam search or high best_of values
   - Solution: Increase `swap_space` parameter

2. **Prefix Sharing Not Activating**:
   - Symptom: Poor cache hit rate despite common prefixes
   - Cause: Token-level differences (whitespace, tokenization)
   - Solution: Normalize prompts before tokenization

3. **Block Size Mismatch**:
   - Symptom: Inefficient memory usage or poor sharing
   - Solution: Experiment with block_size values (8, 16, 32)

4. **Memory Fragmentation Despite PagedAttention**:
   - Symptom: Cannot allocate blocks despite free memory
   - Cause: External fragmentation in physical memory allocator
   - Solution: Periodic garbage collection or defragmentation

## Sources

**Official Documentation:**
- [vLLM Automatic Prefix Caching Details](https://docs.vllm.ai/en/v0.6.3.post1/automatic_prefix_caching/details.html) - Implementation details and eviction policies (accessed 2025-02-02)
- [vLLM Paged Attention Kernel](https://docs.vllm.ai/en/v0.7.0/design/kernel/paged_attention.html) - CUDA kernel implementation deep dive (accessed 2025-02-02)

**Research Papers:**
- [PagedEviction: Structured Block-wise KV Cache Pruning](https://arxiv.org/html/2509.04377v1) - arXiv:2509.04377 (accessed 2025-02-02)
- [A Survey on LLM Acceleration based on KV Cache Management](https://arxiv.org/html/2412.19442v3) - arXiv:2412.19442 (accessed 2025-02-02)
- [Context Switching in LLM Serving](https://arxiv.org/html/2411.18424v1) - arXiv:2411.18424 (accessed 2025-02-02)

**GitHub Issues:**
- [RFC: Enable Memory Tiering for vLLM](https://github.com/vllm-project/vllm/issues/7697) - CPU swapping discussion (accessed 2025-02-02)
- [When is CPU KV cache used and swapping?](https://github.com/vllm-project/vllm/issues/2853) - Swap space usage (accessed 2025-02-02)

**Technical Articles:**
- [KV Cache: The Secret Weapon Making Your LLMs 10x Faster](https://pub.towardsai.net/kv-cache-the-secret-weapon-making-your-llms-10x-faster-ec6953a76e77) - Towards AI (accessed 2025-02-02)
- [KV Cache: The Key to Efficient LLM Inference](https://pub.towardsai.net/kv-cache-the-key-to-efficient-llm-inference-7260a504efed) - Towards AI (accessed 2025-02-02)
- [Why vLLM uses a lot of CPU memory](https://discuss.vllm.ai/t/why-vllm-uses-alot-of-cpu-memory/511) - vLLM Forums (accessed 2025-02-02)

**Additional References:**
- [LLM Inference Series: KV Caching, A Deeper Look](https://medium.com/@plienhar/llm-inference-series-4-kv-caching-a-deeper-look-4ba9a77746c8) - Medium (accessed 2025-02-02)
- [KV Caching in LLM Inference: A Comprehensive Review](https://www.rohan-paul.com/p/kv-caching-in-llm-inference-a-comprehensive) - Rohan's Bytes (accessed 2025-02-02)
- [Transformers Key-Value Caching Explained](https://neptune.ai/blog/transformers-key-value-caching) - Neptune.ai (accessed 2025-02-02)
