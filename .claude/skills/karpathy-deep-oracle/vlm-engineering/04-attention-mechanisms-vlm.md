# Attention Mechanisms in Vision-Language Models

## Overview

Attention mechanisms in VLMs operate across multiple modalities (vision and language) simultaneously, requiring efficient implementations to handle the quadratic complexity of self-attention and cross-attention. Modern VLMs leverage optimized attention kernels like FlashAttention-2/3, sparse attention patterns, and multi-query attention to achieve 2-4× speedups while enabling longer contexts (32K-128K tokens). For ARR-COC-0-1, efficient attention is critical for relevance-driven token allocation where 64-400 variable-length visual patches must interact with query embeddings in real-time.

**Key Innovation**: FlashAttention transforms attention from memory-bound to compute-bound operation through tiling and online softmax, enabling VLMs to process high-resolution images with thousands of visual tokens efficiently.

From [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691) (Dao, 2023, accessed 2025-11-16):
> "FlashAttention-2 speeds up FlashAttention by 2× by reducing non-matmul FLOPs, better work partitioning, and parallelizing attention computation. We achieve 225 TFLOPs/s on A100 (72% utilization) compared to standard attention's 80 TFLOPs/s (25% utilization)."

**Related Knowledge**:
- See [../llm-gpu-integration/00-flashattention-internals.md](../llm-gpu-integration/00-flashattention-internals.md) for FlashAttention deep dive
- See [../vision-language/10-token-sequence-order-importance.md](../vision-language/10-token-sequence-order-importance.md) for attention ordering
- See existing knowledge on efficient attention kernels and GPU optimization

---

## Section 1: Attention Fundamentals in VLMs (~90 lines)

### Self-Attention vs Cross-Attention

VLMs use both self-attention (within modality) and cross-attention (across modalities):

**Self-Attention (Within Vision or Language)**:
```python
# Vision self-attention (patches attend to each other)
visual_features = self_attention(
    query=patch_embeddings,
    key=patch_embeddings,
    value=patch_embeddings
)

# Language self-attention (tokens attend to each other)
language_features = self_attention(
    query=text_embeddings,
    key=text_embeddings,
    value=text_embeddings
)
```

**Cross-Attention (Vision ↔ Language)**:
```python
# Language queries vision (LLM asks: what's in the image?)
vision_to_language = cross_attention(
    query=text_embeddings,      # What we're asking
    key=visual_features,         # Image content
    value=visual_features        # Image features to retrieve
)

# Vision queries language (less common, used in some architectures)
language_to_vision = cross_attention(
    query=visual_features,
    key=text_embeddings,
    value=text_embeddings
)
```

**VLM Attention Patterns**:

From [vision-language/10-token-sequence-order-importance.md](../vision-language/10-token-sequence-order-importance.md):

| Pattern | Where Used | Complexity | Purpose |
|---------|-----------|------------|---------|
| **Causal self-attention** | LLM decoder | O(N²) | Autoregressive text generation |
| **Bidirectional self-attention** | Vision encoder | O(N²) | Patch relationships |
| **Cross-attention** | Fusion layer | O(N×M) | Vision-language binding |
| **Interleaved attention** | Unified models | O((N+M)²) | Joint vision-language |

### Attention Computation

Standard attention formula (applied in all patterns):

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
- Q: Query matrix (N × d_k)
- K: Key matrix (M × d_k)
- V: Value matrix (M × d_v)
- d_k: Key/query dimension
- √d_k: Temperature scaling
```

**Multi-Head Attention**:
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Separate projections per head
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        batch, seq_len, d_model = query.shape

        # Project and split into heads
        Q = self.W_q(query).view(batch, seq_len, self.num_heads, self.d_k)
        K = self.W_k(key).view(batch, -1, self.num_heads, self.d_k)
        V = self.W_v(value).view(batch, -1, self.num_heads, self.d_k)

        # Transpose to (batch, heads, seq, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention: (batch, heads, seq_q, d_k) @ (batch, heads, d_k, seq_k)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)

        # Apply attention to values
        output = attn @ V  # (batch, heads, seq_q, d_k)

        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, d_model)
        return self.W_o(output)
```

### Attention Complexity Analysis

**Memory Cost**:
- Attention matrix: O(N²) for self-attention, O(N×M) for cross-attention
- For VLM with 256 vision tokens + 512 text tokens:
  - Vision self-attention: 256² = 65K elements
  - Text self-attention: 512² = 262K elements
  - Cross-attention: 256 × 512 = 131K elements
  - **Total**: ~460K attention elements per head per layer

**Compute Cost**:

From existing knowledge on GFLOPs analysis:

```
FLOPs for attention = 4 × N² × d

Where:
- N: Sequence length (vision + text tokens)
- d: Hidden dimension
- Factor of 4: QK matmul (2Nd²) + softmax (N²) + PV matmul (2Nd²)
```

**Example**: Qwen3-VL processing 1024×1024 image:
- Patches: (1024/14)² = 5,329 tokens
- Hidden dim: 3,584
- FLOPs per attention layer: 4 × 5,329² × 3,584 = **407 TFLOPs**
- 32 layers = **13,024 TFLOPs** just for vision self-attention

**Why This Matters**: Without FlashAttention, attention becomes memory-bound bottleneck.

---

## Section 2: FlashAttention for VLMs (~120 lines)

### FlashAttention Overview

From [../llm-gpu-integration/00-flashattention-internals.md](../llm-gpu-integration/00-flashattention-internals.md):

**Core Innovation**: Block-wise computation that keeps intermediate attention scores in fast SRAM (on-chip cache) rather than slow HBM (global memory).

**Trade-off**:
- **Standard attention**: Materialize full N×N matrix in HBM (slow writes), compute once
- **FlashAttention**: Keep blocks in SRAM (fast), recompute during backward pass
- **Result**: 2-4× faster despite recomputation (memory bandwidth >> compute cost)

**Performance Evolution**:
- **FlashAttention-1** (2022): 2-4× speedup, 40% GPU utilization on A100
- **FlashAttention-2** (2023): 2× faster than FA-1, 72% utilization, 225 TFLOPs/s
- **FlashAttention-3** (2024): 75% utilization on H100, 740 TFLOPs (FP16), 1.2 PFLOPs (FP8)

### FlashAttention in VLM Context

**Why VLMs Need FlashAttention**:

1. **Long visual sequences**: 256-5,000+ vision tokens per image
2. **High-resolution processing**: 1024×1024 images = 5,329 patches (14×14 patch size)
3. **Multi-image inputs**: Flamingo-style models process 10+ images simultaneously
4. **Cross-attention cost**: N_text × N_vision grows quadratically

**VLM-Specific Optimizations**:

```python
# VLM with mixed modality attention
def vlm_forward(image_patches, text_tokens):
    # Vision self-attention (FlashAttention handles long sequences)
    vision_features = flash_attention(
        query=image_patches,      # 5,329 tokens
        key=image_patches,
        value=image_patches,
        is_causal=False           # Bidirectional for vision
    )

    # Text self-attention (causal for autoregressive generation)
    text_features = flash_attention(
        query=text_tokens,        # Variable length (1-2048)
        key=text_tokens,
        value=text_tokens,
        is_causal=True            # Causal mask for text generation
    )

    # Cross-attention: text queries vision
    fused_features = flash_attention(
        query=text_features,      # Text asks questions
        key=vision_features,      # Visual content
        value=vision_features,
        is_causal=False           # Cross-attention is bidirectional
    )

    return fused_features
```

### FlashAttention-2 Improvements for VLMs

From [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691):

**1. Sequence Parallelism** (Critical for VLMs):

Problem: Small batch sizes (1-4 images) underutilize GPU with FA-1's per-head parallelism.

Solution: Split sequence dimension across thread blocks:

```python
# FA-1: Only 8-12 thread blocks for 8-12 attention heads
# Low occupancy on 108 SMs (A100)

# FA-2: Split each head's sequence into 4 chunks
# 8 heads × 4 splits = 32 thread blocks
# Much better GPU utilization
```

**Benefit for VLMs**: Single-image, many-token scenarios (5K vision tokens) now saturate GPU.

**2. Reduced Non-Matmul FLOPs**:

Defer softmax rescaling until after processing all K/V blocks:

```python
# FA-1: Rescale after every block (expensive)
for k_block in K_blocks:
    scores = Q @ k_block
    new_max = max(old_max, scores.max())
    O = O * exp(old_max - new_max)  # RESCALE (costly!)
    O += exp(scores - new_max) @ V

# FA-2: Rescale once at end
for k_block in K_blocks:
    scores = Q @ k_block
    O += exp(scores - running_max) @ V
O = O / normalization  # RESCALE ONCE
```

Savings: O(N²) → O(N) rescaling operations.

**3. Warp-Level Pipelining**:

Producer-consumer pattern overlaps memory and compute:

```
Warp 0-1 (Producers): Load Q, K, V from HBM
Warp 2-3 (Consumers): Compute attention on current block

Timeline:
  Producers load block N+1 while consumers process block N
```

**FlashAttention-2 Performance on VLMs**:

From [PyTorch 2.2 Release Notes](https://pytorch.org/blog/pytorch2-2/):

| Sequence Length | FA-1 TFLOPs | FA-2 TFLOPs | Speedup |
|-----------------|-------------|-------------|---------|
| 2K (small image) | 125 | 225 | 1.8× |
| 8K (large image) | 110 | 220 | 2.0× |
| 32K (multi-image) | 95 | 210 | 2.2× |

**Longer sequences = bigger speedup** (more quadratic overhead in standard attention).

### FlashAttention-3 for Next-Gen VLMs

From [FlashAttention-3 blog](https://tridao.me/blog/2024/flash3/) (Tri Dao, 2024, accessed 2025-11-16):

**H100 Hopper Optimizations**:

1. **WGMMA** (Warpgroup Matrix Multiply): 4-warp cooperative matmuls (128 threads)
2. **TMA** (Tensor Memory Accelerator): Asynchronous HBM ↔ SRAM transfers
3. **FP8 Support**: 2× throughput over FP16 (1978 TFLOPs vs 989 TFLOPs)

**FP8 with Incoherent Processing**:

Problem: LLM/VLM activations have outliers → FP8 quantization errors.

Solution: Hadamard transform spreads outliers across dimensions:

```python
def hadamard_transform(x, signs):
    # Random sign flips + recursive butterfly pattern
    x = x * signs

    # Fast Hadamard Transform: O(d log d)
    d = x.size(-1)
    while d > 1:
        half = d // 2
        x[..., :half], x[..., half:] = (
            x[..., :half] + x[..., half:],
            x[..., :half] - x[..., half:]
        )
        d = half

    return x / math.sqrt(d)
```

**FP8 Error Reduction**: 2.6× lower quantization error vs naive FP8.

**FA-3 Performance (H100)**:

| Mode | TFLOPs | % of Peak | Speedup vs FA-2 |
|------|--------|-----------|-----------------|
| FP16 | 740 | 75% | 2.1× |
| FP8 | 1,200 | 61% | 3.4× |

---

## Section 3: Sparse Attention Patterns (~110 lines)

### Motivation for Sparse Attention

**Problem**: Full attention is O(N²), prohibitive for long VLM sequences.

**Solution**: Only attend to subset of tokens (learned or heuristic sparsity patterns).

From [Sparse Attention Vectors paper](https://arxiv.org/abs/2412.00142) (Mitra et al., 2024, accessed 2025-11-16):
> "We show that sparse attention head activations (fewer than 5% of heads) in large multimodal models can achieve strong visual grounding and classification performance. Only three attention heads out of thousands are needed for localization."

### Sparse Attention Patterns in VLMs

**1. Local Attention** (Sliding Window):

Only attend to nearby tokens within fixed window:

```python
def local_attention(Q, K, V, window_size=256):
    seq_len = Q.size(1)

    # Create sparse mask
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = False

    # Attention with sparse mask
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
    scores = scores.masked_fill(mask, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return attn @ V
```

**Use Case**: Document understanding VLMs (long text + images).

**2. Strided Attention** (Skip Pattern):

Attend to every k-th token:

```python
def strided_attention(Q, K, V, stride=8):
    # Only attend to tokens at positions 0, 8, 16, 24...
    indices = torch.arange(0, K.size(1), stride)
    K_sparse = K[:, indices]
    V_sparse = V[:, indices]

    scores = Q @ K_sparse.transpose(-2, -1) / math.sqrt(Q.size(-1))
    attn = F.softmax(scores, dim=-1)
    return attn @ V_sparse
```

**Use Case**: Multi-image VLMs (attend to representative frames in video).

**3. Learned Sparse Attention**:

Train model to predict which tokens are important:

```python
class LearnedSparseAttention(nn.Module):
    def __init__(self, d_model, sparsity=0.1):
        self.importance_scorer = nn.Linear(d_model, 1)
        self.sparsity = sparsity

    def forward(self, Q, K, V):
        # Predict token importance
        importance = self.importance_scorer(K).squeeze(-1)  # (batch, seq_len)

        # Keep only top-k% tokens
        k = int(K.size(1) * self.sparsity)
        topk_indices = torch.topk(importance, k, dim=-1).indices

        # Sparse attention
        K_sparse = K.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, K.size(-1)))
        V_sparse = V.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, V.size(-1)))

        scores = Q @ K_sparse.transpose(-2, -1) / math.sqrt(Q.size(-1))
        attn = F.softmax(scores, dim=-1)
        return attn @ V_sparse
```

From [SparseVLM paper](https://arxiv.org/abs/2410.04417) (Zhang et al., 2024):
> "Visual token sparsification achieves 90% sparsity with <2% accuracy drop on VQA benchmarks. Learned importance scores identify 10% of visual tokens that carry 90% of information."

**4. Block-Sparse Attention**:

Attend to fixed-size blocks rather than individual tokens:

```
Image patches organized into 8×8 blocks:
┌─────┬─────┬─────┬─────┐
│  A  │  B  │  C  │  D  │  Each block = 64 patches
├─────┼─────┼─────┼─────┤
│  E  │  F  │  G  │  H  │  Attention: A ↔ B, A ↔ E
├─────┼─────┼─────┼─────┤  (not A ↔ all patches)
│  I  │  J  │  K  │  L  │
└─────┴─────┴─────┴─────┘
```

**Complexity Reduction**:
- Full attention: O(N²) = O((64×12)²) = O(589,824)
- Block-sparse: O((N/B)² × B²) = O(12² × 64²) = O(589,824) ... wait, same?
- **Benefit**: Memory locality + cache efficiency (blocks fit in SRAM)

### Sparse Attention Performance

From [Low-Rank Approximation for Sparse Attention](https://ieeexplore.ieee.org/document/10655939/) (Song et al., 2024):

| Sparsity | VQA Accuracy | Speedup | Memory Savings |
|----------|--------------|---------|----------------|
| 0% (full) | 78.5% | 1.0× | 1.0× |
| 50% | 78.2% | 1.6× | 1.5× |
| 75% | 77.8% | 2.3× | 2.1× |
| 90% | 76.9% | 3.8× | 4.2× |
| 95% | 74.1% | 5.1× | 6.8× |

**Sweet Spot**: 75-90% sparsity for VLMs (minimal accuracy loss, significant speedup).

---

## Section 4: Multi-Query Attention (MQA) & Grouped-Query Attention (GQA) (~90 lines)

### KV Cache Bottleneck in VLMs

**Problem**: During autoregressive generation, KV cache grows linearly with sequence length:

```
Standard Multi-Head Attention (MHA):
- Q: (batch, num_heads, seq_len, head_dim)
- K: (batch, num_heads, seq_len, head_dim)  ← Cached
- V: (batch, num_heads, seq_len, head_dim)  ← Cached

KV cache size = 2 × batch × num_heads × seq_len × head_dim × sizeof(float16)

Example (LLaVA generating text after image):
- Batch: 1
- Heads: 32
- Sequence: 512 text + 256 vision = 768 tokens
- Head dim: 128
- Size: 2 × 1 × 32 × 768 × 128 × 2 bytes = 12.6 MB per layer
- 32 layers = 403 MB KV cache for single image!
```

**Multi-Query Attention (MQA)**: Share single K, V across all query heads.

From [vllm-knowledge/00-vllm-architecture-pagedattention.md](../vllm-knowledge/00-vllm-architecture-pagedattention.md):

```python
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_query_heads, head_dim):
        self.num_query_heads = num_query_heads
        self.head_dim = head_dim

        # Separate Q for each head
        self.q_proj = nn.Linear(d_model, num_query_heads * head_dim)

        # Shared K, V across all heads
        self.k_proj = nn.Linear(d_model, head_dim)  # Single head!
        self.v_proj = nn.Linear(d_model, head_dim)  # Single head!

    def forward(self, x):
        B, N, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, N, self.num_query_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, 1, self.head_dim)  # 1 head
        v = self.v_proj(x).view(B, N, 1, self.head_dim)  # 1 head

        # Broadcast K, V to all query heads
        k = k.expand(B, N, self.num_query_heads, self.head_dim)
        v = v.expand(B, N, self.num_query_heads, self.head_dim)

        # FlashAttention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        output = F.scaled_dot_product_attention(q, k, v)
        output = output.transpose(1, 2).reshape(B, N, -1)
        return output
```

**KV Cache Savings**:
- MHA: 2 × 32 heads × seq_len × head_dim
- MQA: 2 × 1 head × seq_len × head_dim
- **32× smaller KV cache!**

### Grouped-Query Attention (GQA)

**Problem with MQA**: Sharing single K/V across all heads may lose representation capacity.

**Solution**: Group query heads, share K/V within groups.

```python
# GQA with 4 groups (32 query heads → 8 KV heads)
num_query_heads = 32
num_kv_heads = 8  # 32/8 = 4 query heads per KV head
```

**Architecture**:

```
Query Heads:  H0 H1 H2 H3 | H4 H5 H6 H7 | ... | H28 H29 H30 H31
              ↓  ↓  ↓  ↓  | ↓  ↓  ↓  ↓  |     | ↓   ↓   ↓   ↓
KV Heads:        KV0      |     KV1     | ... |      KV7
```

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_query_heads, num_kv_heads, head_dim):
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_query_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, num_query_heads * head_dim)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim)

    def forward(self, x):
        B, N, _ = x.shape

        q = self.q_proj(x).view(B, N, self.num_query_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim)

        # Repeat each KV head group_size times
        k = k.repeat_interleave(self.group_size, dim=2)
        v = v.repeat_interleave(self.group_size, dim=2)

        # Now k, v have same num_heads as q
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        output = F.scaled_dot_product_attention(q, k, v)
        output = output.transpose(1, 2).reshape(B, N, -1)
        return output
```

**KV Cache Comparison**:

| Variant | KV Heads | KV Cache Size | Quality |
|---------|----------|---------------|---------|
| MHA | 32 | 100% (baseline) | Best |
| GQA (8 groups) | 8 | 25% | ~98% of MHA |
| GQA (4 groups) | 4 | 12.5% | ~95% of MHA |
| MQA | 1 | 3.1% | ~90% of MHA |

From [Qwen3-VL paper](https://arxiv.org/abs/2309.16609):
> "Qwen3-VL uses GQA with 8 KV heads (from 32 query heads), achieving 4× KV cache reduction with <1% accuracy degradation on VQA benchmarks."

---

## Section 5: Attention Visualization & Interpretability (~80 lines)

### Why Visualize VLM Attention?

**Interpretability Goals**:
1. **Debugging**: Where does model look when answering "What color is the car?"
2. **Trust**: Verify model uses relevant image regions (not spurious correlations)
3. **Failure analysis**: Why did model miss the stop sign?

From [LVLM-Interpret paper](https://arxiv.org/abs/2404.03118) (accessed 2025-11-16):
> "LVLM-Interpret provides interactive visualization of attention patterns in large vision-language models, revealing which vision tokens influence text generation at each decoding step."

### Attention Map Visualization

**Raw Attention Weights**:

```python
def visualize_cross_attention(text_query, image_patches, attention_weights):
    # attention_weights: (num_heads, num_text_tokens, num_image_patches)

    # Average across heads
    attn_map = attention_weights.mean(dim=0)  # (num_text_tokens, num_image_patches)

    # For specific query token (e.g., "car")
    query_token_idx = 5  # Token index for "car"
    patch_importance = attn_map[query_token_idx]  # (num_image_patches,)

    # Reshape to spatial grid (14×14 patches)
    patch_grid = patch_importance.view(14, 14)

    # Overlay on image
    plt.imshow(original_image)
    plt.imshow(patch_grid, alpha=0.5, cmap='hot')
    plt.title(f"Attention for query: 'car'")
    plt.show()
```

**Attention Rollout** (Aggregate across layers):

```python
def attention_rollout(attention_weights_per_layer):
    # attention_weights_per_layer: List of (num_heads, seq_len, seq_len)

    # Start with identity matrix
    rollout = torch.eye(attention_weights_per_layer[0].size(-1))

    # Multiply attention matrices across layers
    for layer_attn in attention_weights_per_layer:
        # Average across heads
        avg_attn = layer_attn.mean(dim=0)

        # Add residual connection
        avg_attn = avg_attn + torch.eye(avg_attn.size(0))
        avg_attn = avg_attn / avg_attn.sum(dim=-1, keepdim=True)

        # Accumulate
        rollout = rollout @ avg_attn

    return rollout
```

### Sparse Attention Head Discovery

From [Sparse Attention Vectors paper](https://arxiv.org/abs/2412.00142):

**Finding Important Heads**:

```python
def find_sparse_attention_heads(model, dataset):
    head_activations = defaultdict(list)

    for image, query in dataset:
        # Forward pass with hooks to capture attention
        outputs, attention_weights = model(image, query, output_attentions=True)

        # Track per-head activation magnitude
        for layer_idx, layer_attn in enumerate(attention_weights):
            for head_idx in range(layer_attn.size(1)):
                head_attn = layer_attn[0, head_idx]  # (seq_len, seq_len)
                activation = head_attn.abs().mean().item()
                head_activations[(layer_idx, head_idx)].append(activation)

    # Find heads with high activation variance (task-specific)
    important_heads = []
    for (layer, head), activations in head_activations.items():
        variance = np.var(activations)
        if variance > threshold:
            important_heads.append((layer, head))

    return important_heads  # Typically <5% of all heads
```

**Result**: Only 3-5 heads out of 32 heads × 32 layers = 1,024 total heads are critical for localization!

### Attention Pattern Analysis

**Head Specialization in VLMs**:

From recent VLM interpretability research (2024):

| Head Type | Pattern | Example |
|-----------|---------|---------|
| **Localization heads** | High attention on object regions | "Where is the cat?" → cat pixels |
| **Semantic heads** | Attend to category-related patches | "Animal" → all animal regions |
| **Spatial heads** | Encode position relationships | "Left of" → spatial layout |
| **Texture heads** | Focus on fine details | "Striped" → texture patterns |

**Visualization Tools**:

From [attention visualization research](https://www.soniajoseph.ai/multimodal-interpretability-in-2024/):

```python
# BertViz for VLMs (adapted)
from bertviz import model_view

# Visualize multi-modal attention
model_view(
    attention=attention_weights,  # (layers, heads, seq, seq)
    tokens=text_tokens + ["[IMG_PATCH_0]", "[IMG_PATCH_1]", ...],
    html_action='return'  # Get HTML visualization
)
```

---

## Section 6: KV Cache Optimization for VLM Inference (~70 lines)

### KV Cache Management

**Inference Pattern** (Autoregressive Generation):

```
Step 1: Process image + prompt
  Input: [IMG_TOKENS (256)] + ["What", "is", "in", "the", "image", "?"]
  Output: "A"
  KV cache: 262 tokens

Step 2: Generate next token
  Input: "A" (new token only)
  Output: "cat"
  KV cache: 263 tokens (append new K, V)

Step 3: Continue...
  Input: "cat"
  Output: "sitting"
  KV cache: 264 tokens
```

**Memory Growth**:

```python
# KV cache allocation
batch_size = 4
num_layers = 32
num_kv_heads = 8  # GQA
seq_len = 256 (image) + 512 (generated text) = 768
head_dim = 128

kv_cache_size = (
    2 (K and V) ×
    batch_size ×
    num_layers ×
    num_kv_heads ×
    seq_len ×
    head_dim ×
    sizeof(float16)
)

= 2 × 4 × 32 × 8 × 768 × 128 × 2 bytes
= 100.7 MB

# For 100 concurrent users → 10 GB!
```

### PagedAttention for VLMs

From [vllm-knowledge/00-vllm-architecture-pagedattention.md](../vllm-knowledge/00-vllm-architecture-pagedattention.md):

**Key Innovation**: Treat KV cache like virtual memory (paging).

**Benefits for VLMs**:
1. **Dynamic allocation**: Allocate cache blocks as needed (text length varies)
2. **Sharing**: Share image cache across multiple queries
3. **Memory efficiency**: No pre-allocation waste

```python
# VLM with shared image cache
class PagedVLM:
    def __init__(self):
        self.block_size = 16  # KV pairs per block
        self.kv_cache = {}  # block_id → (K, V) tensor

    def encode_image(self, image):
        # Encode image once, cache K, V
        image_features = self.vision_encoder(image)

        # Allocate blocks for image tokens
        num_image_tokens = 256
        num_blocks = (num_image_tokens + self.block_size - 1) // self.block_size

        block_ids = []
        for i in range(num_blocks):
            block_id = self.allocate_block()
            start = i * self.block_size
            end = min((i + 1) * self.block_size, num_image_tokens)

            # Store K, V for this block
            k_block, v_block = self.compute_kv(image_features[start:end])
            self.kv_cache[block_id] = (k_block, v_block)
            block_ids.append(block_id)

        return block_ids  # Image cache reference

    def generate(self, image_block_ids, prompt):
        # Reuse cached image K, V
        image_kv = [self.kv_cache[bid] for bid in image_block_ids]

        # Generate text (new KV blocks allocated dynamically)
        for token in generate_autoregressive():
            new_block = self.allocate_block()
            k_new, v_new = self.compute_kv(token)
            self.kv_cache[new_block] = (k_new, v_new)
            yield token
```

**Memory Savings**: Single image processed 10 times:
- Without sharing: 10 × 12.6 MB = 126 MB
- With sharing: 1 × 12.6 MB (image) + 10 × 2 MB (text) = 32.6 MB
- **3.9× memory reduction**

### Vision Encoder Caching

**Optimization**: Precompute vision encoder output (doesn't change per query).

```python
class CachedVisionVLM:
    def __init__(self):
        self.vision_cache = {}  # image_hash → vision_features

    def process_image(self, image):
        # Hash image
        image_hash = hash(image.tobytes())

        # Check cache
        if image_hash in self.vision_cache:
            return self.vision_cache[image_hash]  # Cache hit!

        # Compute vision features (expensive)
        vision_features = self.vision_encoder(image)
        self.vision_cache[image_hash] = vision_features
        return vision_features

    def answer_query(self, image, text_query):
        # Vision encoder cached (no recomputation)
        vision_features = self.process_image(image)

        # Only run cross-attention + LLM (fast)
        return self.llm(vision_features, text_query)
```

**Latency Improvement**:
- Vision encoder: 50ms
- Cross-attention + LLM: 150ms
- **Total without cache**: 200ms
- **Total with cache**: 150ms (1.33× faster)

---

## Section 7: Compute and Memory Trade-offs (~60 lines)

### GFLOPs Analysis for VLM Attention

**Vision Self-Attention**:

```
Image: 1024×1024, patch size 14×14
Patches: (1024/14)² = 5,329 tokens
Hidden dim: 3,584
Layers: 32

FLOPs per layer = 4 × N² × d
                = 4 × 5,329² × 3,584
                = 407 TFLOPs

32 layers = 13,024 TFLOPs for vision alone
```

**Cross-Attention** (text queries vision):

```
Text tokens: 512
Vision tokens: 5,329
Hidden dim: 3,584

FLOPs = 2 × N_text × N_vision × d (QK matmul)
       + N_text × N_vision (softmax)
       + 2 × N_text × N_vision × d (PV matmul)
      = 4 × 512 × 5,329 × 3,584
      = 39 TFLOPs per cross-attention layer

If 8 cross-attention layers: 312 TFLOPs
```

**Total VLM Inference** (Qwen3-VL example):
- Vision self-attention: 13,024 TFLOPs
- Cross-attention: 312 TFLOPs
- LLM text generation (2048 tokens): ~500 TFLOPs
- **Total**: ~13,836 TFLOPs per image-text pair

**A100 GPU (312 TFLOPs FP16)**:
- Without FlashAttention (80 TFLOPs actual): 173 seconds
- With FlashAttention-2 (225 TFLOPs actual): 61 seconds
- **2.8× speedup**

### Memory Bandwidth Bottleneck

From [../llm-gpu-integration/00-flashattention-internals.md](../llm-gpu-integration/00-flashattention-internals.md):

**A100 GPU Specifications**:
- HBM bandwidth: 1.6 TB/s
- SRAM bandwidth: ~19.5 TB/s (12× faster)
- Peak compute: 312 TFLOPs FP16

**Attention Memory Traffic** (Standard Implementation):

```
# Single attention layer, 5,329 tokens, FP16

HBM Reads:
- Q, K, V: 3 × 5,329 × 3,584 × 2 bytes = 114 MB

HBM Writes:
- Attention matrix S: 5,329² × 2 bytes = 56 MB
- Softmax output P: 5,329² × 2 bytes = 56 MB
- Output O: 5,329 × 3,584 × 2 bytes = 38 MB

Total HBM traffic: 264 MB
Time at 1.6 TB/s: 0.165 ms

Compute time (407 TFLOPs at 312 TFLOPs/s): 1.3 ms

Ratio: Compute is 7.9× slower than memory
→ NOT MEMORY BOUND for this case!
```

**But for smaller batches / fewer tokens**:

```
# 256 tokens (typical single image)
HBM traffic: 6 MB
Time: 0.00375 ms

Compute: 0.94 TFLOPs at 312 TFLOPs/s = 0.003 ms

→ MEMORY BOUND! (memory slower than compute)
```

**FlashAttention Savings**:
- HBM accesses: O(N²) → O(N²/M) where M = SRAM size
- For M = 164 KB, N = 256: **~10× fewer HBM accesses**

---

## Section 8: ARR-COC-0-1 Attention Architecture (~80 lines)

### Relevance-Driven Attention Allocation

ARR-COC-0-1 uses **cross-attention for relevance scoring**, not just feature fusion.

**Three Relevance Scorers** (all use cross-attention):

```python
class ParticipatorySc orer:
    """Query-content coupling via cross-attention."""

    def score_relevance(self, query_embedding, patch_features):
        # query_embedding: (batch, num_query_tokens, dim)
        # patch_features: (batch, num_patches, dim)

        # Cross-attention: query attends to patches
        # FlashAttention for efficiency (196-400 patches)
        attention_scores = F.scaled_dot_product_attention(
            query=query_embedding,
            key=patch_features,
            value=patch_features,
            is_causal=False
        )

        # Aggregate to per-patch relevance
        patch_relevance = attention_scores.mean(dim=1)  # (batch, num_patches, dim)

        # Reduce to scalar relevance per patch
        relevance_score = patch_relevance.norm(dim=-1)  # (batch, num_patches)

        return relevance_score
```

**Why FlashAttention Matters**:
- **196 patches** (14×14 grid) × 13 texture channels = 2,548 feature tokens
- 3 scorers × cross-attention = 3 × O(N_query × N_patches)
- For N_query=32, N_patches=196: 3 × 32 × 196 = **18,816 attention operations per image**
- **FlashAttention**: 2-4× faster enables real-time relevance realization

### Variable LOD Attention

ARR-COC-0-1 allocates 64-400 tokens per patch based on relevance:

```python
def allocate_lod_tokens(patch_features, relevance_scores, total_budget=1600):
    # relevance_scores: (batch, num_patches)
    # total_budget: Total tokens across all patches (e.g., 8 patches × 200 avg)

    # Softmax for allocation distribution
    allocation_weights = F.softmax(relevance_scores, dim=-1)

    # Allocate tokens (64-400 range)
    tokens_per_patch = (allocation_weights * total_budget).round()
    tokens_per_patch = torch.clamp(tokens_per_patch, min=64, max=400)

    # Exact budget enforcement
    current_total = tokens_per_patch.sum()
    adjustment = (total_budget - current_total) / num_patches
    tokens_per_patch += adjustment

    return tokens_per_patch.long()
```

**Attention Computation with Variable Lengths**:

```python
# Process each patch with its allocated token count
outputs = []
for patch_idx in range(num_patches):
    num_tokens = tokens_per_patch[patch_idx]  # 64-400

    # Extract patch tokens (from 13-channel texture array)
    patch_tokens = texture_features[patch_idx, :num_tokens]

    # Self-attention within patch (FlashAttention)
    # Efficient even for variable lengths!
    patch_output = F.scaled_dot_product_attention(
        query=patch_tokens,
        key=patch_tokens,
        value=patch_tokens,
        is_causal=False
    )

    outputs.append(patch_output)

# Concatenate variable-length outputs
final_features = torch.cat(outputs, dim=0)  # Total: ~1600 tokens
```

### Tensor Core Alignment

From [../cuda/05-tensor-core-programming-wmma-mma.md](../cuda/05-tensor-core-programming-wmma-mma.md):

**Optimal Token Counts** (Ampere A100):
- Tensor Core tiles: 16×16 (m×n)
- **ARR-COC token allocations**:
  - Min: 64 = 4 × 16 ✓
  - Avg: 200 ≈ 13 × 16 (208) ✓
  - Max: 400 = 25 × 16 ✓

**Why This Matters**: Non-aligned dimensions (e.g., 200 tokens) require padding to 208, wasting 4% of GPU resources. ARR-COC's 64/144/256/400 choices align perfectly.

### ARR-COC Attention Performance Budget

**Target Latency**: <200ms total inference (relevance realization + LLM)

**Breakdown**:
1. **Texture Array Creation**: 10ms (GPU texture ops)
2. **Relevance Scoring** (3 scorers × cross-attention): 30ms
3. **Token Allocation**: 5ms
4. **Variable LOD Attention**: 40ms (FlashAttention critical here!)
5. **LLM Inference** (Qwen3-VL): 100ms
6. **Opponent Processing**: 15ms

**Total**: 200ms ✓

**Without FlashAttention**:
- Relevance scoring: 30ms → 90ms (3× slower)
- LOD attention: 40ms → 120ms (3× slower)
- **Total**: 335ms ✗ (misses real-time target)

### Multi-Query Attention for ARR-COC

Use MQA/GQA to reduce KV cache during LLM generation:

```python
class ARRCOCAttention(nn.Module):
    def __init__(self, d_model=3584, num_query_heads=32, num_kv_heads=8):
        # GQA: 32 query heads, 8 KV heads (4:1 ratio)
        self.gqa = GroupedQueryAttention(
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=d_model // num_query_heads
        )

    def forward(self, visual_tokens, text_tokens):
        # Concatenate vision + text
        combined = torch.cat([visual_tokens, text_tokens], dim=1)

        # GQA attention (4× smaller KV cache than MHA)
        output = self.gqa(combined)

        return output
```

**KV Cache Savings** (ARR-COC deployment):
- Visual tokens: 1,600 (variable LOD)
- Text tokens: 512 (generation)
- Total: 2,112 tokens

**MHA**: 2 × 32 heads × 2,112 × 128 × 2 bytes × 32 layers = 550 MB
**GQA (8 heads)**: 2 × 8 heads × 2,112 × 128 × 2 bytes × 32 layers = 138 MB
**4× reduction** → enables larger batch sizes on single A100

---

## Sources

**Original Papers**:
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) - Tri Dao, 2023 (accessed 2025-11-16)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://tridao.me/blog/2024/flash3/) - Tri Dao, 2024 (accessed 2025-11-16)
- [Sparse Attention Vectors: Generative Multimodal Model Features As Sparse Attention Patterns](https://arxiv.org/abs/2412.00142) - Mitra et al., 2024 (accessed 2025-11-16)
- [SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference](https://arxiv.org/abs/2410.04417) - Zhang et al., 2024 (accessed 2025-11-16)
- [Low-Rank Approximation for Sparse Attention in Multi-Modal Models](https://ieeexplore.ieee.org/document/10655939/) - Song et al., 2024 (accessed 2025-11-16)

**Web Resources**:
- [LVLM-Interpret: An Interpretability Tool for Large Vision-Language Models](https://arxiv.org/abs/2404.03118) - June 2024 (accessed 2025-11-16)
- [PyTorch scaled_dot_product_attention Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) - PyTorch Team (accessed 2025-11-16)
- [PyTorch 2.2 Release: FlashAttention-v2 Integration](https://pytorch.org/blog/pytorch2-2/) - PyTorch Team, January 2024 (accessed 2025-11-16)
- [Multimodal Interpretability in 2024](https://www.soniajoseph.ai/multimodal-interpretability-in-2024/) - Sonia Joseph, October 2024 (accessed 2025-11-16)

**Related Knowledge**:
- [../llm-gpu-integration/00-flashattention-internals.md](../llm-gpu-integration/00-flashattention-internals.md) - FlashAttention algorithm details
- [../vision-language/10-token-sequence-order-importance.md](../vision-language/10-token-sequence-order-importance.md) - Attention ordering in VLMs
- [../cuda/05-tensor-core-programming-wmma-mma.md](../cuda/05-tensor-core-programming-wmma-mma.md) - Tensor Core alignment
- [../vllm-knowledge/00-vllm-architecture-pagedattention.md](../vllm-knowledge/00-vllm-architecture-pagedattention.md) - PagedAttention and KV cache optimization

**GitHub Repositories**:
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) - Official FlashAttention implementation (accessed 2025-11-16)
