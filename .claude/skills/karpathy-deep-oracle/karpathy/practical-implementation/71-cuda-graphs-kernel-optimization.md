# CUDA Graphs and Kernel Launch Optimization

## Overview

CUDA Graphs are a mechanism to capture a sequence of CUDA operations (kernels, memory copies, etc.) and execute them as a single unit, dramatically reducing CPU overhead from kernel launches. Instead of launching each operation individually, CUDA Graphs bundle multiple GPU operations into a single launchable unit, reducing overhead from microseconds per kernel to near-constant time.

**Why CUDA Graphs Matter for Deep Learning:**
- **Reduced CPU overhead** - Launch overhead drops from ~2μs + 200ns/node to ~2.5μs constant (10+ node graphs)
- **Higher GPU utilization** - Eliminates CPU bottlenecks that cause GPU stalls
- **Better performance scaling** - Critical for models with many small kernels (transformers, VLMs)
- **Consistent timing** - Predictable kernel execution reduces timing skew in distributed training

From [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) (accessed 2025-01-13):
> "It provides a mechanism to launch multiple GPU operations through a single CPU operation, and hence reduces the launching overheads."

From [Constant Time Launch for Straight-Line CUDA Graphs](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/) (accessed 2025-01-13):
> "For straight-line kernel graphs, there has been a significant reduction in time taken during repeat launches. Specifically, the time decreased from 2μs + 200 ns extra time per node to a nearly constant time of 2.5μs + (~1ns per node)."

---

## Section 1: CUDA Graphs Fundamentals (120 lines)

### What Are CUDA Graphs?

**Definition**: A CUDA Graph is a recorded sequence of GPU operations with explicit dependencies that can be launched and replayed with minimal CPU involvement.

**Traditional Launch Model:**
```
CPU: Launch Kernel A → Launch Kernel B → Launch Kernel C → Launch Kernel D
GPU:     [A]              [B]              [C]              [D]
         ↑ gap            ↑ gap            ↑ gap
```

**CUDA Graphs Model:**
```
CPU: Build Graph (once) → Launch Graph
GPU:     [A][B][C][D]  (tightly packed, minimal gaps)
```

### The Kernel Launch Overhead Problem

**CPU Overhead Per Kernel:**
- Python layer: Function call overhead
- PyTorch/TensorFlow layer: Argument processing, dispatch logic
- CUDA driver layer: Kernel setup, parameter marshaling
- Total: 5-50 μs per kernel (depends on complexity)

**When This Becomes Critical:**
- **Short kernels** - When kernel execution time < 100 μs
- **Many kernels** - Transformer layers with 100+ kernel calls per iteration
- **Small batch sizes** - Common in inference (batch=1 for low latency)
- **Multi-GPU** - NCCL collective operations have launch overhead

From vertex-ai-production/01-gpu-optimization-deep.md (lines 20-35):
> "Memory bandwidth is the primary bottleneck for deep learning workloads. Effective use of memory hierarchy is critical."

**Example: Transformer Inference Bottleneck**
```python
# Traditional inference (per token generation)
for _ in range(output_length):
    # Each iteration: 10-20 kernel launches
    attn_output = multi_head_attention(query, key, value)  # 5-8 kernels
    ffn_output = feed_forward(attn_output)                 # 3-5 kernels
    output = layer_norm(ffn_output)                         # 1-2 kernels

# CPU overhead: 10-20 kernels × 10 μs = 100-200 μs
# Actual GPU work: Often < 50 μs for small batches
# GPU utilization: ~20-30% (CPU-bound!)
```

### Graph Capture, Instantiate, Replay

**Three Phases of CUDA Graphs:**

**1. Capture Phase** - Record operations into a graph
```python
import torch

# Warm up (required before capture)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):  # Warm-up iterations
        output = model(input)
torch.cuda.current_stream().wait_stream(s)

# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)
```

**2. Instantiate Phase** - Prepare graph for execution
- Graph is analyzed and optimized
- Memory is allocated for intermediate tensors
- Dependencies are resolved
- Executable graph is created

**3. Replay Phase** - Execute the graph
```python
# Fill input with new data
static_input.copy_(real_input)

# Replay graph (single CPU operation)
g.replay()

# Read output
result = static_output.clone()
```

### Performance Benefits

From [NVIDIA CUDA Graphs Blog](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/) (accessed 2025-01-13):

**CUDA Toolkit 11.8 vs 12.6 Improvements:**
| Metric | Topology | Length | 11.8 | 12.6 | Speedup |
|--------|----------|--------|------|------|---------|
| Instantiation | Straight line | 100 | 168 μs | 127 μs | 32% |
| Instantiation | Straight line | 1025 | 2143 μs | 1526 μs | 40% |
| First Launch CPU | Straight line | 100 | 25 μs | 15 μs | 66% |
| Repeat Launch CPU | Straight line | 100 | 20 μs | 2.5 μs | 87% |
| Repeat Launch CPU | Straight line | 1025 | 200 μs | 2.5 μs | 98% |

**Key Insight**: Repeat launch overhead is now nearly constant (~2.5μs) regardless of graph size for straight-line graphs (10+ nodes).

### Constraints and Requirements

**CUDA Graphs Work Best When:**
- ✓ Static shapes - Tensor dimensions don't change
- ✓ Static control flow - No data-dependent branching
- ✓ No CPU synchronization - No `torch.cuda.synchronize()` in graph
- ✓ Fixed memory addresses - Same tensors used across iterations

**Not Suitable For:**
- ✗ Dynamic shapes (variable sequence lengths)
- ✗ Dynamic batching (changing batch sizes)
- ✗ Conditional execution based on intermediate results
- ✗ Operations requiring CPU-GPU sync

**Memory Considerations:**
- Graphs allocate memory from a private pool
- Memory layout is fixed after capture
- Can increase total memory usage (trade memory for speed)
- Best practice: Use `set_to_none=True` in `optimizer.zero_grad()`

---

## Section 2: PyTorch Integration (140 lines)

### PyTorch CUDA Graph API

PyTorch provides three ways to use CUDA Graphs:

**1. Raw API** - `torch.cuda.CUDAGraph` for manual control
**2. Context Manager** - `torch.cuda.graph()` for convenience
**3. Callable Wrapper** - `torch.cuda.make_graphed_callables()` for partial graphing

### Full Model Graphing with torch.cuda.graph

**Complete Training Loop Example:**

```python
import torch
import torch.nn as nn

# Model setup
N, D_in, H, D_out = 640, 4096, 2048, 1024
model = nn.Sequential(
    nn.Linear(D_in, H),
    nn.Dropout(p=0.2),
    nn.Linear(H, D_out),
    nn.Dropout(p=0.1)
).cuda()

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Static tensors for capture
static_input = torch.randn(N, D_in, device='cuda')
static_target = torch.randn(N, D_out, device='cuda')

# Warm-up on side stream (REQUIRED)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for i in range(3):
        optimizer.zero_grad(set_to_none=True)
        y_pred = model(static_input)
        loss = loss_fn(y_pred, static_target)
        loss.backward()
        optimizer.step()
torch.cuda.current_stream().wait_stream(s)

# Capture graph
g = torch.cuda.CUDAGraph()
optimizer.zero_grad(set_to_none=True)
with torch.cuda.graph(g):
    static_y_pred = model(static_input)
    static_loss = loss_fn(static_y_pred, static_target)
    static_loss.backward()
    optimizer.step()

# Training loop with graph replay
real_inputs = [torch.rand_like(static_input) for _ in range(10)]
real_targets = [torch.rand_like(static_target) for _ in range(10)]

for data, target in zip(real_inputs, real_targets):
    # Copy new data into static tensors
    static_input.copy_(data)
    static_target.copy_(target)

    # Replay: forward + backward + step
    g.replay()

    # Gradients and params are updated
    # Can read static_loss.item() for monitoring
```

### Partial Model Graphing with make_graphed_callables

When model has dynamic components, graph only the static parts:

```python
# Model with dynamic control flow
module1 = nn.Linear(D_in, H).cuda()
module2 = nn.Linear(H, D_out).cuda()
module3 = nn.Linear(H, D_out).cuda()

# Sample inputs for capture
x = torch.randn(N, D_in, device='cuda')
h = torch.randn(N, H, device='cuda', requires_grad=True)

# Make graphed versions (autograd-aware)
module1 = torch.cuda.make_graphed_callables(module1, (x,))
module2 = torch.cuda.make_graphed_callables(module2, (h,))
module3 = torch.cuda.make_graphed_callables(module3, (h,))

# Training loop
for data, target in dataloader:
    optimizer.zero_grad(set_to_none=True)

    tmp = module1(data)  # Runs as graph

    # Dynamic control flow (eager)
    if tmp.sum().item() > 0:
        tmp = module2(tmp)  # Runs as graph
    else:
        tmp = module3(tmp)  # Runs as graph

    loss = loss_fn(tmp, target)
    loss.backward()  # Backward also graphed!
    optimizer.step()
```

### Static vs Dynamic Shapes

**Problem**: Graphs require fixed tensor shapes. Variable sequence lengths break graphing.

**Solution Patterns:**

**1. Padding to Maximum Length**
```python
# Pad all sequences to max_length
def pad_batch(sequences, max_length):
    padded = torch.zeros(len(sequences), max_length, device='cuda')
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    return padded

# Capture with max_length
with torch.cuda.graph(g):
    output = model(padded_input)  # Shape: [batch, max_length, hidden]
```

**2. Multiple Graphs for Common Shapes**
```python
# Capture graphs for common sequence lengths
graphs = {}
for seq_len in [128, 256, 512, 1024]:
    static_input = torch.randn(batch_size, seq_len, device='cuda')
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        output = model(static_input)
    graphs[seq_len] = (g, static_input, output)

# Use appropriate graph at inference
def infer(input_seq):
    seq_len = input_seq.size(1)
    # Round up to nearest graph size
    graph_len = min(l for l in graphs.keys() if l >= seq_len)
    g, static_in, static_out = graphs[graph_len]

    # Pad if necessary
    if seq_len < graph_len:
        input_seq = F.pad(input_seq, (0, graph_len - seq_len))

    static_in.copy_(input_seq)
    g.replay()
    return static_out[:, :seq_len]  # Remove padding
```

### Memory Management and Pools

**Graph Memory Pool Behavior:**
```python
# Graphs use private memory pool
g = torch.cuda.CUDAGraph()

# Allocations during capture come from graph's pool
with torch.cuda.graph(g):
    # This allocation is FIXED for all replays
    intermediate = torch.zeros(batch_size, hidden_size, device='cuda')
    output = model(intermediate)

# Memory is NOT released until graph is destroyed
del g  # Releases graph's memory pool
```

**Best Practices:**
```python
# 1. Use set_to_none for gradients (creates in graph pool)
optimizer.zero_grad(set_to_none=True)

# 2. Pre-allocate static tensors outside graph
static_input = torch.randn(N, D_in, device='cuda')

# 3. Minimize allocations inside graph
# BAD:
with torch.cuda.graph(g):
    temp = torch.zeros(...)  # New allocation every capture

# GOOD:
temp = torch.zeros(..., device='cuda')  # Pre-allocate
with torch.cuda.graph(g):
    temp.zero_()  # Reuse memory
```

### Integration with Mixed Precision (AMP)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Warm up
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=torch.float16):
            output = model(static_input)
            loss = loss_fn(output, static_target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
torch.cuda.current_stream().wait_stream(s)

# Capture with AMP
g = torch.cuda.CUDAGraph()
optimizer.zero_grad(set_to_none=True)
with torch.cuda.graph(g):
    with autocast(dtype=torch.float16):
        static_output = model(static_input)
        static_loss = loss_fn(static_output, static_target)
    scaler.scale(static_loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Replay
for data, target in dataloader:
    static_input.copy_(data)
    static_target.copy_(target)
    g.replay()
```

---

## Section 3: Transformer Inference Optimization (120 lines)

### CUDA Graphs for GPT/BERT Inference

**Challenge**: Transformer inference is CPU-bound at small batch sizes due to:
- Many small kernels (attention, layer norm, FFN)
- Autoregressive generation (one token at a time)
- KV cache updates every iteration

**Solution**: Graph the entire decode step.

### KV Cache with CUDA Graphs

**Static KV Cache Pattern:**
```python
import torch.nn.functional as F

class GraphedTransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len

        # Pre-allocate KV cache (static)
        self.register_buffer(
            'k_cache',
            torch.zeros(1, num_heads, max_seq_len, hidden_size // num_heads)
        )
        self.register_buffer(
            'v_cache',
            torch.zeros(1, num_heads, max_seq_len, hidden_size // num_heads)
        )
        self.register_buffer('cache_position', torch.tensor(0))

    def forward(self, x, position):
        # Compute new K, V
        k_new = self.k_proj(x)
        v_new = self.v_proj(x)

        # Update cache at position (in-place)
        self.k_cache[:, :, position] = k_new
        self.v_cache[:, :, position] = v_new

        # Attention over all cached tokens
        q = self.q_proj(x)
        attn_output = F.scaled_dot_product_attention(
            q,
            self.k_cache[:, :, :position+1],
            self.v_cache[:, :, :position+1]
        )
        return attn_output
```

**Graph Capture for Autoregressive Generation:**
```python
# Prefill phase (not graphed - variable length)
with torch.no_grad():
    for i, token in enumerate(input_tokens):
        output = model(token, position=i)

# Decode phase (graphed - fixed pattern)
g = torch.cuda.CUDAGraph()

# Static tensors
static_token = torch.zeros(1, 1, hidden_size, device='cuda')
static_position = torch.tensor(0, device='cuda')
static_output = torch.zeros(1, 1, hidden_size, device='cuda')

# Capture single decode step
with torch.cuda.graph(g):
    static_output = model(static_token, static_position)

# Generate tokens with graph
generated_tokens = []
for i in range(max_new_tokens):
    # Update position
    static_position.fill_(len(input_tokens) + i)

    # Update input (from previous output)
    static_token.copy_(static_output)

    # Replay graph
    g.replay()

    # Sample next token
    next_token = static_output.argmax(dim=-1)
    generated_tokens.append(next_token.item())
```

### Batch Size Considerations

**Graphing Trade-offs by Batch Size:**

| Batch Size | CPU Overhead | GPU Utilization | Graph Benefit |
|------------|--------------|-----------------|---------------|
| 1 | Very High | 10-20% | **Massive** (5-10×) |
| 4 | High | 30-40% | **Large** (2-3×) |
| 16 | Moderate | 60-70% | **Moderate** (1.5-2×) |
| 64 | Low | 85-95% | **Small** (1.1-1.3×) |

**When to Use Graphs:**
- **Inference** - Almost always beneficial (batch sizes typically 1-16)
- **Training** - Beneficial for small batch sizes or many-kernel models
- **Large batches** - Less critical but still helpful

### Warmup and Graph Capture Strategies

From [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) (accessed 2025-01-13):

**Why Warm-up is Critical:**
```python
# BAD: No warm-up
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(input)  # May capture uninitialized state!

# GOOD: Proper warm-up
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):  # 3 iterations minimum
        output = model(input)
torch.cuda.current_stream().wait_stream(s)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(input)
```

**Warm-up ensures:**
- CUDA kernels are compiled (JIT)
- Memory allocations are stabilized
- Autotuning completes (cuDNN, etc.)
- Internal state is initialized

**Transformer-Specific Warm-up:**
```python
def warmup_transformer(model, batch_size, seq_len, device='cuda'):
    """Warm up transformer before graph capture"""
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warm-up on side stream
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.no_grad():
            for _ in range(10):  # More iterations for complex models
                _ = model(dummy_input)
    torch.cuda.current_stream().wait_stream(s)

    # Force synchronization
    torch.cuda.synchronize()
```

### Performance Benchmarking Results

From [NVIDIA MLPerf v1.0 Results](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) (accessed 2025-01-13):

**Real-World Speedups:**
| Model | GPUs | Speedup from CUDA Graphs |
|-------|------|--------------------------|
| Mask R-CNN | 272 | **1.70×** |
| BERT | 4096 | **1.12×** |
| DLRM (batch=32) | 1 | **1.35×** |
| DLRM (batch=1) | 1 | **2.10×** |

**Mask R-CNN Case Study:**
- Before: GPU idle 70% of time, CPU maxed at 100%
- After: GPU kernels tightly packed, 5× speedup on backbone (31ms → 6ms)
- Overall: 1.7× end-to-end training speedup

**BERT Case Study:**
- Removed CPU-GPU synchronizations (dynamic shapes → static masks)
- Graph captured full model (all layers)
- 1.12× speedup at 4096 GPUs (critical for large-scale training)

---

## Section 4: ARR-COC Integration (70 lines)

### Relevance Scoring with CUDA Graphs

**ARR-COC Pipeline Stages:**
```
Input Image → Texture Extraction → Relevance Scoring → Token Allocation → Compression
              [Many kernels]       [Many kernels]      [Top-K ops]      [Gather ops]
```

**Graphing Opportunities:**
1. **Texture Extraction** - Fixed conv/pooling operations per patch
2. **Relevance Scoring** - Propositional/Perspectival/Participatory scorers
3. **Token Allocation** - Top-K selection, budget assignment
4. **Compression** - Patch gathering, feature extraction

### Token Allocation Kernel Optimization

**Problem**: Top-K relevance selection launches many small kernels.

**Solution**: Graph the entire allocation pipeline.

```python
class GraphedTokenAllocator(nn.Module):
    def __init__(self, num_patches, min_tokens=64, max_tokens=400):
        super().__init__()
        self.num_patches = num_patches
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def forward(self, relevance_scores, total_budget):
        # All operations here will be in the graph

        # 1. Normalize scores
        scores_norm = F.softmax(relevance_scores, dim=-1)

        # 2. Compute token allocation (proportional to relevance)
        allocations = (scores_norm * total_budget).round().long()

        # 3. Clamp to min/max
        allocations = torch.clamp(allocations, self.min_tokens, self.max_tokens)

        # 4. Adjust to meet exact budget
        current_total = allocations.sum()
        diff = total_budget - current_total
        if diff != 0:
            # Distribute difference to patches with highest residuals
            residuals = scores_norm * total_budget - allocations.float()
            _, indices = torch.topk(residuals, k=abs(diff))
            allocations[indices] += torch.sign(diff).long()

        return allocations

# Capture allocation graph
allocator = GraphedTokenAllocator(num_patches=196).cuda()
static_scores = torch.randn(196, device='cuda')
static_budget = torch.tensor(8000, device='cuda')

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_allocations = allocator(static_scores, static_budget)

# Use in ARR-COC pipeline
for image_batch in dataloader:
    # Compute relevance (can also be graphed separately)
    relevance_scores = compute_relevance(image_batch)

    # Update static tensor
    static_scores.copy_(relevance_scores)

    # Replay allocation graph
    g.replay()

    # Use allocations for compression
    compressed = apply_allocations(image_batch, static_allocations)
```

### VLM Inference Speedup

**Multi-Stage Graph Capture:**
```python
# Stage 1: Visual encoder (graphed)
g_visual = torch.cuda.CUDAGraph()
with torch.cuda.graph(g_visual):
    static_image_features = visual_encoder(static_image)

# Stage 2: Relevance realization (graphed)
g_relevance = torch.cuda.CUDAGraph()
with torch.cuda.graph(g_relevance):
    static_prop_scores = propositional_scorer(static_image_features)
    static_persp_scores = perspectival_scorer(static_image_features)
    static_partic_scores = participatory_scorer(static_image_features, static_query)
    static_final_scores = opponent_process(
        static_prop_scores, static_persp_scores, static_partic_scores
    )

# Stage 3: Token allocation (graphed as shown above)

# Stage 4: Language model (may have dynamic length, not graphed)

# Full pipeline
def arr_coc_inference_graphed(image, query):
    # Visual encoding (graphed)
    static_image.copy_(image)
    g_visual.replay()

    # Relevance scoring (graphed)
    static_query.copy_(query)
    g_relevance.replay()

    # Token allocation (graphed)
    static_scores.copy_(static_final_scores)
    g_allocation.replay()

    # Language generation (eager, due to variable length)
    output = language_model.generate(
        static_image_features,
        static_allocations,
        query
    )
    return output
```

**Expected Speedups for ARR-COC:**
- **Relevance scoring**: 2-3× faster (many small kernels)
- **Token allocation**: 3-5× faster (Top-K operations)
- **Overall VLM inference**: 1.5-2× faster (visual stages dominate)

### Production Deployment Patterns

**Graph Management Strategy:**
```python
class ARRCOCGraphCache:
    def __init__(self, image_sizes=[224, 384, 512]):
        self.graphs = {}
        self._build_graphs(image_sizes)

    def _build_graphs(self, image_sizes):
        for size in image_sizes:
            # Build graph for this image size
            num_patches = (size // 16) ** 2

            g = torch.cuda.CUDAGraph()
            static_input = torch.randn(1, 3, size, size, device='cuda')

            with torch.cuda.graph(g):
                features = self.visual_encoder(static_input)
                scores = self.relevance_scorer(features)

            self.graphs[size] = (g, static_input, scores)

    def infer(self, image):
        size = image.shape[-1]
        g, static_in, static_out = self.graphs[size]
        static_in.copy_(image)
        g.replay()
        return static_out.clone()
```

---

## Sources

**PyTorch Documentation:**
- [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) - Official PyTorch blog (accessed 2025-01-13)
- [PyTorch CUDA Graphs API](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs) - API documentation

**NVIDIA Documentation:**
- [Constant Time Launch for Straight-Line CUDA Graphs](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/) - NVIDIA Developer Blog (accessed 2025-01-13)
- [Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/) - NVIDIA CUDA Graphs introduction
- [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) - NVIDIA LLM optimization guide (accessed 2025-01-13)

**Research & Community:**
- [arXiv: Robust Compiler Support for CUDA Graphs in PyTorch](https://arxiv.org/html/2503.19779v1) - PyGraph research (March 2025)
- [Speed, Python: Pick Two - How CUDA Graphs Enable Fast Python](https://fireworks.ai/blog/speed-python-pick-two-how-cuda-graphs-enable-fast-python-code-for-deep-learning) - Fireworks AI blog
- [Reddit r/CUDA: CUDA Graphs vs Kernel Fusion](https://www.reddit.com/r/CUDA/comments/1o2fl3g/cuda_graphs_vs_kernel_fusion_are_we_solving_the/) - Community discussion

**Source Documents:**
- [vertex-ai-production/01-gpu-optimization-deep.md](../vertex-ai-production/01-gpu-optimization-deep.md) - Base GPU optimization knowledge

---

**Document Version**: 1.0
**Created**: 2025-01-13
**Word Count**: ~4,200 words / 450+ lines
**Knowledge Gaps Filled**: CUDA Graphs fundamentals, PyTorch integration, transformer inference optimization, ARR-COC kernel optimization patterns
