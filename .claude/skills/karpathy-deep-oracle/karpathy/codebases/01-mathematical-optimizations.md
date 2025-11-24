# Mathematical Optimizations in nanoGPT

Research findings on Karpathy's mathematical optimization choices.

---

## 1. Attention Scaling: 1/√(d_k)

**Formula**: `scores = (Q @ K.T) / sqrt(d_k)`

**Mathematical Reasoning** (from research):

### Why Scale?

**Problem**: Dot products grow in magnitude proportionally to dimension size
- For random vectors with zero mean and unit variance
- Expected value of dot product between two d-dimensional vectors ≈ d
- Without scaling, scores get pushed into extreme regions of softmax

**Softmax Saturation**:
```
When scores are large (e.g., 10, 15, 20):
- softmax(x) ≈ [0, 0, 1, 0] (near-one-hot)
- Gradients ≈ 0 (vanishing gradients)
- Model can't learn effectively

When scores are scaled (e.g., 0.5, 1.2, 1.8):
- softmax(x) ≈ [0.2, 0.35, 0.45] (smooth distribution)
- Gradients flow properly
- Model learns well
```

**Variance Analysis**:
- Dot product of two random d-dimensional vectors has variance ≈ d
- Dividing by sqrt(d) normalizes variance to ≈ 1
- Keeps scores in reasonable range for softmax

**Why sqrt specifically?**
- Variance of sum of d independent random variables = d * variance_each
- Standard deviation = sqrt(variance) = sqrt(d)
- Dividing by sqrt(d) gives unit variance

### Code in nanoGPT
```python
# model.py line 67
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
```

**References**:
- "Attention is All You Need" paper (Vaswani et al., 2017)
- [Detailed explanation](https://uselessai.in/what-is-scaling-in-transformers-self-attention-you-ll-not-regret-reading-this-d37121f6644e)

---

## 2. Residual Projection Initialization: 0.02/√(2*n_layer)

**Formula**: `std = 0.02 / sqrt(2 * n_layer)`

**Mathematical Reasoning** (from GPT-2 paper + research):

### Why Scale by Layer Depth?

**Problem**: Residual connections accumulate variance
- Each layer adds: `x = x + f(x)`
- After N layers: variance grows by factor of N
- Deep networks (12-48 layers) accumulate too much signal

**Variance Accumulation**:
```
Layer 1: var(x₁) = var(x₀) + var(f(x₀))
Layer 2: var(x₂) = var(x₁) + var(f(x₁)) = var(x₀) + var(f(x₀)) + var(f(x₁))
...
Layer N: var(xₙ) ≈ var(x₀) + N * var(f(x))
```

**Solution**: Pre-scale residual contributions
- Scale residual layer init by 1/sqrt(N) where N = number of layers
- After N additions, total variance ≈ constant
- Math: N * (1/sqrt(N))² = N * (1/N) = 1

**GPT-2 Specific**:
- Base init std = 0.02 (empirically good for transformers)
- Two residual paths per block: attention + MLP
- Total residual layers = 2 * n_layer
- Scaling factor = 1/sqrt(2 * n_layer)

### Code in nanoGPT
```python
# model.py lines 143-145
for pn, p in self.named_parameters():
    if pn.endswith('c_proj.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
```

**Which layers?**
- `c_proj` in CausalSelfAttention (attention output projection)
- `c_proj` in MLP (feed-forward output projection)
- These are the layers that feed INTO residual connections

**References**:
- GPT-2 paper: "We scale the weights of residual layers at initialization by a factor of 1/√N"
- [Annotated GPT-2](https://amaarora.github.io/posts/2020-02-18-annotatedGPT2.html)

---

## 3. Weight Decay Only on 2D Parameters

**Rule**: Weight decay applied only to matrices (2D tensors), not biases/layernorms (1D)

**Mathematical Reasoning**:

### Why Selective Weight Decay?

**Weight Decay Purpose**: L2 regularization to prevent overfitting
- Formula: `loss = loss + λ * ||W||²`
- Penalizes large weights
- Encourages smaller, simpler models

**Parameter Types**:
```
2D Parameters (matrices):
- Linear layer weights: [out_features, in_features]
- Embedding weights: [vocab_size, embedding_dim]
- Represent actual transformations
- Can overfit by learning arbitrary mappings

1D Parameters (vectors):
- Biases: [out_features]
- LayerNorm gamma/beta: [hidden_dim]
- Represent shifts/scales
- Far fewer parameters, minimal overfit risk
```

**Why Skip 1D Parameters?**
1. **Scale of problem**: Matrices have O(d²) params, vectors have O(d)
   - For d=768: matrix ~590K params, vector ~768 params
   - Regularizing vectors doesn't meaningfully reduce capacity

2. **Different roles**:
   - Matrices: learn representations (prone to overfitting)
   - Biases/norms: adjust distributions (helper parameters)

3. **Empirical results**: Decaying all params harms performance
   - LayerNorm parameters need flexibility to normalize properly
   - Biases need freedom to shift activations

### Code in nanoGPT
```python
# model.py lines 268-274
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]
```

**Typical Split** (GPT-2 124M):
- Decayed params: ~85M (matrices)
- Non-decayed params: ~39M (biases, norms)

**References**:
- AdamW paper (Loshchilov & Hutter, 2017)
- Common practice in transformer training

---

## 4. Flash Attention Optimization

**Goal**: Compute exact attention 3x faster with less memory

**Mathematical/Algorithmic Innovation**:

### Standard Attention Problems

**Memory Bottleneck**:
```
Standard attention:
1. Compute QK^T: [batch, heads, seq_len, seq_len] - O(n²) memory
2. Apply softmax: keeps full attention matrix in memory
3. Multiply by V: [batch, heads, seq_len, seq_len] @ [batch, heads, seq_len, d_k]

For seq_len=2048, heads=12:
- Attention matrix: 2048 * 2048 * 12 = 50M elements per example
- With batch_size=64: 3.2B floats = 12.8GB just for attention scores!
```

**Recomputation Trade-off**:
- GPUs are compute-bound, not memory-bound (on modern GPUs)
- Memory access (HBM ↔ SRAM) is slower than computation
- Recomputing values can be faster than loading from memory

### Flash Attention Algorithm

**Key Innovation**: Tiling + online softmax + kernel fusion

**Tiling Strategy**:
```
Instead of:
  scores = Q @ K.T  # Load full Q, K, store full scores
  attn = softmax(scores)  # Load full scores, store full attn
  out = attn @ V  # Load full attn, V

Flash Attention:
  For each block of Q (tile):
    For each block of K, V (tile):
      1. Load Q_block, K_block from HBM to SRAM
      2. Compute scores_block = Q_block @ K_block.T
      3. Load V_block
      4. Accumulate softmax and output online
      5. Discard intermediate results
```

**Online Softmax** (key mathematical trick):
- Don't need full scores to compute softmax
- Can compute incrementally using numerical stability tricks
- Update running max and sum as we process tiles

**Benefits**:
- **Memory**: O(n) instead of O(n²) - never materialize full attention matrix
- **Speed**: ~3x faster due to optimized memory access patterns
- **Exact**: Mathematically identical results to standard attention

### Code in nanoGPT
```python
# model.py lines 62-64
if self.flash:
    # efficient attention using Flash Attention CUDA kernels
    y = torch.nn.functional.scaled_dot_product_attention(...)
```

**PyTorch Integration**:
- PyTorch 2.0+ provides `scaled_dot_product_attention`
- Automatically uses Flash Attention when available
- Falls back to standard attention otherwise

**CUDA Kernel Optimization**:
- Careful memory layout for coalesced access
- Shared memory for tiles
- Thread block synchronization
- Warp-level operations

**References**:
- Flash Attention paper (Dao et al., 2022)
- Flash Attention 2 (Dao, 2023)
- [Implementation walkthrough](https://www.stephendiehl.com/posts/flash_attention/)

---

## 5. Fused AdamW Optimizer

**Optimization**: Single CUDA kernel for entire optimizer step

**Standard AdamW** (multiple kernels):
```
1. Compute gradients (backward pass)
2. For each parameter group:
   - Update first moment: m = β₁*m + (1-β₁)*grad
   - Update second moment: v = β₂*v + (1-β₂)*grad²
   - Bias correction: m_hat = m/(1-β₁^t), v_hat = v/(1-β₂^t)
   - Weight decay: param = param * (1-lr*wd)
   - Update: param = param - lr*m_hat/(sqrt(v_hat) + ε)

Each operation: CPU → GPU kernel launch overhead
Total: ~6-8 kernel launches per parameter
```

**Fused AdamW** (single kernel):
```
Single CUDA kernel does all operations:
- Read grad, param, m, v from memory once
- Compute all updates in registers
- Write back param, m, v once

Eliminates kernel launch overhead
Reduces memory bandwidth by ~3-4x
~20-30% speedup on optimizer step
```

### Code in nanoGPT
```python
# model.py lines 281-285
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device_type == 'cuda'
extra_args = dict(fused=True) if use_fused else dict()
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
```

**When Available**:
- PyTorch 2.0+ on CUDA devices
- Automatically detected and used
- No code changes needed

**References**:
- PyTorch documentation
- NVIDIA Apex optimizer

---

## 6. Model FLOPs Utilization (MFU)

**Metric**: Ratio of achieved FLOPs to theoretical peak FLOPs

**Calculation** (from PaLM paper):

**FLOPs per Token**:
```
Forward pass: 2N (N = number of parameters)
Backward pass: 4N (approximately 2x forward)
Total per token: 6N

Additional attention FLOPs:
  12*L*H*Q*T per token
  where L=layers, H=heads, Q=head_dim, T=seq_len
```

### Code in nanoGPT
```python
# model.py lines 289-303
def estimate_mfu(self, fwdbwd_per_iter, dt):
    N = self.get_num_params()
    cfg = self.config
    L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
    flops_per_token = 6*N + 12*L*H*Q*T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    flops_achieved = flops_per_iter * (1.0/dt)
    flops_promised = 312e12  # A100 bfloat16 peak
    mfu = flops_achieved / flops_promised
    return mfu
```

**Interpretation**:
- MFU = 0.50 (50%): Excellent utilization, near optimal
- MFU = 0.30-0.40: Good, typical for well-optimized training
- MFU < 0.20: Poor, bottlenecks present

**Why Not 100%?**:
- Memory bandwidth limits
- Kernel launch overhead
- Data loading time
- Gradient accumulation steps
- Mixed precision conversions

**Target**: ~40-50% MFU is excellent for transformer training

**References**:
- PaLM paper Appendix B (FLOPs calculation)
- [Performance analysis](https://arxiv.org/abs/2204.02311)

---

## Summary of Optimizations

| Optimization | Purpose | Speedup/Benefit |
|--------------|---------|----------------|
| Attention Scaling (1/√d_k) | Prevent softmax saturation | Stable gradients, better training |
| Residual Init Scaling | Maintain variance across layers | Stable deep network training |
| Selective Weight Decay | Effective regularization | Better generalization |
| Flash Attention | Reduce memory, optimize access | 3x faster, O(n) memory |
| Fused AdamW | Reduce kernel launches | 20-30% optimizer speedup |
| PyTorch Compile | JIT fusion, optimization | 2x overall training speedup |

**Combined Effect**: nanoGPT achieves ~40-50% MFU on A100, comparable to highly optimized frameworks

---

**Research Sources**:
- "Attention is All You Need" (Vaswani et al., 2017)
- "Language Models are Unsupervised Multitask Learners" (GPT-2 paper)
- "Flash Attention" (Dao et al., 2022)
- "Decoupled Weight Decay Regularization" (AdamW paper)
- [Annotated GPT-2](https://amaarora.github.io/posts/2020-02-18-annotatedGPT2.html)
- [Scaling in Attention](https://uselessai.in/what-is-scaling-in-transformers-self-attention-you-ll-not-regret-reading-this-d37121f6644e)
