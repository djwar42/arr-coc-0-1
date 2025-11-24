# Vision Encoder Compression Ratios

**Domain**: Vision-Language Model Efficiency
**Focus**: Compression techniques, accuracy vs efficiency trade-offs, task-specific limits
**Date**: 2025-01-31

---

## Overview

Vision encoder compression reduces the number of visual tokens passed to the language model, improving inference speed and reducing memory usage. The key question: **How much can we compress without hurting accuracy?**

**Input → Output Transformation**:
```
224×224 image (ViT-L/14)
→ 16×16 patches = 256 tokens
→ Compression (various techniques)
→ 64-256 compressed tokens
→ Language model
```

**Why Compress?**:
- **Faster LLM prefill**: Fewer visual tokens = faster context processing
- **Lower memory**: Reduced KV cache for long conversations
- **Better throughput**: More requests fit in GPU memory
- **Edge deployment**: Smaller models for resource-constrained devices

**Compression Ratio Notation**:
- **4× compression**: 256 tokens → 64 tokens
- **16× compression**: 256 tokens → 16 tokens
- **No compression (1×)**: 256 tokens → 256 tokens (identity)

---

## Compression Techniques Overview

### 1. Spatial Pooling (Simple, Fast)

**Method**: Average or max pooling over spatial dimensions

**Example** (2×2 pooling on 16×16 grid):
```
16×16 tokens → 8×8 tokens = 64 tokens
Compression ratio: 4×
```

**Pros**:
- Extremely fast (no learned parameters)
- Deterministic and interpretable

**Cons**:
- Loses fine-grained details
- No adaptive behavior (treats all regions equally)

### 2. Attention-Based Compression (Learned, Flexible)

**Method**: Use learned queries to aggregate vision features via cross-attention

**Examples**:
- **Q-Former** (BLIP-2): 32-64 learned queries
- **Perceiver Resampler** (Flamingo): 64-128 learned queries

**Compression Ratio**:
- 256 tokens → 32 queries = **8× compression**
- 256 tokens → 64 queries = **4× compression**

**Pros**:
- Learned to preserve task-relevant information
- Adaptive (different queries attend to different regions)

**Cons**:
- Slower than pooling (~20-40ms overhead)
- Requires training

### 3. Token Pruning (Dynamic, Query-Aware)

**Method**: Remove less important tokens based on attention scores or learned policies

**Examples**:
- **FastVLM**: Prune 75% of tokens (576 → 144)
- **HIVTP**: Attention-guided pruning (up to 26.5% latency reduction)
- **ToMe** (Token Merging): Merge similar tokens

**Compression Ratio**:
- 576 tokens → 144 tokens = **4× compression**
- 256 tokens → 64 tokens = **4× compression**

**Pros**:
- Preserves high-attention regions
- Query-aware (different questions prune differently)
- Minimal accuracy loss (<1-2%)

**Cons**:
- Requires careful tuning
- May remove task-critical tokens if pruning is too aggressive

### 4. Learned Compression (Neural)

**Method**: Train a compression network (e.g., MLP, small transformer) to map vision features to compressed representation

**Examples**:
- **DeepSeek-OCR**: 16× compression (4096 tokens → 256 tokens)
- **LLaVA MLP Projector**: 1× compression (no reduction, just projection)

**Compression Ratio**: Varies (typically 2-16×)

**Pros**:
- Can learn complex compression strategies
- Task-specific optimization

**Cons**:
- Requires training
- Risk of information bottleneck if too aggressive

---

## Compression Ratio Benchmarks

### 4× Compression (256 → 64 tokens)

**Techniques**:
- 2×2 spatial pooling
- Q-Former with 64 queries
- Token pruning (keep top 25%)

**Accuracy Impact**:
- **VQAv2**: -0.5% to -1.2% (minimal)
- **COCO Captioning**: -0.8 CIDEr (negligible)
- **GQA**: -0.3% to -0.9%

**Latency Improvement**:
- **LLM prefill**: 20-30% faster
- **TTFT**: 12-18% reduction
- **Memory**: 15-20% KV cache reduction

**Verdict**: **Safe for most tasks** ✅

**Example**: BLIP-2 with 32 queries achieves near-parity with full resolution on VQAv2 (65.0% vs 65.4%)

---

### 8× Compression (256 → 32 tokens)

**Techniques**:
- 2×2×2 spatial pooling (three stages)
- Q-Former with 32 queries (BLIP-2 default)
- Aggressive token pruning

**Accuracy Impact**:
- **VQAv2**: -1.5% to -2.5% (acceptable)
- **COCO Captioning**: -2.0 to -3.5 CIDEr
- **GQA**: -2.0% to -3.2%
- **Visual reasoning**: -3% to -5% (more sensitive)

**Latency Improvement**:
- **LLM prefill**: 35-50% faster
- **TTFT**: 20-30% reduction
- **Memory**: 30-40% KV cache reduction

**Verdict**: **Good trade-off for general VQA** ⚖️

**Example**: BLIP-2 achieves 65.0% on VQAv2 with 32 queries vs ~67% with 256 tokens (full resolution)

---

### 16× Compression (256 → 16 tokens)

**Techniques**:
- Extreme spatial pooling
- Very small Q-Former (16 queries)
- Learned compression networks

**Accuracy Impact**:
- **VQAv2**: -4% to -7% (significant)
- **COCO Captioning**: -5 to -8 CIDEr
- **GQA**: -5% to -8%
- **Visual reasoning**: -8% to -12%
- **OCR tasks**: -15% to -25% (severe degradation)

**Latency Improvement**:
- **LLM prefill**: 50-65% faster
- **TTFT**: 30-45% reduction
- **Memory**: 50-60% KV cache reduction

**Verdict**: **Too aggressive for most tasks** ⚠️

**Exception**: DeepSeek-OCR uses 16× compression (4096 → 256) successfully because it starts with much higher resolution (1024×1024)

---

### 64× Compression (1024 → 16 tokens)

**Techniques**:
- Only viable with multi-stage compression
- Example: Spatial pooling + learned compression + token merging

**Accuracy Impact**:
- **VQAv2**: -12% to -20% (unacceptable)
- **COCO Captioning**: -15 to -25 CIDEr
- **Fine-grained tasks**: Catastrophic failure

**Latency Improvement**:
- **LLM prefill**: 70-80% faster
- **TTFT**: 50-60% reduction

**Verdict**: **Not viable for general VLMs** ❌

**Exception**: Extremely specialized tasks where coarse visual understanding suffices

---

## Task-Specific Compression Limits

### VQA (Visual Question Answering)

**Compression Tolerance**: **4-8× (64-128 tokens)**

**Reasoning**:
- Most VQA questions focus on 1-2 objects
- Coarse spatial understanding often sufficient
- Example: "What color is the car?" doesn't need fine-grained detail

**Benchmark Results**:
- **4× compression**: -0.5% to -1.5% on VQAv2
- **8× compression**: -2% to -3% on VQAv2
- **16× compression**: -5% to -8% on VQAv2

**Optimal**: **32-64 tokens** (8-4× compression)

**Sources**:
- BLIP-2 ablations: 32 queries achieve 65.0% on VQAv2
- LLaVA ablations: 576 tokens (no compression) achieve 66.1%

---

### Image Captioning

**Compression Tolerance**: **4-8× (64-128 tokens)**

**Reasoning**:
- Captions describe overall scene, not fine details
- Holistic understanding more important than pixel-level precision

**Benchmark Results** (COCO Captioning, CIDEr score):
- **4× compression**: -1 to -2 CIDEr (140 → 138-139)
- **8× compression**: -3 to -5 CIDEr (140 → 135-137)
- **16× compression**: -8 to -12 CIDEr (140 → 128-132)

**Optimal**: **64-128 tokens** (4-8× compression)

---

### Visual Reasoning (GQA, VCR)

**Compression Tolerance**: **2-4× (128-256 tokens)**

**Reasoning**:
- Requires understanding spatial relationships
- "What is to the left of X?" needs positional information
- More sensitive to compression than simple VQA

**Benchmark Results** (GQA):
- **2× compression**: -0.3% to -0.8%
- **4× compression**: -1.5% to -2.5%
- **8× compression**: -4% to -6%

**Optimal**: **128-256 tokens** (2-4× compression)

**Critical**: Spatial pooling is problematic (loses positional info). Use attention-based compression.

---

### OCR & Document Understanding

**Compression Tolerance**: **1-2× (512-1024 tokens)** — **VERY SENSITIVE**

**Reasoning**:
- Requires reading fine-grained text
- Even mild compression (4×) causes 10-15% accuracy drop
- Needs high-resolution inputs (1024×1024 or higher)

**Benchmark Results** (DocVQA, TextVQA):
- **2× compression**: -5% to -10%
- **4× compression**: -15% to -25%
- **8× compression**: -30% to -50% (catastrophic)

**Optimal**: **512-1024 tokens** (minimal compression)

**Solution**: DeepSeek-OCR uses 1024×1024 input (4096 tokens) → 16× compression → 256 tokens (still 2× more than standard VLMs)

---

### Fine-Grained Recognition (Birds, Cars)

**Compression Tolerance**: **2-4× (128-256 tokens)**

**Reasoning**:
- Requires distinguishing subtle visual differences
- "Is this a Robin or a Sparrow?" needs detailed features
- Color, texture, shape all matter

**Benchmark Results** (CUB-200 birds):
- **2× compression**: -2% to -4%
- **4× compression**: -5% to -8%
- **8× compression**: -12% to -18%

**Optimal**: **128-256 tokens** (2-4× compression)

---

## Compression Method Comparison

### Spatial Pooling

**Compression Ratios**: 4×, 16×, 64× (powers of 2)

**Accuracy (VQAv2)**:
- **4× (64 tokens)**: 64.2% (-1.2% vs no compression)
- **16× (16 tokens)**: 59.8% (-5.6%)

**Latency** (LLaVA-13B on A100):
- **4×**: TTFT 175ms (-20% vs 220ms baseline)
- **16×**: TTFT 155ms (-30%)

**Pros**: Fast, no training
**Cons**: Uniform compression (ignores content)

---

### Q-Former (BLIP-2)

**Compression Ratios**: 4-8× (32-64 queries)

**Accuracy (VQAv2)**:
- **32 queries (8× compression)**: 65.0%
- **64 queries (4× compression)**: 65.4%
- **128 queries (2× compression)**: 65.6%

**Latency Overhead**: +25-40ms (cross-attention)

**Pros**: Learned, adaptive, minimal accuracy loss
**Cons**: Slower than pooling

**Source**: BLIP-2 paper ablations (Table 8)

---

### Token Pruning (FastVLM)

**Compression Ratios**: 4× (576 → 144 tokens)

**Accuracy (VQAv2)**:
- **4× compression**: 65.8% (-0.3% vs no pruning)

**Latency** (LLaVA-13B on A100):
- **TTFT**: 170ms (-23% vs 220ms)

**Pros**: Query-aware, minimal loss, fast
**Cons**: Requires tuning

**Source**: FastVLM paper (CVPR 2025)

---

### Learned Compression (DeepSeek-OCR)

**Compression Ratios**: 16× (4096 → 256 tokens)

**Accuracy (DocVQA)**:
- **16× compression**: 72.3% (comparable to 1024-token baselines)

**Key**: Starts with 4× higher resolution (1024×1024 vs 336×336)

**Pros**: Can handle extreme compression if input resolution is high
**Cons**: Requires high-res input

**Source**: DeepSeek-OCR paper

---

## Memory Impact

### KV Cache Scaling

**Formula**: KV cache size = 2 × num_layers × hidden_dim × num_tokens × batch_size

**Example** (LLaVA-13B, 40 layers, hidden_dim=5120, FP16):
- **576 visual tokens**: 2 × 40 × 5120 × 576 × 2 bytes = **471 MB per request**
- **144 visual tokens** (4× compression): **118 MB per request** (75% reduction)
- **32 visual tokens** (18× compression): **26 MB per request** (94% reduction)

**Multi-Turn Impact** (10-turn conversation, 50 tokens per turn):
- **No compression**: 471 MB (visual) + 102 MB (text) = **573 MB**
- **4× compression**: 118 MB (visual) + 102 MB (text) = **220 MB** (62% reduction)
- **18× compression**: 26 MB (visual) + 102 MB (text) = **128 MB** (78% reduction)

**Batch Serving** (batch size=16, 576 visual tokens):
- **No compression**: 471 MB × 16 = **7.5 GB KV cache**
- **4× compression**: 118 MB × 16 = **1.9 GB KV cache**

**Takeaway**: Compression enables **4-8× larger batch sizes** (more throughput)

---

## Best Practices

### Choosing Compression Ratio by Task

**General Guidelines**:
1. **Simple VQA** → 8× compression (32-64 tokens)
2. **Visual reasoning** → 4× compression (128-144 tokens)
3. **Fine-grained recognition** → 2-4× compression (128-256 tokens)
4. **OCR/Documents** → 1-2× compression (512-1024 tokens)
5. **Video understanding** → 8-16× compression (16-32 tokens per frame)

### Compression Technique Selection

**Latency-critical** (robotics, real-time):
- Use **spatial pooling** or **token pruning**
- Avoid Q-Former (adds 25-40ms)

**Accuracy-critical** (medical, fine-grained):
- Use **Q-Former** or **learned compression**
- Minimal compression (2-4×)

**Balanced** (general deployment):
- Use **4× compression** (Q-Former with 64 queries or FastVLM pruning)
- Achieves <1% accuracy loss with 20-30% latency improvement

### Multi-Resolution Strategy

**Adaptive compression based on query complexity**:
- Simple questions ("What color?") → 8× compression
- Complex questions ("Why is X doing Y?") → 2× compression

**Implementation**: Query classifier → dynamic compression ratio

**Expected Gains**: 30-50% average latency reduction with <1% accuracy loss

---

## Future Directions

### 1. Query-Aware Compression

**Idea**: Compress differently based on the question

**Example**:
- "What color is the car?" → Compress heavily (focus on car region)
- "What is the relationship between X and Y?" → Compress less (preserve spatial info)

**Potential**: 2-4× additional speedup with accuracy preservation

**Related Work**: ARR-COC (this project!) explores query-aware relevance realization

---

### 2. Hierarchical Compression

**Idea**: Multi-level compression (coarse to fine)

**Example**:
- Level 1: 16 tokens (coarse scene understanding)
- Level 2: 64 tokens (region-specific details)
- Level 3: 256 tokens (fine-grained features, only if needed)

**Potential**: 50-70% latency reduction for most queries (early exit at Level 1-2)

---

### 3. Cross-Attention Pruning

**Idea**: Prune visual tokens that receive low attention from LLM

**Challenges**:
- Requires online pruning during generation
- May discard tokens needed for later reasoning steps

**Potential**: 30-50% memory reduction in multi-turn conversations

---

## Summary: Compression Recommendations

### Production Deployment

**Interactive Applications** (<200ms TTFT):
- **Compression ratio**: 4-8×
- **Technique**: FastVLM pruning or Q-Former (64 queries)
- **Accuracy loss**: <1-2%
- **Latency gain**: 20-30%

**Batch Serving** (maximize throughput):
- **Compression ratio**: 8-16×
- **Technique**: Q-Former (32 queries) or aggressive pruning
- **Accuracy loss**: 2-5%
- **Throughput gain**: 4-8× (more requests fit in memory)

**Accuracy-Critical** (medical, legal):
- **Compression ratio**: 1-2×
- **Technique**: Minimal pooling or learned compression
- **Accuracy loss**: <0.5%
- **Latency gain**: 10-15%

### Compression Ratio Sweet Spots

**4× compression (64-144 tokens)**:
- **Accuracy**: 98-99% of full resolution
- **Latency**: 75-80% of full resolution
- **Use case**: General-purpose VLMs

**8× compression (32-64 tokens)**:
- **Accuracy**: 95-97% of full resolution
- **Latency**: 65-70% of full resolution
- **Use case**: Simple VQA, batch serving

**16× compression (16-32 tokens)**:
- **Accuracy**: 85-92% of full resolution
- **Latency**: 55-60% of full resolution
- **Use case**: Video understanding (per-frame compression)

### Key Insights

1. **4× compression is the sweet spot** for most applications
2. **Q-Former** (learned compression) outperforms spatial pooling for same compression ratio
3. **Token pruning** (FastVLM, HIVTP) achieves best latency/accuracy trade-off
4. **Task sensitivity varies**: OCR >> visual reasoning > VQA > captioning
5. **Memory savings enable larger batch sizes** (4-8× more throughput)

---

**References**:
- BLIP-2 paper: Q-Former ablations (32-128 queries)
- FastVLM paper (CVPR 2025): Token pruning benchmarks
- DeepSeek-OCR: 16× compression with high-resolution input
- HIVTP paper (arXiv 2509.23663): Attention-guided pruning
- LLaVA ablation studies: Token budget experiments
- Perceiver papers: Learned compression via cross-attention

**Last Updated**: 2025-01-31
