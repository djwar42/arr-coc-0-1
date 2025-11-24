# KNOWLEDGE DROP: LLM Training Dynamics & GPU Optimization

**Runner**: PART 3
**File Created**: `karpathy/llm-gpu-integration/02-training-dynamics-gpu.md`
**Date**: 2025-02-03 16:45
**Lines**: ~850 lines
**Status**: ✓ Complete

---

## Summary

Created comprehensive guide covering gradient checkpointing (activation recomputation), pipeline parallelism (GPipe/PipeDream/Megatron), ZeRO optimizer stages 1-3, tensor parallelism, and 3D parallelism combining all techniques. Includes ARR-COC specific training optimizations for multi-stage training with texture arrays and relevance scorers.

---

## Key Topics Covered

### 1. Gradient Checkpointing (Activation Recomputation)
- **Memory problem**: Activations dominate memory (307 GB for GPT-3 vs 175 GB params)
- **Solution**: Checkpoint √L layers, recompute rest (9.6× memory reduction)
- **Trade-off**: 33% extra compute for 75% memory savings
- **Strategies**: Full checkpointing, selective attention checkpointing, block checkpointing
- **PyTorch**: `torch.utils.checkpoint.checkpoint()`
- **HuggingFace**: `model.gradient_checkpointing_enable()`

### 2. Pipeline Parallelism
- **Problem**: Naive pipeline has 75% bubble (idle time)
- **GPipe solution**: Micro-batching reduces bubble to 18.75% (16 microbatches, 4 GPUs)
- **PipeDream**: 1F1B schedule reduces memory (4 microbatches in flight vs 8)
- **Megatron-LM**: Virtual pipeline parallelism, interleaved schedules
- **Communication**: P2P between stages, requires NVLink for efficiency

### 3. ZeRO Optimizer (Zero Redundancy Optimizer)
- **Stage 1**: Partition optimizer states → 8× reduction (2.1 TB → 262 GB)
- **Stage 2**: Partition gradients → Additional savings (350 GB → 43.75 GB)
- **Stage 3**: Partition parameters → 86% total reduction (2.45 TB → 350 GB)
- **ZeRO-Infinity**: CPU/NVMe offloading for trillion-parameter models
- **Implementation**: DeepSpeed JSON config, PyTorch FSDP

### 4. Tensor Parallelism
- **Column-parallel**: Split output dimensions (QKV projections)
- **Row-parallel**: Split input dimensions (output projections, all-reduce)
- **Attention**: Split heads across GPUs (embarrassingly parallel)
- **MLP**: Column-parallel (h→4h) + Row-parallel (4h→h)
- **Communication**: 2 all-reduces per layer, requires NVLink (600 GB/s)

### 5. 3D Parallelism (Combining All Techniques)
- **Data Parallelism (DP)**: Replicate model across groups
- **Tensor Parallelism (TP)**: Split layers across GPUs (within node)
- **Pipeline Parallelism (PP)**: Split layers sequentially (across nodes)
- **Example**: GPT-3 175B on 64 GPUs = DP=8 × TP=4 × PP=2
- **Efficiency**: 86% overall (95% TP × 92% PP × 98% DP)

### 6. ARR-COC Multi-Stage Training
- **Stage 1**: Texture extraction with gradient checkpointing (13 channels)
- **Stage 2**: Relevance scorers with BF16 (opponent processing stability)
- **Stage 3**: Quality adapter with gradient accumulation
- **Future**: 3D parallelism for 70B+ VLMs (Qwen-VL 72B + ARR-COC 2B)

---

## Sources Cited

**NVIDIA Documentation:**
- NVIDIA NeMo Activation Recomputation (accessed 2025-02-03)

**DeepSpeed:**
- DeepSpeed ZeRO Tutorial (accessed 2025-02-03)

**Academic Papers:**
- NeurIPS 2024: Optimizing Intermediate Memory for Long Sequences
- Megatron-LM: Efficient Large-Scale Language Model Training

**Blog Posts:**
- Kaitchup: The Unreasonable Impact of Gradient Checkpointing
- Medium: Training Transformer Models Memory Challenges
- Hugging Face: Comprehensive Overview of Optimization Techniques

**Internal References:**
- cuda/01-memory-management-unified.md
- cuda/07-mixed-precision-training-internals.md
- vertex-ai-production/00-distributed-training-patterns.md

---

## Technical Highlights

**Memory Calculations:**
```
GPT-3 175B without optimization:
- Parameters: 350 GB
- Optimizer: 2.1 TB
- Activations: 307 GB
- Total: 2.76 TB per GPU (impossible!)

With ZeRO-3 + Gradient Checkpointing (8 GPUs):
- Parameters: 43.75 GB (÷8)
- Optimizer: 262.5 GB (÷8)
- Gradients: 43.75 GB (÷8)
- Activations: 38.4 GB (÷8 checkpoints)
- Total: 388 GB per GPU (fits!)
```

**Parallelism Selection:**
```python
# Decision tree
if model_size < 13B:
    use_data_parallelism()
elif model_size < 175B:
    use_dp_plus_tp()  # DP + TP (requires NVLink)
elif model_size < 1T:
    use_3d_parallelism()  # DP + TP + PP
else:
    use_3d_plus_zero_infinity()  # + CPU/NVMe offload
```

---

## Integration with Existing Knowledge

**Connects to:**
- Mixed precision training (FP16/BF16/FP8) from cuda/07-mixed-precision-training-internals.md
- GPU memory hierarchy (SRAM vs HBM) from cuda/01-memory-management-unified.md
- Distributed training patterns (DDP) from vertex-ai-production/00-distributed-training-patterns.md

**Complements:**
- FlashAttention kernel optimization (future PART 1)
- Architecture-GPU constraints (future PART 2)
- Inference KV cache optimization (future PART 4)

---

## ARR-COC Specific Insights

**Texture Array Memory Challenge:**
- 13 channels × 200 patches × 32 batch × 224² = 52 GB activations
- Solution: Checkpoint channel groups → 16 GB (69% reduction)

**Opponent Processing Stability:**
- Tension balancing requires BF16 (wide dynamic range)
- Alternative to FP16 for subtract/sigmoid operations

**Future Scaling:**
- 3D parallelism enables 70B+ VLM + ARR-COC components
- Pipeline: Stage 0 (Qwen-VL 0-39) | Stage 1 (Qwen-VL 40-79 + ARR-COC)
- Estimated: 74.8 GB per GPU on 128 A100s (444K tokens/sec)

---

**KNOWLEDGE DROP Complete** ✓
