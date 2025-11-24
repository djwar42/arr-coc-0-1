# Training Optimizations

**See**: `RESEARCH/DeepSeekOCR/TRAINING.md` lines 800-1000

## Mixed Precision

**bfloat16** throughout:
- Activations: bf16
- Gradients: bf16
- Optimizer states: fp32
- **Speedup**: ~2× faster

## Flash Attention 2

Standard attention: O(N²) memory
Flash Attention: O(N) memory

**Benefit**: Longer sequences, larger batches

## Pipeline Parallelism

4-stage pipeline across GPUs:
1. DeepEncoder (SAM+CLIP)
2. Projector
3. MoE layers 0-5
4. MoE layers 6-11 + head

**Benefit**: Distribute memory, increase batch size

## Gradient Checkpointing

Trade compute for memory:
- Don't store all activations
- Recompute during backward
- **Result**: 40% memory savings

## Data Loading

- Parallel data loading (16 workers)
- Prefetching next batch
- Pin memory for fast GPU transfer
- **Throughput**: 70-90B tokens/day

**See TRAINING.md** for infrastructure details!
