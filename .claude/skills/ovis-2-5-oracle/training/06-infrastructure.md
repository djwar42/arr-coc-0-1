# Training Infrastructure

**Category**: Training
**Related**: [00-overview.md](00-overview.md)

## Hardware

**Setup**: 20-40 nodes × 8× A100-80GB
**Total**: 160-320 GPUs
**Time**: 2-3 weeks for full curriculum

## Optimization

**DeepSpeed ZeRO-3**: Memory efficiency
**Flash Attention 2**: 2-3× speedup
**Mixed Precision**: bfloat16
**Gradient Checkpointing**: Memory savings
**Data Packing**: 3-4× throughput

## Parallelism

**Data Parallel (DP)**: Replicate across GPUs
**Tensor Parallel (TP)**: Split large layers
**Context Parallel (CP)**: Split long sequences

## Speed

**P2 Training**: ~400K steps in 10-12 days
**Throughput**: ~70B tokens/day multimodal
