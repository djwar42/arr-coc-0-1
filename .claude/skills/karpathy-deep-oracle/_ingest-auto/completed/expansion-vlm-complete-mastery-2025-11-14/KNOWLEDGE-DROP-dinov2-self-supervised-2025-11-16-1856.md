# KNOWLEDGE DROP: DINOv2 & Self-Supervised Vision

**Date**: 2025-11-16-1856
**Part**: PART 8
**File Created**: `vlm-mastery/07-dinov2-self-supervised.md`
**Lines**: ~730 lines

## What Was Learned

### Core Concepts
- **DINOv2** = self-supervised vision foundation model (no text needed!)
- **Self-distillation**: student network learns to match momentum-averaged teacher
- **Multi-crop training**: 2 global + 8 local crops per image for scale invariance
- **Curated data >> uncurated data**: 142M curated images outperform billion-scale web scrapes

### Key Technical Details

**Training Framework**:
- Momentum teacher (EMA, τ=0.996) provides stable targets
- Student processes all crops, teacher only global crops
- iBOT (masked image modeling) + Koleo regularization prevent collapse
- LayerScale + stochastic depth stabilize 1B parameter ViT training

**Model Variants**:
- ViT-S/14 (22M), ViT-B/14 (86M), ViT-L/14 (300M), ViT-g/14 (1B params)
- Patch size 14×14 → 16×16 grid for 224×224 images
- Each patch gets rich semantic feature (384-1536 dim)

**Dense Prediction Excellence**:
- **Segmentation**: ADE20K 84.5 mIoU (frozen linear probe!)
- **Depth**: NYU Depth v2 δ₁=97.1% (beats specialized depth models)
- **Emergent properties**: Attention heads discover objects without labels

### DINOv2 vs CLIP

**DINOv2 wins**:
- Dense tasks (segmentation, depth, keypoints) - significant gap
- Local semantic features - richer spatial detail
- Fine-grained recognition - breeds, species, medical imaging

**CLIP wins**:
- Zero-shot classification - text prompts ("a photo of X")
- Vision-language alignment - VQA, captioning
- Multimodal retrieval - text→image, image→text

**Hybrid approach**: Fuse DINOv2 + CLIP features for best of both worlds (Prismatic VLMs).

### Distributed Training (FSDP)

**DINOv2-g (1B params) on 64 A100s**:
- FSDP FULL_SHARD (ZeRO-3 equivalent)
- 31MB params + 125MB optimizer states + 31MB grads = 187MB per GPU
- Activations dominate: ~40GB per GPU
- Training time: ~2 weeks for 1.2B samples (1.2M iters × 1024 batch)

**Communication optimizations**:
- Overlap all-gather params with compute
- Overlap reduce-scatter grads with backward pass
- ~11,520 images/sec total throughput (64 GPUs × 180 images/sec)

### torch.compile Speedups

**Compilation benefits**:
- Attention kernel fusion (FlashAttention-2 automatic)
- LayerNorm + Linear fusion
- 1.4× speedup (45ms → 32ms per batch, A100, batch=32)

**Best practices**:
- Use `mode='max-autotune'` for offline compilation (try many kernels)
- Fixed shapes compile faster than dynamic
- Compile separate models for common resolutions (224, 384, 518)

### TPU Training

**TPU advantages**:
- Native BF16 support (better stability than FP16, no loss scaling)
- 256 TPU v5e chips → ~30k images/sec (3× faster than 64 A100s)
- DINOv2-g training: 3-4 days on TPU vs 2 weeks on GPU

**JAX/Flax implementation**:
- `jax.pmap` for data parallelism
- Layer-wise sharding for 1B param model
- BF16 activations, FP32 params/optimizer

### ARR-COC-0-1 Integration (10% section)

**Why DINOv2 for spatial relevance**:
- Dense patch features (14×14 pixels) vs CLIP's global features
- Natural semantic segmentation in attention heads
- Fine-grained relevance maps for token allocation

**Pattern**:
```python
# Extract dense features
visual_features = dinov2_dense(images)  # [B, H_patches, W_patches, 768]

# Compute spatial relevance (query-conditioned)
relevance_map = similarity(visual_features, query_embedding)  # [B, H_p, W_p]

# Allocate tokens: high relevance → 400 tokens, low → 64 tokens
token_allocation = map_relevance_to_budget(relevance_map)
```

**Hybrid approach**: Use CLIP text encoder for query understanding + DINOv2 dense features for spatial precision.

## Sources Cited

**Influential Files (4, 8, 16)**:
- [distributed-training/03-fsdp-vs-deepspeed.md](../distributed-training/03-fsdp-vs-deepspeed.md) - FSDP sharding for 1B param ViT
- [inference-optimization/03-torch-compile-aot-inductor.md](../inference-optimization/03-torch-compile-aot-inductor.md) - torch.compile speedups
- [alternative-hardware/03-tpu-programming-fundamentals.md](../alternative-hardware/03-tpu-programming-fundamentals.md) - TPU/JAX training

**ARR-COC-0-1**:
- [arr_coc/knowing.py](/Users/alfrednorth/Desktop/Code/arr-coc-ovis/RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/knowing.py) - Dense feature extraction for relevance

**Web Research (8 sources)**:
- [DINOv2 paper](https://arxiv.org/abs/2304.07193) (Oquab et al., 2023) - Main technical reference
- [DINO v1 paper](https://arxiv.org/abs/2104.14294) (Caron et al., 2021) - Multi-crop training, momentum teacher
- [Meta AI blog](https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/) - Training details, benchmarks
- [DINOv1 vs DINOv2](https://medium.com/@jimcanary/dinov1-vs-dinov2-evolution-of-self-supervised-vision-transformers-83dd60dd81d3) - Evolution comparison
- [CLIP to DINO paper](https://arxiv.org/abs/2310.08825) (Jiang et al., 2024) - Performance comparison
- [Prismatic VLMs](https://arxiv.org/abs/2402.07865) (Karamcheti et al., 2024) - Hybrid fusion approach
- [DINOv2 GitHub](https://github.com/facebookresearch/dinov2) - Code, models
- [DINOv2 website](https://dinov2.metademolab.com/) - Demos, visualizations

## Impact on VLM Engineering

**When to choose DINOv2**:
1. Dense prediction tasks (segmentation, depth, pose)
2. No text understanding needed (pure vision)
3. Fine-grained visual recognition
4. Local semantic detail matters

**When to fuse DINOv2 + CLIP**:
1. Building VLMs (want both text grounding + spatial precision)
2. Complex visual reasoning (need rich local features)
3. Have compute budget for dual encoders

**Training insights**:
- Data curation > data quantity (142M curated >> 1B+ uncurated)
- Momentum teacher is critical for self-supervised stability
- Multi-crop training essential for scale invariance
- FSDP handles 1B params efficiently across 64+ GPUs

## Next Steps

This knowledge enables:
- Understanding when DINOv2 beats CLIP (dense tasks)
- Implementing self-supervised training pipelines
- Choosing right vision encoder for VLM architecture
- Hybrid fusion strategies for best-of-both-worlds

**Follow-up topics**:
- PART 9: EVA-CLIP (billion-scale vision encoders)
- PART 7: CLIP deep dive (contrastive pretraining)
- Vision encoder fusion strategies (Prismatic patterns)
