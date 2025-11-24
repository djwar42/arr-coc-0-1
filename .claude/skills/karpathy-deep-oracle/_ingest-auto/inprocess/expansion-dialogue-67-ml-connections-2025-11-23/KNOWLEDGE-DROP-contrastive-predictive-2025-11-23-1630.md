# KNOWLEDGE DROP: Contrastive Predictive Coding

**Date**: 2025-11-23 16:30
**Part**: 9
**Status**: COMPLETE

## Summary

Created comprehensive knowledge file on Contrastive Predictive Coding (CPC) and InfoNCE loss, including:
- Full CPC framework architecture and theory
- InfoNCE loss derivation and mutual information connection
- Complete PyTorch implementations (CPC model, InfoNCE loss, CLIP model)
- TRAIN STATION unification: CPC = CLIP = SimCLR = Active Inference
- Performance optimizations and memory-efficient implementations
- ARR-COC connection: Contrastive relevance learning for token allocation

## File Created

**Path**: `ml-predictive-coding/02-contrastive-predictive.md`
**Lines**: ~700 lines
**Sections**: 9 major sections

## Key Content

### Implementations Provided
1. **CPCModel** - Complete CPC with encoder, autoregressive model, and prediction heads
2. **InfoNCELoss** - Standalone loss module with paired/unpaired negative modes
3. **CLIPModel** - Simplified CLIP with vision and text encoders
4. **ContrastiveRelevanceModule** - ARR-COC integration for token relevance

### TRAIN STATION Found
**CPC = CLIP = SimCLR = Prediction = Intelligence**

All contrastive methods optimize the same objective:
```
L = -log p(positive | context) / sum_i p(sample_i | context)
```

This is:
- InfoNCE loss
- Cross-entropy over similarities
- Lower bound on mutual information
- Free energy minimization

### Key Insights
- InfoNCE frames prediction as classification (K+1 way)
- MI bound capped at log(N) for N negatives
- Temperature controls distribution sharpness
- Larger batch = more negatives = tighter bound

## Sources Cited

**Papers:**
- van den Oord et al. 2018 (CPC original)
- Radford et al. 2021 (CLIP)
- Henaff et al. 2020 (Data-Efficient CPC)

**Implementations:**
- RElbers/info-nce-pytorch
- openai/CLIP
- mlfoundations/open_clip

**Web Resources:**
- Medium tutorials on InfoNCE
- W&B contrastive learning guides
- IJCAI/NeurIPS papers on MI estimation

## ARR-COC Connection (10%)

Contrastive learning applies directly to relevance computation:
- Relevant tokens should PREDICT the task/query
- This is exactly what CPC optimizes
- InfoNCE naturally selects "informative" samples

Implementation provided: `ContrastiveRelevanceModule` with:
- `compute_contrastive_relevance()` - similarity-based scores
- `contrastive_relevance_loss()` - train from important token masks
- `allocate_tokens_contrastively()` - budget allocation

## Verification

- File exists and is readable
- Contains all required sections
- Includes working PyTorch code
- Citations properly formatted
- TRAIN STATION clearly identified
