# Knowledge Drop: Multi-Mask Output and IoU Prediction

**Date**: 2025-11-20 21:50
**Part**: 11 of SAM General Expansion
**File Created**: `sam-general/10-multi-mask-iou-prediction.md`
**Lines**: 962

## Summary

Created comprehensive documentation on SAM's multi-mask output and IoU prediction system, covering:

1. **Multi-Mask Output Overview** (~120 lines)
   - Why SAM predicts 3 masks per prompt
   - Architecture with learnable mask tokens
   - Single vs multi-mask inference modes

2. **Ambiguity Handling** (~140 lines)
   - The ambiguity problem in segmentation
   - Minimum-loss training objective
   - Hierarchical granularity (whole/part/subpart)
   - Training for ambiguity with 11 rounds per mask

3. **IoU Prediction Head** (~130 lines)
   - Architecture: 3-layer MLP
   - MSE training against actual IoU
   - Token flow through transformer
   - Practical performance (~0.82-0.88 correlation)

4. **Mask Selection Strategies** (~120 lines)
   - Automatic selection (argmax IoU)
   - Oracle selection (analysis)
   - Application-specific strategies
   - Ensemble methods

5. **Training Objectives** (~100 lines)
   - Combined focal + dice loss
   - Minimum-loss across candidates
   - Stability score computation

6. **Inference Modes** (~110 lines)
   - Interactive mode with refinement
   - Automatic everything mode
   - Batch processing
   - SAM 2 video with occlusion

7. **ARR-COC Integration** (~90 lines, ~10%)
   - Attention granularity analysis
   - Mask quality estimation
   - Hierarchical loss for training

## Key Technical Insights

### Multi-Mask Design
- **3 masks** captures typical whole/part/subpart hierarchy
- Learnable mask tokens generate dynamic linear classifiers
- Each token produces independent per-pixel predictions

### Minimum-Loss Training
```python
# Key insight: only best-matching mask gets gradients
min_loss, min_idx = losses.min(dim=1)
```
This operationalizes "any valid mask" without penalizing alternatives.

### IoU Prediction
- Trained with MSE against actual IoU
- Enables automatic mask ranking
- ~85% accuracy on best-mask selection

### Practical Thresholds
- pred_iou_thresh: 0.88 (quality)
- stability_score_thresh: 0.95 (consistency)

## Sources Cited

**Papers:**
- SAM (Kirillov et al., 2023)
- SAM 2 (Ravi et al., 2024)

**Web Research:**
- Medium article on SAM architecture
- Storrs.io decoder explanation
- GitHub issues on IoU prediction
- Ultralytics documentation

## Integration Value

Multi-mask output provides natural supervision for:
- Multi-scale attention training
- Granularity-aware loss functions
- Quality estimation for generated masks
- Hierarchical feature learning

The IoU prediction head offers a self-supervised quality metric that can transfer to ARR-COC mask evaluation.
