# KNOWLEDGE DROP: Recurrent Predictive Networks

**Date**: 2025-11-23 16:00
**File**: ml-predictive-coding/04-recurrent-predictive.md
**Lines**: ~700

---

## What Was Added

Comprehensive guide to recurrent predictive networks combining RNNs with predictive coding principles.

### Key Content

1. **PredNet Architecture** - Complete PyTorch implementation of Lotter et al.'s deep predictive coding network
2. **PredRNN/PredRNN-V2** - Spatiotemporal LSTMs with zigzag memory flow
3. **Video Prediction Training** - BPTT, scheduled sampling, multi-loss training
4. **Performance Optimization** - Gradient checkpointing, mixed precision, memory efficiency

### Code Implementations

- `ConvLSTMCell` - Convolutional LSTM for spatial-temporal processing
- `PredNetLayer` - Single predictive coding layer with prediction/error
- `PredNet` - Complete hierarchical predictive network
- `SpatioTemporalLSTMCell` - Dual memory (C + M) cell
- `PredRNN` - Spatiotemporal predictive learning
- `MultiScalePredictiveNetwork` - Different timescales per layer

### TRAIN STATION Unification

**Recurrent = Temporal = Sequence = Memory**

All four concepts solve the same problem:
- Compress history into state
- Use state to predict future
- Update based on errors

This connects to:
- Transformers (soft memory addressing)
- State Space Models (structured RNN)
- Active Inference (free energy over time)

### ARR-COC-0-1 Applications

- `TemporalRelevancePredictor` - Predict token relevance over time
- `PredictiveTokenRouter` - Route with error-based updates
- `TemporalContextRelevance` - Build context for better allocation

---

## Sources

**Papers (5400+ citations total):**
- Lotter et al. 2016 - PredNet (1285 citations)
- Wang et al. 2017/2022 - PredRNN (TPAMI)
- Millidge et al. 2024 - Temporal PC
- Ali et al. 2022 - PC as energy efficiency

**Code:**
- [coxlab/prednet](https://github.com/coxlab/prednet)
- [thuml/predrnn-pytorch](https://github.com/thuml/predrnn-pytorch)

---

## Index Updates Needed

Add to ml-predictive-coding section:
- `04-recurrent-predictive.md` - PredNet, PredRNN, temporal prediction

Tags: video-prediction, convlstm, prednet, predrnn, temporal-memory
