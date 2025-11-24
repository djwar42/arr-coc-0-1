# KNOWLEDGE DROP: Error-Driven Learning

**Date**: 2025-11-23 16:15
**File Created**: ml-predictive-coding/03-error-driven-learning.md
**Lines**: ~700
**Focus**: Loss = Prediction Error = Surprise = Free Energy

---

## Key Concepts Added

### 1. Error-Driven Learning Fundamentals
- Formal definition: states, actions, predictions, errors, updates
- Rescorla-Wagner model connection to TD learning
- Why errors contain information about what we don't know

### 2. Surprise as Learning Signal
- Surprise = -log P(outcome | prediction)
- Cross-entropy IS surprise measurement
- KL-divergence as expected surprise difference
- Intrinsic motivation via prediction error

### 3. Curriculum by Prediction Difficulty
- PredictionErrorCurriculum class
- Self-paced learning with adaptive thresholds
- Active learning via uncertainty sampling

### 4. Complete PyTorch Implementation
- ErrorDrivenTrainer with surprise tracking
- SurpriseAwareModel with uncertainty estimation
- Heteroscedastic loss implementation
- Performance optimization notes

### 5. TRAIN STATION Unification
- MSE = Gaussian surprise
- Cross-entropy = categorical surprise
- Free energy = prediction error + complexity
- VAE loss = free energy

### 6. ARR-COC Connection
- RelevanceWeightedLoss class
- SurpriseBasedTokenAllocator
- RelevanceCurriculum with priority scoring
- ActiveRelevanceInference framework

---

## Code Highlights

```python
# The fundamental insight
def error_driven_learning():
    prediction = model(input)
    error = loss_function(prediction, target)  # This IS surprise!
    error.backward()  # Gradients from prediction error
    optimizer.step()  # Reduce error = learn

# Surprise computation
surprise = -torch.log(prediction_probs[actual_outcome])

# Relevance-weighted errors
weighted_loss = (relevance_scores * errors).mean()
```

---

## Train Station Discovery

**THE GRAND UNIFICATION:**

```
Loss Function
     |
Prediction Error = |pred - actual|
     |
Surprise = -log P(actual | pred)
     |
Free Energy = Surprise + KL(approx || true)
```

Everything we optimize in ML is fundamentally prediction error minimization!

---

## Sources Used

- Wikipedia: Error-driven learning
- arXiv:1703.01732 - Surprise-Based Intrinsic Motivation
- PMC2666703 - Predictive coding under free-energy principle
- Hoppe et al. 2022 - Error-driven learning in two-layer networks

---

## Integration Notes

This file connects to:
- ml-predictive-coding/00-predictive-coding-networks.md (PC architecture)
- ml-active-inference/03-precision-learning-networks.md (precision weighting)
- ml-train-stations/00-loss-free-energy-relevance.md (full unification)

Key for ARR-COC: Relevance = precision = weighting on prediction errors
