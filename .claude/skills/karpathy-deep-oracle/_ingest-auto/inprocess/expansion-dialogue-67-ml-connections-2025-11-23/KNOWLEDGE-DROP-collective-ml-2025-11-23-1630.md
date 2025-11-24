# KNOWLEDGE DROP: Collective Intelligence in ML

**Date**: 2025-11-23 16:30
**File Created**: ml-morphogenesis/05-collective-intelligence-ml.md
**Lines**: ~700
**Status**: COMPLETE

---

## Summary

Created comprehensive guide to collective intelligence in machine learning, covering:

1. **Ensemble Methods** - Voting, bagging, boosting as collective decision-making
2. **Mixture of Experts (MoE)** - Learned routing with sparse expert activation
3. **Swarm Intelligence** - PSO for neural network training
4. **Collective Decision Networks** - Voting layers and consensus mechanisms
5. **TRAIN STATION Unification** - Collective = MoE = Ensemble = Swarm = Cells
6. **ARR-COC Connection** - Multi-agent relevance scoring

---

## Key TRAIN STATION

**The Grand Unification**:

```
Collective = MoE = Ensemble = Swarm = Cells

All are: weighted sum of specialists
         output = sum_i w_i(x) * f_i(x)
```

The isomorphisms:
- **Ensemble = MoE** (with uniform gating)
- **MoE = Attention** (with experts as values)
- **Swarm = Distributed optimization**
- **Cells in tissue = Experts in MoE**

---

## Code Implementations

### 1. EnsembleCollective
- Multiple models with learnable or fixed weights
- Diversity loss for negative correlation learning
- Uncertainty via inter-model variance

### 2. MixtureOfExperts
- TopKGating with load balancing
- Sparse expert activation
- Auxiliary loss for balanced routing

### 3. SwitchTransformerLayer
- Top-1 routing (simpler, similarly effective)
- Capacity factor and token dropping
- Router z-loss for stability

### 4. ParticleSwarmOptimizer
- Gradient-free neural network training
- Personal best and global best tracking
- Hybrid PSO-gradient approach

### 5. CollectiveRelevanceScorer
- MoE-style relevance for ARR-COC
- Different experts for faces/text/objects/scene
- Interpretable expert contributions

---

## Performance Insights

**MoE Benefits**:
- 4x pre-train speedup (Switch Transformers)
- Sparse activation = O(top_k) not O(n_experts)
- Scales better than dense models

**MoE Challenges**:
- High VRAM (all experts loaded)
- Fine-tuning prone to overfitting
- Communication overhead in distributed

**PSO Benefits**:
- No gradients needed
- Good for non-differentiable objectives
- Avoids local minima

---

## Sources Cited

- arXiv:2111.14377 - Collective Intelligence for Deep Learning (Ha & Tang, 2021)
- HuggingFace Blog - Mixture of Experts Explained (2023)
- arXiv:2101.03961 - Switch Transformers
- arXiv:1701.06538 - Outrageously Large Neural Networks
- arXiv:2006.16668 - GShard

---

## ARR-COC Application

**Multi-agent relevance scoring**:
- Different experts specialize in different content types
- Router learns to consult appropriate experts per region
- Sparse activation for efficiency
- Interpretable: can see which expert drove the relevance score

This matches brain processing:
- FFA for faces
- Visual word form area for text
- Distributed specialists, not monolithic scorer

---

## Next Steps

- Implement CollectiveRelevanceScorer in ARR-COC
- Experiment with number of experts (4-8 recommended)
- Monitor expert utilization during training
- Use auxiliary loss to prevent expert collapse
