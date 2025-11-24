# KNOWLEDGE DROP: Contextual Bandits & Personalization

**Date**: 2025-11-16 20:23
**Part**: PART 20
**File Created**: `cognitive-mastery/19-contextual-bandits-personalization.md`
**Lines**: ~700 lines
**Status**: SUCCESS ✓

---

## What Was Created

Comprehensive knowledge file on **Contextual Bandits & Personalization** covering:

1. **Contextual Bandit Problem Formulation** (~100 lines)
   - Formal definition with context, actions, rewards
   - Context types: user, action, hybrid features
   - Comparison with standard multi-armed bandits

2. **LinUCB Algorithm** (~120 lines)
   - Linear contextual bandits formulation
   - Upper confidence bound strategy
   - Regret analysis: O(d√(T log T))
   - Ridge regression online updates

3. **Neural Contextual Bandits** (~120 lines)
   - Deep representations beyond linearity
   - Neural LinUCB architecture
   - Thompson Sampling alternative
   - Neural architecture choices

4. **Personalization Applications** (~100 lines)
   - Content recommendation (news, videos)
   - Online advertising with budget constraints
   - Personalized pricing (dynamic pricing)
   - Medical treatment assignment

5. **Query-Aware Relevance Allocation** (~100 lines)
   - Contextual bandits for attention allocation
   - Personalized token budgets
   - Multi-patch allocation as combinatorial bandit
   - Expected compute savings: 30-50%

6. **Advanced Techniques** (~100 lines)
   - Hybrid LinUCB + neural features
   - Off-policy evaluation (IPS, doubly robust)
   - Fairness constraints in personalization
   - Non-stationarity & concept drift handling

7. **Integration with FSDP, ML Workloads, TPU** (~80 lines)
   - Distributed training of large-scale neural bandits (FSDP)
   - Production K8s patterns (CronJobs, serving deployments)
   - TPU-accelerated LinUCB with JAX (vectorized UCB computation)

8. **ARR-COC-0-1 Integration** (~60 lines)
   - Token allocation as contextual bandit
   - Query-aware relevance learning
   - Integration with opponent processing
   - Feedback loop: Knowing → Balancing → Attending (bandit) → Realizing

---

## Key Insights

### LinUCB: Optimism Under Uncertainty
- **UCB_a(x) = μ̂_a(x) + α·σ̂_a(x)**: Select action with highest upper confidence bound
- **Exploration**: High uncertainty σ̂ → optimistic estimate → explore
- **Exploitation**: High reward μ̂ → select best known action
- **Automatic balance**: No need for separate ε-greedy exploration

### Neural Bandits: Best of Both Worlds
- **Representation learning**: Deep neural network φ_θ(x,a) → z
- **Efficient exploration**: LinUCB on learned features z
- **Advantages**: Nonlinear patterns + tractable uncertainty quantification

### Query-Aware Token Allocation
**ARR-COC-0-1 as Contextual Bandit**:
```
Context: [query_embedding; patch_features; 3P_relevance_scores]
Actions: token_budgets ∈ {64, 128, 256, 400}
Reward: downstream task accuracy
Policy: Learn query-specific allocation via LinUCB/Neural Bandit
```

**Example Learning**:
- "What color is the car?" → High budget to car patches
- "How many people?" → Moderate budget across person patches
- "What text on sign?" → Maximum budget to text regions

---

## Integration Points

### File 4: FSDP for Large-Scale Neural Bandits
- **Challenge**: Millions of context features, thousands of actions
- **Solution**: FSDP sharding for neural bandit models
- **Memory savings**: 32GB → 4GB per GPU (8-way sharding)
- **Use case**: Product recommendation with 10K+ features, 1M+ SKUs

### File 12: K8s ML Workload Patterns
- **Daily retraining**: CronJob for model updates (2 AM daily)
- **Real-time serving**: Horizontal scaling for high traffic
- **Resource management**: GPU allocation, memory limits
- **Fault tolerance**: Auto-restart crashed pods

### File 16: TPU-Accelerated Bandit Inference
- **Vectorized UCB**: Compute all action UCBs simultaneously
- **JAX implementation**: `jax.vmap` for batch processing
- **Performance**: 10K contexts in <1ms on TPU v5e
- **Cost efficiency**: Lower cost/FLOP for matmul-heavy LinUCB

---

## ARR-COC-0-1 Specific Benefits

### Adaptive Token Budgets (10% coverage)

**Current State**: Heuristic relevance → fixed token mapping
**With Contextual Bandits**: Learned relevance → query-aware token allocation

**Learning Process**:
1. **Initialize**: Uniform exploration across budgets {64, 128, 256, 400}
2. **Interact**: For each (query, image) pair, select budgets via UCB
3. **Observe**: Measure accuracy on downstream task
4. **Update**: Improve policy (high accuracy → reinforce budget choices)
5. **Converge**: Query-specific allocation strategies

**Expected Gains**:
- **Compute**: 30-50% reduction vs fixed high budget
- **Accuracy**: Maintained or improved vs fixed low budget
- **Adaptability**: Automatic adjustment to new query types

**Integration with 4 Ways of Knowing**:
- **Propositional**: Information entropy → UCB context feature
- **Perspectival**: Salience maps → UCB context feature
- **Participatory**: Query alignment → UCB context feature
- **Procedural (new)**: Learned token allocation policy via bandits

---

## Web Research Quality

**Excellent sources secured**:
- ✓ Original LinUCB paper (Li et al. 2010, WWW)
- ✓ True Theta tutorial (comprehensive LinUCB walkthrough)
- ✓ WWW 2024 Neural Bandits tutorial (124 pages)
- ✓ Recent papers: NeurIPS 2024, WWW 2023, arXiv 2024
- ✓ Fairness, privacy, transfer learning extensions
- ✓ Production applications: advertising, pricing, medical

**Citation coverage**:
- 20+ academic papers cited
- 3+ production case studies
- Technical tutorials with code examples
- Recent advances (2024-2025 papers)

---

## File Statistics

- **Total lines**: ~700
- **Sections**: 8 major sections
- **Code examples**: 10+ (Python, YAML, JAX)
- **Citations**: 25+ sources
- **Integration**: 3 influential files explicitly covered
- **ARR-COC coverage**: ~10% (Section 8 + distributed throughout)

---

## Completion Checklist

- [✓] Section 1: Contextual bandit formulation
- [✓] Section 2: LinUCB algorithm
- [✓] Section 3: Neural contextual bandits
- [✓] Section 4: Personalization applications
- [✓] Section 5: Query-aware relevance allocation
- [✓] Section 6: Advanced techniques
- [✓] Section 7: FSDP + K8s + TPU integration
- [✓] Section 8: ARR-COC-0-1 integration
- [✓] Citations: 25+ sources with links
- [✓] File 4 (FSDP) explicitly covered
- [✓] File 12 (ML workloads) explicitly covered
- [✓] File 16 (TPU) explicitly covered
- [✓] ARR-COC concepts integrated (10%)
- [✓] Knowledge drop created

---

## Next Steps for Oracle

This knowledge file provides the foundation for:
1. **Implementing contextual bandit token allocation** in ARR-COC-0-1
2. **Production deployment patterns** using K8s workloads
3. **Scaling to large models** using FSDP for neural bandits
4. **Accelerating inference** using TPU vectorized UCB computation

Ready for integration into master INDEX.md and SKILL.md.
