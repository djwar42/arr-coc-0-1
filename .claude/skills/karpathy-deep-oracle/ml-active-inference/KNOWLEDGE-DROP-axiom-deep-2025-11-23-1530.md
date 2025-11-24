# KNOWLEDGE DROP: AXIOM Architecture Deep Dive

**Date**: 2025-11-23 15:30
**Runner**: PART 6
**Target**: ml-active-inference/05-axiom-architecture-deep.md

---

## Summary

Created comprehensive deep dive into AXIOM (Active eXpanding Inference with Object-centric Models), the VERSES AI architecture that achieves human-like sample efficiency through Bayesian mixture models instead of neural networks.

---

## Key Insights Acquired

### 1. AXIOM's Four Mixture Models
- **sMM**: Slot Mixture Model - segments pixels into objects
- **iMM**: Identity Mixture Model - assigns type codes
- **tMM**: Transition Mixture Model - switching linear dynamics
- **rMM**: Recurrent Mixture Model - models interactions

### 2. Core Innovation
- No neural networks, gradients, or replay buffers
- Online learning with exact Bayesian updates
- Structure grows/prunes based on data complexity
- 0.3-1.6M parameters vs 420M for DreamerV3

### 3. Performance Results
- Learns games in 10,000 steps (~12 minutes)
- 7x faster updates than BBF
- Higher cumulative reward on 8/10 Gameworld games

---

## TRAIN STATION Found

**AXIOM = Bayesian NN = Uncertainty Quantification**

All are manifestations of representing knowledge as probability distributions:
- Mixture posteriors = Weight distributions = Predictive distributions
- Conjugate updates = Variational inference = Posterior predictive
- Object-centric = Distributed repr. = Task-agnostic

**Key Insight**: To make decisions under uncertainty, you need to know WHAT you don't know. AXIOM provides this naturally through posterior distributions.

---

## ARR-COC Connection (10%)

**Uncertainty in Relevance**:
- Token relevance should be a DISTRIBUTION, not point estimate
- UCB-style routing: mean + exploration bonus from variance
- Expected Free Energy for token allocation decisions
- Online relevance learning without replay buffers

**Direct Applications**:
1. Adaptive token budgets based on uncertainty
2. Thompson sampling for expert selection
3. Precision weighting from confidence levels
4. Streaming relevance updates

---

## Code Implementations Included

1. **SlotMixtureModel**: Complete sMM with pixel segmentation
2. **TransitionMixtureModel**: tMM with linear dynamics modes
3. **RecurrentMixtureModel**: rMM for interaction clustering
4. **AXIOMAgent**: Full agent with planning loop
5. **UncertainRelevanceRouter**: ARR-COC-specific application

---

## Sources Cited

**Primary**:
- arXiv:2505.24784 - AXIOM paper
- VERSES AI research blog
- GitHub: VersesTech/axiom
- GitHub: VersesTech/gameworld

**Background**:
- Active inference literature (Friston, Parr, Da Costa)
- Object-centric learning (Locatello, Greff)
- Bayesian NNs (Gal, Blundell)

---

## File Stats

- **Lines**: ~750
- **Sections**: 7 (Architecture, Differences, Beliefs, Code, Train Station, ARR-COC, Performance)
- **Code Blocks**: 15+ implementations
- **Tables**: 5 comparison tables

---

## Connections to Other Knowledge

Links to existing oracle knowledge:
- `concepts/active-inference.md` - Core framework
- `concepts/free-energy-principle.md` - Theoretical foundation
- `concepts/attention-mechanisms.md` - Comparison point
- Future: `ml-active-inference/00-active-inference-pytorch.md`

---

## Quality Notes

- Heavy on implementation details as requested
- Complete PyTorch-style code for all core components
- Clear TRAIN STATION identification and explanation
- Strong ARR-COC practical applications
- All sources properly cited with access dates

**ML-HEAVY requirements satisfied**: Code in every section, performance benchmarks, implementation details.
