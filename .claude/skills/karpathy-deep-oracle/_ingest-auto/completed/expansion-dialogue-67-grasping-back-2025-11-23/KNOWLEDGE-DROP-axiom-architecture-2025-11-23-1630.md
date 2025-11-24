# KNOWLEDGE DROP: AXIOM Architecture - VERSES AI

**Date**: 2025-11-23 16:30
**PART**: 7
**File Created**: `friston/06-axiom-architecture-versus-ai.md`
**Lines**: ~700

---

## Summary

Created comprehensive knowledge file documenting AXIOM (Active eXpanding Inference with Object-centric Models), a groundbreaking AI architecture developed by VERSES AI that implements active inference principles for sample-efficient learning.

---

## Key Content Created

### Section 1: AXIOM Architecture Overview
- Paradigm shift from transformers to active inference
- Core components: sMM, iMM, tMM, rMM (four mixture models)
- Learn online from single examples vs massive static datasets

### Section 2: Beliefs vs Values Representation
- Transformers: values (fixed embeddings, point estimates)
- AXIOM: beliefs (posterior distributions, full uncertainty)
- Conjugate priors enable exact Bayesian updates without gradients

### Section 3: Why AXIOM Represents Uncertainty
- Free Energy Principle foundation
- Expected Free Energy = -Utility + Information Gain
- Principled exploration-exploitation balance

### Section 4: Implementation Details
- Mathematical formulations for all four mixture models
- Online structure learning algorithm
- Bayesian Model Reduction for generalization

### Section 5: Comparison to Transformer Architecture
- AXIOM: 0.3-1.6M parameters vs DreamerV3: 420M
- 7-12x faster model updates
- Converges in 5k steps vs 10k

### Section 6: Advantages for Active Inference
- Principled exploration
- Continual learning without catastrophic forgetting
- Interpretability by design
- Robustness to cosmetic shifts

### Section 7: Code Examples and Technical Details
- Generative model structure
- Mean-field variational posterior
- Hyperparameters and GitHub links

### Section 8: ARR-COC-0-1 Connection (10%)
- Relevance realization as free energy minimization
- Beliefs vs values for token relevance
- Object-centric parsing for visual relevance
- Active inference for token allocation
- Precision as salience
- Transjective aspect of relevance

---

## Sources Cited

**Primary Sources**:
- arXiv:2505.24784 - AXIOM paper (Heins et al., 2025)
- VERSES AI Research Blog (accessed 2025-11-23)
- Medium article by Philemon Kiprono (August 2025)

**GitHub**:
- https://github.com/VersesTech/axiom
- https://github.com/VersesTech/gameworld

**Key Figures**:
- Karl Friston (Chief Scientist, VERSES AI)
- Conor Heins, Toon Van de Maele, Alexander Tschantz

---

## Key Insights

### Why This Matters

1. **AXIOM proves active inference is practical**: Not just theory, but working implementation that beats deep RL
2. **Beliefs > Values**: Representing uncertainty enables principled exploration and adaptation
3. **Sample efficiency**: 10k steps vs millions for transformers
4. **Compute efficiency**: 400x fewer parameters than comparable systems

### Critical Technical Points

- **Four mixture models** handle segmentation, identification, dynamics, interactions
- **Online structure learning**: Grows/prunes model based on data complexity
- **Bayesian Model Reduction**: Merges clusters for generalization
- **Expected Free Energy**: Unifies utility and information gain

### ARR-COC Connection

**Direct mapping**:
- AXIOM slots = ARR-COC salient regions
- Precision weighting = attention allocation
- Information gain = exploration of visual space
- Utility = task-relevant token allocation

**Implementation implications**:
1. Replace softmax attention with belief distributions
2. Object-centric visual parsing before attention
3. Active token allocation planning
4. Online relevance learning with BMR

---

## Cross-References

**Builds On**:
- `friston/00-free-energy-principle-foundations.md`
- `friston/02-active-inference-perception-action.md`
- `friston/04-precision-weighting-salience.md`

**Related To**:
- `friston/07-bayesian-brain-hypothesis.md` (PART 8)
- `advanced/00-axiom-vs-transformers.md` (PART 37)

**Informs**:
- Token allocation strategies in ARR-COC-0-1
- Uncertainty representation in VLMs
- Object-centric attention mechanisms

---

## Quality Metrics

- **Comprehensiveness**: Full architecture covered with mathematical detail
- **Citations**: All claims linked to sources
- **ARR-COC Integration**: Strong connection (10% of content)
- **Technical Depth**: Implementation-ready detail
- **Accessibility**: Accessible to ML practitioners

---

## Notes for Oracle

- VERSES AI (not "Versus AI") is the company name
- Karl Friston is Chief Scientist at VERSES AI
- AXIOM represents a paradigm shift from backpropagation to Bayesian structure learning
- Gameworld 10k is their custom benchmark (10k steps = ~12 min human gameplay)
- "Genius" is VERSES' commercial platform built on AXIOM principles
