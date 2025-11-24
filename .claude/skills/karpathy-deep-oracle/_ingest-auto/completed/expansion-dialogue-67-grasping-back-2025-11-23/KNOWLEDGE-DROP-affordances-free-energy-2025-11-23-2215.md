# KNOWLEDGE DROP: Affordances + Free Energy

**Date**: 2025-11-23 22:15
**PART**: 35
**File Created**: cross-domain/04-affordances-free-energy.md
**Lines**: ~700

---

## Summary

Created comprehensive knowledge file on the unification of Gibson's ecological psychology with Friston's free energy principle. This is one of the most significant theoretical syntheses in cognitive science, showing that affordances ARE priors and active inference IS affordance-guided action.

---

## Key Content Created

### Main Sections:
1. Ecological Psychology and Predictive Processing: Apparent Tension
2. Affordances as Priors: The Deep Connection
3. Expected Free Energy and Action Selection
4. Action-Relevant Perception: The Core Claim
5. Precision, Salience, and Affordance Relevance
6. Synchronization and Attunement: Beyond Inference
7. Optimal Grip and the Tendency Toward Equilibrium
8. **ARR-COC-0-1: Affordance-Driven Token Allocation** (~10%)
9. Ecological-Enactive vs Helmholtzian Interpretations
10. Markov Blankets and Agent-Environment Boundaries
11. Niche Construction and Extended Affordances
12. Implications for Robotics and AI
13. Mathematical Formalization
14. Comparison: Gibson vs Friston Terminology
15. Key Papers and Sources
16. Summary: The Grand Unification

### Critical Insights:

**The Core Synthesis:**
- Affordances ARE priors (organism's expectations about action possibilities)
- Free energy minimization IS grip optimization
- Active inference IS affordance-guided action
- Precision weighting IS salience
- The organism IS its model

**Key Quote from Bruineberg et al. (2018):**
> "We argue that the free energy principle and the ecological and enactive approach to mind and life make for a much happier marriage of ideas."

**The "Crooked Scientist" Critique:**
> "If my brain is a scientist, it is a crooked and fraudulent scientist - but the only sort of scientist that can survive an inconstant and capricious world."

---

## ARR-COC Integration (Section 8)

### Token Allocation as Active Inference

Mapped concepts:
- Affordance detection → Region relevance scoring
- Precision weighting → Context-dependent attention
- Expected free energy → Token allocation priority
- Optimal grip → Best attention configuration

### Implementation Patterns:

```python
def compute_token_affordance(region, task_context):
    epistemic = information_gain(region, current_belief)
    pragmatic = task_relevance(region, task_context)
    precision = salience_estimate(region, context)
    affordance = precision * (epistemic + pragmatic)
    return affordance
```

### Key Principle:
**Relevance = Affordance Detection**
- What's relevant = What affords action
- Salience = Precision-weighted affordance
- Attention = Active inference policy
- Understanding = Optimal grip achieved

---

## Sources Cited

### Primary Web Research:
1. **Bruineberg et al. (2018)** - The anticipating brain is not a scientist
   - Synthese, 195, 2417-2444
   - Cited by 507
   - https://link.springer.com/article/10.1007/s11229-016-1239-1

2. **Friston et al. (2012)** - Dopamine, affordance and active inference
   - PLoS Computational Biology
   - Cited by 457
   - https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002327

3. **Linson et al. (2018)** - The active inference approach to ecological perception
   - Frontiers in Psychology
   - Cited by 134
   - https://pmc.ncbi.nlm.nih.gov/articles/PMC7805975/

4. **Parr & Friston (2019)** - Generalised free energy and active inference
   - Biological Cybernetics
   - Cited by 363
   - https://link.springer.com/article/10.1007/s00422-019-00805-w

5. **Constant et al. (2020)** - Extended active inference
   - Biology & Philosophy
   - Cited by 121
   - https://pmc.ncbi.nlm.nih.gov/articles/PMC9292365/

### Oracle Knowledge Referenced:
- friston/02-active-inference-perception-action.md
- gibson-affordances/00-ecological-psychology.md
- gibson-affordances/01-direct-perception.md
- gibson-affordances/02-affordance-space-topology.md
- cognitive-mastery/03-affordances-4e-cognition.md

---

## Theoretical Significance

This file documents one of the most important theoretical unifications:

1. **Resolves Gibson vs Bayes debate**: Shows both describe same phenomenon
2. **Ecological interpretation of FEP**: Against Helmholtzian reading
3. **Action-oriented cognition**: Perception serves action, not representation
4. **Deflationary inference**: Synchronization, not hypothesis testing
5. **Embodied priors**: Organism structure encodes expectations

### For VLM Design:
- Attention IS active inference policy
- Token allocation IS precision-weighted affordance detection
- Understanding IS optimal grip on scene

---

## Integration Notes

### Connects To:
- friston/ (free energy principle, active inference)
- gibson-affordances/ (ecological psychology, direct perception)
- cognitive-mastery/ (relevance realization, 4E cognition)
- whitehead/ (prehension as grasping possibilities)

### Part of Cross-Domain Series:
- 00-friston-vervaeke-unification.md
- 01-whitehead-active-inference.md
- 02-predictive-coding-enactivism.md
- 03-morphogenesis-active-inference.md
- **04-affordances-free-energy.md** (THIS FILE)
- 05-temporal-physics-unification.md

---

## Status

**COMPLETE**

- [x] Web research conducted
- [x] Key papers identified and scraped
- [x] Knowledge file created (~700 lines)
- [x] Section 8 ARR-COC integration included (~10%)
- [x] Sources properly cited with URLs
- [x] KNOWLEDGE DROP created
