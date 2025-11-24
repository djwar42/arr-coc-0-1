# KNOWLEDGE DROP: Active Inference - Perception & Action

**Date**: 2025-11-23 14:45
**Runner**: PART 3
**File Created**: friston/02-active-inference-perception-action.md
**Lines**: ~700

## Summary

Created comprehensive knowledge file on Active Inference unifying perception and action under a single free energy minimization framework. Covers the theoretical foundations of Karl Friston's active inference, including expected free energy decomposition into epistemic and pragmatic value, policy selection, temporal planning horizons, and agent-environment coupling.

## Key Concepts Covered

1. **Active Inference Overview**: Action minimizes prediction error just like perception
2. **Perception vs Action Unification**: Single objective (minimize free energy) for both
3. **Expected Free Energy**: G = Risk + Ambiguity = Pragmatic + Epistemic value
4. **Epistemic Value**: Information-seeking, curiosity, exploration
5. **Pragmatic Value**: Goal achievement, exploitation, risk minimization
6. **Temporal Planning Horizons**: Multi-scale planning from milliseconds to years
7. **Agent-Environment Coupling**: Circular causality, Markov blankets, niche construction

## ARR-COC-0-1 Connection (Section 8)

Mapped active inference directly to ARR-COC-0-1's four processes:

| Active Inference | ARR-COC-0-1 | Implementation |
|------------------|-------------|----------------|
| Perception | knowing.py | Score relevance |
| Expected Free Energy | balancing.py | Navigate tensions |
| Policy Selection | attending.py | Allocate tokens |
| Action | realizing.py | Execute compression |

**Key insight**: Participatory knowing IS active inference - query and image mutually define relevance through transjective coupling.

## Sources Used

**Web Research (2024-2025)**:
- Friston et al. (2015) "Active inference and epistemic value" - 920 citations
- Laukkonen et al. (2025) "A beautiful loop" - 13 citations
- Parr & Friston (2019) "Generalised free energy and active inference" - 363 citations
- Bouizegarene et al. (2024) "Narrative as active inference" - 31 citations

**Existing Oracle Knowledge**:
- cognitive-mastery/00-free-energy-principle-foundations.md
- cognitive-foundations/00-active-inference-free-energy.md

## Integration Points

- Builds on PART 1 (Free Energy Foundations) and PART 2 (Predictive Coding)
- Connects to future PART 4 (Markov Blankets)
- Provides foundation for cross-domain unifications (PART 31+)

## Quality Assessment

- **Theoretical depth**: Strong mathematical formulations with intuitive explanations
- **Web research**: 4 major 2024-2025 papers cited with direct quotes
- **ARR-COC-0-1**: 10% (~70 lines) with code examples and mapping table
- **Practical focus**: Emphasis on how active inference resolves exploration-exploitation

## Next Steps

This file provides foundation for:
- Markov blankets and statistical boundaries (PART 4)
- Precision weighting and salience (PART 5)
- Axiom architecture implementing active inference (PART 7)
- Cross-domain unifications with Vervaeke and Whitehead (PART 31-32)

---

**Status**: SUCCESS
**Time**: ~30 minutes
**Oracle ready for consolidation**: After all 6 PARTs in Batch 1 complete
