# KNOWLEDGE DROP: Active Inference & Free Energy Principle

**Runner**: PART 1
**Timestamp**: 2025-11-14 14:40
**Status**: ✓ COMPLETE

## Knowledge File Created

`cognitive-foundations/00-active-inference-free-energy.md` (688 lines)

## Sources Used

**Web Research:**
- [The Free Energy Principle - MIT Open Encyclopedia of Cognitive Science](https://oecs.mit.edu/pub/my8vpqih) (2024)
- [A step-by-step tutorial on active inference](https://www.sciencedirect.com/science/article/pii/S0022249621000973) (Smith et al., 2022)
- [Reinforcement Learning or Active Inference](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0006421) (Friston et al., 2009)
- Google search results: Karl Friston 2024, active inference computational tutorials, variational free energy

**Existing Oracle Knowledge:**
- `.claude/skills/john-vervaeke-oracle/concepts/00-relevance-realization/00-overview.md`
- `.claude/skills/john-vervaeke-oracle/papers/00-Vervaeke-2012-Primary-Paper-Analysis.md`

**ARR-COC-0-1 Implementation Files:**
- `arr_coc/knowing.py` - Perception as Bayesian inference
- `arr_coc/balancing.py` - Opponent processing
- `arr_coc/attending.py` - Resource allocation
- `arr_coc/realizing.py` - Active inference pipeline

## Context

Active inference is the computational implementation of the free energy principle—the theory that living systems maintain their existence by minimizing surprise about sensory inputs. This requires both perception (updating beliefs) and action (changing the world).

**Key Connection**: Active inference IS relevance realization. Both frameworks describe how systems determine what matters from infinite possibilities through:
- Opponent processing (pragmatic ↔ epistemic value)
- Hierarchical prediction error minimization
- Precision-weighted resource allocation
- Transjective coupling between agent and world

## Knowledge Gaps Filled

**Before**: Oracle had Vervaeke's relevance realization but lacked the mathematical/computational foundation

**After**:
- Mathematical formulation of free energy principle
- Variational inference framework
- Active inference as perception + action
- Expected free energy (epistemic vs pragmatic value)
- Precision weighting (attention as gain control)
- Temporal depth and planning horizons
- Connection to machine learning (VAEs, predictive coding, RL)
- **Explicit mapping to ARR-COC-0-1 implementation** (Section 8)

## Sections Created (8 total)

1. **Free Energy Principle Fundamentals** - Surprise minimization, Markov blankets, variational bounds
2. **Active Inference** - Perception as belief updating, action as world changing, unified framework
3. **Generative Models** - Hierarchical structure, learning across timescales, precision-weighted updates
4. **Epistemic vs Pragmatic Value** - Exploration-exploitation as natural consequence of expected free energy
5. **Precision Weighting** - Attention as gain control, precision optimization, salience connection
6. **Temporal Depth** - Planning horizons, multi-scale dynamics, policy selection
7. **Connection to Machine Learning** - VAEs, predictive coding, comparison with reinforcement learning
8. **ARR-COC-0-1 as Active Inference** - Complete mapping of our implementation to active inference framework

## Integration with ARR-COC-0-1

**Section 8 provides explicit connections:**

**Knowing.py** = Perception (Bayesian inference)
- InformationScorer → Propositional knowing → Surprise minimization
- SalienceScorer → Perspectival knowing → Precision estimation
- QueryCouplingScorer → Participatory knowing → Agent-arena coupling

**Balancing.py** = Precision weighting
- Compress ↔ Particularize = Pragmatic ↔ Epistemic value
- Opponent processing = Expected free energy optimization

**Attending.py** = Policy selection
- Token allocation (64-400) = Minimizing expected free energy
- Resource distribution = Precision-weighted prediction error minimization

**Realizing.py** = Active inference execution
- Pipeline orchestration = Complete perception-action loop
- Compression = Action that fulfills predictions

**Generative Model**: ARR-COC-0-1 implements hierarchical predictive coding
- Level 3: Query (abstract goal)
- Level 2: Relevance landscapes (object predictions)
- Level 1: Textures (feature predictions)
- Level 0: Pixels (sensory data)

## Theoretical Foundation

This knowledge file provides the **mathematical and computational foundation** for understanding ARR-COC-0-1:

- **Why** our architecture works (free energy minimization)
- **How** it relates to neuroscience (cortical predictive coding)
- **What** principles guide it (Bayesian inference, expected free energy)
- **Where** it extends (beyond vision to general cognition)

**Key Insight**: ARR-COC-0-1 is not just "inspired by" active inference—it IS active inference applied to vision-language models. Relevance realization and free energy minimization are the same process described in different vocabularies.

## Citations Quality

All web sources accessed 2025-01-14 with full URLs:
- MIT Press (official encyclopedia entry)
- ScienceDirect peer-reviewed tutorial (Smith et al., 2022)
- PLOS ONE original paper (Friston et al., 2009)
- Internal oracle knowledge (Vervaeke framework)
- Implementation files (arr_coc/*.py)

## Next Steps for Oracle

This PART provides theoretical foundation for:
- PART 2: Predictive Processing (cortical implementation)
- PART 3: Bayesian Brain (probabilistic inference)
- PART 4: Attention as Resource Allocation
- Future: Training ARR-COC-0-1 as active inference system
