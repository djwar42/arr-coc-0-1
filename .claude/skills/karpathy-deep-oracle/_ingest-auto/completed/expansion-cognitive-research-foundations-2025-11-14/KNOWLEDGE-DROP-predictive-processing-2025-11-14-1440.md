# KNOWLEDGE DROP: Predictive Processing & Hierarchical Coding

**Date**: 2025-11-14-1440
**PART**: 2 of 24
**File Created**: cognitive-foundations/01-predictive-processing-hierarchical.md
**Lines**: ~700
**Status**: ✓ Complete

---

## What Was Created

Comprehensive knowledge file on **predictive processing and predictive coding** as hierarchical models of brain function. Covers the theoretical framework where the brain acts as a prediction machine, constantly generating internal models and minimizing prediction errors.

## Key Content Sections

1. **Predictive Processing Framework** - Brain as prediction machine, Bayesian inference, historical origins (Helmholtz → Rao & Ballard)

2. **Predictive Coding Architecture** - Error neurons vs prediction neurons, canonical microcircuits (Bastos et al.), laminar organization

3. **Neural Implementation** - fMRI/EEG evidence, intracortical spiking studies, dendritic error computation, precision weighting

4. **Hierarchical Processing** - Top-down predictions vs bottom-up errors, multi-scale temporal/spatial integration

5. **Neuroscience Evidence** - Visual illusions, binocular rivalry, repetition suppression, local-global oddball paradigm

6. **Computational Models** - Free energy principle, variational inference, connection to VAEs and deep learning

7. **Active Inference** - Motor control as predictive coding, exploration vs exploitation, proprioceptive predictions

8. **ARR-COC-0-1 Application** - Propositional knowing = prediction error, hierarchical relevance realization, variable LOD as precision-weighted encoding

## Citations & Sources

**Source Documents**:
- biological-vision/05-cortical-processing-streams.md
- john-vervaeke-oracle/concepts/00-relevance-realization/00-overview.md

**Web Research** (accessed 2025-11-14):
- Costa et al., 2024 - Comprehensive investigation of predictive processing
- Gabhart et al., 2025 - Predictive coding: a more cognitive process than we thought?
- Wikipedia: Predictive coding - Historical and framework overview
- Bastos et al., 2012 - Canonical microcircuits
- Rao & Ballard, 1999 - Original computational model
- Friston & Feldman, 2010 - Attention and free energy
- Adams et al., 2013 - Active inference in motor system
- Mikulasch et al., 2023 - Dendritic error computation
- Millidge et al., 2022 - PC for deep learning

All URLs preserved with access dates.

## ARR-COC-0-1 Integration

**Directly connects to**:
- knowing.py: InformationScorer measures Shannon entropy (prediction error)
- balancing.py: Opponent processing IS precision weighting
- attending.py: Token allocation = hierarchical prediction error minimization
- realizing.py: Active inference pipeline

**Key insight**: ARR-COC-0-1's Vervaekean architecture implements predictive processing principles. Propositional knowing (entropy) = prediction error. Balancing tensions = precision optimization. Variable LOD = precision-weighted encoding.

## Notable Findings

**Critical challenge to classical PC**: Recent spiking studies (Gabhart et al., 2025) show genuine prediction errors emerge in **prefrontal cortex**, not sensory cortex. Suggests predictive processing is more **cognitive** than **sensory**-based.

**Dendritic computation**: Error computation may occur in pyramidal neuron dendrites, not just spiking activity. Apical dendrites receive predictions, basal dendrites receive input, nonlinearities compute error locally.

**Active inference**: Motor control = predictive coding in proprioceptive domain. Motor cortex doesn't send commands, it sends predictions of proprioceptive states.

---

## Quality Checklist

- ✓ 8 major sections with hierarchical structure
- ✓ ~700 lines of content
- ✓ Citations to source documents with line numbers
- ✓ Web research citations with URLs and access dates
- ✓ ARR-COC-0-1 integration section (Section 8)
- ✓ Connections to biological vision knowledge
- ✓ Connections to Vervaeke framework
- ✓ Recent 2024-2025 research included
- ✓ Computational and neural implementation details
- ✓ Clear explanations with examples
