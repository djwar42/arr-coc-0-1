# KNOWLEDGE DROP: Predictive Coding & Hierarchical Message Passing

**Date**: 2025-11-23 15:45
**PART**: 2
**File Created**: friston/01-predictive-coding-message-passing.md
**Lines**: ~700

---

## Summary

Created comprehensive knowledge file on **Friston's predictive coding framework** and **hierarchical message passing** in cortical processing.

---

## Key Content

### 1. Hierarchical Message Passing Architecture
- Two-stream framework: Top-down predictions vs bottom-up errors
- Anatomical correspondence to cortical layers
- Mathematical formulation of update rules

### 2. Prediction Error Minimization
- Core computational principle
- Sparse coding of errors
- Hierarchical depth and abstraction

### 3. Precision Weighting and Gain Control
- Precision as inverse variance
- Attention as precision optimization
- Neural implementation via neuromodulators

### 4. Cortical Microcircuits
- Bastos et al. canonical microcircuit
- Layer-specific computational roles
- Superficial (errors) vs deep (predictions)

### 5. Layer-wise Computation
- Information flow through layers
- Oscillatory signatures (gamma vs alpha/beta)
- Cross-frequency coupling

### 6. Visual Hierarchy (V1 to IT)
- Visual cortex as predictive hierarchy
- Classical and extra-classical receptive fields
- Attention effects across areas

### 7. Mathematical Formulation
- Generative model structure
- Recognition dynamics
- Learning dynamics (Hebbian)

### 8. ARR-COC-0-1 Connection (10%)
- Relevance realization as hierarchical message passing
- Variable LOD as hierarchical precision
- Query-aware hierarchical processing
- Entropy as prediction error
- Opponent processing as precision weighting

---

## Sources Cited

**Primary Papers**:
- Friston, 2009 - Predictive coding under free-energy (1956 citations)
- Bastos et al., 2012 - Canonical Microcircuits (2820 citations)
- Kanai et al., 2015 - Cerebral hierarchies (442 citations)
- Rao & Ballard, 1999 - Predictive coding in visual cortex

**Recent Research**:
- Chao et al., 2018 - Large-Scale Cortical Networks (252 citations)
- Gelens et al., 2024 - Distributed prediction errors
- Jiang et al., 2021 - Predictive Coding Theories
- den Ouden et al., 2012 - Prediction Errors (595 citations)

**Oracle Cross-References**:
- cognitive-mastery/07-predictive-coding-algorithms.md
- cognitive-foundations/01-predictive-processing-hierarchical.md
- friston/00-free-energy-principle-foundations.md

---

## Differentiation from Existing Knowledge

**Existing file** (cognitive-mastery/07-predictive-coding-algorithms.md):
- Focus: Computational implementation, code, algorithms
- Content: Rao-Ballard algorithm, FORCE learning, Python implementations
- ~768 lines

**New file** (friston/01-predictive-coding-message-passing.md):
- Focus: Theoretical framework, neural architecture, Friston's contributions
- Content: Two-stream framework, cortical microcircuits, precision weighting
- ~700 lines

**Complementary coverage**: Implementation vs Theory

---

## ARR-COC-0-1 Connections

Strong mappings established:

| Predictive Coding | ARR-COC-0-1 |
|-------------------|-------------|
| Prediction error | Entropy, saliency, query mismatch |
| Precision weighting | Opponent processing |
| Hierarchical levels | Texture pyramid (RGB → edges → spatial) |
| Top-down predictions | Query-aware modulation |
| Resource allocation | Token budget per patch |
| Active inference | Compression and generation |

---

## Status

**COMPLETE**

- [x] Checked existing knowledge
- [x] Conducted web research
- [x] Created knowledge file (~700 lines)
- [x] Included Section 8: ARR-COC-0-1 (10%)
- [x] Cited all sources with URLs
- [x] Created KNOWLEDGE DROP file
