# KNOWLEDGE DROP: Precision Weighting & Salience

**Date**: 2025-11-23 09:45
**PART**: 5
**File Created**: friston/04-precision-weighting-salience.md
**Lines**: ~700

---

## Summary

Created comprehensive knowledge file on precision weighting and salience in Friston's free energy framework. This file bridges neuroscience concepts (precision, gain control, neuromodulators) with ARR-COC-0-1's token allocation system.

---

## Key Concepts Documented

### Section 1: Precision as Inverse Variance
- Mathematical definition: pi = 1/sigma^2
- Precision in Bayesian inference
- Feldman & Friston equivalence: Attention = E[pi]

### Section 2: Attention as Expected Precision
- Gain control mechanism on PE units
- Synaptic implementation via NMDA
- Top-down vs bottom-up precision

### Section 3: Gain Control Mechanisms
- Neural gain control (multiplicative)
- Contrast gain control
- Normalization model (Reynolds & Heeger)

### Section 4: Salience = Precision-Weighted Prediction Error
- salience = precision * |prediction_error|
- Unsigned vs signed salience
- Alerting function and orienting

### Section 5: Resource Allocation Based on Precision
- Precision as resource allocation signal
- Optimal allocation criterion
- Hierarchical precision allocation

### Section 6: Neural Implementation (Neuromodulators)
- Dopamine: Precision of policy beliefs
- Acetylcholine: Sensory precision
- Norepinephrine: Network gain
- GABA: Predictions via inhibition

### Section 7: Mathematical Formulation
- Precision-weighted PE computation
- Bayesian updates with precision
- Free energy and precision
- Hierarchical message passing

### Section 8: ARR-COC-0-1 Connection (10%)
- Token allocation = Precision weighting
- Precision computation from three knowing modes
- Token budget as gain control
- Salience-driven allocation
- Resource-rational token allocation
- Active inference interpretation
- Empirical predictions
- Vervaeke integration

---

## Sources Used

**Existing Knowledge**:
- cognitive-mastery/01-precision-attention-resource.md

**Web Research**:
- Haarsma et al. 2021 (Mol Psychiatry) - Precision weighting of cortical PE
- den Ouden et al. 2012 (Front Psych) - How PEs shape perception
- Eldar et al. 2013 (Nat Neuro) - Neural gain and attention
- Feldman & Friston 2010 - Attention as expected precision
- Schwartenbeck et al. 2015 - Dopamine encodes certainty
- Perez-Gonzalez et al. 2024 (eLife) - ACh modulates precision
- Lieder & Griffiths 2020 (BBS) - Resource-rational analysis

---

## ARR-COC-0-1 Connections

**Direct implementations**:
1. Token allocation IS precision weighting
2. Three knowing modes provide three precision sources
3. Token budget implements computational gain
4. Query-driven allocation = top-down precision modulation
5. Resource-rational optimization under token budget constraint

**Empirical predictions**:
1. Token allocation should correlate with human gaze
2. Accuracy should scale with token allocation
3. Query should drive reallocation
4. Optimal budget should exist

---

## Cross-References

**Should link to**:
- friston/00-free-energy-principle-foundations.md (when created)
- friston/01-predictive-coding-message-passing.md (when created)
- friston/02-active-inference-perception-action.md (when created)
- cognitive-mastery/01-precision-attention-resource.md (existing)

**Builds on**:
- Vervaeke's 4P knowing framework
- Predictive coding theory
- Resource-rational analysis

---

## Status

**COMPLETE**

File created: friston/04-precision-weighting-salience.md (~700 lines)
All sections documented with sources
ARR-COC-0-1 connection section included (~10%)
Ready for integration into INDEX.md
