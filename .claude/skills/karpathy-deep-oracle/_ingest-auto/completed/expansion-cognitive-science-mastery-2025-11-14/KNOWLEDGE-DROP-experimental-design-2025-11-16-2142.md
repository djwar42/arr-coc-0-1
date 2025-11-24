# KNOWLEDGE DROP: Experimental Design Fundamentals

**Runner**: PART 31
**Date**: 2025-11-16 21:42
**File Created**: `cognitive-mastery/30-experimental-design-fundamentals.md`
**Lines**: ~700
**Status**: ✓ Complete

---

## What Was Created

Comprehensive experimental design foundation covering:

### Core Concepts (Sections 1-2)
- **Independent variables**: Manipulated factors (experimental vs subject variables)
- **Dependent variables**: Measured outcomes
- **Control conditions**: No-treatment, placebo, waitlist controls
- **Control variables**: Held constant to isolate IV effects

### Methodological Techniques (Sections 3-4)
- **Randomization**: Random assignment vs random sampling, eliminating bias
- **Counterbalancing**: Fixed sequences to control order effects (complete vs partial)
- Difference between randomization (all sequences) and counterbalancing (selected sequences)

### Study Designs (Sections 5-6)
- **Between-subjects**: Different groups per condition
  - Advantages: No carryover, clean separation
  - Disadvantages: Individual differences, larger N needed
- **Within-subjects**: Same participants in all conditions
  - Advantages: Controls individual differences, smaller N, more powerful
  - Disadvantages: Carryover effects, practice/fatigue, order effects

### Engineering Integration (Sections 7-9)
- **Tensor parallelism**: Parallel evaluation of experimental conditions
- **Triton Inference Server**: Multi-model A/B testing infrastructure
- **Intel oneAPI**: Cross-platform experimental validation

### ARR-COC Application (Section 10)
- Between-subjects: Compare allocation strategies
- Within-subjects: Test query effects with counterbalancing
- Mixed design: Model type × Query type
- Validation: Correlation with human eye tracking

---

## Web Research Sources

**Primary sources** (accessed 2025-11-16):
1. **iMotions Experimental Design Guide** - Comprehensive overview, variable types, design approaches
2. **Scribbr Independent/Dependent Variables** - Clear definitions, examples, relationship structure
3. **Scribbr Within-Subjects Design** - Detailed comparison, advantages/disadvantages, carryover effects

**Key insights**:
- Modern experiments require careful planning, not random probing
- Within-subjects designs are ~2× more powerful than between-subjects
- Counterbalancing uses fixed sequences; randomization uses all sequences
- Each design type has specific threats to internal validity

---

## Integration with Influential Files

**File 3: Megatron-LM Tensor Parallelism**
- Parallel experimental condition evaluation
- Each GPU runs different condition simultaneously
- Fair comparison with identical hardware environment

**File 7: Triton Inference Server**
- Multi-model deployment for A/B testing
- Dynamic batching ensures fair inference conditions
- Model versioning for easy condition switching

**File 15: Intel oneAPI**
- Cross-platform experimental validation
- Hardware-independent performance testing
- Single codebase reduces implementation confounds

---

## ARR-COC-0-1 Connection (10%)

**Experimental validation framework**:

1. **Between-subjects validation**: Compare ARR-COC adaptive allocation against baselines
   - Control: Same architecture, data, budget
   - Measure: Accuracy, memory, inference time

2. **Within-subjects query study**: Test query-aware relevance
   - All participants see generic/specific/complex queries
   - Counterbalanced order (Latin square)
   - Measure token allocation sensitivity

3. **Mixed design**: Model type (between) × Query type (within)
   - More powerful than pure between-subjects
   - Tests interaction: Does ARR-COC benefit vary by query?

4. **Human attention correlation**: Validate relevance scorers
   - Eye tracking + relevance maps
   - Repeated measures (same participants, multiple images)
   - Expect participatory scorer > saliency for task-driven gaze

**Key principle**: Proper experimental design critical for validating that ARR-COC improvements come from relevance realization, not confounds.

---

## File Statistics

- **Sections**: 10 major sections
- **Subsections**: 25+ detailed subsections
- **Examples**: 15+ concrete examples
- **Code snippets**: 3 implementation examples
- **Citations**: 7 explicit source citations
- **Cross-references**: 3 karpathy oracle files

**Coverage**:
- Variables and hypotheses: 20%
- Randomization and counterbalancing: 20%
- Between-subjects design: 15%
- Within-subjects design: 20%
- Engineering integration: 15%
- ARR-COC applications: 10%

---

## Quality Checks

✓ All sources cited with access dates
✓ Explicit connections to Files 3, 7, 15
✓ ARR-COC application examples (Section 10)
✓ Clear definitions with quotes
✓ Practical implementation guidance
✓ Statistical considerations included
✓ Cross-platform validation covered

---

## Next Steps

This file provides foundation for:
- PART 32: Ablation studies and causal inference
- PART 33: Statistical hypothesis testing
- PART 34: Effect sizes and practical significance
- Experimental validation of ARR-COC components
- A/B testing of relevance scorers in production
