# KNOWLEDGE DROP: Psychophysics & Human Studies

**Created**: 2025-11-14 15:48
**PART**: 14 of 24
**File**: karpathy/research-methodology/01-psychophysics-human-studies.md
**Lines**: ~700

---

## What Was Created

Comprehensive psychophysics and human studies methodology file covering experimental methods for measuring perception and validating AI systems against human ground truth.

**8 Major Sections**:
1. Psychophysics Fundamentals (~90 lines)
2. Classical Psychophysical Methods (~130 lines)
3. Adaptive Psychophysical Methods (~110 lines)
4. Signal Detection Theory (SDT) (~100 lines)
5. Weber's Law and JND (~90 lines)
6. Perceptual Scaling and Magnitude Estimation (~80 lines)
7. Human Studies Protocols and Ethics (~90 lines)
8. ARR-COC-0-1 Human Evaluation Design (~110 lines)

**Total**: ~700 lines

---

## Key Knowledge Added

### Classical Methods (Gold Standards)
- **Method of Limits**: Ascending/descending threshold estimation
- **Method of Constant Stimuli**: Full psychometric function mapping (200-400 trials)
- **Method of Adjustment**: Subject-controlled threshold setting

### Adaptive Methods (Efficiency)
- **Staircase Procedures**: 1-up-2-down converges to 70.7% correct (40-60 trials)
- **QUEST**: Bayesian threshold estimation (20-40 trials)
- **Psi Method**: Entropy minimization, theoretically optimal (15-30 trials)

### Signal Detection Theory
- **d' (d-prime)**: Sensitivity = Z(Hit Rate) - Z(False Alarm Rate)
- **Criterion (c)**: Response bias measurement
- **ROC Curves**: Area under curve = probability correct in 2AFC
- Separates true sensitivity from response bias

### Weber's Law
- **Formula**: ΔI/I = k (JND proportional to stimulus magnitude)
- **Weber Fractions**: Brightness 8%, Line length 3%, Weight 2%
- Applications to token budget JNDs

### Perceptual Scaling
- **Stevens' Power Law**: Ψ = k × I^n
- **Exponents**: Brightness 0.33 (compressive), Pain 3.5 (expansive)
- **Magnitude Estimation**: Direct ratio scaling of perceived intensity

### Human Studies Ethics
- **IRB Levels**: Exempt, Expedited, Full Review
- **Informed Consent**: 8 required elements
- **Data Privacy**: De-identification, encryption, retention policies

### ARR-COC-0-1 Validation Designs
- **Phase 1**: Token budget JND measurement (2AFC staircase)
- **Phase 2**: Relevance magnitude estimation (0-100 scale)
- **Phase 3**: Quality vs token budget scaling (1-7 categories)
- **Eye Tracking**: AUC-Judd, NSS, Pearson correlation metrics
- **Signal Detection**: Relevance detection d' and criterion

---

## Research Sources

**33 sources cited**, including:

**Psychophysics Methods**:
- NIH PMC: Introspective psychophysics, Comparing adaptive procedures
- ScienceDirect: Continuous psychophysics review
- Sci-Hub: Methods in Psychophysics (Wichmann 2018)
- Purdue: Adaptive methods outline
- UW: Lesson on psychometric functions

**Signal Detection Theory**:
- Princeton: SDT lecture slides
- Springer: d' sensitivity measures
- Birmingham: SDT introduction
- APA: d-prime definition

**Weber's Law & JND**:
- Simply Psychology, Verywell Mind: JND explanations
- NIH PMC: Grasping follows Weber's law
- RIT, EBSCO: Weber's law fundamentals

**Adaptive Methods**:
- Treutwein 1995: Classic adaptive procedures review (889 citations)
- Leek 2001: Adaptive procedures in research (1077 citations)
- Psychology Stack Exchange: Efficiency comparisons

**Source Documents**:
- biological-vision/02-eye-tracking-task-attention.md: Eye tracking methods
- biological-vision/00-gestalt-visual-attention.md: Perceptual phenomena

---

## ARR-COC-0-1 Integration

### Relevance Perception Validation
**Token Budget JNDs**:
- Hypothesized Weber fraction k ≈ 0.10 (10%)
- At 200 tokens baseline: JND ≈ 20 tokens
- At 400 tokens baseline: JND ≈ 40 tokens
- Validates meaningful perceptual differences in allocation

**Experimental Design**:
```
Phase 1: JND Measurement
- Method: 2AFC + 2-down-1-up staircase
- Task: "Which image shows more detail in ROI?"
- Converges to 71% correct threshold
- n=20 subjects

Phase 2: Relevance Scaling
- Method: Magnitude estimation (0-100)
- Correlate human ratings with model attention
- Expected Pearson r > 0.7
- n=30 subjects

Phase 3: Quality Assessment
- Method: Category scaling (1-7)
- Token budgets: 64, 128, 200, 300, 400
- Find diminishing returns point
- n=25 subjects
```

### Eye Tracking Validation
**Metrics**:
- **AUC-Judd**: ROC curve for fixation prediction (target > 0.80)
- **NSS**: Normalized scanpath saliency (target > 2.0)
- **Pearson r**: Spatial attention overlap (target > 0.60)
- **KL divergence**: Distribution similarity

**Equipment**: SR Research EyeLink 1000 or Tobii Pro Spectrum (500 Hz)

### Signal Detection Analysis
**Relevance Detection**:
- Task: Detect query-relevant regions (Yes/No)
- Expected d' > 2.0 (high sensitivity)
- Expected c ≈ 0 (neutral criterion)
- Validates ARR-COC-0-1 attention threshold meaningful

---

## Methodology Contributions

### Efficiency Comparison
```
Classical constant stimuli:    200-400 trials
Simple staircase:              40-60 trials
Transformed staircase:         30-50 trials
QUEST:                         20-40 trials
Psi method:                    15-30 trials (optimal)
```

### Sample Size Guidelines
```
Within-subjects (α=0.05, power=0.80):
- Medium effect (d=0.5): n=27
- Large effect (d=0.8): n=12

Between-subjects:
- Medium effect: n=64 per group
- Large effect: n=26 per group
```

### Data Quality Control
- Practice trials to stabilize performance
- Catch trials to detect inattention
- Validation trials with known answers
- Eye tracking accuracy <0.5° error
- Reaction time outlier detection

---

## Novel Applications to VLMs

### Query-Aware Relevance Thresholds
First systematic application of psychophysics to VLM token allocation validation.

### Adaptive Token Budget Optimization
Use magnitude estimation to map quality vs token budget relationship, find optimal allocation policy.

### Multi-Modal Integration Validation
Combine eye tracking (gaze patterns) with magnitude estimation (relevance ratings) to validate cross-modal attention mechanisms.

---

## Connection to Other Knowledge

### Builds On:
- **biological-vision/02-eye-tracking-task-attention.md**: Eye tracking methods, gaze analysis, AOI metrics
- **biological-vision/00-gestalt-visual-attention.md**: Perceptual organization principles

### Enables:
- **research-methodology/03-statistical-analysis-testing.md**: Statistical tests for psychophysics data
- **practical-implementation/56-vision-token-budget-ablations.md**: Human-grounded ablation studies

### Complements:
- **cognitive-foundations/03-attention-resource-allocation.md**: Theoretical attention framework
- **information-theory/00-shannon-entropy-mutual-information.md**: Information-theoretic measures in perception

---

## File Statistics

- **Total lines**: ~700
- **Sections**: 8 major sections
- **Code examples**: 15+ practical examples
- **Formulas**: 20+ mathematical definitions
- **References**: 33 cited sources
- **Tables**: 10+ comparison tables
- **Experimental designs**: 4 complete protocols

---

## Quality Checklist

- [✓] All sources cited with URLs and access dates
- [✓] Source documents referenced (biological-vision files)
- [✓] ARR-COC-0-1 integration in Section 8
- [✓] Mathematical formulas with clear notation
- [✓] Practical examples throughout
- [✓] Ethical considerations addressed
- [✓] Sample size calculations provided
- [✓] Experimental designs fully specified
- [✓] ~700 lines target met

---

## Next Steps

This knowledge enables:
1. Design human validation experiments for ARR-COC-0-1
2. Calculate sample sizes for perceptual studies
3. Choose appropriate psychophysical methods
4. Analyze human-model agreement data
5. Validate token allocation against perception thresholds

**Ready for**: PART 15 (Eye Tracking & Gaze Analysis) or PART 16 (Statistical Analysis & Hypothesis Testing)
