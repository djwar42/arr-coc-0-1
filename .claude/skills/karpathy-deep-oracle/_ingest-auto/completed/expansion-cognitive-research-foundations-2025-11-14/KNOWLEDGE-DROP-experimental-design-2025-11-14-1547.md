# KNOWLEDGE DROP: Experimental Design Fundamentals

**Runner**: PART 13 Worker
**Date**: 2025-11-14 15:47
**Target**: research-methodology/00-experimental-design.md (~700 lines)
**Status**: ✓ COMPLETE

---

## What Was Created

**File**: `.claude/skills/karpathy-deep-oracle/research-methodology/00-experimental-design.md`
**Size**: 732 lines
**Quality**: Comprehensive experimental design guide for ML research

---

## Content Summary

### Three Pillars of Experimental Design
1. **Control**: Holding confounds constant (random assignment, counterbalancing, blocking)
2. **Randomization**: Eliminating systematic bias (simple, block, stratified, cluster)
3. **Replication**: Multiple measurements for power and reliability

### Core Design Types
- **Between-subjects**: Each unit experiences one condition (requires larger n)
- **Within-subjects**: Each unit experiences all conditions (higher power)
- **Factorial designs**: Test multiple IVs simultaneously (2×2, 2×2×2, efficient)

### Variables & Operationalization
- **Independent variables**: Manipulated by experimenter (token budget, vision encoder)
- **Dependent variables**: Measured outcomes (accuracy, latency, human ratings)
- **Operationalization**: Translating abstract concepts to concrete measurements

### Control Conditions
- **Baseline control**: Standard practice reference point
- **Active control**: Controls for non-specific effects
- **Sham control**: Mimics procedure without active ingredient
- **No-treatment control**: Natural baseline

### Threats to Validity
**Internal validity** (can we infer causality?):
- Selection bias, history effects, maturation, testing effects, instrumentation, regression to mean

**External validity** (does it generalize?):
- Construct validity (does measurement match concept?)
- Ecological validity (real-world applicability)
- Population validity (sample representativeness)

### Sample Size & Power Analysis
- **Statistical power**: Probability of detecting real effects (target 80%)
- **Effect size**: Cohen's d (0.2 small, 0.5 medium, 0.8 large), η² for ANOVA
- **Sample size calculation**: Balance power, effect size, alpha, resources
- **ML context**: Large benchmarks (10K+) = very high power for small effects

### ARR-COC-0-1 Experimental Design

**Ablation study structure**:
```
IV: Token allocation strategy (5 levels)
- Fixed 64, Fixed 144, Fixed 256
- Random adaptive (64-400)
- ARR-COC adaptive (64-400)

Task complexity (3 levels):
- Simple VQA, Complex VQA, OCR-heavy

DV: Accuracy, latency, token efficiency
Design: 5×3 mixed factorial (within-subjects for tasks)
```

**Multiple baselines isolate ARR-COC benefits**:
- vs Fixed 256: Overall improvement
- vs Random adaptive: Proves relevance mechanism matters
- vs Uniform adaptive: Proves spatial allocation matters

**Human evaluation**:
- Within-subjects preference judgment (attention map comparisons)
- Counterbalanced presentation order
- Blinded evaluators
- 50 participants × 40 trials = 2,000 judgments

**Statistical analysis example**:
- Paired t-test (within-subjects)
- Effect size (Cohen's d)
- Confidence intervals
- Multiple comparisons correction (Bonferroni)

### Best Practices for ML Experimentation
1. **Pre-register hypotheses**: Prevent p-hacking
2. **Multiple comparisons correction**: Bonferroni, Holm, FDR
3. **Report effect sizes**: Not just p-values
4. **Confidence intervals**: Quantify uncertainty
5. **Random seeds**: Multiple runs, report variance
6. **Computational budget**: Strategic prioritization (pilot → full)

---

## Integration with Existing Knowledge

### From Benchmarking Files

**55-vlm-inference-latency-benchmarks.md**:
- Latency as DV (TTFT, TPOT, E2E)
- Controlled hardware comparisons (A100 vs H100)
- ARR-COC latency hypothesis: Fewer tokens → reduced latency

**56-vision-token-budget-ablations.md**:
- TokenFLEX factorial design (token budget × benchmark)
- Within-subjects ablation (64, 144, 256 tokens on same model)
- Task-dependent effects (OCR: +6.9%, VQA: +0.9%)
- Large effect sizes (η² = 0.15 for token budget main effect)

---

## Key Contributions

### 1. Rigorous Methodology for VLM Evaluation
Comprehensive experimental design framework ensures:
- Valid causal inferences (control confounds)
- Replicable results (random seeds, multiple runs)
- Statistical confidence (power analysis, effect sizes)
- Generalizability (diverse benchmarks, tasks)

### 2. ARR-COC-Specific Design
Detailed ablation study plan with:
- Multiple baseline controls (fixed, random, uniform)
- Factorial design (allocation × task complexity interaction)
- Human evaluation protocol (preference judgments)
- Statistical analysis roadmap (paired t-tests, effect sizes, CIs)

### 3. ML Best Practices
- Pre-registration to prevent p-hacking
- Multiple comparisons correction
- Effect sizes over p-values
- Computational budget optimization (pilot → full)
- Reproducibility protocols (seeds, variance reporting)

---

## Alignment with PART 13 Instructions

✓ **Section 1**: Experimental design principles (control, randomization, replication)
✓ **Section 2**: Independent vs dependent variables (IV/DV definitions, operationalization)
✓ **Section 3**: Between-subjects vs within-subjects (power, counterbalancing)
✓ **Section 4**: Factorial designs (main effects, interactions, ANOVA)
✓ **Section 5**: Control conditions (baseline, active control, sham)
✓ **Section 6**: Confounds and threats to validity (internal, external, construct)
✓ **Section 7**: Sample size and power analysis (effect size, statistical power)
✓ **Section 8**: **ARR-COC-0-1 experiments** (ablation study design, human evaluation)
✓ **CITE**: practical-implementation/55,56 (benchmarking, ablations)

---

## Web Research Sources

**Experimental Design**:
- NIH PMC11741180: Quasi-experimental designs (2024, cited 28×)
- Sage Journals: Factorial designs, main effects, interactions (2024, cited 7×)

**Randomized Controlled Trials**:
- ResearchGate: Factorial ANOVA in various structured designs (2024)
- JMIR: Factorial trial optimization methodology (2024, cited 2×)

**Sample Size & Power**:
- NIH PMC7745163: Sample size, power, effect size (2020, cited 1842×) - foundational
- JTE: Power analyses for sample size determination (2024, cited 18×)
- Statsig: Power analysis determine sample size (2024)

---

## Quality Metrics

**Length**: 732 lines (target: ~700) ✓
**Comprehensiveness**: All 8 sections from PART 13 plan ✓
**Citations**: Both source documents (55, 56) cited ✓
**ARR-COC integration**: Section 8 with detailed experimental design ✓
**Web research**: 4 searches, 7+ papers cited ✓
**Practical examples**: Python code, data tables, statistical analysis ✓

---

## Next Steps (Oracle's Responsibility)

After ALL 24 runners complete:
1. Read all KNOWLEDGE DROP files
2. Update INDEX.md with research-methodology/ entries
3. Update SKILL.md if major methodology changes
4. Move to completed/
5. Git commit with comprehensive message

---

**PART 13 COMPLETE** ✓
