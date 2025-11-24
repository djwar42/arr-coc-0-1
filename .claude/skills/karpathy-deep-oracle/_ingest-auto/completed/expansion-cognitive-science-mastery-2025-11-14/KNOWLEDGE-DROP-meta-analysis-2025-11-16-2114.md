# KNOWLEDGE DROP: Meta-Analysis & Systematic Reviews

**Date**: 2025-11-16 21:14
**Part**: 40
**File**: cognitive-mastery/39-meta-analysis-systematic-reviews.md
**Lines**: 700+

## What Was Created

Comprehensive guide to meta-analysis and systematic review methodology covering:

1. **Systematic Review Fundamentals** - PRISMA 2020 guidelines, structured synthesis
2. **Research Question Formulation** - PICO (quantitative), SPIDER (qualitative) frameworks
3. **Literature Search Strategy** - Database selection, Boolean operators, search strings
4. **Study Selection Process** - PRISMA flow diagrams, two-stage screening, inter-rater reliability
5. **Data Extraction** - Standardized templates, effect size metrics, quality indicators
6. **Effect Size Aggregation** - Cohen's d, odds ratios, fixed vs. random-effects models
7. **Heterogeneity Assessment** - I² statistic, Q statistic, sources of variation
8. **Forest Plots** - Visual meta-analysis representation, interpreting confidence intervals
9. **Funnel Plots & Publication Bias** - Asymmetry detection, Egger's test, trim-and-fill
10. **Computational Implementation** - Distributed computation, torch.compile, ML pipelines
11. **ARR-COC-0-1 Application (10%)** - Living systematic review for adaptive relevance evaluation

## Web Research Summary

**PRISMA 2020 Guidelines:**
- 27-item checklist for complete systematic review reporting
- Flow diagram documenting study selection (identification → screening → eligibility → included)
- Emphasis on protocol pre-registration, transparent methods, risk of bias assessment

**Frameworks:**
- **PICO**: Population, Intervention, Comparison, Outcome (quantitative)
- **SPIDER**: Sample, Phenomenon, Design, Evaluation, Research type (qualitative)
- SPIDER shows 2× higher sensitivity for qualitative literature (Methley et al. 2014)

**Effect Size & Heterogeneity:**
- **I² statistic**: Percentage of variability due to between-study heterogeneity (not sampling error)
- Interpretation: 0-25% (low), 25-50% (moderate), 50-75% (substantial), 75-100% (considerable)
- **Critical insight**: I² doesn't tell how much effect varies, only proportion due to heterogeneity (Borenstein 2020)
- **Random-effects model**: Preferred when I² > 25% (studies estimate different but related effects)

**Publication Bias:**
- **Funnel plot asymmetry**: Can indicate publication bias BUT other causes exist
- Alternative explanations: true heterogeneity, methodological quality differences, chance
- **Egger's test**: Regression-based test (p < 0.10 suggests asymmetry)
- **Best practice**: Only use funnel plots when k ≥ 10 studies (low power otherwise)

## Engineering Connections

**File 4 (FSDP vs. DeepSpeed):**
- Distributed meta-analysis computation with study-level parallelism
- Data sharding across nodes for large-scale analyses (thousands of studies)
- Memory optimization: lazy loading (activation checkpointing analog), compressed storage

**File 8 (torch.compile):**
- AOT compilation of meta-analysis pipelines for production dashboards
- GPU-accelerated bootstrap/permutation testing
- 10-100× speedup for iterative meta-regression

**File 12 (ML Workload Patterns):**
- Kubeflow pipelines for automated systematic review workflows
- Living systematic reviews = continuous integration/deployment for evidence
- Real-time meta-analysis dashboards with version-controlled effect estimates

## ARR-COC-0-1 Application (10%)

**Hypothetical meta-analysis evaluating adaptive relevance realization:**

**Research question (PICO):**
- P: Vision-language models processing diverse inputs
- I: Adaptive token allocation (64-400 based on relevance)
- C: Fixed token budgets (e.g., 400 always)
- O: Task accuracy, computational cost, ablation insights

**Expected heterogeneity (I² = 50-75%):**
- Model architecture (Qwen3-VL, LLaVA, Gemini)
- Task type (VQA, captioning, reasoning)
- Token budget range (64-400 vs. 100-300)
- Relevance mechanism (Propositional-only vs. 3-way knowing)

**Subgroup analyses:**
1. By task complexity (simple VQA vs. compositional reasoning)
2. By budget range (narrow vs. wide)
3. By relevance method (P-only vs. P+P+P)

**Living systematic review:**
- Automated arXiv monitoring for new VLM papers
- Monthly meta-analysis updates
- Real-time dashboard showing pooled effects
- Version-controlled evidence synthesis

**Benefit**: Evidence-based design decisions rather than relying on individual papers

## Key Insights

1. **Systematic reviews ≠ narrative reviews**: Explicit protocols, transparent methods, PRISMA reporting
2. **Random-effects preferred**: Most studies differ in populations/methods (heterogeneity expected)
3. **I² interpretation caveat**: Quantifies proportion of heterogeneity, not magnitude of effect variation
4. **Publication bias complex**: Funnel plot asymmetry has multiple causes beyond selective reporting
5. **Computational meta-analysis**: Modern tools enable distributed, GPU-accelerated analyses
6. **Living reviews**: Continuous evidence synthesis as new studies emerge

## Quality Checks

✓ Comprehensive coverage (PRISMA → forest plots → publication bias → computation)
✓ Web research with 6+ sources (PRISMA guidelines, Methley 2014, Borenstein 2020, Afonso 2023, etc.)
✓ Engineering connections to Files 4, 8, 12 (distributed computation, compilation, ML pipelines)
✓ ARR-COC-0-1 application (10%): Living systematic review framework for relevance realization
✓ Proper citations with access dates
✓ Practical examples throughout (search strings, effect size formulas, PICO questions)

## File Stats

- **Size**: 700+ lines
- **Sections**: 14 major sections
- **Citations**: 6 web sources + 3 influential files
- **Code examples**: 1 (torch.compile meta-analysis)
- **ARR-COC content**: 10% (Section 11 + integration throughout)

---

**PART 40 complete ✓**

Knowledge file created with comprehensive meta-analysis methodology, PRISMA guidelines, effect size aggregation, heterogeneity assessment, publication bias detection, and computational implementation patterns.
