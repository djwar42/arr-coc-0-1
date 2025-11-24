# Meta-Analysis & Systematic Reviews

**Research synthesis methodology for aggregating findings across multiple studies**

Meta-analysis and systematic reviews provide rigorous methodologies for synthesizing research evidence, identifying patterns across studies, and drawing stronger conclusions than individual studies alone. This comprehensive guide covers the complete workflow from protocol design to publication bias detection.

---

## 1. Systematic Review Fundamentals

### What is a Systematic Review?

A systematic review is a structured, transparent approach to identifying, evaluating, and synthesizing all available evidence on a specific research question. Unlike narrative reviews, systematic reviews follow explicit protocols to minimize bias.

**Key characteristics:**
- **Comprehensive search**: Exhaustive literature search across multiple databases
- **Explicit criteria**: Pre-specified inclusion/exclusion criteria
- **Transparent process**: Documented methodology for reproducibility
- **Quality assessment**: Critical appraisal of study quality and bias
- **Structured synthesis**: Systematic organization and summary of findings

**PRISMA 2020 Guidelines:**

The Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) provides a 27-item checklist ensuring complete reporting:

1. **Title**: Identify as systematic review
2. **Abstract**: Structured summary (background, methods, results, conclusions)
3. **Introduction**: Rationale and objectives (PICO/SPIDER framework)
4. **Methods**: Protocol registration, eligibility criteria, search strategy, selection process, data collection, risk of bias assessment, synthesis methods
5. **Results**: Study selection flow diagram, study characteristics, risk of bias, synthesis results
6. **Discussion**: Limitations, conclusions, implications
7. **Funding**: Sources of support

From [PRISMA 2020 Statement](https://www.prisma-statement.org) (accessed 2025-11-16)

---

## 2. Formulating Research Questions

### PICO Framework (Quantitative Research)

**P**opulation/Problem: Who or what is the focus?
**I**ntervention/Exposure: What is being tested?
**C**omparison: What is the alternative?
**O**utcome: What are you measuring?

**Example**: In adults with depression (P), does cognitive behavioral therapy (I) compared to medication (C) reduce symptom severity (O)?

### SPIDER Framework (Qualitative Research)

**S**ample: Who is being studied?
**P**henomenon of Interest: What is being explored?
**D**esign: What methods are used?
**E**valuation: What outcomes are assessed?
**R**esearch type: Qualitative, quantitative, mixed-methods?

From [Methley et al. 2014](https://bmchealthservres.biomedcentral.com/articles/10.1186/s12913-014-0579-0): SPIDER shows higher sensitivity for qualitative literature compared to PICO (accessed 2025-11-16)

**ARR-COC-0-1 Example:**
- **P**: Vision-language models processing complex images
- **I**: Adaptive relevance realization with variable token allocation
- **C**: Fixed-resolution processing (e.g., 400 tokens per image)
- **O**: Task accuracy, computational efficiency, ablation study insights

---

## 3. Literature Search Strategy

### Database Selection

**Core databases:**
- **PubMed/MEDLINE**: Biomedical and life sciences
- **PsycINFO**: Psychology and behavioral sciences
- **Web of Science**: Multidisciplinary citation index
- **Scopus**: Abstract and citation database
- **Google Scholar**: Broad coverage (use for supplementary searches)
- **arXiv**: Preprints (computer science, AI/ML)

**Domain-specific:**
- **IEEE Xplore**: Engineering, computer science
- **ACM Digital Library**: Computing research
- **Papers with Code**: ML research with implementations

### Search String Construction

**Boolean operators:**
- **AND**: Narrows results (all terms must appear)
- **OR**: Broadens results (any term can appear)
- **NOT**: Excludes terms

**Example search string:**
```
("vision-language model*" OR VLM OR "multimodal model*")
AND
("token allocation" OR "adaptive processing" OR "variable resolution")
AND
("attention mechanism*" OR "relevance realization")
```

**Wildcards:**
- `*` (truncation): `process*` → process, processing, processed
- `?` (single character): `wom?n` → woman, women

### Inclusion/Exclusion Criteria

**Inclusion criteria specify:**
- Study design (RCT, observational, qualitative)
- Population characteristics
- Intervention/exposure type
- Outcome measures
- Language (e.g., English only)
- Publication date range

**Exclusion criteria might include:**
- Unpublished dissertations
- Conference abstracts without full text
- Studies with insufficient data for meta-analysis
- Non-peer-reviewed sources

**ARR-COC-0-1 Ablation Study Inclusion:**
- Peer-reviewed papers on adaptive visual processing
- Studies reporting token allocation strategies
- Quantitative performance metrics (accuracy, efficiency)
- Exclude: Non-vision models, fixed-resolution only, theoretical papers without experiments

---

## 4. Study Selection Process

### PRISMA Flow Diagram

Standard four-phase flowchart documenting:

1. **Identification**: Records identified from databases and other sources
2. **Screening**: Records after duplicates removed; records screened (title/abstract)
3. **Eligibility**: Full-text articles assessed for eligibility
4. **Included**: Studies included in qualitative synthesis and meta-analysis

**Typical attrition:**
- 10,000 records identified → 6,000 after duplicate removal
- 6,000 screened → 500 full-text assessed
- 500 assessed → 50 included in meta-analysis

### Screening Process

**Two-stage screening:**
1. **Title/Abstract**: Quick relevance check (2 independent reviewers)
2. **Full-text**: Detailed eligibility assessment (2 independent reviewers)

**Cohen's Kappa** for inter-rater reliability:
- κ > 0.80: Excellent agreement
- κ = 0.60-0.80: Good agreement
- κ < 0.60: Poor agreement (requires discussion and consensus)

---

## 5. Data Extraction

### Standardized Extraction Forms

Create structured templates capturing:

**Study characteristics:**
- Authors, year, journal
- Study design, sample size
- Population demographics
- Intervention/exposure details
- Comparison conditions
- Follow-up duration

**Outcome data:**
- Mean, standard deviation, sample size for continuous outcomes
- Event counts, total N for dichotomous outcomes
- Effect sizes (Cohen's d, odds ratio, risk ratio)
- Subgroup data if available

**Quality indicators:**
- Randomization method
- Blinding procedures
- Attrition rates
- Conflicts of interest

**ARR-COC-0-1 Extraction Template:**
```
Study ID: [Author Year]
Model: [Qwen3-VL, LLaVA, etc.]
Token Budget: [64-400 range or fixed]
Task: [VQA, captioning, reasoning]
Metrics: [Accuracy %, inference time ms]
Relevance Method: [Propositional, Perspectival, Participatory scores]
Ablation: [Which components removed?]
Dataset: [COCO, GQA, TextVQA]
```

---

## 6. Effect Size Aggregation

### Common Effect Size Metrics

**Continuous outcomes:**
- **Cohen's d**: Standardized mean difference
  - d = (M₁ - M₂) / SD_pooled
  - Interpretation: 0.2 (small), 0.5 (medium), 0.8 (large)

- **Hedge's g**: Bias-corrected version of Cohen's d (preferred for small samples)

**Dichotomous outcomes:**
- **Odds Ratio (OR)**: Odds of event in treatment vs. control
- **Risk Ratio (RR)**: Risk of event in treatment vs. control
- **Risk Difference (RD)**: Absolute difference in risk

**Correlation:**
- **Pearson's r**: Correlation coefficient
- **Fisher's Z transformation**: For meta-analysis (normalizes distribution)

### Fixed-Effect vs. Random-Effects Models

**Fixed-effect model:**
- **Assumption**: All studies estimate the SAME underlying effect
- **Weights**: Inverse variance (larger studies weighted more heavily)
- **Use when**: Low heterogeneity (I² < 25%), all studies very similar

**Random-effects model:**
- **Assumption**: Studies estimate DIFFERENT but related effects (drawn from distribution)
- **Weights**: Inverse variance + between-study variance (τ²)
- **Use when**: Moderate to high heterogeneity (I² > 25%), studies differ in populations/methods
- **Interpretation**: Average effect across different populations/settings

From [Borenstein et al. 2020](https://www.sciencedirect.com/science/article/pii/S1836955320300163): I² does not tell us how much effect varies, only the proportion of variance due to heterogeneity (accessed 2025-11-16)

**Most meta-analyses use random-effects** because:
1. Studies rarely identical (different populations, settings, methods)
2. More conservative (wider confidence intervals)
3. Allows generalization beyond included studies

---

## 7. Heterogeneity Assessment

### I² Statistic

**Definition**: Percentage of total variability due to between-study heterogeneity (not sampling error)

**Formula:**
I² = 100% × (Q - df) / Q

Where:
- Q = Cochran's Q statistic (observed variance)
- df = degrees of freedom (k - 1 studies)

**Interpretation (rough guidelines):**
- I² = 0-25%: Low heterogeneity (might not be important)
- I² = 25-50%: Moderate heterogeneity
- I² = 50-75%: Substantial heterogeneity
- I² = 75-100%: Considerable heterogeneity

From [Huedo-Medina et al. 2006](https://www.um.es/metaanalysis/pdf/5008.pdf): I² quantifies heterogeneity as proportion of total variance, widely used but should be interpreted with caution (accessed 2025-11-16)

**Q Statistic:**
- Tests whether observed variance exceeds chance expectations
- Significant Q (p < 0.05) suggests heterogeneity present
- Low power when few studies (k < 10)

### Sources of Heterogeneity

**Clinical heterogeneity:**
- Different populations (age, severity)
- Different interventions (dose, duration)
- Different outcomes (measurement tools)

**Methodological heterogeneity:**
- Study design differences (RCT vs. observational)
- Quality differences (high vs. low risk of bias)
- Publication year (older vs. newer methods)

**Statistical heterogeneity:**
- Variation in effect sizes beyond sampling error
- Detected by I², Q statistic

**Addressing heterogeneity:**
1. **Subgroup analysis**: Explore differences by study characteristics
2. **Meta-regression**: Model effect size as function of study-level predictors
3. **Sensitivity analysis**: Exclude outlier studies, assess robustness
4. **Narrative synthesis**: If too heterogeneous for quantitative pooling

**ARR-COC-0-1 Heterogeneity Sources:**
- Model architecture (different VLM backbones)
- Token budget range (64-400 vs. 100-300)
- Task type (VQA vs. captioning)
- Dataset domain (natural images vs. documents)

---

## 8. Forest Plots

### Anatomy of a Forest Plot

**Visual representation of meta-analysis results:**

```
Study          Effect Size [95% CI]   Weight
Smith 2020     ●——————|              15%
Jones 2021       ●————|              18%
Lee 2022           ●——|              22%
Chen 2023        ●———|               20%
Kim 2024          ●——|               25%
                   |
Overall          ●—|                100%
                   |
         -1    0    1    2
         Favors Control  Favors Treatment
```

**Components:**
- **Study labels**: Author, year
- **Effect size point estimate**: Square or diamond center
- **Confidence interval**: Horizontal line (95% CI typically)
- **Study weight**: Size of square reflects precision (larger = more weight)
- **Overall effect**: Diamond at bottom (width = 95% CI)
- **Line of no effect**: Vertical line at 0 (or 1 for ratios)

**Interpreting forest plots:**
- **Squares left of line**: Effect favors control
- **Squares right of line**: Effect favors treatment
- **Squares crossing line**: Non-significant effect
- **Overall diamond crossing line**: Pooled effect not significant
- **Wide spread of squares**: High heterogeneity

From [PRISMA 2020 Explanation](https://www.bmj.com/content/372/bmj.n160): Forest plots are standard for displaying meta-analysis results graphically (accessed 2025-11-16)

---

## 9. Funnel Plots & Publication Bias

### Funnel Plot Basics

**Y-axis**: Precision (standard error, sample size, or inverse variance)
**X-axis**: Effect size

**Expected pattern (no bias):**
- Symmetric inverted funnel shape
- Small studies (low precision) scatter widely at bottom
- Large studies (high precision) cluster narrowly at top
- Studies centered around true effect

**Asymmetric funnel suggests publication bias:**
- Missing studies in bottom-left (small negative effects not published)
- "File drawer problem": Null results remain unpublished

From [Afonso et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10933152/): Funnel plot asymmetry can indicate publication bias, but other causes exist (accessed 2025-11-16)

### Other Causes of Asymmetry

**Not always publication bias:**
1. **True heterogeneity**: Small studies differ systematically
2. **Poor methodological quality**: Small studies lower quality
3. **Chance**: Random variation especially with few studies
4. **Language bias**: Non-English studies show different effects

**Contour-enhanced funnel plots:**
- Add significance contour lines (p = 0.05, p = 0.01)
- If missing studies in non-significant region → publication bias likely
- If missing studies across all regions → other causes

### Statistical Tests for Publication Bias

**Egger's test:**
- Regression-based test for funnel plot asymmetry
- Significant result (p < 0.10) suggests asymmetry
- **Limitations**: Low power with few studies, assumes linear relationship

**Begg's rank correlation test:**
- Non-parametric alternative to Egger's
- Less powerful but more robust to outliers

**Trim-and-fill method:**
- Imputes missing studies to restore symmetry
- Provides adjusted pooled effect estimate
- **Caveat**: Assumes asymmetry is due to publication bias (may not be true)

**Best practice**: Use funnel plots only when k ≥ 10 studies (Sterne et al. recommendations)

---

## 10. Computational Implementation

### Distributed Meta-Analysis Computation

**Engineering parallel to File 4 (FSDP vs. DeepSpeed):**

When conducting large-scale meta-analyses with thousands of studies:

**Data sharding across compute nodes:**
- **Study-level parallelism**: Distribute studies across workers
- **Resampling parallelism**: Bootstrap/jackknife iterations in parallel
- **Subgroup parallelism**: Different subgroup analyses on different nodes

**Memory optimization patterns:**
- **Lazy loading**: Load study data on-demand (analogous to activation checkpointing)
- **Compression**: Store effect sizes in reduced precision
- **Distributed storage**: Effect size databases sharded across nodes

### Torch.compile for Meta-Analysis (File 8)

**Ahead-of-time compilation of meta-analysis pipelines:**

```python
import torch
import torch.compile

@torch.compile
def random_effects_pooling(effect_sizes, variances, tau_squared):
    """
    Compiled random-effects meta-analysis

    Analogous to compiling VLM forward passes
    """
    weights = 1.0 / (variances + tau_squared)
    pooled_effect = torch.sum(effect_sizes * weights) / torch.sum(weights)
    pooled_variance = 1.0 / torch.sum(weights)
    return pooled_effect, pooled_variance

# AOT Inductor for production meta-analysis dashboards
compiled_pool = torch.compile(random_effects_pooling, mode="max-autotune")
```

**Performance benefits:**
- 10-100× speedup for iterative meta-regression
- GPU acceleration for large-scale sensitivity analyses
- Efficient bootstrap/permutation testing

### ML Workload Patterns (File 12)

**Production meta-analysis systems follow ML orchestration patterns:**

**Kubeflow pipelines for systematic reviews:**
1. **Search pipeline**: Automated database queries
2. **Screening pipeline**: ML-assisted title/abstract screening
3. **Extraction pipeline**: NLP-based data extraction
4. **Analysis pipeline**: Statistical meta-analysis
5. **Reporting pipeline**: Automated PRISMA diagram generation

**Real-world example**: Cochrane uses automation tools for living systematic reviews (continuous updates as new evidence emerges)

---

## 11. ARR-COC-0-1 Meta-Analysis Application (10%)

### Evaluating Adaptive Relevance Realization

**Research question (PICO framework):**
- **P**: Vision-language models processing diverse visual inputs
- **I**: Adaptive token allocation based on relevance realization (64-400 tokens)
- **C**: Fixed token budget baselines (e.g., 400 tokens always)
- **O**: Task accuracy, computational cost, ablation insights

### Hypothetical Meta-Analysis

**Inclusion criteria:**
- Peer-reviewed VLM studies with variable token allocation
- Quantitative performance metrics on standard benchmarks
- Ablation studies isolating relevance components

**Expected heterogeneity sources:**
- **Model architecture**: Qwen3-VL, LLaVA, Gemini (architectural heterogeneity)
- **Task type**: VQA, captioning, reasoning (task heterogeneity)
- **Token budget range**: 64-400 vs. 100-300 (methodological heterogeneity)
- **Relevance mechanism**: Propositional-only vs. 3-way (intervention heterogeneity)

**Effect size metric**: Standardized mean difference in accuracy
- Cohen's d comparing adaptive vs. fixed allocation
- Expected d = 0.3-0.5 (small to medium improvement)
- High I² (50-75%) due to task and architecture diversity

**Subgroup analyses:**
1. **By task complexity**: Simple VQA vs. compositional reasoning
   - Hypothesis: Larger effect for complex tasks (more benefit from adaptation)
2. **By budget range**: Narrow (64-128) vs. wide (64-400)
   - Hypothesis: Wider range shows greater efficiency gains
3. **By relevance method**: Propositional-only vs. 3-way (P+P+P)
   - Hypothesis: 3-way knowing shows larger accuracy improvements

**Publication bias assessment:**
- Funnel plot likely asymmetric (positive results more published)
- Trim-and-fill adjustment to estimate unpublished null results
- Sensitivity analysis excluding industry preprints (potential conflicts)

### Living Systematic Review for ARR-COC

**Continuous evidence synthesis:**
- Automated arXiv searches for new VLM papers
- Monthly meta-analysis updates as evidence accumulates
- Version-controlled effect size estimates
- Real-time dashboard showing current pooled effects

**Analogous to ML deployment:**
- Living review = continuous integration/deployment
- New study = new training run
- Meta-analysis = model ensemble
- Subgroup analysis = ablation study

**Benefits for ARR-COC development:**
1. **Evidence-based design**: Identify which relevance mechanisms most effective
2. **Benchmark contextualization**: Compare ARR-COC improvements to field averages
3. **Gap identification**: Discover under-explored combinations (e.g., few studies on participatory knowing)
4. **Resource allocation**: Prioritize development of high-impact components

---

## 12. Quality Assessment & Risk of Bias

### Risk of Bias Tools

**Cochrane Risk of Bias 2 (RoB 2) for RCTs:**
- Randomization process
- Deviations from intended interventions
- Missing outcome data
- Measurement of outcomes
- Selection of reported results

**ROBINS-I for observational studies:**
- Confounding
- Selection of participants
- Classification of interventions
- Deviations from intended interventions
- Missing data
- Measurement of outcomes
- Selection of reported results

**Quality ratings:**
- Low risk (green)
- Some concerns (yellow)
- High risk (red)

**Sensitivity analysis**: Exclude high-risk studies, re-run meta-analysis to assess robustness

---

## 13. Advanced Topics

### Network Meta-Analysis

**Comparing multiple interventions simultaneously:**
- Traditional meta-analysis: A vs. B
- Network meta-analysis: A vs. B vs. C vs. D (indirect comparisons)
- Requires transitivity assumption (effects generalizable across comparisons)

### Individual Participant Data (IPD) Meta-Analysis

**Pooling raw participant-level data:**
- More powerful than aggregate data meta-analysis
- Allows standardized outcome definitions
- Can explore participant-level moderators
- Requires data sharing agreements (challenging)

### Meta-Regression

**Modeling effect size as function of study-level covariates:**
- Explore sources of heterogeneity
- E.g., effect size = β₀ + β₁(year) + β₂(quality) + ε
- **Limitations**: Observational (confounding possible), low power with few studies

---

## 14. Reporting Standards

### PRISMA 2020 Checklist Items

**Methods section must report:**
- Protocol registration (PROSPERO, OSF)
- Search strategy (databases, dates, search strings)
- Selection process (number of reviewers, disagreement resolution)
- Data extraction process (standardized forms, piloting)
- Risk of bias assessment (tools used, number of assessors)
- Synthesis methods (fixed vs. random effects, software used)
- Subgroup/sensitivity analyses (pre-specified vs. exploratory)

**Results section must include:**
- PRISMA flow diagram (identification → screening → eligibility → included)
- Study characteristics table
- Risk of bias summary (figure or table)
- Forest plot for primary outcome
- Funnel plot if k ≥ 10 studies
- Heterogeneity statistics (I², Q, τ²)
- Sensitivity analyses results

---

## Sources

### Web Research

**PRISMA Guidelines:**
- [PRISMA 2020 Statement](https://www.prisma-statement.org) - Official PRISMA reporting guidelines (accessed 2025-11-16)
- [PRISMA 2020 Explanation & Elaboration](https://www.bmj.com/content/372/bmj.n160) - BMJ 2021, Page et al. - Detailed guidance on forest plots, funnel plots, and bias detection (accessed 2025-11-16)

**Search Frameworks:**
- [PICO, PICOS and SPIDER: Comparison Study](https://bmchealthservres.biomedcentral.com/articles/10.1186/s12913-014-0579-0) - Methley et al. 2014, BMC Health Services Research - SPIDER framework for qualitative research (accessed 2025-11-16)

**Effect Size & Heterogeneity:**
- [I² Statistic Interpretation](https://www.sciencedirect.com/science/article/pii/S1836955320300163) - Borenstein et al. 2020, Journal of Physiotherapy - I² does not tell us how much effect varies (accessed 2025-11-16)
- [Assessing Heterogeneity: Q Statistic or I²](https://www.um.es/metaanalysis/pdf/5008.pdf) - Huedo-Medina et al. 2006, Psychological Methods - I² quantifies between-study variance proportion (accessed 2025-11-16)

**Publication Bias:**
- [Funnel Plot Education Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC10933152/) - Afonso et al. 2023, Sports Medicine - Funnel plot asymmetry causes beyond publication bias (accessed 2025-11-16)

**Systematic Review Guides:**
- [Comprehensive Guide to Systematic Reviews](https://journals.lww.com/md-journal/fulltext/2025/08150/a_comprehensive_guide_to_conduct_a_systematic.68.aspx) - Martinez et al. 2025, Medicine - Complete workflow from protocol to publication (accessed 2025-11-16)

### Engineering Connections

**File 4**: FSDP vs. DeepSpeed - Data sharding for distributed meta-analysis computation (parallel study processing, distributed effect size storage)

**File 8**: torch.compile & AOT Inductor - Ahead-of-time compilation of meta-analysis pipelines for production dashboards (10-100× speedup for iterative analyses)

**File 12**: ML Workload Patterns on K8s - Kubeflow pipelines for automated systematic review workflows (search → screen → extract → analyze → report)

### ARR-COC-0-1 Application

**Meta-analysis framework for evaluating adaptive relevance realization:**
- PICO formulation for VLM token allocation studies
- Heterogeneity assessment across model architectures and tasks
- Subgroup analyses by complexity, budget range, relevance method
- Living systematic review for continuous evidence synthesis
- Publication bias detection for industry vs. academic findings

**Practical benefit**: Evidence-based design decisions for ARR-COC development based on aggregated field evidence rather than individual papers

---

**Next Steps:**
- Implement automated screening using transformer-based classifiers (BERT fine-tuned on abstract relevance)
- Deploy living meta-analysis dashboard with real-time arXiv monitoring
- Conduct network meta-analysis comparing multiple relevance mechanisms (Propositional vs. Perspectival vs. Participatory vs. Combined)
