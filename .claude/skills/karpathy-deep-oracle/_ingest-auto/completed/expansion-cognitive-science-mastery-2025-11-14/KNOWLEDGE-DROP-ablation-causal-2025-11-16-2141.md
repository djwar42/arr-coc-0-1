# KNOWLEDGE DROP: Ablation Studies & Causal Inference

**Runner**: PART 32
**Date**: 2025-11-16 21:41
**File Created**: `cognitive-mastery/31-ablation-studies-causal.md`
**Lines**: ~700

## Summary

Created comprehensive knowledge file on ablation studies and causal inference methodologies, covering Pearl's do-calculus, counterfactual reasoning, and their application to machine learning systems like ARR-COC-0-1.

## Key Content Areas

### 1. Ablation Studies in Machine Learning
- Definition: Systematic removal of components to measure contributions
- Types: Feature, model, and training ablations
- Methodology: Baseline → removal → measurement → attribution
- Challenges: Code redundancy, combinatorial explosion, confounding effects

### 2. Lesion Studies in Neuroscience
- Historical foundation in neuropsychology
- Provides causal evidence (unlike correlational fMRI)
- Reveals necessity of brain regions for functions
- Parallel to ML ablation: Brain regions ↔ network layers

### 3. Pearl's Do-Calculus
- Three fundamental rules for causal inference from DAGs
- Rule 1: Insertion/deletion of observations
- Rule 2: Action/observation exchange (generalizes back-door criterion)
- Rule 3: Insertion/deletion of actions
- Goal: Transform causal expressions (do-operators) to observational expressions

### 4. Counterfactual Reasoning
- Definition: "What would have happened if things were different?"
- Structural Equation Models (SEMs) enable counterfactual queries
- Twin datapoint metaphor: Parallel universe with intervention
- Distinction: Population-level interventions vs individual-level counterfactuals
- Untestable but useful for credit assignment and explanation

### 5. Ablation as Causal Inference
- Ablation = intervention: Removing component C ≡ do(C=0)
- Proper design: Randomization, isolation, statistical testing, interaction analysis
- Common pitfalls: Single runs, simultaneous changes, cherry-picking
- Confounders: Hyperparameters, random seeds, data ordering

### 6. ARR-COC-0-1 Application
- Three ways of knowing ablation (propositional, perspectival, participatory)
- Opponent processing ablation (tension mechanisms)
- LOD allocation ablation (fixed vs adaptive token budgets)
- Counterfactual explanations: Per-prediction component necessity
- Infrastructure: Distributed training enables large-scale ablations (Files 4, 8, 16)

## Web Research Sources

1. **EuroMLSys '25 Paper**: Utilizing LLMs for Ablation Studies (Sheikholeslami et al.)
   - Modern challenges in ML ablation studies
   - Automated ablation frameworks
   - LLM-assisted experimentation

2. **PMC Neuroscience Review**: Lesion Studies in Contemporary Neuroscience (Vaidya et al., 2019)
   - Causal evidence from brain damage
   - Lesion-symptom mapping
   - Connection to computational modeling

3. **Joshua Entrop Blog**: The 3 Rules of Do-Calculus
   - Clear explanation with examples
   - DAG manipulation rules
   - Practical causal effect identification

4. **Ferenc Huszár Blog**: Causal Inference 3 - Counterfactuals
   - Structural equation models
   - Twin datapoint metaphor
   - Counterfactual vs intervention distinction
   - Connection to average causal effects

## Integration with Influential Files

**File 4 (FSDP vs DeepSpeed)**:
- Ablation robustness across distributed frameworks
- Memory efficiency enables architectural variations
- Framework choice shouldn't confound component effects

**File 8 (torch.compile)**:
- Compilation eliminates implementation confounds
- Measures algorithmic differences, not efficiency differences
- Apply compilation equally to all ablation configurations

**File 16 (TPU fundamentals)**:
- Hardware-specific effects in ablation experiments
- XLA compilation interactions with ablations
- Cross-hardware ablation validation

## ARR-COC-0-1 Insights (10%)

### Planned Ablation Framework

**Causality testing**: Each relevance realization component can be systematically disabled to test causal necessity:

- **Knowing ablations**: Disable individual scorers (propositional/perspectival/participatory)
- **Balancing ablations**: Disable opponent processing tensions
- **Attending ablations**: Fix token budgets (no adaptive LOD)

**Counterfactual explanations**: "Would this image still be correctly classified if we disabled the perspectival scorer?" provides per-prediction interpretability beyond aggregate ablation statistics.

**Distributed infrastructure**: ZeRO-3 and pipeline parallelism (Files 1, 2) make large-scale ablation studies computationally tractable, enabling exploration of architectural space.

## Statistical Rigor

- Effect sizes (Cohen's d) for practical significance
- Multiple comparison corrections (Bonferroni, FDR)
- Bayesian ablation analysis with hierarchical priors
- Minimum 3-5 random seeds per configuration

## Publication Standards

- Report all configurations (no cherry-picking)
- Include negative results
- Test interaction effects
- Statistical uncertainty quantification
- Information-theoretic view: Ablation reduces uncertainty about component contributions

## Success Metrics

- ✓ Comprehensive coverage of ablation methodology
- ✓ Clear do-calculus explanation with rules
- ✓ Counterfactual reasoning framework
- ✓ ARR-COC-0-1 integration (10%+)
- ✓ Influential files cited (4, 8, 16)
- ✓ Web sources with links and dates
- ✓ ~700 lines of substantive content
