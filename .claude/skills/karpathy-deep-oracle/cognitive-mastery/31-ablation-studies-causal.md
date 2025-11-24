# Ablation Studies & Causal Inference

## Overview

Ablation studies and causal inference methodologies provide rigorous frameworks for understanding component contributions in machine learning systems and establishing causal relationships in data. These approaches move beyond correlation to establish causation through systematic intervention and counterfactual reasoning.

## Ablation Studies in Machine Learning

### Definition and Purpose

From [Utilizing Large Language Models for Ablation Studies](https://dl.acm.org/doi/10.1145/3721146.3721957) (EuroMLSys '25, accessed 2025-11-16):

> "Ablation studies are typically performed to provide insights into the individual contribution of different building blocks and components of an ML/DL system (e.g., a deep neural network), as well as to justify that certain additions or modifications to an existing ML/DL system can result in the proposed improved performance."

**Core principle**: Remove or disable individual components to measure their contribution to overall system performance.

**Types of ablation**:
- **Feature ablation**: Remove input features systematically
- **Model ablation**: Remove architectural components (layers, attention heads, connections)
- **Training ablation**: Disable specific training procedures (data augmentation, regularization)

### Methodology

**Systematic approach**:
1. **Baseline establishment**: Measure full model performance
2. **Component removal**: Systematically disable each component
3. **Performance measurement**: Quantify impact of removal
4. **Causal attribution**: Attribute performance changes to specific components

**Challenges** (from ACM paper):
- **Tedious redundancy**: Requires maintaining multiple nearly-identical code versions
- **Combinatorial explosion**: Testing all component combinations becomes intractable
- **Confounding effects**: Components may interact non-linearly
- **Statistical validity**: Requires proper experimental design and multiple runs

### Ablation vs Hyperparameter Tuning

**Key distinction**: Ablation studies evaluate architectural/algorithmic choices, while hyperparameter tuning optimizes continuous parameters. Ablation asks "does this component help?" while tuning asks "what's the optimal setting?"

## Lesion Studies in Neuroscience

From [Lesion studies in contemporary neuroscience](https://pmc.ncbi.nlm.nih.gov/articles/PMC6712987/) (PMC, accessed 2025-11-16):

### Historical Foundation

**Classical approach**: Study patients with brain damage to infer function of damaged regions. If region X is damaged and function Y is impaired, region X likely supports function Y.

**Modern advancement**: Combining lesion data with neuroimaging and computational modeling provides stronger causal inference than fMRI alone.

### Causal Inference from Lesions

From the PMC article:

> "Studying the effects of lesions on brain activity provides causal evidence for the contribution of a damaged brain region to functional processes measured in neuroimaging."

**Advantages**:
- Direct causal evidence (unlike correlational fMRI)
- Reveals necessity of brain regions for specific functions
- Tests computational models through forced perturbations

**Limitations**:
- Lesions often affect multiple regions (anatomical bias)
- Plasticity and reorganization after damage
- Cannot isolate single neurons or precise circuits
- Ethical constraints on experimental lesions in humans

### Parallel to ML Ablation

**Conceptual similarity**: Both lesion studies and ML ablation involve systematically removing components to understand their causal contribution. Brain regions ↔ neural network layers, cognitive functions ↔ task performance.

## Causal Graphs and Do-Calculus

From [The 3 rules of do-calculus](https://www.joshua-entrop.com/post/the_3_rules_of_do_calculus.html) (Joshua Entrop, accessed 2025-11-16):

### Pearl's Do-Calculus

**Core insight**: Directed acyclic graphs (DAGs) combined with do-notation enable formal reasoning about interventions and causality.

**Do-notation**: `do(X=x)` represents an intervention that sets variable X to value x, distinct from observing X=x.

**Three fundamental rules**:

**Rule 1 - Insertion/deletion of observations**:
```
P(y|do(x), z, w) = P(y|do(x), w) if Y ⊥⊥ Z|X,W in G_X̄
```
Remove variable Z if it's independent of Y given X and W in the graph where arrows into X are removed.

**Rule 2 - Action/observation exchange**:
```
P(y|do(x), do(z), w) = P(y|do(x), z, w) if Y ⊥⊥ Z|X,W in G_X̄,Z̲
```
Replace intervention do(z) with observation z if Y and Z are independent given X,W in graph with arrows into X removed and arrows out of Z removed. **This generalizes the back-door criterion**.

**Rule 3 - Insertion/deletion of actions**:
```
P(y|do(x), do(z), w) = P(y|do(x), w) if Y ⊥⊥ Z|X,W in G_X̄,Z̄(W)
```
Remove intervention do(z) if Y and Z are independent given X,W in graph with arrows into X removed and arrows into Z (that aren't ancestors of W) removed.

### Practical Application

**Goal**: Transform causal expressions (with do-operators) into purely observational expressions (no do-operators) that can be estimated from data.

**Example - Confounding**:
```
P(y|do(x)) = Σ_z P(y|x,z) P(z)
```
When Z confounds X→Y relationship, condition on Z to identify causal effect.

**Significance**: Allows estimating causal effects from observational data when randomized experiments are infeasible.

## Counterfactual Reasoning

From [Causal Inference 3: Counterfactuals](https://www.inference.vc/causal-inference-3-counterfactuals/) (Ferenc Huszár, accessed 2025-11-16):

### Definition

**Counterfactual query**: "What would have happened if things had been different?"

From the blog post:

> "A very specific definition of counterfactuals: a probabilistic answer to a 'what would have happened if' question."

**Example**: "Given that Hillary Clinton lost the 2016 election and did not visit Michigan, what is the probability she would have won if she had visited Michigan?"

### Structural Equation Models (SEMs)

**Beyond DAGs**: SEMs specify the precise functional relationships between causally connected variables.

**Components**:
- Causal graph G showing dependencies
- Functions f_i for each variable: X_i = f_i(parents(X_i), ε_i)
- Noise variables ε_i capturing randomness

**Intervention modeling**: Replace function f_i with constant assignment for do(X_i = x).

### Counterfactuals vs Interventions

**Key distinction**:

| Aspect | Intervention P(y\|do(x)) | Counterfactual P(y*\|do(x*), x, y, ...) |
|--------|-------------------------|------------------------------------------|
| **Scope** | Population-level effect | Individual-level effect |
| **Question** | What happens to random person if we intervene? | What would have happened to this specific person? |
| **Testability** | Can be tested via RCT | Generally untestable (can't rerun history) |
| **Variables** | Only counterfactual variables | Both observed and counterfactual variables |

From Huszár:

> "Counterfactuals are 'personalized' in the sense that you'd expect the answer to change if you substitute a different person in there."

### Twin Datapoint Metaphor

**Concept**: For each observed datapoint, imagine a "parallel twin" in an alternate universe where intervention occurred. Share the same noise variables ε_i, differ only in intervention and downstream effects.

**Mathematical expression**:
```
p(Y*|do(X*=x̂), X=x, Y=y, Z=z)
```
Predict unobserved counterfactual outcome Y* given observed outcome Y and intervention do(X*=x̂).

**Connection to interventions**:
```
P(y|do(x)) = E_p(X,Y,Z) [P(y*|do(x*=x), X, Y, Z)]
```
Intervention effect is the average of individual counterfactuals over the population.

## Ablation Studies as Causal Inference

### Conceptual Framework

**Ablation as intervention**: Removing component C is equivalent to intervention do(C=0).

**Causal question**: Does component C have a causal effect on performance P?

**Confounders in ablation**:
- Training hyperparameters (learning rate, batch size)
- Random initialization seeds
- Data ordering and augmentation
- Interaction effects with other components

### Proper Ablation Design

**Requirements for causal validity**:

1. **Randomization**: Use multiple random seeds to account for training stochasticity
2. **Isolation**: Change only one component per ablation
3. **Statistical testing**: Quantify uncertainty in performance differences
4. **Interaction analysis**: Test combinations to detect non-additive effects

**Common pitfalls**:
- Single-run ablations (no statistical validity)
- Simultaneous multiple changes (confounded effects)
- Cherry-picking favorable configurations
- Ignoring training dynamics (early stopping, learning rate scheduling)

## ARR-COC-0-1 Ablation Framework

### Planned Ablation Studies

**Three ways of knowing ablation**:
- Remove propositional scorer (entropy): Does statistical information content matter?
- Remove perspectival scorer (salience): Do spatial priors matter?
- Remove participatory scorer (query-content): Does transjective coupling matter?

**Opponent processing ablation**:
- Disable compression↔particularize tension: Effect of fixed patch counts?
- Disable exploit↔explore tension: Effect of fixed uncertainty weighting?
- Disable focus↔diversify tension: Effect of fixed spatial diversity?

**LOD allocation ablation**:
- Fixed 64 tokens per patch: Effect of minimum resolution?
- Fixed 400 tokens per patch: Effect of maximum resolution?
- Uniform allocation: Effect of adaptive allocation?

### Causal Interpretation

**Viewing through do-calculus**:

Each ablation estimates `P(performance|do(component=disabled))` versus baseline `P(performance)`.

**Confounders to control**:
- Dataset (ImageNet, COCO, VQA)
- Base vision encoder (Qwen3-VL)
- Training procedure (distributed training, ZeRO, pipeline parallelism)
- Evaluation metrics (accuracy, F1, BLEU)

**From influential files**:

File 4 (FSDP vs DeepSpeed): Ensures ablation results aren't confounded by distributed training framework choice. Same ablation should yield similar conclusions whether trained with FSDP or DeepSpeed.

File 8 (torch.compile): Ablations measure algorithmic differences, not implementation efficiency. Compile all configurations equally to avoid confounding.

File 16 (TPU programming): If ablating on TPU hardware, ensure XLA compilation doesn't introduce hardware-specific effects that confound component contributions.

### Counterfactual Questions

**Individual prediction counterfactuals**:

"Given this image of a cat that ARR-COC-0-1 correctly identified, and given that the perspectival scorer allocated high precision to the face region, what is the probability it would still be correctly identified if we intervened to disable the perspectival scorer?"

**Answer**: High probability → perspectival scorer not necessary for this case. Low probability → perspectival scorer was crucial.

**Population-level**: Average counterfactuals across test set yields ablation study result.

## Computational Infrastructure for Ablation

### Distributed Ablation Studies

**From File 1 (DeepSeek ZeRO optimizer)**:

Ablation studies require training multiple model variants. ZeRO enables training large models with different architectural choices without running out of memory.

**Memory efficiency for ablations**: Remove attention heads → memory savings. Remove FFN layers → memory savings. ZeRO makes it feasible to explore architectural variations.

### Parallel Ablation Execution

**From File 2 (DeepSpeed pipeline parallelism)**:

Run different ablation configurations on different pipeline stages simultaneously. While stage 1 processes configuration A, stage 2 processes configuration B.

**Throughput advantage**: Pipeline parallelism allows testing multiple ablation hypotheses in parallel across GPU clusters.

### Serving Ablated Models

**From File 6 (TensorRT VLM deployment)**:

After ablation studies identify critical components, deploy optimized models. TensorRT optimizations don't change ablation conclusions but enable faster inference for A/B testing in production.

**Online ablation**: Serve baseline and ablated models simultaneously, route traffic probabilistically, measure real-world performance differences (causal inference in production).

## Statistical Methods for Ablation

### Effect Sizes

**Cohen's d for ablation**:
```
d = (mean_baseline - mean_ablated) / pooled_std
```

**Interpretation**:
- Small effect: d = 0.2
- Medium effect: d = 0.5
- Large effect: d = 0.8

**Practical significance**: Statistical significance (p < 0.05) doesn't imply practical importance. A component might have statistically significant but negligible practical effect.

### Multiple Comparisons

**Problem**: Testing 10 components at α=0.05, expect 0.5 false discoveries by chance.

**Bonferroni correction**:
```
α_adjusted = α / n_comparisons
```

**FDR control**: Benjamini-Hochberg procedure controls false discovery rate while maintaining power.

### Bayesian Ablation Analysis

**From File 11 (Prior knowledge and learning)**:

Bayesian framework for ablation: prior belief about component importance + ablation data → posterior belief.

**Advantages**:
- Quantifies uncertainty about effect sizes
- Naturally handles multiple comparisons
- Enables sequential ablation (update beliefs as data arrives)
- Can incorporate domain knowledge (hierarchical priors)

## Research Publication Standards

### Reporting Ablation Studies

**Minimum requirements**:
1. **All configurations tested**: Don't cherry-pick favorable ablations
2. **Statistical uncertainty**: Report confidence intervals, not just point estimates
3. **Multiple random seeds**: Minimum 3-5 seeds per configuration
4. **Negative results**: Report ablations that didn't improve performance
5. **Interaction effects**: Test key combinations, not just individual ablations

**From File 13 (Shannon entropy and information)**:

Information-theoretic view: Ablation study reduces uncertainty about which components contribute to performance. Entropy of beliefs about components decreases as ablation data accumulates.

## Future Directions

### Automated Ablation

**From influential paper** (LLMs for ablation studies):

Large language models can potentially automate ablation study design:
- Generate ablation configurations from code
- Modify code to disable components systematically
- Analyze results and suggest follow-up experiments
- Report findings in scientific format

**Challenges**: LLMs must understand causal structure of code, not just syntax.

### Causal Discovery from Ablations

**Reverse problem**: Given ablation results, can we reconstruct the causal DAG of component interactions?

**Constraint-based methods**: Use conditional independence tests from ablation data to constrain possible causal graphs.

**Score-based methods**: Score candidate DAGs by how well they explain ablation results.

## Key Takeaways

1. **Ablation = Causal intervention**: Ablation studies are applications of causal inference methodology to machine learning systems

2. **Do-calculus for ablation**: Pearl's do-calculus provides formal framework for reasoning about ablation interventions in presence of confounders

3. **Counterfactuals for interpretation**: Individual predictions can be explained through counterfactual reasoning: "Would this prediction change if component X were disabled?"

4. **Statistical rigor essential**: Proper experimental design, multiple seeds, statistical testing required for valid causal claims

5. **Distributed infrastructure enables scale**: ZeRO, pipeline parallelism, and efficient serving make large-scale ablation studies computationally feasible

6. **ARR-COC-0-1 ablations test causality**: Systematically removing relevance scorers, opponent processes, and LOD allocation tests causal necessity of each component

## Sources

**Source Documents:**
- None directly used (web research only)

**Web Research:**

Primary sources:
- [Utilizing Large Language Models for Ablation Studies in Machine Learning and Deep Learning](https://dl.acm.org/doi/10.1145/3721146.3721957) - EuroMLSys '25, Sheikholeslami et al. (accessed 2025-11-16)
- [Lesion studies in contemporary neuroscience](https://pmc.ncbi.nlm.nih.gov/articles/PMC6712987/) - PMC, Vaidya et al., 2019 (accessed 2025-11-16)
- [The 3 rules of do-calculus](https://www.joshua-entrop.com/post/the_3_rules_of_do_calculus.html) - Joshua Entrop blog, February 2024 (accessed 2025-11-16)
- [Causal Inference 3: Counterfactuals](https://www.inference.vc/causal-inference-3-counterfactuals/) - Ferenc Huszár blog, January 2019 (accessed 2025-11-16)

Search queries:
- "ablation studies machine learning methodology 2024"
- "lesion studies neuroscience causal inference"
- "causal graphs do-calculus Pearl 2024"
- "counterfactual reasoning causal inference"

**Influential Files:**
- File 4: `distributed-training/03-fsdp-vs-deepspeed.md` - Ensures ablation results robust across distributed training frameworks
- File 8: `inference-optimization/03-torch-compile-aot-inductor.md` - Compilation shouldn't confound ablation studies
- File 16: `alternative-hardware/03-tpu-programming-fundamentals.md` - Hardware-specific effects in ablation experiments

**ARR-COC-0-1 Integration:**
Ablation methodology directly applicable to evaluating three ways of knowing (propositional, perspectival, participatory), opponent processing mechanisms, and LOD allocation strategies. Counterfactual reasoning enables per-prediction explanations of relevance allocation decisions.
