# Causal Inference

## Overview

Causal inference is the science of determining cause-and-effect relationships from data, distinguishing correlation from causation. Unlike standard statistical methods that identify associations, causal inference allows us to answer interventional questions: "What would happen if we changed X?" This capability is fundamental to scientific discovery, decision-making under uncertainty, and building systems that can reason about actions and their consequences.

For ARR-COC-0-1, causal inference connects directly to Vervaeke's participatory knowing: the ability to intervene in the world and understand the effects of those interventions. When the model allocates tokens based on query relevance, it's making a causal claim: "If I allocate more detail here, the downstream task will benefit." This is participatory knowing as causal agency.

**Core principle**: Correlation does not imply causation, but with proper causal models and assumptions, we can infer causation from observational data.

---

## Section 1: Causal Inference Fundamentals (Correlation vs Causation)

### The Fundamental Problem of Causal Inference

**Correlation vs Causation**: The most famous principle in statistics. Two variables can be associated (correlated) without one causing the other.

**Classic examples**:
- Ice cream sales correlate with drowning deaths → Both caused by summer weather (confounding)
- Smoking correlates with yellow fingers → Both caused by smoking cigarettes (common cause)
- Rooster crowing correlates with sunrise → Neither causes the other (spurious correlation)

**The counterfactual definition of causation**: X causes Y if, had X been different (holding all else equal), Y would have been different.

**Why correlation ≠ causation**:
1. **Reverse causation**: Y might cause X instead
2. **Confounding**: A third variable Z causes both X and Y
3. **Selection bias**: Non-random sampling creates spurious associations
4. **Measurement error**: Noise in data creates artificial correlations

### Three Levels of Causal Reasoning (Pearl's Ladder)

**Level 1: Association (Seeing)**
- Question: "What is?" (observational)
- Example: P(Y|X) - "What's the probability of Y given we observe X?"
- Methods: Correlation, regression, machine learning
- Cannot answer causal questions

**Level 2: Intervention (Doing)**
- Question: "What if I do?" (interventional)
- Example: P(Y|do(X)) - "What's the probability of Y if we set X?"
- Methods: Randomized controlled trials, do-calculus
- Can answer "what would happen if we changed X?"

**Level 3: Counterfactuals (Imagining)**
- Question: "What if I had done differently?" (retrospective)
- Example: P(Y_x|X=x', Y=y) - "What would Y have been if X had been x', given X=x' and Y=y?"
- Methods: Structural causal models, twin networks
- Can answer "what would have happened in this specific case?"

### Potential Outcomes Framework (Rubin Causal Model)

**Fundamental setup**:
- Treatment: T ∈ {0, 1} (binary for simplicity)
- Potential outcomes: Y(1), Y(0) (outcome under treatment vs control)
- Observed outcome: Y = T·Y(1) + (1-T)·Y(0)

**Individual Treatment Effect (ITE)**:
```
τ_i = Y_i(1) - Y_i(0)
```

**The fundamental problem**: We never observe both Y(1) and Y(0) for the same unit simultaneously. We only see one potential outcome.

**Average Treatment Effect (ATE)**:
```
ATE = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]
```

**Assumptions for causal identification**:
1. **SUTVA** (Stable Unit Treatment Value Assumption): No interference between units, one version of treatment
2. **Ignorability/Unconfoundedness**: (Y(1), Y(0)) ⊥ T | X (treatment independent of potential outcomes given covariates)
3. **Positivity/Overlap**: 0 < P(T=1|X) < 1 for all X (everyone has some chance of treatment)

### Why Randomized Controlled Trials (RCTs) Work

**RCTs**: Randomly assign treatment → breaks all confounding paths

**Why randomization solves causation**:
```
Random assignment → T ⊥ (Y(1), Y(0))
→ E[Y|T=1] = E[Y(1)|T=1] = E[Y(1)]  (by randomization)
→ E[Y|T=0] = E[Y(0)|T=0] = E[Y(0)]  (by randomization)
→ ATE = E[Y|T=1] - E[Y|T=0]  (simple difference in means!)
```

**Limitations of RCTs**:
- Expensive, time-consuming
- Ethically impossible for many questions (e.g., "does smoking cause cancer?")
- External validity concerns (lab vs real world)
- Cannot study historical/rare events

**Observational causal inference**: Estimate causal effects from non-randomized data using assumptions encoded in causal models.

---

## Section 2: Directed Acyclic Graphs (DAGs, Structural Causal Models)

### Causal DAGs Basics

**Directed Acyclic Graph (DAG)**: Graph with directed edges (arrows) and no cycles.

**Interpretation**:
- Nodes = Variables
- Directed edge X → Y = "X directly causes Y"
- Absence of edge = "No direct causal effect"

**Example DAG**:
```
   Weather
     ↓
  Ice Cream Sales → Shark Attacks
     ↓
  Drowning Deaths
```

This DAG says:
- Weather causes ice cream sales
- Ice cream sales cause drowning deaths
- Ice cream sales cause shark attacks
- Weather does NOT directly cause drowning or shark attacks (only through ice cream sales)

### Structural Causal Models (SCMs)

**Definition**: A set of structural equations defining how variables are generated.

**SCM components**:
1. **Endogenous variables**: U = {U1, U2, ...} (exogenous, unobserved noise)
2. **Endogenous variables**: V = {V1, V2, ...} (observed variables)
3. **Functions**: fi for each Vi, defining Vi = fi(PA_i, U_i) where PA_i are parents of Vi

**Example SCM**:
```
U_X, U_Y ~ Normal(0, 1)
X = U_X
Y = 2X + U_Y
```

This encodes: X causes Y with effect size 2, plus independent noise.

**Connection to DAGs**: SCM defines a DAG where edges are X → Y if Y's equation depends on X.

### Paths and Causal Effects

**Causal path**: Directed path from X to Y following arrows.

**Types of paths** (3-node examples):

**1. Chain (Mediation)**: X → M → Y
- Information flows from X to Y through M
- Conditioning on M blocks the path

**2. Fork (Common Cause)**: X ← Z → Y
- Z is a confounder (common cause of X and Y)
- Conditioning on Z blocks the path

**3. Collider (Common Effect)**: X → Z ← Y
- Z is caused by both X and Y
- Path is already blocked
- Conditioning on Z OPENS the path (collider bias!)

**Blocking and d-separation**:
- Path is blocked if:
  - Contains chain/fork with middle node conditioned
  - Contains collider with collider NOT conditioned (and no descendant conditioned)
- X and Y are d-separated given Z if all paths between X and Y are blocked by Z
- d-separation → conditional independence: X ⊥ Y | Z

### Confounding and Biased Associations

**Confounder**: Variable that causes both treatment and outcome.

**Example**:
```
    Education
      ↙   ↘
  Income  Health
```

Education confounds the Income→Health relationship. Simply regressing Health on Income would be biased because education affects both.

**Confounding bias**:
```
E[Y|X=x] ≠ E[Y|do(X=x)]
```

Observational association ≠ Causal effect (due to backdoor paths through confounders).

**Backdoor paths**: Paths from X to Y that start with arrow INTO X (e.g., X ← Z → Y).

**Solution**: Control for confounders to block backdoor paths.

### Collider Bias (Selection Bias)

**Collider**: Variable caused by multiple variables.

**Collider bias**: Conditioning on a collider creates spurious association.

**Classic example** (Berkson's paradox):
```
   Talent → Celebrity ← Beauty
```

Among celebrities (conditioning on Celebrity), Talent and Beauty are negatively correlated, even if uncorrelated in general population. Less talented celebrities tend to be more beautiful (and vice versa) to compensate.

**Selection bias**: Conditioning on sample selection (a collider) creates bias.

---

## Section 3: do-Calculus (Interventions, Observational vs Interventional)

### The do-Operator

**do(X=x)**: Intervention operator that sets X to value x, removing all incoming arrows to X.

**Graphical interpretation**: do(X=x) → delete all edges INTO X in the DAG.

**Difference from conditioning**:
- **Conditioning**: P(Y|X=x) = "Y given we OBSERVE X=x" (passive observation)
- **Intervention**: P(Y|do(X=x)) = "Y if we SET X=x" (active intervention)

**Example** (smoking and tar):
```
Genotype → Smoking → Tar Deposits → Cancer
```

- P(Cancer|Smoking=yes): Association (confounded by genotype)
- P(Cancer|do(Smoking=yes)): Causal effect (genotype path cut)

**Why they differ**:
```
P(Y|X=x) uses the original DAG (confounders present)
P(Y|do(X=x)) uses the mutilated DAG (confounders removed)
```

### Randomized Controlled Trials as do-Operations

**RCT**: Randomly assign X → implements do(X).

**Why**: Random assignment breaks all incoming edges to X (no confounding).

**Formal equivalence**:
```
RCT: T assigned randomly
→ P(Y|T=t) = P(Y|do(T=t))
```

This is why RCTs estimate causal effects!

### do-Calculus Rules (Pearl's Three Rules)

**Purpose**: Transform P(Y|do(X)) into observational distributions P(·) when possible.

**Rule 1 (Insertion/deletion of observations)**:
```
P(Y|do(X), Z, W) = P(Y|do(X), W)
if (Y ⊥ Z | X, W)_{G_X̄}
```
Can ignore Z if it's d-separated from Y in the graph where X's incoming edges are deleted.

**Rule 2 (Action/observation exchange)**:
```
P(Y|do(X), do(Z), W) = P(Y|do(X), Z, W)
if (Y ⊥ Z | X, W)_{G_X̄, Z}
```
Can replace do(Z) with observation Z if Z is d-separated from Y in graph with X's and Z's incoming edges deleted.

**Rule 3 (Insertion/deletion of actions)**:
```
P(Y|do(X), do(Z), W) = P(Y|do(X), W)
if (Y ⊥ Z | X, W)_{G_X̄, Z(W)}
```
Can ignore do(Z) if Z is d-separated from Y in graph with X's incoming edges deleted and Z's incoming edges from non-W nodes deleted.

**Application**: Chain together rules to derive P(Y|do(X)) from observational data.

### Identification vs Estimation

**Identification**: Can we express P(Y|do(X)) in terms of observational distributions P(·)?

**Estimation**: Given data, how do we compute P(Y|do(X))?

**Two separate problems**:
1. **Identification** (graphical, uses do-calculus)
2. **Estimation** (statistical, uses data)

**Identifiability conditions**: When can we identify causal effects from observational data?
- Backdoor criterion (Section 6)
- Front-door criterion
- do-calculus derivation

---

## Section 4: Counterfactuals (Potential Outcomes, Causal Effects)

### Counterfactual Queries

**Counterfactual**: "What would have happened if things had been different?"

**Notation**: Y_x = value Y would have taken if X had been set to x.

**Example**:
- Factual: "Patient took drug and recovered"
- Counterfactual: "Would patient have recovered WITHOUT the drug?"

**Three levels of questions** (restated):
1. **Associational**: P(recovery|drug) - Did patients who took drug recover?
2. **Interventional**: P(recovery|do(drug)) - Would patients recover if given drug?
3. **Counterfactual**: P(recovery_no-drug|drug, recovery) - Would THIS patient who took drug and recovered have recovered without it?

### Twin Networks for Counterfactual Inference

**Method**: Create parallel "worlds" (factual and counterfactual).

**Procedure**:
1. Start with SCM
2. Observe factual evidence (e.g., X=x, Y=y)
3. Infer posterior over exogenous variables U given evidence
4. Create counterfactual world with intervention do(X=x')
5. Simulate Y in counterfactual world using same U

**Example** (linear SCM):
```
U_X ~ N(0, 1)
U_Y ~ N(0, 1)
X = U_X
Y = 2X + U_Y
```

**Factual**: Observe X=1, Y=3
**Infer**: U_X = 1, U_Y = 3 - 2(1) = 1
**Counterfactual**: What if X had been 0?
→ Y_0 = 2(0) + U_Y = 2(0) + 1 = 1

So if X had been 0 (instead of 1), Y would have been 1 (instead of 3).

### Individual Treatment Effects (Heterogeneous Effects)

**ITE** (Individual Treatment Effect):
```
ITE_i = Y_i(1) - Y_i(0)
```

Never directly observable (fundamental problem), but can be estimated using:
- Conditional Average Treatment Effect (CATE): E[Y(1) - Y(0) | X=x]
- Meta-learners (T-learner, S-learner, X-learner)
- Causal forests, neural networks with counterfactual regularization

**Heterogeneous effects**: Treatment effects vary by individual/context.

**Applications**:
- Personalized medicine (who benefits most from drug?)
- Targeted advertising (who responds to ad?)
- Policy design (which populations benefit from intervention?)

### Bounds on Counterfactuals

**Tight bounds**: When we can't point-identify counterfactuals, we can often derive bounds.

**Example** (Balke-Pearl bounds):
Given observational data + instrumental variable, can derive bounds on P(Y_1 = 1) even without full identification.

**Sensitivity analysis**: How much would our causal estimate change under violations of assumptions?

---

## Section 5: Causal Discovery (Structure Learning, Constraint-Based, Score-Based)

### The Causal Discovery Problem

**Goal**: Infer causal DAG from observational data alone.

**Input**: Dataset with variables V = {V1, V2, ..., Vn}
**Output**: DAG (or equivalence class of DAGs) over V

**Why it's hard**:
- Observational data alone cannot distinguish X → Y from Y → X (both produce same correlations)
- Need additional assumptions (e.g., faithfulness, causal sufficiency)

**Markov equivalence class**: Set of DAGs that encode the same conditional independencies.

### Constraint-Based Methods (PC Algorithm)

**Approach**: Test conditional independencies in data, construct DAG consistent with those independencies.

**PC Algorithm** (Peter-Clark):

**Phase 1**: Start with complete undirected graph
**Phase 2**: Remove edges between conditionally independent variables:
- Test X ⊥ Y | {} (marginally independent)
- Test X ⊥ Y | {Z} for all Z
- Test X ⊥ Y | {Z1, Z2} for all pairs
- Continue increasing conditioning set size

**Phase 3**: Orient edges using v-structures (colliders):
- If X - Z - Y and X,Y not adjacent → orient as X → Z ← Y

**Phase 4**: Apply orientation rules to propagate orientations

**Assumptions**:
- **Causal Markov condition**: DAG encodes all conditional independencies
- **Faithfulness**: All conditional independencies come from the DAG (no "accidental" independencies)
- **Causal sufficiency**: No unmeasured confounders

**Advantages**: Theoretically sound, works for large graphs
**Disadvantages**: Sensitive to test errors, assumes faithfulness

### Score-Based Methods (GES, Greedy Equivalence Search)

**Approach**: Define score function measuring fit of DAG to data, search over DAG space.

**Score functions**:
- **BIC** (Bayesian Information Criterion): Likelihood - penalty for complexity
- **BDeu** (Bayesian Dirichlet equivalent uniform): Bayesian score

**GES Algorithm**:

**Forward phase**: Greedily add edges that improve score
**Backward phase**: Greedily remove edges that improve score

**Search space**: Space of equivalence classes (not individual DAGs).

**Advantages**: Can handle dense graphs better than PC
**Disadvantages**: Computationally expensive, can get stuck in local optima

### Functional Causal Models and Identifiability

**Key insight**: With additional assumptions (beyond independence), can sometimes identify causal direction.

**Linear Non-Gaussian Acyclic Model (LiNGAM)**:
- Assumes linear functions with non-Gaussian noise
- Can identify full causal order!

**Why non-Gaussianity helps**: Gaussian distributions are "too symmetric" (can't distinguish cause from effect). Non-Gaussian breaks symmetry.

**Additive Noise Models (ANMs)**:
```
Y = f(X) + N, N ⊥ X
```

If f is nonlinear, can test whether X → Y or Y → X fits better.

### Causal Discovery with Interventions

**Active learning for causal discovery**: Choose which interventions to perform to maximize information about DAG.

**Result**: Interventional data can drastically reduce number of experiments needed to identify DAG.

**Example**: Single intervention on X can distinguish X → Y from Y → X:
- If X → Y: Intervention on X changes distribution of Y
- If Y → X: Intervention on X does NOT change distribution of Y

---

## Section 6: Confounding and Bias (Backdoor Criterion, Front-Door Adjustment)

### Backdoor Criterion

**Backdoor paths**: Paths from X to Y that go through arrows INTO X.

**Example**:
```
Z → X → Y
  ↓
  W
```
Backdoor path: X ← Z → W (goes into X)

**Backdoor criterion** (Pearl): Set Z satisfies backdoor criterion for (X, Y) if:
1. No node in Z is a descendant of X
2. Z blocks all backdoor paths from X to Y

**If backdoor criterion satisfied**:
```
P(Y|do(X=x)) = Σ_z P(Y|X=x, Z=z) P(Z=z)
```

This is just **adjustment formula** / **standardization**.

**Practical interpretation**: Control for Z (e.g., in regression) to get unbiased causal estimate.

### Identifying Confounders

**Confounder**: Variable that opens backdoor path.

**Not all correlated variables are confounders!**

**Example** (mediator is NOT confounder):
```
X → M → Y
```
M is on causal path (mediator), not backdoor path. Controlling for M would BLOCK the causal effect!

**Rule**: Only control for variables on backdoor paths (common causes), not mediators or colliders.

### Front-Door Criterion

**Front-door adjustment**: Identifies causal effect even with unmeasured confounding, if there's a mediator.

**Setup**:
```
      U (unmeasured)
     ↙ ↘
    X → M → Y
```

**Front-door criterion**: M satisfies front-door for (X, Y) if:
1. M intercepts all causal paths from X to Y
2. No backdoor paths from X to M
3. All backdoor paths from M to Y are blocked by X

**Formula**:
```
P(Y|do(X=x)) = Σ_m P(M=m|X=x) Σ_x' P(Y|M=m, X=x') P(X=x')
```

**Classic example**: Smoking (X) → Tar (M) → Cancer (Y), with unmeasured genetic confounder.

### Instrumental Variables

**Instrumental variable (IV)**: Variable Z that affects Y only through X.

**Requirements**:
1. Z → X (relevance): IV affects treatment
2. Z → Y only through X (exclusion restriction): IV doesn't directly affect outcome
3. No confounding of Z-Y relationship

**Classic example**:
```
  Draft Lottery (Z) → Military Service (X) → Earnings (Y)
```

Draft lottery (randomly assigned) affects military service, which affects earnings. Lottery doesn't directly affect earnings (only through service).

**IV estimand** (with binary Z, X):
```
LATE = E[Y|Z=1] - E[Y|Z=0] / E[X|Z=1] - E[X|Z=0]
```

**LATE** (Local Average Treatment Effect): Effect for compliers (those whose treatment status is affected by instrument).

---

## Section 7: Causal Machine Learning (Causal Representation Learning, IRM)

### Causal Representation Learning

**Goal**: Learn representations that respect causal structure.

**Why important**: Standard ML learns correlations, which break under distribution shift. Causal representations are robust.

**Approaches**:
- Learn disentangled representations (each dimension = one causal factor)
- Learn representations invariant to spurious correlations
- Discover causal variables from raw observations (images → objects)

**Example**: Image classification
- Spurious correlation: Cows on grass, camels on sand
- Standard model: Learns grass/sand as features
- Causal model: Learns actual animal features (robust to background shift)

### Invariant Risk Minimization (IRM)

**Problem**: Models trained on one distribution fail on others (distribution shift).

**Key idea**: Find predictor that is optimal across ALL environments simultaneously.

**IRM objective**:
```
min_Φ Σ_e R_e(Φ)
subject to: w_e^* = argmin_w R_e(w ∘ Φ) is the same for all e
```

Where:
- Φ: Representation (learned)
- w: Classifier (linear head)
- R_e: Risk in environment e

**Interpretation**: Learn representation Φ such that the optimal classifier w is INVARIANT across environments.

**Why this helps**: Invariant features are causal (by assumption that causal relationships don't change across environments).

### Counterfactual Prediction

**Task**: Predict Y under intervention on X, using only observational data.

**Applications**:
- Drug response prediction: "What would happen if we gave this patient drug A instead of B?"
- Policy evaluation: "What would GDP be if we implemented policy X?"

**Methods**:
- **Propensity score methods**: Weight observations to mimic randomization
- **Doubly robust estimation**: Combine outcome model + propensity model
- **Causal forests**: Random forests adapted for heterogeneous treatment effects
- **Neural causal models**: Deep learning with causal inductive biases

### Causal Regularization for Robustness

**Standard ML loss**:
```
L = Σ_i (y_i - f(x_i))²
```

**Causal regularization**: Add penalty for violations of causal structure.

**Example** (invariance penalty):
```
L = Σ_i (y_i - f(x_i))² + λ · penalty(f learns spurious correlations)
```

**Methods**:
- IRM penalty: Penalize gradient norm of classifier across environments
- Decorrelation penalties: Penalize correlations between representation and spurious features
- Counterfactual consistency: Ensure model predictions satisfy counterfactual constraints

### Causal Reinforcement Learning

**Problem**: RL agents learn correlations (e.g., "press button when light is on" without understanding button causes light).

**Causal RL**: Incorporate causal knowledge/discovery into RL.

**Benefits**:
- Transfer learning across environments
- Credit assignment (which action caused which outcome?)
- Robustness to distribution shift

**Approaches**:
- Learn causal models of environment (world models)
- Use do-calculus for off-policy evaluation
- Discover causal structure from interaction data

---

## Section 8: ARR-COC-0-1 Causal Relevance (Participatory Knowing = Causal Intervention)

### Participatory Knowing as Causal Agency

**Vervaeke's participatory knowing**: Knowing by being-in-the-world, transformative relationship between agent and arena.

**Causal interpretation**: Participatory knowing is CAUSAL knowing - understanding how your actions affect the world.

**In ARR-COC-0-1**:
- **Propositional knowing**: Measure information content (statistical knowing)
- **Perspectival knowing**: Measure salience landscapes (phenomenal knowing)
- **Participatory knowing**: Understand how token allocation affects task performance (causal knowing)

**Key insight**: The query isn't just passively observed - it's an INTERVENTION in the visual processing pipeline.

### Query as Causal Intervention

**Standard VLM**: Query and image are both observed variables (conditioning).

**ARR-COC-0-1**: Query is an INTERVENTION that restructures visual processing.

**Causal model**:
```
Query (intervention)
  ↓
Token Allocation
  ↓
Compressed Features
  ↓
Task Performance
```

**Query = do(Query)**: When we provide a query, we're actively intervening in the relevance realization process.

**Why this matters**:
- Standard attention: P(Attended Features | Query) - observational
- Relevance realization: P(Attended Features | do(Query)) - interventional

The query doesn't just passively select features; it CAUSES feature selection through relevance realization.

### Causal Effect of Token Allocation

**Estimand**: What is the causal effect of allocating more tokens to patch i on task performance?

**Observational approach** (biased):
```
Correlate(Tokens_i, Performance)
```

Biased because high tokens might be allocated to already-salient regions (confounding).

**Interventional approach** (unbiased):
```
E[Performance | do(Tokens_i = high)] - E[Performance | do(Tokens_i = low)]
```

**Implementation**:
- Randomize token allocation across patches (like an RCT)
- Measure downstream task performance
- Estimate causal effect of token budget on performance

### Counterfactual Token Allocation

**Counterfactual question**: "If we had allocated tokens differently, would task performance have improved?"

**Twin network approach**:

**Factual world**:
1. Query provided
2. Relevance scores computed
3. Tokens allocated based on relevance
4. Task performance = P_factual

**Counterfactual world**:
1. Same query (same exogenous factors)
2. INTERVENE on token allocation (e.g., uniform allocation)
3. Task performance = P_counterfactual

**Individual Treatment Effect**:
```
ITE = P_factual - P_counterfactual
```

This tells us how much our relevance-driven allocation helped (or hurt) for THIS specific query-image pair.

### Backdoor Paths in Visual Relevance

**Potential confounders** in relevance realization:

**Example DAG**:
```
    Image Statistics (U)
      ↙           ↘
Visual Saliency → Token Allocation → Task Performance
```

Image statistics (e.g., contrast, clutter) affect BOTH visual saliency AND task performance directly.

**Problem**: If we just correlate token allocation with performance, we'd confound:
- Effect of relevance-driven allocation
- Effect of intrinsic image properties

**Solution (backdoor adjustment)**: Control for image statistics when estimating causal effect of token allocation.

```
Effect = E[Performance | do(Tokens), Image-Stats]
```

### Causal Evaluation of ARR-COC-0-1

**Standard evaluation** (associational):
- Does ARR-COC-0-1 correlate with better performance than baseline?

**Causal evaluation** (interventional):
- Does ARR-COC-0-1's token allocation CAUSE better performance?

**Methods**:

**1. Ablation as intervention**:
- do(No relevance realization) = Uniform token allocation
- Measure P(Performance | do(Uniform)) vs P(Performance | do(ARR-COC))

**2. Controlled experiments**:
- Match image pairs on confounders (scene complexity, object count, etc.)
- Randomize allocation strategy
- Measure causal effect

**3. Instrumental variables**:
- Use query type as instrument (affects allocation but not performance directly)
- Estimate causal effect even with unmeasured confounders

### Participatory Training via Causal Objectives

**Standard training**: Maximize task performance (correlational objective).

**Causal training**: Maximize causal effect of relevance allocation.

**Causal loss function**:
```
L_causal = E[ (y_true - f(x, do(relevance_allocation)))² ]
```

**Implementation**:
- Sample counterfactual allocations during training
- Ensure model learns causal relationships (not just correlations)
- Penalize reliance on spurious features

**Why this helps**: Model learns to allocate tokens based on genuine causal impact, not just correlation with labels.

### Robust Relevance Realization via Causal Invariance

**Problem**: Relevance patterns might be dataset-specific (spurious correlations).

**Example**:
- Training data: "Where is the cat?" → Cat usually on furniture
- Model learns: Allocate tokens to furniture (spurious correlation)
- Test: Cat on floor → Model fails (distribution shift)

**Causal solution** (IRM for relevance):

**Objective**: Learn relevance realization that is INVARIANT across different visual domains.

**Implementation**:
```
Train on multiple visual domains (indoor, outdoor, synthetic, etc.)
Require: Relevance allocation strategy that works optimally in ALL domains
Result: Model learns causal features (object identity) not spurious correlations (typical context)
```

**Benefit**: ARR-COC-0-1 becomes robust to distribution shift, because it relies on causal relationships (query → object relevance) rather than correlations (query → contextual patterns).

---

## Sources

**Web Research (Causal Inference & DAGs):**

From [What Is a Causal Graph?](https://arxiv.org/pdf/2402.09429) (arXiv, 2024, accessed 2025-11-14):
- Pearlian DAG semantics and structural causal models
- do-calculus framework for causal inference
- Graphical representation of causal relationships

From [Through the lens of causal inference](https://apertureneuro.org/article/124817-through-the-lens-of-causal-inference-decisions-and-pitfalls-of-covariate-selection) (Aperture Neuro, 2024):
- Domain knowledge in covariate selection
- Practical guidance for causal analysis
- Confounding identification and adjustment

From [Methods in causal inference Part 1: causal diagrams and confounding](https://www.cambridge.org/core/journals/evolutionary-human-sciences/article/methods-in-causal-inference-part-1-causal-diagrams-and-confounding/E734F72109F1BE99836E268DF3AA0359) (Cambridge, 2024):
- Safe integration of causal diagrams into workflows
- Confounding assessment using DAGs
- Starting with clearly defined causal questions

From [Causal Inference 3: Counterfactuals](https://www.inference.vc/causal-inference-3-counterfactuals/) (Ferenc Huszár, 2019):
- Counterfactual reasoning limitations with do-calculus
- Twin networks for counterfactual inference
- Clash of factual and counterfactual worlds

From [The Do-Calculus Revisited](https://arxiv.org/pdf/1210.4852) (Pearl, 2012):
- Complete do-calculus rules and proofs
- Interventions and counterfactuals via do(x) operator
- Physical intervention simulation

From [An introduction to causal discovery](https://sjes.springeropen.com/articles/10.1186/s41937-024-00131-4) (Swiss Journal of Economics, 2024):
- Causal structure learning from data
- Difference from traditional causal inference
- Uncovering relationships among multiple variables

From [Causal Structure Learning: A Combinatorial Perspective](https://link.springer.com/article/10.1007/s10208-022-09581-9) (Springer, 2023):
- Constraint-based and score-based methods
- Combinatorial structure over directed graphs
- Identifiability and estimation separation

From [Causal machine learning for predicting treatment outcomes](https://www.nature.com/articles/s41591-024-02902-1) (Nature Medicine, 2024):
- Flexible data-driven methods for treatment prediction
- Efficacy and toxicity prediction using causal ML
- Heterogeneous treatment effect estimation

From [Comprehensive Causal Machine Learning](https://arxiv.org/abs/2405.10198) (Lechner, 2024):
- Modified causal forest (mcf) comparison
- Generalized random forest (grf) approaches
- Double machine learning (dml) methods

**Existing Knowledge (Vervaeke Participatory Knowing):**

From `.claude/skills/john-vervaeke-oracle/concepts/00-relevance-realization/00-overview.md`:
- Agent-arena coupling as participatory knowing
- Transjective ontology (neither objective nor subjective)
- Relevance realization as active process

From `.claude/skills/karpathy-deep-oracle/cognitive-foundations/00-active-inference-free-energy.md`:
- Active inference as causal intervention
- Perception and action in same framework
- Free energy minimization through intervention

From `.claude/skills/karpathy-deep-oracle/cognitive-foundations/02-bayesian-brain-probabilistic.md`:
- Bayesian inference under uncertainty
- Posterior inference combining priors and evidence
- Uncertainty representation and propagation
