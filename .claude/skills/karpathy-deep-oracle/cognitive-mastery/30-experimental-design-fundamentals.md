# Experimental Design Fundamentals

**Location**: `cognitive-mastery/30-experimental-design-fundamentals.md`

**Purpose**: Comprehensive foundation in experimental design principles for cognitive-behavioral research, covering variables, control conditions, randomization, counterbalancing, and between/within-subjects designs with engineering implementation considerations.

---

## Overview

Experimental design is the systematic process of planning experiments to test hypotheses and establish cause-and-effect relationships. In cognitive science and machine learning research, proper experimental design is critical for drawing valid conclusions about model performance, human behavior, and cognitive processes.

From [iMotions Experimental Design Guide](https://imotions.com/blog/learning/research-fundamentals/experimental-design/) (accessed 2025-11-16):
> "Experiments are the professional way to answer questions, identify cause and effect or determine predictors and outcomes. These insights help us understand how and why things are what they are and can ultimately be used to change the world by improving the good and overcoming the bad."

**Core Principle**: Modern experiments are not random probing but careful planning of data collection, proper control of variables, and appropriate statistical analysis.

---

## Section 1: Independent and Dependent Variables

### 1.1 Independent Variables (IV)

From [Scribbr Variables Guide](https://www.scribbr.com/methodology/independent-and-dependent-variables/) (accessed 2025-11-16):
> "The independent variable is the cause. Its value is independent of other variables in your study."

**Definition**: The independent variable is what you manipulate, control, or vary in an experimental study to explore its effects.

**Also called**:
- Explanatory variables (they explain an event or outcome)
- Predictor variables (they can be used to predict the value of a dependent variable)
- Factors (in experimental design terminology)
- Right-hand-side variables (in regression equations)

**Two main types**:

1. **Experimental independent variables** - Can be directly manipulated by researchers
   - Example: Different learning rates in neural network training (0.001, 0.01, 0.1)
   - Example: Presence/absence of attention mechanism in model architecture

2. **Subject variables** - Cannot be manipulated but can categorize research subjects
   - Example: Gender, age, education level
   - Example: Prior experience with task domain
   - Note: Creates quasi-experimental designs, not true experiments

### 1.2 Dependent Variables (DV)

From [Scribbr Variables Guide](https://www.scribbr.com/methodology/independent-and-dependent-variables/) (accessed 2025-11-16):
> "The dependent variable is the effect. Its value depends on changes in the independent variable."

**Definition**: The outcome you measure that changes as a result of independent variable manipulation.

**Also called**:
- Response variables (they respond to a change in another variable)
- Outcome variables (they represent the outcome you want to measure)
- Left-hand-side variables (in regression equations)

**Examples in ML/VLM research**:
- Model accuracy, precision, recall, F1 score
- Inference time, memory usage
- Human annotation agreement rates
- Eye fixation duration on image patches
- Cognitive load measurements (EEG, pupil dilation)

### 1.3 Relationship Between Variables

**Hypothesis structure**: Independent variable → Dependent variable

Example: "Increasing visual token budget (IV) from 64 to 400 tokens improves image captioning accuracy (DV) on complex scenes."

**Key principle**: The independent variable must come **before** the dependent variable in time to establish causation.

---

## Section 2: Control Conditions and Variables

### 2.1 Control Conditions

From [iMotions Experimental Design Guide](https://imotions.com/blog/learning/research-fundamentals/experimental-design/) (accessed 2025-11-16):
> "A control condition allows researchers to compare results of the dependent variable at the different levels of the independent variable to a standard to establish a relationship between the two."

**Types of control conditions**:

1. **No-treatment control** - Participants receive no intervention
   - Example: Baseline VLM without adaptive token allocation

2. **Placebo control** - Participants receive inactive treatment
   - Example: VLM with fixed random token allocation (appears adaptive but isn't)

3. **Waitlist control** - Participants receive treatment later
   - Common in longitudinal intervention studies

### 2.2 Control Variables

**Purpose**: Hold constant factors that could influence outcomes but aren't of primary interest.

**In VLM experiments**:
- Same base model architecture across conditions
- Identical training data and preprocessing
- Fixed computational budget (FLOPs)
- Consistent evaluation metrics and datasets
- Same random seed for reproducibility

**Critical principle**: Change only ONE independent variable at a time to isolate its effect.

---

## Section 3: Randomization

### 3.1 Purpose of Randomization

From [iMotions Experimental Design Guide](https://imotions.com/blog/learning/research-fundamentals/experimental-design/) (accessed 2025-11-16):
> "The assignment to experimental groups is done in a randomized fashion, such that all respondents have the same probability for ending up in the available experimental groups. There should not be any bias to assign specific respondents to one group or the other."

**Why randomize?**
- Eliminates systematic bias in group assignment
- Balances out unknown confounding variables
- Ensures groups are comparable at baseline
- Strengthens causal inference

### 3.2 Random Assignment vs Random Sampling

**Random sampling**: How you select participants from population
- Affects external validity (generalizability)

**Random assignment**: How you assign participants to experimental conditions
- Affects internal validity (causal inference)
- More critical for experimental control

### 3.3 Randomization in ML Experiments

**Training data splits**:
- Randomly split into train/val/test sets
- Stratified randomization to maintain class balance
- Fixed random seed for reproducibility

**Hyperparameter search**:
- Random search over hyperparameter space
- Prevents systematic bias toward certain configurations

**Stimulus presentation order**:
- Randomize order of test images across trials
- Prevents learning effects and order bias

---

## Section 4: Counterbalancing

### 4.1 What is Counterbalancing?

From [Scribbr Within-Subjects Design](https://www.scribbr.com/methodology/within-subjects-design/) (accessed 2025-11-16):
> "Randomization means using many different possible sequences for treatments, while counterbalancing means using a limited number of sequences across the group."

**Purpose**: Control for order effects when same participants experience multiple conditions.

**Key difference from randomization**:
- **Counterbalancing**: Fixed sequences, each used equally often
- **Randomization**: All possible sequences, frequency not controlled

### 4.2 Types of Counterbalancing

**Complete counterbalancing**: All possible orders of conditions
- With 3 conditions (A, B, C): 6 possible orders (3! = 6)
- ABC, ACB, BAC, BCA, CAB, CBA
- Each order used equal number of times
- Becomes impractical with many conditions (4! = 24, 5! = 120)

**Partial counterbalancing**: Subset of possible orders
- Latin square design: Each condition appears once in each position
- Balanced for first-order carryover effects

**Example for VLM evaluation**:
Testing 4 different attention mechanisms (A, B, C, D):
- Group 1: A → B → C → D
- Group 2: B → C → D → A
- Group 3: C → D → A → B
- Group 4: D → A → B → C

Each mechanism appears equally often in each position (1st, 2nd, 3rd, 4th).

### 4.3 When to Use Counterbalancing

**Use when**:
- Same participants tested under multiple conditions
- Order of conditions might affect outcomes
- Practice effects or fatigue are concerns

**Not needed when**:
- Using between-subjects design
- Order effects theoretically impossible
- Conditions presented simultaneously

---

## Section 5: Between-Subjects Design

### 5.1 Definition

From [Scribbr Within-Subjects Design](https://www.scribbr.com/methodology/within-subjects-design/) (accessed 2025-11-16):
> "In a between-subjects design, every participant experiences only one condition, and researchers assess group differences between participants in various conditions."

**Structure**: Different groups of participants for each experimental condition.

**Example**: Testing three VLM architectures
- Group A: Tests VLM with Q-Former
- Group B: Tests VLM with perceiver resampler
- Group C: Tests VLM with cross-attention pooling

### 5.2 Advantages

**Clean separation**:
- No carryover effects between conditions
- No practice or fatigue effects
- Simpler to implement and analyze

**Ecological validity**:
- More realistic for one-time deployment scenarios
- Users typically don't compare multiple systems directly

### 5.3 Disadvantages

**Individual differences**:
- Groups may differ in ways that affect outcomes
- Age, experience, ability vary between groups
- Requires larger sample sizes to detect effects

**Less statistical power**:
- Between-group variance includes individual differences
- Need more participants to achieve same power as within-subjects

**Higher cost**:
- More participants needed
- More time and resources required

### 5.4 Requirements

**Random assignment**: Critical for internal validity
- Ensures groups comparable at baseline
- Balances out individual differences

**Large sample size**: Compensate for individual variation
- Typically need 2× participants vs within-subjects
- Power analysis determines exact N needed

---

## Section 6: Within-Subjects Design

### 6.1 Definition

From [Scribbr Within-Subjects Design](https://www.scribbr.com/methodology/within-subjects-design/) (accessed 2025-11-16):
> "In a within-subjects design, or a within-groups design, all participants take part in every condition."

**Also called**:
- Repeated measures design
- Dependent groups design
- Crossover design

**Structure**: Same participants experience all experimental conditions.

**Example**: VLM attention comparison
- All participants rate images processed by:
  1. Standard uniform attention
  2. Foveated attention (high center, low periphery)
  3. Relevance-weighted attention

### 6.2 Advantages

From [Scribbr Within-Subjects Design](https://www.scribbr.com/methodology/within-subjects-design/) (accessed 2025-11-16):

**Smaller sample required**:
> "Within-subjects designs help you detect causal or correlational relationships between variables with relatively small samples."

**Controls individual differences**:
> "In contrast, there are no variations in individual differences between conditions in a within-subjects design because the same individuals participate in all conditions."

**Statistically powerful**:
> "A within-subjects design is more statistically powerful than a between-subjects design, because individual variation is removed."

**Practical advantages**:
- Each participant is their own control
- Fewer participants needed (cost effective)
- Direct comparison within individuals

### 6.3 Disadvantages

From [Scribbr Within-Subjects Design](https://www.scribbr.com/methodology/within-subjects-design/) (accessed 2025-11-16):

**Carryover effects**:
- **Practice effects**: Performance improves with familiarity
- **Fatigue effects**: Performance declines with exhaustion
- **Order effects**: Position in sequence affects outcomes
- **Contrast effects**: Earlier conditions influence judgment of later ones

**Time-related threats**:
- **History**: External events influence outcomes
- **Maturation**: Natural changes over time (learning, aging)
- **Subject attrition**: Participants drop out differentially

**Demand characteristics**:
- Participants may guess study purpose
- Behavior changes based on perceived expectations

### 6.4 Controlling for Order Effects

**Counterbalancing** (described in Section 4):
- Each condition appears equally often in each position
- Balances out systematic order effects

**Randomization**:
- Completely random order for each participant
- Distributes order effects randomly across conditions

**Time delays**:
- Insert delays between conditions
- Reduces carryover effects
- May not be practical for all studies

---

## Section 7: Tensor Parallelism for Experimental Parallelization

From [karpathy/distributed-training/02-megatron-lm-tensor-parallelism.md](../karpathy/distributed-training/02-megatron-lm-tensor-parallelism.md):
> "Tensor parallelism (TP) splits individual layers horizontally across GPUs... enabling training models that don't fit in single GPU memory while minimizing communication overhead."

**Application to experimental design**:

**Parallel condition evaluation**:
- Split experimental conditions across GPUs
- Each GPU evaluates subset of conditions simultaneously
- Aggregate results for statistical analysis

**Example**: Testing 8 different token allocation strategies
- TP=8: Each GPU runs one strategy
- Simultaneous evaluation reduces wall-clock time
- Identical hardware ensures fair comparison

**Implementation considerations**:
```python
# Pseudocode for parallel experimental conditions
def run_parallel_experiment(conditions, data, tp_size=8):
    # Split conditions across tensor parallel groups
    conditions_per_gpu = len(conditions) // tp_size

    results = []
    for tp_rank in range(tp_size):
        # Each GPU evaluates its assigned conditions
        local_conditions = conditions[
            tp_rank * conditions_per_gpu:
            (tp_rank + 1) * conditions_per_gpu
        ]

        local_results = evaluate_conditions(
            local_conditions, data, tp_rank
        )
        results.append(local_results)

    # All-gather results for analysis
    return aggregate_results(results)
```

**Benefits**:
- Faster experimental iteration
- Identical computational environment across conditions
- Easy to add/remove conditions by adjusting TP size

---

## Section 8: Triton Inference Server for A/B Testing

From [karpathy/inference-optimization/02-triton-inference-server.md](../karpathy/inference-optimization/02-triton-inference-server.md):
> "Triton Inference Server is NVIDIA's open-source inference serving platform designed for production deployment of ML models. Unlike vLLM (LLM-specific) or TensorFlow Serving (framework-specific), Triton is framework-agnostic and multimodal-capable."

**Application to experimental design**:

**Multi-model A/B testing**:
- Deploy multiple experimental conditions as separate models
- Triton routes requests randomly or via counterbalancing
- Collect metrics for each condition simultaneously

**Dynamic batching for fair comparison**:
From [karpathy/inference-optimization/02-triton-inference-server.md](../karpathy/inference-optimization/02-triton-inference-server.md):
> "Dynamic Batching = Server-side request combining to create batches dynamically... Triton delays requests briefly to accumulate more requests into larger batches."

**Benefits for experimental design**:
- Consistent inference conditions across experimental groups
- Automatic load balancing
- Real-time metrics collection
- Easy condition switching (model versioning)

**Example configuration**:
```protobuf
# Model A: Baseline attention
model {
  name: "vlm_baseline"
  platform: "pytorch"
  version_policy { specific { versions: 1 }}
}

# Model B: Relevance-weighted attention
model {
  name: "vlm_relevance"
  platform: "pytorch"
  version_policy { specific { versions: 1 }}
}

# Ensemble for A/B routing
ensemble_scheduling {
  step {
    model_name: "vlm_baseline"
    model_version: 1
  }
  step {
    model_name: "vlm_relevance"
    model_version: 1
  }
}
```

---

## Section 9: Intel oneAPI for Cross-Platform Experiments

From [karpathy/alternative-hardware/02-intel-oneapi-ml.md](../karpathy/alternative-hardware/02-intel-oneapi-ml.md):
> "Intel oneAPI is an open, unified programming model that enables developers to accelerate applications across CPUs, GPUs, and other accelerators using a single codebase."

**Application to experimental design**:

**Hardware as independent variable**:
- Compare model performance across different hardware
- NVIDIA GPUs vs Intel Data Center GPUs vs AMD GPUs
- Control for software differences using oneAPI

**Cross-platform reproducibility**:
```python
import torch
import intel_extension_for_pytorch as ipex

def run_experiment(model, data, device='xpu'):
    """Run experiment on Intel or NVIDIA hardware"""
    model = model.to(device)  # 'xpu' for Intel, 'cuda' for NVIDIA
    data = data.to(device)

    # Optimize for target hardware
    if device == 'xpu':
        model = ipex.optimize(model)

    # Measure performance
    results = evaluate(model, data)
    return results
```

**Benefits**:
- Fair comparison across hardware platforms
- Single codebase reduces implementation confounds
- Validates generalizability of findings

**Hardware independence testing**:
- Ensures experimental results not artifacts of specific hardware
- Critical for deployment to heterogeneous environments

---

## Section 10: ARR-COC-0-1 Experimental Validation

**Application to ARR-COC relevance realization**:

### 10.1 Between-Subjects Design for ARR-COC

**Independent variable**: Relevance realization approach
- Group A: Fixed uniform token allocation (64 tokens/patch)
- Group B: Uniform LOD pyramid (no relevance weighting)
- Group C: ARR-COC adaptive allocation (64-400 tokens)

**Dependent variables**:
- Image captioning accuracy (BLEU, METEOR, CIDEr)
- Visual question answering accuracy
- Memory usage (GB)
- Inference time (ms)

**Control variables**:
- Same base VLM architecture (Qwen3-VL)
- Identical training data and procedure
- Same evaluation datasets
- Fixed total computational budget

### 10.2 Within-Subjects Design for Query Effects

**Structure**: All participants evaluate same images under different query conditions

**Independent variable**: Query specificity
- Generic: "Describe this image"
- Specific: "What color is the car in the foreground?"
- Complex: "Compare the architectural styles visible"

**Dependent variables**:
- Token allocation distribution
- Gaze pattern correlation with allocations
- Response accuracy

**Counterbalancing**: Latin square design for query order
```
Participant 1: Generic → Specific → Complex
Participant 2: Specific → Complex → Generic
Participant 3: Complex → Generic → Specific
```

### 10.3 Mixed Design: Query × Model Type

**Between-subjects factor**: Model type (different groups)
- Baseline uniform allocation
- ARR-COC adaptive allocation

**Within-subjects factor**: Query type (same participants)
- All participants see all query types

**Advantages**:
- Tests both model superiority (between) and query sensitivity (within)
- More powerful than pure between-subjects for query effects
- Efficient use of participants

**Statistical analysis**: Mixed ANOVA
- Main effect of model type
- Main effect of query type
- Interaction: Does ARR-COC benefit depend on query type?

### 10.4 Validating Relevance Scorers

**Experiment**: Correlation with human attention

**Design**: Within-subjects (repeated measures)
- Same participants view images while:
  1. Eye tracking records gaze patterns
  2. ARR-COC generates relevance maps
  3. Propositional scorer (Shannon entropy) runs
  4. Perspectival scorer (saliency) runs
  5. Participatory scorer (query-content) runs

**Dependent variables**:
- Correlation between scorer outputs and eye fixations
- Temporal alignment (do scorers predict next fixation?)
- Explanation accuracy (can scorers explain past fixations?)

**Randomization**:
- Random image presentation order
- Multiple fixation maps per image for reliability

**Expected outcome**: Participatory scorer (query-aware) should show higher correlation with task-driven eye movements than bottom-up saliency alone.

---

## Sources

**Web Research**:
- [iMotions Experimental Design Guide](https://imotions.com/blog/learning/research-fundamentals/experimental-design/) (accessed 2025-11-16)
- [Scribbr: Independent vs. Dependent Variables](https://www.scribbr.com/methodology/independent-and-dependent-variables/) (accessed 2025-11-16)
- [Scribbr: Within-Subjects Design](https://www.scribbr.com/methodology/within-subjects-design/) (accessed 2025-11-16)

**Karpathy Oracle Files**:
- [karpathy/distributed-training/02-megatron-lm-tensor-parallelism.md](../karpathy/distributed-training/02-megatron-lm-tensor-parallelism.md) - Tensor parallelism for parallel experimental conditions
- [karpathy/inference-optimization/02-triton-inference-server.md](../karpathy/inference-optimization/02-triton-inference-server.md) - Multi-model A/B testing infrastructure
- [karpathy/alternative-hardware/02-intel-oneapi-ml.md](../karpathy/alternative-hardware/02-intel-oneapi-ml.md) - Cross-platform experimental validation

**ARR-COC-0-1 Connection**:
Experimental design principles guide validation of relevance realization framework, from between-subjects comparisons of allocation strategies to within-subjects studies of query effects and human attention correlation.
