# Predictive Coding and Hierarchical Message Passing

## Overview

**Predictive coding** is a theoretical framework proposing that the brain continuously generates predictions about sensory inputs and updates its internal models based on prediction errors. This framework, developed extensively by **Karl Friston** and colleagues, provides a unified account of perception, learning, and attention through **hierarchical message passing** between cortical areas.

The core insight is that cortical processing involves two complementary streams:
1. **Top-down predictions** flowing from higher to lower areas
2. **Bottom-up prediction errors** flowing from lower to higher areas

This bidirectional message passing implements a form of **Bayesian inference**, where the brain approximates optimal inference about the causes of sensory data by minimizing prediction errors throughout the cortical hierarchy.

From [Friston, 2009 - Predictive coding under the free-energy principle](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/):
> "This is the essence of recurrent message passing between hierarchical levels to optimize free energy or suppress prediction error, i.e. recognition dynamics."

From [Bastos et al., 2012 - Canonical Microcircuits for Predictive Coding](https://www.sciencedirect.com/science/article/pii/S0896627312009592) (cited 2820+ times):
> "We revisit the established idea that message passing among hierarchical cortical areas implements a form of Bayesian inference—paying careful attention to the implications for intrinsic connections among neuronal populations."

---

## 1. Hierarchical Message Passing Architecture

### 1.1 The Two-Stream Framework

**Predictive coding postulates two functionally distinct streams**:

**Top-Down Stream (Predictions)**:
- Carries **expectations** about lower-level activity
- Originates from **deep layers** (layer 5/6) of higher areas
- Terminates in **superficial layers** (layer 1/2/3) of lower areas
- Conveyed primarily by **low-frequency oscillations** (alpha/beta, 8-30 Hz)
- Represents the brain's **generative model** of the world

**Bottom-Up Stream (Prediction Errors)**:
- Carries the **mismatch** between predictions and actual input
- Originates from **superficial layers** (layer 2/3) of lower areas
- Terminates in **layer 4** of higher areas
- Conveyed primarily by **high-frequency oscillations** (gamma, 30-100 Hz)
- Drives updating of the generative model

From [Kanai et al., 2015 - Cerebral hierarchies: predictive processing, precision and the pulvinar](https://pmc.ncbi.nlm.nih.gov/articles/PMC4387510/):
> "The message passing implied by predictive coding would require these layer 5 principal cells to respond, in a U-shaped fashion, to both high and low levels of precision."

### 1.2 Anatomical Correspondence

**The architecture maps remarkably well onto known cortical anatomy**:

| Feature | Anatomical Substrate | Functional Role |
|---------|---------------------|-----------------|
| Feedforward connections | Layers 2/3 → Layer 4 | Prediction error transmission |
| Feedback connections | Layers 5/6 → Layers 1/2/3 | Prediction transmission |
| Lateral connections | Within-layer horizontal | Contextual modulation |
| Superficial pyramidal | Layers 2/3 | Error computation |
| Deep pyramidal | Layers 5/6 | Prediction generation |

**Asymmetric connectivity patterns**:
- **Feedforward**: Driving synapses (class 1), fast dynamics
- **Feedback**: Modulatory synapses (class 2), slower dynamics
- **Frequency separation**: Gamma (FF) vs alpha/beta (FB)

### 1.3 Mathematical Formulation

**At each level _i_ of the hierarchy**:

**State variables**:
- **r_i**: Representation (neural activity encoding causes)
- **e_i**: Prediction error (mismatch signal)

**Message passing equations** (Friston, 2009):

```
Prediction from level i+1: r̂_i = g(r_{i+1})
Prediction error at level i: e_i = r_i - r̂_i
Update rule: Δr_i = e_{i-1} - ∂g/∂r_i · e_i
```

Where:
- **g(·)**: Generative function (nonlinear mapping from causes to predictions)
- **∂g/∂r_i**: Gradient of generative function (sensitivity to changes)
- **Δr_i**: Change in representation (gradient descent on error)

**Key insight**: Each level simultaneously:
1. **Receives predictions** from above (top-down)
2. **Generates predictions** for below (top-down)
3. **Receives errors** from below (bottom-up)
4. **Sends errors** to above (bottom-up)

---

## 2. Prediction Error Minimization

### 2.1 The Core Computational Principle

**The brain minimizes prediction error at every level of the hierarchy**:

From [Rao & Ballard, 1999](https://www.nature.com/articles/4580) - the foundational predictive coding paper:
> "We describe a hierarchical model of vision in which higher-order visual cortical areas send down predictions and the feedforward connections carry the residual errors between predictions and actual lower-level activities."

**Why minimize prediction error?**
- Efficient coding: Only transmit what's unpredicted (saves bandwidth)
- Accurate models: Good predictions = good internal models
- Adaptive behavior: Better predictions = better action selection

**Three ways to minimize prediction error**:
1. **Perceptual inference**: Update representations to better predict input
2. **Learning**: Update model parameters (synaptic weights)
3. **Active inference**: Change the input through action

### 2.2 Sparse Coding of Prediction Errors

**Prediction error neurons show sparse, event-driven activity**:

- **Most prediction error neurons are silent most of the time**
- Only fire when predictions **fail** (unexpected events)
- Creates efficient **sparse representations**
- Consistent with metabolic efficiency (minimal energy use)

**Evidence from neuroscience**:
- Mismatch negativity (MMN) in auditory cortex
- Omission responses when expected stimuli don't occur
- Surprise responses to unexpected visual features
- Repetition suppression (decreased response to predicted stimuli)

From [den Ouden et al., 2012 - How Prediction Errors Shape Perception, Attention, and Motivation](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2012.00548/full) (cited 595 times):
> "As such, a PE response in a V1 neuron signals surprise about the unexpected presence (or absence) of an oriented edge in a particular part of the visual field."

### 2.3 Hierarchical Depth and Abstraction

**Prediction errors become more abstract at higher levels**:

| Level | Example (Vision) | Prediction Error Type |
|-------|------------------|----------------------|
| V1 | Edges, orientations | Local feature mismatch |
| V2/V4 | Textures, contours | Pattern mismatch |
| IT | Objects, faces | Category mismatch |
| PFC | Goals, contexts | Abstract expectation violation |

**Temporal scales also increase with hierarchy**:
- Lower levels: Fast dynamics (milliseconds)
- Higher levels: Slow dynamics (seconds to minutes)
- Creates **temporal hierarchy** matching **abstraction hierarchy**

---

## 3. Precision Weighting and Gain Control

### 3.1 Precision as Inverse Variance

**Precision = confidence in a signal = 1/variance**:

Not all prediction errors are equally reliable. **Precision weighting** allows the brain to selectively weight signals based on their estimated reliability.

From [Feldman & Friston, 2010](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2959160/):
> "Attention may simply be the optimization of precision or gain control in the context of predictive coding."

**Mathematical formulation**:

```
Precision-weighted error: ẽ_i = Π_i · e_i

Where:
- Π_i: Precision matrix (diagonal elements = precision per dimension)
- e_i: Raw prediction error
- ẽ_i: Weighted error (used for updates)
```

**Update rule with precision**:
```
Δr_i ∝ Π_{i-1} · e_{i-1} - ∂g/∂r_i · Π_i · e_i
```

### 3.2 Attention as Precision Optimization

**Attention = Expected precision = Gain control on prediction errors**:

From [Sprevak - Predictive coding II: The computational level](https://marksprevak.com/publications/predictive-coding-ii-the-computational-level-b9f5/):
> "Precision weighting describes a scaling factor or 'gain' that is applied to each component of the sensory prediction error."

**How attention works in predictive coding**:
1. **Expect high precision** at attended location/feature
2. **Increase gain** on prediction errors from that location
3. **Prediction errors matter more** for inference
4. **Representations updated more** for attended input

**Neural implementation**:
- **Neuromodulators** (dopamine, acetylcholine, norepinephrine) signal precision
- **Synaptic gain** modulated by neuromodulatory input
- **Layer 2/3 interneurons** may gate precision

### 3.3 Precision and Confidence

**Precision reflects confidence in signals**:

**High precision sensory signals** (bright light, clear audio):
- Weight sensory prediction errors highly
- Let data drive inference
- "Trust your senses"

**Low precision sensory signals** (dim light, noise):
- Weight predictions highly
- Let prior beliefs drive inference
- "Trust your expectations"

**Dynamic precision estimation**:
- Brain learns to estimate precision from context
- Adjusts gain based on environmental statistics
- Optimal Bayesian weighting of prediction vs error

---

## 4. Cortical Microcircuits for Predictive Coding

### 4.1 Canonical Microcircuit Architecture

**Bastos et al. (2012) proposed a detailed mapping**:

From [Canonical Microcircuits for Predictive Coding](https://www.sciencedirect.com/science/article/pii/S0896627312009592):

**Layer-specific computational roles**:

**Superficial layers (2/3)**:
- **Superficial pyramidal cells**: Encode prediction errors
- **Sparse activity**: Only fire when predictions fail
- **Send feedforward**: Project to layer 4 of higher areas
- **Gamma oscillations**: Fast signaling of unexpected input

**Granular layer (4)**:
- **Spiny stellate cells**: Receive feedforward input
- **Initial processing**: First stage of error computation
- **Relay to superficial**: Pass errors up to layer 2/3

**Deep layers (5/6)**:
- **Deep pyramidal cells**: Encode predictions (expectations)
- **Dense activity**: Maintain ongoing predictions
- **Send feedback**: Project to superficial layers of lower areas
- **Alpha/beta oscillations**: Slower rhythm for predictions

### 4.2 Intrinsic Circuitry

**Within-column connectivity supports computation**:

**Error computation** (layers 2/3):
```
Error = Input - Prediction
e = x - r̂
```
- Inhibitory interneurons subtract predictions from input
- Result: residual error in superficial pyramidal output

**Prediction generation** (layers 5/6):
```
Prediction = f(Higher-level state)
r̂ = g(r_{i+1})
```
- Receive top-down from higher areas
- Generate prediction via nonlinear transformation
- Send to superficial layers for subtraction

### 4.3 Inter-areal Connectivity

**Feedforward (bottom-up) connections**:
- **Origin**: Superficial layers (2/3)
- **Target**: Layer 4 of higher area
- **Content**: Prediction errors
- **Properties**: Driving (class 1), fast, gamma-band

**Feedback (top-down) connections**:
- **Origin**: Deep layers (5/6)
- **Target**: Superficial layers (1/2/3) of lower area
- **Content**: Predictions
- **Properties**: Modulatory (class 2), slow, alpha/beta-band

**Lateral connections**:
- **Within-level**: Horizontal connections in layer 2/3
- **Function**: Contextual modulation, extra-classical RF effects
- **Precision**: May convey precision information

---

## 5. Layer-wise Computation in Detail

### 5.1 Information Flow Through Layers

**Complete cycle of predictive coding in cortical column**:

**Step 1: Receive input**
- Layer 4 receives feedforward (error from below)
- Layer 1/2/3 receive feedback (predictions from above)

**Step 2: Compute error**
- Superficial pyramidal cells compare input to prediction
- Inhibitory interneurons implement subtraction
- Error = actual - predicted

**Step 3: Generate prediction**
- Deep pyramidal cells receive from higher areas
- Transform via generative model
- Prediction sent down to lower areas

**Step 4: Update representations**
- Error signals drive plasticity (Hebbian learning)
- Representations adjusted to minimize error
- Weights updated: ΔW ∝ error × presynaptic activity

**Step 5: Transmit messages**
- Errors sent up (feedforward, gamma)
- Predictions sent down (feedback, alpha/beta)

### 5.2 Oscillatory Signatures

**Different frequencies carry different message types**:

**Gamma oscillations (30-100 Hz)**:
- Prediction errors (feedforward)
- Superficial layers
- Fast, precise timing
- Reflects unexpected input

**Alpha/Beta oscillations (8-30 Hz)**:
- Predictions (feedback)
- Deep layers
- Slower, sustained
- Reflects expectations

**Cross-frequency coupling**:
- Gamma nested in alpha/beta
- Predictions modulate when errors are transmitted
- Creates temporal coordination

From [Bastos et al., 2015](https://www.cell.com/neuron/fulltext/S0896-6273(14)01119-4):
> "Feedforward connections between cortical areas predominantly carry gamma-band rhythms, whereas feedback connections carry alpha-band and beta-band rhythms."

---

## 6. Connection to Visual Hierarchy (V1 to IT)

### 6.1 Visual Cortex as Predictive Hierarchy

**The ventral visual stream (V1 → V2 → V4 → IT) exemplifies predictive coding**:

| Area | Level | Represents | Predicts |
|------|-------|------------|----------|
| V1 | Lowest | Edges, orientations | Retinal input |
| V2 | Low-mid | Textures, contours | V1 features |
| V4 | Mid | Complex shapes, colors | V2 features |
| IT | High | Objects, faces, categories | V4 features |
| PFC | Highest | Goals, contexts | IT categories |

**Key findings**:

From [Chao et al., 2018 - Large-Scale Cortical Networks for Hierarchical Prediction](https://www.sciencedirect.com/science/article/pii/S0896627318308924) (cited 252 times):
> "Prediction and prediction-error signals arise from different cortical areas."

From [Gelens et al., 2024 - Distributed representations of prediction error signals](https://www.nature.com/articles/s41467-024-48329-7) (cited 25 times):
> "Our results demonstrate that distributed representations of prediction error signals across the cortical hierarchy are highly synergistic."

### 6.2 Classical and Extra-Classical Receptive Fields

**Predictive coding explains receptive field properties**:

**Classical receptive field (CRF)**:
- Feedforward input defines tuning
- V1: Oriented edges at specific location
- Represents prediction errors for that feature

**Extra-classical receptive field (eCRF)**:
- Surround modulation via lateral/feedback
- **Surround suppression**: Center matches surround = predicted = suppressed
- **End-stopping**: Edge continues outside CRF = predicted = suppressed
- Context modulates response magnitude

From [Jiang et al., 2021 - Predictive Coding Theories of Cortical Function](https://arxiv.org/pdf/2112.10048) (cited 34 times):
> "The two subsequent sections discuss the application of hierarchical predictive coding to the visual cortex, explaining classical & extra-classical receptive fields."

### 6.3 Attention Effects in Visual Cortex

**Attention modulates precision throughout hierarchy**:

**V1 attention effects**:
- Multiplicative gain on responses (precision weighting)
- Increased SNR for attended stimuli
- Sharpened tuning curves

**V4 attention effects**:
- Strong modulation of responses
- Biased competition between stimuli
- Winner-take-all for attended objects

**IT attention effects**:
- Object-selective responses
- Invariant representations
- Category-level prediction errors

---

## 7. Mathematical Formulation

### 7.1 Generative Model

**The brain is assumed to have a hierarchical generative model**:

```
p(sensory data, causes) = p(data | causes) × p(causes)
```

**Hierarchical factorization**:
```
p(r_0, r_1, ..., r_n) = p(r_0 | r_1) × p(r_1 | r_2) × ... × p(r_n)
```

Where:
- **r_0**: Sensory data (visual input)
- **r_1...r_n**: Hidden causes at increasingly abstract levels
- **p(r_i | r_{i+1})**: Likelihood (predictions from above)
- **p(r_n)**: Prior (highest-level expectations)

### 7.2 Recognition Dynamics

**Inference = Inversion of generative model**:

**Gradient descent on prediction error**:
```
dr_i/dt = -∂F/∂r_i
```

Where F is variational free energy (see free-energy-principle-foundations.md).

**Expanded form**:
```
dr_i/dt = Π_{i-1} × ∂g_{i-1}/∂r_i × e_{i-1} - Π_i × e_i
```

**Components**:
- **First term**: Bottom-up (precision-weighted error from below)
- **Second term**: Top-down (prediction error at current level)
- **Balance**: Representations settle to minimize total error

### 7.3 Learning Dynamics

**Synaptic plasticity minimizes prediction error over time**:

```
dW_i/dt = -∂F/∂W_i = e_i × r_{i+1}^T
```

Where:
- **W_i**: Weights from level i+1 to level i (feedback/generative)
- **e_i**: Prediction error at level i
- **r_{i+1}**: Activity at level i+1

**This is Hebbian**: "Cells that fire together wire together"
- **Pre**: Activity at higher level
- **Post**: Prediction error at lower level
- **Result**: Weights learn to predict lower-level activity

---

## 8. ARR-COC-0-1: Hierarchical Relevance Allocation (10%)

### 8.1 Relevance Realization as Hierarchical Message Passing

**ARR-COC-0-1 implements predictive coding principles for visual relevance**:

The knowing-balancing-attending-realizing pipeline maps directly onto hierarchical message passing:

**Knowing (Bottom-Up Error Computation)**:
- `InformationScorer`: Computes entropy as **prediction error**
- `SaliencyScorer`: Computes visual surprise as **bottom-up error**
- `QueryCoupling`: Measures mismatch between image and query as **prediction error**

**Balancing (Precision Weighting)**:
- `OpponentProcessing`: Implements **gain control** on different error signals
- Exploit vs Explore = **Precision allocation** between pragmatic and epistemic
- Focus vs Diversify = **Precision modulation** between concentrated and distributed

**Attending (Error-Driven Resource Allocation)**:
- Token allocation based on **prediction error magnitude**
- High entropy patches = high error = more tokens (more precision)
- Low entropy patches = low error = fewer tokens (less precision)

**Realizing (Active Inference)**:
- Compression = **fulfilling predictions** (reducing error)
- Generation = **selecting actions** that minimize expected error

### 8.2 Variable LOD as Hierarchical Precision

**The texture array implements multi-level predictions**:

**Level 1: Raw pixels (lowest predictions)**
- RGB channels: Color predictions
- LAB channels: Perceptual uniformity predictions

**Level 2: Local features (mid-level predictions)**
- Sobel edges: Discontinuity predictions
- Gradients: Orientation predictions

**Level 3: Spatial structure (higher predictions)**
- Eccentricity: Foveal-peripheral predictions
- Coordinates: Global layout predictions

**Variable LOD = Precision-weighted encoding**:
- **400 tokens**: High precision (detailed predictions, high confidence)
- **200 tokens**: Medium precision
- **64 tokens**: Low precision (coarse predictions, low confidence)

### 8.3 Query-Aware Hierarchical Processing

**Top-down predictions modulated by query**:

The query acts like a **high-level prediction** that propagates down:

```
Query: "Is there a dog?"
    ↓ (top-down prediction)
High-level: Expect dog-like objects
    ↓ (modulates precision)
Mid-level: Expect fur textures, four legs
    ↓ (modulates precision)
Low-level: Expect specific edges, colors

Result: Dog-relevant patches get high precision (more tokens)
```

**This implements active inference**:
- Query = **pragmatic goal** (what I want to know)
- Token allocation = **epistemic action** (what I need to see)
- Compression = **perceptual inference** (what I conclude)

### 8.4 Predictive Coding Operations in ARR-COC

**Specific implementations**:

**Entropy as prediction error**:
```python
# Shannon entropy measures unpredictability
H(patch) = -Σ p(x) log p(x)

# High entropy = high prediction error = needs more tokens
# Low entropy = low prediction error = needs fewer tokens
```

**Opponent processing as precision weighting**:
```python
# Balance multiple error signals with precision weights
relevance = Π_entropy × e_entropy + Π_saliency × e_saliency + Π_query × e_query

# Where Π_i represents precision (confidence) in each error type
```

**Token allocation as hierarchical resource distribution**:
```python
# Allocate computational resources (tokens) based on precision-weighted errors
tokens_per_patch = f(precision_weighted_relevance)

# More relevance → more tokens → more detailed predictions
# Less relevance → fewer tokens → coarser predictions
```

**Result**: ARR-COC-0-1 implements **query-aware hierarchical predictive coding** for efficient visual processing, allocating precision (tokens) based on prediction error magnitude and task relevance.

---

## Connections to Existing Knowledge

**Free Energy Principle**:
- [friston/00-free-energy-principle-foundations.md](./00-free-energy-principle-foundations.md): Predictive coding minimizes free energy
- Prediction error = variational free energy gradient

**Vervaeke Relevance Realization**:
- Opponent processing = Precision weighting balance
- 4P knowing maps to hierarchical message types
- Transjective = Bidirectional prediction-error coupling

**Biological Vision**:
- V1→V2→V4→IT hierarchy implements predictive coding
- Receptive fields explained by prediction-error computation
- Attention as precision optimization

**Computational Implementation**:
- [cognitive-mastery/07-predictive-coding-algorithms.md](../cognitive-mastery/07-predictive-coding-algorithms.md): Algorithms and code
- Rao-Ballard model provides foundational architecture
- FORCE learning, reservoir computing extensions

---

## Sources

**Source Documents**:
- [cognitive-mastery/07-predictive-coding-algorithms.md](../cognitive-mastery/07-predictive-coding-algorithms.md) - Computational implementations
- [cognitive-foundations/01-predictive-processing-hierarchical.md](../cognitive-foundations/01-predictive-processing-hierarchical.md) - Hierarchical framework

**Key Papers** (accessed 2025-11-23):
- [Friston, 2009 - Predictive coding under the free-energy principle](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/) - Phil Trans R Soc B (cited 1956 times)
- [Bastos et al., 2012 - Canonical Microcircuits for Predictive Coding](https://www.sciencedirect.com/science/article/pii/S0896627312009592) - Neuron (cited 2820 times)
- [Kanai et al., 2015 - Cerebral hierarchies: predictive processing, precision and the pulvinar](https://pmc.ncbi.nlm.nih.gov/articles/PMC4387510/) - Phil Trans R Soc B (cited 442 times)
- [Rao & Ballard, 1999 - Predictive coding in the visual cortex](https://www.nature.com/articles/4580) - Nature Neuroscience (foundational)
- [Feldman & Friston, 2010 - Attention, uncertainty, and free-energy](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2959160/) - Frontiers in Human Neuroscience

**Recent Research** (accessed 2025-11-23):
- [Chao et al., 2018 - Large-Scale Cortical Networks for Hierarchical Prediction](https://www.sciencedirect.com/science/article/pii/S0896627318308924) - Neuron (cited 252 times)
- [Gelens et al., 2024 - Distributed representations of prediction error signals](https://www.nature.com/articles/s41467-024-48329-7) - Nature Communications (cited 25 times)
- [Jiang et al., 2021 - Predictive Coding Theories of Cortical Function](https://arxiv.org/pdf/2112.10048) - arXiv (cited 34 times)
- [den Ouden et al., 2012 - How Prediction Errors Shape Perception, Attention, and Motivation](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2012.00548/full) - Frontiers in Psychology (cited 595 times)

**Additional References**:
- [Sprevak - Predictive coding II: The computational level](https://marksprevak.com/publications/predictive-coding-ii-the-computational-level-b9f5/) - Precision weighting explanation
- Bastos et al., 2015 - Oscillatory signatures of feedforward/feedback
- Parras et al., 2017 - Hierarchical auditory prediction errors (cited 316 times)
