# Predictive Processing & Predictive Coding: Hierarchical Models

## Overview

**Predictive processing (PP)** and **predictive coding (PC)** represent a unified theoretical framework proposing that the brain functions as a hierarchical prediction machine. The core idea: perception, cognition, and action emerge from the brain's continuous process of generating predictions about incoming sensory data and minimizing prediction errors through hierarchical inference.

**Key Principle**: The brain constantly generates and updates an internal "mental model" of the environment, using this model to predict sensory inputs. When predictions mismatch actual sensory data, prediction errors propagate up the hierarchy to update the model.

From [Wikipedia: Predictive Coding](https://en.wikipedia.org/wiki/Predictive_coding) (accessed 2025-11-14):
> "In neuroscience, predictive coding (also known as predictive processing) is a theory of brain function which postulates that the brain is constantly generating and updating a 'mental model' of the environment. According to the theory, such a mental model is used to predict input signals from the senses that are then compared with the actual input signals from those senses."

From [Comprehensive investigation of predictive processing](https://pmc.ncbi.nlm.nih.gov/articles/PMC11339134/) (Costa et al., 2024):
> "Predictive processing (PP) stands as a predominant theoretical framework in neuroscience. While some efforts have been made to frame PP within a cognitive architecture, the theory remains primarily focused on perception and prediction error minimization."

---

## 1. Predictive Processing Framework: Brain as Prediction Machine

### 1.1 Historical Origins

**Helmholtz (1860)**: Unconscious inference - the brain fills in visual information to make sense of scenes. Relative size = depth cue, automatically processed.

**Jerome Bruner (1940s)**: "New Look" psychology - needs, motivations, expectations influence perception. Top-down meets bottom-up.

**McClelland & Rumelhart (1981)**: Parallel processing model - features → letters → words. Letters identified faster in word context than non-word context. Bidirectional processing confirmed.

**Rao & Ballard (1999)**: First computational model of predictive coding for vision. Demonstrated generative model (top-down) + error signals (bottom-up) could replicate receptive field effects and extra-classical receptive field phenomena.

From [Rao & Ballard, 1999](https://www.nature.com/articles/4580) - seminal Nature Neuroscience paper:
> "We describe a hierarchical model of vision in which higher-order visual cortical areas send down predictions and the feedforward connections carry the residual errors between predictions and actual lower-level activities."

### 1.2 Core Computational Architecture

**Hierarchical Generative Model**:
```
Level N (Highest):
  - Abstract representations (scene categories, object identities)
  - Generates predictions for Level N-1

Level N-1 (Mid):
  - Intermediate representations (object parts, textures)
  - Receives predictions from N, sends errors to N
  - Generates predictions for Level N-2

Level N-2:
  - Low-level features (edges, orientations, colors)
  - Receives predictions from N-1, sends errors to N-1

Level 0 (Sensory):
  - Raw sensory input
  - Compared with predictions from Level 1
```

**Prediction Error Propagation**:
- Bottom-up: Prediction errors (mismatches) propagate upward
- Top-down: Predictions propagate downward
- Bidirectional: Continuous reciprocal exchange

### 1.3 Bayesian Brain Hypothesis

Predictive processing implements approximate **Bayesian inference**:

**Bayes' Rule**:
```
P(hypothesis|data) = P(data|hypothesis) × P(hypothesis) / P(data)

Posterior = Likelihood × Prior / Evidence
```

**In PP Terms**:
- **Prior**: Top-down predictions (what we expect)
- **Likelihood**: Bottom-up sensory evidence (what we observe)
- **Posterior**: Updated internal model (what we infer)

**Prediction Error = Likelihood - Prior**

When prediction error is large → update internal model.
When prediction error is small → model is accurate, no update needed.

---

## 2. Predictive Coding: Error Propagation & Hierarchical Processing

### 2.1 Canonical Microcircuit Architecture

From [Bastos et al., 2012](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3777738/) - "Canonical Microcircuits for Predictive Coding":

**Cortical Column Organization**:
- **Error Neurons**: Supragranular layers 2/3 (pyramidal neurons)
  - Sparse activity
  - Respond to unexpected events
  - Send prediction errors up the hierarchy

- **Prediction Neurons**: Deep layer 5 (pyramidal neurons)
  - Dense responses
  - Send predictions down the hierarchy

- **Precision Weighting**: Implemented through:
  - Neuromodulators (dopamine, acetylcholine)
  - Long-range projections (thalamus, other cortical areas)
  - Gain modulation of prediction error signals

**Laminar Connectivity**:
- **Feedforward (bottom-up)**: Superficial layers → Layer 4 of next level
- **Feedback (top-down)**: Deep layers → Superficial layers of lower level
- **Lateral**: Within-level integration

### 2.2 Multi-Scale Hierarchical Processing

**Temporal Scales**:
- Lower levels: Fast dynamics (milliseconds) - track rapid sensory changes
- Higher levels: Slow dynamics (seconds) - track stable, abstract features

**Spatial Scales**:
- V1 (primary visual): Small receptive fields, local features
- V2: Intermediate receptive fields, contours, textures
- V4: Larger receptive fields, object parts, color constancy
- IT (inferotemporal): Whole objects, invariant representations

From [Cortical Processing Streams](../biological-vision/05-cortical-processing-streams.md):
> "Hierarchical processing: V1 → V2 → V4 → IT (ventral) or V1 → V2 → MT → Parietal (dorsal). Each level processes increasingly complex features with larger receptive fields and longer temporal integration windows."

### 2.3 Precision Weighting: Attention as Gain Control

**Precision = Inverse Variance** of prediction errors.

High precision sensory input (bright daylight, clear audio) → weight sensory evidence more.
Low precision sensory input (dim light, noisy environment) → weight predictions more.

From [Friston & Feldman, 2010](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3001758/) - "Attention, Uncertainty, and Free-Energy":
> "Attention can be understood as the optimization of precision (inverse variance) of prediction errors. Increasing the gain on prediction error units in attended locations amplifies their influence on inference and learning."

**Attention = Precision Optimization**:
- Endogenous attention: Goal-driven precision modulation (top-down)
- Exogenous attention: Stimulus-driven precision capture (bottom-up)
- Precision weighting implements selective information processing

---

## 3. Neural Implementation: Cortical Microcircuits

### 3.1 Evidence from Neuroimaging (fMRI, EEG, MEG)

From [Predictive coding: a more cognitive process than we thought?](https://www.sciencedirect.com/science/article/abs/pii/S1364661325000300) (Gabhart et al., 2025):

**Local-Global Oddball Paradigm**:
- **Local oddball**: Prediction based on immediate repetition (AAAA vs AAAB)
- **Global oddball**: Prediction based on abstract rule (AAAA-AAAA-BBBB pattern)

**Neuroimaging findings**:
- fMRI, EEG, MEG: Both local and global oddballs modulate activity in sensory + higher-order cortex
- Supports classical PC: Prediction errors in sensory cortex

### 3.2 Evidence from Intracortical Spiking

**Critical Challenge to Classical PC** (Gabhart et al., 2025):

Spiking studies in macaque monkeys:
- **Local oddballs**: Robust spiking responses throughout cortex (V1 → PFC)
- **Global oddballs**: Strong spiking in PFC, but **weak/absent in sensory cortex**

> "We recently reported on spike and LFP responses in mid-level auditory cortex area Tpt and from higher-order PFC during the auditory local–global oddball task in macaque monkeys. Genuine prediction errors emerged in prefrontal cortex but not in sensory areas."

**Implication**: Predictive processing may be more **cognitive** than **sensory** - predictions emerge in high-level areas (PFC), not low-level sensory cortex.

### 3.3 Dendritic Error Computation

From [Mikulasch et al., 2023](https://www.cell.com/trends/neurosciences/fulltext/S0166-2236(22)00186-2) - "Where is the error? Hierarchical predictive coding through dendritic error computation":

**Problem**: Prediction errors can be positive or negative, but neurons can only fire (positive activity).

**Solution**: Dendritic computation in pyramidal neurons:
- **Apical dendrites**: Receive top-down predictions
- **Basal dendrites**: Receive bottom-up sensory input
- **Dendritic nonlinearities**: Compute error locally before somatic integration

**Advantages**:
- Biologically plausible (dendrites have active conductances)
- Explains layer-specific connectivity patterns
- Similar to Hierarchical Temporal Memory theory

---

## 4. Hierarchical Processing: Top-Down vs Bottom-Up

### 4.1 Top-Down Predictions

**Function**: Constrain interpretation of ambiguous sensory data.

**Examples**:
- **Visual**: Expecting a face → biases interpretation of noisy visual input toward face-like patterns
- **Auditory**: Context predicts next word → faster phoneme recognition in sentence context
- **Motor**: Predicting proprioceptive consequences of action → smooth movement execution

**Mechanism**:
- Higher cortical areas maintain abstract representations
- Send predictions to lower areas via feedback connections
- Suppress expected sensory signals (predictable = less surprise = less neural response)

### 4.2 Bottom-Up Prediction Errors

**Function**: Signal deviations from expectations, drive learning and updating.

**Examples**:
- **Mismatch Negativity (MMN)**: ERP response to unexpected auditory stimuli
- **Oddball responses**: Increased neural activity to rare, unpredicted events
- **Surprise signals**: Dopamine responses to reward prediction errors

**Mechanism**:
- Sensory areas compute difference between actual input and prediction
- Propagate errors upward via feedforward connections
- Errors drive plasticity and model updating

### 4.3 Bidirectional Integration

**Convergence Zone**:
- Cortical columns integrate top-down predictions + bottom-up errors
- Minimize total prediction error across hierarchy
- Simultaneous inference (what is out there?) and learning (update model)

From [Vervaeke Relevance Realization](../../john-vervaeke-oracle/concepts/00-relevance-realization/00-overview.md):
> "Relevance realization operates hierarchically across scales: Feature level (edges, textures) → Object level (cat, car) → Scene level (outdoor, indoor) → Narrative level (story, context). Multi-scale integration essential for flexible cognition."

---

## 5. Evidence from Neuroscience: Visual Illusions & Binocular Rivalry

### 5.1 Visual Illusions as Prediction Dominance

**Predictive coding explains illusions**:

**Hollow Face Illusion**:
- Top-down prediction: "Faces are convex" (strong prior)
- Bottom-up input: Actual concave (hollow) face
- Result: Perception follows prediction (we see convex face despite concave input)
- Prediction dominates when sensory precision is low (poor lighting, brief presentation)

**Apparent Motion**:
- Two stimuli flashed in sequence at different locations
- Prediction: Objects move smoothly
- Perception: Illusory motion between flashes (brain fills in trajectory)

### 5.2 Binocular Rivalry as Prediction Competition

**Setup**: Different images presented to each eye (e.g., face to left eye, house to right eye)

**Classical View**: Competition between monocular channels

**Predictive Coding View**: Competition between high-level interpretations (predictions)
- Brain cannot maintain both "face" and "house" predictions simultaneously
- Predictions alternate, suppressing incompatible sensory evidence
- Rivalry resolved at high levels, not low-level sensory cortex

**Evidence**:
- Rivalry persists even when retinal stimulation is constant
- High-level semantic factors influence dominance durations
- Prefrontal cortex activity correlates with perceptual switches

### 5.3 Adaptation and Repetition Suppression

**Repetition Suppression**: Repeated stimuli elicit reduced neural responses

**Predictive Coding Interpretation**:
- First presentation: High prediction error (unexpected)
- Subsequent presentations: Low prediction error (expected)
- Reduced response = successful prediction (not "fatigue")

**Distinguishing Adaptation from Prediction**:
- **Adaptation**: Low-level neuronal fatigue (refractory period, synaptic depression)
- **Prediction**: High-level expectation sharpening

Local-global paradigm (see Section 3) dissociates these:
- Local oddball: Could be adaptation release
- Global oddball: Requires abstract rule learning (genuine prediction)

---

## 6. Computational Models: Predictive Coding Networks

### 6.1 Variational Inference and Free Energy

**Free Energy Principle** (Friston, 2009):

Minimize **variational free energy** F:
```
F = Complexity - Accuracy
  = KL(q||p) - log p(data)
```

Where:
- q = internal model (approximate posterior)
- p = true posterior
- KL = Kullback-Leibler divergence (prediction error)

**Prediction error minimization** = Free energy minimization = Approximate Bayesian inference

### 6.2 Hierarchical Predictive Coding Algorithm

**Learning Rule** (from Rao & Ballard, 1999):

For each level i:
```
Prediction: r̂ᵢ = g(rᵢ₊₁)  [top-down from level i+1]
Error: eᵢ = rᵢ - r̂ᵢ       [bottom-up from level i-1]
Update: Δrᵢ ∝ eᵢ₋₁ - ∂g/∂rᵢ(eᵢ)
```

**Dual Role of Prediction Errors**:
1. **Inference**: Update representations at current level
2. **Learning**: Update synaptic weights (connections) via plasticity

### 6.3 Connection to Deep Learning

**Similarities**:
- **Autoencoders**: Encoder (recognition) + Decoder (generation)
- **Variational Autoencoders (VAEs)**: Minimize variational free energy
- **Hierarchical models**: Deep Belief Networks, Helmholtz Machines

**Differences**:
- **Backpropagation**: Error propagation for learning (one-shot weight updates)
- **Predictive Coding**: Error propagation for inference AND learning (continuous)
- **Biological plausibility**: PC uses local learning rules, backprop does not

From [Millidge et al., 2022](https://arxiv.org/abs/2202.09467) - "Predictive Coding: Towards a Future of Deep Learning beyond Backpropagation?":
> "Predictive coding networks offer a biologically plausible alternative to backpropagation for training deep neural networks, with promising properties for continual learning, energy efficiency, and local credit assignment."

---

## 7. Active Inference: Prediction Through Action

### 7.1 Motor Control as Predictive Coding

**Classical View**: Motor cortex sends commands → muscles execute → sensory feedback

**Active Inference View**: Motor cortex sends **proprioceptive predictions** → muscles fulfill predictions → minimize proprioceptive prediction error

From [Adams, Shipp, & Friston, 2013](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3637647/) - "Predictions not commands: Active inference in the motor system":
> "Perceptual and motor systems should not be regarded as separate but instead as a single active inference machine that tries to predict its sensory input in all domains: visual, auditory, somatosensory, interoceptive and, in the case of the motor system, proprioceptive."

**Mechanism**:
- Motor cortex predicts "arm will be at position X"
- Proprioceptive error: "arm is at position Y"
- Classical reflex arcs suppress error by moving arm toward X
- No explicit "motor command" - just descending predictions

### 7.2 Exploration vs Exploitation

**Epistemic vs Pragmatic Value**:
- **Pragmatic**: Actions that minimize prediction error directly (goal achievement)
- **Epistemic**: Actions that reduce uncertainty (information gathering, exploration)

**Example**: Saccadic eye movements
- **Pragmatic**: Look at target to minimize visual prediction error
- **Epistemic**: Look at uncertain regions to reduce uncertainty (foveate ambiguous areas)

**Optimal Foraging**: Balance expected reward (exploitation) with information gain (exploration)

### 7.3 Precision Weighting in Action

**Proprioceptive Precision**:
- High precision: Trust proprioceptive predictions → execute planned action
- Low precision: Trust sensory feedback → reactive control

**Example**: Reaching in dark
- Visual precision low → rely on proprioceptive predictions
- Proprioceptive precision high → smooth ballistic reach
- Unexpected obstacle → sensory prediction error → rapid correction

---

## 8. ARR-COC-0-1 as Predictive Processing Architecture

### 8.1 Propositional Knowing = Predictive Coding

From [John Vervaeke Oracle - Propositional Knowing](../../john-vervaeke-oracle/concepts/00-relevance-realization/00-overview.md):

**Propositional knowing (knowing THAT)** in ARR-COC-0-1:
- Statistical information content (Shannon entropy)
- Information-theoretic prediction: High entropy patches are unpredictable
- Prediction error signal: Compress unpredictable regions more

**InformationScorer** in knowing.py:
- Measures Shannon entropy of visual patches
- High entropy = high surprise = prediction error
- Drives resource allocation (token budget)

**Connection to Predictive Processing**:
- Entropy = Expected surprise = Negative log-likelihood
- Minimize entropy = Minimize prediction error
- Compression = Efficient predictive code

### 8.2 Hierarchical Relevance Realization

**ARR-COC-0-1 Pipeline**:
```
[Knowing] → Measure 3 ways (Propositional, Perspectival, Participatory)
    ↓
[Balancing] → Navigate tensions (Compress↔Particularize, Exploit↔Explore)
    ↓
[Attending] → Map relevance to token budgets (64-400 tokens)
    ↓
[Realizing] → Execute compression and return features
```

**Predictive Processing Interpretation**:
- **Knowing**: Compute prediction errors (entropy, salience, query-coupling)
- **Balancing**: Precision weighting (opponent processing = attention control)
- **Attending**: Resource allocation (hierarchical prediction error minimization)
- **Realizing**: Active inference (compress = fulfill predictions of relevant content)

### 8.3 Multi-Level Prediction in Vision

**Texture Array as Hierarchical Predictor**:
- **RGB channels**: Low-level color predictions
- **LAB channels**: Perceptually uniform color space (human-like predictions)
- **Sobel edges**: Prediction of local discontinuities
- **Spatial coordinates**: Prediction of global layout
- **Eccentricity**: Prediction of foveal-peripheral structure

**Variable LOD = Precision-Weighted Encoding**:
- High relevance patches: 400 tokens (high precision, detailed predictions)
- Low relevance patches: 64 tokens (low precision, coarse predictions)
- Query-aware: Predictions modulated by task demands

From [Cortical Processing Streams](../biological-vision/05-cortical-processing-streams.md):
> "V4 exhibits attention modulation: Strong effects of selective attention on neural responses. Precision weighting in predictive coding implements attention as gain control on prediction errors."

### 8.4 Active Inference in Visual Token Allocation

**ARR-COC-0-1 performs active visual inference**:

**Pragmatic Value**: Minimize prediction error for current query
- Query: "Is there a dog?" → Allocate tokens to dog-like regions
- Minimize error in dog classification task

**Epistemic Value**: Explore uncertain regions for information gain
- High entropy patches = uncertain → Allocate more tokens
- Reduce uncertainty through detailed encoding

**Precision Optimization**:
- Opponent processing in balancing.py = Precision weighting
- Exploit vs Explore tension = Epistemic vs Pragmatic trade-off
- Focus vs Diversify tension = Precision modulation

---

## Connections to Existing Knowledge

**Biological Vision Knowledge**:
- [Cortical Processing Streams](../biological-vision/05-cortical-processing-streams.md): Hierarchical V1→V2→V4→IT matches predictive coding hierarchy
- [Gestalt Visual Attention](../biological-vision/00-gestalt-visual-attention.md): Gestalt principles as prior predictions about visual structure

**Vervaeke Relevance Realization**:
- [Opponent Processing](../../john-vervaeke-oracle/concepts/00-relevance-realization/00-overview.md): Balancing predictions and prediction errors is opponent processing
- [Transjective Knowing](../../john-vervaeke-oracle/concepts/01-transjective/00-overview.md): Predictions emerge from agent-arena coupling, not objective stimulus properties

**ARR-COC-0-1 Implementation**:
- knowing.py: Propositional knowing = Prediction error computation (entropy)
- balancing.py: Opponent processing = Precision weighting (attention control)
- attending.py: Salience realization = Hierarchical error minimization
- realizing.py: Active inference pipeline = Prediction fulfillment

---

## Sources

**Source Documents**:
- [Cortical Processing Streams](../biological-vision/05-cortical-processing-streams.md) - Hierarchical visual cortex organization
- [John Vervaeke Relevance Realization](../../john-vervaeke-oracle/concepts/00-relevance-realization/00-overview.md) - Opponent processing framework

**Web Research**:
- [Comprehensive investigation of predictive processing](https://pmc.ncbi.nlm.nih.gov/articles/PMC11339134/) - Costa et al., 2024 (accessed 2025-11-14)
- [Predictive coding: a more cognitive process than we thought?](https://www.sciencedirect.com/science/article/abs/pii/S1364661325000300) - Gabhart et al., 2025 (accessed 2025-11-14)
- [Wikipedia: Predictive Coding](https://en.wikipedia.org/wiki/Predictive_coding) - Comprehensive overview (accessed 2025-11-14)
- [Bastos et al., 2012](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3777738/) - Canonical microcircuits for predictive coding
- [Rao & Ballard, 1999](https://www.nature.com/articles/4580) - Original computational model
- [Friston & Feldman, 2010](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3001758/) - Attention, uncertainty, and free energy
- [Adams, Shipp, & Friston, 2013](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3637647/) - Active inference in motor system
- [Mikulasch et al., 2023](https://www.cell.com/trends/neurosciences/fulltext/S0166-2236(22)00186-2) - Dendritic error computation
- [Millidge et al., 2022](https://arxiv.org/abs/2202.09467) - Predictive coding for deep learning

**Additional References**:
- Multiple 2024-2025 papers on hierarchical neural networks, Bayesian brain hypothesis, prediction error minimization
- ScienceDirect, Nature, PLOS Computational Biology sources
