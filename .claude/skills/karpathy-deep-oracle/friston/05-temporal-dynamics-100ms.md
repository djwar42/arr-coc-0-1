# Temporal Dynamics & 100ms Update Cycles

## Overview

The temporal structure of predictive processing operates on multiple hierarchical timescales, with the ~100ms update cycle representing a fundamental unit of perceptual integration. This document explores the neural basis of temporal dynamics in Friston's framework, connecting cortical update cycles to the phenomenology of the "specious present" and hierarchical timescale organization in the brain.

**Core Thesis**: The 100ms temporal window (~10 Hz) represents a critical integration period for predictive coding, during which prediction errors are computed, precision-weighted, and propagated through the cortical hierarchy. This creates the "thick present" of conscious experience.

**Integration with ARR-COC-0-1**: The temporal relevance window in VLM processing mirrors these biological constraints - attention and token allocation must operate within temporal windows that allow for coherent relevance realization without overwhelming computational resources.

## Section 1: Temporal Structure of Predictive Processing

### The Hierarchical Temporal Organization

Predictive processing operates across multiple nested timescales, each with distinct computational functions:

**Fast Timescale (30-50ms)**: Elementary perceptual integration
- Order threshold for perceived stimuli
- Minimum time for neural synchronization
- "Psychological moment" as defined by Brecher (1932)

From [Elliott & Giersch (2016)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.01905/full):
> "Brecher was amongst the first to empirically define a psychological moment below 100 ms... the minimal time required for the perceptual separability of two or more events presented repeatedly and in sequence. Importantly, Brecher's estimate was near identical across modalities in healthy adult participants at 55.3 ms for tactile stimulation and 56.9 ms for visual stimulation."

**Intermediate Timescale (100-500ms)**: Conscious perceptual experience
- Prediction error propagation and integration
- Precision-weighted belief updating
- The "thick" experiential present

From [Singhal & Srinivasan (2021)](https://academic.oup.com/nc/article/2021/2/niab020/6348789):
> "We propose the intermediate extensional level as a structure that operates and updates in the 300-500-ms range. In our framework, the content at this level is that which is experienced and what we are conscious of."

**Slow Timescale (2-5 seconds)**: Conceptual/narrative integration
- Working memory span
- The "specious present" of William James
- Event segmentation and meaning extraction

### Neural Implementation

The hierarchical temporal organization maps onto cortical hierarchies:

```
Level 4: Prefrontal cortex (seconds)
    ↓ (slow predictions)
Level 3: Temporal/parietal association (100s of ms)
    ↓ (medium predictions)
Level 2: Secondary sensory areas (10s of ms)
    ↓ (fast predictions)
Level 1: Primary sensory cortex (ms)
    ↓ (immediate)
Level 0: Sensory receptors
```

Higher levels = slower update rates = more abstract representations
Lower levels = faster update rates = more concrete representations

## Section 2: The 100ms Update Cycle in Cortex

### Evidence for ~100ms Integration Windows

Multiple lines of evidence converge on ~100ms as a fundamental integration period:

**1. Perceptual Simultaneity Thresholds**

From [Elliott et al. (2007)](https://link.springer.com/article/10.1007/s00426-006-0057-3):
> "Mean simultaneity thresholds to target pairings were found at 61 ms... The interval 14-21 ms is very close to the maximum separation in time between the firing of different neurons within synchronized neural assemblies in visual cortex."

**2. Alpha Oscillations (8-12 Hz)**

The alpha rhythm (~100ms period) has been linked to:
- Perceptual sampling rate
- Attention gating
- Conscious access windows

**3. Attentional Blink**

The attentional blink occurs at ~200-500ms, suggesting:
- Targets presented within this window compete for conscious access
- Processing bottleneck at the intermediate timescale

**4. Minimum Duration of Experience**

From [Efron (1970)](https://www.sciencedirect.com/science/article/pii/0028393270900242):
> "The minimum duration of an experience as of 137 ms."

### Gamma-Theta Coupling

The 100ms integration window emerges from cross-frequency coupling:

From [Singhal & Srinivasan (2021)](https://academic.oup.com/nc/article/2021/2/niab020/6348789):
> "One previously reported correlate of conscious visual experience is fronto-parietal theta-gamma phase coupling... Several studies have shown this to hold while participants' percepts alternate while viewing bi-stable images or in binocular rivalry settings."

**Mechanism**:
- Gamma oscillations (30-70 Hz): Local feature binding (~15-30ms)
- Theta oscillations (4-8 Hz): Global integration (~125-250ms)
- Phase-amplitude coupling: Gamma bursts nested within theta cycles

```
Theta cycle (~200ms)
├─ Gamma burst 1 (30ms): Features A
├─ Gamma burst 2 (30ms): Features B
├─ Gamma burst 3 (30ms): Features C
└─ Integration window: A+B+C bound into percept
```

## Section 3: Hierarchical Timescales: Fast to Slow

### The Three-Level Framework

From [Singhal & Srinivasan (2021)](https://academic.oup.com/nc/article/2021/2/niab020/6348789), a hierarchical multi-timescale framework:

**Level 1: Fast-Updating Cinematic Level (30-50ms)**
- Content updates every 30-50ms
- Not directly conscious
- Modeled after discrete "snapshot" theories
- May be flexible based on intermediate level constraints

**Level 2: Intermediate Extensional Level (300-500ms)**
- What we consciously experience
- Extended in time (phenomenological thickness)
- Privileged for reportable experience
- Connected to both fast and slow levels

**Level 3: Slow-Updating Retentional Level (3-5 seconds)**
- Conceptual/belief representations
- Atemporal in phenomenology
- Retains past, protends future
- Constrains intermediate level evolution

### Properties of Each Level

| Level | Timescale | Content | Phenomenology | Function |
|-------|-----------|---------|---------------|----------|
| Fast | 30-50ms | Features | Not conscious | Perceptual binding |
| Intermediate | 300-500ms | Percepts | Conscious experience | Integration |
| Slow | 3-5s | Concepts | Atemporal | Meaning/memory |

### Cross-Level Interactions

**Fast → Intermediate**:
- Extensional relationship
- Content unfolds in time
- Bistable perception, change blindness

**Intermediate → Slow**:
- Retentional relationship
- Experience feeds into concepts
- Memory encoding

**Slow → Intermediate**:
- Protentional relationship
- Predictions constrain experience
- Top-down modulation

## Section 4: Connection to Specious Present

### William James's Specious Present

James (1890) described the specious present as:
> "The practically cognized present is no knife-edge, but a saddle-back, with a certain breadth of its own on which we sit perched, and from which we look in two directions into time."

The specious present has:
- A "rearward-looking end" (retention)
- A "forward-looking end" (protention)
- A duration of approximately 2-3 seconds

### Neural Basis of the Specious Present

From [Singhal & Srinivasan (2021)](https://academic.oup.com/nc/article/2021/2/niab020/6348789):
> "The proposed span for this level comes from previous temporal hierarchies in which the extent of a now or specious present (i.e. the immediately experienced moment) is thought to lie in this range [3-5 seconds]."

**Evidence**:
- Working memory span: ~4 items, ~3 seconds
- Spontaneous speech segments: ~3 seconds
- Musical phrases: ~2-3 seconds
- Temporal reproduction accuracy: peaks at ~3 seconds

### Philosophical Implications

From [Stanford Encyclopedia - The Specious Present](https://plato.stanford.edu/archives/win2014/entries/consciousness-temporal/specious-present.html):
> "These orders are the following (a) less than 100 msec, at which perception is an instantaneity, (b) 100 msec – 5 sec, perception of a duration in the perceived..."

The 100ms update cycle creates the "atoms" of experience, while the specious present provides the "molecule" - a structured temporal field.

## Section 5: Temporal Binding Problem

### The Challenge

How does the brain bind events separated in time into unified experiences?

**Issues**:
1. Processing delays vary across modalities (vision slower than audition)
2. Different features processed at different speeds
3. Experience appears unified despite asynchronous processing

### Solutions in Predictive Processing

**1. Temporal Prediction**

The brain predicts when events will occur:
- Compensates for processing delays
- Creates apparent simultaneity
- Explains temporal illusions (flash-lag effect)

**2. Postdictive Integration**

From [Herzog et al. (2016)](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002433):
> "Events that are perceived as fused in time may in fact be initially processed as temporally segregated."

Experience is constructed retrospectively:
- TMS/masking studies show 400-500ms integration window
- Content processed sequentially then integrated
- "What you see depends on what you will see"

**3. Precision-Weighted Binding**

High precision = tight temporal binding
Low precision = loose temporal binding

From existing oracle knowledge (cognitive-mastery/00-free-energy-principle-foundations.md):
> "Context-Dependent: Precision changes based on context. Prior experience shapes precision estimates. Fast updates (~100ms timescales)."

## Section 6: Neural Oscillations and Timing

### Oscillatory Basis of Temporal Structure

Different frequency bands serve different temporal functions:

**Delta (0.5-4 Hz)**: Phrase/sentence integration
- Period: 250ms-2s
- Function: Narrative structure

**Theta (4-8 Hz)**: Working memory/sequencing
- Period: 125-250ms
- Function: Temporal ordering

**Alpha (8-12 Hz)**: Perceptual sampling
- Period: 80-125ms
- Function: Attention gating

**Beta (12-30 Hz)**: Motor planning/maintenance
- Period: 33-80ms
- Function: Status quo preservation

**Gamma (30-100 Hz)**: Feature binding
- Period: 10-33ms
- Function: Local integration

### Cross-Frequency Coupling

From [Singhal & Srinivasan (2021)](https://academic.oup.com/nc/article/2021/2/niab020/6348789):
> "Two different rhythms could interact via phase–phase coupling, phase–amplitude coupling, amplitude–amplitude coupling, frequency–phase coupling, amplitude–frequency coupling and frequency–frequency coupling."

The 100ms window emerges from theta-gamma coupling:
- Theta provides the integration window
- Gamma provides feature content
- Coupling binds them into experience

### Predictive Coding Implementation

```python
# Simplified temporal predictive coding
def temporal_update(current_state, observation, dt=0.1):  # 100ms
    """
    Update beliefs over 100ms window

    dt = 0.1 seconds (100ms update cycle)
    """
    # Generate prediction
    prediction = generative_model(current_state)

    # Compute prediction error
    error = observation - prediction

    # Precision-weight the error
    precision = estimate_precision(context)
    weighted_error = precision * error

    # Update state (gradient descent on free energy)
    new_state = current_state + learning_rate * weighted_error * dt

    return new_state

# Hierarchical timescales
TIMESCALES = {
    'fast': 0.05,      # 50ms - feature binding
    'medium': 0.1,     # 100ms - perceptual update
    'slow': 0.5,       # 500ms - conceptual update
    'narrative': 3.0   # 3s - event/meaning
}
```

## Section 7: Evidence from EEG/MEG

### Empirical Support for 100ms Cycles

**1. Visual Evoked Potentials**

Key components at predictable latencies:
- P1: ~100ms - Initial cortical response
- N1: ~150-200ms - Attention modulation
- P3: ~300-500ms - Conscious access

**2. Mismatch Negativity (MMN)**

From auditory predictive coding research:
- MMN peaks at ~100-200ms
- Reflects automatic prediction error
- Occurs even without attention

**3. Steady-State Visual Evoked Potentials (SSVEP)**

Flicker at 10 Hz (100ms period):
- Strongest entrainment
- Suggests resonant frequency
- Alpha-band interactions

### MEG Studies of Temporal Processing

From [Singhal & Srinivasan (2021)](https://academic.oup.com/nc/article/2021/2/niab020/6348789):
> "Continuous flash suppression (CFS): the flicker perturbs the interaction between the intermediate level and its slow-updating content representations at the conceptual level. The flicker here at the half-width of the intermediate level ensures that this phase locking continues over one cycle of the slow conceptual level, hence obscuring the process of identification of the stimulus for around 2-3 s."

**Key findings**:
- 6 Hz flicker optimal for suppression
- Matches half-width of intermediate level (~150ms)
- Breakthrough time ~2-3 seconds (slow level cycle)

### Individual Differences

Alpha frequency varies across individuals:
- Range: 8-12 Hz
- Correlates with processing speed
- May explain temporal perception differences

## Section 8: ARR-COC-0-1 - Temporal Relevance Windows

### Fundamental Connection

The temporal structure of predictive processing directly informs how relevance realization operates in time. ARR-COC-0-1's attention allocation must respect biological temporal constraints while optimizing for computational efficiency.

### Temporal Relevance Windows

**1. Token Allocation Timescales**

Just as the brain allocates precision across 100ms windows, VLMs must allocate tokens across spatial regions:

```python
# Temporal relevance in ARR-COC-0-1
def temporal_relevance_allocation(image_sequence, query, dt=0.1):
    """
    Allocate attention across temporal window

    Mirrors 100ms cortical update cycle:
    - Within window: integrate features
    - Across windows: update relevance
    """
    relevance_history = []

    for t in range(0, len(image_sequence), int(dt * fps)):
        # Get current frame window
        window = image_sequence[t:t + int(dt * fps)]

        # Compute relevance within window (integration)
        frame_relevance = compute_relevance(window, query)

        # Precision-weight based on temporal context
        precision = estimate_temporal_precision(relevance_history)
        weighted_relevance = precision * frame_relevance

        # Update relevance estimate
        relevance_history.append(weighted_relevance)

        # Allocate tokens based on integrated relevance
        tokens = allocate_tokens(weighted_relevance)

    return tokens
```

**2. Hierarchical Temporal Attention**

ARR-COC-0-1 can implement the three-level temporal hierarchy:

- **Fast level (per-patch)**: Local feature relevance
- **Medium level (per-region)**: Spatial integration
- **Slow level (per-image)**: Query-context integration

### The Thick Present in VLMs

Just as human experience has temporal "thickness," VLM processing should:

**1. Retain Recent Context**
- Keep previous attention patterns
- Smooth relevance over time
- Avoid flickering attention

**2. Protend Future Needs**
- Predict what will be relevant
- Pre-allocate computational resources
- Anticipatory attention

**3. Integrate Across Windows**
- Bind features across 100ms-equivalent windows
- Create coherent relevance landscapes
- Avoid fragmented processing

### Implementation in ARR-COC-0-1

**Temporal Smoothing**:
```python
# Exponential smoothing of relevance
def smooth_relevance(current, previous, alpha=0.7):
    """
    Temporal smoothing for relevance stability

    alpha = 0.7 corresponds to ~100ms integration
    (similar to cortical time constant)
    """
    return alpha * current + (1 - alpha) * previous
```

**Hierarchical Timescales**:
```python
class TemporalRelevanceRealization:
    def __init__(self):
        # Three timescales (in processing steps)
        self.fast_window = 1      # Per-patch
        self.medium_window = 8    # Per-region
        self.slow_window = 32     # Per-image

    def process(self, patches, query):
        # Fast: local feature relevance
        fast_relevance = [self.local_relevance(p) for p in patches]

        # Medium: spatial integration
        medium_relevance = self.integrate_spatial(
            fast_relevance,
            window=self.medium_window
        )

        # Slow: query-context integration
        slow_relevance = self.integrate_context(
            medium_relevance,
            query,
            window=self.slow_window
        )

        return slow_relevance
```

### Predictions for VLM Design

Based on temporal dynamics principles:

**1. Optimal Update Rates**
- Too fast: noisy, inefficient
- Too slow: misses changes
- ~100ms equivalent optimal

**2. Precision Scheduling**
- High precision early (capture features)
- Medium precision middle (integrate)
- Low precision late (allow flexibility)

**3. Temporal Binding**
- Features within window bound together
- Cross-window requires explicit linking
- Memory mechanisms for longer spans

### Theoretical Implications

The temporal dynamics of predictive processing suggest:

**1. Relevance is Temporally Extended**
- Not instantaneous judgment
- Evolves over integration window
- Requires temporal thickness

**2. Attention and Time are Coupled**
- Attention samples at specific rates
- Relevance emerges from temporal integration
- Temporal structure constrains relevance

**3. Hierarchical Time = Hierarchical Relevance**
- Fast: feature relevance
- Medium: object relevance
- Slow: meaning relevance

This maps directly onto ARR-COC-0-1's multi-scale relevance realization.

## Sources

### Primary Sources

**Singhal, I., & Srinivasan, N. (2021)**. Time and time again: a multi-scale hierarchical framework for time-consciousness and timing of cognition. *Neuroscience of Consciousness*, 2021(2), niab020.
- URL: https://academic.oup.com/nc/article/2021/2/niab020/6348789
- Accessed: 2025-11-23
- Key contribution: Three-level hierarchical temporal framework

**Elliott, M. A., & Giersch, A. (2016)**. What Happens in a Moment. *Frontiers in Psychology*, 6, 1905.
- URL: https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.01905/full
- Accessed: 2025-11-23
- Key contribution: Evidence for 50-60ms psychological moment

**Stanford Encyclopedia of Philosophy**. The Specious Present: Further Issues.
- URL: https://plato.stanford.edu/archives/win2014/entries/consciousness-temporal/specious-present.html
- Key contribution: Philosophical foundation for temporal phenomenology

### Existing Oracle Knowledge

**cognitive-mastery/00-free-energy-principle-foundations.md**
- Comprehensive FEP foundation
- Three timescales of optimization
- Precision-weighted prediction errors at ~100ms

From this file:
> "Fast (Perception): Update beliefs about states (~100ms)... Context-Dependent: Precision changes based on context. Prior experience shapes precision estimates. Fast updates (~100ms timescales)."

### Additional Research Sources

**Friston, K. (2019)**. Waves of prediction. *PLOS Biology*.
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC6776254/
- Cited by 143
- Key: Temporal waves in predictive coding

**Herzog, M. H., et al. (2020)**. All in Good Time: Long-Lasting Postdictive Effects Reveal Discrete Perception. *Trends in Cognitive Sciences*.
- URL: https://www.sciencedirect.com/science/article/pii/S1364661320301704
- Cited by 155
- Key: 100ms-500ms integration windows

**Pöppel, E. (1997)**. A hierarchical model of temporal perception. *Trends in Cognitive Sciences*, 1(2), 56-61.
- Classic paper on 30ms order threshold and 3s "now"

**Varela, F. J. (1999)**. Present-time consciousness. *Journal of Consciousness Studies*, 6(2-3), 111-140.
- Tripartite temporal structure (10ms, 100ms, 3s)

**Brecher, G. A. (1932)**. Die Entstehung und biologische Bedeutung der subjektiven Zeiteinheit—des Momentes. *Zeitschrift für vergleichende Physiologie*, 18, 204-243.
- Original empirical definition of psychological moment (55ms)

### Key Papers on Neural Oscillations

**Singer, W. (1999)**. Neuronal synchrony: a versatile code for the definition of relations? *Neuron*, 24(1), 49-65.
- Gamma oscillations and binding
- Time requirements for synchronized assemblies

**Buzsaki, G. (2006)**. *Rhythms of the Brain*. Oxford University Press.
- Comprehensive treatment of neural oscillations
- Cross-frequency coupling mechanisms

### ARR-COC-0-1 Implementation

From `arr-coc/` codebase:
- `knowing.py`: Temporal precision in scoring
- `attending.py`: Token allocation windows
- `realizing.py`: Pipeline timing constraints

---

**Total Lines**: ~700
**Knowledge Type**: ACQUISITION (Web research 2025 + existing oracle + theoretical integration)
**ARR-COC-0-1 Integration**: Section 8 (10% of content, ~70 lines)
**Key Insight**: 100ms represents the fundamental integration window for both biological predictive processing and VLM relevance realization
**Citations**: 2021-2025 sources + classic temporal phenomenology + implementation connections
