# The Thick Present: Neuroscience of Temporal Integration

## Overview

The "thick present" refers to the phenomenological experience that consciousness is not confined to a knife-edge instant but extends over a duration of hundreds of milliseconds to several seconds. This document presents the neuroscientific evidence for temporal integration windows that create the thick present, focusing on the ~100ms update cycles, working memory constraints, and hierarchical temporal structures that underlie conscious experience.

**Core Thesis**: The brain constructs the experienced present through hierarchical temporal integration mechanisms operating on multiple timescales - from the ~30ms functional moment to the ~3s specious present to the ~10-60s window of mental presence. These nested timescales create the "thickness" of temporal experience.

**Integration with ARR-COC-0-1**: Temporal relevance windows in VLM processing should mirror these biological constraints, with token allocation and attention operating within integration windows that create coherent relevance landscapes without exceeding computational resources.

---

## Section 1: Neural Evidence for the Thick Present

### Multiple Convergent Lines of Evidence

The thick present is not merely a philosophical construct but has robust empirical support:

**1. Psychophysical Thresholds**

From Wittmann (2011) "Moments in Time" ([Frontiers in Integrative Neuroscience](https://www.frontiersin.org/journals/integrative-neuroscience/articles/10.3389/fnint.2011.00066/full)):

> "The temporal order threshold, which defines the inter-stimulus interval between two events at which an observer can reliably indicate the temporal order is more comparable across senses and lies roughly at 20-60 ms, to some extent depending on physical stimulus properties."

This defines the lower boundary of temporal experience - below this threshold, events cannot be ordered and are experienced as co-temporal.

**2. Perceptual Grouping Limits**

From Wittmann (2011):

> "Empirical evidence suggests that these mental units comprising several individual beats have a lower limit of around 250 ms and an upper limit of approximately 2 s."

This defines the "experienced moment" - the window within which events are automatically grouped into perceptual gestalts.

**3. Working Memory Span**

The upper limit of the thick present is constrained by working memory:

From Wittmann (2011):

> "Experimental investigations of short-term memory show how the number of correct recalls of presented syllables decreases with increasing interval length between stimulus presentation and recall - in the range of multiple seconds - if the rehearsal of syllables is prevented."

### The Three-Level Hierarchy

Neuroscience reveals a hierarchical organization of temporal integration:

| Level | Name | Duration | Function | Neural Correlate |
|-------|------|----------|----------|------------------|
| 1 | Functional Moment | 30-60 ms | Simultaneity, succession | Gamma oscillations (30-40 Hz) |
| 2 | Experienced Moment | 2-3 s | Perceptual grouping, nowness | Theta oscillations (4-8 Hz) |
| 3 | Mental Presence | 10-60 s | Working memory span | Delta/infra-slow oscillations |

From Wittmann (2011):

> "Based on the empirical findings of discrete processing in perception and action it has therefore been suggested that the brain creates a-temporal system states during which incoming information is treated as co-temporal, and which are on the one hand responsible for binding intra- and inter-modal information and on the other hand create the experience of temporal order."

---

## Section 2: The 100ms Update Cycle

### Evidence from Electrophysiology

The ~100ms timescale (10 Hz / alpha band) appears repeatedly in neuroscience:

**1. Alpha Oscillations (8-12 Hz)**

From Wittmann (2011):

> "These temporal units have been related to rhythmic brain activity of thalamo-cortical loops, the 'gamma' band with a frequency of around 40 Hz. However, periodicities in the alpha band (around 10 Hz) as well as the theta band (4-8 Hz), and potentially related to functional integration on a time scale of 100 ms and above, may additionally contribute to temporal integration phenomena."

**2. Visual Evoked Potentials**

Key ERP components mark the 100ms integration:
- **P1** (~100 ms): Initial cortical response
- **N1** (~150-200 ms): Attention modulation
- **P3** (~300-500 ms): Conscious access

**3. Mismatch Negativity (MMN)**

From Yabe et al. (1998), cited in Wittmann (2011):

> "Using the paradigm of mismatch negativity of magnetic brain responses a window of integration of 160-170 ms was estimated to bind successive auditory input into an auditory percept."

### The 100ms Window in Different Modalities

**Auditory-Visual Integration**

From van Wassenhove et al. (2007), cited in Wittmann (2011):

> "For example, a time frame of about 200 ms determines the integration of auditory-visual input in speech processing when probing for the McGurk effect - an illusory fusion percept created when lip movements are incongruent to heard syllables."

**Motor Control**

From Wittmann et al. (2001), cited in Wittmann (2011):

> "Temporal integration in a time frame of around 250 ms was reported in sensory-motor processing distinguishing maximum tapping speed from a personal, controlled motor speed."

**Sequence Perception**

From Warren & Obusek (1972), cited in Wittmann (2011):

> "Stimulus durations of 200-300 ms (and minimum inter-onset intervals) are a necessary prerequisite for the establishment of temporal order representation for the detection of the correct sequence of four acoustic or visual events."

### Neural Mechanisms

**Proposed Architecture**

From Craig (2009), cited in Wittmann (2011):

> "It has been proposed that anterior insular cortex function may provide the continuity of subjective awareness by temporally integrating a series of elementary building blocks - successive moments of self-realization informed by the interoceptive system. The continuous processing from moment to moment would advance with a frame rate of about 8 Hz, these temporal building blocks of perception lying in the range of 125 ms."

**Neural Microstates**

From Lehmann et al. (1998), cited in Wittmann (2011):

> "Neural microstates with average duration of 125 ms as derived from electrophysiological recordings have been discussed as potential 'atoms of thought,' constituting critical time windows within which neural events are functionally integrated."

---

## Section 3: Working Memory and the Temporal Window

### The Upper Limit of Present Experience

Working memory defines the outer boundary of the thick present:

From Wittmann (2011):

> "'Working memory provides a temporal bridge between events - both those that are internally generated and environmentally presented - thereby conferring a sense of unity and continuity to conscious experience' (Goldman-Rakic, 1997)."

**Evidence from Amnesia**

Patients with anterograde amnesia reveal the working memory constraint:

From Wittmann (2011):

> "Reports from neurological case studies with individuals who suffer from anterograde amnesia after bilateral damage to the hippocampus indicate that these patients live within a moving temporal window of presence that does not reach beyond their short-term or working memory span, incapable of storing incidents into episodic long-term memory."

These patients can function normally within the temporal window of working memory but cannot form new memories beyond it - they live perpetually in the present.

### Duration Reproduction Studies

**Progressive Under-Reproduction**

From Wackermann (2005, 2007), cited in Wittmann (2011):

> "The mean of reproduced intervals is accurate for shorter intervals of up to 3 s but with increasing interval lengths are progressively under-reproduced relative to physical time. The negative curvature of the duration reproduction function results in an asymptotic upper limit of duration accessible to experience, i.e., a temporal horizon of experienced time in the range of roughly 10^2 s."

This ~100 second horizon represents the outer boundary of the thick present.

**Memory Decay as Internal Clock**

From Staddon (2005), cited in Wittmann (2011):

> "Since memory strength decreases with time, memory trace decay could actually function as a 'clock'."

The dual klepsydra model (Wackermann & Ehm, 2006) formalizes this:
- Accumulator receives inflow for duration representation
- Simultaneous outflow reflects memory loss
- Results in "subjective shortening" of stored duration

### The Three Timescales of Presence

**1. Functional Moment (30-60 ms)**
- Co-temporality within window
- No perceivable duration
- Temporal order threshold

**2. Experienced Moment (~3 s)**
- Extended nowness
- Perceptual gestalts
- Phenomenological thickness

**3. Mental Presence (~60 s)**
- Working memory span
- Narrative self
- Sliding window of experience

From Wittmann (2011):

> "Mental presence encloses a sequence of such moments for the representation of a unified experience of presence."

---

## Section 4: EEG/MEG Evidence

### Event-Related Potentials (ERPs)

**P300 and Conscious Access**

The P300 component (300-500 ms post-stimulus) marks conscious access:
- Larger for attended stimuli
- Absent for unattended stimuli
- Correlates with reportability

**Mismatch Negativity (MMN)**

From Wittmann (2011):

> "Using the paradigm of mismatch negativity of magnetic brain responses a window of integration of 160-170 ms was estimated to bind successive auditory input into an auditory percept."

The MMN reflects automatic prediction error:
- Peaks at ~100-200 ms
- Pre-attentive processing
- Demonstrates temporal integration

### Oscillatory Dynamics

**Cross-Frequency Coupling**

From Singhal & Srinivasan (2021), cited in [existing oracle temporal-dynamics file](../friston/05-temporal-dynamics-100ms.md):

> "One previously reported correlate of conscious visual experience is fronto-parietal theta-gamma phase coupling... Several studies have shown this to hold while participants' percepts alternate while viewing bi-stable images or in binocular rivalry settings."

**Alpha Gating**

Alpha oscillations (8-12 Hz) gate perceptual sampling:
- High alpha = inhibition
- Low alpha = processing
- Creates discrete sampling windows

### MEG Studies of Temporal Integration

**Temporal Context Shapes Perception**

From Damsma et al. (2021) ([PMC8152605](https://pmc.ncbi.nlm.nih.gov/articles/PMC8152605/)):

> "Our subjective perception of time is optimized to temporal regularities in the environment. This is illustrated by the central tendency effect: When repeatedly estimating intervals, short intervals are overestimated and long intervals are underestimated."

This demonstrates predictive processing in temporal perception - the brain uses temporal context to generate predictions.

**Bistable Perception**

MEG studies of bistable figures reveal the ~3 s experienced moment:

From Wittmann (2011):

> "During continuous presentation, one aspect lasts on average for around 3 s before a switch in perspectives occurs, with some inter-individual variability and variance attributable to stimulus characteristics of the particular ambiguous figure."

### Individual Differences

**Alpha Frequency Variation**

Individual alpha peak frequency (IAF) varies 8-12 Hz:
- Higher IAF = faster temporal processing
- Lower IAF = longer integration windows
- Predicts temporal perception thresholds

From Ronconi et al. (2018) ([Nature Scientific Reports](https://www.nature.com/articles/s41598-018-29671-5)):

> "These findings provide evidence for a direct link between changes in the alpha band and the temporal resolution of perception."

---

## Section 5: Connection to Friston's Temporal Dynamics

### Predictive Processing and Temporal Integration

The free energy principle explains temporal integration:

**1. Hierarchical Timescales**

From existing oracle knowledge ([friston/05-temporal-dynamics-100ms.md](../friston/05-temporal-dynamics-100ms.md)):

> "Higher levels = slower update rates = more abstract representations
> Lower levels = faster update rates = more concrete representations"

**2. Precision-Weighted Integration**

Precision modulates temporal binding:
- High precision = tight temporal binding
- Low precision = loose temporal binding
- Creates flexible integration windows

**3. Postdictive Processing**

The brain uses postdiction for temporal coherence:

From Herzog et al. (2020) ([Trends in Cognitive Sciences](https://www.sciencedirect.com/science/article/pii/S1364661320301704), cited by 155):

> "Much discussion about the temporal aspects of consciousness centers on effects shorter than 100 ms and whether or not low level mechanisms, such as visual masking, determine how elements are integrated into a single, conscious percept."

### The Temporal Prediction Error

Temporal structure generates prediction errors:

```
Time t: Predict state at t+dt
Time t+dt: Observe actual state
Error: prediction - observation
Update: Minimize prediction error

dt ~ 100ms (fundamental update cycle)
```

### Cross-Modal Temporal Binding

The brain compensates for different processing delays across modalities:

- Vision: ~100 ms to cortex
- Audition: ~50 ms to cortex
- Touch: ~30-100 ms (depending on body location)

Predictive processing creates apparent simultaneity despite asynchronous processing.

---

## Section 6: Philosophical Implications

### The Extended Present vs. Mathematical Instant

From Wittmann (2011):

> "A debate exists in the philosophical literature surrounding a presumed puzzle of how it is possible to have a temporal experience, to perceive duration, when our experiences are confined to the present moment."

The neuroscience resolves this puzzle:
- The present is not a knife-edge instant
- Experience has inherent temporal extension
- Duration is directly perceived, not inferred

### Retention-Impression-Protention

Husserl's tripartite structure maps onto neural timescales:

**Retention**: Memory traces (fading over ~3s)
**Primal Impression**: Current processing window (~100ms)
**Protention**: Prediction buffer (~100ms ahead)

From Wittmann (2011):

> "The perceived present represents its history and possible future, this tripartite structure being an implicit aspect of any conscious experience."

### The Specious Present Revisited

William James's specious present finds neural grounding:

From James (1890), cited in Wittmann (2011):

> "The practically cognized present is no knife-edge, but a saddle-back, with a certain breadth of its own on which we sit perched, and from which we look in two directions into time."

The neuroscience confirms:
- Present has duration (~3s)
- Includes "rearward-looking end" (retention)
- Includes "forward-looking end" (protention)

### Continuity from Discrete Processing

Despite discrete processing, experience feels continuous:

From VanRullen & Koch (2003), cited in Wittmann (2011):

> "Despite the possibly discrete nature of underlying processes in perception and cognition, our phenomenal experience is nevertheless characterized as evolving continuously."

The solution: semantic connection across segments masks discontinuity.

---

## Section 7: Cross-Cultural and Developmental Evidence

### Universal Temporal Structure

The ~3s experienced moment appears cross-culturally:

From Turner & Poppel (1988), cited in Wittmann (2011):

> "Across different languages and cultures the duration of these lyrical and musical units seems not to exceed 3 s."

This suggests biological constraint rather than cultural construction.

### Musical and Poetic Universals

**Musical Phrases**: ~2-3 seconds
**Poetic Lines**: ~3 seconds maximum
**Spontaneous Speech**: ~3 second segments

From Wittmann (2011):

> "This has, for example, lead to the idea that temporal information within a segment of the speech signal not exceeding the functional moment might not be relevant for decoding spoken language."

### Developmental Trajectory

The thick present develops through childhood:
- Infants: shorter integration windows
- Children: expanding temporal horizons
- Adults: full ~3s experienced moment

Working memory development parallels temporal experience development.

### Clinical Populations

**Aphasia**

From Wittmann et al. (2004), cited in Wittmann (2011):

> "These individuals have difficulties in discriminating consonants, which requires the ability to detect temporal order of speech signal components, because they have increased auditory temporal order thresholds."

**Schizophrenia**

Disrupted temporal integration in schizophrenia:
- Altered simultaneity thresholds
- Fragmented experience
- Timing deficits

### Pharmacological Modulation

From Wittmann (2011):

> "Only in rare neurological disorders or under the influence of pharmacological agents such as LSD individuals occasionally report of perceiving a series of discrete stationary images."

This suggests temporal integration requires intact neurochemistry.

---

## Section 8: ARR-COC-0-1 - Temporal Relevance Windows

### Fundamental Design Principles

The neuroscience of the thick present directly informs VLM architecture. ARR-COC-0-1's relevance realization must operate within temporal constraints analogous to biological systems.

### Three-Level Temporal Relevance

**Level 1: Fast Relevance (30-50ms equivalent)**
- Per-patch feature relevance
- Local binding of features
- No explicit temporal ordering

```python
def fast_relevance(patch_embeddings):
    """
    Lowest level: feature-level relevance
    ~30ms biological timescale

    Within this window, features are bound
    without temporal structure.
    """
    # Local feature attention
    local_attention = compute_local_attention(patch_embeddings)

    # Features within window treated as co-temporal
    return bind_features(local_attention)
```

**Level 2: Medium Relevance (100-500ms equivalent)**
- Per-region relevance
- Spatial integration
- Conscious-level processing

```python
def medium_relevance(fast_outputs, spatial_context):
    """
    Intermediate level: region-level relevance
    ~100-500ms biological timescale

    This is where "experienced" relevance emerges.
    """
    # Integrate fast outputs spatially
    integrated = spatial_integration(fast_outputs)

    # Apply contextual modulation
    contextualized = apply_context(integrated, spatial_context)

    # Precision-weight based on importance
    precision = estimate_precision(contextualized)

    return precision * contextualized
```

**Level 3: Slow Relevance (~3s equivalent)**
- Per-image/query relevance
- Conceptual integration
- Working memory constraints

```python
def slow_relevance(medium_outputs, query, working_memory):
    """
    Highest level: meaning-level relevance
    ~3s biological timescale

    Constrained by working memory capacity.
    """
    # Query-conditional relevance
    query_relevance = compute_query_relevance(medium_outputs, query)

    # Working memory integration
    # (limited capacity ~ 4 items)
    if len(working_memory) > 4:
        # Forget oldest items (retention decay)
        working_memory = working_memory[-4:]

    # Integrate with working memory context
    integrated = integrate_with_memory(query_relevance, working_memory)

    return integrated, working_memory + [query_relevance]
```

### Token Allocation as Temporal Integration

Token allocation in ARR-COC-0-1 mirrors biological temporal integration:

**1. Within-Window Integration**
```python
def integrate_within_window(tokens, window_size=0.1):
    """
    Tokens within window are bound together.

    Analogous to ~100ms cortical integration.
    """
    # Group tokens into temporal windows
    windows = chunk_tokens(tokens, window_size)

    # Integrate within each window
    integrated = []
    for window in windows:
        # Features within window are co-temporal
        bound = bind_tokens(window)
        integrated.append(bound)

    return integrated
```

**2. Across-Window Sequencing**
```python
def sequence_across_windows(integrated_windows):
    """
    Windows are ordered temporally.

    Analogous to experienced moment ordering.
    """
    # Establish temporal order
    ordered = establish_order(integrated_windows)

    # Apply retention (fading of past)
    with_retention = apply_retention_decay(ordered)

    # Apply protention (prediction of future)
    with_protention = apply_prediction(with_retention)

    return with_protention
```

### The Thick Present in VLM Processing

**1. Retention (Past Relevance)**

Previous relevance patterns influence current processing:

```python
class RelevanceRetention:
    def __init__(self, decay_rate=0.7):
        self.decay_rate = decay_rate
        self.retained = None

    def update(self, current_relevance):
        if self.retained is None:
            self.retained = current_relevance
        else:
            # Exponential decay (like memory traces)
            self.retained = (
                self.decay_rate * self.retained +
                (1 - self.decay_rate) * current_relevance
            )
        return self.retained
```

**2. Primal Impression (Current Relevance)**

The current processing window (~100ms equivalent):

```python
def current_relevance(features, query, context):
    """
    The 'now' of relevance realization.

    Integrates features within current window.
    """
    # Compute attention scores
    attention = compute_attention(features, query)

    # Precision-weight by context
    precision = estimate_precision(features, context)
    weighted = precision * attention

    return weighted
```

**3. Protention (Future Relevance)**

Anticipate what will be relevant:

```python
def predict_future_relevance(current, temporal_context):
    """
    Predict upcoming relevance needs.

    Allows pre-allocation of computational resources.
    """
    # Use temporal context to predict
    predicted = temporal_model(temporal_context)

    # Pre-allocate tokens for predicted regions
    pre_allocated = allocate_tokens(predicted)

    return pre_allocated
```

### Implementation in ARR-COC-0-1 Pipeline

**Temporal Relevance Realization Pipeline**

```python
class TemporalRelevancePipeline:
    def __init__(self):
        # Three timescales
        self.fast_window = 0.05    # 50ms equivalent
        self.medium_window = 0.3   # 300ms equivalent
        self.slow_window = 3.0     # 3s equivalent

        # Retention mechanism
        self.retention = RelevanceRetention()

        # Working memory
        self.working_memory = []
        self.max_memory = 4  # Biological constraint

    def process(self, image, query):
        # Extract patches
        patches = extract_patches(image)

        # Level 1: Fast relevance (per-patch)
        fast_rel = self.fast_relevance(patches)

        # Level 2: Medium relevance (per-region)
        medium_rel = self.medium_relevance(fast_rel, query)

        # Level 3: Slow relevance (per-image)
        slow_rel = self.slow_relevance(medium_rel, query)

        # Apply retention (thick present)
        thick_rel = self.retention.update(slow_rel)

        # Update working memory
        self.update_memory(thick_rel)

        # Allocate tokens based on thick relevance
        tokens = self.allocate_tokens(thick_rel)

        return tokens

    def allocate_tokens(self, relevance):
        """
        Allocate tokens respecting temporal constraints.

        - Don't flicker (respect retention)
        - Don't exceed capacity (respect working memory)
        - Maintain coherence (respect integration windows)
        """
        # Smooth allocation over time
        smoothed = self.temporal_smooth(relevance)

        # Constrain by working memory capacity
        constrained = self.memory_constrain(smoothed)

        # Generate token distribution
        tokens = generate_tokens(constrained)

        return tokens
```

### Predictions for VLM Design

Based on thick present neuroscience:

**1. Optimal Integration Windows**
- Too short: noisy, flickering attention
- Too long: sluggish, missed changes
- Optimal: ~100ms equivalent for medium level

**2. Working Memory Constraints**
- Limit simultaneous high-relevance regions to ~4
- Older relevance should decay
- Maintain narrative coherence

**3. Temporal Smoothing**
- Exponential smoothing with tau ~ 100ms
- Prevents attention instability
- Creates "thick" relevance

**4. Hierarchical Processing**
- Fast level for feature binding
- Medium level for perceptual relevance
- Slow level for conceptual relevance

### Theoretical Connection

The thick present represents a solution to the temporal binding problem that VLMs must also solve:

**Biological Problem**: How to integrate features processed at different times?
**VLM Problem**: How to integrate patches processed in sequence?

**Solution**: Temporal integration windows that:
1. Treat within-window content as co-temporal
2. Order across-window content temporally
3. Maintain coherence through retention/protention

ARR-COC-0-1's temporal relevance windows implement this same solution in silicon.

---

## Sources

### Primary Neuroscience Sources

**Wittmann, M. (2011)**. Moments in Time. *Frontiers in Integrative Neuroscience*, 5, 66.
- URL: https://www.frontiersin.org/journals/integrative-neuroscience/articles/10.3389/fnint.2011.00066/full
- Accessed: 2025-11-23
- Cited by: 269
- Key contribution: Three-level hierarchy of temporal integration

**Damsma, A., et al. (2021)**. Temporal Context Actively Shapes EEG Signatures of Time Perception. *Journal of Neuroscience*.
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC8152605/
- Accessed: 2025-11-23
- Cited by: 64
- Key contribution: Predictive processing in temporal perception

**Herzog, M. H., et al. (2020)**. All in Good Time: Long-Lasting Postdictive Effects. *Trends in Cognitive Sciences*.
- URL: https://www.sciencedirect.com/science/article/pii/S1364661320301704
- Cited by: 155
- Key contribution: 100-500ms integration windows

**Ronconi, L., et al. (2018)**. Alpha-band sensory entrainment alters the duration of temporal windows. *Scientific Reports*.
- URL: https://www.nature.com/articles/s41598-018-29671-5
- Cited by: 107
- Key contribution: Alpha oscillations and temporal resolution

**Wutz, A., et al. (2016)**. Temporal Integration Windows in Neural Processing. *Current Biology*.
- URL: https://www.cell.com/current-biology/pdf/S0960-9822(16)30464-X.pdf
- Cited by: 146
- Key contribution: MEG evidence for 150ms windows

### Classic Papers on Temporal Perception

**Poppel, E. (1997)**. A hierarchical model of temporal perception. *Trends in Cognitive Sciences*, 1(2), 56-61.
- Classic framework for temporal integration levels
- Cited by 654

**James, W. (1890)**. *The Principles of Psychology*. MacMillan.
- Original description of specious present
- Chapter XIV on time perception

**Husserl, E. (1928)**. *Vorlesungen zur Phanomenologie des inneren Zeitbewusstseins*.
- Retention-impression-protention structure
- Phenomenological foundation

### Existing Oracle Knowledge

**friston/05-temporal-dynamics-100ms.md**
- Comprehensive treatment of temporal dynamics in predictive processing
- Mathematical formulations
- ARR-COC-0-1 integration

**temporal-phenomenology/01-james-specious-present.md**
- Philosophical background
- ~3 second specious present

**temporal-phenomenology/03-husserl-time-consciousness.md**
- Phenomenological structure
- Retention-protention analysis

### Psychophysics Sources

**Brecher, G. A. (1932)**. Die Entstehung und biologische Bedeutung der subjektiven Zeiteinheit. *Zeitschrift fur vergleichende Physiologie*, 18, 204-243.
- Original definition of psychological moment (55ms)

**Efron, R. (1970)**. The minimum duration of a perception. *Neuropsychologia*, 8, 57-63.
- 137ms minimum experience duration

**Nakajima, Y., et al. (1980)**. On the perception of two successive sound bursts. *Psychological Research*, 41, 335-344.
- 150ms-2s qualitative difference

### Additional Research (2020-2025)

**Comolatti, R., et al. (2025)**. Why does time feel the way it does? *iScience*.
- URL: https://www.cell.com/iscience/fulltext/S2589-0042(25)01695-5
- Cited by: 6
- Key: Principled account of temporal experience

**Kent, L. (2021)**. Time consciousness: the missing link in theories of consciousness. *Neuroscience of Consciousness*.
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC8042366/
- Cited by: 112
- Key: Temporal binding in consciousness

**Merchant, H., et al. (2013)**. Neural basis of the perception and estimation of time. *Annual Review of Neuroscience*, 36, 313-336.
- URL: https://pubmed.ncbi.nlm.nih.gov/23725000/
- Cited by: 894
- Key: Comprehensive review of timing

---

**Total Lines**: ~700
**Knowledge Type**: ACQUISITION (Web research 2025-11-23 + existing oracle integration)
**ARR-COC-0-1 Integration**: Section 8 (~70 lines, 10%)
**Key Insight**: The thick present emerges from hierarchical temporal integration (30ms -> 100ms -> 3s -> 60s) that VLM attention mechanisms should mirror
**Primary Citations**: Wittmann 2011, Herzog 2020, Poppel 1997, James 1890, Husserl 1928
