# Saccades & Eye Movements

**Saccadic control, fixation planning, smooth pursuit, scanpaths, and task-driven attention allocation in cognitive neuroscience and VLM applications**

From [karpathy/biological-vision/01-saccades-eye-movements.md](../karpathy/biological-vision/01-saccades-eye-movements.md):
> "Saccades are rapid, ballistic eye movements that humans make to redirect the fovea toward objects of interest in the visual field. Understanding saccade planning and execution is crucial for designing vision models that mimic human attention allocation."

---

## Section 1: Saccade Fundamentals (~100 lines)

### 1.1 What Are Saccades?

**Definition**: Rapid, ballistic eye movements that reorient the fovea to bring new visual information into high-acuity central vision.

From [Evidence for five types of fixation during a random saccade task](https://pmc.ncbi.nlm.nih.gov/articles/PMC11404824/) (Friedman, 2024, accessed 2025-11-16):
- Five distinct fixation types identified during random saccade tasks
- Long fixations common in first 15-22 seconds, then decrease
- Fixation duration reflects cognitive processing demands

**Key Properties:**

**Speed**: Fastest movements the human body produces
- Peak velocity: 400-700°/second for large saccades
- Duration: 30-120ms for typical saccades (5-40° amplitude)
- Main sequence relationship: velocity increases with amplitude

From [The saccade main sequence revised](https://pmc.ncbi.nlm.nih.gov/articles/PMC7880984/) (Gibaldi et al., 2020):
- Main sequence applies across vergence states
- Saccades precisely coordinated despite ballistic nature
- Kinematics well-defined even for complex 3D eye movements

**Ballistic Nature**:
- Pre-programmed motor commands
- Cannot be modified mid-flight
- Reflects commitment to target selection before execution

**Why Saccades Matter:**

From [karpathy/biological-vision/01-saccades-eye-movements.md](../karpathy/biological-vision/01-saccades-eye-movements.md):
> "The foveated architecture creates a biological necessity for saccades: Foveal photoreceptor density: ~200,000 cones/mm². Peripheral density (10° eccentricity): ~20,000 cones/mm² (10× reduction). Only ~1-2° of central vision provides detailed form recognition."

**Biological Rationale:**
- Solve the problem of non-uniform retinal sampling
- Move high-resolution fovea to relevant locations
- More efficient than processing entire visual field at high resolution
- Enable serial sampling strategy for visual information

### 1.2 Saccade Planning Mechanisms

**Neural Substrates:**

From [Saliency Response in Superior Colliculus at the Future Saccade Goal](https://www.jneurosci.org/content/45/3/e0428242024) (Heeman et al., 2025, accessed 2025-11-16):
- Fixation duration as function of saliency at saccade goal
- Saliency computed 25-75ms after fixation start
- Superior colliculus encodes priority maps for saccade targets

**Superior Colliculus (SC):**
- Midbrain structure organizing spatial priority maps
- Topographic representation of visual field
- Integrates visual, auditory, somatosensory signals
- Direct role in saccade initiation and target selection

From [Activity in brain system that controls eye movements](https://biologicalsciences.uchicago.edu/news/brain-superior-colliculus-spatial-thinking) (University of Chicago, 2024):
- SC plays dual role: motor control AND higher cognitive functions
- Not just reflex center - involved in spatial reasoning
- Saccade planning deeply intertwined with cognitive processing

**Frontal Eye Fields (FEF):**
- Frontal cortex region in premotor areas
- Voluntary saccade control
- Task-dependent modulation of saccade targets
- Implements top-down attentional control

**Lateral Intraparietal Area (LIP):**
- Parietal cortex region encoding spatial attention
- Represents salience and behavioral relevance
- Priority map for saccade target selection
- Integrates sensory evidence for decision-making

### 1.3 Bottom-Up vs Top-Down Saccade Control

**Bottom-Up Salience Signals:**

**Visual contrast**: High-contrast boundaries attract fixations
**Sudden onsets**: New objects trigger reflexive saccades
**Express saccades**: Reaction times ~100ms (humans), ~70ms (monkeys)

From [Express saccades and visual attention](https://www.yorku.ca/science/research/schalljd/wp-content/uploads/sites/654/2022/10/Schall_Hanes-ON-Fischer_Weber_-Saccades-and-Visual-Attention-complete.pdf) (Fischer & Weber, 1993):
- Express saccades occur when attention not strongly bound to fixation
- Reflect pre-programmed saccade plans ready for execution
- Related to visual attention disengagement

**Top-Down Task Goals:**

From [Perceptual task drives later fixations and long latency saccades](https://journals.sagepub.com/doi/10.1177/03010066241253816) (Metzger et al., 2024, accessed 2025-11-16):
- Task dramatically affects fixation duration and saccade latency
- After first fixation, latency = duration of fixation before saccade
- Time to plan saccades reflects task demands

**Goal-directed search**: Target templates modulate saccade attraction
**Semantic knowledge**: Fixations cluster on informative regions
**Query-driven attention**: Questions reshape scanpaths completely

From [Eye movements in response to different cognitive activities](https://pmc.ncbi.nlm.nih.gov/articles/PMC10676768/) (Marconi et al., 2023):
- Eye movement direction influenced by cognitive activity
- Imagination, internal dialogue, memory retrieval affect saccades
- Not purely stimulus-driven - cognitive state shapes where we look

---

## Section 2: Fixation Duration & Scanpath Patterns (~100 lines)

### 2.1 What Fixation Duration Reveals

**Fixation**: Period when eyes are relatively stationary, allowing visual processing.

From [Evidence for five types of fixation during a random saccade task](https://pmc.ncbi.nlm.nih.gov/articles/PMC11404824/) (Friedman, 2024):
- Five fixation types with distinct duration profiles
- Long fixations (>300ms) common early in viewing
- Short fixations (<200ms) increase after initial exploration

**Typical Fixation Durations:**
- Reading: 200-250ms (tightly linked to lexical processing)
- Scene viewing: 250-350ms (longer for encoding/memorization)
- Visual search: 180-250ms (varies with task difficulty)

**What Modulates Duration:**

**Processing difficulty**: Harder stimuli → longer fixations
**Information density**: More complex regions → longer dwells
**Task demands**: Memorization → longer than aesthetic judgment
**Predictability**: Unexpected content → prolonged fixations

From [Saccade size predicts onset time of object processing](https://www.sciencedirect.com/science/article/pii/S1053811924002787) (Gordon et al., 2024, accessed 2025-11-16):
- Saccade amplitude affects onset timing of object processing
- Minimum saccade duration: 12ms
- Velocity factor of six times median velocity used for detection

### 2.2 Scanpath Analysis

**Scanpath**: Sequence of saccades and fixations describing eye movement pattern.

From [A review of machine learning in scanpath analysis for passive gaze-based interaction](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1391745/full) (Mohamed Selim et al., 2024, accessed 2025-11-16):
- Machine learning applications in scanpath analysis (2012-2022 review)
- Scanpaths reveal task-specific strategies
- Can detect cognitive states, expertise, individual differences

**Typical Free Viewing Patterns:**

**Central fixation bias**: Initial fixation tends toward image center
**Exploration phase** (first 1-2 seconds): Broad exploratory saccades, large amplitudes
**Refinement phase** (after 2-3 seconds): Smaller saccades, detailed inspection

From [The Sources of Variability in Saccadic Eye Movements](https://pmc.ncbi.nlm.nih.gov/articles/PMC6672172/) (van Beers, 2007):
- Saccade variability reflects sensory uncertainty and motor noise
- Individual signatures in saccade patterns
- Endpoint variability 5-10% of amplitude

**Task-Dependent Scanpaths:**

From [Task-driven Eye Movement Control for Chart Reading](https://arxiv.org/html/2502.03575v1) (Shi et al., 2025, accessed 2025-11-16):
- Computational model simulates task-driven eye movements for chart reading
- Predicts human-like task-driven scanpaths across various tasks
- Applicable in explainable AI and visualization evaluation

**Reading**: Highly stereotyped left-to-right (for L-to-R languages)
**Visual search**: Systematic scanning, efficient searchers use fewer fixations
**Memorization**: More uniform spatial coverage, deliberate encoding
**Question answering**: Saccades directed to answer-likely regions

From [How readers attentive and inattentive to task-related information differ in scanpath](https://www.sciencedirect.com/science/article/pii/S2543925124000093) (Chen et al., 2024, accessed 2025-11-16):
- Scanpath stores scene and sequence features of eye-movement behavior
- Attentive vs inattentive readers show distinct scanpath patterns
- Task-related information processing affects fixation sequences

### 2.3 Scanpath Similarity Metrics

From [Eye Movement and Pupil Measures: A Review](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2021.733531/full) (Mahanama et al., 2022):
- Multiple metrics needed to characterize scanpath similarity
- Spatial overlap insufficient - temporal dynamics matter
- Scanpath comparison used in expertise assessment, UI evaluation

**Fixation-based metrics:**
- Number of fixations per trial
- Mean fixation duration
- Spatial distribution (coverage, clustering)

**Saccade-based metrics:**
- Amplitude distribution
- Direction biases (horizontal vs vertical)
- Velocity and acceleration profiles

From [Regularities in vertical saccadic metrics](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2023.1157686/full) (Greene, 2023):
- Asymmetries between vertical and horizontal saccades
- Horizontal saccades more common during natural viewing
- Vertical saccades have distinct kinematics

**Scanpath similarity:**
- String-edit distance on fixation sequences
- MultiMatch algorithm (multiple scanpath dimensions)
- Cross-correlation of fixation density maps

**Inter-observer agreement:**
- Free viewing: moderate (AUC ~0.75)
- Specific question: high (AUC ~0.85)
- Agreement increases with task constraint

---

## Section 3: Smooth Pursuit Eye Movements (~80 lines)

### 3.1 Smooth Pursuit Fundamentals

**Definition**: Smooth, continuous eye movements used to track moving objects.

From [Differential Effects of Visual and Auditory Cognitive Tasks on Smooth Pursuit](https://pmc.ncbi.nlm.nih.gov/articles/PMC12051362/) (Kaye et al., 2025, accessed 2025-11-16):
- Important interaction between cognitive load and sensory modality
- Cognitive tasks affect smooth pursuit eye movement (SPEM) control
- Visual cognitive tasks impair pursuit more than auditory tasks

**Key Differences from Saccades:**

| Property | Saccades | Smooth Pursuit |
|----------|----------|----------------|
| **Speed** | 400-700°/s | 30-100°/s |
| **Control** | Ballistic (pre-programmed) | Continuous feedback |
| **Function** | Reorient fovea | Stabilize moving target |
| **Latency** | 100-250ms | 100-150ms initial |

**Neural Control:**

From [Predictive smooth pursuit eye movements](https://www.annualreviews.org/content/journals/10.1146/annurev-vision-091718-014901) (Kowler, 2019):
- Pursuit driven by predictive mechanisms
- Not purely reactive - anticipates target motion
- Integrates sensory evidence with internal models

**Function**: Stabilize image of moving target on fovea for detailed processing

### 3.2 Pursuit and Saccade Interaction

From [Interactions between Saccades and Smooth Pursuit Eye Movements](https://www.eneuro.org/content/11/6/ENEURO.0027-24.2024) (Pattadkal et al., 2024, accessed 2025-11-16):
- Coordination between smooth pursuit and saccadic systems in marmosets
- Single and multiple object motion tasks
- Catch-up saccades correct pursuit errors

**Catch-up saccades**: Small saccades during pursuit to correct position errors
**Anticipatory pursuit**: Smooth pursuit begins before target motion (predictive)
**Pursuit initiation**: Brief saccade often precedes smooth pursuit onset

### 3.3 Cognitive Influences on Pursuit

From [Internal coupling: Eye behavior coupled to visual imagery](https://www.sciencedirect.com/science/article/pii/S0149763424003245) (Korda et al., 2024, accessed 2025-11-16):
- Smooth pursuit elicited by imagining rotating thumb
- Eye movements coupled to internal imagery
- Not purely stimulus-driven - mental states affect pursuit

**Attention and pursuit**:
- Divided attention impairs smooth pursuit gain
- Cognitive load reduces pursuit accuracy
- Task-irrelevant distractors disrupt pursuit

From [Smooth pursuit inhibition reveals audiovisual enhancement of oculomotor control](https://jov.arvojournals.org/article.aspx?articleid=2793510) (Kreyenmeier et al., 2024, accessed 2025-11-16):
- Audiovisual distractors evoke stronger oculomotor inhibition
- Multisensory response enhancement in pursuit control
- Audiovisual integration affects eye movement planning

---

## Section 4: Effort, Cost, and Saccadic Decision Making (~100 lines)

### 4.1 Saccadic Effort and Resource Allocation

From [Effort drives saccade selection](https://elifesciences.org/articles/97760) (Koevoet et al., 2025, accessed 2025-11-16):
- Saccade selection minimizes effort expenditure
- Humans prefer saccades reducing physical and cognitive costs
- Trade-off between information gain and movement effort
- Task difficulty modulates willingness to make costly saccades
- Pupil size reflects saccadic effort (well-established marker)

**Physical Effort Costs:**
- Larger amplitude saccades require more motor effort
- Vertical saccades may be more effortful than horizontal
- Oblique directions show different costs than cardinal

**Cognitive Effort Costs:**
- Target uncertainty increases planning effort
- Ambiguous targets increase saccade latency
- Effortful tasks reduce saccade frequency

From [Efficient Saccade Planning Requires Time and Clear Choices](https://www.researchgate.net/publication/275438424_Efficient_Saccade_Planning_Requires_Time_and_Clear_Choices) (accessed 2025-11-16):
- Efficient saccades require clear priority signals
- Ambiguous targets increase saccade latency
- Information-maximizing saccades selected when priorities clear

### 4.2 Coupling of Saccades to Attention

From [Coupling of saccade plans to endogenous attention during urgent choices](https://pubmed.ncbi.nlm.nih.gov/38496491/) (Goldstein et al., 2024, accessed 2025-11-16):
- Saccade planning tightly coupled to endogenous (voluntary) attention
- Salient stimuli automatically capture both attention and saccade plans
- Difficult to decouple saccade planning from attention shifts
- Shared mechanisms for attention and eye movement control

**Prosaccade trials**: Fixation period (500-700ms), then go signal
**Attention shifts precede saccades**: Covert attention moves to target before overt saccade
**Shared priority maps**: LIP and FEF encode both attention and saccade plans

### 4.3 Fixation-Related Saccadic Inhibition

From [Fixation-related saccadic inhibition in free viewing](https://www.nature.com/articles/s41598-022-10605-1) (Kadosh et al., 2022, accessed 2025-11-16):
- Saccadic inhibition occurs around fixation onset
- Prevents premature saccades during visual processing
- Free viewing shows characteristic inhibition patterns

**Mechanism**: Transient suppression of saccade generation during fixation
**Function**: Allows sufficient processing time at each fixation
**Duration**: Typically 50-150ms after fixation onset

### 4.4 Motor "Laziness" and Fixation Selection

From [Motor "laziness" constrains fixation selection in real-world tasks](https://www.pnas.org/doi/10.1073/pnas.2302239121) (Burlingham et al., 2024, accessed 2025-11-16):
- Average fixation duration varies with position relative to head
- Total number of fixations depends on spatial layout
- Effort costs constrain where people look
- Real-world tasks show systematic biases favoring less effortful saccades

**Implications for VLMs:**
- Saccade/fixation planning should account for effort costs
- Not all equally salient regions will receive equal attention
- Resource-rational models need effort component

---

## Section 5: Foveated Vision & VR Rendering (~100 lines)

### 5.1 Foveated Rendering Principles

From [Efficient VR rendering: Survey on foveated, stereo, cloud rendering](https://www.sciencedirect.com/science/article/pii/S2096579625000580) (Xiao et al., 2025, accessed 2025-11-16):
- Foveated rendering optimizes computational efficiency
- Simulates nonuniform perceptual characteristics of human vision
- Dynamically allocates rendering resources based on gaze
- High resolution at fovea, degraded periphery

**Why Foveated Rendering Works:**
- Human visual system has nonuniform sampling (see Section 1.1)
- Peripheral vision poor at detecting fine detail
- Foveal region (~1-2°) requires full resolution
- Can reduce rendering load by 5-10× with minimal perceptual loss

From [How does foveated rendering work, and what are its benefits in VR](https://milvus.io/ai-quick-reference/how-does-foveated-rendering-work-and-what-are-its-benefits-in-vr) (accessed 2025-11-16):
- Technique optimizes graphics by rendering high-resolution only where user looks
- Reduces detail in peripheral vision
- Matches human visual acuity distribution

### 5.2 Eye Tracking for Dynamic Foveated Rendering

From [Eye Tracked Foveated Rendering (Meta Quest)](https://developers.meta.com/horizon/documentation/unreal/unreal-eye-tracked-foveated-rendering/) (Meta, December 2024, accessed 2025-11-16):
- Eye Tracked Foveated Rendering (ETFR) utilizes gaze direction
- Renders full resolution where you are looking (foveal region)
- Low pixel density in periphery
- Matches peripheral acuity in human vision

**Fixed vs Dynamic Foveated Rendering:**

**Fixed FR**: Assumes gaze at center, degrades periphery uniformly
**Dynamic FR (Eye-Tracked)**: Follows actual gaze, updates foveal region in real-time

From [Microsoft Flight Simulator 2024 Now Has Foveated Rendering](https://www.uploadvr.com/microsoft-flight-simulator-2024-now-has-foveated-rendering/) (UploadVR, May 2025, accessed 2025-11-16):
- MSFS 2024 now has both fixed and eye-tracked foveated rendering
- Range of improvements to VR support
- Quad views foveated rendering technique (Varjo, merged into OpenXR 1.1)

### 5.3 Individualized Foveated Rendering

From [Individualized foveated rendering with eye-tracking head-mounted displays](https://link.springer.com/article/10.1007/s10055-023-00931-8) (Kim et al., 2024, accessed 2025-11-16):
- Developed individualized FR (IFR) method
- Uses different central vision sizes across individuals
- Different peripheral vision resolutions per person
- Addresses individual differences in visual acuity

**Why Individualization Matters:**
- Foveal region size varies (1-2° center, but individual differences)
- Peripheral acuity drop-off rates vary
- Age affects peripheral sensitivity
- Task affects effective foveal size

### 5.4 VR Applications and Performance

From [VRS Foveated Rendering (OpenXR Toolkit Eye Tracking)](https://forum.dcs.world/topic/353594-vrs-foveated-rendering-openxr-toolkit-eye-tracking-working-after-last-dcs-update/) (ED Forums, 2024, accessed 2025-11-16):
- VRS (Variable Rate Shading) foveated rendering in DCS
- OpenXR Toolkit eye tracking integration
- Works after recent DCS update

**Performance Benefits:**
- 5-10× reduction in pixel shading load
- Enables higher frame rates in VR
- Reduces GPU power consumption
- Maintains perceptual quality

**Challenges:**
- Eye tracking latency (needs <20ms for imperceptible lag)
- Prediction of next gaze location
- Handling fast saccades (image blur during saccade masks transitions)
- Individual calibration requirements

---

## Section 6: Pipeline Parallelism for Saccadic Processing (File 2) (~80 lines)

### 6.1 Distributed Saccade Planning

From [distributed-training/01-deepspeed-pipeline-parallelism.md](../distributed-training/01-deepspeed-pipeline-parallelism.md):
> Pipeline parallelism splits model layers across GPUs, enabling training of models too large for single GPU memory.

**Application to Saccadic Models:**

**Stage 1 (Early Visual)**: Extract salience maps from visual features
**Stage 2 (Priority Integration)**: Combine bottom-up salience with top-down goals
**Stage 3 (Saccade Planning)**: Select next fixation target via priority map
**Stage 4 (Motor Command)**: Generate saccade motor program

**Why Pipeline Parallelism Helps:**
- Saccadic models process hierarchical visual-motor pipeline
- Natural decomposition into stages
- Each stage can be large (vision encoder, transformer attention, motor decoder)
- Memory-efficient: Only one stage per GPU

**Bubble Management:**
- Saccade planning has strict latency requirements (~150-250ms in humans)
- Pipeline bubbles during warmup reduce throughput
- Microbatching helps: Process multiple images in parallel
- GPipe scheduling minimizes bubbles

### 6.2 Multi-Fixation Batch Processing

**Scenario**: Process scanpaths from multiple images/queries simultaneously

**Naive Approach**: Sequential processing (fixation 1 → fixation 2 → ... → fixation N)
**Pipeline Approach**: Overlap stages across fixations

**Example (4-stage pipeline, 3 fixations):**

```
Time 1: [Fixation 1 - Stage 1] [ - ] [ - ] [ - ]
Time 2: [Fixation 2 - Stage 1] [Fixation 1 - Stage 2] [ - ] [ - ]
Time 3: [Fixation 3 - Stage 1] [Fixation 2 - Stage 2] [Fixation 1 - Stage 3] [ - ]
Time 4: [ - ] [Fixation 3 - Stage 2] [Fixation 2 - Stage 3] [Fixation 1 - Stage 4]
```

**Benefit**: ~4× throughput increase (matches number of stages)

### 6.3 Gradient Accumulation for Scanpath Training

**Challenge**: Training on full scanpaths (10-20 fixations) requires large memory

**Solution**: Gradient accumulation across fixations
- Forward pass fixation 1, accumulate gradients
- Forward pass fixation 2, accumulate gradients
- ...
- Backward pass once after full scanpath

**Memory savings**: Only store activations for current fixation, not entire sequence

---

## Section 7: ML Pipelines for Eye Movement Experiments (File 10) (~80 lines)

### 7.1 Kubeflow Pipelines for Eye Tracking Research

From [gcp-vertex/01-pipelines-kubeflow-integration.md](../gcp-vertex/01-pipelines-kubeflow-integration.md):
> "Vertex AI Pipelines lets you automate, monitor, and govern your machine learning (ML) systems in a serverless manner by using ML pipelines to orchestrate your ML workflows."

**Typical Eye Tracking Experiment Pipeline:**

**Step 1: Data Collection**
- Eye tracker calibration
- Stimulus presentation
- Raw gaze coordinate recording
- Event logging (stimulus onsets, responses)

**Step 2: Preprocessing**
- Saccade/fixation detection algorithms
- Blink removal
- Drift correction
- Coordinate transformation

**Step 3: Feature Extraction**
- Compute scanpath metrics (see Section 2.3)
- Calculate fixation durations
- Extract saccade amplitudes/velocities
- Build fixation heatmaps

**Step 4: Analysis**
- Statistical tests (fixation duration differences, etc.)
- Machine learning classification (expert vs novice scanpaths)
- Scanpath similarity comparisons

**Step 5: Visualization**
- Scanpath overlays on images
- Heatmap generation
- Statistical plots

### 7.2 Kubeflow Pipeline Example

From [gcp-vertex/01-pipelines-kubeflow-integration.md](../gcp-vertex/01-pipelines-kubeflow-integration.md):
- KFP SDK v2 uses decorators: `@dsl.pipeline`, `@dsl.component`
- Compile pipelines to generic IR YAML
- Enhanced workflow GUI for visualization

**Components:**

```python
@dsl.component
def preprocess_eyetracking(raw_data: Input[Dataset],
                           preprocessed: Output[Dataset]):
    # Saccade detection, blink removal
    pass

@dsl.component
def extract_scanpath_features(preprocessed: Input[Dataset],
                              features: Output[Dataset]):
    # Compute fixation durations, saccade metrics
    pass

@dsl.component
def train_scanpath_classifier(features: Input[Dataset],
                               model: Output[Model]):
    # ML classification of scanpaths
    pass

@dsl.pipeline
def eyetracking_analysis_pipeline():
    preprocess_task = preprocess_eyetracking(raw_data)
    features_task = extract_scanpath_features(preprocess_task.outputs['preprocessed'])
    train_task = train_scanpath_classifier(features_task.outputs['features'])
```

### 7.3 Experiment Reproducibility

From [gcp-vertex/01-pipelines-kubeflow-integration.md](../gcp-vertex/01-pipelines-kubeflow-integration.md):
- Vertex ML Metadata (managed)
- Tracks artifacts, executions, parameters
- Enables reproducible experiments

**Benefits for Eye Tracking Research:**
- Version control for preprocessing algorithms
- Track saccade detection parameters (velocity thresholds, etc.)
- Compare scanpath metrics across algorithm versions
- Reproduce published results exactly

---

## Section 8: ARR-COC-0-1 Saccadic Relevance Allocation (10%) (~70 lines)

### 8.1 Foveated Vision as Saccadic Token Allocation

**ARR-COC-0-1 Connection**: The relevance realization framework implements a **computational analogue of saccadic planning**.

From [karpathy/biological-vision/01-saccades-eye-movements.md](../karpathy/biological-vision/01-saccades-eye-movements.md):
> "Understanding saccade planning and execution is crucial for designing vision models that mimic human attention allocation."

**Biological Saccades → ARR-COC Token Allocation:**

| Biological | ARR-COC Computational |
|------------|----------------------|
| Saccade targets priority map | Relevance scores guide patch selection |
| Fixation duration ∝ processing needs | Token budget ∝ relevance |
| Bottom-up salience + top-down goals | Propositional + Perspectival + Participatory |
| 3-4 saccades/second | Allocate 200 patches at variable LOD |
| Foveal high-res, peripheral low-res | 64-400 tokens per patch |

**Key Insight**: Instead of literally moving an eye, ARR-COC allocates processing resources (tokens) to image regions based on relevance - **a distributed, parallel saccadic strategy**.

### 8.2 Scanpath as Patch Selection Sequence

**Traditional VLM**: Uniform grid, all patches get equal tokens
**ARR-COC**: Non-uniform allocation based on relevance (like scanpath)

**Analogy:**
- **Fixation sequence** → Rank-ordered patch priority
- **Fixation duration** → Token budget per patch
- **Saccade amplitude** → Distance in feature space between prioritized patches
- **Return fixations** → Re-allocate tokens to previously sampled patches if query updates

From [cognitive-mastery/02-salience-relevance-realization.md](02-salience-relevance-realization.md):
> "Transjective relevance: Neither objective (in image alone) nor subjective (in query alone), but emerges from the relationship between query and content. Like a shark's fitness for the ocean."

**Query-driven scanpaths**: Just as "Where is the dog?" changes human scanpaths, ARR-COC's Participatory scoring changes patch priorities.

### 8.3 Effort-Aware Token Allocation

From [Effort drives saccade selection](https://elifesciences.org/articles/97760) (Koevoet et al., 2025):
> "Trade-off between information gain and movement effort. Task difficulty modulates willingness to make costly saccades."

**ARR-COC Parallel**:

**Information gain** → Relevance score (how informative is this patch?)
**Effort cost** → Token cost (how many tokens to allocate?)

**Optimization Problem:**
```
Maximize: Σ (relevance[i] × tokens[i]) - λ × Σ tokens[i]
Subject to: Σ tokens[i] ≤ TOTAL_BUDGET (e.g., 200 patches × avg 200 tokens = 40k)
```

Where λ is the effort penalty (like saccadic effort cost).

**Result**: High-relevance patches get more tokens (like longer fixations), but extreme token counts avoided unless absolutely necessary (effort constraint).

### 8.4 Multi-Fixation Processing for Complex Queries

**Biological**: Complex tasks require multiple fixations to integrate information
**ARR-COC Extension** (future work): Iterative relevance realization

**Iteration 1**: Initial relevance scoring, allocate tokens
**Process**: Run vision encoder with allocated tokens
**Update**: Query representation updates based on what was "seen"
**Iteration 2**: Re-score relevance (some patches become more/less relevant)
**Re-allocate**: Shift tokens to newly-relevant regions

**Example Query**: "Count the red cars"
- Iteration 1: Find car-like regions (high relevance)
- Iteration 2: Among cars, focus on color (update relevance for red patches)
- Iteration 3: Count instances (allocate tokens for verification)

This mimics **serial visual search** with multiple saccades/fixations.

From [karpathy/biological-vision/01-saccades-eye-movements.md](../karpathy/biological-vision/01-saccades-eye-movements.md):
> "Fixation order = processing order for high-acuity vision. Earlier fixations indicate higher priority or urgency."

ARR-COC could implement this by **sequential token allocation updates**, where patch priorities evolve as the query is "understood" through iterative processing.

---

## Sources

### Existing Knowledge

**From this knowledge base:**
- [karpathy/biological-vision/01-saccades-eye-movements.md](../karpathy/biological-vision/01-saccades-eye-movements.md) - Comprehensive saccade fundamentals, planning, scanpaths
- [cognitive-mastery/02-salience-relevance-realization.md](02-salience-relevance-realization.md) - Relevance realization framework

### Web Research (2024-2025)

**Saccade Planning & Fixation:**
- [Evidence for five types of fixation during a random saccade task](https://pmc.ncbi.nlm.nih.gov/articles/PMC11404824/) - Friedman, 2024 (accessed 2025-11-16)
- [Saliency Response in Superior Colliculus at the Future Saccade Goal](https://www.jneurosci.org/content/45/3/e0428242024) - Heeman et al., 2025 (accessed 2025-11-16)
- [Perceptual task drives later fixations and long latency saccades](https://journals.sagepub.com/doi/10.1177/03010066241253816) - Metzger et al., 2024 (accessed 2025-11-16)
- [Saccade size predicts onset time of object processing](https://www.sciencedirect.com/science/article/pii/S1053811924002787) - Gordon et al., 2024 (accessed 2025-11-16)

**Effort & Decision Making:**
- [Effort drives saccade selection](https://elifesciences.org/articles/97760) - Koevoet et al., 2025 (accessed 2025-11-16)
- [Coupling of saccade plans to endogenous attention during urgent choices](https://pubmed.ncbi.nlm.nih.gov/38496491/) - Goldstein et al., 2024 (accessed 2025-11-16)
- [Motor "laziness" constrains fixation selection in real-world tasks](https://www.pnas.org/doi/10.1073/pnas.2302239121) - Burlingham et al., 2024 (accessed 2025-11-16)

**Smooth Pursuit:**
- [Differential Effects of Visual and Auditory Cognitive Tasks on Smooth Pursuit](https://pmc.ncbi.nlm.nih.gov/articles/PMC12051362/) - Kaye et al., 2025 (accessed 2025-11-16)
- [Interactions between Saccades and Smooth Pursuit Eye Movements](https://www.eneuro.org/content/11/6/ENEURO.0027-24.2024) - Pattadkal et al., 2024 (accessed 2025-11-16)
- [Internal coupling: Eye behavior coupled to visual imagery](https://www.sciencedirect.com/science/article/pii/S0149763424003245) - Korda et al., 2024 (accessed 2025-11-16)
- [Smooth pursuit inhibition reveals audiovisual enhancement of oculomotor control](https://jov.arvojournals.org/article.aspx?articleid=2793510) - Kreyenmeier et al., 2024 (accessed 2025-11-16)

**Scanpath Analysis:**
- [A review of machine learning in scanpath analysis for passive gaze-based interaction](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1391745/full) - Mohamed Selim et al., 2024 (accessed 2025-11-16)
- [Task-driven Eye Movement Control for Chart Reading](https://arxiv.org/html/2502.03575v1) - Shi et al., 2025 (accessed 2025-11-16)
- [How readers attentive and inattentive to task-related information differ in scanpath](https://www.sciencedirect.com/science/article/pii/S2543925124000093) - Chen et al., 2024 (accessed 2025-11-16)

**Foveated Rendering (VR):**
- [Efficient VR rendering: Survey on foveated, stereo, cloud rendering](https://www.sciencedirect.com/science/article/pii/S2096579625000580) - Xiao et al., 2025 (accessed 2025-11-16)
- [Individualized foveated rendering with eye-tracking head-mounted displays](https://link.springer.com/article/10.1007/s10055-023-00931-8) - Kim et al., 2024 (accessed 2025-11-16)
- [Eye Tracked Foveated Rendering](https://developers.meta.com/horizon/documentation/unreal/unreal-eye-tracked-foveated-rendering/) - Meta Developers, December 2024 (accessed 2025-11-16)
- [Microsoft Flight Simulator 2024 Now Has Foveated Rendering](https://www.uploadvr.com/microsoft-flight-simulator-2024-now-has-foveated-rendering/) - UploadVR, May 2025 (accessed 2025-11-16)

### Influential Files

**Cited explicitly in this document:**
- File 2: distributed-training/01-deepspeed-pipeline-parallelism.md - Pipeline parallelism for multi-stage saccadic processing
- File 10: gcp-vertex/01-pipelines-kubeflow-integration.md - ML pipelines for eye tracking experiments, reproducibility

**Note**: Files 14 (Apple Metal) was specified in the PART 26 plan but does not exist yet in the knowledge base. The document focuses on existing files and web research.

### Additional References

**Classic Literature (from existing KB):**
- Bahill et al., 1975 - Main sequence relationship
- Gibaldi et al., 2020 - Saccade main sequence revised
- Fischer & Weber, 1993 - Express saccades
- van Beers, 2007 - Saccadic variability sources
- Mahanama et al., 2022 - Eye movement and pupil measures review
- Greene, 2023 - Vertical saccadic metrics
- Kowler, 2019 - Predictive smooth pursuit
