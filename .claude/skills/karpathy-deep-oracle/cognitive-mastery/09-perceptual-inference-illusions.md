# Perceptual Inference & Illusions

## Overview

Perception is not passive reception of sensory data but active **Bayesian inference**—the brain constructs interpretations of the world by combining prior expectations with uncertain sensory evidence. Visual illusions reveal this process: they arise when the brain makes optimal inferences based on statistical regularities that don't match the current stimulus. Bistable perception (where ambiguous images alternate between interpretations) and binocular rivalry (conflicting images to each eye) demonstrate how the brain actively resolves uncertainty through inference mechanisms rather than simply "reading out" sensory inputs.

**Core principle**: What we perceive is not raw sensory data but the brain's best guess about the causes of that data, based on learned priors and likelihood functions.

## Section 1: Helmholtz's Unconscious Inference

### Historical Foundation

**Hermann von Helmholtz (1867)** proposed that perception involves **unconscious inferences**: the brain automatically interprets ambiguous sensory signals by applying learned knowledge about the world.

From [Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/archives/fall2024/entries/hermann-helmholtz/) (accessed 2025-11-16):
> "Helmholtz argues that the brain adjusts the retinal images by a process of 'unconscious inferences.' Helmholtz contends that a child's brain learns to coordinate retinal images with touch sensations and motor commands."

**Key insight**: Perception = Inference from incomplete data

**Three components**:
1. **Sensory data** (retinal image, sound waves) - inherently ambiguous
2. **Prior knowledge** (learned regularities from experience)
3. **Unconscious conclusion** (what we perceive)

**Example - Size constancy**:
- Retinal image of distant person is small
- Prior: People don't shrink with distance
- Inference: Person is normal-sized but far away

### Modern Revival: Perception as Bayesian Inference

**Contemporary neuroscience** rediscovered Helmholtz's ideas through Bayesian frameworks:

```
Posterior (percept) ∝ Likelihood (sensory data | world state) × Prior (world state)
```

**What this means**:
- **Likelihood**: How probable is this sensory signal given different possible world states?
- **Prior**: How probable are different world states based on past experience?
- **Posterior**: The brain's best guess about what caused the sensation

From [Perception as Visual Inference](https://www.dissonances.blog/p/perception-as-visual-inference) (Galen, 2024):
> "Illusions are interesting because our perceptual deviations from reality give us hints about the inference process. Maybe you stare at the image for a while and the brain's inference switches to a different interpretation."

### Literal vs Analogical Inference

**Philosophical debate**: Are perceptual inferences truly inferences (involving propositional content and logic) or merely computation resembling inference?

From [Literal Perceptual Inference](https://philarchive.org/rec/KIELPI) (Kiefer, 2017):
> "Theories of perception that appeal to Helmholtz's idea of unconscious inference ('Helmholtzian' theories) should be taken literally. Perception involves genuine inference over internal representations with semantic content."

**Literal view**:
- Perception manipulates mental representations with semantic content
- Follows inference rules (Bayesian updating)
- Genuinely computes probabilities

**Analogical view**:
- Perception is computation that resembles inference
- No actual propositional attitudes involved
- Metaphorical description of neural processing

**Current consensus**: Perception implements Bayesian inference algorithmically, whether or not it involves conscious-like "thoughts."

## Section 2: Bayesian Perception Framework

### Core Bayesian Equation

**Perception as inverse inference**:

```
P(World | Sensation) = [P(Sensation | World) × P(World)] / P(Sensation)
```

**Components**:
- **P(World | Sensation)**: Posterior - what we perceive
- **P(Sensation | World)**: Likelihood - generative model
- **P(World)**: Prior - learned expectations
- **P(Sensation)**: Marginal likelihood (normalization)

### Generative vs Recognition Models

**Generative model** (top-down):
- Brain's internal model of how world states cause sensations
- "If there's a chair, what retinal images would I expect?"
- Encodes causal structure of the world

**Recognition model** (bottom-up):
- Inverts generative model to infer causes from effects
- "Given this retinal image, what's the most likely object?"
- Often approximated (exact inversion intractable)

**Perception cycle**:
1. Top-down: Generate prediction from current hypothesis
2. Bottom-up: Compute prediction error (data - prediction)
3. Update: Revise hypothesis to reduce error
4. Repeat until convergence

### Optimal Inference Under Uncertainty

**Why Bayesian?** The world is inherently ambiguous:

**Inverse problem**:
- Many world states can produce same sensation
- Retinal image is 2D projection of 3D world
- Infinite 3D scenes map to one 2D image

**Noise**:
- Sensory receptors are noisy (photon shot noise, thermal noise)
- Neural transmission is stochastic
- Brain must infer signal from noise

**Solution**: Use priors to disambiguate

From [Human Visual Motion Perception Shows Hallmarks of Bayesian Inference](https://www.nature.com/articles/s41598-021-82175-7) (Yang et al., 2021):
> "The Bayesian ideal observer suggests statistically correct structural inference by humans. Analysis via an ideal observer was made possible by our novel experimental paradigm allowing precise quantification of uncertainty."

**Hallmarks of Bayesian perception**:
- Perceptual biases reflect natural scene statistics (priors)
- Confidence scales with stimulus uncertainty
- Cue integration weighted by reliability (precision-weighting)

## Section 3: Visual Illusions as Optimal Inference

### Illusions Reveal Priors

**Illusions occur when**:
- Stimulus violates typical statistical regularities
- Optimal inference given normal priors → incorrect percept
- Brain is "right to be wrong" (inference is correct, world is unusual)

From [Illusions, Perception and Bayes](https://www.researchgate.net/publication/11334330_Illusions_perception_and_Bayes) (Weiss et al., 2002):
> "A new model shows that a range of visual illusions in humans can be explained as rational inferences about the odds that a motion stimulus on the retina was caused by an object moving in the world."

### Motion Illusions

**Aperture problem**:
- Viewing moving line through small aperture
- Infinite velocities consistent with local motion
- Brain assumes: Most likely velocity is perpendicular to edge
- **Prior**: Objects move rigidly, not sliding along edges

**Barberpole illusion**:
- Diagonal stripes inside rectangular aperture
- Perceived motion: Along pole's long axis
- Actual motion: Perpendicular to stripes
- **Prior**: Motion direction biased by global shape

**Motion aftereffect** (waterfall illusion):
- Stare at downward motion → static scene appears to move upward
- Adaptation shifts neural baseline
- **Bayesian account**: Recalibrated prior expects downward motion

### Lightness/Brightness Illusions

**Checker-shadow illusion** (Adelson):
- Two squares (A and B) appear different brightness
- Physically identical luminance
- **Prior**: Shadows darken surfaces uniformly
- Inference: B is brighter but in shadow

**White's illusion**:
- Gray patches on black vs white backgrounds
- Same luminance, different perceived brightness
- **Prior**: Surface reflectance varies smoothly
- Inference: Assign lightness based on surrounds

**Mach bands**:
- Illusory bright and dark bands at luminance gradients
- Enhance edge perception
- **Prior**: Edges are informationally important
- Inference: Sharpen transitions for better segmentation

### Size and Depth Illusions

**Müller-Lyer illusion**:
- Lines with inward vs outward arrows
- Same length, different perceived length
- **Prior**: Outward corners (edges of room) farther than inward
- Inference: Applies 3D interpretation to 2D figure

**Ponzo illusion**:
- Converging lines suggest depth (railroad tracks)
- Upper line appears longer
- **Prior**: Parallel lines in world converge in image with distance
- Inference: Upper line is farther and therefore larger

**Ames room**:
- Distorted room appears rectangular
- People appear to change size as they walk
- **Prior**: Rooms have rectangular geometry
- Inference: Perceive room as normal, people as giants/tiny

From [Bayesian Confidence in Optimal Decisions](https://psycnet.apa.org/record/2025-04700-001) (Calder-Travis et al., 2024):
> "Our results favor the hypothesis that confidence reflects the strength of accumulated evidence penalized by the time taken to reach the decision."

## Section 4: Bistable Perception

### What is Bistable Perception?

**Definition**: Ambiguous stimuli that spontaneously alternate between two or more distinct interpretations despite unchanging sensory input.

**Classic examples**:
- **Necker cube**: Wire-frame cube flips between two 3D orientations
- **Rubin's vase**: Face-vase figure-ground reversal
- **Spinning dancer**: Silhouette appears to spin clockwise or counterclockwise

**Key properties**:
- Stochastic transitions (unpredictable timing)
- Mean dominance duration ~2-4 seconds
- Individual differences in switch rate
- Attention and volition can modulate (partially)

### Neural Mechanisms

From [An Accumulating Neural Signal Underlying Binocular Rivalry](https://www.jneurosci.org/content/43/50/8777) (Nie et al., 2023):
> "During binocular rivalry, conflicting images are presented one to each eye and perception alternates stochastically between them. We found neurons in lateral intraparietal cortex (LIP) accumulate evidence consistent with perceptual decisions."

**Accumulation model**:
- Evidence for each interpretation accumulates over time
- Stochastic noise causes fluctuations
- When one accumulator crosses threshold → perceptual switch
- Mutual inhibition between alternatives

**Brain regions involved**:
- **Visual cortex (V1-V4)**: Represents currently dominant percept
- **Parietal cortex (LIP)**: Accumulates evidence for switch
- **Prefrontal cortex (dlPFC)**: Top-down biasing, working memory
- **Subcortical (SC, thalamus)**: Attention and selection

### Why Does Perception Alternate?

**Competing theories**:

**1. Adaptation/fatigue**:
- Neurons encoding current percept adapt (reduce response)
- Competing representation gains advantage
- **Evidence**: Prolonging one percept increases next dominance duration

**2. Noise accumulation**:
- Random neural fluctuations eventually tip balance
- Stochastic process with probabilistic transitions
- **Evidence**: Switch timing follows gamma distribution

**3. Exploratory sampling**:
- Brain actively samples alternative interpretations
- Information-seeking behavior (epistemic value)
- **Evidence**: Switches bring new information about stimulus

From [Parietal Theta Burst TMS Does Not Modulate Bistable Perception](https://academic.oup.com/nc/article/2024/1/niae009/7636050) (Schauer et al., 2024):
> "Our results suggest that continuous theta burst stimulation (cTBS) is particularly unreliable in modulating bistable perception when applied over parietal cortex."

**Bayesian account**: Perception samples from posterior distribution
- Multiple interpretations have similar posterior probability
- Brain explores hypothesis space
- Uncertainty drives exploration

## Section 5: Binocular Rivalry

### Definition and Phenomenology

**Binocular rivalry**: Present different images to each eye → perception alternates between them

**Setup**:
- Left eye: Horizontal grating
- Right eye: Vertical grating
- Percept: Alternates between horizontal and vertical (suppression of one eye)

**Not** binocular fusion:
- Unlike stereopsis (depth from disparity)
- Images too different to merge
- Winner-take-all competition

From [Cardiac Afferent Signals Can Facilitate Visual Dominance in Binocular Rivalry](https://pubmed.ncbi.nlm.nih.gov/39356552/) (Veillette et al., 2024):
> "We presented separate grating stimuli to each eye as in a classic binocular rivalry paradigm. Cardiac signals (heartbeat-evoked potentials) facilitated visual dominance of one eye over the other."

### Sites of Competition

**Where does rivalry occur?**

**Early visual cortex (V1)**:
- Monocular neurons (receive input from one eye)
- Suppressed eye shows reduced V1 activity
- **Evidence**: Rivalry can be retinotopically specific

**Higher visual areas (V4, IT)**:
- Binocular neurons integrate both eyes
- Stronger modulation by rivalry
- **Evidence**: Object-level rivalry (e.g., face vs house)

**Frontoparietal attention networks**:
- Correlate with perceptual transitions
- May trigger switches rather than represent content

From [Divergent Neural Mechanisms of Selective Attention in Bistable Perception](https://www.sciencedirect.com/science/article/pii/S1053811925004677) (2025):
> "Human bistable perception phenomena, such as dichotic listening (DL) and binocular rivalry (BR), provide ideal experimental paradigms for studying selective attention across sensory modalities."

**Hierarchical competition**:
- Low-level rivalry: Retinotopic features compete
- High-level rivalry: Objects/faces compete
- Both can occur simultaneously

### Interocular Suppression

**Continuous flash suppression** (CFS):
- One eye: Static target image
- Other eye: Rapid sequence of Mondrian patterns
- Target suppressed from awareness for extended periods (>1 second)

**Applications**:
- Study unconscious processing (suppressed stimuli still processed?)
- Measure "breaking suppression" (what reaches awareness?)
- Tool for studying consciousness

**Key findings**:
- High-level features (faces, words) can break suppression faster
- Emotional stimuli penetrate suppression
- Suggests unconscious semantic processing

## Section 6: Prediction Error and Perceptual Updating

### Predictive Coding Framework

**Core idea**: Perception minimizes prediction error across cortical hierarchy

**Bidirectional processing**:
- **Top-down**: Predictions flow from higher to lower areas
- **Bottom-up**: Prediction errors flow from lower to higher areas
- **Update**: Minimize mismatch

From [Perceptual Inference: A Matter of Predictions and Errors](https://www.cell.com/current-biology/fulltext/S0960-9822(16)30854-5) (Kok, 2016):
> "A recent study finds that separate populations of neurons in inferotemporal cortex code for perceptual predictions and prediction errors, providing strong support for predictive coding theories."

**Equations**:
```
Prediction: ŷ = f(higher-level representation)
Error: ε = y - ŷ
Update: Δ representation ∝ ε
```

### Neural Correlates of Prediction and Error

From [Distinguishing Neural Correlates of Prediction Errors on Perception](https://pubmed.ncbi.nlm.nih.gov/39785692/) (Dijkstra et al., 2025):
> "We develop a novel visual perception paradigm that probes such inferences by manipulating both expectations about stimulus content (stimulus predictions) and expectations about stimulus presence (temporal predictions)."

**Experimental dissociation**:
- **Expected stimulus**: Lower neural response (prediction correct, small error)
- **Unexpected stimulus**: Higher response (large prediction error)
- **Precision-weighting**: Errors weighted by confidence

**Prediction suppression**:
- Predicted stimuli evoke weaker responses (repetition suppression)
- Unpredicted stimuli evoke stronger responses
- **Interpretation**: Brain encodes deviations from predictions efficiently

**Mismatch negativity** (MMN):
- EEG component to unexpected auditory stimuli
- Automatic detection of violations
- Evidence for prediction error signaling

### High-Level Predictions in Low-Level Cortex

From [High-Level Visual Prediction Errors in Early Visual Cortex](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002829) (Richter et al., 2024):
> "Our results suggest that high-level predictions constrain sensory processing in earlier areas, thereby aiding perceptual inference. Predictions about object identity modulate responses in V1."

**Key finding**:
- Object-level predictions (e.g., expecting a face) modulate V1 activity
- Not just local feature predictions
- Top-down semantic information shapes early sensory processing

**Implications**:
- Early vision not purely bottom-up
- Perception is deeply context-dependent from earliest stages
- "Seeing" involves high-level knowledge immediately

## Section 7: Perceptual Confidence and Metacognition

### Confidence as Posterior Probability

**Bayesian view**: Confidence reflects certainty of perceptual inference

```
Confidence = P(hypothesis | data)
```

**Factors affecting confidence**:
- **Evidence strength**: Strong sensory signal → high confidence
- **Prior strength**: Strong prior → confident even with weak signal
- **Ambiguity**: Multiple hypotheses with similar probability → low confidence

**Measurement**:
- Confidence ratings (e.g., 1-6 scale)
- Wagering tasks (bet on perceptual judgment)
- Post-decision choice (opt out of difficult trials)

### Neural Basis of Confidence

**Decision variable** models:
- Confidence correlates with decision variable strength
- Larger margin between alternatives → higher confidence
- Implemented via accumulator models (drift-diffusion)

**Metacognitive regions**:
- **Prefrontal cortex**: Explicit confidence reports
- **Anterior cingulate**: Uncertainty monitoring
- **Parietal cortex**: Accumulation and confidence

From [Neural Prediction Errors Distinguish Perception and Misperception](https://www.jneurosci.org/content/38/27/6076) (Blank et al., 2018):
> "Our findings suggest that the strength of neural prediction error representations distinguishes correct perception and misperception. Larger errors predict incorrect perceptual decisions."

**Confidence errors**:
- **Overconfidence**: High confidence despite incorrect decision
- **Underconfidence**: Low confidence despite correct decision
- Individual differences in calibration

### Metacognitive Accuracy

**Can we accurately judge our own perception?**

**Type 1 performance**: Perceptual decision accuracy
**Type 2 performance**: Metacognitive accuracy (confidence calibration)

**Dissociation**:
- Type 1 can be high, Type 2 low (good perception, poor insight)
- Lesions can selectively impair metacognition

**Calibration curve**:
- Plot: Confidence vs accuracy
- Ideal: Perfect correspondence
- Typical: Overconfidence for hard tasks, underconfidence for easy

## Section 8: Computational Implementation (Files 2, 10, 14: Pipeline Parallelism, Kubeflow, Apple Metal)

### Distributed Hierarchical Inference

**File 2: [DeepSpeed Pipeline Parallelism](../distributed-training/01-deepspeed-pipeline-parallelism.md)** for hierarchical perceptual models:

**Hierarchical predictive coding**:
- Each layer predicts layer below
- Pipeline stages = cortical hierarchy levels
- V1 → V2 → V4 → IT → PFC

**Implementation**:
```python
# Pipeline parallel predictive coding
class HierarchicalPercept(nn.Module):
    def __init__(self):
        self.v1 = FeatureLayer()   # Stage 1
        self.v2 = ObjectLayer()    # Stage 2
        self.v4 = CategoryLayer()  # Stage 3

    def forward(self, image):
        # Forward predictions
        pred_v1 = self.v2.predict_v1()
        pred_v2 = self.v4.predict_v2()

        # Compute errors
        err_v1 = self.v1(image) - pred_v1
        err_v2 = self.v2(err_v1) - pred_v2

        # Update via backprop
        return loss(err_v1, err_v2)
```

**Pipeline efficiency**:
- Stages process different images simultaneously
- Mimics cortical streaming (feedforward + feedback)
- Memory-efficient for deep hierarchies

### Orchestrating Perceptual Experiments

**File 10: [Kubeflow ML Pipelines](../orchestration/01-kubeflow-ml-pipelines.md)** for large-scale psychophysics:

**Experimental pipeline**:
1. **Stimulus generation**: Create ambiguous images (Necker cubes, rival gratings)
2. **Model inference**: Bayesian observer model
3. **Human data collection**: Behavioral experiments (via web)
4. **Model fitting**: Estimate priors, likelihood functions
5. **Validation**: Compare human vs model predictions

**Kubeflow components**:
```yaml
# Binocular rivalry experiment pipeline
apiVersion: argoproj.io/v1alpha1
kind: Workflow
spec:
  templates:
  - name: rivalry-experiment
    steps:
    - - name: generate-stimuli
        template: create-gratings
    - - name: run-model
        template: bayesian-observer
    - - name: collect-human-data
        template: web-experiment
    - - name: fit-parameters
        template: likelihood-estimation
```

**Advantages**:
- Reproducible experiments
- Version control for stimuli and models
- Scalable to thousands of participants
- Automated analysis pipelines

### Real-Time Perceptual Inference on Edge Devices

**File 14: [Apple Metal ML](../alternative-hardware/01-apple-metal-ml.md)** for interactive demos:

**On-device Bayesian perception**:
- Run predictive coding models on M4 chip
- Real-time bistable perception demo
- Low latency for interactive psychophysics

**Metal implementation**:
```swift
// Real-time prediction error minimization
func perceptInference(image: MTLTexture) -> MTLTexture {
    // Top-down prediction
    let prediction = generativeModel.predict()

    // Compute prediction error (Metal shader)
    let error = predictionErrorShader.encode(
        input: image,
        prediction: prediction
    )

    // Update representation (gradient descent)
    let percept = updateShader.encode(
        current: representation,
        error: error,
        learningRate: 0.1
    )

    return percept
}
```

**Applications**:
- AR/VR perceptual illusions
- Interactive museum exhibits
- Educational neuroscience demos
- Low-power wearables (Apple Watch experiments)

## Section 9: ARR-COC-0-1 as Perceptual Inference System (10%)

### Visual Inference Through Compression

**ARR-COC-0-1 implements perceptual inference**:
- **Generative model**: Compression network predicts original image from compressed representation
- **Recognition model**: Encoder infers compressed representation from image
- **Prediction error**: Reconstruction loss (original - decoded)

**Bayesian interpretation**:
```python
# ARR-COC-0-1 as Bayesian perceiver
P(compressed_rep | image, query) ∝
    P(image | compressed_rep) × P(compressed_rep | query)
```

**Components**:
- **Likelihood P(image | compressed_rep)**: Decoder quality (reconstruction error)
- **Prior P(compressed_rep | query)**: Query-dependent relevance (participatory knowing)
- **Posterior P(compressed_rep | image, query)**: What to represent

### Illusions and Compression Artifacts

**Compression creates illusions**:
- Lossy compression = Prior-biased inference
- Low-relevance regions heavily compressed → lose detail
- High-relevance regions preserved → retain detail

**ARR-COC-0-1 "illusions"**:
- **Prior bias**: Learned compression patterns favor typical scenes
- **Unusual scenes**: Compressed incorrectly (like human illusions)
- **Query bias**: Participatory knowing creates expectation-driven distortions

**Example**:
- Query: "Find the cat"
- Cat-like textures boosted (high LOD)
- Cat-like noise in background over-represented
- Similar to pareidolia (seeing faces in clouds)

### Bistability in Ambiguous Queries

**Ambiguous query → multiple relevance interpretations**:

**Example**:
- Image: Dog and cat both present
- Query: "Find the animal"
- **Bistability**: Relevance alternates between dog and cat regions

**Implementation**:
```python
# Simulated bistable relevance
relevance_dog = participatory_scorer(patch_dog, query)
relevance_cat = participatory_scorer(patch_cat, query)

if abs(relevance_dog - relevance_cat) < threshold:
    # Ambiguous - implement stochastic switching
    if random.random() < switch_probability:
        allocate_tokens(patch_dog, 400)
        allocate_tokens(patch_cat, 64)
    else:
        allocate_tokens(patch_dog, 64)
        allocate_tokens(patch_cat, 400)
```

**Analogy to human bistability**:
- Both interpretations viable (similar posterior probability)
- Stochastic transitions driven by noise
- Attention (query strength) can bias dominance

### Prediction Error Minimization via Adaptive Compression

**ARR-COC-0-1 training = Learn to minimize prediction error**:

**Predictive coding analogy**:
- **Top-down prediction**: Decoder predicts image from compressed tokens
- **Bottom-up error**: Reconstruction loss (actual - predicted image)
- **Update**: Encoder learns to minimize error via better compression

**Adaptive LOD**:
- High prediction error regions → increase LOD (400 tokens)
- Low prediction error regions → decrease LOD (64 tokens)
- Dynamic allocation based on reconstruction quality

**Quality adapter as metacognition**:
- Learns to predict which compression strategies work
- Confidence = Expected reconstruction quality
- High confidence → aggressive compression
- Low confidence → conservative compression (more tokens)

**Future enhancement**:
```python
# Uncertainty-driven token allocation
class UncertaintyAllocator:
    def allocate(self, patch, query):
        # Epistemic uncertainty (model uncertainty)
        epistemic = model_variance(patch)

        # Aleatoric uncertainty (data noise)
        aleatoric = prediction_entropy(patch)

        # High uncertainty → more tokens (explore)
        if epistemic > threshold:
            return 400  # Need more data to resolve
        else:
            return 64   # Confident in compression
```

## Sources

### Source Documents

**Existing Knowledge**:
- [cognitive-foundations/03-attention-resource-allocation.md](../cognitive-foundations/03-attention-resource-allocation.md) - Attention bottleneck, biased competition, resource allocation
- [cognitive-foundations/00-active-inference-free-energy.md](../cognitive-foundations/00-active-inference-free-energy.md) - Free energy principle, precision-weighting, Bayesian brain
- [cognitive-foundations/02-bayesian-brain-probabilistic.md](../cognitive-foundations/02-bayesian-brain-probabilistic.md) - Bayesian inference, hierarchical models

**Influential Files**:
- [distributed-training/01-deepspeed-pipeline-parallelism.md](../distributed-training/01-deepspeed-pipeline-parallelism.md) - File 2: Pipeline parallel hierarchical models
- [orchestration/01-kubeflow-ml-pipelines.md](../orchestration/01-kubeflow-ml-pipelines.md) - File 10: ML experiment orchestration
- [alternative-hardware/01-apple-metal-ml.md](../alternative-hardware/01-apple-metal-ml.md) - File 14: Real-time inference on Apple Silicon

### Web Research

**Bayesian Perception & Illusions**:
- Weiss, Y. et al. (2002). "[Illusions, Perception and Bayes](https://www.researchgate.net/publication/11334330_Illusions_perception_and_Bayes)." ResearchGate. (accessed 2025-11-16) - Visual illusions as optimal inference
- Yang, S. et al. (2021). "[Human Visual Motion Perception Shows Hallmarks of Bayesian Inference](https://www.nature.com/articles/s41598-021-82175-7)." Nature Scientific Reports. (accessed 2025-11-16) - Bayesian ideal observer validation
- Galen (2024). "[Perception as Visual Inference](https://www.dissonances.blog/p/perception-as-visual-inference)." Dissonances Blog. (accessed 2025-11-16) - Accessible explanation of Bayesian perception
- Calder-Travis, J. et al. (2024). "[Bayesian Confidence in Optimal Decisions](https://psycnet.apa.org/record/2025-04700-001)." APA PsycNet. (accessed 2025-11-16) - Confidence as evidence strength

**Bistable Perception & Binocular Rivalry**:
- Nie, S. et al. (2023). "[An Accumulating Neural Signal Underlying Binocular Rivalry](https://www.jneurosci.org/content/43/50/8777)." Journal of Neuroscience, 43(50). (accessed 2025-11-16) - LIP accumulation during rivalry
- Veillette, J.P. et al. (2024). "[Cardiac Afferent Signals Can Facilitate Visual Dominance in Binocular Rivalry](https://pubmed.ncbi.nlm.nih.gov/39356552/)." NIH PubMed. (accessed 2025-11-16) - Heartbeat effects on perception
- Schauer, G. et al. (2024). "[Parietal Theta Burst TMS Does Not Modulate Bistable Perception](https://academic.oup.com/nc/article/2024/1/niae009/7636050)." Oxford Academic. (accessed 2025-11-16) - TMS and bistability
- (2025). "[Divergent Neural Mechanisms of Selective Attention in Bistable Perception](https://www.sciencedirect.com/science/article/pii/S1053811925004677)." ScienceDirect. (accessed 2025-11-16) - Dichotic listening vs binocular rivalry

**Prediction Error & Perceptual Inference**:
- Kok, P. (2016). "[Perceptual Inference: A Matter of Predictions and Errors](https://www.cell.com/current-biology/fulltext/S0960-9822(16)30854-5)." Cell Press Current Biology. (accessed 2025-11-16) - Separate prediction and error neurons
- Dijkstra, N. et al. (2025). "[Distinguishing Neural Correlates of Prediction Errors on Perception](https://pubmed.ncbi.nlm.nih.gov/39785692/)." NIH PubMed. (accessed 2025-11-16) - Stimulus vs temporal predictions
- Richter, D. et al. (2024). "[High-Level Visual Prediction Errors in Early Visual Cortex](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002829)." PLOS Biology. (accessed 2025-11-16) - Object predictions modulate V1
- Blank, H. et al. (2018). "[Neural Prediction Errors Distinguish Perception and Misperception](https://www.jneurosci.org/content/38/27/6076)." Journal of Neuroscience, 38(27). (accessed 2025-11-16) - Error strength predicts accuracy

**Helmholtz & Unconscious Inference**:
- Hatfield, G. (2002). "[Perception as Unconscious Inference](https://philpapers.org/rec/HATPAU)." PhilPapers. (accessed 2025-11-16) - Historical and philosophical analysis
- Kiefer, A. (2017). "[Literal Perceptual Inference](https://philarchive.org/rec/KIELPI)." PhilArchive. (accessed 2025-11-16) - Taking inference literally
- [Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/archives/fall2024/entries/hermann-helmholtz/) (2024). "Hermann von Helmholtz." (accessed 2025-11-16) - Comprehensive overview
- Jimenez, M. et al. (2025). "[Advancing the Study of Unconscious Perception](https://pmc.ncbi.nlm.nih.gov/articles/PMC12405385/)." NIH PMC. (accessed 2025-11-16) - Modern predictive coding roots

### Additional References

- Summerfield, C. et al. (2008). "Expectation in Perceptual Decision Making." *Neuron*
- Hohwy, J. (2016). "Bayesian Learning Rate, Prediction Error, and Perceptual Inference." *NeuroImage*
- Davis, M. et al. "Prediction Error for Bayesian Inference in Speech Perception."
- Vincent, B.T. (2015). "A Tutorial on Bayesian Models of Perception." *ScienceDirect*

---

**Total Lines**: ~700
**Knowledge Type**: ACQUISITION (Web research + existing knowledge integration)
**ARR-COC-0-1 Integration**: Section 9 (10%) explicitly connects perceptual inference to compression
**Citations**: Comprehensive web sources (20+ papers) + internal oracle knowledge + pipeline/orchestration files
