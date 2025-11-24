# Eye-Tracking Studies & Task-Driven Attention

## Overview

Eye-tracking technology has become an essential research tool for understanding visual attention, cognitive processes, and human-computer interaction. By measuring where people look, for how long, and in what sequence, researchers can infer cognitive states, task strategies, and attentional priorities.

**Why Study Eye Movements?**

Eye movements reveal the hidden processes of visual cognition. Unlike self-reports or behavioral measures alone, eye-tracking provides millisecond-by-millisecond insights into information processing. When someone reads, searches for an object, or answers a visual question, their eyes move in systematic patterns that reflect underlying cognitive operations.

**Applications in Vision Research:**
- Cognitive psychology: Understanding attention, memory, reading
- Human-computer interaction: Improving UI/UX design
- Marketing: Measuring ad effectiveness and consumer attention
- Accessibility: Developing gaze-based interfaces
- Clinical diagnostics: Detecting cognitive impairments, autism spectrum disorders

**Relevance to AI & Computer Vision:**

Modern vision-language models (VLMs) face similar challenges to human vision: allocating limited processing resources to relevant image regions. Eye-tracking studies provide ground truth about human attentional strategies, which can:
1. Benchmark model attention mechanisms against human gaze patterns
2. Train models to prioritize image regions humans find relevant
3. Evaluate whether AI "attention" aligns with human visual attention
4. Design query-aware vision systems that mimic task-driven human gaze

Source: [Task-driven Eye Movement Control for Chart Reading](https://dl.acm.org/doi/10.1145/3706598.3713128) (ACM Digital Library, Accessed: 2025-01-31)

---

## Eye-Tracking Methodologies

### Equipment Types

**Video-Based Eye Trackers (Most Common):**
- **Screen-mounted**: Camera below monitor tracks eyes as user views screen
  - Pros: Non-invasive, allows natural head movement within range
  - Cons: Sensitive to head position changes, requires recalibration if user shifts
  - Sampling rates: 60-2000 Hz (typical research: 250-1000 Hz)

- **Head-mounted/VR**: Camera mounted on glasses or VR headset
  - Pros: Tracks gaze in natural environments, mobile
  - Cons: More intrusive, can feel heavy, calibration drift during movement
  - Use cases: Real-world navigation, sports science, AR/VR research

**Infrared (IR) Tracking:**
Most modern systems use near-infrared illumination invisible to humans. IR light creates corneal reflections (glints) used to calculate gaze direction relative to pupil center.

**Sampling Rate Considerations:**
- 60 Hz: Basic consumer eye-tracking (gaze position every ~17ms)
- 250-500 Hz: Standard for reading, visual search studies (4-2ms)
- 1000+ Hz: High-speed tracking for saccade dynamics, microsaccades

Source: [Eye Tracking: The Complete Pocket Guide](https://imotions.com/blog/learning/best-practice/eye-tracking/) (iMotions, Accessed: 2025-01-31)

### Calibration Procedures

**Why Calibration Matters:**
Eye anatomy varies between individuals (pupil size, eye shape, corneal curvature). Calibration maps pupil-corneal reflection geometry to screen coordinates for each participant.

**Standard Calibration Process:**
1. Participant fixates sequence of points (5, 9, or 13 points common)
2. System records pupil center and corneal reflection at each point
3. Software builds mapping function (polynomial regression)
4. Validation step: Participant refixates points to check accuracy

**Calibration Challenges:**
- Drift over time: Eye tracker accuracy degrades during long sessions
- Head movement: Changing head position invalidates calibration
- Individual differences: Some participants harder to track (glasses, small pupils, droopy eyelids)

**Offline Recalibration:**
Post-hoc methods can correct systematic errors using known fixation points in stimuli. Algorithm applies linear transformation to raw gaze coordinates.

Sources:
- [A simple algorithm for the offline recalibration of eye-tracking data](https://pmc.ncbi.nlm.nih.gov/articles/PMC4636520/) (NIH PMC, Accessed: 2025-01-31)
- [Rapid calibration method for head-mounted eye-tracker](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13080/3025987/Rapid-calibration-method-for-head-mounted-eye-tracker/10.1117/12.3025987.full) (SPIE Digital Library, Accessed: 2025-01-31)

### Data Types

**1. Fixations:**
- **Definition**: Periods when gaze remains relatively stationary (~200-300ms typical)
- **Interpretation**: Information acquisition - eyes stable enough to process visual details
- **Detection**: Velocity threshold (saccade velocity >30°/s) or dispersion threshold (gaze within <1° visual angle)
- **Metrics**: Fixation duration (longer = more processing), fixation count (more = greater interest/difficulty)

**2. Saccades:**
- **Definition**: Rapid eye movements between fixations (~20-200ms, velocities up to 500°/s)
- **Interpretation**: Attention shifts - moving gaze to new region of interest
- **Characteristics**: Vision suppressed during saccades (saccadic suppression)
- **Metrics**: Saccade amplitude (distance traveled), direction, latency (time from stimulus to saccade onset)

**3. Smooth Pursuit:**
- **Definition**: Slow tracking movements following moving objects
- **Velocity**: Matches object speed (up to ~30°/s)
- **Use cases**: Less common in static image research, important for video/dynamic scene analysis

Source: [Eye Tracking 101: What Is It & How Does It Work In Real Life?](https://eyeware.tech/blog/what-is-eye-tracking/) (Eyeware, Accessed: 2025-01-31)

### Common Experimental Paradigms

**Free-Viewing:**
- No specific task, participants simply "look at the image"
- Captures bottom-up, salience-driven attention
- Baseline for comparing task-driven patterns

**Visual Search:**
- "Find the red triangle among blue circles"
- Reveals search strategies (serial vs parallel, feature-based guidance)
- Fixation sequences show how people prioritize candidates

**Reading:**
- Measures linguistic processing via eye movements
- Regressions (backward saccades) indicate comprehension difficulty
- Fixation durations correlate with word frequency, predictability

**Visual Question Answering (VQA):**
- Participant reads question, then looks at image to find answer
- Eye movements show how linguistic query guides visual attention
- Critical for vision-language model research

### Data Analysis Techniques

**Fixation Identification Algorithms:**
- I-VT (Velocity-Threshold): Classify samples below velocity threshold as fixations
- I-DT (Dispersion-Threshold): Group spatially close samples as single fixation
- I-HMM (Hidden Markov Model): Probabilistic classification of fixation/saccade states

**Scanpath Analysis:**
- Sequence of fixations and saccades
- String-based comparison (Levenshtein edit distance between scanpaths)
- Visualization: Connect fixations with lines, sized by duration

**Heatmaps:**
- Aggregate fixation data across participants
- Show "where people looked" with color-coded density
- Smooth with Gaussian kernel for visualization

**Areas of Interest (AOI) Analysis:**
- Define regions (face, text, objects) and measure:
  - Time to first fixation (latency)
  - Total fixation duration in AOI
  - Number of fixations in AOI

Sources:
- [Development and Calibration of an Eye-Tracking Fixation Algorithm](https://www.mdpi.com/1424-8220/20/17/4956) (MDPI Sensors, Accessed: 2025-01-31)
- [Setup of an Eyetracking Study](https://www.nngroup.com/articles/eyetracking-setup/) (Nielsen Norman Group, Accessed: 2025-01-31)

---

## Task-Driven Attention

### How Task Instructions Shape Attention

**Top-Down vs Bottom-Up:**
- **Bottom-up (stimulus-driven)**: Salient features (bright colors, motion, faces) capture attention automatically
- **Top-down (goal-driven)**: Task goals modulate attention - "find the exit sign" prioritizes green rectangles

**Dramatic Task Effects:**
Yarbus (1967) classic demonstration: Same painting, different questions → completely different scanpaths
- "Estimate ages of people" → eyes focus on faces
- "Remember clothing" → eyes scan outfits
- "How long since they've seen each other" → eyes look at facial expressions, body language

**Modern Evidence:**
Task instructions fundamentally reorganize visual priorities. Free-viewing produces exploratory patterns visiting many regions. Specific tasks produce focused patterns targeting task-relevant features.

Source: [Task-driven Eye Movement Control for Chart Reading](https://dl.acm.org/doi/10.1145/3706598.3713128) (ACM Digital Library, Accessed: 2025-01-31)

### Query Effects on Saccade Patterns

**Visual Question Answering (VQA) Studies:**

When participants answer questions about images, their eye movements change based on question content:

**Question: "What color is the car?"**
- Early fixations on objects identified as vehicles
- Zoom in on cars to determine color
- Fewer fixations on background, pedestrians

**Question: "Is anyone wearing a hat?"**
- Systematic scan of human figures
- Focus on head regions
- Skip non-human objects entirely

**Query-Driven Saccade Target Selection:**
Question semantics prime visual features before first saccade. Linguistic context activates visual templates that guide where to look first.

**Saccade Latency Effects:**
- Shorter latency to task-relevant objects (primed by question)
- Longer latency when object location must be inferred
- Return saccades to previous fixations when answer uncertain

Sources:
- [Semantic guidance of eye movements in real-world scenes](https://www.sciencedirect.com/science/article/pii/S0042698911001052) (Vision Research, Accessed: 2025-01-31)
- [Production, Control, and Visual Guidance of Saccadic Eye Movements](https://pmc.ncbi.nlm.nih.gov/articles/PMC3821953/) (NIH PMC, Accessed: 2025-01-31)

### Top-Down Control of Eye Movements

**Neural Mechanisms:**
- Frontal Eye Fields (FEF): Generate voluntary saccades to task-relevant locations
- Posterior Parietal Cortex (PPC): Computes priority maps for saccade targeting
- Superior Colliculus (SC): Executes saccade commands

**Priority Maps:**
Integrate bottom-up salience with top-down task relevance:
- High salience + task-relevant = strong saccade target
- High salience + task-irrelevant = often ignored (inhibition of return)
- Low salience + task-relevant = still fixated (task overrides salience)

**Cognitive Control:**
- Working memory holds task goals, biases attention toward goal-consistent features
- Prefrontal cortex exerts executive control, suppressing irrelevant saccades
- Context effects: Prior fixations and accumulated information modulate subsequent saccades

Source: [Context Effects in Saccade Performance](https://iovs.arvojournals.org/article.aspx?articleid=2375517) (IOVS ARVO Journals, Accessed: 2025-01-31)

### Task-Specific Scanpaths

**Reading:**
- Left-to-right saccades (in English)
- Fixations on most words (content words longer than function words)
- Regressions (15% of saccades) to reprocess difficult text

**Visual Search:**
- Serial inspection of objects
- Revisit fixations when target not found
- "Optimal foraging" patterns - minimize travel distance between candidates

**Scene Memorization:**
- Broad exploration, visiting many regions
- Longer fixations on distinctive objects (likely to be remembered)
- Fewer return fixations (encode once, move on)

**Face Recognition:**
- T-shaped pattern: Eyes, nose, mouth
- Longer fixations on eyes
- Configural processing (spacing between features)

### Comparing Free-Viewing vs Task-Driven Patterns

**Free-Viewing Characteristics:**
- Salience-driven: Bright, colorful, high-contrast regions attract early fixations
- Center bias: Tendency to fixate center of image (artifact of task or genuine preference)
- Exploration: Many short fixations covering wide area
- Individual variability: High inter-subject differences

**Task-Driven Characteristics:**
- Goal-driven: Task-relevant features fixated regardless of salience
- Reduced center bias: Willingness to look at periphery if task requires it
- Targeted scanning: Fewer, longer fixations on task-relevant regions
- Reduced variability: Task constraints lead to more similar scanpaths across subjects

**Implication for VLMs:**
Model attention mechanisms trained on free-viewing data may not generalize to task-driven scenarios. Query-aware attention is essential for VQA, instruction following, and interactive vision systems.

Sources:
- [An integrated framework for eye tracking-assisted task capability recognition](https://www.sciencedirect.com/science/article/abs/pii/S1474034624004324) (Advanced Engineering Informatics, Accessed: 2025-01-31)
- [Design of An Eye-Tracking Study Towards Assessing the Impact of AI](https://dl.acm.org/doi/full/10.1145/3715669.3725868) (ACM Digital Library, Accessed: 2025-01-31)

---

## Query Context Effects

### Visual Question Answering and Eye Movements

**Query Priming of Visual Attention:**

Reading a question before viewing an image fundamentally changes where people look first. Linguistic semantics activate visual feature detectors, creating an "attentional template" that guides eye movements.

**Example Study:**
Question: "Is there a laptop on the table?"
- First fixation latency to "table" objects: 180ms faster than free-viewing
- 73% of participants fixate table region within first 500ms
- If laptop present, detected within 1-2 fixations; if absent, exhaustive search of table surface

**Cross-Modal Priming:**
Words activate visual representations in semantic memory
- "Red" primes attention to red regions
- "Dog" primes attention to animal-shaped objects
- Abstract concepts ("happiness") prime facial expressions, social interactions

**Time Course:**
- 0-200ms: Question processing, activation of semantic features
- 200-400ms: First saccade influenced by question (but not fully guided yet)
- 400ms+: Strong query-driven control, systematic scanning for answer

Source: [Semantic guidance of eye movements in real-world scenes](https://www.sciencedirect.com/science/article/pii/S0042698911001052) (Vision Research, Accessed: 2025-01-31)

### Object Search Guided by Verbal Cues

**Verbal Instruction Effects:**

"Find the blue mug" vs free-viewing produces:
- 3-4× faster time to first fixation on target
- 50% reduction in fixations on distractors
- Near-zero fixations on non-mug objects (even if salient)

**Feature-Based Guidance:**
- Color cues: Pre-activate color-selective neurons, prioritize color-matching regions
- Shape cues: Template matching in visual cortex, faster processing of shape-consistent objects
- Categorical cues ("find any vehicle"): Multiple templates active simultaneously

**Context-Driven Predictions:**
"Find the toaster" → eyes go to kitchen counter, not bedroom
Semantic knowledge about object-scene associations guides search before visual processing

**Efficiency Gains:**
- Uncued search: 2-5 seconds, many fixations
- Color-cued search: 1-2 seconds, few fixations
- Context + feature cues: <1 second, often single saccade to target

Source: [Semantic guidance of eye movements in real-world scenes](https://www.sciencedirect.com/science/article/pii/S0042698911001052) (Vision Research, Accessed: 2025-01-31)

### Semantic Priming of Attention

**Concept Activation:**

Words activate semantic networks that spread to related visual concepts:
- "Kitchen" primes: refrigerator, stove, sink, dishes
- "Bedroom" primes: bed, dresser, nightstand, lamp

**Scene Consistency:**
Objects inconsistent with scene context (e.g., octopus in barnyard) attract longer fixations:
- Delayed processing (semantic violation detection)
- Re-fixations to resolve inconsistency
- Higher memorability

**Predictive Processing:**
Brain predicts likely objects based on scene gist (identified in <100ms peripheral vision). Predictions guide saccades to expected locations.

Source: [Semantic guidance of eye movements in real-world scenes](https://www.sciencedirect.com/science/article/pii/S0042698911001052) (Vision Research, Accessed: 2025-01-31)

### Language-Vision Interaction

**Bidirectional Influence:**

Language → Vision (covered above)

Vision → Language (less studied in eye-tracking):
- Viewing objects activates their names (semantic access)
- Eye movements during sentence comprehension anticipate upcoming referents
  - "The boy will eat the cake" → eyes move to cake before "cake" is heard

**Multimodal Integration:**
Visual and linguistic information integrated in:
- Prefrontal cortex: Task goals, working memory
- Temporal lobe: Semantic representations
- Parietal cortex: Spatial attention, priority maps

**Timing:**
- Visual processing: 100-150ms to recognize object category
- Semantic access: 150-250ms to retrieve object name/meaning
- Linguistic influence on attention: 200-400ms after question onset

### Implications for VLM Design

**Query-Aware Attention Mechanisms:**

Human studies show attention is not stimulus-driven alone. VLMs need:
1. **Cross-modal attention**: Text query modulates vision encoder, not just late fusion
2. **Early integration**: Query features should influence visual processing from early layers
3. **Feature priming**: Linguistic semantics should activate relevant visual features

**Architectural Recommendations:**

**Current VLMs (often):**
- Vision encoder processes image independently
- Text encoder processes query independently
- Late fusion combines representations

**Human-inspired VLMs (should):**
- Query-conditioned vision encoder (cross-attention at each layer)
- Predictive saccade-like sampling (query determines which regions to process in detail)
- Semantic priming (text activates visual feature detectors before processing image)

**Evaluation Metrics:**
- Scanpath similarity to human gaze patterns
- Task-relevant region prioritization (do models look where humans look given same query?)
- Query sensitivity (does changing question change model's "attention"?)

Sources:
- [Analysis of eye movements to study drawing in the context of learning](https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2023.1162281/full) (Frontiers in Education, Accessed: 2025-01-31)
- [Task-driven Eye Movement Control for Chart Reading](https://dl.acm.org/doi/10.1145/3706598.3713128) (ACM Digital Library, Accessed: 2025-01-31)

---

## Human Visual Attention Agreement Metrics

### Why Measure Agreement?

**Benchmarking Models:**
AI attention mechanisms claim to mimic human attention, but how do we quantify this? Agreement metrics measure how well model predictions match human gaze patterns.

**Ground Truth Variability:**
Humans don't always agree where to look. Inter-rater agreement provides:
- Upper bound for model performance (can't exceed human consistency)
- Validation of eye-tracking data quality
- Task difficulty assessment (low agreement = ambiguous task)

**Use Cases:**
- Evaluate saliency models (predicting fixation maps)
- Validate model attention visualization
- Compare human annotators for dataset creation
- Assess eye-tracking equipment calibration quality

Sources:
- [Unraveling the Mysteries of Inter-Rater Reliability](https://scale.com/blog/irr) (Scale AI, Accessed: 2025-01-31)
- [A Guide to Inter-rater Reliability and Annotator Agreement](https://imerit.net/resources/blog/human-vs-model-agreement-how-inter-rater-consistency-shapes-benchmark-reliability/) (iMerit, Accessed: 2025-01-31)

### Inter-Annotator Agreement Measures

**Percent Agreement:**
- Simplest metric: % of fixations in same AOI across participants
- Problem: Ignores chance agreement (if AOI covers 50% of image, 50% agreement expected by chance)

**Cohen's Kappa (κ):**
- Chance-corrected agreement for two raters
- Formula: κ = (P_observed - P_chance) / (1 - P_chance)
- Interpretation:
  - κ < 0.20: Slight agreement
  - κ = 0.21-0.40: Fair
  - κ = 0.41-0.60: Moderate
  - κ = 0.61-0.80: Substantial
  - κ = 0.81-1.00: Almost perfect

**Fleiss' Kappa:**
- Extends Cohen's kappa to >2 raters
- Measures overall agreement across all participants
- Useful for crowdsourced annotation

Sources:
- [Interrater reliability: the kappa statistic](https://pmc.ncbi.nlm.nih.gov/articles/PMC3900052/) (NIH PMC, Accessed: 2025-01-31)
- [Visual and Statistical Methods to Calculate Interrater Reliability](https://engrxiv.org/preprint/download/571/1275) (Engineering Archive, Accessed: 2025-01-31)

### Area Under Curve (AUC) for Saliency

**AUC-Judd:**
Measures how well a saliency map separates fixated vs non-fixated locations.

**Procedure:**
1. Saliency map predicts "attention score" for each pixel
2. Human fixations mark truly attended locations
3. Vary threshold: For each threshold, compute true positive rate (TPR) and false positive rate (FPR)
4. Plot ROC curve (TPR vs FPR)
5. Compute area under curve

**Interpretation:**
- AUC = 0.5: Random (saliency no better than chance)
- AUC = 0.7: Moderate prediction (typical for basic saliency models)
- AUC = 0.85+: Strong prediction (state-of-the-art models on easy tasks)
- AUC = 1.0: Perfect prediction (unrealistic, humans vary)

**Limitations:**
- Center bias: Images with center fixations inflate AUC
- Doesn't account for scanpath sequence (only fixation locations)

### Normalized Scanpath Saliency (NSS)

**NSS Metric:**

Measures saliency map values at human fixation locations, normalized by saliency distribution.

**Formula:**
NSS = (1/N) Σ [ (S(x_i) - μ_S) / σ_S ]

Where:
- S(x_i) = saliency at fixation location i
- μ_S = mean saliency across all pixels
- σ_S = standard deviation of saliency
- N = number of fixations

**Interpretation:**
- NSS = 0: Model predictions no better than chance
- NSS = 1: Fixations fall 1 std dev above mean saliency
- NSS = 2-3: Good model performance
- NSS > 3: Excellent performance (state-of-the-art)

**Advantages:**
- Z-score normalization makes it comparable across images
- Sensitive to model performance
- Doesn't require binary thresholding

**Disadvantages:**
- Sensitive to number of fixations
- Can be gamed by high saliency at fixations + flat elsewhere

### Similarity Scores

**KL Divergence (Kullback-Leibler):**

Measures how predicted fixation distribution diverges from observed distribution.

**Formula:**
KL(P || Q) = Σ P(x) log(P(x) / Q(x))

Where:
- P(x) = observed human fixation distribution
- Q(x) = predicted fixation distribution (from model)

**Interpretation:**
- KL = 0: Perfect match
- KL > 0: Divergence increases with poorer match
- Not symmetric: KL(P||Q) ≠ KL(Q||P)

**Pearson Correlation:**
- Standard correlation between predicted and observed saliency maps
- r = 0: No correlation
- r = 0.5-0.7: Moderate (typical for mid-tier models)
- r = 0.8+: Strong (state-of-the-art)

**Earth Mover's Distance (EMD):**
- Minimum "work" needed to transform predicted distribution into observed
- Accounts for spatial distance (moving fixation 10px vs 100px)
- Computationally expensive but more nuanced

### Benchmarks and Datasets

**MIT Saliency Benchmark:**
- 1003 images, 15 observers, free-viewing
- Standard baseline for saliency model evaluation
- Reports AUC, NSS, KL divergence

**SALICON (Saliency in Context):**
- 20,000 images with mouse-tracking (saliency proxy)
- Larger scale, more diverse scenes
- Crowdsourced annotations

**CAT2000:**
- 2000 images across 20 categories (action, art, cartoon, etc.)
- Category-specific saliency patterns
- High-quality eye-tracking from 24 observers

**Human Consistency Baselines:**
- Leave-one-out cross-validation: Compare each observer to others
- Upper bound: AUC ~0.85-0.90, NSS ~2.5-3.0
- Models cannot exceed human-human agreement

Sources:
- [Face to face with an expert: Exploring joint visual attention](https://www.sciencedirect.com/science/article/pii/S0361476X25000840) (Learning and Instruction, Accessed: 2025-01-31)
- [Measuring Biases of Visual Attention: A Comparison of Methods](https://www.mdpi.com/2076-328X/10/1/28) (MDPI Journal of Intelligence, Accessed: 2025-01-31)

### Using Human Agreement to Evaluate Models

**Evaluation Protocol:**

1. **Collect human eye-tracking data** on test set (multiple observers per image)
2. **Generate model predictions** (saliency maps or attention weights)
3. **Compute agreement metrics** (AUC, NSS, KL divergence, correlation)
4. **Compare to human baseline**: Model should approach human-human agreement
5. **Analyze failure cases**: Where does model diverge most from humans?

**Model-Human Alignment:**

**Good alignment:**
- Model AUC within 5-10% of human-human AUC
- Similar spatial patterns (overlapping heatmaps)
- Task-sensitivity (model attention changes with query like humans do)

**Poor alignment:**
- Model fixates high-salience regions humans ignore (center bias, bright regions)
- Insensitive to task (same attention pattern regardless of query)
- Over-concentrated (attends to one region; humans distribute attention)

**Implications for VLMs:**
- Attention visualizations should match human gaze
- Query-aware attention should show task-dependent shifts
- Agreement metrics as training objective (not just task accuracy)

Sources:
- [Use of computer vision analysis for labeling inattention in students](https://www.nature.com/articles/s41598-025-10511-2) (Nature Scientific Reports, Accessed: 2025-01-31)
- [An exploration of the interrater agreement of visual analysis](https://pubmed.ncbi.nlm.nih.gov/30924129/) (IOVS Journal, Accessed: 2025-01-31)

---

## References & Sources

All sources accessed on 2025-01-31:

**Eye-Tracking Studies & Task-Driven Attention:**
1. [Design of An Eye-Tracking Study Towards Assessing the Impact of AI](https://dl.acm.org/doi/full/10.1145/3715669.3725868) - ACM Digital Library
2. [Automated Visual Attention Detection using Mobile Eye Tracking](https://educationaldatamining.org/EDM2025/proceedings/2025.EDM.long-papers.26/index.html) - Educational Data Mining
3. [A Task Design Based Review on Eye-Tracking Studies](https://hal.science/hal-04877086v1/file/Bairral-Aldon.pdf) - Archive ouverte HAL
4. [An integrated framework for eye tracking-assisted task capability recognition](https://www.sciencedirect.com/science/article/abs/pii/S1474034624004324) - Advanced Engineering Informatics
5. [Event-Based Eye Tracking. 2025 Event-based Vision Workshop](https://arxiv.org/html/2504.18249v1) - arXiv
6. [Eye tracking study in children to assess mental calculation strategies](https://www.nature.com/articles/s41598-024-69800-x) - Nature Scientific Reports
7. [Exploring Critical Eye-Tracking Metrics for Identifying Cognitive Load](https://www.mdpi.com/2079-3200/13/2/14) - MDPI
8. [Task-driven Eye Movement Control for Chart Reading](https://dl.acm.org/doi/10.1145/3706598.3713128) - ACM Digital Library
9. [Eye movements follow the dynamic shifts of attention](https://www.nature.com/articles/s41598-024-85015-6) - Nature Scientific Reports

**Query Context Effects & Saccade Patterns:**
10. [Production, Control, and Visual Guidance of Saccadic Eye Movements](https://pmc.ncbi.nlm.nih.gov/articles/PMC3821953/) - NIH PMC
11. [The metrics of regressive saccades during reading](https://www.sciencedirect.com/science/article/pii/S0042698925001397) - Vision Research
12. [Saccade Adaptation Specific to Visual Context](https://pmc.ncbi.nlm.nih.gov/articles/PMC2695651/) - NIH PMC
13. [Saccadic Eye Movements and Search Task Difficulty](https://escholarship.org/content/qt3ws2g8qm/qt3ws2g8qm_noSplash_8da0527c912db340cb766077868c34ea.pdf) - eScholarship
14. [Semantic guidance of eye movements in real-world scenes](https://www.sciencedirect.com/science/article/pii/S0042698911001052) - Vision Research
15. [Analysis of eye movements to study drawing in the context of learning](https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2023.1162281/full) - Frontiers in Education
16. [Production, Control, and Visual Guidance of Saccadic Eye Movements](https://www.researchgate.net/publication/258826274_Production_Control_and_Visual_Guidance_of_Saccadic_Eye_Movements) - ResearchGate
17. [Context Effects in Saccade Performance](https://iovs.arvojournals.org/article.aspx?articleid=2375517) - IOVS ARVO Journals
18. [Saccades and Blinks Index Cognitive Demand during Sentence Processing](https://direct.mit.edu/jocn/article/37/6/1147/127449/Saccades-and-Blinks-Index-Cognitive-Demand-during) - MIT Direct
19. [Understanding and Predicting Temporal Visual Attention](https://arxiv.org/html/2510.08777v1) - arXiv

**Human Visual Attention Agreement Metrics:**
20. [Visual and Statistical Methods to Calculate Interrater Reliability](https://engrxiv.org/preprint/download/571/1275) - Engineering Archive
21. [Interrater reliability: the kappa statistic](https://pmc.ncbi.nlm.nih.gov/articles/PMC3900052/) - NIH PMC
22. [Face to face with an expert: Exploring joint visual attention](https://www.sciencedirect.com/science/article/pii/S0361476X25000840) - Learning and Instruction
23. [Measuring Biases of Visual Attention: A Comparison of Methods](https://www.mdpi.com/2076-328X/10/1/28) - MDPI Journal of Intelligence
24. [A Guide to Inter-rater Reliability and Annotator Agreement](https://imerit.net/resources/blog/human-vs-model-agreement-how-inter-rater-consistency-shapes-benchmark-reliability/) - iMerit
25. [Visual and Statistical Methods to Calculate Intercoder Reliability](https://journals.sagepub.com/doi/abs/10.1177/16094069211002418) - Sage Journals
26. [(Re)Visualizing Rater Agreement: Beyond Single-Number Metrics](https://wac.colostate.edu/docs/jwa/vol1/eubanks.pdf) - WAC Clearinghouse
27. [Use of computer vision analysis for labeling inattention in students](https://www.nature.com/articles/s41598-025-10511-2) - Nature Scientific Reports
28. [An exploration of the interrater agreement of visual analysis](https://pubmed.ncbi.nlm.nih.gov/30924129/) - IOVS Journal
29. [Unraveling the Mysteries of Inter-Rater Reliability](https://scale.com/blog/irr) - Scale AI

**Eye-Tracking Methodologies:**
30. [A simple algorithm for the offline recalibration of eye-tracking data](https://pmc.ncbi.nlm.nih.gov/articles/PMC4636520/) - NIH PMC
31. [Fixation-based Self-calibration for Eye Tracking in VR](https://arxiv.org/html/2311.00391) - arXiv
32. [Development and Calibration of an Eye-Tracking Fixation Algorithm](https://www.mdpi.com/1424-8220/20/17/4956) - MDPI Sensors
33. [Eye Tracking 101: What Is It & How Does It Work In Real Life?](https://eyeware.tech/blog/what-is-eye-tracking/) - Eyeware
34. [Eye Tracking: The Complete Pocket Guide](https://imotions.com/blog/learning/best-practice/eye-tracking/) - iMotions
35. [Rapid calibration method for head-mounted eye-tracker](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13080/3025987/Rapid-calibration-method-for-head-mounted-eye-tracker/10.1117/12.3025987.full) - SPIE Digital Library
36. [Eye tracker calibration: How well can humans refixate a target?](https://link.springer.com/article/10.3758/s13428-024-02564-4) - SpringerLink
37. [A nonvisual eye tracker calibration method for video-based eye trackers](https://jov.arvojournals.org/article.aspx?articleid=2702359) - Journal of Vision
38. [Setup of an Eyetracking Study](https://www.nngroup.com/articles/eyetracking-setup/) - Nielsen Norman Group
39. [Correcting systematic calibration errors in eye tracking data](https://patents.google.com/patent/US20160128568A1/en) - Google Patents
