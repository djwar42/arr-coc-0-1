# Saccade Planning & Eye Movement Patterns

## Overview

Saccades are rapid, ballistic eye movements that humans make to redirect the fovea toward objects of interest in the visual field. Unlike smooth pursuit or vergence movements, saccades are the fastest movements the human body produces, reaching peak velocities of 400-700 degrees/second for large movements. Understanding saccade planning and execution is crucial for designing vision models that mimic human attention allocation.

**Why humans use saccadic vision:**

The human visual system evolved with non-uniform sampling across the retina. High-acuity vision is restricted to the fovea (central 1-2° of visual field), while peripheral vision provides lower resolution context. Saccades solve the fundamental problem of how to efficiently sample a high-resolution world with limited neural resources - by rapidly moving the fovea to relevant locations rather than processing the entire visual field at high resolution simultaneously.

**Foveated vision system:**

The foveated architecture creates a biological necessity for saccades:
- Foveal photoreceptor density: ~200,000 cones/mm²
- Peripheral density (10° eccentricity): ~20,000 cones/mm² (10× reduction)
- Only ~1-2° of central vision provides detailed form recognition
- Saccades bring new regions into foveal view 3-4 times per second during active viewing

**Importance for attention research:**

Saccade patterns reveal the underlying priority maps that guide visual attention. The sequence and timing of saccades provide direct behavioral readout of:
- What information the visual system deems relevant
- How task goals modulate bottom-up salience
- The interplay between exploratory and goal-directed attention
- Predictive mechanisms that anticipate future visual input

From [The main sequence, a tool for studying human eye movements](https://www.sciencedirect.com/science/article/pii/0025556475900759) (Bahill et al., 1975, accessed 2025-01-31):
- Established quantitative relationships between saccade amplitude, duration, and peak velocity
- The "main sequence" describes stereotyped kinematics: peak velocity increases with amplitude up to ~20°
- Duration ranges from 30-120ms for typical saccades (5-40° amplitude)

From [The saccade main sequence revised](https://pmc.ncbi.nlm.nih.gov/articles/PMC7880984/) (Gibaldi et al., 2020, accessed 2025-01-31):
- Modern eye-tracking confirms main sequence applies across vergence states
- Saccades are precisely coordinated despite their ballistic nature
- Kinematics are well-defined even for complex 3D eye movements

## Saccade Planning Mechanisms

Saccade planning involves a distributed neural network that transforms visual input into motor commands. The process integrates bottom-up salience signals with top-down task goals to select the next fixation target.

### Neural Substrates

**Superior colliculus (SC):**
- Midbrain structure organizing spatial priority maps
- Topographic representation of visual field
- Integrates visual, auditory, and somatosensory signals
- Direct role in saccade initiation and target selection

From [Activity in brain system that controls eye movements](https://biologicalsciences.uchicago.edu/news/brain-superior-colliculus-spatial-thinking) (University of Chicago, 2024, accessed 2025-01-31):
- Superior colliculus plays dual role: motor control AND higher cognitive functions
- Not just reflex center - involved in spatial reasoning and decision-making
- Suggests saccade planning deeply intertwined with cognitive processing

**Frontal eye fields (FEF):**
- Frontal cortex region in premotor areas
- Voluntary saccade control
- Task-dependent modulation of saccade targets
- Implements top-down attentional control

**Lateral intraparietal area (LIP):**
- Parietal cortex region encoding spatial attention
- Represents salience and behavioral relevance
- Priority map for saccade target selection
- Integrates sensory evidence for decision-making

### Bottom-Up Salience Signals

Bottom-up factors that automatically capture saccades:

**Visual contrast and edges:**
- High-contrast boundaries attract fixations
- Orientation discontinuities draw attention
- Color pop-out effects trigger rapid saccades

**Sudden onsets:**
- New objects appearing in periphery capture attention
- Motion transients trigger reflexive saccades
- Latencies as short as 100-120ms for "express saccades"

From [Express saccades and visual attention](https://www.yorku.ca/science/research/schalljd/wp-content/uploads/sites/654/2022/10/Schall_Hanes-ON-Fischer_Weber_-Saccades-and-Visual-Attention-complete.pdf) (Fischer & Weber, 1993, accessed 2025-01-31):
- Express saccades: reaction times ~100ms (humans), ~70ms (monkeys)
- Related to visual attention disengagement
- Occur when attention is not strongly bound to fixation point
- Reflect pre-programmed saccade plans ready for execution

**Salience based on feature contrast:**
- Itti-Koch style saliency models predict some fixation patterns
- Center-surround differences in color, intensity, orientation
- Limited predictive power without task context

### Top-Down Task Goals

Task instructions dramatically reshape saccade patterns, overriding bottom-up salience:

**Goal-directed search:**
- Target templates modulate which features attract saccades
- Searching for "red objects" increases saccades to red regions
- Task relevance can completely override visual salience

**Semantic knowledge:**
- Fixations cluster on semantically informative regions
- Face viewing: eyes, nose, mouth preferentially fixated
- Scene viewing: objects relevant to current task receive more fixations

**Query-driven attention:**
- Questions about image content reshape scanpaths
- "Is there a person?" → fixations to person-likely regions
- "What color is the car?" → fixations to car, then detailed inspection

From [Eye movements in response to different cognitive activities](https://pmc.ncbi.nlm.nih.gov/articles/PMC10676768/) (Marconi et al., 2023, accessed 2025-01-31):
- Eye movement direction influenced by cognitive activity
- Imagination, internal dialogue, memory retrieval affect saccade patterns
- Not purely stimulus-driven - cognitive state shapes where we look

### Gestalt Understanding Guiding Saccade Targets

Gestalt perception - the ability to perceive whole scenes before fixating details - profoundly influences saccade planning. Global scene structure constrains which local regions attract saccades.

**Scene gist extraction:**
- Peripheral vision (pre-saccade) provides coarse scene category
- Within 100-150ms, observers extract basic scene properties
- "Indoor/outdoor," "natural/urban," "cluttered/sparse" computed from first glance
- This gist guides subsequent saccade targets

**Gestalt grouping principles guide saccades:**

From search results on [gestalt properties and eye tracking](https://www.researchgate.net/figure/The-Number-of-saccades-and-saccade-duration-of-different-gestalt-properties_fig3_369780520) (accessed 2025-01-31):
- Gestalt principles (proximity, similarity, continuity) impact saccade sequences
- Grouped elements receive coordinated saccades
- Perceptual organization pre-structures saccade planning

**Proximity:** Objects close together treated as single saccade target cluster
**Similarity:** Similar colors/shapes group perceptually, saccades scan groups as units
**Continuity:** Smooth contours followed by sequential saccades along the path
**Closure:** Incomplete shapes completed perceptually, guiding saccade completion patterns

**Implications for saccade planning:**
- First fixation often targets region of high global relevance
- Subsequent saccades explore within perceptually coherent groups
- Scanpaths respect gestalt boundaries (e.g., don't randomly jump between objects)
- Global context from periphery pre-programs likely saccade sequences

From [Individual differences in first saccade latency](https://pubmed.ncbi.nlm.nih.gov/40799365/) (Leonard et al., 2024, accessed 2025-01-31):
- First saccade latency predicts how gestalt closure guides attention
- Individual variation in using global structure to plan saccades
- Closed contours formed by oriented elements guide initial saccades

### Priority Maps for Saccade Selection

The brain maintains dynamic priority maps that combine bottom-up and top-down signals:

**Priority = f(Salience, Relevance, History)**

**Salience component:** Low-level visual features (contrast, color, motion)
**Relevance component:** Task goals, semantic importance, behavioral value
**History component:** Inhibition of return (IOR), previously fixated locations deprioritized

**Winner-take-all selection:**
- Priority map feeds into competitive selection mechanism
- Highest priority location wins and triggers saccade
- After saccade, transient inhibition at that location (IOR)
- Ensures efficient exploration rather than re-fixation

**Neural implementation:**

From [Saccades trigger predictive updating of attentional topography](https://pmc.ncbi.nlm.nih.gov/articles/PMC5927612/) (Marino et al., 2018, accessed 2025-01-31):
- Saccades trigger predictive remapping of attention
- Priority maps updated in retinotopic coordinates before saccade lands
- Anticipatory mechanism maintains spatial continuity despite retinal image shifts

**Temporal dynamics:**
- Priority computation takes 150-250ms before saccade initiation
- Parallel accumulation of evidence for multiple potential targets
- Decision threshold reached when one location dominates

From [Efficient Saccade Planning Requires Time and Clear Choices](https://www.researchgate.net/publication/275438424_Efficient_Saccade_Planning_Requires_Time_and_Clear_Choices) (accessed 2025-01-31):
- Efficient saccades require clear priority signals
- Ambiguous targets increase saccade latency
- Task structure providing unambiguous choices speeds saccade planning
- Information-maximizing saccades selected when priorities clear

## Saccade Sequence Patterns

Saccade sequences - called scanpaths - reveal the unfolding of visual attention over time. Patterns in these sequences expose the strategies observers use to extract information.

### Typical Patterns in Free Viewing

Free viewing (no specific task) produces characteristic scanpath patterns:

**Central fixation bias:**
- Initial fixation tends toward image center
- Photographer bias: important content often centered
- Viewing strategies adapt to learned statistics of image composition

**Exploration phase:**
- First 1-2 seconds: broad exploratory saccades
- Large amplitude movements covering wide spatial extent
- Sampling semantically rich regions (objects, faces, text)

**Refinement phase:**
- After 2-3 seconds: smaller amplitude saccades
- Detailed inspection of previously identified regions of interest
- Increasing revisits to already-fixated locations

From [The Sources of Variability in Saccadic Eye Movements](https://pmc.ncbi.nlm.nih.gov/articles/PMC6672172/) (van Beers, 2007, accessed 2025-01-31):
- Saccade variability reflects both sensory uncertainty and motor noise
- Even in free viewing, saccade patterns show individual signatures
- Variability in saccade endpoints 5-10% of amplitude

**Statistical regularities:**

From [The main sequence, a tool for studying human eye movements](https://www.sciencedirect.com/science/article/pii/0025556475900759) (Bahill et al., 1975, accessed 2025-01-31):
- Amplitude distribution: most saccades 5-15° during natural viewing
- Small corrective saccades common (1-3°)
- Very large saccades (>40°) rare except during initial orientation

### Task-Dependent Patterns

Task instructions fundamentally reshape saccade patterns:

**Reading:**
- Highly stereotyped left-to-right saccades (for left-to-right languages)
- Amplitude 7-9 character spaces on average
- Regression saccades (backward jumps) 10-15% of movements
- Fixation duration 200-250ms, tightly linked to lexical processing

**Visual search:**
- Systematic scanning patterns emerge with practice
- Efficient searchers: fewer fixations, more direct paths to target
- Inefficient searchers: more random, revisit locations

From [Effort drives saccade selection](https://elifesciences.org/articles/97760) (Koevoet et al., 2025, accessed 2025-01-31):
- Saccade selection minimizes effort expenditure
- Humans prefer saccades that reduce physical and cognitive costs
- Trade-off between information gain and movement effort
- Task difficulty modulates willingness to make costly saccades

**Scene memorization:**
- More uniform spatial coverage than free viewing
- Deliberate scanning to encode spatial layout
- Longer fixation durations (250-350ms) for encoding

**Question answering:**
- Saccades directed to regions likely containing answer
- Query-specific scanpaths distinct from free viewing same image
- Example: "How many people?" → systematic counting saccades

### First Fixation Biases

Where the eyes land first reveals a lot about rapid scene understanding:

**Center bias:**
- First fixation often within central 5-10° of screen
- Partly due to task structure (images presented centrally)
- Partly strategic (central regions statistically informative)

**Semantic bias:**
- First saccade often lands on most semantically important object
- Faces strongly attract first fixation if present
- Text regions draw early fixations

**Salience vs. meaning:**
- Bottom-up salience predicts first fixation moderately well
- By second and third fixations, semantic task relevance dominates
- First fixation blend of rapid salience detection and gist-based relevance

From [Pre-saccadic attention spreads to stimuli forming a perceptual group](https://www.sciencedirect.com/science/article/pii/S0010945221001313) (Shurygina, 2021, accessed 2025-01-31):
- Pre-saccadic attention spreads along saccade direction
- Perceptually grouped stimuli receive enhanced pre-saccadic processing
- Attention not just spotlight - shaped by gestalt grouping before saccade lands

### Return Saccades

Return saccades - going back to previously fixated locations - serve specific functions:

**Verification saccades:**
- Return to location to confirm initial percept
- Common when initial fixation brief or ambiguous
- Second look often longer duration, more detailed processing

**Comparison saccades:**
- Alternate between two regions for comparison
- "Is A bigger than B?" → saccades between A and B
- Relational judgments require serial inspection

**Inhibition of return (IOR):**
- Transient bias against returning to just-fixated location
- Peaks 200-300ms after saccade
- Ensures efficient exploration by discouraging immediate re-fixation
- Can be overridden by task demands

**Strategic re-fixations:**
- Task-relevant locations revisited despite IOR
- Integration of information across multiple fixations
- Building spatial memory requires revisiting anchor points

### Scanpath Analysis Methods

Researchers quantify scanpath patterns with various metrics:

**Fixation-based metrics:**
- Number of fixations per trial/image
- Mean fixation duration
- Spatial distribution of fixations (coverage, clustering)

**Saccade-based metrics:**
- Saccade amplitude distribution
- Saccade direction biases (horizontal vs. vertical)
- Saccade velocity and acceleration profiles

From [Regularities in vertical saccadic metrics](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2023.1157686/full) (Greene, 2023, accessed 2025-01-31):
- Asymmetries between vertical and horizontal saccades
- Horizontal saccades more common during natural viewing
- Vertical saccades distinct kinematics, possibly different control mechanisms

**Scanpath similarity:**
- Comparing scanpaths between observers or conditions
- String-edit distance on fixation sequences
- MultiMatch algorithm comparing multiple scanpath dimensions
- Cross-correlation of fixation density maps

From [Eye Movement and Pupil Measures: A Review](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2021.733531/full) (Mahanama et al., 2022, accessed 2025-01-31):
- Scanpath: sequence of fixations and saccades describing eye movement pattern
- Scanpaths reveal task-specific strategies
- Could be used to detect cognitive states, expertise, individual differences

**Information-theoretic metrics:**
- Entropy of fixation distribution (randomness vs. stereotypy)
- Mutual information between task and scanpath patterns
- Surprise: unexpected fixations given statistical model

## Eye Movement Order & Cognitive Relevance

The order in which regions are fixated is not random - it reveals the priority structure of visual attention and the unfolding of cognitive processing.

### What Fixation Order Reveals About Cognition

**Serial processing bottleneck:**
- Humans can only deeply process one region at a time (foveal limitation)
- Fixation order = processing order for high-acuity vision
- Earlier fixations indicate higher priority or urgency

**Active inference:**
- Each fixation tests a hypothesis about scene content
- Saccade order reflects uncertainty reduction strategy
- Most informative regions fixated first

From [Worth a Glance: Using Eye Movements to Investigate Memory](https://pmc.ncbi.nlm.nih.gov/articles/PMC2995997/) (Hannula, 2010, accessed 2025-01-31):
- Eye movements reveal memory for previous experience
- Viewing patterns differ for novel vs. familiar scenes
- Memory influences early saccade targeting
- Relational memory affects which objects get compared via saccades

**Predictive processing:**
- Fixation order reflects predictions about what to expect where
- Violated expectations trigger surprise-driven saccades
- Familiar scenes: stereotyped scanpaths (predicted locations)
- Novel scenes: more exploratory, less predictable orders

### Task-Driven Prioritization

Different tasks produce completely different fixation orders on the same image:

**Aesthetic judgment task:**
- Saccades distributed for holistic evaluation
- Composition, balance, color harmony assessed via distributed sampling
- Longer fixations, less systematic scanning

**Object counting task:**
- Systematic serial scanning
- Each object fixated exactly once
- Efficient path minimizes saccade distances

**Memorization task:**
- Thorough coverage of all regions
- Deliberate fixations to encode spatial layout
- Return saccades to verify encoding

**Question answering:**

From search results on [query context effects saccade patterns](https://www.google.com/search?q=eye+movement+order+cognitive+relevance) (accessed 2025-01-31):
- Query context dramatically changes saccade order
- "Where is the dog?" → direct saccades to animal-likely regions
- "What color is the car?" → first locate car, then detailed fixation
- Language-vision interaction guides priority

### Semantic Importance Affecting Saccades

Not all objects are equal - semantic meaning determines fixation priority:

**Objects vs. background:**
- Foreground objects fixated more than background
- Saccades cluster on objects even when not explicitly task-relevant
- Background texture rarely fixated unless searching for camouflaged target

**Animate vs. inanimate:**
- Living things (people, animals) attract earlier fixations
- Faces especially powerful attentional magnets
- Evolutionary priority for social and threat-relevant stimuli

**Contextual relevance:**
- Objects congruent with scene context fixated normally
- Incongruent objects ("octopus in living room") attract prolonged fixations
- Semantic violations detected surprisingly early (by 2nd-3rd fixation)

**Text prioritization:**
- Text regions almost always fixated if present
- High information density makes text high priority
- Even in non-reading tasks, text draws fixations

From [Coupling of saccade plans to endogenous attention](https://pubmed.ncbi.nlm.nih.gov/38496491/) (Goldstein et al., 2024, accessed 2025-01-31):
- Saccade planning tightly coupled to endogenous (voluntary) attention
- Salient stimuli automatically capture both attention and saccade plans
- Difficult to decouple saccade planning from attention shifts
- Shared mechanisms for attention and eye movement control

### Individual Differences in Scanpaths

Not everyone looks at scenes the same way:

**Expertise effects:**
- Experts show more efficient scanpaths in their domain
- Radiologists: faster target detection, fewer fixations
- Chess masters: fixations to strategically relevant positions
- Expertise = optimized saccade sequences

**Cognitive style:**
- Field-dependent vs. field-independent processing
- Some observers focus locally (object details)
- Others process globally (scene relationships)
- Reflected in relative fixation distribution

**Task strategy:**
- Speed-accuracy tradeoff affects saccade patterns
- Speed emphasis: fewer fixations, larger saccades
- Accuracy emphasis: more thorough inspection, smaller saccades

From [Individual differences in first saccade latency](https://pubmed.ncbi.nlm.nih.gov/40799365/) (Leonard et al., 2024, accessed 2025-01-31):
- First saccade latency varies systematically between individuals
- Correlates with how gestalt closure guides attention
- Stable individual signatures in saccade planning

**Developmental differences:**
- Children show less systematic scanpaths than adults
- More random exploration, less task-optimized
- Scanpath maturation continues through adolescence

### Measuring Agreement in Eye Movements

How similar are different people's scanpaths? Measuring inter-observer agreement:

**Fixation overlap:**
- Simple metric: do observers fixate same regions?
- Thresholded spatial proximity (e.g., within 2° = match)
- Reports proportion of shared fixations

**Area Under Curve (AUC):**
- Treat one observer's fixations as ground truth
- Other observer's fixation map as predicted saliency
- ROC analysis: AUC indicates predictive power
- Typical inter-observer AUC: 0.70-0.85

**Normalized Scanpath Saliency (NSS):**
- Average saliency value at fixation locations
- Normalized by saliency map statistics
- NSS > 1 indicates above-chance agreement
- Typical inter-observer NSS: 1.5-2.5

**String-edit distance:**
- Represent scanpath as sequence of fixated regions
- Compute edit distance between sequences
- Captures temporal order, not just spatial overlap

From [Eye Movement and Pupil Measures: A Review](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2021.733531/full) (Mahanama et al., 2022, accessed 2025-01-31):
- Multiple metrics needed to fully characterize scanpath similarity
- Spatial overlap insufficient - temporal dynamics matter
- Scanpath comparison used in expertise assessment, UI evaluation

**Task effects on agreement:**
- Free viewing: moderate agreement (AUC ~0.75)
- Specific question: high agreement (AUC ~0.85)
- Aesthetic judgment: low agreement (AUC ~0.65)
- Agreement increases with task constraint

**Implications for model evaluation:**
- Human agreement sets upper bound for model performance
- Models shouldn't be expected to exceed inter-observer agreement
- Task-dependent agreement means models need task conditioning

## Computational Models and Applications

Understanding saccade planning informs vision model design:

**Active vision models:**
- Sequential attention mechanisms
- Learned policies for where to look next
- Reinforcement learning for saccade control

From [Predictive coding as a model of biased competition in visual attention](https://www.sciencedirect.com/science/article/pii/S0042698908001466) (Spratling, 2008, accessed 2025-01-31):
- Predictive coding framework for attention
- Top-down predictions bias competition for saccade targets
- Error signals drive attention to unpredicted/surprising locations
- Saccades minimize prediction error

**Foveated rendering:**
- Real-time graphics systems mimicking human vision
- High resolution at gaze point, degraded periphery
- Requires accurate saccade prediction

**Attention-aware architectures:**
- Vision transformers with learned saccade-like attention
- Differentiable visual search
- Meta-learning for efficient visual exploration

From [Visual Attention Saccadic Models Learn to Emulate Gaze](https://inria.hal.science/hal-01650322/file/LeMeur_SaccadicModelFromChildhoodToAdulthood_Final.pdf) (Le Meur et al., 2017, accessed 2025-01-31):
- Computational saccadic models can emulate human gaze patterns
- Models trained on adult data also predict developmental changes
- Saccadic models capture task-dependent viewing strategies
- Successfully predict where humans look in complex scenes

**Applications to VLMs:**
- Query-conditional visual attention
- Relevance-guided token allocation
- Dynamic resolution based on predicted importance
- Bio-inspired multi-fixation processing

## Sources

### Web Research

**Academic Papers:**

- [The main sequence, a tool for studying human eye movements](https://www.sciencedirect.com/science/article/pii/0025556475900759) - Bahill et al., 1975 (Accessed: 2025-01-31)

- [The saccade main sequence revised: A fast and repeatable tool for oculomotor analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC7880984/) - Gibaldi et al., 2020 (Accessed: 2025-01-31)

- [The Sources of Variability in Saccadic Eye Movements](https://pmc.ncbi.nlm.nih.gov/articles/PMC6672172/) - van Beers, 2007 (Accessed: 2025-01-31)

- [Effort drives saccade selection](https://elifesciences.org/articles/97760) - Koevoet et al., 2025 (Accessed: 2025-01-31)

- [Regularities in vertical saccadic metrics: new insights, and implications](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2023.1157686/full) - Greene, 2023 (Accessed: 2025-01-31)

- [Express saccades and visual attention](https://www.yorku.ca/science/research/schalljd/wp-content/uploads/sites/654/2022/10/Schall_Hanes-ON-Fischer_Weber_-Saccades-and-Visual-Attention-complete.pdf) - Fischer & Weber, 1993 (Accessed: 2025-01-31)

- [Individual differences in first saccade latency predict overt visual attention](https://pubmed.ncbi.nlm.nih.gov/40799365/) - Leonard et al., 2024 (Accessed: 2025-01-31)

- [Pre-saccadic attention spreads to stimuli forming a perceptual group](https://www.sciencedirect.com/science/article/pii/S0010945221001313) - Shurygina, 2021 (Accessed: 2025-01-31)

- [Saccades trigger predictive updating of attentional topography](https://pmc.ncbi.nlm.nih.gov/articles/PMC5927612/) - Marino et al., 2018 (Accessed: 2025-01-31)

- [Eye movements in response to different cognitive activities](https://pmc.ncbi.nlm.nih.gov/articles/PMC10676768/) - Marconi et al., 2023 (Accessed: 2025-01-31)

- [Coupling of saccade plans to endogenous attention during urgent choices](https://pubmed.ncbi.nlm.nih.gov/38496491/) - Goldstein et al., 2024 (Accessed: 2025-01-31)

- [Worth a Glance: Using Eye Movements to Investigate the Cognitive Neuroscience of Memory](https://pmc.ncbi.nlm.nih.gov/articles/PMC2995997/) - Hannula, 2010 (Accessed: 2025-01-31)

- [Eye Movement and Pupil Measures: A Review](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2021.733531/full) - Mahanama et al., 2022 (Accessed: 2025-01-31)

- [Predictive coding as a model of biased competition in visual attention](https://www.sciencedirect.com/science/article/pii/S0042698908001466) - Spratling, 2008 (Accessed: 2025-01-31)

- [Visual Attention Saccadic Models Learn to Emulate Gaze Patterns from Childhood to Adulthood](https://inria.hal.science/hal-01650322/file/LeMeur_SaccadicModelFromChildhoodToAdulthood_Final.pdf) - Le Meur et al., 2017 (Accessed: 2025-01-31)

**News/Research Articles:**

- [Activity in brain system that controls eye movements highlights importance of superior colliculus in spatial thinking](https://biologicalsciences.uchicago.edu/news/brain-superior-colliculus-spatial-thinking) - University of Chicago, 2024 (Accessed: 2025-01-31)

- [Efficient Saccade Planning Requires Time and Clear Choices](https://www.researchgate.net/publication/275438424_Efficient_Saccade_Planning_Requires_Time_and_Clear_Choices) - ResearchGate (Accessed: 2025-01-31)

- [The Number of saccades and saccade duration of different gestalt properties](https://www.researchgate.net/figure/The-Number-of-saccades-and-saccade-duration-of-different-gestalt-properties_fig3_369780520) - ResearchGate (Accessed: 2025-01-31)

**Search Queries Used:**
- "saccade planning gestalt understanding 2023-2025" (Google)
- "saccade sequence patterns human vision studies" (Google)
- "eye movement order cognitive relevance" (Google)
- "predictive saccade models visual attention" (Google)

### Additional References

See also:
- [00-gestalt-visual-attention.md](00-gestalt-visual-attention.md) - How global context guides saccade planning
- [karpathy/attention-mechanisms/](../attention-mechanisms/) - Computational models of attention
