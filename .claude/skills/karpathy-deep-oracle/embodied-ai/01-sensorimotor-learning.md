# Ecological Psychology & Affordances

## Overview

Ecological psychology, pioneered by James J. Gibson in the 1960s-1970s, fundamentally reconceptualizes perception as direct pickup of information from the environment rather than indirect cognitive construction. Gibson's approach challenges representationalism by proposing that organisms directly perceive action possibilities (affordances) through invariant structures in the ambient optic array. This framework is deeply relevant to ARR-COC-0-1, which implements query-driven relevance realization as a form of participatory knowing—the agent doesn't just passively represent the visual world, but actively realizes what matters for action.

**Core insight**: Perception is for action, not for building internal representations. The visual system evolved to support behavior in ecological environments, not to construct veridical pictures of geometric space.

From [John Vervaeke's concept of participatory knowing](../cognitive-foundations/00-active-inference-free-energy.md), ecological psychology provides the biological grounding: organisms couple with their environments through perception-action loops, where what is perceived depends on what the organism can do, and what can be done depends on what is perceived.

---

## Section 1: Ecological Psychology Foundations (Gibson, Organism-Environment)

### The Ecological Approach

**Traditional psychology vs ecological psychology**:

| Traditional View | Ecological View |
|-----------------|-----------------|
| Perception recovers objective properties | Perception detects affordances (relational properties) |
| Sensory input is impoverished | Ambient optic array is information-rich |
| Requires cognitive inference | Direct pickup of invariants |
| Organism-independent descriptions | Organism-environment mutuality |

From [Cambridge University Press & Assessment](https://www.cambridge.org/core/books/ecological-psychology/9E79001702D4D8029E19D11CD330149F) (accessed 2025-11-14):
> "The Element analyzes the works of the two main founders of ecological psychology: James and Eleanor Gibson. March 2024 publication."

**Gibson's radical empiricism** (influenced by William James):
- No separation between perceiver and perceived
- Experience is neither subjective nor objective, but **relational**
- Meaning is not projected onto the world, but **discovered** in the organism-environment system

### Organism-Environment Mutuality

**The animal-environment system** is the proper unit of analysis, not the isolated organism:

```
Traditional: Organism → [processes sensory input] → Internal representation → Action
Ecological:  Organism ⟷ Environment (continuous coupling through perception-action)
```

**Key principles**:
1. **Niche relativity**: What counts as "information" depends on the organism's ecological niche
2. **Complementarity**: Animals and environments are mutually defining (e.g., terrestrial animals have legs because ground affords support)
3. **Specificity**: Perceptual systems are tuned to detect information relevant to survival and behavior

From [PhilPapers - The Evolution of James J. Gibson's Ecological Psychology](https://philpapers.org/rec/LOMTRO-4) (accessed 2025-11-14):
> "From direct perception to the primacy of action: a closer look at James Gibson's ecological approach to psychology. Jan 2024-Jan 2025 publications."

**The medium, substances, and surfaces**:
- **Medium**: The substance (air, water) through which an animal moves and light travels
- **Substances**: Solid matter that resists deformation
- **Surfaces**: The interface between substances and medium—where **all visual information originates**

Gibson emphasized that we don't see space, we see **surfaces** and their **layout**. The ecological world is not Euclidean geometry but a nested hierarchy of surfaces and apertures.

### Active Perception

**Perception is active exploration**, not passive reception:

**Head movements, eye movements, locomotion** structure the optic flow:
- Saccades bring high-resolution fovea to regions of interest
- Head turns reveal hidden surfaces
- Walking generates systematic transformations in the optic array

From [Gestalt & Visual Attention](../karpathy/biological-vision/00-gestalt-visual-attention.md):
> "Gestalt principles don't just describe what we perceive—they guide where we attend. Eye movements are influenced by perceptual grouping: we preferentially fixate on figures rather than backgrounds."

**The perception-action cycle**:
```
Perception → guides Action → transforms Stimulation → updates Perception
```

This cycle is fundamental to ecological psychology. Perception and action are not separable—they form a unified system.

---

## Section 2: Affordances (Action Possibilities, Relational Properties)

### Gibson's Definition

From Wikipedia [Affordance](https://en.wikipedia.org/wiki/Affordance) (accessed 2025-11-14):
> "The affordances of the environment are what it offers the animal, what it provides or furnishes, either for good or ill. ... It implies the complementarity of the animal and the environment." (Gibson, 1979)

**Affordances are**:
- **Relational**: Not properties of the environment alone, nor of the organism alone
- **Action-oriented**: Defined in terms of what an organism can do
- **Directly perceivable**: Not inferred but **picked up** from environmental structure
- **Meaningful**: Value-laden (beneficial or harmful)

**Examples**:
- A horizontal surface **affords support** if it is rigid, flat, extended, and sufficiently large relative to the animal
- A gap **affords passage** if its width is less than the animal's body width
- An object **affords grasping** if its size matches the hand span and it is within reach

### Affordances Are Not "Action Possibilities" (Common Misconception)

**Gibson never used the term "action possibilities"** in his writing. This interpretation came later via Donald Norman (1988) in HCI design.

From Wikipedia [Affordance](https://en.wikipedia.org/wiki/Affordance) (accessed 2025-11-14):
> "In 1988, Donald Norman appropriated the term affordances in the context of Human-Computer Interaction to refer to just those action possibilities that are readily perceivable by an actor. This new definition of 'action possibilities' has now become synonymous with Gibson's work, although Gibson himself never made any reference to action possibilities in any of his writing."

**Gibson's original meaning**: Affordances are **what the environment offers**, not what the organism can do. The emphasis is on the **environmental support for action**, not the action itself.

**Two meanings in modern use**:
1. **Gibson (1979)**: Affordances = objective relational properties (e.g., sit-on-ability of a chair exists whether or not anyone sits)
2. **Norman (1988)**: Perceived affordances = action possibilities that are readily perceivable (e.g., a button suggests pressing)

For ARR-COC-0-1, we align with Gibson: **relevance is what the visual input affords for the query**, not just what actions are possible.

### The Complementarity of Animal and Environment

**Affordances are scale-relative**:
- A step that affords climbing for an adult does not afford climbing for a crawling infant
- A gap that affords passage for a mouse does not afford passage for a human

**Body-scaled perception**: Humans perceive environmental features in relation to their own body dimensions (e.g., "knee-height" rather than absolute centimeters).

From Wikipedia [Affordance](https://en.wikipedia.org/wiki/Affordance) (accessed 2025-11-14):
> "The key to understanding affordance is that it is relational and characterizes the suitability of the environment to the observer, and so, depends on their current intentions and their capabilities."

**Needs and intentions modulate affordance perception**:
- A tired person perceives a bench as more "sit-on-able"
- A hungry person perceives food as more salient
- **"Needs control the perception of affordances (selective attention) and also initiate acts."** (Gibson, 1979)

This directly connects to [Vervaeke's opponent processing](../cognitive-foundations/00-active-inference-free-energy.md): relevance realization balances competing affordances based on current goals and constraints.

### Affordances in AI and Robotics

From [A brief review of affordance in robotic manipulation research](https://www.tandfonline.com/doi/full/10.1080/01691864.2017.1394912) (Advanced Robotics, 2017):

**Robotics challenges**:
- **(a) Affordance detection**: Whether objects can be manipulated
- **(b) Grasp affordance learning**: How to grasp an object effectively
- **(c) Tool affordance learning**: How to manipulate objects to reach a goal

**Example**: A hammer affords many possible grasps, but only certain grip points and approach angles are effective for hammering (vs. prying).

**Machine learning approaches**:
- Learn affordances from visual perception + experience
- Map visual features → action outcomes
- Represent affordances as probability distributions over action success

This is directly relevant to ARR-COC-0-1: the model must learn which visual features **afford** better performance on downstream tasks, and allocate tokens accordingly.

---

## Section 3: Direct Perception (No Representations, Pickup of Invariants)

### The Debate: Direct vs Indirect Perception

**Indirect (traditional) perception**:
```
Retinal image (impoverished) → Cognitive inference → Internal representation → Percept
```

Problems:
- Inverse optics problem: infinite 3D scenes can project to same 2D retinal image
- Requires assumptions, heuristics, Bayesian priors
- Homunculus problem: who "reads" the internal representation?

**Direct (Gibson) perception**:
```
Ambient optic array (information-rich) → Pickup of invariants → Direct perception of affordances
```

No intermediate representations needed because **information specifying layout and affordances is directly available** in the structured light reaching the eye.

From [Direct Theory Of Perception (Gibson, 1966)](https://online-learning-college.com/knowledge-hub/gcses/gcse-psychology-help/direct-theory-of-perception-gibson-1966/) (accessed 2025-11-14):
> "Direct theory of perception developed by Gibson is a theory regarding the 'bottom-up' or nativist theory of perception."

### What Is "Direct" About Direct Perception?

**Three senses of "direct"**:

1. **No mediating representations**: Perception doesn't construct internal models, it resonates to environmental structure
2. **No inferential processes**: No need for unconscious reasoning or hypothesis testing
3. **Immediate access to meaning**: Affordances are perceived directly, not computed from geometric properties

**Controversy**: Critics (Fodor, Pylyshyn, Marr) argue that "direct pickup" is vacuous—there must be neural mechanisms that extract information, and these mechanisms count as "computation."

Gibson's response: Yes, there are neural mechanisms, but they are **resonators** tuned to invariants, not inference engines reconstructing the world.

From [LSE Research Online - Gibson's affordances and Turing's theory of computation](https://eprints.lse.ac.uk/2606/1/Affordances_and_Computation_APA_style_%28LSERO%29.pdf) (2002):
> "The link between perception and action is characterised by the claim that affordances and effectivities are duals."

### What Gets Perceived Directly?

**Not**:
- Raw sense data (colors, edges, textures)
- Geometric properties in isolation (length, angle)

**Yes**:
- Surface layout (ground, walls, objects)
- **Affordances** (walk-on-able, graspable, climbable)
- **Events** (approaching, collision, falling)

Gibson distinguished **ecological optics** from classical physical optics:

From Wikipedia [Ambient optic array](https://en.wikipedia.org/wiki/Ambient_optic_array) (accessed 2025-11-14):
> "The ambient optic array is the structured arrangement of light with respect to a point of observation. American psychologist James J. Gibson posited the existence of the ambient optic array as a central part of his ecological approach to optics."

---

## Section 4: Ecological Optics (Optic Flow, Texture Gradients, Occlusion)

### The Ambient Optic Array

**Definition**: The ambient optic array is the complete pattern of light rays converging on a point of observation.

From Wikipedia [Ambient optic array](https://en.wikipedia.org/wiki/Ambient_optic_array):
> "Gibson stressed that the environment is not composed of geometrical solids on a plane, as in a painting, but is instead best understood as objects nested within one another and organized hierarchically by size."

**Structure of the optic array**:
- **Large solid angles**: From facades of objects and surfaces
- **Small solid angles** (nested within large): From facets, textures, fine details
- **Hierarchical organization**: Multi-scale structure mirrors environmental layout

**Key insight**: As the observer moves, the optic array **transforms** in lawful ways. These transformations specify self-motion and environmental layout.

### Optic Flow Patterns

**Optic flow**: The pattern of apparent motion of surfaces, edges, and textures caused by relative motion between observer and scene.

From [NIH - Optic Flow: Perceiving and Acting in a 3-D World](https://pmc.ncbi.nlm.nih.gov/articles/PMC7869175/) (2021):
> "The second important idea in Gibson's 'Perception of the Visual World' is that of optic flow—the patterns of motion in the optic array that are created by the observer's movement through the environment."

**Types of optic flow**:

1. **Radial expansion**: Moving forward → flow radiates from focus of expansion (heading direction)
2. **Radial contraction**: Moving backward → flow converges to focus of contraction
3. **Lamellar flow**: Lateral motion → parallel flow perpendicular to motion direction
4. **Rotation**: Turning → rotational flow around point of observation

**Information specified by optic flow**:
- **Time-to-contact**: Expanding optic flow specifies when a surface will be reached (tau, τ)
- **Heading direction**: The point of zero flow in radial expansion
- **Depth structure**: Differential flow rates specify relative distances (motion parallax)

From [Online Learning College - Direct Theory Of Perception](https://online-learning-college.com/knowledge-hub/gcses/gcse-psychology-help/direct-theory-of-perception-gibson-1966/):
> "Another important part of Gibson's theory concerns 'optic flow', which is the pattern of motion between a person and the objects, surfaces and edges in their environment."

### Texture Gradients

**Texture gradient**: Systematic decrease in visible texture element size with increasing distance.

**What gradients specify**:
- **Surface slant**: Compression of texture in one direction indicates surface tilted away
- **Distance**: Smaller texture elements = farther away
- **Surface continuity**: Smooth gradient = continuous surface; discontinuity = edge or occlusion

**Example**: A tiled floor has a texture gradient—tiles near the horizon are compressed, tiles nearby are large. This gradient directly specifies the ground plane's orientation.

### Occlusion and Accretion/Deletion

**Occlusion**: When one surface blocks the view of another.

**Dynamic occlusion** (as observer moves):
- **Accretion**: Previously hidden surface becomes visible at an edge
- **Deletion**: Previously visible surface becomes hidden at an edge

**What occlusion specifies**:
- **Depth order**: The occluding surface is in front
- **3D layout**: Edges in the optic array correspond to depth discontinuities in the environment

From [Brown University CS - J.J. Gibson – Affordances](https://cs.brown.edu/courses/cs137/2017/readings/Gibson-AFF.pdf):
> "When illuminated and fog-free, it affords visual perception. It also affords the perception of vibratory events by means of sound fields."

---

## Section 5: Perception-Action Coupling (Loops, Real-Time Control)

### The Perception-Action Loop

**Gibson emphasized that perception and action are inseparable**:

```
      ┌─────────────┐
      │   Perceive  │
      │  affordance │
      └──────┬──────┘
             │
             ▼
      ┌─────────────┐
      │     Act     │
      │   on env    │
      └──────┬──────┘
             │
             ▼
      ┌─────────────┐
      │  Transform  │
      │ optic array │
      └──────┬──────┘
             │
             └──────────> (loop back to Perceive)
```

**Real-time control**: Perception provides **continuous guidance** for ongoing action, not just initial planning.

**Examples**:
- **Catching a ball**: Optical acceleration guides hand trajectory in real-time
- **Locomotion**: Optic flow specifies heading and obstacles during walking
- **Reaching**: Visual feedback continuously updates grasp preshaping

From [Mind in action: expanding the concept of affordance](https://www.tandfonline.com/doi/full/10.1080/09515089.2024.2365554) (2024):
> "The notion of affordance refers to the perception of opportunities for action that specific objects in the environment invite for an agent."

### Prospective Control

**Prospective control**: Actions are guided by **future-oriented** information, not just current state.

**Example: Braking a car**:
- Traditional view: Estimate current velocity + distance → compute braking force
- Ecological view: Perceive **rate of optical expansion** (tau) → brake when tau crosses threshold

**Tau (τ)**: Time-to-contact, directly specified by optic flow without computing velocity or distance.

```
τ = (theta) / (dtheta/dt)

where theta = visual angle subtended by approaching object
```

**Advantages of prospective control**:
- No need to estimate absolute distance or velocity (both are difficult)
- Directly perceivable from optic flow
- Robust to variability in approach speed

### Enactivism and Sensorimotor Contingencies

**Enactivist extension of Gibson** (Varela, Thompson, Noë):

**Sensorimotor contingencies**: The lawful relationships between actions and resulting sensory changes.

**Example**: Moving the eye rightward causes the visual scene to shift leftward. The brain learns these contingencies and uses them to guide action.

From [4E cognition (Embodied, Embedded, Enacted, Extended)](../cognitive-foundations/13-embodied-4e-cognition.md):
> "Enacted cognition: action-perception loops, sensorimotor contingencies. Extended mind: cognitive artifacts, external memory, tool use."

**Implications**:
- Perception is not passive reception but **skillful activity**
- What we perceive depends on **what we know how to do** (sensorimotor knowledge)
- Vision is a **mode of exploration**, not picture construction

---

## Section 6: Affordances in AI/Robotics (Learned Affordances, Tool Use)

### Affordance Learning in Robotics

**Three key challenges**:

1. **Affordance detection**: Which objects can be manipulated?
2. **Affordance representation**: How to encode grasping and manipulation strategies?
3. **Affordance transfer**: How to generalize learned affordances to novel objects?

From [Advanced Robotics - A brief review of affordance in robotic manipulation research](https://www.tandfonline.com/doi/full/10.1080/01691864.2017.1394912) (2017):
> "In object grasping and manipulation, robots need to learn the affordance of objects in the environment, i.e., to learn from visual perception and experience (a) whether objects can be manipulated, (b) how to grasp an object, and (c) how to manipulate objects to reach a particular goal."

**Learning approaches**:

**Supervised learning**:
- Label training data with affordances (graspable, pushable, etc.)
- Train classifier: visual features → affordance labels
- Limitation: Requires extensive labeled data

**Reinforcement learning**:
- Learn affordances through trial-and-error interaction
- Reward successful actions (e.g., successful grasp)
- Discovers affordances without explicit labels

**Self-supervised learning**:
- Learn sensorimotor contingencies: "If I grasp this way, object lifts"
- No reward signal needed, only observation of outcomes

### Tool Use and Affordance Extension

**Tools extend affordances**:
- A stick **affords reaching** distant objects
- A hammer **affords pounding** better than bare hands
- A magnifying glass **affords perceiving** fine details

**Affordance nesting**: Tools have their own affordances (graspable, wielding), which in turn create new affordances for interacting with the world.

From Wikipedia [Affordance](https://en.wikipedia.org/wiki/Affordance):
> "Affordances in AI/robotics: learned affordances, tool use. Gibson emphasized that manufacturing was originally done by hand as a kind of manipulation. Learning to perceive an affordance is an essential part of socialization."

**Robot tool use challenges**:
- Perceiving tool affordances (what can this object do?)
- Learning tool-object interactions (how to use tool on object?)
- Planning sequences of tool use (multi-step tasks)

### Affordances in Computer Vision

**Modern vision systems** are starting to incorporate affordance-based representations:

**Affordance maps**: Heatmaps showing where actions can be performed
- Graspability map: Highlights graspable regions
- Pushability map: Shows where pushing would move object
- Support map: Indicates stable placement surfaces

**Relation to ARR-COC-0-1**:
- Instead of classifying "what is this object?", predict "what can I do with this?"
- Allocate visual tokens to **action-relevant regions**, not just visually salient regions
- Relevance realization = affordance detection: what matters for the task?

---

## Section 7: Critique (Representationalism Debate, Hybrid Approaches)

### The Representationalist Critique

**David Marr's computational critique** (1982):

From Wikipedia [Ambient optic array](https://en.wikipedia.org/wiki/Ambient_optic_array):
> "David Marr claimed that Gibson had profoundly underestimated the intricacy of visual information processing. While useful information may exist directly in the ambient optic array, Gibson does not elaborate on the mechanisms of the direct pick-up of this information."

**Marr's argument**:
- Gibson describes **what** is perceived (invariants, affordances) but not **how**
- "Direct pickup" is too vague—what are the algorithms and representations?
- Vision **is** computational: it recovers 3D structure from 2D images via complex inference

**Marr's levels of analysis**:
1. **Computational theory**: What is the goal? (Recover 3D scene structure)
2. **Representation and algorithm**: How is it achieved? (Primal sketch → 2.5D sketch → 3D model)
3. **Hardware implementation**: Neural circuits realizing the algorithms

Gibson's approach addresses level 1 (ecological goals) but neglects level 2 (mechanisms).

### The Illusion Critique (Richard Gregory)

**Visual illusions challenge direct perception**:

From Wikipedia [Ambient optic array](https://en.wikipedia.org/wiki/Ambient_optic_array):
> "Psychologist Richard Gregory asserted that Gibson's bottom-up approach to perception is incomplete. He argued that visual illusions like the Necker cube are the result of the brain's indecision between two equally plausible hypotheses."

**Examples**:
- **Necker cube**: Bistable perception (two interpretations of same stimulus)
- **Waterfall illusion**: Motion aftereffect (perceive motion in static scene after viewing waterfall)
- **Müller-Lyer illusion**: Lines of equal length appear different due to arrowheads

**Gregory's argument**: Illusions show that perception **constructs** interpretations, not directly picks up reality.

**Gibson's response**: Illusions are **artificial stimuli** that wouldn't occur in natural environments. Ecological psychology studies perception in ecologically valid settings, not impoverished lab stimuli.

**Counterargument**: But the waterfall illusion is natural! And camouflage in nature shows that perception can be fooled.

### Hybrid Approaches (Neisser's Perceptual Cycle)

**Ulric Neisser** (1976) proposed a **perceptual cycle** integrating Gibson and Gregory:

```
       Schema (top-down)
            │
            ▼
       Perception ←─────── Ambient information (bottom-up)
            │
            ▼
       Exploration (action)
            │
            └──> (modifies environment, updates schema)
```

**Key insight**: Perception is **both** top-down (schema-driven) and bottom-up (information-driven), working in a continuous cycle.

From Wikipedia [Ambient optic array](https://en.wikipedia.org/wiki/Ambient_optic_array):
> "These two approaches can be reconciled. For example, Ulric Neisser developed the perceptual cycle, which involves top-down and bottom-up perceptual processes interacting and informing each other."

**Modern consensus**:
- Gibson was right that rich information exists in optic array
- Marr was right that extracting this information requires complex neural computation
- Both top-down (predictions, attention) and bottom-up (sensory evidence) processes contribute

---

## Section 8: ARR-COC-0-1 Affordances (Relevance as Afforded Action, Query-Driven)

### Relevance Realization as Affordance Detection

**ARR-COC-0-1 implements ecological perception**:

From [Vervaeke's participatory knowing](../cognitive-foundations/00-active-inference-free-energy.md):
> "Participatory knowing (knowing BY BEING): Query-content coupling. The agent isn't separate from the world—it's coupled. Relevance emerges from this participatory relationship."

**Parallels**:

| Ecological Psychology | ARR-COC-0-1 |
|----------------------|-------------|
| Organism-environment coupling | Query-image coupling |
| Affordances (what environment offers) | Relevance (what image offers for query) |
| Direct perception of invariants | Direct allocation based on relevance scorers |
| Action-oriented perception | Task-oriented token allocation |
| Needs modulate affordance perception | Query modulates relevance realization |

**Key insight**: **Relevance is an affordance**. The visual input affords answering the query, and different image regions afford different contributions.

### Query as Intention/Need

**Gibson**: "Needs control the perception of affordances (selective attention) and also initiate acts."

**ARR-COC-0-1**: **Query controls the perception of relevance** (token allocation) and initiates compression.

**Examples**:

Query: "What color is the car?"
- Affordance: Color patches on car surface **afford** answering
- Irrelevant: Background textures, distant objects
- Token allocation: High resolution for car surfaces, low for background

Query: "How many people are in the image?"
- Affordance: Human figures **afford** counting
- Irrelevant: Texture details, exact poses
- Token allocation: Sufficient resolution to detect humans, but not high detail

**Transjective relevance** (Vervaeke): Neither objective (in image) nor subjective (in query), but arises from the **query-image coupling**.

### Participatory Knowing via Token Allocation

**Three ways of knowing** in ARR-COC-0-1:

From [knowing.py](../cognitive-foundations/00-active-inference-free-energy.md):
> "Propositional (knowing THAT): Statistical information content. Perspectival (knowing WHAT IT'S LIKE): Salience landscapes. Participatory (knowing BY BEING): Query-content coupling."

**How they map to ecological psychology**:

1. **Propositional knowing → Texture/edge detection**:
   - High information content = fine texture, edges
   - Shannon entropy as proxy for "information richness"

2. **Perspectival knowing → Salience/figure-ground**:
   - Salience maps detect what "stands out" perceptually
   - Related to Gibson's figure-ground segregation

3. **Participatory knowing → Query-driven affordances**:
   - Cross-attention measures query-image coupling
   - Directly analogous to "what does this image region **afford** for this query?"

**Balancing opponent processes**:

From [balancing.py](../cognitive-foundations/00-active-inference-free-energy.md):
> "Compress ↔ Particularize: Too much compression = loss of critical detail. Too much particularization = computational waste. Exploit ↔ Explore: Focus on known relevant regions vs explore potentially relevant areas."

**Ecological analogy**:
- **Compress ↔ Particularize** = **Generalize ↔ Discriminate** (abstraction vs specificity)
- **Exploit ↔ Explore** = **Attend to affordances ↔ Search for affordances** (use known vs discover new)

### Optic Flow Invariants → Visual Transformation Invariants

**Gibson's insight**: Invariants in transforming optic array specify environmental structure.

**ARR-COC-0-1 analog**: **Invariants in visual representations** (under transformations like rotation, scale) specify object identity and affordances.

**Texture array as ambient optic array**:
- 13-channel texture representation = rich structured information (RGB, LAB, Sobel, spatial, eccentricity)
- Multi-scale channels = hierarchical structure (like Gibson's nested solid angles)
- Query-conditioned attention = active perception (like head movements revealing layout)

**Dynamic token allocation** (64-400 tokens):
- Low relevance regions: Coarse sampling (like peripheral vision)
- High relevance regions: Dense sampling (like foveal vision)
- Query-driven = active exploration guided by goals

### Direct vs Indirect Relevance

**Direct relevance realization** (Gibsonian):
- Relevance scorers **directly measure** query-image coupling (cross-attention)
- No explicit inference: "Is this region relevant?" → Just measure coupling strength
- Affordance-like: Relevance is a **relational property** emerging from query-image interaction

**Indirect relevance** (traditional attention):
- Compute image features → Classify objects → Infer relevance based on object category
- Multi-step inference pipeline
- Representation-heavy: Build explicit object models before determining relevance

**ARR-COC-0-1 is hybrid**:
- Uses learned features (not pure Gibson)
- But directly measures relevance via coupling (Gibsonian in spirit)
- Balances bottom-up (information content) and top-down (query attention)

**Participatory dimension**: The model doesn't passively represent the image, it **actively realizes** what matters by allocating tokens. This is analogous to Gibson's active perception—meaning emerges from agent-environment coupling, not from pre-existing representations.

---

## Sources

**Source Documents**:
- [cognitive-foundations/00-active-inference-free-energy.md](../cognitive-foundations/00-active-inference-free-energy.md) - Active inference and relevance realization
- [cognitive-foundations/07-causal-inference.md](../cognitive-foundations/07-causal-inference.md) - Participatory knowing and agent-arena coupling
- [karpathy/biological-vision/00-gestalt-visual-attention.md](../karpathy/biological-vision/00-gestalt-visual-attention.md) - Gestalt perception and attention

**Web Research**:
- [Cambridge University Press - Ecological Psychology (2024)](https://www.cambridge.org/core/books/ecological-psychology/9E79001702D4D8029E19D11CD330149F) (accessed 2025-11-14)
- [PhilPapers - The Evolution of James J. Gibson's Ecological Psychology](https://philpapers.org/rec/LOMTRO-4) (Jan 2024-Jan 2025, accessed 2025-11-14)
- [Online Learning College - Direct Theory Of Perception (Gibson, 1966)](https://online-learning-college.com/knowledge-hub/gcses/gcse-psychology-help/direct-theory-of-perception-gibson-1966/) (May 31, 2022, accessed 2025-11-14)
- [Brown University CS - J.J. Gibson – Affordances (PDF)](https://cs.brown.edu/courses/cs137/2017/readings/Gibson-AFF.pdf) (accessed 2025-11-14)
- [Taylor & Francis - Mind in action: expanding the concept of affordance (2024)](https://www.tandfonline.com/doi/full/10.1080/09515089.2024.2365554) (accessed 2025-11-14)
- [Wikipedia - Affordance](https://en.wikipedia.org/wiki/Affordance) (accessed 2025-11-14)
- [Wikipedia - Ambient optic array](https://en.wikipedia.org/wiki/Ambient_optic_array) (accessed 2025-11-14)
- [Advanced Robotics - A brief review of affordance in robotic manipulation research (2017)](https://www.tandfonline.com/doi/full/10.1080/01691864.2017.1394912) (accessed 2025-11-14)
- [NIH PMC - Optic Flow: Perceiving and Acting in a 3-D World (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7869175/) (accessed 2025-11-14)
- [LSE Research Online - Gibson's affordances and Turing's theory of computation (2002)](https://eprints.lse.ac.uk/2606/1/Affordances_and_Computation_APA_style_%28LSERO%29.pdf) (accessed 2025-11-14)

**Additional References**:
- Gibson, J.J. (1966). The Senses Considered as Perceptual Systems
- Gibson, J.J. (1979). The Ecological Approach to Visual Perception
- Norman, D. (1988). The Design of Everyday Things
- Marr, D. (1982). Vision: A Computational Investigation
- Neisser, U. (1976). Cognition and Reality
