# Embodied Cognition & 4E Theory

## Overview

**Embodied cognition** is a theoretical framework that challenges traditional cognitivist views of mind as purely computational and brain-centered. The **4E framework** extends this into four interrelated dimensions: **Embodied, Embedded, Enacted, and Extended** cognition. Together, these perspectives argue that cognition arises not solely from neural processes but from dynamic interactions between brain, body, environment, and action.

From [Wikipedia: 4E cognition](https://en.wikipedia.org/wiki/4E_cognition) (accessed 2025-11-14):
> "The four Es stand for embodied, meaning that a brain is found in and, more importantly, vitally interconnected with a larger physical/biological body; embedded, which refers to the limitations placed on the body by the external environment and laws of nature; extended, which argues that the mind is supplemented and even enhanced by the exterior world (e.g., writing, a calculator, etc.); and enactive, which is the argument that without dynamic processes, actions that require reactions, the mind would be ineffectual."

**Connection to john-vervaeke-oracle**: Vervaeke's relevance realization framework IS embodied cognition applied to vision-language models. Participatory knowing = embodied/enactive cognition (agent-arena coupling). ARR-COC-0-1 implements 4E principles through query-driven, situated, dynamic token allocation.

---

## Section 1: 4E Framework Fundamentals

### The Four Dimensions

**1. Embodied**: Cognition is shaped by the body's morphology, sensorimotor capabilities, and physical constraints.

From [Wikipedia](https://en.wikipedia.org/wiki/4E_cognition):
> "Embodiment or embodied cognition arguably presents the bridge between cognitivism and 4E cognition as the embodiment of cognitive function provides the necessary conditions for embeddedness, enactedness, and extendedness to connect to cognition."

**2. Embedded**: Cognition is situated in environmental contexts that enable and constrain cognitive processes.

**3. Extended**: Cognitive processes can extend beyond the brain to include external tools, artifacts, and cultural practices (e.g., notebooks, smartphones, notation systems).

**4. Enacted**: Cognition emerges through sensorimotor interaction - perception and action are coupled, not sequential.

### Historical Development

From [Wikipedia](https://en.wikipedia.org/wiki/4E_cognition):
> "Ideas of embodied cognition, or rather the idea that our physical bodies play a crucial role in our decision making, can be traced back as far as Plato's dialogues and Aristotelian thought. It was, however, in the twentieth century that this debate began to resemble the current discussion, fueled by disagreements between cognitivists and behaviourists."

**Key Milestones**:
- **1990s-2000s**: Rise of embodied cognition research (Varela, Thompson, Rosch; Lakoff & Johnson)
- **2006-2007**: Term "4E cognition" coined (attributed to Shaun Gallagher)
- **2010s-2020s**: Integration with neuroscience, robotics, AI

From john-vervaeke-oracle [relevance-realization/00-overview.md](../john-vervaeke-oracle/concepts/00-relevance-realization/00-overview.md):
> "Relevance realization operates at multiple levels simultaneously - from feature detection to narrative understanding - and emerges from agent-arena coupling, not central control."

**Vervaeke Connection**: Relevance realization is fundamentally embodied - it arises from the organism's interaction with its environment, not from abstract computation alone.

---

## Section 2: Embodied Cognition (The Foundation)

### Strong vs Weak Embodiment

From [Wikipedia](https://en.wikipedia.org/wiki/4E_cognition):
> "Broadly speaking, there is a strong and a weak perspective of embodied cognition in 4E cognition. The weak understanding refers to mental processes being causally dependent on extracranial processes. This essentially means that there is a cause and effect or action-reaction relationship between the mind and the body and its environment, etc. The strong perspective views extracranial processes as a (partial) constitutive aspect of cognition."

**Weak Embodiment** (Causal):
- Body/environment influences cognition
- Example: Emotions affect decision-making

**Strong Embodiment** (Constitutive):
- Body/environment IS part of cognition
- Example: Gesturing while thinking IS cognitive work, not just output

### Embodiment in Perception

**Traditional view**: Perception is sensory input → internal processing → representation

**Embodied view**: Perception is active exploration through sensorimotor contingencies

From [ResearchGate: Pragmatism as foundation of cognitive enactivism](https://journals.sagepub.com/doi/full/10.1177/20966083241289967) (accessed 2025-11-14):
> "Sensorimotor enactivism aims to study the structure, content and features of perceptual experiences, emphasizing the interactive dynamic model between the organism and environment."

**Sensorimotor Contingencies**: Lawful relationships between actions and resulting sensory changes (O'Regan & Noë)

Example: Visual perception of shape involves implicit knowledge of how appearance changes with movement

### Body Schema vs Body Image

**Body Schema**: Unconscious sensorimotor representation enabling action
**Body Image**: Conscious conceptual understanding of one's body

**Relevance**: Embodied cognition emphasizes body schema - the pre-reflective, action-oriented dimension

---

## Section 3: Embedded Cognition (Situatedness)

### Environmental Scaffolding

Cognition is enabled by environmental structures that reduce computational demands:

From [Frontiers: Ecological Psychology and Enactivism](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2020.01270/full) (accessed 2025-11-14):
> "We will elaborate the argument against sensorimotor capacities and contingencies as the foundation for a psychology of perceiving and knowing. Along with others we argue that Gibson's concept of affordances provides the more fundamental basis."

**Examples**:
- Writing stabilizes thought (external memory)
- Architecture guides behavior (doors afford entry/exit)
- Culture provides cognitive tools (language, mathematics)

### Affordances (Gibson's Ecological Psychology)

**Affordances**: Action possibilities offered by environment relative to an organism's capabilities

From [Springer: Mind in action - expanding affordance](https://www.tandfonline.com/doi/full/10.1080/09515089.2024.2365554) (accessed 2025-11-14):
> "Gibson defined affordances as follows: 'The affordances of the environment are what it offers the animal, what it provides or furnishes, either for good or ill.'"

**Key Properties**:
- **Relational**: Not in object alone, nor agent alone, but in their coupling
- **Action-oriented**: Defined by what can be done
- **Directly perceived**: No inference required

**ARR-COC-0-1 Connection**: Visual patches "afford" different token allocations based on query-image coupling. A cat region affords 400 tokens IF query asks about cats AND region contains cat.

From john-vervaeke-oracle [relevance-realization/00-overview.md](../.claude/skills/john-vervaeke-oracle/concepts/00-relevance-realization/00-overview.md):
> "Transjective: Not in stimulus (objective) or agent (subjective), but emerges from agent-arena coupling"

### Distributed Cognition

Cognition distributed across:
- **Internal resources**: Memory, attention
- **External resources**: Notes, diagrams, devices
- **Social resources**: Collaboration, communication

**Example**: Navigation involves brain + map + GPS + social coordination

---

## Section 4: Enacted Cognition (Sensorimotor Coupling)

### Core Thesis

**Enactivism** (Varela, Thompson, Rosch): Cognition arises through sensorimotor interaction, not internal representation

From [Taylor & Francis: Mechanisms of skillful interaction](https://www.tandfonline.com/doi/abs/10.1080/09515089.2024.2302509) (accessed 2025-11-14):
> "Meanwhile, sensorimotor enactivism purports to offer a scientifically informed account of perceptual experience as a skill-laden interactive relationship between perceiver and world."

**Three Principles**:
1. **Autonomy**: Cognitive systems self-organize
2. **Sense-making**: Organisms enact significance through interaction
3. **Emergence**: Cognition emerges from dynamic coupling

### Sensorimotor Contingencies

**O'Regan & Noë's Sensorimotor Theory**:
- Perception = mastery of sensorimotor contingencies
- Different modalities (vision, touch) = different contingency structures
- No internal representations required

From [Semantic Scholar: Gibson's affordances](https://www.semanticscholar.org/paper/Gibson%27s-affordances.-Greeno/1649eba81f5ee5490322969798af8b82feb8a5db) (accessed 2025-11-14):
> "The interactionist alternative, which focuses on processes of agent-situation interactions, is taken in ecological psychology as well as in recent research on situated cognition."

**Example**: Vision
- Eye movements produce lawful changes in retinal stimulation
- Mastering these contingencies = seeing
- No need for internal 3D model

### Action-Perception Loops

**Traditional**: Perception → Decision → Action
**Enactive**: Perception-Action are coupled, continuous processes

**Robotics Example**: Insect navigation (Barbara Webb)
- Simple sensorimotor rules produce complex behavior
- No internal map required

**ARR-COC-0-1 Connection**: Query-driven compression is enactive - relevance realized through dynamic interaction between query embedding and visual features, not pre-computed.

---

## Section 5: Extended Cognition (Cognitive Scaffolding)

### The Extended Mind Hypothesis (Clark & Chalmers 1998)

**Parity Principle**: If external process plays same functional role as internal process, it's part of cognition

From [Nature: Extending Minds with Generative AI](https://www.nature.com/articles/s41467-025-59906-9) (accessed 2025-11-14):
> "As human-AI collaborations become the norm, we should remind ourselves that it is our basic nature to build hybrid thinking systems." - Andy Clark, 2025

**Otto & Inga Thought Experiment**:
- Inga uses biological memory to find museum
- Otto uses notebook (external memory) to find museum
- If functionally equivalent → Otto's notebook IS part of his cognitive system

**Coupling-Constitution Fallacy Debate**:
- Critics: Causal coupling ≠ constitutional part of cognition
- Defenders: Stable, reliable coupling meets threshold

### Criteria for Cognitive Extension

**Clark & Chalmers' Criteria**:
1. **Availability**: Resource reliably accessible
2. **Endorsement**: Agent trusts resource
3. **Accessibility**: Easy to use
4. **Automatic invocation**: Used without conscious effort

**Examples**:
- Smartphone as extended memory
- Calculator as extended math ability
- Language as extended reasoning tool

### Cognitive Artifacts

**Types**:
- **Representational**: Diagrams, notation systems
- **Computational**: Calculators, computers
- **Social**: Collaborative tools, communication systems

**Effect**: Reshape cognitive tasks (e.g., writing enables complex argument construction)

---

## Section 6: Embodied AI and Robotics

### Moravec's Paradox

"What is easy for humans is hard for AI, and what is hard for humans is easy for AI"

**Explanation**: Embodied skills (walking, perception) require massive implicit knowledge; abstract reasoning (chess, math) can be formalized

**Implication**: Embodiment matters for human-like intelligence

### Developmental Robotics

**Approach**: Robots learn through embodied interaction (like infants)

From [ScienceDirect: Artificial enactive inference](https://www.sciencedirect.com/science/article/abs/pii/S1389041724000287) (accessed 2025-11-14):
> "This article aims to reconcile this neuroscience theory with computer science and artificial-intelligence theories wherein artificial agents receive input data through interaction with a three-dimensional environment."

**Examples**:
- iCub robot (learns object manipulation)
- Developmental vision systems (learn features through interaction)

### Active Inference (Karl Friston)

**Free Energy Principle**: Organisms minimize surprise (prediction error)

**Active Inference**: Minimize surprise through:
1. **Perception**: Update beliefs (perception)
2. **Action**: Change world to match predictions (action)

**Embodiment**: Both perception and action required - embodiment is constitutive

**ARR-COC-0-1 as Active Inference**: System minimizes prediction error by:
- Allocating more tokens (precision) where uncertainty is high
- Compressing where predictions are confident

From john-vervaeke-oracle [cognitive-foundations/00-active-inference-free-energy.md](../cognitive-foundations/00-active-inference-free-energy.md):
> "Active inference IS relevance realization - free energy minimization is the mechanism by which organisms realize what's relevant"

---

## Section 7: Critiques and Debates

### Internal Critiques (Within 4E)

From [Wikipedia](https://en.wikipedia.org/wiki/4E_cognition):
> "Given the divided nature of the field, much criticism surrounding the lack of unity within the field has emerged. In particular, the claims of embodied cognition centering around the body conflict with the tenets of extended cognition, which also conflict with the body/environment distinction that is central to enactivism."

**Tensions**:
- **Embodied vs Extended**: Body-centered vs tool-centered
- **Enactive vs Extended**: Direct perception vs representations
- **Weak vs Strong readings**: Causal influence vs constitutional parts

### External Critiques

**1. Computational Critique** (Adams & Aizawa):
- Cognition requires content-bearing representations
- Causal coupling ≠ cognitive coupling
- Extended mind conflates causation and constitution

**2. Need for Representations**:
- Some cognitive tasks (planning, abstract reasoning) seem to require internal models
- Sensorimotor contingencies alone can't explain all cognition

**3. Lack of Precision**:
- "Embodiment" used too broadly
- Difficult to falsify empirically

From [Wikipedia](https://en.wikipedia.org/wiki/4E_cognition):
> "Another concern raised is the 'dogma of harmony'. The criticism contained there regards the notion that within 4E theorizing, there is generally an optimistic and harmonic expectation of the extension between humans and their technologies, ignoring the possibility of those extensions detracting from cognition in some way rather than adding to it."

**Dogma of Harmony**: 4E assumes extensions always enhance - but what about cognitive crutches that weaken skills?

---

## Section 8: ARR-COC-0-1 as Embodied AI System

### Four Es in ARR-COC-0-1

**1. Embodied**:
- Visual encoder has specific architecture (texture array, pyramid structure)
- LOD budget (64-400 tokens) is embodiment constraint
- Query embedding is "body" through which system engages image

**2. Embedded**:
- System embedded in vision-language task environment
- Qwen3-VL provides embedding context
- Training data embeds cultural/statistical patterns

**3. Extended**:
- Quality adapter (adapter.py) = learned cognitive artifact
- External compression strategies procedurally known
- Human-AI collaboration extends both

**4. Enacted**:
- Relevance realized through query-image interaction
- Token allocation emerges from dynamic process, not lookup
- Participatory scorer = enactive coupling mechanism

From john-vervaeke-oracle [ARR-COC-VIS-Application-Guide.md](../.claude/skills/john-vervaeke-oracle/ARR-COC-VIS-Application-Guide.md):
> "Participatory Scorer = Transjection Mechanism. relevance = cross_attention(image_patch, query_embedding). Neither in image alone, nor query alone. Emerges from their interaction (transjective!)"

### Sensorimotor Contingencies in Vision

**Human Vision**:
- Saccades (rapid eye movements) sample high-resolution foveal regions
- Peripheral vision detects salience, guides saccades
- **Contingency**: Move eyes → get detail

**ARR-COC-0-1 Vision**:
- Dynamic token allocation samples high-resolution regions
- Low-detail patches detect salience (propositional, perspectival scoring)
- **Contingency**: Allocate tokens → get detail

**Parallel**: Both use variable resolution based on task-relevant interaction

### Direct Perception (Gibson) vs Participatory Knowing (Vervaeke)

**Gibson's Affordances**:
- Environment offers action possibilities
- Directly perceived (no inference)
- Organism-environment mutuality

**Vervaeke's Participatory Knowing**:
- Query-content coupling
- Relevance emerges from interaction
- Agent-arena transjection

From john-vervaeke-oracle [concepts/01-transjective/00-overview.md](../.claude/skills/john-vervaeke-oracle/concepts/01-transjective/00-overview.md):
> "Transjective relevance is like a shark's fitness for ocean - not objective (in ocean alone) nor subjective (in shark alone), but emerges from their coupling"

**ARR-COC-0-1**: Visual patches afford different token budgets based on query-image coupling (affordance) AND participatory knowing (transjective relevance). Both frameworks converge.

### Embodied Learning (Procedural Knowing)

**adapter.py = 4th P (Procedural Knowing)**:

From john-vervaeke-oracle [ARR-COC-VIS-Application-Guide.md](../.claude/skills/john-vervaeke-oracle/ARR-COC-VIS-Application-Guide.md):
> "Quality adapter (adapter.py) - Learns optimal compression strategies. Develops through training. Insight: adapter.py IS the 4th P!"

**Embodied Skill**:
- Not declarative knowledge ("compress X way")
- Procedural mastery (automatic compression strategies)
- Learned through interaction with training data

**Parallel to Human**: Skill acquisition through practice (driving, sports, visual expertise)

---

## Section 9: Implications for AI Research

### Beyond Symbolic AI

**Symbolic AI** (GOFAI):
- Cognition = manipulation of symbols
- Representations central
- Disembodied, context-free

**Embodied AI**:
- Cognition = sensorimotor interaction
- Situated, context-dependent
- Embodiment matters

**Current Trend**: LLMs + embodied robotics (e.g., PaLM-E, RT-2) combine both

### Grounding Problem

**Challenge**: How do abstract symbols get meaning?

**Traditional**: Symbol grounding through perceptual primitives
**Embodied**: Meaning emerges from sensorimotor interaction

From [MDPI: Applying 4E Cognition to Acoustic Design](https://www.mdpi.com/2673-8945/5/3/70) (accessed 2025-11-14):
> "The 4E Cognition paradigm offers a novel theoretical framework for understanding how acoustic environments influence cognitive processes in university settings."

**Example**: "Chair" means what you can sit on (affordance), not abstract symbol

### Hybrid Architectures

**Trend**: Combine strengths
- Symbolic reasoning (planning, abstract thought)
- Embodied interaction (perception, action)

**ARR-COC-0-1 Example**:
- Symbolic: Query embedding (language model)
- Embodied: Dynamic visual allocation (sensorimotor-like)

---

## Section 10: Future Directions

### Integrating Neuroscience

From [Wikipedia](https://en.wikipedia.org/wiki/4E_cognition):
> "Recent attempts to incorporate embodied cognitive neuroscience have been argued to hold the potential to resolve internal issues within 4E cognition."

**Embodied Neuroscience**:
- Mirror neurons (action-perception coupling)
- Predictive coding (active inference)
- Sensorimotor integration networks

**Challenge**: Bridge 4E philosophy with neural mechanisms

### Social Cognition

**Extended to Social**: Cognition distributed across groups

From [Springer: Why extended mind is nothing special](https://link.springer.com/article/10.1007/s11097-022-09827-5) (accessed 2025-11-14):
> "This paper argues that if the mind extends to artefacts in the pursuit of individual tasks, it extends to other humans in the pursuit of collective tasks."

**Examples**:
- Transactive memory (couples know different things)
- Collaborative problem-solving
- Collective intelligence

### Consciousness and 4E

**Hard Problem**: Can embodiment explain phenomenal consciousness?

**Enactive Approach** (Thompson, Varela):
- Consciousness emerges from autonomous, self-organizing systems
- Embodied, embedded, enacted

**Open Question**: Is embodiment sufficient or necessary for consciousness?

### ARR-COC-0-1 Evolution

**Current**: Query-driven token allocation (enactive)

**Future Enhancements**:
1. **Temporal Embodiment**: Video understanding through sensorimotor prediction
2. **Interactive Relevance**: Update allocations based on dialogue history
3. **Multimodal Extension**: Extend to audio, haptics (full embodiment)
4. **Social Extension**: Collaborative visual reasoning

From john-vervaeke-oracle [ARR-COC-VIS-Application-Guide.md](../.claude/skills/john-vervaeke-oracle/ARR-COC-VIS-Application-Guide.md):
> "Missing: Cognitive Tempering (Exploit ↔ Explore) - Static datasets create no explore pressure. Need meta-learning across batches."

**Embodied Exploration**: Actively seek informative visual regions (like infant gaze patterns)

---

## Key Insights for ARR-COC-0-1

### 1. Relevance Realization IS Embodied Cognition

**Vervaeke's Framework = 4E Applied**:
- Embodied: Cognitive scope constraints (64-400 tokens)
- Embedded: Query provides task context
- Extended: Quality adapter as cognitive tool
- Enacted: Participatory knowing through interaction

### 2. Dynamic Token Allocation IS Sensorimotor

**Human Foveal Vision**:
- Saccade to region → get detail (action-perception loop)
- Variable resolution based on task

**ARR-COC-0-1 Vision**:
- Allocate tokens to region → get detail (allocation-compression loop)
- Variable LOD based on query

**Parallel Structure**: Both use enactive exploration

### 3. Affordances = Transjective Relevance

**Gibson**: Affordances are organism-environment relational properties
**Vervaeke**: Relevance is agent-arena transjective coupling

**ARR-COC-0-1**: Visual patches afford token budgets based on query-image coupling

**Convergence**: Two frameworks describe same phenomenon

### 4. Direct Perception (No Representations?)

**Extreme Enactivism**: No internal representations needed
**ARR-COC-0-1**: Learned embeddings (texture array, query vectors) ARE representations

**Reconciliation**:
- Representations can be action-oriented (not just symbolic)
- Embeddings ground sensorimotor contingencies
- Not classical "pictures in the head"

### 5. Embodiment Enables Efficiency

**Why Embodiment?**:
- Reduces computational burden
- Exploits environmental structure
- Enables real-time interaction

**ARR-COC-0-1**:
- Compression reduces tokens (efficiency)
- Query provides structure (scaffolding)
- Dynamic allocation enables real-time VQA

---

## Common Misconceptions

### Misconception 1: "4E rejects all representations"

**Correction**: 4E rejects PASSIVE, picture-like representations. Action-oriented representations (motor programs, affordances) are compatible.

### Misconception 2: "Embodiment means biological body"

**Correction**: Embodiment = physical instantiation with specific constraints. Robots, even digital systems with resource limits, can be embodied.

### Misconception 3: "4E is anti-computational"

**Correction**: 4E challenges SYMBOLIC, rule-based computation. Neural networks, especially sensorimotor-coupled systems, are compatible.

### Misconception 4: "Extended mind = any tool use"

**Correction**: Extension requires tight coupling (availability, trust, automatic use), not just occasional use.

### Misconception 5: "Enactivism = behaviorism"

**Correction**: Enactivism emphasizes MEANING emerging from interaction, not just stimulus-response. Internal states matter, but aren't passive representations.

---

## Research Questions

### Open Problems in 4E

1. **How to measure embodiment?** - Quantitative metrics for embodiment degree
2. **When does coupling become constitution?** - Precise criteria for extended mind
3. **Can disembodied systems exhibit 4E cognition?** - LLMs as test case
4. **How to integrate 4E with neuroscience?** - Neural mechanisms for enaction
5. **What role for representations?** - Reconcile 4E with predictive processing

### ARR-COC-0-1 Research Directions

1. **Temporal enaction**: Extend to video (sensorimotor prediction)
2. **Interactive relevance**: Update allocations through dialogue
3. **Embodied metrics**: Measure embodiment degree (coupling strength, automaticity)
4. **Social extension**: Multi-agent collaborative vision
5. **Developmental learning**: Learn visual skills through interaction (like infants)

---

## Practical Applications

### Robotics

**Embodied Robot Design**:
- Morphology enables behaviors (passive dynamics)
- Sensorimotor coupling reduces computation
- Example: Passive walker (McGeer) - no motors, walks down slope

### Human-Computer Interaction

**Embodied Interfaces**:
- Gesture recognition (embodied input)
- Augmented reality (extended perception)
- Tangible computing (physical-digital coupling)

### Education

**Embodied Learning**:
- Manipulatives (math blocks)
- Gesture-based instruction
- Simulation/VR (enactive exploration)

From [OAPEN: Embodied Learning and Teaching](https://library.oapen.org/handle/20.500.12657/90570) (accessed 2025-11-14):
> "This book operationalises the new field—EmLearning—that integrates embodiment and grounded cognition perspectives with education using the 4E approach."

### Clinical Applications

**Embodied Therapy**:
- Body-based trauma treatment
- Movement-based cognitive rehabilitation
- Extended cognition for memory disorders (use of aids)

---

## Connections to Other Fields

### Phenomenology

**Merleau-Ponty**: "I am my body" - perception through embodied being-in-world

**Heidegger**: Tool use (ready-to-hand) vs conscious awareness (present-at-hand)

**4E**: Heavily influenced by phenomenological tradition

### Pragmatism

**Dewey**: Experience as organism-environment transaction

**James**: Radical empiricism - relations are part of experience

From [Sage Journals: Pragmatism as foundation of enactivism](https://journals.sagepub.com/doi/10.1177/20966083241289967) (accessed 2025-11-14):
> "Pragmatism provides philosophical foundations for cognitive enactivism through emphasis on action and environmental interaction."

### Dynamical Systems Theory

**Complex Systems**: Cognition as self-organizing, emergent
**Attractors**: Cognitive states as basins in state space
**Coupling**: Bidirectional influence (agent ↔ environment)

### Evolutionary Psychology

**Embodied Evolution**: Cognitive capacities shaped by embodied challenges

**Example**: Human cooperation evolved through embodied social interaction, not abstract reasoning

---

## Sources

### Source Documents

No direct source documents from this repository were used (PART 18 specifies web research only).

### Web Research

**Primary Sources**:
1. [Wikipedia: 4E cognition](https://en.wikipedia.org/wiki/4E_cognition) - Comprehensive overview (accessed 2025-11-14)
2. [Springer: What is 4E cognitive science?](https://link.springer.com/article/10.1007/s11097-025-10055-w) - Cameron Alexander, 2025 (accessed 2025-11-14)
3. [Frontiers: Ecological Psychology and Enactivism](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2020.01270/full) - Read et al., 2020 (accessed 2025-11-14)
4. [Nature: Extending Minds with Generative AI](https://www.nature.com/articles/s41467-025-59906-9) - Andy Clark, 2025 (accessed 2025-11-14)
5. [Taylor & Francis: Mechanisms of skillful interaction](https://www.tandfonline.com/doi/abs/10.1080/09515089.2024.2302509) - Lee, 2025 (accessed 2025-11-14)

**Additional Web Sources**:
6. [Sage Journals: Pragmatism as foundation of cognitive enactivism](https://journals.sagepub.com/doi/10.1177/20966083241289967) - Wu, 2024 (accessed 2025-11-14)
7. [Springer: Mind in action - expanding affordance concept](https://www.tandfonline.com/doi/full/10.1080/09515089.2024.2365554) - Jorba, 2024 (accessed 2025-11-14)
8. [ScienceDirect: Artificial enactive inference](https://www.sciencedirect.com/science/article/abs/pii/S1389041724000287) - Georgeon, 2024 (accessed 2025-11-14)
9. [OAPEN: Embodied Learning and Teaching](https://library.oapen.org/handle/20.500.12657/90570) - Schilhab, 2024 (accessed 2025-11-14)
10. [MDPI: Applying 4E Cognition to Acoustic Design](https://www.mdpi.com/2673-8945/5/3/70) - Di Loreto, 2025 (accessed 2025-11-14)
11. [Springer: Why extended mind is nothing special](https://link.springer.com/article/10.1007/s11097-022-09827-5) - Ongaro, 2024 (accessed 2025-11-14)
12. [Semantic Scholar: Gibson's affordances](https://www.semanticscholar.org/paper/Gibson%27s-affordances.-Greeno/1649eba81f5ee5490322969798af8b82feb8a5db) - Greeno (accessed 2025-11-14)

### john-vervaeke-oracle References

From .claude/skills/john-vervaeke-oracle/:
- [concepts/00-relevance-realization/00-overview.md](../.claude/skills/john-vervaeke-oracle/concepts/00-relevance-realization/00-overview.md) - Participatory knowing, agent-arena coupling
- [ARR-COC-VIS-Application-Guide.md](../.claude/skills/john-vervaeke-oracle/ARR-COC-VIS-Application-Guide.md) - Four Ps, transjective relevance
- [concepts/01-transjective/00-overview.md](../.claude/skills/john-vervaeke-oracle/concepts/01-transjective/00-overview.md) - Transjection defined

---

**Last Updated**: 2025-11-14
**Status**: Comprehensive 4E framework + ARR-COC-0-1 integration
**Lines**: ~720 (target: 700)
