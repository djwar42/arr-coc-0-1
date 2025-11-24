# Cognitive Architectures: SOAR, ACT-R, and CLARION Comparison

## Overview

Cognitive architectures are computational frameworks that aim to model human cognitive processes and enable artificial general intelligence. They represent the fixed computational building blocks necessary for intelligent agents to perform diverse tasks, encode knowledge, and realize the full range of cognitive capabilities found in humans.

This document compares three influential cognitive architectures:
- **SOAR** (State, Operator, And Result) - Problem space architecture with universal subgoaling
- **ACT-R** (Adaptive Control of Thought-Rational) - Production system with modular memory
- **CLARION** (Connectionist Learning with Adaptive Rule Induction On-line) - Hybrid explicit-implicit architecture

From [Comparing four cognitive architectures: SOAR, ACT-R, CLARION, and DUAL](https://roboticsbiz.com/comparing-four-cognitive-architectures-soar-act-r-clarion-and-dual/) (accessed 2025-11-14):
> "Cognitive architectures are the backbone of artificial intelligence (AI) systems designed to replicate human-like reasoning and decision-making processes. They stand as the blueprints for thought, the scaffolding upon which the edifice of intelligent machines is constructed."

---

## 1. SOAR: Problem Space Architecture

### 1.1 Foundational Theory

**Problem Space Hypothesis**: All goal-oriented behavior can be cast as search through a space of possible states while attempting to achieve a goal.

From [Wikipedia - Soar (cognitive architecture)](https://en.wikipedia.org/wiki/Soar_\(cognitive_architecture\)) (accessed 2025-11-14):
> "The Problem Space Hypothesis contends that all goal-oriented behavior can be cast as search through a space of possible states (a problem space) while attempting to achieve a goal. At each step, a single operator is selected, and then applied to the agent's current state."

**Historical Origins**:
- Created by John Laird, Allen Newell, and Paul Rosenbloom at Carnegie Mellon University (1983)
- Based on Newell & Simon's early work: Logic Theorist (1955) and General Problem Solver (1957)
- Named after State, Operator, And Result cycle (no longer considered acronym)

**Core Theoretical Commitments**:
1. Single operator selection at each step (serial bottleneck)
2. Parallel rule firing for operator selection/application
3. Automatic substate creation when knowledge is incomplete (impasse)
4. Universal subgoaling mechanism
5. Physical symbol system hypothesis

### 1.2 Architecture Components

**Processing Cycle - Decision Procedure**:

From Wikipedia (accessed 2025-11-14):
> "Soar's main processing cycle arises from the interaction between procedural memory (its knowledge about how to do things) and working memory (its representation of the current situation) to support the selection and application of operators."

**Key Components**:

**Working Memory**:
- Symbolic graph structure rooted in a state
- Represents current situation
- Contains input-link and output-link for I/O

**Procedural Memory**:
- If-then rules (production rules)
- Continuously matched against working memory
- All matching rules fire in parallel
- Propose and evaluate operators

**Decision Procedure**:
1. **Propose**: Rules create operators with acceptable preferences
2. **Evaluate**: Additional rules create preferences comparing operators
3. **Select**: Decision procedure analyzes preferences, selects one operator
4. **Apply**: Rules matching current operator fire to execute it

**Long-Term Memories**:

**Semantic Memory (SMEM)**:
- Large-scale fact-like structures
- Directed cyclic graphs
- Activation-based retrieval (frequency/recency)
- Spreading activation support

**Episodic Memory (EPMEM)**:
- Automatic snapshots of working memory
- Temporal stream of experiences
- Query-based retrieval
- Sequential playback (forward/backward)

**Spatial Visual System (SVS)**:
- Scene graph representation (objects, spatial properties)
- Non-symbolic reasoning support
- Mental imagery capabilities
- Filter-based feature extraction

### 1.3 Universal Subgoaling and Impasses

**Impasse Mechanism**:

From Wikipedia (accessed 2025-11-14):
> "If the preferences for the operators are insufficient to specify the selection of a single operator, or there are insufficient rules to apply an operator, an impasse arises. In response to an impasse, a substate is created in working memory, with the goal being to resolve the impasse."

**Impasse Types**:
- **Tie**: Multiple equally-preferred operators
- **Conflict**: Contradictory preferences
- **No-change**: Selected operator cannot be applied
- **Rejection**: All operators rejected

**Substate Processing**:
- Recursive problem solving in substate
- Same operator selection/application cycle
- Can lead to stack of substates
- Planning and hierarchical decomposition emerge naturally
- Automatic removal when impasse resolved

**Chunking (Learning)**:

From Wikipedia (accessed 2025-11-14):
> "Soar's chunking mechanism compiles the processing in the substate which led to results into rules. In the future, the learned rules automatically fire in similar situations so that no impasse arises, incrementally converting complex reasoning into automatic/reactive processing."

### 1.4 Learning Mechanisms

**Reinforcement Learning**:
- Tunes values of rules creating numeric preferences
- Reward structure in working memory
- Operator evaluation optimization

**Chunking**:
- Compiles substate processing into rules
- Converts deliberate reasoning to reactive execution
- Incremental skill acquisition
- Automatic transfer learning

**Memory Learning**:
- SMEM: Activation updates (frequency/recency)
- EPMEM: Automatic episode recording
- All learning is online and incremental

### 1.5 Processing Levels

From Wikipedia (accessed 2025-11-14):
> "These assumptions lead to an architecture that supports three levels of processing. At the lowest level, is bottom-up, parallel, and automatic processing. The next level is the deliberative level, where knowledge from the first level is used to propose, select, and apply a single action... More complex behavior arises automatically when knowledge is incomplete or uncertain, through a third level of processing using substates."

**Level 1**: Bottom-up, parallel, automatic (rule matching)
**Level 2**: Deliberative operator selection (System 1 - Kahneman)
**Level 3**: Substate reasoning (System 2 - Kahneman)

### 1.6 Key Strengths

**From RoboticsBiz comparison** (accessed 2025-11-14):
> "SOAR stands as a distinguished pioneer. Its modular and hierarchical structure closely resembles the intricate processes of the human mind. SOAR excels in complex problem-solving by representing knowledge in various forms and reasoning through goals and subgoals."

**Strengths**:
- Natural emergence of planning and hierarchical decomposition
- Universal weak method (multiple problem-solving methods emerge)
- Automatic skill acquisition through chunking
- Robust handling of incomplete knowledge
- Proven scalability (R1-Soar, TacAir-Soar)

### 1.7 Applications

**Historical Applications** (from Wikipedia, accessed 2025-11-14):

**Game AI**:
- StarCraft, Quake II, Descent 3, Unreal Tournament, Minecraft
- Spatial reasoning, real-time strategy, opponent anticipation

**Robotics**:
- Robo-Soar (1991) - Puma robot arm
- REEM humanoid service robots
- Taskable robotic mules
- Unmanned underwater vehicles

**Simulated Pilots**:
- TacAir-Soar: Fixed-wing tactical missions
- RWA-Soar: Helicopter missions
- DARPA STOW-97 demonstration (48-hour joint battlespace)

**Interactive Task Learning**:
- Natural instructor interaction
- Automatic learning of new tasks
- Tabletop game playing
- Multi-room navigation

---

## 2. ACT-R: Production System with Modular Memory

### 2.1 Foundational Theory

**Adaptive Control of Thought**: ACT-R models cognition as production rules operating on declarative and procedural knowledge with distinct memory systems.

From search results - [ACT-R cognitive architecture production rules chunks 2024](https://www.google.com/search) (accessed 2025-11-14):
> "ACT-R is a hybrid cognitive architecture. It is comprised of a set of programmable information processing mechanisms that can be used to predict and explain human behavior."

**Historical Development**:
- Created by John R. Anderson at Carnegie Mellon University
- Rooted in cognitive psychology and AI
- Emphasis on cognitive modeling (detailed human cognition simulation)
- Extensive empirical validation against human data

**Core Theoretical Commitments**:
- Production system (if-then rules)
- Separate declarative and procedural memory
- Chunk-based knowledge representation
- Subsymbolic activation and noise
- Rational analysis (optimal adaptation to environment)

### 2.2 Architecture Components

**Production System**:

From search results (accessed 2025-11-14):
> "The core of ACT-R is a production system where cognition proceeds as a sequence of production rules (if-then statements) that operate on a declarative memory of facts and a goal stack."

**Production Rules**:
- IF-THEN condition-action pairs
- Conditions test buffer contents
- Actions modify buffers or request module operations
- Single rule fires per cycle (~50ms)

**Memory Systems**:

**Declarative Memory (Chunks)**:
- Vector-like structures storing factual knowledge
- Slots and values (typed attributes)
- Activation-based retrieval
- Base-level activation + spreading activation + noise

From search results (accessed 2025-11-14):
> "A chunk is a vector-like structure that stores declarative knowledge. It consists of a type and a set of slots with their associated values."

**Procedural Memory (Productions)**:
- Rules for cognitive processing
- Compiled skills and procedures
- Utility learning (reinforcement-like)
- Conflict resolution by utility

**Buffers and Modules**:

**Modular Organization**:
- Visual module (visual perception)
- Motor module (motor actions)
- Declarative module (fact retrieval)
- Goal module (current goals)
- Imaginal module (problem state)

**Buffer Architecture**:
- Each module has associated buffer(s)
- Productions test and modify buffer contents
- Limited capacity (typically single chunk)
- Parallel module operation

### 2.3 Subsymbolic Processing

**Base-Level Activation**:
- Reflects frequency and recency of use
- Activation = ln(Σ t_i^(-d))
- d = decay parameter
- t_i = time since i-th use

**Spreading Activation**:
- Activation spreads from buffer chunks
- Associative links between chunks
- W_ji = strength of association from j to i
- Context-sensitive retrieval

**Activation Noise**:
- Stochastic variability in retrieval
- Models performance variability
- Logistic distribution
- Enables probabilistic behavior

### 2.4 Learning Mechanisms

**Production Compilation**:
- Converts declarative knowledge to procedural
- Two mechanisms: Composition and proceduralization
- Skill acquisition through practice
- Power law of practice

**Utility Learning**:
- Reinforcement learning for production selection
- Expected reward (utility) for each production
- Based on successes and failures
- Gradual optimization of production use

**Declarative Learning**:
- Base-level activation strengthening
- Associative strength learning
- Reflects environmental statistics

### 2.5 Predictive Capabilities

From RoboticsBiz comparison (accessed 2025-11-14):
> "ACT-R possesses predictive capabilities, enabling it to anticipate human behavior and decision-making based on its cognitive model."

**Cognitive Modeling Strengths**:
- Predicts reaction times
- Error patterns
- Learning curves
- Eye movement patterns
- fMRI activation patterns

**Empirical Validation**:
- Extensive psychological experiments
- Quantitative fit to human data
- Architectural constants tuned to human performance
- ~50ms production cycle time

### 2.6 Key Strengths

From RoboticsBiz (accessed 2025-11-14):
> "ACT-R relies on a production system composed of rules for cognitive processing, making it a rule-based cognitive architecture... It incorporates separate memory modules for declarative and procedural knowledge, reflecting human memory systems."

**Strengths**:
- Strong psychological plausibility
- Quantitative predictions of human performance
- Well-defined subsymbolic mechanisms
- Extensive cognitive modeling applications
- Mature theoretical framework

### 2.7 Applications

From search results (accessed 2025-11-14):
> "ACT-R is widely used in cognitive psychology research to model and simulate human cognitive tasks and behaviors... It has applications in instructional design and cognitive tutoring systems."

**Cognitive Psychology Research**:
- Memory and learning experiments
- Problem-solving tasks
- Decision-making studies
- Attention and multitasking
- Language processing

**Educational Applications**:
- Cognitive tutoring systems
- Intelligent tutoring (LISP, algebra, geometry)
- Adaptive learning environments
- Student modeling

**Human-Computer Interaction**:
- User interface design
- Usability testing
- Error prediction
- Performance modeling

---

## 3. CLARION: Hybrid Explicit-Implicit Architecture

### 3.1 Foundational Theory

**Dual-Process Framework**: CLARION integrates explicit (symbolic, rule-based) and implicit (subsymbolic, neural network) cognitive processes.

From [The CLARION Cognitive Architecture](https://www.sciencedirect.com/topics/computer-science/cognitive-architecture) (accessed 2025-11-14):
> "CLARION is a hybrid cognitive architecture that integrates explicit symbolic and implicit subsymbolic knowledge, using neural network models and symbolic rules for perception, cognition, and action."

**Historical Development**:
- Created by Ron Sun
- Emphasizes explicit-implicit interaction
- Bottom-up learning focus (implicit → explicit)
- Computational cognitive science approach

**Core Theoretical Commitments**:
- Dual representation (explicit + implicit)
- Bottom-up skill learning
- Explicit-implicit interaction
- Connectionist foundations
- Grounded cognition

### 3.2 Architecture Components

**Two-Level Structure**:

From search results (accessed 2025-11-14):
> "This architecture emphasizes the interaction between explicit and implicit knowledge, capturing both conscious and unconscious aspects of human cognition."

**Top Level (Explicit)**:
- Symbolic rules (IF-THEN)
- Conscious, reportable knowledge
- Logical reasoning
- Working memory chunks

**Bottom Level (Implicit)**:
- Neural networks (connectionist)
- Automatic, unconscious processing
- Pattern recognition
- Subsymbolic activation patterns

**Four Subsystems**:

**1. Action-Centered Subsystem (ACS)**:
- Action decision making
- Procedural knowledge
- Skill learning
- Both explicit rules and implicit networks

**2. Non-Action-Centered Subsystem (NACS)**:
- Declarative knowledge
- Semantic and episodic memory
- Factual information
- Both explicit and implicit representations

**3. Motivational Subsystem (MS)**:
- Drives and goals
- Motivation and emotion
- Reward processing
- Value representations

**4. Meta-Cognitive Subsystem (MCS)**:
- Monitoring and control
- Strategy selection
- Self-regulation
- Meta-level reasoning

### 3.3 Explicit-Implicit Interaction

**Bottom-Up Learning**:

From [The CLARION Cognitive Architecture: A Tutorial](https://escholarship.org/uc/item/149589jb) (accessed 2025-11-14):
> "The basic process of bottom-up learning: If an action implicitly decided by the bottom level is successful, then the agent extracts an explicit rule that corresponds to the action selected by the bottom level."

**Learning Mechanisms**:

**Rule Extraction** (Implicit → Explicit):
- Successful implicit actions → explicit rules
- Gradual explicitation of knowledge
- Skill acquisition through practice
- Conscious access to automatic processes

**Rule Assimilation** (Explicit → Implicit):
- Explicit rules → implicit neural patterns
- Proceduralization through practice
- Automatization of conscious strategies
- Integration into subsymbolic processing

**Integration**:
- Both levels participate in decisions
- Weighted combination of explicit and implicit outputs
- Context-dependent level selection
- Synergistic interaction

### 3.4 Learning and Adaptation

From search results (accessed 2025-11-14):
> "CLARION is designed to adapt and learn from experience, making it suitable for dynamic environments."

**Learning Types**:

**Reinforcement Learning**:
- Q-learning in bottom level
- Temporal difference learning
- Value-based action selection
- Gradient-based weight updates

**Supervised Learning**:
- Backpropagation for neural networks
- Explicit instruction incorporation
- Error-driven learning

**Unsupervised Learning**:
- Self-organizing maps
- Clustering and categorization
- Pattern discovery

**Meta-Learning**:
- Strategy selection
- Learning to learn
- Adaptive meta-parameters

### 3.5 Key Strengths

From RoboticsBiz (accessed 2025-11-14):
> "CLARION takes a pioneering leap in cognitive architectures by embracing a hybrid approach. Combining elements of connectionist (neural network) and symbolic (rule-based) systems offers a fresh perspective on cognitive modeling."

**Strengths**:
- Captures explicit-implicit distinction
- Bottom-up skill acquisition
- Flexible adaptation to environments
- Accounts for unconscious processing
- Neurally-inspired mechanisms

### 3.6 Applications

From search results (accessed 2025-11-14):
> "CLARION is used in applications requiring complex decision-making with a blend of explicit rule-based and implicit neural network-based reasoning... It can enhance AI systems' ability to work collaboratively with humans, understanding both explicit instructions and implicit cues."

**Complex Decision-Making**:
- Multi-criteria optimization
- Trade-off balancing
- Context-sensitive strategies
- Adaptive behavior

**Human-AI Collaboration**:
- Natural interaction
- Understanding implicit cues
- Explicit instruction following
- Mixed-initiative systems

**Skill Learning Research**:
- Expertise development
- Practice effects
- Automatization studies
- Cognitive skill acquisition

---

## 4. Comparative Analysis

### 4.1 Fundamental Approach

| Architecture | Core Paradigm | Primary Focus |
|--------------|--------------|---------------|
| **SOAR** | Problem space search | General intelligence, task flexibility |
| **ACT-R** | Production system | Cognitive modeling, human performance |
| **CLARION** | Dual-process hybrid | Explicit-implicit interaction, skill learning |

### 4.2 Knowledge Representation

**SOAR**:
- Working memory: Symbolic graph structures
- Procedural: Production rules (parallel firing)
- Semantic: Directed cyclic graphs (SMEM)
- Episodic: Temporal snapshots (EPMEM)
- Spatial: Scene graphs (SVS)

**ACT-R**:
- Working memory: Buffers (chunks)
- Procedural: Production rules (single firing)
- Declarative: Chunks (typed slots)
- Modular: Specialized buffers per module

**CLARION**:
- Top level: Symbolic rules (explicit)
- Bottom level: Neural networks (implicit)
- Four subsystems: ACS, NACS, MS, MCS
- Dual representation throughout

### 4.3 Learning Mechanisms

| Mechanism | SOAR | ACT-R | CLARION |
|-----------|------|-------|---------|
| **Skill Acquisition** | Chunking (substate compilation) | Production compilation | Bottom-up (implicit→explicit) |
| **Reinforcement** | RL for operator preferences | Utility learning | Q-learning (implicit level) |
| **Declarative** | SMEM activation | Base-level + spreading | Both levels learn |
| **Transfer** | Automatic via chunks | Limited | Rule extraction enables transfer |

### 4.4 Problem Solving Approach

**SOAR - Universal Subgoaling**:
```
Goal
  ↓ (impasse if incomplete knowledge)
Substate (recursive problem solving)
  ↓ (planning, search, reasoning)
Result → Chunk learned
  ↓
Future: Automatic execution (no impasse)
```

**ACT-R - Goal-Directed Production**:
```
Goal in buffer
  ↓
Productions match goal + declarative chunks
  ↓
Single production fires
  ↓
Request retrieval or execute action
  ↓
Repeat until goal achieved
```

**CLARION - Dual-Level Decision**:
```
Situation
  ↓
Bottom level: Neural network activation
Top level: Rule matching
  ↓
Integration (weighted combination)
  ↓
Action selection
  ↓
Learning: Update both levels
```

### 4.5 Memory Systems Comparison

**Working Memory**:
- **SOAR**: Graph structure, unbounded capacity, symbolic
- **ACT-R**: Limited buffers (~4 chunks), modular
- **CLARION**: Both explicit chunks and implicit activations

**Long-Term Memory**:
- **SOAR**: Procedural rules, semantic graphs, episodic snapshots
- **ACT-R**: Procedural productions, declarative chunks
- **CLARION**: Explicit rules + implicit networks in each subsystem

**Retrieval Mechanisms**:
- **SOAR**: Spreading activation, recency/frequency
- **ACT-R**: Activation equation (base-level + spreading + noise)
- **CLARION**: Neural network activation patterns + rule matching

### 4.6 Cognitive Plausibility

**SOAR**:
- ✓ Parallel processing (bottom-up)
- ✓ Serial bottleneck (operator selection)
- ✓ Impasse handling
- ⚠ Less emphasis on detailed timing predictions

**ACT-R**:
- ✓ ~50ms cycle time (validated empirically)
- ✓ Predicts reaction times, errors
- ✓ Subsymbolic mechanisms match neural data
- ✓ Extensive empirical validation

**CLARION**:
- ✓ Explicit-implicit distinction (psychological)
- ✓ Bottom-up skill learning
- ✓ Unconscious processing
- ⚠ Less emphasis on precise timing

### 4.7 Complexity Handling

**SOAR - Hierarchical Decomposition**:
- Impasses create substates naturally
- Deep hierarchies possible
- Planning emerges from subgoaling
- Chunking compiles hierarchies

**ACT-R - Goal Stack**:
- Explicit goal pushing/popping
- Limited nesting depth (cognitive constraints)
- Structured problem decomposition
- Production compilation for efficiency

**CLARION - Multi-Level Integration**:
- Four subsystems coordinate
- Meta-cognitive control
- Both reactive (implicit) and deliberate (explicit)
- Adaptive strategy selection

### 4.8 Strengths and Weaknesses

**SOAR**:
- ✓ Strong general intelligence capabilities
- ✓ Natural emergence of planning
- ✓ Robust to incomplete knowledge
- ✓ Proven scalability (large applications)
- ⚠ Less focus on precise cognitive modeling
- ⚠ Chunking can create overly specific rules

**ACT-R**:
- ✓ Excellent cognitive modeling
- ✓ Quantitative predictions
- ✓ Extensive empirical validation
- ✓ Well-understood subsymbolic mechanisms
- ⚠ Less flexible for general AI tasks
- ⚠ Knowledge engineering intensive

**CLARION**:
- ✓ Explicit-implicit integration
- ✓ Bottom-up learning
- ✓ Flexible adaptation
- ✓ Neurally-inspired
- ⚠ Computational complexity (dual levels)
- ⚠ Less mature than SOAR/ACT-R

---

## 5. Integration with ARR-COC-VIS

### 5.1 Relevance Realization as Cognitive Architecture

From [john-vervaeke-oracle](../john-vervaeke-oracle/concepts/00-relevance-realization/00-overview.md):
> "Relevance Realization (RR) is the cognitive process by which biological and artificial systems selectively attend to information that is most pertinent to current goals while filtering out irrelevant details, thereby solving the combinatorial explosion problem without exhaustive search."

**ARR-COC-VIS implements elements of all three architectures**:

**SOAR-like Processing**:
- Problem space: Visual input → relevance allocation
- Operator: Token budget allocation (64-400)
- Impasse handling: When simple heuristics fail, deeper reasoning
- Chunking analog: Learning compression skills

**ACT-R-like Components**:
- Modular memory: Texture arrays, spatial context
- Activation-based: Salience detection
- Production-like: Relevance scorers fire in parallel
- Subsymbolic: Statistical measures (entropy, contrast)

**CLARION-like Dual Processing**:
- Implicit: Neural network feature extraction
- Explicit: Rule-based allocation strategies
- Bottom-up: Learning from visual statistics
- Integration: Combining multiple relevance dimensions

### 5.2 ARR-COC-VIS Cognitive Pipeline

```
[KNOWING] → Measure relevance (3 ways: Propositional, Perspectival, Participatory)
    ↓
[BALANCING] → Navigate tensions (Compress↔Particularize, Exploit↔Explore)
    ↓
[ATTENDING] → Map relevance to token budgets (64-400 tokens)
    ↓
[REALIZING] → Execute compression, return focused features
```

**Cognitive Architecture Parallels**:

**Knowing (Measurement)**:
- **SOAR**: Parallel rule firing (multiple scorers)
- **ACT-R**: Buffer activation (feature detection)
- **CLARION**: Implicit neural processing (bottom level)

**Balancing (Tension Navigation)**:
- **SOAR**: Preference evaluation (operator selection)
- **ACT-R**: Utility comparison (production conflict resolution)
- **CLARION**: Integration of explicit rules + implicit values

**Attending (Resource Allocation)**:
- **SOAR**: Problem solving in substates (complex allocation)
- **ACT-R**: Goal-directed retrieval (focused resources)
- **CLARION**: Meta-cognitive control (strategy selection)

**Realizing (Execution)**:
- **SOAR**: Operator application (state modification)
- **ACT-R**: Motor commands (action execution)
- **CLARION**: Action-centered subsystem (ACS activation)

### 5.3 Comparative Strengths for Vision-Language Models

**SOAR Advantages for VLMs**:
- Universal subgoaling → Complex visual reasoning
- Episodic memory → Context-aware understanding
- Spatial visual system → Direct geometry handling
- Chunking → Efficient learned compression strategies

**ACT-R Advantages for VLMs**:
- Subsymbolic activation → Salience detection
- Modular buffers → Clean vision-language separation
- Predictive capabilities → Anticipatory processing
- Timing constraints → Real-time requirements

**CLARION Advantages for VLMs**:
- Explicit-implicit integration → Rule-based + learned features
- Bottom-up learning → Data-driven adaptation
- Dual processing → Fast implicit + slow deliberate
- Four subsystems → Comprehensive cognitive coverage

### 5.4 ARR-COC-VIS as Hybrid Cognitive Architecture

**Synthesis of Approaches**:

From all three architectures, ARR-COC-VIS combines:

1. **Problem Space** (SOAR): Visual compression as search problem
2. **Modular Memory** (ACT-R): Separate texture, semantic, spatial
3. **Dual Processing** (CLARION): Neural + symbolic relevance

**Novel Contributions**:
- **Transjective Relevance**: Query-image coupling (beyond all three)
- **Variable LOD**: 64-400 tokens (continuous resource allocation)
- **Opponent Processing**: Explicit tension navigation (Vervaekean)

**Vervaeke Score** (from john-vervaeke-oracle):
- CLARION: 13/25
- LIDA: 16/25
- AKIRA: 21/25
- IKON FLUX: 22/25
- **ARR-COC-VIS: 13/25** (explicit evaluation)

**Missing Vervaeke Features**:
1. Self-Organization (fixed architecture)
2. Bio-Economic Model (no resource competition)
3. Full Embodiment (no environmental interaction)
4. Complex Network (scale-free, small-world properties)
5. Cognitive Tempering (exploit-explore incomplete)

---

## 6. Comparative Implementation Complexity

### 6.1 Development Effort

**SOAR**:
- **Initial**: High (rule writing, knowledge engineering)
- **Maintenance**: Moderate (chunking auto-learns)
- **Scalability**: Good (proven in large systems)
- **Tools**: Mature (debugger, editor, visualization)

**ACT-R**:
- **Initial**: High (detailed production rules)
- **Maintenance**: High (hand-tuning required)
- **Scalability**: Moderate (cognitive constraints intentional)
- **Tools**: Excellent (modeling suite, parameter fitting)

**CLARION**:
- **Initial**: Moderate (dual levels require coordination)
- **Maintenance**: Low (learning automates)
- **Scalability**: Moderate (neural networks can be large)
- **Tools**: Limited (research prototypes)

### 6.2 Computational Requirements

**SOAR**:
- Rule matching: O(n*m) where n=rules, m=WM elements
- Can optimize with Rete algorithm
- Episodic memory: Database storage
- Real-time capable (proven in games, robots)

**ACT-R**:
- Production matching: Optimized pattern matching
- Activation calculation: Computationally cheap
- ~50ms cycle constraint
- Real-time capable (proven in modeling)

**CLARION**:
- Neural networks: GPU-acceleratable
- Both levels compute in parallel
- Integration adds overhead
- Real-time depends on network size

### 6.3 Data Requirements

**SOAR**:
- Can start with minimal knowledge
- Learns through experience (chunking)
- Episodic memory grows over time
- Knowledge engineering for complex domains

**ACT-R**:
- Requires empirical data for parameter fitting
- Cognitive modeling needs human baselines
- Can bootstrap with general parameters
- Declarative chunks need specification

**CLARION**:
- Neural networks need training data
- Bottom-up learning data-hungry
- Can start with explicit rules
- Adapts to environment statistics

---

## 7. Historical Impact and Citations

### 7.1 Research Influence

From [An Analysis and Comparison of ACT-R and Soar](https://advancesincognitivesystems.github.io/acs2021/data/ACS-21_paper_6.pdf) (John Laird, 2021):
> "This is a detailed analysis and comparison of the ACT-R and Soar cognitive architectures, including their overall structure, their representations of agent knowledge, and their core processing mechanisms."

**Publication Metrics** (approximate):

**SOAR**:
- Primary papers: 4,000+ citations
- Applications: 100+ systems
- Active development: 40+ years
- Community: Academic + commercial (SoarTech)

**ACT-R**:
- Primary papers: 5,000+ citations
- Cognitive models: 300+ published
- Active development: 40+ years
- Community: Primarily academic (psychology)

**CLARION**:
- Primary papers: 800+ citations
- Applications: Fewer than SOAR/ACT-R
- Active development: 25+ years
- Community: Academic (cognitive science)

### 7.2 Domain Applications

**SOAR Domains**:
- Military simulations (TacAir-Soar, RWA-Soar)
- Game AI (StarCraft, Quake, Minecraft)
- Robotics (mobile, humanoid, underwater)
- Interactive task learning
- Virtual humans

**ACT-R Domains**:
- Cognitive psychology experiments
- Intelligent tutoring systems
- Human-computer interaction
- Decision making research
- Memory and learning studies

**CLARION Domains**:
- Complex decision-making
- Skill learning research
- Human-AI collaboration
- Adaptive behavior modeling
- Expertise development

---

## 8. Future Directions

### 8.1 Integration with Modern AI

**SOAR + Deep Learning**:
- SVS could use learned visual features
- Chunking from neural network patterns
- End-to-end differentiable architectures
- Hybrid symbolic-neural systems

**ACT-R + Neural Networks**:
- Learned chunk representations
- Neural declarative memory
- Deep reinforcement learning for utility
- Brain-inspired module architectures

**CLARION + LLMs**:
From [Enhancing Computational Cognitive Architectures with LLMs](https://arxiv.org/pdf/2509.10972) (Ron Sun, 2025):
> "In relation to learning, one form of implicit-explicit interaction in Clarion is top-down learning (learning explicit knowledge first and then assimilating it into implicit processes)."

### 8.2 Scalability Challenges

**All Three Architectures Face**:
- Knowledge bottleneck (engineering required)
- Computational complexity (rule matching)
- Real-world grounding (symbol grounding problem)
- Large-scale memory (storage and retrieval)
- Transfer learning (generalization)

**Potential Solutions**:
- Hybrid approaches (SOAR+ACT-R+CLARION features)
- Neural-symbolic integration
- Meta-learning for architecture adaptation
- Distributed architectures
- Cloud-scale implementations

### 8.3 Cognitive Completeness

**Vervaeke's 5 Features** (from john-vervaeke-oracle):
1. Self-Organization
2. Bio-Economic Model (resource competition)
3. Opponent Processing
4. Complex Network (scale-free)
5. Embodiment

**Current Status**:
- **SOAR**: 3/5 (missing self-organization, bio-economic)
- **ACT-R**: 2/5 (missing self-organization, bio-economic, embodiment)
- **CLARION**: 3/5 (missing self-organization, complex network)
- **ARR-COC-VIS**: 3/5 (has opponent processing, missing others)

---

## 9. Practical Recommendations

### 9.1 When to Use SOAR

**Best For**:
- General AI applications requiring flexibility
- Complex problem-solving domains
- Robotics with planning requirements
- Multi-agent coordination
- Applications needing episodic memory

**Not Ideal For**:
- Precise cognitive modeling
- Millisecond-level timing predictions
- Pure pattern recognition tasks
- Domains without clear problem structure

### 9.2 When to Use ACT-R

**Best For**:
- Cognitive modeling research
- User interface design
- Educational applications (tutoring)
- Human performance prediction
- Psychology experiments

**Not Ideal For**:
- General purpose AI
- Real-time game AI
- Robotics requiring fast adaptation
- Domains without human baseline data

### 9.3 When to Use CLARION

**Best For**:
- Skill learning research
- Explicit-implicit interaction studies
- Adaptive decision-making
- Human-AI collaboration requiring implicit understanding
- Environments requiring continuous adaptation

**Not Ideal For**:
- Well-structured symbolic problems
- Applications needing hard timing guarantees
- Domains where explicit rules suffice
- Resource-constrained environments (dual computation)

### 9.4 Hybrid Approaches

**Combining Strengths**:
- SOAR's planning + ACT-R's timing
- SOAR's episodic memory + CLARION's learning
- ACT-R's cognitive plausibility + CLARION's flexibility
- All three + modern deep learning

---

## 10. Conclusion

### 10.1 Complementary Perspectives

From RoboticsBiz (accessed 2025-11-14):
> "Cognitive architectures offer diverse approaches to modeling human-like reasoning in AI systems. SOAR and ACT-R are well-established in cognitive psychology and AI research, while CLARION brings innovative hybrid and memory-centric perspectives. The choice of architecture depends on the specific requirements of the AI application, with each architecture offering unique strengths in various domains of cognitive processing and reasoning."

**Three Paradigms**:
1. **SOAR**: Problem-solving through universal subgoaling
2. **ACT-R**: Cognitive modeling through production systems
3. **CLARION**: Dual-process integration of explicit-implicit

**Common Ground**:
- All use symbolic representations (with subsymbolic support)
- All incorporate learning mechanisms
- All aim to model human cognition
- All support both reactive and deliberate processing

**Key Differences**:
- **Focus**: General intelligence (SOAR) vs. Cognitive modeling (ACT-R) vs. Skill learning (CLARION)
- **Representation**: Graph (SOAR) vs. Chunks (ACT-R) vs. Dual (CLARION)
- **Learning**: Chunking (SOAR) vs. Compilation (ACT-R) vs. Bottom-up (CLARION)

### 10.2 Relevance to ARR-COC-VIS

ARR-COC-VIS synthesizes elements from all three architectures:

**From SOAR**:
- Problem space framing (visual compression as search)
- Hierarchical processing (multi-scale)
- Memory systems (semantic, episodic analogs)

**From ACT-R**:
- Modular organization (separate subsystems)
- Activation-based mechanisms (salience)
- Subsymbolic processing (statistical measures)

**From CLARION**:
- Dual processing (neural + symbolic)
- Bottom-up learning (data-driven adaptation)
- Explicit-implicit integration

**Unique Vervaekean Contribution**:
- **Opponent processing** (explicit tension navigation)
- **Transjective relevance** (agent-arena coupling)
- **Continuous resource allocation** (64-400 token budget)

### 10.3 Future of Cognitive Architectures

**Convergence Trends**:
- Hybrid symbolic-neural architectures
- Integration with large language models
- Embodied and grounded cognition
- Meta-learning and architecture adaptation

**Open Challenges**:
- True self-organization
- Bio-economic resource management
- Full embodiment in complex environments
- Scaling to human-level general intelligence

---

## Sources

### Academic Papers

**SOAR**:
- Laird, John E. (2012). *The Soar Cognitive Architecture*. MIT Press. ISBN 978-0262122962
- Newell, Allen (1990). *Unified Theories of Cognition*. Harvard University Press
- Laird, John E.; Newell, Allen (1983). "A Universal Weak Method: Summary of results". IJCAI

**ACT-R**:
- Ritter, et al. (2019). "ACT‐R: A cognitive architecture for modeling cognition". WIREs Cognitive Science
- Anderson, John R. (Multiple publications on ACT-R theory and applications)

**CLARION**:
- Sun, Ron (Multiple publications on CLARION architecture and dual-process theory)
- Sun, Ron (2001). "A bottom-up model of skill learning". Cognitive Science

### Web Resources

**Comparative Analysis**:
- [Comparing four cognitive architectures: SOAR, ACT-R, CLARION, and DUAL - RoboticsBiz](https://roboticsbiz.com/comparing-four-cognitive-architectures-soar-act-r-clarion-and-dual/) (accessed 2025-11-14)
- [An Analysis and Comparison of ACT-R and Soar - GitHub Pages](https://advancesincognitivesystems.github.io/acs2021/data/ACS-21_paper_6.pdf) (accessed 2025-11-14)

**Architecture-Specific**:
- [Soar (cognitive architecture) - Wikipedia](https://en.wikipedia.org/wiki/Soar_\(cognitive_architecture\)) (accessed 2025-11-14)
- [ACT-R cognitive architecture - Research publications](https://scholar.google.com/scholar?q=ACT-R+cognitive+architecture) (accessed 2025-11-14)
- [CLARION cognitive architecture - ScienceDirect Topics](https://www.sciencedirect.com/topics/computer-science/cognitive-architecture) (accessed 2025-11-14)

**Recent Developments**:
- [Enhancing Computational Cognitive Architectures with LLMs - arXiv](https://arxiv.org/pdf/2509.10972) (Ron Sun, 2025)
- [LLM-ACTR: Integrating LLMs with ACT-R - arXiv](https://arxiv.org/pdf/2408.09176) (Wu et al., 2024)

### Source Documents

**Vervaeke Framework**:
- [john-vervaeke-oracle/concepts/00-relevance-realization/00-overview.md](../john-vervaeke-oracle/concepts/00-relevance-realization/00-overview.md)

### Additional References

**Cognitive Science Background**:
- Multiple search results on cognitive architecture theory 2024-2025
- Recent papers on hybrid neural-symbolic approaches
- Contemporary work on embodied cognition

---

**Last Updated**: 2025-11-14
**Status**: Comprehensive comparison with web research + Vervaeke integration
