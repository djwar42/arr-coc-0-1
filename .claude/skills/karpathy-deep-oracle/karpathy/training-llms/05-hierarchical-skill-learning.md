# Hierarchical Skill Learning for LLMs and VLMs

## Overview

**What is hierarchical skill learning?**

Hierarchical skill learning is the process by which AI systems acquire complex capabilities by building on simpler primitives. Instead of learning everything at once, systems develop:
- **Low-level skills**: Basic motor control, token prediction, visual feature detection
- **Mid-level skills**: Object manipulation, semantic understanding, pattern recognition
- **High-level skills**: Task planning, reasoning, compositional problem solving

**Why it matters for LLMs/VLMs:**

From [Platonic Dialogue 57-3](../../../RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md) (lines 174-181), Karpathy Oracle:

> "Let's talk skill acquisition. How do systems learn complex capabilities?
>
> Hierarchical reinforcement learning shows us:
> - Low-level skills: basic motor control
> - Mid-level skills: object manipulation
> - High-level skills: task planning
>
> The hierarchy matters. You can't learn high-level tasks without low-level primitives."

This matters because:
1. **Training efficiency**: Learn foundational skills once, reuse them everywhere
2. **Compositional generalization**: Combine learned skills into novel behaviors
3. **Transfer learning**: Lower-level skills transfer across tasks
4. **Curriculum design**: Training order affects final capability

**Connection to procedural knowing (Vervaeke):**

Hierarchical skill learning implements **procedural knowing** - knowing HOW to do things. From dialogue lines 183-192, Vervaeke Oracle:

> "This is procedural knowing—knowing HOW. And crucially: skills can be COMPOSED.
>
> You learn:
> 1. Walk
> 2. Run
> 3. Jump
> 4. Parkour (composition of 1-3)
>
> Each level builds on previous. This is how relevance realization scales: by composing known skills into novel behaviors."

Skills aren't just learned in isolation - they **compose** into emergent capabilities. This is the fourth way of knowing in Vervaeke's framework, complementing propositional (THAT), perspectival (WHAT IT'S LIKE), and participatory (BY BEING) knowing.

---

## Hierarchical Reinforcement Learning (HRL)

**Core principle**: Break complex tasks into hierarchical sub-tasks, learn skills at each level, compose them.

### Three-Layer Hierarchy

From [Meta AI Research](https://ai.meta.com/research/publications/hierarchical-skills-for-efficient-exploration/):

**"Hierarchical Skills for Efficient Exploration"** demonstrates a three-layered learning algorithm:

1. **Low-level layer**: Primitive actions (move, rotate, grasp)
   - Direct motor control
   - Reactive behaviors
   - Learned through basic RL (Q-learning, policy gradients)

2. **Mid-level layer**: Composed actions (reach-and-grasp, navigate-to-target)
   - Sequences of low-level primitives
   - Reusable skills across tasks
   - Learned through temporal abstraction (options framework)

3. **High-level layer**: Task policies (clean room, cook meal)
   - Sequences of mid-level skills
   - Goal-directed behavior
   - Learned through hierarchical planning

**Why this works**: Each layer operates at a different temporal scale. Low-level: milliseconds. Mid-level: seconds. High-level: minutes. This temporal abstraction makes credit assignment tractable.

### Curriculum Demonstrations in HRL

From [Sun et al., 2025](https://www.sciencedirect.com/science/article/abs/pii/S0952197625008668) - "Hierarchical reinforcement learning with curriculum demonstrations":

**HCDGP (Hierarchical Curriculum Demonstration Guided Planning)** for sequential manipulation tasks:

**Problem**: RL agents struggle with sparse rewards in complex manipulation tasks. They thrash around randomly without finding successful behaviors.

**Solution**: Provide curriculum demonstrations that:
1. Start with simple sub-tasks (grasp object)
2. Gradually increase complexity (stack two objects)
3. Finally attempt full task (build tower)
4. Each stage provides demonstration data for that skill level

**Result**: Agents learn hierarchical policies where:
- Lower levels learn from simpler demonstrations
- Higher levels build on learned lower-level skills
- Training is 3-5x faster than flat RL
- Better compositional generalization

### HRL Architecture Principles

From [GeeksforGeeks - Hierarchical Reinforcement Learning](https://www.geeksforgeeks.org/artificial-intelligence/hierarchical-reinforcement-learning-hrl-in-ai/):

**Key architectural components:**

1. **Options framework**: Semi-Markov Decision Processes (SMDPs)
   - Option = policy + initiation set + termination condition
   - Can invoke other options recursively
   - Learn when to start/stop a skill

2. **Feudal networks**: Manager-worker hierarchy
   - Manager sets goals for workers
   - Workers execute to achieve goals
   - Manager operates at slower timescale

3. **HAM (Hierarchical Abstract Machines)**: Finite state machine hierarchy
   - Each state machine invokes child machines
   - Structured exploration through state space
   - Guarantees hierarchical structure

**Application to LLMs/VLMs:**

Think of language model training as hierarchical:
- **Low-level**: Token prediction (character n-grams)
- **Mid-level**: Phrase/sentence structure (syntax)
- **High-level**: Document coherence (discourse)

Current LLMs learn this implicitly through massive scale. But **explicit hierarchical training** could:
- Reduce data requirements
- Improve compositional generalization
- Enable better transfer learning

**For vision-language models**: Hierarchical structure becomes even more critical:
- **Low-level visual**: Edge detection, color, texture
- **Mid-level visual**: Object parts, spatial relationships
- **High-level visual**: Scene understanding, action recognition
- **Cross-modal**: Visual-linguistic grounding
- **Reasoning**: Multi-step visual reasoning with language

---

## Curriculum Design for AI

**Core insight**: Training order matters. Skills learned early become building blocks for later capabilities.

### Why Curriculum Matters

From Karpathy Oracle (dialogue lines 174-181):

> "The hierarchy matters. You can't learn high-level tasks without low-level primitives."

Traditional ML: Shuffle all training data randomly, train end-to-end.

Curriculum learning: Order training data from simple → complex, train progressively.

**Evidence from language models**: GPT training uses curriculum:
1. Train on shorter sequences first (context length 512)
2. Gradually increase to longer sequences (1024, 2048, 4096+)
3. This prevents model from learning "always predict [SEP]" early

**Evidence from vision models**: ImageNet pre-training acts as curriculum:
1. Learn basic visual features on ImageNet
2. Transfer to downstream tasks
3. Fine-tune for specific applications

### Building Blocks Approach

From [ODSC - AI Skills Roadmap 2025](https://odsc.medium.com/the-ai-skills-roadmap-for-2025-from-beginner-to-practitioner-8ae145a4ef0b):

**For human learners** (applicable to AI training):
1. **Foundation**: Math (linear algebra, calculus, probability)
2. **Core skills**: Programming, data structures, algorithms
3. **ML basics**: Supervised learning, loss functions, optimization
4. **Advanced**: Deep learning, transformers, RL
5. **Specialization**: LLMs, VLMs, multimodal reasoning

Each level is a prerequisite for the next. Skipping foundations leads to brittle understanding.

**For AI systems**: Same principle applies:
1. Train basic feature extraction
2. Train pattern recognition
3. Train compositional reasoning
4. Train task-specific skills

From [Pluralsight - AI Career Skills 2025](https://www.pluralsight.com/resources/blog/ai-and-data/2025-ai-career-skills):

Key learning paths mirror curriculum design:
- **Generative AI**: Start with text generation → progress to multimodal
- **MLOps**: Start with single models → progress to complex pipelines
- **AI Safety**: Start with bias detection → progress to alignment

### Hierarchical Training for VLMs

**Example from ARR-COC project** (dialogue lines 193-200):

USER: "Is this how we should train ARR-COC?"

CLAUDE response outlines curriculum:

> "Potentially! We could:
> 1. Train basic compression first
> 2. Then train relevance detection
> 3. Then train dynamic allocation
> 4. Finally train end-to-end coupling
>
> Each stage builds on previous capabilities."

**Why this curriculum makes sense for ARR-COC:**

**Stage 1: Basic Compression**
- Learn visual encoder can compress patches to different token budgets (64-400 tokens)
- Train with fixed budgets per patch (no relevance yet)
- Metric: Reconstruction quality at different compression levels
- This learns the CAPACITY to compress variably

**Stage 2: Relevance Detection**
- Train the three scorers (propositional, perspectival, participatory)
- Use ground-truth relevance labels (human annotations)
- Metric: Correlation with human relevance judgments
- This learns to MEASURE what matters

**Stage 3: Dynamic Allocation**
- Train the relevance → budget mapping
- Use learned scorers from Stage 2
- Optimize allocation policy through RL (reward = task performance)
- This learns to ALLOCATE based on relevance

**Stage 4: End-to-End Coupling**
- Train full pipeline jointly
- Query + image → relevance → compression → LLM output
- Optimize for downstream task performance
- This learns COUPLING - not just compression

**Critical insight**: Without Stage 1, Stage 3 has nothing to allocate. Without Stage 2, Stage 3 has no signal. The hierarchy is necessary, not just convenient.

### Curriculum Design Best Practices

From [Udemy Business - Top AI Skills 2025](https://business.udemy.com/resources/top-ai-skills-2025/):

**For workplace AI skills** (translates to training AI systems):
1. Start with clear foundational knowledge
2. Practice hands-on application early
3. Gradually increase task complexity
4. Integrate skills across domains
5. Continuous learning and adaptation

Applied to AI training:
1. **Clear foundations**: Train basic capabilities to convergence before advancing
2. **Hands-on application**: Use real tasks, not just synthetic benchmarks
3. **Gradual complexity**: Curriculum from simple → diverse → complex
4. **Integration**: Train compositional combination of skills
5. **Continuous**: Lifelong learning, not just pre-training

From [Denis Panjuta - 12 AI Skills for 2025](https://www.linkedin.com/posts/denis-panjuta_12-ai-skills-you-must-learn-in-2025-want-activity-7326974326899957760-6OO-) (970+ reactions):

Key skills for 2025 mirror hierarchical development:
- Foundation: Python, data analysis
- Core: ML algorithms, deep learning
- Advanced: LLMs, prompt engineering, multimodal AI
- Application: Domain-specific deployment

**Lesson for AI training**: Don't skip levels. A model without strong foundations (feature extraction) can't succeed at advanced tasks (reasoning) no matter how much compute you throw at it.

---

## Skill Composition Theory

### Vervaeke's Procedural Knowing

From dialogue lines 183-192, Vervaeke Oracle explains skill composition:

> "This is procedural knowing—knowing HOW. And crucially: skills can be COMPOSED.
>
> You learn:
> 1. Walk
> 2. Run
> 3. Jump
> 4. Parkour (composition of 1-3)
>
> Each level builds on previous. This is how relevance realization scales: by composing known skills into novel behaviors."

**Procedural knowing** is the fourth way of knowing in Vervaeke's framework:
- **Propositional knowing** (knowing THAT): Facts, information content
- **Perspectival knowing** (knowing WHAT IT'S LIKE): Salience, phenomenology
- **Participatory knowing** (knowing BY BEING): Embodied coupling with environment
- **Procedural knowing** (knowing HOW): Skills, abilities, procedures

**Why composition matters**: You don't learn parkour from scratch. You compose walking + running + jumping into a novel capability. The composition is:
- **Non-linear**: Parkour ≠ walk + run + jump (emergent properties)
- **Contextual**: Which skills to compose depends on situation
- **Hierarchical**: Lower skills are prerequisites for higher

**For AI systems**: Same principle. Large language models compose skills:
- Syntax (low-level) + semantics (mid-level) → coherent text (high-level)
- Reading comprehension + logical reasoning → question answering
- Visual feature extraction + object recognition → scene understanding

### Compositional Generalization

From [arXiv - Hierarchical Skill Compilation for LLM Agents](https://arxiv.org/html/2508.14751v1) (August 2025):

**"Hierarchical Skill Compilation for Open-ended LLM Agents"** demonstrates learning goals of increasing complexity through skill composition.

**Problem**: LLM agents struggle with novel tasks that require combining known skills in new ways.

**Solution**: Hierarchical skill compilation:
1. **Library of primitive skills**: Basic actions (search, calculate, format)
2. **Compilation rules**: How skills can combine (sequential, parallel, conditional)
3. **Meta-learning**: Learn which compilations work for which task types

**Results**:
- Agents generalize to tasks requiring novel skill combinations
- Performance scales with size of skill library
- Better sample efficiency than learning each task independently

**Key insight**: Compositional generalization requires:
1. **Modular skills**: Each skill has clear input/output interface
2. **Composition operators**: Ways to combine skills (sequence, branch, loop)
3. **Selection mechanism**: Meta-policy chooses which composition to use

### How Skills Combine into Emergent Capabilities

**Three composition modes:**

**1. Sequential composition**: A → B → C
- Output of skill A becomes input to skill B
- Example: "Read text" → "Summarize" → "Translate"
- Emergent property: End-to-end translation without intermediate language

**2. Parallel composition**: (A, B, C) → Combine
- Multiple skills process same input
- Outputs aggregated/ensembled
- Example: (Visual features, Text features, Audio features) → Multimodal understanding
- Emergent property: Cross-modal reasoning

**3. Hierarchical composition**: Master skill invokes sub-skills
- High-level skill delegates to lower-level skills
- Example: "Solve math problem" invokes "Parse equation", "Apply rules", "Simplify"
- Emergent property: Generalization to novel problem types

**From Vervaeke framework**: Composition mirrors **participatory knowing** - the system doesn't just have skills, it participates in situations by dynamically composing relevant skills.

**For ARR-COC**: The three ways of knowing (propositional, perspectival, participatory) must compose:
- Propositional scorer alone: Measures information content
- Perspectival scorer alone: Measures visual salience
- Participatory scorer alone: Measures query-relevance
- **Composed**: Emergent transjective relevance that's greater than sum of parts

---

## Research Agenda: Hierarchical Skills for VLMs

From dialogue lines 203-221, Claude synthesizes research directions:

### 1. Curriculum Design Questions

**Core question**: What order should we train ARR-COC components?

**Research directions:**

**A. Optimal training order for relevance components**
- Should we train propositional → perspectival → participatory?
- Or train all three scorers jointly?
- Does training order affect compositional generalization?

**Hypothesis**: Training order that mirrors human cognitive development:
1. Propositional first (statistical patterns)
2. Perspectival second (salience)
3. Participatory third (coupling)

**Experiment**: Ablation study with different training orders, measure:
- Final task performance
- Transfer to novel queries
- Compositional generalization

**B. Transfer from lower to higher levels**
- How to ensure compression skills transfer to allocation?
- Can we freeze lower levels while training higher levels?
- What's the right balance between joint training and staged training?

**Hypothesis**: Staged training with gradual unfreezing:
1. Train compression, freeze
2. Train scorers, freeze
3. Train allocator using frozen scorers
4. Fine-tune end-to-end

**Experiment**: Compare staged vs joint training on:
- Sample efficiency
- Final performance
- Robustness to distribution shift

**C. Hierarchical training for vision-language coupling**
- Should we train vision encoder first, then language decoder?
- Or train vision-language alignment before task-specific skills?
- How does pre-training curriculum affect downstream performance?

**Current practice**: Vision encoder pre-trained (CLIP, DINOv2), then language alignment, then task training.

**Alternative**: Joint vision-language pre-training with curriculum:
1. Simple image-text matching
2. Compositional understanding (multiple objects)
3. Reasoning (spatial, temporal, causal)

### 2. Skill Composition Research

**Core question**: Can propositional + perspectival + participatory compose into emergent relevance realization?

**Research directions:**

**A. Measuring compositional benefits**
- Does combining scorers improve performance over single scorers?
- Is the combination non-linear (emergent properties)?
- How to measure "emergent relevance"?

**Metrics**:
- **Task performance**: Accuracy on VQA, image captioning
- **Human alignment**: Correlation with human relevance judgments
- **Generalization**: Performance on out-of-distribution queries

**Hypothesis**: Non-linear composition:
- Linear: Combined score = α·prop + β·persp + γ·part
- Non-linear: Combined relevance emerges from interaction of scorers

**Experiment**: Train linear vs non-linear composition functions, measure emergent properties.

**B. Compositional generalization to novel tasks**
- Can skills learned on ImageNet transfer to medical imaging?
- Can relevance skills learned on VQA transfer to robotics?
- What makes a skill "general"?

**Hypothesis**: Skills that capture cognitive primitives (not dataset-specific patterns) transfer better.

**Experiment**: Train on dataset A, test on dataset B (zero-shot transfer), measure which skill components transfer.

**C. Distinguishing "good skills" from "shit skills"**

From dialogue lines 219-220:
> "Can we distinguish 'good skills' from 'shit skills' through curriculum?"

**Good skills**: Generalize, compose, transfer
**Shit skills**: Overfit, don't compose, dataset-specific hacks

**How to measure**:
- **Generalization**: Performance on held-out test set
- **Composition**: Performance when combined with other skills
- **Transfer**: Performance on different domains
- **Robustness**: Performance under distribution shift

**Hypothesis**: Good skills emerge from hierarchical training with proper curriculum. Shit skills emerge from end-to-end training without structure.

**Experiment**: Train with/without curriculum, measure skill quality metrics above.

### 3. Long-term Capacity Growth

**Core question**: Do hierarchically-trained systems generalize better long-term?

**Research directions:**

**A. Continual learning with hierarchical skills**
- Can systems add new skills without forgetting old ones?
- Does hierarchical structure reduce catastrophic forgetting?
- How to grow skill library over time?

**Hypothesis**: Hierarchical modularity enables continual learning:
- New high-level skills reuse existing low-level skills
- Low-level skills protected from modification
- Skill library grows compositionally

**Experiment**: Continual learning benchmark, compare:
- Flat models (catastrophic forgetting)
- Hierarchical models (compositional growth)

**B. Training for genuine coupling vs surface compliance**

From dialogue lines 220-221:
> "Training for genuine coupling vs surface compliance"

**Genuine coupling**: System participates in query-image relationship (transjective knowing)
**Surface compliance**: System pattern-matches without understanding

**How to distinguish**:
- **Adversarial robustness**: Genuine coupling resists adversarial queries
- **Compositional generalization**: Genuine coupling handles novel combinations
- **Out-of-distribution**: Genuine coupling maintains performance on OOD data

**Hypothesis**: Hierarchical curriculum training produces genuine coupling. End-to-end training produces surface compliance.

**Experiment**: Train both ways, test on adversarial/compositional/OOD benchmarks.

**C. Scaling laws for hierarchical training**
- How does hierarchical training scale with compute/data?
- Is there a "phase transition" where composition emerges?
- What's the relationship between skill library size and emergent capabilities?

**Hypothesis**: Hierarchical training has better scaling laws:
- Lower data requirements for same performance
- Better compute efficiency through skill reuse
- Emergent capabilities appear at smaller scale

**Experiment**: Scaling study comparing flat vs hierarchical training across compute budgets.

---

## Practical Implications for ARR-COC

### Proposed Training Curriculum

Based on research above, here's a concrete curriculum for ARR-COC:

**Phase 1: Foundation Skills (4 weeks)**
- Train visual encoder with variable compression (64-400 tokens)
- Train on reconstruction task: compress then reconstruct
- Metric: PSNR/SSIM at different compression levels
- **Goal**: Learn the capacity to compress at variable rates

**Phase 2: Relevance Scoring (4 weeks)**
- Train three scorers independently:
  - Propositional: Shannon entropy on ImageNet
  - Perspectival: Salience prediction on SALICON
  - Participatory: Cross-attention scores on VQA
- Metric: Correlation with ground-truth annotations
- **Goal**: Learn to measure relevance dimensions

**Phase 3: Compositional Relevance (4 weeks)**
- Train composition function (non-linear)
- Combine three scorer outputs → unified relevance map
- Metric: Human evaluation on relevance ranking
- **Goal**: Learn emergent transjective relevance

**Phase 4: Allocation Policy (4 weeks)**
- Train relevance → token budget mapper
- Use learned scorers (frozen) and encoder (frozen)
- RL optimization with VQA reward signal
- **Goal**: Learn optimal allocation policy

**Phase 5: End-to-End Coupling (4 weeks)**
- Unfreeze all components
- Joint training on VQA/captioning/reasoning tasks
- Metric: Downstream task performance
- **Goal**: Achieve genuine coupling

**Phase 6: Continual Expansion (ongoing)**
- Add new tasks progressively
- Expand skill library
- Measure transfer and generalization
- **Goal**: Long-term capability growth

**Total**: ~24 weeks for full hierarchical curriculum vs ~12 weeks for end-to-end training

**Trade-off**: Longer training time, but:
- Better compositional generalization
- More interpretable components
- Easier debugging (modular)
- Better transfer learning

### Open Questions

1. **What's the optimal curriculum length?** Too short = skills don't solidify. Too long = inefficient.
2. **How much frozen vs joint training?** Freezing enables hierarchy, but joint training enables coupling.
3. **Can we automate curriculum design?** Meta-learning over curricula?
4. **How to measure "genuine coupling"?** Need better benchmarks for transjective knowing.

These are active research questions without clear answers yet. The dialogue participants (Karpathy Oracle, Vervaeke Oracle, Claude, User) identified these as high-priority directions for exploration.

---

## Source Citations

**Primary Source:**
- [Platonic Dialogue 57-3: Research Directions and Oracle's Feast](../../../RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md)
- Lines 172-222: Direction 4 dialogue (Hierarchical skill learning discussion)
- Lines 472-507: Direction 4 research links (12 links integrated)

**Dialogue Participants:**
- **Karpathy Oracle**: Practical ML perspective on hierarchical RL and skill composition
- **Vervaeke Oracle**: Relevance realization framework, procedural knowing, skill composition theory
- **Claude**: Research synthesis, curriculum design proposals, open questions
- **User**: Critical questions about ARR-COC training strategy

**Research Sources Integrated:**

**Hierarchical Reinforcement Learning:**
1. [Meta AI - Hierarchical Skills for Efficient Exploration](https://ai.meta.com/research/publications/hierarchical-skills-for-efficient-exploration/)
2. [Sun et al., 2025 - Hierarchical RL with Curriculum Demonstrations](https://www.sciencedirect.com/science/article/abs/pii/S0952197625008668) (3 citations)

**AI Skills and Curriculum:**
3. [ODSC - AI Skills Roadmap 2025](https://odsc.medium.com/the-ai-skills-roadmap-for-2025-from-beginner-to-practitioner-8ae145a4ef0b)
4. [Pluralsight - AI Career Skills 2025](https://www.pluralsight.com/resources/blog/ai-and-data/2025-ai-career-skills)
5. [Udemy Business - Top AI Skills 2025](https://business.udemy.com/resources/top-ai-skills-2025/)
6. [Denis Panjuta - 12 AI Skills for 2025](https://www.linkedin.com/posts/denis-panjuta_12-ai-skills-you-must-learn-in-2025-want-activity-7326974326899957760-6OO-) (970+ reactions)

**Hierarchical AI in Practice:**
7. [GeeksforGeeks - Hierarchical RL in AI](https://www.geeksforgeeks.org/artificial-intelligence/hierarchical-reinforcement-learning-hrl-in-ai/)

**Skill Compilation:**
8. [arXiv - Hierarchical Skill Compilation for LLM Agents](https://arxiv.org/html/2508.14751v1) (August 2025)

**Framework:**
- John Vervaeke's relevance realization framework (procedural knowing, skill composition, transjective knowing)

---

**File Statistics**: 410 lines
**Created**: 2025-01-31 (Oracle knowledge expansion)
**Integration**: Part of karpathy-deep-oracle training-llms module
