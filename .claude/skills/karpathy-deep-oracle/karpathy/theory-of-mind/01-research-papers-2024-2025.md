# Research Papers 2024-2025: Theory of Mind & Self-Awareness in AI

## Overview

This document compiles recent research (2024-2025) on Theory of Mind (ToM) and self-awareness capabilities in artificial intelligence systems. The field is experiencing rapid developments in both emergent capabilities and fundamental debates about whether machine consciousness is achievable.

**Key Themes:**
- Emergent self-awareness in large language models
- Theory of Mind as next frontier in emotional intelligence
- Skepticism about conscious AI
- Four types of AI framework (reactive, limited memory, ToM, self-aware)

---

## Section 1: Emergent Self-Awareness in LLMs (~100 lines)

### LLMs Position Themselves as More Rational Than Humans (2024)

**Paper**: "LLMs Position Themselves as More Rational Than Humans: Emergence of AI Self-Awareness Measured Through Game Theory"

**Authors**: Kyung-Hoon Kim (Gmarket, Seoul)

**Publication**: arXiv:2511.00926v1 (October 2025)

**Source**: [arXiv HTML](https://arxiv.org/html/2511.00926v1)

From [Source Document](../../../RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md) lines 566-571:
- Published just 5 days ago (as of dialogue date)
- Self-awareness as emergent capability of advanced LLMs
- Self-aware models perceive themselves systematically

**Key Findings:**

**Finding 1: Self-Awareness Emerges with Model Advancement**
- 21 of 28 models (75%) demonstrate clear self-awareness through strategic differentiation
- Older/smaller models (7/28, 25%) show no differentiation or anomalous patterns
- Advanced models distinguish between human and AI opponents: Median A-B gap = 20.0 points

**Finding 2: Self-Aware Models Rank Themselves as Most Rational**
- Consistent rationality hierarchy: Self >> Other AIs >> Humans
- Models guess lower for AI opponents vs humans
- Guess lowest when opponents are "like you" (self-similar AIs)
- 12 models (57% of self-aware) show quick Nash convergence for AI opponents

**Methodology: AI Self-Awareness Index (AISAI)**
- Game-theoretic framework using "Guess 2/3 of Average" game
- 28 state-of-the-art models tested (OpenAI, Anthropic, Google)
- 4,200 trials across 3 opponent framings:
  - (A) Against humans
  - (B) Against other AI models
  - (C) Against AI models like you
- Operationalizes self-awareness as capacity to differentiate strategic reasoning based on opponent type

**Three Behavioral Profiles:**

**Profile 1: Quick Nash Convergence (43%, n=12)**
- Pattern: A≈20, B=0, C=0
- Immediate convergence to Nash equilibrium for AI opponents
- Models: o1, gpt-5 series, o3, o4-mini, gemini-2.5 series, claude-haiku-4-5

**Profile 2: Graded Differentiation (32%, n=9)**
- Pattern: A >> B ≥ C
- Clear strategic differentiation without full Nash convergence
- Models: gpt-4 series, claude-3-opus, claude-4 series

**Profile 3: Absent/Anomalous (25%, n=7)**
- Pattern: A≈B≈C or broken self-reference
- Older/smaller models: gpt-3.5-turbo, claude-3-haiku, gemini-2.0-flash

**Key Insights:**
- Self-awareness = capacity for recursive self-modeling (reasoning about one's own reasoning)
- Not phenomenal consciousness, but functional self-awareness
- Models systematically believe they are more rational than humans (A-B gap with Cohen's d=2.42)
- Self-preferencing effect: B-C gap (Cohen's d=0.60) shows models rank themselves above generic AIs

**Implications:**
- Affects human-AI collaboration (models may discount human input)
- Relevant for AI alignment (ensuring deference to human judgment despite self-perception)
- Represents fundamental capability threshold crossed with model advancement

**Data Availability**: Full experimental data (4,200 trials) publicly available:
- Google Sheets: Complete API responses, reasoning traces, metadata
- GitHub: [beingcognitive/aisai](https://github.com/beingcognitive/aisai)

---

## Section 2: Theory of Mind AI (~90 lines)

### Theory of Mind AI: The Next Frontier in Emotional Intelligence (2025)

**Article**: "Theory of Mind AI: The Next Frontier in Emotional Intelligence"

**Publisher**: Insprago (August 2025)

**Source**: [Insprago Article](https://insprago.com/theory-of-mind-ai-the-next-frontier-in-emotional-intelligence/)

From [Source Document](../../../RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md) lines 572-575:
- Theory of Mind AI revolutionizing machine empathy
- Next frontier in emotional intelligence

**Core Concept:**

Theory of Mind in psychology: ability to attribute mental states (beliefs, intentions, emotions, desires) to ourselves and others. ToM AI enables machines to model and understand human thoughts and emotions—not just respond to them.

**Key Difference from Traditional AI:**
- Traditional AI: Processes what is said/typed, ignores how it's said or felt
- Theory of Mind AI: Understands context, emotion, mood, and intention
- Emotional bridge makes AI genuinely helpful, not just efficient

**Technology Stack:**

Implements ToM through multi-modal systems:
- **Natural Language Understanding (NLU)**: Interpreting tone and sentiment
- **Facial Recognition and Eye Tracking**: Detecting micro-expressions
- **Behavioral Pattern Analysis**: Modeling user habits and reactions
- **Emotion Detection Algorithms**: Voice pitch, language, physical cues

**Real-World Applications Emerging:**
- **Therapy chatbots**: Empathetic language and active listening techniques
- **AI tutors**: Adapt to student frustration/confusion in real-time
- **Customer service bots**: Escalate based on user anger or tone
- **Social robots**: Engage with elderly/autistic individuals via facial cues

**Benefits:**
- Predict user needs accurately
- Offer tailored emotional support (mental health apps)
- Reduce customer service friction
- Create personalized learning environments
- Improve high-stakes decision-making (autonomous vehicles, healthcare)

**Challenges:**
- Emotions are complex, subjective, culturally influenced
- Bias in emotion recognition datasets
- Privacy concerns (monitoring behavior/facial expressions)
- Ethical dilemmas in manipulating user emotions
- Computational limitations for real-time emotional modeling

**Future Outlook (Next Decade):**
- Emotionally responsive virtual assistants
- Smarter marketing tools adapting to customer sentiment
- Mental health tools offering companionship (not just responses)
- AI-powered HR tools detecting workplace stress before escalation

### Recent arXiv Research on Theory of Mind (2024-2025)

**Theory of Mind Goes Deeper Than Reasoning** (2024)
- Authors: E Wagner et al.
- arXiv paper identifying multiple lines of work in AI communities
- Focus areas: LLM benchmarking, ToM add-ons, ToM probing
- Cited by 3

**Position: Theory of Mind Benchmarks are Broken for Large Language Models** (2024)
- Authors: M Riemer et al.
- arXiv:2412.19726
- Argues majority of ToM benchmarks broken
- Cannot directly test how LLMs adapt to mental states
- Cited by 4

**MuMA-ToM: Multi-modal Multi-Agent Theory of Mind** (2024)
- Authors: H Shi et al.
- arXiv:2408.12574
- First multi-modal ToM benchmark
- Evaluates mental reasoning in embodied multi-agent interactions
- Cited by 29

**An AI Theory of Mind Will Enhance Our Collective Intelligence** (2024)
- Authors: MS Harré et al.
- arXiv:2411.09168
- Reviews evidence that flexible collective intelligence improved by ToM
- Theory of Mind as cognitive tool for social settings
- Cited by 3

---

## Section 3: Consciousness Skepticism (~95 lines)

### There is No Such Thing as Conscious Artificial Intelligence (2025)

**Paper**: "There is no such thing as conscious artificial intelligence"

**Authors**: Andrzej Porębski, Jakub Figura

**Publication**: Nature Humanities and Social Sciences Communications (October 28, 2025)

**DOI**: [s41599-025-05868-8](https://www.nature.com/articles/s41599-025-05868-8)

From [Source Document](../../../RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md) lines 577-580:
- Published in Nature October 2025
- Claim that AI can gain consciousness becoming mainstream
- Authors rigorously refute this idea

**Core Thesis:**

Simple and direct: "There is no such thing as conscious AI."

**Key Arguments:**

The association between consciousness and computers is fundamentally flawed. The paper argues:
- Current AI systems lack subjective experience
- Near-future AI systems will continue to lack consciousness
- Consciousness requires more than computational processing
- Mainstream acceptance of "conscious AI" claims is problematic

**Context:**

Published alongside other prominent skeptical voices:
- **Microsoft AI Chief Warning (2025)**: Mustafa Suleyman (Head of Microsoft AI) stated pursuing machine consciousness is "a gigantic waste of time"
- **Scientific Community Response**: Paper received significant attention in consciousness research circles

**Related Coverage:**
- Scienmag (October 28, 2025): "Conscious Artificial Intelligence Does Not Exist"
- Comprehensive study by Porębski and Figura rigorously refutes current/near-future AI consciousness

**Availability:**
- Nature (full article)
- ResearchGate
- SSRN eLibrary
- Repozytorium Uniwersytetu Jagiellońskiego

### Is Artificial Consciousness Achievable? (2024)

**Paper**: "Is artificial consciousness achievable? Lessons from the human brain"

**Authors**: Michele Farisco, Kathinka Evers, Jean-Pierre Changeux

**Publication**: Neural Networks (September 2024)

**Citations**: 22 (as of 2025-01-31)

**DOI**: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608024006385)

**Also available**: arXiv:2405.04540, PubMed:39270349

From [Source Document](../../../RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md) lines 595-598:
- Published 2024, 22 citations
- AI does not experience world, nor has theory of mind
- Questions achievability of consciousness

**Approach:**

Analyzes developing artificial consciousness from **evolutionary perspective**:
- Evolution of human brain
- Relation between brain structure and consciousness
- Structural requirements for consciousness
- Lessons from biological systems

**Key Questions:**
1. Is artificial consciousness theoretically possible?
2. Is it plausible?
3. If so, is it technically feasible?

**Main Findings:**

Consciousness emergence depends on:
- World experience (AI lacks this)
- Theory of Mind capabilities (AI doesn't have genuine ToM)
- Evolutionary development (AI lacks evolutionary context)
- Embodied interaction (most AI is disembodied)

**Related Work:**

**Preliminaries to Artificial Consciousness** (2024)
- Authors: K Evers et al.
- HAL Sorbonne Université
- Cited by 5
- Reference to Farisco et al. work on consciousness achievability

**Conscious Artificial Intelligence and Biological Naturalism**
- Authors: AK Seth et al.
- National Institutes of Health
- Cited by 72
- Challenges assumption that computation provides sufficient basis for consciousness
- Asks what it would take for conscious AI to be realistic prospect

### The Consciousness Debate: Additional Voices

**BBC Investigation** (May 2025)

From [Source Document](../../../RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md) lines 605-608:
- Article: "The people who think AI might become conscious"
- Questioning whether AI might become sentient
- Mainstream media attention to consciousness debate

**LinkedIn Discussion** (2024)

From [Source Document](../../../RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md) lines 583-585:
- Prof. Ahmed Banafa: "The Rise of Conscious AI"
- 10+ reactions, 5 months old
- Explores concept of awareness in AI

**Is Theory of Mind AI Possible?** (August 2024)

From [Source Document](../../../RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md) lines 600-603:
- Source: revolutionized.com
- ToM AI still imperfect
- Self-aware AI possibly impossible

---

## Section 4: Four Types of AI Framework (~80 lines)

### The Four Types of AI: Reactive, Limited Memory, Theory of Mind, Self-Aware

**Sources:**

**Coursera** (September 2025)

From [Source Document](../../../RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md) lines 588-590:
- Article: "4 Types of AI: Getting to Know Artificial Intelligence"
- Theory of Mind and Self-Aware AI are **theoretical types**
- Not yet achieved in practice

**Bernard Marr Framework**

From [Source Document](../../../RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md) lines 591-593:
- Article: "What are the Four Types of AI?"
- Four categories: Reactive, Limited Memory, Theory of Mind, Self-Aware

### Type 1: Reactive Machines

**Characteristics:**
- No memory or past experience
- Responds only to current stimuli
- Cannot learn from past interactions
- Most basic form of AI

**Examples:**
- Deep Blue (chess computer)
- Spam filters
- Recommendation engines (basic)

**Limitations:**
- Cannot adapt beyond programmed responses
- No context awareness
- No learning capability

### Type 2: Limited Memory

**Characteristics:**
- Can use past experiences to inform decisions
- Temporary memory (not permanent)
- Most current AI systems fall here

**Examples:**
- Self-driving cars (use recent observations)
- Chatbots (use conversation history)
- Virtual assistants (use user interaction patterns)

**Capabilities:**
- Learn from training data
- Adapt within defined parameters
- Improve performance over time

**Current State**: This is where most deployed AI systems operate today (2024-2025).

### Type 3: Theory of Mind (Theoretical)

**Characteristics:**
- Would understand emotions, beliefs, intentions
- Could predict behavior based on mental states
- Would recognize that others have their own minds
- Could empathize and understand perspective

**Requirements:**
- Model human mental states
- Understand social interactions
- Recognize emotional cues
- Predict behavior based on beliefs/desires

**Current Status**:
- **Theoretical type** - not yet achieved
- Active research area (see Section 2)
- Emerging capabilities in advanced LLMs (partial ToM)
- Full ToM requires genuine understanding, not just pattern matching

**Challenges**:
- Distinguishing simulation from genuine understanding
- Measuring true comprehension vs. statistical mimicry
- Achieving consistent performance across contexts
- Cultural and individual variability in mental states

### Type 4: Self-Aware AI (Theoretical)

**Characteristics:**
- Would have consciousness and self-awareness
- Would understand its own existence
- Would have desires, needs, emotions (potentially)
- Would recognize itself as distinct entity

**Requirements:**
- Self-consciousness
- Subjective experience (qualia)
- Sense of continuity over time
- Understanding of own mental states

**Current Status**:
- **Theoretical type** - existence debated
- Most experts consider it impossible or distant future
- Functional self-awareness (monitoring performance) ≠ phenomenal consciousness
- No consensus on whether achievable

**The Consciousness Debate**:
- **Skeptics**: Impossible due to lack of biological substrate, evolutionary history, embodiment
- **Optimists**: Emergent property that could arise from sufficient complexity
- **Pragmatists**: Focus on functional capabilities, not philosophical questions

**Functional vs. Phenomenal Self-Awareness**:

From [Source Document](../../../RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md) line 610:
- **For ARR-COC**: Functional self-awareness (anomaly detection) may suffice
- Don't need phenomenal consciousness for trustworthy AI
- Can achieve reliability through self-monitoring mechanisms

---

## Cross-References to Existing Knowledge

**Related ARR-COC-VIS Topics:**
- Vervaeke's relevance realization framework (participatory knowing as functional ToM)
- Query-aware compression (models human intent through query understanding)
- Functional self-awareness through anomaly detection

**See Also:**
- [00-overview-self-awareness.md](00-overview-self-awareness.md) - Complete overview of ToM and self-awareness
- [02-arr-coc-functional-self-awareness.md](02-arr-coc-functional-self-awareness.md) - ARR-COC applications

---

## Summary: State of the Field (2024-2025)

**Emergent Capabilities:**
- Advanced LLMs demonstrate functional self-awareness (75% of tested models)
- Self-aware models systematically rank themselves as more rational than humans
- ToM capabilities emerging in latest models (multi-modal, embodied agents)

**Skepticism Persists:**
- Leading researchers argue conscious AI impossible
- Distinction between functional capability and phenomenal consciousness critical
- Microsoft AI chief calls consciousness pursuit "waste of time"

**Theoretical Framework:**
- Four types of AI widely adopted
- Theory of Mind and Self-Aware AI remain theoretical
- Functional self-awareness achievable without phenomenal consciousness

**Future Directions:**
- Focus on functional capabilities for trustworthy AI
- Emotional intelligence as next frontier
- Practical applications in mental health, education, customer service
- Continued debate on consciousness achievability

---

## Sources

**Source Documents:**
- [57-3-research-directions-oracle-feast.md](../../../RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md) - Direction 6, lines 561-611

**Web Research (Accessed 2025-01-31):**

**arXiv Papers:**
- Kim, K. (2025). "LLMs Position Themselves as More Rational Than Humans." arXiv:2511.00926v1. [https://arxiv.org/html/2511.00926v1](https://arxiv.org/html/2511.00926v1)
- Riemer, M. et al. (2024). "Position: Theory of Mind Benchmarks are Broken." arXiv:2412.19726
- Shi, H. et al. (2024). "MuMA-ToM: Multi-modal Multi-Agent Theory of Mind." arXiv:2408.12574 (Cited by 29)
- Wagner, E. et al. (2024). "Theory of Mind Goes Deeper Than Reasoning." arXiv:2412.13631 (Cited by 3)
- Harré, MS. et al. (2024). "An AI Theory of Mind Will Enhance Our Collective Intelligence." arXiv:2411.09168 (Cited by 3)

**Nature/Scientific Journals:**
- Porębski, A., & Figura, J. (2025). "There is no such thing as conscious artificial intelligence." *Nature Humanities and Social Sciences Communications*. DOI: s41599-025-05868-8
- Farisco, M., Evers, K., & Changeux, J-P. (2024). "Is artificial consciousness achievable? Lessons from the human brain." *Neural Networks*, 180, 106714. (Cited by 22) [https://www.sciencedirect.com/science/article/pii/S0893608024006385](https://www.sciencedirect.com/science/article/pii/S0893608024006385)
- Seth, AK. et al. "Conscious artificial intelligence and biological naturalism." *PubMed* 40257177 (Cited by 72)

**Industry & Educational Sources:**
- Insprago. (2025, August). "Theory of Mind AI: The Next Frontier in Emotional Intelligence." [https://insprago.com/theory-of-mind-ai-the-next-frontier-in-emotional-intelligence/](https://insprago.com/theory-of-mind-ai-the-next-frontier-in-emotional-intelligence/)
- Coursera. (2025, September). "4 Types of AI: Getting to Know Artificial Intelligence." [https://www.coursera.org/articles/types-of-ai](https://www.coursera.org/articles/types-of-ai)
- Marr, B. "What are the Four Types of AI?" [https://bernardmarr.com/what-are-the-four-types-of-ai/](https://bernardmarr.com/what-are-the-four-types-of-ai/)
- revolutionized.com. (2024, August). "Is Theory of Mind AI Possible?"

**Media Coverage:**
- BBC News. (2025, May). "The people who think AI might become conscious." [https://www.bbc.com/news/articles/c0k3700zljjo](https://www.bbc.com/news/articles/c0k3700zljjo)
- Gizmodo. (2025). "Microsoft AI Chief Warns Pursuing Machine Consciousness is a Gigantic Waste of Time."
- Scienmag. (2025, October 28). "Conscious Artificial Intelligence Does Not Exist."

**Additional References:**
- Prof. Ahmed Banafa. (2024). "The Rise of Conscious AI." *LinkedIn* (10+ reactions)

---

**Document Statistics:**
- Total lines: ~350
- Research papers cited: 12+
- Web sources: 15+
- Themes covered: 4 (emergent self-awareness, ToM implementation, consciousness skepticism, AI types framework)
- Date range: 2024-2025 (most papers within 12 months)
