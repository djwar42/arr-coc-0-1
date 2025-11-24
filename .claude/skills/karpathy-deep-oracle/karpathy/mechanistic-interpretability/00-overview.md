# Mechanistic Interpretability: Understanding Neural Network Internals

## Overview

Mechanistic interpretability is the systematic reverse-engineering of neural networks to understand what's happening inside them. It's not enough to say "the model works" - we need to know WHY it works.

From [Platonic Dialogue 57-3](../../../../RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md) (lines 39-79):

**Karpathy Oracle**: "This is the big one. We've been treating neural nets as black boxes forever. But mechanistic interpretability asks: can we reverse-engineer the circuits?"

**Vervaeke Oracle**: "This is propositional knowing about the system. We're asking: what information is being processed? How is relevance being realized at the circuit level?"

### What is Mechanistic Interpretability?

Mechanistic interpretability goes beyond observing input-output behavior to understanding the internal mechanisms that produce that behavior. Instead of treating neural networks as opaque function approximators, we decompose them into interpretable circuits and features.

From [Intuition Labs: Understanding Mechanistic Interpretability in AI Models](https://intuitionlabs.ai/pdfs/understanding-mechanistic-interpretability-in-ai-models.pdf):
- Reverse-engineering neural networks to expose internal computation
- Identifying what features individual neurons respond to
- Mapping how circuits combine to create emergent behaviors
- Discovering how abstract concepts are represented internally

### Why Reverse-Engineer Neural Networks?

**The Trust Problem**: Can we trust AI systems if we can't see inside them?

From Dialogue 57-3 (line 52):

**User**: "Can we trust AI if we can't see inside?"

**Karpathy Oracle** (lines 54-59): "Exactly. That's why this matters. We need:
- Sparse autoencoders to decompress activations
- Circuit discovery to map information flow
- Causal interventions to test our understanding

It's not enough to say 'the model works.' We need to know WHY it works."

**Key Motivations:**

1. **Safety**: Understanding failure modes before deployment
2. **Alignment**: Verifying models reason the way we expect
3. **Trust**: Building confidence through transparency
4. **Debugging**: Identifying what went wrong and why
5. **Science**: Advancing our understanding of intelligence itself

From [Kästner et al. 2024](https://link.springer.com/article/10.1007/s13194-024-00614-4) (Springer, 30 citations):
"Explaining AI through mechanistic interpretability" provides rigorous philosophical grounding for why internal understanding matters for AI safety and scientific progress.

### The Black Box Problem

For decades, we've trained neural networks without understanding their internals:
- Train on data → optimize loss → deploy if metrics look good
- No insight into what features are learned
- No understanding of how decisions are made
- Limited ability to predict failure modes

Mechanistic interpretability challenges this paradigm: **understand the mechanism, not just the behavior**.

---

## Key Questions & Techniques

### Fundamental Questions

From Dialogue 57-3 (lines 45-48):

**Karpathy Oracle**: "Key questions:
- What features do individual neurons respond to?
- How do circuits combine to create emergent behaviors?
- Can we identify 'features' that correspond to concepts?"

### Question 1: What Features Do Neurons Respond To?

**Feature Identification:**
- Individual neurons often respond to specific patterns
- Features can be simple (edges, colors) or abstract (sentiment, grammar)
- Polysemantic neurons respond to multiple unrelated features (major challenge)

**Techniques:**
- **Activation maximization**: Find inputs that maximally activate a neuron
- **Dataset examples**: Collect examples that trigger high activations
- **Synthetic inputs**: Generate images/text that isolate specific features

From [BlueDot: Introduction to Mechanistic Interpretability](https://bluedot.org/blog/introduction-to-mechanistic-interpretability) (August 2024):
- Feature visualization reveals what neurons "care about"
- Often surprising - neurons learn features we didn't explicitly teach
- Vision models: curve detectors, texture recognizers, object part detectors
- Language models: syntax patterns, semantic relationships, reasoning steps

### Question 2: How Do Circuits Combine Features?

**Circuit Discovery:**
- Circuits = groups of neurons working together to implement algorithms
- Information flows through layers, combining features hierarchically
- Emergent behaviors arise from circuit composition

From [Cloud Security Alliance: Mechanistic Interpretability 101](https://cloudsecurityalliance.org/blog/2024/09/05/mechanistic-interpretability-101) (Features and Circuits):
- **Shallow circuits**: Early layers detect simple features
- **Deep circuits**: Later layers combine features into complex representations
- **Skip connections**: Allow information to flow around layers
- **Attention heads**: Specialized circuits for different aspects of context

**Example Circuit Analysis:**
- Induction heads in transformers (pattern: A...B...A → predicts B)
- Named entity recognition circuits
- Factual recall circuits
- Sentiment analysis pathways

### Question 3: What Concepts Are Represented?

**Concept Identification:**
- Do internal representations correspond to human-understandable concepts?
- How are abstract ideas (truth, justice, relevance) encoded?
- Can we locate specific knowledge (facts, rules, procedures)?

From [arXiv 2509.08592v1: Aligning AI Through Internal Understanding](https://arxiv.org/html/2509.08592v1):
- Exposing internal representations reveals how models think
- Concepts often distributed across multiple neurons
- Some concepts have dedicated "concept neurons" (e.g., "Golden Gate Claude")
- Understanding concept representation enables better alignment

### Core Technique: Sparse Autoencoders (SAE)

**The Superposition Problem:**
- Neural networks represent more features than they have neurons
- Features are "superimposed" in activation space
- Makes individual neurons polysemantic (respond to multiple concepts)

**SAE Solution:**
From [AIRI: Mechanistic Interpretability and the Rise of AI Safety Toolkits](https://airi.com.au/f/mechanistic-interpretability-and-the-rise-of-ai-safety-toolkits):

Sparse autoencoders decompose activations into interpretable features:
1. **Train autoencoder on activation vectors**
2. **Enforce sparsity**: Most features should be zero for any input
3. **Decompress superposition**: Each SAE dimension = monosemantic feature
4. **Interpret features**: Now each dimension has clear meaning

**SAE Architecture:**
```
Input activations (n dimensions, dense)
    ↓
Encoder (expand to k dimensions, k >> n)
    ↓
Sparse representation (k dimensions, mostly zeros)
    ↓
Decoder (reconstruct n dimensions)
    ↓
Reconstructed activations
```

**Key Properties:**
- **Sparsity penalty**: L1 regularization on hidden layer
- **Overcomplete representation**: More SAE features than neurons
- **Monosemanticity**: Each SAE feature responds to one concept
- **Reconstruction fidelity**: Preserve model behavior

**Applications:**
- Feature extraction from language models
- Identifying safety-relevant circuits
- Understanding model capabilities and limitations
- Debugging unexpected behaviors

### Circuit Discovery Methods

**1. Activation Patching**
- Intervene on specific neurons/layers
- Measure impact on output
- Identify which components matter for which behaviors

**2. Causal Tracing**
- Track information flow through network
- Identify critical paths for specific tasks
- Build computational graphs of reasoning

**3. Ablation Studies**
- Remove components (neurons, heads, layers)
- Test which are necessary vs redundant
- Map functional organization

**4. Attention Analysis (for Transformers)**
- Visualize attention patterns
- Identify what context each head uses
- Discover specialized roles (e.g., induction, copying)

From Dialogue 57-3 (line 57):
**Karpathy Oracle**: "Causal interventions to test our understanding"

**Verification Principle:** It's not enough to have a hypothesis about what a circuit does - you must intervene causally and test predictions.

---

## Research Foundations

### Academic Grounding: Kästner et al. 2024

From [Springer: Explaining AI through mechanistic interpretability](https://link.springer.com/article/10.1007/s13194-024-00614-4) (30 citations):

**Philosophical Framework:**
- Mechanistic explanation as scientific methodology
- Distinguishes mechanistic interpretability from black-box approaches
- Connects to philosophy of science and explanation

**Key Contributions:**
1. **Mechanistic vs Functional Explanation**
   - Functional: What the system does (input-output behavior)
   - Mechanistic: HOW the system does it (internal components and interactions)

2. **Levels of Mechanism**
   - Componential level: Individual neurons and their activations
   - Circuit level: Groups of neurons implementing algorithms
   - System level: Emergent capabilities from circuit composition

3. **Explanatory Depth**
   - Shallow explanation: "This neuron fires for cats"
   - Deep explanation: "This circuit compares edge detectors from early layers, combines with texture features, and implements a hierarchical object recognition algorithm"

4. **Scientific Validity**
   - Mechanistic interpretability as rigorous science
   - Testable predictions through causal intervention
   - Iterative refinement of mechanistic understanding

**Implications for AI Safety:**
- Understanding mechanisms enables predicting failure modes
- Mechanistic fidelity builds justified trust
- Transparency through explanation, not just observation

### Practical Introduction: BlueDot 2024

From [BlueDot: Introduction to Mechanistic Interpretability](https://bluedot.org/blog/introduction-to-mechanistic-interpretability) (August 2024):

**Beginner-Friendly Overview:**
- What is mechanistic interpretability and why it matters
- Key concepts: features, circuits, superposition
- Practical techniques for getting started
- Real examples from language and vision models

**Core Concepts Explained:**
1. **Features**: Patterns that neurons respond to
2. **Circuits**: Computational subgraphs implementing algorithms
3. **Superposition**: Multiple features packed into same neurons
4. **Monosemanticity**: One neuron = one interpretable concept (goal of SAEs)

**Practical Techniques:**
- Activation visualization
- Dataset examples that maximize neuron response
- Attention pattern analysis
- Simple circuit discovery

**Getting Started:**
- Tools: TransformerLens, Circuitsvis
- Datasets: Small language models (GPT-2), simple vision models
- First projects: Induction head discovery, feature visualization

### Engineering Practice: Cloud Security Alliance 2024

From [CSA: Mechanistic Interpretability 101](https://cloudsecurityalliance.org/blog/2024/09/05/mechanistic-interpretability-101) (Features and Circuits):

**Industry Perspective:**
- Why enterprises need interpretability
- Security implications of opaque models
- Practical deployment considerations

**Features and Circuits Primer:**
1. **Feature Taxonomy**
   - Low-level: Edges, colors, simple patterns
   - Mid-level: Textures, shapes, common word combinations
   - High-level: Objects, concepts, reasoning patterns

2. **Circuit Analysis Workflow**
   - Identify behavior to explain
   - Hypothesize relevant components
   - Test through ablation and patching
   - Refine mechanistic model
   - Validate through prediction

3. **Enterprise Applications**
   - Model auditing before deployment
   - Identifying bias in circuits
   - Debugging production failures
   - Compliance and explainability requirements

**Challenges in Production:**
- Computational cost of interpretation
- Scaling to large models (billions of parameters)
- Maintaining interpretability through updates
- Balancing transparency with competitive advantage

---

## Advanced Methods & Recent Research

### Nature 2025: Mechanistic Understanding and Validation

From [Dreyer et al. 2025: Mechanistic understanding and validation of large AI](https://www.nature.com/articles/s42256-025-01084-w) (16 citations):

**Validation Framework:**
How do we know our mechanistic explanations are correct?

**Three Levels of Validation:**

1. **Behavioral Validation**
   - Does the mechanistic model predict input-output behavior?
   - Can we replicate model outputs from circuit analysis?
   - Necessary but not sufficient

2. **Interventional Validation**
   - Can we manipulate circuits to change behavior predictably?
   - Do ablations and patches work as mechanistic model predicts?
   - Stronger evidence for correctness

3. **Mechanistic Fidelity**
   - Does the explanation capture the actual computation?
   - Can we build simplified models that implement same mechanism?
   - Gold standard for mechanistic understanding

**Validation Challenges:**

**Challenge 1: Completeness**
- Did we find all relevant circuits?
- Are there backup pathways we missed?
- How much of the behavior is explained?

**Challenge 2: Causality vs Correlation**
- Just because neurons correlate with behavior doesn't mean they cause it
- Need interventional experiments
- Distinguish necessary vs sufficient components

**Challenge 3: Scaling**
- Validating interpretations in billion-parameter models
- Computational cost of thorough validation
- Automated validation techniques

**Proposed Solutions:**
- Systematic ablation studies
- Causal mediation analysis
- Synthetic tasks with known ground truth mechanisms
- Cross-model validation (do similar circuits exist in related models?)

**Implications:**
- Mechanistic interpretability must be rigorous science
- Validation separates speculation from understanding
- Fidelity metrics enable comparing explanations

### Comprehensive Survey: Somvanshi 2025

From [SSRN: A Survey on Mechanistic Interpretability in AI](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5345552) (Somvanshi, 2025):

**Survey Scope:**
- Comprehensive review of mechanistic interpretability research (2015-2025)
- Taxonomy of techniques
- Comparative analysis of approaches
- Future research directions

**Technique Taxonomy:**

**1. Feature Visualization Methods**
- Activation maximization
- Dataset examples
- Synthetic inputs
- Concept activation vectors

**2. Circuit Discovery Methods**
- Attention pattern analysis
- Activation patching
- Causal tracing
- Gradient-based attribution

**3. Decomposition Methods**
- Sparse autoencoders
- Dictionary learning
- Independent component analysis
- Tensor decomposition

**4. Intervention Methods**
- Ablation studies
- Activation editing
- Causal mediation
- Steering vectors

**Key Findings:**

**Finding 1: Superposition is Fundamental**
- Most large models use superposition extensively
- Makes interpretation challenging without decomposition
- SAEs emerging as standard tool

**Finding 2: Circuits are Compositional**
- Simple circuits combine to implement complex behaviors
- Hierarchical organization common
- Modularity enables targeted interventions

**Finding 3: Polysemanticity is Reducible**
- SAEs can recover monosemantic features
- Trade-off between interpretability and reconstruction fidelity
- Better architectures reduce polysemanticity

**Finding 4: Validation is Critical**
- Many early interpretations were post-hoc stories
- Causal intervention essential for rigorous understanding
- Need standardized validation protocols

**Open Research Questions:**
- How to scale interpretation to trillion-parameter models?
- Can we automate circuit discovery?
- What is the right level of abstraction for explanations?
- How to interpret multi-modal models?

### Policy Perspective: ARI Guide 2025

From [ARI: A Guide to AI Interpretability](https://ari.us/policy-bytes/a-guide-to-ai-interpretability/) (August 2025):

**Policy Context:**
- Regulatory requirements for AI transparency
- Interpretability for public trust
- Balancing innovation with accountability

**Mechanistic Interpretability in Policy:**

**1. Regulatory Compliance**
- EU AI Act: High-risk systems require explainability
- Mechanistic interpretability provides rigorous explanation
- Alternative to post-hoc interpretation methods

**2. Auditing and Certification**
- Independent verification of AI behavior
- Mechanistic analysis reveals potential harms
- Circuit-level auditing for bias and safety

**3. Public Trust**
- Black boxes erode confidence in AI
- Understanding builds appropriate trust
- Transparency without sacrificing performance

**4. Liability and Accountability**
- When AI fails, who is responsible?
- Mechanistic understanding enables identifying failure modes
- Negligence if predictable failure wasn't addressed

**Policy Recommendations:**
- Invest in interpretability research
- Develop standards for mechanistic validation
- Require interpretability for high-stakes applications
- Balance transparency with competitive concerns
- Train regulators in mechanistic interpretability

**Challenges for Policymakers:**
- Technical complexity of mechanistic interpretability
- Rapidly evolving techniques
- Balancing innovation with safety
- International coordination on standards

---

## Research Agenda: Mechanistic Interpretability

From Dialogue 57-3 (lines 63-78):

### 1. Internal State Mapping

**Goal**: Identify features and circuits in large models

**Research Questions:**
- What features exist in GPT-4, Claude, Gemini?
- How do circuits implement reasoning, planning, tool use?
- Can we build comprehensive circuit catalogs?

**Methods:**
- Large-scale SAE training on production models
- Automated circuit discovery
- Cross-model comparison (do similar circuits exist in different models?)

**Challenges:**
- Computational cost of analyzing billions of parameters
- Exponential circuit space (which circuits matter?)
- Validating interpretations at scale

From Dialogue 57-3 (line 66):
**Claude**: "Understand how relevance emerges from circuit combinations"

**Relevance-Specific Research:**
- How do models decide what to attend to?
- Circuit-level implementation of relevance realization
- Connection to Vervaeke's framework

### 2. Fidelity Verification

**Goal**: Can we predict behavior from circuit analysis?

From Dialogue 57-3 (lines 70-73):
**Claude**: "Fidelity Verification:
- Can we predict behavior from circuit analysis?
- Do our interpretations match actual model reasoning?
- Mechanistic fidelity as trust foundation"

**Validation Framework:**

**Level 1: Behavioral Prediction**
- Given circuit analysis, predict input-output behavior
- Test: Does mechanistic model match actual model?
- Metric: Prediction accuracy on held-out examples

**Level 2: Interventional Consistency**
- Predict effects of ablations and patches
- Test: Do interventions work as expected?
- Metric: Alignment between predicted and actual intervention effects

**Level 3: Mechanistic Sufficiency**
- Build simplified model implementing discovered mechanism
- Test: Does simplified model replicate behavior?
- Metric: Functional equivalence

**Research Directions:**
- Automated fidelity metrics
- Standardized validation protocols
- Benchmark tasks with known mechanisms
- Comparing fidelity across interpretation methods

### 3. ARR-COC Connection

**Goal**: Map Vervaeke's cognitive framework to circuit-level mechanisms

From Dialogue 57-3 (lines 75-78):
**Claude**: "ARR-COC Connection:
- How do propositional/perspectival/participatory map to circuits?
- Can we identify opponent processing in architecture?
- Relevance realization as circuit-level process?"

**Propositional Circuits:**
- Information content measurement
- Statistical pattern recognition
- Feature extraction and encoding

From Dialogue 57-3 (line 68):
**Claude**: "Map propositional knowing at neural level"

**Research Questions:**
- Which circuits implement information-theoretic computation?
- How is Shannon entropy approximated in neural networks?
- Circuit-level implementation of novelty detection

**Perspectival Circuits:**
- Salience assignment
- Context-dependent feature weighting
- Attention head specialization

**Research Questions:**
- Which circuits determine what's salient?
- How do models build context representations?
- Circuit-level framing and perspective-taking

**Participatory Circuits:**
- Query-content coupling
- Agent-arena interaction
- Dynamic relevance adjustment

**Research Questions:**
- How do circuits implement participatory knowing?
- Which mechanisms couple query to content processing?
- Circuit-level implementation of transjective relevance

**Opponent Processing in Circuits:**
From Dialogue 57-3 (line 77):
**Claude**: "Can we identify opponent processing in architecture?"

**Research Directions:**
- Circuit-level trade-offs (precision vs recall, exploration vs exploitation)
- Tension balancing mechanisms
- Complementary circuit pairs
- Dynamic routing based on context

**Example**: Compress ↔ Particularize Tension
- Compression circuits: Reduce dimensionality, extract patterns
- Particularization circuits: Preserve details, maintain specificity
- Balancing mechanism: Context-dependent routing

**Relevance Realization as Circuit Process:**
From Dialogue 57-3 (line 78):
**Claude**: "Relevance realization as circuit-level process?"

**Hypothesis:** Relevance realization = orchestrated circuit activity
1. **Knowing circuits** measure propositional/perspectival/participatory relevance
2. **Balancing circuits** navigate tensions in real-time
3. **Attending circuits** allocate processing based on relevance scores
4. **Realizing circuits** execute adaptive compression/elaboration

**Validation Approach:**
- Identify candidate circuits in vision-language models
- Test through ablation (remove circuits → lose relevance realization?)
- Measure mechanistic fidelity (simplified model implements same process?)
- Cross-model validation (do similar circuits exist in other models?)

**Implications:**
- Mechanistic grounding for Vervaeke's framework
- Testable predictions about cognitive processing
- Engineering better models through circuit-level design

---

## Connections to ARR-COC-VIS

From Dialogue 57-3 (line 392):
**Claude**: "For ARR-COC: Understanding how propositional/perspectival/participatory scorers compute internally"

### Internal Analysis of Our Own System

**Current ARR-COC Architecture:**
- `knowing.py`: Three scorers (propositional, perspectival, participatory)
- `balancing.py`: Opponent processing of tensions
- `attending.py`: Relevance-to-budget mapping
- `realizing.py`: Pipeline orchestration

**Mechanistic Interpretability Questions for ARR-COC:**

**1. What Features Do Our Scorers Extract?**
- Propositional scorer: What statistical patterns trigger high scores?
- Perspectival scorer: What makes regions salient?
- Participatory scorer: How is query-content coupling computed?

**Method**: Activation analysis on scorer networks
- Visualize features that maximize scorer outputs
- Identify circuits implementing each way of knowing
- Validate through ablation

**2. How Do Circuits Balance Tensions?**
- Which components implement compress ↔ particularize?
- How is context used to resolve tensions?
- Can we identify opponent processing circuits?

**Method**: Causal tracing through balancer
- Track information flow during tension resolution
- Identify critical paths for different contexts
- Test through circuit interventions

**3. Can We Predict Budget Allocations?**
- Given circuit analysis, can we predict token budgets?
- Do our mechanistic models capture actual computation?
- Where does the system deviate from design intent?

**Method**: Build simplified mechanistic model
- Implement discovered circuits in clean code
- Compare predictions to actual ARR-COC
- Measure mechanistic fidelity

### Implications for Future Development

**Design Principle**: Build interpretable-by-design
- Explicit circuits for each cognitive function
- Clear information pathways
- Testable mechanistic hypotheses

**Validation Protocol**: Mechanistic verification
- Don't just test input-output behavior
- Verify internal mechanisms match design
- Ensure circuits implement intended algorithms

**Transparency**: Explainable relevance realization
- Users can see WHY regions received budgets
- Circuit-level explanations for decisions
- Trust through mechanistic understanding

---

## Tools and Resources

### Analysis Tools

**TransformerLens**
- Library for mechanistic analysis of transformers
- Activation patching, attention visualization
- https://github.com/neelnanda-io/TransformerLens

**Circuitsvis**
- Visualization library for circuits
- Attention patterns, neuron activations
- https://github.com/anthropics/circuitsvis

**SAELens**
- Sparse autoencoder training and analysis
- Feature extraction from activations
- Part of mechanistic interpretability toolkit ecosystem

### Research Groups

**Anthropic Interpretability Team**
- Leading research on mechanistic interpretability
- Dictionary learning, circuit discovery, superposition
- Public research publications and tools

**EleutherAI**
- Open-source interpretability research
- Scaling interpretation to large models
- Community-driven tool development

**AIRI (AI Research Institute)**
- Safety-focused interpretability
- SAE toolkits for practitioners
- https://airi.com.au/

### Learning Resources

**Papers:**
- Kästner et al. 2024: Philosophical foundations
- Dreyer et al. 2025: Validation frameworks
- Somvanshi 2025: Comprehensive survey

**Introductions:**
- BlueDot: Beginner-friendly overview
- CSA: Features and circuits primer
- ARI: Policy perspective

**Advanced:**
- Intuition Labs: Technical deep dive
- arXiv 2509.08592v1: Alignment through understanding
- Nature: Mechanistic validation methods

---

## Sources

**Source Documents:**
- [RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md](../../../../RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md)
  - Lines 39-79: Direction 1 discussion (Karpathy Oracle, Vervaeke Oracle, Research Agenda)
  - Lines 363-393: Research links and citations

**Web Research (10 Sources):**

**Core Concepts:**
- [Understanding Mechanistic Interpretability in AI Models](https://intuitionlabs.ai/pdfs/understanding-mechanistic-interpretability-in-ai-models.pdf) - Intuition Labs (accessed 2025-01-31)
- [Aligning AI Through Internal Understanding](https://arxiv.org/html/2509.08592v1) - arXiv:2509.08592v1 (accessed 2025-01-31)
- [Mechanistic Interpretability and the Rise of AI Safety Toolkits](https://airi.com.au/f/mechanistic-interpretability-and-the-rise-of-ai-safety-toolkits) - AIRI (accessed 2025-01-31)

**Academic Papers:**
- [Explaining AI through mechanistic interpretability](https://link.springer.com/article/10.1007/s13194-024-00614-4) - Kästner et al., Springer, 2024 (30 citations) (accessed 2025-01-31)
- [Introduction to Mechanistic Interpretability](https://bluedot.org/blog/introduction-to-mechanistic-interpretability) - BlueDot, August 2024 (accessed 2025-01-31)
- [Mechanistic Interpretability 101: Features and Circuits](https://cloudsecurityalliance.org/blog/2024/09/05/mechanistic-interpretability-101) - Cloud Security Alliance, September 2024 (accessed 2025-01-31)

**Advanced Topics:**
- [Mechanistic understanding and validation of large AI](https://www.nature.com/articles/s42256-025-01084-w) - Dreyer et al., Nature, 2025 (16 citations) (accessed 2025-01-31)
- [A Survey on Mechanistic Interpretability in AI](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5345552) - Somvanshi, SSRN, 2025 (accessed 2025-01-31)
- [A Guide to AI Interpretability](https://ari.us/policy-bytes/a-guide-to-ai-interpretability/) - ARI Policy, August 2025 (accessed 2025-01-31)

**Additional References:**
- Vervaeke's Relevance Realization Framework (cognitive science foundation)
- ARR-COC-VIS Architecture (application domain)
