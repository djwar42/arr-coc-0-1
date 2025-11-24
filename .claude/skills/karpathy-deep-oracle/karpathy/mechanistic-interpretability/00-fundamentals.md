# Mechanistic Interpretability: Fundamentals

## Overview

### What is Mechanistic Interpretability?

Mechanistic interpretability is the study of **how neural networks compute their outputs by reverse-engineering their internal mechanisms** - treating trained models like compiled programs and attempting to recover their underlying algorithms. Instead of treating models as black boxes or using surface-level explanation techniques (like saliency maps), mechanistic interpretability aims to translate learned weights and activations into **human-understandable algorithms and circuits**.

From [Understanding Mechanistic Interpretability in AI Models](https://intuitionlabs.ai/articles/mechanistic-interpretability-ai-llms) (IntuitionLabs, accessed 2025-01-31):

> "Mechanistic interpretability aims to 'reverse engineer' neural networks into human understandable algorithms... much like deciphering a compiled program. Instead of treating a model as a black box, it aims to translate the network's learned weights and activations into human-understandable algorithms."

This approach differs fundamentally from:
- **Behavioral interpretability**: Understanding input-output relationships
- **Attributional interpretability**: Identifying which inputs influence decisions (saliency maps, feature importances)
- **Mechanistic interpretability**: Uncovering the actual **causal circuitry** transforming inputs into outputs

The field treats neural networks like unknown programs where:
- Learned parameters = machine code
- Architecture = CPU
- Activations = program state

The goal: recover the **logic** and **pseudocode-level description** of network operations.

### Core Hypotheses

From the foundational Anthropic work, mechanistic interpretability rests on three key hypotheses:

**1. Features Hypothesis**: Neural networks learn to encode **features** (meaningful concepts/patterns) as directions in activation space. These features are the basic units of computation - properties represented reliably across neurons.

**2. Circuits Hypothesis**: Features connect via weighted connections to form **circuits** (subnetworks) implementing specific algorithms or sub-tasks. A circuit is a computational subgraph that performs a recognizable function.

**3. Universality Hypothesis**: Analogous features and circuits **recur across different models and tasks**, suggesting transferable structure in how networks learn representations.

If these hypotheses hold, we can reverse-engineer networks **feature by feature, circuit by circuit** - building a complete map from weights to human-comprehensible concepts.

### Why Mechanistic Interpretability Matters

**Scientific Understanding**: Explain how deep learning actually works - what algorithms emerge from gradient descent, why models generalize, what causes phase transitions during training.

**AI Safety & Alignment**: Open the "black box" to verify models reason as intended. Detect deceptive strategies, misaligned goals, or unintended objectives before they cause harm. Critical as models become more powerful than humans.

**Debugging & Reliability**: Diagnose why models make mistakes at the circuit level. Enable surgical fixes (editing specific weights) rather than broad retraining. Reduce errors and hallucinations.

**Model Control**: Steer behavior by modulating identified circuits. Amplify safe goals, dampen problematic patterns. Enable fine-grained control beyond prompting or fine-tuning.

**Transparency & Governance**: Provide explanations for audits, regulation, and accountability. Enable oversight of superintelligent systems through automated interpretability.

From [BlueDot Impact - How Mechanistic Interpretability Actually Works](https://bluedot.org/projects/breaking-down-complexity-how-mechanistic-interpretability-actually-works) (accessed 2025-01-31):

> "This approach to alignment represents a fundamental shift in how we think about making AI systems safe and controllable."

---

## Reverse-Engineering Neural Networks

### The Compilation Analogy

Think of a trained neural network as a **compiled program**:

```
Source Code (High-level algorithm)
    ↓ [Training/Compilation]
Machine Code (Model weights)
    ↓ [Execution/Inference]
Program State (Activations)
    ↓
Output
```

Mechanistic interpretability attempts the reverse journey:
- **Decompilation**: Weights → Interpretable algorithms
- **State inspection**: Activations → Meaningful variables/features
- **Circuit tracing**: Subnetworks → Computational modules

### From Neurons to Features to Circuits

**Neurons**: Individual units in a layer (e.g., 768 neurons in a transformer MLP layer)

**Features**: Meaningful directions in activation space. A feature might be "text is in French" or "discussing medical topics" - represented as a vector direction that can span multiple neurons.

**Problem - Superposition**: Networks often encode **more features than neurons** by compressing them together. A single neuron may represent multiple unrelated concepts (polysemanticity). Features interfere with each other, causing cascading errors.

**Circuits**: Connected subgraphs of features that implement algorithms. For example, an "induction head circuit" in language models:
1. Head A attends backward to find repeated tokens
2. Head B copies the subsequent token forward
3. Result: In-context learning of patterns like "A B ... A → B"

From [An Extremely Opinionated Annotated List of Mechanistic Interpretability Papers](https://www.lesswrong.com/posts/NfFST5Mio7BCAQHPA/an-extremely-opinionated-annotated-list-of-my-favourite) (Neel Nanda, LessWrong, July 2024):

> "For any given activation (eg the output of MLP13), we believe that there's a massive dictionary of concepts/features the model knows of. Each feature has a corresponding vector, and model activations are a sparse linear combination of these meaningful feature vectors."

### Key Principles

**1. The Residual Stream is Central** (Transformer Models)

The residual stream is the primary "communication channel" in transformers. All layers read from and write to this stream additively:

```
Input Embedding
  ↓ (add to stream)
Attention Layer 1 → updates stream
  ↓ (add to stream)
MLP Layer 1 → updates stream
  ↓ (add to stream)
... (repeat)
  ↓
Output Unembedding
```

Each component makes **additive contributions** that can be analyzed independently and composed.

**2. Attention Heads are Modular**

Attention heads can be decomposed into:
- **QK circuit** (Query-Key): **Where** to attend (which tokens are relevant)
- **OV circuit** (Output-Value): **What** to do once attending (what information to move)

These circuits operate semi-independently, enabling separate analysis.

**3. Features are Linear Representations**

The **linear representation hypothesis**: Important features are represented as linear directions in activation space. This means:
- Features can be found with linear probes
- Features can be manipulated with vector addition/subtraction
- Features compose linearly (mostly)

Evidence from multiple domains (language, vision, games like Othello) supports this hypothesis, though superposition complicates matters.

**4. Computational Paths Through the Network**

Rather than analyzing every intermediate activation, we can trace **paths** from input to output:
- Input tokens → Attention → MLP → ... → Logits
- Each path is a **composition of matrices**
- Helps identify which sequences of operations matter for specific behaviors

### The Challenge of Superposition

**Why Superposition Happens**:
- Models have limited capacity (e.g., 768 dimensions in a layer)
- The world has far more concepts than dimensions (thousands of languages, topics, entities, patterns)
- Networks **compress** multiple features into the same neuron space
- Features stored in **overlapping** directions

**Consequences**:
- **Polysemantic neurons**: One neuron activates for bizarre, unrelated triggers (e.g., "French text" AND "verb tenses")
- **Distributed features**: Important concepts spread across many neurons
- **Interference**: Features interfere with each other, causing subtle errors

From [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) (Anthropic, 2022), superposition depends on:
- **Feature importance**: More important features get less compressed
- **Feature sparsity**: Sparse features (rarely active) can be compressed more
- **Feature correlation**: Correlated features interfere less; anti-correlated features superpose more

**Solution Approaches**:
1. **Sparse Autoencoders (SAEs)**: Learn an overcomplete basis that disentangles features (see next section)
2. **Probing**: Train classifiers to detect specific features in activations
3. **Activation patching**: Identify causal features through interventions

---

## Sparse Autoencoders for Internal States

### The Core Idea

Sparse Autoencoders (SAEs) are a breakthrough technique for **untangling superposition**. They learn an interpretable, sparse representation of model activations.

**Architecture**:
```
Model Activation (e.g., 768-dim)
    ↓
Encoder + ReLU → Sparse Features (e.g., 16,384-dim)
    ↓
Decoder → Reconstructed Activation (768-dim)
```

**Training Objective**:
- Minimize reconstruction loss: `||x - decoder(sparse_features)||²`
- Add L1 penalty on sparse features: `λ * ||sparse_features||₁`

The L1 penalty encourages **sparsity**: most features are zero, only a few activate per input.

**Result**: Each dimension in the sparse feature space corresponds to an **interpretable, monosemantic feature** - one that reliably represents a single concept.

### Why SAEs Work

From [Neel Nanda's annotated list](https://www.lesswrong.com/posts/NfFST5Mio7BCAQHPA/an-extremely-opinionated-annotated-list-of-my-favourite):

> "Though not directly trained to be interpretable, the hope is that each unit (or feature) corresponds to an interpretable feature. The encoder + ReLU learns the sparse feature coefficients, and the decoder is a dictionary of feature vectors. Empirically, it seems to work."

**Key Properties**:
- **Overcomplete**: More features than original dimensions (e.g., 16k features for 768-dim activations)
- **Sparse**: Only 5-20 features active per token on average
- **Interpretable**: Each feature activates for semantically related inputs
- **Disentangled**: Different concepts separated into different features

### Practical Applications

**1. Feature Discovery**

Train SAEs on every layer of a model, examine which tokens activate each feature:
- "French language" feature activates on French text
- "Medical terminology" feature activates on clinical discussions
- "Unsafe code" feature activates on security warnings

**2. Circuit Analysis**

Use SAE features as nodes in circuits instead of neurons:
- More interpretable than polysemantic neurons
- Clearer causal relationships
- Can trace information flow through feature activations

**3. Model Steering**

Amplify or suppress specific features during inference:
- Boost "truthful" features → reduce hallucinations
- Suppress "refusal" features → jailbreak safety filters
- Adjust "sentiment" features → control tone

From [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) (Anthropic, 2024):
- Demonstrated SAEs on Claude 3 Sonnet (near-frontier model)
- Found abstract features (e.g., "Golden Gate Bridge") that causally control outputs
- "Golden Gate Claude" demo: model obsessively discusses bridge when feature amplified

**4. Quality Metrics**

Evaluating SAE quality is an open problem. Current approaches:
- **Reconstruction loss**: How well do decoded activations match originals?
- **Sparsity**: How many features fire per input?
- **Interpretability**: Can humans understand feature activations?
- **Causal impact**: Do features matter for model outputs?

### Architecture Variants

**Standard SAE**:
- Linear encoder, ReLU activation, linear decoder
- L1 penalty on hidden activations

**Gated SAE** (2024):
- Separate gating mechanism decides which features activate
- Same reconstruction quality at **half the sparsity**
- Comparable or better interpretability

**Top-K SAE** (2024):
- Keep only top K features per input, zero the rest
- Simpler than L1 penalty, equally effective
- Used by OpenAI for GPT-4 SAEs

**Transcoders** (2024):
- Replace entire MLP layer with interpretable SAE-like module
- Map MLP input → sparse features → MLP output
- Enables true **weights-based** circuit analysis (not just activations)

### Scaling Laws and Evidence

From [Scaling and Evaluating Sparse Autoencoders](https://cdn.openai.com/papers/sparse-autoencoders.pdf) (OpenAI, 2024):

**Scaling to Frontier Models**:
- SAEs successfully applied to GPT-4 (largest model tested)
- Scaling laws: more features + more training = better reconstruction
- Feature completeness: sigmoid relationship between concept frequency in training and probability SAE learns it

**Universality Evidence**:
- Similar features appear across different models
- Features transfer between model scales (e.g., GPT-2 to GPT-4)
- Some features are multimodal (appear in both text and images in CLIP)

### Current Limitations

**1. Incomplete Coverage**: SAEs don't capture all model computation. Some information remains in the reconstruction error (what SAE can't explain).

**2. Validation Challenge**: Hard to verify features are "truly" interpretable. Humans might misinterpret or cherry-pick convincing examples.

**3. Computational Cost**: Training SAEs on every layer of large models is expensive. Inference with SAEs adds overhead.

**4. Interference Remains**: Even with SAEs, some features still entangle. Perfect disentanglement may be impossible.

**5. Evaluation Metrics**: No consensus on measuring SAE quality. Trade-offs between sparsity, reconstruction, and interpretability.

From [Towards Principled Evaluations of Sparse Autoencoders](https://arxiv.org/abs/2405.08366) (Makelov & Lange et al, 2024):
- Proposed **sparse control** metric: Can sparse feature changes cause specific output changes?
- Proposed **probe accuracy** metric: Can features predict expected properties?
- Evaluated on well-studied IOI circuit where ground truth partially known

---

## Connection to ARR-COC Vision

### Mechanistic Interpretability for Propositional Knowing

In the ARR-COC framework, **propositional knowing** (knowing THAT) involves measuring statistical information content. Mechanistic interpretability reveals **how** this measurement happens at the circuit level.

**Question**: When ARR-COC's InformationScorer computes Shannon entropy over visual patches, what internal circuits enable this computation?

**Potential Applications**:
1. **Verify scoring fidelity**: Use mechanistic interpretability to confirm scorers measure what we think they measure
2. **Debug relevance failures**: When relevance realization fails, trace which circuits produced incorrect salience
3. **Optimize architectures**: Identify redundant or interfering circuits to improve efficiency
4. **Trustworthy compression**: Prove compression preserves critical information by analyzing learned circuits

### Mechanistic Understanding of Opponent Processing

ARR-COC's **balancing** module navigates tensions (compress ↔ particularize, exploit ↔ explore). Can we identify these tension-balancing mechanisms at the circuit level?

**Research Questions**:
- Do opponent processing pairs emerge as distinct circuit modules?
- Can we find "tension neurons" that arbitrate between competing pressures?
- How does the model learn to balance context-dependent trade-offs?

From dialogue 57-3: Understanding the **transjective** (agent-arena coupling) at the circuit level could reveal:
- How query features compose with image features
- Which circuits implement participatory knowing (query-content coupling)
- Why certain query-image pairs produce surprising relevance patterns

### Training Interpretability

Mechanistic interpretability can illuminate **how relevance realization emerges during training**:

From [In-Context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) (Anthropic, 2022):
- Induction heads form suddenly during training (phase transition)
- Correlates with jump in in-context learning capability
- Suggests specific circuits unlock new capabilities

**ARR-COC Parallel**: Do relevance circuits emerge suddenly? Can we predict when the model will "understand" salience mapping?

### Fidelity Verification Research Agenda

**Goal**: Use mechanistic interpretability to verify ARR-COC does what it claims.

**Approach**:
1. Train SAEs on ARR-COC's intermediate activations (after knowing, after balancing, after attending)
2. Identify features corresponding to:
   - High-information patches
   - Salient regions (perspectival knowing)
   - Query-relevant areas (participatory knowing)
3. Use activation patching to verify causal relationships:
   - Do information features → influence token allocation?
   - Do salience features → guide attending module?
   - Do query features → modulate compression?

**Validation Questions**:
- Are propositional, perspectival, and participatory scorers truly independent?
- Do tension balancers actually navigate trade-offs, or just implement fixed heuristics?
- Is relevance → LOD mapping (64-400 tokens) justified by internal feature importance?

### Practical Debugging Tools

**Scenario**: ARR-COC allocates too few tokens to an important face in an image.

**Mechanistic Debugging**:
1. Train SAEs on attending module activations
2. Identify "face detection" features - do they fire on this image?
3. Use activation patching:
   - Patch face features from high-attention example → does allocation increase?
   - Ablate competing features → does face get more tokens?
4. Trace backward through circuit:
   - Do perspectival scorers detect the face?
   - Do tension balancers suppress it (explore vs exploit)?
   - Is participatory knowing ignoring it (query-image mismatch)?

Result: **Surgical fix** to specific circuit rather than retraining entire model.

---

## Research Directions

### Open Problems in Mechanistic Interpretability

From [Understanding Mechanistic Interpretability in AI Models](https://intuitionlabs.ai/articles/mechanistic-interpretability-ai-llms):

**1. Scalability**: Current techniques struggle with frontier models (100B+ parameters). Need automated, efficient methods.

**2. Superposition**: Still partially unsolved. SAEs help but don't fully disentangle all features. Need better factorization techniques.

**3. Evaluation**: No consensus metrics for interpretability quality. Risk of confirmation bias (cherry-picking convincing features).

**4. Completeness**: Can we verify we've found **all** relevant circuits? Or only the easy-to-see ones ("streetlight interpretability")?

**5. Automation**: Manual circuit discovery doesn't scale. Need AI systems to interpret AI systems.

### Opportunities for ARR-COC Integration

**1. Multi-Modal Mechanistic Interpretability**

ARR-COC processes vision + language. Mechanistic interpretability research on CLIP (vision-language models) is relevant:
- Multimodal neurons: Single neurons fire for concept in both modalities (e.g., "Spider-Man" in text and images)
- Cross-modal circuits: How do text queries influence visual processing?

**2. Hierarchical Feature Analysis**

ARR-COC uses hierarchical vision processing (patches → regions → full image). Mechanistic interpretability can reveal:
- Which features emerge at each level?
- How do high-level features depend on low-level circuits?
- Do features compose hierarchically or interact laterally?

**3. Dynamic Resource Allocation**

ARR-COC's variable LOD (64-400 tokens per patch) is a form of **dynamic compute allocation**. Mechanistic interpretability can study:
- What circuits decide token budgets?
- How do models learn to allocate compute efficiently?
- Can we transfer allocation circuits between tasks?

### Methodological Innovations Needed

**For ARR-COC Analysis**:
1. **Query-conditional SAEs**: Train SAEs that vary with query context (not just image content)
2. **Budget-aware circuit analysis**: Trace how token allocation decisions propagate through the network
3. **Vervaekean feature taxonomy**: Categorize SAE features by type of knowing (propositional, perspectival, participatory)

**Measurement Challenges**:
- How to quantify "relevance realization" as a circuit-level phenomenon?
- Can we identify transjective features (neither purely objective nor subjective)?
- What does "opponent processing" look like in weight space?

---

## Sources

### Primary Web Research

**1. IntuitionLabs - Understanding Mechanistic Interpretability in AI Models**
- URL: https://intuitionlabs.ai/articles/mechanistic-interpretability-ai-llms
- Accessed: 2025-01-31
- Comprehensive overview of mechanistic interpretability fundamentals, techniques, applications, and limitations
- 35-minute read covering definitions, methods (probing, activation patching, circuit analysis, SAEs), case studies (Anthropic, OpenAI, DeepMind), and ethical considerations

**2. BlueDot Impact - Breaking Down Complexity: How Mechanistic Interpretability Actually Works**
- URL: https://bluedot.org/projects/breaking-down-complexity-how-mechanistic-interpretability-actually-works
- Accessed: 2025-01-31
- Student project overview highlighting practical applications and alignment approaches
- Emphasis on SAEs as tools for AI safety and control

**3. LessWrong - An Extremely Opinionated Annotated List of My Favourite Mechanistic Interpretability Papers v2**
- URL: https://www.lesswrong.com/posts/NfFST5Mio7BCAQHPA/an-extremely-opinionated-annotated-list-of-my-favourite
- Author: Neel Nanda (Google DeepMind)
- Published: July 7, 2024
- Annotated reading list covering foundational work (Transformer Circuits), superposition, SAEs, activation patching, narrow circuits, and bonus topics
- Includes practical guidance on what to read deeply vs skim

### Foundational Papers (Referenced)

**Anthropic Research**:
- A Mathematical Framework for Transformer Circuits (2021)
- Toy Models of Superposition (2022)
- In-Context Learning and Induction Heads (2022)
- Towards Monosemanticity (2023)
- Scaling Monosemanticity (2024)

**OpenAI Research**:
- Logit Lens (nostalgebraist)
- Multimodal Neurons in Artificial Neural Networks (Goh et al)
- Language Models Can Explain Neurons in Language Models (Bills et al)
- Scaling and Evaluating Sparse Autoencoders (Gao et al, 2024)

**Google DeepMind Research**:
- The Hydra Effect (McGrath et al, 2023)
- AtP* - Attribution Patching improvements (Kramar et al)
- Does Circuit Analysis Interpretability Scale? (Lieberum et al)

**Academic Research**:
- Sparse Feature Circuits (Marks et al, David Bau's group, 2024)
- Transcoders Find Interpretable LLM Feature Circuits (Dunefsky, Chlenski et al, 2024)
- Towards Principled Evaluations of Sparse Autoencoders (Makelov, Lange et al, 2024)
- Gated SAEs (Rajamanoharan et al, 2024)

### Additional References

**Technique Papers**:
- Causal Scrubbing (Redwood Research)
- Automated Circuit Discovery (Conmy et al, 2023)
- Distributed Alignment Search (Geiger et al)
- Interpretability Illusion for Subspace Activation Patching (Makelov & Lange)

**Circuit Analysis Examples**:
- Indirect Object Identification (Wang et al, Redwood, 2022)
- Copy Suppression (McDougall, Conmy, Rushing et al)
- Progress Measures for Grokking via Mechanistic Interpretability (Nanda et al)

**Steering and Control**:
- Activation Addition (Turner et al)
- Inference-Time Interventions (Li et al, Harvard)
- Representation Engineering (Zou et al, CAIS)
- Refusal is Mediated by a Single Direction (Arditi et al, 2024)

### Related Resources

- ARENA Tutorial on Sparse Autoencoders: https://arena3-chapter1-transformer-interp.streamlit.app/
- ARENA Tutorial on Activation Patching: https://arena3-chapter1-transformer-interp.streamlit.app/
- Transformer Circuits Thread: https://transformer-circuits.pub/
- Anthropic Interpretability Research: https://www.anthropic.com/research#interpretability
- Neel Nanda's YouTube Tutorials: https://www.youtube.com/c/NeelNanda

---

**Note**: This fundamentals document synthesizes concepts from multiple sources to provide a comprehensive introduction to mechanistic interpretability. For deeper technical understanding, consult the original papers listed in the sources section. The ARR-COC connections represent research directions proposed in dialogue 57-3, not established results.
