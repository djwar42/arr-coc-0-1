# ARR-COC Connection: Mechanistic Interpretability for Relevance Realization

**Topic**: Applying mechanistic interpretability to understand ARR-COC's relevance realization circuits
**Purpose**: Research agenda for reverse-engineering how ARR-COC realizes relevance internally
**Date**: 2025-01-31

---

## Overview

Mechanistic interpretability provides tools to reverse-engineer neural networks by decomposing activations into interpretable features and circuits. For ARR-COC-VIS (Adaptive Relevance Realization - Contexts Optical Compression), this creates unique opportunities to validate that the architecture truly implements Vervaekean relevance realization rather than merely mimicking it through learned patterns.

**Key Question**: Does ARR-COC genuinely navigate opponent processes and realize transjective relevance, or does it approximate these concepts through statistical patterns?

**This file establishes**:
1. How ARR-COC's propositional knowing relates to interpretable circuits
2. Research agenda for validating Vervaekean principles at circuit level
3. Practical applications for debugging and improving relevance realization

---

## Section 1: Propositional Knowing at Circuit Level

### ARR-COC Architecture Overview

From [ARR-COC-VIS README.md](../../../../README.md), the architecture implements four ways of knowing:

**Vervaekean Core** (Lines 154-188):
```
knowing.py - Three of four ways of knowing
  ├── InformationScorer (Propositional: knowing THAT via Shannon entropy)
  ├── SalienceScorer (Perspectival: knowing WHAT IT'S LIKE via archetypes)
  ├── CouplingScorer (Participatory: knowing BY BEING via coupling)
  └── RelevanceIntegrator (Integrates all three dimensions)

balancing.py - Navigating cognitive tensions
  ├── ScopeTension (Compress ↔ Particularize)
  ├── TemperingTension (Exploit ↔ Explore)
  ├── PriorityTension (Focus ↔ Diversify)
  └── TensionBalancer (Resolves all tensions)

attending.py - Realizing salience (relevance → token budgets)
  ├── TierMapper (Maps relevance to token tiers)
  └── BudgetCalculator (Calculates final token budgets)

realizing.py - Orchestrating complete pipeline
  └── RelevanceAllocator (Main pipeline conductor)
```

### Propositional Knowing: Information Entropy as Measurable Circuit

The **InformationScorer** implements propositional knowing through Shannon entropy calculation:

**What it measures**: Statistical information content in visual patches
**Method**: Entropy H(X) = -Σ p(x) log p(x) over feature distributions
**Purpose**: Quantifies "knowing THAT" - factual information density

**Circuit-level questions**:
1. **Feature decomposition**: What specific features does InformationScorer activate on?
   - Edge detectors? Texture patterns? Semantic objects?
   - Can we identify entropy-sensitive neurons using sparse autoencoders (SAEs)?

2. **Information flow**: How do entropy scores influence downstream decisions?
   - Direct pathway to TensionBalancer?
   - Indirect modulation through RelevanceIntegrator?
   - Can we trace causal paths using activation patching?

3. **Integration circuits**: How are propositional, perspectival, and participatory scores combined?
   - Linear combination in RelevanceIntegrator?
   - Non-linear gating mechanisms?
   - Attention-like weighting based on query context?

### Perspectival and Participatory Circuits

**SalienceScorer (Perspectival)**:
- Measures symbolic/archetypal density using Jungian principles
- Creates salience landscapes: "what it's like" to attend to different patches
- Circuit question: Are there interpretable "archetype detectors" (faces, text, patterns)?

**CouplingScorer (Participatory)**:
- Measures query-content coupling through cross-attention
- Implements transjective relevance: emerges from agent-arena relationship
- Circuit question: Can we identify "coupling neurons" that fire only for specific query-content pairs?

### Verification Through Mechanistic Interpretability

**Propositional circuit hypothesis**: InformationScorer should contain:
- Entropy calculation neurons (logarithm, summation operations)
- Feature distribution analyzers (variance, spread detectors)
- Information density integrators (combining local to global information)

**Testable predictions**:
1. SAE decomposition reveals entropy-computation features
2. Activation patching of InformationScorer disrupts budget allocation predictably
3. Circuit discovery finds direct paths from entropy scores to token budgets

---

## Section 2: Research Agenda for ARR-COC

### Core Research Question

**Can we validate that ARR-COC implements genuine Vervaekean relevance realization?**

This requires showing that:
1. Opponent processing exists as interpretable circuits
2. Transjective relevance emerges from query-content coupling (not stored patterns)
3. Four ways of knowing are mechanistically distinct (not collapsed into single scoring)

### Research Direction 1: Opponent Processing Circuit Discovery

**Goal**: Identify circuits implementing the three opponent processes

**Cognitive Scope (Compress ↔ Particularize)**:
- **Hypothesis**: TensionBalancer contains opponent neurons that balance general vs. specific allocation
- **Method**:
  - Use SAEs to decompose TensionBalancer activations
  - Look for feature pairs with anti-correlated firing
  - Test: One feature activates on broad/general queries → promotes compression
  - Test: Opposing feature activates on specific queries → promotes particularization
- **Validation**: Ablate compression features → should increase token budgets
- **Validation**: Ablate particularization features → should decrease token budgets

**Cognitive Tempering (Exploit ↔ Explore)**:
- **Hypothesis**: Query uncertainty modulates exploration behavior
- **Method**:
  - Compare activations on confident queries ("Extract the table") vs. exploratory queries ("What's interesting here?")
  - Identify features that correlate with query uncertainty
  - Test if these features modulate token distribution variance
- **Validation**: High-uncertainty queries should show more diverse token allocations

**Cognitive Prioritization (Focus ↔ Diversify)**:
- **Hypothesis**: Query specificity controls allocation concentration
- **Method**:
  - Compare specific queries ("Find the signature") vs. broad queries ("Describe the document")
  - Measure token allocation entropy (H = -Σ p_i log p_i over patches)
  - Identify features that modulate allocation sharpness
- **Validation**: Focused queries should show low allocation entropy (concentrated budgets)

### Research Direction 2: Transjective Relevance Validation

**Goal**: Prove relevance emerges from query-content coupling, not memorized patterns

**Critical test**: Does CouplingScorer contain genuine coupling circuits or lookup tables?

**Genuine coupling indicators**:
1. **Compositional**: Novel query-content combinations produce sensible relevance scores
2. **Causal**: Editing query features causally changes content relevance rankings
3. **Symmetric**: Query←→Content coupling is bidirectional (not query-dominant)

**Method: Activation steering experiments**:
1. Collect query-content pairs with known relevance (e.g., "extract table" + table-containing image)
2. Extract CouplingScorer activations for high-relevance pairs
3. Use activation steering to inject "table-extraction coupling" into unrelated queries
4. Test: Does relevance to tables increase even for queries like "find signatures"?
5. If yes → genuine coupling circuit (steerable)
6. If no → memorized patterns (not transferable)

**Method: Causal intervention**:
1. Use activation patching to swap CouplingScorer activations between two queries
2. Test if relevance patterns follow the patched activations
3. If relevance changes → causal coupling circuit
4. If relevance unchanged → downstream processing ignores coupling (bad!)

### Research Direction 3: Fidelity Verification for Trust Foundation

**Goal**: Verify ARR-COC's relevance decisions are faithful to actual content

**Motivation**: From [Dialogue 57-3](../../../../RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md):
> "Mechanistic interpretability helps verify that AI systems are actually doing what we think they're doing - not just producing plausible-looking outputs."

**Fidelity risk**: ARR-COC might allocate high tokens to visually salient patches (bright colors, high contrast) rather than semantically relevant patches (text, tables, diagrams).

**Circuit-level fidelity test**:
1. **Surface feature detectors**: Identify features that activate on low-level patterns (edges, colors, textures)
2. **Semantic feature detectors**: Identify features that activate on high-level patterns (text, tables, faces)
3. **Measure influence**: Use causal tracing to determine which features drive token budgets
4. **Fidelity score**: Ratio of semantic influence to surface influence
5. **Good fidelity**: Token budgets primarily driven by semantic features
6. **Poor fidelity**: Token budgets driven by surface features (visual bias)

**Example fidelity test**:
- Input: Document with large colorful header (high surface salience) and small dense table (low surface salience, high semantic relevance for query "extract data")
- Expected: High tokens to table, low tokens to header
- Fidelity verification: Trace which features drive the allocation
  - If semantic "table detector" features dominate → faithful
  - If surface "colorful region" features dominate → unfaithful

### Research Direction 4: Four Ways of Knowing - Mechanistic Distinctness

**Goal**: Prove the four ways of knowing are mechanistically distinct, not collapsed

**Hypothesis**: Propositional, perspectival, participatory, and procedural knowing use different circuits

**Method: Representational dissimilarity analysis**:
1. Extract activations from InformationScorer, SalienceScorer, CouplingScorer, QualityAdapter
2. Compute representational dissimilarity matrices (RDMs) for each module
3. Test correlation between RDMs
4. **Prediction**: Low correlation → distinct representations → mechanistically separate
5. **Failure mode**: High correlation → collapsed into single scoring function

**Method: Ablation studies**:
1. Ablate InformationScorer (propositional) → should disrupt compression on information-dense patches
2. Ablate SalienceScorer (perspectival) → should disrupt allocation on visually salient patches
3. Ablate CouplingScorer (participatory) → should disrupt query-specific allocation
4. **Prediction**: Each ablation has unique behavioral signature
5. **Failure mode**: Ablations have similar effects → redundant modules

**Method: Circuit tracing**:
1. Use activation patching to trace information flow from scorers to token budgets
2. Identify which scorer dominates in different scenarios:
   - Information-dense queries ("extract all text") → InformationScorer dominant?
   - Visually-guided queries ("find the diagram") → SalienceScorer dominant?
   - Specific queries ("what is the signature?") → CouplingScorer dominant?
3. **Prediction**: Different scorers dominate different queries → mechanistic specialization

---

## Section 3: Practical Applications

### Application 1: Debugging Relevance Misallocations

**Problem**: ARR-COC sometimes allocates tokens poorly (e.g., high budget to background, low budget to relevant content)

**Mechanistic debugging workflow**:
1. **Identify failure case**: Query + image where allocation is incorrect
2. **Extract activations**: Run through ARR-COC, save all intermediate activations
3. **SAE decomposition**: Decompose activations into interpretable features
4. **Feature analysis**: Which features are firing? Which should be firing?
5. **Causal intervention**: Patch activations to test hypotheses
6. **Root cause**: Identify which module/feature causes misallocation
7. **Fix**: Fine-tune to correct the problematic circuit

**Example**:
- Failure: Query "extract table" allocates high tokens to document header
- Analysis: SalienceScorer features fire strongly on header (large text, bold)
- Hypothesis: Perspectival knowing (visual salience) overrides participatory knowing (query relevance)
- Test: Ablate SalienceScorer features → does allocation improve?
- Fix: Adjust RelevanceIntegrator to down-weight perspectival when query is specific

### Application 2: Understanding Why Relevance Realizations Occur

**Problem**: ARR-COC makes allocation decisions, but we don't know why

**Mechanistic explanation workflow**:
1. **Generate allocation**: Run ARR-COC on query + image
2. **Identify key decisions**: Which patches got high/low budgets?
3. **Attribution analysis**: Which features/circuits drove those decisions?
4. **Natural language explanation**: Convert circuit analysis to human-readable explanation

**Example**:
- Query: "Find mentions of quantum computing"
- Image: Research paper with text
- Allocation: High tokens to abstract and conclusion, low tokens to methods
- Mechanistic explanation:
  - CouplingScorer features fire strongly on "quantum" + "computing" text patterns
  - InformationScorer features detect high information density in abstract/conclusion
  - TensionBalancer resolves toward focused allocation (high priority for specific query)
  - Natural language: "The system allocated high tokens to abstract and conclusion because it detected query-relevant keywords ('quantum', 'computing') in regions with high information density. The focused nature of the query led to concentrated allocation rather than broad exploration."

### Application 3: Opponent Processing Visualization

**Problem**: Opponent processing is abstract - hard to verify it's working

**Visualization workflow**:
1. **Run ARR-COC on diverse queries**: Broad ("describe"), focused ("find X"), exploratory ("what's interesting?")
2. **Extract tension states**: For each query, extract TensionBalancer activations
3. **Visualize in tension space**: Plot queries in 3D space (Scope, Tempering, Priority axes)
4. **Validate opponent processing**: Check if queries cluster sensibly
   - Broad queries → low priority (diversify)
   - Focused queries → high priority (focus)
   - Uncertain queries → high tempering (explore)
   - Confident queries → low tempering (exploit)

**Interactive tool**:
- Input: Custom query
- Output: Real-time tension state visualization
- Shows: Which opponent processes dominate for this query
- Benefit: Users can understand how ARR-COC will process their query before running

### Application 4: Improving Training Through Circuit Analysis

**Problem**: ARR-COC training loss decreases, but we don't know what it's learning

**Circuit-informed training workflow**:
1. **Baseline circuit analysis**: Run mechanistic interpretability on untrained ARR-COC
2. **Train for N steps**: Standard training
3. **Post-training circuit analysis**: Rerun mechanistic interpretability
4. **Compare circuits**: What changed? What stayed the same?
5. **Validate improvements**: Are new circuits implementing desired behaviors?

**Example findings**:
- **Good**: Training strengthened CouplingScorer circuits → better query-content matching
- **Bad**: Training strengthened surface feature detectors → visual bias (fix with regularization)
- **Unexpected**: Training created new opponent processing circuits → emergent behavior (investigate!)

### Application 5: Building Trust Through Transparency

**Problem**: Users don't trust ARR-COC's token allocations - they seem arbitrary

**Transparency features using mechanistic interpretability**:

**Feature 1: Allocation explanations**
- For each patch, show which knowing contributed most:
  - "This patch received 256 tokens because: 60% participatory (query match), 30% propositional (high information), 10% perspectival (moderate salience)"

**Feature 2: Opponent processing dashboard**
- Show real-time tension states:
  - "Your query triggered: High focus (specific request), High exploit (confident matching), Medium compression (balanced allocation)"

**Feature 3: Circuit-level audit trail**
- For critical applications (medical, legal), provide full mechanistic trace:
  - "Token allocation driven by features F1 (table detector), F2 (text density), F3 (query coupling)"
  - "No surface features (color, brightness) influenced allocation"
  - "Fidelity score: 0.87 (semantic features dominate)"

---

## Research Priorities

### High Priority (Foundation)

1. **SAE decomposition of knowing.py modules**: Identify interpretable features in InformationScorer, SalienceScorer, CouplingScorer
2. **Circuit tracing for opponent processing**: Validate TensionBalancer implements genuine opponent circuits
3. **Fidelity verification**: Ensure semantic features (not surface features) drive allocations

### Medium Priority (Validation)

4. **Transjective relevance testing**: Prove coupling emerges from query-content interaction (not memorization)
5. **Four ways distinctness**: Show propositional/perspectival/participatory are mechanistically separate
6. **Failure mode analysis**: Identify common misallocation patterns and their circuit causes

### Low Priority (Enhancement)

7. **Interactive visualization**: Build tools for real-time opponent processing visualization
8. **Automated explanations**: Generate natural language explanations from circuit analysis
9. **Circuit-informed training**: Use mechanistic insights to guide training improvements

---

## Connections to Broader Research

### Mechanistic Interpretability Literature

This research agenda builds on:
- **Sparse autoencoders (SAEs)**: Decomposing activations into interpretable features
- **Activation patching**: Causal intervention to test circuit hypotheses
- **Circuit discovery**: Identifying computational subgraphs (features → features → outputs)
- **Representation analysis**: Understanding what networks learn internally

### ARR-COC Specific Innovations

**Novel contributions**:
1. **Cognitive architecture interpretability**: Most mechanistic interpretability focuses on language models - ARR-COC applies it to explicit cognitive architecture
2. **Vervaekean validation**: Using interpretability to validate philosophical principles (opponent processing, transjective relevance)
3. **Multi-modal circuits**: Understanding how vision, language, and relevance realization interact at circuit level

### Future Directions

**Beyond ARR-COC**:
- Apply mechanistic interpretability to other Vervaekean architectures
- Develop general tools for "cognitive architecture auditing"
- Build interpretability-driven training methods that encourage desired circuits

---

## Conclusion

Mechanistic interpretability provides powerful tools for validating and improving ARR-COC-VIS:

1. **Validation**: Prove opponent processing and transjective relevance exist as real circuits (not approximations)
2. **Debugging**: Identify and fix relevance misallocations by understanding circuit-level failures
3. **Trust**: Build transparency through mechanistic explanations of allocation decisions
4. **Improvement**: Guide training toward desired cognitive circuits using interpretability insights

**Next steps**: Begin with SAE decomposition of knowing.py modules to identify propositional, perspectival, and participatory feature circuits.

---

## Sources

**Source Documents**:
- [README.md](../../../../README.md) - ARR-COC-VIS architecture and Vervaekean principles (lines 1-941)
- [Dialogue 57-3](../../../../RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md) - Mechanistic interpretability research direction

**Related Oracle Knowledge**:
- Mechanistic interpretability fundamentals (to be created: 00-fundamentals.md)
- Academic foundations (to be created: 01-academic-foundations.md)
- Production applications (to be created: 02-advanced-production.md)

---

**Created**: 2025-01-31
**Purpose**: Research agenda for applying mechanistic interpretability to ARR-COC relevance realization
**Status**: Initial research directions established - experimental validation needed
