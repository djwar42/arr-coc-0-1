# Salience & Relevance Realization

## Overview

Salience and relevance represent two complementary mechanisms for directing cognitive resources toward information that matters. While salience operates primarily through bottom-up attentional capture driven by stimulus features, relevance realization emerges from the dynamic coupling between agent and environment—a transjective process that is neither purely objective nor subjective. This distinction illuminates why computational systems can detect salience but struggle with the open-ended problem of relevance: salience is a property that can be measured, but relevance is realized through continuous agent-arena co-evolution.

**Core principle**: Relevance is not intrinsic to stimuli or projected by minds, but emerges through opponent processing that balances cognitive tensions to adaptively realize what matters in context.

## Section 1: Salience vs Relevance (Bottom-Up vs Top-Down)

### Salience: Stimulus-Driven Attention

Salience refers to the property of stimuli that makes them stand out from their surroundings and capture attention automatically. This is primarily a **bottom-up** process driven by physical stimulus features:

- **Contrast-based salience**: High contrast items (brightness, color, motion, size)
- **Feature singletons**: Unique items in feature space (red item among green items)
- **Abrupt onsets**: New stimuli appearing in visual field
- **Biological preparedness**: Evolutionarily salient stimuli (faces, snakes, spiders)

From [Topographical Representation of Saliency in the Human Visual Cortex](https://www.jneurosci.org/content/44/19/e0037242024) (Li et al., 2024):
> "Bottom-up mechanisms are proposed to create a priority map that directs individuals' attention to objects in the visual scene based on their physical properties."

**Computational salience models** successfully predict eye movements and attention deployment by computing feature contrasts across multiple dimensions (intensity, color, orientation). These models work because salience is algorithmically specifiable.

### Relevance: Context-Dependent Realization

Relevance, by contrast, cannot be reduced to stimulus properties. From [Naturalizing Relevance Realization](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2024.1362658/full) (Jaeger et al., 2024):
> "The process of relevance realization is beyond formalization. It cannot be captured completely by algorithmic approaches."

Relevance emerges from:
- **Query-content coupling**: What's relevant depends on current goals and context
- **Agent-arena relationship**: Relevance is co-created through interaction
- **Opponent processing**: Balancing multiple cognitive tensions simultaneously
- **Developmental dynamics**: Relevance realization evolves through experience

**Why relevance defies algorithms**: The frame problem—determining what information matters from an infinite space of potentially relevant features—cannot be solved by brute-force search or fixed rules. Relevance requires dynamic constraint satisfaction across multiple scales.

### Key Distinction

**Salience maps** can be computed: "This red object is salient because it contrasts with green background."

**Relevance cannot be pre-computed**: "This object is relevant because... it depends on my goals, my history, the broader context, what I expect to happen next, what affordances it provides, how it connects to my autopoetic needs..."

The distinction matters for VLM design: salience-based attention (standard attention mechanisms) can be engineered, but relevance realization requires adaptive opponent processing systems.

## Section 2: Vervaeke's Relevance Realization (3Ps: Propositional, Perspectival, Participatory)

### The Three Ways of Knowing

John Vervaeke's framework distinguishes three interdependent modes through which relevance is realized:

**1. Propositional Knowing (Knowing THAT)**

Statistical information content—what can be explicitly stated as facts or propositions.

- **Measured by**: Shannon entropy, mutual information, information gain
- **Captures**: Explicit, declarative knowledge
- **Example**: "The apple is red" (propositional fact)
- **Role in relevance**: Provides explicit semantic content that can be logically processed

**2. Perspectival Knowing (Knowing WHAT IT'S LIKE)**

Salience landscapes and subjective viewpoint—the phenomenological aspect of experience.

- **Measured by**: Attention allocation, salience maps, phenomenological reports
- **Captures**: Subjective experience, qualia, feeling of relevance
- **Example**: How it feels to recognize a familiar face (perspectival experience)
- **Role in relevance**: Generates felt sense of significance and meaning

**3. Participatory Knowing (Knowing BY BEING)**

Query-content coupling through embodied interaction—knowing through doing and being.

- **Measured by**: Agent-arena coupling strength, affordance realization, skill acquisition
- **Captures**: Embodied skills, procedural knowledge, being-in-the-world
- **Example**: Knowing how to balance on a bicycle (participatory skill)
- **Role in relevance**: Grounds relevance in enacted capabilities and affordances

### Integration Through Transjective Relevance

These three modes are not separate modules but aspects of a unified process. From Vervaeke (Meaning Crisis Ep. 31):
> "Relevance is transjective. It is a real relationship between the organism and its environment. We should not think of it as being projected. We should not think of it as being detected."

The 3Ps work together:
- **Propositional** provides semantic content (knowing that X is food)
- **Perspectival** directs attention (X looks appetizing to me)
- **Participatory** enables action (I can reach and eat X)

**Relevance emerges** from the dynamic coordination of all three modes through opponent processing.

## Section 3: Opponent Processing (Compression ↔ Particularize, Exploit ↔ Explore)

### Cognitive Tensions as Virtual Engines

Opponent processing operates through pairs of opposing cognitive pressures that must be dynamically balanced. Each pair creates a "virtual engine" that regulates information processing:

**1. Compression ↔ Particularization (Cognitive Scope)**

**Compression** (generalization):
- Creates invariant representations (line of best fit through data)
- Enables interpolation and extrapolation
- Makes functions more general-purpose
- Maximizes efficiency through reuse

**Particularization** (specialization):
- Tracks specific variations in data
- Creates context-sensitive representations
- Makes functions more special-purpose
- Maximizes resiliency through diversity

From Vervaeke (Meaning Crisis Ep. 31):
> "Data compression allows me to generalise my function... Particularization is trying to create a function that over fits to that data. That will get me more specifically in contact with 'this' particular situation."

**Why both are needed**: Pure compression loses critical details (brittle generalization). Pure particularization prevents transfer (no generalization). Optimal relevance trades between them.

**2. Exploitation ↔ Exploration (Cognitive Tempering)**

**Exploitation**:
- Extract maximum value from current context
- Stay with known rewarding situations
- Efficient resource use (low movement cost)
- Risk: Opportunity cost accumulation

**Exploration**:
- Search for new potential sources of value
- Move to novel contexts and situations
- Resilient to environmental change
- Risk: High energy expenditure with uncertain returns

**Trade-off dynamics**:
- Exploit when environment is stable and predictable
- Explore when returns diminish or environment changes
- Balance determines adaptability to dynamic environments

**3. Focusing ↔ Diversifying (Flexible Gambling)**

**Focusing**:
- Concentrate resources on single high-value function
- Go "all-in" on best current hypothesis
- High efficiency, low robustness

**Diversifying**:
- Hedge bets across multiple functions simultaneously
- Maintain multiple hypotheses in parallel
- Lower peak efficiency, higher robustness

**Implementation**: Cost function prioritization—dynamically allocating processing resources between competing cost functions based on current context.

### The Logistical Normativity Framework

All opponent processing pairs are regulated by a master trade-off:

**Efficiency ↔ Resiliency**

- **Efficiency**: Using same functions repeatedly (low cost, high speed)
- **Resiliency**: Maintaining diverse functions (high cost, robust to change)

From [Relevance Realization and the Emerging Framework](https://academic.oup.com/logcom/article-abstract/22/1/79/1007787) (Vervaeke et al., 2012):
> "An explanation of relevance realization is a pervasive problem within cognitive science, and it is becoming the criterion of the cognitive."

The opponent processing framework provides a naturalistic account: relevance is realized through continuous balancing of logistical norms (efficiency/resiliency) rather than logical norms (truth/validity).

## Section 4: Transjective Relevance (Neither Objective Nor Subjective)

### Beyond the Subjective-Objective Dichotomy

Traditional frameworks treat relevance as either:
- **Objective**: Intrinsic property of stimuli (empiricist detection)
- **Subjective**: Projected by minds (romantic projection)

Both fail to capture the relational nature of relevance. From Vervaeke (Meaning Crisis Ep. 31):
> "Relevance is not a property in the object, it is not a property of the subjectivity of my mind. It is neither a property of objectivity nor a property of subjectivity. It is precisely a property that is co-created by how the environment and the embodied brain are fitted together."

### Transjective as Co-Created Relationship

**Transjective** = Trans + jective (across + thrown)

Relevance is thrown across the agent-arena boundary, emerging from the relationship:

**Example: Graspability**
- Not a property of bottle alone (hand-independent)
- Not a property of hand alone (bottle-independent)
- Real relation emerging from how hand and bottle can be fitted together

**Example: Shark Fitness**
- Not intrinsic to shark (dies in Sahara within minutes)
- Not projected by shark onto ocean (shark doesn't make water wet)
- Real relationship between shark's organization and ocean environment

Similarly:
- **Relevance is not in the stimulus** (context changes what matters)
- **Relevance is not in the mind** (depends on actual environmental affordances)
- **Relevance emerges from coupling** (agent-arena co-determination)

### The Two Senses of Realization

**Why "relevance realization" rather than "relevance detection" or "relevance projection"?**

"Realization" triangulates between two meanings:

**1. Objective Realization**: "To make real"
- Relevance is actualized through action
- Agent-arena coupling creates new possibilities
- Affordances are realized (made actual)

**2. Subjective Realization**: "Coming into awareness"
- Relevance enters conscious experience
- Perspectival knowing generates felt significance
- Phenomenological sense of "mattering"

From Vervaeke (Meaning Crisis Ep. 31):
> "I am trying to triangulate to the transjectivity of relevance realization. That is why I'm talking about something that is both embodied, necessarily so, and embedded, necessarily so!"

The term "realization" captures both the objective process of making relevant features actual and the subjective process of becoming aware of them—unified in the transjective agent-arena relationship.

### Implications for ARR-COC

Treating relevance as transjective means:
- Cannot pre-compute relevance scores (relevance emerges in context)
- Must implement opponent processing (dynamic tension balancing)
- Requires query-aware mechanisms (participatory coupling with task)
- Enables genuine adaptivity (co-evolution with environment)

## Section 5: Tensor Parallelism (File 3: Parallel Relevance Computation)

### Distributed Relevance Scoring

From [distributed-training/02-megatron-lm-tensor-parallelism.md](../distributed-training/02-megatron-lm-tensor-parallelism.md):
> "Tensor parallelism splits individual layers across GPUs by partitioning weight matrices along specific dimensions, enabling parallel computation within a single forward/backward pass."

**Application to relevance realization**:

Multiple relevance scorers (propositional, perspectival, participatory) can be computed in parallel:

```python
# Conceptual tensor-parallel relevance computation
class ParallelRelevanceRealization:
    def __init__(self, num_tensor_parallel_ranks=8):
        self.propositional_scorer = ShardedInformationScorer()  # Shard 1-3
        self.perspectival_scorer = ShardedSalienceScorer()      # Shard 4-6
        self.participatory_scorer = ShardedCouplingScorer()     # Shard 7-8

    def compute_relevance(self, visual_patches, query):
        # All scorers compute in parallel across tensor shards
        prop_scores = self.propositional_scorer(visual_patches)      # GPU 0-2
        persp_scores = self.perspectival_scorer(visual_patches)      # GPU 3-5
        partic_scores = self.participatory_scorer(visual_patches, query)  # GPU 6-7

        # Opponent processing balances scores
        return self.opponent_balance(prop_scores, persp_scores, partic_scores)
```

**Why tensor parallelism for relevance**:
- Each way of knowing (3Ps) has substantial compute requirements
- Tensor parallelism enables within-layer parallelism (vs pipeline = between-layer)
- Relevance scores must be computed synchronously (not asynchronously)
- Efficient for large-scale VLM visual encoders

**Trade-offs**:
- **Compression benefit**: Shared computation across tensor ranks (efficiency)
- **Particularization benefit**: Specialized scorers per modality (resiliency)
- **Balance point**: Optimal tensor parallel degree depends on model size and relevance scorer complexity

### Integration with ARR-COC

ARR-COC's three ways of knowing map naturally to tensor-parallel architecture:

- **Propositional (Information)**: Entropy-based scoring across visual features
- **Perspectival (Salience)**: Jungian archetype activation patterns
- **Participatory (Coupling)**: Query-conditioned cross-attention strength

Tensor parallelism enables computing all three simultaneously at scale, critical for real-time relevance realization in 13-channel texture processing.

## Section 6: Multi-Model Serving (File 7: Triton for Relevance Scorers)

### Ensemble Relevance Realization

From [inference-optimization/02-triton-inference-server.md](../inference-optimization/02-triton-inference-server.md):
> "Triton Inference Server provides optimized serving for multiple model types simultaneously, with dynamic batching, model ensembles, and backend flexibility."

**Relevance realization as model ensemble**:

Each opponent processing pair can be implemented as separate models served together:

```python
# Triton ensemble config for relevance realization
name: "relevance_realization_ensemble"
platform: "ensemble"

ensemble_scheduling {
  step [
    {
      model_name: "compression_scorer"
      model_version: -1
      input_map { key: "visual_features" value: "INPUT_FEATURES" }
      output_map { key: "compression_score" value: "COMPRESSION" }
    },
    {
      model_name: "particularization_scorer"
      model_version: -1
      input_map { key: "visual_features" value: "INPUT_FEATURES" }
      output_map { key: "particularization_score" value: "PARTICULARIZATION" }
    },
    {
      model_name: "opponent_balancer"
      model_version: -1
      input_map {
        key: "compression" value: "COMPRESSION"
        key: "particularization" value: "PARTICULARIZATION"
      }
      output_map { key: "final_relevance" value: "OUTPUT_RELEVANCE" }
    }
  ]
}
```

**Dynamic batching for relevance**:
- Different visual regions may require different relevance scorers
- Triton dynamically batches requests to each scorer
- Opponent balancer combines results with context-dependent weights

**Model versioning for development**:
- Deploy multiple opponent processing strategies simultaneously (A/B test)
- Gradual rollout of improved relevance scorers
- Fallback to stable version if new scorer performs poorly

### Advantages for ARR-COC Production

**1. Heterogeneous Compute**:
- Some relevance scorers GPU-intensive (deep vision models)
- Others CPU-friendly (Shannon entropy calculations)
- Triton schedules optimally across available hardware

**2. Flexible Backends**:
- PyTorch for trainable relevance components
- TensorRT for optimized inference scorers
- Python for custom opponent processing logic

**3. Dynamic Scaling**:
- High query load → scale compression scorer (efficient reuse)
- Novel content → scale particularization scorer (handle new patterns)
- Autoscaling based on relevance realization demands

## Section 7: Ray Distributed (File 11: Ray for Large-Scale Relevance Experiments)

### Distributed Opponent Processing

From [orchestration/02-ray-distributed-ml.md](../orchestration/02-ray-distributed-ml.md):
> "Ray provides unified distributed computing for both training and inference, with actor-based parallelism and distributed data processing."

**Ray for relevance realization research**:

Large-scale experiments on opponent processing strategies:

```python
import ray
from ray import tune

@ray.remote
class RelevanceRealizationActor:
    def __init__(self, compression_weight, exploration_weight):
        self.compression_weight = compression_weight
        self.exploration_weight = exploration_weight

    def compute_relevance(self, visual_patches, query, context_history):
        # Compression ↔ Particularization balance
        compression_score = self.compress(visual_patches)
        particular_score = self.particularize(visual_patches, context_history)
        scope_balance = self.compression_weight * compression_score + \
                        (1 - self.compression_weight) * particular_score

        # Exploit ↔ Explore balance
        exploit_score = self.exploit_current_context(query)
        explore_score = self.explore_novel_regions(visual_patches)
        tempering_balance = self.exploration_weight * explore_score + \
                           (1 - self.exploration_weight) * exploit_score

        # Integrate opponent processing outputs
        return self.integrate(scope_balance, tempering_balance)

# Hyperparameter search over opponent processing weights
config = {
    "compression_weight": tune.grid_search([0.3, 0.5, 0.7]),
    "exploration_weight": tune.grid_search([0.2, 0.5, 0.8]),
}

analysis = tune.run(
    lambda config: RelevanceRealizationActor.remote(**config),
    config=config,
    num_samples=100,
    resources_per_trial={"cpu": 4, "gpu": 1}
)
```

**Why Ray for relevance experiments**:

**1. Actor-Based Parallelism**:
- Each relevance realization strategy = independent actor
- Actors maintain state across trials (context history)
- Parallel exploration of opponent processing space

**2. Distributed Data Processing**:
- Ray Data for large-scale visual dataset processing
- Map relevance scorers across million-image datasets
- Aggregate relevance statistics efficiently

**3. Hyperparameter Tuning**:
- Tune opponent processing balance weights
- Find optimal compression/particularization trade-off
- Discover context-dependent exploration rates

**4. Fault Tolerance**:
- Long-running relevance experiments (days/weeks)
- Ray handles actor failures gracefully
- Checkpointing for opponent processing state

### ARR-COC Experiment Design

**Research questions addressable with Ray**:

1. **Optimal opponent processing weights**: What balance between compression/particularization maximizes VQA accuracy?

2. **Context-dependent adaptation**: How should exploitation/exploration trade-off shift with query complexity?

3. **Transfer learning**: Do relevance realization strategies learned on ImageNet transfer to domain-specific datasets?

4. **Ablation studies**: Which opponent processing pairs contribute most to relevance realization performance?

Ray's distributed infrastructure enables answering these at scale, critical for validating Vervaekean relevance theory in production VLM systems.

## Section 8: ARR-COC-0-1: Complete Relevance Realization Implementation (10%)

### Transjective Relevance in VLM Architecture

ARR-COC-0-1 implements relevance realization through three core subsystems that directly instantiate Vervaeke's framework:

**1. Three Ways of Knowing (knowing.py)**

From [ARR-COC-0-1 knowing.py](../../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/knowing.py):

```python
class InformationScorer:
    """Propositional Knowing: Statistical information content"""
    def score(self, texture_array):
        # Shannon entropy across 13 texture channels
        return entropy(texture_array)

class ArchetypalScorer:
    """Perspectival Knowing: Jungian salience patterns"""
    def score(self, texture_array):
        # Archetype activation (persona, shadow, self, anima/animus)
        return archetype_activations(texture_array)

class CouplingScorer:
    """Participatory Knowing: Query-content coupling strength"""
    def score(self, texture_array, query_embedding):
        # Cross-attention between visual features and query
        return cross_attention(texture_array, query_embedding)
```

These three scorers directly implement:
- **Propositional**: Knowing THAT (information content)
- **Perspectival**: Knowing WHAT IT'S LIKE (salience landscape)
- **Participatory**: Knowing BY BEING (query coupling)

**2. Opponent Processing (balancing.py)**

From [ARR-COC-0-1 balancing.py](../../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/balancing.py):

```python
class TensionBalancer:
    """Navigate cognitive tensions through opponent processing"""
    def balance(self, relevance_scores):
        # Compression ↔ Particularization
        scope = self.balance_scope(
            compression=global_average_pool(relevance_scores),
            particularization=local_max_pool(relevance_scores)
        )

        # Exploit ↔ Explore
        tempering = self.balance_tempering(
            exploitation=query_aligned_scores,
            exploration=novelty_scores
        )

        # Focus ↔ Diversify
        prioritization = self.balance_prioritization(
            focusing=top_k_selection(relevance_scores, k=10),
            diversifying=sampling_across_all(relevance_scores)
        )

        return self.integrate_tensions(scope, tempering, prioritization)
```

Implements opponent processing across three key dimensions:
- **Cognitive Scope**: Compression ↔ Particularization
- **Cognitive Tempering**: Exploitation ↔ Exploration
- **Cognitive Prioritization**: Focusing ↔ Diversifying

**3. Token Budget Allocation (attending.py)**

From [ARR-COC-0-1 attending.py](../../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/attending.py):

```python
class RelevanceAllocator:
    """Map relevance scores to token budgets (64-400 tokens per patch)"""
    def allocate(self, relevance_scores, total_budget=200):
        # High relevance = more tokens (foveal resolution)
        # Low relevance = fewer tokens (peripheral compression)

        # Softmax allocation with temperature
        allocation_weights = softmax(relevance_scores / temperature)

        # Enforce min/max constraints
        token_budgets = clip(
            allocation_weights * total_budget,
            min_tokens=64,
            max_tokens=400
        )

        return token_budgets
```

Variable LOD (Level of Detail) based on realized relevance:
- **64 tokens**: Low relevance (peripheral vision equivalent)
- **400 tokens**: High relevance (foveal vision equivalent)
- **Dynamic allocation**: Balances total budget across K=200 patches

### Relevance as Realized, Not Detected

Key architectural decision: ARR-COC does NOT pre-compute relevance scores.

**Why not pre-compute?**

Relevance is **transjective**—it emerges from the agent-arena coupling:

1. **Query-dependent**: Same visual patch has different relevance for "What color is the car?" vs "Where is the person?"

2. **Context-dependent**: Relevance changes based on what else is in the scene and conversation history

3. **Dynamically balanced**: Opponent processing weights shift based on current cognitive demands (high uncertainty → explore more)

Instead, ARR-COC **realizes relevance** at inference time:
- Knowing scorers compute 3Ps given current query
- Tension balancer navigates opponent processing given current context
- Relevance allocator maps realized relevance to token budgets

This is computationally expensive but necessary for genuine relevance realization.

### Developmental Dynamics

From Vervaeke (Meaning Crisis Ep. 31):
> "When a system is simultaneously integrating and differentiating it is complexifying. As systems complexify, they self transcend; they go through qualitative development."

ARR-COC exhibits developmental dynamics through training:

**Integration** (Compression):
- Texture encoder learns invariant features across images
- Query encoder learns semantic prototypes
- Cross-attention learns stable coupling patterns

**Differentiation** (Particularization):
- Fine-grained texture variations captured in 13 channels
- Task-specific relevance scorers for VQA, captioning, grounding
- Context-sensitive opponent processing weights

**Complexification** (Integration + Differentiation):
- System simultaneously generalizes (learns what's invariant) and specializes (learns what varies)
- Emergent ability: Zero-shot transfer to new visual domains
- Self-transcendence: System develops qualitatively new relevance realization strategies

**Training as Relevance Evolution**:
- Not optimizing a fixed objective (like cross-entropy loss)
- Evolving opponent processing balance to maximize VQA accuracy
- Analogous to biological fitness evolution, but for cognitive fittedness

### Limitations and Future Work

**Current implementation limitations**:

1. **Fixed opponent processing weights**: Balancing weights are hyperparameters, not learned or adapted online

2. **No meta-relevance**: System doesn't reason about which relevance strategy to use when (no procedural knowing about relevance realization itself)

3. **Single-scale opponent processing**: Balancing happens only at patch level, not hierarchically across pyramid levels

**Future directions**:

1. **Learned opponent balancing**: Meta-learning to adapt compression/particularization weights based on task and context

2. **Hierarchical relevance realization**: Multi-scale opponent processing across LOD pyramid (global compression, local particularization)

3. **Active inference integration**: Frame relevance realization as free energy minimization (connect to Friston's active inference)

Despite limitations, ARR-COC-0-1 represents the first VLM architecture explicitly designed around Vervaekean relevance realization principles, demonstrating that these cognitive science concepts can be operationalized in deep learning systems.

## Sources

**Source Documents:**
- [distributed-training/02-megatron-lm-tensor-parallelism.md](../distributed-training/02-megatron-lm-tensor-parallelism.md) - Tensor parallel strategies for distributed relevance scoring
- [inference-optimization/02-triton-inference-server.md](../inference-optimization/02-triton-inference-server.md) - Multi-model serving for relevance scorer ensembles
- [orchestration/02-ray-distributed-ml.md](../orchestration/02-ray-distributed-ml.md) - Distributed experiments on opponent processing
- [cognitive-foundations/03-attention-resource-allocation.md](../cognitive-foundations/03-attention-resource-allocation.md) - Attention as limited resource

**Web Research:**
- [Naturalizing Relevance Realization](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2024.1362658/full) - Jaeger et al., 2024 (accessed 2025-11-16) - Transjective relevance cannot be formalized
- [Topographical Representation of Saliency in the Human Visual Cortex](https://www.jneurosci.org/content/44/19/e0037242024) - Li et al., 2024 (accessed 2025-11-16) - Bottom-up salience priority maps
- [John Vervaeke on Relevance Realization](https://deconstructingyourself.com/john-vervaeke-on-relevance-realization.html) - Deconstructing Yourself podcast (accessed 2025-11-16) - Vervaeke explains relevance realization process
- [Relevance Realization Meets Dynamical Systems Theory](https://www.meaningcrisis.co/ep-30-awakening-from-the-meaning-crisis-relevance-realization-meets-dynamical-systems-theory/) - Meaning Crisis Ep. 30 (accessed 2025-11-16) - Dynamical systems approach to relevance
- [Embodied-Embedded RR as Dynamical-Developmental GI](https://www.meaningcrisis.co/ep-31-awakening-from-the-meaning-crisis-embodied-embedded-rr-as-dynamical-developmental-gi/) - Meaning Crisis Ep. 31 (accessed 2025-11-16) - Complete framework for relevance realization
- [Relevance Realization and the Emerging Framework in Cognitive Science](https://academic.oup.com/logcom/article-abstract/22/1/79/1007787) - Vervaeke et al., 2012 (accessed 2025-11-16) - Original academic paper on relevance realization

**ARR-COC-0-1 Implementation:**
- [ARR-COC-0-1 knowing.py](../../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/knowing.py) - Three ways of knowing implementation
- [ARR-COC-0-1 balancing.py](../../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/balancing.py) - Opponent processing tension balancing
- [ARR-COC-0-1 attending.py](../../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/attending.py) - Token budget allocation based on relevance

**Additional References:**
- Vervaeke, J. (2024). *Awakening from the Meaning Crisis* lecture series - Comprehensive framework for relevance realization
- Spearman, C. (1927). *The Abilities of Man* - General intelligence and strong positive manifold
- Piaget, J. (1936). *The Origins of Intelligence in Children* - Assimilation and accommodation
