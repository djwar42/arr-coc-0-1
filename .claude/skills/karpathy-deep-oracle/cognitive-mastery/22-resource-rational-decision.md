# Resource-Rational Decision Making

## Overview

Resource-rational decision making formalizes the idea that human cognition operates optimally under computational constraints—doing the best possible given limited time, memory, and processing capacity. This framework bridges bounded rationality with optimization theory, showing that apparent cognitive biases often represent rational solutions to the problem of expensive computation.

**Core principle**: Agents maximize utility subject to the cost of computation itself, treating thinking as a scarce resource to be allocated efficiently.

From [Resource-rational decision making](https://www.sciencedirect.com/science/article/abs/pii/S2352154621000371) (Bhui, Lai & Gershman, 2021, *Current Opinion in Behavioral Sciences*):
> "Resource rationality refines traditional views of bounded rationality by conceiving of judgment as optimal under cognitive costs or constraints, rather than merely feasible."

## Section 1: Bounded Rationality - Herbert Simon's Foundation

### The Classical Problem

**Perfect rationality** assumes:
- Unlimited computational resources
- Complete information access
- Infinite time for deliberation
- No cognitive costs

Real agents face:
- **Time constraints**: Decisions needed before optimal computation finishes
- **Memory limits**: Cannot store all relevant information
- **Processing bottlenecks**: Serial computation, attention limits
- **Uncertainty**: Incomplete models of the world

### Simon's Bounded Rationality (1955)

From [Models of Bounded Rationality](https://www.jstor.org/stable/1914185) (Simon, 1955):
> "The capacity of the human mind for formulating and solving complex problems is very small compared with the size of the problems whose solution is required for objectively rational behavior."

**Satisficing** instead of optimizing:
```
Instead of: Find argmax_x U(x)  # Optimal solution
Do: Find any x such that U(x) > threshold  # Good enough
```

**Key insight**: Rationality must account for the costs of being rational.

### Limitations of Classical Bounded Rationality

1. **Descriptive, not normative**: Says what people do, not what they should do
2. **No optimization principle**: "Good enough" is vague
3. **Missing cost-benefit tradeoff**: When is satisficing better than optimizing?

Resource rationality addresses these gaps.

## Section 2: Computational Constraints - The Real Bottlenecks

### Types of Computational Constraints

**1. Time Constraints**
- Limited deliberation time before action required
- Deadlines, urgency, opportunity costs
- Anytime algorithms must return progressively better solutions

**2. Memory Constraints**
- Finite working memory capacity (7±2 items, Cowan 2001)
- Episodic memory retrieval costs
- Cannot store complete world models

**3. Processing Constraints**
- Serial attention bottleneck
- Limited parallel processing
- Neural computation energy costs (~20W brain power budget)

**4. Information Access Costs**
- Sensory acquisition effort (eye movements, exploration)
- Communication bandwidth limits
- Environmental sampling expenses

### Measuring Computational Costs

**Time cost**:
```
Cost_time(algorithm) = wall_clock_time × opportunity_cost_per_second
```

**Energy cost**:
```
Cost_energy(computation) = neural_firing_rate × metabolic_expense
```

**Information cost** (bits):
```
Cost_info(representation) = -log₂ P(representation)  # Shannon entropy
```

From [Thermodynamics as a theory of decision-making](https://royalsocietypublishing.org/doi/10.1098/rspa.2012.0683) (Ortega & Braun, 2013):
> "Information-processing costs can be formalized as the KL divergence between prior and posterior distributions, providing a thermodynamic bound on computation."

## Section 3: Anytime Algorithms - Progressive Refinement Under Time Pressure

### Anytime Algorithm Properties

**Definition**: An algorithm that returns progressively better solutions as computation time increases.

**Key characteristics**:
1. **Interruptible**: Can be stopped at any time
2. **Monotonic quality**: Later solutions never worse than earlier
3. **Diminishing returns**: Quality improvements slow over time
4. **Measurable quality**: Can assess solution goodness at any point

From [Anytime sorting algorithms](https://dl.acm.org/doi/10.24963/ijcai.2024/785) (Caizergues et al., 2024, *IJCAI*):
> "Anytime algorithms provide tentative estimates at each step, allowing agents to trade off solution quality against computation time."

### Examples of Anytime Algorithms

**1. Monte Carlo Tree Search (MCTS)**
```python
def mcts_anytime(state, time_budget):
    """Return best action, improving with more time."""
    root = TreeNode(state)
    start_time = time.time()

    while time.time() - start_time < time_budget:
        # One iteration: select, expand, simulate, backpropagate
        leaf = select_leaf(root)
        reward = simulate_rollout(leaf)
        backpropagate(leaf, reward)

    return root.best_child()  # Better with more iterations
```

**2. Iterative Deepening Search**
- Depth-limited search: depth 1, 2, 3, ...
- Each iteration completes before going deeper
- Always has a solution ready (even if suboptimal)

**3. Approximate Bayesian Inference**
- Variational inference: ELBO optimization
- More gradient steps → better posterior approximation
- Can stop early for fast (noisy) estimates

### Anytime Decision Making in Cognition

**Visual search**: Allocate gaze time progressively
- First fixation: coarse scene understanding (300ms)
- Additional fixations: refine object recognition
- Stop when confidence threshold reached OR time runs out

**Memory retrieval**: Iterative recall
- Fast: retrieve most accessible items first
- Slow: search deeper, activate related memories
- Anytime: can respond based on partial retrieval

From [Exploiting Anytime Algorithms for Collaborative Service Composition](https://www.mdpi.com/2073-431X/13/6/130) (Nogueira et al., 2024, *Computer Science*):
> "Anytime algorithms hold significant promise in decision making under time constraints, producing progressively better solutions as they are allotted more computation time."

## Section 4: Metalevel Reasoning - Deciding How to Decide

### The Metalevel/Object-Level Distinction

**Object level**: The original decision problem
- Example: "Should I take this job offer?"
- Computations: evaluate salary, commute, career growth, work-life balance

**Metalevel**: Decisions about which object-level computations to run
- Example: "Should I spend time researching company culture, or decide now?"
- Computations: estimate value of information, cost of delay

From [Principles of Metareasoning](https://2024.sci-hub.se/196/44a218a9eb38b3f5b964793fff03b414/russell1991.pdf) (Russell & Wefald, 1991):
> "The metalevel is a second decision-making process whose application domain consists of the object-level computations themselves."

### Type II Rationality (I. J. Good, 1971)

**Type I Rationality**: Choosing optimal actions given beliefs
- Traditional decision theory: max_a E[U(a)|beliefs]

**Type II Rationality**: Choosing optimal computations given resource costs
- Metalevel decision theory: max_c E[Gain(c)] - Cost(c)

```
Metalevel decision:
    For each possible computation c:
        Estimate: Gain(c) = Expected improvement in decision quality
        Estimate: Cost(c) = Time, memory, energy required
        If Gain(c) > Cost(c): Execute c
        Else: Stop deliberation, act now
```

**Introspection paradox**: Perfect metalevel reasoning requires infinite regress
- To decide whether to compute X, must compute value of computing X
- But that computation itself has a cost...
- **Solution**: Use cheap heuristics at metalevel (don't optimize metalevel perfectly)

### Metalevel Heuristics - Fast & Frugal

**Value of Information (VOI)**: Expected improvement from gathering information
```
VOI(information) = E[U(optimal_decision | info)] - E[U(current_decision)]

If VOI(info) > Cost(info): Gather info
Else: Decide now with current knowledge
```

**Value of Computation (VOC)**: Expected improvement from more thinking
```
VOC(computation) = E[U_after_computation] - U_current - Cost_computation

If VOC(next_step) > 0: Continue thinking
Else: Stop, act on best current option
```

From [Rationality and Intelligence](https://people.eecs.berkeley.edu/~russell/papers/aij-cnt.pdf) (Russell & Subramanian, 1995):
> "Rational metareasoning formalizes the intuition that the metalevel can 'do the right thinking'—selecting object-level computations that maximize expected utility gain per unit cost."

## Section 5: Cost of Computation - Quantifying Cognitive Expense

### Neural Metabolic Costs

**Brain energy budget**: ~20W total power consumption
- 2% of body mass, 20% of metabolic energy
- ~10^11 neurons, each ~10^4 synapses
- Action potentials: expensive (ATP hydrolysis)

**Energy per spike**: ~10^-9 J
- High firing rates → high metabolic cost
- Neural efficiency: sparse coding, predictive coding

From [Brain energy budget](https://www.nature.com/articles/nrn3941) (Attwell & Laughlin, 2001, *Nature Reviews Neuroscience*):
> "Signaling costs dominate brain energy consumption—75% of ATP goes to action potentials and synaptic transmission."

### Information-Theoretic Costs

**Rate-distortion theory** (Shannon, 1959):
- Can't compress below entropy without distortion
- Lossy compression trades accuracy for efficiency
- Cognitive representations: compressed world models

**Free energy principle** (Friston, 2010):
```
Free Energy = -log P(sensory_data | model) + KL(beliefs || prior)
              \_____________________________/    \_________________/
                    Prediction error                Complexity cost
```

Minimizing free energy = Maximizing accuracy while minimizing model complexity.

### Opportunity Costs

**Time is finite**: Every second spent deliberating is lost
- In foraging: While deciding, competitors eat your food
- In markets: While analyzing, prices change
- In survival: While planning, predator approaches

**Optimal stopping**: When to stop thinking and act
```
Expected value of continuing = P(better_decision) × Improvement - Time_cost

Stop when: Expected_value ≤ 0
```

**Secretary problem** example:
- 37% rule: Explore first 37%, then exploit (pick first better)
- Balances information gain vs opportunity loss

## Section 6: Distributed Tensor-Parallel Metalevel Reasoning (File 3: Megatron-LM)

### Parallel Metalevel Computations

From [distributed-training/02-megatron-lm-tensor-parallelism.md](../distributed-training/02-megatron-lm-tensor-parallelism.md):

**Tensor parallelism** enables resource-rational decision making at scale by distributing metalevel reasoning across GPUs:

```python
# Metalevel: Decide which computations to parallelize
def resource_rational_parallel(computation_graph, gpu_count):
    """Allocate computations to minimize latency under resource constraints."""

    for node in computation_graph:
        # Metalevel reasoning: Is tensor parallel worth it?
        communication_cost = estimate_allreduce_time(node.size, gpu_count)
        speedup = node.compute_time / gpu_count

        if speedup > communication_cost:  # VOC > 0
            node.tensor_parallel = True
        else:
            node.tensor_parallel = False  # Keep sequential (lower cost)
```

**Resource-rational tensor sharding**:
- **Computation gain**: N-way parallel → N× speedup (ideally)
- **Communication cost**: All-reduce collective (bandwidth-bound)
- **Optimal decision**: Shard only when computation >> communication

**Example**: GPT-3 attention layers
- 12,288-dimensional embeddings, 96 attention heads
- Tensor parallel across 8 A100 GPUs
- Compute: 100ms/layer sequential → 15ms parallel (85ms gain)
- Communication: 5ms all-reduce
- **Net gain**: 80ms/layer → rational to parallelize

### Anytime Inference with Tensor Parallelism

**Progressive refinement** in distributed models:
1. **Depth 1**: Return after first transformer layer (fast, coarse)
2. **Depth 12**: Return after 12 layers (medium quality)
3. **Depth 96**: Full GPT-3 depth (highest quality, slowest)

Metalevel decides depth based on time budget.

## Section 7: Inference Server Resource Allocation (File 7: Triton Inference Server)

### Dynamic Batching as Metalevel Optimization

From [inference-optimization/02-triton-inference-server.md](../inference-optimization/02-triton-inference-server.md):

**Triton's dynamic batching**: Metalevel reasoning about when to process requests

```python
# Metalevel decision: When to execute batch?
class ResourceRationalBatcher:
    def should_execute_batch(self, current_batch, time_since_first):
        """Optimize latency vs throughput tradeoff."""

        # Gain from waiting: More requests → better GPU utilization
        potential_gain = (max_batch - len(current_batch)) * compute_per_sample

        # Cost of waiting: Latency for current requests increases
        waiting_cost = time_since_first * len(current_batch) * latency_penalty

        # Metalevel decision
        if waiting_cost > potential_gain:
            return True  # Execute now (anytime: good enough)
        else:
            return False  # Wait for more requests
```

**Resource-rational serving**:
- **Small batches**: Low latency, poor GPU utilization (idle cycles wasted)
- **Large batches**: High throughput, high latency (users wait)
- **Optimal**: Dynamic cutoff based on latency SLA and request rate

**Example**: Vision model serving (ResNet-50)
- Batch size 1: 5ms latency, 30% GPU utilization (wasteful)
- Batch size 64: 50ms latency, 95% utilization (too slow)
- **Adaptive**: Start at 8 (12ms), grow to 32 if queue builds (25ms, 85% utilization)

### Model Selection Metalevel Reasoning

**Which model to serve?** Metalevel cost-benefit:
```
Cheap model (DistilBERT): 10ms latency, 85% accuracy
Expensive model (BERT-Large): 50ms latency, 92% accuracy

VOC(expensive_model) = 7% accuracy gain - 40ms time cost
If deadline < 50ms: Use DistilBERT (anytime constraint)
If accuracy critical: Use BERT-Large (quality threshold)
```

Triton enables A/B testing this metalevel decision at runtime.

## Section 8: Intel oneAPI Hardware Resource Constraints (File 15: Intel oneAPI)

### Hardware Heterogeneity and Rational Allocation

From [alternative-hardware/02-intel-oneapi-ml.md](../alternative-hardware/02-intel-oneapi-ml.md):

**Intel platforms**: CPUs, GPUs, FPGAs, Gaudi accelerators
- Each has different compute/memory/power tradeoffs
- Metalevel: Allocate workloads to minimize total cost

```python
# Metalevel reasoning: Which device for this computation?
def resource_rational_device_selection(task):
    """Select device that maximizes utility per watt-second."""

    devices = {
        'cpu': {'compute': 1.0, 'power': 100W, 'latency': 10ms},
        'gpu': {'compute': 10.0, 'power': 300W, 'latency': 5ms},
        'fpga': {'compute': 5.0, 'power': 50W, 'latency': 8ms}
    }

    for device, specs in devices.items():
        utility = task.quality(specs['compute']) / specs['latency']
        cost = specs['power'] * specs['latency']

        rational_score[device] = utility / cost

    return max(rational_score, key=rational_score.get)
```

**Bounded rationality on edge devices**:
- Jetson Nano: 5W power budget → must be extremely selective
- Metalevel: "Is this frame worth processing, or skip to save energy?"
- Anytime: "How much can I compress the model and still meet accuracy threshold?"

**Example**: Real-time object detection on battery
- YOLO-v8 (full): 45 FPS, 15W power → battery lasts 2 hours
- YOLO-v8 (tiny): 120 FPS, 5W power → battery lasts 6 hours
- **Rational**: Use tiny model, accept 5% accuracy drop for 3× battery life

### Memory Hierarchy as Computational Constraint

**Intel Sapphire Rapids**: L1 (64KB), L2 (2MB), L3 (105MB), DRAM (512GB), SSD (2TB)
- Access latency: L1 (1ns), L2 (5ns), L3 (20ns), DRAM (100ns), SSD (100μs)
- **Metalevel**: Decide what to cache where (limited capacity)

```python
# Resource-rational cache management
def rational_cache_placement(data_item):
    """Place in cache level that maximizes utility per byte."""

    access_frequency = estimate_reuse(data_item)
    data_size = len(data_item)

    for cache_level in ['L3', 'L2', 'L1']:
        capacity_left = cache_level.capacity - cache_level.current_usage
        latency_gain = DRAM_latency - cache_level.latency

        utility = access_frequency * latency_gain
        cost = data_size  # Bytes consumed

        if utility / cost > threshold and capacity_left > data_size:
            return cache_level

    return 'DRAM'  # Default fallback
```

**Bounded optimality**: Can't perfectly predict future access patterns
- Use cheap heuristics: LRU (recency), LFU (frequency), Belady (oracle)
- Intel's Cache Allocation Technology (CAT): Partition L3 rationally across workloads

## Section 9: ARR-COC-0-1 - Token Allocation as Resource-Rational Vision (10%)

### Vision Token Budget as Computational Constraint

**ARR-COC-0-1 problem**: Allocate 200 visual tokens (K=200) across image patches

This IS resource-rational decision making:
- **Resource**: Limited token budget (200 tokens, ~8MB activations)
- **Decision**: Which patches get more detail (64-400 tokens per patch)?
- **Constraint**: Total budget fixed, must maximize task utility

From [RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/attending.py](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/attending.py):

```python
class ResourceRationalAllocator:
    """Allocate tokens to maximize expected utility under budget constraint.

    This implements resource rationality:
    - Computational cost: Tokens (proportional to FLOPs, memory)
    - Utility: Query-relevant visual information extracted
    - Optimization: Maximize information gain per token spent
    """

    def allocate_lod(self, relevance_scores, total_budget=200):
        """Metalevel reasoning: Decide LOD for each patch."""

        patches = []
        for patch_idx, relevance in enumerate(relevance_scores):
            # Anytime algorithm: Start with minimum (64 tokens)
            lod = 64

            # Iteratively increase LOD while VOC > 0
            while total_budget > 0:
                # Value of Computation: More tokens → better features
                info_gain = estimate_information_gain(patch_idx, lod + 64)
                token_cost = 64

                voc = info_gain / token_cost  # Utility per token

                # Metalevel decision
                if voc > threshold and lod < 400:
                    lod += 64
                    total_budget -= 64
                else:
                    break  # Stop refining this patch (anytime cutoff)

            patches.append((patch_idx, lod))

        return patches
```

### Connecting to Resource Rationality Theory

**1. Bounded rationality**: Can't process full image at max resolution
- Vision transformer: 1024×1024 image → 256×256 patches = 65,536 tokens
- Budget: 200 tokens → Must compress 327× (massive bounded constraint)

**2. Anytime decision**: Progressive refinement of patch details
- 64 tokens: Basic shape, color (fast, coarse)
- 128 tokens: Texture, edges (medium quality)
- 400 tokens: Fine details, text (slow, high quality)

**3. Metalevel reasoning**: Relevance scorers decide where to look
- Propositional: Information content (Shannon entropy)
- Perspectival: Salience (Jungian archetypes, anomaly)
- Participatory: Query coupling (cross-attention)

**4. Cost of computation**:
```
Cost(lod) = tokens × (FLOPs_per_token + Memory_per_token)
LOD=64:  64 × (12M FLOPs + 4KB memory) = 768M FLOPs, 256KB
LOD=400: 400 × (12M FLOPs + 4KB memory) = 4.8B FLOPs, 1.6MB
```

**5. Optimal under constraints**:
- Not "best possible" (that would be 400 tokens everywhere)
- But "best given 200-token budget" (resource-rational optimum)

### Human Foveated Vision as Resource Rationality

**Human retina**: Non-uniform sampling
- Fovea: 5° central vision, 50% of V1 neurons (high resolution)
- Periphery: 95° outer vision, 50% of V1 neurons (low resolution)
- **Why?**: Optic nerve bandwidth constraint (~1M axons)

**Resource-rational explanation**:
- Transmitting full retinal image: ~100M photoreceptors × 60Hz = 6GB/s
- Optic nerve capacity: ~10MB/s (600× compression needed)
- **Solution**: Allocate bandwidth where task-relevant (fovea for reading, periphery for motion)

**ARR-COC mirrors this**:
- Token budget = Optic nerve bandwidth
- LOD allocation = Fovea/periphery distinction
- Relevance realization = Saccade planning (where to look next)

From [Platonic Dialogue Part 46](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/46-mvp-be-doing.md):
> "The vision system doesn't transmit raw pixels—it transmits features at variable granularity, guided by top-down task demands. ARR-COC implements this computationally: relevance determines resolution."

### Metalevel Learning in ARR-COC

**The adapter module** (4th P: Procedural knowing) learns metalevel policies:

```python
class QualityAdapter:
    """Learn to predict optimal LOD allocation (metalevel policy).

    Trained on:
    - Input: Image statistics, query embedding
    - Output: Expected utility of each LOD level per patch
    - Loss: Task performance (answer accuracy) - token cost
    """

    def learn_metalevel_policy(self, experience):
        """Reinforcement learning for resource allocation."""

        for episode in experience:
            # Object level: What LOD did we use?
            lod_allocation = episode.lod_decisions

            # Outcome: How well did it work?
            task_accuracy = episode.final_score
            token_cost = sum(lod_allocation)

            # Metalevel reward: Accuracy per token spent
            metalevel_reward = task_accuracy / token_cost

            # Update policy to maximize resource rationality
            self.update_weights(episode.features, metalevel_reward)
```

**Result**: Learns heuristics like "faces need high LOD" or "backgrounds tolerate low LOD"
- These are **fast metalevel heuristics** (no infinite regress)
- Approximations of true VOC, but much cheaper to compute

## Section 10: Practical Resource-Rational Algorithms

### Approximate Inference - Good Enough is Optimal

**Variational Bayesian Inference**: Trade accuracy for speed

Perfect Bayesian inference:
```
P(hypothesis | data) = P(data | hypothesis) × P(hypothesis) / P(data)
                       \_____________________________________/
                              Intractable integral
```

Resource-rational approximation (variational inference):
```
Find Q(hypothesis) that minimizes KL(Q || P)
Subject to: Q must be computationally cheap (factorized, Gaussian, etc.)

Trade accuracy for tractability
```

**When to stop optimization?** Anytime stopping criterion:
```
ELBO_improvement = ELBO[iteration_t] - ELBO[iteration_{t-1}]
Compute_cost = wall_clock_time × opportunity_cost

If ELBO_improvement < Compute_cost: STOP
```

### Heuristics as Resource-Rational Solutions

**Fast-and-frugal trees** (Gigerenzer):
- Make decisions with minimal information lookup
- Each node: Simple binary test (cheap computation)
- Reaches decision in O(depth) time, not O(N log N)

Example: Heart attack diagnosis (Green & Mehr, 1997):
```
If chest_pain == "crushing": ADMIT (sensitivity 95%)
Else if ECG_abnormal: ADMIT
Else: DISCHARGE

Accuracy: 92% (vs 88% for complex logistic regression)
Speed: 3 decisions vs 20+ features analyzed
```

**Why it works**: VOC of additional tests is low
- Most diagnostic power in first few variables
- Diminishing returns on extra features
- Resource-rational to stop early

### Pruning Search Trees - When to Give Up

**Alpha-beta pruning** in game trees:
- Metalevel: Estimate if subtree can improve current best move
- If provably worse: Skip entire subtree (save computation)
- Anytime: Can interrupt search anytime, always have best move so far

**Progressive widening** (MCTS):
```
def expand_node(node, compute_budget):
    """Add children only if VOC > 0."""

    visit_count = node.visits
    exploration_constant = sqrt(2)

    # Metalevel: Is it worth exploring a new child?
    exploration_bonus = exploration_constant × sqrt(log(visit_count))
    best_child_value = max(child.value for child in node.children)

    if exploration_bonus > best_child_value and budget > 0:
        add_new_child(node)  # Expected gain > 0
    else:
        refine_existing_children(node)  # Exploit current knowledge
```

**Resource-rational exploration**: UCB balances explore/exploit under time constraints

## Section 11: Neuroscience Evidence for Resource Rationality

### Dopamine and Computational Cost Signals

**Dopamine** encodes reward prediction error, but also **effort costs**:

From [Effort-based decision making](https://www.nature.com/articles/nn.3917) (Salamone & Correa, 2012, *Nature Neuroscience*):
> "Dopamine signals both the value of rewards and the cost of effortful actions needed to obtain them."

**Computational effort** = Neural metabolic cost:
- Frontal cortex: High firing rates during deliberation → high energy burn
- Dopamine depletion: Reduces willingness to exert cognitive effort
- Depression: Low dopamine → "cognitive exhaustion" (high subjective cost of thinking)

**Resource-rational interpretation**:
```
Decision = argmax_action [E[Reward(action)] - Effort_cost(action)]

Where Effort_cost includes:
- Physical: Motor energy
- Cognitive: Deliberation time, working memory load
- Opportunity: Other actions foregone
```

### Prefrontal Cortex as Metalevel Controller

**PFC lesions** impair metalevel reasoning:
- Perseveration: Can't switch strategies when current one fails
- Distractibility: Can't filter irrelevant computations
- Impulsivity: Can't delay gratification (no deliberation)

**Intact PFC**: Implements adaptive control
- Decides when to engage cognitive control (costly)
- Monitors conflict, error likelihood
- Recruits additional processing only when expected gain > cost

From [The Computational and Neural Basis of Cognitive Control](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4084861/) (Botvinick & Cohen, 2014):
> "Cognitive control reflects a resource allocation decision: engage controlled processing when automatic processing is insufficient."

**EEG evidence**: N2, P3 components scale with decision difficulty
- Harder decisions → larger late potentials → more prolonged deliberation
- Metalevel: Allocating more time when expected VOC is high

## Section 12: Critique and Limitations

### When Does Resource Rationality Fail?

**1. Systematic biases**: Not all heuristics are optimal
- Availability bias: Recent events over-weighted (not always rational)
- Anchoring: First number "sticks" even when irrelevant
- Counterargument: Maybe optimal for environments with autocorrelation?

**2. Irrational resource allocation**:
- Worrying excessively about unlikely events (catastrophizing)
- Overthinking simple decisions (analysis paralysis)
- These violate resource rationality principles

**3. Measurement challenges**:
- Hard to quantify "cognitive cost" objectively
- Post-hoc curve fitting: Can always find a cost function that explains behavior
- Need independent measures of resource consumption

### The Rational Process Models Debate

From [Tight resource-rational analysis](https://www.sciencedirect.com/science/article/pii/S1389041724000330) (Dimov et al., 2024, *Cognitive Systems Research*):
> "Resource rationality risks becoming unfalsifiable: if behavior looks irrational, just add a new cost term. Need tighter constraints from neuroscience and algorithmic complexity theory."

**Open question**: Which resource constraints are "real" vs post-hoc?
- Brain energy budget: Real, measurable
- "Attention is costly": But how costly, precisely?
- "Working memory limited": But limits vary with training, context

**Resolution**: Integrate with neuroscience (metabolic imaging, neural recordings)

## Sources

### Cognitive Science Literature

**Resource Rationality Foundation**:
- Lieder, F., & Griffiths, T. L. (2020). [Resource-rational analysis: Understanding human cognition as the optimal use of limited computational resources](https://cocosci.princeton.edu/papers/lieder_resource.pdf). *Behavioral and Brain Sciences*, 43, e1. (1036 citations)
- Bhui, R., Lai, L., & Gershman, S. J. (2021). [Resource-rational decision making](https://www.sciencedirect.com/science/article/abs/pii/S2352154621000371). *Current Opinion in Behavioral Sciences*, 41, 15-21. (213 citations)
- Gershman, S. J., Horvitz, E. J., & Tenenbaum, J. B. (2015). [Computational rationality: A converging paradigm for intelligence in brains, minds, and machines](https://www.science.org/doi/10.1126/science.aac6076). *Science*, 349(6245), 273-278.

**Bounded Rationality**:
- Simon, H. A. (1955). A behavioral model of rational choice. *Quarterly Journal of Economics*, 69(1), 99-118.
- Gigerenzer, G., & Goldstein, D. G. (1996). Reasoning the fast and frugal way: Models of bounded rationality. *Psychological Review*, 103(4), 650-669.

**Metalevel Reasoning**:
- Russell, S., & Wefald, E. (1991). [Principles of metareasoning](https://2024.sci-hub.se/196/44a218a9eb38b3f5b964793fff03b414/russell1991.pdf). *Artificial Intelligence*, 49(1-3), 361-395.
- Russell, S., & Subramanian, D. (1995). [Provably bounded-optimal agents](https://people.eecs.berkeley.edu/~russell/papers/aij-cnt.pdf). *Journal of Artificial Intelligence Research*, 2, 575-609.
- Good, I. J. (1971). Twenty-seven principles of rationality. In V. P. Godambe & D. A. Sprott (Eds.), *Foundations of Statistical Inference*. Holt, Rinehart & Winston.

**Anytime Algorithms**:
- Caizergues, E., et al. (2024). [Anytime sorting algorithms](https://dl.acm.org/doi/10.24963/ijcai.2024/785). *Proceedings of IJCAI 2024*.
- Nogueira, L., et al. (2024). [Exploiting anytime algorithms for collaborative service composition](https://www.mdpi.com/2073-431X/13/6/130). *Computer Science*, 13(6), 130.
- Zilberstein, S. (1996). Using anytime algorithms in intelligent systems. *AI Magazine*, 17(3), 73-83.

**Information-Theoretic Costs**:
- Ortega, P. A., & Braun, D. A. (2013). [Thermodynamics as a theory of decision-making with information-processing costs](https://royalsocietypublishing.org/doi/10.1098/rspa.2012.0683). *Proceedings of the Royal Society A*, 469(2153), 20120683.
- Friston, K. (2010). The free-energy principle: A unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.
- Attwell, D., & Laughlin, S. B. (2001). An energy budget for signaling in the grey matter of the brain. *Journal of Cerebral Blood Flow & Metabolism*, 21(10), 1133-1145.

**Neuroscience Evidence**:
- Salamone, J. D., & Correa, M. (2012). [The mysterious motivational functions of mesolimbic dopamine](https://www.nature.com/articles/nn.3917). *Neuron*, 76(3), 470-485.
- Botvinick, M. M., & Cohen, J. D. (2014). [The computational and neural basis of cognitive control: Charted territory and new frontiers](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4084861/). *Cognitive Science*, 38(6), 1249-1285.

**Critique**:
- Dimov, C. M., et al. (2024). [Tight resource-rational analysis](https://www.sciencedirect.com/science/article/pii/S1389041724000330). *Cognitive Systems Research*, 86, 101239.

### ML Systems Engineering (Influential Files)

**File 3: Tensor Parallel Resource Allocation**:
- [distributed-training/02-megatron-lm-tensor-parallelism.md](../distributed-training/02-megatron-lm-tensor-parallelism.md) - Metalevel decisions about computation/communication tradeoffs in distributed training

**File 7: Inference Server Batching Optimization**:
- [inference-optimization/02-triton-inference-server.md](../inference-optimization/02-triton-inference-server.md) - Dynamic batching as resource-rational latency/throughput optimization

**File 15: Hardware Heterogeneity Constraints**:
- [alternative-hardware/02-intel-oneapi-ml.md](../alternative-hardware/02-intel-oneapi-ml.md) - Device selection and memory hierarchy as bounded rationality problems

### ARR-COC-0-1 Implementation

**Vision Token Allocation**:
- [RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/attending.py](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/attending.py) - LOD allocation as resource-rational visual processing

**Platonic Dialogue**:
- [RESEARCH/PlatonicDialogues/46-mvp-be-doing/46-mvp-be-doing.md](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/46-mvp-be-doing.md) - Philosophical grounding of ARR-COC in cognitive science

### Additional References

- Cowan, N. (2001). The magical number 4 in short-term memory: A reconsideration of mental storage capacity. *Behavioral and Brain Sciences*, 24(1), 87-114.
- Green, L., & Mehr, D. R. (1997). What alters physicians' decisions to admit to the coronary care unit? *Journal of Family Practice*, 45(3), 219-226.
