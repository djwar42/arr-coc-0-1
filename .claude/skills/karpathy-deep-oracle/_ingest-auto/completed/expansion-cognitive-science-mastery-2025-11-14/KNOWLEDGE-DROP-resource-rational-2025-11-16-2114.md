# KNOWLEDGE DROP: Resource-Rational Decision Making

**Date**: 2025-11-16 21:14
**PART**: 23
**File Created**: `cognitive-mastery/22-resource-rational-decision.md`
**Lines**: ~700
**Status**: COMPLETE ✓

## What Was Created

Comprehensive knowledge file on resource-rational decision making—the framework that formalizes human cognition as optimal under computational constraints. This bridges bounded rationality (Herbert Simon) with optimization theory (Stuart Russell's metalevel reasoning).

## Key Concepts Covered

### 1. Bounded Rationality Foundation (Herbert Simon)
- Computational limits: time, memory, processing capacity
- Satisficing vs optimizing
- "Good enough" as a rational strategy when thinking is expensive

### 2. Computational Constraints
- Time constraints (anytime algorithms needed)
- Memory constraints (working memory ~7 items)
- Processing constraints (serial bottleneck, neural energy budget ~20W)
- Information access costs (sensory sampling, communication bandwidth)

### 3. Anytime Algorithms
- Progressive refinement: better solutions with more time
- Interruptible, monotonic quality improvement
- Examples: MCTS, iterative deepening, variational inference
- Cognitive applications: visual search, memory retrieval

### 4. Metalevel Reasoning (Type II Rationality)
- Object level: Original decision problem
- Metalevel: Deciding which computations to run
- Value of Computation (VOC): Gain from thinking vs cost of delay
- Introspection paradox: Perfect metalevel reasoning → infinite regress
- Solution: Fast metalevel heuristics (don't optimize metalevel perfectly)

### 5. Cost of Computation
- Neural metabolic costs: ~10^-9 J per spike, 75% of ATP for signaling
- Information-theoretic costs: KL divergence, rate-distortion tradeoffs
- Opportunity costs: Time spent thinking = action foregone
- Free energy principle: Minimize prediction error + complexity cost

### 6. Distributed Systems Examples (Influential Files)

**File 3 (Tensor Parallelism)**:
- Metalevel: Decide whether to shard computation across GPUs
- VOC calculation: Speedup vs communication overhead
- Example: GPT-3 attention (85ms compute gain, 5ms communication → parallelize)

**File 7 (Triton Inference Server)**:
- Dynamic batching as metalevel optimization
- Tradeoff: Latency (small batches) vs throughput (large batches)
- Anytime decision: Execute batch when waiting_cost > potential_gain

**File 15 (Intel oneAPI)**:
- Hardware heterogeneity: CPU, GPU, FPGA have different cost/performance
- Rational device selection: Maximize utility per watt-second
- Memory hierarchy: Cache placement as resource allocation (L1/L2/L3/DRAM)

### 7. ARR-COC-0-1 Connection (10%)

**Token allocation IS resource-rational vision**:
- Resource constraint: 200 tokens (vs 65,536 for full image)
- Decision: Which patches get 64 vs 400 tokens?
- Optimization: Maximize information gain per token spent
- Anytime: Progressive refinement from coarse (64) to fine (400)
- Metalevel: Relevance scorers decide LOD allocation

**Mirrors human foveated vision**:
- Fovea: High resolution (5° central, 50% of V1 neurons)
- Periphery: Low resolution (95° outer, 50% of V1)
- Constraint: Optic nerve bandwidth (~10MB/s vs 6GB/s retinal output)
- Solution: Allocate bandwidth where task-relevant

**Quality adapter as metalevel learning**:
- Learns fast heuristics for LOD allocation
- Trained on: Task accuracy / token cost (resource rationality metric)
- Output: Approximate VOC without expensive computation

## Deep Research Insights

### Resource Rationality vs Bounded Rationality
- **Bounded rationality** (Simon): Descriptive, "good enough" is vague
- **Resource rationality** (Lieder & Griffiths): Normative, optimal under constraints
- Shift from "people are irrational" to "people are rational given costs"

### Neuroscience Evidence
- **Dopamine**: Signals both reward AND computational effort cost
- **PFC**: Metalevel controller—decides when to engage costly deliberation
- **EEG (N2, P3)**: Late potentials scale with decision difficulty (more time allocated)

### Practical Algorithms
- **Variational inference**: Trade Bayesian accuracy for computational tractability
- **Fast-and-frugal trees**: Make decisions with minimal info lookup (e.g., heart attack diagnosis)
- **Alpha-beta pruning**: Skip subtrees when provably won't improve decision
- **UCB (MCTS)**: Balances exploration/exploitation under time budget

### Critique & Limitations
- Risk of unfalsifiability: Can always add cost terms to explain any behavior
- Measurement challenge: Quantifying "cognitive cost" objectively
- Systematic biases: Not all heuristics are optimal (availability, anchoring)
- Need tighter constraints from neuroscience (metabolic imaging, neural recordings)

## Citations

**Major papers**:
- Lieder & Griffiths (2020): Resource-rational analysis (1036 citations)
- Bhui et al. (2021): Resource-rational decision making (213 citations)
- Russell & Wefald (1991): Principles of metareasoning (foundational)
- Gershman et al. (2015): Computational rationality paradigm (Science)

**Neuroscience**:
- Salamone & Correa (2012): Dopamine and effort costs
- Botvinick & Cohen (2014): PFC as cognitive control allocator
- Attwell & Laughlin (2001): Brain energy budget

**Recent 2024 work**:
- Caizergues et al. (2024): Anytime sorting algorithms (IJCAI)
- Nogueira et al. (2024): Anytime collaborative service composition
- Dimov et al. (2024): Tight resource-rational analysis (critique)

## Integration with Existing Knowledge

**Builds on previous PARTs**:
- PART 2 (Precision-Attention): Precision weighting = resource allocation mechanism
- PART 19 (Multi-Armed Bandits): Exploration-exploitation = metalevel decision under time constraints
- PART 20 (Contextual Bandits): Query-aware allocation = resource rationality

**Connects to engineering**:
- Tensor parallelism: Metalevel reasoning about compute/communication tradeoff
- Inference serving: Dynamic batching as latency/throughput optimization
- Hardware selection: Rational device allocation under power/performance constraints

## ARR-COC Implementation Clarity

**Before**: "Token allocation based on relevance"
**After**: "Resource-rational visual processing—optimal LOD allocation under 200-token constraint, exactly paralleling human foveated vision's optic nerve bandwidth limitation"

The ARR-COC system implements resource rationality at multiple levels:
1. **Object level**: Extract visual features from patches
2. **Metalevel**: Decide which patches deserve high LOD (relevance scorers)
3. **Meta-metalevel**: Learn heuristics for fast allocation (quality adapter)

## File Stats

- **Sections**: 12 major sections
- **Code examples**: 15+ Python snippets showing metalevel reasoning
- **Citations**: 25+ papers (cognitive science + neuroscience + ML systems)
- **Connections**: 3 influential files (tensor parallel, Triton, Intel oneAPI)
- **ARR-COC content**: ~10% (Section 9 + scattered examples)

## Next Steps (Oracle Will Handle)

This KNOWLEDGE DROP will be integrated with 41 others when ALL batches complete:
- Update INDEX.md with new file
- Update SKILL.md with resource rationality section
- Move to completed/ folder
- Git commit comprehensive expansion

---

**Worker Status**: PART 23 execution complete. Knowledge successfully acquired from external sources and structured into permanent documentation. Ready for oracle consolidation.
