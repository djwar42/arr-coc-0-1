# KNOWLEDGE DROP: Planning & Model-Based Control

**Timestamp**: 2025-11-16 21:14
**Part**: PART 22 of Cognitive Science Mastery expansion
**File Created**: `cognitive-mastery/21-planning-model-based-control.md`
**Lines**: ~730 lines
**Status**: ✓ COMPLETE

---

## What Was Created

Comprehensive knowledge file on **Planning & Model-Based Control** covering:

### Core Content (90%)

1. **Model-Based RL Fundamentals** (~100 lines)
   - Learned dynamics models (deterministic, probabilistic, ensemble, latent)
   - Planning algorithms (Random Shooting, CEM, MPC, MCTS)
   - Dyna architecture (real + simulated experience)

2. **Monte Carlo Tree Search** (~120 lines)
   - MCTS algorithm (Selection, Expansion, Simulation, Backprop)
   - AlphaZero: MCTS + deep learning (PUCT, self-play)
   - MuZero: Planning without environment rules (latent dynamics)

3. **Planning as Inference** (~90 lines)
   - Control as probabilistic inference (optimality variable)
   - Variational inference for planning (ELBO maximization)
   - Message passing (belief propagation on MDP graphs)

4. **World Models for Planning** (~90 lines)
   - Latent world models (Ha & Schmidhuber)
   - Dreamer: Actor-critic in imagination (RSSM)
   - Transformer world models (IRIS)

5. **Model Errors & Compounding** (~70 lines)
   - Model exploitation problem
   - Solutions: Short horizons, ensembles, pessimism, uncertainty-aware planning
   - Decision-focused learning

6. **Engineering: Pipeline Planning & Serving** (~100 lines)
   - File 2 (DeepSpeed): Planning across pipeline stages
   - File 6 (TensorRT): Fast planning for VLM serving (batched MCTS, quantized models)
   - File 14 (Apple Metal): On-device planning (MPS, adaptive depth)

7. **Research Topics** (~80 lines)
   - Sample-efficient model learning
   - Compositional world models
   - Causal models for planning
   - Hierarchical planning (options, feudal RL)

### ARR-COC-0-1 Connection (10%)

8. **Relevance Allocation as Planning** (~80 lines)
   - Look-ahead relevance planning (multi-turn VQA)
   - Model-based relevance realization (MCTS for token allocation)
   - Meta-learning planning policies
   - Expected free energy for allocation decisions

---

## Key Insights

### Model-Based vs Model-Free Trade-Offs

**Model-Based Advantages**:
- High sample efficiency (learn from imagination)
- Interpretable (can inspect plans)
- Rapid adaptation (re-plan with new goals)

**Model-Based Challenges**:
- Compounding model errors
- Computational cost (many queries)
- Requires accurate dynamics model

### AlphaZero → MuZero Evolution

**AlphaZero** (2017):
- Requires known environment rules
- Perfect for board games (Chess, Go, Shogi)
- MCTS + neural network guidance

**MuZero** (2019):
- Learns latent dynamics model
- Works for unknown environments (Atari)
- Plans in abstract state space (no reconstruction)

### Planning as Inference

**Core Insight**: Optimal control = inference in graphical model
- Introduce optimality variable O_t
- P(a_t | s_t, O_{1:T}) ∝ exp(Q(s_t, a_t))
- All variational inference tricks apply to planning

**Enables**:
- Belief propagation algorithms for planning
- Connection to active inference (expected free energy)
- Probabilistic robustness (posterior over models)

### Engineering for Real-Time Planning

**Critical for Production**:
1. **Batched MCTS**: Evaluate multiple leaf nodes in parallel (10-100× speedup)
2. **Quantized Models**: INT8 dynamics for faster inference
3. **Speculative Planning**: Cache common subplans
4. **Adaptive Depth**: Adjust planning horizon based on compute budget

**Latency Budget** (100ms VLM serving):
- Vision: 30ms
- Planning: 50ms
- Execution: 20ms

---

## ARR-COC-0-1 Applications

### Current (Reactive)
```
Query arrives → Measure relevance → Allocate tokens → Generate response
```

### Future (Anticipatory)
```
Query sequence → Predict future relevance → Plan allocations → Optimize long-term value
```

### Example: Multi-Turn VQA

**Without Planning**:
- Q1: "What objects?" → Allocate broadly
- Q2: "What color car?" → Re-allocate (wasteful)
- Q3: "License plate?" → Re-allocate again (inefficient)

**With Planning**:
- Predict query sequence pattern
- Q1 allocation considers Q2/Q3 needs
- Early exploration enables later exploitation
- 30-50% token savings across sequence

### MCTS for Token Allocation

**State**: Current allocation `{patch_i: budget_i}`
**Action**: Re-allocate tokens between patches
**Simulation**: Predict query performance
**Value**: Expected accuracy - token cost

**Planning enables**:
- Anticipatory allocation
- Amortized encoding
- User model adaptation

---

## Web Research Summary

**4 searches performed**:
1. Model-based RL 2024 planning
2. MCTS deep learning 2024
3. Planning as inference 2024
4. AlphaZero MuZero algorithms 2024

**Key papers scraped**:
- Decision-Focused MBRL (arXiv:2304.03365)
- What Type of Inference is Planning? (NeurIPS 2024)
- Multiagent Gumbel MuZero (AAAI 2024)
- MuZero paper (DeepMind)

**Citations**: 20+ papers with arXiv IDs, DOIs, access dates

---

## Connections to Existing Knowledge

**Cognitive Mastery Files**:
- 00-free-energy-principle-foundations.md (active inference)
- 18-multi-armed-bandits.md (tree-structured bandits)
- 06-bayesian-inference-deep.md (Bayesian planning)
- 13-mutual-information-correlation.md (information gain)
- 14-rate-distortion-theory.md (state compression)
- 05-cybernetics-control-theory.md (control hierarchies)

**Cognitive Foundations**:
- 09-reinforcement-learning-fundamentals.md (RL basics)

**Influenced By** (from plan):
- File 2: Pipeline parallelism (planning across stages)
- File 6: TensorRT VLM (fast planning for serving)
- File 14: Apple Metal (on-device planning)

---

## File Statistics

- **Total lines**: ~730
- **Sections**: 10 major sections
- **Subsections**: 30+ subsections
- **Code blocks**: 15+ examples
- **Tables**: 1 comparison table
- **Citations**: 20+ papers with full URLs
- **Cross-references**: 8 internal links to other cognitive-mastery files

---

## Quality Checklist

- [✓] Comprehensive coverage of planning & model-based control
- [✓] AlphaZero, MuZero, MCTS thoroughly explained
- [✓] Planning as inference framework included
- [✓] Engineering considerations (Files 2,6,14) integrated
- [✓] ARR-COC-0-1 connection (10%) with concrete examples
- [✓] Web research citations with access dates
- [✓] Cross-references to related cognitive files
- [✓] Clear progression from fundamentals to applications
- [✓] Code examples and mathematical formulations
- [✓] Open research questions identified

---

## Next Steps

1. Oracle will review this KNOWLEDGE DROP
2. Checkbox will be marked complete in ingestion.md
3. Continue to PART 23 (Resource-Rational Decision Making)
4. After all 42 PARTs complete: Consolidate → Update INDEX.md → Update SKILL.md → Git commit

**PART 22: COMPLETE** ✓
