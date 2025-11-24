# KNOWLEDGE DROP: Expected Free Energy Planning

**File**: ml-active-inference/02-expected-free-energy-planning.md
**Created**: 2025-11-23 14:45
**Lines**: ~700
**Focus**: EFE computation, epistemic/pragmatic value, MCTS tree search, PyTorch implementation

---

## What Was Created

Comprehensive guide to Expected Free Energy (EFE) as the central planning objective in active inference. Covers mathematical foundations, decompositions, tree search integration, and complete PyTorch implementations.

---

## Key Sections

### 1. EFE Computation (~100 lines)
- Mathematical definition and decomposition
- Contrast with Variational Free Energy (VFE)
- Free Energy of the Future (FEF) alternative
- Key insight: EFE = FEF - Information_Gain

### 2. Epistemic + Pragmatic Value (~100 lines)
- Exploration-exploitation decomposition
- Risk and ambiguity alternative decomposition
- Automatic scheduling of exploration
- Connection to Bayesian surprise

### 3. Tree Search with EFE (~120 lines)
- MCTS for active inference
- UCB-EFE connection
- Active Inference Tree Search algorithm
- Boosting MCTS with free energy

### 4. PyTorch Implementation (~250 lines)
- `EFEPlanningAgent` class
- `MCTSNode` for tree search
- `ActiveInferenceMCTS` integration
- Training utilities and VFE loss
- Performance considerations

### 5. TRAIN STATION: EFE = UCB = Thompson (~100 lines)
- Mathematical proof of equivalence
- Precision as exploration parameter
- Code demonstration
- Grand unification table

### 6. ARR-COC Connection (~100 lines)
- Token allocation as EFE minimization
- Hierarchical planning for pyramids
- Precision weighting for budget
- `EFETokenRouter` implementation sketch

---

## TRAIN STATION Discovery

**EFE = UCB = Thompson Sampling**

All three exploration strategies are topologically equivalent:
- EFE epistemic value = UCB exploration bonus = Thompson posterior variance
- EFE pragmatic value = UCB exploitation = Thompson posterior mean
- Precision parameter = UCB coefficient c = Thompson variance scaling

This unifies:
- Neuroscience (active inference)
- AI planning (MCTS/bandits)
- Bayesian decision theory (Thompson)

---

## Code Highlights

```python
# Core EFE computation
def compute_efe(prior_mean, prior_logvar, action, horizon=1):
    # PRAGMATIC: -E[ln p~(o)]
    pragmatic = -torch.sum(pred_obs * log_preference, dim=-1)

    # EPISTEMIC: -entropy (uncertainty to resolve)
    prior_entropy = 0.5 * torch.sum(1 + next_logvar, dim=-1)
    epistemic = -prior_entropy

    return pragmatic + epistemic

# Action selection via EFE
action_probs = F.softmax(-precision * efe_tensor, dim=-1)
```

---

## Sources Used

- Millidge et al. 2021 "Whence the Expected Free Energy?" (Neural Computation)
- Fountas et al. 2020 "Deep Active Inference Using Monte-Carlo Methods" (NeurIPS)
- Friston et al. 2015 "Active Inference and Epistemic Value" (920+ citations)
- Dao et al. 2025 "Boosting MCTS with Free Energy Minimization"
- GitHub: zfountas/deep-active-inference-mc

---

## ARR-COC Relevance

Token allocation directly maps to active inference action selection:
- **Actions** = Token allocation decisions
- **Expected Free Energy** = Expected Relevance Gain
- **Epistemic value** = Information gain from processing region
- **Pragmatic value** = Task-relevant value of region

Implementation approach: `EFETokenRouter` using precision-weighted softmax over region EFE values.

---

## Performance Notes

**Memory**: ~2KB per MCTS node (state mean + logvar + statistics)
**Speed**: 100 simulations x 5 horizon = 500 forward passes
**GPU**: Batch all state transitions, parallelize EFE computation
**Numerical**: Clamp log-variances, use stable KL computation

---

*KNOWLEDGE DROP complete*
