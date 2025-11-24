# Knowledge Expansion: Cognitive Foundations & Research Methodology (24 runners in 6 batches)

**Date**: 2025-11-14
**Goal**: Theoretical + Experimental foundations FOR ARR-COC-0-1
**Strategy**: 24 runners, 4 at a time (6 batches)
**Total**: ~16,800 lines across 24 files
**Focus**: GOING WIDE - Cognitive science, active inference, research methodology, information theory

---

## 🚀 HOW TO EXECUTE THIS EXPANSION

**BATCH EXECUTION SYSTEM** (Recommended: 4 runners per batch, but flexible)

### Why Batches?
- **Quality Control**: Review results between batches
- **Token Management**: Avoid overwhelming context windows
- **Error Recovery**: Fix issues before continuing
- **Progress Tracking**: Clear milestones

### Recommended: 4 Runners Per Batch
- ✅ **4 runners**: Optimal balance (quality + speed)
- ⚠️ **6 runners**: Acceptable if experienced
- ❌ **8+ runners**: Not recommended (too much to review)

### Execution Pattern
1. **Launch Batch**: Run 4 runners in parallel
2. **Review Results**: Check KNOWLEDGE DROP files
3. **Fix Issues**: Retry any failures
4. **Next Batch**: Continue to next 4 runners
5. **Consolidate**: Big integration at the END of ALL batches

### Worker Instructions
- ✅ **Create KNOWLEDGE DROPS**: Every runner creates KNOWLEDGE-DROP-*.md
- ✅ **Check existing knowledge**: Read relevant files FIRST
- ✅ **Follow the plan**: Execute steps as written
- ✅ **Return results**: Report success/failure clearly

### Oracle Instructions (Consolidation)
After ALL batches complete:
1. **Read all KNOWLEDGE DROP files**
2. **Update INDEX.md** with all new files
3. **Update SKILL.md** (if major changes)
4. **Move to completed/**
5. **Git commit** with comprehensive message

---

## 📋 THE 16 INFLUENTIAL FILES (Explicit Reference)

**Distributed Training (4 files)**:
1. `distributed-training/00-deepspeed-zero-optimizer.md` - Multi-GPU memory optimization
2. `distributed-training/01-deepspeed-pipeline-parallelism.md` - Pipeline parallel patterns
3. `distributed-training/02-megatron-lm-tensor-parallelism.md` - Tensor parallel strategies
4. `distributed-training/03-fsdp-vs-deepspeed.md` - Distributed framework comparison

**Inference Optimization (4 files)**:
5. `inference-optimization/00-tensorrt-fundamentals.md` - GPU inference acceleration
6. `inference-optimization/01-tensorrt-vlm-deployment.md` - VLM serving optimization
7. `inference-optimization/02-triton-inference-server.md` - Multi-model GPU serving
8. `inference-optimization/03-torch-compile-aot-inductor.md` - PyTorch compilation

**Orchestration (4 files)**:
9. `orchestration/00-kubernetes-gpu-scheduling.md` - K8s GPU workloads
10. `orchestration/01-kubeflow-ml-pipelines.md` - ML pipeline orchestration
11. `orchestration/02-ray-distributed-ml.md` - Ray for distributed compute
12. `orchestration/03-ml-workload-patterns-k8s.md` - Production ML patterns

**Alternative Hardware (4 files)**:
13. `alternative-hardware/00-amd-rocm-ml.md` - AMD GPU alternatives
14. `alternative-hardware/01-apple-metal-ml.md` - Apple Silicon patterns
15. `alternative-hardware/02-intel-oneapi-ml.md` - Intel accelerator strategies
16. `alternative-hardware/03-tpu-programming-fundamentals.md` - TPU architecture

---

## ⚠️ EXECUTION PLAN: 6 BATCHES OF 4 RUNNERS

**CRITICAL**: Run ONLY 4 runners at a time! Review results between batches.

- **Batch 1**: PARTs 1-4 (Active Inference & Predictive Processing)
- **Batch 2**: PARTs 5-8 (Information Theory & Bayesian Methods)
- **Batch 3**: PARTs 9-12 (Decision Making & Resource Allocation)
- **Batch 4**: PARTs 13-16 (Experimental Design & Research Methodology)
- **Batch 5**: PARTs 17-20 (Cognitive Architectures & Embodied AI)
- **Batch 6**: PARTs 21-24 (Paper Writing & Publication)

---

# BATCH 1: Active Inference & Predictive Processing (4 runners, ~2,800 lines)

## PART 1: Active Inference & Free Energy Principle (~700 lines)

- [✓] PART 1: Create cognitive-foundations/00-active-inference-free-energy.md (Completed 2025-11-14 14:40)

**Step 0: Check Existing Knowledge**
- [ ] Read john-vervaeke-oracle/ knowledge (relevance realization framework)
- [ ] Read karpathy/knowing.py, balancing.py, attending.py, realizing.py concepts

**Influenced by**: (Vervaeke knowledge) - Active inference IS relevance realization

**Step 1: Web Research**
- [ ] Search: "Karl Friston free energy principle 2024"
- [ ] Search: "active inference tutorial computational"
- [ ] Search: "variational free energy minimization"
- [ ] Search: "active inference vs reinforcement learning"

**Step 2: Create Knowledge File**
- [ ] Section 1: Free Energy Principle fundamentals (surprise minimization, variational inference)
- [ ] Section 2: Active inference (perception + action to minimize prediction error)
- [ ] Section 3: Generative models (internal models of the world, hierarchical)
- [ ] Section 4: Epistemic vs pragmatic value (exploration vs exploitation)
- [ ] Section 5: Precision weighting (attention as gain control, confidence)
- [ ] Section 6: Temporal depth (planning horizons, multi-scale dynamics)
- [ ] Section 7: Connection to machine learning (VAEs, predictive coding networks)
- [ ] Section 8: **ARR-COC-0-1 as active inference** (relevance realization = free energy minimization)
- [ ] **CITE**: john-vervaeke-oracle/ (relevance realization); knowing.py concepts

**Step 3: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-active-inference-2025-11-14-[TIME].md

---

## PART 2: Predictive Processing & Predictive Coding (~700 lines)

- [✓] PART 2: Create cognitive-foundations/01-predictive-processing-hierarchical.md (Completed 2025-11-14 14:40)

**Step 0: Check Existing Knowledge**
- [ ] Read biological-vision/ (cortical processing, visual streams)
- [ ] Read john-vervaeke-oracle/ (opponent processing, salience)

**Influenced by**: (Biological vision knowledge) - Predictive coding in visual cortex

**Step 1: Web Research**
- [ ] Search: "predictive processing brain 2024"
- [ ] Search: "predictive coding hierarchical neural networks"
- [ ] Search: "prediction error minimization perception"
- [ ] Search: "Bayesian brain hypothesis evidence"

**Step 2: Create Knowledge File**
- [ ] Section 1: Predictive processing framework (brain as prediction machine)
- [ ] Section 2: Predictive coding (error propagation, top-down vs bottom-up)
- [ ] Section 3: Hierarchical processing (multi-level predictions, temporal scales)
- [ ] Section 4: Precision weighting of prediction errors (attention allocation)
- [ ] Section 5: Neural implementation (cortical microcircuits, pyramidal neurons)
- [ ] Section 6: Evidence from neuroscience (visual illusions, binocular rivalry)
- [ ] Section 7: Computational models (predictive coding networks, neural implementation)
- [ ] Section 8: **ARR-COC-0-1 predictive architecture** (propositional knowing as prediction)
- [ ] **CITE**: biological-vision/; john-vervaeke-oracle/ (opponent processing)

**Step 3: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-predictive-processing-2025-11-14-[TIME].md

---

## PART 3: Bayesian Brain & Probabilistic Inference (~700 lines)

- [✓] PART 3: Create cognitive-foundations/02-bayesian-brain-probabilistic.md (Completed 2025-11-14 14:40)

**Step 0: Check Existing Knowledge**
- [ ] Read john-vervaeke-oracle/ (uncertainty, transjective knowing)

**Influenced by**: (Vervaeke knowledge) - Relevance under uncertainty

**Step 1: Web Research**
- [ ] Search: "Bayesian brain hypothesis 2024"
- [ ] Search: "probabilistic inference perception"
- [ ] Search: "Bayesian cue integration multisensory"
- [ ] Search: "uncertainty representation brain"

**Step 2: Create Knowledge File**
- [ ] Section 1: Bayesian brain hypothesis (perceptual inference as Bayesian inference)
- [ ] Section 2: Prior beliefs and likelihood (generative models, sensory evidence)
- [ ] Section 3: Posterior inference (combining priors + evidence)
- [ ] Section 4: Bayesian cue integration (optimal multisensory fusion)
- [ ] Section 5: Uncertainty representation (distributional codes, probabilistic population codes)
- [ ] Section 6: Empirical evidence (psychophysics, neural correlates)
- [ ] Section 7: Computational models (Bayesian networks, particle filters, variational inference)
- [ ] Section 8: **ARR-COC-0-1 Bayesian relevance** (query-aware priors, posterior token allocation)
- [ ] **CITE**: john-vervaeke-oracle/ (uncertainty, transjective)

**Step 3: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-bayesian-brain-2025-11-14-[TIME].md

---

## PART 4: Attention as Resource Allocation (~700 lines)

- [✓] PART 4: Create cognitive-foundations/03-attention-resource-allocation.md (Completed 2025-11-14 14:42)

**Step 0: Check Existing Knowledge**
- [ ] Read biological-vision/00-gestalt-visual-attention.md
- [ ] Read biological-vision/02-eye-tracking-task-attention.md
- [ ] Read john-vervaeke-oracle/ (attending.py concepts)

**Influenced by**: (Biological attention knowledge) - Attention mechanisms

**Step 1: Web Research**
- [ ] Search: "attention resource allocation cognitive neuroscience 2024"
- [ ] Search: "limited capacity attention bottleneck"
- [ ] Search: "attention control biased competition"
- [ ] Search: "endogenous exogenous attention"

**Step 2: Create Knowledge File**
- [ ] Section 1: Attention as limited resource (capacity constraints, bottleneck theories)
- [ ] Section 2: Biased competition model (neural competition for representation)
- [ ] Section 3: Endogenous vs exogenous attention (goal-driven vs stimulus-driven)
- [ ] Section 4: Feature-based vs spatial attention (what vs where)
- [ ] Section 5: Attention control networks (fronto-parietal, dorsal/ventral attention)
- [ ] Section 6: Attention and working memory (resource sharing, trade-offs)
- [ ] Section 7: Computational models (normalization models, priority maps)
- [ ] Section 8: **ARR-COC-0-1 token allocation** (attention budget 64-400, relevance-driven)
- [ ] **CITE**: biological-vision/00,02; john-vervaeke-oracle/ (attending.py)

**Step 3: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-attention-resource-2025-11-14-[TIME].md

---

# BATCH 2: Information Theory & Bayesian Methods (4 runners, ~2,800 lines)

## PART 5: Information Theory Deep Dive (~700 lines)

- [✓] PART 5: Create information-theory/00-shannon-entropy-mutual-information.md (Completed 2025-11-14 15:16)

**Step 0: Check Existing Knowledge**
- [ ] Read john-vervaeke-oracle/ (propositional knowing, information content)
- [ ] Read karpathy/knowing.py (InformationScorer using Shannon entropy)

**Influenced by**: (Vervaeke propositional knowing) - Information theoretic measures

**Step 1: Web Research**
- [ ] Search: "information theory entropy mutual information 2024"
- [ ] Search: "KL divergence cross entropy neural networks"
- [ ] Search: "rate distortion theory compression"
- [ ] Search: "information bottleneck principle"

**Step 2: Create Knowledge File**
- [ ] Section 1: Shannon entropy (uncertainty, surprise, self-information)
- [ ] Section 2: Mutual information (shared information, independence)
- [ ] Section 3: KL divergence (relative entropy, distribution distance)
- [ ] Section 4: Cross-entropy (loss functions, classification)
- [ ] Section 5: Rate-distortion theory (lossy compression, optimal trade-offs)
- [ ] Section 6: Information bottleneck (relevant information extraction)
- [ ] Section 7: Applications to ML (loss functions, regularization, VAEs)
- [ ] Section 8: **ARR-COC-0-1 information measures** (propositional knowing = entropy, compression)
- [ ] **CITE**: john-vervaeke-oracle/ (propositional); knowing.py (InformationScorer)

**Step 3: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-information-theory-2025-11-14-[TIME].md

---

## PART 6: Bayesian Methods & Inference (~700 lines)

- [ ] PART 6: Create cognitive-foundations/05-bayesian-methods-inference.md

**Step 0: Check Existing Knowledge**
- [ ] Read training-llms/ (optimization, uncertainty)

**Influenced by**: (Training knowledge) - Bayesian optimization and uncertainty

**Step 1: Web Research**
- [ ] Search: "Bayesian inference MCMC variational 2024"
- [ ] Search: "Bayesian neural networks uncertainty"
- [ ] Search: "Bayesian optimization hyperparameter tuning"
- [ ] Search: "variational inference deep learning"

**Step 2: Create Knowledge File**
- [ ] Section 1: Bayesian inference fundamentals (posterior = prior × likelihood)
- [ ] Section 2: MCMC methods (Metropolis-Hastings, Gibbs sampling, HMC)
- [ ] Section 3: Variational inference (ELBO, mean-field approximation)
- [ ] Section 4: Bayesian neural networks (weight uncertainty, epistemic uncertainty)
- [ ] Section 5: Bayesian optimization (Gaussian processes, acquisition functions)
- [ ] Section 6: Uncertainty quantification (epistemic vs aleatoric)
- [ ] Section 7: Applications to ML (Bayesian deep learning, uncertainty-aware predictions)
- [ ] Section 8: **ARR-COC-0-1 uncertainty** (relevance under uncertainty, Bayesian token allocation)
- [ ] **CITE**: training-llms/ (optimization)

**Step 3: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-bayesian-methods-2025-11-14-[TIME].md

---

## PART 7: Statistical Learning Theory (~700 lines)

- [✓] PART 7: Create cognitive-foundations/06-statistical-learning-theory.md (Completed 2025-11-14 15:17)

**Step 0: Check Existing Knowledge**
- [✓] Read training-llms/ (generalization, overfitting)

**Influenced by**: (Training knowledge) - Generalization theory

**Step 1: Web Research**
- [ ] Search: "statistical learning theory VC dimension 2024"
- [ ] Search: "PAC learning generalization bounds"
- [ ] Search: "Rademacher complexity neural networks"
- [ ] Search: "bias-variance tradeoff deep learning"

**Step 2: Create Knowledge File**
- [ ] Section 1: Statistical learning framework (ERM, hypothesis classes)
- [ ] Section 2: VC dimension (capacity, shattering, generalization bounds)
- [ ] Section 3: PAC learning (sample complexity, learnability)
- [ ] Section 4: Rademacher complexity (data-dependent bounds)
- [ ] Section 5: Bias-variance tradeoff (underfitting vs overfitting)
- [ ] Section 6: Regularization theory (ridge, lasso, elastic net)
- [ ] Section 7: Deep learning generalization (implicit regularization, flat minima)
- [ ] Section 8: **ARR-COC-0-1 generalization** (token budget vs accuracy tradeoff)
- [ ] **CITE**: training-llms/ (generalization)

**Step 3: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-statistical-learning-2025-11-14-[TIME].md

---

## PART 8: Causal Inference (~700 lines)

- [✓] PART 8: Create cognitive-foundations/07-causal-inference.md (Completed 2025-11-14 15:18)

**Step 0: Check Existing Knowledge**
- [ ] Read john-vervaeke-oracle/ (participatory knowing, agent-arena coupling)

**Influenced by**: (Vervaeke participatory) - Causal intervention and agency

**Step 1: Web Research**
- [ ] Search: "causal inference Pearl DAGs 2024"
- [ ] Search: "do-calculus interventions counterfactuals"
- [ ] Search: "causal discovery structure learning"
- [ ] Search: "causal machine learning"

**Step 2: Create Knowledge File**
- [ ] Section 1: Causal inference fundamentals (correlation vs causation)
- [ ] Section 2: Directed Acyclic Graphs (DAGs, structural causal models)
- [ ] Section 3: do-calculus (interventions, observational vs interventional)
- [ ] Section 4: Counterfactuals (potential outcomes, causal effects)
- [ ] Section 5: Causal discovery (structure learning, constraint-based, score-based)
- [ ] Section 6: Confounding and bias (backdoor criterion, front-door adjustment)
- [ ] Section 7: Causal machine learning (causal representation learning, IRM)
- [ ] Section 8: **ARR-COC-0-1 causal relevance** (participatory knowing = causal intervention)
- [ ] **CITE**: john-vervaeke-oracle/ (participatory knowing)

**Step 3: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-causal-inference-2025-11-14-[TIME].md

---

# BATCH 3: Decision Making & Resource Allocation (4 runners, ~2,800 lines)

## PART 9: Multi-Armed Bandits (~700 lines)

- [✓] PART 9: Create cognitive-foundations/08-multi-armed-bandits.md (Completed 2025-11-14 15:29)

**Step 0: Check Existing Knowledge**
- [ ] Read john-vervaeke-oracle/ (opponent processing: exploit vs explore)

**Influenced by**: (Vervaeke opponent processing) - Exploration-exploitation

**Step 1: Web Research**
- [ ] Search: "multi-armed bandits UCB Thompson sampling 2024"
- [ ] Search: "exploration exploitation tradeoff"
- [ ] Search: "contextual bandits personalization"
- [ ] Search: "regret bounds bandit algorithms"

**Step 2: Create Knowledge File**
- [ ] Section 1: Multi-armed bandit problem (exploration vs exploitation)
- [ ] Section 2: Classical algorithms (epsilon-greedy, UCB, Thompson sampling)
- [ ] Section 3: Contextual bandits (context-aware decisions, LinUCB)
- [ ] Section 4: Regret bounds (cumulative regret, optimal algorithms)
- [ ] Section 5: Bayesian bandits (posterior sampling, Thompson sampling)
- [ ] Section 6: Non-stationary bandits (changing reward distributions)
- [ ] Section 7: Applications (A/B testing, recommendation systems, resource allocation)
- [ ] Section 8: **ARR-COC-0-1 token allocation** (bandit for dynamic token budget)
- [ ] **CITE**: john-vervaeke-oracle/ (exploit vs explore tension)

**Step 3: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-multi-armed-bandits-2025-11-14-[TIME].md

---

## PART 10: Reinforcement Learning Fundamentals (~700 lines)

- [✓] PART 10: Create cognitive-foundations/09-reinforcement-learning-fundamentals.md (Completed 2025-11-14 15:31)

**Step 0: Check Existing Knowledge**
- [✓] Read john-vervaeke-oracle/ (procedural knowing, learning how)

**Influenced by**: (Vervaeke procedural knowing) - Learning through interaction

**Step 1: Web Research**
- [ ] Search: "reinforcement learning Q-learning DQN 2024"
- [ ] Search: "policy gradient methods REINFORCE"
- [ ] Search: "actor-critic algorithms PPO"
- [ ] Search: "model-based vs model-free RL"

**Step 2: Create Knowledge File**
- [ ] Section 1: RL fundamentals (MDP, agent, environment, reward)
- [ ] Section 2: Value-based methods (Q-learning, DQN, Double DQN)
- [ ] Section 3: Policy gradient methods (REINFORCE, A2C, PPO)
- [ ] Section 4: Actor-critic architectures (advantage, baseline)
- [ ] Section 5: Model-based RL (world models, planning, Dyna)
- [ ] Section 6: Exploration strategies (epsilon-greedy, entropy regularization)
- [ ] Section 7: Applications to ML (RLHF, reward shaping, curriculum learning)
- [ ] Section 8: **ARR-COC-0-1 RL training** (reward for relevance allocation quality)
- [ ] **CITE**: john-vervaeke-oracle/ (procedural knowing)

**Step 3: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-reinforcement-learning-2025-11-14-[TIME].md

---

## PART 11: Decision Theory & Utility (~700 lines)

- [ ] PART 11: Create cognitive-foundations/10-decision-theory-utility.md

**Step 0: Check Existing Knowledge**
- [ ] Read john-vervaeke-oracle/ (balancing.py, tension navigation)

**Influenced by**: (Vervaeke balancing) - Decision making under constraints

**Step 1: Web Research**
- [ ] Search: "decision theory expected utility 2024"
- [ ] Search: "prospect theory behavioral economics"
- [ ] Search: "multi-criteria decision making"
- [ ] Search: "Pareto optimality tradeoffs"

**Step 2: Create Knowledge File**
- [ ] Section 1: Decision theory fundamentals (outcomes, probabilities, utilities)
- [ ] Section 2: Expected utility theory (rational choice, von Neumann-Morgenstern)
- [ ] Section 3: Prospect theory (loss aversion, reference dependence, framing)
- [ ] Section 4: Multi-criteria decision making (Pareto optimality, TOPSIS)
- [ ] Section 5: Decision under uncertainty (ambiguity, Ellsberg paradox)
- [ ] Section 6: Temporal discounting (hyperbolic discounting, intertemporal choice)
- [ ] Section 7: Applications (risk assessment, cost-benefit analysis)
- [ ] Section 8: **ARR-COC-0-1 decision making** (balancing tensions, opponent processing)
- [ ] **CITE**: john-vervaeke-oracle/ (balancing.py)

**Step 3: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-decision-theory-2025-11-14-[TIME].md

---

## PART 12: Resource Allocation & Optimization (~700 lines)

- [✓] PART 12: Create cognitive-foundations/11-resource-allocation-optimization.md (Completed 2025-11-14 15:30)

**Step 0: Check Existing Knowledge**
- [✓] Read john-vervaeke-oracle/ (attending.py, salience realization)
- [✓] Read orchestration/03-ml-workload-patterns-k8s.md (resource scheduling)

**Influenced by**: Files 12, (Vervaeke attending) - Optimal resource allocation

**Step 1: Web Research**
- [ ] Search: "resource allocation optimization 2024"
- [ ] Search: "linear programming convex optimization"
- [ ] Search: "knapsack problem dynamic programming"
- [ ] Search: "online algorithms competitive analysis"

**Step 2: Create Knowledge File**
- [ ] Section 1: Resource allocation problems (objectives, constraints)
- [ ] Section 2: Linear programming (simplex, dual, integer programming)
- [ ] Section 3: Convex optimization (gradient descent, KKT conditions)
- [ ] Section 4: Dynamic programming (knapsack, optimal substructure)
- [ ] Section 5: Online algorithms (competitive ratio, ski rental problem)
- [ ] Section 6: Approximation algorithms (greedy, local search)
- [ ] Section 7: Applications (compute allocation, bandwidth, attention budgets)
- [ ] Section 8: **ARR-COC-0-1 token budget** (optimal allocation 64-400 tokens)
- [ ] **CITE**: john-vervaeke-oracle/ (attending.py); orchestration/03 (scheduling)

**Step 3: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-resource-allocation-2025-11-14-[TIME].md

---

# BATCH 4: Experimental Design & Research Methodology (4 runners, ~2,800 lines)

## PART 13: Experimental Design Fundamentals (~700 lines)

- [✓] PART 13: Create research-methodology/00-experimental-design.md (Completed 2025-11-14 15:47)

**Step 0: Check Existing Knowledge**
- [ ] Read practical-implementation/55-vlm-inference-latency-benchmarks.md (benchmarking)
- [ ] Read practical-implementation/56-vision-token-budget-ablations.md (ablations)

**Influenced by**: (Benchmarking and ablation knowledge) - Rigorous experiments

**Step 1: Web Research**
- [ ] Search: "experimental design psychology research 2024"
- [ ] Search: "randomized controlled trials RCT"
- [ ] Search: "factorial design ANOVA"
- [ ] Search: "within-subjects between-subjects design"

**Step 2: Create Knowledge File**
- [ ] Section 1: Experimental design principles (control, randomization, replication)
- [ ] Section 2: Independent vs dependent variables (manipulation, measurement)
- [ ] Section 3: Between-subjects vs within-subjects (power, counterbalancing)
- [ ] Section 4: Factorial designs (main effects, interactions, ANOVA)
- [ ] Section 5: Control conditions (baseline, active control, sham)
- [ ] Section 6: Confounds and threats to validity (internal, external, construct)
- [ ] Section 7: Sample size and power analysis (effect size, statistical power)
- [ ] Section 8: **ARR-COC-0-1 experiments** (ablation study design, human evaluation)
- [ ] **CITE**: practical-implementation/55,56 (benchmarking, ablations)

**Step 3: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-experimental-design-2025-11-14-[TIME].md

---

## PART 14: Psychophysics & Human Studies (~700 lines)

- [✓] PART 14: Create research-methodology/01-psychophysics-human-studies.md (Completed 2025-11-14 15:48)

**Step 0: Check Existing Knowledge**
- [ ] Read biological-vision/02-eye-tracking-task-attention.md (eye tracking methods)
- [ ] Read biological-vision/00-gestalt-visual-attention.md (perceptual phenomena)

**Influenced by**: (Eye tracking and perception knowledge) - Human perception studies

**Step 1: Web Research**
- [ ] Search: "psychophysics methods 2024"
- [ ] Search: "signal detection theory d-prime"
- [ ] Search: "just noticeable difference Weber's law"
- [ ] Search: "method of constant stimuli adaptive staircase"

**Step 2: Create Knowledge File**
- [ ] Section 1: Psychophysics fundamentals (stimulus-response relationships)
- [ ] Section 2: Classical methods (method of limits, constant stimuli, adjustment)
- [ ] Section 3: Adaptive methods (staircase procedures, QUEST, psi-method)
- [ ] Section 4: Signal detection theory (d-prime, criterion, ROC curves)
- [ ] Section 5: Weber's law and JND (difference thresholds, scaling)
- [ ] Section 6: Perceptual scales (magnitude estimation, Stevens' power law)
- [ ] Section 7: Human studies protocols (IRB, informed consent, ethics)
- [ ] Section 8: **ARR-COC-0-1 human evaluation** (relevance perception thresholds)
- [ ] **CITE**: biological-vision/02 (eye tracking); biological-vision/00 (perception)

**Step 3: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-psychophysics-human-2025-11-14-[TIME].md

---

## PART 15: Eye Tracking & Gaze Analysis (~700 lines)

- [ ] PART 15: Create research-methodology/02-eye-tracking-gaze-analysis.md

**Step 0: Check Existing Knowledge**
- [ ] Read biological-vision/02-eye-tracking-task-attention.md (eye tracking)
- [ ] Read biological-vision/01-saccades-eye-movements.md (saccade patterns)

**Influenced by**: (Eye tracking knowledge) - Gaze-based attention measurement

**Step 1: Web Research**
- [ ] Search: "eye tracking methodology 2024"
- [ ] Search: "fixation detection algorithms"
- [ ] Search: "saccade detection velocity threshold"
- [ ] Search: "areas of interest AOI analysis"

**Step 2: Create Knowledge File**
- [ ] Section 1: Eye tracking hardware (video-based, SR Research, Tobii)
- [ ] Section 2: Calibration procedures (9-point, drift correction, validation)
- [ ] Section 3: Fixation detection (velocity threshold, dispersion threshold)
- [ ] Section 4: Saccade detection (algorithms, microsaccades, drift)
- [ ] Section 5: Areas of Interest (AOI) analysis (dwell time, first fixation)
- [ ] Section 6: Gaze metrics (fixation duration, saccade amplitude, scanpath)
- [ ] Section 7: Data quality (accuracy, precision, loss, artifacts)
- [ ] Section 8: **ARR-COC-0-1 gaze validation** (compare model attention to human gaze)
- [ ] **CITE**: biological-vision/02 (eye tracking); biological-vision/01 (saccades)

**Step 3: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-eye-tracking-gaze-2025-11-14-[TIME].md

---

## PART 16: Statistical Analysis & Hypothesis Testing (~700 lines)

- [✓] PART 16: Create experimental-design/03-benchmark-datasets-evaluation.md (Completed 2025-11-14 15:48)

**Step 0: Check Existing Knowledge**
- [✓] Read practical-implementation/56-vision-token-budget-ablations.md (ablation analysis - folder empty)

**Influenced by**: (Ablation analysis knowledge) - Statistical testing

**Step 1: Web Research**
- [✓] Search: "statistical hypothesis testing t-test ANOVA machine learning 2024"
- [✓] Search: "effect size Cohen's d eta-squared practical significance 2024"
- [✓] Search: "multiple comparisons correction Bonferroni FDR permutation tests 2024"
- [✓] Search: "benchmark dataset evaluation vision-language models VQA GQA 2024"

**Step 2: Create Knowledge File**
- [✓] Section 1: Experimental design fundamentals (control, randomization, replication, factorial designs)
- [✓] Section 2: Vision-language benchmark datasets (VQA v2, GQA, TextVQA, NaturalBench)
- [✓] Section 3: Statistical hypothesis testing (t-tests, ANOVA, non-parametric tests)
- [✓] Section 4: Effect sizes and practical significance (Cohen's d, eta-squared)
- [✓] Section 5: Multiple comparisons correction (Bonferroni, FDR, permutation tests)
- [✓] Section 6: Post-hoc tests for ANOVA (Tukey HSD, Bonferroni, Dunnett, Games-Howell)
- [✓] Section 7: Reporting statistical results (APA style, visualization best practices)
- [✓] Section 8: **ARR-COC-0-1 statistical validation** (2×3 factorial design, token budget ablation)
- [✓] Section 9: Advanced topics (Bayesian testing, equivalence testing, meta-analysis)
- [✓] **CITE**: 15+ web sources (statistical testing, effect sizes, VQA benchmarks)

**Step 3: Create KNOWLEDGE DROP**
- [✓] Create KNOWLEDGE-DROP-benchmark-evaluation-2025-11-14-15-48.md

---

# BATCH 5: Cognitive Architectures & Embodied AI (4 runners, ~2,800 lines)

## PART 17: Cognitive Architectures (~700 lines)

- [✓] PART 17: Create cognitive-architectures/00-soar-act-r-comparison.md (Completed 2025-11-14 16:45)

**Step 0: Check Existing Knowledge**
- [✓] Read john-vervaeke-oracle/ (four ways of knowing, relevance realization pipeline)

**Influenced by**: (Vervaeke architecture) - Cognitive system design

**Step 1: Web Research**
- [ ] Search: "cognitive architectures ACT-R SOAR 2024"
- [ ] Search: "CLARION cognitive architecture"
- [ ] Search: "cognitive systems integration"
- [ ] Search: "working memory production systems"

**Step 2: Create Knowledge File**
- [ ] Section 1: Cognitive architecture principles (modularity, integration, learning)
- [ ] Section 2: ACT-R (adaptive control of thought, production rules, chunks)
- [ ] Section 3: SOAR (state, operator, and result, universal subgoaling)
- [ ] Section 4: CLARION (hybrid explicit-implicit, skill learning)
- [ ] Section 5: Working memory models (capacity, maintenance, manipulation)
- [ ] Section 6: Memory systems (declarative, procedural, episodic, semantic)
- [ ] Section 7: Integration challenges (symbol grounding, knowledge representation)
- [ ] Section 8: **ARR-COC-0-1 as cognitive architecture** (knowing → balancing → attending → realizing)
- [ ] **CITE**: john-vervaeke-oracle/ (cognitive pipeline)

**Step 3: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-cognitive-architectures-2025-11-14-[TIME].md

---

## PART 18: Embodied Cognition & 4E Cognition (~700 lines)

- [✓] PART 18: Create embodied-ai/00-embodied-cognition-theory.md (Completed 2025-11-14 15:56)

**Step 0: Check Existing Knowledge**
- [ ] Read john-vervaeke-oracle/ (participatory knowing, agent-arena coupling)

**Influenced by**: (Vervaeke participatory) - Embodied and embedded cognition

**Step 1: Web Research**
- [ ] Search: "embodied cognition 4E framework 2024"
- [ ] Search: "enactivism sensorimotor contingencies"
- [ ] Search: "extended mind hypothesis Clark Chalmers"
- [ ] Search: "ecological psychology affordances Gibson"

**Step 2: Create Knowledge File**
- [ ] Section 1: 4E cognition (Embodied, Embedded, Enacted, Extended)
- [ ] Section 2: Embodied cognition (body shapes mind, sensorimotor grounding)
- [ ] Section 3: Embedded cognition (environment scaffolds cognition)
- [ ] Section 4: Enacted cognition (action-perception loops, sensorimotor contingencies)
- [ ] Section 5: Extended mind (cognitive artifacts, external memory, tool use)
- [ ] Section 6: Affordances (Gibson, action possibilities, direct perception)
- [ ] Section 7: Critique and integration (computationalism vs embodiment)
- [ ] Section 8: **ARR-COC-0-1 embodied relevance** (query-driven, situated, participatory)
- [ ] **CITE**: john-vervaeke-oracle/ (participatory knowing)

**Step 3: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-embodied-4e-cognition-2025-11-14-[TIME].md

---

## PART 19: Ecological Psychology & Affordances (~700 lines)

- [✓] PART 19: Create embodied-ai/01-sensorimotor-learning.md (Completed 2025-11-14 15:57)

**Step 0: Check Existing Knowledge**
- [ ] Read john-vervaeke-oracle/ (participatory knowing, agent-arena coupling)
- [ ] Read biological-vision/ (direct perception, gestalt)

**Influenced by**: (Participatory and biological vision) - Ecological perception

**Step 1: Web Research**
- [ ] Search: "ecological psychology James Gibson 2024"
- [ ] Search: "affordances perception action"
- [ ] Search: "direct perception ecological optics"
- [ ] Search: "invariants optic flow"

**Step 2: Create Knowledge File**
- [ ] Section 1: Ecological psychology foundations (Gibson, organism-environment)
- [ ] Section 2: Affordances (action possibilities, relational properties)
- [ ] Section 3: Direct perception (no representations, pickup of invariants)
- [ ] Section 4: Ecological optics (optic flow, texture gradients, occlusion)
- [ ] Section 5: Perception-action coupling (loops, real-time control)
- [ ] Section 6: Affordances in AI/robotics (learned affordances, tool use)
- [ ] Section 7: Critique (representationalism debate, hybrid approaches)
- [ ] Section 8: **ARR-COC-0-1 affordances** (relevance as afforded action, query-driven)
- [ ] **CITE**: john-vervaeke-oracle/ (participatory); biological-vision/ (perception)

**Step 3: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-ecological-psychology-2025-11-14-[TIME].md

---

## PART 20: Consciousness & Meta-Cognition (~700 lines)

- [✓] PART 20: Create cognitive-foundations/15-consciousness-metacognition.md (Completed 2025-01-14 15:57)

**Step 0: Check Existing Knowledge**
- [✓] Read john-vervaeke-oracle/ (perspectival knowing, salience landscapes)

**Influenced by**: (Vervaeke perspectival) - Subjective experience and awareness

**Step 1: Web Research**
- [ ] Search: "theories of consciousness 2024"
- [ ] Search: "global workspace theory Baars"
- [ ] Search: "integrated information theory IIT"
- [ ] Search: "metacognition monitoring control"

**Step 2: Create Knowledge File**
- [ ] Section 1: Consciousness theories (Global Workspace, IIT, Higher-Order)
- [ ] Section 2: Global Workspace Theory (broadcast, access consciousness)
- [ ] Section 3: Integrated Information Theory (phi, cause-effect structure)
- [ ] Section 4: Phenomenal vs access consciousness (qualia, reportability)
- [ ] Section 5: Metacognition (monitoring, control, confidence)
- [ ] Section 6: Self-awareness (introspection, theory of mind)
- [ ] Section 7: Neural correlates (NCC, neural synchrony, thalamo-cortical)
- [ ] Section 8: **ARR-COC-0-1 meta-awareness** (perspectival knowing, salience landscapes)
- [ ] **CITE**: john-vervaeke-oracle/ (perspectival knowing)

**Step 3: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-consciousness-metacognition-2025-11-14-[TIME].md

---

# BATCH 6a: Paper Writing & Publication - Part A (2 runners, ~1,400 lines)

## PART 21: Academic Paper Structure & Writing (~700 lines)
## PART 22: Figure Design & Data Visualization (~700 lines)

---

# BATCH 6b: Paper Writing & Publication - Part B (2 runners, ~1,400 lines)

## PART 23: Peer Review & Publication Process (~700 lines)
## PART 24: Research Ethics & Reproducibility (~700 lines)

---

# ORIGINAL BATCH 6 (Split into 6a and 6b above)

## PART 21: Academic Paper Structure & Writing (~700 lines)

- [ ] PART 21: Create research-methodology/04-academic-paper-writing.md

**Step 0: Check Existing Knowledge**
- [ ] Read karpathy/academic-research/ (paper structure, citations)

**Influenced by**: (Academic research knowledge) - Scientific writing

**Step 1: Web Research**
- [ ] Search: "academic paper writing structure 2024"
- [ ] Search: "IMRAD format research papers"
- [ ] Search: "scientific writing clarity conciseness"
- [ ] Search: "LaTeX academic templates"

**Step 2: Create Knowledge File**
- [ ] Section 1: Paper structure (IMRAD: Introduction, Methods, Results, Discussion)
- [ ] Section 2: Abstract writing (concise, self-contained, structured)
- [ ] Section 3: Introduction (motivation, gap, contribution, organization)
- [ ] Section 4: Related work (positioning, comparison, synthesis)
- [ ] Section 5: Methods (reproducibility, clarity, enough detail)
- [ ] Section 6: Results (figures, tables, statistical reporting)
- [ ] Section 7: Discussion (interpretation, limitations, future work)
- [ ] Section 8: **ARR-COC-0-1 paper** (cognitive architecture + VLM implementation)
- [ ] **CITE**: karpathy/academic-research/ (paper structure)

**Step 3: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-academic-writing-2025-11-14-[TIME].md

---

## PART 22: Figure Design & Data Visualization (~700 lines)

- [✓] PART 22: Create research-methodology/05-figure-design-visualization.md (Completed 2025-11-14 21:10)

**Step 0: Check Existing Knowledge**
- [ ] Read practical-implementation/55-vlm-inference-latency-benchmarks.md (benchmark visualization)

**Influenced by**: (Benchmarking visualization) - Scientific figures

**Step 1: Web Research**
- [ ] Search: "scientific figure design best practices 2024"
- [ ] Search: "data visualization Edward Tufte"
- [ ] Search: "matplotlib seaborn academic figures"
- [ ] Search: "colorblind-friendly palettes"

**Step 2: Create Knowledge File**
- [ ] Section 1: Figure design principles (clarity, simplicity, honesty)
- [ ] Section 2: Chart types (line, bar, scatter, heatmap, when to use)
- [ ] Section 3: Color palettes (colorblind-safe, perceptually uniform)
- [ ] Section 4: Typography and labels (legible, informative, consistent)
- [ ] Section 5: Error bars and uncertainty (standard error, confidence intervals)
- [ ] Section 6: Multi-panel figures (layout, consistency, sub-figure labels)
- [ ] Section 7: Tools (matplotlib, seaborn, ggplot2, TikZ, Inkscape)
- [ ] Section 8: **ARR-COC-0-1 figures** (attention maps, relevance allocation, ablations)
- [ ] **CITE**: practical-implementation/55 (benchmarking visualization)

**Step 3: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-figure-design-2025-11-14-[TIME].md

---

## PART 23: Peer Review & Publication Process (~700 lines)

- [✓] PART 23: Create research-methodology/06-peer-review-publication.md (Completed 2025-11-15 00:30)

**Step 0: Check Existing Knowledge**
- [ ] Read karpathy/academic-research/ (publication venues, review process)

**Influenced by**: (Academic research knowledge) - Publication workflow

**Step 1: Web Research**
- [ ] Search: "peer review process academic publishing 2024"
- [ ] Search: "responding to reviewer comments"
- [ ] Search: "conference vs journal publication ML"
- [ ] Search: "preprints arXiv submission"

**Step 2: Create Knowledge File**
- [ ] Section 1: Publication venues (conferences, journals, workshops)
- [ ] Section 2: Peer review process (single-blind, double-blind, open review)
- [ ] Section 3: Review criteria (novelty, rigor, clarity, significance)
- [ ] Section 4: Responding to reviewers (rebuttal, revision, point-by-point)
- [ ] Section 5: Preprints (arXiv, bioRxiv, open access)
- [ ] Section 6: Ethics (authorship, plagiarism, conflicts of interest)
- [ ] Section 7: Conference presentation (poster, talk, demo)
- [ ] Section 8: **ARR-COC-0-1 publication strategy** (NeurIPS, ICLR, CVPR targets)
- [ ] **CITE**: karpathy/academic-research/ (publication venues)

**Step 3: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-peer-review-publication-2025-11-14-[TIME].md

---

## PART 24: Research Ethics & Reproducibility (~700 lines)

- [✓] PART 24: Create research-methodology/07-ethics-reproducibility.md (Completed 2025-11-14 02:32)

**Step 0: Check Existing Knowledge**
- [ ] Read mlops-production/00-monitoring-cicd-cost-optimization.md (reproducibility)

**Influenced by**: (MLOps reproducibility) - Research integrity

**Step 1: Web Research**
- [ ] Search: "research ethics AI ML 2024"
- [ ] Search: "reproducibility crisis machine learning"
- [ ] Search: "ML Code Completeness Checklist"
- [ ] Search: "IRB human subjects research"

**Step 2: Create Knowledge File**
- [ ] Section 1: Research ethics fundamentals (integrity, honesty, respect)
- [ ] Section 2: Human subjects research (IRB, informed consent, privacy)
- [ ] Section 3: Data ethics (bias, fairness, representation, consent)
- [ ] Section 4: Reproducibility (code release, random seeds, hyperparameters)
- [ ] Section 5: ML Code Completeness (Papers with Code checklist)
- [ ] Section 6: Replication vs reproduction (exact vs conceptual)
- [ ] Section 7: Open science (data sharing, code repositories, preregistration)
- [ ] Section 8: **ARR-COC-0-1 reproducibility** (code release, checkpoints, documentation)
- [ ] **CITE**: mlops-production/00 (reproducibility practices)

**Step 3: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-ethics-reproducibility-2025-11-14-[TIME].md

---

## Summary

**Total**: 24 PARTs across 6 batches
**Execution**: Run 4 runners at a time, review between batches
**Expected**: ~16,800 lines total
**New folders**: cognitive-foundations/ (16 files), research-methodology/ (8 files)
**Focus**: WIDE theoretical + experimental foundations FOR ARR-COC-0-1

**16 Influential Files Explicitly Referenced**:
- Distributed: 00-deepspeed-zero, 01-deepspeed-pipeline, 02-megatron-lm, 03-fsdp-vs-deepspeed
- Inference: 00-tensorrt-fundamentals, 01-tensorrt-vlm, 02-triton-server, 03-torch-compile
- Orchestration: 00-kubernetes-gpu, 01-kubeflow-pipelines, 02-ray-distributed, 03-ml-workload-patterns
- Hardware: 00-amd-rocm, 01-apple-metal, 02-intel-oneapi, 03-tpu-programming

**ARR-COC-0-1 Integration Throughout**:
Every file has Section 8 connecting theory to arr-coc-0-1 implementation!

**Batch Schedule**:
1. ✅ Batch 1 (PARTs 1-4: Active Inference & Predictive Processing) → Review → Continue
2. ✅ Batch 2 (PARTs 5-8: Information Theory & Bayesian Methods) → Review → Continue
3. ✅ Batch 3 (PARTs 9-12: Decision Making & Resource Allocation) → Review → Continue
4. ✅ Batch 4 (PARTs 13-16: Experimental Design & Research Methodology) → Review → Continue
5. ✅ Batch 5 (PARTs 17-20: Cognitive Architectures & Embodied AI) → Review → Continue
6. ✅ Batch 6 (PARTs 21-24: Paper Writing & Publication) → COMPLETE!

**After each batch**: Oracle updates INDEX.md incrementally, commits progress, reviews quality before continuing to next batch.

**Perfect for ARR-COC-0-1 because:**
- Active inference = relevance realization (free energy minimization)
- Bayesian brain = uncertainty in relevance allocation
- Multi-armed bandits = dynamic token budget optimization
- Information theory = propositional knowing (entropy measures)
- Experimental design = rigorous evaluation of relevance allocation
- Eye tracking = validate model attention against human gaze
- Paper writing = publish ARR-COC-0-1 research!
