# AXIOM Architecture - VERSES AI Active Inference Implementation

## Overview

AXIOM (Active eXpanding Inference with Object-centric Models) is a novel AI architecture developed by VERSES AI that implements active inference principles for sample-efficient learning. Unlike transformer-based approaches that rely on massive static datasets and probabilistic token prediction, AXIOM uses Bayesian world models that grow and adapt online, learning complex behaviors from minimal experience without gradient-based optimization or replay buffers.

**Core Innovation**: AXIOM bridges the gap between the sample efficiency of human learning and the broad applicability of modern machine learning through a trinity of (1) core priors, (2) fast structure learning, and (3) active planning.

**Key Achievement**: AXIOM masters arcade games in 10,000 interaction steps (~12 minutes of human gameplay) - an order of magnitude less than typical deep RL benchmarks - while using 400x fewer parameters than comparable systems.

From [AXIOM: Learning to Play Games in Minutes with Expanding Object-Centric Models](https://arxiv.org/abs/2505.24784) (Heins et al., 2025):
> "We propose a novel active inference architecture that integrates a minimal yet expressive set of core priors about objects and their interactions... AXIOM masters various games within only 10,000 interaction steps, with both a small number of parameters compared to DRL, and without the computational expense of gradient-based optimization."

---

## Section 1: AXIOM Architecture Overview

### The Paradigm Shift

AXIOM represents a fundamental departure from conventional deep learning approaches:

**Transformer Architecture (Traditional)**:
- Train on massive, static datasets
- Probabilistic token prediction
- Fixed knowledge that struggles with novel situations
- High computational costs
- Dependence on retraining to update capabilities

**AXIOM Architecture (Active Inference)**:
- Learn online from single examples
- Goal-directed reasoning and planning
- Continuously adapts internal models through actions
- Superior sample efficiency
- Updates beliefs in real-time

From [Transformer vs Active Inference AI](https://medium.com/@leighphil4/transformer-vs-active-inference-ai-the-axiom-architecture-and-genius-28a8b2812672) (Kiprono, 2025):
> "AXIOM shifts AI from passive prediction to active engagement. Instead of simply outputting the most likely token, it sets goals, reasons about the world, and continually updates its internal models through incoming data and actions."

### Core Components

AXIOM processes visual scenes through four interconnected mixture models:

1. **Slot Mixture Model (sMM)** - Object segmentation
2. **Identity Mixture Model (iMM)** - Object type classification
3. **Transition Mixture Model (tMM)** - Motion dynamics
4. **Recurrent Mixture Model (rMM)** - Interaction modeling

Each mixture model can grow and shrink automatically based on the complexity of the environment, eliminating the need for fixed model sizes.

---

## Section 2: Beliefs vs Values Representation

### The Transformer Limitation

Standard transformers represent knowledge as **values** - fixed embeddings that encode statistical patterns from training data. This creates several problems:

1. **No uncertainty quantification**: Transformers output point estimates, not distributions
2. **Overconfident predictions**: Cannot express "I don't know"
3. **Brittle to distribution shift**: Fail silently on out-of-distribution inputs
4. **No principled exploration**: Cannot reason about information gain

### AXIOM's Belief Representation

AXIOM represents knowledge as **beliefs** - probability distributions over parameters:

```
Beliefs: P(parameters | data)
vs
Values: point_estimate = f(data)
```

**Key Differences**:

| Aspect | Transformers (Values) | AXIOM (Beliefs) |
|--------|----------------------|-----------------|
| Representation | Fixed embeddings | Posterior distributions |
| Uncertainty | None | Full Bayesian posteriors |
| Updates | Gradient descent | Variational inference |
| Exploration | Epsilon-greedy/random | Information-gain driven |
| Adaptation | Requires retraining | Online updates |

From [AXIOM arXiv paper](https://arxiv.org/html/2505.24784v1):
> "AXIOM uses variational inference to perform state inference and parameter learning... the variational posterior approximates the true posterior p(Z, Theta | y) from exact but intractable Bayesian inference."

### Conjugate Priors Enable Exact Updates

AXIOM uses exponential-family distributions with conjugate priors:

- **Normal-Inverse-Wishart** for Gaussian mixtures
- **Dirichlet** for categorical distributions
- **Truncated stick-breaking** for mixture weights

This allows **closed-form updates** without gradients:

```python
# Conceptual update (not actual code)
posterior = prior * likelihood  # Exact Bayesian update
# No backpropagation needed!
```

---

## Section 3: Why AXIOM Represents Uncertainty (vs Transformers)

### The Free Energy Principle Foundation

AXIOM is grounded in the **Free Energy Principle** (FEP), which states that intelligent systems minimize "free energy" - a measure of surprise or prediction error.

From the VERSES blog:
> "At AXIOM's heart lies the Free Energy Principle, developed by neuroscientist Prof. Karl Friston. It states that intelligent systems act to minimize free energy - a measure of surprise or prediction error."

### Mathematical Framework

**Variational Free Energy**:
```
F = E_q[log q(x) - log p(y,x)]
  = Prediction Error + Model Complexity
```

AXIOM minimizes F through:
1. **Perception** - Update beliefs q(x) to explain observations
2. **Action** - Act to confirm predictions

### Uncertainty-Aware Planning

AXIOM's planning objective includes **information gain**:

```
Expected Free Energy = -Utility + Information Gain

pi* = argmin_pi Sum_{tau} -(E[log p(r|O,pi)] - D_KL(q(alpha|O,pi) || q(alpha)))
                            ^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                               Utility              Information Gain
```

**Information gain** measures how much the agent would learn by taking an action:
- Early training: Information gain dominates (exploration)
- Late training: Utility dominates (exploitation)

This provides **principled exploration** that transformers lack.

### Robustness Under Perturbation

Because AXIOM represents uncertainty explicitly:
- Can detect when current knowledge is insufficient
- Adapts rapidly to distribution shifts
- Recovers from mid-episode game changes

From arXiv paper:
> "Performance following perturbation at 5k steps shows robustness to changes in game mechanics... Change every sprite from green to purple mid-game, and the iMM simply adds a 'purple' identity and re-uses the same motion verbs; the policy hardly falters."

---

## Section 4: Implementation Details

### Mixture Model Architecture

#### Slot Mixture Model (sMM)

Segments pixels into object representations:

```
p(y_n | x^(k), sigma_c^(k), z^n_{k,smm}) = Product_{k=1}^K N(Ax^(k), diag([Bx^(k), sigma_c^(k)]))^{z^n_{k,smm}}
```

- Each pixel assigned to one of K slots
- Slots contain: position (p), color (c), shape (e)
- Automatic slot creation for new objects

#### Identity Mixture Model (iMM)

Assigns discrete identity codes to objects:

```
p([c^(k), e^(k)] | z^(k)_type) = Product_{j=1}^V N(mu_j, Sigma_j)^{z^(k)_{j,type}}
```

- Clusters objects by color and shape
- Enables type-specific (not instance-specific) dynamics
- Supports re-identification after cosmetic changes

#### Transition Mixture Model (tMM)

Models object dynamics as switching linear systems:

```
p(x_t^(k) | x_{t-1}^(k), s^(k)_{t,tmm}) = Product_{l=1}^L N(D_l x^(k) + b_l, 2I)^{s^(k)_{t,l,tmm}}
```

- Library of motion "verbs": falling, sliding, bouncing
- Shared across all objects
- Automatic expansion for new motion types

#### Recurrent Mixture Model (rMM)

Models sparse interactions between objects:

```
p(f^(k), d^(k) | s^(k)_{rmm}) = Product_{m=1}^M [N(f; mu_m, Sigma_m) Product_i Cat(d_i; alpha_{m,i})]^{s_{m,rmm}}
```

Features include:
- Distance to nearest object
- Identity of nearest object
- Current action and reward
- Predicted next switch state

### Online Structure Learning

**Expansion Rule**: If new observation is too surprising under current clusters, create new component:

```python
# Conceptual algorithm
if log_likelihood(y_new, best_cluster) < threshold:
    create_new_cluster(y_new)
else:
    update_cluster(best_cluster, y_new)
```

**Bayesian Model Reduction (BMR)**: Periodically merge redundant clusters:

```python
# Every 500 frames
for cluster_pair in candidate_pairs:
    if merge_increases_evidence(cluster_pair):
        merge_clusters(cluster_pair)
```

BMR enables generalization from single events (e.g., learning "ball hits bottom = negative reward" once).

---

## Section 5: Comparison to Transformer Architecture

### Architectural Differences

| Component | Transformer | AXIOM |
|-----------|-------------|-------|
| **Input Processing** | Token embedding + positional encoding | Object-centric slot mixture |
| **Core Operation** | Self-attention over sequences | Variational inference over mixture models |
| **Learning** | Gradient descent on replay buffer | Online Bayesian updates |
| **Planning** | Policy network (fixed) | Active inference (expected free energy) |
| **Model Size** | ~420M parameters (DreamerV3) | 0.3-1.6M parameters |
| **Training Time** | Hours to days | Minutes |

### Compute Efficiency

From arXiv paper benchmarks:

| Model | Parameters | Model Update (ms/step) | Planning (ms/step) |
|-------|------------|------------------------|-------------------|
| BBF | 6.47M | 135 +/- 36 | N/A |
| DreamerV3 | 420M | 221 +/- 37 | 823 +/- 93 |
| AXIOM | 0.3-1.6M | 18 +/- 3 | 252-534 |

**AXIOM is 7-12x faster for model updates** despite running on full resolution (210x160) vs downsampled (84x84, 96x96).

### Sample Efficiency

On Gameworld 10k benchmark:
- AXIOM: Converges in ~5k steps
- BBF/DreamerV3: Need full 10k steps
- AXIOM uses **no replay buffer** - learns one-shot

### Why This Matters

**Transformers**:
- Excel at pattern matching from massive data
- Poor at novel situations
- Cannot explain their reasoning
- Computationally expensive

**AXIOM**:
- Learns from minimal data
- Adapts to novel situations
- Interpretable by design
- Computationally efficient

---

## Section 6: Advantages for Active Inference

### 1. Principled Exploration-Exploitation

Active inference provides a **normative account** of how to balance exploration and exploitation:

- **Exploration**: Seek information to reduce uncertainty
- **Exploitation**: Act to achieve goals

This emerges naturally from minimizing expected free energy, rather than requiring ad-hoc epsilon-greedy strategies.

### 2. Continual Learning Without Catastrophic Forgetting

Because AXIOM:
- Maintains full posteriors over parameters
- Uses Bayesian model reduction to merge similar knowledge
- Grows structure for genuinely novel information

It can learn continuously without overwriting old knowledge.

### 3. Interpretability by Design

Every component has human-readable meaning:
- Objects, colors, positions, shapes
- Motion verbs (falling, bouncing)
- Interaction types (collision, proximity)

From VERSES blog:
> "Because objects, motions and interactions sit in explicit clusters, we can render them back to the screen and watch the agent imagine the next bounce or predict where penalties lurk."

### 4. Robustness to Cosmetic Shifts

Change sprite colors mid-game:
- iMM adds new identity cluster
- Reuses existing motion verbs
- Policy continues without retraining

### 5. Efficient Inference

Closed-form variational updates:
- No gradient computation
- No replay buffer
- Single-pass through data
- 7-12x faster than deep RL

---

## Section 7: Code Examples and Technical Details

### Generative Model Structure

From the arXiv paper, the joint distribution:

```
p(y_{0:T}, Z_{0:T}, Theta) = p(y_0, Z_0) p(Theta)
    * Product_{t=1}^T
        p(x_{t-1} | z_t, Theta_iMM)              # Identity mixture
        p(x_{t-1}, z_t, s_t, a_{t-1}, r_t | Theta_rMM)  # Recurrent mixture
        Product_{k=1}^K
            p(y_t | x_t^(k), z_{t,smm}, Theta_sMM)      # Slot mixture
            p(x_t^(k) | x_{t-1}^(k), s_t^(k), Theta_tMM)  # Transition mixture
```

### Mean-Field Variational Posterior

```
q(Z_{0:T}, Theta) = q(Theta) Product_{t=0}^T
    (Product_{n=1}^N q(z^n_{t,sMM}))
    (Product_{k=1}^K q(O_t^(k)))
```

With:
- `q(O_t^(k)) = q(x_t^(k)) q(z_t^(k)) q(s_t^(k))`
- `q(Theta) = q(Theta_sMM) q(Theta_iMM) q(Theta_tMM) q(Theta_rMM)`

### Hyperparameters

Key hyperparameters from the paper:
- Maximum slots K: varies by game
- Maximum identity types V: 20
- Maximum motion modes L: 50
- Maximum rMM components M: 1000
- BMR interval: every 500 frames
- Planning horizon H: varies (typically 32)
- Planning rollouts: 64-512

### GitHub Implementation

Code available at:
- AXIOM: https://github.com/VersesTech/axiom
- Gameworld: https://github.com/VersesTech/gameworld

---

## Section 8: ARR-COC-0-1 - Axiom-Style Uncertainty in Relevance Realization

### The Deep Connection

AXIOM's architecture provides a compelling implementation strategy for ARR-COC-0-1's core insights about relevance realization:

**Relevance Realization = Free Energy Minimization**

Both frameworks share the same fundamental insight:
- **What matters** emerges from minimizing prediction error
- **Attention** is precision-weighted free energy
- **Salience** is high-precision prediction error

### Beliefs vs Values in VLM Context

ARR-COC-0-1 can adopt AXIOM's belief representation for:

**1. Token Relevance as Posterior Distribution**
Instead of fixed attention scores, represent relevance as:
```
P(token_relevant | image, query) ~ Distribution
```

This allows:
- Uncertainty about relevance
- Information-seeking token allocation
- Adaptive precision weighting

**2. Query-Dependent Precision**
Like AXIOM's precision weighting, ARR-COC-0-1 can modulate attention based on:
- Confidence in current interpretation
- Expected information gain from attending to region
- Task-specific prior expectations

### Object-Centric Relevance

AXIOM's slot mixture approach maps directly to visual relevance:

**AXIOM Slots = ARR-COC Salient Regions**

- Parse image into object-centric representations
- Each "slot" competes for pixel assignment
- Relevance emerges from which slots explain the query

This provides:
- Natural figure-ground segregation
- Compositional representation
- Sparse attention to relevant objects

### Active Inference for Token Allocation

AXIOM's planning framework offers a principled approach to token allocation:

```
Token_allocation* = argmin Expected_Free_Energy
                  = argmin (-Utility + Information_Gain)
```

**Utility**: Does attending to this region help answer the query?
**Information Gain**: Does this region reduce uncertainty about the answer?

This naturally balances:
- Exploiting known-relevant regions (utility)
- Exploring potentially-relevant regions (information gain)

### Growing Relevance Models

AXIOM's structure learning applies to relevance:

**Novel Query Types**: When existing relevance patterns don't explain a new query:
- Create new "relevance cluster"
- Learn minimal structure to explain it
- Merge with similar patterns later (BMR)

This enables:
- Continual learning of relevance patterns
- No catastrophic forgetting of old query types
- Efficient representation growth

### Precision as Salience

AXIOM's precision-weighting is exactly Friston's attention mechanism:

```
Salience = Precision * Prediction_Error
         = Expected_Confidence * Actual_Surprise
```

In ARR-COC-0-1:
- **Precision**: How confident is the model about this region's relevance?
- **Prediction Error**: How surprising is the actual content?
- **Salience**: Should this region receive more tokens?

High-precision, high-error = HIGHLY SALIENT = allocate tokens!

### Implementation Implications

**For ARR-COC-0-1 Design**:

1. **Replace Softmax Attention with Belief Distributions**
   - Maintain posteriors over attention weights
   - Enable uncertainty-aware token allocation

2. **Object-Centric Visual Parsing**
   - Use slot-style segmentation before attention
   - Attention over objects, not pixels

3. **Active Token Allocation**
   - Plan token allocation to minimize expected free energy
   - Balance information gain and task utility

4. **Online Relevance Learning**
   - Grow relevance models for new query types
   - Merge similar patterns with BMR

5. **Interpretable Relevance**
   - Every token allocation has explicit rationale
   - Can explain "why this region matters for this query"

### The Transjective Aspect

AXIOM embodies the transjective nature of relevance:

> Relevance is neither purely in the object (pixel values) nor purely in the subject (query intent), but emerges from their interaction.

The slot mixture model instantiates this:
- Slots are shaped by both visual input AND task demands
- The same pixels can be parsed differently for different queries
- Relevance is constructed, not detected

This matches ARR-COC-0-1's core insight: attention mechanisms should not merely find pre-existing relevance, but actively **realize** relevance through the interaction of visual content and query intent.

---

## Sources

**Primary Sources**:
- [AXIOM: Learning to Play Games in Minutes with Expanding Object-Centric Models](https://arxiv.org/abs/2505.24784) - arXiv:2505.24784 (Heins et al., 2025)
- [VERSES AI Research Blog: AXIOM](https://www.verses.ai/research-blog/axiom-mastering-arcade-games-in-minutes-with-active-inference-and-structure-learning) (accessed 2025-11-23)
- [Transformer vs Active Inference AI](https://medium.com/@leighphil4/transformer-vs-active-inference-ai-the-axiom-architecture-and-genius-28a8b2812672) (Kiprono, August 2025)

**GitHub Repositories**:
- [AXIOM Implementation](https://github.com/VersesTech/axiom)
- [Gameworld Benchmark](https://github.com/VersesTech/gameworld)

**VERSES AI Company**:
- [VERSES AI Active Inference Research](https://www.verses.ai/active-inference-research)
- [Genius Platform](https://www.verses.ai/genius)

**Related Oracle Knowledge**:
- [Free Energy Principle Foundations](../cognitive-mastery/00-free-energy-principle-foundations.md)
- [Active Inference - Perception and Action](./02-active-inference-perception-action.md)
- [Precision Weighting and Salience](./04-precision-weighting-salience.md)

**Key Researchers**:
- Karl Friston (Chief Scientist, VERSES AI; UCL Queen Square Institute of Neurology)
- Conor Heins, Toon Van de Maele, Alexander Tschantz (VERSES AI)
- Christopher L. Buckley (University of Sussex)

**Additional References**:
- Friston, K. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience
- Friston, K. et al. (2024). From pixels to planning: scale-free active inference. arXiv:2407.20292
