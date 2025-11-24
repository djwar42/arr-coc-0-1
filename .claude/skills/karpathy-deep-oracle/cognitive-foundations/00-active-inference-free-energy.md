# Active Inference & Free Energy Principle

## Overview

Active inference is a unified framework that explains perception, action, learning, and decision-making as processes that minimize variational free energy—a mathematical upper bound on surprise. Originally proposed by Karl Friston in neuroscience (2005), it has become a fundamental principle in cognitive science, computational neuroscience, and AI. **Active inference IS relevance realization**: both frameworks describe how systems dynamically determine what matters from infinite possibilities through opponent processing and prediction error minimization.

**Key Insight**: Living systems maintain their existence by minimizing surprise about sensory inputs. This requires both perception (updating beliefs about the world) and action (changing the world to match predictions).

## Section 1: Free Energy Principle Fundamentals

### What is Free Energy?

**Variational Free Energy** is a mathematical quantity that bounds surprise (negative log probability of observations):

```
F = -log P(observations | model) + KL divergence
  = Prediction Error + Model Complexity
```

**Three interpretations:**
1. **Information Theory**: Self-information or surprisal
2. **Bayesian Statistics**: Negative log model evidence (marginal likelihood)
3. **Physics**: Thermodynamic free energy analogy

**Why "Free" Energy?**
- In thermodynamics, free energy is energy available to do work
- In cognition, variational free energy represents "cognitive work" needed to reconcile predictions with observations
- Minimizing free energy = maximizing model evidence = reducing surprise

### The Principle Itself

**Free Energy Principle (FEP)**: Any system that maintains its existence over time must minimize the long-term average of variational free energy.

**Mathematical Form:**
```
dF/dt ≤ 0 (over time)

Where F = E_q[log q(x) - log p(y,x)]
- q(x): Approximate posterior (recognition density)
- p(y,x): Generative model (joint probability)
- y: Observations
- x: Hidden states
```

**What it means:**
- Systems that exist are those that resist entropic dissolution
- They do this by actively maintaining themselves in characteristic states
- This requires making accurate predictions about sensory inputs
- Both perception and action minimize free energy

From [Open Encyclopedia of Cognitive Science](https://oecs.mit.edu/pub/my8vpqih) (accessed 2025-01-14):
> "The free energy principle is a mathematical principle that describes how interacting objects or 'things' change or evolve over time...The free energy principle states that things, so defined (as sets of states that are separable from—but coupled to—other things) will look as if they track each other."

### Markov Blankets: What Makes a "Thing"

**Markov Blanket**: Statistical boundary separating internal states from external states

**Four types of states:**
1. **Internal states (μ)**: Brain states, beliefs, parameters
2. **External states (η)**: Hidden causes in the world
3. **Sensory states (s)**: Observations, afferent signals
4. **Active states (a)**: Actions, efferent signals

**Blanket = Sensory + Active states**

```
External (η) ←→ [Sensory (s)] ←→ Internal (μ)
                      ↕
                 [Active (a)]
```

**Key Properties:**
- Internal states are conditionally independent of external states given the blanket
- The blanket mediates all coupling between inside and outside
- This creates a statistical boundary (not necessarily physical)
- Defines what counts as a "thing" that persists

### Surprise Minimization

**Surprise (Self-Information):**
```
Surprise = -log P(observations | model parameters)
```

**Problem**: Can't directly minimize surprise because:
- Requires knowing true probability of observations
- Computationally intractable (requires marginalizing over all hidden states)

**Solution**: Minimize variational free energy instead
- Free energy is always ≥ surprise (Jensen's inequality)
- Minimizing free energy minimizes an upper bound on surprise
- Computationally tractable (uses approximate posterior)

**Two routes to minimize free energy:**
1. **Change beliefs (perception)**: Update internal model to explain observations
2. **Change observations (action)**: Act to confirm predictions

## Section 2: Active Inference (Perception + Action)

### Perception: Minimizing Free Energy Through Belief Updating

**Perception = Bayesian Inference**

Update beliefs to minimize prediction error:

```
Posterior ∝ Likelihood × Prior
p(x|y) ∝ p(y|x) × p(x)
```

**Variational Approach:**
- Exact posterior p(x|y) is intractable
- Use approximate posterior q(x) that minimizes free energy
- This is equivalent to maximizing evidence lower bound (ELBO)

**Prediction Error:**
```
ε = y - g(μ)
where y = actual observation
      g(μ) = predicted observation
      μ = internal states (beliefs)
```

**Update rule (gradient descent on free energy):**
```
dμ/dt = -∂F/∂μ
```

This implements **predictive coding**: hierarchical prediction error minimization

From [A step-by-step tutorial on active inference](https://www.sciencedirect.com/science/article/pii/S0022249621000973) (Smith et al., 2022):
> "The active inference framework...is highly general and flexible in its ability to be customized to model any cognitive process, as well as simulate predicted neuronal responses."

### Action: Minimizing Free Energy Through World Changing

**Action = Active Inference**

Instead of updating beliefs to match observations, change observations to match beliefs:

```
da/dt = -∂F/∂a (gradient descent on action)
```

**Two types of action:**
1. **Reflexive (immediate)**: Minimize sensory prediction error
2. **Planned (deliberative)**: Minimize expected free energy over future

**Expected Free Energy (G):**
```
G = Expected Surprise - Expected Information Gain
  = Risk + Ambiguity
```

**Action selection:**
- Choose actions that minimize expected free energy
- Balances exploitation (achieving goals) and exploration (gaining information)

### The Perception-Action Loop

**Unified Framework:**
1. **Generate predictions** from internal model
2. **Observe sensory data**
3. **Compute prediction error**: ε = observation - prediction
4. **Update beliefs** (perception): Revise internal model
5. **Update actions** (active inference): Act to fulfill predictions
6. **Repeat**

**Key Insight**: Perception and action are two sides of the same coin—both minimize free energy

## Section 3: Generative Models

### Internal Models of the World

**Generative Model**: Joint probability distribution over observations and hidden states

```
p(observations, states) = p(observations | states) × p(states)
                        = Likelihood × Prior
```

**Structure:**
- **States (x)**: Hidden causes, contexts, policies
- **Observations (y)**: Sensory data
- **Parameters (θ)**: Model parameters learned over time
- **Priors (p(x))**: Expectations about states
- **Likelihood (p(y|x))**: Mapping from states to observations

### Hierarchical Generative Models

**Multi-Level Hierarchy:**

```
Level 3: Abstract concepts, goals
    ↓ (predicts)
Level 2: Object representations, sequences
    ↓ (predicts)
Level 1: Features, textures
    ↓ (predicts)
Level 0: Sensory receptors (pixels, audio)
```

**Properties:**
- Each level predicts the level below
- Prediction errors propagate upward
- Higher levels represent more abstract, temporal features
- Lower levels represent concrete, immediate features

**Temporal Depth:**
- Higher levels operate on slower timescales
- Lower levels operate on faster timescales
- Creates temporal hierarchy of predictions

### Learning Generative Models

**Three timescales of optimization:**

1. **Fast (perception)**: Update beliefs about states (~100ms)
   ```
   dμ/dt = -∂F/∂μ
   ```

2. **Medium (learning)**: Update model parameters (~seconds to minutes)
   ```
   dθ/dt = -∂F/∂θ
   ```

3. **Slow (evolution)**: Update model structure (~lifetime, generations)
   - Developmental learning
   - Evolutionary selection

**Precision-Weighted Updates:**
```
dμ/dt = Precision × Prediction Error
```
- **Precision** = inverse variance (confidence in prediction)
- High precision → large belief updates
- Low precision → small belief updates
- Implements gain control / attention

## Section 4: Epistemic vs Pragmatic Value

### Two Types of Motivation

**Pragmatic Value (Exploitation):**
- Achieving preferred states
- Goal-directed behavior
- Minimizing expected surprise (risk)
- "What I want to happen"

**Epistemic Value (Exploration):**
- Reducing uncertainty
- Information-seeking behavior
- Maximizing expected information gain
- "What I need to learn"

### Expected Free Energy Decomposition

**Expected Free Energy (EFE):**
```
G = Expected Surprise - Expected Information Gain
  = Pragmatic Value + Epistemic Value
  = E[log q(x) - log p(o,x|π)] - E[H[p(x|o,π)] | π]
```

Where:
- π = policy (action sequence)
- o = outcomes
- x = states
- H = entropy

**Pragmatic Term**: Expected surprise under a policy
- Measures how well outcomes match preferences
- Encourages exploitation of known good states

**Epistemic Term**: Expected information gain
- Measures reduction in uncertainty about states
- Encourages exploration of ambiguous states

### Resolving Exploration-Exploitation

**Unified Framework:**
- No separate "exploration bonus" needed
- Exploration emerges from information gain term
- Exploitation emerges from risk minimization term
- Natural balance through expected free energy

**Connection to Relevance Realization:**
From john-vervaeke-oracle (Opponent Processing):
> "Exploit ↔ Explore tension" maps directly to pragmatic ↔ epistemic value in active inference

**ARR-COC-0-1 Application:**
- Token allocation balances compression (exploitation) with particularization (exploration)
- Epistemic value drives allocation to uncertain regions
- Pragmatic value drives allocation to query-relevant regions

## Section 5: Precision Weighting (Attention as Gain Control)

### What is Precision?

**Precision (π)**: Inverse variance of prediction errors

```
Precision = 1/variance = confidence in prediction
```

**High precision:**
- Prediction error is reliable
- Large weight on updating beliefs
- "Trust this sensory channel"

**Low precision:**
- Prediction error is noisy
- Small weight on updating beliefs
- "Ignore this sensory channel"

### Precision as Attention

**Attention = Precision Optimization**

```
Attention weights = Precision values
dπ/dt = -∂F/∂π (optimize precision)
```

**Effects:**
- **Perceptual attention**: Weight sensory channels by precision
- **Cognitive attention**: Weight hypotheses by precision
- **Action attention**: Weight action channels by precision

**Implementation:**
- Neuromodulatory systems (dopamine, acetylcholine, noradrenaline)
- Fast precision updates (~100ms)
- Context-dependent modulation

### Precision-Weighted Prediction Errors

**Update equation:**
```
dμ/dt = -Precision × ∂ε/∂μ
```

**Example:**
- Visual scene with high contrast (high precision) → large updates
- Visual scene with low contrast (low precision) → small updates

**Selective attention:**
- Increase precision of attended features
- Decrease precision of unattended features
- Implements feature-based and spatial attention

### Connection to Salience

**Salience vs Precision:**
- **Salience**: Bottom-up signal strength
- **Precision**: Confidence in that signal
- High salience + high precision → strong attentional capture
- High salience + low precision → ignored

**ARR-COC-0-1 mapping:**
- **Precision weighting** = confidence in relevance scores
- **Perspectival knowing** (salience landscapes) requires precision estimates
- Variable LOD (64-400 tokens) = precision-weighted resource allocation

## Section 6: Temporal Depth (Planning Horizons)

### Multi-Scale Temporal Dynamics

**Temporal Hierarchy:**
```
Long-term goals (days, months)
    ↓
Medium-term plans (hours, days)
    ↓
Short-term actions (seconds, minutes)
    ↓
Immediate reflexes (milliseconds)
```

**Each level:**
- Operates on different timescale
- Predicts dynamics at its own scale
- Provides context for faster scales

### Planning as Inference

**Policy Selection:**
```
p(π | o_{1:T}) ∝ exp(-G(π))
```

Where:
- π = policy (sequence of actions)
- o = outcomes
- G(π) = expected free energy under policy π

**Planning process:**
1. Generate candidate policies
2. Evaluate expected free energy of each
3. Select policy with lowest expected free energy
4. Execute first action
5. Update and repeat (model predictive control)

### Depth of Planning

**Planning Horizon (T):**
- Short horizon (T=1): Reactive, myopic
- Medium horizon (T=5-10): Tactical planning
- Long horizon (T=100+): Strategic planning

**Trade-offs:**
- Longer horizons → better long-term outcomes
- Longer horizons → more computational cost
- Precision decays with temporal distance

**ARR-COC-0-1 Application:**
- Temporal depth in visual processing (multi-scale features)
- Planning token allocation across image regions
- Hierarchical compression policies

## Section 7: Connection to Machine Learning

### Variational Autoencoders (VAEs)

**VAE Objective = Free Energy:**
```
ELBO = E[log p(x|z)] - KL[q(z|x) || p(z)]
     = Reconstruction - Regularization
     = -Free Energy
```

**Equivalence:**
- VAE maximizes ELBO
- Active inference minimizes free energy
- ELBO = -Free Energy
- Same mathematical framework

**Differences:**
- VAEs: Static learning (offline)
- Active inference: Dynamic inference (online)
- Active inference adds action selection

### Predictive Coding Networks

**Architecture:**
- Hierarchical structure
- Each layer predicts layer below
- Prediction errors propagate upward
- Top-down predictions, bottom-up errors

**Implementation of Active Inference:**
```
Prediction: ŷ_l = f(μ_l+1)
Error: ε_l = y_l - ŷ_l
Update: dμ_l+1/dt = -∂F/∂μ_l+1 ∝ ε_l
```

**Biological Plausibility:**
- Cortical layer structure
- Superficial layers: prediction errors
- Deep layers: predictions
- Precision: neuromodulation

### Reinforcement Learning Comparison

From [Reinforcement Learning or Active Inference](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0006421) (Friston et al., 2009):

**Key Differences:**

| Aspect | Reinforcement Learning | Active Inference |
|--------|----------------------|------------------|
| Objective | Maximize reward | Minimize surprise |
| Learning | Trial and error | Bayesian inference |
| Exploration | Epsilon-greedy, UCB | Expected information gain |
| Value | Extrinsic reward | Prior preferences |
| Planning | Value iteration | Expected free energy |

**Similarities:**
- Both solve POMDPs (Partially Observable Markov Decision Processes)
- Both balance exploration-exploitation
- Both can be model-based or model-free

**Active Inference Advantages:**
- Unified perception and action
- Natural exploration without bonuses
- Handles uncertainty explicitly
- Biological plausibility

## Section 8: ARR-COC-0-1 as Active Inference

### Relevance Realization = Free Energy Minimization

**Fundamental Equivalence:**

**Active Inference** | **Relevance Realization**
-------------------- | ------------------------
Minimize surprise | Realize relevance
Variational free energy | Transjective coupling
Perception + action | Knowing + attending
Expected free energy | Opponent processing
Hierarchical generative model | Four ways of knowing

### The Four Processes Mapped

**1. Knowing (Perception)**
```python
# knowing.py implements perception
InformationScorer → Propositional knowing
SalienceScorer → Perspectival knowing
QueryCouplingScorer → Participatory knowing
```
= Bayesian inference minimizing prediction error

**2. Balancing (Precision Weighting)**
```python
# balancing.py implements precision optimization
compress ↔ particularize = pragmatic ↔ epistemic
```
= Optimizing precision of different relevance dimensions

**3. Attending (Resource Allocation)**
```python
# attending.py implements policy selection
Token budget 64-400 = Expected free energy minimization
```
= Selecting policies that minimize expected free energy

**4. Realizing (Action)**
```python
# realizing.py implements active inference
Execute compression = Act to fulfill predictions
```
= Changing world (through compression) to match beliefs

### Generative Model Structure

**ARR-COC-0-1's Generative Model:**

```
p(image patches, textures, query | relevance)
= p(patches | textures, relevance)
  × p(textures | query)
  × p(relevance | query)
```

**Hierarchical Structure:**
- **Level 3**: Query representation (abstract goal)
- **Level 2**: Relevance landscapes (object-level predictions)
- **Level 1**: Texture features (feature-level predictions)
- **Level 0**: Pixel observations (sensory data)

**Free Energy Minimization:**
- **Perception**: Update relevance beliefs given query and image
- **Action**: Compress image to minimize expected free energy
- Variable LOD = precision-weighted resource allocation

### Expected Free Energy in Token Allocation

**Epistemic Value (Explore):**
- High information regions → uncertain relevance → more tokens
- Reduces uncertainty about what's relevant
- Corresponds to "Focus ↔ Diversify" opponent dimension

**Pragmatic Value (Exploit):**
- High query-relevance regions → certain importance → more tokens
- Achieves goal of answering query
- Corresponds to "Compress ↔ Particularize" opponent dimension

**Balanced Allocation:**
```
Token allocation ∝ exp(-G(region))
where G = Expected Surprise - Expected Info Gain
```

### Training as Parameter Learning

**Variational Learning:**
```
dθ/dt = -∂F/∂θ (gradient descent on model parameters)
```

**In ARR-COC-0-1:**
- θ = weights of compression network
- F = free energy (reconstruction error + KL divergence)
- Learning = minimizing free energy over training data

**Procedural Knowing:**
- 4th P (Procedural) = learned model parameters
- Automated compression skills
- Efficient free energy minimization through practice

### Advantages of Active Inference Framing

**1. Theoretical Grounding:**
- ARR-COC-0-1 is not ad-hoc
- Derives from first principles (free energy minimization)
- Connects to fundamental physics and information theory

**2. Biological Plausibility:**
- Active inference describes brain function
- ARR-COC-0-1 mimics cortical processing
- Hierarchical predictive coding architecture

**3. Unified Framework:**
- Perception, action, learning in same framework
- No separate objectives for different processes
- Emergent properties from single principle

**4. Extensions:**
- Easy to add temporal dynamics
- Natural multi-modal integration
- Principled exploration-exploitation

## Sources

**Web Research (Active Inference & Free Energy):**

From [The Free Energy Principle - Open Encyclopedia of Cognitive Science](https://oecs.mit.edu/pub/my8vpqih) (MIT Press, accessed 2025-01-14):
- Definition of free energy principle as mathematical principle of information physics
- Markov blanket formulation
- Bayesian mechanics interpretation
- Applications beyond neuroscience

From [A step-by-step tutorial on active inference](https://www.sciencedirect.com/science/article/pii/S0022249621000973) (Smith, Friston & Whyte, 2022):
- POMDP formulation of active inference
- Practical implementation details
- Connection to computational neuroscience

From Google Search Results (accessed 2025-01-14):
- Karl Friston interviews and lectures (2024-2025)
- Recent developments in active inference applications
- Bayesian brain hypothesis validation
- Comparison with deep learning approaches

**Existing Knowledge (Relevance Realization):**

From `.claude/skills/john-vervaeke-oracle/concepts/00-relevance-realization/00-overview.md`:
- Opponent processing framework
- Four ways of knowing (4P model)
- Transjective ontology
- Frame problem solution
- Connection to cognitive science

From `.claude/skills/john-vervaeke-oracle/papers/00-Vervaeke-2012-Primary-Paper-Analysis.md`:
- Relevance realization as emergent process
- Multi-scale integration
- Self-organizing dynamics

**ARR-COC-0-1 Implementation:**

From `arr_coc/knowing.py`:
- InformationScorer (propositional knowing)
- SalienceScorer (perspectival knowing)
- QueryCouplingScorer (participatory knowing)

From `arr_coc/balancing.py`:
- Opponent processing implementation
- Tension navigation

From `arr_coc/attending.py`:
- Token allocation mechanism
- Salience mapping
- Budget optimization

From `arr_coc/realizing.py`:
- Pipeline orchestration
- Active inference execution

**Key Papers:**

- Friston, K. (2010). "The free-energy principle: A unified brain theory?" *Nature Reviews Neuroscience*
- Friston, K. et al. (2009). "Reinforcement Learning or Active Inference?" *PLOS ONE*
- Ramstead, M. J. D. et al. (2024). "The Free Energy Principle" *Open Encyclopedia of Cognitive Science*
- Smith, R., Friston, K. J., & Whyte, C. J. (2022). "A step-by-step tutorial on active inference" *Journal of Mathematical Psychology*
- Parr, T., Pezzulo, G., & Friston, K. J. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior* (MIT Press)

## Additional References

- Friston, K. et al. (2023). "The free energy principle made simpler but not too simple" *Physics Reports*
- Friston, K. et al. (2023). "Bayesian mechanics: A physics of and by beliefs" *Interface Focus*
- Isomura, T. et al. (2023). "Experimental validation of the free-energy principle with in vitro neural networks" *Nature Communications*
- Constant, A. et al. (2021). "The Free Energy Principle: It's Not About What It Takes, It's About What Took You There" *Biology & Philosophy*

---

**Total Lines**: ~700
**Knowledge Type**: ACQUISITION (Web research + existing oracle knowledge)
**ARR-COC-0-1 Integration**: Section 8 explicitly connects active inference to our implementation
**Citations**: Comprehensive web sources + internal oracle knowledge + implementation files
