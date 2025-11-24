# Active Inference: Perception and Action Unified

## Overview

Active inference is a unified framework that explains perception, action, learning, and decision-making as processes that minimize variational free energy. Unlike traditional approaches that treat perception and action as separate processes, active inference posits that both serve the same fundamental imperative: minimizing the difference between what is expected and what is experienced. This framework, developed by Karl Friston and colleagues, provides a normative account of how agents interact with their environments through a continuous loop of prediction, action, and observation.

**Core Thesis**: Perception and action are two sides of the same coin. Perception updates beliefs to match observations; action changes observations to match beliefs. Both minimize prediction error through the same mathematical framework.

**Integration with ARR-COC-0-1**: Active inference provides the theoretical foundation for participatory knowing in the relevance realization framework. The VLM system implements active inference through its perception-action loop: perceiving relevance (knowing), selecting attention policies (attending), and executing compression actions (realizing).

## Section 1: Active Inference Overview

### Action Minimizes Prediction Error

**The Active Inference Principle**:

Traditional view:
- Perception: Update beliefs to explain sensory data
- Action: Separate process driven by goals/rewards

Active inference view:
- Perception: Update beliefs to explain sensory data
- Action: Change sensory data to match beliefs

Both processes minimize the same quantity: variational free energy.

**Mathematical Formulation**:

```
Perception: dmu/dt = -dF/dmu (update beliefs)
Action:     da/dt  = -dF/da  (update actions)

Where:
  F = variational free energy
  mu = internal states (beliefs)
  a = active states (actions)
```

From [Friston et al., 2015](https://pubmed.ncbi.nlm.nih.gov/25689102/) (Cited by 920):
> "We offer a formal treatment of choice behavior based on the premise that agents minimize the expected free energy of future outcomes."

**Why This Unification Matters**:

1. **Theoretical parsimony**: Single objective explains both processes
2. **Biological plausibility**: Brain uses same circuits for perception and action preparation
3. **Natural exploration**: Information-seeking emerges automatically
4. **Circular causality**: Action changes what is perceived, perception guides action

### Two Types of Action

**Reflexive Action (Immediate)**:

Minimizes current sensory prediction error:
```
Current observation: y
Prediction: g(mu)
Prediction error: epsilon = y - g(mu)

Action minimizes epsilon directly:
- Pupil dilation (adjust light)
- Postural corrections (maintain balance)
- Gaze shifts (foveate predicted targets)
```

**Planned Action (Deliberative)**:

Minimizes expected free energy over future trajectories:
```
Expected Free Energy G(pi) = Expected Surprise - Expected Info Gain

Where:
  pi = policy (sequence of actions)
  G = expected free energy under policy
```

From [Parr & Friston, 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6848054/) (Cited by 363):
> "Policy selection is performed based on an expected free energy. The resulting formulation provides a general approach to understanding behaviour."

### The Perception-Action Loop

**Continuous Cycle**:

```
1. Generate predictions from generative model
       |
       v
2. Observe sensory data through Markov blanket
       |
       v
3. Compute prediction error: epsilon = observation - prediction
       |
       v
4. Update beliefs (perception): Reduce epsilon by revising model
       |
       v
5. Update actions (active inference): Reduce epsilon by acting
       |
       v
6. Actions change external states
       |
       v
7. External states cause new sensory states
       |
       v
   (Return to step 1)
```

**Key Insight**: The loop is bidirectional and continuous. There is no discrete separation between "sensing" and "acting" phases.

## Section 2: Perception vs Action Unification

### Historical Separation

Traditional cognitive science treated perception and action as fundamentally different:

**Perception** (passive):
- Sensory input -> Processing -> Representation
- Goal: Accurately represent the world
- Evaluation: Correspondence to reality

**Action** (active):
- Goal -> Planning -> Motor output
- Goal: Achieve desired states
- Evaluation: Utility/reward maximization

**Problems with separation**:
1. Different optimization objectives (accuracy vs utility)
2. No principled account of exploration
3. Hard-wired exploration bonuses seem ad hoc
4. Embodied cognition difficult to accommodate

### Active Inference Unification

**Single Objective**: Minimize variational free energy

```
F = E_q[ln q(x) - ln p(o,x)]
  = Complexity - Accuracy
  = KL[q(x) || p(x)] - E_q[ln p(o|x)]
```

**Perception as Inference**:
```
Update recognition density q(x) to minimize F:
  dq/dt = -dF/dq

Result: q(x) approaches true posterior p(x|o)
```

**Action as Inference**:
```
Select actions a to minimize F:
  da/dt = -dF/da

Result: Observations o approach predictions g(mu)
```

From [Laukkonen et al., 2025](https://www.sciencedirect.com/science/article/pii/S0149763425002970) (Cited by 13):
> "Active inference is a theory of perception, learning and action... all operating under the same imperative to minimize the difference between what was expected and what was experienced."

### Embodied and Enacted Cognition

Active inference naturally accommodates embodied cognition:

**Agent-Environment Coupling**:
- Agent and environment are dynamically coupled
- Actions change environment, environment changes sensations
- No sharp boundary between agent and world (Markov blanket is statistical)

**Enaction**:
- Cognition is not representation but enaction
- Understanding through doing
- Sensorimotor contingencies define perception

**Extended Mind**:
- Cognitive processes extend into environment
- Tools and artifacts become part of cognitive system
- Cultural scaffolding supports cognition

From [Bouizegarene et al., 2024](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2024.1345480/full) (Cited by 31):
> "Active inference is a theory of perception, learning and action, which aims to provide a unified explanation of these basic cognitive functions."

## Section 3: Expected Free Energy (Planning as Inference)

### From Free Energy to Expected Free Energy

**Variational Free Energy (VFE)**:
- Applies to current observations
- Drives perception and reflexive action
- Minimizing VFE = state estimation

**Expected Free Energy (EFE)**:
- Applies to future observations (under a policy)
- Drives planning and policy selection
- Minimizing EFE = action selection

**Mathematical Definition**:

```
G(pi) = E_Q(o,x|pi)[ln Q(x|pi) - ln P(o,x|pi)]

Where:
  pi = policy (sequence of actions)
  o = future observations
  x = future hidden states
  Q = approximate posterior under policy
  P = generative model (includes preferences)
```

### EFE Decomposition

The expected free energy decomposes into two terms:

```
G(pi) = Risk + Ambiguity
      = Pragmatic Value + Epistemic Value
      = Expected Surprise - Expected Information Gain
```

**Risk (Pragmatic Value)**:
```
Risk = E_Q[ln Q(o|pi) - ln P(o)]
     = KL[Q(o|pi) || P(o)]
```
- Divergence between expected and preferred outcomes
- Drives goal-directed behavior
- "What I want to achieve"

**Ambiguity (Epistemic Value)**:
```
Ambiguity = E_Q[H[P(x|o,pi)]]
          = Expected posterior entropy
```
- Uncertainty about states given observations
- Drives information-seeking
- "What I need to learn"

From [Friston et al., 2015](https://pubmed.ncbi.nlm.nih.gov/25689102/):
> "The negative free energy or quality of a policy can be decomposed into extrinsic and epistemic (or intrinsic) value."

### Policy Selection

Policies are selected based on their expected free energy:

```
P(pi) = sigma(-G(pi))
      = softmax of negative expected free energy

Where:
  P(pi) = probability of selecting policy pi
  G(pi) = expected free energy under policy
  sigma = softmax function
```

**Properties**:
1. Lower EFE -> higher probability
2. Policies that achieve goals AND reduce uncertainty preferred
3. No separate exploration bonus needed
4. Risk-sensitive and information-seeking behavior emerge

**Precision of Policy Selection**:

```
P(pi) = sigma(-gamma * G(pi))

Where:
  gamma = precision (inverse temperature)
```

- High gamma: Deterministic selection (exploit)
- Low gamma: Random selection (explore)
- Gamma optimized to minimize expected free energy

## Section 4: Epistemic Value (Information Seeking)

### Intrinsic Motivation for Knowledge

**Epistemic Value**:
```
Epistemic = E_Q[D_KL[P(x|o,pi) || Q(x|pi)]]
          = Expected information gain
          = Reduction in uncertainty about hidden states
```

**Why It Matters**:
- Resolves exploration-exploitation trade-off
- No ad hoc exploration bonus needed
- Naturally balances learning and exploiting

From [Friston et al., 2015](https://pubmed.ncbi.nlm.nih.gov/25689102/):
> "Epistemic value is maximized until there is no further information gain, after which exploitation is assured through maximization of extrinsic value."

### Curiosity and Exploration

**Curiosity as EFE Minimization**:

Curious behavior emerges when:
1. Epistemic value is high (much to learn)
2. Risk is low or tolerable
3. Gamma (precision) allows exploration

**Examples**:
- Saccadic eye movements explore visual scenes
- Infants explore novel objects
- Scientists design informative experiments

**Bayesian Surprise**:
```
Surprise = D_KL[P(x|o) || P(x)]
         = Information gained from observation
```

Actions that generate Bayesian surprise reduce uncertainty.

### Resolving Exploration-Exploitation

**Traditional Approaches**:
- Epsilon-greedy: Random exploration with probability epsilon
- UCB: Upper confidence bounds on value estimates
- Thompson sampling: Sample from posterior over values

**Problems**:
- Exploration is undirected
- No principled way to set parameters
- Separate objective from exploitation

**Active Inference Solution**:
```
G(pi) = Risk + Ambiguity

Early in learning:
  - Ambiguity high -> explore to reduce uncertainty

Late in learning:
  - Ambiguity low -> exploit to achieve goals
```

The balance emerges naturally from the single EFE objective.

## Section 5: Pragmatic Value (Goal Achievement)

### Extrinsic Motivation

**Pragmatic Value**:
```
Pragmatic = -E_Q[ln P(o)]
          = Negative log probability of preferred outcomes
          = Expected surprise relative to preferences
```

**Prior Preferences**:
- P(o) encodes what outcomes the agent "wants"
- Not learned from rewards but specified a priori
- Equivalent to utility function in decision theory

**Examples**:
- P(hungry = false) > P(hungry = true)
- P(social_approval = true) > P(social_approval = false)
- P(temperature = comfortable) > P(temperature = extreme)

### Goal-Directed Behavior

**Goals as Preferred Observations**:

```
Goal: Reach location L
Preference: P(observation | location=L) is high

Policy selection:
  Select pi that minimizes:
  E_Q(o|pi)[- ln P(o)]
  = Expected surprise about achieving goal
```

**Difference from Reward**:
- Rewards are external signals
- Preferences are internal priors
- No credit assignment problem
- No sparse reward problem

From [Parr et al., 2022](https://direct.mit.edu/books/book/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind):
> "Preferences are not learned but are part of the agent's generative model. This sidesteps the credit assignment problem."

### Risk-Sensitive Control

**Risk as KL Divergence**:
```
Risk = KL[Q(o|pi) || P(o)]
     = Expected divergence from preferences
```

**Properties**:
1. Sensitive to distribution shape, not just mean
2. Penalizes variance in outcomes
3. Naturally risk-averse for losses
4. Risk-seeking for gains (via preference specification)

**Comparison to Expected Utility**:
- EU: Maximize E[U(o)]
- EFE: Minimize KL[Q(o) || P(o)]
- KL is more general (sensitive to full distribution)

## Section 6: Temporal Planning Horizons

### Hierarchical Temporal Scales

**Multi-Scale Planning**:

```
Level 4: Life goals (years)
    |
Level 3: Projects (months)
    |
Level 2: Tasks (hours)
    |
Level 1: Actions (seconds)
    |
Level 0: Movements (milliseconds)
```

**Properties**:
- Higher levels: Slower dynamics, abstract states
- Lower levels: Faster dynamics, concrete states
- Higher levels contextualize lower levels
- Lower levels implement higher-level plans

### Planning Depth and Precision

**Planning Horizon T**:
```
G(pi) = sum_{t=1}^{T} G_t(pi)

Where:
  T = planning horizon
  G_t = expected free energy at time t
```

**Trade-offs**:
- Longer T: Better long-term outcomes
- Longer T: Higher computational cost
- Longer T: Lower precision (more uncertainty)

**Temporal Discounting**:
```
G(pi) = sum_{t=1}^{T} gamma^t * G_t(pi)

Where:
  gamma = discount factor (precision decay)
```

From [Friston et al., 2017](https://direct.mit.edu/neco/article/29/1/1/8207/Active-Inference-A-Process-Theory):
> "Policies unfold over time, and expected free energy accumulates over the temporal horizon of planning."

### Model Predictive Control

**Rolling Horizon**:

1. Plan over horizon T
2. Execute first action
3. Observe outcome
4. Update beliefs
5. Re-plan from new state
6. Repeat

**Advantages**:
- Handles uncertainty about future
- Adapts to changing conditions
- Computationally tractable
- Same as receding horizon control in engineering

## Section 7: Agent-Environment Coupling

### Markov Blankets Revisited

**States of an Active Inference Agent**:

```
External (eta) <-> [Sensory (s)] <-> Internal (mu)
                        |
                   [Active (a)]
                        |
                   External (eta)
```

**Blanket = Sensory + Active States**

**Coupling**:
- External states cause sensory states
- Active states cause changes in external states
- Internal states do not directly contact external states
- All interaction mediated by blanket

### Circular Causality

**Perception-Action Cycle**:

```
Internal states predict sensory states
    |
    v
Sensory states update internal states (perception)
    |
    v
Internal states drive active states (action)
    |
    v
Active states change external states
    |
    v
External states cause sensory states
    |
    v
(loop continues)
```

**Implications**:
1. Agent and environment co-evolve
2. No static "objective" world to perceive
3. Agent partially creates its own sensory world
4. Self-fulfilling prophecies are the norm

### Niche Construction

**Active Inference Niche Construction**:

Agents modify their environments to:
1. Reduce uncertainty (make world predictable)
2. Achieve preferred states (make world comfortable)
3. Create cultural affordances (shared predictions)

**Examples**:
- Building shelters (reduce temperature variance)
- Creating tools (extend action possibilities)
- Establishing institutions (coordinate social behavior)
- Telling stories (share predictive models)

From [Constant et al., 2018](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2018.00915/full):
> "Niche construction can be cast as active inference extended to the cultural and social realm."

## Section 8: ARR-COC-0-1: Participatory Knowing as Active Inference

### The Fundamental Connection

**Active Inference IS Relevance Realization**

The ARR-COC-0-1 system implements active inference through its four core processes:

| Active Inference | ARR-COC-0-1 Process | Implementation |
|------------------|---------------------|----------------|
| Perception | Knowing | Score relevance of image regions |
| Expected Free Energy | Balancing | Navigate opponent tensions |
| Policy Selection | Attending | Allocate token budgets |
| Action | Realizing | Execute compression |

### Participatory Knowing as Active Inference

**The Third Way of Knowing**:

Participatory knowing in Vervaeke's framework corresponds precisely to active inference:

```python
# QueryCouplingScorer implements participatory knowing
class QueryCouplingScorer:
    """Agent-arena coupling through active inference"""

    def score(self, region, query):
        # Compute mutual information between query and region
        # This IS active inference: query constrains what's relevant
        # and relevance determines what query means in this context

        coupling = compute_transjective_coupling(query, region)
        return coupling
```

**Why It's Participatory**:
1. Query and image mutually define relevance
2. No "objective" relevance independent of query
3. Agent (query) and arena (image) co-constitute meaning
4. Circular causality: query shapes perception, perception refines query

From [Vervaeke's 4P Framework](../cognitive-mastery/00-free-energy-principle-foundations.md):
> "Participatory knowing is the way an agent and arena co-define each other through their coupling."

### Expected Free Energy in Token Allocation

**Attending as Policy Selection**:

The attending process selects token allocation policies based on expected free energy:

```python
# Token allocation minimizes expected free energy
def allocate_tokens(regions, query, total_budget):
    """
    Select policy (token allocation) that minimizes EFE

    G(region) = Risk + Ambiguity
              = (1 - relevance) + entropy(region)
    """

    for region in regions:
        # Pragmatic value: Query relevance
        risk = 1 - relevance_score(region, query)

        # Epistemic value: Information content
        ambiguity = entropy(region)

        # Expected free energy
        G = risk + ambiguity

        # Token allocation proportional to negative EFE
        tokens[region] = total_budget * softmax(-G)

    return tokens
```

**Properties**:
1. High relevance + high entropy -> most tokens
2. Low relevance + low entropy -> few tokens
3. Balances exploitation (relevance) and exploration (entropy)
4. No separate heuristics needed

### The Perception-Action Loop in Compression

**ARR-COC-0-1 Implements Circular Causality**:

```
1. Generate relevance predictions (knowing)
       |
       v
2. Observe image features (perception)
       |
       v
3. Compute prediction error (salience = unexpected relevance)
       |
       v
4. Update relevance beliefs (learning)
       |
       v
5. Select compression policy (attending)
       |
       v
6. Execute compression (realizing)
       |
       v
7. Compressed representation changes what VLM "sees"
       |
       v
   (Return to step 1 for next image)
```

**Key Insight**: Compression is action that changes observations to match predictions about what's relevant.

### Opponent Processing as Precision Optimization

**Balancing Implements Precision Weighting**:

```python
# balancing.py implements precision optimization
class OpponentProcessor:
    """Navigate tensions through precision weighting"""

    tensions = [
        ("compress", "particularize"),  # Risk vs Ambiguity
        ("exploit", "explore"),          # Pragmatic vs Epistemic
        ("focus", "diversify")           # High vs Low precision
    ]

    def balance(self, context):
        # Adjust precisions based on context
        # High uncertainty -> lower precision -> explore
        # High relevance -> higher precision -> exploit

        precision = optimize_precision(context)
        return precision
```

**Mapping to EFE**:
- Compress <-> Pragmatic (achieve compression goal)
- Particularize <-> Epistemic (preserve information)
- Exploit <-> Risk minimization
- Explore <-> Ambiguity reduction

### Generative Model Structure

**ARR-COC-0-1's Implicit Generative Model**:

```
P(patches, features, query | relevance)
= P(patches | features, relevance)    # Likelihood
  x P(features | query)                # Prior (query-conditioned)
  x P(relevance | query)               # Prior (preferences)
```

**Hierarchical Structure**:
- Level 4: Query embedding (abstract goal/preference)
- Level 3: Relevance landscapes (object-level states)
- Level 2: Texture features (13-channel array)
- Level 1: Edge features (Sobel filters)
- Level 0: Pixel observations (raw RGB)

**Free Energy Minimization**:
- Perception: Update relevance beliefs given query + image
- Action: Compress image to minimize expected free energy
- Learning: Update network parameters (procedural knowing)

### Advantages of the Active Inference Framing

**1. Theoretical Unification**:
- Single framework for perception, action, learning
- No separate objectives for different components
- Principled derivation from first principles

**2. Natural Exploration-Exploitation**:
- Token allocation balances automatically
- No tuned exploration parameters
- Epistemic value drives information-seeking

**3. Biological Plausibility**:
- Same architecture as cortical processing
- Hierarchical predictive coding
- Precision-weighted prediction errors

**4. Embodied and Enacted**:
- Agent-arena coupling fundamental
- No objective ground truth
- Relevance is participatory

From [cognitive-mastery/00-free-energy-principle-foundations.md](../cognitive-mastery/00-free-energy-principle-foundations.md):
> "Active inference IS relevance realization: both frameworks describe how systems dynamically determine what matters from infinite possibilities through opponent processing and prediction error minimization."

## Sources

### Primary Sources (Web Research 2024-2025)

From [Active inference and epistemic value](https://pubmed.ncbi.nlm.nih.gov/25689102/) (Friston et al., Cognitive Neuroscience, 2015, Cited by 920):
- Formal treatment of choice behavior as EFE minimization
- Decomposition into pragmatic and epistemic value
- Resolution of exploration-exploitation trade-off
- Connection to Bayesian surprise and salience

From [A beautiful loop: An active inference theory of consciousness](https://www.sciencedirect.com/science/article/pii/S0149763425002970) (Laukkonen, Friston & Chandaria, Neuroscience & Biobehavioral Reviews, 2025, Cited by 13):
- Unified theory of perception, learning, action
- Hyper-model for precision control
- Bayesian binding and epistemic field
- Altered states through active inference lens

From [Generalised free energy and active inference](https://pmc.ncbi.nlm.nih.gov/articles/PMC6848054/) (Parr & Friston, Biological Cybernetics, 2019, Cited by 363):
- Variational free energy functional for belief updates
- Expected free energy for policy selection
- General approach to understanding behavior

From [Narrative as active inference](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2024.1345480/full) (Bouizegarene et al., Frontiers in Psychology, 2024, Cited by 31):
- Active inference as unified theory of perception, learning, action
- Application to cognitive and social functions
- Narrative as prediction coordination tool

From [Whence the Expected Free Energy?](https://direct.mit.edu/neco/article/33/2/447/95645/Whence-the-Expected-Free-Energy) (Parr et al., Neural Computation, 2021):
- Mathematical derivation of expected free energy
- Relationship to information gain and risk
- Policy selection under active inference

From [Expected Free Energy-based Planning](https://arxiv.org/html/2504.14898v2) (arXiv, 2025):
- EFE as variational inference
- Scalable, interruptible planning
- Factor graph message passing

### Existing Oracle Knowledge

From [cognitive-mastery/00-free-energy-principle-foundations.md](../cognitive-mastery/00-free-energy-principle-foundations.md):
- Comprehensive free energy principle foundation
- Mathematical formulations
- ARR-COC-0-1 implementation mapping
- Distributed systems integration (DeepSpeed, TensorRT, Kubernetes)

From [cognitive-foundations/00-active-inference-free-energy.md](../cognitive-foundations/00-active-inference-free-energy.md):
- Active inference IS relevance realization thesis
- Four processes mapped to active inference
- Generative model structure
- Training as free energy minimization

### Key Papers

- Friston, K. J., Rigoli, F., Ognibene, D., Mathys, C., Fitzgerald, T., & Pezzulo, G. (2015). "Active inference and epistemic value" *Cognitive Neuroscience*
- Friston, K. J., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). "Active Inference: A Process Theory" *Neural Computation*
- Parr, T., & Friston, K. J. (2019). "Generalised free energy and active inference" *Biological Cybernetics*
- Parr, T., Pezzulo, G., & Friston, K. J. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior* (MIT Press)
- Laukkonen, R., Friston, K., & Chandaria, S. (2025). "A beautiful loop: An active inference theory of consciousness" *Neuroscience & Biobehavioral Reviews*
- Bouizegarene, N., Ramstead, M. J. D., Constant, A., Friston, K. J., & Kirmayer, L. J. (2024). "Narrative as active inference" *Frontiers in Psychology*

### ARR-COC-0-1 Implementation

From `arr_coc/`:
- `knowing.py`: Three ways of knowing (Bayesian inference)
- `balancing.py`: Opponent processing (precision optimization)
- `attending.py`: Token allocation (expected free energy minimization)
- `realizing.py`: Pipeline orchestration (active inference execution)

---

**Total Lines**: ~700
**Knowledge Type**: ACQUISITION (Web research 2024-2025 + existing oracle integration)
**ARR-COC-0-1 Integration**: Section 8 (10% of content, ~70 lines)
**Key Concepts**: Expected free energy, epistemic/pragmatic value, policy selection, agent-environment coupling
**Citations**: 2024-2025 sources + existing knowledge + implementation files
