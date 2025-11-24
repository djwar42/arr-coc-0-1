# Free Energy Principle: Deep Foundations

## Overview

The Free Energy Principle (FEP) is a unifying mathematical principle in theoretical neuroscience and physics that describes how self-organizing systems maintain their existence over time. Originally proposed by Karl Friston in the mid-2000s, FEP provides a normative framework for understanding perception, action, learning, and decision-making under a single imperative: minimize variational free energy.

**Core Thesis**: Any system that maintains its structural and functional integrity over time must actively minimize variational free energy - a mathematical upper bound on surprise (self-information). This applies universally from quantum particles to cells to brains to societies.

**Fundamental Insight**: Things that exist do so by virtue of minimizing surprise about their sensory observations. In other words, living systems act to confirm their own existence by behaving in unsurprising (characteristic) ways given the kind of thing they are.

From [MIT Open Encyclopedia of Cognitive Science](https://oecs.mit.edu/pub/my8vpqih) (Ramstead, 2024):
> "The free energy principle is a mathematical principle that describes how interacting objects or 'things' (defined in a specific way) change or evolve over time... things defined (as sets of states separable from - but coupled to - other things) will look as if they track each other."

---

## Section 1: Free Energy Principle Overview (Surprise Minimization)

### What Does It Mean to Be Alive?

The FEP answers the fundamental question: What enables living organisms to stay alive?

From [A Gentle Introduction to the Free Energy Principle](https://awjuliani.medium.com/a-gentle-introduction-to-the-free-energy-principle-03f219853177) (Juliani, 2023):
> "The FEP states that living organisms are dynamical systems that are separated from their environment by an interface... According to the FEP, all agents act, as best as they can, to minimize the errors in their guesses about the environment."

**Key Insight**: Organisms never have unmediated access to the environment. They must always infer (guess) the true state of the world based on limited sensory information. Free energy measures how wrong these guesses are.

### Surprise and Self-Information

**Surprise (Surprisal)** is defined mathematically as the negative log probability of an observation:

```
Surprise = -log P(observations | model)
         = Self-information
```

**Why Minimize Surprise?**

- High surprise = improbable state = organism in danger
- Low surprise = expected state = organism maintaining homeostasis
- Organisms that minimize surprise stay in characteristic states
- Characteristic states = states compatible with continued existence

**Example - The Tree in Wind**:

From Juliani (2023):
> "We find that certain species of trees which live in windy regions are capable of growing in the direction of the wind. Here we have all the components of the FEP at work. The tree has a desired homeostasis... This expectation is violated when the wind blows against the tree... In response, the tree 'acts' in order to minimize the free energy by slowly growing in the direction of the wind."

The tree cannot directly minimize surprise (it cannot know the probability distribution of all possible wind patterns). Instead, it minimizes free energy - an upper bound on surprise that is tractable to compute.

### The Problem with Direct Surprise Minimization

**Why can't we directly minimize surprise?**

1. **Intractability**: Computing P(observations) requires integrating over ALL possible hidden states
2. **Unknown true distribution**: We don't have access to the true probability distribution
3. **High-dimensional state spaces**: The integral is computationally infeasible

```
P(observations) = integral over all hidden states of P(observations, states)
                = Requires knowing everything about the world!
```

**Solution**: Minimize an upper bound (free energy) instead.

---

## Section 2: Mathematical Formulation - F = E_q[ln p(x,theta)] - H[q(theta)]

### Variational Free Energy Definition

**Core Equation**:

```
F = E_q[log q(theta) - log p(observations, theta)]
  = E_q[log q(theta)] - E_q[log p(observations, theta)]
  = -H[q(theta)] - E_q[log p(observations, theta)]
```

Where:
- F = Variational free energy
- q(theta) = Approximate posterior (recognition density) - our beliefs
- p(observations, theta) = Generative model (joint probability)
- H[q] = Entropy of approximate posterior
- E_q[...] = Expectation under q

### Alternative Decompositions

**Decomposition 1: Complexity vs Accuracy**

```
F = KL[q(theta) || p(theta)] - E_q[log p(observations | theta)]
  = Complexity - Accuracy
```

- **Complexity**: KL divergence between posterior and prior (how much beliefs deviate from prior)
- **Accuracy**: Expected log likelihood (how well model explains observations)

Minimizing F = maximizing accuracy while minimizing complexity (Occam's razor built-in!)

**Decomposition 2: Energy vs Entropy**

```
F = E_q[-log p(observations, theta)] - H[q(theta)]
  = Energy - Entropy
```

This connects to statistical physics:
- **Energy**: Expected negative log probability (how improbable is this configuration?)
- **Entropy**: Disorder/uncertainty in beliefs

**Decomposition 3: Bound on Surprise**

```
F = -log p(observations) + KL[q(theta) || p(theta | observations)]
  = Surprise + KL divergence
```

Since KL divergence >= 0:
```
F >= -log p(observations) = Surprise
```

**Therefore: F is an upper bound on surprise!**

Minimizing F minimizes this bound, indirectly minimizing surprise.

### Why "Free" Energy?

The name comes from thermodynamics (Helmholtz free energy):

**Thermodynamic Free Energy**:
```
F_thermo = U - TS
         = Internal energy - Temperature * Entropy
         = Energy available to do useful work
```

**Variational Free Energy**:
```
F_variational = Energy - Entropy
              = "Cognitive work" to reconcile predictions with observations
```

In both cases, free energy represents the energy available for useful work - whether physical or cognitive.

---

## Section 3: Variational Inference and ELBO

### The Variational Inference Problem

**Goal**: Compute the posterior p(theta | observations)

**Problem**: Exact Bayesian inference is intractable:
```
p(theta | observations) = p(observations | theta) * p(theta) / p(observations)
                        = p(observations | theta) * p(theta) / integral[p(observations | theta) * p(theta)]
```

The denominator (marginal likelihood / model evidence) is intractable for complex models.

**Solution**: Approximate the posterior with a tractable distribution q(theta) and minimize KL divergence:
```
q*(theta) = argmin KL[q(theta) || p(theta | observations)]
```

### ELBO (Evidence Lower Bound)

**Key Insight**: Minimizing free energy = Maximizing ELBO

```
log p(observations) = F + KL[q(theta) || p(theta | observations)]
                    >= F  (since KL >= 0)

Therefore:
ELBO = -F = E_q[log p(observations, theta)] - E_q[log q(theta)]
     = E_q[log p(observations | theta)] - KL[q(theta) || p(theta)]
```

**ELBO = negative Free Energy = lower bound on log model evidence**

### Connection to Machine Learning

From [arXiv:2410.19315](https://arxiv.org/html/2410.19315v2) (2024):
> "ELBO is exactly equal to negative free energy (ELBO = -F)"

In variational autoencoders (VAEs) and other deep generative models:
- Maximize ELBO = Minimize Free Energy
- Same mathematical objective!

**VAE Loss**:
```
L_VAE = -ELBO = -E_q[log p(x|z)] + KL[q(z|x) || p(z)]
      = Reconstruction loss + KL regularization
      = Free energy!
```

This connects FEP directly to modern deep learning - VAEs are implementing free energy minimization!

### Derivation of Free Energy Bound

**Starting Point**: Jensen's Inequality

For any convex function f and random variable X:
```
f(E[X]) <= E[f(X)]
```

For log (concave function), inequality flips:
```
log(E[X]) >= E[log X]
```

**Derivation**:

```
log p(y) = log integral[p(y,x) dx]
         = log integral[q(x) * p(y,x)/q(x) dx]
         = log E_q[p(y,x)/q(x)]
         >= E_q[log p(y,x)/q(x)]        (Jensen's inequality)
         = E_q[log p(y,x)] - E_q[log q(x)]
         = -F
```

Therefore: `log p(y) >= -F`, which means `F >= -log p(y) = Surprise`.

---

## Section 4: Markov Blankets as Statistical Boundaries

### Definition of a Markov Blanket

A **Markov blanket** is a statistical boundary that separates a system's internal states from external states, mediating ALL interactions between them.

From [The Markov Blankets of Life](https://royalsocietypublishing.org/doi/10.1098/rsif.2017.0792) (Kirchhoff et al., 2018):
> "A Markov blanket defines the boundaries of a system in a statistical sense... Internal states are conditionally independent of external states given the blanket."

### Four Types of States

**1. Internal States (mu)**
- Hidden inside the Markov blanket
- Include brain states, beliefs, model parameters
- Never directly interact with external states

**2. External States (eta)**
- Hidden causes in the world
- Everything outside the blanket
- Never directly observed by internal states

**3. Sensory States (s)**
- Part of the blanket
- Receive information from external states
- Send information to internal states
- "Afferent" signals (incoming)

**4. Active States (a)**
- Part of the blanket
- Receive commands from internal states
- Change external states through action
- "Efferent" signals (outgoing)

### Markov Blanket Architecture

```
                 External States (eta)
                        |
                        v
                 [Sensory States (s)]
                        |
                        v
                 Internal States (mu)
                        |
                        v
                 [Active States (a)]
                        |
                        v
                 External States (eta)
```

**Blanket = Sensory + Active states**

The blanket mediates ALL coupling between internal and external - there is no direct connection!

### Conditional Independence Property

**Mathematical Definition**:

```
p(mu | s, a, eta) = p(mu | s, a)
```

Given the blanket states (s, a), internal states mu are conditionally independent of external states eta.

**In English**: If you know everything about the blanket states, knowing external states gives you NO additional information about internal states.

### Why Markov Blankets Define "Things"

**What makes something a "thing"?**

A thing exists if and only if:
1. It has internal states
2. These internal states are separated from other states by a Markov blanket
3. The blanket mediates all interactions with external states

From Ramstead (2024):
> "The free energy principle provides researchers with a stipulative definition of thingness, in terms of dynamical or causal coupling among states."

**Examples**:
- **Cell**: Cell membrane is the Markov blanket
- **Brain**: Sensory organs and motor outputs form the blanket
- **Organism**: Skin, sensory systems, and motor systems form the blanket
- **Society**: Communication channels form the blanket

### Hierarchical (Nested) Markov Blankets

Markov blankets exist at multiple scales, nested within each other:

```
Quantum states <- Molecules <- Cells <- Organs <- Organisms <- Societies
```

Each level has its own Markov blanket:
- **Neurons**: Cell membrane separates internal from external
- **Brain regions**: Functional connectivity defines boundaries
- **Organisms**: Skin, sensory organs, motor systems
- **Social groups**: Communication channels, cultural boundaries

**Scale Invariance**: The FEP applies at EVERY scale!

---

## Section 5: Self-Organization Through Free Energy Minimization

### Self-Evidencing Systems

**Core Concept**: Systems with Markov blankets "self-evidence" - they maximize evidence for their own existence.

From [Bayesian Brain Computing Interview](https://academic.oup.com/nsr/article/11/5/nwae025/7571549) (Lu & Friston, 2024):
> "Self-evidencing refers to the imperative - for perception, cognition and action - to maximize (i.e. gather) evidence for the brain's generative (a.k.a. world) model of the sensed world."

**Mathematical Equivalence**:
```
Minimize Free Energy = Maximize Model Evidence = Maximize p(observations | model)
```

By minimizing free energy, systems gather evidence that they are the kind of system they are - they "prove" their own existence through their characteristic behavior.

### Two Routes to Minimize Free Energy

**Route 1: Change Beliefs (Perception)**

Update internal model q(theta) to better explain observations:
```
dq/dt = -partial_F / partial_q
```

This is gradient descent on free energy with respect to beliefs.

**Example**: Seeing a shape in the dark
- Initial belief: "It's a person"
- Get closer, more visual information
- Update belief: "It's actually a shrub"
- Free energy decreases as belief better matches observations

**Route 2: Change Observations (Action)**

Act on the world to make observations match predictions:
```
da/dt = -partial_F / partial_a
```

This is gradient descent on free energy with respect to actions.

**Example**: Setting and achieving goals

From Juliani (2023):
> "When we set a goal for ourselves, we are making a special kind of prediction about the world. This prediction is one that we know goes against the current sensory evidence... what we do instead is to set out to act in the world in order to ensure that the sensory evidence corresponds to our prediction."

### The Perception-Action Loop

```
1. Generate predictions from generative model
2. Observe sensory data through Markov blanket
3. Compute prediction error: epsilon = observation - prediction
4. Route A (Perception): Update beliefs to reduce epsilon
5. Route B (Action): Act to make observations match predictions
6. Repeat continuously
```

**Key Insight**: Perception and action are TWO SIDES OF THE SAME COIN - both minimize free energy!

### Exploration vs Exploitation

**Paradox**: If organisms minimize surprise, why do they explore novel environments?

**Resolution**: Long-term free energy minimization

From Juliani (2023):
> "Anyone who has lived with a cat can attest to both their unique spatial curiosity as well as their general tendency to be easily frightened. This apparent paradox is resolved if we understand their behavior through the lens of free energy minimization. They do not simply oscillate between curiosity and fear, the former is rather in service of preventing the latter!"

**Expected Free Energy** accounts for this:
```
G = Expected Surprise - Expected Information Gain
  = Risk + Ambiguity
  = Pragmatic Value + Epistemic Value
```

- **Pragmatic value**: Achieve goals (exploitation)
- **Epistemic value**: Reduce uncertainty (exploration)

Exploration minimizes FUTURE free energy by reducing uncertainty about the environment.

### Homeostasis as Prediction

Organisms maintain homeostasis by treating their preferred states as predictions:

- **Body temperature** = prediction about future observations
- **Deviation** = prediction error
- **Action** = minimize prediction error by thermoregulating

From Juliani (2023):
> "All living agents have some sense of what that homeostasis for them should be, even if it is only represented implicitly within the structure of the agent itself."

---

## Section 6: Connection to Thermodynamics

### Statistical Physics Origins

The name "free energy" comes from thermodynamics - specifically Helmholtz free energy:

**Helmholtz Free Energy**:
```
F_Helmholtz = U - TS
            = Internal Energy - Temperature * Entropy
```

At thermal equilibrium, systems minimize Helmholtz free energy.

### Principle of Least Action

Classical mechanics: Systems follow paths of least action
```
S = integral[L dt]   (L = Lagrangian)
```

**FEP extends this to statistical systems**:

From Ramstead (2024):
> "The free energy principle says that the motion of things in the space of possible states (or paths) minimize surprisal - in the same way that classical systems follow paths of least action that minimize an energy function (called a Lagrangian)."

### Bayesian Mechanics

The FEP gives rise to a new mechanics - **Bayesian mechanics**:

From Ramstead (2024):
> "This leads to a new family of mechanics, akin to classical and quantum mechanics, called Bayesian mechanics. In Bayesian mechanics, the time evolution of things minimizes a quantity called variational free energy."

**Classical Mechanics**: F = ma (Newton's law)
**Quantum Mechanics**: i*hbar * d_psi/dt = H*psi (Schrodinger equation)
**Bayesian Mechanics**: dF/dt <= 0 (Free energy principle)

### Maximum Entropy Connection

The FEP is closely related to the principle of maximum entropy:

From Ramstead (2024):
> "The free energy principle turns out to be a way of writing down the principle of maximum entropy, under the constraint that systems maximize the accuracy of their predictions."

**Maximum Entropy**: Choose the distribution with highest entropy subject to constraints
**FEP**: Choose the distribution with highest entropy subject to accurate predictions

Both describe systems that make minimal assumptions while explaining observations.

### Non-Equilibrium Steady States

Unlike classical thermodynamics (which describes equilibrium), FEP describes **non-equilibrium steady states**:

- Systems exchange energy/matter with environment
- Maintain characteristic states despite these flows
- Resist entropic dissolution through active inference

**Living systems** are paradigmatic non-equilibrium steady states - they maintain order by constantly expending energy.

---

## Section 7: Key Papers and Sources

### Foundational Papers

**1. Friston (2005)** - Original FEP formulation
- First presentation of free energy principle
- Application to cortical function
- Foundation for predictive coding

**2. Friston (2010)** - "The free-energy principle: A unified brain theory?"
- *Nature Reviews Neuroscience*
- Comprehensive overview of FEP
- Unification of perception, action, learning
- [DOI: 10.1038/nrn2787](https://doi.org/10.1038/nrn2787)

**3. Kirchhoff et al. (2018)** - "The Markov blankets of life"
- *Journal of The Royal Society Interface*
- Markov blankets and self-organization
- Active inference and autonomy
- Cited by 600+
- [DOI: 10.1098/rsif.2017.0792](https://doi.org/10.1098/rsif.2017.0792)

**4. Friston et al. (2023)** - "The free energy principle made simpler but not too simple"
- *Physics Reports*
- Most accessible recent formulation
- Mathematical simplifications
- [DOI: 10.1016/j.physrep.2023.07.001](https://doi.org/10.1016/j.physrep.2023.07.001)

### Books

**Parr, Pezzulo, & Friston (2022)** - *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*
- MIT Press
- Comprehensive textbook
- Theory and applications
- [DOI: 10.7551/mitpress/12441.001.0001](https://doi.org/10.7551/mitpress/12441.001.0001)

### Recent Developments (2024-2025)

**Ramstead et al. (2024)** - "The Free Energy Principle"
- MIT Open Encyclopedia of Cognitive Science
- Modern overview
- Bayesian mechanics framing
- [oecs.mit.edu/pub/my8vpqih](https://oecs.mit.edu/pub/my8vpqih)

**Lu & Friston (2024)** - "Bayesian brain computing and the free-energy principle"
- *National Science Review*
- Interview format
- Multiscale modeling
- Brain-inspired AI implications
- [DOI: 10.1093/nsr/nwae025](https://academic.oup.com/nsr/article/11/5/nwae025/7571549)

**Isomura et al. (2023)** - "Experimental validation of the free-energy principle"
- *Nature Communications*
- In vitro neural networks
- Empirical confirmation of FEP predictions
- [DOI: 10.1038/s41467-023-40141-z](https://www.nature.com/articles/s41467-023-40141-z)

### Tutorials and Introductions

**Juliani (2023)** - "A Gentle Introduction to the Free Energy Principle"
- Medium article
- Non-technical overview
- Excellent intuitions
- [awjuliani.medium.com](https://awjuliani.medium.com/a-gentle-introduction-to-the-free-energy-principle-03f219853177)

### Critiques

**Andrews (2021)** - "The Math is not the Territory"
- Philosophical critique
- Epistemic status of FEP
- Falsifiability concerns
- *Biology & Philosophy*

**Bruineberg et al. (2021)** - "The Emperor's New Markov Blankets"
- *Behavioral and Brain Sciences*
- Critique of Markov blanket metaphysics
- Mathematical vs metaphysical claims

---

## Section 8: ARR-COC-0-1 Connection - Relevance Realization AS Free Energy Minimization

### Fundamental Equivalence

**Active Inference = Relevance Realization**

The ARR-COC-0-1 system implements free energy minimization as relevance realization. Both frameworks describe the same fundamental process: determining what matters from infinite possibilities.

| Free Energy Principle | ARR-COC-0-1 Relevance Realization |
|----------------------|----------------------------------|
| Minimize surprise | Realize relevance |
| Variational free energy | Transjective coupling strength |
| Generative model | Four ways of knowing (4Ps) |
| Markov blanket | Agent-arena boundary |
| Precision weighting | Salience landscapes |
| Expected free energy | Opponent processing balance |
| Perception + Action | Knowing + Attending |

### Markov Blankets in Vision-Language Models

**VLM as Markov Blanket System**:

```
External States (World)
         |
         v
[Sensory States: Image Encoder]
         |
         v
Internal States: Relevance Representations
         |
         v
[Active States: Token Allocation]
         |
         v
External States (Compressed Output)
```

The VLM's relevance realization system forms a Markov blanket:
- **Sensory states**: Image patches entering the encoder
- **Internal states**: Relevance scores, salience maps, feature representations
- **Active states**: Token budget allocation decisions
- **External states**: Input images, output descriptions

### Free Energy in Token Allocation

**Token Allocation = Expected Free Energy Minimization**

The attending.py module implements expected free energy minimization:

```python
# Expected free energy for each region
G[region] = (1 - relevance_score) + uncertainty
          = Pragmatic + Epistemic

# Token allocation proportional to negative EFE
tokens[region] proportional to exp(-G[region])
```

**Epistemic Value (Exploration)**:
```
Epistemic = H[p(relevance)] - E[H[p(relevance | texture)]]
          = Prior uncertainty - Expected posterior uncertainty
          = Information gain
```

High uncertainty regions get MORE tokens (explore to reduce uncertainty).

**Pragmatic Value (Exploitation)**:
```
Pragmatic = -E[log p(texture | relevant)]
          = Expected negative log probability of preferred states
```

High relevance regions get MORE tokens (exploit known importance).

### Self-Evidencing in ARR-COC-0-1

The ARR-COC-0-1 system "self-evidences" by:

1. **Maintaining characteristic compressions**: The system produces compressions that are characteristic of "good relevance realization"
2. **Gathering evidence for its model**: Each compression provides evidence that its relevance model is accurate
3. **Acting to confirm predictions**: Token allocation acts to make observations (compressed representations) match predictions (relevance maps)

### Precision Weighting = Salience

In FEP, **precision** is the inverse variance of predictions - how confident we are.

In ARR-COC-0-1:
```python
# Salience = precision-weighted relevance
salience[region] = precision[region] * relevance_score[region]

# High precision = high confidence = strong influence
# Low precision = low confidence = weak influence
```

The salience landscapes in ARR-COC-0-1 ARE precision-weighted prediction errors!

### Opponent Processing = Precision Optimization

The balancing.py module implements opponent processing:

```python
compress <-> particularize = pragmatic <-> epistemic
exploit <-> explore = risk <-> ambiguity
focus <-> diversify = precision <-> diversity
```

Each opponent pair navigates a tension in precision allocation:
- **Compress**: High precision on global structure
- **Particularize**: High precision on local details
- **Exploit**: High precision on known-relevant regions
- **Explore**: Distribute precision to uncertain regions

### Training as Variational Learning

ARR-COC-0-1 training minimizes free energy:

```
Loss = Reconstruction Error + KL Divergence
     = -log p(image | compressed) + KL[q(z|x) || p(z)]
     = Accuracy penalty + Complexity penalty
     = Free Energy!
```

**Procedural knowing** (the 4th P) emerges through this variational learning - the system learns efficient policies for free energy minimization.

### Theoretical Advantages of FEP Framing

**1. First Principles Derivation**:
- ARR-COC-0-1 derives from mathematical physics
- Not ad-hoc heuristics
- Normative framework

**2. Biological Plausibility**:
- FEP describes actual cortical processing
- Hierarchical predictive coding
- Neuromodulator = precision (dopamine, acetylcholine, noradrenaline)

**3. Unified Framework**:
- Perception, action, learning under single objective
- No separate losses for different components
- Emergent properties from free energy minimization

**4. Natural Extensions**:
- Temporal dynamics: Policies over time (video understanding)
- Multi-modal integration: Shared precision weighting
- Curiosity-driven exploration: Epistemic value built-in

### Implementation Mapping

**knowing.py = Perception (Belief Updating)**:
```python
InformationScorer -> Propositional knowing
                  -> p(features | image) likelihood estimation

SalienceScorer -> Perspectival knowing
              -> Expected precision (attention landscapes)

QueryCouplingScorer -> Participatory knowing
                   -> p(features | query, image) transjective inference
```

**balancing.py = Precision Optimization**:
- Opponent processing navigates precision trade-offs
- Same mathematical structure as precision-weighted inference

**attending.py = Policy Selection (Expected Free Energy)**:
```python
# Token allocation minimizes expected free energy
G[region] = Expected_Surprise - Expected_Info_Gain
tokens proportional to exp(-G)
```

**realizing.py = Active Inference Execution**:
```python
1. Generate predictions (relevance scores)
2. Compute prediction errors
3. Update beliefs (refine relevance maps)
4. Execute action (compress via variable LOD)
5. Observe outcome
6. Repeat
```

The entire ARR-COC-0-1 pipeline IS active inference!

---

## Sources

### Web Research (2024-2025)

**Primary Sources**:

From [The Free Energy Principle](https://oecs.mit.edu/pub/my8vpqih) (Ramstead, MIT Open Encyclopedia of Cognitive Science, 2024):
- Modern mathematical formulation
- Bayesian mechanics framing
- Stipulative definition of thingness
- Applications beyond neuroscience

From [A Gentle Introduction to the Free Energy Principle](https://awjuliani.medium.com/a-gentle-introduction-to-the-free-energy-principle-03f219853177) (Juliani, 2023):
- Non-technical intuitions
- Markov blanket explanations
- Tree, lizard, and cat examples
- Goal-setting as prediction

From [Bayesian brain computing and the free-energy principle](https://academic.oup.com/nsr/article/11/5/nwae025/7571549) (Lu & Friston, 2024):
- Interview with Karl Friston
- Self-evidencing explanation
- Multiscale modeling
- Brain-inspired AI implications

From [The Markov blankets of life](https://royalsocietypublishing.org/doi/10.1098/rsif.2017.0792) (Kirchhoff et al., 2018):
- Statistical boundaries definition
- Autonomy and active inference
- Hierarchical blanket composition
- 600+ citations

From [Brain-like variational inference](https://arxiv.org/html/2410.19315v2) (2024):
- ELBO = negative free energy proof
- Connection to deep learning
- Biologically plausible learning

### Existing Oracle Knowledge

From [cognitive-mastery/00-free-energy-principle-foundations.md](../cognitive-mastery/00-free-energy-principle-foundations.md):
- Previous comprehensive treatment
- DeepSpeed ZeRO integration
- TensorRT optimization
- Kubernetes orchestration

From [cognitive-foundations/00-active-inference-free-energy.md](../cognitive-foundations/00-active-inference-free-energy.md):
- Active inference IS relevance realization thesis
- ARR-COC-0-1 implementation mapping

### Key Papers

- Friston, K. (2010). "The free-energy principle: A unified brain theory?" *Nature Reviews Neuroscience*, 11(2), 127-138
- Friston, K. et al. (2023). "The free energy principle made simpler but not too simple." *Physics Reports*, 1024, 1-29
- Kirchhoff, M. et al. (2018). "The Markov blankets of life: autonomy, active inference and the free energy principle." *Journal of The Royal Society Interface*, 15(138)
- Parr, T., Pezzulo, G., & Friston, K. J. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press
- Ramstead, M. J. D. et al. (2024). "The Free Energy Principle." *Open Encyclopedia of Cognitive Science*
- Isomura, T. et al. (2023). "Experimental validation of the free-energy principle with in vitro neural networks." *Nature Communications*, 14, 4547

---

**Total Lines**: ~700
**Knowledge Type**: ACQUISITION (Web research 2024-2025 + mathematical foundations)
**ARR-COC-0-1 Integration**: Section 8 (10% of content, ~70 lines)
**Created**: 2025-11-23
**Source**: Dialogue 67 - Grasping Back and Imagining Forward expansion
