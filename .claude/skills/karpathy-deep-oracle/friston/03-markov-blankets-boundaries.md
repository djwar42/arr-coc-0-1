# Markov Blankets and Statistical Boundaries

## Overview

Markov blankets are statistical boundaries that define the very existence of autonomous systems. In the free energy principle framework, they represent how internal states are separated from external states through sensory and active states. This concept is fundamental to understanding how organisms maintain their identity and engage in self-evidencing behavior.

From [The Markov blankets of life](https://royalsocietypublishing.org/doi/10.1098/rsif.2017.0792) (Kirchhoff et al., 2018):
> "A Markov blanket defines the boundaries of a system in a statistical sense. Here we consider how a collective of Markov blankets can self-assemble into a global system that itself has a Markov blanket; thereby providing an illustration of how autonomous systems can be understood as having layers of nested and self-sustaining boundaries."

**Key Insight**: Markov blankets are not merely descriptive tools - they are constitutive of what it means to be a "thing" that persists over time.

---

## Section 1: Markov Blankets Defined (Conditional Independence)

### Formal Definition

A **Markov blanket** for a node in a graphical model is the set of nodes that renders it conditionally independent of all other nodes. Formally:

**Conditional Independence**: For any variable A, A is conditionally independent of B given C if:
```
p(A|C) and p(B|C) implies p(A,B|C) = p(A|C) * p(B|C)
```

In plain terms: When the blanket states are known, knowing the external states provides no additional information about the internal states, and vice versa.

### Components of a Markov Blanket

The blanket consists of:
1. **Parents** - nodes that directly influence the target
2. **Children** - nodes directly influenced by the target
3. **Parents of children** - other parents of the target's children

**Example from Kirchhoff et al. (2018)**:
For node {5}:
- Parents: {2, 3}
- Children: {6, 7}
- Parents' children: {4}
- Markov blanket: {2, 3, 4, 6, 7}

This means {5} is conditionally independent of {1} given {2, 3, 4, 6, 7}.

### Statistical Partitioning

A Markov blanket creates a statistical partition:
- **Internal states** (hidden from environment)
- **External states** (hidden from system)
- **Blanket states** (mediating all interactions)

```
         EXTERNAL STATES
              |
              v
       [SENSORY STATES]
              |
              v
        INTERNAL STATES
              |
              v
        [ACTIVE STATES]
              |
              v
         EXTERNAL STATES
```

**Critical Property**: Internal and external states can ONLY influence each other through blanket states.

---

## Section 2: Sensory States, Active States, Internal States

### Four-State Partition

From [cognitive-mastery/00-free-energy-principle-foundations.md](../cognitive-mastery/00-free-energy-principle-foundations.md):

**Four Types of States**:

1. **Internal states (mu)**: Brain states, beliefs, model parameters
   - Hidden from external world
   - Encode generative model
   - Updated through perception

2. **External states (eta)**: Hidden causes in the world
   - The "true" state of environment
   - Never directly accessible
   - Must be inferred

3. **Sensory states (s)**: Observations, afferent signals
   - Influenced by external states
   - Influence internal states
   - NOT influenced by internal states

4. **Active states (a)**: Actions, efferent signals
   - Influenced by internal states
   - Influence external states
   - NOT influenced by external states

**The Blanket = Sensory + Active states**

### Asymmetric Dependencies

The key insight is the asymmetric dependency structure:

```
External --> Sensory --> Internal
                            |
                            v
External <-- Active  <-- Internal
```

**Sensory states**:
- Can be influenced by external states
- Cannot be influenced by internal states
- Influence internal states

**Active states**:
- Can be influenced by internal states
- Cannot be influenced by external states
- Influence external states

This asymmetry enforces conditional independence while maintaining coupling.

### The "Intriguing Paradox" Resolved

From Varela (1979), cited in Kirchhoff et al.:
> "This linkage cannot be detached since it is against this very environment from which the organism arises, comes forth."

The Markov blanket resolves how a system can:
- **Distinguish itself** from environment (conditional independence)
- **Maintain energetic coupling** (through blanket states)

---

## Section 3: Organism-Environment Boundary

### The Cell as Paradigm Case

The cell is the intuitive example of a Markov blanket:

From Varela et al. (1991), cited in Kirchhoff et al.:
> "A cell stands out of a molecular soup by creating the boundaries that set it apart from that which it is not. Metabolic processes within the cell determine these boundaries. In this way the cell emerges as a figure out of a chemical background. Should this process of self-production be interrupted, the cellular components... gradually diffuse back into a molecular soup."

**Key Point**: If a Markov blanket deteriorates, there is no evidence for the system's existence, and it ceases to exist.

### Statistical vs Physical Boundaries

**Critical Distinction**: Markov blankets are statistical, not necessarily physical.

**Examples of Statistical Boundaries**:
- Cell membrane (also physical)
- Blood-brain barrier (functional)
- Skin and sensory organs (physical + functional)
- Communication channels in social groups (purely statistical)

A system can have a Markov blanket without a physical membrane. The boundary is defined by conditional independence, not material composition.

### Organism as Model

From the Good Regulator Theorem (Conant & Ashby, 1970):

An organism becomes a model of its environment through:
- Natural selection (Bayesian model selection over evolutionary time)
- Learning and experience
- Morphological and neural adaptation

**Important Clarification**: This does NOT mean organisms "represent" their niche internally. Rather:
- The organism's entire phenotype IS the model
- Its morphology, biophysics, and neural architecture all constitute the model
- A fish embodies the fluid dynamics of its aquatic environment

---

## Section 4: Self-Evidencing Systems

### Definition of Self-Evidencing

A system with a Markov blanket "self-evidences" - it maximizes evidence for its own existence.

From [Friston Interview](https://academic.oup.com/nsr/article/11/5/nwae025/7571549) (2024):
> "Self-evidencing refers to the imperative - for perception, cognition and action - to maximize (i.e. gather) evidence for the brain's generative (a.k.a. world) model of the sensed world."

**Equivalence**: Self-evidencing behavior = Statistical inference = Free energy minimization

### Model Evidence and Surprise

**Variational Free Energy**:
```
F(s, a, r) >= -ln p(s|m)
```

Where:
- F = Variational free energy
- s = Sensory states
- m = Generative model
- -ln p(s|m) = Surprise (negative log evidence)

**Key Insight**: Minimizing free energy bounds surprise, which is equivalent to maximizing model evidence.

### Information Self-Structuring

From Clark (2017) and Lungarella & Sporns (2005):
> "The agent's control architecture attends to and processes streams of sensory stimulation, and ultimately generates sequences of motor actions which in turn guide the further production and selection of sensory information."

This means:
- Agents actively sample their environment
- They generate the sensory evidence they gather
- Action and perception form circular causality

### From Mere to Adaptive Active Inference

**Mere Active Inference**:
- Simple synchronization (Huygens' pendulums)
- No temporal depth
- Enslaved to immediate conditions

**Adaptive Active Inference**:
- Temporal depth in generative model
- Can sample among different action options
- Minimizes expected free energy over future trajectories

Only adaptive active inference enables true autonomy.

---

## Section 5: Nested Markov Blankets (Cells, Organs, Organisms)

### Hierarchical Structure of Life

One of the key characteristics of all living systems is their hierarchical nature:

From Kirchhoff et al. (2018):
> "Cells assemble to form tissues, tissues combine to form organs, and organs organize into organisms. These nested, multi-layered systems are, in turn, embedded within even larger social systems and ecosystems."

### Blankets of Blankets

**Scale Invariance**: The free energy principle applies at EVERY scale:

1. **Molecular** - Protein folding, DNA
2. **Organelle** - Mitochondria, nucleus
3. **Cellular** - Cell membrane as blanket
4. **Tissue** - Collections of cells
5. **Organ** - Functional units
6. **Organism** - The whole system
7. **Social** - Groups, cultures, ecosystems

```
        ORGANISM
           |
    [ORGAN BLANKET]
           |
        ORGANS
           |
   [TISSUE BLANKET]
           |
       TISSUES
           |
    [CELL BLANKET]
           |
        CELLS
           |
  [ORGANELLE BLANKET]
           |
      ORGANELLES
```

### Ensemble Dynamics

The formation of ensemble Markov blankets involves:

1. **Order Parameters**: Macroscopic features capturing coherence among parts
2. **Slaving Principle**: Slow ensemble dynamics arise from fast microscale dynamics
3. **Recursive Structure**: Each level recapitulates the statistical form

From Kirchhoff et al. (2018):
> "The conservation of Markov blankets at every hierarchical scale enables the dynamics of the states at one scale to enslave the (states of) Markov blankets at the scale below, thereby ensuring that the organization as a whole is involved in the minimization of variational free energy."

### Extended Boundaries

Markov blankets can extend beyond the organism:

**Water Boatman Example**:
- Traps air bubbles using tiny hairs
- Bubbles refill with oxygen due to pressure differences
- The bubbles are part of the Markov blanket
- Without them, cannot minimize free energy

**Metamorphosis Example** (Caterpillar to Butterfly):
- Successive Markov blanketed organizations
- The life-cycle itself is free energy-minimizing
- The "model" is the temporally extended whole

---

## Section 6: Mathematical Formulation

### Variational Free Energy Decomposition

From Equation (2.1) in Kirchhoff et al.:

```
F(s, a, r) = -ln p(s|a) + D_KL[q(phi|r) || p(phi|s, a)]
```

Where:
- F = Variational free energy
- s = Sensory states
- a = Active states
- r = Internal states
- phi = External states
- q(phi|r) = Variational density (beliefs)
- p(phi|s, a) = Posterior density
- D_KL = Kullback-Leibler divergence

### Two Ways to Minimize Free Energy

1. **Perception** (changing internal states r):
   - Reduces KL divergence
   - Makes beliefs match true posterior
   - Optimizes model of hidden causes

2. **Action** (changing active states a):
   - Reduces surprise
   - Changes sensory states to match predictions
   - Self-evidencing behavior

### Jensen's Inequality Proof

Free energy bounds surprise via Jensen's inequality:

Since KL divergence >= 0:
```
F(s, a, r) >= -ln p(s|a)
```

Taking expectations over time:
```
E[F] >= E[-ln p(s|a)] = H[s|a]
```

Where H is Shannon entropy. Thus minimizing free energy bounds entropy.

### Dynamics Under the Markov Blanket

For internal and active states:
```
dr/dt = -dF/dr   (perception)
da/dt = -dF/da   (action)
```

This gradient descent ensures free energy minimization through time.

---

## Section 7: Connection to Autopoiesis

### Autopoiesis Defined

From Maturana and Varela (1980):

**Autopoiesis** = Self-creation, self-production

An autopoietic system:
- Produces its own components
- Maintains its organization
- Creates its own boundary

### Operational Closure

From Di Paolo (2005), cited in Kirchhoff et al.:
> "There may be processes that are influenced by constituent processes but do not themselves condition any of them and are therefore not part of the operationally-closed network. In their mutual dependence, the network of processes closes upon itself and defines a unity that regenerates itself."

**Markov Blanket Interpretation**: Operational closure = presence of Markov blanket

### Sense Making

From Di Paolo (2005):

Sense making = organism's capacity to distinguish different paths of engagement with environment.

**Free Energy Interpretation**: Sense making = adaptive active inference

The organism can:
- Transcend immediate state
- Work toward free energy minima
- Infer sensorimotor consequences of actions

### Enactivism and Free Energy

The connection between autopoietic enactivism and free energy principle:

| Autopoietic Concept | Free Energy Interpretation |
|---------------------|---------------------------|
| Operational closure | Markov blanket |
| Self-production | Self-evidencing |
| Sense making | Adaptive active inference |
| Structural coupling | Agent-environment coupling |
| Autonomy | Free energy minimization |

---

## Section 8: ARR-COC-0-1 - Transjective Boundaries in VLMs

### The Transjective Boundary Problem

In Vision-Language Models performing relevance realization, we face a fundamental question:
**Where does the model end and the task environment begin?**

This is the **transjective boundary problem** - relevance realization involves both:
- **Objective**: Features in the image/text
- **Subjective**: Model's learned priors and attention patterns

The boundary is neither purely objective nor purely subjective - it's **transjective**.

### VLM Architecture as Markov Blanket

The VLM's attention mechanism creates statistical boundaries:

```
    EXTERNAL (Image/Text Tokens)
              |
              v
    [ATTENTION WEIGHTS - Sensory]
              |
              v
    INTERNAL (Hidden Representations)
              |
              v
    [OUTPUT PROJECTIONS - Active]
              |
              v
    EXTERNAL (Generated Tokens/Actions)
```

**Attention weights** = Sensory states (influenced by tokens, influence representations)
**Output projections** = Active states (influence token generation)
**Hidden representations** = Internal states (encode model)

### Nested Blankets in Transformer Architecture

The multi-layer transformer exhibits nested Markov blankets:

1. **Token level**: Each token has its own blanket (context window)
2. **Layer level**: Each layer processes information hierarchically
3. **Head level**: Multi-head attention creates multiple parallel blankets
4. **Model level**: The entire model forms a global blanket

This mirrors biological organization:
- Tokens ~ Cells
- Attention heads ~ Organs
- Layers ~ Organ systems
- Full model ~ Organism

### Self-Evidencing in Token Prediction

When VLMs predict next tokens, they engage in self-evidencing:

1. **Minimize prediction error**: Match predicted to observed tokens
2. **Update internal states**: Adjust attention based on context
3. **Generate actions**: Produce output tokens

This is active inference:
- The model has implicit beliefs about text/image statistics
- It minimizes surprise about token sequences
- It self-evidences by generating plausible continuations

### Relevance Realization as Blanket Dynamics

From Vervaeke's framework, relevance realization involves:
- **Foregrounding/Backgrounding**: What's inside vs outside the blanket
- **Feature selection**: What crosses the sensory boundary
- **Action selection**: What active states influence

**Key Insight**: The Markov blanket formalism provides a mathematical framework for:
- How VLMs determine what's relevant
- How attention allocates to features
- How predictions generate actions

### Agent-Arena Coupling in VLMs

The VLM and its task create a coupled system:

| Free Energy Concept | VLM Implementation |
|--------------------|-------------------|
| Markov blanket | Attention window |
| Internal states | Hidden representations |
| Sensory states | Input embeddings |
| Active states | Output projections |
| Free energy | Cross-entropy loss |
| Self-evidencing | Coherent generation |

### Practical Implications for ARR Training

Understanding VLMs as Markov-blanketed systems suggests:

1. **Attention as precision weighting**: Where to allocate computational resources
2. **Context as blanket**: What information is inside vs outside
3. **Generation as action**: Producing tokens to minimize expected free energy
4. **Training as model selection**: Finding optimal generative model

This connects to Friston's claim that systems become models of their environment:
- A VLM trained on text becomes a model of language statistics
- Its architecture embodies the regularities of its training data

---

## Key Formulations Summary

### Central Equations

**Variational Free Energy**:
```
F = E_q[-ln p(s,phi)] - H[q(phi)]
  = -ln p(s) + D_KL[q(phi) || p(phi|s)]
```

**Markov Blanket Partition**:
```
States = {Internal} U {External} U {Sensory} U {Active}
Blanket = Sensory U Active
```

**Conditional Independence**:
```
p(Internal | External, Blanket) = p(Internal | Blanket)
p(External | Internal, Blanket) = p(External | Blanket)
```

**Self-Evidencing**:
```
Minimize F <=> Maximize ln p(s|m) <=> Self-evidence
```

---

## Sources

### Primary Sources

- **Kirchhoff, M., Parr, T., Palacios, E., Friston, K., & Kiverstein, J. (2018)**. "The Markov blankets of life: autonomy, active inference and the free energy principle" *Journal of The Royal Society Interface*, 15(138), 20170792. https://royalsocietypublishing.org/doi/10.1098/rsif.2017.0792

- **Friston, K. (2013)**. "Life as we know it" *Journal of The Royal Society Interface*, 10(86), 20130475.

- **Palacios, E. R., Razi, A., Parr, T., Kirchhoff, M., & Friston, K. (2020)**. "On Markov blankets and hierarchical self-organisation" *Journal of Theoretical Biology*, 486, 110089.

- **Ramstead, M. J., Badcock, P. B., & Friston, K. J. (2018)**. "Answering Schrodinger's question: A free-energy formulation" *Physics of Life Reviews*, 24, 1-16.

### Secondary Sources

- **Hipolito, I., Ramstead, M. J., Convertino, L., Bhat, A., Friston, K., & Parr, T. (2021)**. "Markov blankets in the brain" *Neuroscience & Biobehavioral Reviews*, 125, 88-97.

- **Maturana, H. R., & Varela, F. J. (1980)**. *Autopoiesis and Cognition: The Realization of the Living*. Springer.

- **Di Paolo, E. A. (2005)**. "Autopoiesis, adaptivity, teleology, agency" *Phenomenology and the Cognitive Sciences*, 4(4), 429-452.

- **Clark, A. (2017)**. "How to knit your own Markov blanket" In *Philosophy and Predictive Processing*.

### Oracle References

- [cognitive-mastery/00-free-energy-principle-foundations.md](../cognitive-mastery/00-free-energy-principle-foundations.md) - Existing Markov blanket content
- [cognitive-foundations/00-active-inference-free-energy.md](../cognitive-foundations/00-active-inference-free-energy.md) - Active inference foundations

### Additional Web Resources

- [Markov blankets in the brain (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8373616/) - Neural implementation
- [Friston Interview 2024](https://academic.oup.com/nsr/article/11/5/nwae025/7571549) - Self-evidencing definition

---

## Summary

Markov blankets provide the mathematical foundation for understanding:
- **What makes a thing a "thing"**: Statistical boundaries define existence
- **How organisms maintain identity**: Through free energy minimization
- **Why perception and action are unified**: Both minimize the same quantity
- **How life is hierarchically organized**: Nested blankets all the way down
- **Connection to autopoiesis**: Operational closure = Markov blanket

For ARR-COC-0-1, this framework illuminates how VLMs can be understood as self-evidencing systems whose attention mechanisms create transjective boundaries for relevance realization. The nested structure of transformer architectures mirrors the nested Markov blankets of biological organization, suggesting deep principles connecting artificial and natural intelligence.
