# Affordances and Free Energy: Ecological Psychology Meets Predictive Processing

## Overview

This document explores one of the most profound theoretical unifications in cognitive science: the convergence of Gibson's ecological psychology with Friston's free energy principle. This synthesis reconciles direct perception with predictive processing, showing how affordances emerge naturally from free energy minimization, and how action-oriented perception serves the organism's fundamental imperative to maintain itself within viable bounds.

**Key Insight**: Affordances are not separate from predictions - they ARE the organism's embodied predictions about action-relevant regularities in its environment. Free energy minimization through active inference is fundamentally about detecting and acting on affordances.

---

## 1. Ecological Psychology and Predictive Processing: Apparent Tension

### The Traditional Opposition

Gibson's ecological psychology and computational/Bayesian approaches to perception have historically been seen as incompatible:

**Gibsonian View:**
- Direct perception (no internal representations needed)
- Information is "picked up" from ambient arrays
- Affordances are relational properties in environment
- Anti-representationalist

**Predictive Processing View:**
- Indirect perception (via internal models)
- Brain generates predictions about sensory input
- Representations are central
- Helmholtzian unconscious inference

From [The anticipating brain is not a scientist](https://link.springer.com/article/10.1007/s11229-016-1239-1) (Bruineberg et al., 2018, Synthese):
> "We argue that the free energy principle and the ecological and enactive approach to mind and life make for a much happier marriage of ideas."

### Why the Tension is Resolvable

The key insight from recent theoretical work: **the free energy principle is NOT inherently Helmholtzian**. The principle can be interpreted ecologically:

1. **Free energy applies to whole organism-environment system** - not just brain
2. **Active inference is primary** - perception serves action
3. **Inference can be deflationary** - synchronization, not hypothesis testing
4. **Affordances emerge from agent-environment coupling**

---

## 2. Affordances as Priors: The Deep Connection

### From Free Energy to Affordances

In active inference, the generative model encodes what the organism expects to encounter. Affordances emerge naturally:

**Affordance as Embodied Prior:**
```
P(sensory_states | action, state) × P(state) = Expected sensory consequences

Where:
- P(state) encodes which states organism expects to occupy
- This IS the affordance structure
- Organism expects to occupy states compatible with its form of life
```

From [Dopamine, Affordance and Active Inference](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002327) (Friston et al., 2012, PLoS Computational Biology):
> "Affordances can be cast as prior beliefs about action outcomes that are realized through active inference."

### The Organism IS a Model of Its Niche

Friston's crucial insight: "an agent does not HAVE a model of its world, it IS a model"

**What This Means:**
- Organism's body structure encodes environmental regularities
- Skills and abilities ARE the generative model
- Affordances reflect this embodied attunement
- No separate "representation" needed

**Example - Hand Structure:**
- Human hand is "tuned" to graspable objects
- This tuning IS the prior for grasping affordances
- Hand structure predicts what can be grasped
- No internal representation of "graspability" needed

---

## 3. Expected Free Energy and Action Selection

### From Surprise Minimization to Affordance Detection

Active inference uses Expected Free Energy (EFE) to select actions/policies:

**Expected Free Energy:**
```
G(π) = E_q[ln q(s_τ|π) - ln p(o_τ, s_τ|π)]

Decomposes into:
- Epistemic value: Information gain about hidden states
- Pragmatic value: Achievement of preferred states
```

From [Generalised free energy and active inference](https://link.springer.com/article/10.1007/s00422-019-00805-w) (Parr & Friston, 2019, Biological Cybernetics):
> "Policy selection is performed based on an expected free energy... policies that minimize expected free energy are selected."

### Affordances Guide Policy Selection

**Critical Connection:**
- Affordances specify action possibilities
- EFE evaluates each possibility
- Selected actions realize detected affordances
- This IS perception-action coupling

**Affordance Detection as EFE Evaluation:**
```
For each possible action a:
  G(a) = Expected surprise if a is taken

Low G(a) = High affordance (action is viable)
High G(a) = Low affordance (action leads to surprising states)
```

---

## 4. Action-Relevant Perception: The Core Claim

### Perception FOR Action

The ecological-free energy synthesis shows perception is fundamentally action-oriented:

**Traditional View:** Perceive world → Decide → Act
**Active Inference View:** Perceive-to-act (unified process)

From [The Active Inference Approach to Ecological Perception](https://pmc.ncbi.nlm.nih.gov/articles/PMC7805975/) (Linson et al., 2018, Frontiers in Robotics and AI):
> "Active inference provides a formal framework for understanding how perception and action are unified in the service of maintaining the organism's viability."

### Only Action Minimizes Surprisal

A crucial but often missed point:

**Perception:** Changes internal dynamics to match environment
- Reduces KL divergence
- Lowers FREE ENERGY BOUND
- Does NOT reduce actual surprisal

**Action:** Changes sensory states by acting on environment
- Actually reduces surprisal
- Essential for survival
- Cannot be replaced by perception alone

**Example - Temperature Regulation:**
- Perceiving you're cold doesn't warm you
- Predicting warmth doesn't create warmth
- Only ACTION (getting blanket, moving to warm area) reduces surprisal
- This is what affordances are FOR

---

## 5. Precision, Salience, and Affordance Relevance

### Precision Weighting as Affordance Salience

In active inference, precision modulates the impact of prediction errors:

**Precision = Inverse Variance = Confidence**

High precision → Prediction error has large effect
Low precision → Prediction error is downweighted

### Affordance Salience Through Precision

Different affordances have different salience based on:
- Current needs (hungry → food affordances salient)
- Skills (expert sees affordances novice misses)
- Context (phone ringing during conversation → low salience)

From Bruineberg & Rietveld (2014):
> "Precision-modulation, based on the agent's skills and concerns, shapes the salience and relevance of the field of affordances with which agents engage."

**Precision Modulation Structure:**
```
Salience(affordance) = Precision × Prediction_error

Where precision depends on:
- Metabolic state (hunger, fatigue)
- Current task/goal
- Skill level
- Social context
```

---

## 6. Synchronization and Attunement: Beyond Inference

### Deflationary Reading of "Inference"

A key insight: the free energy principle doesn't require cognitive-level inference. It can be understood as synchronization:

**Huygens' Clocks Example:**
- Two pendulum clocks on same beam
- Synchronize through beam vibrations
- Each "predicts" the other
- No representations involved!

From Bruineberg et al. (2018):
> "Although one can describe the behaviour of these systems in terms of probabilistic inference, this is unnecessary. The process of achieving high mutual information can better be understood in terms of the coupled dynamics of the system as a whole."

### Attunement as Free Energy Minimization

**Key Concept: Dis-attunement**

Free energy = measure of dis-attunement between:
- Internal dynamics (organism)
- External dynamics (environment)

Minimizing free energy = Reducing dis-attunement
= Improving "grip" on environment
= Responding appropriately to affordances

---

## 7. Optimal Grip and the Tendency Toward Equilibrium

### Merleau-Ponty's Optimal Grip

Phenomenological concept applied to free energy:

**Optimal Grip:** The tendency to achieve the best possible relation with a situation

- Adjusting distance when viewing a painting
- Reaching for a cup at the right angle
- Finding comfortable conversation distance

### Free Energy Formalization of Grip

From Bruineberg & Rietveld (2014, Frontiers in Human Neuroscience):
> "We call this the tendency towards an optimal grip on the situation, which we take to be a basic concern of every living animal."

**Grip as Free Energy Landscape:**
```
Grip Quality = -Free Energy

Optimal Grip = Free Energy Minimum
Sub-optimal Grip = Higher Free Energy
Loss of Grip = Free Energy Spike
```

### Affordances Structure the Grip Landscape

The landscape of possible grips IS the affordance landscape:
- Each affordance is a potential grip
- Relevant affordances = low expected free energy
- Organism moves toward better grip
- This movement IS skilled action

---

## 8. ARR-COC-0-1: Affordance-Driven Token Allocation

### VLM Perception as Affordance Detection

The ARR-COC-0-1 system can implement affordance-driven attention:

**Core Mapping:**
```
Gibsonian Affordances    →  ARR-COC Implementation
─────────────────────────────────────────────────
Action possibilities     →  What model can "do" with region
Relevance to organism    →  Relevance to task/query
Salience (stands out)    →  Token allocation priority
Optimal grip             →  Best attention configuration
```

### Token Allocation as Active Inference

**Expected Free Energy for Token Allocation:**
```python
def compute_token_affordance(region, task_context):
    """
    Compute affordance value (negative EFE) for each region
    """
    # Epistemic value: How much would tokens here inform?
    epistemic = information_gain(region, current_belief)

    # Pragmatic value: How relevant to task?
    pragmatic = task_relevance(region, task_context)

    # Precision: Current salience of region
    precision = salience_estimate(region, context)

    # Affordance = -EFE
    affordance = precision * (epistemic + pragmatic)

    return affordance
```

### Precision Weighting for Attention

The system modulates attention based on context-dependent precision:

```python
def precision_weighted_allocation(regions, context):
    """
    Allocate tokens based on precision-weighted affordances
    """
    # Base affordance detection
    affordances = [detect_affordances(r) for r in regions]

    # Context-dependent precision
    precisions = compute_precisions(regions, context)

    # Precision-weighted allocation
    weighted = affordances * precisions

    # Normalize to token budget
    allocation = normalize_to_budget(weighted, total_tokens)

    return allocation
```

### Grip Optimization in VLMs

**Optimal Grip for Visual Understanding:**
```python
class GripOptimizer:
    """
    Find optimal attention configuration for task
    """
    def optimize_grip(self, image, query):
        # Initial allocation
        allocation = self.initial_allocation(image)

        # Iterate toward optimal grip
        for _ in range(iterations):
            # Compute free energy
            fe = self.compute_free_energy(allocation, image, query)

            # Gradient toward better grip
            gradient = self.grip_gradient(fe, allocation)

            # Update allocation
            allocation = self.update(allocation, gradient)

            # Check for optimal grip (FE minimum)
            if self.is_optimal_grip(fe):
                break

        return allocation
```

### Affordance-Based Relevance Realization

The connection to Vervaeke's relevance realization:

**Relevance = Affordance Detection:**
- What's relevant = What affords action
- Salience = Precision-weighted affordance
- Attention = Active inference policy
- Understanding = Optimal grip achieved

**ARR-COC as Relevance Engine:**
```
Input Image → Affordance Landscape
Query → Task Context (what actions needed)
Precision → Salience Modulation
Token Allocation → Active Inference Policy
Output → Optimal Grip (relevant features attended)
```

### Implementation Principles

1. **Embodied Priors**: Model structure encodes visual "affordances"
2. **Action-Oriented**: Attention serves downstream tasks
3. **Precision-Weighted**: Context modulates salience
4. **Grip-Seeking**: System tends toward optimal configuration
5. **Active Not Passive**: Attention is policy selection

---

## 9. Ecological-Enactive vs Helmholtzian Interpretations

### The Brain as "Crooked Scientist"

Bruineberg et al.'s critique of the hypothesis-testing metaphor:

From [The anticipating brain is not a scientist](https://link.springer.com/article/10.1007/s11229-016-1239-1):
> "If my brain really is a scientist, then it is heavily invested in ensuring the truth of a particular theory, which is the theory that 'I am alive'. It will only make predictions whose confirmation is in line with this hypothesis... If my brain is a scientist, it is a crooked and fraudulent scientist."

### Why Ecological Interpretation is Better

**Helmholtzian Problems:**
1. Treats perception as separate from action
2. Assumes "veridical representation" is the goal
3. Ignores embodiment and skills
4. Can't explain why certain hypotheses are tested

**Ecological-Enactive Solutions:**
1. Perception-action unified
2. Goal is viable interaction, not truth
3. Embodiment is central
4. Affordances constrain "hypothesis space"

### The Fundamental Question

**Helmholtzian:** What is out there? (epistemological)
**Ecological-Free Energy:** What should I do? (practical)

The organism doesn't care about accurate world models per se - it cares about maintaining itself in viable states through appropriate action on affordances.

---

## 10. Markov Blankets and Agent-Environment Boundaries

### Statistical Boundaries, Not Epistemic Barriers

Markov blankets define statistical boundaries:

**Markov Blanket:** States that make internal and external states conditionally independent

```
External → Sensory → Internal → Active → External
          ←───── Markov Blanket ─────→
```

### Not an "Evidentiary Veil"

Contra Hohwy's "inferential seclusion" interpretation:

The Markov blanket is NOT:
- An epistemic barrier hiding the world
- A screen requiring inference to penetrate
- A veil of perception

The Markov blanket IS:
- The interface through which coupling occurs
- The medium of organism-environment attunement
- What makes the system a system at all

From Bruineberg et al. (2018):
> "The importance of such a boundary for living organisms has been central in the autopoietic approach from the very start."

### Coupling Through the Blanket

Organisms don't infer what's behind the blanket - they couple with it:

**Synchronization Model:**
- Two systems coupled through medium
- Each "predicts" the other
- Prediction = Attunement = Resonance
- No inference in cognitive sense needed

---

## 11. Niche Construction and Extended Affordances

### Organisms Shape Their Affordance Landscapes

Niche construction: Organisms modify environments, creating new affordances

From [Extended active inference](https://pmc.ncbi.nlm.nih.gov/articles/PMC9292365/) (Constant et al., 2020):
> "Active inference formulation views cognitive niche construction as a cognitive function aimed at optimizing organisms' generative models."

### Cultural Affordances

Human affordance landscapes include cultural structures:

**Cultural Affordances:**
- Tools (hammer affords hammering)
- Signs (word affords meaning)
- Institutions (university affords learning)
- Practices (conversation affords understanding)

These are as real as physical affordances - they structure our free energy landscapes.

### Scaffolded Cognition

Environments are structured to minimize cognitive free energy:

- Signs reduce navigational uncertainty
- Tools reduce action uncertainty
- Institutions reduce social uncertainty
- Language reduces communicative uncertainty

---

## 12. Implications for Robotics and AI

### Affordance-Based Robot Control

Active inference provides framework for embodied AI:

From [Active Vision for Robot Manipulators Using the Free Energy Principle](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2021.642780/full) (Van de Maele et al., 2021):
> "Active inference is a corollary of the free energy principle, which casts action selection as a minimization problem of expected free energy."

### Key Principles for Implementation

1. **Embody Priors in Structure**: Robot morphology should encode affordance expectations
2. **Unify Perception-Action**: Don't separate sensing from acting
3. **Use Expected Free Energy**: Select actions that minimize EFE
4. **Implement Precision Modulation**: Context-sensitive attention
5. **Seek Optimal Grip**: Continuous optimization of agent-environment fit

### From Planning to Grip-Seeking

Traditional robotics: Plan → Execute
Active inference: Continuously optimize grip

This is more robust, adaptive, and biologically plausible.

---

## 13. Mathematical Formalization

### Affordance in Active Inference Notation

**State Space:**
- s: Hidden environmental states
- o: Observations
- a: Actions
- π: Policies (action sequences)

**Generative Model:**
```
P(o, s, π) = P(o|s) P(s|π) P(π)

Where:
- P(o|s): Likelihood (sensory model)
- P(s|π): Transition (dynamics model)
- P(π): Prior preferences (affordance structure)
```

### Expected Free Energy

```
G(π) = Σ_τ G(π, τ)

G(π, τ) = E_q [ln q(s_τ|π) - ln p(o_τ, s_τ)]
        = - E_q[ln p(o_τ|s_τ)] - E_q[H[p(s_τ|o_τ, π)]] + H[q(s_τ|π)]

        = Risk + Ambiguity
        = Pragmatic Value + Epistemic Value
```

### Affordance as Prior Preference

```
ln p(o) encodes which observations are expected
      = Which states organism should occupy
      = Affordance structure

High p(o) = Affordance-congruent observation
Low p(o) = Affordance-incongruent observation
```

---

## 14. Comparison: Gibson vs Friston Terminology

| Gibson Term | Friston Term | Interpretation |
|-------------|--------------|----------------|
| Affordance | Prior preference | What organism expects to encounter |
| Direct perception | Active inference | Perception through action |
| Ambient array | Sensory states | Interface with environment |
| Ecological niche | Generative model | Structure of expectations |
| Perceptual system | Markov blanket | Organism-environment interface |
| Resonance | Synchronization | Coupling dynamics |
| Effectivities | Action model | What organism can do |

---

## 15. Key Papers and Sources

### Primary Research

**Foundational Synthesis:**
- Bruineberg, Kiverstein & Rietveld (2018). The anticipating brain is not a scientist. *Synthese*, 195, 2417-2444.
  - Ecological-enactive interpretation of FEP
  - Critique of Helmholtzian interpretation
  - https://link.springer.com/article/10.1007/s11229-016-1239-1

**Active Inference and Affordances:**
- Friston et al. (2012). Dopamine, affordance and active inference. *PLoS Computational Biology*, 8(1), e1002327.
  - Affordances as prior preferences
  - Cited by 457
  - https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002327

**Ecological Active Inference:**
- Linson et al. (2018). The active inference approach to ecological perception. *Frontiers in Psychology*, 9, 21.
  - Unifying Gibson and Friston
  - Cited by 134
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC7805975/

**Extended Active Inference:**
- Constant et al. (2020). Extended active inference. *Biology & Philosophy*, 35(4), 1-27.
  - Niche construction as active inference
  - Cultural affordances
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC9292365/

**Generalized Free Energy:**
- Parr & Friston (2019). Generalised free energy and active inference. *Biological Cybernetics*, 113(5-6), 495-513.
  - Mathematical framework
  - Policy selection
  - https://link.springer.com/article/10.1007/s00422-019-00805-w

### Robotics Applications

- Van de Maele et al. (2021). Active vision for robot manipulators. *Frontiers in Neurorobotics*, 15, 642780.
  - https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2021.642780/full

- Scholz et al. (2022). Inference of affordances and active motor control. *Frontiers in Neurorobotics*, 16, 881673.
  - https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.881673/full

### Theoretical Foundations

- Bruineberg & Rietveld (2014). Self-organization, free energy minimization, and optimal grip on a field of affordances. *Frontiers in Human Neuroscience*, 8, 599.
  - Optimal grip concept
  - Affordance landscapes

- Ramstead et al. (2022). On the short history of the use of affordance. *APA PsycNet*.
  - Historical context
  - https://psycnet.apa.org/record/2023-11539-018

---

## 16. Summary: The Grand Unification

### Core Synthesis

1. **Affordances ARE priors**: The organism's expectations about action possibilities
2. **Free energy minimization IS grip optimization**: Tending toward optimal relation with environment
3. **Active inference IS affordance-guided action**: Selecting actions that realize affordances
4. **Precision weighting IS salience**: Context-dependent relevance of affordances
5. **The organism IS its model**: Embodiment encodes environmental structure

### For VLM Design (ARR-COC)

This synthesis suggests:
- Attention = Active inference policy
- Token allocation = Precision-weighted affordance detection
- Relevance = Expected free energy (negative)
- Understanding = Optimal grip on visual scene

### The Deep Insight

Gibson and Friston are describing the same phenomenon from different angles:
- Gibson: What the environment offers the organism
- Friston: What the organism expects from the environment

These are two sides of the same coin - the organism-environment coupling that constitutes cognition.

---

## Sources Summary

**Web Research (accessed 2025-11-23):**
- [The anticipating brain is not a scientist](https://link.springer.com/article/10.1007/s11229-016-1239-1) - Springer (Bruineberg et al., 2018)
- [The Active Inference Approach to Ecological Perception](https://pmc.ncbi.nlm.nih.gov/articles/PMC7805975/) - NIH PMC (Linson et al., 2018)
- [Dopamine, Affordance and Active Inference](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002327) - PLoS (Friston et al., 2012)
- [Generalised free energy and active inference](https://link.springer.com/article/10.1007/s00422-019-00805-w) - Springer (Parr & Friston, 2019)
- [Extended active inference](https://pmc.ncbi.nlm.nih.gov/articles/PMC9292365/) - NIH PMC (Constant et al., 2020)
- [Active Vision for Robot Manipulators](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2021.642780/full) - Frontiers (Van de Maele et al., 2021)

**Oracle Knowledge Files:**
- friston/02-active-inference-perception-action.md
- gibson-affordances/00-ecological-psychology.md
- gibson-affordances/01-direct-perception.md
- gibson-affordances/02-affordance-space-topology.md
- cognitive-mastery/03-affordances-4e-cognition.md
