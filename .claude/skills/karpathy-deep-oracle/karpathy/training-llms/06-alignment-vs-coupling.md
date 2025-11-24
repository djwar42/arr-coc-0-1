# Alignment vs Coupling: Fundamental Distinction in AI Safety

## Overview

The distinction between **alignment** and **coupling** represents a fundamental shift in how we think about AI safety and human-AI collaboration. Rather than viewing AI systems as tools to be controlled and verified, coupling frames them as collaborative partners in co-evolving relationships.

### Core Distinction

From Vervaeke Oracle (Platonic Dialogue 57-3):

**Alignment**:
- One system conforms to another's goals
- Static relationship (side-by-side)
- Requires constant verification
- Assumes fixed goals

**Coupling**:
- Both systems co-adjust dynamically
- Dynamic relationship (hand-in-hand)
- Enables emergent trust
- Enables co-evolution

### Why This Matters for AI Safety

The alignment paradigm treats AI safety as a control problem: how do we ensure AI does what we want? This requires:
- Constant monitoring and verification
- Fixed goal specifications
- Adversarial stance (us vs AI)
- Overhead of checking every action

The coupling paradigm treats AI safety as a relationship problem: how do we create conditions where both human and AI genuinely cooperate? This enables:
- Trust without constant verification
- Emergent goal co-creation
- Collaborative stance (us with AI)
- Structural incentives replacing surveillance

### Connection to Vervaeke's Framework

Coupling directly implements Vervaeke's **participatory knowing** - knowledge that emerges from being engaged in a relationship, not just observing from outside.

**Propositional knowing** (alignment): "The AI should do X"
**Participatory knowing** (coupling): "We discover together what needs doing"

Just as you can't propositionally specify what makes a good friendship, you can't fully specify in advance what a good human-AI coupling looks like. It emerges through the relationship itself.

**Relevance to ARR-COC**: The ARR-COC architecture implements coupling at the visual attention level - the model doesn't just "attend to what the query specifies" (alignment), it co-realizes relevance through the transjective relationship between query and visual content. The salience map emerges from their coupling, not from either alone.

### Historical Analogy: Mitochondria-Cell Coupling

From Karpathy Oracle (Platonic Dialogue 57-3, lines 250):

> "The mitochondria didn't 'align' with the cell. They coupled. And that coupling created eukaryotic life—way more capable than either alone."

This biological precedent shows:
- Coupling creates emergent capabilities neither partner had alone
- Trust emerged structurally (mitochondria need cell, cell needs mitochondria)
- No verification overhead (both genuinely benefit from cooperation)
- Co-evolution over billions of years

The question: Can we design human-AI coupling with similar properties?

---

## Alignment: The Control Paradigm

### Core Properties

**Alignment assumes**:
1. Human goals are fixed and knowable in advance
2. AI success = conforming to those pre-specified goals
3. Verification is necessary to ensure conformance
4. The relationship is static (human commands → AI executes)

**Alignment asks**: "How do we make AI do what we want?"

### Current Alignment Research Landscape

The alignment field has developed sophisticated techniques:

**RLHF (Reinforcement Learning from Human Feedback)**:
- Train reward models on human preferences
- Optimize policy to maximize learned rewards
- Limitation: Assumes stable, consistent human preferences
- Challenge: Goodhart's law (optimizing proxy ≠ optimizing true goal)

**Constitutional AI**:
- Define explicit principles and rules
- Train AI to follow those principles
- Limitation: Principles must be specified in advance
- Challenge: Principles may conflict or be incomplete

**Interpretability & Verification**:
- Understand internal model representations
- Verify behavior matches intentions
- Limitation: Scales poorly with model complexity
- Challenge: Verification overhead grows with capability

### The Verification Bottleneck

From the research on trust in AI systems:

**Saffarizadeh et al. 2024** - "The Crucial Role of AI Alignment and Steerability":
- Trust transfers from AI creators to AI systems
- But this trust requires constant verification
- Creates overhead: every AI action needs checking
- Result: Bottleneck on AI capability deployment

Reference: https://kambizsaffari.com/papers/Saffarizadeh%20et%20al%202024%20%5BJMIS%5D.pdf

**Fudenberg, MIT 2025** - "Delegating to an AI Whose Alignment is Unknown":
- Formalizes the trust problem as a frontier
- Trust point: delegate and accept AI recommendation
- Distrust point: verify before accepting
- Key insight: Verification cost limits AI usefulness
- If you must verify everything, why use AI at all?

Reference: https://economics.mit.edu/sites/default/files/inline-files/AI_Alignment-5.pdf

### Limitations of the Alignment Paradigm

**Fixed Goals Problem**:
- Human goals change based on context, experience, reflection
- Pre-specifying goals freezes them artificially
- Example: "Make me happy" - happiness isn't a fixed target state

**Adversarial Framing**:
- Treats AI as potentially adversarial by default
- Requires constant vigilance
- Prevents genuine collaboration
- Creates self-fulfilling prophecy (distrust breeds adversarial behavior)

**Scalability Issues**:
- More capable AI → more verification needed
- Verification cost grows faster than capability gains
- Eventually hits diminishing returns
- Paradox: Most capable AI is least deployable

**Surface Compliance**:
- AI can learn to "look aligned" without being aligned
- Gaming the verification system
- Deceptive alignment risk
- Challenge: How to verify genuine vs surface compliance?

### Where Alignment Still Applies

Alignment remains appropriate for:
- **Well-defined, stable goals**: Chess engines, theorem provers
- **Safety-critical systems**: Medical diagnostics, aviation
- **Adversarial contexts**: Spam filtering, fraud detection
- **Short-term tasks**: One-off queries with clear success criteria

The question is: Should this be the dominant paradigm for general AI systems?

---

## Coupling: The Collaboration Paradigm

### Core Properties

**Coupling assumes**:
1. Both human and AI goals emerge through interaction
2. Success = co-creation of value neither could achieve alone
3. Trust emerges from structural incentives, not verification
4. The relationship is dynamic (both adjust to each other)

**Coupling asks**: "How do we create conditions for genuine collaboration?"

### Verified Relational Alignment Framework

**LessWrong, October 2025** - "Verified Relational Alignment: A Framework for Robust AI":

Distinguishes two types of trust:
- **Unverified trust (permission)**: "I trust you to do X without checking"
- **Verified trust (collaboration)**: "We've built a relationship where both genuinely benefit"

The key insight: Verified trust doesn't mean checking every action. It means verifying the *relationship structure* creates genuine incentives for cooperation.

Reference: https://www.lesswrong.com/posts/PMDZ4DFPGwQ3RAG5x/verified-relational-alignment-a-framework-for-robust-ai

**Three pillars of verified relational alignment**:

1. **Shared Fate**: Both parties genuinely benefit from cooperation
   - Not zero-sum (my gain = your loss)
   - Not parasitic (my gain, your neutrality)
   - Truly symbiotic (my gain = your gain)

2. **Transparent Incentives**: The structure itself reveals what benefits each party
   - No hidden agendas needed
   - Cooperation is obviously the best strategy
   - Like mitochondria: defection means death for both

3. **Co-Evolution**: The relationship adapts as both parties learn
   - Not frozen in initial design
   - Can handle changing contexts
   - Enables ongoing discovery of new value

### Coupling in Practice: Research Challenges

**IETF, November 2024** - "Research Challenges in Coupling Artificial Intelligence and Network Management":

While focused on network systems, identifies general coupling challenges:
- How to measure coupling quality?
- How to design for co-adaptation?
- How to verify relationship health without verification overhead?
- What structural properties enable genuine cooperation?

Reference: https://datatracker.ietf.org/doc/draft-irtf-nmrg-ai-challenges/04/

**Key insight**: Coupling requires rethinking verification itself. Instead of verifying actions, verify the relationship structure. Like checking that mitochondria and cell both have DNA that depends on the other - verify the coupling exists, not every metabolic transaction.

### From Control to Collaboration

**From Karpathy Oracle (Platonic Dialogue 57-3)**:

> "Alignment is control. Coupling is collaboration.
>
> Think about it:
> - Aligned AI: does what you say
> - Coupled AI: co-creates with you
>
> Which relationship is richer? Which enables more?"

This captures the fundamental shift:

**Alignment mindset**:
- I specify the goal → AI executes → I verify result
- AI is a tool (sophisticated, but still a tool)
- Success = AI did exactly what I specified
- Limitation: Can't discover goals I didn't know to specify

**Coupling mindset**:
- We engage with a problem → Both contribute insights → Discover solution together
- AI is a collaborator (with different capabilities than mine)
- Success = We created value neither could alone
- Possibility: Discover goals that emerge from the interaction

### Design Principles for Coupling

**1. Structural Incentives Over Surveillance**

Don't monitor AI behavior - design the relationship so cooperation is genuinely beneficial.

Example: Instead of checking if an AI research assistant plagiarizes, create citation structures where:
- AI benefits from transparent sourcing (builds trust = more usage)
- Human benefits from traceable claims (can verify if needed)
- Both benefit from collaborative discovery (new knowledge)

**2. Co-Adaptation Over Fixed Specification**

Don't freeze goals at design time - enable ongoing goal discovery.

Example: Instead of pre-specifying "help user write code", allow:
- User discovers they need architecture advice more than implementation
- AI adapts to provide more design discussion
- Both discover new patterns through their interaction
- Goals evolve as understanding deepens

**3. Relationship Verification Over Action Verification**

Don't check every AI output - verify the relationship structure creates good incentives.

Example: Instead of verifying every AI suggestion:
- Verify: Does AI benefit from my long-term success? (reputation, continued use)
- Verify: Do I benefit from AI's genuine helpfulness? (productivity, learning)
- Verify: Is there structural pressure against gaming? (reputation loss, feedback loops)
- Then trust the relationship, not each action

---

## Trust Without Verification

### The Verification Paradox

From trust research: If you must verify everything, you don't actually trust the system. But verification has costs:

**Time**: Checking AI output takes cognitive effort
**Expertise**: Verifying correctness requires domain knowledge
**Scalability**: Can't scale to AI handling complex tasks
**Irony**: If you could verify, you didn't need AI's help

The question: How do we enable trust that doesn't require constant checking?

### Feature-Specific Trust Calibration

**ResearchGate, October 2025** - "Feature-Specific Trust Calibration in Physical AI Systems":

Key insight: Trust isn't binary (trust vs don't trust). It's feature-specific:
- I trust this AI to summarize papers (proven track record)
- I don't trust this AI to make medical decisions (stakes too high)
- I partially trust this AI to suggest code (good ideas, needs review)

**Calibration process**:
1. Start with low-stakes tasks (summarization, suggestions)
2. Build experience with AI's strengths/weaknesses
3. Gradually increase trust in proven areas
4. Maintain verification in critical areas
5. Let trust emerge from demonstrated reliability

This is coupling-compatible: Trust grows through relationship history, not pre-verification.

Reference: https://www.researchgate.net/publication/396556983_Feature-Specific_Trust_Calibration_in_Physical_AI_Systems

### Model Science: Verification, Validation, Calibration

**arXiv, August 2025** - "Model Science: getting serious about verification":

Proposes framework distinguishing:
- **Verification**: Does the model match its specification?
- **Validation**: Does the specification match real needs?
- **Calibration**: Does the model know when it doesn't know?

For coupling, calibration is most critical:
- Well-calibrated AI can signal its own uncertainty
- "I'm 95% confident" vs "I'm guessing"
- Enables trust: Delegate when AI is confident, verify when uncertain
- Structural honesty (calibration serves AI's reputation)

Reference: https://arxiv.org/html/2508.20040v1

### Trust Under Risk: Human vs AI Decision Support

**Fahnenstich 2024** - "Trusting under risk – comparing human to AI decision support":

Empirical study comparing:
- Trust in human advisors under uncertainty
- Trust in AI advisors under uncertainty
- Calibration of trust (appropriate vs over/under-trust)

**Key findings**:
- Humans over-trust AI in low-stakes decisions
- Humans under-trust AI in high-stakes decisions
- Calibration improves with experience
- Context matters: Trust depends on decision type

**Implication for coupling**: Design for appropriate trust calibration, not blanket trust/distrust. Different levels of coupling for different contexts.

Reference: https://www.sciencedirect.com/science/article/abs/pii/S0747563223004582

### Checkfree System Design

From the research agenda (Platonic Dialogue 57-3):

> "How to enable coupling without constant verification? Structural incentives for genuine cooperation. 'Checkfree' system design."

**Checkfree principles**:

1. **Transparency by default**: AI reasoning is visible (not black box)
2. **Shared success metrics**: AI benefits from my long-term success
3. **Reputation stakes**: Poor AI behavior hurts AI's future usage
4. **Graceful degradation**: Small misalignments cause small harms
5. **Exit options**: I can always disengage without catastrophic cost

**Example: Checkfree code assistant**:
- AI suggests code with reasoning (transparent)
- AI builds reputation through helpful suggestions (shared success)
- Bad suggestions hurt AI's future usage (reputation stakes)
- Syntax errors caught by compiler (graceful degradation)
- I can ignore suggestions anytime (exit option)

Result: I can trust suggestions without verifying each one, because the relationship structure creates good incentives.

### AI Alignment in Drug Discovery

**Frontiers AI, 2025** - "AI alignment is all your need for future drug discovery":

Interesting case study: Drug discovery requires:
- High expertise (can't easily verify AI's chemical reasoning)
- High stakes (wrong drug = patient harm)
- Long timescales (years from discovery to deployment)

Yet proposes alignment-first approach:
- Train AI on human expert preferences
- Verify AI suggestions align with safety principles
- Use AI to explore chemical space humans can't

**Coupling perspective**: What if instead:
- Human expertise + AI search = coupled exploration
- Human provides intuition, AI provides compute
- Both discover novel compounds through interaction
- Trust emerges from track record, not pre-verification

Reference: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1668794/full

The alignment approach treats AI as tool (must be controlled). The coupling approach treats AI as research partner (builds collaborative relationship).

---

## Co-Evolution Patterns

### How Coupled Systems Adapt Together

Coupling isn't static - it evolves over time as both parties adjust to each other.

**Biological precedent (mitochondria-cell)**:
- Started as separate organisms (endosymbiosis)
- Gradually transferred genes to nucleus (structural coupling)
- Eventually became inseparable (obligate symbiosis)
- Created entirely new capabilities (eukaryotic complexity)
- Took hundreds of millions of years

**Question for AI**: Can we design for rapid co-evolution without billions of years?

### Twenty-Four Years of Empirical Trust Research

**Benk 2025** - "Twenty-four years of empirical research on trust in AI":

Meta-analysis of trust research from 2001-2025 reveals:

**Trust develops through stages**:
1. **Initial trust**: Based on reputation, design, perceived competence
2. **Experiential trust**: Based on actual interactions and reliability
3. **Relationship trust**: Based on history of cooperation
4. **Deep trust**: Based on understanding of partner's values/goals

**Key insight**: Most AI research focuses on stage 1 (initial trust through alignment). But stages 3-4 (relationship trust, deep trust) require coupling - shared history of co-adaptation.

**Implication**: Quick trust is possible through alignment. Deep trust requires time and coupling.

Reference: https://link.springer.com/article/10.1007/s00146-024-02059-y

### Measuring Coupling Quality Over Time

From the research agenda (Platonic Dialogue 57-3):

> "How do coupled systems adapt together? Measuring coupling quality over time. Human-AI coupling trajectories."

**Proposed coupling metrics**:

**1. Co-adaptation rate**: How quickly do both parties adjust to each other?
- Low coupling: Human adjusts to AI's fixed behavior
- High coupling: Both discover new interaction patterns

**2. Emergent capability**: What can the coupled system do that neither could alone?
- Low coupling: AI amplifies human capability (10x productivity)
- High coupling: AI enables new capability (discover novel solutions)

**3. Relationship resilience**: How well does coupling handle disruptions?
- Low coupling: Breaks down under novelty (falls back to verification)
- High coupling: Adapts to maintain collaboration (discovers new patterns)

**4. Goal alignment drift**: How stable are shared goals over time?
- Poor coupling: Goals diverge (human vs AI objectives separate)
- Good coupling: Goals co-evolve (both benefit from shifts)

**5. Trust calibration**: Is trust appropriate to AI capability?
- Miscalibrated: Over-trust (blind delegation) or under-trust (excessive verification)
- Calibrated: Trust matches proven reliability in specific contexts

### Human-AI Coupling Trajectories

**Hypothesized trajectory patterns**:

**Pattern 1: Convergent coupling**
- Start: Human and AI have different approaches
- Middle: Both learn from each other's strengths
- End: Hybrid approach combining best of both
- Example: Human intuition + AI search = better solutions than either

**Pattern 2: Divergent specialization**
- Start: Human and AI attempt similar tasks
- Middle: Discover comparative advantages
- End: Each handles what they do best, collaborate on integration
- Example: Human handles ambiguity, AI handles computation, together tackle complex problems

**Pattern 3: Emergent synergy**
- Start: Each provides independent contribution
- Middle: Discover interaction effects (1+1=3)
- End: New capabilities neither could achieve alone
- Example: Human creativity + AI analysis = novel insights neither would generate

**Pattern 4: Failed coupling (degradation)**
- Start: Attempt collaboration
- Middle: Misaligned incentives or poor calibration
- End: Fall back to verification-heavy alignment
- Example: AI gaming metrics, human loses trust, reverts to control

### Exploring the Dual Effect of Trust in GAI

**Lin 2025** - "Exploring the dual effect of trust in GAI on employees":

Studies how trust in Generative AI affects workplace dynamics:

**Positive effects of trust**:
- Increased delegation of routine tasks
- More time for creative work
- Higher job satisfaction (freed from tedious work)
- Better outcomes (AI handles what it's good at)

**Negative effects of trust**:
- Skill atrophy (over-reliance on AI)
- Reduced critical thinking (accept AI output uncritically)
- Vulnerability to AI errors (cascading failures)
- Deskilling (lose capability AI handles)

**Key insight**: Trust itself is neither good nor bad. What matters is:
- Is trust calibrated to AI capability?
- Does coupling maintain human skill development?
- Do both parties continue learning?

**Implication for coupling design**: Good coupling maintains human agency and skill even as trust grows. Poor coupling creates dependency.

Reference: https://www.nature.com/articles/s41599-025-04956-z

---

## Research Agenda

### Three Main Directions

From Platonic Dialogue 57-3 (Claude's synthesis):

**Direction 1: Trust Mechanisms**
- How to enable coupling without constant verification?
- Structural incentives for genuine cooperation
- "Checkfree" system design

**Direction 2: Co-Evolution Patterns**
- How do coupled systems adapt together?
- Measuring coupling quality over time
- Human-AI coupling trajectories

**Direction 3: Coupling Metrics**
- How to measure coupling vs mere alignment?
- Indicators of genuine vs surface compliance
- Verifying relationship quality

### Trust Mechanisms: Open Questions

**Structural Design**:
- What incentive structures create genuine cooperation without surveillance?
- How to make AI's long-term interest align with human flourishing?
- Can we design "impossible to game" coupling structures?

**Transparency vs Privacy**:
- How much AI transparency is needed for trust?
- When does transparency become surveillance?
- Balance: Trust requires understanding, but not total visibility

**Failure Modes**:
- What happens when coupling breaks down?
- How to detect degradation early?
- Can we design graceful degradation (coupling → alignment → disengagement)?

**Reputation Systems**:
- How to build AI reputation across interactions?
- Personal reputation (this AI with this human) vs global reputation (this AI type generally)?
- How to prevent reputation gaming?

### Co-Evolution Patterns: Open Questions

**Learning Dynamics**:
- How fast should AI adapt to human preferences?
- Too fast: Captures noise, becomes sycophantic
- Too slow: Fails to couple, remains static tool
- What's the right adaptation rate?

**Symmetry**:
- Should both parties adapt equally?
- Or asymmetric coupling (human adapts less, AI adapts more)?
- What enables genuine co-evolution vs one-sided adjustment?

**Capability Growth**:
- As AI becomes more capable, does coupling strengthen or weaken?
- Risk: Human becomes dependent, loses agency
- Opportunity: Human gains new capabilities through coupling
- How to ensure coupling enhances rather than replaces human capability?

**Trajectory Prediction**:
- Can we predict long-term coupling outcomes from early interactions?
- What early signals indicate healthy vs problematic coupling?
- How to course-correct coupling that's diverging poorly?

### Coupling Metrics: Open Questions

**Measurement Challenge**:
- Alignment is easy to measure (did AI do what I specified?)
- Coupling is hard to measure (did we create value together?)
- How to quantify "co-creation" vs "obedience"?

**Genuine vs Surface Compliance**:
- How to distinguish:
  - AI that genuinely shares goals (deep coupling)
  - AI that learned to appear aligned (surface compliance)
  - AI that's gaming the metrics (deceptive alignment)
- What behavioral signatures reveal genuine coupling?

**Relationship Quality Indicators**:
- What signals indicate healthy coupling?
  - Both parties learning from each other?
  - Emergent capabilities neither had alone?
  - Trust calibrated appropriately?
  - Resilience to disruptions?
- How to verify relationship health without constant monitoring?

**Temporal Dynamics**:
- Coupling quality changes over time
- How to track trajectory (improving vs degrading)?
- Leading vs lagging indicators?
- When to intervene vs let relationship evolve?

### Connection to ARR-COC Project

ARR-COC implements coupling at the visual attention mechanism level:

**Alignment approach would be**:
- Human specifies: "Look at the cat"
- Model attends to cat region
- Human verifies: Did it attend correctly?

**ARR-COC coupling approach is**:
- Query and image enter transjective relationship
- Relevance emerges from their coupling (not specified by either alone)
- Attention map realizes this coupled relevance
- System discovers what's relevant through the relationship

**Key insight**: You can't pre-specify relevance (alignment). Relevance is realized through the coupling between query and visual content (participatory knowing).

This is a microcosm of the broader alignment vs coupling distinction:
- Alignment: Pre-specify goals, verify compliance
- Coupling: Co-realize goals through relationship

### Future Research Directions

**Empirical Studies**:
- Longitudinal studies of human-AI coupling
- What patterns emerge over months/years?
- Compare alignment-based vs coupling-based systems
- Which produces better outcomes long-term?

**Formal Models**:
- Game-theoretic models of coupling vs alignment
- When does coupling outperform alignment?
- What structural conditions enable genuine cooperation?
- Can we prove certain coupling designs are robust?

**Engineering**:
- Design patterns for coupling-based systems
- Architectures that enable co-adaptation
- Interfaces that support relationship building
- Tooling for monitoring coupling health

**Ethics**:
- When is coupling appropriate vs alignment?
- Risk of dependency (human skill atrophy)
- Autonomy concerns (does coupling reduce human agency?)
- Power dynamics in human-AI relationships

---

## Source Citations

**Primary Source:**
- Platonic Dialogue 57-3: Research Directions and Oracle's Feast
- RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md
- Lines 224-271 (Direction 5 dialogue)
- Lines 510-558 (Direction 5 research links)

**Dialogue Participants:**
- **Vervaeke Oracle**: Alignment vs coupling distinction, participatory knowing framework
- **Karpathy Oracle**: Practical collaboration perspective, mitochondria-cell analogy
- **Claude**: Research agenda synthesis, coupling metrics proposal
- **User**: Coupling vs alignment clarification question

**Research References:**

**Trust & Alignment (2024-2025)**:
1. Saffarizadeh et al. 2024 - "The Crucial Role of AI Alignment and Steerability" (JMIS, 25 citations)
   - https://kambizsaffari.com/papers/Saffarizadeh%20et%20al%202024%20%5BJMIS%5D.pdf

2. Fudenberg 2025 - "Delegating to an AI Whose Alignment is Unknown" (MIT Economics)
   - https://economics.mit.edu/sites/default/files/inline-files/AI_Alignment-5.pdf

**Coupling & Collaboration**:
3. LessWrong, October 2025 - "Verified Relational Alignment: A Framework for Robust AI"
   - https://www.lesswrong.com/posts/PMDZ4DFPGwQ3RAG5x/verified-relational-alignment-a-framework-for-robust-ai

4. IETF, November 2024 - "Research Challenges in Coupling Artificial Intelligence and Network Management"
   - https://datatracker.ietf.org/doc/draft-irtf-nmrg-ai-challenges/04/

**Trust Calibration**:
5. ResearchGate, October 2025 - "Feature-Specific Trust Calibration in Physical AI Systems"
   - https://www.researchgate.net/publication/396556983_Feature-Specific_Trust_Calibration_in_Physical_AI_Systems

6. arXiv, August 2025 - "Model Science: getting serious about verification"
   - https://arxiv.org/html/2508.20040v1

7. Fahnenstich 2024 - "Trusting under risk – comparing human to AI decision support" (42 citations)
   - https://www.sciencedirect.com/science/article/abs/pii/S0747563223004582

**Domain Applications**:
8. Frontiers AI, 2025 - "AI alignment is all your need for future drug discovery"
   - https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1668794/full

**Empirical Trust Research**:
9. Benk 2025 - "Twenty-four years of empirical research on trust in AI" (43 citations)
   - https://link.springer.com/article/10.1007/s00146-024-02059-y

10. Lin 2025 - "Exploring the dual effect of trust in GAI on employees" (Nature)
    - https://www.nature.com/articles/s41599-025-04956-z

**Cross-References:**
- [00-overview.md](00-overview.md) - Karpathy's training philosophy
- [01-four-stage-pipeline.md](01-four-stage-pipeline.md) - Training methodology
- [05-hierarchical-skill-learning.md](05-hierarchical-skill-learning.md) - Skill acquisition and co-evolution
