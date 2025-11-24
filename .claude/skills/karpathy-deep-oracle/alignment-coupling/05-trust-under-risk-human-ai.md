# Trust Under Risk: Human vs AI Decision Support

## Overview: Trust Under Uncertainty

Trust is not a static property - it's a dynamic process that must adapt to uncertainty and risk. When humans work alongside decision support agents (whether human or AI), the stakes matter profoundly. A recommendation that costs nothing if wrong faces different trust dynamics than one that could harm lives or destroy value.

This fundamental insight - that **risk shapes trust behavior** - has only recently begun receiving serious empirical attention. Most trust research treats trust as context-independent: either you trust a system or you don't. But real-world deployment demands understanding how trust operates **under pressure**, when consequences loom large.

The critical question: Do humans trust AI decision support differently than human advisors when stakes are high? And if so, what does this mean for designing vision-language models that must earn trust in safety-critical domains?

### Key Findings from Recent Research

Fahnenstich et al. (2024) conducted a controlled study comparing human trust behavior toward human vs AI decision support agents under varying risk levels. Their findings challenge common assumptions:

**Surprising Result**: Under high risk, participants exhibited **increased trust behavior** toward AI support agents, while trust in human advisors remained unchanged.

**Responsibility Attribution**: Participants perceived human support agents as more responsible for negative outcomes than AI agents, regardless of risk level.

**Trust Attitude vs Trust Behavior**: While self-reported trust attitudes showed no difference between human and AI agents, actual behavioral reliance patterns diverged significantly under risk.

This disconnect between what people say they trust and how they actually behave reveals a crucial design challenge: building AI systems that deserve the trust they receive, especially when stakes are highest.

### Why This Matters for VLM Development

Vision-language models increasingly support high-stakes decisions:
- Medical diagnosis assistance (radiological interpretation)
- Autonomous vehicle perception (pedestrian detection)
- Industrial inspection (defect identification in safety-critical components)
- Security systems (threat assessment in surveillance)

In these domains, **coupling quality under risk** determines whether human-AI collaboration enhances or undermines safety. If humans overtrust AI recommendations when consequences are severe, catastrophic failures become inevitable. If they undertrust, the AI provides no value.

The alignment vs coupling distinction becomes critical: aligned systems maintain static trust relationships, while coupled systems enable **dynamic trust calibration** that adapts to risk context.

### Connection to ARR-COC Framework

ARR-COC's relevance realization approach offers a pathway to trust-worthy VLM behavior:

**Risk-Aware Relevance**: The system must recognize when uncertainty is high and stakes are critical, allocating cognitive resources accordingly.

**Uncertainty Signaling**: Rather than hiding confidence intervals, surface them through visual attention patterns that communicate "I'm unsure about this region."

**Participatory Knowing Under Risk**: Query-content coupling must incorporate risk assessment - when the query implies high-stakes decision support, relevance patterns should shift toward conservative, verification-friendly compression.

This isn't just about calibrated confidence scores - it's about **structural coupling** between human risk perception and AI relevance realization.

---

## Human-AI Trust Comparison: The Fahnenstich Study

### Study Design

Fahnenstich, Rieger, and Roesler (2024) examined trust dynamics through a visual estimation task where participants received support from either a human or AI agent. The experimental design manipulated two key variables:

**Support Agent Type**:
- Human advisor (presented as another participant)
- AI advisor (explicitly identified as algorithmic decision support)

**Risk Level**:
- Low risk: Incorrect estimates result in minor inconvenience
- High risk: Incorrect estimates result in serious negative consequences for others

**Dependent Measures**:
1. **Trust Attitude**: Self-reported trust in the advisor (questionnaire)
2. **Trust Behavior**: Actual reliance on advisor's recommendations (decision choices)
3. **Perceived Responsibility**: Attribution of accountability for outcomes

### Core Findings

#### Finding 1: Risk Increases AI Trust Behavior (Not Human Trust)

**Behavioral Pattern**:
- Participants with AI support: Significantly increased reliance under high risk
- Participants with human support: No change in reliance across risk levels

This asymmetry suggests fundamentally different trust mechanisms operating for human vs AI advisors.

**Possible Explanations**:

**Self-Confidence vs Trust Trade-off**: Under high risk, participants may have experienced heightened self-doubt, leading them to defer more strongly to the AI's "objective" calculation. With human advisors, the trade-off becomes less clear - the other person might be equally uncertain, so self-confidence remains a viable alternative.

**Responsibility Diffusion**: When consequences are severe, participants may unconsciously shift responsibility to the AI agent. Since AI is perceived as less responsible for outcomes (see Finding 3), increased reliance becomes psychologically easier.

**Perceived Consistency Under Pressure**: Humans are known to perform worse under stress. AI systems, by contrast, maintain consistent performance regardless of stakes. Under high risk, this consistency becomes more valuable, driving increased reliance.

**Algorithm Appreciation vs Aversion**: While "algorithm aversion" (reduced trust after observing AI errors) is well-documented, "algorithm appreciation" may emerge under risk. If the AI hasn't yet made an error, its computational nature becomes an asset rather than a limitation.

#### Finding 2: Trust Attitude Remains Unchanged

Despite the behavioral shift, self-reported trust attitudes showed:
- No difference between human and AI support agents
- No difference across risk levels

**Interpretation**: Participants' conscious assessment of trust didn't shift, but their actual reliance behavior did. This highlights the distinction between:

**Explicit Trust**: What people consciously report believing
**Implicit Trust**: How people actually behave when making decisions

For system design, this means:
- User satisfaction surveys may not capture real trust dynamics
- Behavioral metrics (reliance rates, verification frequency, override patterns) provide ground truth
- Trust calibration must target behavior, not just attitudes

#### Finding 3: Humans Seen as More Responsible Than AI

Across all conditions, participants rated human support agents as significantly more responsible for negative outcomes than AI agents.

**Risk Level**: Did not influence perceived responsibility (both low and high risk showed same pattern)

**Implications**:

**Accountability Gap**: When AI makes mistakes in high-stakes domains, who bears responsibility? If users don't perceive the AI as accountable, this creates moral hazard - the system can fail without clear attribution.

**Design Challenge**: Coupling requires **shared accountability**. If the AI is viewed as a tool rather than a collaborator, true co-evolution becomes impossible.

**Regulatory Tension**: Legal frameworks increasingly demand AI accountability, but human psychology resists attributing responsibility to non-human agents.

### Theoretical Integration

#### Why Increased AI Trust Under Risk?

The Fahnenstich findings suggest a **risk-dependent trust asymmetry** that previous research missed:

**Low Stakes**: Human and AI advisors receive similar trust (behavioral parity)
**High Stakes**: AI advisors receive increased trust (behavioral divergence)

This pattern contradicts the "algorithm aversion" literature, which predicts humans will avoid AI after observing errors, especially in consequential domains.

**Resolution**: Algorithm aversion may be a **post-error phenomenon**, while the Fahnenstich study examined trust before errors occurred. The full picture likely involves:

1. **Pre-error phase**: Algorithm appreciation under risk (Fahnenstich)
2. **Post-error phase**: Algorithm aversion after mistakes (prior literature)
3. **Recovery phase**: Trust recalibration based on explanation quality

This suggests **trust dynamics follow a lifecycle** rather than static levels:

```
High Stakes + No Prior Errors → Increased AI Reliance (algorithm appreciation)
                ↓
        AI Makes Mistake
                ↓
High Stakes + Recent Error → Decreased AI Reliance (algorithm aversion)
                ↓
        AI Provides Explanation + Subsequent Accuracy
                ↓
High Stakes + Calibrated Experience → Appropriate Reliance (coupling)
```

#### The Role of Uncertainty Communication

Neither human nor AI advisors in the Fahnenstich study communicated uncertainty explicitly. Both provided point estimates without confidence intervals.

**Hypothesis**: If AI advisors surfaced uncertainty (e.g., "My confidence is 60% on this estimate"), would the risk-dependent trust asymmetry persist?

**Prediction**: Uncertainty-aware AI might maintain trust advantage under risk while avoiding over-reliance. Humans naturally communicate hedging ("I think it's around 500, but could be 450-550"). Algorithmic point estimates create false precision that invites over-trust.

**VLM Design Implication**: Visual attention patterns in ARR-COC could serve as uncertainty signals. Rather than producing uniform-confidence outputs, the system could communicate "I spent 400 tokens on this region because it's critical and I'm uncertain" vs "I spent 64 tokens here because it's clearly background."

### Comparison to Related Work

#### Algorithm Aversion (Dietvorst et al. 2015)

**Finding**: People avoid algorithmic forecasts after seeing the algorithm err, even when it outperforms human forecasters.

**Contrast with Fahnenstich**: Algorithm aversion requires **observed errors**. The Fahnenstich study measured trust before errors occurred, finding algorithm appreciation instead.

**Reconciliation**: Trust in AI follows a **U-shaped curve**:
1. High initial trust (no errors observed)
2. Sharp drop after first error (algorithm aversion)
3. Gradual recovery with calibrated experience

The Fahnenstich study captured phase 1, while algorithm aversion research examines phase 2.

#### Trust in Automation (Hoff & Bashir 2015)

**Framework**: Trust operates at three levels:
- Dispositional (personality-based)
- Situational (context-specific)
- Learned (experience-based)

**Fahnenstich Contribution**: Identifies **risk as a situational moderator** that interacts with agent type. Previous research treated risk as affecting trust uniformly, but the human vs AI asymmetry reveals more nuanced dynamics.

#### Reliance on Decision Aids (van Dongen & van Maanen 2013)

**Framework**: Reliance depends on:
- Task difficulty
- Advisor reliability
- User confidence
- Cost of errors

**Fahnenstich Addition**: Agent type (human vs AI) modulates how cost of errors affects reliance. High error cost increases AI reliance but not human reliance.

### Open Questions

**1. Would Results Generalize to Visual Tasks?**

The Fahnenstich study used numerical estimation. VLM applications involve spatial reasoning, object recognition, scene understanding - fundamentally different cognitive processes.

**Hypothesis**: Visual domain expertise might moderate the effect. Radiologists may trust AI chest X-ray analysis differently than novices, especially under high stakes.

**2. How Does Explanation Quality Interact with Risk?**

The study provided no explanations for advisor recommendations. Would XAI (explainable AI) methods change trust dynamics?

**Prediction**: Uncertainty-aware explanations might reduce over-reliance under risk, while confidence-boosting explanations might exacerbate it.

**3. Does Trust Asymmetry Persist After Errors?**

The study measured trust before errors. What happens after the AI makes a mistake in a high-stakes context?

**Prediction**: Risk-dependent trust asymmetry reverses - humans become more forgiving of human errors than AI errors when consequences are severe.

**4. How Long-Term Interaction Patterns Evolve?**

Single-session studies miss **co-evolution dynamics**. Over weeks or months, do humans develop different trust calibration with AI vs human collaborators?

**Coupling Hypothesis**: True coupling requires extended interaction where both parties adapt. Short-term studies capture alignment dynamics (static trust) but miss coupling potential (adaptive trust).

---

## Coupling Under Risk: Dynamic Trust Adjustment

### From Static Trust to Adaptive Coupling

The Fahnenstich findings reveal a fundamental limitation of current trust models: they treat trust as a **static relationship** that changes only through discrete events (errors, successes, explanations).

Coupling offers an alternative: **trust as continuous co-adjustment** where both human and AI adapt their behavior based on ongoing interaction, with risk context modulating the adaptation rate.

#### Static Trust (Alignment Paradigm)

**Characteristics**:
- Fixed reliability levels
- Human adjusts reliance based on observed performance
- AI maintains consistent behavior regardless of human trust state
- Trust calibration is **unidirectional**: human → AI

**Risk Response**: Humans increase or decrease reliance based on perceived stakes, but AI remains unchanged.

**Fahnenstich Example**: Participants increased reliance on AI under high risk, but the AI provided identical support regardless of risk level. This creates **asymmetric adaptation** - only the human adjusts.

#### Dynamic Trust (Coupling Paradigm)

**Characteristics**:
- Adaptive reliability (AI modulates confidence based on task difficulty)
- Both human and AI adjust behavior based on partner state
- Trust calibration is **bidirectional**: human ↔ AI
- Risk awareness is **shared**: both parties recognize stakes

**Risk Response**: AI modulates output precision, uncertainty signaling, and verification affordances based on detected risk level. Human adjusts reliance based on both AI behavior and risk context.

**ARR-COC Implementation**: Relevance realization operates differently in high-stakes queries:
- Allocate more tokens to critical regions (increased scrutiny)
- Surface uncertainty through attention pattern heterogeneity
- Avoid overconfident compression when stakes are high

### Uncertainty-Aware Relevance Realization

#### The Core Problem

Standard VLMs produce **uniform confidence** outputs: every generated token receives similar certainty, regardless of whether the model is making a clear determination or guessing.

This creates **false precision** that invites over-reliance, especially under risk. When a VLM says "The mass in the left lung appears benign," it provides no signal about whether this judgment rests on clear visual features or ambiguous patterns.

#### ARR-COC Solution: Relevance Patterns as Uncertainty Signals

**Key Insight**: The cognitive work required to establish relevance correlates with uncertainty.

**Clear Cases**: Low uncertainty regions require minimal tokens (64) - relevance is easily realized
**Ambiguous Cases**: High uncertainty regions require maximal tokens (400) - relevance is hard to realize

By surfacing this allocation pattern, the system communicates **"I had to work hard to understand this region, which means uncertainty is high."**

**Implementation**:

```python
# Standard VLM output (no uncertainty signal)
output = {
    "description": "Chest X-ray shows left lung opacity",
    "confidence": 0.87  # Opaque scalar
}

# ARR-COC output (relevance pattern as uncertainty signal)
output = {
    "description": "Chest X-ray shows left lung opacity",
    "relevance_map": {
        "left_lung_region": {
            "tokens_allocated": 380,  # High allocation → high uncertainty
            "compression_ratio": 2.1,  # Low compression → difficult to summarize
            "information_score": 0.92,  # High information content
            "perspectival_score": 0.45, # Low salience → not visually obvious
            "participatory_score": 0.88  # High query relevance
        }
    },
    "interpretation": "This region demanded substantial cognitive resources despite not being visually salient - indicates uncertain but critical finding."
}
```

The user doesn't just see a confidence score - they see **the cognitive work the system performed**, which provides calibrated trust affordances.

#### Risk-Modulated Compression

**Standard Approach**: Uniform compression strategy across all queries

**ARR-COC Coupling**: Compression aggressiveness adapts to detected risk level

**Detection Methods**:

**1. Query Analysis**:
- "Is this cancer?" → High risk detected
- "What's in this image?" → Low risk detected

**2. Domain Recognition**:
- Medical image modality → Increase scrutiny
- Casual photography → Standard compression

**3. Uncertainty Estimation**:
- High inter-patch variance → Allocate conservatively
- Low variance → Aggressive compression acceptable

**Compression Strategy Adaptation**:

```python
# Low risk / high certainty
compression_range = (64, 200)  # Allow aggressive compression
uncertainty_threshold = 0.3     # Collapse ambiguous regions

# High risk / high uncertainty
compression_range = (200, 400) # Conservative compression
uncertainty_threshold = 0.1     # Preserve ambiguous regions
```

This creates **context-aware coupling**: the system recognizes when stakes are high and adjusts its cognitive allocation accordingly.

### Verification-Friendly Architecture

#### The Transparency Problem

Black-box VLMs provide answers without revealing reasoning paths. This makes verification costly - the human must independently analyze the image to confirm the AI's judgment.

Under high risk, this cost becomes prohibitive. If verification requires equal effort to initial analysis, the AI provides no efficiency gain.

#### ARR-COC Verification Affordances

**Key Idea**: Make the AI's reasoning path **inspectable** by surfacing relevance allocation decisions.

**Verification Workflow**:

1. **VLM produces initial analysis**: "Suspicious mass detected in left lung"
2. **Human inspects relevance map**: See which regions received high token allocation
3. **Selective verification**: Focus verification effort on high-allocation regions
4. **Trust calibration**: If relevance patterns align with human judgment, trust increases; if misaligned, trust decreases

**Example**:

```
VLM: "No significant abnormalities detected"

Relevance Map:
- Left lung upper lobe: 64 tokens (minimal attention)
- Right lung: 64 tokens (minimal attention)
- Heart: 120 tokens (standard attention)
- Mediastinum: 64 tokens (minimal attention)

Human Verification:
- Quickly scan high-allocation regions (heart)
- Spot-check low-allocation regions
- NOTICE: Left lung upper lobe has subtle opacity that VLM missed
- RESULT: VLM's relevance realization failed - trust decreases
```

**Contrast**:

```
VLM: "Possible left lung opacity - uncertain"

Relevance Map:
- Left lung upper lobe: 400 tokens (maximum attention, high uncertainty)
- Right lung: 64 tokens
- Heart: 120 tokens
- Mediastinum: 64 tokens

Human Verification:
- Focus immediately on left lung (VLM flagged uncertainty)
- Deep inspection confirms subtle opacity
- VLM's relevance realization succeeded AND surfaced uncertainty
- RESULT: Trust increases (appropriate caution demonstrated)
```

The second case exemplifies **coupling under risk**: the VLM recognized uncertainty, allocated cognitive resources appropriately, and surfaced the ambiguity to invite human verification.

### Feature-Specific Trust Calibration

#### Beyond Global Trust Metrics

The Fahnenstich study measured **overall trust** in the advisor. But real-world AI systems exhibit **heterogeneous reliability** - accurate on some tasks, unreliable on others.

**Example**: A VLM might excel at object recognition but struggle with spatial relationships, or perform well on common objects but fail on rare categories.

**Implication**: Trust must be **feature-specific** rather than global. Humans should learn to trust the VLM for certain judgments while remaining skeptical for others.

#### ARR-COC's Multi-Dimensional Relevance

The three ways of knowing provide natural dimensions for feature-specific trust:

**Propositional Trust**: Reliability of information content assessments
- "This VLM accurately identifies high-entropy regions"

**Perspectival Trust**: Reliability of salience judgments
- "This VLM's attention patterns align with human radiologist focus"

**Participatory Trust**: Reliability of query-content coupling
- "This VLM correctly interprets what I'm asking about"

**Trust Calibration Strategy**:

Rather than a single trust score, humans develop **multidimensional trust profiles**:

```python
trust_profile = {
    "propositional_trust": 0.85,  # Usually correct about information density
    "perspectival_trust": 0.62,   # Sometimes misses subtle salient features
    "participatory_trust": 0.91,  # Excellent query understanding
    "overall_trust": 0.79         # Computed from dimension-specific values
}
```

This enables **selective reliance**: Trust the VLM's query interpretation (participatory) but verify its salience judgments (perspectival) more carefully.

#### Longitudinal Trust Dynamics

Feature-specific trust develops over **extended interaction**:

**Week 1**: User treats VLM as uniformly reliable/unreliable (global trust)
**Week 4**: User notices VLM excels at information density but occasionally misses salient details
**Week 12**: User develops calibrated reliance - trusts propositional knowing, verifies perspectival knowing

This represents **co-evolution**: the human's interaction strategy adapts to the VLM's reliability profile, while the VLM (if capable of learning) adapts to the human's verification patterns.

**Coupling Metric**: Trust calibration quality = alignment between actual VLM reliability and human trust behavior across dimensions.

---

## Design Implications: Building Trust-Worthy VLMs

### Core Principle: Earn Trust Through Legibility

The Fahnenstich findings reveal a dangerous pattern: humans increase reliance on AI under high risk **before errors occur**. This creates potential for catastrophic over-trust if the AI doesn't deserve the confidence placed in it.

**Design Challenge**: Build VLMs that are **trust-worthy** (deserve trust) rather than merely **trusted** (receive trust).

### Implication 1: Surface Uncertainty Structurally

**Don't**: Provide opaque confidence scores (0.87) that obscure reasoning

**Do**: Make cognitive work visible through relevance allocation patterns

**Implementation**:

```python
class UncertaintyAwareVLM:
    def __init__(self):
        self.relevance_realizer = ARRCOCRealizer()

    def analyze_with_uncertainty(self, image, query):
        # Realize relevance and track allocation
        relevance_map = self.relevance_realizer.realize(image, query)

        # Compute uncertainty from allocation patterns
        uncertainty_signals = {
            "high_token_regions": [r for r in relevance_map if r.tokens > 300],
            "low_compression_regions": [r for r in relevance_map if r.compression < 1.5],
            "high_variance_regions": [r for r in relevance_map if r.uncertainty > 0.7]
        }

        # Generate output with uncertainty context
        return {
            "analysis": self.generate_description(relevance_map),
            "uncertainty_signals": uncertainty_signals,
            "verification_guidance": self.suggest_verification_focus(uncertainty_signals)
        }
```

**User Experience**:

Instead of: "Chest X-ray shows left lung opacity (confidence: 87%)"

Provide: "Chest X-ray shows left lung opacity. This region required substantial analysis (380 tokens allocated) despite low visual salience, suggesting uncertain but potentially significant finding. Recommend verification of left upper lobe."

### Implication 2: Risk-Adaptive Behavior

**Don't**: Maintain uniform compression strategy regardless of stakes

**Do**: Detect risk context and adjust cognitive resource allocation

**Risk Detection Methods**:

```python
def detect_risk_level(query: str, image_metadata: dict) -> RiskLevel:
    """Estimate risk level from query and context."""

    # Query-based detection
    high_risk_keywords = ["cancer", "diagnosis", "critical", "safety", "threat"]
    if any(keyword in query.lower() for keyword in high_risk_keywords):
        risk_score = 0.8

    # Domain-based detection
    if image_metadata.get("modality") in ["X-ray", "CT", "MRI"]:
        risk_score = max(risk_score, 0.7)

    # Uncertainty-based escalation
    if self.estimate_uncertainty(image) > 0.6:
        risk_score = max(risk_score, 0.6)

    return RiskLevel(risk_score)
```

**Adaptive Compression**:

```python
def allocate_tokens_risk_aware(self, patch_relevance, risk_level):
    """Adjust token allocation based on risk context."""

    if risk_level.is_high():
        # Conservative compression in high-risk contexts
        return self.allocate_conservative(patch_relevance)
    else:
        # Aggressive compression acceptable in low-risk contexts
        return self.allocate_standard(patch_relevance)
```

### Implication 3: Verification Affordances

**Don't**: Produce final answers that require complete re-analysis to verify

**Do**: Structure outputs to enable **efficient selective verification**

**Verification-Friendly Output Structure**:

```python
{
    "primary_finding": "Left lung opacity detected",

    "critical_regions": [
        {
            "location": "left_upper_lobe",
            "tokens_allocated": 380,
            "uncertainty": "high",
            "verification_priority": 1,
            "visual_cues": "Subtle increased density compared to right lung"
        }
    ],

    "normal_regions": [
        {
            "location": "right_lung",
            "tokens_allocated": 64,
            "uncertainty": "low",
            "verification_priority": 3,
            "rationale": "Clear normal lung markings"
        }
    ],

    "suggested_verification_workflow": [
        "1. Inspect left upper lobe (high uncertainty, critical finding)",
        "2. Compare to prior images if available",
        "3. Spot-check right lung (low priority, high confidence)"
    ]
}
```

This structure enables **trust through transparency**: the human can verify efficiently by focusing on regions where the VLM invested heavy cognitive resources or flagged uncertainty.

### Implication 4: Shared Accountability Signals

**Challenge**: Humans perceive AI as less responsible than human advisors (Fahnenstich Finding 3)

**Goal**: Design interactions that communicate **shared responsibility** rather than tool-like delegation

**Implementation Strategies**:

**Collaborative Framing**:
- Instead of: "AI recommends X"
- Use: "Our analysis suggests X" (shared ownership)

**Commitment Signaling**:
- Surface the cognitive work invested: "I examined this region carefully using 400 tokens"
- Indicate confidence limits: "I can reliably assess common patterns, but this case is unusual"

**Verification Requests**:
- Explicitly invite verification: "I'm uncertain about this region - please verify"
- Request collaboration: "I detect ambiguity here - what's your assessment?"

**Rationale**: If the VLM frames itself as a **collaborator seeking joint truth** rather than an **oracle providing answers**, humans may develop more appropriate trust calibration and stronger accountability attribution.

### Implication 5: Coupling Metrics Over Alignment Metrics

**Standard AI Evaluation**: Measure accuracy, precision, recall on held-out test sets

**Coupling Evaluation**: Measure **joint human-AI performance** and **trust calibration quality**

**Proposed Metrics**:

**1. Trust Calibration Error**:
```python
trust_calibration_error = abs(human_reliance_rate - ai_actual_accuracy)
```
Perfect calibration: Humans rely on AI exactly as often as it's correct

**2. Verification Efficiency**:
```python
verification_efficiency = (correct_verifications / total_verifications) / baseline_efficiency
```
Higher values indicate VLM effectively guides verification to uncertain regions

**3. Co-Evolution Rate**:
```python
co_evolution_rate = rate_of_trust_calibration_improvement_over_time
```
Faster calibration indicates better coupling dynamics

**4. Risk-Appropriate Reliance**:
```python
risk_appropriate_reliance = correlation(risk_level, verification_rate)
```
Strong positive correlation indicates humans appropriately increase verification under risk

These metrics shift focus from "Is the AI accurate?" to "Does the human-AI partnership produce well-calibrated decisions?"

---

## Sources

**Primary Research:**

Fahnenstich, H., Rieger, T., & Roesler, E. (2024). Trusting under risk – comparing human to AI decision support agents. *Computers in Human Behavior*, 153, 108107. https://doi.org/10.1016/j.chb.2023.108107
- Key findings: Risk increases AI trust behavior, humans seen as more responsible than AI
- 42 citations as of 2025
- Experimental design: Visual estimation task, 104 participants, human vs AI support agents
- Main contribution: Identified risk-dependent trust asymmetry between human and AI advisors

**Related Trust Research:**

Dietvorst, B. J., Simmons, J. P., & Massey, C. (2015). Algorithm aversion: People erroneously avoid algorithms after seeing them err. *Journal of Experimental Psychology: General*, 144(1), 114-126.
- Algorithm aversion framework (post-error trust reduction)

Hoff, K. A., & Bashir, M. (2015). Trust in automation: Integrating empirical evidence on factors that influence trust. *Human Factors: The Journal of the Human Factors and Ergonomics Society*, 57(3), 407-434.
- Three-level trust framework (dispositional, situational, learned)

van Dongen, K., & van Maanen, P. P. (2013). A framework for explaining reliance on decision aids. *International Journal of Human-Computer Studies*, 71(4), 410-424.
- Decision aid reliance framework

**ARR-COC Context:**

This knowledge file synthesizes trust research for application to Adaptive Relevance Realization vision-language models. The connection to Vervaeke's framework:

- **Propositional Knowing**: Information-theoretic reliability assessment
- **Perspectival Knowing**: Salience-based trust calibration
- **Participatory Knowing**: Query-content coupling under risk
- **Procedural Knowing**: Learned trust calibration patterns

For ARR-COC implementation details, see:
- `karpathy/training-llms/06-alignment-vs-coupling.md` (alignment vs coupling distinction)
- `alignment-coupling/00-alignment-vs-coupling-fundamental-distinction.md` (theoretical foundations)
- `alignment-coupling/03-feature-specific-trust-calibration.md` (multidimensional trust)

**Access Date**: 2025-01-31
