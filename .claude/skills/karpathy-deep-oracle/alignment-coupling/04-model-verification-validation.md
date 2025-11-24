# Model Verification and Validation

## Section 1: Overview - Model Science Framework (~80 lines)

### What is Model Science?

The emergence of foundation models—large, general-purpose AI systems trained on diverse datasets—requires a paradigm shift from **Data Science** to **Model Science**. While Data Science focuses on data collection, cleaning, and exploration as the central activity, Model Science places the **trained model** at the core of analysis.

**Key Distinction:**

- **Data Science**: Data is king. Build many models around a single well-defined dataset to find the best fit.
- **Model Science**: Model is king. Use multiple datasets (training, validation, test, monitoring, out-of-domain, synthetic) to understand and verify a single model's behavior.

From [Model Science: getting serious about verification, explanation and control of AI systems](https://arxiv.org/html/2508.20040v1) (Biecek & Samek, August 2025):

> "The growing adoption of foundation models calls for a paradigm shift from Data Science to Model Science. Unlike data-centric approaches, Model Science places the trained model at the core of analysis, aiming to interact, verify, explain, and control its behavior across diverse operational contexts."

### Why Model Science Matters for AI Safety

Foundation models like LLMs and vision-language models (VLMs):
- Have tens of gigabytes of parameters
- Serve millions of users from a single base model
- Perform dozens of tasks from one foundation
- Are deployed in high-stakes domains (healthcare, legal, finance, education)

**The Problem**: Standard benchmark evaluation doesn't reveal real-world failures. Recent studies show that SOTA models exhibit severe flaws when subjected to rigorous, context-specific validation:

**Speech-to-Text Hallucinations** ([Koenecke et al., 2024](https://arxiv.org/html/2508.20040v1#bib.bibx31)):
- OpenAI's Whisper produces hallucinated sentences in ~1% of transcripts
- 38% of hallucinations contain harmful/violent language
- Worse for people with aphasia (model "fills in" pauses with fabricated statements)
- Risk: False legal evidence, medical miscommunication

**Legal Hallucinations** ([Dahl et al., 2024](https://arxiv.org/html/2508.20040v1#bib.bibx14)):
- ChatGPT-3.5, PaLM-2, LLaMA-2: 69-88% invented non-existent case law
- Specialized tools (Lexis+ AI): 17-34% hallucination rate
- Contradicts claims of "passing bar exams"

**Healthcare Bias** ([Zack et al., 2024](https://arxiv.org/html/2508.20040v1#bib.bibx64)):
- GPT-4 showed persistent gender and racial biases (Lancet Digital Health)
- In 37% of paired clinical scenarios differing only by gender/race: distinct diagnoses
- Overestimated sarcoidosis in Black patients, hepatitis B in Asian patients
- Reflects historical healthcare biases

**Code Security** ([Majdinasab et al., 2024](https://arxiv.org/html/2508.20040v1#bib.bibx39)):
- GitHub Copilot: 35% of code suggestions contain security weaknesses
- 42 distinct MITRE CWEs (Common Weakness Enumerations)
- 11 ranked in CWE Top-25 most dangerous vulnerabilities
- Includes command injection, insecure randomness

**Education Quality** ([Ngo et al., 2024](https://arxiv.org/html/2508.20040v1#bib.bibx46)):
- ChatGPT-3.5 medical questions: only 32% fully correct and explanatory
- Risk: Propagating misconceptions to medical students

**Adultification Bias** ([Castleman & Korolova, 2025](https://arxiv.org/html/2508.20040v1#bib.bibx11)):
- LLMs and text-to-image models depict Black girls as older and more sexualized than White peers
- Potential harm in educational/disciplinary systems

### Verification vs Validation vs Calibration

Model Science framework has **four pillars**:

1. **Verification**: Does it work? (This document's focus)
2. **Explanation**: How does it really work? (Understanding internal mechanisms)
3. **Control**: How should it work? (Alignment and steering)
4. **Interface**: How to interact with it? (User trust and calibration)

**Verification** answers: "Does the model perform correctly across diverse operational contexts?"

**Validation** answers: "Does the model solve the right problem for the right reasons?"

**Calibration** answers: "Does the model's confidence match its actual accuracy?"

All three are essential for trustworthy AI deployment.

---

## Section 2: Verification Framework - Model Evaluation Levels (MEL) (~180 lines)

### The Right-for-the-Wrong-Reason Problem

Before discussing verification levels, we must address a critical failure mode: **models can be accurate but unreliable**.

**Clever Hans Effect** ([Lapuschkin et al., 2019](https://arxiv.org/html/2508.20040v1#bib.bibx34)):
- Named after a horse that appeared to do arithmetic by reading trainer's cues
- Image classifier correctly recognized horses... by detecting copyright tags in horse photos
- Not learning horse features, learning coincidental correlation

**Wolf vs Husky Classifier** ([Ribeiro et al., 2016](https://arxiv.org/html/2508.20040v1#bib.bibx51)):
- LIME interpretability revealed the model wasn't detecting animals
- It was detecting **snow in the background** (huskies photographed in snow)
- Right answer (husky), wrong reason (background context)

**Unsupervised Learning Shortcuts** ([Kauffmann et al., 2025](https://arxiv.org/html/2508.20040v1#bib.bibx27)):
- Even unsupervised representation learning methods form spurious associations
- Grouping images by embedded text logos or high-frequency noise
- Accidentally aligns with task but fails when cues change

**Implication for ARR-COC**: Relevance realization must be verified not just for accuracy, but for **realizing relevance through the right mechanisms** (Propositional + Perspectival + Participatory knowing, not spurious shortcuts).

### Five Levels of Model Evaluation (MEL)

The Model Science framework proposes five levels of increasing verification rigor. Most models are only tested at Levels 0-2, but high-stakes deployment requires Levels 3-5.

#### **MEL 0: No Explicit Evaluation**

- No verification performed
- Includes **unintended use**: applying models to tasks they weren't designed for
- Common in rapid prototyping or when foundation models are repurposed

**Risk**: Unknown failure modes, undetected biases, catastrophic errors in production.

**ARR-COC Context**: Using a VLM for medical imaging without any validation of relevance realization quality.

#### **MEL 1: Training Data Evaluation**

- Model evaluated on the same data used for training
- Common in statistical modeling (linear models, R², p-values for coefficients)
- Assumes strong assumptions about data generation mechanism and sample representativeness

**Valid Use Cases**:
- Linear regression with well-understood distributions
- Statistical inference with controlled experimental design
- Exploratory analysis of model fit

**Invalid Use Cases**:
- Deep learning models (high risk of overfitting)
- Foundation models (training data distribution unknown)
- Any model deployed in dynamic environments

**Risk**: Overconfidence from overfitting. Model memorizes training data rather than learning generalizable patterns.

**ARR-COC Context**: Testing relevance realization only on the exact image-query pairs seen during training.

#### **MEL 2: Separate Test Set (Similar Distribution)**

- Random train/test split from the same dataset
- Variants: cross-validation, out-of-bag sampling (bootstrap)
- Assumes **test distribution ≈ training distribution**
- Assumes **future reality ≈ past reality**

**Valid Assumptions**:
- Static environments (product recommendations, stable user behavior)
- Short deployment windows (model retrained frequently)
- Controlled domains (lab experiments)

**Invalid Assumptions**:
- Dynamic environments (COVID pandemic, wars, economic shifts)
- Evolving user behavior (adversarial users, distribution shift)
- Long-term deployment without retraining

**Risk**: Model performs well in testing but fails when reality changes.

**ARR-COC Context**: Testing on held-out image-query pairs from the same dataset used for training. Doesn't test generalization to new visual domains or novel query types.

#### **MEL 3: Disjoint Data Evaluation**

- Training and test data are **deliberately different**
- Tests for **generalization** across meaningful dimensions

**Out-of-Time Validation** ([Elena et al., 2021](https://arxiv.org/html/2508.20040v1#bib.bibx18)):
- Train on data from time period A, test on time period B
- Used by financial institutions for credit scoring models
- Detects temporal distribution shift

**Out-of-Region Validation**:
- Train on geographic region A, test on region B
- Tests cultural, linguistic, environmental differences

**Out-of-Device Validation**:
- Train on data from Device A, test on Device B
- Tests sensor variations, hardware differences

**Out-of-Dataset Validation** ([Ho et al., 2020](https://arxiv.org/html/2508.20040v1#bib.bibx23)):
- For AutoML systems: train on Dataset A, test on Dataset B
- Tests meta-learning generalization

**Why MEL 3 is Critical**:
- Audit teams at financial institutions require this for high-impact models
- Reveals whether model learned genuine patterns vs dataset-specific artifacts
- Exposes brittleness to distribution shift

**Risk**: Even MEL 3 doesn't catch adversarial manipulation or deliberate attacks.

**ARR-COC Context**:
- **Out-of-time**: Train on 2023 visual data, test on 2025 visual data
- **Out-of-domain**: Train on natural images, test on synthetic/artistic images
- **Out-of-query-type**: Train on object detection queries, test on spatial reasoning queries
- **Out-of-visual-complexity**: Train on simple scenes, test on cluttered/occluded scenes

#### **MEL 4: Adversarial Testing (Black-Box)**

- Actively search for failure cases, even if rare in training data
- Adversary doesn't know model internals, can only query and observe outputs
- Simulates **real-world adversarial users**

**Use Cases**:

**Insurance Pricing**:
- Users manipulate mileage, number of drivers, days abroad to reduce premiums
- Model must be robust to strategic misreporting

**Hate Speech Detection**:
- Users actively test scenarios where model fails
- Adversarial prompt engineering to bypass filters
- Model must handle obfuscation, synonyms, context manipulation

**Spam Filtering**:
- Spammers probe model to find blind spots
- Model must resist adversarial text generation

**Vision Models**:
- Adversarial patches ([Lovisotto et al., 2022](https://arxiv.org/html/2508.20040v1#bib.bibx37))
- Small perturbations causing misclassification
- Tests robustness to adversarial inputs

**Techniques**:
- Red teaming ([Ganguli et al., 2022](https://arxiv.org/html/2508.20040v1#bib.bibx20))
- Adversarial prompt engineering
- Edge case generation
- Fuzzing inputs

**ARR-COC Context**:
- Adversarial queries designed to trick relevance realization
- Visual adversarial patches that exploit attention mechanisms
- Query-image pairs that trigger spurious correlations
- Testing whether relevance realization is robust to adversarial coupling

#### **MEL 5: White-Box Adversarial Testing**

- Adversary has **full access to model internals**
- Can track decision paths, compute gradients, inspect weights
- Strongest form of adversarial testing

**Use Cases**:

**Open-Source Models**:
- Llama 2 ([Touvron et al., 2023](https://arxiv.org/html/2508.20040v1#bib.bibx58)) - publicly available
- Adversaries can analyze full model architecture
- Can design targeted attacks using gradient information

**Gandalf Challenge**:
- Designed to deceive LLMs to steal secrets
- Tests undesirable behavior extraction
- https://gandalf.lakera.ai/

**Gradient-Based Attacks**:
- Fast Gradient Sign Method (FGSM)
- Projected Gradient Descent (PGD)
- Adversarial training as defense

**Model Stealing**:
- Extracting model parameters through queries
- Reconstructing training data
- Privacy and IP concerns

**ARR-COC Context**:
- White-box attacks on relevance realization mechanism
- Gradient-based adversarial examples for knowing scorers
- Analyzing whether opponent processing can be manipulated
- Testing whether balancing mechanisms are robust to targeted perturbations

### Summary: MEL Hierarchy

| Level | Evaluation Type | Adversary Access | Typical Users | ARR-COC Application |
|-------|----------------|------------------|---------------|---------------------|
| 0 | None | N/A | Rapid prototyping | Unverified VLM deployment |
| 1 | Training data | N/A | Statistical modeling | Testing on training image-query pairs |
| 2 | Similar test set | N/A | Most ML projects | Held-out test set from same distribution |
| 3 | Disjoint data | N/A | Audit teams, production | Out-of-domain, out-of-time testing |
| 4 | Black-box adversarial | Query only | Security testing | Adversarial query-image pairs |
| 5 | White-box adversarial | Full model access | Red teams, researchers | Gradient-based attacks on relevance realization |

**Key Insight**: Most models are only evaluated at MEL 2, but high-stakes deployment (healthcare, legal, finance, safety-critical systems) requires **MEL 3-5**.

---

## Section 3: VLM Verification - Applying Model Science to ARR-COC (~100 lines)

### Verifying ARR-COC Relevance Realization

ARR-COC is a **vision-language model** that implements **relevance realization** through:
1. **Three Ways of Knowing**: Propositional, Perspectival, Participatory
2. **Opponent Processing**: Balancing cognitive tensions (Compress↔Particularize, Exploit↔Explore, Focus↔Diversify)
3. **Dynamic Token Allocation**: 64-400 tokens per patch based on query-aware relevance
4. **Transjective Coupling**: Query-content relationship (not objective or subjective)

**Verification Challenges**:

#### **1. Right-for-the-Right-Reason Verification**

Standard accuracy metrics don't reveal whether ARR-COC realizes relevance through the intended mechanism or spurious shortcuts.

**What to Verify**:
- Do Propositional scores measure information content (Shannon entropy)?
- Do Perspectival scores measure salience landscapes (Jungian archetypes)?
- Do Participatory scores measure query-content coupling (cross-attention)?
- Or do scorers learn dataset-specific artifacts?

**Verification Methods**:

**Explanation-Based Verification** ([Baniecki & Biecek, 2024](https://arxiv.org/html/2508.20040v1#bib.bibx2)):
- Use interpretability tools to inspect scorer decision paths
- Identify spurious correlations (e.g., "snow detection" instead of "husky detection")
- Feature attribution to verify Propositional/Perspectival/Participatory mechanisms

**Adversarial Explanation Testing** ([Baniecki & Biecek, 2025](https://arxiv.org/html/2508.20040v1#bib.bibx3)):
- Subtle input manipulations to test scorer robustness
- Example: Can we make a bird image be "explained" using car prototypes?
- Tests whether scorers are intrinsically interpretable or exploitable

**Counterfactual Analysis** ([Płudowski et al., 2025](https://arxiv.org/html/2508.20040v1#bib.bibx49)):
- Generate minimal image changes that flip relevance scores
- Do changes align with human understanding of relevance?
- Or do they exploit brittle model features?

#### **2. Validating Coupling Quality**

Coupling is **transjective**: emerges from query-content relationship, not from query alone or content alone.

**What to Validate**:
- Does relevance realization adapt to different queries on the same image?
- Does it adapt to the same query on different images?
- Is the coupling **dynamic** (adjusts to both) or **static** (fixed patterns)?

**Validation Methods**:

**Query Ablation Testing**:
- Test same image with semantically different queries
- Verify relevance scores change meaningfully
- Example: "Find the car" vs "Find the person" should allocate different patches

**Content Ablation Testing**:
- Test same query with semantically different images
- Verify relevance scores adapt to content
- Example: "Find the car" on urban scene vs forest scene

**Cross-Modal Coupling Analysis** ([Gandelsman et al., 2024](https://arxiv.org/html/2508.20040v1#bib.bibx19)):
- Decompose CLIP-style image-text representations
- Identify which attention heads specialize in coupling vs spurious features
- Remove heads that learn shortcuts rather than genuine coupling

**Verified Relational Alignment** (LessWrong framework, October 2025):
- Framework for robust AI alignment through relational trust
- Distinguish permission-based trust (unverified) from collaboration-based trust (verified)
- ARR-COC coupling should demonstrate collaboration-based trust: both query and content co-adjust

#### **3. Calibrating Token Budgets**

ARR-COC allocates 64-400 tokens per patch based on realized relevance. Are these allocations **calibrated** to actual task requirements?

**What to Calibrate**:
- Does high relevance → high token allocation → better task performance?
- Do low-relevance patches with high token budgets indicate miscalibration?
- Is token allocation robust across different visual domains?

**Calibration Methods**:

**Expected Calibration Error (ECE)**:
- Traditional metric for classification confidence calibration
- Adapted for relevance scores: Does predicted relevance match empirical relevance?

**Selective Prediction**:
- Allow model to abstain on low-confidence predictions
- Test whether token allocation correlates with abstention decisions

**Feature-Specific Trust Calibration** (ResearchGate publication, October 2025):
- From [Feature-Specific Trust Calibration in Physical AI Systems](https://www.researchgate.net/publication/396556983_Feature-Specific_Trust_Calibration_in_Physical_AI_Systems)
- Trust varies by feature: Propositional vs Perspectival vs Participatory
- Calibrate trust per scorer, not globally
- Example: Propositional scorer may be well-calibrated, but Perspectival scorer overconfident

#### **4. MEL 3-5 Verification for ARR-COC**

**MEL 3: Out-of-Domain Testing**:
- **Out-of-visual-domain**: Natural images → Medical images → Satellite images
- **Out-of-query-domain**: Object detection → Spatial reasoning → Temporal understanding
- **Out-of-complexity**: Simple scenes → Cluttered scenes → Occluded objects
- Verify relevance realization generalizes beyond training distribution

**MEL 4: Black-Box Adversarial**:
- Adversarial query-image pairs designed to trick relevance realization
- Test whether opponent processing is robust to adversarial coupling
- Example: Query that triggers over-allocation to irrelevant patches

**MEL 5: White-Box Adversarial**:
- Gradient-based attacks on knowing scorers
- Can adversary manipulate Propositional/Perspectival/Participatory scores?
- Test whether balancing mechanisms can be exploited
- Verify opponent processing isn't vulnerable to gradient-based manipulation

### Verification Checklist for ARR-COC

**Before Deployment**:
- [ ] **MEL 2**: Test on held-out image-query pairs from same distribution (baseline)
- [ ] **MEL 3**: Test on out-of-domain images, out-of-domain queries, out-of-complexity scenes
- [ ] **Explanation Verification**: Use interpretability tools to verify scorers learn intended mechanisms
- [ ] **Coupling Validation**: Verify transjective coupling through query/content ablations
- [ ] **Calibration**: Measure feature-specific trust for each scorer (Propositional, Perspectival, Participatory)
- [ ] **MEL 4**: Red team with adversarial query-image pairs
- [ ] **MEL 5**: White-box gradient attacks on relevance realization mechanism

**In Production**:
- [ ] **Monitoring**: Track relevance score distributions over time (distribution shift detection)
- [ ] **Out-of-Time Validation**: Periodic testing on new data
- [ ] **User Feedback**: Collect human judgments on relevance quality
- [ ] **Adversarial Monitoring**: Detect and log adversarial query patterns

---

## Section 4: Standards, Checklists, and Production Readiness (~40 lines)

### Why Standards Matter

From the Model Science paper:

> "Clear verification procedures, standards, and checklists are essential for building trustworthy AI systems."

**Examples of Standards Preventing Failures**:

**Medical Imaging COVID-19 Models** ([Hryniewska et al., 2021](https://arxiv.org/html/2508.20040v1#bib.bibx25)):
- Many errors from poor data handling and weak validation
- Structured checklist proposed for responsible model development
- Prevents misleading results in high-stakes healthcare

**Credit Scoring Models** ([Bücker et al., 2022](https://arxiv.org/html/2508.20040v1#bib.bibx10)):
- Transparency and auditability critical for financial regulation
- Without clear validation rules, advanced algorithms can't be trusted
- Regulators require MEL 3+ verification

**Generative AI Standards** ([Knott et al., 2023](https://arxiv.org/html/2508.20040v1#bib.bibx29)):
- Public release of generative models should require built-in detection mechanisms
- Protects information ecosystem from AI-generated misinformation
- Media/tech companies expected to follow detection obligations ([Knott et al., 2024](https://arxiv.org/html/2508.20040v1#bib.bibx30))

### ARR-COC Verification Standard

**Minimum Requirements for Deployment**:

1. **MEL 3 Verification**: Out-of-domain testing across visual domains, query types, complexity levels
2. **Explanation Verification**: Interpretability analysis confirming scorers learn intended mechanisms (not spurious shortcuts)
3. **Coupling Validation**: Query/content ablation tests demonstrating transjective coupling
4. **Feature-Specific Calibration**: Per-scorer trust calibration (Propositional, Perspectival, Participatory)
5. **Adversarial Robustness**: MEL 4 black-box testing, MEL 5 white-box testing if model is open-source

**Nice-to-Have (Advanced Verification)**:
- Counterfactual explanation analysis
- Cross-modal coupling decomposition (attention head analysis)
- Verified relational alignment framework (collaboration-based trust)
- Long-term out-of-time validation

**Documentation Requirements**:
- Verification report with MEL levels tested
- Known failure modes and limitations
- Calibration metrics per scorer
- Adversarial robustness results
- Deployment recommendations (safe domains, risky domains)

### V&V Checklist for Production Readiness

**Verification (Does it work?)**:
- [ ] MEL 2: Standard test set performance documented
- [ ] MEL 3: Out-of-domain generalization tested and documented
- [ ] MEL 4: Adversarial robustness tested (black-box)
- [ ] MEL 5: Gradient attack robustness tested (if open-source)

**Validation (Right problem, right reasons?)**:
- [ ] Explanation analysis confirms intended mechanisms
- [ ] No Clever Hans effects detected
- [ ] Coupling quality validated through ablations
- [ ] Feature-specific trust calibrated

**Calibration (Confidence matches accuracy?)**:
- [ ] Expected Calibration Error (ECE) measured
- [ ] Per-scorer calibration documented
- [ ] Token allocation calibration verified

**Control (Can we steer it?)**:
- [ ] Alignment mechanisms tested (if applicable)
- [ ] Model behavior controllable via query engineering
- [ ] Fail-safe mechanisms implemented (fallback to safe defaults)

**Interface (Can users trust it?)**:
- [ ] Explanation interfaces provide faithful, understandable rationales
- [ ] User trust calibrated (neither over-trust nor under-trust)
- [ ] Interactive debugging tools available for developers

---

## Source Citations

**Primary Web Research**:
- [Model Science: getting serious about verification, explanation and control of AI systems](https://arxiv.org/html/2508.20040v1) - Biecek & Samek, August 2025 (ArXiv preprint)
  - Comprehensive framework for Model Science vs Data Science
  - Five Model Evaluation Levels (MEL 0-5)
  - Four Pillars: Verification, Explanation, Control, Interface
  - Right-for-the-wrong-reason problem (Clever Hans effect)
  - Standards and checklists for trustworthy AI

**Referenced Studies (from Model Science paper)**:
- [Koenecke et al., 2024](https://arxiv.org/html/2508.20040v1#bib.bibx31) - "Careless Whisper: Speech-to-Text Hallucination Harms"
- [Dahl et al., 2024](https://arxiv.org/html/2508.20040v1#bib.bibx14) - "Large Legal Fictions" (LLM legal hallucinations)
- [Zack et al., 2024](https://arxiv.org/html/2508.20040v1#bib.bibx64) - GPT-4 gender/racial bias (Lancet Digital Health)
- [Majdinasab et al., 2024](https://arxiv.org/html/2508.20040v1#bib.bibx39) - GitHub Copilot security weaknesses
- [Ngo et al., 2024](https://arxiv.org/html/2508.20040v1#bib.bibx46) - ChatGPT medical question quality
- [Castleman & Korolova, 2025](https://arxiv.org/html/2508.20040v1#bib.bibx11) - Adultification bias in LLMs
- [Lapuschkin et al., 2019](https://arxiv.org/html/2508.20040v1#bib.bibx34) - "Unmasking Clever Hans predictors"
- [Ribeiro et al., 2016](https://arxiv.org/html/2508.20040v1#bib.bibx51) - LIME interpretability (wolf vs husky)
- [Kauffmann et al., 2025](https://arxiv.org/html/2508.20040v1#bib.bibx27) - Clever Hans in unsupervised learning
- [Elena et al., 2021](https://arxiv.org/html/2508.20040v1#bib.bibx18) - Out-of-time validation framework
- [Ho et al., 2020](https://arxiv.org/html/2508.20040v1#bib.bibx23) - Out-of-dataset validation for AutoML
- [Ganguli et al., 2022](https://arxiv.org/html/2508.20040v1#bib.bibx20) - Red teaming LLMs (Anthropic)
- [Touvron et al., 2023](https://arxiv.org/html/2508.20040v1#bib.bibx58) - LLaMA 2 open-source model
- [Lovisotto et al., 2022](https://arxiv.org/html/2508.20040v1#bib.bibx37) - Adversarial patches on ViTs
- [Baniecki & Biecek, 2024](https://arxiv.org/html/2508.20040v1#bib.bibx2) - Adversarial attacks on XAI
- [Baniecki & Biecek, 2025](https://arxiv.org/html/2508.20040v1#bib.bibx3) - "Birds look like cars" (adversarial interpretability)
- [Płudowski et al., 2025](https://arxiv.org/html/2508.20040v1#bib.bibx49) - MASCOTS counterfactual explanations
- [Gandelsman et al., 2024](https://arxiv.org/html/2508.20040v1#bib.bibx19) - Interpreting CLIP's representations
- [Hryniewska et al., 2021](https://arxiv.org/html/2508.20040v1#bib.bibx25) - COVID-19 medical imaging checklist
- [Bücker et al., 2022](https://arxiv.org/html/2508.20040v1#bib.bibx10) - Credit scoring transparency
- [Knott et al., 2023](https://arxiv.org/html/2508.20040v1#bib.bibx29) - Generative AI detection standards
- [Knott et al., 2024](https://arxiv.org/html/2508.20040v1#bib.bibx30) - AI content detection obligations

**Additional Sources (Referenced in PART 5 Instructions)**:
- [Verified Relational Alignment Framework](https://www.lesswrong.com/posts/PMDZ4DFPGwQ3RAG5x/verified-relational-alignment-a-framework-for-robust-ai) - LessWrong, October 2025
- [Feature-Specific Trust Calibration in Physical AI Systems](https://www.researchgate.net/publication/396556983_Feature-Specific_Trust_Calibration_in_Physical_AI_Systems) - ResearchGate, October 2025

**Dialogue Source**:
- RESEARCH/PlatonicDialogues/57-coupling-intelligence/57-3-research-directions-oracle-feast.md
- Direction 5: Alignment vs Coupling discussion (lines 224-271, 510-558)

**Cross-References**:
- See `alignment-coupling/00-alignment-vs-coupling-fundamental-distinction.md` for paradigm shift context
- See `alignment-coupling/01-verified-relational-alignment.md` for collaboration-based trust framework
- See `alignment-coupling/03-feature-specific-trust-calibration.md` for per-scorer calibration methods
- See `game-theory/00-game-theory-fundamentals.md` for opponent processing verification

**ARR-COC Project Context**:
- This verification framework applies to ARR-COC's Vervaekean relevance realization architecture
- Three Ways of Knowing (Propositional, Perspectival, Participatory) require separate verification
- Opponent processing (balancing tensions) must be tested for adversarial robustness
- Dynamic token allocation (64-400 tokens) requires calibration validation
- Transjective coupling (query-content relationship) requires MEL 3-5 testing across domains
