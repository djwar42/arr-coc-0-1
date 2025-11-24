# Psychophysics & Human Studies: Methods for Measuring Perception

## Overview

Psychophysics is the scientific study of the relationship between physical stimuli and the sensations and perceptions they produce. Founded by Gustav Fechner in 1860, psychophysics provides quantitative methods for measuring subjective perceptual experiences, bridging the gap between the physical world and conscious experience. These methods are essential for evaluating vision-language models against human perception, validating attention mechanisms, and understanding query-aware relevance allocation.

**Core Question**: How do we measure what humans perceive, and how can we use these measurements to evaluate AI systems?

**Relevance to ARR-COC-0-1**: Human psychophysics provides ground truth for relevance perception. By measuring just-noticeable differences in token allocation quality, signal detection thresholds for relevance judgments, and perceptual scales for compression artifacts, we can validate that the model's relevance realization matches human perception.

From [Introspective psychophysics for the study of subjective experience](https://pubmed.ncbi.nlm.nih.gov/39569467/) (NIH PMC, Accessed 2025-11-14):
> "This next generation of psychophysics will establish a powerful tool for building and testing precise explanatory models of phenomenology across clarity, vividness, confidence, and more."

---

## Section 1: Psychophysics Fundamentals

### What is Psychophysics?

Psychophysics quantifies the stimulus-response relationship:
- **Physical Stimulus** (S): Measurable property (brightness, weight, sound intensity)
- **Perceptual Response** (R): Subjective experience or judgment
- **Psychophysical Function**: R = f(S)

**Three Fundamental Questions**:
1. **Detection**: What is the minimum stimulus intensity detectable?
2. **Discrimination**: What is the smallest difference between two stimuli that can be perceived?
3. **Scaling**: How does perceived intensity relate to physical intensity?

### Historical Context

**Gustav Fechner (1860)**: "Elements of Psychophysics"
- First systematic attempt to measure mental phenomena
- Proposed Fechner's Law: S = k log(I), where S is sensation, I is intensity
- Established psychophysics as bridge between physics and psychology

**Ernst Weber (1834)**: Weber's Law
- Just-noticeable difference (JND) is proportional to stimulus magnitude
- ΔI/I = constant (Weber fraction)
- Foundation for modern discrimination experiments

**Stanley Smith Stevens (1957)**: Stevens' Power Law
- S = kI^n, where n varies by modality
- Better fit to perceptual data than logarithmic law
- Different exponents for different senses (brightness n≈0.33, pain n≈3.5)

From [Continuous psychophysics: past, present, future](https://www.sciencedirect.com/science/article/abs/pii/S1364661325000051) (ScienceDirect, Accessed 2025-11-14):
> "Continuous target-tracking psychophysics is an innovative experimental paradigm that has emerged as a powerful tool for studying perception, cognition, and visually guided behavior."

### Modern Applications

**Vision Science**:
- Contrast sensitivity functions
- Color discrimination thresholds
- Motion detection limits
- Visual acuity measurements

**Human-Computer Interaction**:
- UI element discriminability
- Just-noticeable differences in display quality
- Perceptual compression thresholds

**AI Evaluation**:
- Human-model agreement metrics
- Perceptual quality assessment
- Attention mechanism validation
- Relevance perception thresholds

---

## Section 2: Classical Psychophysical Methods

### Method of Limits

**Procedure**:
1. Start with clearly detectable (or undetectable) stimulus
2. Gradually decrease (or increase) intensity in small steps
3. Subject reports when they can no longer detect (or first detect) stimulus
4. Repeat with ascending and descending series
5. Threshold = average of transition points

**Example - Visual Detection**:
```
Ascending series:  0 1 2 3 4 5 6 7 8 9
Response:          N N N N Y Y Y Y Y Y
                           ↑
                      Threshold ≈ 4.5

Descending series: 9 8 7 6 5 4 3 2 1 0
Response:          Y Y Y Y Y N N N N N
                           ↑
                      Threshold ≈ 4.5
```

**Advantages**:
- Fast, intuitive
- Good for clinical testing
- Simple to implement

**Disadvantages**:
- **Habituation**: Subject expects same response to continue
- **Anticipation errors**: Premature response changes
- **Hysteresis**: Different thresholds for ascending vs descending
- No measure of response variability

From [Methods in Psychophysics](https://2024.sci-hub.se/6766/17e0e80bd3770ea807f2de6941b413f4/wichmann2018.pdf) (Sci-Hub, Accessed 2025-11-14):
> "In this chapter on methods in psychophysics we stress the importance of a time-proven, careful, and somewhat conservative approach."

### Method of Constant Stimuli

**Procedure**:
1. Select 5-9 fixed stimulus intensities spanning detection range
2. Present each intensity multiple times (20-50 trials per level) in random order
3. Subject makes binary response (detected/not detected, same/different)
4. Plot percent detected vs intensity (psychometric function)
5. Threshold = intensity detected 50% of the time

**Example - Contrast Detection**:
```
Contrast Level:     2%   4%   6%   8%  10%  12%  14%
Trials per level:   40   40   40   40   40   40   40
% Detected:         10%  25%  48%  72%  88%  96%  99%
                              ↑
                    Threshold ≈ 6% (50% detection)
```

**Advantages**:
- **Gold standard** for threshold estimation
- Full psychometric function
- No order effects (randomized presentation)
- Enables psychometric function fitting
- Provides slope estimate (discriminability)

**Disadvantages**:
- **Time-consuming**: Many trials needed
- **Inefficient**: Many trials far from threshold
- Fixed stimulus set may miss optimal range

**Psychometric Function**:
- Sigmoid shape: P(detect) = Ψ(x)
- Common models: Cumulative Gaussian, Weibull, logistic
- Parameters: Threshold (α), slope (β), lapse rate (λ), guess rate (γ)
- Fit via maximum likelihood estimation

From [Comparing adaptive procedures for estimating the psychometric function](https://pmc.ncbi.nlm.nih.gov/articles/PMC3634902/) (NIH PMC, Accessed 2025-11-14):
> "A classic procedure for estimating the psychometric function utilizes the method of constant stimuli. In this procedure, the percent correct is estimated at each of several stimulus levels, and the psychometric function is estimated by fitting an appropriate function to these data."

### Method of Adjustment

**Procedure**:
1. Subject directly controls stimulus intensity (e.g., brightness knob)
2. Instructed to adjust until stimulus is "just barely detectable" or "equal to reference"
3. Record final setting
4. Repeat 10-20 times
5. Threshold = average setting

**Example - Brightness Matching**:
```
Reference brightness: 50 cd/m²
Subject adjustments:  48, 52, 49, 51, 50, 49, 48, 51, 50, 49
Mean: 49.7 cd/m²
SD: 1.34 cd/m²
```

**Advantages**:
- **Very fast**: Single trial gives estimate
- Subject has control, reduces frustration
- Natural task for matching experiments
- Good for initial threshold estimation

**Disadvantages**:
- No forced-choice, subject can avoid responding
- Response bias affects results
- Motor precision limits accuracy
- No measure of sensitivity separate from criterion

**Applications**:
- Color matching (metamers)
- Brightness matching
- Loudness balancing
- Size estimation

---

## Section 3: Adaptive Psychophysical Methods

### Why Adaptive Methods?

Classical methods waste trials far from threshold. Adaptive methods place most stimuli near threshold, improving efficiency by 3-5×.

**Key Principle**: Use previous responses to select next stimulus intensity, concentrating trials where information is most valuable.

From [Adaptive psychophysical procedures](https://link.springer.com/content/pdf/10.3758/BF03194543.pdf) (Springer, Accessed 2025-11-14):
> "However, in the method of constant stimuli, the threshold is extracted from a fully sampled function, making the measurement of this single point on the psychometric function very inefficient."

### Staircase Procedures

**Simple Up-Down Staircase**:
1. Start above threshold
2. After correct response: decrease intensity (make harder)
3. After incorrect response: increase intensity (make easier)
4. Continue for fixed number of reversals (8-12)
5. Threshold = mean of reversal points

**Example - Tone Detection**:
```
Intensity: 40→35→30→25→30→25→20→25→20→15→20→15→20
Response:  C  C  C  I  C  I  C  I  C  I  C  I  C
                     ↑     ↑     ↑     ↑     ↑
                  Reversals: 30, 25, 25, 20, 20
                  Threshold ≈ 24 dB
```

**Transformed Up-Down Methods**:
- **1-up-2-down**: Increase after 1 incorrect, decrease after 2 correct
  - Converges to 70.7% correct
  - Good for 2AFC tasks
- **1-up-3-down**: Converges to 79.4% correct
- **2-up-1-down**: Converges to 29.3% correct
- General formula: Convergence = 0.5^(1/n_down)

**Advantages**:
- Fast convergence (20-40 trials)
- Adaptive to individual differences
- Real-time feedback for subject
- Simple to implement

**Disadvantages**:
- Only estimates threshold, not full function
- Can get stuck in local minima
- Early trials bias final estimate
- No measure of slope

From [Adaptive Psychophysical Methods Outline](https://engineering.purdue.edu/~ece511/LectureNotes/pp12.pdf) (Purdue, Accessed 2025-11-14):
> "Unlike classical procedures (method of limits, method of constant stimuli, signal detection), adaptive method places most of the stimuli at intensity levels close to the threshold that is being measured."

### Bayesian Adaptive Methods

**QUEST (Quick Estimation by Sequential Testing)**:
- Maintains probability distribution over threshold values
- Uses Bayes' theorem to update after each trial
- Selects next stimulus to maximize expected information gain
- Converges faster than staircases (10-30 trials)

**Procedure**:
1. Initialize prior distribution P(θ) for threshold θ
2. For each trial:
   - Select stimulus x maximizing expected information
   - Present stimulus, get response r
   - Update posterior: P(θ|r) ∝ P(r|θ,x) × P(θ)
3. Final threshold estimate: mode or mean of posterior

**Psi Method**:
- Extends QUEST to estimate full psychometric function
- Estimates threshold, slope, lapse rate simultaneously
- Uses entropy minimization to select optimal stimuli
- Most efficient adaptive procedure (theory optimal)

**Advantages**:
- **Theoretically optimal** efficiency
- Robust to outliers
- Can estimate multiple parameters
- Incorporates prior knowledge

**Disadvantages**:
- Computationally intensive
- Requires specifying priors
- Complex implementation
- May not feel "natural" to subjects

From [Lesson 4: Measuring psychometric functions](http://courses.washington.edu/matlab1/Lesson_4.html) (UW, Accessed 2025-11-14):
> "We'll cover two ways of choosing the stimulus strength for each trial - the method of 'constant stimuli' and a '3-down 1-up staircase method'."

### Choosing an Adaptive Method

**Decision Tree**:
```
Need full psychometric function?
├─ YES → Method of constant stimuli OR Bayesian adaptive (Psi)
└─ NO → Need just threshold?
    ├─ YES → Fast needed?
    │   ├─ YES → QUEST OR 2-down-1-up staircase
    │   └─ NO → Method of constant stimuli (gold standard)
    └─ NO → Method of adjustment (exploratory)
```

**Efficiency Comparison** (trials to reach 1% threshold precision):
- Method of constant stimuli: 200-400 trials
- Simple staircase: 40-60 trials
- Transformed staircase: 30-50 trials
- QUEST: 20-40 trials
- Psi method: 15-30 trials

---

## Section 4: Signal Detection Theory (SDT)

### The Problem with Classical Thresholds

Classical psychophysics assumes a fixed threshold: stimuli above threshold are always detected, below never detected. Reality: detection is probabilistic and influenced by response criterion.

**Two Factors in Detection**:
1. **Sensitivity (d')**: True ability to discriminate signal from noise
2. **Response Criterion (β)**: Willingness to say "yes" (bias)

### SDT Framework

**Components**:
- **Signal + Noise (SN)**: Internal response when stimulus present
- **Noise (N)**: Internal response when stimulus absent
- Both distributions are Gaussian (usually)
- **d' (d-prime)**: Separation between distributions (sensitivity)
- **Criterion (c)**: Decision boundary (bias)

**Four Possible Outcomes**:
```
                   Stimulus Present    Stimulus Absent
Response "Yes"     HIT                 FALSE ALARM
Response "No"      MISS                CORRECT REJECTION
```

**Sensitivity Calculation**:
```
d' = Z(Hit Rate) - Z(False Alarm Rate)

Where Z is the inverse of the cumulative standard normal distribution.
```

**Example**:
```
Hits: 80/100 (80%)
False Alarms: 20/100 (20%)

d' = Z(0.80) - Z(0.20)
   = 0.84 - (-0.84)
   = 1.68

Interpretation: Signal distribution is 1.68 standard deviations above noise.
```

From [Signal Detection Theory and the 'yes/no' experiment](https://courses.washington.edu/matlab1/Lesson_8.html) (UW, Accessed 2025-11-14):
> "This shows that in theory we can estimate D-Prime from a simple yes/no experiment."

### Response Bias Measures

**Criterion (c)**:
```
c = -0.5 × [Z(Hit Rate) + Z(False Alarm Rate)]
```
- c = 0: Neutral (equal evidence required)
- c > 0: Conservative (prefer saying "no")
- c < 0: Liberal (prefer saying "yes")

**Beta (β)**:
```
β = exp(c × d')
```
- β = 1: Neutral
- β > 1: Conservative
- β < 1: Liberal

**Example - Conservative Observer**:
```
Hits: 60/100 (60%)
False Alarms: 5/100 (5%)

d' = Z(0.60) - Z(0.05) = 0.25 - (-1.64) = 1.89
c = -0.5 × [0.25 + (-1.64)] = 0.70

Interpretation: Good sensitivity (d'=1.89), but conservative criterion (c=0.70).
```

From [Signal detection theory and psychophysics](https://link.springer.com/article/10.3758/s13428-022-01913-5) (Springer, Accessed 2025-11-14):
> "Signal detection theory and psychophysics provide a framework for understanding how observers make decisions under uncertainty."

### ROC Curves

**Receiver Operating Characteristic (ROC)**:
- Plot Hit Rate vs False Alarm Rate
- Each point represents different criterion
- Area under curve (AUC) = probability correct in 2AFC
- AUC = Φ(d'/√2), where Φ is cumulative normal

**Interpretation**:
- AUC = 0.5: Chance performance (d'=0)
- AUC = 0.75: d' ≈ 1.35
- AUC = 0.90: d' ≈ 2.56
- AUC = 1.0: Perfect discrimination

**Applications to VLM Evaluation**:
- Measure model's ability to detect task-relevant regions
- Separate sensitivity from response bias
- Compare human vs model attention discrimination
- Evaluate relevance allocation thresholds

From [Signal detection theory and vestibular thresholds](https://pmc.ncbi.nlm.nih.gov/articles/PMC3096492/) (NIH PMC, Accessed 2025-11-14):
> "The theoretical psychophysical function for detection (Yes/No) of unidirectional stimuli tasks ranges between 0% yes for very small magnitudes and 100% yes for large magnitudes."

---

## Section 5: Weber's Law and Just Noticeable Difference (JND)

### Weber's Law

**Statement**: The just-noticeable difference (JND) is a constant proportion of the stimulus magnitude.

**Formula**:
```
ΔI/I = k

Where:
- ΔI = Just-noticeable difference
- I = Standard stimulus intensity
- k = Weber fraction (constant for each modality)
```

**Example - Weight Discrimination**:
```
If k = 0.02 for weight (2%)

Standard weight: 100g → JND = 2g (need 102g to notice difference)
Standard weight: 500g → JND = 10g (need 510g to notice difference)
Standard weight: 1000g → JND = 20g (need 1020g to notice difference)
```

From [Just Noticeable Difference (JND) in Psychology](https://www.verywellmind.com/what-is-the-just-noticeable-difference-2795306) (Verywell Mind, Accessed 2025-11-14):
> "The just noticeable difference, also known as the difference threshold, is the minimum level of stimulation that a person can detect 50% of the time."

### Weber Fractions by Modality

**Visual Modality**:
- Brightness: k ≈ 0.08 (8%)
- Line length: k ≈ 0.03 (3%)
- Contrast: k ≈ 0.01-0.02 (1-2%)

**Auditory**:
- Loudness: k ≈ 0.10 (10%)
- Pitch: k ≈ 0.003 (0.3%)

**Tactile**:
- Weight: k ≈ 0.02 (2%)
- Electric shock: k ≈ 0.01 (1%)

**Taste/Smell**:
- Salt concentration: k ≈ 0.08 (8%)
- Odor concentration: k ≈ 0.10 (10%)

### Violations of Weber's Law

**Near Absolute Threshold**:
- Weber's law breaks down for very weak stimuli
- JND becomes constant (not proportional)
- Transition from Weber to constant JND

**Very Intense Stimuli**:
- Weber fraction increases at high intensities
- Sensory saturation effects
- Safety/pain thresholds

**Modified Weber's Law**:
```
ΔI/(I + I₀) = k

Where I₀ is a constant accounting for threshold effects.
```

From [Grasping follows Weber's law](https://pmc.ncbi.nlm.nih.gov/articles/PMC9669808/) (NIH PMC, Accessed 2025-11-14):
> "Weber's law states that the just noticeable difference (JND) between two stimuli increases with stimulus magnitude."

### Applications to ARR-COC-0-1

**Token Budget JNDs**:
- What is the minimum change in token allocation that affects perceived relevance?
- If baseline = 200 tokens, is 204 tokens noticeably better?
- Weber's law predicts k ≈ 0.05-0.10 (10-20 token JND at 200 baseline)

**Compression Quality Thresholds**:
- How much LOD reduction before artifacts are noticeable?
- 400 tokens → 380 tokens (5% reduction): likely detectable
- 400 tokens → 360 tokens (10% reduction): definitely detectable

**Experimental Design**:
```
Baseline token allocation: 64, 128, 200, 300, 400
Weber fraction k ≈ 0.10 (hypothesized)

Expected JNDs:
- 64 tokens: JND ≈ 6 tokens
- 128 tokens: JND ≈ 13 tokens
- 200 tokens: JND ≈ 20 tokens
- 300 tokens: JND ≈ 30 tokens
- 400 tokens: JND ≈ 40 tokens

Discrimination experiment:
"Which image has better detail in the region of interest?"
Present pairs: (200, 220), (200, 240), (200, 260)
Measure discrimination threshold
```

---

## Section 6: Perceptual Scaling and Magnitude Estimation

### Stevens' Power Law

**Formula**:
```
Ψ = k × I^n

Where:
- Ψ = Perceived magnitude
- I = Physical intensity
- k = Scaling constant
- n = Exponent (varies by modality)
```

**Exponents by Modality**:
```
Compressive (n < 1):          Expansive (n > 1):
- Brightness: 0.33            - Electric shock: 3.5
- Loudness: 0.67              - Pain: 3.5
- Smell: 0.55                 - Length: 1.0 (linear)
- Heaviness: 1.45             - Numerosity: 1.0
```

**Interpretation**:
- **n < 1 (compressive)**: Doubling physical intensity less than doubles perception
  - Brightness doubles when luminance increases 8× (2³)
- **n > 1 (expansive)**: Doubling physical intensity more than doubles perception
  - Pain doubles when stimulus increases 1.23× (2^(1/3.5))
- **n = 1 (linear)**: Direct proportionality (rare)

### Magnitude Estimation

**Procedure**:
1. Present reference stimulus with assigned value (e.g., "10")
2. Present test stimuli in random order
3. Subject assigns number proportional to perceived magnitude
4. No maximum/minimum constraint
5. Plot log(response) vs log(intensity)
6. Slope = exponent n

**Example - Brightness Estimation**:
```
Reference: 50 cd/m² = "10"

Test:        Luminance    Avg Response    Log(Lum)   Log(Resp)
Stimulus 1:  10 cd/m²     4.2            1.00       0.62
Stimulus 2:  25 cd/m²     7.1            1.40       0.85
Stimulus 3:  75 cd/m²     12.8           1.88       1.11
Stimulus 4:  150 cd/m²    18.5           2.18       1.27

Log-log regression:
Slope = 0.35 ≈ brightness exponent (0.33)
```

From [Just Noticeable Difference (JND) in Psychology: Examples](https://www.simplypsychology.org/what-is-the-just-noticeable-difference.html) (Simply Psychology, Accessed 2025-11-14):
> "Using Weber's law, the person conducting the experiment could now predict the size of the observer's difference threshold for a light spot of any intensity."

### Category Scaling

**Procedure**:
1. Define discrete categories (e.g., 1-7 scale)
2. Anchor endpoints with labels ("not at all" to "extremely")
3. Subject assigns stimuli to categories
4. Analyze distribution of responses

**Example - Image Quality Assessment**:
```
Scale:
1 = Very poor
2 = Poor
3 = Fair
4 = Good
5 = Excellent

Token Budget:     64    128   200   300   400
Mean Rating:      2.1   3.2   4.1   4.6   4.8
SD:              0.8   0.7   0.6   0.5   0.4

Diminishing returns above 300 tokens (ceiling effect).
```

**Advantages**:
- Natural for subjects
- Easy to analyze (ordinal statistics)
- Good for quality assessment
- Handles ceiling/floor effects

**Disadvantages**:
- Restricted range
- Category boundaries arbitrary
- Ordinal not interval scale
- Individual differences in category use

### Applications to VLM Evaluation

**Relevance Magnitude Estimation**:
```
Task: "Rate how relevant each image region is to the query"
Query: "Where is the dog?"

Region A (contains dog): Assigned "100"
Region B (dog's toy): Subject rates "45"
Region C (background): Subject rates "5"
Region D (unrelated object): Subject rates "2"

Scaling reveals relative relevance structure.
Compare to model attention weights.
```

**Quality vs Token Budget Scaling**:
```
Psychophysical experiment:
- Vary token budget: 64, 128, 200, 300, 400
- Subjects rate perceived detail quality (1-10 scale)
- Fit power law: Quality = k × (Tokens)^n
- Find diminishing returns point
- Optimize token budget allocation policy
```

---

## Section 7: Human Studies Protocols and Ethics

### Institutional Review Board (IRB)

**Purpose**: Protect human subjects in research
- Ensure informed consent
- Minimize risks
- Protect vulnerable populations
- Maintain confidentiality

**IRB Review Levels**:
1. **Exempt**: Minimal risk (surveys, public data analysis)
2. **Expedited**: Low risk (brief behavioral tasks, no deception)
3. **Full Review**: Greater than minimal risk (vulnerable populations, deception, medical)

**Psychophysics studies typically qualify for expedited review** (non-invasive perceptual tasks).

### Informed Consent

**Required Elements**:
1. **Purpose**: Study goals in plain language
2. **Procedures**: What will participant do?
3. **Duration**: How long will it take?
4. **Risks**: Potential discomfort (eye strain, fatigue)
5. **Benefits**: Contribution to science, compensation
6. **Confidentiality**: Data protection measures
7. **Voluntary**: Can withdraw at any time
8. **Contact**: Researcher and IRB contact information

**Example - VQA Study Consent**:
```
Title: "Visual Attention in Image Question Answering"

You will view images and answer questions about them (e.g., "Where is the dog?").
We will record your eye movements using a camera below the screen.

Duration: 45 minutes
Compensation: $15

Risks: Minimal. Possible eye strain (can take breaks).
Benefits: Contribute to AI research.

Data: Eye tracking and responses recorded. No identifying information stored.
All data anonymized and stored securely.

Voluntary: You may stop at any time without penalty.

Contact: [Researcher name, email, phone]
IRB: [University IRB contact]

I have read and understood the above.

Signature: _______________ Date: ___________
```

### Participant Recruitment

**Sample Size Determination**:
- Power analysis: Detect effect size d with power 1-β at significance α
- Typical: n = 20-30 for within-subjects psychophysics
- Larger samples for between-subjects or small effects

**Inclusion/Exclusion Criteria**:
```
Inclusion:
- Age 18-65
- Normal or corrected-to-normal vision (20/20)
- Fluent English speaker (for VQA tasks)
- No history of neurological disorders

Exclusion:
- Uncorrected vision problems
- Color blindness (for color tasks)
- Medications affecting perception
- Previous participation in similar studies (to avoid practice effects)
```

**Compensation**:
- Typical: $10-15/hour
- Bonus for performance (avoid biasing strategy)
- Course credit (undergraduate subject pools)

### Data Collection Best Practices

**Environmental Control**:
- Consistent lighting (photopic: 300-500 lux)
- Viewing distance fixed (typically 57cm = 1° visual angle per cm)
- Screen calibrated (luminance, color)
- Quiet testing room (auditory tasks)

**Session Structure**:
```
1. Welcome and consent (5 min)
2. Instructions and practice (5-10 min)
3. Calibration (eye tracking: 2 min)
4. Block 1: Experimental trials (15 min)
5. Break (5 min)
6. Recalibration if needed (2 min)
7. Block 2: Experimental trials (15 min)
8. Debriefing and payment (5 min)

Total: ~50 minutes
```

**Quality Control**:
- Practice trials to stabilize performance
- Catch trials to detect inattention
- Validation trials with known answers
- Monitor reaction times for outliers
- Check calibration accuracy (eye tracking: <0.5° error)

### Data Privacy and Confidentiality

**De-identification**:
- Assign participant IDs (P001, P002, ...)
- Separate consent forms (with names) from data
- Store consent forms in locked cabinet or encrypted folder
- No names/emails in data files

**Data Storage**:
- Encrypted hard drive or secure server
- Access restricted to research team
- Retention period specified (typically 5 years post-publication)
- Destruction protocol for old data

**Sharing Data**:
- Anonymized data can be shared (e.g., Open Science Framework)
- Aggregate data in publications
- Individual data only with explicit consent

From [Methods in Psychophysics](https://2024.sci-hub.se/6766/17e0e80bd3770ea807f2de6941b413f4/wichmann2018.pdf) (Sci-Hub, Accessed 2025-11-14):
> "A careful and somewhat conservative approach is essential when conducting human psychophysics experiments to ensure reproducibility and ethical standards."

---

## Section 8: ARR-COC-0-1 Human Evaluation Design

### Validating Relevance Perception Thresholds

**Research Question**: Does ARR-COC-0-1's token allocation match human relevance judgments?

**Experimental Design**:

**Phase 1: Token Budget JND Measurement**
```
Method: 2-Alternative Forced Choice (2AFC) with staircase
Task: "Which image shows more detail in the [query-relevant region]?"

Stimuli:
- Reference: 200 tokens allocated to ROI
- Comparison: 200 + ΔT tokens (adaptive staircase)

Procedure:
1. Present query: "Where is the dog?"
2. Show two images side-by-side:
   - Left: ARR-COC-0-1 with 200 tokens to dog region
   - Right: ARR-COC-0-1 with 200+ΔT tokens to dog region
3. Subject selects image with better dog detail
4. 2-down-1-up staircase converges to 71% correct (JND threshold)

Expected Result:
JND ≈ 20-30 tokens (10-15% Weber fraction)
Validates meaningful perceptual differences in token allocation.
```

**Phase 2: Relevance Scaling**
```
Method: Magnitude estimation
Task: Rate relevance of each image region to query (0-100 scale)

Stimuli:
- 50 images with VQA questions
- ARR-COC-0-1 generates attention map
- 5-8 regions per image

Procedure:
1. Show image + query
2. Highlight region (1 of 5-8)
3. Subject rates relevance (0 = irrelevant, 100 = highly relevant)
4. Repeat for all regions
5. Correlate human ratings with model attention weights

Expected Result:
Pearson r > 0.7 between human relevance and model attention.
Validates that high-attention regions perceived as relevant.
```

**Phase 3: Token Budget Optimization**
```
Method: Category scaling + quality assessment
Task: "Rate overall image understanding quality (1-7 scale)"

Stimuli:
- VQA questions answered by ARR-COC-0-1
- Vary total token budget: 64, 128, 200, 300, 400
- 40 questions × 5 budgets = 200 trials

Procedure:
1. Show image + question
2. Show ARR-COC-0-1 answer (correct/incorrect not revealed)
3. Subject rates confidence in model's understanding (1-7)
4. Analyze quality vs token budget relationship

Expected Result:
Quality saturates around 250-300 tokens (diminishing returns).
Informs optimal token budget policy.
```

### Eye Tracking Validation Study

**Research Question**: Do human gaze patterns match ARR-COC-0-1 attention allocation?

**Equipment**:
- SR Research EyeLink 1000 or Tobii Pro Spectrum
- Sampling rate: 500 Hz minimum
- Accuracy: <0.5° visual angle

**Design**:
```
Task: Visual Question Answering with eye tracking

Stimuli:
- 100 images with questions
- Query presented 2 seconds before image
- Image visible for 5 seconds (free viewing)

Procedure:
1. Calibration (9-point)
2. Read question aloud (ensures comprehension)
3. Fixation cross (500ms)
4. Image appears (5000ms, free viewing)
5. Answer question (multiple choice)
6. Repeat for 100 trials (5 breaks)

Analysis:
1. Extract fixation locations and durations
2. Generate human attention heatmap (Gaussian smoothing, σ=1°)
3. Compare to ARR-COC-0-1 attention map
4. Metrics:
   - AUC-Judd: ROC curve for fixation prediction
   - NSS (Normalized Scanpath Saliency): Z-score at fixations
   - Pearson correlation: Spatial overlap
   - KL divergence: Distribution similarity

Expected Result:
- AUC > 0.80 (strong prediction of fixations)
- NSS > 2.0 (fixations on high-attention regions)
- Pearson r > 0.60 (spatial agreement)
- Model outperforms bottom-up saliency baseline
```

From [Eye-Tracking Studies & Task-Driven Attention](../biological-vision/02-eye-tracking-task-attention.md):
> "Task instructions fundamentally reorganize visual priorities. Free-viewing produces exploratory patterns visiting many regions. Specific tasks produce focused patterns targeting task-relevant features."

### Signal Detection Analysis of Relevance Judgments

**Research Question**: Can humans reliably detect task-relevant vs irrelevant regions (signal detection sensitivity)?

**Design**:
```
Task: Detect query-relevant regions (Yes/No detection)

Stimuli:
- 200 trials: 100 relevant regions, 100 irrelevant regions
- Relevance defined by ARR-COC-0-1 attention threshold (top 25%)
- Brief presentation (500ms) to avoid extensive search

Procedure:
1. Present query: "Find the bicycle"
2. Show image region (500ms)
3. Subject responds: "Relevant" or "Not relevant"
4. Record hits, misses, false alarms, correct rejections

Analysis:
Compute d' (sensitivity):
d' = Z(Hit Rate) - Z(False Alarm Rate)

Compute criterion (c):
c = -0.5 × [Z(Hit Rate) + Z(False Alarm Rate)]

Expected Result:
- High sensitivity: d' > 2.0 (humans reliably detect relevance)
- Neutral criterion: c ≈ 0 (balanced yes/no responding)
- Validates that ARR-COC-0-1 attention threshold meaningful
```

### Sample Size and Power Analysis

**Typical Psychophysics Study**:
```
Effect size (Cohen's d):
- Small: d = 0.2
- Medium: d = 0.5
- Large: d = 0.8

Power analysis (α=0.05, power=0.80):
- Within-subjects design:
  - Medium effect (d=0.5): n = 27
  - Large effect (d=0.8): n = 12

- Between-subjects design:
  - Medium effect (d=0.5): n = 64 per group
  - Large effect (d=0.8): n = 26 per group

Recommendation for ARR-COC-0-1 validation:
- Phase 1 (JND): n=20 (within-subjects staircase)
- Phase 2 (Scaling): n=30 (magnitude estimation variability)
- Phase 3 (Quality): n=25 (within-subjects categorical)
- Eye tracking: n=30 (typical for gaze-prediction studies)
```

**Statistical Tests**:
- JND threshold: One-sample t-test vs theoretical prediction
- Relevance correlation: Pearson correlation + Fisher z-transform
- Token budget quality: Repeated-measures ANOVA
- Eye tracking agreement: Bootstrap confidence intervals for AUC/NSS

### Ethical Considerations Specific to ARR-COC-0-1

**Potential Risks**:
- **Eye strain**: Long viewing sessions (mitigate with breaks)
- **Fatigue**: 200+ trials (limit session to 60 minutes max)
- **Boredom**: Repetitive task (vary stimuli, provide feedback)

**Minimal Risk Determination**:
- Non-invasive visual tasks
- No deception
- Standard psychophysics paradigms
- IRB likely grants expedited approval

**Data Use Transparency**:
```
Consent language:
"Your eye movements and quality ratings will be used to evaluate
an AI system's visual attention. This helps us understand whether
the AI focuses on image regions that humans find important.
Your data will be anonymized and may be shared with other
researchers studying human perception."
```

---

## References & Sources

All sources accessed on 2025-11-14:

**Psychophysics Methods Overview:**
1. [Introspective psychophysics for the study of subjective experience](https://pubmed.ncbi.nlm.nih.gov/39569467/) - NIH PMC
2. [Continuous psychophysics: past, present, future](https://www.sciencedirect.com/science/article/abs/pii/S1364661325000051) - ScienceDirect
3. [Methods in Psychophysics](https://2024.sci-hub.se/6766/17e0e80bd3770ea807f2de6941b413f4/wichmann2018.pdf) - Sci-Hub
4. [Continuous psychophysics shows millisecond-scale visual processing](https://jburge.psych.upenn.edu/ewExternalFiles/BurgeCormack_JOV_2024-1.pdf) - University of Pennsylvania
5. [Fechner Day 2024](https://ispsychophysics.org/fechner-day-2024/) - International Society for Psychophysics
6. [Informing Machine Perception With Psychophysics](https://ieeexplore.ieee.org/document/10496416) - IEEE Xplore
7. [Psychophysics - Latest research and news](https://www.nature.com/subjects/psychophysics) - Nature
8. [Attention, Perception, & Psychophysics](https://www.psychonomic.org/page/app) - Psychonomic Society

**Signal Detection Theory:**
9. [Signal Detection Theory and the 'yes/no' experiment](https://courses.washington.edu/matlab1/Lesson_8.html) - University of Washington
10. [Sensitivity at the optimal criterion location](https://link.springer.com/article/10.3758/s13428-022-01913-5) - Springer
11. [Detection theory](https://en.wikipedia.org/wiki/Detection_theory) - Wikipedia
12. [Psychophysics & Signal Detection Theory](https://pillowlab.princeton.edu/teaching/sp2019/slides/Lec03_Psychophysics.pdf) - Princeton
13. [d prime - APA Dictionary of Psychology](https://dictionary.apa.org/d-prime) - APA
14. [Sensitivity and Bias - introduction to Signal Detection](https://www.birmingham.ac.uk/Documents/college-les/psych/vision-laboratory/sdtintro.pdf) - University of Birmingham
15. [Signal detection theory and vestibular thresholds](https://pmc.ncbi.nlm.nih.gov/articles/PMC3096492/) - NIH PMC
16. [Signal Detection Theory - an overview](https://www.sciencedirect.com/topics/computer-science/signal-detection-theory) - ScienceDirect

**Weber's Law and JND:**
17. [Just-noticeable difference](https://en.wikipedia.org/wiki/Just-noticeable_difference) - Wikipedia
18. [Just Noticeable Difference (JND) in Psychology: Examples](https://www.simplypsychology.org/what-is-the-just-noticeable-difference.html) - Simply Psychology
19. [Just Noticeable Difference (JND) in Psychology](https://www.verywellmind.com/what-is-the-just-noticeable-difference-2795306) - Verywell Mind
20. [Weber's Law](https://www.cis.rit.edu/people/faculty/montag/vandplite/pages/chap_3/ch3p1.html) - RIT
21. [Just-noticeable difference | Research Starters](https://www.ebsco.com/research-starters/physics/just-noticeable-difference) - EBSCO
22. [Grasping follows Weber's law](https://pmc.ncbi.nlm.nih.gov/articles/PMC9669808/) - NIH PMC
23. [Weber's Law of Just Noticeable Differences](https://k-carlson180.medium.com/webers-law-of-just-noticeable-differences-af4f6a792d46) - Medium

**Adaptive Methods:**
24. [Adaptive Psychophysical Methods Outline](https://engineering.purdue.edu/~ece511/LectureNotes/pp12.pdf) - Purdue
25. [Comparing adaptive procedures for estimating the psychometric function](https://pmc.ncbi.nlm.nih.gov/articles/PMC3634902/) - NIH PMC
26. [Adaptive procedures in psychophysical research](https://link.springer.com/content/pdf/10.3758/BF03194543.pdf) - Springer
27. [Lesson 4: Measuring psychometric functions](http://courses.washington.edu/matlab1/Lesson_4.html) - University of Washington
28. [Are staircases more efficient than the method of constant stimuli](https://psychology.stackexchange.com/questions/8524/are-staircases-more-efficient-than-the-method-of-constant-stimuli) - Psychology Stack Exchange
29. [Psychophysics](https://en.wikipedia.org/wiki/Psychophysics) - Wikipedia
30. [Staircase procedure and constant stimuli method](https://psycnet.apa.org/record/1991-17580-001) - APA PsycNet
31. [Adaptive psychophysical procedures](https://www.sciencedirect.com/science/article/pii/004269899500016X) - ScienceDirect (Treutwein 1995)
32. [Psychophysical Methods](http://www.appstate.edu/~steelekm/classes/psy4215/Documents/Ehrenstein&Ehrenstein1999-PsychoPhysicalMethods.pdf) - Appalachian State

**Source Documents Referenced:**
33. [Eye-Tracking Studies & Task-Driven Attention](../biological-vision/02-eye-tracking-task-attention.md) - eye tracking methods, gaze analysis, human attention metrics
34. [Gestalt Perception & Visual Attention](../biological-vision/00-gestalt-visual-attention.md) - perceptual organization, grouping principles

**Connection to ARR-COC-0-1:**
These psychophysical methods provide the experimental foundation for validating ARR-COC-0-1's relevance realization against human perception. By measuring just-noticeable differences in token allocation quality, signal detection thresholds for relevance judgments, and perceptual scaling of compression artifacts, we can ensure that the model's attention allocation matches human visual priorities. Eye tracking validation confirms that query-aware token budgets align with natural human gaze patterns during visual question answering.
