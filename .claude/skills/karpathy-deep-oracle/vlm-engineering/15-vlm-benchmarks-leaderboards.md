# VLM Benchmarks & Leaderboards

## Overview

Vision-language model evaluation has evolved into a complex ecosystem of specialized benchmarks and competitive leaderboards tracking progress across diverse capabilities. This landscape includes task-specific benchmarks (VQA, captioning, reasoning), multi-task evaluation suites (MMMU, MMBench, MME), and dynamic leaderboards (Vision Arena, Open VLM Leaderboard) that enable apples-to-apples model comparison. Understanding this evaluation infrastructure is critical for assessing VLM capabilities, identifying failure modes, and guiding architectural improvements.

**Key Challenge**: Benchmark saturation. As models rapidly improve, existing benchmarks reach near-perfect performance, necessitating continuous development of more challenging evaluation protocols. GPT-4V achieving 56% on MMMU while scoring 76% on "easy" questions highlights how difficulty stratification reveals model limitations even as aggregate scores improve.

## VQA Benchmarks

### VQAv2 (2017)

**Overview**: Visual Question Answering v2, the de facto standard for open-ended VQA evaluation with 1.1M questions on 200K COCO images.

**Evaluation Protocol**:
- 10 human answers per question (crowd-sourced)
- Consensus voting: Answer accuracy = min(# humans said answer / 3, 1.0)
- Penalizes language biases through answer distribution balancing
- Train/val/test splits with held-out test server

**Question Types**:
- Yes/No (38%): "Is there a cat?" → Binary classification
- Number (12%): "How many people?" → Counting
- Other (50%): "What color is the car?" → Open vocabulary

**Strengths**:
- Large-scale, diverse question coverage
- Human consensus scoring reduces annotation noise
- Balanced answer distribution prevents trivial priors

**Limitations**:
- COCO images bias toward common objects/scenes
- Many questions answerable from image statistics alone
- Approaching saturation (SOTA ~80% accuracy)

**ARR-COC-0-1 Relevance**: VQAv2 tests whether relevance realization correctly allocates visual attention. Questions like "What is the person on the left doing?" require spatial reasoning + object recognition, exposing whether 64-400 token budgets capture task-relevant regions.

### GQA (2019)

**Overview**: 22M questions on 113K images, emphasizing compositional reasoning and visual grounding.

**Key Innovation**: Structured question representation using scene graphs, enabling fine-grained analysis of reasoning capabilities.

**Question Complexity**:
- Compositional: "What color is the shirt of the person holding the umbrella?"
- Spatial: "Is the book to the left or right of the laptop?"
- Logical: "Are both cars the same color?"

**Evaluation Metrics**:
- Binary accuracy (Yes/No questions)
- Open accuracy (multi-choice)
- Consistency score (checks logical consistency across related questions)
- Grounding score (does model attend to correct objects?)

**Advantages**:
- Compositional structure enables diagnostic evaluation
- Grounding annotations reveal attention correctness
- Tests multi-hop reasoning (2-3 reasoning steps)

**Limitations**:
- Synthetic question generation can introduce artifacts
- Scene graph annotations expensive, limits diversity
- Still based on COCO domain

**ARR-COC-0-1 Relevance**: GQA's compositional reasoning tests whether opponent processing correctly balances compress vs particularize tensions. Multi-hop questions like "What is on the table near the window?" require maintaining intermediate visual evidence across reasoning steps.

### OKVQA (2019)

**Overview**: Outside Knowledge VQA - 14K questions requiring external world knowledge beyond visual recognition.

**Question Categories**:
- Knowledge-based: "What sport is this?" (requires knowing sport rules)
- Reasoning: "Why is the person wearing a helmet?" (requires safety knowledge)
- Common sense: "What time of day is it?" (infer from lighting/shadows)

**Key Challenge**: Requires integrating vision with factual/commonsense knowledge databases.

**Evaluation**: Same as VQAv2 (10 human answers, consensus voting)

**Strengths**:
- Tests knowledge integration beyond pure perception
- Realistic information-seeking scenarios
- Exposes memorization vs reasoning tradeoffs

**Limitations**:
- Knowledge requirements implicit, hard to diagnose
- External knowledge scope unbounded
- Small dataset size (14K questions)

**ARR-COC-0-1 Relevance**: Tests whether relevance realization integrates with LLM world knowledge. Questions requiring visual evidence + factual knowledge (e.g., "What country is this currency from?") test vision-language coupling strength.

### TextVQA (2019)

**Overview**: 45K questions on 28K images requiring reading text in images (OCR + reasoning).

**Challenge Type**: "What does the sign say?", "What brand is shown?"

**Evaluation**:
- Requires extracting text from natural images (scene text)
- Combines OCR with reasoning about text content
- Tests VLM's ability to process fine-grained visual details

**Strengths**:
- Realistic text-in-the-wild scenarios
- Tests high-resolution perception capabilities
- Requires precise spatial localization

**Limitations**:
- Heavy OCR component may favor models with explicit text recognition
- Limited to text-present images
- Relatively narrow capability scope

**ARR-COC-0-1 Relevance**: Critical test for adaptive LOD allocation. Reading small text requires high token budgets (256-400 tokens) for relevant patches, while background regions can use minimal budgets (64 tokens). Tests relevance realization's spatial precision.

### VizWiz (2018)

**Overview**: 31K questions from visually impaired users, emphasizing real-world visual assistance scenarios.

**Unique Aspects**:
- Images taken by blind/low-vision users (often poor quality)
- Questions reflect genuine information needs ("What is this product?", "What color is this shirt?")
- ~30% unanswerable due to image quality issues

**Evaluation**: 10 human answers, includes "unanswerable" category

**Strengths**:
- Real-world distribution (not curated datasets)
- Tests robustness to image quality degradation
- Socially impactful use case

**Limitations**:
- Small dataset size
- Heavy bias toward object recognition/reading tasks
- Less emphasis on complex reasoning

**ARR-COC-0-1 Relevance**: Tests robustness of relevance allocation under visual degradation. Blurry/poorly-framed images require adaptive strategies - can the model detect low-quality regions and allocate more tokens to compensate?

## Captioning Benchmarks

### COCO Captions (2015)

**Overview**: 330K images with 5 human-written captions each, the standard benchmark for image captioning.

**Evaluation Metrics**:
- **BLEU** (1-4): N-gram overlap with references
- **METEOR**: Unigram matching with stemming/synonyms
- **ROUGE-L**: Longest common subsequence
- **CIDEr**: Consensus-based TF-IDF weighting (most correlated with human judgment)
- **SPICE**: Semantic propositional content matching (scene graphs)

**Caption Characteristics**:
- Average length: 10-12 words
- Describe salient objects, actions, spatial relationships
- Multiple valid descriptions per image

**Strengths**:
- Large-scale, diverse image coverage
- Multiple references reduce caption variance
- Well-established baseline comparisons

**Limitations**:
- Short captions lack detailed descriptions
- Metrics favor n-gram overlap over semantic correctness
- Tendency toward generic descriptions ("A man standing in a room")

**ARR-COC-0-1 Relevance**: Captioning tests holistic relevance realization - must identify salient objects/actions across entire image. Generic captions ("A person and a dog") vs detailed captions ("A golden retriever catching a frisbee in a park") reveal whether relevance allocation captures fine-grained details.

### Flickr30K (2014)

**Overview**: 31K images with 5 captions each, focused on everyday scenes and events.

**Characteristics**:
- More action-oriented than COCO ("people doing things")
- Richer entity descriptions (people, clothing, activities)
- Popular for image-text retrieval tasks

**Evaluation**: Same metrics as COCO Captions (BLEU, CIDEr, SPICE)

**Strengths**:
- Diverse everyday scenarios
- Detailed entity descriptions enable fine-grained evaluation
- Complementary to COCO's object-centric focus

**Limitations**:
- Smaller scale than COCO
- Limited domain diversity (mostly people-centric scenes)
- Caption quality variance across annotators

**ARR-COC-0-1 Relevance**: Tests whether relevance realization captures human-centric details (clothing, actions, interactions). Requires balancing people-region token budgets vs background context.

### NoCaps (2019)

**Overview**: Novel Object Captioning - 15K images with objects outside COCO training distribution.

**Key Innovation**: Tests generalization to novel object categories not seen during training.

**Evaluation Splits**:
- In-domain: Objects from COCO categories
- Near-domain: Related but distinct categories
- Out-domain: Completely novel object classes

**Metrics**: Same as COCO, reported separately per domain split

**Strengths**:
- Directly tests compositional generalization
- Reveals overfitting to training object categories
- Challenging evaluation (SOTA ~90 CIDEr in-domain, ~70 out-domain)

**Limitations**:
- Smaller dataset than COCO
- Novel object definition somewhat arbitrary
- Still evaluates with n-gram metrics

**ARR-COC-0-1 Relevance**: Critical test for compositional generalization. Can relevance realization handle novel object categories without retraining? Tests whether opponent processing strategies (compress/particularize) transfer across domains.

## Multi-Task Benchmarks

### MMMU (2024)

**Overview**: Massive Multi-discipline Multimodal Understanding - 11.5K college-level questions across 6 disciplines, 30 subjects, 183 subfields.

**Disciplines**:
1. Art & Design (paintings, diagrams, design blueprints)
2. Business (charts, financial tables, graphs)
3. Science (chemical structures, physics diagrams, experiments)
4. Health & Medicine (medical images, anatomy, pathology)
5. Humanities & Social Science (maps, historical timelines, documents)
6. Tech & Engineering (circuit diagrams, blueprints, code screenshots)

**Image Types** (30 heterogeneous formats):
- Diagrams (3,184 questions)
- Tables (2,267)
- Plots/Charts (840)
- Chemical structures (573)
- Photographs (770)
- Geometric shapes (336)
- Sheet music (335)
- Medical images (272)
- MRI/CT/X-rays (198)
- Mathematical notations (133)

**Difficulty Levels**:
- Easy (GPT-4V: 76.1% accuracy)
- Medium (GPT-4V: 55.6%)
- Hard (GPT-4V: 42.3%)

**Key Insight**: Performance gap narrows in "Hard" category - even advanced models struggle with expert-level tasks.

**Evaluation**: Zero-shot, multiple choice + open-ended QA

**MMMU-Pro (2024)**: Robust version with increased difficulty, reducing benchmark saturation concerns.

**Error Analysis** (GPT-4V):
- 40% perception errors (missed visual details)
- 30% knowledge errors (incorrect domain knowledge)
- 20% reasoning errors (logical mistakes)
- 10% other (ambiguity, unclear questions)

**Strengths**:
- Expert-level evaluation (college knowledge required)
- Unprecedented heterogeneity (30 image types)
- Diagnostic evaluation across disciplines
- Human expert performance baseline

**Limitations**:
- Expensive to create (manual curation by domain experts)
- May favor models with broad pretraining knowledge
- Cultural/educational system biases

**ARR-COC-0-1 Relevance**: Ultimate test for relevance realization across diverse visual formats. Chemical structures require precise detail (high tokens), while conceptual diagrams may need less. Tests whether opponent processing adapts to image type complexity.

From [MMMU Benchmark](https://mmmu-benchmark.github.io/) (accessed 2025-11-16):
- 11.5K meticulously collected questions from college exams, quizzes, textbooks
- GPT-4V achieves only 56% overall accuracy
- Performance on specialized image types like music sheets, chemical structures near random guessing
- Tests expert-level perception + deliberate reasoning with domain knowledge

### MMBench (2023)

**Overview**: Systematically-designed objective benchmark for robust VLM evaluation with ~3,000 multiple-choice questions.

**Design Principles**:
- CircularEval: Rotate options to detect position bias
- Multi-language support (English + Chinese)
- Hierarchical ability taxonomy

**Ability Categories**:
- Coarse perception (object recognition, OCR, counting)
- Fine-grained perception (attribute recognition, spatial relationships)
- Logic reasoning (future prediction, physical relations)
- Knowledge reasoning (commonsense, cultural knowledge)

**Evaluation Strategy**: ChatGPT-based matching for open-ended answers (correlates well with exact match)

**Strengths**:
- Systematic coverage of VLM capabilities
- Robust to answer biases (circular evaluation)
- Enables diagnostic analysis (per-category performance)

**Limitations**:
- Relatively small dataset (~3K questions)
- Multiple-choice format easier than open-ended
- ChatGPT matching introduces evaluation dependency

**ARR-COC-0-1 Relevance**: Hierarchical ability taxonomy aligns with relevance realization levels. Fine-grained perception questions (attribute recognition) require high-token allocation to specific regions, while coarse perception (object recognition) may succeed with lower budgets.

### MME (2023)

**Overview**: Comprehensive evaluation benchmark measuring perception AND cognition abilities separately.

**Structure**:
- **Perception Tasks** (10 tasks): Existence, count, position, color, OCR, commonsense reasoning, numerical calculation, text translation, code reasoning
- **Cognition Tasks** (4 tasks): Commonsense reasoning, numerical calculation, text translation, code reasoning

**Scoring**: Accuracy + accuracy_plus (stricter, penalizes false positives)

**Key Innovation**: Separate perception vs cognition scores reveal whether failures stem from vision or reasoning.

**Task Examples**:
- Existence: "Is there a dog in the image?" (tests basic object detection)
- Position: "Is the cat on the left or right?" (tests spatial reasoning)
- OCR: "What does the text say?" (tests text recognition)
- Commonsense: "What season is it?" (tests visual reasoning)

**Strengths**:
- Diagnostic separation of perception vs cognition
- Simple yes/no format reduces evaluation complexity
- Fast evaluation (smaller dataset than MMMU)

**Limitations**:
- Binary questions less challenging than open-ended
- Limited task diversity compared to MMMU
- May favor models with strong binary classification

**ARR-COC-0-1 Relevance**: Perception/cognition separation tests whether relevance allocation serves downstream reasoning. High perception scores + low cognition scores suggest visual tokens capture details but fail to support reasoning.

### SEED-Bench (2023)

**Overview**: 19K multiple-choice questions with human-verified answers, emphasizing spatial-temporal understanding.

**Dimensions**:
- Scene understanding (indoor, outdoor, lighting)
- Instance identity (celebrities, landmarks, brands)
- Instance attributes (color, shape, material)
- Instance location (spatial relationships, depth)
- Instance counting
- Spatial relation (left/right, above/below)
- Instance interaction (human-object, object-object)
- Visual reasoning (physical, social)
- Text understanding (OCR, scene text)

**Evaluation**: Accuracy on multiple-choice questions with 4 options

**Strengths**:
- Fine-grained dimension coverage
- Human-verified ground truth (high quality)
- Balanced difficulty across dimensions

**Limitations**:
- Multiple-choice format (easier than open-ended)
- COCO-centric domain
- Limited video/temporal reasoning

**ARR-COC-0-1 Relevance**: Spatial relation questions directly test relevance allocation precision. "Is the cup to the left of the book?" requires precisely allocating tokens to both objects + their relationship, testing opponent processing's spatial awareness.

## Reasoning Benchmarks

### CLEVR (2017)

**Overview**: 700K synthetic questions on 100K rendered 3D scenes, testing compositional visual reasoning.

**Question Structure**:
- Compositional templates: "What size is the cylinder that is left of the brown metal thing?"
- Reasoning chains: Query → filter → relate → count/compare
- Unambiguous ground truth (synthetic generation)

**Question Types**:
- Existence: "Is there a red cube?"
- Counting: "How many metal spheres are there?"
- Comparison: "Is the red cube larger than the blue sphere?"
- Querying: "What color is the large metal cube?"

**Strengths**:
- Perfect ground truth (no annotation noise)
- Systematic compositional structure
- Diagnostic evaluation (can trace reasoning steps)
- Controls for visual biases (synthetic rendering)

**Limitations**:
- Synthetic images (limited real-world transfer)
- Simple object types (cubes, spheres, cylinders)
- Questions follow rigid templates

**ARR-COC-0-1 Relevance**: Tests pure compositional reasoning without visual complexity confounds. Can relevance realization handle multi-step reasoning ("the cylinder left of the brown thing") without real-world visual variability?

### NLVR2 (2019)

**Overview**: Natural Language Visual Reasoning - 107K sentence-image pair examples requiring reasoning about image pairs.

**Task**: Given two images and a sentence, determine if sentence is true for the image pair.

**Examples**:
- "One image shows exactly two dogs" → Compare counts across images
- "The left image contains more apples than the right" → Cross-image comparison
- "Both images show the same type of animal" → Category matching

**Evaluation**: Binary accuracy (true/false)

**Strengths**:
- Tests cross-image reasoning (not single-image VQA)
- Natural language descriptions (not templates)
- Requires maintaining state across images

**Limitations**:
- Binary classification easier than open-ended QA
- Limited to image pairs (not arbitrary N-way comparison)
- Smaller scale than VQAv2

**ARR-COC-0-1 Relevance**: Tests whether relevance realization maintains separate allocations for multiple images. Cross-image comparison requires balanced token budgets across both images, testing opponent processing's resource distribution.

### Winoground (2022)

**Overview**: 400 examples testing compositional understanding through hard negatives.

**Task Format**: Given two images and two captions, match correct image-caption pairs.

**Key Challenge**: Captions differ by single word/phrase that completely changes meaning:
- Caption 1: "A dog chasing a cat"
- Caption 2: "A cat chasing a dog"
- Image 1: dog→cat chase
- Image 2: cat→dog chase

**Evaluation**: Accuracy matching both pairs correctly (very challenging, ~30% for many VLMs)

**Strengths**:
- Tests fine-grained compositional understanding
- Hard negatives prevent shortcut learning
- Exposes word order/syntax sensitivity

**Limitations**:
- Very small dataset (400 examples)
- Binary matching task (limited evaluation scope)
- May be too difficult for current models

**ARR-COC-0-1 Relevance**: Extreme test for relevance precision. Single-word differences require precisely allocating tokens to relationship-critical regions (who's chasing whom?), testing whether opponent processing captures subtle compositional differences.

## Zero-Shot Transfer Evaluation

### Cross-Dataset Generalization

**Paradigm**: Train on dataset A, evaluate on dataset B without fine-tuning.

**Common Transfers**:
- VQAv2 → GQA (tests compositional generalization)
- VQAv2 → VizWiz (tests robustness to image quality)
- COCO Captions → Flickr30K (tests domain shift)
- VQAv2 → OKVQA (tests knowledge integration)

**Metrics**: Direct accuracy drop vs in-domain performance

**Typical Results**:
- VQAv2 → GQA: 5-10% accuracy drop (compositional gap)
- VQAv2 → VizWiz: 10-15% drop (quality robustness gap)
- COCO → Flickr: 2-5 CIDEr drop (domain shift minimal)

**Insights**:
- Models overfit to dataset-specific biases
- Compositional generalization remains challenging
- Robustness to distribution shift varies widely

**ARR-COC-0-1 Relevance**: Tests whether relevance realization strategies transfer across datasets. If opponent processing learns dataset-specific priors (e.g., "VQA images always have centered objects"), zero-shot transfer suffers.

## Leaderboard Analysis

### Vision Arena (LMArena)

**Overview**: Crowdsourced pairwise comparison platform for VLMs (551K+ votes across 81 models as of Nov 2025).

**Methodology**:
- Users submit images + questions
- Two anonymous models generate answers
- Users vote for better response
- Elo ratings computed from pairwise battles

**Top Models (Nov 2025)**:
1. Gemini 2.5 Pro (1249 Elo)
2. GPT-4o latest (1240 Elo)
3. GPT-4.5 preview (1228 Elo)
4. Gemini 2.5 Flash preview (1224 Elo)
5. GPT-5 chat (1222 Elo)

**Observations**:
- Proprietary models dominate top rankings
- Open-source leaders: Qwen3-VL-235B (1204 Elo, rank 12)
- Large gap between top proprietary and open models (~40 Elo)

**Strengths**:
- Real user preference (not benchmark gaming)
- Diverse question distribution (not curated)
- Continuous evaluation (new models added dynamically)

**Limitations**:
- User preference ≠ objective correctness
- Vote quality varies (casual users vs experts)
- Presentation bias (formatting, verbosity influence votes)

From [Vision Arena Leaderboard](https://lmarena.ai/leaderboard/vision) (accessed 2025-11-16):
- 551,420 total votes across 81 models
- Gemini 2.5 Pro leads with 1249 Elo score
- Open-source best: Qwen3-VL-235B at rank 12 (1204 Elo)
- Large proprietary/open gap (~40-50 Elo points)

**ARR-COC-0-1 Relevance**: Real-world preference testing for relevance-driven architectures. If ARR-COC-0-1 produces visually-grounded, query-relevant answers, human voters should prefer them over generic responses.

### Open VLM Leaderboard (HuggingFace)

**Overview**: Standardized benchmark aggregation across multiple datasets with reproducible evaluation.

**Evaluated Benchmarks**:
- MMMU, MMBench, MME
- TextVQA, VQAv2, GQA
- COCO Captions (CIDEr)
- AI2D (science diagrams)

**Aggregate Scoring**: Weighted average across benchmarks (equal weighting or custom)

**Features**:
- Open-source models only (reproducible evaluation)
- Standardized evaluation protocol
- Regular updates with new model submissions

**Current Leaders** (open-source):
- InternVL-Chat-V1.5 series
- Qwen-VL models
- LLaVA-NeXT series

**Strengths**:
- Reproducible (open weights + public eval code)
- Comprehensive coverage (7+ benchmarks)
- Community-driven model submissions

**Limitations**:
- Open-source only (no GPT-4V, Gemini comparison)
- Aggregate scores hide per-task performance
- Benchmark selection bias (easier datasets weighted equally to harder)

From [Open VLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) (accessed 2025-11-16):
- Aggregates scores across MMMU, MMBench, TextVQA, VQAv2, and others
- Open-source models only with reproducible evaluation
- Enables fair comparison across architectural approaches

**ARR-COC-0-1 Relevance**: Provides baseline comparison for ARR-COC-0-1 against state-of-the-art open VLMs. If relevance realization reduces token budgets while maintaining accuracy, ARR-COC-0-1 could achieve competitive scores with better efficiency.

### Benchmark Saturation Issues

**Problem**: As models improve rapidly, benchmarks reach near-ceiling performance, reducing discriminative power.

**Examples**:
- VQAv2: SOTA ~80% (approaching human agreement ~83%)
- COCO Captions: CIDEr >140 (exceeds some human references)
- GQA: SOTA ~65% (improvements slowing)

**Saturation Indicators**:
- Year-over-year improvements <2%
- Top-5 models within 1% accuracy
- Error analysis reveals mostly annotation noise, not model failures

**Responses to Saturation**:
1. **Harder benchmarks**: MMMU-Pro, Winoground (intentionally challenging)
2. **Adversarial examples**: Hard negatives, counterfactuals
3. **Procedural generation**: Infinite test sets (e.g., synthetic CLEVR variants)
4. **Real-world deployment**: Production A/B testing over static benchmarks

**From Research**:

From [Benchmark Saturation Study](https://gradientscience.org/platinum-benchmarks/) (accessed 2025-11-16):
- Benchmarks considered saturated even before 100% accuracy
- Year-over-year improvements <2% signal practical saturation
- Creating harder versions (MMMU-Pro) extends benchmark lifespan

From [VLM Evaluation Challenges](https://arxiv.org/html/2501.02189v3) (accessed 2025-11-16):
- Modern VLM benchmarks moving beyond simple VQA to complex multi-task evaluation
- Emphasis on compositional reasoning, expert knowledge, multi-image understanding
- Saturation driving development of more challenging protocols

**ARR-COC-0-1 Relevance**: Benchmark saturation motivates efficiency-focused evaluation. Instead of chasing marginal accuracy gains on saturated benchmarks, ARR-COC-0-1 aims for competitive accuracy with 3-5× token reduction. New efficiency metrics (accuracy per token, FLOPs per point) become relevant evaluation axes.

## Creating Custom Benchmarks

### When to Create Custom Benchmarks

**Scenarios**:
1. **Domain-specific applications**: Medical imaging, satellite analysis, robotics
2. **New capabilities**: Novel architectural features requiring targeted evaluation
3. **Deployment requirements**: Production-specific failure modes not covered by academic benchmarks

**Example**: ARR-COC-0-1 Custom Benchmark
- **Relevance allocation quality**: Does model allocate high tokens to query-relevant regions?
- **Opponent processing efficiency**: Compress/particularize balance across image types
- **Adaptive LOD correctness**: Token budgets match visual complexity (64 for simple, 400 for complex)

### Benchmark Design Principles

**1. Clear Evaluation Criteria**:
- Unambiguous success/failure (exact match, F1, human agreement)
- Reproducible scoring (no subjective judgment)
- Stratified difficulty (easy/medium/hard splits)

**2. Diverse Coverage**:
- Multiple task types (perception, reasoning, knowledge)
- Varied image types (photos, diagrams, charts)
- Balanced difficulty distribution

**3. Quality Control**:
- Expert annotation (domain specialists for specialized benchmarks)
- Multiple annotators per example (consensus voting)
- Adversarial validation (catch annotation errors)

**4. Avoiding Biases**:
- Answer distribution balancing (prevent trivial priors)
- Hard negatives (force fine-grained discrimination)
- Out-of-distribution test set (prevent overfitting)

### Example: ARR-COC-0-1 Benchmark Suite

**Goal**: Evaluate relevance realization quality + efficiency tradeoffs

**Tasks**:

**1. Relevance Allocation Evaluation**:
- Given image + query, collect human annotations of "relevant regions"
- Measure IoU between high-token ARR-COC patches and human annotations
- Metrics: Precision@K (top-K patches hit relevant regions), Recall@K

**2. Adaptive LOD Correctness**:
- Create image set with varying complexity (simple shapes → complex scenes)
- Ground truth: Expert-labeled "minimum tokens needed" per region
- Measure correlation between ARR-COC budgets and ground truth
- Metrics: Spearman correlation, budget efficiency (accuracy / avg tokens)

**3. VQA Accuracy vs Token Budget**:
- Evaluate VQAv2, GQA, TextVQA at different token budgets (64, 144, 256, 400)
- Compare ARR-COC vs fixed-budget baselines
- Metrics: Pareto frontier (accuracy vs tokens), efficiency ratio

**4. Opponent Processing Analysis**:
- Analyze compress/particularize decisions across image types
- Diagrams should favor compress (low tokens, spatial relationships)
- Photos should favor particularize (high tokens, fine details)
- Metrics: Token allocation distribution by image type

**5. Cross-Dataset Generalization**:
- Train on VQAv2, evaluate on GQA, TextVQA, VizWiz
- Measure whether relevance strategies transfer
- Metrics: Zero-shot accuracy, relative performance drop

**Expected Results**:
- ARR-COC-0-1 achieves comparable accuracy to fixed-budget baselines
- Uses 40-60% fewer tokens on average (3-5× efficiency gain)
- Relevance allocation correlates with human judgments (IoU > 0.7)
- Adaptive LOD matches image complexity (Spearman r > 0.8)

## Best Practices for VLM Benchmarking

### 1. Multi-Benchmark Evaluation

**Don't rely on single benchmark** - performance may be dataset-specific.

**Recommended Suite**:
- **VQA**: VQAv2 (general), GQA (compositional), TextVQA (OCR)
- **Captioning**: COCO Captions (standard), NoCaps (generalization)
- **Multi-task**: MMMU (expert-level), MMBench (diagnostic)
- **Reasoning**: CLEVR (synthetic control), NLVR2 (cross-image)

### 2. Stratified Analysis

**Report performance by**:
- Difficulty level (easy/medium/hard)
- Question type (yes/no, counting, open-ended)
- Image type (photos, diagrams, charts)
- Answer length (short vs long captions)

### 3. Error Analysis

**Go beyond aggregate accuracy**:
- Sample 100-200 failures
- Categorize errors (perception, knowledge, reasoning)
- Identify systematic failure modes
- Trace errors to architectural components

**Example Error Categories** (from MMMU GPT-4V analysis):
- **Perception errors** (40%): Missed visual details, OCR failures
- **Knowledge errors** (30%): Incorrect domain knowledge, hallucinated facts
- **Reasoning errors** (20%): Logical mistakes, compositional failures
- **Other** (10%): Ambiguous questions, annotation errors

### 4. Human Baselines

**Include human performance**:
- Expert humans (domain specialists)
- Crowd workers (general population)
- Inter-annotator agreement (noise ceiling)

**Why**: Contextualizes model performance. If models approach human agreement levels, benchmark may be saturated.

### 5. Efficiency Metrics

**Beyond accuracy, measure**:
- **Tokens per sample**: Visual + text token counts
- **Latency**: Inference time (ms per sample)
- **FLOPs**: Computational cost
- **Memory**: Peak GPU memory usage

**Efficiency ratios**:
- Accuracy per token: `accuracy / avg_tokens`
- Points per TFLOP: `accuracy_delta / TFLOP_delta`

### 6. Zero-Shot Evaluation

**Prefer zero-shot over fine-tuned** for generalization assessment:
- Tests out-of-the-box capabilities
- Reveals overfitting to training distributions
- More realistic for practical deployment

**When to use fine-tuning**:
- Domain-specific applications (medical, satellite)
- Benchmark requires task-specific prompts
- Comparing adaptation efficiency (few-shot, LoRA)

## ARR-COC-0-1 Benchmark Strategy

### Evaluation Protocol

**Primary Benchmarks**:
1. **VQAv2**: Test general VQA capability
2. **GQA**: Test compositional reasoning
3. **TextVQA**: Test OCR + high-resolution perception
4. **COCO Captions**: Test holistic scene understanding

**Efficiency Benchmarks**:
1. **Token budget ablations**: 64, 144, 256, 400 tokens per patch
2. **Pareto frontier analysis**: Accuracy vs tokens across benchmarks
3. **Relevance allocation quality**: IoU with human annotations
4. **Adaptive LOD correctness**: Correlation with ground truth budgets

**Comparison Baselines**:
- **Fixed-budget VLMs**: LLaVA (256 tokens), Qwen-VL (256 tokens)
- **Adaptive models**: Flamingo (Perceiver ~256 tokens), BLIP-2 (Q-Former 32 queries)
- **SOTA VLMs**: GPT-4V, Gemini Pro (proprietary, no token control)

### Success Criteria

**Accuracy Targets**:
- VQAv2: ≥75% (competitive with fixed-budget baselines)
- GQA: ≥65% (tests compositional generalization)
- TextVQA: ≥70% (tests adaptive high-res allocation)
- COCO CIDEr: ≥130 (tests holistic understanding)

**Efficiency Targets**:
- Average tokens: 180 (vs 256 for baselines = 30% reduction)
- Pareto frontier: Dominate baselines (higher accuracy at same tokens)
- Relevance IoU: ≥0.7 (correlates with human judgments)
- LOD correlation: ≥0.8 (matches complexity requirements)

**Ablation Studies**:
1. Remove opponent processing → measure impact on compress/particularize balance
2. Fixed LOD (no adaptation) → measure impact on efficiency
3. Remove relevance scoring → measure impact on allocation quality
4. Vary K (number of patches) → characterize K vs accuracy tradeoff

### Reporting Standards

**Full Evaluation Report Includes**:
1. **Aggregate scores**: Per-benchmark accuracy, average tokens
2. **Stratified analysis**: Performance by difficulty, question type, image type
3. **Efficiency metrics**: Accuracy per token, Pareto curves
4. **Error analysis**: 100+ sampled failures with categorization
5. **Ablation studies**: Component contribution analysis
6. **Qualitative examples**: Successful relevance allocations + failure cases

**Visualization**:
- Pareto frontier plots (accuracy vs tokens)
- Relevance allocation heatmaps (tokens per patch)
- Error distribution charts (perception, knowledge, reasoning)
- Cross-benchmark comparison radar charts

## Sources

**Benchmark Documentation**:
- [MMMU Benchmark](https://mmmu-benchmark.github.io/) - 11.5K college-level multimodal questions (accessed 2025-11-16)
- [Vision Arena Leaderboard](https://lmarena.ai/leaderboard/vision) - Crowdsourced VLM ranking via pairwise battles (accessed 2025-11-16)
- [Open VLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) - HuggingFace benchmark aggregation (accessed 2025-11-16)

**Web Research**:
- [Benchmark Saturation Analysis](https://gradientscience.org/platinum-benchmarks/) - Study on benchmark saturation indicators (accessed 2025-11-16)
- [VLM Evaluation Survey](https://arxiv.org/html/2501.02189v3) - Comprehensive overview of VLM benchmark landscape (accessed 2025-11-16)
- [VLM Benchmarks Overview](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/vlm.html) - EvalScope VLM benchmark list (accessed 2025-11-16)

**Reddit Discussions**:
- [LocalLLaMA VLM Rankings Discussion](https://www.reddit.com/r/LocalLLaMA/comments/1l9zym8/whats_your_local_vision_model_rankings_and_local/) - Community rankings of vision models (accessed 2025-11-16)
- [Benchmark Gaming Issues](https://www.reddit.com/r/LocalLLaMA/comments/1hs6ftc/killed_by_llm_i_collected_data_on_ai_benchmarks/) - Discussion on benchmark saturation and gaming (accessed 2025-11-16)

**Additional Resources**:
- [Clarifai VLM Benchmark Guide](https://www.clarifai.com/blog/best-vision-language-models-vlms-for-image-classification-performance-benchmarks) - Practical VLM benchmarking (accessed 2025-11-16)
- [LearnOpenCV Evaluation Metrics](https://learnopencv.com/vlm-evaluation-metrics/) - Comprehensive VLM metrics guide (accessed 2025-11-16)
