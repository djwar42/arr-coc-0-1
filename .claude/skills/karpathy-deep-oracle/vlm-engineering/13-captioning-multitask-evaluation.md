# VLM Captioning & Multi-Task Evaluation: Metrics, Protocols, and Benchmarks

## Overview

Image captioning and multi-task evaluation represent critical dimensions of VLM assessment that go beyond simple accuracy metrics. Unlike discriminative tasks (VQA, classification), generative captioning requires evaluating semantic richness, linguistic fluency, and factual grounding simultaneously. Multi-task benchmarks provide holistic views of model capabilities across diverse vision-language challenges.

**Key Challenge**: A model excelling at logical reasoning (VQA) may struggle with semantic richness in captioning. Similarly, high VQA accuracy doesn't guarantee low hallucination rates or strong retrieval performance.

From [LearnOpenCV VLM Evaluation Guide](https://learnopencv.com/vlm-evaluation-metrics/) (accessed 2025-11-16):
- Different tasks require different evaluation approaches
- Captioning metrics measure both surface similarity (BLEU) and semantic correctness (SPICE)
- Multi-task benchmarks reveal model strengths/weaknesses across diverse capabilities

---

## Section 1: Image Captioning Evaluation Metrics (200 lines)

### 1.1 BLEU (Bilingual Evaluation Understudy)

**Purpose**: Measures n-gram overlap between generated and reference captions.

**Method**:
- Originally designed for machine translation
- Geometric mean of modified n-gram precisions with brevity penalty
- Standard variants: BLEU-1, BLEU-2, BLEU-3, BLEU-4 (cumulative)

**Equation**:

```
BLEU = BP × exp(Σ(n=1 to N) w_n × log p_n)

where:
- BP = Brevity Penalty = {1 if c > r, e^(1-r/c) if c ≤ r}
- c = candidate caption length
- r = effective reference length
- p_n = modified n-gram precision for n-grams of size n
- w_n = weights (typically 0.25 for BLEU-4)
```

From [MS COCO Captions Paper](https://www.cs.cmu.edu/~jeanoh/16-785/papers/chen-arxiv2015-mscoco-metrics.pdf) (Chen et al., 2015):
- BLEU-4 is standard for caption evaluation
- Requires 5 reference captions per image (MS-COCO standard)
- Cumulative score: considers 1-grams through 4-grams
- Brevity penalty prevents gaming metric with short captions

**Strengths**:
- Fast to compute
- Widely adopted baseline
- Works well for comparing similar caption styles

**Weaknesses**:
- Focuses on surface similarity, not semantics
- Can't detect hallucinations
- Penalizes valid paraphrases
- Doesn't reward creativity or descriptiveness

**Example**:
```
Reference: "A dog playing with a frisbee in the park"
Generated: "A dog catching a frisbee outdoors"
BLEU-4: ~0.40 (decent overlap despite different wording)

Generated (hallucination): "A cat playing with a ball in the park"
BLEU-4: ~0.50 (higher score despite wrong object!)
```

---

### 1.2 METEOR (Metric for Evaluation with Explicit ORdering)

**Purpose**: Addresses BLEU's limitations by incorporating synonyms, stemming, and recall.

**Method**:
- Unigram-based matching with WordNet synonyms
- Harmonic mean of precision and recall (favors recall)
- Fragmentation penalty for word order differences

**Equation**:

```
METEOR = F_mean × (1 - Penalty)

F_mean = (10 × P × R) / (R + 9P)

where:
- P = unigram precision (exact/stem/synonym matches)
- R = unigram recall
- Penalty = fragmentation penalty for word order
```

From [METEOR Paper](https://aclanthology.org/W05-0909.pdf) (Banerjee & Lavie, 2005):
- Correlates better with human judgment than BLEU
- Recall-weighted: F_mean uses 9:1 ratio favoring recall
- Matches on three levels: exact, stemmed, synonym (via WordNet)

**Strengths**:
- Handles paraphrases better than BLEU
- Considers word order via fragmentation penalty
- Better human judgment correlation

**Weaknesses**:
- Still unigram-based (ignores phrase structure)
- Slower than BLEU (WordNet lookups)
- Language-dependent (requires WordNet)

---

### 1.3 ROUGE-L (Longest Common Subsequence)

**Purpose**: Measures sentence-level similarity via longest common subsequence.

**Method**:
- Finds longest matching word sequence (allows gaps)
- F-score combining LCS-based precision and recall
- Focuses on word order preservation

**Equation**:

```
F_lcs = ((1 + β²) × R_lcs × P_lcs) / (R_lcs + β² × P_lcs)

where:
- R_lcs = LCS(X,Y) / m  (recall: coverage of reference)
- P_lcs = LCS(X,Y) / n  (precision: relevance of candidate)
- X = reference caption (length m)
- Y = candidate caption (length n)
- LCS(X,Y) = length of longest common subsequence
- β = balance parameter (β=1 for F1)
```

From [ROUGE Paper](https://aclanthology.org/W04-1013.pdf) (Lin, 2004):
- Originally designed for summarization evaluation
- Captures fluency better than BLEU (requires contiguous matching)
- ROUGE-L focuses on sentence-level structure

**Strengths**:
- Rewards grammatical coherence
- No pre-defined n-gram size
- Better for longer captions

**Weaknesses**:
- Ignores non-sequential matches
- Doesn't capture semantics
- Can reward verbose captions

---

### 1.4 CIDEr (Consensus-based Image Description Evaluation)

**Purpose**: Weights n-grams by TF-IDF to focus on distinctive/important words.

**Method**:
- TF-IDF weighted n-gram matching
- Downweights common words ("the", "a"), upweights specific terms
- Average cosine similarity across reference captions

**Equation**:

```
CIDEr_n(c, R) = (1/|R|) × Σ(r∈R) [g^n(c) · g^n(r)] / [||g^n(c)|| × ||g^n(r)||]

CIDEr(c, R) = Σ(n=1 to N) w_n × CIDEr_n(c, R)

where:
- c = candidate caption
- R = set of reference captions
- g^n(c) = TF-IDF weighted n-gram vector for candidate
- g^n(r) = TF-IDF weighted n-gram vector for reference
- N = max n-gram length (typically 4)
- w_n = uniform weights (e.g., 0.25 for n=1..4)
```

From [CIDEr Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vedantam_CIDEr_Consensus-Based_Image_2015_CVPR_paper.pdf) (Vedantam et al., 2015):
- Designed specifically for image captioning (not MT)
- TF-IDF emphasizes image-specific content words
- Aggregates consensus across multiple references
- Correlation with human judgment: 0.92 (higher than BLEU, METEOR)

**Strengths**:
- Best correlation with human judgment
- Focuses on semantically important words
- Robust to caption length variations
- Standard metric for MS-COCO leaderboard

**Weaknesses**:
- Requires large reference corpus for TF-IDF
- Doesn't detect hallucinations
- Can be gamed by copying reference word distributions

---

### 1.5 SPICE (Semantic Propositional Image Caption Evaluation)

**Purpose**: Evaluates semantic content via scene graph matching.

**Method**:
- Parse captions into scene graphs (objects, attributes, relations)
- Extract semantic tuples from graphs
- F1-score over tuple matching (precision & recall)

**Equation**:

```
SPICE(c, R) = F1(c, R) = (2 × P(c,R) × R(c,R)) / (P(c,R) + R(c,R))

where:
- G(c) = scene graph for candidate caption
- G(R) = union of scene graphs for references
- T(G) = semantic tuples extracted from graph
- P(c,R) = |T(G(c)) ∩ T(G(R))| / |T(G(c))|  (precision)
- R(c,R) = |T(G(c)) ∩ T(G(R))| / |T(G(R))|  (recall)
```

From [SPICE Paper](https://arxiv.org/abs/1607.08822) (Anderson et al., 2016):
- Parses captions using dependency parsing + semantic role labeling
- Extracts tuples: objects (dog), attributes (brown dog), relations (dog on grass)
- Measures semantic correctness, not surface similarity
- Better correlation with human judgment than n-gram metrics

**Strengths**:
- Evaluates meaning, not just wording
- Detects missing/incorrect objects
- Can identify attribute/relation errors
- Closer to human evaluation process

**Weaknesses**:
- Depends on parser quality
- Slower than n-gram metrics
- Errors compound (parsing → tuple extraction → matching)
- Doesn't evaluate fluency/grammar

---

### 1.6 Metric Comparison Summary

| Metric | Focus | Speed | Human Corr. | Detects Hallucination? |
|--------|-------|-------|-------------|------------------------|
| **BLEU** | N-gram overlap | Fast | Moderate (0.6-0.7) | No |
| **METEOR** | Synonyms, stemming | Medium | Good (0.7-0.8) | No |
| **ROUGE-L** | Sequence matching | Fast | Moderate (0.65-0.75) | No |
| **CIDEr** | TF-IDF weighted | Medium | Excellent (0.92) | No |
| **SPICE** | Semantic content | Slow | Excellent (0.90) | Partial |

**Standard Practice** (from MS-COCO evaluation server):
- Report all five metrics
- CIDEr is primary metric for ranking
- SPICE provides semantic grounding check
- BLEU-4 for historical comparison

---

## Section 2: MS-COCO Captioning Evaluation Protocol (150 lines)

### 2.1 Dataset Overview

**MS-COCO (Microsoft Common Objects in Context) Captions**:
- 330,000+ images of complex real-world scenes
- 5 independent human-written reference captions per image
- Train: 118k images (590k captions)
- Val: 5k images (25k captions)
- Test: 40k images (200k captions, held out)

From [MS COCO Captions Collection Paper](https://www.cs.cmu.edu/~jeanoh/16-785/papers/chen-arxiv2015-mscoco-metrics.pdf) (Chen et al., 2015):
- Reference diversity crucial for robust evaluation
- 5 captions capture multiple valid descriptions
- Average caption length: 11.8 words
- Vocabulary: ~9,000 unique words

**Why 5 References?**
- Single reference insufficient: "A dog" vs "A canine" vs "The puppy"
- Multiple references create evaluation target distribution
- Allows metrics to reward any valid description
- Reduces annotator bias

---

### 2.2 Official Evaluation Protocol

**Submission Format**:
```json
[
  {
    "image_id": 123456,
    "caption": "A person riding a skateboard on a ramp"
  },
  {
    "image_id": 789012,
    "caption": "Two dogs playing with a frisbee in a park"
  }
]
```

**Evaluation Server**:
- Hosted on [EvalAI](https://eval.ai/web/challenges/challenge-page/355/overview)
- Submissions evaluated against held-out test set
- Returns all 5 standard metrics (BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr, SPICE)
- Leaderboard ranks by CIDEr score

From [MS COCO Evaluation API](https://github.com/tylin/coco-caption):
- Open-source Python implementation
- Handles multiple references automatically
- Normalizes text (lowercase, punctuation removal)
- Computes per-image scores and corpus-level averages

**Normalization Steps**:
1. Lowercase all text
2. Remove punctuation
3. Tokenize (Penn Treebank tokenizer)
4. Remove extra whitespace

---

### 2.3 Multi-Reference Evaluation Logic

**Key Insight**: Metrics aggregate over multiple references differently.

**BLEU, METEOR, ROUGE-L**:
- Compare candidate against each reference
- Take maximum score (best match)
- Rationale: Any valid description should be rewarded

**CIDEr**:
- Average cosine similarity across all references
- Rationale: Consensus-based scoring

**SPICE**:
- Union scene graph from all references
- Single precision/recall computation
- Rationale: Semantic content should match any valid description

**Example**:
```python
# Candidate caption
candidate = "A black dog playing fetch"

# 5 Reference captions
references = [
    "A dog chasing a ball in the park",
    "A black labrador retrieving a tennis ball",
    "A playful dog running after a thrown ball",
    "A dog playing fetch outdoors",
    "An energetic dog catching a ball"
]

# BLEU: max over 5 references
bleu_scores = [bleu(candidate, ref) for ref in references]
final_bleu = max(bleu_scores)  # Take best match

# CIDEr: average over 5 references
cider_scores = [cider(candidate, ref) for ref in references]
final_cider = mean(cider_scores)  # Consensus

# SPICE: union graph matching
union_graph = merge_graphs([parse(ref) for ref in references])
final_spice = f1_score(parse(candidate), union_graph)
```

---

### 2.4 Corpus-Level vs Image-Level Metrics

**Corpus-Level (Standard)**:
- Average metric across all test images
- Reported on leaderboards
- Formula: `Σ score(image_i) / N`

**Image-Level (Diagnostic)**:
- Per-image breakdown for error analysis
- Identifies specific failure modes
- Useful for debugging

**Example Output**:
```
Corpus-Level Scores:
- BLEU-4: 0.356
- METEOR: 0.277
- ROUGE-L: 0.568
- CIDEr: 1.133
- SPICE: 0.214

Image-Level Examples:
Image 123456: BLEU=0.45, CIDEr=1.2, SPICE=0.28 (Good)
Image 789012: BLEU=0.12, CIDEr=0.4, SPICE=0.08 (Poor - hallucination)
```

---

### 2.5 Common Pitfalls

**Pitfall 1: Caption Length Bias**
- Very short captions can achieve high precision
- Brevity penalty in BLEU mitigates this
- METEOR's recall-weighting helps

**Pitfall 2: Copying Reference Style**
- Models can overfit to caption templates
- High BLEU doesn't guarantee semantic correctness
- Use SPICE for grounding check

**Pitfall 3: Hallucination Not Detected**
- CIDEr rewards TF-IDF weighted words
- Can score high by mentioning common objects
- Combine with POPE or CHAIR for hallucination detection

From [Image Captioning Evaluation Paper](https://aclanthology.org/2024.ecnlp-1.9.pdf) (Li et al., 2024):
- Current metrics inadequate for modern VLMs
- Need better hallucination detection
- Human evaluation still gold standard

---

## Section 3: Flickr30K and Other Caption Benchmarks (100 lines)

### 3.1 Flickr30K Captions

**Dataset**:
- 31,783 images from Flickr
- 5 captions per image (158k total)
- Focus: People engaged in activities
- Simpler scenes than MS-COCO

From [Flickr30K Paper](https://www.ijcai.org/proceedings/2025/1180.pdf) (accessed 2025-11-16):
- Created for caption generation and image-text retrieval
- Standard split: 29k train, 1k val, 1k test
- Average caption length: 12.3 words
- Smaller scale than COCO, but higher quality annotations

**Evaluation Protocol**:
- Same metrics as COCO (BLEU, METEOR, ROUGE-L, CIDEr, SPICE)
- 5 reference captions per image
- Often used for image-text retrieval (Recall@K)

---

### 3.2 NoCaps (Novel Object Captioning)

**Purpose**: Test generalization to novel objects unseen during training.

**Dataset**:
- 15,100 images from Open Images
- 5 captions per image
- Split by object novelty:
  - In-domain: Only COCO objects
  - Near-domain: Some novel objects
  - Out-domain: Majority novel objects

From [NoCaps Paper](https://arxiv.org/abs/1812.08658) (Agrawal et al., 2019):
- Tests compositional generalization
- Exposes caption dataset bias
- Standard evaluation: CIDEr per domain

**Key Finding**:
- Models trained on COCO struggle with out-domain (CIDEr drops 30-50%)
- Demonstrates limitation of closed-vocabulary captioning

---

### 3.3 Conceptual Captions (CC3M, CC12M)

**Dataset**:
- CC3M: 3.3M image-caption pairs from web
- CC12M: 12M pairs (larger, noisier)
- Automatically collected via alt-text filtering

**Use Case**:
- Pre-training vision-language models
- Not evaluation benchmark (noisy captions)
- Provides scale for contrastive learning

**Evaluation**:
- Typically not used for caption generation benchmarking
- Used for image-text retrieval tasks
- Zero-shot transfer to COCO/Flickr30K

---

## Section 4: Visual Grounding Evaluation (100 lines)

### 4.1 RefCOCO Family (Referring Expression Comprehension)

**Task**: Localize object given textual description.

**Datasets**:
- **RefCOCO**: 142k expressions, 50k objects, location-based ("left person")
- **RefCOCO+**: No location words, attribute-based ("person in blue shirt")
- **RefCOCOg**: Longer, grammatically complex descriptions

**Evaluation Metric**: Accuracy @ IoU Threshold

```
IoU = Area(Predicted Box ∩ Ground-Truth Box) / Area(Predicted Box ∪ Ground-Truth Box)

Accuracy@0.5 = (Predictions with IoU > 0.5) / Total Expressions × 100%
```

From [RefCOCO Paper](https://arxiv.org/abs/1608.00272) (Yu et al., 2016):
- Standard threshold: IoU ≥ 0.5
- Sometimes report Precision@0.5, 0.6, 0.7, 0.8, 0.9
- Separate evaluation on val/testA/testB splits

**RefCOCO Splits**:
- Val: Same split across all three datasets
- TestA: Multiple people (harder)
- TestB: Multiple objects of same category (harder)

---

### 4.2 Visual Genome Regions

**Task**: Dense captioning - generate descriptions for all image regions.

**Dataset**:
- 108k images
- 5.4M region descriptions
- Average 50 regions per image

**Evaluation**:
- Meteor score for region captions
- Mean Average Precision (mAP) for region detection

---

### 4.3 GROOViST (Grounding Objects in Visual Storytelling)

**Purpose**: Evaluate object grounding in generated stories.

From [GROOViST Paper](https://aclanthology.org/anthology-files/anthology-files/pdf/emnlp/2023.emnlp-main.202.pdf) (Surikuchi et al., 2023):
- Measures visual grounding quality in story generation
- Checks if mentioned objects actually appear in images
- Novel metric for vision-language storytelling

**Method**:
- Extract object mentions from generated text
- Verify object presence in corresponding images
- Compute grounding precision/recall

---

## Section 5: Image-Text Retrieval Evaluation (100 lines)

### 5.1 Task Overview

**Bidirectional Retrieval**:
- **Image Retrieval**: Given text query, retrieve relevant images
- **Text Retrieval**: Given image query, retrieve relevant captions

**Standard Datasets**:
- COCO (5k test images, 25k captions)
- Flickr30K (1k test images, 5k captions)

---

### 5.2 Recall@K Metric

**Definition**: Percentage of queries where correct match appears in top-K results.

```
Recall@K = (Queries with correct match in top-K) / Total Queries × 100%
```

**Standard K values**: 1, 5, 10

**Example**:
```
Image Query → Text Retrieval:
- Top-1 caption matches? → Recall@1
- Correct caption in top-5? → Recall@5
- Correct caption in top-10? → Recall@10

Text Query → Image Retrieval:
- Target image rank 1? → Recall@1
- Target image in top-5? → Recall@5
```

From [CLIP Paper](https://arxiv.org/abs/2103.00020) (Radford et al., 2021):
- Standard benchmark for vision-language models
- COCO: 5 captions per image creates ambiguity
- Report bidirectional: I2T and T2I separately

---

### 5.3 Mean Reciprocal Rank (MRR)

**Formula**:

```
MRR = (1/N) × Σ(i=1 to N) 1/rank_i

where:
- rank_i = position of correct match for query i
- N = total queries
```

**Interpretation**:
- MRR = 1.0: All queries rank correct match first
- MRR = 0.5: Average rank = 2
- More informative than Recall@K (considers exact rank)

---

### 5.4 NDCG (Normalized Discounted Cumulative Gain)

**Purpose**: Account for multiple correct matches (COCO has 5 captions per image).

**Formula**:

```
DCG@K = Σ(i=1 to K) rel_i / log₂(i+1)

NDCG@K = DCG@K / IDCG@K

where:
- rel_i = relevance score at position i (1 if correct, 0 otherwise)
- IDCG@K = ideal DCG (best possible ranking)
```

---

## Section 6: Multi-Task VLM Benchmarks (150 lines)

### 6.1 VLUE (Vision-Language Understanding Evaluation)

**Purpose**: Unified benchmark for diverse VL tasks.

From [VLUE Paper](https://arxiv.org/abs/2205.15237) (Zhou et al., 2022):
- First multi-task benchmark for vision-language understanding
- Covers 8 tasks: VQA, NLVR2, Image-Text Retrieval, Visual Entailment, etc.
- Tests generalization across task types

**Tasks**:
1. Visual Question Answering (VQA)
2. Image-Text Retrieval (COCO, Flickr30K)
3. Visual Reasoning (NLVR2)
4. Visual Entailment (SNLI-VE)
5. Referring Expression Comprehension (RefCOCO)
6. Visual Commonsense Reasoning (VCR)
7. Region Captioning (Visual Genome)
8. Image Captioning (COCO)

**Evaluation**:
- Task-specific metrics (accuracy, recall@K, CIDEr, etc.)
- Aggregate score across all tasks
- Diagnostic: Per-task performance breakdown

---

### 6.2 MMBench (Multi-Modality Benchmark)

**Purpose**: Fine-grained ability assessment across 20 dimensions.

**Dataset**:
- 2,974 multiple-choice questions
- 20 ability dimensions (perception, OCR, reasoning, etc.)
- Circular evaluation protocol (eliminates position bias)

From [MMBench Paper](https://github.com/open-compass/MMBench) (accessed 2025-11-16):
- Uses ChatGPT-based choice extraction for free-form outputs
- Provides ability-wise performance breakdown
- English and Chinese versions

**Evaluation**:
- Accuracy per dimension
- Overall accuracy (aggregate)
- Circular evaluation: Test N times with shifted options

---

### 6.3 SEED-Bench (Spatiotemporal Evaluation)

**Dataset**:
- 24,000 multiple-choice questions
- 27 evaluation dimensions
- Covers spatial, temporal, and reasoning tasks

**Dimensions Include**:
- Scene understanding (recognition, localization, counting)
- Instance reasoning (attributes, relations, physics)
- Visual reasoning (logic, mathematics, science)
- Text understanding (OCR, scene text reasoning)

From [SEED-Bench Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_SEED-Bench_Benchmarking_Multimodal_Large_Language_Models_CVPR_2024_paper.pdf) (Li et al., 2024):
- Systematically designed for comprehensive evaluation
- Fine-grained performance breakdown
- Standard for modern VLM evaluation

---

### 6.4 LLaVA-Bench (In-the-Wild)

**Purpose**: Evaluate instruction-following and conversational abilities.

**Dataset**:
- 90 high-quality image-instruction pairs
- 3 categories:
  - Conversation (multi-turn dialogues)
  - Detail Description (detailed captioning)
  - Complex Reasoning (multi-step reasoning)

**Evaluation**:
- GPT-4 as judge (rates 1-10)
- Compares against reference answers
- Qualitative assessment of generation quality

From [LLaVA Paper](https://github.com/haotian-liu/LLaVA) (Liu et al., 2023):
- Tests open-ended generation quality
- Complements multiple-choice benchmarks
- Standard for conversational VLMs

---

## Section 7: ARR-COC-0-1 Multi-Task Evaluation (100 lines)

### 7.1 Relevance-Aware Captioning Evaluation

**ARR-COC-0-1 Hypothesis**:
- Query-aware relevance realization improves caption quality
- Adaptive token allocation (64-400 per patch) focuses on salient regions
- Should generate more accurate, detailed captions

**Evaluation Protocol**:

1. **Standard Caption Metrics** (MS-COCO):
   - BLEU-4, METEOR, ROUGE-L, CIDEr, SPICE
   - Compare against baseline VLMs (uniform token allocation)

2. **Relevance-Conditioned Captioning**:
   - Prompt: "Describe the image, focusing on {region/object}"
   - Measure: Does relevance allocation improve region description accuracy?
   - Metric: SPICE tuples for specified region

3. **Hallucination Detection**:
   - POPE (object presence questions)
   - CHAIR (hallucinated objects in captions)
   - Hypothesis: Relevance realization reduces hallucination

4. **Token Budget Efficiency**:
   - Accuracy vs total tokens used
   - Compare: ARR-COC (adaptive) vs baseline (fixed)
   - Metric: CIDEr per 100 tokens

**Example Ablation**:
```
Task: Caption generation on COCO val set

Baseline (uniform 256 tokens/patch):
- CIDEr: 1.20
- SPICE: 0.22
- Total tokens: 12,800 (50 patches × 256)

ARR-COC (adaptive 64-400 tokens/patch):
- CIDEr: 1.28 (+6.7%)
- SPICE: 0.25 (+13.6%)
- Total tokens: 9,600 (avg 192/patch, -25% tokens)
- Efficiency: 1.28/96 = 0.0133 CIDEr per 100 tokens (+42%)
```

---

### 7.2 VQA + Captioning Joint Evaluation

**Multi-Task Assessment**:

1. **VQA Accuracy** (VQAv2):
   - Standard accuracy metric
   - Test: Does relevance allocation improve VQA?

2. **Caption Quality** (COCO):
   - CIDEr, SPICE metrics
   - Test: Does VQA training improve captions?

3. **Joint Performance**:
   - Aggregate score: `0.5 × VQA_acc + 0.5 × CIDEr_norm`
   - Test: Multi-task learning benefit

**ARR-COC-0-1 Expectation**:
- Relevance realization benefits both tasks
- Query-aware allocation: VQA focuses on relevant patches
- Scene understanding: Improves caption semantic richness

---

### 7.3 Grounding Evaluation (RefCOCO)

**Hypothesis**: Opponent processing improves spatial reasoning.

**Evaluation**:
1. RefCOCO accuracy @ IoU 0.5
2. Attention visualization: Does model focus on correct region?
3. Ablation: Opponent processing vs standard attention

**Expected Results**:
- Higher accuracy on RefCOCO+ (attribute-based, no location words)
- Better performance on complex expressions (RefCOCOg)
- Visualization: Relevance maps align with ground-truth boxes

---

## Section 8: Best Practices for Caption Evaluation (100 lines)

### 8.1 Reporting Standards

**Minimum Requirements**:
1. Dataset version and split (COCO 2014 Karpathy split vs 2017 official)
2. All 5 standard metrics (BLEU-4, METEOR, ROUGE-L, CIDEr, SPICE)
3. Number of test samples
4. Inference settings (beam search width, max length, etc.)

**Example Report**:
```
MS-COCO 2014 Karpathy Test Split (5k images)
Beam search: width=5, max_length=20

Metrics:
- BLEU-4: 0.356
- METEOR: 0.277
- ROUGE-L: 0.568
- CIDEr: 1.133 ← Primary metric
- SPICE: 0.214

Inference: 1.2s/image (NVIDIA A100)
```

---

### 8.2 Ablation Studies

**Essential Ablations**:

1. **Vision Encoder**: CLIP vs DINOv2 vs EVA
2. **Token Budget**: 64, 144, 256, 576, 1024 tokens
3. **Fusion Method**: Early, mid, late, Q-Former, Perceiver
4. **Training Data**: COCO only vs COCO + CC3M
5. **Prompt Format**: Zero-shot vs few-shot examples

**ARR-COC-0-1 Specific**:
- Relevance allocation: Propositional only vs all three ways
- Opponent processing: With vs without tension balancing
- LOD range: 64-400 vs 128-256 vs fixed

---

### 8.3 Error Analysis

**Categorize Failures**:

1. **Hallucination**: Object mentioned but not in image
2. **Missing Objects**: Important objects not mentioned
3. **Wrong Attributes**: Incorrect color, size, etc.
4. **Wrong Relations**: Incorrect spatial/action relationships
5. **Generic Descriptions**: "A scene with objects" (too vague)

**Example Error Analysis**:
```
100 Random Test Samples:

Hallucination: 12% (CHAIR_s = 0.12)
- 8%: Spurious objects (cat → dog)
- 4%: Wrong attributes (red car → blue car)

Missing Objects: 18%
- 10%: Background objects
- 8%: Small objects (<5% image area)

Generic: 5%
- Vague captions lacking specificity
```

---

### 8.4 Human Evaluation

**When Needed**:
- Automatic metrics don't capture caption quality
- Evaluating creative/stylistic generation
- Cross-lingual captioning (no good automatic metrics)

**Standard Protocol**:
1. Sample 100-500 image-caption pairs
2. 3-5 human annotators per pair
3. Rate 1-5 on:
   - **Accuracy**: Objects/attributes correct?
   - **Fluency**: Grammatically correct, natural?
   - **Relevance**: Describes important content?
   - **Specificity**: Detailed vs generic?

4. Compute:
   - Inter-annotator agreement (Fleiss' kappa)
   - Mean rating per dimension
   - Correlation with automatic metrics

From [THumB Paper](https://www.semanticscholar.org/paper/Microsoft-COCO-Captions%3A-Data-Collection-and-Server-Chen-Fang/696ca58d93f6404fea0fc75c62d1d7b378f47628) (Kasai et al., 2022):
- Established rubric-based human evaluation protocol
- Found CLIPScore correlates well with human judgment
- Recommends hybrid: automatic + human on sample

---

## Sources

**Primary Papers**:
- [MS COCO Captions: Data Collection and Evaluation Server](https://www.cs.cmu.edu/~jeanoh/16-785/papers/chen-arxiv2015-mscoco-metrics.pdf) - Chen et al., 2015 (accessed 2025-11-16)
- [BLEU: A Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040.pdf) - Papineni et al., 2002
- [METEOR: An Automatic Metric for MT Evaluation](https://aclanthology.org/W05-0909.pdf) - Banerjee & Lavie, 2005
- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf) - Lin, 2004
- [CIDEr: Consensus-based Image Description Evaluation](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vedantam_CIDEr_Consensus-Based_Image_2015_CVPR_paper.pdf) - Vedantam et al., 2015
- [SPICE: Semantic Propositional Image Caption Evaluation](https://arxiv.org/abs/1607.08822) - Anderson et al., 2016

**Multi-Task Benchmarks**:
- [VLUE: A Multi-Task Benchmark for VL Understanding](https://arxiv.org/abs/2205.15237) - Zhou et al., 2022 (accessed 2025-11-16)
- [MMBench: Is Your Multi-modal Model an All-around Player?](https://github.com/open-compass/MMBench) - OpenCompass (accessed 2025-11-16)
- [SEED-Bench: Benchmarking Multimodal LLMs](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_SEED-Bench_Benchmarking_Multimodal_Large_Language_Models_CVPR_2024_paper.pdf) - Li et al., 2024 (accessed 2025-11-16)

**Grounding & Retrieval**:
- [RefCOCO: Modeling Context in Referring Expressions](https://arxiv.org/abs/1608.00272) - Yu et al., 2016
- [GROOViST: Grounding Objects in Visual Storytelling](https://aclanthology.org/anthology-files/anthology-files/pdf/emnlp/2023.emnlp-main.202.pdf) - Surikuchi et al., 2023 (accessed 2025-11-16)

**Web Resources**:
- [LearnOpenCV VLM Evaluation Metrics Guide](https://learnopencv.com/vlm-evaluation-metrics/) - Comprehensive tutorial (accessed 2025-11-16)
- [MS COCO Evaluation API](https://github.com/tylin/coco-caption) - Official implementation
- [Image Captioning Evaluation Survey](https://aclanthology.org/2024.ecnlp-1.9.pdf) - Li et al., 2024 (accessed 2025-11-16)

**Additional References**:
- [NoCaps: Novel Object Captioning](https://arxiv.org/abs/1812.08658) - Agrawal et al., 2019
- [Flickr30K Paper](https://www.ijcai.org/proceedings/2025/1180.pdf) - (accessed 2025-11-16)
- [THumB: Rubric-based Human Evaluation](https://www.semanticscholar.org/paper/Microsoft-COCO-Captions%3A-Data-Collection-and-Server-Chen-Fang/696ca58d93f6404fea0fc75c62d1d7b378f47628) - Kasai et al., 2022
