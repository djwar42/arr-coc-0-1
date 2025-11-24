# VQA Evaluation & Metrics

**Comprehensive guide to evaluating Visual Question Answering systems, covering accuracy metrics, benchmark protocols, ablation studies, and ARR-COC-0-1 evaluation strategies.**

---

## Section 1: VQA Accuracy Metrics

### 1.1 Official VQA Accuracy Formula

From [VQA Evaluation Code](https://visualqa.org/evaluation.html) (accessed 2025-11-16):

The official VQA v2.0 accuracy metric accounts for answer ambiguity through human consensus:

```
VQA Accuracy = min(# humans that said answer / 3, 1.0)
```

**Key principles:**
- At least 3 humans must agree for 100% accuracy
- Partial credit for 1-2 agreements (33%, 67%)
- Accounts for answer ambiguity and subjectivity
- Each answer scored independently

**Implementation:**
```python
def vqa_accuracy(prediction, ground_truth_answers):
    """
    prediction: single predicted answer string
    ground_truth_answers: list of 10 human answers
    """
    # Normalize prediction
    prediction = normalize_answer(prediction)

    # Count how many humans gave this answer
    matches = sum([1 for gt in ground_truth_answers
                   if normalize_answer(gt) == prediction])

    # Apply VQA accuracy formula
    return min(matches / 3.0, 1.0)

# Average over all questions
total_accuracy = sum([vqa_accuracy(pred, gt)
                     for pred, gt in zip(predictions, ground_truths)])
average_accuracy = total_accuracy / len(predictions)
```

From [VQA: Visual Question Answering Paper (Antol et al., 2015)](https://arxiv.org/pdf/1505.00468) (accessed 2025-11-16):
- Multiple-choice task: 18 candidate answers created for each question
- Open-ended task: Free-form text generation
- Accuracy computed based on agreement with 10 human annotators

### 1.2 Answer Normalization

From [VQA Evaluation Tools](https://github.com/GT-Vision-Lab/VQA) (accessed 2025-11-16):

**Standard normalization pipeline:**

```python
import re

def normalize_answer(answer):
    """Normalize VQA answer for fair comparison."""
    # Convert to lowercase
    answer = answer.lower()

    # Remove articles
    answer = re.sub(r'\b(a|an|the)\b', ' ', answer)

    # Remove punctuation
    answer = re.sub(r'[^\w\s]', '', answer)

    # Remove extra whitespace
    answer = ' '.join(answer.split())

    return answer
```

**Number handling:**
- "5" and "five" should match
- Implement number-to-word and word-to-number conversion
- Handle approximate numbers ("about 10" vs "10")

**Synonym matching:**
- "car" vs "automobile"
- "couch" vs "sofa"
- Use WordNet or custom synonym dictionary

### 1.3 Answer Type Analysis

From [Answer-Type Prediction for VQA (Kafle & Kanan, 2016)](https://openaccess.thecvf.com/content_cvpr_2016/papers/Kafle_Answer-Type_Prediction_for_CVPR_2016_paper.pdf) (accessed 2025-11-16):

**Evaluate separately by answer type:**

```python
# Answer types in VQA v2.0
answer_types = ["yes/no", "number", "other"]

for ans_type in answer_types:
    subset = [ex for ex in examples if ex['answer_type'] == ans_type]
    accuracy = compute_vqa_accuracy(subset)
    print(f"{ans_type}: {accuracy:.2f}%")
```

**Typical distributions:**
- Yes/no questions: ~30% of dataset, highest accuracy (~80-85%)
- Number questions: ~10% of dataset, mid-range accuracy (~45-55%)
- Other questions: ~60% of dataset, lowest accuracy (~55-65%)

### 1.4 Consistency and Agreement Metrics

From [On the Human-level Performance of VQA (Zhou et al., 2025)](https://aclanthology.org/2025.coling-main.277.pdf) (accessed 2025-11-16):

**Human agreement analysis:**
- Computed average accuracy of humans' answers for each question type
- Cohen's kappa coefficient for inter-annotator agreement
- Percentage agreement across annotators

**Key findings:**
- Human-human agreement varies by question type
- Complex reasoning questions show lower agreement
- Simple factual questions show high agreement (>90%)

**Measuring model-human alignment:**

```python
def cohen_kappa(predictions, ground_truths):
    """
    Measure agreement between model and human consensus.
    """
    from sklearn.metrics import cohen_kappa_score

    # Convert to binary: correct vs incorrect
    correct = [vqa_accuracy(pred, gt) > 0.5
               for pred, gt in zip(predictions, ground_truths)]

    # Compare with human majority vote
    human_majority = [get_majority_answer(gt) for gt in ground_truths]

    return cohen_kappa_score(correct, human_majority)
```

---

## Section 2: Benchmark Evaluation Protocols

### 2.1 VQA v2.0 Evaluation Protocol

From [VQA Challenge](https://visualqa.org/challenge.html) (accessed 2025-11-16):

**Dataset splits:**
- Train: 82,783 MS COCO images, 443,757 questions
- Validation: 40,504 MS COCO images, 214,354 questions
- Test-dev: Subset for development (limited submissions per day)
- Test-standard: Full test set for final evaluation (very limited submissions)

**Submission format:**
```json
[
  {
    "question_id": 262148000,
    "answer": "down"
  },
  {
    "question_id": 262148001,
    "answer": "blue"
  }
]
```

**Evaluation server protocol:**
- Test-dev: Development experiments, live leaderboard
- Test-standard: Final benchmark results, official leaderboard
- Multiple-choice vs open-ended tasks

### 2.2 Human Performance Baselines

From [Reliable Visual Question Answering (Shakib, 2022)](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-137.pdf) (accessed 2025-11-16):

**Human baselines on VQA v2.0:**
- Human accuracy: ~90-95% (with proper training)
- Best models (2024): ~71-75% accuracy
- Gap to human performance: ~20 percentage points

**Reliability analysis:**
- Models achieve 71% accuracy but often lack confidence calibration
- Abstaining option improves reliability (answer only when confident)
- Human-model agreement varies significantly by question type

### 2.3 GQA Evaluation Protocol

From [VQA Benchmark Datasets](https://paperswithcode.com/datasets?q=&v=lst&o=newest&task=visual-question-answering-1) (accessed 2025-11-16):

**GQA-specific metrics:**
- Accuracy: Standard VQA accuracy
- Consistency: Answers remain consistent under question paraphrasing
- Plausibility: Answers are semantically plausible
- Distribution: Answer distribution matches expected distribution

**Structured reasoning evaluation:**
```python
# GQA provides question semantic programs
def evaluate_gqa(predictions, ground_truths, semantic_programs):
    """
    Evaluate compositional reasoning on GQA.
    """
    results = {
        'accuracy': compute_vqa_accuracy(predictions, ground_truths),
        'consistency': compute_consistency(predictions, paraphrases),
        'plausibility': compute_plausibility(predictions, semantic_programs)
    }
    return results
```

### 2.4 TextVQA and OCR-VQA

From [VizWiz VQA](https://vizwiz.org/tasks-and-datasets/vqa/) (accessed 2025-11-16):

**TextVQA evaluation:**
- Task: Answer questions requiring reading text in images
- Metric: Exact match accuracy (after normalization)
- Challenges: OCR errors, text localization, reasoning over text

**VizWiz evaluation:**
- Task: VQA for blind/low-vision users
- Metric: VQA accuracy + answerable/unanswerable classification
- Special considerations: Image quality issues, real-world constraints

---

## Section 3: Ablation Studies for VQA

### 3.1 Vision Encoder Ablations

From [karpathy-deep-oracle/practical-implementation/56-vision-token-budget-ablations.md](../karpathy-deep-oracle/practical-implementation/56-vision-token-budget-ablations.md):

**Comparing vision encoders:**
- CLIP ViT-B/32 vs ViT-L/14 vs DINOv2
- Frozen vs trainable encoders
- Resolution ablations (224×224 vs 384×384 vs 448×448)

**Methodology:**
```python
def ablate_vision_encoder(encoders, dataset):
    """
    Systematic ablation of vision encoder choices.
    """
    results = {}

    for encoder_name, encoder in encoders.items():
        # Freeze encoder
        encoder.eval()
        encoder.requires_grad_(False)

        # Evaluate on VQA
        accuracy = evaluate_vqa(model_with_encoder(encoder), dataset)

        results[encoder_name] = {
            'accuracy': accuracy,
            'params': count_parameters(encoder),
            'tokens_per_image': encoder.num_tokens
        }

    return results
```

**Typical findings:**
- Larger encoders (ViT-L) improve accuracy by 3-5%
- Higher resolution (+50%) improves accuracy by 2-4%
- Frozen encoders sufficient for many tasks (faster training)

### 3.2 Token Budget Ablations

From [karpathy-deep-oracle/practical-implementation/51-vision-token-budgets.md](../karpathy-deep-oracle/practical-implementation/51-vision-token-budgets.md):

**Token budget impact on VQA accuracy:**

```python
# Test different token budgets
token_budgets = [64, 144, 256, 576, 1024]

for budget in token_budgets:
    model = VLM_Model(vision_tokens=budget)
    accuracy = evaluate_vqa(model, vqa_val)
    latency = measure_latency(model, vqa_val)

    print(f"Budget: {budget} tokens")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Latency: {latency:.1f}ms")
    print(f"  Tokens/accuracy: {budget/accuracy:.2f}")
```

**Expected trends:**
- 64 tokens: ~60-65% accuracy (fastest, lowest quality)
- 256 tokens: ~68-72% accuracy (balanced)
- 576 tokens: ~70-74% accuracy (diminishing returns)
- 1024 tokens: ~71-75% accuracy (slow, marginal gains)

### 3.3 Fusion Method Ablations

From [karpathy-deep-oracle/practical-implementation/57-qformer-learned-queries-ablation.md](../karpathy-deep-oracle/practical-implementation/57-qformer-learned-queries-ablation.md):

**Comparing fusion approaches:**
- Early fusion: Concatenate vision + text tokens directly
- Q-Former: Learnable queries attend to vision features
- Perceiver Resampler: Latent queries compress vision
- Late fusion: Separate vision and text pathways

**Ablation template:**
```python
fusion_methods = {
    'early': EarlyFusion(),
    'qformer': QFormer(num_queries=64),
    'perceiver': PerceiverResampler(num_latents=128),
    'late': LateFusion()
}

for method_name, fusion in fusion_methods.items():
    model = VLM(vision_encoder, fusion, language_model)
    results = evaluate_full_suite(model)

    print(f"{method_name}:")
    print(f"  VQA v2: {results['vqa_v2']:.2f}%")
    print(f"  GQA: {results['gqa']:.2f}%")
    print(f"  Params: {count_parameters(fusion)/1e6:.1f}M")
```

### 3.4 Training Objective Ablations

From [karpathy-deep-oracle/practical-implementation/60-vision-encoder-compression-ratios.md](../karpathy-deep-oracle/practical-implementation/60-vision-encoder-compression-ratios.md):

**Comparing pre-training objectives:**
- Contrastive (CLIP-style): Image-text matching
- Masked language modeling: Predict masked tokens
- Image-text matching (binary): Match vs non-match
- Multi-task: Combine multiple objectives

**Evaluation:**
```python
objectives = ['contrastive', 'mlm', 'itm', 'multitask']

for objective in objectives:
    model = pretrain_vlm(objective=objective, steps=100000)

    # Zero-shot evaluation
    zero_shot = evaluate_vqa(model, vqa_val, fine_tune=False)

    # Fine-tuned evaluation
    fine_tuned = evaluate_vqa(model, vqa_val, fine_tune=True)

    print(f"{objective}:")
    print(f"  Zero-shot: {zero_shot:.2f}%")
    print(f"  Fine-tuned: {fine_tuned:.2f}%")
```

---

## Section 4: Error Analysis and Failure Modes

### 4.1 Common VQA Failure Modes

From [Human-Adversarial VQA (Sheng et al., 2021)](https://proceedings.neurips.cc/paper/2021/file/aa97d584861474f4097cf13ccb5325da-Paper.pdf) (accessed 2025-11-16):

**AdVQA dataset findings:**
- 46,807 adversarial examples that fool SOTA models
- Human-in-the-loop generation reveals model weaknesses
- Common failures: Counting, spatial reasoning, attribute binding

**Failure categories:**
```python
def categorize_failures(predictions, ground_truths, images, questions):
    """
    Classify VQA failures by error type.
    """
    failures = []

    for pred, gt, img, q in zip(predictions, ground_truths, images, questions):
        if vqa_accuracy(pred, gt) < 0.5:  # Incorrect
            error_type = classify_error(pred, gt, img, q)
            failures.append({
                'question': q,
                'prediction': pred,
                'ground_truth': gt,
                'error_type': error_type
            })

    # Aggregate by error type
    error_counts = {}
    for f in failures:
        error_counts[f['error_type']] = error_counts.get(f['error_type'], 0) + 1

    return error_counts
```

**Error types:**
- **Hallucination**: Model generates plausible but incorrect answers
- **Visual grounding failure**: Doesn't look at relevant image regions
- **Language bias**: Relies on question priors, ignores image
- **Counting errors**: Difficulty with precise counting (>3 objects)
- **Spatial reasoning**: Left/right, above/below confusion
- **Attribute binding**: Incorrectly associates attributes with objects

### 4.2 Bias Detection

From [VQA v2.0 Paper](https://arxiv.org/pdf/1505.00468) (accessed 2025-11-16):

**Language priors test:**
```python
def measure_language_bias(model, dataset):
    """
    Test if model relies on question-only priors.
    """
    # Baseline: Answer questions without seeing image
    text_only_acc = evaluate_vqa(model, dataset, no_image=True)

    # Full model: Answer with image
    full_acc = evaluate_vqa(model, dataset, no_image=False)

    # Multimodal gain
    mm_gain = full_acc - text_only_acc

    print(f"Text-only accuracy: {text_only_acc:.2f}%")
    print(f"Full accuracy: {full_acc:.2f}%")
    print(f"Multimodal gain: {mm_gain:.2f}%")

    # Red flag if text-only ≈ full (model ignoring vision)
    if mm_gain < 5.0:
        print("WARNING: Model may be relying on language priors!")

    return mm_gain
```

### 4.3 Consistency Analysis

From [Mind the Uncertainty in Human Disagreement (Lan et al., 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/32468/34623) (accessed 2025-11-16):

**Human uncertainty and model consistency:**

```python
def evaluate_consistency(model, paraphrase_pairs):
    """
    Test if model gives consistent answers to paraphrased questions.
    """
    consistent = 0
    total = len(paraphrase_pairs)

    for (q1, q2), image in paraphrase_pairs:
        ans1 = model.predict(image, q1)
        ans2 = model.predict(image, q2)

        if normalize_answer(ans1) == normalize_answer(ans2):
            consistent += 1

    consistency_rate = consistent / total
    return consistency_rate
```

**Key insights:**
- Human disagreement reflects question ambiguity
- Models should reflect uncertainty when humans disagree
- Consistency across paraphrases is critical for reliability

---

## Section 5: Advanced Evaluation Techniques

### 5.1 VQAScore for Generative Models

From [VQAScore: Evaluating Text-to-Visual Generation (Lin et al., 2024)](https://linzhiqiu.github.io/papers/vqascore/) (accessed 2025-11-16):

**Using VQA models to evaluate image generation:**

```python
def vqa_score(generated_image, text_prompt, vqa_model):
    """
    Score generated image using VQA model.

    Ask questions derived from text prompt, measure accuracy.
    """
    # Extract facts from prompt
    questions = extract_questions_from_prompt(text_prompt)

    # Ask VQA model about generated image
    scores = []
    for question, expected_answer in questions:
        predicted = vqa_model.predict(generated_image, question)
        score = soft_match(predicted, expected_answer)
        scores.append(score)

    # Average score
    return np.mean(scores)
```

**Applications:**
- Text-to-image generation evaluation
- Image editing evaluation
- Video generation evaluation
- Correlates better with human judgment than CLIP score

### 5.2 Improving VQA Metrics with LLMs

From [Improving Automatic VQA Evaluation Using LLMs (Mañas et al., 2024)](https://arxiv.org/html/2310.02567v2) (accessed 2025-11-16):

**LLM-as-judge for VQA:**

```python
def llm_vqa_metric(prediction, ground_truth, question, llm):
    """
    Use LLM to judge if prediction matches ground truth.

    Better handles synonyms, paraphrases, and semantic equivalence.
    """
    prompt = f"""
    Question: {question}
    Ground truth answer: {ground_truth}
    Model prediction: {prediction}

    Does the prediction correctly answer the question?
    Consider synonyms and semantic equivalence.
    Answer: YES or NO
    """

    response = llm.generate(prompt)
    return 1.0 if "YES" in response else 0.0
```

**Advantages:**
- Better synonym matching ("car" vs "automobile")
- Semantic equivalence ("5" vs "five")
- Contextual understanding
- Correlates better with human judgment

### 5.3 Multi-Metric Evaluation

From [W&B VLM Evaluation](../karpathy-deep-oracle/karpathy/gradio/13-wandb-vlm-metrics.md):

**Comprehensive VQA evaluation suite:**

```python
def comprehensive_vqa_eval(model, datasets):
    """
    Evaluate VQA model across multiple metrics and datasets.
    """
    results = {}

    # Standard accuracy
    for dataset_name, dataset in datasets.items():
        accuracy = evaluate_vqa_accuracy(model, dataset)
        results[f'{dataset_name}_accuracy'] = accuracy

    # Consistency
    consistency = evaluate_consistency(model, paraphrase_pairs)
    results['consistency'] = consistency

    # Language bias
    mm_gain = measure_language_bias(model, datasets['vqa_v2'])
    results['multimodal_gain'] = mm_gain

    # Error analysis
    error_dist = categorize_failures(model, datasets['vqa_v2'])
    results['error_distribution'] = error_dist

    # Human agreement
    human_corr = compute_human_correlation(model, datasets['vqa_v2'])
    results['human_correlation'] = human_corr

    return results
```

---

## Section 6: Leaderboards and Benchmarking

### 6.1 Major VQA Leaderboards

From [VQA Leaderboards](https://visualqa.org/challenge.html) and [Open VLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) (accessed 2025-11-16):

**Official VQA v2.0 leaderboard (2024 SOTA):**
- PaliGemma-3B (fine-tuned, 448px): 96.39%
- Zhipu AI: 95.71%
- PaLI (Google): 96.13%
- BEiT-3 (Microsoft): ~95%

**Open VLM Leaderboard:**
- Tracks performance across multiple benchmarks
- VQA v2.0, GQA, TextVQA, VizWiz, OKVQA
- Model size, architecture, training data transparency

**Leaderboard considerations:**
```python
# Track multiple benchmarks
benchmarks = ['vqa_v2', 'gqa', 'textvqa', 'vizwiz', 'okvqa']

for benchmark in benchmarks:
    score = evaluate_benchmark(model, benchmark)
    print(f"{benchmark}: {score:.2f}%")

# Aggregate score (weighted average)
weights = {'vqa_v2': 0.3, 'gqa': 0.3, 'textvqa': 0.2,
           'vizwiz': 0.1, 'okvqa': 0.1}
aggregate = sum(weights[b] * evaluate_benchmark(model, b)
                for b in benchmarks)
```

### 6.2 Specialized Leaderboards

From [IDP Leaderboard](https://idp-leaderboard.org/qa-benchmark) and [ScienceQA](https://scienceqa.github.io/leaderboard.html) (accessed 2025-11-16):

**Document VQA (DocVQA):**
- IDP Leaderboard: Unified benchmark for document understanding
- Metrics: ANLS (Average Normalized Levenshtein Similarity)
- Focus: OCR + reasoning over documents

**ScienceQA:**
- Multi-modal science question answering
- Categories: Natural science, social science, language science
- Evaluation: Accuracy across categories, with/without explanations

**OK-VQA (Outside Knowledge VQA):**
- Requires external knowledge beyond image
- Metrics: VQA accuracy + knowledge retrieval accuracy
- Challenges: Grounding answers in both image and world knowledge

---

## Section 7: Statistical Rigor in VQA Evaluation

### 7.1 Proper Statistical Testing

From [karpathy-deep-oracle/experimental-design/03-benchmark-datasets-evaluation.md](../karpathy-deep-oracle/experimental-design/03-benchmark-datasets-evaluation.md):

**Comparing two VQA models:**

```python
from scipy import stats

def compare_vqa_models(model_a, model_b, dataset):
    """
    Statistically compare two VQA models.
    """
    # Get predictions
    acc_a = []
    acc_b = []

    for example in dataset:
        pred_a = model_a.predict(example['image'], example['question'])
        pred_b = model_b.predict(example['image'], example['question'])

        acc_a.append(vqa_accuracy(pred_a, example['answers']))
        acc_b.append(vqa_accuracy(pred_b, example['answers']))

    # Paired t-test (same images tested by both models)
    t_stat, p_value = stats.ttest_rel(acc_a, acc_b)

    # Effect size (Cohen's d)
    diff = np.array(acc_a) - np.array(acc_b)
    cohen_d = np.mean(diff) / np.std(diff)

    print(f"Model A: {np.mean(acc_a):.2f}% ± {np.std(acc_a):.2f}")
    print(f"Model B: {np.mean(acc_b):.2f}% ± {np.std(acc_b):.2f}")
    print(f"Difference: {np.mean(diff):.2f}%")
    print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
    print(f"Cohen's d: {cohen_d:.3f}")

    if p_value < 0.05:
        print("✓ Statistically significant difference")
    else:
        print("✗ No significant difference")

    return {'t_stat': t_stat, 'p_value': p_value, 'cohen_d': cohen_d}
```

### 7.2 Bootstrap Confidence Intervals

```python
def bootstrap_vqa_ci(model, dataset, n_bootstrap=10000):
    """
    Compute bootstrap confidence intervals for VQA accuracy.
    """
    accuracies = []

    for example in dataset:
        pred = model.predict(example['image'], example['question'])
        acc = vqa_accuracy(pred, example['answers'])
        accuracies.append(acc)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(accuracies, size=len(accuracies), replace=True)
        bootstrap_means.append(np.mean(sample))

    # 95% confidence interval
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    mean_acc = np.mean(accuracies)

    print(f"VQA Accuracy: {mean_acc:.2f}%")
    print(f"95% CI: [{ci_lower:.2f}%, {ci_upper:.2f}%]")

    return mean_acc, (ci_lower, ci_upper)
```

---

## Section 8: ARR-COC-0-1 VQA Evaluation Strategy

### 8.1 Custom Metrics for Relevance Realization

**ARR-COC-0-1 requires specialized evaluation beyond standard VQA accuracy:**

```python
def evaluate_arr_coc_vqa(model, dataset):
    """
    Comprehensive evaluation for ARR-COC-0-1 VQA performance.
    """
    results = {
        'vqa_accuracy': [],
        'relevance_allocation': [],
        'token_efficiency': [],
        'query_awareness': []
    }

    for example in dataset:
        # Standard VQA accuracy
        pred, metadata = model.predict(example['image'], example['question'],
                                       return_metadata=True)
        acc = vqa_accuracy(pred, example['answers'])
        results['vqa_accuracy'].append(acc)

        # Relevance allocation quality
        allocations = metadata['token_allocations']

        # Check allocation range (64-400 tokens per patch)
        in_range = all(64 <= tokens <= 400 for tokens in allocations.values())
        results['relevance_allocation'].append(int(in_range))

        # Token efficiency (accuracy per token)
        total_tokens = sum(allocations.values())
        efficiency = acc / (total_tokens / 1000)  # Per 1000 tokens
        results['token_efficiency'].append(efficiency)

        # Query awareness (allocation varies with query)
        allocation_variance = np.var(list(allocations.values()))
        query_aware = allocation_variance > 100  # Threshold
        results['query_awareness'].append(int(query_aware))

    # Aggregate metrics
    summary = {
        'vqa_accuracy': np.mean(results['vqa_accuracy']),
        'relevance_valid': np.mean(results['relevance_allocation']),
        'token_efficiency': np.mean(results['token_efficiency']),
        'query_awareness': np.mean(results['query_awareness'])
    }

    return summary
```

### 8.2 Ablation Studies for ARR-COC-0-1

**Token budget ablation:**

```python
# Test different token budgets
budgets = [64, 100, 150, 200, 300, 400]

for budget in budgets:
    # Set global token budget
    model.set_global_budget(budget)

    # Evaluate VQA
    vqa_acc = evaluate_vqa_accuracy(model, vqa_val)
    latency = measure_latency(model, vqa_val)

    print(f"Budget: {budget} tokens/patch")
    print(f"  VQA Accuracy: {vqa_acc:.2f}%")
    print(f"  Latency: {latency:.1f}ms")
    print(f"  Efficiency: {vqa_acc/budget:.4f} acc/token")
```

**Relevance scorer ablation:**

```python
# Test different relevance scoring methods
scorers = {
    'propositional': PropositionalScorer(),  # Shannon entropy
    'perspectival': PerspectivalScorer(),    # Jungian archetypes
    'participatory': ParticipatoryScorer(),  # Cross-attention
    'combined': CombinedScorer()             # All three
}

for scorer_name, scorer in scorers.items():
    model.set_relevance_scorer(scorer)

    vqa_acc = evaluate_vqa_accuracy(model, vqa_val)
    print(f"{scorer_name}: {vqa_acc:.2f}%")
```

### 8.3 ARR-COC-0-1 Benchmark Suite

**Recommended evaluation protocol:**

```python
def arr_coc_benchmark_suite(model):
    """
    Complete ARR-COC-0-1 evaluation across multiple benchmarks.
    """
    results = {}

    # 1. VQA v2.0 (primary benchmark)
    results['vqa_v2'] = {
        'accuracy': evaluate_vqa_accuracy(model, vqa_v2_val),
        'ci_95': bootstrap_vqa_ci(model, vqa_v2_val),
        'by_question_type': evaluate_by_question_type(model, vqa_v2_val)
    }

    # 2. GQA (compositional reasoning)
    results['gqa'] = {
        'accuracy': evaluate_vqa_accuracy(model, gqa_val),
        'consistency': evaluate_consistency(model, gqa_paraphrases)
    }

    # 3. TextVQA (attention to text)
    results['textvqa'] = {
        'accuracy': evaluate_vqa_accuracy(model, textvqa_val),
        'text_region_attention': measure_text_attention(model, textvqa_val)
    }

    # 4. Relevance realization metrics
    results['relevance'] = {
        'allocation_valid': evaluate_relevance_allocation(model, vqa_v2_val),
        'query_awareness': evaluate_query_awareness(model, vqa_v2_val),
        'token_efficiency': evaluate_token_efficiency(model, vqa_v2_val)
    }

    # 5. Ablation studies
    results['ablations'] = {
        'token_budget': ablate_token_budget(model, vqa_v2_val),
        'relevance_scorer': ablate_relevance_scorer(model, vqa_v2_val),
        'opponent_processing': ablate_opponent_processing(model, vqa_v2_val)
    }

    return results
```

---

## Sources

### Source Documents

From [karpathy-deep-oracle/karpathy/gradio/18-wandb-evaluations.md](../karpathy-deep-oracle/karpathy/gradio/18-wandb-evaluations.md):
- W&B Evaluations framework for VQA evaluation
- Custom scorers for VQA metrics
- Human-in-the-loop evaluation patterns

From [karpathy-deep-oracle/karpathy/gradio/13-wandb-vlm-metrics.md](../karpathy-deep-oracle/karpathy/gradio/13-wandb-vlm-metrics.md):
- VQA accuracy tracking with W&B
- Relevance score tracking for ARR-COC
- Token allocation metrics
- Debugging visualizations for VQA

From [karpathy-deep-oracle/karpathy/practical-implementation/50-vqav2-training-protocols.md](../karpathy-deep-oracle/karpathy/practical-implementation/50-vqav2-training-protocols.md):
- VQA v2.0 dataset structure
- VQA accuracy formula and implementation
- Answer normalization and encoding strategies
- Soft label construction for multi-annotator answers

From [karpathy-deep-oracle/experimental-design/03-benchmark-datasets-evaluation.md](../karpathy-deep-oracle/experimental-design/03-benchmark-datasets-evaluation.md):
- Statistical testing for VQA model comparison
- Experimental design fundamentals
- Effect sizes and practical significance
- Bootstrap confidence intervals

### Web Research

**VQA Evaluation Fundamentals:**
- [VQA Evaluation Code](https://visualqa.org/evaluation.html) - Official VQA accuracy formula (accessed 2025-11-16)
- [VQA: Visual Question Answering (Antol et al., 2015)](https://arxiv.org/pdf/1505.00468) - Original VQA paper and evaluation protocol (accessed 2025-11-16)
- [VQA Challenge](https://visualqa.org/challenge.html) - VQA v2.0 benchmark and leaderboard (accessed 2025-11-16)

**VQA Metrics and Analysis:**
- [VQAScore: Evaluating Text-to-Visual Generation (Lin et al., 2024)](https://linzhiqiu.github.io/papers/vqascore/) - Using VQA for image generation evaluation (accessed 2025-11-16)
- [Improving Automatic VQA Evaluation Using LLMs (Mañas et al., 2024)](https://arxiv.org/html/2310.02567v2) - LLM-as-judge for VQA (accessed 2025-11-16)
- [Answer-Type Prediction for VQA (Kafle & Kanan, 2016)](https://openaccess.thecvf.com/content_cvpr_2016/papers/Kafle_Answer-Type_Prediction_for_CVPR_2016_paper.pdf) - Answer type analysis (accessed 2025-11-16)

**Human Evaluation and Agreement:**
- [On the Human-level Performance of VQA (Zhou et al., 2025)](https://aclanthology.org/2025.coling-main.277.pdf) - Human performance baselines and agreement (accessed 2025-11-16)
- [Mind the Uncertainty in Human Disagreement (Lan et al., 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/32468/34623) - Human uncertainty in VQA evaluation (accessed 2025-11-16)

**Adversarial and Failure Analysis:**
- [Human-Adversarial VQA (Sheng et al., 2021)](https://proceedings.neurips.cc/paper/2021/file/aa97d584861474f4097cf13ccb5325da-Paper.pdf) - AdVQA dataset and failure modes (accessed 2025-11-16)
- [Reliable Visual Question Answering (Shakib, 2022)](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-137.pdf) - Reliability and abstention in VQA (accessed 2025-11-16)

**Leaderboards and Benchmarks:**
- [Open VLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) - Multi-benchmark VLM evaluation (accessed 2025-11-16)
- [IDP Leaderboard](https://idp-leaderboard.org/qa-benchmark) - Document VQA benchmarking (accessed 2025-11-16)
- [ScienceQA Leaderboard](https://scienceqa.github.io/leaderboard.html) - Science question answering (accessed 2025-11-16)
- [VizWiz VQA](https://vizwiz.org/tasks-and-datasets/vqa/) - VQA for accessibility (accessed 2025-11-16)
- [OK-VQA](https://okvqa.allenai.org/) - Outside knowledge VQA (accessed 2025-11-16)

**Additional References:**
- [VQA GitHub - Evaluation Tools](https://github.com/GT-Vision-Lab/VQA) - Official evaluation code (accessed 2025-11-16)
- [Papers with Code - VQA Datasets](https://paperswithcode.com/datasets?q=&v=lst&o=newest&task=visual-question-answering-1) - VQA dataset collection (accessed 2025-11-16)

---

**Document Version**: 1.0
**Created**: 2025-11-16
**Purpose**: VQA evaluation metrics and protocols for ARR-COC-0-1
**Lines**: ~700+
**Related**: vlm-engineering/00-vlm-architectures-survey.md, practical-implementation/50-vqav2-training-protocols.md
