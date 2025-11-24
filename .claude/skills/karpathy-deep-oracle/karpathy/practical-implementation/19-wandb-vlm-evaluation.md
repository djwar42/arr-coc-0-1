# W&B VLM Evaluation: Vision-Language Model Metrics and Logging

Comprehensive guide to evaluating vision-language models using W&B, covering VQA metrics, image captioning evaluation, visual grounding assessment, and production monitoring patterns.

---

## Section 1: VQA (Visual Question Answering) Evaluation (150 lines)

### VQA Task Overview

Visual Question Answering evaluates a model's ability to answer natural language questions about image content. VQA requires both visual understanding and language reasoning capabilities.

**Common VQA Benchmarks:**
- **VQA v2**: ~1.1M questions on COCO images (balanced to reduce language bias)
- **GQA**: 22M questions with compositional reasoning emphasis
- **OKVQA**: Outside knowledge VQA requiring external knowledge
- **TextVQA**: Questions about text in images (OCR + reasoning)

From [VQAScore: Evaluating Vision-Language Models](https://blog.ml.cmu.edu/2024/10/07/vqascore-evaluating-and-improving-vision-language-generative-models/) (CMU, accessed 2025-01-31):
- Traditional VQA accuracy often insufficient for generative models
- VQAScore metric uses VQA models themselves as evaluators
- Achieves higher correlation with human judgments than CLIP-based metrics

### VQA Evaluation Metrics

**1. Exact Match Accuracy**
```python
def vqa_accuracy(prediction, ground_truth):
    """VQA accuracy with soft matching (at least 3 annotators agree)"""
    # VQA v2 uses min(#humans_said_answer / 3, 1.0)
    return min(ground_truth.count(prediction) / 3.0, 1.0)
```

**2. Answer Distribution Analysis**
- Yes/No questions: Check for language bias (should be ~50/50)
- Number questions: Evaluate numerical reasoning
- Color/object questions: Test visual grounding

**3. Question Type Breakdown**
Evaluate performance by question category:
- **What**: Object recognition
- **Where**: Spatial reasoning
- **How many**: Counting
- **Why**: Causal reasoning (hardest)

From [Open-ended VQA Benchmarking](https://arxiv.org/abs/2402.07270) (arXiv, accessed 2025-01-31):
- Open-ended VQA more challenging than multiple-choice
- Fine-grained evaluation reveals model strengths/weaknesses
- Consistency metrics important (same question, different phrasings)

### Logging VQA Results to W&B Tables

**Basic VQA Logging Pattern:**
```python
import wandb

run = wandb.init(project="vlm-vqa-eval")

# Create table with MUTABLE mode for enrichment
vqa_table = wandb.Table(
    columns=["image", "question", "prediction", "ground_truth",
             "correct", "question_type"],
    log_mode="MUTABLE"
)

for sample in vqa_dataset:
    image = wandb.Image(sample["image"])
    prediction = model.answer_question(sample["image"], sample["question"])

    # Calculate accuracy
    gt_answers = sample["answers"]  # List of 10 human answers
    accuracy = min(gt_answers.count(prediction) / 3.0, 1.0)

    vqa_table.add_data(
        image,
        sample["question"],
        prediction,
        ", ".join(gt_answers),
        accuracy >= 0.5,  # Binary correct/incorrect
        sample["question_type"]
    )

run.log({"vqa_results": vqa_table})
```

From [W&B Tables Documentation](https://docs.wandb.ai/models/tables/log_tables) (accessed 2025-01-31):
- MUTABLE mode allows adding columns after initial log
- INCREMENTAL mode for streaming large evaluation runs
- Tables support rich media types (wandb.Image, wandb.Audio, wandb.Html)

**Advanced: Confidence and Error Analysis**
```python
# Add confidence scores and failure modes
vqa_table.add_column("confidence", confidence_scores)
vqa_table.add_column("failure_mode", failure_classifications)

# Failure modes: visual_error, reasoning_error, language_bias, etc.
run.log({"vqa_results": vqa_table})
```

### Aggregated VQA Metrics

**Log summary statistics:**
```python
# Overall metrics
metrics = {
    "vqa/overall_accuracy": overall_acc,
    "vqa/yes_no_accuracy": yes_no_acc,
    "vqa/number_accuracy": number_acc,
    "vqa/other_accuracy": other_acc,

    # Question type breakdown
    "vqa/what_accuracy": what_acc,
    "vqa/where_accuracy": where_acc,
    "vqa/how_many_accuracy": count_acc,
    "vqa/why_accuracy": why_acc,

    # Answer distribution (check for bias)
    "vqa/yes_ratio": yes_count / total_yes_no,
    "vqa/no_ratio": no_count / total_yes_no
}

run.log(metrics)
```

### BERTScore for VQA Answer Similarity

From [BERTScore for VQA](https://arxiv.org/html/2507.22369v1) (arXiv, accessed 2025-01-31):
- BERTScore provides semantic similarity beyond exact match
- Useful for open-ended VQA where multiple phrasings are correct
- Correlation with human judgments: 0.7235 (higher than exact match)

```python
from torchmetrics.text.bert import BERTScore

bertscore = BERTScore(model_name_or_path="microsoft/deberta-xlarge-mnli")

for pred, ref_answers in zip(predictions, references):
    # Calculate BERTScore against all reference answers
    scores = bertscore([pred] * len(ref_answers), ref_answers)
    max_score = max(scores['f1'])

    vqa_table.add_data(..., max_score)
```

---

## Section 2: Image Captioning Metrics (150 lines)

### Image Captioning Evaluation Overview

Image captioning generates natural language descriptions of image content. Evaluation requires both n-gram overlap metrics and semantic similarity measures.

From [Re-evaluating Automatic Metrics for Image Captioning](https://aclanthology.org/E17-1019.pdf) (ACL Anthology, accessed 2025-01-31):
- Study evaluated BLEU, ROUGE, METEOR, CIDEr, SPICE, WMD
- WMD (Word Mover's Distance) shows strong advantages
- No single metric perfectly captures human judgment
- Recommend using multiple complementary metrics

### N-gram Overlap Metrics

**1. BLEU (Bilingual Evaluation Understudy)**
- Originally for machine translation
- Measures n-gram precision (1-gram to 4-gram)
- Range: 0-1 (higher is better)
- **Limitation**: Doesn't account for recall, penalizes brevity

```python
from pycocoevalcap.bleu.bleu import Bleu

bleu_scorer = Bleu(n=4)  # BLEU-1 through BLEU-4
score, _ = bleu_scorer.compute_score(references, predictions)
# score is list: [BLEU-1, BLEU-2, BLEU-3, BLEU-4]
```

**2. METEOR (Metric for Evaluation of Translation with Explicit ORdering)**
- Considers synonyms and stemming
- Balances precision and recall
- Accounts for word order through fragmentation penalty
- **Better correlation** with human judgment than BLEU

**3. ROUGE-L (Longest Common Subsequence)**
- Measures longest common subsequence
- Captures sentence-level structure
- Range: 0-1 (F-measure)

From [Evaluation Metrics for Video Captioning Survey](https://www.sciencedirect.com/science/article/pii/S2666827023000415) (ScienceDirect, accessed 2025-01-31):
- ROUGE-L effective for capturing fluency
- F-measure formulation balances precision/recall
- Works well for longer captions

### Semantic Similarity Metrics

**4. CIDEr (Consensus-based Image Description Evaluation)**
- Uses TF-IDF weighting for n-grams
- Measures consensus with human references
- **Specifically designed for image captioning**
- High correlation with human judgments

From [Learning to Evaluate Image Captioning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Cui_Learning_to_Evaluate_CVPR_2018_paper.pdf) (CVPR 2018, accessed 2025-01-31):
- CIDEr-D (with stemming) most widely used
- Weights rare n-grams higher (more informative)
- Can be gamed with n-gram copying

**5. SPICE (Semantic Propositional Image Caption Evaluation)**
- Uses scene graph matching
- Evaluates semantic content, not just n-grams
- Captures objects, attributes, relationships
- **Better captures human judgment** on semantic correctness

From [SPICE: Semantic Propositional Image Caption Evaluation](https://panderson.me/images/SPICE.pdf) (accessed 2025-01-31):
- Converts captions to scene graphs
- Compares object, attribute, and relation tuples
- F-score over semantic propositions
- Less sensitive to synonym choice than BLEU/METEOR

```python
from pycocoevalcap.spice.spice import Spice

spice_scorer = Spice()
score, _ = spice_scorer.compute_score(references, predictions)
# Returns overall SPICE score (0-1 range)
```

**6. BERTScore for Semantic Similarity**
- Uses contextual embeddings from BERT
- Measures token-level semantic similarity
- Better handles paraphrasing than n-gram metrics

```python
from torchmetrics.text.bert import BERTScore

bertscore = BERTScore(model_name_or_path="microsoft/deberta-xlarge-mnli")
scores = bertscore(predictions, references)
# scores: dict with 'precision', 'recall', 'f1'
```

**7. CLIPScore (Reference-free Metric)**

From [CLIPScore: A Reference-free Evaluation Metric](https://aclanthology.org/2021.emnlp-main.595.pdf) (ACL Anthology, accessed 2025-01-31):
- Measures image-text alignment using CLIP
- **No reference captions needed**
- Correlates with human judgment on caption quality
- Complementary to reference-based metrics

```python
from torchmetrics.multimodal.clip_score import CLIPScore

clip_scorer = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")

for image, caption in zip(images, predictions):
    score = clip_scorer(image, caption)
    # Score: similarity between image and text in CLIP space
```

### Logging Captioning Results to W&B

**Comprehensive Captioning Evaluation:**
```python
import wandb
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

run = wandb.init(project="vlm-captioning-eval")

# Create table for per-image results
caption_table = wandb.Table(
    columns=["image", "prediction", "references",
             "bleu4", "meteor", "cider", "spice", "clip_score"],
    log_mode="MUTABLE"
)

# Initialize scorers
scorers = {
    'BLEU': Bleu(n=4),
    'METEOR': Meteor(),
    'ROUGE': Rouge(),
    'CIDEr': Cider(),
    'SPICE': Spice()
}

for sample in dataset:
    image = wandb.Image(sample["image"])
    prediction = model.generate_caption(sample["image"])
    references = sample["references"]  # List of 5 reference captions

    # Calculate metrics (need dict format for scorers)
    pred_dict = {0: [prediction]}
    ref_dict = {0: references}

    scores = {}
    for name, scorer in scorers.items():
        score, _ = scorer.compute_score(ref_dict, pred_dict)
        if isinstance(score, list):
            scores[name] = score[-1]  # Use BLEU-4
        else:
            scores[name] = score

    # Add CLIPScore
    clip_score = clip_scorer(sample["image"], prediction)

    caption_table.add_data(
        image,
        prediction,
        "\n".join(references),
        scores['BLEU'],
        scores['METEOR'],
        scores['CIDEr'],
        scores['SPICE'],
        clip_score.item()
    )

run.log({"caption_results": caption_table})

# Log aggregate metrics
aggregate_metrics = {
    "captioning/bleu4": avg_bleu4,
    "captioning/meteor": avg_meteor,
    "captioning/rouge_l": avg_rouge,
    "captioning/cider": avg_cider,
    "captioning/spice": avg_spice,
    "captioning/clip_score": avg_clip_score,
    "captioning/bertscore_f1": avg_bertscore
}

run.log(aggregate_metrics)
```

### Metric Interpretation Guide

| Metric | Range | Good Score | What It Measures | Best For |
|--------|-------|------------|------------------|----------|
| BLEU-4 | 0-1 | >0.30 | N-gram precision | Fluency, exact word match |
| METEOR | 0-1 | >0.25 | Precision/recall with synonyms | Semantic similarity |
| ROUGE-L | 0-1 | >0.50 | Longest common subsequence | Sentence structure |
| CIDEr | 0-10+ | >1.0 | TF-IDF weighted consensus | Overall caption quality |
| SPICE | 0-1 | >0.20 | Scene graph matching | Semantic correctness |
| BERTScore | 0-1 | >0.85 | Contextual embedding similarity | Paraphrase detection |
| CLIPScore | 0-1 | >0.75 | Image-text alignment | Reference-free quality |

---

## Section 3: ARR-COC Specific Evaluation (150 lines)

### ARR-COC Evaluation Philosophy

ARR-COC (Adaptive Relevance Realization - Contexts Optical Compression) uses Vervaeke's cognitive framework. Evaluation must assess:
1. **Relevance realization quality** - Are the right regions attended to?
2. **Token budget efficiency** - Optimal compression vs information retention
3. **Three ways of knowing** - Propositional, Perspectival, Participatory

### Relevance Realization Metrics

**1. Attention Map Correlation**

Measure whether ARR-COC attends to query-relevant regions:

```python
import wandb
import numpy as np
from scipy.stats import spearmanr

def evaluate_relevance_maps(model, dataset):
    """Evaluate correlation between ARR-COC relevance and ground truth."""

    relevance_table = wandb.Table(
        columns=["image", "query", "relevance_map", "ground_truth_map",
                 "correlation", "budget_allocated"],
        log_mode="INCREMENTAL"
    )

    correlations = []

    for sample in dataset:
        # Get ARR-COC relevance map
        relevance_map = model.get_relevance_map(
            sample["image"],
            sample["query"]
        )

        # Ground truth: human attention map or object importance
        gt_map = sample["attention_gt"]

        # Calculate correlation
        corr, _ = spearmanr(
            relevance_map.flatten(),
            gt_map.flatten()
        )
        correlations.append(corr)

        # Get token budget allocation per patch
        budget_map = model.get_token_budgets(
            sample["image"],
            sample["query"]
        )

        relevance_table.add_data(
            wandb.Image(sample["image"]),
            sample["query"],
            wandb.Image(relevance_map),
            wandb.Image(gt_map),
            corr,
            budget_map.mean()
        )

    return np.mean(correlations), relevance_table
```

**2. Token Budget Efficiency**

Evaluate compression quality vs downstream task performance:

```python
def evaluate_token_efficiency(model, vqa_dataset, budgets=[64, 128, 256, 400]):
    """Test performance at different token budgets."""

    efficiency_metrics = {}

    for budget in budgets:
        model.set_max_tokens(budget)

        # Run VQA evaluation
        vqa_acc = evaluate_vqa(model, vqa_dataset)

        efficiency_metrics[f"vqa_acc_tokens_{budget}"] = vqa_acc
        efficiency_metrics[f"tokens_per_image"] = budget

    # Calculate efficiency ratio
    # Higher accuracy with fewer tokens = better
    for budget in budgets:
        acc = efficiency_metrics[f"vqa_acc_tokens_{budget}"]
        efficiency_metrics[f"efficiency_ratio_{budget}"] = acc / budget

    return efficiency_metrics
```

**3. Query-Aware Compression Quality**

Measure how well compression adapts to query content:

```python
def evaluate_query_adaptation(model, dataset):
    """Evaluate if token allocation changes appropriately with query."""

    adaptation_scores = []

    for sample in dataset:
        image = sample["image"]

        # Two different queries for same image
        query1 = sample["query_general"]  # "What is in this image?"
        query2 = sample["query_specific"]  # "What color is the car?"

        # Get token budgets for each query
        budget1 = model.get_token_budgets(image, query1)
        budget2 = model.get_token_budgets(image, query2)

        # For specific query, relevant regions should get MORE tokens
        relevant_region = sample["specific_region_mask"]

        # Calculate adaptation score
        budget_diff = budget2[relevant_region] - budget1[relevant_region]
        adaptation = budget_diff.mean()

        adaptation_scores.append(adaptation)

    return {
        "arr_coc/query_adaptation": np.mean(adaptation_scores),
        "arr_coc/adaptation_variance": np.std(adaptation_scores)
    }
```

### Ablation Studies: Three Ways of Knowing

Evaluate contribution of each "way of knowing":

```python
def ablation_three_ways(model, dataset):
    """Ablate each of the three ways of knowing."""

    ablation_table = wandb.Table(
        columns=["configuration", "vqa_accuracy", "caption_quality",
                 "avg_tokens"],
        log_mode="MUTABLE"
    )

    configurations = [
        {"propositional": True, "perspectival": True, "participatory": True},
        {"propositional": False, "perspectival": True, "participatory": True},
        {"propositional": True, "perspectival": False, "participatory": True},
        {"propositional": True, "perspectival": True, "participatory": False},
    ]

    for config in configurations:
        model.set_knowing_config(config)

        # Evaluate on VQA
        vqa_acc = evaluate_vqa(model, dataset)

        # Evaluate on captioning
        caption_metrics = evaluate_captioning(model, dataset)

        # Track token usage
        avg_tokens = model.get_avg_tokens_used()

        config_name = "+".join([k for k, v in config.items() if v])

        ablation_table.add_data(
            config_name,
            vqa_acc,
            caption_metrics['cider'],
            avg_tokens
        )

    wandb.log({"arr_coc/ablation_study": ablation_table})
```

### Comparative Evaluation vs Baselines

Compare ARR-COC against standard vision transformers:

```python
def comparative_evaluation(arr_coc_model, baseline_models, dataset):
    """Compare ARR-COC against baselines."""

    comparison_table = wandb.Table(
        columns=["model", "vqa_accuracy", "caption_cider",
                 "tokens_used", "latency_ms", "params_m"],
        log_mode="MUTABLE"
    )

    models = {
        "ARR-COC (Ours)": arr_coc_model,
        "ViT-L/14": baseline_models["vit_large"],
        "CLIP ViT-B/32": baseline_models["clip_base"],
        "Ovis 1.5": baseline_models["ovis"]
    }

    for name, model in models.items():
        # Evaluate
        vqa_acc = evaluate_vqa(model, dataset)
        caption_metrics = evaluate_captioning(model, dataset)

        # Resource metrics
        tokens = model.get_avg_tokens() if hasattr(model, 'get_avg_tokens') else "N/A"
        latency = benchmark_latency(model, dataset[:100])
        params = count_parameters(model) / 1e6

        comparison_table.add_data(
            name,
            vqa_acc,
            caption_metrics['cider'],
            tokens,
            latency,
            params
        )

    wandb.log({"arr_coc/model_comparison": comparison_table})
```

### ARR-COC Validation Checklist

From ARR-COC validation requirements:

**Required Evaluations:**
- [ ] VQA accuracy (VQA v2, GQA)
- [ ] Captioning metrics (BLEU, CIDEr, SPICE)
- [ ] Relevance map correlation with human attention
- [ ] Token budget efficiency analysis
- [ ] Query adaptation measurement
- [ ] Ablation of three ways of knowing
- [ ] Comparison vs fixed-resolution baselines
- [ ] Latency and throughput benchmarks

**Logging Pattern:**
```python
# Log ARR-COC specific metrics
arr_coc_metrics = {
    "arr_coc/vqa_accuracy": vqa_acc,
    "arr_coc/caption_cider": cider_score,
    "arr_coc/relevance_correlation": relevance_corr,
    "arr_coc/avg_tokens_per_image": avg_tokens,
    "arr_coc/token_efficiency_ratio": vqa_acc / avg_tokens,
    "arr_coc/query_adaptation_score": adaptation_score,

    # Ablation results
    "arr_coc/ablation_propositional": prop_contribution,
    "arr_coc/ablation_perspectival": persp_contribution,
    "arr_coc/ablation_participatory": part_contribution,

    # Resource usage
    "arr_coc/latency_ms": latency,
    "arr_coc/throughput_imgs_per_sec": throughput,
    "arr_coc/memory_mb": memory_usage
}

wandb.log(arr_coc_metrics)
```

### Visual Grounding Evaluation

For referring expression comprehension:

From [Visual Grounding Evaluation Metrics](https://arxiv.org/abs/2509.10345) (arXiv, accessed 2025-01-31):
- Pointing accuracy: IoU between predicted and ground truth boxes
- Grounding accuracy@0.5: IoU >= 0.5 threshold
- RefCOCO, RefCOCO+, RefCOCOg benchmarks standard

```python
def evaluate_visual_grounding(model, refcoco_dataset):
    """Evaluate referring expression grounding."""

    grounding_table = wandb.Table(
        columns=["image", "expression", "pred_box", "gt_box",
                 "iou", "correct_at_50"],
        log_mode="INCREMENTAL"
    )

    ious = []
    correct_at_50 = 0

    for sample in refcoco_dataset:
        # Get predicted bounding box
        pred_box = model.ground_expression(
            sample["image"],
            sample["expression"]
        )

        gt_box = sample["bbox"]

        # Calculate IoU
        iou = calculate_iou(pred_box, gt_box)
        ious.append(iou)

        if iou >= 0.5:
            correct_at_50 += 1

        # Visualize with boxes
        img_with_boxes = draw_boxes(
            sample["image"],
            [pred_box, gt_box],
            ["Predicted", "Ground Truth"]
        )

        grounding_table.add_data(
            wandb.Image(img_with_boxes),
            sample["expression"],
            str(pred_box),
            str(gt_box),
            iou,
            iou >= 0.5
        )

    metrics = {
        "grounding/accuracy@0.5": correct_at_50 / len(refcoco_dataset),
        "grounding/mean_iou": np.mean(ious),
        "grounding/median_iou": np.median(ious)
    }

    wandb.log(metrics)
    wandb.log({"grounding_results": grounding_table})
```

---

## Sources

**VQA Evaluation:**
- [VQAScore: Evaluating Vision-Language Models](https://blog.ml.cmu.edu/2024/10/07/vqascore-evaluating-and-improving-vision-language-generative-models/) - CMU Blog (accessed 2025-01-31)
- [Open-ended VQA Benchmarking](https://arxiv.org/abs/2402.07270) - arXiv:2402.07270 (accessed 2025-01-31)
- [BERTScore for VQA Evaluation](https://arxiv.org/html/2507.22369v1) - arXiv:2507.22369 (accessed 2025-01-31)

**Image Captioning Metrics:**
- [Re-evaluating Automatic Metrics for Image Captioning](https://aclanthology.org/E17-1019.pdf) - ACL Anthology (accessed 2025-01-31)
- [SPICE: Semantic Propositional Image Caption Evaluation](https://panderson.me/images/SPICE.pdf) - Anderson et al. (accessed 2025-01-31)
- [CLIPScore: A Reference-free Evaluation Metric](https://aclanthology.org/2021.emnlp-main.595.pdf) - ACL Anthology (accessed 2025-01-31)
- [Evaluation Metrics for Video Captioning Survey](https://www.sciencedirect.com/science/article/pii/S2666827023000415) - ScienceDirect (accessed 2025-01-31)

**Visual Grounding:**
- [Visual Grounding Evaluation](https://arxiv.org/abs/2509.10345) - arXiv:2509.10345 (accessed 2025-01-31)
- [Grounding Referring Expressions](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Grounding_Referring_Expressions_CVPR_2018_paper.pdf) - CVF Open Access (accessed 2025-01-31)
- [DRef Benchmark](https://openreview.net/forum?id=jruZzZJWGt) - OpenReview (accessed 2025-01-31)

**W&B Documentation:**
- [Log Tables - W&B Documentation](https://docs.wandb.ai/models/tables/log_tables) - Weights & Biases (accessed 2025-01-31)
- [W&B Evaluations Framework](https://wandb.ai/site/evaluations/) - Weights & Biases (accessed 2025-01-31)

**General VLM Evaluation:**
- [The Ultimate Guide to VLM Evaluation Metrics](https://learnopencv.com/vlm-evaluation-metrics/) - LearnOpenCV (accessed 2025-01-31)
- [Multimodal Model Evaluation Framework](https://github.com/EvolvingLMMs-Lab/lmms-eval) - GitHub (accessed 2025-01-31)

**Additional References:**
- [Learning to Evaluate Image Captioning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Cui_Learning_to_Evaluate_CVPR_2018_paper.pdf) - CVPR 2018 (accessed 2025-01-31)
