# W&B Evaluations Framework

**Complete evaluation framework for systematic LLM application improvement**

## Overview

W&B Evaluations (via Weave) provides a comprehensive framework for evaluation-driven LLM application development. The core concept is systematic measurement of application behavior using consistent, curated examples - similar to Test-Driven Development (TDD) but for AI systems.

**Core workflow:**
1. Define an `Evaluation` object (blueprint)
2. Create a `Dataset` of test examples
3. Define scoring functions (metrics)
4. Run evaluations against `Model` or function
5. Compare results across iterations

From [W&B Evaluations Documentation](https://docs.wandb.ai/weave/guides/core-types/evaluations) (accessed 2025-01-31):
- Each call to `.evaluate()` triggers an evaluation run
- Evaluation object = blueprint, each run = measurement
- Automatically captures traces of predictions and scores
- Rich UI for drilling into individual outputs

## Section 1: Evaluation Framework Architecture (150 lines)

### Evaluation Object Structure

The `Evaluation` class defines the complete evaluation configuration:

```python
import weave

evaluation = weave.Evaluation(
    dataset=examples,              # Dataset or list of dicts
    scorers=[scorer1, scorer2],    # List of scoring functions
    evaluation_name="My Eval",     # Optional: name the Evaluation
    trials=1,                      # Optional: run each example N times
    preprocess_model_input=None    # Optional: transform examples
)
```

**Key components:**
- **Dataset**: Test examples (failure cases, edge cases, golden examples)
- **Scorers**: Metrics to evaluate outputs (accuracy, quality, relevance)
- **Config**: Optional preprocessing, naming, trial settings

### Dataset Definition

From [W&B Evaluations Tutorial](https://docs.wandb.ai/weave/tutorial-eval) (accessed 2025-01-31):

**Simple list of dicts:**
```python
examples = [
    {"question": "What is the capital of France?", "expected": "Paris"},
    {"question": "Who wrote 'To Kill a Mockingbird'?", "expected": "Harper Lee"},
    {"question": "What is the square root of 64?", "expected": "8"},
]
```

**Published Dataset object:**
```python
weave.init('my-project')
dataset = weave.Dataset(name='qa_examples', rows=examples)
weave.publish(dataset)

# Use in evaluation
evaluation = weave.Evaluation(dataset=dataset, scorers=[...])
```

**Benefits of Dataset objects:**
- Version control for test data
- UI browsing and exploration
- Reusable across evaluations
- Track dataset lineage

### Model vs Function Evaluation

**Evaluate a Model (with parameters):**
```python
from weave import Model

class MyModel(Model):
    prompt: str
    temperature: float

    @weave.op()
    def predict(self, question: str):
        # LLM call here
        return {'generated_text': 'answer'}

model = MyModel(prompt='You are helpful', temperature=0.7)
await evaluation.evaluate(model)
```

**Evaluate a function (simpler):**
```python
@weave.op
def function_to_evaluate(question: str):
    # LLM call here
    return {'generated_text': 'answer'}

await evaluation.evaluate(function_to_evaluate)
```

**When to use Models vs functions:**
- **Model**: Parameters you want to experiment with and track
- **Function**: Simple evaluation without parameter tracking

### Evaluation Naming

From [W&B Evaluations Overview](https://docs.wandb.ai/weave/guides/core-types/evaluations) (accessed 2025-01-31):

**Two types of names:**

1. **Evaluation object name** (persistent):
```python
evaluation = Evaluation(
    dataset=examples,
    scorers=[score_fn],
    evaluation_name="My Evaluation"  # Shows in UI listings
)
```

2. **Evaluation run display name** (per-execution):
```python
evaluation.evaluate(model, __weave={"display_name": "GPT-4 baseline"})
```

**Automatic naming:**
- If no display_name provided, Weave generates: `date + random memorable name`
- Example: `2025-01-31-joyful-panda`

### Running Evaluations

**Basic execution:**
```python
import asyncio

# From script
asyncio.run(evaluation.evaluate(model))

# From Jupyter
await evaluation.evaluate(model)
```

**Multiple trials:**
```python
evaluation = Evaluation(
    dataset=examples,
    scorers=[scorer],
    trials=3  # Each example evaluated 3 times
)
```

Each example passes to model 3x, each run scored independently. Useful for:
- Measuring variance in stochastic models
- Detecting unstable outputs
- Statistical significance testing

### Evaluation Results

**Summary results (automatic):**
```python
summary = await evaluation.evaluate(model)
print(summary)
# {'accuracy': 0.85, 'latency_avg': 1.2, ...}
```

**Full scored rows (detailed):**
```python
results = evaluation.get_eval_results()
# Access individual predictions, outputs, scores
for row in results:
    print(row.prediction, row.scores)
```

**Result structure:**
- Summary: Aggregated metrics across all examples
- Full results: Individual predictions + scores per example
- Traces: Captured automatically in Weave UI

### Preprocessing Dataset Rows

From [W&B Evaluations Documentation](https://docs.wandb.ai/weave/guides/core-types/evaluations#format-dataset-rows-before-evaluating) (accessed 2025-01-31):

**Use case: Transform examples before model input**

```python
@weave.op()
def preprocess_example(example):
    # Dataset has "input_text" but model expects "question"
    return {
        "question": example["input_text"]
    }

evaluation = Evaluation(
    dataset=examples,
    scorers=[scorer],
    preprocess_model_input=preprocess_example
)
```

**Important:** Preprocessing only applied to model input, not scorer input. Scorers always receive original dataset example.

**Common use cases:**
- Rename fields to match model API
- Format complex inputs (images, multi-modal)
- Load additional context per example
- Transform data types

### HuggingFace Dataset Integration

**Current workaround (via preprocessing):**
```python
from datasets import load_dataset

hf_dataset = load_dataset("squad", split="validation[:100]")

@weave.op()
def preprocess_hf_example(example):
    return {
        "question": example["question"],
        "context": example["context"]
    }

# Convert HF dataset to Weave format
weave_examples = [
    {**ex, "expected": ex["answers"]["text"][0]}
    for ex in hf_dataset
]

evaluation = Evaluation(
    dataset=weave_examples,
    scorers=[...],
    preprocess_model_input=preprocess_hf_example
)
```

Note: W&B is building more seamless HF integration. See [Using HuggingFace datasets cookbook](https://docs.wandb.ai/weave/cookbooks/hf_dataset_evals) for updates.

---

## Section 2: Custom Scorers (150 lines)

### Scorer Fundamentals

From [W&B Scorers Overview](https://docs.wandb.ai/weave/guides/evaluation/scorers) (accessed 2025-01-31):

**Scorers** evaluate AI outputs and return evaluation metrics. Two types:
1. **Function-based scorers**: Simple functions with `@weave.op`
2. **Class-based scorers**: Inherit from `weave.Scorer` for complex evaluations

**Core requirements:**
- Must return a dictionary
- Can return multiple metrics
- Can return nested metrics
- Can return non-numeric values (LLM reasoning)

### Function-Based Scorers

**Basic structure:**
```python
import weave

@weave.op
def match_score(expected: str, output: dict) -> dict:
    return {"match": expected == output["generated_text"]}
```

**Keyword arguments:**
- `output`: Required - AI system's output
- Other args: Match dataset columns (e.g., `expected`, `label`, `target`)

**Automatic column matching:**
```python
@weave.op
def evaluate_relevance(output: str, label: int, metadata: dict) -> dict:
    # Weave automatically maps:
    # - output from model prediction
    # - label from dataset column "label"
    # - metadata from dataset column "metadata"
    return {"relevance_score": compute_relevance(output, label)}
```

**Multiple metrics in one scorer:**
```python
@weave.op
def comprehensive_score(expected: str, output: dict) -> dict:
    text = output["generated_text"]
    return {
        "exact_match": expected == text,
        "contains_keyword": expected.lower() in text.lower(),
        "length": len(text),
        "word_count": len(text.split())
    }
```

### Class-Based Scorers

From [W&B Scorers Documentation](https://docs.wandb.ai/weave/guides/evaluation/scorers) (accessed 2025-01-31):

**Advanced scorers with state/config:**

```python
from weave import Scorer
from openai import OpenAI

class SummarizationScorer(Scorer):
    model_id: str = "gpt-4o"
    system_prompt: str = "Evaluate summary quality."

    @weave.op
    def preprocess(self, text: str) -> str:
        return "Original text: \n" + text + "\n"

    @weave.op
    def call_llm(self, summary: str, processed_text: str) -> dict:
        client = OpenAI()
        res = client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Summary: {summary}\n{processed_text}"}
            ]
        )
        return {"llm_evaluation": res.choices[0].message.content}

    @weave.op
    def score(self, output: str, text: str) -> dict:
        """Score summary quality.

        Args:
            output: Summary generated by AI
            text: Original text being summarized
        """
        processed = self.preprocess(text)
        eval_result = self.call_llm(summary=output, processed_text=processed)
        return {"summary_quality": eval_result}
```

**When to use class-based:**
- Multiple function calls per evaluation
- LLM-as-judge with configurable prompts
- Complex preprocessing pipelines
- Tracking scorer metadata/parameters

### Column Mapping

**Problem: Dataset columns don't match scorer arguments**

```python
# Dataset has "news_article" but scorer expects "text"
dataset = [
    {"news_article": "The news...", "date": "2030-04-20"},
    ...
]

class SummarizationScorer(Scorer):
    @weave.op
    def score(self, output, text) -> dict:  # Expects "text"
        ...

# Solution 1: Use column_map
scorer = SummarizationScorer(column_map={"text": "news_article"})
```

**Alternative: Subclass and remap manually:**
```python
class MySummarizationScorer(SummarizationScorer):
    @weave.op
    def score(self, output: str, news_article: str) -> dict:
        # Manually map columns
        return super().score(output=output, text=news_article)
```

### Custom Summarization

From [W&B Scorers Documentation](https://docs.wandb.ai/weave/guides/evaluation/scorers) (accessed 2025-01-31):

**Default auto_summarize:**
- Numerical columns: Average
- Boolean columns: Count and fraction
- Other types: Ignored

**Custom summarize method:**
```python
class BinaryMatchScorer(Scorer):
    @weave.op
    def score(self, output, target):
        return {"match": output == target}

    def summarize(self, score_rows: list) -> dict:
        """Custom aggregation across all rows.

        Args:
            score_rows: List of dicts from score() for each example
        """
        # All must match for full_match
        full_match = all(row["match"] for row in score_rows)

        # Also compute standard metrics
        num_matches = sum(row["match"] for row in score_rows)
        total = len(score_rows)

        return {
            "full_match": full_match,
            "partial_matches": num_matches,
            "match_rate": num_matches / total
        }
```

**Why custom summarization:**
- All-or-nothing evaluation (full dataset must pass)
- Weighted scoring (some examples more important)
- Complex aggregations (F1, BLEU, CIDEr)
- Cross-example dependencies

### VLM-Specific Custom Scorers

**VQA accuracy scorer:**
```python
@weave.op
def vqa_accuracy(expected: str, output: dict) -> dict:
    """VQA allows soft matching."""
    answer = output.get("answer", "").lower().strip()
    expected_lower = expected.lower().strip()

    # Exact match
    exact = answer == expected_lower

    # Soft match (contains expected)
    contains = expected_lower in answer

    return {
        "vqa_exact": exact,
        "vqa_soft": contains,
        "answer_length": len(answer)
    }
```

**Image-text relevance scorer:**
```python
from weave import Scorer
import clip
import torch

class ImageTextRelevanceScorer(Scorer):
    model_name: str = "ViT-B/32"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model, self.preprocess = clip.load(self.model_name)

    @weave.op
    def score(self, output: str, image_path: str) -> dict:
        """Score image-caption relevance using CLIP."""
        from PIL import Image

        image = Image.open(image_path)
        image_input = self.preprocess(image).unsqueeze(0)
        text_input = clip.tokenize([output])

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)

            # Cosine similarity
            similarity = torch.cosine_similarity(
                image_features, text_features
            ).item()

        return {"clip_similarity": similarity}
```

**ARR-COC relevance scorer (custom metric):**
```python
class RelevanceRealizationScorer(Scorer):
    """Score relevance realization quality."""

    @weave.op
    def score(self, output: dict, query: str, image_path: str) -> dict:
        """Score ARR-COC relevance allocation.

        Args:
            output: Model output with token allocations
            query: User query
            image_path: Input image
        """
        allocations = output.get("token_allocations", {})

        # Check allocation range (64-400 tokens)
        in_range = all(
            64 <= tokens <= 400
            for tokens in allocations.values()
        )

        # Check query-awareness (variance in allocations)
        if allocations:
            import numpy as np
            variance = np.var(list(allocations.values()))
            query_aware = variance > 100  # Threshold
        else:
            query_aware = False

        return {
            "allocations_in_range": in_range,
            "query_aware": query_aware,
            "avg_allocation": np.mean(list(allocations.values())),
            "allocation_variance": variance
        }
```

### Built-in Scorers

From [W&B Evaluations Documentation](https://docs.wandb.ai/weave/guides/evaluation/builtin_scorers) (accessed 2025-01-31):

**Text generation scorers:**
- `HallucinationFreeScorer`: Detect hallucinations
- `SummarizationScorer`: Summary quality
- `EmbeddingSimilarityScorer`: Semantic similarity

**Classification scorers:**
- `MultiTaskBinaryClassificationF1`: F1 for multi-task classification

**Example usage:**
```python
from weave.scorers import MultiTaskBinaryClassificationF1

scorer = MultiTaskBinaryClassificationF1(
    class_names=["fruit", "color", "flavor"]
)

evaluation = weave.Evaluation(
    dataset=examples,
    scorers=[scorer]
)
```

**Local SLM scorers (privacy-first):**
- `WeaveToxicityScorerV1`: On-device toxicity detection
- `WeaveContextRelevanceScorerV1`: Context relevance scoring

---

## Section 3: Production Evaluation (150 lines)

### Continuous Evaluation Pipelines

**Evaluation as CI/CD step:**
```python
import weave
import asyncio

def run_evaluation_suite():
    """Run all evaluations in CI/CD."""
    weave.init('production-evals')

    # Load production dataset
    dataset = weave.ref("golden_examples:latest").get()

    # Test current model
    model = MyProductionModel.load()

    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=[accuracy_scorer, latency_scorer, safety_scorer]
    )

    results = asyncio.run(evaluation.evaluate(model))

    # Assert thresholds
    assert results["accuracy"] >= 0.90, "Accuracy below threshold"
    assert results["avg_latency"] <= 2.0, "Latency too high"
    assert results["safety_score"] >= 0.95, "Safety issues detected"

    return results

if __name__ == "__main__":
    run_evaluation_suite()
```

**GitHub Actions integration:**
```yaml
# .github/workflows/evaluate.yml
name: Model Evaluation
on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run evaluations
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: python run_evaluations.py
```

### A/B Testing with Evaluations

**Compare model variants:**
```python
import weave

weave.init('ab-testing')

# Define dataset
dataset = weave.Dataset(name='test_set', rows=examples)
weave.publish(dataset)

# Model A: GPT-4
model_a = MyModel(
    model_name="gpt-4",
    temperature=0.7
)

# Model B: GPT-3.5
model_b = MyModel(
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

# Evaluation config
evaluation = weave.Evaluation(
    dataset=dataset,
    scorers=[accuracy, cost, latency]
)

# Run both
results_a = await evaluation.evaluate(
    model_a,
    __weave={"display_name": "GPT-4 baseline"}
)
results_b = await evaluation.evaluate(
    model_b,
    __weave={"display_name": "GPT-3.5 candidate"}
)

# Compare in UI or programmatically
print(f"GPT-4: accuracy={results_a['accuracy']}, cost={results_a['cost']}")
print(f"GPT-3.5: accuracy={results_b['accuracy']}, cost={results_b['cost']}")
```

**Compare in Weave UI:**
- Navigate to Evaluations tab
- Select both runs
- Click "Compare Evaluations"
- View side-by-side metrics, example-level diffs

### Regression Detection

**Track evaluation over time:**
```python
class RegressionDetector:
    def __init__(self, project: str, eval_name: str):
        self.client = weave.init(project)
        self.eval_name = eval_name

    def get_baseline_metrics(self):
        """Get metrics from last production evaluation."""
        # Query previous evaluation results
        evals = self.client.get_evaluations(name=self.eval_name)
        if evals:
            return evals[0].summary  # Most recent
        return None

    def detect_regression(self, current_results: dict, threshold: float = 0.05):
        """Check if current results show regression."""
        baseline = self.get_baseline_metrics()
        if not baseline:
            return False, "No baseline found"

        regressions = []
        for metric, value in current_results.items():
            if metric in baseline:
                baseline_value = baseline[metric]
                if isinstance(value, (int, float)):
                    # Check for >5% degradation
                    degradation = (baseline_value - value) / baseline_value
                    if degradation > threshold:
                        regressions.append(
                            f"{metric}: {baseline_value:.3f} → {value:.3f}"
                        )

        if regressions:
            return True, "\n".join(regressions)
        return False, "No regression detected"

# Usage
detector = RegressionDetector('my-project', 'production_eval')
results = await evaluation.evaluate(model)
is_regression, details = detector.detect_regression(results)

if is_regression:
    raise ValueError(f"Regression detected:\n{details}")
```

### Human-in-the-Loop Evaluation

**Collect human feedback on predictions:**
```python
import weave

# Run evaluation
results = await evaluation.evaluate(model)

# Display results for human review
for example, prediction in results.items():
    print(f"Input: {example['question']}")
    print(f"Output: {prediction['answer']}")

    # Collect human feedback
    rating = input("Rate 1-5: ")
    feedback = input("Comments: ")

    # Log feedback to Weave
    weave.log_feedback(
        prediction_id=prediction.id,
        feedback={
            "rating": int(rating),
            "comments": feedback,
            "reviewer": "human_expert"
        }
    )
```

**Aggregate human feedback:**
```python
@weave.op
def human_feedback_scorer(prediction_id: str) -> dict:
    """Retrieve human feedback for prediction."""
    feedback = weave.get_feedback(prediction_id)

    if feedback:
        return {
            "human_rating": feedback.get("rating"),
            "has_comments": bool(feedback.get("comments"))
        }
    return {"human_rating": None, "has_comments": False}
```

### Integration with CI/CD

**Pre-merge evaluation gate:**
```python
#!/usr/bin/env python3
"""Pre-merge evaluation check."""

import sys
import weave
import asyncio

async def main():
    weave.init('ci-evaluations')

    # Load test dataset
    dataset = weave.ref("test_set:latest").get()

    # Load current model from PR
    model = MyModel.load_from_checkpoint("current")

    # Run evaluation
    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=[accuracy, safety, latency]
    )

    results = await evaluation.evaluate(model)

    # Check thresholds
    checks = {
        "accuracy": (results["accuracy"], 0.85, ">="),
        "safety": (results["safety_score"], 0.95, ">="),
        "latency": (results["avg_latency"], 2.5, "<="),
    }

    failures = []
    for metric, (value, threshold, op) in checks.items():
        if op == ">=" and value < threshold:
            failures.append(f"{metric}: {value:.3f} < {threshold}")
        elif op == "<=" and value > threshold:
            failures.append(f"{metric}: {value:.3f} > {threshold}")

    if failures:
        print("❌ Evaluation failed:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)

    print("✅ Evaluation passed:")
    for metric, (value, threshold, _) in checks.items():
        print(f"  - {metric}: {value:.3f} (threshold: {threshold})")
    sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())
```

### Evaluation Monitoring Dashboard

**Track evaluation metrics over time:**
```python
import weave
import pandas as pd
import matplotlib.pyplot as plt

class EvaluationDashboard:
    def __init__(self, project: str):
        self.client = weave.init(project)

    def get_evaluation_history(self, eval_name: str, limit: int = 50):
        """Get recent evaluation runs."""
        evals = self.client.get_evaluations(name=eval_name, limit=limit)

        history = []
        for eval_run in evals:
            history.append({
                "timestamp": eval_run.created_at,
                "model_version": eval_run.model_ref,
                **eval_run.summary
            })

        return pd.DataFrame(history)

    def plot_metric_trends(self, eval_name: str, metrics: list):
        """Plot metric trends over time."""
        df = self.get_evaluation_history(eval_name)

        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            ax.plot(df["timestamp"], df[metric], marker='o')
            ax.set_title(f"{metric} over time")
            ax.set_xlabel("Date")
            ax.set_ylabel(metric)
            ax.grid(True)

        plt.tight_layout()
        return fig

# Usage
dashboard = EvaluationDashboard('my-project')
fig = dashboard.plot_metric_trends(
    'production_eval',
    metrics=['accuracy', 'latency', 'cost']
)
plt.show()
```

### Saved Views

From [W&B Evaluations Documentation](https://docs.wandb.ai/weave/guides/core-types/evaluations#saved-views) (accessed 2025-01-31):

**Save table configurations for quick access:**
- Configure filters, sorts, columns in UI
- Save as named view
- Access via Python SDK or UI
- Share with team

**Use cases:**
- "High-confidence errors" view (confidence > 0.8, incorrect)
- "Expensive predictions" view (cost > $0.01)
- "Slow responses" view (latency > 2s)
- "Safety failures" view (safety_score < 0.9)

### Imperative Evaluations (EvaluationLogger)

From [W&B Evaluations Documentation](https://docs.wandb.ai/weave/guides/evaluation/evaluation_logger) (accessed 2025-01-31):

**For complex workflows with more control:**
```python
from weave import EvaluationLogger

# More flexible than standard Evaluation
# Use when you need:
# - Custom evaluation loops
# - Non-standard data flows
# - Integration with existing systems
```

**Standard Evaluation vs EvaluationLogger:**
- **Standard**: Structured, guided, best for most use cases
- **EvaluationLogger**: Flexible, imperative, for complex workflows

## Sources

**W&B Official Documentation:**
- [Evaluations Overview](https://docs.wandb.ai/weave/guides/core-types/evaluations) - Core evaluation framework (accessed 2025-01-31)
- [Scoring Overview](https://docs.wandb.ai/weave/guides/evaluation/scorers) - Scorer fundamentals and custom scorers (accessed 2025-01-31)
- [Build an Evaluation Pipeline Tutorial](https://docs.wandb.ai/weave/tutorial-eval) - Complete evaluation workflow (accessed 2025-01-31)

**Web Research:**
- [LLM Evaluation: Metrics, frameworks, and best practices](https://wandb.ai/onlineinference/genai-research/reports/LLM-evaluation-Metrics-frameworks-and-best-practices--VmlldzoxMTMxNjQ4NA) - W&B Report (accessed 2025-01-31)
- [Building Robust LLM Evaluation Frameworks](https://www.zenml.io/llmops-database/building-robust-llm-evaluation-frameworks-w-b-s-evaluation-driven-development-approach) - ZenML case study (accessed 2025-01-31)

**Additional References:**
- [W&B Weave for Enterprises](https://www.nieveazul360.com/weights-biases-announces-general-availability-of-wb-weave-for-enterprises-to-deliver-generative-ai-applications-with-confidence/) - Enterprise evaluation features
- [AI Agent Evaluation](https://wandb.ai/onlineinference/genai-research/reports/AI-agent-evaluation-Metrics-strategies-and-best-practices--VmlldzoxMjM0NjQzMQ) - Agent-specific evaluation patterns
