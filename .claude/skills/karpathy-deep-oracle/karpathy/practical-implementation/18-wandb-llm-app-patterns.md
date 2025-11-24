# W&B LLM Application Tracking Patterns

**Comprehensive guide to tracking, monitoring, and optimizing LLM applications with Weights & Biases**

## Overview

LLM applications present unique tracking challenges compared to traditional ML models: non-deterministic outputs, complex multi-step chains, prompt versioning, token costs, and retrieval-augmented generation (RAG) pipelines. W&B Weave provides specialized tooling for LLM observability, from development through production deployment.

This guide covers production-ready patterns for:
- Prompt engineering workflows and A/B testing
- Token usage and cost tracking
- RAG pipeline monitoring (retrieval + generation)
- Chain-of-thought and multi-step reasoning
- User feedback collection and continuous improvement

**Key difference**: Unlike traditional ML tracking (metrics, hyperparameters), LLM tracking focuses on **traces** (execution flows), **prompts** (versioned templates), **completions** (generated text), and **costs** (token usage).

---

## Section 1: Prompt Engineering Tracking (~150 lines)

### Prompt Template Versioning

**Core pattern**: Track prompts as versioned objects, not hardcoded strings.

From [W&B Prompts Documentation](https://docs.wandb.ai/weave/guides/core-types/prompts) (accessed 2025-01-31):

**String prompts** for simple templates:
```python
import weave

weave.init('prompt-experiments')

# Version 1: Basic instruction
system_prompt_v1 = weave.StringPrompt(
    "You are a helpful financial advisor. Answer briefly."
)
weave.publish(system_prompt_v1, name="advisor_prompt")

# Version 2: Enhanced with persona
system_prompt_v2 = weave.StringPrompt(
    "You are Warren Buffett. Explain complex financial concepts "
    "using simple analogies and folksy wisdom. Keep responses under 100 words."
)
weave.publish(system_prompt_v2, name="advisor_prompt")  # Creates v2
```

**Message prompts** for conversations:
```python
conversation_prompt = weave.MessagesPrompt([
    {
        "role": "system",
        "content": "You analyze {domain} using first-principles thinking."
    },
    {
        "role": "user",
        "content": "{question}"
    }
])
weave.publish(conversation_prompt, name="analyst_prompt")

# Usage with parameters
messages = conversation_prompt.format(
    domain="quantum computing",
    question="Explain decoherence"
)
```

**Key benefits**:
- Automatic versioning (Weave creates v1, v2, v3...)
- Diff comparison in UI
- Rollback to previous versions
- A/B test across versions

### A/B Testing Prompts

**Pattern**: Run parallel evaluations with different prompt versions.

```python
import weave
from openai import OpenAI

weave.init('prompt-ab-test')

class PromptVariant(weave.Model):
    prompt: weave.StringPrompt
    model_name: str = "gpt-4o"
    temperature: float = 0.7

    @weave.op()
    def predict(self, question: str) -> dict:
        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": self.prompt.format(question=question)
            }],
            temperature=self.temperature
        )
        return {
            "answer": response.choices[0].message.content,
            "prompt_version": self.prompt.ref.digest[:8]  # Track which prompt
        }

# Load different versions
variant_a = PromptVariant(
    prompt=weave.ref("advisor_prompt:v1").get()
)
variant_b = PromptVariant(
    prompt=weave.ref("advisor_prompt:v2").get()
)

# Test both on same questions
test_questions = [
    {"question": "Should I invest in index funds?"},
    {"question": "Explain compound interest"},
]

eval_a = weave.Evaluation(dataset=test_questions, scorers=[quality_scorer])
eval_b = weave.Evaluation(dataset=test_questions, scorers=[quality_scorer])

# Compare results in UI
results_a = await eval_a.evaluate(variant_a)
results_b = await eval_b.evaluate(variant_b)
```

### Tracking Prompt Parameters

**Pattern**: Log temperature, top_p, max_tokens alongside prompts.

From [W&B Weave Quickstart](https://docs.wandb.ai/weave/quickstart) (accessed 2025-01-31):

```python
import weave
from openai import OpenAI

weave.init('parameter-tracking')

@weave.op()
def generate_with_params(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 500
) -> dict:
    """Automatically logs all parameters via @weave.op()"""
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    return {
        "text": response.choices[0].message.content,
        "finish_reason": response.choices[0].finish_reason
    }

# Each call logged with parameters
result = generate_with_params(
    "Explain neural networks",
    temperature=0.3,  # Logged
    max_tokens=200    # Logged
)
```

### Completion Quality Metrics

**Pattern**: Track output quality alongside inputs.

```python
@weave.op()
def assess_completion_quality(output: dict, expected_length: int) -> dict:
    """Custom quality metrics for completions"""
    text = output.get("text", "")

    return {
        "length": len(text),
        "word_count": len(text.split()),
        "meets_length_req": len(text.split()) >= expected_length,
        "has_code": "```" in text,
        "has_citations": "[" in text and "]" in text,
        "finish_reason": output.get("finish_reason"),
        "truncated": output.get("finish_reason") == "length"
    }

# Use in evaluation
evaluation = weave.Evaluation(
    dataset=examples,
    scorers=[assess_completion_quality]
)
```

### Cost Per Prompt Calculation

From [W&B Cost Tracking](https://docs.wandb.ai/weave/guides/tracking/costs) (accessed 2025-01-31):

**Automatic cost tracking** (OpenAI, Anthropic, etc.):
```python
import weave

weave.init('cost-tracking')

# Costs automatically calculated from token usage
@weave.op()
def expensive_prompt(query: str) -> str:
    # W&B automatically logs:
    # - prompt_tokens
    # - completion_tokens
    # - total_cost (based on model pricing)
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content

# View costs in UI or query programmatically
```

**Custom model cost tracking**:
```python
client = weave.init("custom-cost-tracking")

# Add pricing for custom models
client.add_cost(
    llm_id="my-fine-tuned-gpt4",
    prompt_token_cost=0.00003,      # $30 per 1M tokens
    completion_token_cost=0.00006   # $60 per 1M tokens
)

# Different pricing after certain date
from datetime import datetime
client.add_cost(
    llm_id="my-fine-tuned-gpt4",
    prompt_token_cost=0.00004,  # Price increase
    completion_token_cost=0.00008,
    effective_date=datetime(2025, 4, 1)
)
```

**Query total costs for project**:
```python
@weave.op()
def get_project_costs(project_name: str) -> dict:
    """Calculate total LLM costs for project"""
    client = weave.init(project_name)

    total_cost = 0
    total_requests = 0

    # Fetch all traced calls with costs
    calls = list(client.get_calls(
        filter={"trace_roots_only": True},
        include_costs=True
    ))

    for call in calls:
        if call.summary.get("weave", {}).get("costs"):
            for model, cost_data in call.summary["weave"]["costs"].items():
                total_requests += cost_data["requests"]
                total_cost += cost_data["prompt_tokens_total_cost"]
                total_cost += cost_data["completion_tokens_total_cost"]

    return {
        "total_cost": total_cost,
        "total_requests": total_requests,
        "avg_cost_per_request": total_cost / total_requests if total_requests > 0 else 0
    }
```

---

## Section 2: RAG Pipeline Monitoring (~150 lines)

### Retrieval Performance Tracking

From [W&B RAG Tutorial](https://docs.wandb.ai/weave/tutorial-rag) (accessed 2025-01-31):

**Core RAG pattern** with retrieval instrumentation:

```python
import weave
from openai import OpenAI
import numpy as np

weave.init('rag-monitoring')

# Track retrieval step separately
@weave.op()
def retrieve_documents(
    query: str,
    top_k: int = 3
) -> dict:
    """Retrieval step with performance metrics"""
    # Get query embedding
    client = OpenAI()
    query_embedding = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding

    # Compute similarities (from vector DB)
    similarities = compute_similarities(query_embedding, document_embeddings)

    # Get top-k documents
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    retrieved_docs = [documents[i] for i in top_indices]

    return {
        "documents": retrieved_docs,
        "scores": [similarities[i] for i in top_indices],
        "top_k": top_k,
        "num_candidates": len(similarities),
        "avg_score": np.mean([similarities[i] for i in top_indices]),
        "score_variance": np.var([similarities[i] for i in top_indices])
    }
```

### Retrieved Context Logging

**Pattern**: Log full context passed to LLM for debugging.

```python
class RAGModel(weave.Model):
    system_message: str
    model_name: str = "gpt-4o"
    top_k: int = 3

    @weave.op()
    def predict(self, question: str) -> dict:
        # Retrieval (automatically traced)
        retrieval_result = retrieve_documents(question, top_k=self.top_k)

        # Format context
        context = "\n\n".join([
            f"Document {i+1}:\n{doc}"
            for i, doc in enumerate(retrieval_result["documents"])
        ])

        # Generation
        client = OpenAI()
        prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ]
        )

        # Log everything for inspection
        return {
            "answer": response.choices[0].message.content,
            "context": context,  # Full context visible in trace
            "retrieval_scores": retrieval_result["scores"],
            "num_docs_used": len(retrieval_result["documents"])
        }
```

### Generation Quality with Context

**Pattern**: Evaluate if retrieved context was actually useful.

From [W&B RAG Tutorial](https://docs.wandb.ai/weave/tutorial-rag):

```python
@weave.op()
async def context_precision_score(question: str, output: dict) -> dict:
    """LLM judge: Was retrieved context useful for answer?"""

    judge_prompt = f"""Given the question, answer, and context, determine if the
context was useful in arriving at the answer.

Question: {question}
Context: {output['context']}
Answer: {output['answer']}

Was the context useful? Respond with JSON: {{"verdict": 1}} for yes, {{"verdict": 0}} for no.
"""

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": judge_prompt}],
        response_format={"type": "json_object"}
    )

    verdict = json.loads(response.choices[0].message.content)
    return {"context_was_useful": verdict["verdict"] == 1}

# Evaluate RAG pipeline
evaluation = weave.Evaluation(
    dataset=rag_test_cases,
    scorers=[context_precision_score, answer_correctness_scorer]
)

await evaluation.evaluate(rag_model)
```

### End-to-End Latency Breakdown

**Pattern**: Track time spent in retrieval vs generation.

```python
import time

class TimedRAGModel(weave.Model):
    system_message: str
    model_name: str = "gpt-4o"

    @weave.op()
    def predict(self, question: str) -> dict:
        start_time = time.time()

        # Retrieval phase
        retrieval_start = time.time()
        retrieval_result = retrieve_documents(question)
        retrieval_time = time.time() - retrieval_start

        # Generation phase
        generation_start = time.time()
        context = format_context(retrieval_result["documents"])
        response = generate_answer(question, context, self.model_name)
        generation_time = time.time() - generation_start

        total_time = time.time() - start_time

        return {
            "answer": response,
            "latency_breakdown": {
                "retrieval_ms": retrieval_time * 1000,
                "generation_ms": generation_time * 1000,
                "total_ms": total_time * 1000,
                "retrieval_pct": (retrieval_time / total_time) * 100
            }
        }
```

### Context Relevance Scoring

**Pattern**: Measure relevance of retrieved documents.

```python
@weave.op()
async def context_relevance_scorer(question: str, output: dict) -> dict:
    """Score how relevant each retrieved document was"""

    relevance_scores = []

    for i, doc in enumerate(output.get("retrieved_docs", [])):
        judge_prompt = f"""Rate the relevance of this document to the question.

Question: {question}
Document: {doc}

Scale: 0 (irrelevant) to 5 (highly relevant)
Output JSON: {{"score": <number>}}
"""

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheaper for scoring
            messages=[{"role": "user", "content": judge_prompt}],
            response_format={"type": "json_object"}
        )

        score_data = json.loads(response.choices[0].message.content)
        relevance_scores.append(score_data["score"])

    return {
        "avg_relevance": np.mean(relevance_scores) if relevance_scores else 0,
        "min_relevance": min(relevance_scores) if relevance_scores else 0,
        "all_relevant": all(s >= 3 for s in relevance_scores)  # Threshold
    }
```

---

## Section 3: Production LLM Patterns (~150 lines)

### Multi-Model Comparison

**Pattern**: Test multiple LLM providers on same tasks.

From [W&B LLM Tracking Patterns](https://wandb.ai/onlineinference/genai-research/reports/A-guide-to-LLM-debugging-tracing-and-monitoring--VmlldzoxMzk1MjAyOQ) (accessed 2025-01-31):

```python
import weave
from openai import OpenAI
import anthropic

weave.init('multi-model-comparison')

class MultiModelOrchestrator(weave.Model):
    """Route to different LLMs based on task"""

    @weave.op()
    def predict(self, task: str, query: str) -> dict:
        if task == "coding":
            return self.gpt4_predict(query)
        elif task == "analysis":
            return self.claude_predict(query)
        else:
            return self.gpt35_predict(query)  # Cheap default

    @weave.op()
    def gpt4_predict(self, query: str) -> dict:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": query}]
        )
        return {
            "answer": response.choices[0].message.content,
            "model": "gpt-4o",
            "provider": "openai"
        }

    @weave.op()
    def claude_predict(self, query: str) -> dict:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": query}]
        )
        return {
            "answer": response.content[0].text,
            "model": "claude-3.5-sonnet",
            "provider": "anthropic"
        }

    @weave.op()
    def gpt35_predict(self, query: str) -> dict:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": query}]
        )
        return {
            "answer": response.choices[0].message.content,
            "model": "gpt-3.5-turbo",
            "provider": "openai"
        }

# Compare costs and quality across models
orchestrator = MultiModelOrchestrator()
```

### Fallback Strategies Tracking

**Pattern**: Monitor fallback patterns when primary model fails.

```python
import weave
from typing import Optional

class ResilientLLMClient(weave.Model):
    """LLM with automatic fallback"""

    primary_model: str = "gpt-4o"
    fallback_model: str = "gpt-3.5-turbo"
    max_retries: int = 3

    @weave.op()
    def predict(self, query: str) -> dict:
        # Try primary model
        try:
            return self._call_model(self.primary_model, query, attempt="primary")
        except Exception as e:
            # Log failure
            print(f"Primary model failed: {e}")

            # Fallback to cheaper model
            try:
                return self._call_model(
                    self.fallback_model,
                    query,
                    attempt="fallback",
                    fallback_reason=str(e)
                )
            except Exception as e2:
                return {
                    "answer": None,
                    "error": str(e2),
                    "attempt": "failed_all"
                }

    @weave.op()
    def _call_model(
        self,
        model: str,
        query: str,
        attempt: str,
        fallback_reason: Optional[str] = None
    ) -> dict:
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}]
        )

        return {
            "answer": response.choices[0].message.content,
            "model_used": model,
            "attempt": attempt,
            "fallback_reason": fallback_reason
        }

# Track fallback patterns in traces
```

### Rate Limit and Retry Monitoring

**Pattern**: Track retry attempts and rate limit hits.

```python
import weave
import time
from tenacity import retry, stop_after_attempt, wait_exponential

class RateLimitAwareLLM(weave.Model):
    """Track rate limits and retries"""

    @weave.op()
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def predict_with_retry(self, query: str) -> dict:
        start_time = time.time()
        attempt_count = 0

        try:
            attempt_count += 1
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": query}]
            )

            return {
                "answer": response.choices[0].message.content,
                "attempts": attempt_count,
                "total_time": time.time() - start_time,
                "rate_limited": False
            }

        except Exception as e:
            if "rate_limit" in str(e).lower():
                # Log rate limit hit
                return {
                    "answer": None,
                    "attempts": attempt_count,
                    "total_time": time.time() - start_time,
                    "rate_limited": True,
                    "error": str(e)
                }
            raise  # Retry on rate limits
```

### User Feedback Integration

From [W&B LLM Observability](https://wandb.ai/onlineinference/genai-research/reports/LLM-observability-Enhancing-AI-systems-with-W-B-Weave--VmlldzoxMjY4MjMwNQ) (accessed 2025-01-31):

**Pattern**: Link user feedback to specific traces.

```python
import weave

weave.init('user-feedback')

class FeedbackEnabledModel(weave.Model):
    """LLM that collects user feedback"""

    @weave.op()
    def predict(self, query: str) -> dict:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": query}]
        )

        answer = response.choices[0].message.content

        # Return trace ID for feedback linking
        return {
            "answer": answer,
            "trace_id": weave.get_current_call().id  # Link to this trace
        }

# Later: User provides feedback
@weave.op()
def log_user_feedback(trace_id: str, feedback: dict):
    """Associate feedback with specific LLM call"""
    client = weave.init('user-feedback')

    # Add feedback annotation to trace
    call = client.get_call(trace_id)
    call.feedback.add_note(
        note=f"User feedback: {feedback['rating']}/5",
        metadata={
            "helpful": feedback.get("helpful"),
            "issues": feedback.get("issues", []),
            "timestamp": time.time()
        }
    )

# Usage
result = model.predict("Explain quantum entanglement")
# User rates response
log_user_feedback(
    trace_id=result["trace_id"],
    feedback={
        "rating": 4,
        "helpful": True,
        "issues": ["too technical"]
    }
)
```

### Continuous Improvement Pipeline

**Pattern**: Use feedback to retrain/improve prompts.

```python
@weave.op()
def analyze_feedback_patterns(project_name: str) -> dict:
    """Analyze user feedback to identify prompt improvement areas"""
    client = weave.init(project_name)

    # Get all calls with feedback
    calls_with_feedback = list(client.get_calls(
        filter={"has_feedback": True}
    ))

    low_rated = []
    common_issues = {}

    for call in calls_with_feedback:
        feedback = call.feedback
        rating = feedback.metadata.get("rating", 0)

        if rating < 3:
            low_rated.append({
                "query": call.inputs.get("query"),
                "answer": call.output.get("answer"),
                "issues": feedback.metadata.get("issues", [])
            })

        # Count issue types
        for issue in feedback.metadata.get("issues", []):
            common_issues[issue] = common_issues.get(issue, 0) + 1

    return {
        "total_feedback": len(calls_with_feedback),
        "low_rated_count": len(low_rated),
        "common_issues": sorted(
            common_issues.items(),
            key=lambda x: x[1],
            reverse=True
        ),
        "examples_to_review": low_rated[:10]
    }

# Use insights to update prompts
insights = analyze_feedback_patterns("production-llm")
# If "too technical" is common issue → simplify system prompt
```

---

## Sources

**W&B Documentation:**
- [Track LLM Inputs & Outputs](https://docs.wandb.ai/weave/quickstart) - Weave quickstart guide (accessed 2025-01-31)
- [Cost Tracking](https://docs.wandb.ai/weave/guides/tracking/costs) - Custom and automatic cost calculation (accessed 2025-01-31)
- [Prompts Guide](https://docs.wandb.ai/weave/guides/core-types/prompts) - Prompt versioning and management (accessed 2025-01-31)
- [RAG Tutorial](https://docs.wandb.ai/weave/tutorial-rag) - Model-based evaluation of RAG applications (accessed 2025-01-31)

**W&B Articles:**
- [LLM Debugging, Tracing, and Monitoring](https://wandb.ai/onlineinference/genai-research/reports/A-guide-to-LLM-debugging-tracing-and-monitoring--VmlldzoxMzk1MjAyOQ) - Comprehensive LLM observability guide (accessed 2025-01-31)
- [LLM Observability with W&B Weave](https://wandb.ai/onlineinference/genai-research/reports/LLM-observability-Enhancing-AI-systems-with-W-B-Weave--VmlldzoxMjY4MjMwNQ) - Production LLM monitoring (accessed 2025-01-31)

**Additional References:**
- [OpenAI Monitoring with W&B Weave](https://cookbook.openai.com/examples/third_party/openai_monitoring_with_wandb_weave) - OpenAI Cookbook integration (accessed 2025-01-31)
- [Prompt Engineering Monitoring](https://ploomber.io/blog/prompts-weights-and-biases/) - Ploomber guide on W&B for prompts (accessed 2025-01-31)

**Related Oracle Files:**
- [10-wandb-basics.md](../gradio/10-wandb-basics.md) - W&B fundamentals
- [11-wandb-gradio-integration.md](../gradio/11-wandb-gradio-integration.md) - Gradio + W&B patterns
- [17-wandb-production-monitoring.md](17-wandb-production-monitoring.md) - General production monitoring patterns
