# Thinking Mode: Reflective Reasoning

**Category**: Architecture
**Related**: [00-overview.md](00-overview.md), [04-qwen3-llm.md](04-qwen3-llm.md)
**Code**: `ovis/model/modeling_ovis.py` - `generate()` with `enable_thinking=True`

## Overview

**Thinking Mode** is Ovis 2.5's advanced feature that enables **reflective reasoning** - the model explicitly generates its thinking process before providing the final answer.

**Key Innovation**: Two-phase generation with self-correction and reflection, similar to human problem-solving.

## Motivation

**Standard Generation** (Linear):
```
Input → Model → Output (direct)
```
Problem: No space for reflection, correction, or step-by-step reasoning.

**Thinking Mode** (Reflective):
```
Input → Model → Thinking Process → Self-Reflection → Final Answer
```
Benefit: Model can reason, catch mistakes, and refine approach.

## Architecture

### Two-Phase Generation

```
Phase 1: Thinking
┌─────────────────────────────────┐
│ Generate thinking process       │
│ Budget: thinking_budget tokens  │
│ Format: <think>...</think>      │
│ Purpose: Reasoning, reflection  │
└─────────────────────────────────┘
         ↓
Phase 2: Answer
┌─────────────────────────────────┐
│ Generate final answer           │
│ Budget: remaining tokens        │
│ Format: Clean answer (no tags)  │
│ Purpose: User-facing output     │
└─────────────────────────────────┘
```

### Tag Format

**Thinking tags** (model-generated):
```
<think>
Let me analyze this step by step:

1. First observation: ...
2. Second insight: ...
3. Wait, I need to reconsider...
4. Correction: ...
5. Final reasoning: ...
</think>

[Final answer here]
```

**Important**: Tags are part of model output, not input instructions.

## Implementation

### Enabling Thinking Mode

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "AIDC-AI/Ovis2.5-9B",
    trust_remote_code=True
)

# Enable thinking mode
response, _, _ = model.chat(
    prompt="<image>\nSolve this math problem",
    images=[image],
    max_new_tokens=3072,        # Total tokens
    enable_thinking=True,        # ← Enable reflective reasoning
    thinking_budget=2048         # ← Tokens for thinking phase
)
```

### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `enable_thinking` | bool | Enable/disable thinking mode | False |
| `thinking_budget` | int | Max tokens for thinking phase | 2048 |
| `max_new_tokens` | int | Total tokens (thinking + answer) | 3072 |

**Budget Allocation**:
- Thinking Phase: Up to `thinking_budget` tokens
- Answer Phase: `max_new_tokens - thinking_budget` tokens

## Example Output

### Math Problem

**Input**: "Solve: If x² + 5x + 6 = 0, what are the values of x?"

**Output**:
```
<think>
I need to solve the quadratic equation x² + 5x + 6 = 0.

Method 1: Factoring
Looking for two numbers that multiply to 6 and add to 5.
3 and 2 work: 3 × 2 = 6, 3 + 2 = 5

So: (x + 3)(x + 2) = 0

Therefore: x + 3 = 0 OR x + 2 = 0
Which gives: x = -3 OR x = -2

Let me verify:
x = -3: (-3)² + 5(-3) + 6 = 9 - 15 + 6 = 0 ✓
x = -2: (-2)² + 5(-2) + 6 = 4 - 10 + 6 = 0 ✓

Both solutions check out.
</think>

The values of x are **x = -3** and **x = -2**.
```

### Chart Analysis

**Input**: "Analyze this sales chart"

**Output**:
```
<think>
Let me examine the chart systematically:

1. Axes:
   - X-axis: Months (Jan-Dec)
   - Y-axis: Sales in thousands

2. Key observations:
   - Peak: July at 450k
   - Low: January at 200k
   - Trend: Upward from Jan to Jul, then decline

3. Patterns:
   - Q1 (Jan-Mar): Slow period, around 200-250k
   - Q2 (Apr-Jun): Growth phase, 300-400k
   - Q3 (Jul-Sep): Peak and decline, 450-350k
   - Q4 (Oct-Dec): Stabilization, around 300k

4. Insights:
   - Summer months show highest sales
   - Beginning of year is weakest
   - Possible seasonal business pattern
</think>

The chart shows annual sales data with a clear seasonal pattern. Sales peak in July (450,000) and bottom out in January (200,000). The pattern suggests a summer-focused business with Q2-Q3 being the strongest periods.
```

## When to Use Thinking Mode

### ✅ Use When:
- **Complex reasoning**: Multi-step math, logic puzzles
- **Analysis tasks**: Chart interpretation, document analysis
- **Ambiguous questions**: Need to clarify assumptions
- **Verification needed**: Want model to check its work
- **STEM problems**: Math, physics, coding

### ❌ Don't Use When:
- **Simple queries**: "What color is the sky?"
- **Speed critical**: Need fast responses
- **Short answers**: Yes/no questions
- **Factual lookups**: "What's the capital of France?"

## Training Data

### Phase P3: Instruction Tuning with Thinking

**Thinking-Style Data** (subset of training):
```json
{
  "image": "math_problem.jpg",
  "query": "Solve this equation",
  "answer": "<think>\nLet me work through this:\n1. ...\n2. ...\n</think>\n\nThe answer is X."
}
```

**Key Aspects**:
- Human-annotated thinking processes
- Step-by-step reasoning examples
- Self-correction demonstrations
- Multiple reasoning strategies

### Phase P4: DPO (Preference)

**Preference Pairs**:
- ✅ Preferred: Clear thinking → correct answer
- ❌ Rejected: Confused thinking → wrong answer
- ✅ Preferred: Self-correction → improved answer
- ❌ Rejected: No reflection → hasty answer

**Purpose**: Teach model to value quality reasoning.

### Phase P5: GRPO (RL)

**Reward Signal**:
- Correct final answer: +1
- Correct reasoning process: +0.5
- Self-correction: +0.3
- Logical consistency: +0.2

**Purpose**: Optimize reasoning quality beyond supervised learning.

## Performance Impact

### Benchmarks with Thinking Mode

| Dataset | Standard | +Thinking | Improvement |
|---------|----------|-----------|-------------|
| MathVista | 62.3% | 68.7% | +6.4% |
| ChartQA | 74.2% | 78.9% | +4.7% |
| DocVQA | 81.5% | 82.1% | +0.6% |
| STEM-QA | 59.8% | 66.2% | +6.4% |

**Observation**: Largest gains on reasoning-heavy tasks.

### Speed Tradeoff

```
Standard Mode:
- Tokens generated: 50-200
- Time: 2-8 seconds

Thinking Mode:
- Tokens generated: 500-2500
- Time: 20-100 seconds

Tradeoff: 5-10× slower, but higher accuracy on complex tasks
```

## Parsing Output

### Extracting Thinking and Answer

```python
def parse_thinking_output(response):
    """
    Extract thinking process and final answer
    """
    import re

    # Find thinking section
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)

    if think_match:
        thinking = think_match.group(1).strip()
        # Remove thinking tags for final answer
        answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        return {
            'thinking': thinking,
            'answer': answer,
            'has_thinking': True
        }
    else:
        return {
            'thinking': None,
            'answer': response,
            'has_thinking': False
        }

# Usage
response, _, _ = model.chat(
    prompt="<image>\nAnalyze this chart",
    images=[image],
    enable_thinking=True
)

parsed = parse_thinking_output(response)
print("Thinking:", parsed['thinking'])
print("\nAnswer:", parsed['answer'])
```

## Best Practices

### 1. Budget Allocation

```python
# For complex problems
enable_thinking=True
thinking_budget=2048      # More thinking space
max_new_tokens=3072       # Total budget

# For moderate problems
enable_thinking=True
thinking_budget=1024      # Less thinking needed
max_new_tokens=2048

# For simple problems
enable_thinking=False     # Skip thinking overhead
max_new_tokens=512
```

### 2. Prompt Engineering

**Good prompts for thinking mode**:
```
"Analyze this chart step by step"
"Solve this problem, showing your work"
"Explain your reasoning for this answer"
"Think through this carefully before answering"
```

**Poor prompts** (don't need thinking):
```
"What color is this?"
"Is there a table in the image?"
"Yes or no: ..."
```

### 3. Quality Control

```python
# Check if thinking actually helped
parsed = parse_thinking_output(response)

if parsed['has_thinking']:
    # Verify thinking is substantive (not just filler)
    thinking_words = len(parsed['thinking'].split())

    if thinking_words < 50:
        print("Warning: Thinking seems shallow")

    # Check for self-correction (good sign)
    if 'wait' in parsed['thinking'].lower() or 'actually' in parsed['thinking'].lower():
        print("Model self-corrected (good!)")
```

## Limitations

### 1. Not Always Necessary
- Simple queries don't benefit from thinking
- Adds unnecessary latency
- Use selectively

### 2. Thinking Quality Varies
- Some problems: Deep, insightful reasoning
- Other problems: Superficial filler
- Model still learning optimal thinking depth

### 3. No Guarantee of Correctness
- Thinking ≠ correct answer
- Model can reason incorrectly
- Always verify critical outputs

## Related Topics

- [00-overview.md](00-overview.md) - System architecture
- [04-qwen3-llm.md](04-qwen3-llm.md) - Language model capabilities
- [../training/03-phase-p3-instruction.md](../training/03-phase-p3-instruction.md) - Thinking data
- [../training/04-phases-p4-p5-rl.md](../training/04-phases-p4-p5-rl.md) - RL optimization
- [../examples/01-thinking-mode.md](../examples/01-thinking-mode.md) - Code examples
- [../usage/02-advanced-features.md](../usage/02-advanced-features.md) - Advanced usage

## Code References

**Main Generation**: `ovis/model/modeling_ovis.py` - `generate()` method
**Chat Interface**: `ovis/model/modeling_ovis.py` - `chat()` method
**Demo**: `ovis/serve/infer_think_demo.py` - Complete thinking mode examples
