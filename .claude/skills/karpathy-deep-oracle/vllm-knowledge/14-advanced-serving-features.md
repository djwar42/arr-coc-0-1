# vLLM Advanced Serving Features

## Overview

vLLM provides sophisticated serving features that go beyond basic text generation, enabling fine-grained control over model outputs, structured formats, and decoding strategies. These features are essential for production deployments where predictability, format compliance, and advanced sampling techniques are required.

This document covers six major advanced serving capabilities in vLLM:
1. Guided decoding (regex, grammar constraints, JSON schemas)
2. Beam search implementation and optimization
3. Log probabilities (logprobs) and token probabilities
4. Parallel sampling strategies
5. Stopping criteria and conditions
6. Seed-based deterministic sampling

## 1. Guided Decoding (Structured Outputs)

### What is Guided Decoding?

From [vLLM Blog - Structured Decoding Introduction](https://blog.vllm.ai/2025/01/14/struct-decode-intro.html) (accessed 2025-11-02):

> Guided decoding is to LLMs what validation is to APIs - it acts as a guarantee that what comes out matches what you expect. Guided decoding ensures structure integrity that allows developers to integrate LLMs into their application with ease!

Guided decoding constrains the output of a language model to follow specific formats by applying logit masks during token generation. Instead of allowing free-form text, the model is restricted to produce only valid outputs according to user-defined rules.

### Why Guided Decoding Matters

From [Red Hat Developer - Structured Outputs in vLLM](https://developers.redhat.com/articles/2025/06/03/structured-outputs-vllm-guiding-ai-responses) (accessed 2025-11-02):

LLMs excel at generating coherent text but can be unpredictable when specific formats are required. Key benefits of guided decoding:

1. **Reliability**: Outputs are predictable and machine-readable
2. **Compatibility**: Seamless integration with APIs, databases, or other systems
3. **Efficiency**: No need for extensive post-processing to validate or fix outputs

Without constraints, asking a model to generate JSON might produce: "While I think this is a good structure, here's a JSON object..." which fails to parse. With guided decoding, you're guaranteed valid JSON.

### How Guided Decoding Works

From [vLLM Blog - Structured Decoding Introduction](https://blog.vllm.ai/2025/01/14/struct-decode-intro.html):

At generation time, vLLM modifies the probability distribution for next tokens by applying bias (logit masks) based on schemas. The system uses finite-state machines (FSM) or pushdown automata (PDA) to track the current state during decoding and filter out invalid tokens.

**Technical Flow**:
1. User provides a constraint (regex, JSON schema, grammar, or choice list)
2. vLLM compiles the constraint into a state machine
3. During each decoding step, the state machine determines which tokens are valid
4. Invalid tokens receive logit bias (typically -inf) to prevent sampling
5. Model generates only from valid token candidates
6. State machine transitions based on generated tokens

### Backend Systems

vLLM supports multiple backends for guided decoding, each with different performance characteristics:

From [vLLM Blog - Structured Decoding Introduction](https://blog.vllm.ai/2025/01/14/struct-decode-intro.html):

**XGrammar Backend**:
- Uses pushdown automata (PDA) for context-free grammars
- Excellent caching for repeated schemas
- Grammar compilation moved to C with pthread optimization
- Best for: Long generations with reused schemas
- Performance: Up to 5x improvement in time-per-output-token (TPOT) under load

**Guidance Backend** (llguidance):
- Calculates constraints on per-token basis
- Faster time-to-first-token (TTFT)
- Better for dynamic, unpredictable schemas
- Best for: Multi-tenant workloads with unique schemas per request

**Outlines Backend** (legacy):
- Token-level FSM construction
- Slower decoding (one token at a time)
- Batch processing bottlenecks
- Being phased out in favor of XGrammar/Guidance

vLLM uses "auto" mode by default to intelligently select the best backend based on the request characteristics.

### Guided Choice Constraints

From [HPE Developer - Using Structured Outputs in vLLM](https://developer.hpe.com/blog/using-structured-outputs-in-vllm/) (accessed 2025-11-02):

The simplest form - restrict output to predefined options:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="-")

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"}
    ],
    extra_body={"guided_choice": ["positive", "negative"]},
)
print(completion.choices[0].message.content)
# Output: positive
```

**Use cases**: Classification tasks, enumerated selections, multiple-choice scenarios.

### Guided Regex Constraints

Constrain output to match regex patterns (email addresses, dates, identifiers):

```python
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Generate an example email address for Alan Turing at Enigma. End in .com.",
        }
    ],
    extra_body={"guided_regex": r"\w+@\w+\.com\n", "stop": ["\n"]},
)
print(completion.choices[0].message.content)
# Output: alan.turing@enigma.com
```

**Use cases**: Email validation, phone numbers, custom ID formats, dates (YYYY-MM-DD), license keys.

### Guided JSON Schema

From [HPE Developer - Using Structured Outputs in vLLM](https://developer.hpe.com/blog/using-structured-outputs-in-vllm/):

Enforce valid JSON based on schema - the most powerful structured output mode:

```python
from pydantic import BaseModel
from enum import Enum

class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"

class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType

json_schema = CarDescription.model_json_schema()

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {"role": "user", "content": "Generate a JSON for the most iconic car from the 90s."}
    ],
    extra_body={"guided_json": json_schema},
)
print(completion.choices[0].message.content)
# Output:
# {
#   "brand": "Toyota",
#   "model": "Supra",
#   "car_type": "coupe"
# }
```

**Schema Features**:
- Type enforcement (string, integer, boolean, arrays, nested objects)
- Required vs optional fields
- Enumerations for restricted values
- Nested structures with complex hierarchies

**Use cases**: API responses, database records, tool calling, configuration files, structured data extraction.

### Guided Grammar (EBNF)

From [HPE Developer - Using Structured Outputs in vLLM](https://developer.hpe.com/blog/using-structured-outputs-in-vllm/):

Extended Backus-Naur Form (EBNF) grammars for complex structures like SQL queries:

```python
completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[
        {"role": "user", "content": "Generate a SQL query to find all users older than 30."}
    ],
    extra_body={
        "guided_grammar": """
        query ::= "SELECT" fields "FROM users WHERE" condition;
        fields ::= "name, age" | "*";
        condition ::= "age >" number;
        number ::= [0-9]+;
        """
    },
)
print(completion.choices[0].message.content)
# Output: SELECT * FROM users WHERE age > 30;
```

**Grammar syntax**:
- `::=` defines production rules
- `|` represents alternatives
- `[]` for character classes
- `+` for one or more repetitions
- Terminal strings in quotes

**Use cases**: SQL generation, programming language syntax, domain-specific languages, protocol messages.

### Python API for Guided Decoding

From [vLLM Documentation - Sampling Parameters](https://docs.vllm.ai/en/stable/api/vllm/sampling_params.html) (accessed 2025-11-02):

```python
from vllm import LLM, SamplingParams

# JSON schema example
sampling_params = SamplingParams(
    temperature=0.7,
    guided_decoding=GuidedDecodingParams(
        guided_json={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
    )
)

# Regex example
sampling_params = SamplingParams(
    guided_decoding=GuidedDecodingParams(
        guided_regex=r"\d{4}-\d{2}-\d{2}"
    )
)

# Choice example
sampling_params = SamplingParams(
    guided_decoding=GuidedDecodingParams(
        guided_choice=["red", "blue", "green"]
    )
)
```

### Performance Considerations

From [vLLM Blog - Structured Decoding Introduction](https://blog.vllm.ai/2025/01/14/struct-decode-intro.html):

**V0 vs V1 Architecture**:
- V0: Guided decoding could stall entire engine during schema compilation
- V1: Non-blocking initialization, minimal overhead

**Backend Trade-offs**:

| Backend | TTFT | TPOT | Best For |
|---------|------|------|----------|
| XGrammar | Moderate | Excellent (cached) | Repeated schemas, long outputs |
| Guidance | Excellent | Good | Dynamic schemas, multi-tenant |
| Outlines | Poor | Poor | Legacy (deprecated) |

**Optimization Tips**:
1. Reuse schemas across requests for XGrammar caching
2. Use simpler constraints when possible (choice > regex > JSON > grammar)
3. Enable backend auto-selection for optimal performance
4. Monitor time-to-first-token for user experience

### Future: Jump Decoding

From [vLLM Blog - Structured Decoding Introduction](https://blog.vllm.ai/2025/01/14/struct-decode-intro.html):

Jump decoding is an upcoming optimization where vLLM can skip ahead when output is deterministic:

```json
{ "name": "Alice" }
```

Once `{` is chosen, the next token **must** be `"`, then `name`, etc. No need to sample each step individually - the system can "jump" through the deterministic sequence, significantly accelerating generation and reducing GPU load.

## 2. Beam Search Implementation

### What is Beam Search?

Beam search is a heuristic search algorithm that explores multiple candidate sequences simultaneously, keeping the top-k most probable sequences at each step. Unlike greedy decoding (which picks the single best token) or sampling (which randomly selects), beam search maintains multiple hypotheses to find higher-quality outputs.

From [vLLM Documentation - Sampling Parameters](https://docs.vllm.ai/en/v0.8.1/api/inference_params.html) (accessed 2025-11-02):

**Key parameters**:
- `use_beam_search`: Enable beam search mode
- `best_of`: Number of sequences to generate from which to select the best
- `beam_width`: Number of beams to maintain (V0 only)
- `length_penalty`: Penalize sequences based on length
- `early_stopping`: Control when beam search terminates

### Basic Beam Search Usage

```python
from vllm import SamplingParams

# Enable beam search with best_of parameter
sampling_params = SamplingParams(
    n=1,  # Return 1 final sequence
    best_of=5,  # Generate 5 candidates, return best
    use_beam_search=True,
    temperature=0.0,  # Must be 0 for beam search
    max_tokens=100
)

# With length penalty to prefer longer sequences
sampling_params = SamplingParams(
    use_beam_search=True,
    best_of=3,
    length_penalty=1.2,  # Values > 1 encourage longer sequences
    early_stopping=True
)
```

**Important constraints**:
- Temperature must be 0 (deterministic)
- Top-p and top-k are ignored in beam search mode
- Incompatible with parallel sampling

### Beam Search Algorithm

From [GitHub Issue #20316 - Optimize beam search code](https://github.com/vllm-project/vllm/issues/20316) (accessed 2025-11-02):

Traditional beam search in vLLM worked as follows:

1. Start with beam_width initial beams
2. For each beam, generate logprobs for top-k tokens
3. Create beam_width × 2 × beam_width candidate sequences by concatenating all tokens with all beams
4. Sort all candidates by cumulative log probability
5. Keep top beam_width sequences
6. Repeat until max_tokens or early_stopping condition

**Performance issue**: The concatenation operation in step 3 is highly time-consuming, executing `beam_width × 2 × beam_width` times per decoding step.

### Beam Search Optimization

From [GitHub Issue #20316 - Optimize beam search code](https://github.com/vllm-project/vllm/issues/20316):

Optimized approach avoids expensive concatenation:

```python
# Optimized beam search (conceptual)
new_beams = []
all_beams_token_id = []
all_beams_logprob = []

# Iterate through all beam inference results
for i, result in enumerate(output):
    current_beam = all_beams[i]
    if result.outputs[0].logprobs is not None:
        logprobs = result.outputs[0].logprobs[0]
        all_beams_token_id.extend(list(logprobs.keys()))
        all_beams_logprob.extend([
            current_beam.cum_logprob + obj.logprob
            for obj in logprobs.values()
        ])

# Convert to numpy for efficient operations
all_beams_token_id = np.array(all_beams_token_id)
all_beams_logprob = np.array(all_beams_logprob)

# Handle EOS tokens
if not ignore_eos:
    eos_idx = np.where(all_beams_token_id == tokenizer.eos_token_id)[0]
    for idx in eos_idx:
        current_beam = all_beams[idx // logprobs_num]
        # Add to completed sequences
        completed.append(BeamSearchSequence(...))
    # Set EOS probabilities to -inf
    all_beams_logprob[eos_idx] = -np.inf

# Select top-k without creating all candidates first
topn_idx = np.argpartition(np.negative(all_beams_logprob), beam_width)[:beam_width]

for idx in topn_idx:
    current_beam = all_beams[idx // logprobs_num]
    token_id = int(all_beams_token_id[idx])
    new_beams.append(BeamSearchSequence(
        tokens=current_beam.tokens + [token_id],
        cum_logprob=float(all_beams_logprob[idx]),
        ...
    ))

all_beams = new_beams
```

**Performance improvement**: ~40% reduction in processing time by using numpy's `argpartition` instead of creating and sorting all candidates.

### Length Penalty

From [vLLM Documentation - Sampling Parameters](https://docs.vllm.ai/en/v0.4.0.post1/dev/sampling_params.html) (accessed 2025-11-02):

```python
# Length penalty formula applied to scores
score = log_prob / (length ** length_penalty)

# length_penalty < 1.0: Prefer shorter sequences
sampling_params = SamplingParams(
    use_beam_search=True,
    length_penalty=0.8
)

# length_penalty > 1.0: Prefer longer sequences
sampling_params = SamplingParams(
    use_beam_search=True,
    length_penalty=1.2
)

# length_penalty = 1.0: No length preference (default)
```

### Early Stopping

Controls when beam search terminates:

```python
# Stop as soon as beam_width complete sentences are found
sampling_params = SamplingParams(
    use_beam_search=True,
    early_stopping=True
)

# Never stop early, always generate max_tokens
sampling_params = SamplingParams(
    use_beam_search=True,
    early_stopping=False
)
```

**Early stopping conditions**:
- `True`: Stop when `beam_width` sequences hit EOS or max_tokens
- `False`: Continue until all beams reach max_tokens
- Useful for tasks requiring complete sentences vs fixed-length generation

### Beam Search vs Sampling

| Feature | Beam Search | Sampling |
|---------|-------------|----------|
| Determinism | Fully deterministic | Stochastic |
| Diversity | Low (similar sequences) | High (varied outputs) |
| Quality | Higher average quality | Variable quality |
| Speed | Slower (multiple beams) | Faster (single path) |
| Use Cases | Translation, summarization | Creative writing, dialogue |
| Temperature | Must be 0 | Any value > 0 |

## 3. Log Probabilities (Logprobs) and Token Probabilities

### What are Logprobs?

Log probabilities (logprobs) provide insight into the model's confidence for each generated token. They represent the log of the probability that the model assigned to tokens, both chosen tokens and alternative candidates.

From [vLLM Documentation - Inference Parameters](https://docs.vllm.ai/en/v0.8.1/api/inference_params.html) (accessed 2025-11-02):

> logprobs – Number of log probabilities to return per output token. When set to None, no probability is returned. If set to a non-None value, the result includes the log probabilities of the specified number of most likely tokens, as well as the chosen tokens.

**Key insight**: The API always returns the logprob of the sampled token, so there may be up to `logprobs + 1` elements in the response.

### Output Logprobs

```python
from vllm import SamplingParams

# Request top-5 alternative tokens at each step
sampling_params = SamplingParams(
    temperature=0.8,
    logprobs=5,  # Return 5 most likely alternatives
    max_tokens=50
)

# Output structure (conceptual)
# {
#   "choices": [{
#     "logprobs": {
#       "tokens": ["the", "cat", "sat"],
#       "token_logprobs": [-0.1, -0.3, -0.2],
#       "top_logprobs": [
#         {
#           "the": -0.1,   # chosen token
#           "a": -1.2,
#           "my": -2.1,
#           "this": -2.5,
#           "that": -3.0
#         },
#         # ... for each token
#       ]
#     }
#   }]
# }
```

**Use cases**:
- Confidence scoring for generated text
- Detecting hallucinations (low probability tokens)
- Reranking multiple outputs
- Debugging model behavior
- Uncertainty quantification

### Prompt Logprobs

From [vLLM Forums - What is the purpose of prompt logprobs?](https://discuss.vllm.ai/t/what-is-the-purpose-of-prompt-logprobs/1714) (accessed 2025-11-02):

> The purpose of prompt_logprobs in vLLM is to return the log probabilities (logprobs) of each token in the input prompt, allowing users to understand how the model perceives the prompt.

```python
sampling_params = SamplingParams(
    prompt_logprobs=5,  # Return top-5 alternatives for each prompt token
    max_tokens=100
)

# Use cases for prompt logprobs:
# 1. Perplexity calculation
# 2. Prompt optimization (finding low-probability tokens)
# 3. Input validation (detecting OOV or unusual tokens)
# 4. Prefix caching effectiveness measurement
```

**Prompt logprobs enable**:
- Measuring how "natural" the prompt is to the model
- Identifying problematic tokens in the prompt
- Computing perplexity scores for evaluation
- Analyzing the model's understanding of context

### Logprobs Implementation Details

From [vLLM Documentation - Sampling Parameters](https://docs.vllm.ai/en/stable/api/vllm/sampling_params.html) (accessed 2025-11-02):

```python
class SamplingParams:
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None

    # Implementation note from docs:
    # "Note that the implementation follows the OpenAI API:
    # The API will always return the log probability of the
    # sampled token, so there may be up to logprobs+1 elements
    # in the response."
```

**OpenAI API compatibility**: vLLM follows OpenAI's conventions for logprobs, ensuring drop-in replacement capability for existing applications.

### Computing Probabilities from Logprobs

```python
import math

# Convert log probability to probability
logprob = -0.5
probability = math.exp(logprob)  # e^(-0.5) ≈ 0.6065

# Convert probability to logprob
probability = 0.8
logprob = math.log(probability)  # ln(0.8) ≈ -0.2231

# Common logprob ranges:
# -0.01 to -0.1:  Very confident (90-99% probability)
# -0.5 to -1.0:   Confident (37-61% probability)
# -2.0 to -3.0:   Low confidence (5-14% probability)
# < -5.0:         Very uncertain (< 1% probability)
```

### Logprobs for Reranking

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3-8B-Instruct")

# Generate multiple candidates with logprobs
sampling_params = SamplingParams(
    n=5,  # Generate 5 candidates
    logprobs=1,  # Get chosen token probabilities
    temperature=0.8,
    max_tokens=50
)

outputs = llm.generate(prompts, sampling_params)

# Rerank by average logprob (higher is better)
for output in outputs.outputs:
    tokens = output.logprobs
    avg_logprob = sum(t[output.token_ids[i]].logprob
                      for i, t in enumerate(tokens)) / len(tokens)
    print(f"Text: {output.text}")
    print(f"Avg logprob: {avg_logprob}")
```

### Detecting Low-Confidence Outputs

```python
def detect_hallucinations(output, threshold=-3.0):
    """Detect potentially hallucinated tokens based on low logprobs."""
    suspicious_tokens = []

    for i, logprob_dict in enumerate(output.logprobs):
        token_id = output.token_ids[i]
        logprob = logprob_dict[token_id].logprob

        if logprob < threshold:
            token_text = logprob_dict[token_id].decoded_token
            suspicious_tokens.append({
                'position': i,
                'token': token_text,
                'logprob': logprob,
                'probability': math.exp(logprob)
            })

    return suspicious_tokens

# Usage
suspicious = detect_hallucinations(output, threshold=-2.5)
for token in suspicious:
    print(f"Low confidence at position {token['position']}: "
          f"'{token['token']}' (prob={token['probability']:.2%})")
```

### Special Tokens in Logprobs

From [GitHub Issue #4772 - Unexpected Special Tokens in prompt_logprobs](https://github.com/vllm-project/vllm/issues/4772) (accessed 2025-11-02):

Be aware that logprobs dictionaries may include special tokens (BOS, EOS, PAD) that can affect downstream processing:

```python
# Filter special tokens from logprobs
def filter_special_tokens(logprobs, tokenizer):
    special_token_ids = set(tokenizer.all_special_ids)

    filtered = []
    for logprob_dict in logprobs:
        filtered_dict = {
            k: v for k, v in logprob_dict.items()
            if k not in special_token_ids
        }
        filtered.append(filtered_dict)

    return filtered
```

## 4. Parallel Sampling

### What is Parallel Sampling?

Parallel sampling generates multiple independent output sequences from a single prompt simultaneously, enabling efficient exploration of the output space.

From [vLLM Documentation - Sampling Parameters](https://docs.vllm.ai/en/v0.5.0/dev/sampling_params.html) (accessed 2025-11-02):

```python
sampling_params = SamplingParams(
    n=5,  # Generate 5 independent sequences
    temperature=0.8,
    max_tokens=100
)

# Returns 5 different outputs for the same prompt
outputs = llm.generate(["Tell me a story"], sampling_params)
for i, output in enumerate(outputs[0].outputs):
    print(f"Output {i+1}: {output.text}")
```

### Parallel Sampling Implementation

From [vLLM Documentation - parallel_sampling](https://docs.vllm.ai/en/v0.10.2/api/vllm/v1/engine/parallel_sampling.html) (accessed 2025-11-02):

vLLM's parallel sampling maintains:
- Parent request ID
- Individual sampling params per child sequence
- State tracking for each parallel sample
- Efficient batch processing of all sequences together

```python
# Internal structure (conceptual)
class ParallelSamplingRequest:
    parent_request_id: str
    num_samples: int  # n parameter
    sampling_params: SamplingParams
    child_requests: List[ChildRequest]

    def generate_child_sampling_params(self):
        """Generate sampling params for each child request."""
        # Each child gets same params but independent random state
        pass
```

### Parallel Sampling vs Best-Of

| Feature | Parallel Sampling (`n`) | Best-Of |
|---------|------------------------|---------|
| Returns | All n sequences | Only best 1 (or n best) |
| Selection | No filtering | Sorted by probability |
| Use Case | Explore alternatives | Find highest quality |
| Overhead | Return all outputs | Extra sorting step |
| Parameter | `n=5` | `best_of=5, n=1` |

```python
# Parallel sampling: Get all 5 outputs
sampling_params = SamplingParams(n=5, temperature=0.8)

# Best-of sampling: Generate 5, return best 1
sampling_params = SamplingParams(best_of=5, n=1, temperature=0.8)

# Best-of sampling: Generate 10, return best 3
sampling_params = SamplingParams(best_of=10, n=3, temperature=0.8)
```

### Use Cases for Parallel Sampling

1. **Self-consistency**: Generate multiple reasoning paths and take majority vote
```python
sampling_params = SamplingParams(n=10, temperature=0.7, max_tokens=200)
outputs = llm.generate(["Solve: 2x + 5 = 13"], sampling_params)

# Parse answers and take majority vote
answers = [extract_answer(out.text) for out in outputs[0].outputs]
final_answer = max(set(answers), key=answers.count)
```

2. **Diverse generations**: Provide users with multiple options
```python
# Generate 3 different email subject lines
sampling_params = SamplingParams(n=3, temperature=0.9)
outputs = llm.generate(["Email subject for product launch"], sampling_params)
for i, out in enumerate(outputs[0].outputs):
    print(f"Option {i+1}: {out.text}")
```

3. **A/B testing**: Compare different variations automatically
```python
sampling_params = SamplingParams(n=5, temperature=0.8)
# Automatically get 5 variations for testing
```

### Optimizing Parallel Sampling

From [GitHub Issue #16373 - Optimize parallel sampling by batching add_request calls](https://github.com/vllm-project/vllm/issues/16373) (accessed 2025-11-02):

**Performance optimization**: Batch all parallel samples together rather than adding them sequentially to the engine. This allows:
- Prefilling all samples simultaneously
- Eliminating latency from staggered scheduling
- Better GPU utilization

```python
# Inefficient (sequential)
for i in range(n):
    engine.add_request(child_request_i)  # Each waits for previous

# Efficient (batched)
engine.add_requests_batch([child_request_0, ..., child_request_n])
```

## 5. Stopping Criteria

### Stop Strings

From [vLLM Documentation - Sampling Parameters](https://docs.vllm.ai/en/v0.8.1/api/inference_params.html) (accessed 2025-11-02):

```python
sampling_params = SamplingParams(
    stop=[".", "\n", "END"],  # Stop on these strings
    include_stop_str_in_output=False,  # Exclude stop strings from output
    max_tokens=100
)

# Example: Generate until period
sampling_params = SamplingParams(
    stop=["."],
    max_tokens=200
)
# Output: "The cat sat on the mat"  (stops at first period)
```

**Behavior**:
- Generation stops immediately when any stop string is generated
- Multiple stop strings are OR'd together (stop at first match)
- `include_stop_str_in_output` controls whether stop string appears in result

### Stop Token IDs

```python
# Stop at specific token IDs (EOS or custom)
sampling_params = SamplingParams(
    stop_token_ids=[tokenizer.eos_token_id, 128009],  # EOS + custom
    max_tokens=100
)

# Common use: Stop at EOS but also at newlines
stop_ids = [tokenizer.eos_token_id] + tokenizer.encode("\n\n")
sampling_params = SamplingParams(stop_token_ids=stop_ids)
```

**Note**: Stop tokens are included in output unless they are special tokens (like EOS).

### Ignore EOS

```python
# Continue generating after EOS token
sampling_params = SamplingParams(
    ignore_eos=True,
    max_tokens=200
)

# Use case: Force model to generate exactly max_tokens
# Useful for fixed-length outputs or when EOS is premature
```

### Min Tokens

From [vLLM Documentation - Sampling Parameters](https://docs.vllm.ai/en/v0.8.1/api/inference_params.html):

```python
# Prevent early stopping before min_tokens
sampling_params = SamplingParams(
    min_tokens=50,  # Must generate at least 50 tokens
    max_tokens=200,
    stop=["."]  # But these are ignored until min_tokens reached
)

# Use case: Ensure minimum output length
# Story generation, minimum response length, etc.
```

**Behavior**: EOS and stop_token_ids are ignored until `min_tokens` are generated.

### Early Stopping (Beam Search)

```python
# Beam search specific
sampling_params = SamplingParams(
    use_beam_search=True,
    early_stopping=True  # Stop when beam_width sequences complete
)
```

See Beam Search section for details.

### Truncate Prompt Tokens

```python
# Limit prompt length (left truncation)
sampling_params = SamplingParams(
    truncate_prompt_tokens=512,  # Use only last 512 prompt tokens
    max_tokens=100
)

# Use case: Handle prompts exceeding context window
# Keeps most recent context (right side of prompt)
```

### Custom Stopping Logic

```python
# Using logits processors for custom stopping
def custom_stop_processor(input_ids, logits):
    """Stop if generated specific pattern."""
    if detect_pattern(input_ids):
        # Set all logits to -inf except EOS
        logits[:] = float('-inf')
        logits[eos_token_id] = 0
    return logits

sampling_params = SamplingParams(
    logits_processors=[custom_stop_processor],
    max_tokens=200
)
```

## 6. Seed-Based Deterministic Sampling

### Random Seed for Reproducibility

From [vLLM Documentation - Sampling Parameters](https://docs.vllm.ai/en/v0.8.1/api/inference_params.html) (accessed 2025-11-02):

```python
sampling_params = SamplingParams(
    temperature=0.8,  # Non-zero temperature
    seed=42,  # Fixed seed for reproducibility
    max_tokens=100
)

# Same prompt + same seed + same temperature = same output
output1 = llm.generate(["Hello"], sampling_params)
output2 = llm.generate(["Hello"], sampling_params)
assert output1[0].outputs[0].text == output2[0].outputs[0].text
```

### Seed Behavior

From [Medium - From Probabilistic to Predictable](https://medium.com/@adnanmasood/from-probabilistic-to-predictable-engineering-near-deterministic-llm-systems-for-consistent-6e8e62cf45f6) (accessed 2025-11-02):

> If temperature=0, the seed shouldn't matter (no sampling), but if you had temperature > 0, that seed would ensure the same random choices are made.

**Key principles**:
- `temperature=0`: Deterministic greedy decoding, seed has no effect
- `temperature>0`: Seed controls random number generator for sampling
- Same seed + same prompt + same parameters = reproducible outputs
- Different seeds = different samples from same distribution

### Use Cases for Seeded Sampling

1. **Testing and debugging**:
```python
# Reproducible outputs for unit tests
test_sampling_params = SamplingParams(
    temperature=0.7,
    seed=12345,
    max_tokens=50
)
# Always generates same output for validation
```

2. **Controlled A/B testing**:
```python
# Test different prompts with same randomness
params_v1 = SamplingParams(seed=100, temperature=0.8)
params_v2 = SamplingParams(seed=100, temperature=0.8)

output_v1 = llm.generate(["Prompt V1"], params_v1)
output_v2 = llm.generate(["Prompt V2"], params_v2)
# Same random decisions, only prompt changed
```

3. **Reproducible research**:
```python
# Document seed for experiment reproducibility
experiment_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    seed=42,  # Document this in paper
    max_tokens=200
)
```

4. **Deterministic multi-turn conversations**:
```python
conversation_seed = 98765

for turn in conversation:
    params = SamplingParams(
        seed=conversation_seed,
        temperature=0.7
    )
    response = llm.generate([turn], params)
    conversation_seed += 1  # Increment for each turn
```

### Seed Limitations

**Not fully deterministic across**:
- Different hardware (GPU models, drivers)
- Different vLLM versions
- Different batch sizes or tensor parallel settings
- Different quantization schemes

```python
# Seed provides sampling reproducibility, but:
# - Hardware differences may cause slight numerical variations
# - Cross-platform reproducibility not guaranteed
# - Within same environment: highly reproducible
```

### Combining Seed with Temperature

```python
# Fully deterministic (greedy)
params = SamplingParams(temperature=0.0, seed=42)
# Seed has no effect, always picks argmax

# Reproducible sampling
params = SamplingParams(temperature=0.8, seed=42)
# Same seed = same samples every time

# Different samples from same distribution
params1 = SamplingParams(temperature=0.8, seed=42)
params2 = SamplingParams(temperature=0.8, seed=43)
# Different outputs, same randomness level
```

### Per-Request Seeding

```python
# Different seed for each request
import random

seeds = [random.randint(0, 2**32-1) for _ in range(batch_size)]

outputs = []
for i, prompt in enumerate(prompts):
    params = SamplingParams(
        temperature=0.8,
        seed=seeds[i],
        max_tokens=100
    )
    output = llm.generate([prompt], params)
    outputs.append(output)

# Log seeds for potential reproduction
print(f"Used seeds: {seeds}")
```

## Advanced Combinations

### Structured Output + Logprobs

```python
# Get confidence scores for structured outputs
sampling_params = SamplingParams(
    logprobs=3,
    guided_decoding=GuidedDecodingParams(
        guided_json=schema
    )
)

# Analyze confidence in JSON generation
output = llm.generate(["Extract entities"], sampling_params)
for token_logprobs in output.logprobs:
    # Examine probability of JSON tokens
    pass
```

### Beam Search + Min Tokens

```python
# Force minimum length in beam search
sampling_params = SamplingParams(
    use_beam_search=True,
    best_of=5,
    min_tokens=30,  # Ensure substantive response
    length_penalty=1.1
)
```

### Parallel Sampling + Seed + Logprobs

```python
# Reproducible diverse outputs with confidence
base_seed = 42
params = SamplingParams(
    n=5,
    temperature=0.8,
    logprobs=5,
    seed=base_seed
)

# All 5 outputs are reproducible and include confidence scores
outputs = llm.generate(["Explain AI"], params)
```

## Performance Best Practices

### Guided Decoding
- Reuse schemas across requests for XGrammar caching
- Use simplest constraint type that meets requirements
- Monitor TTFT for user experience impact
- Let vLLM auto-select backend

### Beam Search
- Use for quality-critical tasks (translation, summarization)
- Avoid for latency-sensitive applications
- Consider best_of for quality without full beam overhead
- Optimize with numpy operations (argpartition vs sort)

### Logprobs
- Request only needed logprobs (higher values = more overhead)
- Use for confidence scoring, not every generation
- Filter special tokens in post-processing
- Combine with reranking for quality improvements

### Parallel Sampling
- Batch parallel samples for efficiency
- Use for self-consistency and diversity
- Prefer over sequential requests
- Monitor memory usage (n sequences in parallel)

### Stopping Criteria
- Use stop strings for natural termination
- Set min_tokens to prevent premature stopping
- Combine stop strategies (strings + token IDs)
- Include_stop_str based on use case

### Seed-Based Sampling
- Document seeds for reproducibility
- Use same environment for strict reproducibility
- Increment seeds for conversation turns
- Combine with temperature for controlled randomness

## Sources

### Web Research

**vLLM Official Blog**:
- [Structured Decoding in vLLM: a gentle introduction](https://blog.vllm.ai/2025/01/14/struct-decode-intro.html) - vLLM Blog (accessed 2025-11-02)

**vLLM Official Documentation**:
- [Inference Parameters - vLLM](https://docs.vllm.ai/en/v0.8.1/api/inference_params.html) (accessed 2025-11-02)
- [vllm.sampling_params](https://docs.vllm.ai/en/stable/api/vllm/sampling_params.html) (accessed 2025-11-02)
- [Sampling Parameters - vLLM (v0.4.0)](https://docs.vllm.ai/en/v0.4.0.post1/dev/sampling_params.html) (accessed 2025-11-02)
- [Sampling Parameters - vLLM (v0.5.0)](https://docs.vllm.ai/en/v0.5.0/dev/sampling_params.html) (accessed 2025-11-02)
- [parallel_sampling - vLLM](https://docs.vllm.ai/en/v0.10.2/api/vllm/v1/engine/parallel_sampling.html) (accessed 2025-11-02)

**Red Hat Developer**:
- [Structured outputs in vLLM: Guiding AI responses](https://developers.redhat.com/articles/2025/06/03/structured-outputs-vllm-guiding-ai-responses) - Red Hat Developer (accessed 2025-11-02)

**HPE Developer Portal**:
- [Using structured outputs in vLLM](https://developer.hpe.com/blog/using-structured-outputs-in-vllm/) - HPE Developer Portal (accessed 2025-11-02)

**GitHub Issues and Discussions**:
- [Issue #20316: Optimize beam search code](https://github.com/vllm-project/vllm/issues/20316) - vllm-project/vllm (accessed 2025-11-02)
- [Issue #4772: Unexpected Special Tokens in prompt_logprobs Output](https://github.com/vllm-project/vllm/issues/4772) - vllm-project/vllm (accessed 2025-11-02)
- [Issue #16373: Optimize parallel sampling by batching add_request calls](https://github.com/vllm-project/vllm/issues/16373) - vllm-project/vllm (accessed 2025-11-02)

**vLLM Forums**:
- [What is the purpose of prompt logprobs?](https://discuss.vllm.ai/t/what-is-the-purpose-of-prompt-logprobs/1714) - vLLM Forums (accessed 2025-11-02)

**Medium Articles**:
- [From Probabilistic to Predictable: Engineering Near-Deterministic LLM Systems](https://medium.com/@adnanmasood/from-probabilistic-to-predictable-engineering-near-deterministic-llm-systems-for-consistent-6e8e62cf45f6) - Medium, Adnan Masood (accessed 2025-11-02)

### Additional References

**Related Technologies**:
- XGrammar: [https://github.com/mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar) - Optimized grammar-based decoding backend
- Guidance (llguidance): [https://github.com/guidance-ai/llguidance](https://github.com/guidance-ai/llguidance) - Per-token constraint calculation
- Outlines: [https://github.com/dottxt-ai/outlines](https://github.com/dottxt-ai/outlines) - FSM-based guided generation

**arXiv Papers**:
- Willard, B. T., & Louf, R. (2023). Efficient Guided Generation for Large Language Models. arXiv:2307.09702
