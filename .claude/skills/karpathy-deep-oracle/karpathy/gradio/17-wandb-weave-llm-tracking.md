# W&B Weave: LLM Observability & Application Tracking

**Comprehensive guide to Weights & Biases Weave for LLM application tracking, tracing, and observability**

---

## Overview

W&B Weave is a toolkit for developing, tracking, and improving Generative AI applications built by Weights & Biases. It provides comprehensive observability for LLM applications through automatic tracing, evaluation tools, and production monitoring capabilities.

### What is Weave vs Classic W&B?

**Classic W&B (Training Focus):**
- Experiment tracking for model training
- Hyperparameter optimization
- Model versioning and artifacts
- Training run comparisons

**W&B Weave (LLM App Focus):**
- LLM call tracing (inputs, outputs, metadata)
- Token usage and cost tracking
- Latency and performance monitoring
- Evaluation frameworks for LLM apps
- Production inference monitoring

**Key Difference:** Classic W&B tracks training experiments; Weave tracks LLM application execution and production usage.

From [W&B Weave Documentation](https://docs.wandb.ai/weave) (accessed 2025-01-31):
> "W&B Weave helps you build better language model apps. Use Weave to track, test, and improve your apps"

---

## Section 1: Weave Fundamentals

### Core Concepts

**Calls** are the fundamental building block in Weave. A Call represents:
- Single execution of a function
- Input parameters (arguments)
- Output value (return)
- Metadata (duration, exceptions, LLM usage, cost)

**Traces** are collections of Calls in the same execution context:
- Form a tree structure (parent/child relationships)
- Similar to OpenTelemetry spans
- Enable end-to-end flow visualization

**Ops** are functions/methods decorated with `@weave.op()`:
- Automatically tracked by Weave
- Produce Calls when executed
- Can be customized with display names and attributes

### Installation and Setup

**Install Weave:**
```bash
pip install weave
```

**Initialize tracking:**
```python
import weave

# Initialize Weave for your project
weave.init('project-name')

# For team projects: 'team-name/project-name'
weave.init('wandb-team/my-llm-app')
```

**Authentication:**
- Create account at https://wandb.ai
- Get API key from https://wandb.ai/authorize
- Weave prompts for API key on first run
- Or set `WANDB_API_KEY` environment variable

From [W&B Weave Quickstart](https://docs.wandb.ai/weave/quickstart) (accessed 2025-01-31):
> "Weave automatically tracks calls made to OpenAI, Anthropic and many more LLM libraries and logs their LLM metadata, token usage and cost"

### Automatic vs Manual Tracing

**Automatic Tracing (Integrated Libraries):**

Weave automatically instruments popular LLM libraries:
- OpenAI
- Anthropic
- Cohere
- Mistral
- LangChain
- LlamaIndex

Just initialize Weave - no decorators needed:
```python
import weave
from openai import OpenAI

weave.init('my-project')
client = OpenAI()

# Automatically traced!
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**Manual Tracing (Custom Functions):**

Decorate functions you want to track:
```python
import weave

weave.init('my-project')

@weave.op()
def extract_entities(text: str) -> dict:
    # Your custom logic
    return {"entities": ["entity1", "entity2"]}

result = extract_entities("Sample text")
```

**When to use each:**
- Automatic: LLM API calls, supported frameworks
- Manual: Custom logic, preprocessing, post-processing, business logic

### Three Ways to Create Calls

**1. Automatic tracking (preferred for LLM libraries):**
```python
import weave
from openai import OpenAI

weave.init('project')
client = OpenAI()

# Automatically creates Call
response = client.chat.completions.create(...)
```

**2. Decorator-based tracking (custom functions):**
```python
@weave.op()
def my_function(input: str) -> str:
    return f"Processed: {input}"

result = my_function("data")
```

**3. Manual Call tracking (advanced use cases):**
```python
client = weave.init('project')

call = client.create_call(
    op="custom_operation",
    inputs={"param": "value"}
)

# Your logic here

client.finish_call(call, output="result")
```

From [W&B Tracing Quickstart](https://docs.wandb.ai/weave/guides/tracking/tracing) (accessed 2025-01-31):
> "Calls are the fundamental building block in Weave. They represent a single execution of a function, including inputs, outputs, and metadata"

---

## Section 2: LLM Application Tracking

### Tracking Prompts and Completions

**Basic prompt tracking:**
```python
import weave
from openai import OpenAI

weave.init('llm-app')
client = OpenAI()

@weave.op()
def generate_response(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

# Automatically logs prompt, response, and metadata
result = generate_response("Explain quantum computing")
```

**What gets tracked:**
- Full prompt text
- System messages
- User messages
- Model parameters (temperature, top_p, max_tokens)
- Complete response
- Token counts
- Latency
- Cost (automatically calculated)

### Token Usage and Cost Calculation

**Automatic token tracking:**

Weave automatically captures:
- `prompt_tokens`: Input token count
- `completion_tokens`: Output token count
- `total_tokens`: Sum of prompt + completion
- Estimated cost based on model pricing

**Accessing token data:**
```python
@weave.op()
def analyze_costs(queries: list[str]) -> dict:
    total_cost = 0
    for query in queries:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": query}]
        )
        # Token usage automatically logged in trace
    return {"queries_processed": len(queries)}
```

**Cost tracking in UI:**
- View costs per call
- Aggregate costs across traces
- Filter by date range
- Export cost reports

From [Medium: Weights & Biases Learn LLMs](https://medium.com/@nomannayeem/weights-biases-learn-llms-the-instrument-everything-way-423c6090c0f3) (accessed 2025-01-31):
> "Weights & Biases (W&B) is your ML/LLM control room: it tracks experiments, versions data and models, and shows prompt traces, quality, cost, and latency in one place"

### Latency and Performance Metrics

**Automatic timing:**

Every Call captures:
- `started_at`: Timestamp when call began
- `ended_at`: Timestamp when call finished
- `duration`: Calculated elapsed time

**Performance monitoring:**
```python
@weave.op()
def batch_process(items: list[str]) -> list[str]:
    results = []
    for item in items:
        # Each iteration timed separately
        result = llm_call(item)
        results.append(result)
    return results
```

**Latency insights:**
- Identify slow calls
- Compare performance across models
- Track performance degradation
- Filter by duration threshold

### Chain and Agent Tracing

**Nested call tracking:**

Weave automatically creates parent-child relationships:
```python
@weave.op()
def retrieve_context(query: str) -> str:
    # Database lookup
    return "relevant context"

@weave.op()
def generate_answer(query: str) -> str:
    # Parent call
    context = retrieve_context(query)  # Child call

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Context: {context}"},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content
```

**Trace tree visualization:**
```
generate_answer
├── retrieve_context
└── openai.chat.completions.create
```

**Agent workflow tracking:**
```python
@weave.op()
def agent_step(state: dict, action: str) -> dict:
    # Each agent step is a separate Call
    observation = execute_action(action)
    new_state = update_state(state, observation)
    return new_state

@weave.op()
def run_agent(task: str) -> str:
    state = initialize_state(task)
    for _ in range(max_steps):
        action = decide_action(state)  # Traced
        state = agent_step(state, action)  # Traced
        if is_complete(state):
            break
    return state["result"]
```

### Error Tracking and Debugging

**Exception capture:**

Weave automatically logs exceptions:
```python
@weave.op()
def risky_operation(data: str) -> str:
    if not data:
        raise ValueError("Empty input")
    return process(data)

# Exception automatically captured in Call
try:
    result = risky_operation("")
except ValueError as e:
    print(f"Error logged in Weave: {e}")
```

**Call metadata includes:**
- `exception`: Error message and type
- `status`: "success", "error", or "running"
- Full stack trace
- Input values that caused error

**Debugging workflow:**
1. Filter traces by status = "error"
2. Review exception message
3. Inspect inputs that caused failure
4. Replay with modifications
5. Verify fix in new trace

From [W&B Guide to LLM Debugging, Tracing, and Monitoring](https://wandb.ai/onlineinference/genai-research/reports/A-guide-to-LLM-debugging-tracing-and-monitoring--VmlldzoxMzk1MjAyOQ) (accessed 2025-01-31):
> "Tutorial: using W&B Weave for debugging, tracing, and monitoring. In this tutorial, we will explore how to integrate Weights & Biases Weave"

### Call Attributes and Metadata

**Adding custom attributes:**
```python
import weave

@weave.op()
def production_call(query: str) -> str:
    with weave.attributes({'env': 'production', 'version': 'v2.1'}):
        response = client.chat.completions.create(...)
        return response.choices[0].message.content
```

**Attributes use cases:**
- Environment tags (dev, staging, production)
- Version tracking (model version, prompt version)
- User identifiers
- Request metadata (source, priority)

**Call Summary:**

Add metrics during execution:
```python
@weave.op()
def analyze_sentiment(text: str) -> dict:
    call = weave.get_current_call()

    # Your analysis logic
    result = {"sentiment": "positive", "confidence": 0.95}

    # Add to summary
    call.summary["confidence"] = result["confidence"]
    call.summary["sentiment_category"] = result["sentiment"]

    return result
```

**Summary vs Attributes:**
- Attributes: Set before call starts (frozen during execution)
- Summary: Set during/after execution (mutable)

---

## Section 3: Framework Integration

### LangChain Integration

**Automatic tracing:**

Weave automatically instruments LangChain:
```python
import weave
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

weave.init('langchain-app')

# Automatically traced
llm = OpenAI(model="gpt-4")
prompt = PromptTemplate.from_template("Translate {text} to French")
chain = LLMChain(llm=llm, prompt=prompt)

result = chain.run(text="Hello, world!")
```

**What gets traced:**
- Chain execution
- Individual LLM calls
- Prompt construction
- Tool usage (if using agents)
- Retrieval steps (if using RAG)

From [CrewAI Weave Documentation](https://docs.crewai.com/en/observability/weave) (accessed 2025-01-31):
> "Weights & Biases (W&B) Weave is a framework for tracking, experimenting with, evaluating, deploying, and improving LLM-based applications"

### LlamaIndex Integration

**Automatic tracing:**
```python
import weave
from llama_index import VectorStoreIndex, SimpleDirectoryReader

weave.init('llamaindex-app')

# Load documents
documents = SimpleDirectoryReader('data').load_data()

# Build index (automatically traced)
index = VectorStoreIndex.from_documents(documents)

# Query (automatically traced)
query_engine = index.as_query_engine()
response = query_engine.query("What is the main topic?")
```

**Traced components:**
- Document loading
- Embedding generation
- Index construction
- Query execution
- Retrieval results

### OpenAI API Direct Tracking

**Comprehensive tracking:**
```python
import weave
from openai import OpenAI

weave.init('openai-app')
client = OpenAI()

# All parameters automatically logged
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"}
    ],
    temperature=0.8,
    max_tokens=100,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stream=False
)
```

**Tracked metadata:**
- All request parameters
- Response content
- Token usage
- Model name
- Finish reason
- Cost estimate

### Anthropic Claude Tracking

**Automatic instrumentation:**
```python
import weave
from anthropic import Anthropic

weave.init('claude-app')
client = Anthropic()

# Automatically traced
message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain AI safety"}
    ]
)
```

### Custom Model Tracking

**Wrapping custom APIs:**
```python
import weave
import requests

weave.init('custom-llm')

@weave.op()
def call_custom_llm(prompt: str) -> str:
    response = requests.post(
        "https://custom-llm-api.com/generate",
        json={
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 500
        }
    )
    return response.json()["text"]

result = call_custom_llm("Generate a summary")
```

**Local model tracking:**
```python
import weave
from transformers import pipeline

weave.init('local-models')

generator = pipeline('text-generation', model='gpt2')

@weave.op()
def generate_text_local(prompt: str) -> str:
    outputs = generator(
        prompt,
        max_length=100,
        num_return_sequences=1
    )
    return outputs[0]['generated_text']

result = generate_text_local("Once upon a time")
```

### Gradio + Weave Integration

**Track Gradio app interactions:**
```python
import weave
import gradio as gr
from openai import OpenAI

weave.init('gradio-app')
client = OpenAI()

@weave.op()
def chatbot_response(message: str, history: list) -> str:
    # Add environment metadata
    with weave.attributes({'interface': 'gradio', 'user_session': 'session_id'}):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": message}]
        )
        return response.choices[0].message.content

# Gradio interface
demo = gr.ChatInterface(
    fn=chatbot_response,
    title="LLM Chatbot with Weave Tracking"
)

demo.launch()
```

**What gets tracked:**
- Every user interaction
- Input messages
- Generated responses
- Session metadata
- Performance metrics

From [W&B Weave Documentation](https://docs.wandb.ai/weave) (accessed 2025-01-31):
> "Connect Weave to your code with: Python SDK, TypeScript SDK, Service API. Weave works with many language model providers, local models, and tools"

---

## Section 4: Advanced Features

### Multi-threaded Call Tracing

**Parallel execution tracking:**
```python
import weave
from weave import ThreadPoolExecutor

@weave.op()
def process_item(item: str) -> str:
    # Process individual item
    return f"Processed: {item}"

@weave.op()
def batch_process(items: list[str]) -> list[str]:
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_item, items))
    return results

# Creates hierarchical trace with parallel children
weave.init('parallel-app')
batch_process(["item1", "item2", "item3", "item4"])
```

**Trace structure:**
```
batch_process (parent)
├── process_item("item1") (parallel)
├── process_item("item2") (parallel)
├── process_item("item3") (parallel)
└── process_item("item4") (parallel)
```

### Generator Function Tracing

**Sync generator tracking:**
```python
import weave
from typing import Generator

weave.init('generator-app')

@weave.op()
def stream_responses(queries: list[str]) -> Generator[str, None, None]:
    for query in queries:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": query}]
        )
        yield response.choices[0].message.content

# Must consume generator to log outputs
results = list(stream_responses(["Q1", "Q2", "Q3"]))
```

**Async generator tracking:**
```python
@weave.op()
async def async_stream_responses(queries: list[str]):
    for query in queries:
        response = await async_client.chat.completions.create(...)
        yield response.choices[0].message.content

# Consume async generator
results = [r async for r in async_stream_responses(queries)]
```

### Display Name Customization

**Dynamic display names:**
```python
from datetime import datetime

@weave.op(call_display_name=lambda call: f"{call.func_name}_{datetime.now()}")
def timestamped_function():
    return "result"

# Or with custom attributes
def custom_display_name(call):
    model = call.attributes.get("model", "unknown")
    version = call.attributes.get("version", "v1")
    return f"{model}_{version}"

@weave.op(call_display_name=custom_display_name)
def versioned_call():
    return "result"

with weave.attributes({"model": "gpt-4", "version": "v2.1"}):
    versioned_call()  # Display: "gpt-4_v2.1"
```

### Post-processing for Readability

**Markdown rendering:**
```python
import weave

def postprocess_inputs(query: str) -> weave.Markdown:
    formatted = f"""
**Search Query:**
```
{query}
```
"""
    return {"search_box": weave.Markdown(formatted), "query": query}

def postprocess_output(result: dict) -> weave.Markdown:
    formatted = f"""
# {result["title"]}

{result["content"]}

[Read more]({result["url"]})
"""
    return weave.Markdown(formatted)

@weave.op(
    postprocess_inputs=postprocess_inputs,
    postprocess_output=postprocess_output
)
def search_and_display(query: str) -> dict:
    # Your search logic
    return {"title": "Result", "content": "...", "url": "https://..."}
```

### Configuring Autopatch

**Disable specific integrations:**
```python
# Disable all autopatching
weave.init('project', autopatch_settings={"disable_autopatch": True})

# Disable specific integration
weave.init('project', autopatch_settings={
    "openai": {"enabled": False}
})

# Custom postprocessing for autopatched calls
def redact_pii(inputs: dict) -> dict:
    if "email" in inputs:
        inputs["email"] = "[REDACTED]"
    return inputs

weave.init('project', autopatch_settings={
    "openai": {
        "op_settings": {
            "postprocess_inputs": redact_pii
        }
    }
})
```

### Disabling Tracing

**Environment variable (unconditional):**
```bash
export WEAVE_DISABLED=true
python app.py
```

**Client initialization (conditional):**
```python
import weave

# Disable for entire session
client = weave.init('project', settings={"disabled": True})
```

**Context manager (scoped):**
```python
from weave.trace.context.call_context import set_tracing_enabled

@weave.op()
def my_op():
    return "result"

# Temporarily disable
with set_tracing_enabled(False):
    my_op()  # Not traced

my_op()  # Traced normally
```

### Querying and Exporting Calls

**Python API:**
```python
import weave

client = weave.init('project')

# Get all calls
calls = client.get_calls()

# Filter by operation
calls = client.get_calls(filter={"op_name": "my_function"})

# Filter by status
error_calls = client.get_calls(filter={"status": "error"})

# Delete calls
call_ids = [c.id for c in calls[:100]]
client.delete_calls(call_ids=call_ids)
```

**Export formats:**
- JSON
- CSV
- Pandas DataFrame
- Python code snippets
- CURL commands

From [Reddit: Any thoughts on Weave from WandB?](https://www.reddit.com/r/mlops/comments/1i764f6/any_thoughts_on_weave_from_wandb/) (accessed 2025-01-31):
> "I really like Weave. It does exactly what it promises: with just a few lines of code and a decorator, you get great LLM monitoring"

---

## Sources

**W&B Official Documentation:**
- [W&B Weave Overview](https://docs.wandb.ai/weave) - Main documentation page (accessed 2025-01-31)
- [Track LLM inputs & outputs Quickstart](https://docs.wandb.ai/weave/quickstart) - Getting started guide (accessed 2025-01-31)
- [Tracing Quickstart](https://docs.wandb.ai/weave/guides/tracking/tracing) - Comprehensive tracing guide (accessed 2025-01-31)

**Web Research:**
- [Medium: Weights & Biases Learn LLMs](https://medium.com/@nomannayeem/weights-biases-learn-llms-the-instrument-everything-way-423c6090c0f3) - LLM tracking overview (accessed 2025-01-31)
- [W&B Guide to LLM Debugging, Tracing, and Monitoring](https://wandb.ai/onlineinference/genai-research/reports/A-guide-to-LLM-debugging-tracing-and-monitoring--VmlldzoxMzk1MjAyOQ) - Debugging tutorial (accessed 2025-01-31)
- [Reddit: Any thoughts on Weave from WandB?](https://www.reddit.com/r/mlops/comments/1i764f6/any_thoughts_on_weave_from_wandb/) - Community feedback (accessed 2025-01-31)
- [CrewAI Weave Documentation](https://docs.crewai.com/en/observability/weave) - Integration example (accessed 2025-01-31)

**Additional References:**
- [W&B Weave GitHub Repository](https://github.com/wandb/weave) - Source code and examples
- [PyPI Weave Package](https://pypi.org/project/weave/) - Package information
- [Comparing Custom LLM Evals with Weave, LangSmith, and Others](https://downpat.substack.com/p/comparing-custom-llm-evals-with-weave) - Platform comparison (accessed 2025-01-31)
