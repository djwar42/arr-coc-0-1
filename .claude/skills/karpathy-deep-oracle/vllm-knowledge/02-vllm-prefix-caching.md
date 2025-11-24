# vLLM Automatic Prefix Caching

## Overview

**Automatic Prefix Caching (APC)** in vLLM is an optimization technique that eliminates redundant computation of shared prompt prefixes by caching and reusing Key-Value (KV) blocks across multiple requests. When a new request shares the same token sequence prefix as previous requests, vLLM automatically reuses the cached KV blocks, avoiding duplicate computation of the shared portion.

### What is Prefix Caching?

Prefix caching addresses a fundamental inefficiency in LLM inference: when multiple requests share common prompt prefixes (system prompts, document contexts, few-shot examples), naively recomputing the KV cache for these shared sequences wastes computational resources. APC solves this by:

1. **Identifying shared prefixes** - Detecting when requests have identical token sequences at the beginning
2. **Caching KV blocks** - Storing computed Key-Value tensors in memory
3. **Automatic reuse** - Transparently reusing cached blocks when prefix matches occur
4. **Memory management** - Using eviction policies to manage cache size and optimize hit rates

### Automatic vs Manual

**Automatic Prefix Caching** operates transparently without user intervention. When enabled, vLLM:
- Automatically detects prefix matches using block-level hashing
- Manages cache eviction using LRU (Least Recently Used) policy
- Handles partial matches and block alignment
- Requires only a single configuration flag: `enable_prefix_caching=True`

**Manual prefix caching** would require users to explicitly specify which prefixes to cache and manage cache lifetimes - vLLM does not expose this complexity.

### Key Use Cases

**1. Long Document Question Answering**
```python
# Same long document, multiple questions
document = "..." # 10,000 tokens of context
questions = [
    "What is the main argument?",
    "Who are the key figures mentioned?",
    "What evidence supports the conclusion?"
]
# Only document is computed once, questions computed separately
```

**2. Multi-Turn Conversations**
```python
# Shared system prompt + conversation history
system_prompt = "You are a helpful AI assistant..." # 500 tokens
conversation_history = [...] # Growing history
user_message = "New question" # Only new part computed
```

**3. Few-Shot Learning with Examples**
```python
# Same examples, different queries
examples = """
Example 1: ...
Example 2: ...
Example 3: ...
""" # 2,000 tokens cached
query = "Classify this: ..." # Only query computed
```

**4. Batch Processing with Shared Context**
```python
# RAG with same retrieved context
retrieved_docs = "..." # 5,000 tokens
queries = ["query1", "query2", "query3"]
# Context cached once, queries processed in batch
```

From [vLLM Documentation - Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching.html) (accessed 2025-02-02):
- "The core idea is simple – we cache the kv-cache blocks of processed requests, and reuse these blocks when a new request comes in with the same prefix as previous requests"

From [Medium: vLLM Prefix Caching](https://medium.com/byte-sized-ai/vllm-prefix-caching-vllms-automatic-prefix-caching-vs-chunkattention-749108317621) (accessed 2025-02-02):
- "The prefix KV caching mechanism in vLLM enhances large language model inference by reusing previously computed key-value pairs from attention layers"

---

## Block-Level Hashing Algorithm

vLLM's prefix caching uses a **block-level hashing** approach rather than token-level matching. This design choice balances flexibility with implementation simplicity.

### Hash-Based Block Identification

Each KV block is uniquely identified by:
```python
block_hash = hash(prefix_tokens + block_tokens)
```

Where:
- **prefix_tokens**: All tokens before this block
- **block_tokens**: Tokens within this specific block (typically 16-32 tokens per block)

From [GitHub Issue #2614 - RFC: Automatic Prefix Caching](https://github.com/vllm-project/vllm/issues/2614) (accessed 2025-02-02):
- "Every block in the KV cache can be uniquely identified by hash(prefix tokens, tokens in this block)"
- "We can add another indirection in vLLM's KV cache management: Logical block table → hash table → physical block table"

### Architecture: Three-Level Mapping

vLLM implements prefix caching using three layers of indirection:

```
Request Sequence
    ↓
Logical Block Table (per request)
    ↓
Hash Table (global, maps hash → physical block)
    ↓
Physical Block Table (actual KV cache storage)
```

**Example:**
```
Request 1: "The quick brown fox jumps over"
  Logical blocks: [0, 1, 2]
  Hashes: [hash1, hash2, hash3]
  Physical blocks: [phys_10, phys_11, phys_12]

Request 2: "The quick brown fox sleeps under"
  Logical blocks: [0, 1, 2]
  Hashes: [hash1, hash2, hash4]  # First two match!
  Physical blocks: [phys_10, phys_11, phys_20]  # Reuse first two
```

### Block Metadata

Each cached block maintains:
- **Block hash**: Unique identifier
- **Reference count**: Number of sequences using this block
- **Last accessed time**: For LRU eviction
- **Total access count**: Usage frequency statistics
- **Prefix length**: Position in the token sequence
- **Completion status**: Whether block is fully populated

### Complete vs Partial Blocks

vLLM distinguishes between:

**Complete blocks**: Fully populated with tokens (e.g., 16/16 tokens)
- Eligible for caching and sharing
- Stored in global hash table
- Can be reused across requests

**Partial blocks**: Not yet full (e.g., 7/16 tokens)
- Not added to global hash table
- Cannot be shared until complete
- Managed separately per request

From [Medium: SGLang vs vLLM - Block-Level Hashing](https://medium.com/byte-sized-ai/prefix-caching-sglang-vs-vllm-token-level-radix-tree-vs-block-level-hashing-b99ece9977a1) (accessed 2025-02-02):
- "vLLM implements prefix caching using block-level hashing where all blocks containing the same tokens and prefix are cacheable"
- "Only complete KV blocks are cacheable. Partial blocks are kept out of the hash table until they become complete"

### Hash Function Characteristics

**Performance overhead**: ~100-200ns per token (~6ms for 50k tokens of context)

The hash function must:
- Be fast (computed during prefill phase)
- Be deterministic (same tokens → same hash)
- Handle collisions gracefully
- Work with multi-modal inputs (tokens + image embeddings)

---

## Configuration & Usage

### Enabling Prefix Caching

**Python API**:
```python
from vllm import LLM, SamplingParams

# Enable prefix caching
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_prefix_caching=True,
    gpu_memory_utilization=0.9
)

# Generate as normal - caching is automatic
prompts = [
    "Long shared context... Question 1?",
    "Long shared context... Question 2?",
    "Long shared context... Question 3?"
]

outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
```

**OpenAI-Compatible Server**:
```bash
vllm serve meta-llama/Llama-2-7b-hf \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.9
```

**Docker**:
```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-2-7b-hf \
    --enable-prefix-caching
```

### Important Configuration Notes

1. **No additional parameters required** - Just set `enable_prefix_caching=True`
2. **Works with all sampling parameters** - Temperature, top-p, top-k all compatible
3. **Transparent to API clients** - No client-side changes needed
4. **Compatible with continuous batching** - Integrates with vLLM's scheduling

### Best Practices

**1. GPU Memory Allocation**
```python
# Reserve memory for cache
llm = LLM(
    model="...",
    enable_prefix_caching=True,
    gpu_memory_utilization=0.9,  # Leave room for cache
    max_num_seqs=256  # Balance batch size with cache memory
)
```

**2. Prompt Structure for Maximum Cache Hits**
```python
# ✅ GOOD: Shared prefix at the beginning
shared_context = "System: You are a helpful assistant...\n\nContext: ..."
queries = [
    shared_context + "\n\nUser: Question 1",
    shared_context + "\n\nUser: Question 2",
    shared_context + "\n\nUser: Question 3"
]

# ❌ BAD: Different prefixes
queries = [
    "User: Question 1\n\nContext: ...",
    "Context: ...\n\nUser: Question 2",  # Different order!
    "System: ...\nUser: Question 3"
]
```

**3. Block-Aligned Prefixes**
```python
# Blocks are typically 16-32 tokens
# For best cache utilization, shared prefix should be >> block_size

# ✅ GOOD: 1000+ token shared prefix
# High probability of multiple complete blocks matching

# ⚠️ SUBOPTIMAL: 20 token shared prefix
# Only 1-2 blocks, minimal cache benefit
```

### Limitations

1. **Exact match required**: Prefix must match exactly at token level
   - Even a single different token breaks the cache chain
   - Tokenizer-dependent (same text, different tokenizer = different tokens)

2. **Block granularity**: Caching happens at block level (16-32 tokens)
   - Short prefixes may not align with block boundaries
   - Partial block at end of prefix cannot be cached

3. **Memory overhead**: Cached blocks consume GPU memory
   - May reduce batch size capacity
   - Monitor `gpu_memory_utilization` carefully

4. **Eviction under memory pressure**: LRU policy may evict useful blocks
   - No manual control over which blocks to keep
   - High-frequency prefixes should be cached, but not guaranteed

5. **No cross-request guarantees in distributed settings**:
   - Tensor parallelism: Cache shared across GPUs in same instance
   - Pipeline parallelism: Cache per pipeline stage
   - Separate instances: No cache sharing

From [vLLM Documentation - Features: Automatic Prefix Caching](https://docs.vllm.ai/en/v0.10.2/features/automatic_prefix_caching.html) (accessed 2025-02-02):
- "Automatic Prefix Caching (APC in short) caches the KV cache of existing queries, so that a new query can directly reuse the KV cache if it shares the same prefix"

---

## Performance Analysis

### Cache Hit Rates

Cache hit rate depends on workload characteristics:

**High hit rate scenarios (>80%)**:
- Multi-turn conversations with consistent system prompts
- Batch processing with shared document context
- Few-shot learning with fixed examples
- RAG applications querying same documents

**Medium hit rate scenarios (40-80%)**:
- Mixed workloads with some prefix overlap
- Long contexts with varied suffixes
- Multi-user chat with common instructions

**Low hit rate scenarios (<40%)**:
- Completely unique prompts per request
- Short contexts (<100 tokens)
- Highly variable prompt structures

### Latency Reduction

From [Runpod: SGLang vs vLLM Benchmarks](https://www.runpod.io/blog/sglang-vs-vllm-kv-cache) (accessed 2025-02-02):

**7k token context benchmark (DeepSeek-R1-Distill-Llama-70B on 2x H100 SXM)**:

| Test Type | Duration | Prompt Tokens | Response Tokens | Speed (tok/s) | Speedup |
|-----------|----------|---------------|-----------------|---------------|---------|
| Fresh (no cache) | 5.253s | 6,801 | 150 | 28.6 | Baseline |
| Cache hit 1 | 4.572s | 6,801 | 150 | 32.8 | 1.15x |
| Cache hit 2 | 4.510s | 6,801 | 150 | 33.3 | 1.16x |
| Small context | 4.124s | 241 | 149 | 36.1 | - |

**Key insights**:
- **~15% latency reduction** on cache hits for large contexts (7k tokens)
- Cache performance approaches small prompt performance
- Benefit scales with prefix length (longer prefix = larger speedup)

### Memory Savings

Prefix caching provides memory efficiency through KV cache reuse:

**Example calculation**:
```
Model: Llama-2-7b (32 layers, 32 heads, 128 head_dim)
Prefix length: 2048 tokens
Precision: FP16

KV cache per token = 2 (K+V) × 32 layers × 32 heads × 128 dim × 2 bytes
                    = 524,288 bytes = 512 KB per token

Prefix cache size = 2048 tokens × 512 KB = 1 GB

With 10 concurrent requests sharing prefix:
- Without caching: 10 × 1 GB = 10 GB
- With caching: 1 GB (shared) + 10 × suffix_cache

Memory saved: 9 GB for this prefix alone
```

**Memory management**:
- Cached blocks have reference count > 0
- Eviction only occurs when GPU memory pressure detected
- LRU policy ensures frequently used prefixes remain cached

### Speedup Measurements

Speedup varies by:
1. **Prefix length**: Longer prefixes → larger benefit
2. **Request frequency**: More requests sharing prefix → better amortization
3. **Hardware**: Memory bandwidth affects cache lookup speed

**Rule of thumb**:
- Prefix < 100 tokens: Minimal benefit (<5% speedup)
- Prefix 100-1000 tokens: Moderate benefit (10-20% speedup)
- Prefix > 1000 tokens: Significant benefit (15-30% speedup)
- Prefix > 5000 tokens: Maximum benefit (~20-25% speedup, approaches overhead floor)

### Performance Overhead

From [vLLM Documentation - Prefix Caching Design](https://docs.vllm.ai/en/latest/design/prefix_caching.html) (accessed 2025-02-02):

**Hash computation overhead**: ~100-200ns per token
- For 50k token context: ~6ms total
- Negligible compared to attention computation (100ms+ for 50k tokens)

**Memory access overhead**: Minimal
- Hash table lookup: O(1) expected
- Physical block table access: Single indirection

**Cache miss penalty**: None
- If prefix not in cache, compute normally
- No additional overhead beyond hash computation

---

## Production Patterns

### Pattern 1: Multi-Turn Chat with System Prompt

**Scenario**: Customer support chatbot with long system instructions

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-13b-chat-hf",
    enable_prefix_caching=True
)

system_prompt = """You are a customer support agent for TechCorp.
Guidelines:
- Be polite and professional
- Reference our knowledge base
- Escalate to human if needed
[... 500 more tokens of instructions ...]
"""

def handle_conversation(conversation_history, user_message):
    # System prompt + history is cached across turns
    prompt = system_prompt + "\n\n"
    for turn in conversation_history:
        prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"
    prompt += f"User: {user_message}\nAssistant:"

    output = llm.generate(prompt, SamplingParams(max_tokens=200))
    return output[0].outputs[0].text

# First turn: Full computation
response1 = handle_conversation([], "How do I reset my password?")

# Second turn: System prompt + turn 1 cached
conversation_history = [{"user": "...", "assistant": response1}]
response2 = handle_conversation(conversation_history, "Thanks! What about 2FA?")

# Third turn: Even more cached
```

**Cache behavior**:
- Turn 1: System prompt blocks computed and cached
- Turn 2: System prompt cache HIT, turn 1 computed and cached
- Turn 3: System + turn 1 + turn 2 all cache HITs
- Each turn adds to the cached prefix

### Pattern 2: RAG with Context Reuse

**Scenario**: Multiple questions about the same retrieved documents

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    enable_prefix_caching=True,
    max_num_seqs=32  # Batch size
)

def query_with_context(retrieved_docs, questions):
    # Build context once
    context = "Context:\n"
    for doc in retrieved_docs:
        context += f"- {doc}\n"
    context += "\n\nAnswer the following question based on the context above.\n\n"

    # Create batch of prompts with shared context
    prompts = [context + f"Question: {q}\nAnswer:" for q in questions]

    outputs = llm.generate(prompts, SamplingParams(max_tokens=150))
    return [out.outputs[0].text for out in outputs]

# Retrieve documents once
docs = retrieve_documents("machine learning")  # 3000 tokens

# Ask multiple questions - context is cached
questions = [
    "What is gradient descent?",
    "Explain overfitting",
    "What are neural networks?"
]

answers = query_with_context(docs, questions)
# Context blocks cached after first question
# Questions 2-3 reuse cached context
```

**Cache efficiency**:
- Context: 3000 tokens → ~200 blocks cached
- Question: ~20 tokens → 1-2 blocks per question
- Cache hit rate: ~99% of computation reused

### Pattern 3: Few-Shot Learning with Fixed Examples

**Scenario**: Classification task with same examples, different inputs

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="gpt2",
    enable_prefix_caching=True
)

# Few-shot examples (cached)
examples = """Classify the sentiment of the following text as positive, negative, or neutral.

Example 1:
Text: "This movie was amazing!"
Sentiment: positive

Example 2:
Text: "I hated every minute of it."
Sentiment: negative

Example 3:
Text: "It was okay, nothing special."
Sentiment: neutral

Example 4:
Text: "Absolutely fantastic experience!"
Sentiment: positive

"""  # ~200 tokens, fully cached after first use

def classify_sentiment(text):
    prompt = examples + f"Text: \"{text}\"\nSentiment:"
    output = llm.generate(prompt, SamplingParams(max_tokens=5))
    return output[0].outputs[0].text.strip()

# Examples cached after first call
sentiments = [
    classify_sentiment("Great service!"),       # Cache MISS on examples
    classify_sentiment("Terrible quality"),     # Cache HIT on examples
    classify_sentiment("Average product"),      # Cache HIT on examples
    classify_sentiment("Love it!")              # Cache HIT on examples
]
```

### Pattern 4: Batch Processing with Streaming

**Scenario**: Real-time inference with shared prefix across batch

```python
from vllm import LLM, SamplingParams
import asyncio

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_prefix_caching=True,
    max_num_seqs=64
)

shared_instructions = """[Long system instructions...]
[Context about the domain...]
[Guidelines and constraints...]
"""  # 1000+ tokens

async def process_batch(requests):
    prompts = [
        shared_instructions + f"\n\nUser request: {req}"
        for req in requests
    ]

    # All requests share cached prefix
    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=200, temperature=0.7)
    )

    return [out.outputs[0].text for out in outputs]

# Continuous processing
while True:
    batch = await get_next_batch()  # Get requests from queue
    results = await process_batch(batch)
    await send_results(results)
    # Shared instructions remain cached across batches
```

From [vLLM Documentation - Example Workloads](https://docs.vllm.ai/en/v0.10.2/features/automatic_prefix_caching.html) (accessed 2025-02-02):
- "Long document query, where the user repeatedly queries the same document with different questions"
- "Multi-turn conversations with persistent system prompts and conversation history"

---

## Troubleshooting

### Issue 1: Low Cache Hit Rate

**Symptoms**:
- Expected speedup not observed
- Monitoring shows few cache reuses
- Similar prompts not hitting cache

**Diagnosis**:
```python
# Check if prefixes are truly identical
prompt1 = "Context: ..." + "Question: What is X?"
prompt2 = "Context: ..." + "Question: What is Y?"

# Tokenize to verify
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("model_name")

tokens1 = tokenizer.encode(prompt1)
tokens2 = tokenizer.encode(prompt2)

# Find common prefix length
common_len = 0
for t1, t2 in zip(tokens1, tokens2):
    if t1 == t2:
        common_len += 1
    else:
        break

print(f"Common prefix: {common_len} tokens")
print(f"Block alignment: {common_len // 16} complete blocks")
```

**Solutions**:
1. **Ensure exact token match**:
   ```python
   # Use same formatting consistently
   template = "Context: {context}\n\nQuestion: {question}"
   # Not: Sometimes "Context:", sometimes "Context :" (with space)
   ```

2. **Check tokenizer consistency**:
   ```python
   # Use same tokenizer instance
   # Verify fast_tokenizer vs slow_tokenizer settings
   tokenizer = AutoTokenizer.from_pretrained(
       model_name,
       use_fast=True  # Consistent setting
   )
   ```

3. **Verify block alignment**:
   ```python
   # Prefix length should be >> block_size
   # Aim for 100+ token shared prefix for meaningful benefit
   ```

### Issue 2: Cache Misses Due to Token Variations

**Symptoms**:
- Prompts look identical but don't hit cache
- Whitespace or formatting differences

**Common causes**:
```python
# ❌ BAD: Inconsistent whitespace
prompt1 = "Context:\n\nThis is the document."
prompt2 = "Context:\n This is the document."  # Single newline!

# ❌ BAD: Case sensitivity
prompt1 = "Context: the quick brown fox"
prompt2 = "Context: The quick brown fox"  # Capital T!

# ❌ BAD: Trailing whitespace
prompt1 = "Context: ..."
prompt2 = "Context: ...  "  # Extra spaces!

# ✅ GOOD: Exact match
def format_prompt(context, question):
    return f"Context: {context.strip()}\n\nQuestion: {question.strip()}\nAnswer:"
```

### Issue 3: Memory Pressure Evicting Cache

**Symptoms**:
- High cache hit rate initially, then drops
- GPU memory utilization near 100%
- Frequent cache evictions in logs

**Solutions**:
```python
# 1. Reduce batch size to leave room for cache
llm = LLM(
    model="...",
    enable_prefix_caching=True,
    max_num_seqs=32,  # Lower than default
    gpu_memory_utilization=0.85  # More conservative
)

# 2. Use quantization to free memory
llm = LLM(
    model="...",
    enable_prefix_caching=True,
    quantization="awq",  # or "gptq"
    gpu_memory_utilization=0.9
)

# 3. Monitor memory usage
from vllm import LLM
llm = LLM(...)

# Check cache statistics (if available in version)
# Look for cache_hit_rate, eviction_count in metrics
```

### Issue 4: Prefix Caching Not Working with Multi-Modal Inputs

**Symptoms**:
- Image + text prompts don't benefit from caching
- Cache hits expected but not occurring

**Diagnosis**:
```python
# Multi-modal inputs include both tokens and embeddings
# Hash must account for image embeddings, not just tokens

# Check if your version supports multi-modal prefix caching
# (Feature added in later vLLM versions)
```

**Solutions**:
1. Update to latest vLLM version with multi-modal support
2. Structure prompts to maximize text-only prefix:
   ```python
   # ✅ GOOD: Text prefix before image
   prompt = {
       "text": "System: You are a helpful assistant.\n\nAnalyze this image:",
       "image": image_data
   }
   # Text portion can be cached

   # ❌ BAD: Image first
   prompt = {
       "image": image_data,
       "text": "Analyze this image"
   }
   # No cacheable prefix
   ```

### Issue 5: Debugging Cache Behavior

**Enable verbose logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

llm = LLM(
    model="...",
    enable_prefix_caching=True
)

# Check logs for cache hit/miss information
# Look for messages like:
# "Cache hit for blocks [0, 1, 2, 3]"
# "Cache miss, computing blocks [4, 5]"
```

**Monitor cache statistics**:
```bash
# Server mode with metrics endpoint
vllm serve model_name \
    --enable-prefix-caching \
    --port 8000

# Query metrics
curl http://localhost:8000/metrics | grep cache
# Look for: vllm:cache_hit_rate, vllm:cache_size_bytes
```

### Issue 6: Preemption Affecting Cache

**Symptoms**:
- Sequences preempted and lose cache benefits
- Inconsistent performance under load

**Explanation**:
- When GPU memory fills, vLLM may preempt in-progress sequences
- Preempted sequences lose their KV cache (including cached blocks)
- On resumption, must recompute even cached portions

**Solutions**:
1. **Increase memory headroom**:
   ```python
   llm = LLM(
       model="...",
       enable_prefix_caching=True,
       gpu_memory_utilization=0.8,  # More conservative
       max_num_seqs=16  # Smaller batch size
   )
   ```

2. **Use swap space** (if supported):
   ```python
   llm = LLM(
       model="...",
       enable_prefix_caching=True,
       swap_space=4  # GB of CPU memory for swapping
   )
   ```

From [GitHub Issue #2614 - Automatic Prefix Caching RFC](https://github.com/vllm-project/vllm/issues/2614) (accessed 2025-02-02):
- "The preemption policy is kept the same as before"
- "Better preemption strategy for OOM cases" listed as P2 future work

---

## Comparison: vLLM vs SGLang RadixAttention

Both vLLM and SGLang implement prefix caching, but use different algorithms with distinct trade-offs.

### vLLM: Block-Level Hashing

**Architecture**:
- Hash table maps `hash(prefix + block) → physical block`
- Requires exact block-level match
- Manages cache with LRU eviction policy

**Strengths**:
- Simple, efficient hash table lookup
- Low memory overhead
- Fast eviction decisions
- Extensible to sliding window attention and attention sinks

**Weaknesses**:
- Requires exact prefix match at block boundaries
- Less flexible for partial prefix matches
- Manual optimization may be needed for maximum hit rates

**Best for**:
- Batch inference with templated prompts
- Predictable request patterns
- High-throughput scenarios with exact prefix matches
- Fine-grained control over caching

### SGLang: RadixAttention (Token-Level Radix Tree)

**Architecture**:
- Radix tree structure for flexible prefix matching
- Token-level granularity
- Automatic tree pruning and management

**Strengths**:
- Flexible partial prefix matching
- Zero configuration optimization
- Better for dynamic conversation flows
- Handles branching conversations naturally

**Weaknesses**:
- More complex tree maintenance
- Higher memory overhead per cached entry
- Slower lookup for very deep trees

**Best for**:
- Unpredictable dialog flows
- Multi-turn conversations with variation
- Customer support chatbots
- Coding assistants with evolving context

### Performance Comparison

From [Runpod: SGLang vs vLLM Benchmarks](https://www.runpod.io/blog/sglang-vs-vllm-kv-cache) (accessed 2025-02-02):

**7k context benchmark (DeepSeek-R1-Distill-Llama-70B on 2x H100 SXM)**:

| Engine | Fresh | Cache Hit | Cache Hit 2 | Small Context |
|--------|-------|-----------|-------------|---------------|
| **vLLM** | 5.253s (28.6 tok/s) | 4.572s (32.8 tok/s) | 4.510s (33.3 tok/s) | 4.124s (36.1 tok/s) |
| **SGLang** | 5.093s (29.5 tok/s) | 4.287s (35.0 tok/s) | 4.295s (34.9 tok/s) | 4.154s (36.1 tok/s) |

**Key insights**:
- Fresh context: Similar performance (~2% difference)
- Cache hits: SGLang ~7% faster (10-20% vs 15% improvement over fresh)
- SGLang cache performance closer to small-context baseline
- Both achieve significant speedup, SGLang slightly better for multi-turn

### Decision Guide

**Choose vLLM prefix caching when**:
- Running batch inference with templated prompts
- Exact prefix patterns are consistent
- Need maximum throughput control
- Working with structured workflows
- Prefer explicit configuration

**Choose SGLang RadixAttention when**:
- Building conversational AI with varied flows
- Customer support, tutoring, coding assistance
- Large context windows with partial overlaps
- Prioritizing zero-configuration
- Unpredictable conversation branches

From [Medium: SGLang vs vLLM Comparison](https://medium.com/byte-sized-ai/prefix-caching-sglang-vs-vllm-token-level-radix-tree-vs-block-level-hashing-b99ece9977a1) (accessed 2025-02-02):
- "vLLM excels when you can predict and structure your caching patterns"
- "SGLang shines in unpredictable, dynamic scenarios where conversation flows vary"

---

## Sources

**vLLM Official Documentation:**
- [Automatic Prefix Caching - Design](https://docs.vllm.ai/en/latest/design/prefix_caching.html)
- [Automatic Prefix Caching - Features](https://docs.vllm.ai/en/v0.10.2/features/automatic_prefix_caching.html)

**GitHub:**
- [RFC: Automatic Prefix Caching - Issue #2614](https://github.com/vllm-project/vllm/issues/2614) - Original design discussion

**Web Research (accessed 2025-02-02):**
- [Medium: vLLM Prefix Caching vs ChunkAttention](https://medium.com/byte-sized-ai/vllm-prefix-caching-vllms-automatic-prefix-caching-vs-chunkattention-749108317621) by Don Moon
- [Medium: SGLang vs vLLM - Block-Level Hashing](https://medium.com/byte-sized-ai/prefix-caching-sglang-vs-vllm-token-level-radix-tree-vs-block-level-hashing-b99ece9977a1) by Don Moon
- [Runpod: SGLang vs vLLM Performance Benchmarks](https://www.runpod.io/blog/sglang-vs-vllm-kv-cache)

**Related Papers:**
- RadixAttention paper referenced in comparison sections
- Attention Sinks and sliding window attention mentioned in design notes
