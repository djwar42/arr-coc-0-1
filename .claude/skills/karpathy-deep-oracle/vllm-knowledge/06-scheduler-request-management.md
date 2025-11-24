# vLLM Scheduler and Request Management

## Overview

The vLLM scheduler is the central orchestration component responsible for batching requests, managing the KV cache, and maximizing GPU utilization through continuous batching. It determines which requests advance at each engine step, balances prefill and decode workloads, and handles preemption when resources are constrained.

This document covers the complete lifecycle of requests through vLLM's scheduling system, from arrival to completion.

## Request Lifecycle

### Request States

Requests in vLLM V1 progress through several states:

**WAITING_FOR_FSM** → **WAITING** → **RUNNING** → **FINISHED**

From [Life of an inference request (vLLM V1)](https://www.ubicloud.com/blog/life-of-an-inference-request-vllm-v1) (accessed 2025-02-02):

1. **WAITING_FOR_FSM**: Request waiting for grammar compilation (guided decoding only)
2. **WAITING**: Request is ready to be scheduled but not yet executing
3. **RUNNING**: Request is actively being processed (prefill or decode)
4. **FINISHED**: Request has completed generation

### Request Flow Through the System

From [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) (accessed 2025-02-02):

**Step 1: Request Arrival**
```python
# Request enters through AsyncLLM
llm.generate(prompts, sampling_params)

# Tokenization happens asynchronously
tokens = tokenizer.encode(prompt)

# Request sent to EngineCore via IPC
engine_core.add_request(request_id, tokens, sampling_params)
```

**Step 2: Scheduler Queuing**
- Request wrapped in `Request` object with status `WAITING`
- Added to scheduler's `waiting` deque (FCFS) or priority heap (priority scheduling)
- Request includes metadata: `request_id`, `arrival_time`, `priority`, `sampling_params`

**Step 3: Engine Step Loop**

From the Ubicloud blog post:
```python
# Main engine loop
while has_requests:
    # Schedule: select requests for this step
    scheduled_requests = scheduler.schedule()

    # Forward pass: execute model
    outputs = model_executor.execute_model(scheduled_requests)

    # Postprocess: check completion, detokenize
    for request in outputs:
        if is_finished(request):
            cleanup_kv_cache(request)
            return_output(request)
```

## Scheduler Architecture

### Core Components

From the vLLM blog anatomy post:

**Scheduler Components:**
1. **Policy**: FCFS (first-come-first-serve) or Priority-based
2. **Waiting Queue**: Deque (FCFS) or heap (priority) of pending requests
3. **Running List**: Active requests being processed
4. **KV Cache Manager**: Manages GPU memory allocation for attention KV tensors

### Scheduling Algorithm

The V1 scheduler performs **continuous batching** with mixed prefill/decode execution.

From [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html):

**Priority Order:**
1. **Decode requests first** (from running queue) - these are memory-bandwidth-bound
2. **Prefill requests second** (from waiting queue) - these are compute-bound
3. **Token budget constraint**: `max_num_batched_tokens` (default varies by model)

**Scheduling Logic:**

```python
def schedule(self):
    scheduled = []
    token_budget = self.max_num_batched_tokens

    # Step 1: Schedule decode requests (already running)
    for req in self.running:
        num_tokens = compute_decode_tokens(req)  # Usually 1, more for speculative
        if not kv_cache_manager.allocate_slots(req, num_tokens):
            attempt_preemption(req)
            continue
        scheduled.append(req)
        token_budget -= num_tokens

    # Step 2: Schedule prefill requests (from waiting)
    while self.waiting and token_budget > 0:
        req = self.waiting.pop()

        # Check prefix cache hits
        num_cached = kv_cache_manager.get_computed_blocks(req)
        num_new = req.num_prompt_tokens - num_cached

        # Chunked prefill: limit tokens per step
        if num_new > self.long_prefill_token_threshold:
            num_new = self.long_prefill_token_threshold

        if num_new > token_budget:
            num_new = token_budget  # Partial prefill

        if not kv_cache_manager.allocate_slots(req, num_new):
            break  # Out of memory

        scheduled.append(req)
        token_budget -= num_new

        # Move to running if we started processing
        if num_new > 0:
            req.status = RUNNING
            self.running.append(req)

    return scheduled
```

### Continuous Batching

From the vLLM anatomy blog post, continuous batching allows **iteration-level scheduling**:

**Traditional Static Batching:**
- Wait for N requests to arrive
- Process entire batch start-to-finish
- All requests finish together

**Continuous Batching (vLLM):**
- Add new requests at every step
- Remove completed requests at every step
- Different requests at different stages (prefill vs decode)

**Example Timeline:**

```
Step 0: [R1 prefill: 3 toks] [R2 prefill: 5 toks] [R3 prefill: 2 toks]  (budget: 10)
Step 1: [R1 decode: 1 tok]   [R2 decode: 1 tok]   [R3 prefill: 8 toks]  (budget: 10)
Step 2: [R1 decode: 1 tok]   [R2 decode: 1 tok]   [R3 decode: 1 tok]    [R4 prefill: 7 toks]
Step 3: [R2 decode: 1 tok]   [R3 decode: 1 tok]   [R4 prefill: 8 toks]  (R1 finished!)
```

Key insight from the blog: "All tokens from selected requests are combined into a single large tensor and processed layer by layer. GPUs process these in parallel using SIMD(T) across CUDA cores."

## KV Cache Management

### PagedAttention Blocks

From [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180):

vLLM treats GPU memory like an OS's virtual memory paging system:

**Block Specification:**
- Default block size: 16 tokens
- Block memory: `2 * block_size * num_kv_heads * head_size * dtype_bytes`
- Example (Llama-2-7B, bf16): `2 * 16 * 32 * 128 * 2 = 262,144 bytes` per block per layer

**Free Block Queue:**
- Doubly-linked list of available blocks
- Typically hundreds of thousands of blocks depending on VRAM
- Managed by `KVCacheManager` coordinator

### Allocation Process

From the vLLM anatomy blog:

```python
def allocate_slots(request, num_new_tokens):
    """Allocate KV cache blocks for new tokens."""

    # 1. Compute blocks needed
    num_blocks_needed = ceil(num_new_tokens / block_size)

    # 2. Check availability
    if len(free_block_queue) < num_blocks_needed:
        return False  # Out of memory, trigger preemption

    # 3. Allocate blocks
    allocated_blocks = []
    for _ in range(num_blocks_needed):
        block_id = free_block_queue.pop_left()
        allocated_blocks.append(block_id)

    # 4. Store mapping
    req_to_blocks[request.id].extend(allocated_blocks)

    return True
```

**Block Table Structure:**

```python
# Example: Request with 35 tokens needs 3 blocks (16 tokens each)
req_to_blocks[request_id] = [block_42, block_17, block_391]

# Slot mapping for paged attention kernel:
# Token 0-15  → block_42, slots 0-15
# Token 16-31 → block_17, slots 0-15
# Token 32-34 → block_391, slots 0-2
```

### Prefix Caching Integration

From the anatomy blog, prefix caching reuses KV blocks across requests:

**Hash-based Block Identification:**
```python
def hash_request_tokens(tokens, block_size=16):
    """Compute hash for each complete block of tokens."""
    block_hashes = []
    prev_hash = None

    for i in range(0, len(tokens), block_size):
        block_tokens = tokens[i:i+block_size]
        if len(block_tokens) < block_size:
            break  # Skip incomplete block

        # Hash combines: prev_hash + current_tokens + metadata
        block_hash = compute_hash(prev_hash, block_tokens, lora_id, mm_hash)
        block_hashes.append(BlockHash(block_hash, block_tokens))
        prev_hash = block_hash

    return block_hashes
```

**Cache Lookup:**
```python
def get_computed_blocks(request):
    """Find cached blocks that can be reused."""
    block_hashes = hash_request_tokens(request.tokens)

    num_cached = 0
    for block_hash in block_hashes:
        if block_hash in cached_block_hash_to_block:
            num_cached += block_size
        else:
            break  # Must match prefix exactly

    return num_cached
```

**Memory Reclamation:**

Blocks become invalid when popped from `free_block_queue` for reallocation:
```python
def allocate_fresh_block():
    block_id = free_block_queue.pop_left()

    # Clear old prefix cache entry if exists
    if block_id in block_to_hash:
        old_hash = block_to_hash[block_id]
        del cached_block_hash_to_block[old_hash]
        del block_to_hash[block_id]

    return block_id
```

## Preemption Mechanisms

### When Preemption Occurs

From [Optimization and Tuning - vLLM](https://docs.vllm.ai/en/latest/configuration/optimization.html) (accessed 2025-02-02):

Preemption triggers when:
1. **Out of KV cache blocks**: `allocate_slots()` returns `False`
2. **Priority inversion**: Higher priority request arrives (priority scheduling only)
3. **Forced preemption**: Explicit request to clear resources

### Preemption Modes

vLLM V1 supports **recompute preemption** (V0 supported swap preemption):

**Recompute Preemption:**
- KV cache blocks are freed and returned to `free_block_queue`
- Request moved from `running` → `waiting`
- When rescheduled, prefill is recomputed from scratch
- More efficient for short sequences or when swapping is slow

**Swap Preemption (V0 only):**
- KV cache blocks swapped to CPU memory
- Request state preserved
- When rescheduled, blocks swapped back from CPU
- Useful for very long contexts, but adds I/O overhead

From the vLLM docs:

> "In vLLM V1, the default preemption mode is RECOMPUTE rather than SWAP. Preempted requests are recomputed when sufficient KV cache space becomes available again."

### Preemption Algorithm

From [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html):

```python
def attempt_preemption(request):
    """Try to free KV cache by preempting low-priority requests."""

    # V1: Recompute preemption for decode requests
    if request.is_decode():
        # Find lowest priority running requests
        candidates = sorted(
            self.running,
            key=lambda r: (r.priority, -r.arrival_time)  # Lower priority first, older first
        )

        blocks_needed = compute_blocks_needed(request)
        blocks_freed = 0

        for candidate in candidates:
            if candidate.priority >= request.priority:
                break  # Don't preempt same/higher priority

            # Free this request's KV blocks
            blocks = req_to_blocks[candidate.id]
            for block in blocks:
                free_block_queue.append(block)
            blocks_freed += len(blocks)

            # Move back to waiting
            candidate.status = WAITING
            self.running.remove(candidate)
            self.waiting.append(candidate)

            if blocks_freed >= blocks_needed:
                return True  # Freed enough

        return False  # Couldn't free enough
```

**Warning Messages:**

From the docs, when recompute preemption occurs:
```
WARNING 05-09 12:34:56 scheduler.py:123] Sequence group seq_group_id is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space.
```

## Scheduling Policies

### FCFS (First-Come-First-Serve)

From [How vLLM Prioritizes a Subset of Requests](https://hackernoon.com/how-vllm-prioritizes-a-subset-of-requests) (accessed 2025-02-02):

> "In vLLM, we adopt the first-come-first-serve (FCFS) scheduling policy for all requests, ensuring fairness and preventing starvation."

**FCFS Implementation:**
```python
class FCFS(Policy):
    def get_priority(self, now: float, request: Request) -> float:
        """Priority based solely on arrival time."""
        return request.arrival_time  # Earlier = higher priority (lower value)
```

**Waiting Queue:**
- Python `deque` (double-ended queue)
- Append new requests to right
- Pop from left for FIFO ordering

**Running Queue:**
- Python `list`
- Sorted by arrival time during preemption decisions

### Priority Scheduling

From [RFC: Priority Scheduling · Issue #6077](https://github.com/vllm-project/vllm/issues/6077) (accessed 2025-02-02):

**Motivation:**
1. **Batch + Interactive co-location**: Interactive requests preempt batch requests
2. **Fairness maintenance**: Dynamically adjust priorities to prevent starvation
3. **SLO optimization**: Prioritize requests close to deadline

**Implementation (PR #5958):**

```python
class PriorityPolicy(Policy):
    def get_priority(self, now: float, request: Request) -> float:
        """Dynamic priority based on request metadata."""
        return request.priority  # User-provided or dynamically computed
```

**Key Changes:**
1. Priority field added to `SequenceGroup`
2. Waiting queue uses heap instead of deque
3. Joint sorting of waiting + running queues
4. Forced preemption with KV cache preservation

**Priority APIs:**

```python
# Static priority at request time
llm.generate(prompts, sampling_params, priority=10)

# Dynamic priority adjustment
scheduler.update_priority(request_id, new_priority)
```

### Advanced Scheduling Strategies

From recent research papers and the priority RFC:

**1. Andes (Arxiv 2404.16283):**
- Adjust priorities based on Quality of Experience (QoE)
- Most starved requests get highest priority
- Maintains inter-token latency fairness

**2. VTC (Arxiv 2401.00588):**
- Virtual token credit system
- Requests accumulate credits over time
- Credits spent on GPU time

**3. Learning to Rank (vLLM-LTR):**

From [Efficient LLM Scheduling by Learning to Rank](https://hao-ai-lab.github.io/blogs/vllm-ltr/) (accessed 2025-02-02):

> "vLLM-LTR approximates Shortest Job First (SJF) scheduling using learning to rank. Instead of FCFS, it estimates completion time and prioritizes shorter jobs."

**Benefits:**
- Reduces average latency by 1.8-2.5x
- Approximates optimal SJF without perfect knowledge
- ML model predicts request duration from prompt features

## Chunked Prefill

### Motivation

From the vLLM anatomy blog:

Long prompts can monopolize GPU resources:
- Single 10,000 token prefill blocks all other requests
- Increases head-of-line blocking latency
- Reduces throughput for interactive workloads

**Solution:** Split long prefills into chunks.

### Implementation

```python
# Configuration
long_prefill_token_threshold = 4096  # Chunk size

# Scheduling logic
def schedule_prefill(request, token_budget):
    num_prompt_tokens = request.num_prompt_tokens
    num_computed = request.num_computed_tokens
    num_remaining = num_prompt_tokens - num_computed

    # Apply chunking
    if num_remaining > long_prefill_token_threshold:
        num_to_schedule = long_prefill_token_threshold
    else:
        num_to_schedule = min(num_remaining, token_budget)

    # Schedule partial prefill
    schedule_tokens(request, num_to_schedule)
    request.num_computed_tokens += num_to_schedule

    # Request stays in running until fully prefilled
    if request.num_computed_tokens < num_prompt_tokens:
        request.status = RUNNING  # Will continue next step
```

**Example Timeline:**

```
Prompt: 10,000 tokens, chunk_size: 4096

Step 0: Prefill tokens 0-4095     (4096 tokens)
Step 1: Prefill tokens 4096-8191  (4096 tokens)
Step 2: Prefill tokens 8192-9999  (1808 tokens) + sample first output token
Step 3: Decode 1 token
...
```

**Key Insight from blog:**
> "Chunked prefill enables better mixing of prefill and decode requests, reducing latency variance for interactive workloads."

## SLO-Aware Scheduling

### Service Level Objectives

From [SLO-Aware Scheduling for Large Language Model Inferences](https://arxiv.org/html/2504.14966v1) (accessed 2025-02-02):

**SLO Metrics:**
- **TTFT** (Time to First Token): Max 100ms for interactive
- **TPOT** (Time Per Output Token): Max 50ms for interactive
- **E2E Latency**: Max 500ms for batch workloads

**Scheduling Objective:**
Maximize throughput while meeting SLO constraints.

### Implementation Approaches

**1. Batch Size Limiting:**
```python
# vLLM+Priority approach
if urgent_requests:
    max_batch_size = min_batch_for_latency_target
else:
    max_batch_size = max_batch_for_throughput
```

**2. Request Ordering:**
```python
def priority_by_slo_slack(request, now):
    """Priority = remaining slack time."""
    deadline = request.arrival_time + request.slo_budget
    return deadline - now  # Tighter deadline = higher priority
```

**3. Preemption Policy:**

From the SLO paper:
> "When urgent requests arrive, preempt non-urgent requests during decoding to ensure SLO compliance."

### AdaServe (EuroSys 2026)

From [AdaServe: SLO-Driven Scheduling](https://www.cs.cmu.edu/~zhihaoj2/papers/AdaServe_EuroSys26.pdf) (accessed 2025-02-02):

**Multi-SLO Optimization:**
- Different request classes have different SLOs
- Adaptive batch sizing per SLO class
- Dynamic preemption based on SLO urgency

**Algorithm:**
```python
def adaptive_schedule(requests):
    # Group by SLO class
    urgent = [r for r in requests if r.slack_time < threshold]
    normal = [r for r in requests if r.slack_time >= threshold]

    # Prioritize urgent
    batch = []
    for r in sorted(urgent, key=lambda x: x.slack_time):
        if batch_tokens + r.tokens <= urgent_batch_limit:
            batch.append(r)

    # Fill remaining capacity with normal
    for r in normal:
        if batch_tokens + r.tokens <= max_batch_tokens:
            batch.append(r)

    return batch
```

## Request Metadata

### Core Metadata

From [Life of an inference request](https://www.ubicloud.com/blog/life-of-an-inference-request-vllm-v1):

```python
class Request:
    request_id: str              # Unique identifier
    arrival_time: float          # Unix timestamp
    prompt_token_ids: List[int]  # Tokenized prompt
    sampling_params: SamplingParams
    priority: float              # For priority scheduling

    # Runtime state
    status: RequestStatus        # WAITING, RUNNING, FINISHED
    num_computed_tokens: int     # For chunked prefill
    output_token_ids: List[int]  # Generated tokens

    # KV cache
    kv_block_ids: List[int]      # Allocated KV blocks
    block_hashes: List[BlockHash]  # For prefix caching
```

### Sampling Parameters

```python
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 16
    stop_token_ids: List[int] = []
    stop_strings: List[str] = []
    ignore_eos: bool = False

    # Guided decoding
    guided_decoding: Optional[GuidedDecodingParams] = None

    # Speculative decoding
    num_speculative_tokens: int = 0
```

### Stop Conditions

From the anatomy blog:

**Request finishes when:**
1. `len(output_tokens) >= max_tokens` (length limit)
2. Sampled token is EOS token (unless `ignore_eos=True`)
3. Sampled token in `stop_token_ids`
4. Generated text contains any `stop_string`

**Stop String Handling:**
```python
def check_stop_strings(output_text, stop_strings):
    """Truncate at first stop string occurrence."""
    for stop_str in stop_strings:
        if stop_str in output_text:
            idx = output_text.index(stop_str)
            return output_text[:idx], True  # Truncated, stop=True
    return output_text, False  # Not stopped
```

## Performance Metrics

### Scheduler Metrics

From [Metrics - vLLM](https://docs.vllm.ai/en/v0.8.5/design/v1/metrics.html) (accessed 2025-02-02):

**Server-Level Metrics:**
- `vllm:num_requests_running`: Active requests
- `vllm:num_requests_waiting`: Queued requests
- `vllm:gpu_cache_usage_perc`: KV cache utilization
- `vllm:num_preemptions_total`: Cumulative preemptions

**Request-Level Metrics:**
- `vllm:request_ttft_seconds`: Time to first token
- `vllm:request_tpot_seconds`: Time per output token
- `vllm:request_e2e_seconds`: End-to-end latency
- `vllm:request_prompt_tokens`: Input length
- `vllm:request_generation_tokens`: Output length

**Goodput:**
> "Throughput that meets SLOs. Only tokens from requests meeting TTFT/TPOT/E2E targets are counted."

### Roofline Performance Model

From the anatomy blog:

**Prefill (Compute-Bound):**
- Dominated by matrix multiplications
- Performance: `tokens/sec ≈ TFLOPS / (2 * params)`
- Scales with batch size until GPU saturation

**Decode (Memory-Bound):**
- Dominated by weight loading from HBM
- Performance: `tokens/sec ≈ bandwidth / (bytes_per_token)`
- Benefits from batch size below saturation point `B_sat`

**Saturation Batch Size:**
```
B_sat ≈ HBM_bandwidth / compute_intensity
```

Below `B_sat`: Latency roughly constant (bandwidth-bound)
Above `B_sat`: Latency grows linearly (compute-bound)

## Code Implementation Examples

### V1 Scheduler Core

From [vllm/v1/core/sched/scheduler.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py):

```python
class Scheduler:
    def __init__(self, config):
        self.policy = config.scheduling_policy  # "fcfs" or "priority"
        self.max_num_batched_tokens = config.max_num_batched_tokens

        self.waiting = deque() if self.policy == "fcfs" else []  # heap
        self.running = []

        self.kv_cache_manager = KVCacheManager(config)

    def schedule(self) -> ScheduleOutput:
        """Schedule requests for next engine step."""

        # Add new requests from input queue
        while not self.input_queue.empty():
            req = self.input_queue.get()
            self._add_request(req)

        scheduled = []
        token_budget = self.max_num_batched_tokens

        # Priority 1: Decode requests (running)
        for req in list(self.running):
            num_tokens = self._get_num_new_tokens(req)

            if not self.kv_cache_manager.allocate_slots(req, num_tokens):
                # Out of memory, try preemption
                if self._attempt_preemption(req):
                    if not self.kv_cache_manager.allocate_slots(req, num_tokens):
                        continue  # Still can't allocate, skip
                else:
                    continue  # Preemption failed, skip

            scheduled.append(req)
            token_budget -= num_tokens

        # Priority 2: Prefill requests (waiting)
        while self.waiting and token_budget > 0:
            req = self._pop_waiting()

            num_cached = self.kv_cache_manager.get_computed_blocks(req)
            num_new = req.num_prompt_tokens - num_cached

            # Chunked prefill
            if num_new > self.long_prefill_token_threshold:
                num_new = self.long_prefill_token_threshold

            num_new = min(num_new, token_budget)

            if not self.kv_cache_manager.allocate_slots(req, num_new):
                break  # Out of memory, stop scheduling

            scheduled.append(req)
            token_budget -= num_new

            # Move to running
            req.status = RequestStatus.RUNNING
            self.running.append(req)

        return ScheduleOutput(scheduled)
```

### KV Cache Manager

```python
class KVCacheManager:
    def __init__(self, config):
        self.block_size = config.block_size  # 16
        self.num_blocks = config.num_gpu_blocks

        # Free block pool (doubly-linked list)
        self.free_block_queue = deque(range(self.num_blocks))

        # Block mappings
        self.req_to_blocks = {}  # request_id -> [block_ids]
        self.cached_block_hash_to_block = {}  # hash -> block_id
        self.block_to_hash = {}  # block_id -> hash

    def allocate_slots(self, request, num_tokens):
        """Allocate KV cache blocks for tokens."""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_block_queue) < num_blocks_needed:
            return False  # Out of memory

        allocated = []
        for _ in range(num_blocks_needed):
            block_id = self.free_block_queue.popleft()

            # Clear old prefix cache entry
            if block_id in self.block_to_hash:
                old_hash = self.block_to_hash[block_id]
                del self.cached_block_hash_to_block[old_hash]
                del self.block_to_hash[block_id]

            allocated.append(block_id)

        if request.id not in self.req_to_blocks:
            self.req_to_blocks[request.id] = []
        self.req_to_blocks[request.id].extend(allocated)

        return True

    def free(self, request):
        """Free all KV blocks for request."""
        if request.id not in self.req_to_blocks:
            return

        blocks = self.req_to_blocks[request.id]
        for block_id in blocks:
            self.free_block_queue.append(block_id)

        del self.req_to_blocks[request.id]

    def get_computed_blocks(self, request):
        """Get number of cached tokens from prefix cache."""
        block_hashes = hash_request_tokens(request.tokens, self.block_size)

        num_cached = 0
        for block_hash in block_hashes:
            if block_hash in self.cached_block_hash_to_block:
                num_cached += self.block_size
            else:
                break

        return num_cached
```

## Best Practices

### Scheduler Configuration

**For Maximum Throughput:**
```python
# Large batch size, aggressive chunking
max_num_batched_tokens = 16384
long_prefill_token_threshold = 2048
enable_prefix_caching = True
```

**For Low Latency:**
```python
# Smaller batch size, minimal chunking
max_num_batched_tokens = 2048
long_prefill_token_threshold = 8192
scheduling_policy = "priority"
```

**For Mixed Workloads:**
```python
# Balanced configuration
max_num_batched_tokens = 8192
long_prefill_token_threshold = 4096
enable_prefix_caching = True
scheduling_policy = "priority"

# Use priorities:
# - Interactive: priority=10
# - Batch: priority=1
```

### Monitoring and Tuning

**Watch for Preemption:**
```bash
# High preemption rate indicates insufficient KV cache
grep "preempted by" vllm.log | wc -l
```

**Monitor Queue Depths:**
```python
# Prometheus metrics
vllm:num_requests_waiting > 100  # Backlog building up
vllm:gpu_cache_usage_perc > 90   # Close to OOM
```

**Adjust Token Budget:**
```python
# If TTFT too high, reduce batch size
max_num_batched_tokens = max_num_batched_tokens * 0.8

# If throughput too low, increase batch size
max_num_batched_tokens = max_num_batched_tokens * 1.2
```

### Common Pitfalls

**1. Token Budget Too Large:**
- Symptoms: High TTFT, preemption storms
- Solution: Reduce `max_num_batched_tokens`

**2. Chunked Prefill Too Aggressive:**
- Symptoms: Long prompts take many steps
- Solution: Increase `long_prefill_token_threshold`

**3. FCFS with Mixed Workloads:**
- Symptoms: Batch jobs block interactive requests
- Solution: Enable priority scheduling

**4. Insufficient KV Cache:**
- Symptoms: Frequent preemptions, low throughput
- Solution: Increase `gpu_memory_utilization` or reduce batch size

## Sources

### Primary Documentation
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) - vLLM Blog (accessed 2025-02-02)
- [Life of an inference request (vLLM V1): How LLMs are served efficiently at scale](https://www.ubicloud.com/blog/life-of-an-inference-request-vllm-v1) - Ubicloud Blog (accessed 2025-02-02)
- [Optimization and Tuning - vLLM](https://docs.vllm.ai/en/latest/configuration/optimization.html) - vLLM Documentation (accessed 2025-02-02)
- [Metrics - vLLM](https://docs.vllm.ai/en/v0.8.5/design/v1/metrics.html) - vLLM Documentation (accessed 2025-02-02)

### Research Papers
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - Kwon et al., 2023
- [SLO-Aware Scheduling for Large Language Model Inferences](https://arxiv.org/html/2504.14966v1) - arXiv (accessed 2025-02-02)
- [AdaServe: SLO-Driven Scheduling for LLM Serving](https://www.cs.cmu.edu/~zhihaoj2/papers/AdaServe_EuroSys26.pdf) - CMU (accessed 2025-02-02)

### GitHub Resources
- [RFC: Priority Scheduling · Issue #6077](https://github.com/vllm-project/vllm/issues/6077) - vLLM GitHub (accessed 2025-02-02)
- [vllm/v1/core/sched/scheduler.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) - vLLM Source Code

### Blog Posts and Tutorials
- [How vLLM Prioritizes a Subset of Requests](https://hackernoon.com/how-vllm-prioritizes-a-subset-of-requests) - Hackernoon (accessed 2025-02-02)
- [Efficient LLM Scheduling by Learning to Rank](https://hao-ai-lab.github.io/blogs/vllm-ltr/) - Hao AI Lab (accessed 2025-02-02)
- [Under the Hood of vLLM: Memory, Scheduling & Batching Strategies](https://www.javacodegeeks.com/2025/10/under-the-hood-of-vllm-memory-scheduling-batching-strategies.html) - Java Code Geeks (accessed 2025-02-02)

### Additional References
- [Continuous Batching and Selective Batching, Orca](https://medium.com/byte-sized-ai/inference-optimizations-1-continuous-batching-03408c673098) - Medium (accessed 2025-02-02)
- [LLM Inference Optimisation - Continuous Batching and vLLM](https://samuel-jenkins-ml.com/llm-inference-optimisation-continuous-batching-and-v-llm/) - Samuel Jenkins ML (accessed 2025-02-02)
