# vLLM Request Lifecycle Internals

## Overview

This document provides a comprehensive deep dive into vLLM's request lifecycle, covering the complete journey of an inference request from submission to completion. Understanding these internals is critical for debugging, optimization, and contributing to vLLM.

**Target audience**: Engineers working on vLLM internals, performance optimization, or building custom inference systems.

**Analysis based on**: vLLM V1 engine architecture (commit 42172ad, August 2025)

---

## Request States and State Transitions

### Core Request States

vLLM manages requests through a well-defined state machine. Each request transitions through these states during its lifecycle:

**Primary States:**

1. **WAITING** - Request is in the scheduler's waiting queue, not yet allocated KV cache blocks
2. **WAITING_FOR_FSM** - Request with guided decoding is waiting for grammar compilation (FSM = Finite State Machine)
3. **RUNNING** - Request has allocated KV cache and is actively being processed
4. **PREEMPTED** - Request was evicted from running queue to free KV cache for higher priority requests
5. **FINISHED** - Request completed successfully (hit stop condition)
6. **ABORTED** - Request was cancelled or encountered an error

### State Transition Diagram

```
                    ┌─────────────┐
                    │   WAITING   │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │ (guided    │ (normal    │
              │  decode)   │  request)  │
              ▼            ▼            │
    ┌──────────────────┐  │            │
    │ WAITING_FOR_FSM  │  │            │
    └────────┬─────────┘  │            │
             │            │            │
             │ (FSM ready)│            │
             └────────────┼────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │   RUNNING    │◄─────┐
                   └──────┬───────┘      │
                          │              │
              ┌───────────┼──────────┐   │
              │           │          │   │
              ▼           ▼          ▼   │
       ┌──────────┐  ┌────────┐  ┌──────────┐
       │ FINISHED │  │ ABORTED│  │PREEMPTED │
       └──────────┘  └────────┘  └────┬─────┘
                                      │
                                      └──────┘
                                  (back to RUNNING
                                   when resources
                                   available)
```

### State Transition Triggers

From [vLLM Blog - Anatomy](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) (accessed 2025-11-02):

**WAITING → RUNNING**:
- Scheduler calls `allocate_slots()` successfully
- KV cache blocks are available
- Token budget allows processing

**RUNNING → PREEMPTED**:
- KV cache blocks needed for higher priority requests
- Recompute preemption policy triggers (V1 uses recompute, V0 had swap)
- Request's blocks are freed back to `free_block_queue`

**RUNNING → FINISHED**:
Stop conditions met (any of):
- Exceeds `max_model_length` or request's own `max_tokens`
- Sampled token is EOS token (unless `ignore_eos=True`)
- Sampled token matches any `stop_token_ids`
- Stop strings appear in output (truncated before stop string)

**RUNNING → ABORTED**:
- Client cancels request
- Error during processing
- Timeout exceeded

**WAITING_FOR_FSM → WAITING**:
- Async grammar compilation completes
- Request moved from `skipped_waiting_requests` to `waiting` queue
- Status updated to enable scheduling

---

## Input Processing Pipeline

### Step 1: Request Submission

From [vLLM Docs - Input Processing Pipeline](https://docs.vllm.ai/en/v0.5.4/dev/input_processing/input_processing_pipeline.html) (accessed 2025-11-02):

```python
# User code
from vllm import LLM, SamplingParams

llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)
```

**Internal flow**:

1. **LLM.generate()** receives raw prompts
2. For each prompt, **create unique request ID** and capture arrival timestamp
3. Call **input preprocessor** to handle tokenization

### Step 2: Tokenization and Validation

```python
# Internal: vllm/entrypoints/llm.py
def _validate_and_add_requests(self, prompts, params):
    for i, prompt in enumerate(prompts):
        request_id = f"request-{uuid.uuid4()}"
        arrival_time = time.time()

        # Tokenize if needed
        processed_input = self.input_processor.process(prompt)
        # Returns: {"prompt": str, "prompt_token_ids": List[int], "type": "text"}
```

**Processing steps** (from vLLM docs):
1. **Input data passed to LLMEngine** (or AsyncLLMEngine)
2. **Tokenize data if necessary** - raw text → token IDs
3. **Process inputs using INPUT_REGISTRY.process_input**:
   - For text-only: straightforward tokenization
   - For multimodal: add placeholder tokens to reserve KV cache for embeddings
4. **Create EngineCoreRequest** with:
   - `request_id` (unique identifier)
   - `prompt` and `prompt_token_ids`
   - `sampling_params` (temperature, top_p, max_tokens, etc.)
   - `arrival_time` (for metrics and scheduling)
   - `priority` (if using priority scheduling policy)

### Step 3: Request Wrapping and Queue Insertion

From [vLLM Blog - Anatomy](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) (accessed 2025-11-02):

```python
# Internal: engine core wraps in Request object
request = Request(
    request_id=request_id,
    prompt=prompt,
    prompt_token_ids=prompt_token_ids,
    sampling_params=sampling_params,
    arrival_time=arrival_time
)
request.status = RequestStatus.WAITING

# Add to scheduler's waiting queue
if scheduling_policy == "fcfs":
    self.waiting.append(request)  # FCFS: append to end
elif scheduling_policy == "priority":
    heapq.heappush(self.waiting, (request.priority, request))  # Priority: heap
```

**Queue policies**:
- **FCFS (First Come First Served)**: Simple list, append/pop
- **Priority**: Min-heap ordered by priority value (lower = higher priority)

---

## Execution Pipeline

### Scheduler: The Heart of Request Management

The scheduler is called on every engine step and decides which requests to process.

#### Scheduling Algorithm

From [vLLM Blog - Anatomy](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) (accessed 2025-11-02):

**Phase 1: Process RUNNING requests (decode priority)**

```python
# Pseudo-code from scheduler
def schedule_running_requests(self):
    for request in self.running:
        # 1. Compute tokens to generate
        num_new_tokens = self._compute_num_new_tokens(request)
        # (Usually 1, but can be >1 with speculative decoding)

        # 2. Allocate KV cache slots
        success = self.kv_cache_manager.allocate_slots(
            request_id=request.request_id,
            num_tokens=num_new_tokens
        )

        if not success:
            # Attempt preemption of lower priority requests
            self._attempt_preemption(request)

        # 3. Update token budget
        self.token_budget -= num_new_tokens
```

**Phase 2: Process WAITING requests (prefill)**

```python
def schedule_waiting_requests(self):
    while self.waiting and self.token_budget > 0:
        request = self._pop_waiting_request()  # FCFS or priority

        # 1. Check prefix cache (if enabled)
        num_computed_blocks = self.kv_cache_manager.get_computed_blocks(request)

        # 2. Allocate KV cache for new tokens
        num_new_tokens = len(request.prompt_token_ids) - (num_computed_blocks * block_size)
        success = self.kv_cache_manager.allocate_slots(
            request_id=request.request_id,
            num_tokens=num_new_tokens
        )

        if not success:
            # Not enough blocks, skip for this step
            self.skipped_waiting_requests.append(request)
            continue

        # 3. Move to running
        request.status = RequestStatus.RUNNING
        self.running.append(request)

        # 4. Update token budget
        self.token_budget -= num_new_tokens
```

#### KV Cache Allocation Details

From [vLLM Blog - Anatomy](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) (accessed 2025-11-02):

**allocate_slots() implementation**:

```python
def allocate_slots(self, request_id, num_tokens):
    # 1. Compute number of blocks needed
    n = ceil(num_tokens / self.block_size)  # block_size typically 16

    # 2. Check availability
    if len(self.free_block_queue) < n:
        return False  # Not enough blocks

    # 3. Allocate blocks
    allocated_blocks = []
    for _ in range(n):
        block = self.free_block_queue.popleft()  # FIFO from deque
        allocated_blocks.append(block)

    # 4. Store mapping
    self.req_to_blocks[request_id] = allocated_blocks

    return True
```

**Block structure**:
- Each block stores 16 tokens by default (configurable)
- Block size (bytes) = `2 * block_size * num_kv_heads * head_size * dtype_bytes`
  - Example: `2 * 16 * 32 * 128 * 2 = 262,144 bytes` per block for bf16

### Forward Pass Execution

From [vLLM Blog - Anatomy](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) (accessed 2025-11-02):

**ModelExecutor.execute_model() flow**:

```python
def execute_model(self, scheduler_output):
    # Delegate to Worker (single GPU) or MultiProcExecutor (multi-GPU)
    return self.worker.execute_model(scheduler_output)
```

**Worker.execute_model() → ModelRunner flow**:

**Step 1: Update States**
```python
# Prune finished requests from input_batch
self.input_batch.remove_finished(finished_request_ids)

# Update KV cache block metadata
for request_id, blocks in scheduler_output.allocated_blocks.items():
    self.input_batch.update_kv_blocks(request_id, blocks)
```

**Step 2: Prepare Inputs**
```python
# Copy CPU → GPU buffers
input_ids_tensor = torch.tensor(scheduler_output.token_ids).cuda()

# Compute positions (for RoPE, etc.)
positions = self._compute_positions(scheduler_output)

# Build slot_mapping for paged attention
slot_mapping = self._build_slot_mapping(scheduler_output)
# Maps each token position → KV cache memory location

# Construct attention metadata
attn_metadata = self._build_attention_metadata(
    block_tables=scheduler_output.block_tables,
    context_lens=scheduler_output.context_lens
)
```

**Continuous batching example** (from vLLM blog):

```
Request A: [101, 202, 303]  (3 tokens, decode)
Request B: [404, 505, ...]  (8 tokens, chunked prefill)
Request C: [606, 707]      (2 tokens, decode)

Flattened sequence:
token_ids = [101, 202, 303, 404, 505, ..., 606, 707]
positions = [  5,   6,   7,   0,   1, ...,  10,  11]
slot_mapping = [block_A[5], block_A[6], block_A[7],
                block_B[0], block_B[1], ...,
                block_C[10], block_C[11]]
```

**Step 3: Forward Pass**
```python
if self.enforce_eager:
    # Standard PyTorch eager execution
    hidden_states = self.model(
        input_ids=input_ids_tensor,
        positions=positions,
        kv_caches=self.kv_cache,
        attn_metadata=attn_metadata
    )
else:
    # Replay captured CUDA graph (faster)
    self._replay_cuda_graph(input_ids_tensor, positions, attn_metadata)
    hidden_states = self.graph_output_buffer
```

**Step 4: Gather Last-Token States**
```python
# Extract hidden states for final position of each sequence
last_token_indices = self._compute_last_token_indices(scheduler_output)
logits = self.lm_head(hidden_states[last_token_indices])  # Shape: [batch_size, vocab_size]
```

**Step 5: Sample**
```python
sampled_token_ids = self.sampler.sample(
    logits=logits,
    sampling_params=scheduler_output.sampling_params
)
# Applies temperature, top_p, top_k, etc.
```

### Postprocessing and State Updates

From [vLLM Blog - Anatomy](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) (accessed 2025-11-02):

```python
def postprocess_step(self, sampled_token_ids, scheduler_output):
    for i, request_id in enumerate(scheduler_output.request_ids):
        request = self.get_request(request_id)

        # 1. Append sampled token
        token_id = sampled_token_ids[i]
        request.output_token_ids.append(token_id)

        # 2. Detokenize (incremental)
        new_text = self.tokenizer.decode([token_id])
        request.output_text += new_text

        # 3. Check stop conditions
        if self._check_stop_conditions(request, token_id):
            request.status = RequestStatus.FINISHED

            # 4. Cleanup: free KV cache blocks
            blocks = self.kv_cache_manager.get_blocks(request_id)
            self.kv_cache_manager.free(blocks)
            # Returns blocks to free_block_queue

            # 5. Return output early (streaming)
            self._send_output(request)
```

**Stop condition checking**:
```python
def _check_stop_conditions(self, request, token_id):
    # Length limit
    if len(request.output_token_ids) >= request.sampling_params.max_tokens:
        return True

    # EOS token
    if token_id == self.tokenizer.eos_token_id:
        if not request.sampling_params.ignore_eos:
            return True

    # Stop token IDs
    if token_id in request.sampling_params.stop_token_ids:
        return True

    # Stop strings (check decoded output)
    for stop_str in request.sampling_params.stop:
        if stop_str in request.output_text:
            # Truncate output at stop string
            request.output_text = request.output_text.split(stop_str)[0]
            return True

    return False
```

---

## Output Streaming

### Synchronous Engine (Offline Inference)

From [vLLM Blog - Anatomy](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) (accessed 2025-11-02):

**Batch completion**:
```python
def generate(self, prompts, sampling_params):
    # Add all requests to waiting queue
    self._add_requests(prompts, sampling_params)

    outputs = []
    while self.has_unfinished_requests():
        step_outputs = self.step()  # Execute one engine step

        # Collect finished requests
        for output in step_outputs:
            if output.finished:
                outputs.append(output)

    return outputs
```

**No streaming in synchronous mode** - outputs returned only when fully complete.

### Asynchronous Engine (Online Serving)

From [vLLM Blog - Anatomy](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) and [Zerohertz Blog - vLLM Deep Dive](https://zerohertz.github.io/vllm-openai-2/) (accessed 2025-11-02):

**Streaming architecture**:

```
AsyncLLM
  ├─ DPLBAsyncMPClient
  │    ├─ outputs_queue (asyncio.Queue)
  │    ├─ process_outputs_socket (asyncio task)
  │    └─ output_handler (asyncio task)
  │
  └─ EngineCore (per replica)
       ├─ input_thread (blocks on socket)
       ├─ main_thread (engine loop)
       └─ output_thread (sends to socket)
```

**Output flow**:

1. **Main thread** completes engine step, produces outputs
2. **Output queue** receives: `output_queue.put_nowait(request_output)`
3. **Output thread** wakes: `output = output_queue.get()`, sends via socket
4. **process_outputs_socket** (asyncio) receives from socket → `outputs_queue.put(output)`
5. **output_handler** (asyncio) reads `outputs_queue.get()` → sends to client

**Incremental streaming**:
```python
# FastAPI endpoint
@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    if request.stream:
        return StreamingResponse(
            generate_stream(request),
            media_type="text/event-stream"
        )

async def generate_stream(request):
    async for output in engine.generate_stream(request):
        # output contains incremental tokens
        yield f"data: {json.dumps(output.to_dict())}\n\n"

    yield "data: [DONE]\n\n"
```

**Per-step outputs**:
```python
# Each engine step can produce multiple outputs
class RequestOutput:
    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    outputs: List[CompletionOutput]  # Can have multiple for beam search
    finished: bool

class CompletionOutput:
    index: int
    text: str  # Incremental text since last output
    token_ids: List[int]  # All tokens generated so far
    cumulative_logprob: float
    logprobs: Optional[List[Dict[int, float]]]
    finish_reason: Optional[str]  # "stop", "length", None
```

---

## Error Handling

### Error Types and Recovery

From [vLLM Docs - Troubleshooting](https://docs.vllm.ai/en/stable/usage/troubleshooting.html) and [GitHub Issues](https://github.com/vllm-project/vllm/issues) (accessed 2025-11-02):

**1. CUDA Out-of-Memory (OOM)**

```python
# During KV cache initialization
try:
    num_gpu_blocks = self._profile_num_available_blocks()
except torch.cuda.OutOfMemoryError:
    logger.error("Not enough GPU memory for KV cache")
    # Reduce gpu_memory_utilization or use smaller batch
```

**Recovery**: Automatic reduction not available, requires config adjustment.

**2. Invalid KV Cache Blocks**

From vLLM scheduler code:

```python
def _identify_invalid_blocks(self, requests):
    """Scan requests for invalid cached blocks."""
    for request in requests:
        if hasattr(request, 'cached_block_hashes'):
            for block_hash in request.cached_block_hashes:
                block = self.cached_block_hash_to_block.get(block_hash)
                if block and block.is_invalidated:
                    # Mark block for recompute
                    request.num_computed_blocks = 0
```

**Recovery**: Automatic recomputation of invalidated prefix cache blocks.

**3. Streaming Output Errors**

From [GitHub Issue #22341](https://github.com/vllm-project/vllm/issues/22341) (accessed 2025-11-02):

```python
# Error: "Caught handled exception, but response already started"
# Cause: Error after streaming began, can't send error response

# Mitigation in vLLM:
try:
    async for output in self.generate_stream(request):
        yield output
except Exception as e:
    if not response_started:
        # Can still send error response
        raise HTTPException(status_code=500, detail=str(e))
    else:
        # Log error, send DONE with error flag
        logger.error(f"Error during streaming: {e}")
        yield {"error": str(e), "done": True}
```

**4. Request Abortion**

```python
def abort_request(self, request_id: str):
    """Client-initiated cancellation."""
    request = self.get_request(request_id)
    if request:
        request.status = RequestStatus.ABORTED

        # Free resources
        blocks = self.kv_cache_manager.get_blocks(request_id)
        self.kv_cache_manager.free(blocks)

        # Remove from queues
        self._remove_from_queues(request_id)
```

**5. Engine Loop Crash**

From [GitHub Discussion #9418](https://github.com/vllm-project/vllm/discussions/9418) (accessed 2025-11-02):

```python
# Error: "RuntimeError: Engine loop has died"
# Cause: Uncaught exception in main engine thread

# Detection:
def check_engine_health(self):
    if not self.engine_thread.is_alive():
        raise RuntimeError("Engine loop has died")
```

**Recovery**: Requires engine restart, no automatic recovery.

### Preemption and Recompute

From [vLLM Blog - Anatomy](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) (accessed 2025-11-02):

**Recompute preemption (V1)**:

```python
def _attempt_preemption(self, high_priority_request):
    # Find low priority running requests
    preemptable = [
        req for req in self.running
        if req.priority > high_priority_request.priority
    ]

    if not preemptable:
        return False

    # Sort by priority (highest first = lowest priority value)
    preemptable.sort(key=lambda r: r.priority, reverse=True)

    # Preempt until enough blocks freed
    blocks_needed = self._compute_blocks_needed(high_priority_request)
    blocks_freed = 0

    for req in preemptable:
        req.status = RequestStatus.PREEMPTED
        blocks = self.kv_cache_manager.get_blocks(req.request_id)
        self.kv_cache_manager.free(blocks)
        blocks_freed += len(blocks)

        self.running.remove(req)
        self.waiting.append(req)  # Back to waiting queue

        if blocks_freed >= blocks_needed:
            break

    return blocks_freed >= blocks_needed
```

**Note**: V0 engine supported swap preemption (swap KV cache to CPU), V1 uses recompute only.

---

## Advanced Features Impact on Lifecycle

### Chunked Prefill

From [vLLM Blog - Anatomy](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) (accessed 2025-11-02):

**Effect on request lifecycle**:

```python
# Request with 100 tokens, chunk_size = 32
Step 1: Process tokens [0:32]   → RUNNING (chunked prefill)
Step 2: Process tokens [32:64]  → RUNNING (chunked prefill)
Step 3: Process tokens [64:96]  → RUNNING (chunked prefill)
Step 4: Process tokens [96:100] → RUNNING (chunked prefill)
Step 5: Sample token 100        → RUNNING (decode)
Step 6: Sample token 101        → RUNNING (decode)
...
```

**Scheduler handling**:
```python
if len(request.prompt_token_ids) > self.long_prefill_token_threshold:
    # Cap tokens per step
    num_tokens_this_step = min(
        self.long_prefill_token_threshold,
        len(request.prompt_token_ids) - request.num_processed_tokens
    )
    request.num_tokens_to_process = num_tokens_this_step
else:
    # Process all remaining tokens
    request.num_tokens_to_process = len(request.prompt_token_ids)
```

### Prefix Caching

From [vLLM Blog - Anatomy](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) (accessed 2025-11-02):

**Effect on request lifecycle**:

```python
# First request: "system_prompt + user_query_1"
Step 1: Hash blocks for full prompt → no cache hits
Step 2: Allocate KV blocks for all tokens
Step 3: Forward pass computes KVs, stored in blocks
Step 4: Store block_hash → block mapping in cache

# Second request: "system_prompt + user_query_2"
Step 1: Hash blocks for full prompt → cache hits for system_prompt blocks!
Step 2: Allocate KV blocks only for user_query_2 tokens
Step 3: Forward pass skips system_prompt tokens (KVs already computed)
Step 4: Reuse existing blocks, compute only new ones
```

**Lifecycle modification**:
- `get_computed_blocks()` returns > 0 for cached prefix
- `allocate_slots()` allocates fewer blocks
- Forward pass processing reduces to non-cached tokens only
- Significant latency reduction for repeated prefixes

### Speculative Decoding

From [vLLM Blog - Anatomy](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) (accessed 2025-11-02):

**Effect on request lifecycle**:

```python
# After first decode step
Step 1: Sample token from large model → token_1
Step 2: Draft model proposes k=3 tokens → [draft_1, draft_2, draft_3]
Step 3: Store in request.spec_token_ids

# Next engine step
Step 4: Forward pass with [token_1, draft_1, draft_2, draft_3]
Step 5: Large model produces logits for all 4 positions
Step 6: Rejection sampler accepts/rejects draft tokens
Step 7: If all accepted: 4 tokens in one step (3x speedup!)
```

**Scheduler changes**:
```python
# Allocate KV slots for speculative tokens
num_new_tokens = 1 + len(request.spec_token_ids)  # Original + draft tokens
```

**Lifecycle impact**: Multiple tokens per step, but only one forward pass of large model.

---

## Performance Metrics

### Request-Level Metrics

From [Red Hat vLLM Documentation](https://docs.redhat.com/fr/documentation/red_hat_ai_inference_server/3.0/html-single/vllm_server_arguments/index) (accessed 2025-11-02):

**Time-based metrics**:

- `vllm:request_waiting_time_seconds` - Histogram of time in WAITING state
- `vllm:request_inference_time_seconds` - Histogram of time in RUNNING state
- `vllm:e2e_request_latency_seconds` - End-to-end latency (WAITING + RUNNING)
- `vllm:time_to_first_token_seconds` - TTFT latency
- `vllm:time_per_output_token_seconds` - TPOT latency (ITL average)

**Count metrics**:

- `vllm:num_requests_waiting` - Current waiting queue size
- `vllm:num_requests_running` - Current running queue size
- `vllm:num_requests_preempted` - Total preempted requests (cumulative)
- `vllm:request_success_total` - Finished successfully
- `vllm:request_abort_total` - Aborted/cancelled

### System-Level Metrics

- `vllm:gpu_cache_usage_perc` - KV cache utilization (0-1)
- `vllm:num_gpu_blocks_total` - Total KV cache blocks
- `vllm:num_gpu_blocks_free` - Available KV cache blocks
- `vllm:num_preemptions_total` - Cumulative preemption count
- `vllm:prompt_tokens_total` - Total input tokens processed
- `vllm:generation_tokens_total` - Total output tokens generated

---

## Code Examples

### Example 1: Tracing a Single Request

```python
import logging
from vllm import LLM, SamplingParams

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

prompts = ["Explain quantum computing"]
sampling_params = SamplingParams(temperature=0.8, max_tokens=50)

# Trace request lifecycle
outputs = llm.generate(prompts, sampling_params)

# Expected log output:
# [DEBUG] Request request-abc123: WAITING → added to waiting queue
# [DEBUG] Scheduler step 1: Processing waiting queue
# [DEBUG] Request request-abc123: Allocated 3 KV blocks
# [DEBUG] Request request-abc123: WAITING → RUNNING
# [DEBUG] Forward pass: 1 request, 45 tokens (prefill)
# [DEBUG] Sampled token: 1234
# [DEBUG] Request request-abc123: Output token 1/50
# [DEBUG] Forward pass: 1 request, 1 token (decode)
# ...
# [DEBUG] Request request-abc123: Stop condition met (max_tokens)
# [DEBUG] Request request-abc123: RUNNING → FINISHED
# [DEBUG] Freed 3 KV blocks for request-abc123
```

### Example 2: Monitoring Request States

```python
import time
import asyncio
from vllm import AsyncLLM, SamplingParams

async def monitor_requests():
    llm = AsyncLLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Submit multiple requests
    prompts = [f"Question {i}" for i in range(10)]
    sampling_params = SamplingParams(temperature=0.8, max_tokens=100)

    # Monitor while generating
    async def monitor_loop():
        while True:
            stats = await llm.get_stats()
            print(f"Waiting: {stats['num_waiting']}, "
                  f"Running: {stats['num_running']}, "
                  f"Finished: {stats['num_finished']}")
            await asyncio.sleep(0.1)

    # Run concurrently
    monitor_task = asyncio.create_task(monitor_loop())
    outputs = await llm.generate(prompts, sampling_params)
    monitor_task.cancel()

    return outputs

# Run
asyncio.run(monitor_requests())
```

### Example 3: Custom Error Handling

```python
from vllm import LLM, SamplingParams
from vllm.engine.llm_engine import LLMEngine

class MonitoredLLM(LLM):
    def generate(self, prompts, sampling_params):
        try:
            return super().generate(prompts, sampling_params)
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM Error: {e}")
            print("Try reducing batch size or gpu_memory_utilization")
            raise
        except RuntimeError as e:
            if "Engine loop has died" in str(e):
                print("Engine crashed - requires restart")
                self._restart_engine()
            raise

    def _restart_engine(self):
        # Recreate engine (all state lost)
        self.llm_engine = LLMEngine.from_engine_args(self.engine_args)

# Usage
llm = MonitoredLLM(model="meta-llama/Llama-2-7b-hf")
outputs = llm.generate(prompts, sampling_params)
```

---

## Debugging Tips

### 1. Enable Verbose Logging

```bash
export VLLM_LOGGING_LEVEL=DEBUG
python your_script.py
```

### 2. Trace Specific Request

```python
# Patch engine to log specific request
import logging
logger = logging.getLogger("vllm.engine")

original_step = engine.step

def traced_step():
    logger.info(f"Step start - waiting: {len(engine.scheduler.waiting)}, "
                f"running: {len(engine.scheduler.running)}")
    result = original_step()
    logger.info(f"Step end - outputs: {len(result)}")
    return result

engine.step = traced_step
```

### 3. Monitor KV Cache Usage

```python
# Get KV cache stats
def print_kv_stats(engine):
    total = engine.scheduler.kv_cache_manager.num_blocks
    free = len(engine.scheduler.kv_cache_manager.free_block_queue)
    used = total - free
    utilization = used / total * 100

    print(f"KV Cache: {used}/{total} blocks ({utilization:.1f}% used)")

# Call after each step
while engine.has_unfinished_requests():
    engine.step()
    print_kv_stats(engine)
```

### 4. Detect Preemption

```python
# Monitor preemption metrics
from prometheus_client import REGISTRY

preemption_counter = None
for metric in REGISTRY.collect():
    if metric.name == "vllm:num_preemptions_total":
        preemption_counter = metric

if preemption_counter:
    prev_count = preemption_counter.samples[0].value
    while True:
        time.sleep(1)
        curr_count = preemption_counter.samples[0].value
        if curr_count > prev_count:
            print(f"Preemption detected! Count: {curr_count}")
            prev_count = curr_count
```

---

## Sources

**Official Documentation:**
- [vLLM Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview.html) (accessed 2025-11-02)
- [vLLM Input Processing Pipeline](https://docs.vllm.ai/en/v0.5.4/dev/input_processing/input_processing_pipeline.html) (accessed 2025-11-02)
- [vLLM Troubleshooting Guide](https://docs.vllm.ai/en/stable/usage/troubleshooting.html) (accessed 2025-11-02)
- [Red Hat AI Inference Server - vLLM Arguments](https://docs.redhat.com/fr/documentation/red_hat_ai_inference_server/3.0/html-single/vllm_server_arguments/index) (accessed 2025-11-02)

**Technical Deep Dives:**
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) - Aleksa Gordic, vLLM Blog (accessed 2025-11-02)
- [Code Review: Deep Dive into vLLM's Architecture and OpenAI API Compatibility](https://zerohertz.github.io/vllm-openai-2/) - Zerohertz Blog (accessed 2025-11-02)
- [Life of an Inference Request (vLLM V1): How LLMs are Served](https://www.ubicloud.com/blog/life-of-an-inference-request-vllm-v1) - Ubicloud Blog (accessed 2025-11-02)

**Community Resources:**
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm) - Commit 42172ad (August 2025)
- [vLLM GitHub Issues - Streaming Errors](https://github.com/vllm-project/vllm/issues/22341) (accessed 2025-11-02)
- [vLLM GitHub Discussions - Engine Loop Crashes](https://github.com/vllm-project/vllm/discussions/9418) (accessed 2025-11-02)

**Additional References:**
- [What happens behind vllm serve](https://otterpeeks.com/dives/behind-vllm-serve/) - Otter Peeks (accessed 2025-11-02)
- [How vLLM Does It - Continuous Batching](https://voice.ai/hub/tts/vllm-continuous-batching/) - Voice.ai (accessed 2025-11-02)
- [vLLM Metrics API Documentation](https://docs.vllm.ai/en/latest/serving/metrics.html) (accessed 2025-11-02)

---

**Document Status**: Complete (750+ lines, comprehensive coverage of request lifecycle internals with citations and code examples)

**Last Updated**: 2025-11-02
