# vLLM Engine & Worker Architecture

## Overview

vLLM's engine and worker architecture is a sophisticated multi-process system designed for high-throughput LLM inference. The system orchestrates distributed GPU execution through a driver-worker model, with careful attention to minimizing GPU idle time and CPU overhead.

**Key architectural components:**
- LLMEngine: Main synchronous/asynchronous inference coordinator
- EngineCore: Core scheduling and execution logic
- Workers: GPU process executors (UniProcExecutor, MultiProcExecutor)
- ModelRunner: Manages forward passes and execution
- Scheduler: Request scheduling and KV cache management

From [vLLM Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview.html) (accessed 2025-01-31):
The LLMEngine and AsyncLLMEngine classes are central to vLLM's functioning, handling model inference and asynchronous request processing.

## Section 1: LLMEngine Overview (~140 lines)

### Engine Initialization and Configuration

The LLMEngine is the fundamental building block of vLLM. On its own, it enables high-throughput inference in offline settings before web serving scaffolding is added.

**Main components initialized during construction:**

1. **vLLM Config** - Contains all configuration knobs for model, cache, parallelism, etc.
2. **Processor** - Converts raw inputs → EngineCoreRequests via validation, tokenization, processing
3. **Engine Core Client** - In simple cases uses InprocClient (≈ EngineCore); scales to DPLBAsyncMPClient for distributed serving
4. **Output Processor** - Converts raw EngineCoreOutputs → RequestOutput for users

From [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://aleksagordic.com/blog/vllm) (accessed 2025-01-31):
> The LLM engine is the fundamental building block of vLLM. On its own, it already enables high-throughput inference - but only in an offline setting. You can't serve it to customers over the web yet.

**Engine core sub-components:**

```python
# Core components structure
EngineCore:
  - Model Executor (UniProcExecutor or MultiProcExecutor)
  - Structured Output Manager (for guided decoding)
  - Scheduler:
      - Policy: FCFS or Priority-based
      - Queues: waiting, running
      - KV Cache Manager (heart of paged attention)
```

**KV cache manager** maintains a `free_block_queue` - a pool of available KV-cache blocks (often hundreds of thousands depending on VRAM size and block size).

Block size calculation for standard transformer:
```
block_size = 2 (key/value) * block_size_tokens (default=16)
             * num_kv_heads * head_size * dtype_num_bytes (e.g. 2 for bf16)
```

### Main Inference Loop Architecture

The engine's generate function follows this flow:

**Phase 1: Request Feeding**
1. Create unique request ID and capture arrival time
2. Tokenize prompt via input preprocessor
3. Pack into EngineCoreRequest with priority, sampling params
4. Add to scheduler's waiting queue (append for FCFS, heap-push for priority)

**Phase 2: Execution Loop**
While requests exist, repeatedly call `step()` with three stages:

```python
def step():
    # 1. Schedule
    scheduled_requests = scheduler.schedule()

    # 2. Forward Pass
    outputs = model_executor.execute_model(scheduled_requests)

    # 3. Postprocess
    process_outputs(outputs)  # append tokens, check stop conditions
```

From [Inside vLLM](https://aleksagordic.com/blog/vllm) (accessed 2025-01-31):
> Because the forward pass flattens the batch into a single sequence and custom kernels handle it efficiently, continuous batching is fundamentally supported even in the synchronous engine.

**Stop conditions checked:**
- Request exceeds length limit (max_model_length or max_tokens)
- Sampled token is EOS ID (unless ignore_eos enabled)
- Token matches stop_token_ids
- Stop strings present in output (truncates at first occurrence)

### Request Handling Pipeline

**Two main request types:**

1. **Prefill requests** - Forward pass over all prompt tokens
   - Usually compute-bound (depends on hardware and prompt length)
   - Samples single token from final position's probability distribution
   - Can use chunked prefill for long prompts

2. **Decode requests** - Forward pass over most recent token only
   - Memory-bandwidth-bound (loading weights + KV cache for one token)
   - All earlier KV vectors cached

The V1 scheduler can mix prefill and decode in same step (V0 could only do one type per step).

**Scheduling priority:**
1. Decode requests (already in running queue) processed first
2. Prefill requests from waiting queue processed second
3. Token budget tracked and decremented for each scheduled request

### Output Streaming Mechanism

In streaming mode, intermediate tokens are sent as generated. The output processor:

1. Receives sampled token IDs from forward pass
2. Detokenizes incrementally
3. Checks stop conditions
4. Emits RequestOutput objects with:
   - Generated text (incremental or full)
   - Finish reason (if complete)
   - Token IDs
   - Logprobs (if requested)

From [vLLM V1 Engine Architecture](https://github.com/vllm-project/vllm/issues/8779) (accessed 2025-01-31):
> De-tokenizer moves to the driver process. Async de-tokenization can be regarded as part of async scheduling.

**Performance consideration:** De-tokenization is expensive, especially for large batch sizes. V1 architecture moves this to driver process for async execution.

## Section 2: Worker Architecture (~180 lines)

### GPU Worker Processes (Ray-based Distribution)

vLLM manages distributed runtime with either Ray or Python native multiprocessing. Multiprocessing is used by default on single nodes with sufficient GPUs; Ray is required for multi-node inference.

From [Distributed Inference and Serving](https://docs.vllm.ai/en/v0.7.2/serving/distributed_serving.html) (accessed 2025-01-31):
> We manage the distributed runtime with either Ray or python native multiprocessing. Multiprocessing can be used when deploying on a single node, multi-node inferencing currently requires Ray.

**Process architecture:**

```
Single GPU (no distributed):
  LLMEngine → UniProcExecutor → Single Worker Process → GPU 0

Multi-GPU (TP=4):
  LLMEngine → MultiProcExecutor → 4 Worker Processes → GPUs 0-3
                                    ├─ Worker 0 (rank 0, driver)
                                    ├─ Worker 1 (rank 1)
                                    ├─ Worker 2 (rank 2)
                                    └─ Worker 3 (rank 3)

Multi-Node (TP=4, PP=2):
  LLMEngine → MultiProcExecutor → 8 Worker Processes across 2 nodes
```

**V1 Architecture Improvement:**
From [vLLM V1 Issue](https://github.com/vllm-project/vllm/issues/8779) (accessed 2025-01-31):
> Driver process + SPMD workers: When TP=n & PP=m, vLLM engine will have n*m + 1 processes in total. Even when using a single GPU, we will have 2 processes. The driver process will have the scheduler, memory manager, etc. The workers are stateful, maintaining most of the request states.

### Worker Initialization Sequence

Each worker (whether in UniProcExecutor or MultiProcExecutor) goes through three key initialization procedures:

**1. Init Device**
```python
def init_device():
    # Assign CUDA device (e.g. "cuda:0")
    # Check model dtype support (e.g. bf16)
    # Verify VRAM availability given gpu_memory_utilization (e.g. 0.8 = 80%)
    # Set up distributed settings (DP/TP/PP/EP)
    # Instantiate ModelRunner (holds sampler, KV cache, forward-pass buffers)
    # Instantiate InputBatch (holds CPU-side buffers, block tables)
```

**2. Load Model**
```python
def load_model():
    # Instantiate model architecture
    # Load model weights
    # Call model.eval() (PyTorch inference mode)
    # Optional: torch.compile() on model
```

**3. Initialize KV Cache**
```python
def initialize_kv_cache():
    # Get per-layer KV-cache spec
    # Run dummy/profiling forward pass
    # Take GPU memory snapshot
    # Compute how many KV blocks fit in available VRAM
    # Allocate, reshape, bind KV cache tensors to attention layers
    # Prepare attention metadata (e.g. FlashAttention backend)
    # Unless --enforce-eager: capture CUDA graphs for warmup batch sizes
```

From [Inside vLLM](https://aleksagordic.com/blog/vllm) (accessed 2025-01-31):
> CUDA graphs record the whole sequence of GPU work into a DAG. Later during fwd pass we launch/replay pre-baked graphs and cut on kernel launch overhead and thus improve latency.

### Model Loading Strategy

**Weight loading approaches:**

1. **Standard loading** - Load full model weights on each worker
2. **Tensor parallel sharding** - Each worker loads only its shard
3. **Pipeline parallel stages** - Each worker loads only its pipeline stage layers
4. **Quantization-aware loading** - Load quantized weights (GPTQ, AWQ, etc.)

**Optimization: Reduce redundant loading**
- In tensor parallel mode, weights are sharded across workers
- Only load necessary shard per worker to save memory
- Use memory-mapped weights when possible to reduce loading time

### Worker Communication Patterns

**MultiProcExecutor communication flow:**

```python
class MultiProcExecutor:
    def __init__():
        # 1. Initialize rpc_broadcast_mq (shared memory message queue)
        self.rpc_broadcast_mq = MessageQueue()

        # 2. Spawn worker processes
        for rank in range(world_size):
            worker_proc = WorkerProc.make_worker_process(rank)

        # 3. Each worker creates response queue
        # 4. Workers block on rpc_broadcast_mq.dequeue()

    def execute_model(inputs):
        # 5. Enqueue to rpc_broadcast_mq (non-blocking, broadcast)
        self.rpc_broadcast_mq.enqueue(inputs)

        # 6. Wait on designated output rank's response queue
        result = worker_response_mq.dequeue()  # blocking
        return result
```

From [Inside vLLM](https://aleksagordic.com/blog/vllm) (accessed 2025-01-31):
> The parent first creates a reader and writer pipe. The new process runs WorkerProc.worker_main, which instantiates a worker. Each worker determines whether it is the driver (rank 0 in the TP group) or a regular worker.

**Key insights from V1 design:**
- Workers are stateful (maintain request states)
- Driver only sends "diffs" (new requests, scheduled IDs, new block IDs)
- No need to send token IDs, sampling params every step for in-flight requests
- Clean up of SeqGroupMetadata data structure overhead

**Communication overhead reduction:**
From [V1 Architecture](https://github.com/vllm-project/vllm/issues/8779) (accessed 2025-01-31):
> Input broadcasting is expensive. Instead of sending request information from scheduler to workers every step, the workers should be stateful and maintain most of the request states.

## Section 3: ModelRunner & Executor (~160 lines)

### ModelRunner Responsibilities

The ModelRunner sits between the Worker and the actual model, managing all aspects of forward pass execution.

**Core responsibilities:**

1. **State Management** - Track request states, KV cache blocks, sampling metadata
2. **Input Preparation** - Build model inputs from request metadata
3. **Forward Execution** - Run model with custom kernels (PagedAttention)
4. **Output Processing** - Gather logits, sample tokens
5. **Memory Management** - Update KV cache, handle block allocation

**ModelRunner structure:**

```python
class ModelRunner:
    def __init__():
        self.model = None  # Loaded during worker.load_model()
        self.kv_cache = None  # Allocated during worker.init_kv_cache()
        self.input_batch = InputBatch()  # CPU-side buffers
        self.attention_backend = None  # e.g. FlashAttention

    def execute_model(scheduler_output):
        # 1. Update states
        # 2. Prepare inputs
        # 3. Forward pass
        # 4. Gather last-token states
        # 5. Sample
```

### Input Preparation and Batching

**Five main steps in prepare_inputs:**

**Step 1: Update States**
```python
# Prune finished requests from input_batch
# Update KV cache blocks per request
# Update misc forward-pass metadata
```

**Step 2: Prepare Inputs**
```python
# Copy buffers CPU→GPU
# Compute positions for each token
# Build slot_mapping (maps token positions to KV cache blocks)
# Construct attention metadata (for paged attention kernels)
```

From [Inside vLLM](https://aleksagordic.com/blog/vllm) (accessed 2025-01-31):
> All sequences are flattened and concatenated into one long "super sequence". Position indices and attention masks ensure each sequence only attends to its own tokens, which enables continuous batching without right-padding.

**Example: Continuous batching with paged attention**

```
Request A: tokens [1,2,3,4] → blocks [B0, B1]
Request B: tokens [5,6] → blocks [B2]

Flattened input_ids: [1,2,3,4,5,6]
Positions: [0,1,2,3,0,1]
Slot mapping: [B0_slot0, B0_slot1, ..., B2_slot0, B2_slot1]

Attention: Each sequence attends only to its own tokens via block tables
```

**Step 3: Build Attention Metadata**
```python
# For prefill: query length, key length per sequence
# For decode: always query_len=1, key_len=previous_tokens
# Block tables: mapping from logical token position to physical KV blocks
```

**V1 Optimization:**
From [V1 Architecture](https://github.com/vllm-project/vllm/issues/8779) (accessed 2025-01-31):
> Preparing the model & sampler inputs (e.g., block table) is expensive. We should cache the inputs of the previous steps, and build new inputs incrementally from the cached inputs, if possible.

### Execution Engine

**Two execution modes:**

**1. Eager Mode** (--enforce-eager)
```python
def forward_eager(input_ids, positions, kv_caches):
    # Standard PyTorch forward pass
    hidden_states = model.forward(
        input_ids=input_ids,
        positions=positions,
        kv_caches=kv_caches
    )
    return hidden_states
```

**2. CUDA Graph Mode** (default)
```python
def forward_cuda_graph(input_ids, positions, kv_caches):
    # Copy inputs into pre-allocated CUDA graph buffers
    cuda_graph_input_buffer.copy_(input_ids)

    # Replay captured CUDA graph (eliminates kernel launch overhead)
    cuda_graph.replay()

    # Read outputs from pre-allocated output buffer
    return cuda_graph_output_buffer
```

**CUDA Graph benefits:**
- Eliminates per-kernel launch overhead
- Records full execution DAG
- Significantly improves latency for small batches
- Captured during warmup for different batch sizes

**Step 4: Gather Last-Token States**
```python
# Extract hidden states at each sequence's final position
last_token_indices = compute_last_token_indices(seq_lens)
last_hidden_states = hidden_states[last_token_indices]

# Compute logits (vocabulary projection)
logits = model.lm_head(last_hidden_states)
```

### Output Collection

**Step 5: Sample Tokens**

```python
def sample(logits, sampling_params):
    # Apply temperature, top-p, top-k
    # Apply guided decoding masks (if FSM enabled)
    # Sample from distribution
    # Return: (next_token_ids, logprobs_optional)
```

**Sampling overhead:**
From [V1 Architecture](https://github.com/vllm-project/vllm/issues/8779) (accessed 2025-01-31):
> Sampler is expensive. The GPU operations themselves are not very expensive. However, "pythonizing" the sampler outputs is expensive. Plus, the sampler can launch many small GPU kernels with CPU-GPU synchronizations.

**Output structure:**
```python
@dataclass
class SamplerOutput:
    sampled_token_ids: torch.Tensor  # [batch_size]
    sampled_token_probs: Optional[torch.Tensor]
    logprobs: Optional[torch.Tensor]  # if requested
```

**Post-sampling processing:**
1. Append sampled tokens to request sequences
2. Update KV cache block occupancy
3. Check stop conditions
4. Free completed request resources
5. Return outputs to engine

**V1 async scheduling improvement:**
From [V1 Architecture](https://github.com/vllm-project/vllm/issues/8779) (accessed 2025-01-31):
> Async single-step scheduling, instead of multi-step scheduling. Scheduler will schedule the n+1-th step, while the worker is executing the n-th step.

This allows overlapping scheduling with GPU execution, minimizing idle time.

## Section 4: Distributed Engine (~140 lines)

### Multi-GPU Coordination

vLLM supports distributed tensor-parallel and pipeline-parallel inference following Megatron-LM's tensor parallel algorithm.

**Decision tree for distributed strategy:**

From [Distributed Serving Guide](https://docs.vllm.ai/en/v0.7.2/serving/distributed_serving.html) (accessed 2025-01-31):

1. **Single GPU** - No distributed inference if model fits
2. **Single-Node Multi-GPU** - Use tensor parallelism (TP = num_gpus)
3. **Multi-Node Multi-GPU** - Use TP + pipeline parallelism (PP)

**Example configurations:**

```bash
# Single node, 4 GPUs (TP=4)
vllm serve model --tensor-parallel-size 4

# Two nodes, 8 GPUs total (TP=4, PP=2)
vllm serve model --tensor-parallel-size 4 --pipeline-parallel-size 2

# Multi-node, tensor parallel only (TP=16 across 2 nodes)
vllm serve model --tensor-parallel-size 16
```

### Tensor Parallelism Integration

**Tensor parallelism shards model across GPUs:**

```
Original model:
  Linear(4096 → 11008)  # 45M parameters

TP=4 sharding:
  GPU 0: Linear(4096 → 2752)  # 11M parameters
  GPU 1: Linear(4096 → 2752)  # 11M parameters
  GPU 2: Linear(4096 → 2752)  # 11M parameters
  GPU 3: Linear(4096 → 2752)  # 11M parameters
```

**Communication pattern:**
```python
# Column-parallel linear
x = [batch, seq, 4096]
local_out = linear_shard(x)  # [batch, seq, 2752] on each GPU

# All-reduce to get full output
out = all_reduce(local_out)  # [batch, seq, 11008]
```

**TP group setup:**
```python
def init_distributed_environment():
    # Initialize process group (NCCL for GPU)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=tp_size,
        rank=local_rank
    )

    # Create TP communication group
    tp_group = torch.distributed.new_group(ranks=list(range(tp_size)))
```

From [Distributed Serving](https://docs.vllm.ai/en/v0.7.2/serving/distributed_serving.html) (accessed 2025-01-31):
> The tensor parallel size should be the number of GPUs in each node, and the pipeline parallel size should be the number of nodes.

### Pipeline Parallelism Support

**Pipeline parallelism splits model layers across nodes:**

```
4-layer model, PP=2:

Node 0 (Stage 0):
  - Embedding
  - Layer 0
  - Layer 1

Node 1 (Stage 1):
  - Layer 2
  - Layer 3
  - LM Head
```

**Communication pattern:**
```python
# Forward pass
if stage == 0:
    hidden = embed(input_ids)
    hidden = layer_0(hidden)
    hidden = layer_1(hidden)
    send_to_next_stage(hidden)  # Send to Node 1
else:
    hidden = recv_from_prev_stage()  # Receive from Node 0
    hidden = layer_2(hidden)
    hidden = layer_3(hidden)
    logits = lm_head(hidden)
```

**PP scheduling consideration:**
From [V1 Architecture](https://github.com/vllm-project/vllm/issues/8779) (accessed 2025-01-31):
> Needs a special care for PP, since the output token IDs from the last stage should be sent to the first stage.

This is because the last PP stage produces sampled tokens needed by first stage for next decode step.

### Ray Distributed Framework

**Ray is used for multi-node coordination:**

```python
import ray

@ray.remote(num_gpus=1)
class RayWorker:
    def __init__(self, model_config):
        self.worker = Worker(model_config)

    def execute_model(self, inputs):
        return self.worker.execute_model(inputs)

# Create workers across nodes
workers = [
    RayWorker.remote(model_config)
    for _ in range(tp_size * pp_size)
]

# Execute distributed
futures = [
    worker.execute_model.remote(inputs)
    for worker in workers
]
results = ray.get(futures)
```

**Multi-node setup process:**

From [Distributed Serving](https://docs.vllm.ai/en/v0.7.2/serving/distributed_serving.html) (accessed 2025-01-31):

**Head node:**
```bash
bash run_cluster.sh \
    vllm/vllm-openai \
    ip_of_head_node \
    --head \
    /path/to/huggingface/home \
    -e VLLM_HOST_IP=ip_of_this_node
```

**Worker nodes:**
```bash
bash run_cluster.sh \
    vllm/vllm-openai \
    ip_of_head_node \
    --worker \
    /path/to/huggingface/home \
    -e VLLM_HOST_IP=ip_of_this_node
```

**Network requirements:**
- High-speed interconnect (Infiniband recommended for cross-node TP)
- Proper NCCL configuration for GPU-Direct RDMA
- All nodes must access same model weights (shared filesystem or pre-download)

**Verifying efficient communication:**
From [Distributed Serving](https://docs.vllm.ai/en/v0.7.2/serving/distributed_serving.html) (accessed 2025-01-31):
> If you find `[send] via NET/Socket` in the logs, it means NCCL uses raw TCP Socket, which is not efficient for cross-node tensor parallel. If you find `[send] via NET/IB/GDRDMA` in the logs, it means NCCL uses Infiniband with GPU-Direct RDMA, which is efficient.

## Section 5: Profiling Engine Performance (~100 lines)

### Engine Bottlenecks Identification

From [V1 Architecture Lessons Learned](https://github.com/vllm-project/vllm/issues/8779) (accessed 2025-01-31):

**Critical bottlenecks identified:**

1. **CPU Performance**
   > "To achieve high GPU utilization, we should care about everything happening on the CPU. Python is slow. Fast GPUs like H100 do not necessarily have fast CPUs."

2. **Scheduling Overhead**
   > "For every step, the vLLM scheduler goes over the whole self.running queue and performs some operations for each request. And this is written in Python."

3. **Input Broadcasting**
   > "Input broadcasting is expensive. Instead of sending request information from scheduler to workers every step, the workers should be stateful."

4. **Input Preparation**
   > "Preparing the model & sampler inputs (e.g., block table) is expensive. We should cache the inputs and build new inputs incrementally."

5. **De-tokenization**
   > "For every step, vLLM de-tokenizes the generated output token IDs and checks the stop criteria. The overhead becomes significant for large batch sizes."

6. **Sampler Operations**
   > "The GPU operations themselves are not very expensive. However, 'pythonizing' the sampler outputs is expensive. Plus, the sampler can launch many small GPU kernels with CPU-GPU synchronizations."

### Worker Utilization Monitoring

**Key metrics to track:**

```python
# GPU utilization
gpu_utilization = gpu_time / (gpu_time + gpu_idle_time)

# Request queue depth
waiting_queue_size = len(scheduler.waiting)
running_queue_size = len(scheduler.running)

# KV cache utilization
kv_cache_usage = allocated_blocks / total_blocks

# Throughput metrics
tokens_per_second = total_tokens / elapsed_time
requests_per_second = completed_requests / elapsed_time
```

**Profiling tools:**

From [Inside vLLM](https://aleksagordic.com/blog/vllm) (accessed 2025-01-31):
> vLLM provides a `vllm bench {serve,latency,throughput}` CLI that wraps vllm/benchmarks/{server,latency,throughput}.py

**Benchmark types:**

1. **Latency benchmark**
   - Short input (default 32 tokens)
   - Small batch (default 8)
   - Measures end-to-end latency

2. **Throughput benchmark**
   - Fixed prompts (default 1000 ShareGPT samples)
   - QPS=Inf mode (submit all at once)
   - Reports tokens/requests per second

3. **Serve benchmark**
   - Simulates real workload
   - Poisson/Gamma distributed arrivals
   - Measures TTFT, ITL, E2E latency
   - Optional max concurrency limit

### Communication Overhead Analysis

**Sources of communication overhead:**

1. **Tensor Parallel All-Reduce**
   ```python
   # Every TP operation requires synchronization
   output = all_reduce(local_output)  # Blocking collective
   ```

2. **Pipeline Parallel Send/Recv**
   ```python
   # Sequential dependency between stages
   if pp_rank > 0:
       hidden = recv_from_prev()  # Blocking
   hidden = forward(hidden)
   if pp_rank < pp_size - 1:
       send_to_next(hidden)  # Blocking
   ```

3. **KV Cache Transfer (Disaggregated P/D)**
   - Prefill writes KV to external store
   - Decode reads KV from external store
   - Network bandwidth critical

**Measuring communication:**

```python
import torch.distributed as dist

# Profile all-reduce time
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
dist.all_reduce(tensor, group=tp_group)
end.record()
torch.cuda.synchronize()

allreduce_time_ms = start.elapsed_time(end)
```

**Optimization strategies:**

1. **Overlap communication with computation**
   - Use async all-reduce when possible
   - Pipeline micro-batches in PP

2. **Reduce communication frequency**
   - V1: Stateful workers reduce input broadcasting
   - Cache and incrementally update inputs

3. **Use efficient interconnect**
   - Infiniband with GPU-Direct RDMA for multi-node
   - NVLink for intra-node TP

4. **Minimize CPU-GPU synchronization**
   - Batch small GPU operations
   - Use CUDA graphs to eliminate kernel launch overhead

**Performance analysis workflow:**

```bash
# 1. Profile with PyTorch profiler
python -m torch.utils.bottleneck your_script.py

# 2. Check NCCL communication
NCCL_DEBUG=INFO vllm serve model ...

# 3. GPU profiling with nsys
nsys profile -o output.qdrep vllm serve model ...

# 4. Analyze with vLLM benchmarks
vllm bench throughput --model model-name \
    --tensor-parallel-size 4 \
    --input-len 128 --output-len 256
```

From [Inside vLLM](https://aleksagordic.com/blog/vllm) (accessed 2025-01-31):
> Below a saturation batch B_sat, the step time is dominated by HBM bandwidth (streaming weights layer-by-layer), so step latency is nearly flat. Beyond B_sat, kernels become compute-bound and step time grows roughly with B.

This roofline model explains why batch size affects latency vs throughput tradeoff differently in prefill (compute-bound) vs decode (memory-bound) workloads.

## Sources

**Official Documentation:**
- [vLLM Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview.html) - Official architecture documentation (accessed 2025-01-31)
- [Distributed Inference and Serving](https://docs.vllm.ai/en/v0.7.2/serving/distributed_serving.html) - Multi-GPU and multi-node deployment guide (accessed 2025-01-31)
- [vLLM GPU Model Runner API](https://docs.vllm.ai/en/stable/api/vllm/v1/worker/gpu_model_runner.html) - ModelRunner API reference (accessed 2025-01-31)

**Technical Deep Dives:**
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://aleksagordic.com/blog/vllm) by Aleksa Gordić (accessed 2025-01-31) - Comprehensive technical analysis of vLLM internals
- [vLLM's V1 Engine Architecture](https://github.com/vllm-project/vllm/issues/8779) - GitHub Issue #8779 describing V1 design goals and lessons learned (accessed 2025-01-31)

**Additional References:**
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm) - Main source code repository (accessed 2025-01-31)
- Megatron-LM Tensor Parallel Paper: "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" - https://arxiv.org/pdf/1909.08053.pdf
