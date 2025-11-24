# Batch Embedding Generation at Scale

## Overview

Batch embedding generation is the process of computing vector embeddings for large volumes of text or other data using GPU-accelerated models. Efficient batch processing is critical for production embedding pipelines, enabling high throughput while maximizing GPU utilization and minimizing cost per token.

This guide covers batch sizing strategies, GPU optimization techniques, distributed processing patterns, and production-ready implementations for generating embeddings at scale.

## Why Batch Processing Matters

### Throughput vs Latency Trade-off

**Batch processing prioritizes throughput over per-request latency:**
- Single requests: Low latency (~10-50ms) but poor GPU utilization
- Batched requests: Higher per-request latency (~100-500ms) but 10-100x better throughput
- **Use batching for**: Offline indexing, bulk processing, data pipelines
- **Avoid batching for**: Real-time user queries requiring <100ms response

### GPU Utilization

GPUs achieve peak efficiency when processing data in parallel:
- **Memory bandwidth bound**: Small models bottlenecked by data transfer
- **Compute bound**: Large models bottlenecked by matrix operations
- **Optimal batch size**: Balances both constraints to maximize throughput

From [Sentence Transformers efficiency documentation](https://sbert.net/docs/sentence_transformer/usage/efficiency.html):
> Batch size significantly impacts embedding generation throughput, primarily by balancing hardware utilization and computational efficiency.

## Optimal Batch Sizing

### GPU Memory Constraints

**Batch size is limited by available GPU VRAM:**

```python
# Estimate VRAM usage
model_size_gb = 1.5  # e.g., all-MiniLM-L6-v2
batch_size = 32
sequence_length = 512
embedding_dim = 384

# Approximate VRAM calculation
activations_gb = (batch_size * sequence_length * embedding_dim * 4) / (1024**3)
total_vram_gb = model_size_gb + activations_gb + 0.5  # +0.5 for overhead

print(f"Estimated VRAM usage: {total_vram_gb:.2f} GB")
```

**GPU-specific guidelines:**
- **T4 (16GB)**: batch_size=32-64 for 512-token sequences
- **A10 (24GB)**: batch_size=64-128 for 512-token sequences
- **A100 (40/80GB)**: batch_size=128-256 for 512-token sequences
- **H100 (80GB)**: batch_size=256-512 for 512-token sequences

### Padding Overhead

**Variable-length sequences require padding, which wastes computation:**

```python
# Example: Padding overhead
texts = [
    "Short text",           # 2 tokens
    "Medium length text",  # 3 tokens
    "Very long text " * 50 # 150 tokens
]

# Batch padded to max length (150 tokens)
# Actual tokens: 2 + 3 + 150 = 155
# Padded tokens: 150 * 3 = 450
# Overhead: (450 - 155) / 450 = 65% wasted computation
```

**Strategies to reduce padding:**
1. **Sort by length**: Group similar-length texts in same batch
2. **Dynamic batching**: Pack batches to target token count, not fixed batch size
3. **Bucketing**: Pre-sort into length buckets (0-64, 64-128, 128-256, 256-512 tokens)

### Sweet Spot Analysis

From [Zilliz AI FAQ on batch sizing](https://zilliz.com/ai-faq/what-is-the-impact-of-batch-size-on-embedding-generation-throughput):
> The optimal batch size depends on your specific hardware, model architecture, and use case. There's no universal value.

**Finding the sweet spot:**

```python
import time
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
texts = ["Sample text for benchmarking"] * 1000

# Benchmark different batch sizes
batch_sizes = [8, 16, 32, 64, 128, 256]
results = {}

for bs in batch_sizes:
    start = time.time()
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        embeddings = model.encode(batch, batch_size=bs)
    elapsed = time.time() - start
    throughput = len(texts) / elapsed
    results[bs] = throughput
    print(f"Batch size {bs}: {throughput:.1f} texts/sec")

# Find optimal
optimal_bs = max(results, key=results.get)
print(f"\nOptimal batch size: {optimal_bs}")
```

**Typical results (A100 GPU, 512-token sequences):**
- Batch size 8: 400 texts/sec
- Batch size 16: 750 texts/sec
- Batch size 32: 1,200 texts/sec
- Batch size 64: 1,800 texts/sec ← **Sweet spot**
- Batch size 128: 1,850 texts/sec (diminishing returns)
- Batch size 256: OOM error

## Sentence Transformers Implementation

### Basic Batch Encoding

**Using the built-in batch processing:**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

# List of texts to embed
texts = [
    "First document to embed",
    "Second document to embed",
    # ... thousands more
]

# Batch encoding (handles batching internally)
embeddings = model.encode(
    texts,
    batch_size=64,  # Process 64 texts at a time
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True  # L2 normalization for cosine similarity
)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding shape: {embeddings[0].shape}")
```

### GPU Optimization Techniques

**1. Float16 Precision (FP16)**

From [Sentence Transformers efficiency guide](https://sbert.net/docs/sentence_transformer/usage/efficiency.html):
> Float16 (fp16, half precision) can speed up inference on GPUs at minimal loss of model accuracy.

```python
model = SentenceTransformer(
    'all-MiniLM-L6-v2',
    device='cuda',
    model_kwargs={"torch_dtype": "float16"}
)
# or: model.half()

# ~1.5-2x speedup with negligible accuracy loss
embeddings = model.encode(texts, batch_size=64)
```

**2. BFloat16 Precision (BF16)**

```python
model = SentenceTransformer(
    'all-MiniLM-L6-v2',
    device='cuda',
    model_kwargs={"torch_dtype": "bfloat16"}
)
# or: model.bfloat16()

# Similar speedup to FP16, better accuracy preservation
embeddings = model.encode(texts, batch_size=64)
```

**3. Multi-GPU Encoding**

```python
# Automatically use all available GPUs
pool = model.start_multi_process_pool()

embeddings = model.encode_multi_process(
    texts,
    pool,
    batch_size=64,
    chunk_size=5000  # Split into chunks of 5000 texts per GPU
)

model.stop_multi_process_pool(pool)
```

### Memory-Efficient Streaming

**For datasets too large to fit in RAM:**

```python
def batch_generator(file_path, batch_size=64):
    """Yield batches of texts from a file."""
    batch = []
    with open(file_path, 'r') as f:
        for line in f:
            batch.append(line.strip())
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:  # Last partial batch
            yield batch

# Process in streaming fashion
embeddings_list = []
for batch in batch_generator('large_corpus.txt', batch_size=64):
    batch_embeddings = model.encode(batch, batch_size=len(batch))
    embeddings_list.append(batch_embeddings)

# Concatenate results
all_embeddings = np.vstack(embeddings_list)
```

## Distributed Processing with Ray

From [Anyscale's Ray Data blog post](https://www.anyscale.com/blog/scaling-embedding-generation-pipelines-from-pandas-to-ray-data):
> Ray Data provides a 10x performance improvement over pandas-based pipelines with minimal code changes.

### Ray Data Pipeline

**1. Basic Ray Data implementation:**

```python
import ray
from sentence_transformers import SentenceTransformer

ray.init()

# Load data as Ray Dataset
ds = ray.data.read_text("s3://bucket/documents/*.txt")

# Define embedding function
def embed_batch(batch):
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    texts = batch["text"]
    embeddings = model.encode(texts.tolist(), batch_size=len(texts))
    batch["embeddings"] = embeddings.tolist()
    return batch

# Apply transformation (naive approach - loads model per batch)
ds = ds.map_batches(embed_batch, batch_size=64)

# Write results
ds.write_parquet("s3://bucket/embeddings/")
```

**2. Optimized with Stateful Actors (10x faster):**

```python
class EmbeddingActor:
    """Stateful actor that loads model once and reuses it."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(
            model_name,
            device='cuda',
            model_kwargs={"torch_dtype": "float16"}
        )

    def __call__(self, batch):
        texts = batch["text"].tolist()
        embeddings = self.model.encode(texts, batch_size=len(texts))
        batch["embeddings"] = embeddings.tolist()
        return batch

# Use stateful actors
ds = ds.map_batches(
    EmbeddingActor,
    fn_constructor_kwargs={"model_name": "all-MiniLM-L6-v2"},
    batch_size=64,
    num_gpus=1,      # 1 GPU per actor
    concurrency=4    # Run 4 actors in parallel (if 4 GPUs available)
)

ds.write_parquet("s3://bucket/embeddings/")
```

**3. Dynamic batching for variable-length texts:**

```python
# Chunk documents before embedding
def chunk_documents(row):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len
    )

    chunks = splitter.split_text(row["text"])
    return [{"text": chunk, "doc_id": row["id"]} for chunk in chunks]

# Process pipeline
ds = ray.data.read_text("s3://bucket/documents/*.txt")
ds = ds.flat_map(chunk_documents)  # 1 doc → many chunks
ds = ds.map_batches(
    EmbeddingActor,
    fn_constructor_kwargs={"model_name": "all-MiniLM-L6-v2"},
    batch_size=64,
    num_gpus=1,
    concurrency=4
)
ds.write_parquet("s3://bucket/embeddings/")
```

### Multi-Node Multi-GPU Setup

**Scaling to a cluster:**

```python
# Launch Ray cluster (via Ray on Kubernetes, Anyscale, etc.)
ray.init(address="ray://cluster-head:10001")

ds = ray.data.read_text("s3://bucket/large-corpus/*.txt")

# Automatically distribute across all nodes and GPUs
ds = ds.map_batches(
    EmbeddingActor,
    fn_constructor_kwargs={"model_name": "BAAI/bge-large-en-v1.5"},
    batch_size=32,       # Smaller batches for larger model
    num_gpus=1,          # 1 GPU per actor
    concurrency=16       # Total actors across cluster (e.g., 4 nodes × 4 GPUs)
)

ds.write_parquet("s3://bucket/embeddings/")
```

**Ray automatically handles:**
- Data sharding across nodes
- GPU scheduling and assignment
- Fault tolerance (retries failed tasks)
- Load balancing across heterogeneous nodes

## Production Patterns

### Checkpointing and Resumption

```python
import pickle
from pathlib import Path

def process_with_checkpoints(
    input_path: str,
    output_path: str,
    checkpoint_dir: str,
    batch_size: int = 64
):
    """Process embeddings with checkpointing for fault tolerance."""

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)
    state_file = checkpoint_path / "state.pkl"

    # Load checkpoint if exists
    if state_file.exists():
        with open(state_file, 'rb') as f:
            processed_ids = pickle.load(f)
        print(f"Resuming from checkpoint: {len(processed_ids)} already processed")
    else:
        processed_ids = set()

    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    # Read input
    with open(input_path, 'r') as f_in, open(output_path, 'a') as f_out:
        batch = []
        batch_ids = []

        for line in f_in:
            doc = json.loads(line)
            doc_id = doc['id']

            # Skip if already processed
            if doc_id in processed_ids:
                continue

            batch.append(doc['text'])
            batch_ids.append(doc_id)

            if len(batch) == batch_size:
                # Generate embeddings
                embeddings = model.encode(batch, batch_size=len(batch))

                # Write results
                for doc_id, emb in zip(batch_ids, embeddings):
                    result = {'id': doc_id, 'embedding': emb.tolist()}
                    f_out.write(json.dumps(result) + '\n')
                    processed_ids.add(doc_id)

                # Save checkpoint every 10 batches
                if len(processed_ids) % (batch_size * 10) == 0:
                    with open(state_file, 'wb') as f:
                        pickle.dump(processed_ids, f)
                    print(f"Checkpoint saved: {len(processed_ids)} processed")

                batch = []
                batch_ids = []

        # Process last batch
        if batch:
            embeddings = model.encode(batch, batch_size=len(batch))
            for doc_id, emb in zip(batch_ids, embeddings):
                result = {'id': doc_id, 'embedding': emb.tolist()}
                f_out.write(json.dumps(result) + '\n')

    # Clean up checkpoint
    state_file.unlink()
    print("Processing complete!")
```

### Error Handling and Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def embed_with_retry(model, texts, batch_size=64):
    """Embed texts with automatic retry on failure."""
    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False
        )
        return embeddings
    except RuntimeError as e:
        if "out of memory" in str(e):
            # Reduce batch size and retry
            torch.cuda.empty_cache()
            new_batch_size = max(1, batch_size // 2)
            print(f"OOM error, reducing batch size to {new_batch_size}")
            return embed_with_retry(model, texts, new_batch_size)
        else:
            raise
```

### Monitoring and Metrics

```python
import time
from collections import defaultdict

class EmbeddingMetrics:
    """Track embedding generation metrics."""

    def __init__(self):
        self.start_time = time.time()
        self.total_texts = 0
        self.total_tokens = 0
        self.batch_times = []
        self.errors = defaultdict(int)

    def record_batch(self, num_texts, num_tokens, elapsed_time):
        self.total_texts += num_texts
        self.total_tokens += num_tokens
        self.batch_times.append(elapsed_time)

    def record_error(self, error_type):
        self.errors[error_type] += 1

    def get_stats(self):
        elapsed = time.time() - self.start_time
        return {
            'total_texts': self.total_texts,
            'total_tokens': self.total_tokens,
            'throughput_texts_per_sec': self.total_texts / elapsed,
            'throughput_tokens_per_sec': self.total_tokens / elapsed,
            'avg_batch_time': sum(self.batch_times) / len(self.batch_times),
            'errors': dict(self.errors)
        }

# Usage
metrics = EmbeddingMetrics()

for batch in batches:
    start = time.time()
    try:
        embeddings = model.encode(batch)
        elapsed = time.time() - start
        metrics.record_batch(
            num_texts=len(batch),
            num_tokens=sum(len(t.split()) for t in batch),
            elapsed_time=elapsed
        )
    except Exception as e:
        metrics.record_error(type(e).__name__)

print(metrics.get_stats())
```

## Performance Benchmarks

### Sentence Transformers Throughput

From community benchmarks and [Medium article on batch size tuning](https://medium.com/@vici0549/it-is-crucial-to-properly-set-the-batch-size-when-using-sentence-transformers-for-embedding-models-3d41a3f8b649):

**all-MiniLM-L6-v2 (22M params) on A100 GPU:**
- Batch size 16: ~800 texts/sec (512 tokens each)
- Batch size 32: ~1,400 texts/sec
- Batch size 64: ~2,000 texts/sec ← **Optimal**
- Batch size 128: ~2,100 texts/sec (marginal gain)

**BAAI/bge-large-en-v1.5 (335M params) on A100 GPU:**
- Batch size 8: ~200 texts/sec
- Batch size 16: ~350 texts/sec
- Batch size 32: ~500 texts/sec ← **Optimal**
- Batch size 64: ~520 texts/sec (marginal gain)

### vLLM Optimization Case Study

From [Snowflake's vLLM optimization blog](https://medium.com/snowflake/scaling-vllm-for-embeddings-16x-throughput-and-cost-reduction-f2b4d4c8e1bf):

**Key optimizations achieved 16x throughput improvement:**

1. **Encoding as Little-Endian Bytes**:
   - Reduced serialization overhead from 90% to <10% of total time
   - Used NumPy vectorization: `embedding_tensor.numpy().astype(dtype="<f4").tobytes()`

2. **Disaggregated Tokenization**:
   - Separated tokenization (CPU) from inference (GPU)
   - Enabled pipeline parallelism across requests
   - **Result**: GPU no longer waits idle during tokenization

3. **Multiple Model Replicas on One GPU**:
   - Runs multiple instances of embedding model on single GPU
   - Better utilizes GPU streaming multiprocessors (SMs)
   - **Result**: Increased throughput from 14k to 230k tokens/sec on A10

**Before vs After (snowflake-arctic-embed-m-v1.5 on H200):**
- **Short sequences (50 tokens)**: 16x speedup
- **Long sequences (512 tokens)**: 4.2x speedup
- **Cost reduction**: 16x lower cost per trillion tokens

### GPU Utilization Patterns

**Monitoring GPU utilization:**

```bash
# Monitor GPU usage during embedding generation
nvidia-smi dmon -s mu -d 1

# Expected patterns:
# Good utilization: 80-95% GPU usage, consistent over time
# Poor utilization: <50% GPU usage, spiky pattern
# Memory bottleneck: High memory bandwidth, low compute %
# Compute bottleneck: High compute %, moderate memory bandwidth
```

**Interpreting metrics:**
- **GPU Util 90%+**: Batch size well-tuned
- **GPU Util 50-70%**: Increase batch size or reduce tokenization overhead
- **GPU Util <50%**: Likely CPU-bound (tokenization, data loading)
- **Memory Util 95%+**: At VRAM limit, can't increase batch size further

## Comparison: CPU vs GPU

**When to use CPU for embeddings:**
- Small models (<100M params)
- Batch size <16
- Sporadic requests (GPU warmup overhead not worth it)
- Cost-sensitive scenarios with ample CPU capacity

**When to use GPU for embeddings:**
- Medium to large models (>100M params)
- Batch size >32
- Continuous high-throughput workloads
- Tight latency requirements (<100ms per batch)

**Typical CPU performance (32-core Intel Xeon):**
- all-MiniLM-L6-v2, batch_size=8: ~40 texts/sec
- With ONNX quantization (int8): ~120 texts/sec (3x speedup)

**Typical GPU performance (A100):**
- all-MiniLM-L6-v2, batch_size=64: ~2,000 texts/sec (50x faster than CPU)

## Best Practices Checklist

**Batch sizing:**
- ✓ Profile batch sizes from 8 to 256, measure throughput
- ✓ Monitor GPU memory usage, stay below 90% VRAM
- ✓ Sort inputs by length to reduce padding overhead
- ✓ Use dynamic batching for variable-length inputs

**GPU optimization:**
- ✓ Use FP16 or BF16 precision for 1.5-2x speedup
- ✓ Leverage multi-GPU for large-scale processing
- ✓ Pre-tokenize inputs to enable pipeline parallelism
- ✓ Keep model loaded in stateful actors/workers

**Production readiness:**
- ✓ Implement checkpointing for fault tolerance
- ✓ Add retry logic with exponential backoff
- ✓ Monitor throughput, GPU utilization, error rates
- ✓ Use streaming/chunked processing for large datasets

**Distributed processing:**
- ✓ Use Ray Data for scaling beyond single machine
- ✓ Configure concurrency based on available GPUs
- ✓ Enable autoscaling for variable workloads
- ✓ Store embeddings in columnar format (Parquet) for efficient downstream use

## Sources

**Web Research:**

- [Sentence Transformers: Speeding up Inference](https://sbert.net/docs/sentence_transformer/usage/efficiency.html) (accessed 2025-02-02)
- [Anyscale Blog: Scaling Embedding Generation Pipelines from Pandas to Ray Data](https://www.anyscale.com/blog/scaling-embedding-generation-pipelines-from-pandas-to-ray-data) (accessed 2025-02-02)
- [Snowflake Medium: Scaling vLLM for Embeddings: 16x Throughput](https://medium.com/snowflake/scaling-vllm-for-embeddings-16x-throughput-and-cost-reduction-f2b4d4c8e1bf) (accessed 2025-02-02)
- [Zilliz AI FAQ: Optimal Batch Size for Embeddings](https://zilliz.com/ai-faq/what-is-the-optimal-batch-size-for-generating-embeddings) (accessed 2025-02-02)
- [Medium: Batch Size Tuning for Sentence Transformers](https://medium.com/@vici0549/it-is-crucial-to-properly-set-the-batch-size-when-using-sentence-transformers-for-embedding-models-3d41a3f8b649) (accessed 2025-02-02)
- [Milvus AI FAQ: Impact of Batch Size on Throughput](https://milvus.io/ai-quick-reference/how-can-you-do-batch-processing-of-sentences-for-embedding-to-improve-throughput-when-using-sentence-transformers) (accessed 2025-02-02)
- [GitHub: Sentence Transformers Batch Size Discussion](https://github.com/UKPLab/sentence-transformers/issues/2551) (accessed 2025-02-02)
- [Ray Documentation: Distributed Training](https://docs.ray.io/en/latest/ray-overview/examples/e2e-multimodal-ai-workloads/notebooks/02-Distributed-Training.html) (accessed 2025-02-02)

**Additional References:**
- [NVIDIA Developer: cuEmbed for Accelerating Embedding Lookups](https://developer.nvidia.com/blog/accelerating-embedding-lookups-with-cuembed)
- [Tecton Blog: Building High Performance Embeddings Engine](https://www.tecton.ai/blog/embeddings-engine/)
- [Reddit r/MachineLearning: Generating Embeddings for Large Datasets](https://www.reddit.com/r/MachineLearning/comments/1ah2z4b/pgenerating_embeddings_for_a_large_dataset_in_the/)
