# Vector Index Optimization & Tuning

## Overview

Vector index optimization is the process of tuning index parameters to achieve optimal performance across three competing factors: recall (accuracy), query latency (speed), and memory usage (cost). This guide covers systematic approaches to optimizing FAISS, HNSW, and IVF indexes for production deployments.

The optimization challenge: improving any one factor typically degrades others. High recall requires more computation (slower), larger indexes use more memory (costly), and faster searches sacrifice accuracy (lower recall). Production systems must find the sweet spot for their specific use case.

## Index Performance Factors

### The Performance Triangle

Vector indexes navigate three competing objectives:

**Recall vs Latency**: Higher recall requires searching more candidates, increasing query time. A 99% recall target may be 10x slower than 90% recall.

**Latency vs Memory**: Memory-efficient indexes (like PQ compression) add decompression overhead, increasing query latency by 20-50%.

**Memory vs Recall**: Aggressive compression reduces memory by 8-32x but can drop recall by 5-15% compared to uncompressed indexes.

Production systems must prioritize based on requirements:
- Real-time search (<100ms): Optimize latency, accept 90-95% recall
- High-accuracy retrieval: Target 98-99% recall, tolerate 200-500ms latency
- Cost-sensitive deployments: Maximize compression, accept accuracy trade-offs

### Key Performance Metrics

**Recall@K**: Percentage of true nearest neighbors found in top-K results. Gold standard for accuracy measurement.

**Queries Per Second (QPS)**: Throughput metric. Production systems target 100-1000 QPS depending on scale.

**P50/P95/P99 Latency**: Median, 95th, and 99th percentile query times. P95/P99 matter more than P50 for user experience.

**Memory Footprint**: RAM usage per million vectors. Critical for cost optimization and deployment feasibility.

**Build Time**: Index construction duration. Affects iteration speed during development and reindexing frequency in production.

From [FAISS Index IO Documentation](https://github.com/facebookresearch/faiss/wiki/Index-IO,-cloning-and-hyper-parameter-tuning) (accessed 2025-02-02):
- FAISS provides `AutoTuneCriterion` for systematic parameter exploration
- `OperatingPoints` object tracks Pareto-optimal parameter combinations
- `ParameterSpace` scans indexes for tunable parameters automatically

## HNSW Optimization

HNSW (Hierarchical Navigable Small World) graphs offer excellent recall-latency trade-offs through three primary parameters.

### M Parameter: Graph Connectivity

**M** controls bidirectional links per node. Higher M creates a denser graph with more routing options.

**Trade-offs**:
- Low M (4-16): 0.5-1.5GB for 1M vectors, fast build, moderate recall
- Medium M (32-64): 2-5GB for 1M vectors, balanced performance
- High M (128-512): 5-20GB for 1M vectors, maximum recall, expensive

**Tuning guidelines**:
- Start with M=32 for most applications
- Increase to M=64 if recall <95% and memory allows
- Reduce to M=16 for memory-constrained deployments
- Never exceed M=512 (diminishing returns, memory explosion)

From [Pinecone HNSW Guide](https://www.pinecone.io/learn/series/faiss/hnsw/) (accessed 2025-02-02):
- M affects both search quality and memory usage linearly
- M_max automatically set to M, M_max0 set to 2*M
- Higher layers use longer-range links, lower layers use shorter links

### efConstruction: Build Quality

**efConstruction** controls candidate exploration during index construction. Higher values build better graphs.

**Trade-offs**:
- Low efConstruction (40-100): Fast build, may have suboptimal graph structure
- Medium efConstruction (200-400): Balanced build time and quality
- High efConstruction (500-1000): Slow build, maximum graph quality

**Tuning guidelines**:
- Set efConstruction ≥ 2 * M as baseline
- Increase if recall plateaus when raising efSearch
- Build-time impact is one-time cost, prioritize quality over speed
- Values >1000 rarely improve results

**Key insight**: efConstruction affects build time significantly but has minimal impact on search time. It's a "set high and forget" parameter.

### efSearch: Query Depth

**efSearch** controls candidate exploration during search. Primary knob for recall-latency trade-off.

**Trade-offs**:
- Low efSearch (16-64): <10ms queries, 85-92% recall
- Medium efSearch (128-256): 10-50ms queries, 95-98% recall
- High efSearch (512-1024): 50-200ms queries, 99%+ recall

**Tuning guidelines**:
- Start with efSearch = K (number of results returned)
- Increase efSearch until recall meets target
- Monitor P95 latency, not just P50
- Different queries can use different efSearch values

From [Medium HNSW Optimization](https://medium.com/@bakingai/optimize-hnsw-parameters-in-faiss-for-better-searches-d9ed0bdf7fef) (accessed 2025-02-02):
- Set efConstruction higher than efSearch for best results
- Monitor recall and latency together during tuning
- efSearch is the only parameter that should be tuned at query time

### HNSW Memory Optimization

**Memory usage**: Base storage + graph overhead

For 1M 128-dim float32 vectors:
- Base vectors: 1M * 128 * 4 bytes = 512MB
- HNSW graph (M=32): ~2GB additional
- HNSW graph (M=64): ~4GB additional
- Total: 2.5-4.5GB for M=32-64

**Optimization strategies**:
1. **Product Quantization**: Compress vectors to 64-128 bytes, reducing base storage by 4-8x
2. **Smaller M values**: Linear reduction in graph overhead
3. **Memory-mapped indexes**: Keep graph on disk, page in as needed (slower but cheaper)

**HNSW + PQ example**:
```
Original: 2.5GB (IndexHNSWFlat, M=32)
With PQ: 650MB (IndexHNSWPQ, M=32, 16-byte codes)
Savings: 74% reduction, ~5% recall drop
```

## IVF Optimization

IVF (Inverted File) indexes partition the vector space into cells, enabling faster search by scanning only a subset of cells.

### nlist: Number of Cells

**nlist** determines space partitioning granularity. More cells = smaller cells = faster search but worse recall.

**Trade-offs**:
- Low nlist (100-1000): Large cells, slower but thorough
- Medium nlist (2000-8000): Balanced for 10M-100M vectors
- High nlist (16000+): Tiny cells, fast but may miss nearby vectors

**Tuning guidelines**:
```
Vectors        Recommended nlist    Rationale
1M             1,000-4,000          ~250-1000 vectors/cell
10M            4,000-16,000         ~625-2500 vectors/cell
100M           16,000-64,000        ~1600-6250 vectors/cell
1B             64,000-256,000       ~4000-15600 vectors/cell
```

**Rule of thumb**: Target 1000-5000 vectors per cell for optimal balance.

From [FAISS Indexes Wiki](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) (accessed 2025-02-02):
- nlist affects both build and search performance
- Smaller nlist = more accurate coarse quantizer search
- Larger nlist = faster query after coarse quantizer

### nprobe: Search Width

**nprobe** controls how many cells are searched. Primary query-time parameter.

**Trade-offs**:
- Low nprobe (1-4): Very fast, 60-80% recall
- Medium nprobe (16-64): Balanced, 90-96% recall
- High nprobe (128-256): Thorough, 97-99% recall

**Tuning guidelines**:
- Start with nprobe = sqrt(nlist) / 2
- Double nprobe if recall is insufficient
- Monitor nprobe * cell_size for actual vectors scanned
- Diminishing returns after nprobe > nlist / 4

**Example tuning sequence**:
```python
nlist = 4096
# Start conservative
nprobe = 32  # Recall: 92%, Latency: 8ms

# Increase for higher recall
nprobe = 64  # Recall: 95%, Latency: 15ms
nprobe = 128 # Recall: 97%, Latency: 28ms

# Typically stop here, further increases inefficient
nprobe = 256 # Recall: 98%, Latency: 52ms
```

### IVF Training Strategies

**Training data requirements**:
- Minimum: 30 * nlist vectors (e.g., 120K for nlist=4096)
- Recommended: 100 * nlist vectors for stable centroids
- Maximum: Diminishing returns after 256 * nlist

**Training quality factors**:
- Use representative sample from full dataset
- K-means iterations: 25-50 typically sufficient
- Multiple training runs with different seeds can help
- Monitor quantization error during training

From [FAISS Index Factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory) (accessed 2025-02-02):
- IVF indexes require training on representative data
- Poor training degrades recall more than suboptimal nprobe
- Consider stratified sampling for imbalanced datasets

### IVF Build Optimization

**Parallel construction**:
- FAISS supports multi-threaded `add()` operations
- Speedup scales linearly up to ~16 threads
- Diminishing returns after 32 threads
- Watch memory: each thread needs temporary buffers

**Batch insertion**:
```python
# Efficient: batch insertion
index.add(vectors)  # Add all at once

# Inefficient: per-vector insertion
for v in vectors:
    index.add(v)  # 10-100x slower
```

**Incremental updates**:
- IVF indexes support incremental `add()` after training
- Cells become imbalanced with many additions
- Rebalancing: Retrain + rebuild every 10-50% growth
- Alternative: Use IndexIVFFlat with periodic retraining

## Memory Optimization

### Quantization Techniques

**Product Quantization (PQ)**:
- Divides vectors into subvectors, quantizes each independently
- Compression: 32x (float32 → 1 byte per subvector with 256 codebook)
- Recall impact: 5-10% drop typical, depends on data
- Speed impact: 1.5-2x slower due to distance computation overhead

**Scalar Quantization (SQ)**:
- Converts float32 to int8/uint8 per dimension
- Compression: 4x (32 bits → 8 bits)
- Recall impact: 1-3% drop, minimal for normalized vectors
- Speed impact: Minimal, can be faster with SIMD instructions

**Binary Quantization**:
- Reduces to 1-bit per dimension
- Compression: 32x (float32 → 1 bit)
- Recall impact: 10-20% drop, requires high dimensions (>128)
- Speed impact: Very fast with Hamming distance

From [OpenSearch Memory-Optimized Vectors](https://docs.opensearch.org/latest/field-types/supported-field-types/knn-memory-optimized/) (accessed 2025-02-02):
- Binary vectors reduce memory by 32x with acceptable recall
- Best for high-dimensional spaces (512+)
- Combine with rescoring for accuracy recovery

### Memory-Mapped Indexes

**On-disk indexes**:
- Store index on disk, memory-map for access
- Effectively unlimited capacity
- 2-10x slower than in-memory (depends on access patterns)
- OS page cache critical for performance

**Implementation**:
```python
# Write index to disk
faiss.write_index(index, "large.index")

# Memory-map from disk
index = faiss.read_index("large.index",
                          faiss.IO_FLAG_MMAP)
```

**Best practices**:
- Use for cold/archive indexes searched infrequently
- SSDs dramatically better than HDDs (10-50x faster random access)
- Monitor page cache hit rates
- Consider tiered storage (hot in RAM, warm on SSD, cold on HDD)

### Compression Trade-off Analysis

**Memory reduction vs recall loss**:
```
Method              Compression    Recall Drop    Speed Impact
IndexFlatL2         1x (baseline)  0%            1x (fastest)
IndexSQ8            4x             1-3%          1.0-1.2x
IndexPQ16           8x             5-10%         1.5-2x
IndexPQ32           16x            8-15%         1.5-2x
Binary              32x            15-25%        0.5-0.8x (Hamming)
```

**Selection criteria**:
- <5% recall drop acceptable: Use SQ8
- <10% recall drop OK: Use PQ with 8-16 byte codes
- Memory critical: Use PQ32 or binary, accept accuracy loss
- No accuracy compromise: Use IndexFlat, scale horizontally

From [Redis Vector Quantization](https://redis.io/docs/latest/develop/ai/search-and-query/vectors/svs-compression/) (accessed 2025-02-02):
- Advanced methods like LVQ (Locally-adaptive Vector Quantization) improve compression-accuracy trade-off
- LeanVec and other learned compression methods can match PQ quality with 2x better compression

## Monitoring & Tuning

### Key Metrics to Track

**Search performance**:
```
Metric               Target Range       Alert Threshold
P50 Latency          <50ms             >100ms
P95 Latency          <200ms            >500ms
P99 Latency          <500ms            >1000ms
QPS                  Project-specific  <50% capacity
Error Rate           <0.1%             >1%
```

**Index quality**:
```
Metric               Target Range       Alert Threshold
Recall@10            >95%              <90%
Recall@100           >90%              <85%
nDCG                 >0.90             <0.80
```

**Resource usage**:
```
Metric               Target Range       Alert Threshold
Memory Usage         <80% allocated    >90%
CPU Utilization      <70%              >85%
Disk I/O (mmap)      <60% bandwidth    >80%
```

From [Qdrant Vector Search Optimization](https://qdrant.tech/articles/vector-search-resource-optimization/) (accessed 2025-02-02):
- Monitor recall drift over time as data distribution changes
- Track query patterns to optimize for common access patterns
- Use compression and partitioning together for maximum efficiency

### A/B Testing Strategies

**Parameter comparison**:
1. Deploy two index configurations side-by-side
2. Route 50% traffic to each variant
3. Measure recall, latency, and user engagement metrics
4. Run for 24-48 hours to capture daily patterns
5. Choose winner based on primary metric

**A/B testing framework**:
```python
# Variant A: High recall (current)
index_a = IndexHNSWFlat(d, M=64)
index_a.hnsw.efSearch = 256

# Variant B: Lower latency (test)
index_b = IndexHNSWFlat(d, M=32)
index_b.hnsw.efSearch = 128

# Route queries randomly
if random.random() < 0.5:
    results, latency = search(index_a, query)
    log_metrics("variant_a", latency, recall)
else:
    results, latency = search(index_b, query)
    log_metrics("variant_b", latency, recall)
```

**Metrics to compare**:
- Recall@K (compute with small held-out set)
- P95 latency (user experience critical)
- QPS capacity (throughput)
- Memory usage (cost)
- Click-through rate (if applicable)

From [Dynatrace AI Observability](https://docs.dynatrace.com/docs/analyze-explore-automate/dynatrace-for-ai-observability/get-started/sample-use-cases/ab-model-testing) (accessed 2025-02-02):
- A/B testing crucial for assessing impact of changes to vector databases
- Monitor training dataset changes, algorithm updates, and parameter tweaks
- Track embedding quality drift over time

### Dynamic Parameter Adjustment

**Adaptive efSearch**:
```python
def adaptive_search(index, query, target_latency_ms=50):
    """Dynamically adjust efSearch based on latency budget."""
    ef_search = 64  # Start conservative

    while ef_search <= 512:
        start = time.time()
        results = index.search(query, k=10)
        latency_ms = (time.time() - start) * 1000

        if latency_ms < target_latency_ms * 0.8:
            # Under budget, can increase quality
            ef_search *= 2
            index.hnsw.efSearch = ef_search
        else:
            # At or over budget, use current setting
            break

    return results
```

**Load-based adjustment**:
```python
def load_adaptive_params(current_qps, capacity_qps):
    """Reduce quality under high load to maintain throughput."""
    load_factor = current_qps / capacity_qps

    if load_factor < 0.5:
        return {"nprobe": 128, "efSearch": 256}
    elif load_factor < 0.8:
        return {"nprobe": 64, "efSearch": 128}
    else:
        # High load: prioritize throughput
        return {"nprobe": 32, "efSearch": 64}
```

**Time-of-day optimization**:
- Peak hours: Reduce parameters for lower latency
- Off-peak: Increase parameters for better recall
- Use cron jobs or monitoring triggers
- Maintain separate indexes if switching overhead is high

From [VDTuner Research](https://arxiv.org/html/2404.10413v1) (accessed 2025-02-02):
- VDTuner dynamically scores different index types during tuning
- Automated performance tuning can save weeks of manual optimization
- Consider query patterns and data characteristics together

### Monitoring Tools & Practices

**Metrics collection**:
```python
import time
from dataclasses import dataclass

@dataclass
class SearchMetrics:
    query_id: str
    latency_ms: float
    recall_at_10: float
    num_candidates: int
    index_type: str
    timestamp: float

def monitored_search(index, query, ground_truth=None):
    """Wrap search with comprehensive monitoring."""
    start = time.time()
    results, distances = index.search(query, k=10)
    latency_ms = (time.time() - start) * 1000

    # Compute recall if ground truth available
    recall = None
    if ground_truth is not None:
        recall = compute_recall(results, ground_truth)

    metrics = SearchMetrics(
        query_id=hash(query),
        latency_ms=latency_ms,
        recall_at_10=recall,
        num_candidates=index.hnsw.efSearch if hasattr(index, 'hnsw') else None,
        index_type=type(index).__name__,
        timestamp=time.time()
    )

    # Ship to monitoring system
    log_metrics(metrics)

    return results, distances
```

**Dashboard recommendations**:
- Real-time latency percentiles (P50/P95/P99)
- Recall tracking with sampling
- QPS and throughput graphs
- Memory and CPU usage
- Error rates and timeouts
- Parameter value tracking (efSearch, nprobe changes)

### Continuous Optimization

**Scheduled retuning**:
1. Weekly: Analyze metric trends
2. Monthly: A/B test parameter improvements
3. Quarterly: Consider index architecture changes
4. Annually: Evaluate new index types and algorithms

**Trigger-based retuning**:
- Recall drops below threshold: Increase search parameters
- Latency spikes: Reduce parameters or add capacity
- Memory pressure: Consider compression or scaling out
- New data patterns: Retrain IVF/HNSW from scratch

**Reindexing strategy**:
```python
# Blue-green deployment for zero-downtime reindex
def safe_reindex(old_index, new_vectors, new_params):
    # Build new index
    new_index = build_optimized_index(new_vectors, new_params)

    # Warm up with sample queries
    warmup_queries = sample_recent_queries(n=1000)
    for q in warmup_queries:
        new_index.search(q, k=10)

    # Switch traffic atomically
    swap_index(old_index, new_index)

    # Monitor for regressions
    monitor_metrics(duration=3600)  # 1 hour

    # Rollback if issues detected
    if detect_regression():
        swap_index(new_index, old_index)
        alert("Reindex rolled back")
```

From [AWS Aurora pgvector Optimization](https://aws.amazon.com/blogs/database/supercharging-vector-search-performance-and-relevance-with-pgvector-0-8-0-on-amazon-aurora-postgresql/) (accessed 2025-02-02):
- pgvector 0.8.0 delivers up to 9x faster query processing
- Continuous monitoring and tuning critical for maintaining performance
- Balance index optimization with query pattern analysis

## Build Optimization

### Parallel Construction

**Multi-threading**:
```python
# Set number of threads for index construction
faiss.omp_set_num_threads(16)

# Train in parallel
index.train(training_vectors)  # Uses all threads

# Add vectors in parallel
index.add(vectors)  # Automatically parallelized
```

**Speedup analysis**:
```
Threads    Build Time    Speedup    Notes
1          100 min       1.0x       Baseline
4          28 min        3.6x       Near-linear
8          15 min        6.7x       Good scaling
16         9 min         11.1x      Diminishing returns
32         7 min         14.3x      Limited by memory bandwidth
```

**Best practices**:
- Use physical cores, not hyperthreads
- Monitor memory bandwidth saturation
- Stop at 16-32 threads for most workloads
- Consider distributed training for very large datasets

### Batch Insertion Strategies

**Optimal batch sizes**:
```python
# Good: Add in large batches
batch_size = 100_000
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i+batch_size]
    index.add(batch)

# Bad: Add one at a time
for vector in vectors:
    index.add(vector)  # 10-100x slower!
```

**Memory considerations**:
- Batch size limited by available RAM
- Target: 10-20% of total dataset per batch
- Larger batches better for IVF (reduces quantization overhead)
- HNSW can handle smaller batches efficiently

### Incremental Updates

**Adding new vectors**:
```python
# IVF indexes: Can add after training
index = IndexIVFFlat(quantizer, d, nlist)
index.train(training_vectors)
index.add(training_vectors)

# Later: Add new vectors
index.add(new_vectors)  # Works, but may degrade quality

# Rebalance periodically
if index.ntotal > original_size * 1.5:
    reindex(index)  # Retrain for balanced cells
```

**HNSW incremental addition**:
- HNSW handles incremental adds naturally
- Quality degrades slightly with many additions
- Consider rebuilding every 2-3x growth
- No retraining needed (no clustering)

From [FAISS Index Factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory) (accessed 2025-02-02):
- Index factory syntax enables quick experimentation with different configurations
- Composite indexes combine multiple optimization techniques
- Example: "IVF4096,PQ32" creates IVF with 4096 cells and 32-byte PQ codes

## Production Patterns

### Multi-Stage Search

**Coarse-to-fine strategy**:
```python
def multi_stage_search(query, k=10):
    # Stage 1: Fast initial candidates (low recall)
    index_fast.hnsw.efSearch = 64
    candidates, _ = index_fast.search(query, k=100)

    # Stage 2: Rerank with precise index (high recall)
    index_precise = IndexFlatL2(d)
    index_precise.add(vectors[candidates])
    final_results, _ = index_precise.search(query, k=10)

    return final_results
```

**Benefits**:
- Reduces P99 latency by limiting worst-case search
- Improves recall without full scan
- Can use different index types per stage

### Cascading Thresholds

**Adaptive quality**:
```python
def cascading_search(query, k=10, min_recall=0.95):
    # Try fast first
    efSearch = 32
    while efSearch <= 512:
        index.hnsw.efSearch = efSearch
        results, scores = index.search(query, k)

        # Check if confident (high distances)
        if min(scores) < confidence_threshold:
            return results  # High confidence, return

        # Try harder
        efSearch *= 2

    return results  # Return best effort
```

### Index Sharding

**Horizontal scaling**:
```python
# Partition by vector ID
num_shards = 4
shards = [IndexHNSWFlat(d, M=32) for _ in range(num_shards)]

# Distribute vectors
for i, vector in enumerate(vectors):
    shard_id = i % num_shards
    shards[shard_id].add(vector)

# Query all shards
def sharded_search(query, k=10):
    all_results = []
    for shard in shards:
        results, distances = shard.search(query, k)
        all_results.extend(zip(results[0], distances[0]))

    # Merge and return top-k
    all_results.sort(key=lambda x: x[1])
    return all_results[:k]
```

**Scaling benefits**:
- Linear QPS scaling with shards
- Reduced memory per node
- Fault tolerance (continue with N-1 shards)
- Diminishing returns after 8-16 shards (coordination overhead)

### Hot/Cold Tiering

**Tiered storage**:
```
Tier     Storage    Size        Access Pattern      Performance
Hot      RAM        10-20%      Recent, popular     <50ms P95
Warm     SSD        30-40%      Moderate access     <200ms P95
Cold     HDD/S3     40-60%      Rare access         <1s P95
```

**Implementation**:
```python
class TieredIndex:
    def __init__(self):
        self.hot = IndexHNSWFlat(d, M=32)   # In RAM
        self.warm = read_index("warm.index", IO_FLAG_MMAP)  # SSD
        self.cold = S3Index("s3://bucket/cold.index")  # S3

    def search(self, query, k=10):
        # Try hot first
        results = self.hot.search(query, k)
        if len(results) >= k:
            return results

        # Fall back to warm
        results_warm = self.warm.search(query, k - len(results))
        results.extend(results_warm)

        # Last resort: cold
        if len(results) < k:
            results_cold = self.cold.search(query, k - len(results))
            results.extend(results_cold)

        return results[:k]
```

From [Milvus Efficient Filtering](https://milvus.io/blog/how-to-filter-efficiently-without-killing-recall.md) (accessed 2025-02-02):
- Efficient filtering critical for production vector search
- Innovative optimizations in Milvus enable fast filtered search
- Consider pre-filtering vs post-filtering trade-offs

## Sources

**Source Documents:**
None (this guide is based entirely on web research)

**Web Research:**
- [FAISS Index IO and Parameter Tuning](https://github.com/facebookresearch/faiss/wiki/Index-IO,-cloning-and-hyper-parameter-tuning) - GitHub (accessed 2025-02-02): Comprehensive guide to FAISS auto-tuning, parameter spaces, and optimization reliability

- [Pinecone HNSW Guide](https://www.pinecone.io/learn/series/faiss/hnsw/) - Pinecone Learning (accessed 2025-02-02): Detailed HNSW implementation, parameter explanations, and performance benchmarks with Faiss

- [Medium HNSW Optimization](https://medium.com/@bakingai/optimize-hnsw-parameters-in-faiss-for-better-searches-d9ed0bdf7fef) - Baking AI (accessed 2025-02-02): Practical HNSW parameter tuning tips with M, efConstruction, and efSearch guidelines

- [FAISS Indexes Wiki](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) - GitHub (accessed 2025-02-02): Complete reference for FAISS index types, parameters, and selection guidelines

- [Qdrant Vector Search Optimization](https://qdrant.tech/articles/vector-search-resource-optimization/) - Qdrant Blog (accessed 2025-02-02): Advanced compression, partitioning, and resource optimization techniques

- [VDTuner Automated Tuning](https://arxiv.org/html/2404.10413v1) - arXiv (accessed 2025-02-02): Research on automated vector database performance tuning with dynamic index selection

- [AWS Aurora pgvector Performance](https://aws.amazon.com/blogs/database/supercharging-vector-search-performance-and-relevance-with-pgvector-0-8-0-on-amazon-aurora-postgresql/) - AWS Blog (accessed 2025-02-02): Production optimization lessons achieving 9x faster query processing

- [Dynatrace AI Observability](https://docs.dynatrace.com/docs/analyze-explore-automate/dynatrace-for-ai-observability/get-started/sample-use-cases/ab-model-testing) - Dynatrace Docs (accessed 2025-02-02): A/B testing frameworks for vector databases and AI model versioning

- [OpenSearch Memory-Optimized Vectors](https://docs.opensearch.org/latest/field-types/supported-field-types/knn-memory-optimized/) - OpenSearch Docs (accessed 2025-02-02): Binary quantization and memory reduction strategies for production deployments

- [Redis Vector Quantization](https://redis.io/docs/latest/develop/ai/search-and-query/vectors/svs-compression/) - Redis Docs (accessed 2025-02-02): LVQ, LeanVec, and advanced compression methods for memory optimization

- [Milvus Efficient Filtering](https://milvus.io/blog/how-to-filter-efficiently-without-killing-recall.md) - Milvus Blog (accessed 2025-02-02): Production patterns for filtered vector search without recall degradation

**Additional References:**
- [FAISS Index Factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory) - Quick index experimentation and composite index creation syntax
