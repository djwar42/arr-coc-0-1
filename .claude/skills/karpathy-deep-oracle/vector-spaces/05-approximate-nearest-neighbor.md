# Approximate Nearest Neighbor Search (ANN)

## Overview

Approximate Nearest Neighbor (ANN) search is a fundamental technique for efficiently finding similar items in high-dimensional vector spaces. Unlike exact search methods that guarantee finding the true nearest neighbors, ANN algorithms trade some accuracy for dramatic speed improvements, making them essential for billion-scale vector search in production systems.

**Why ANN?** Exact k-NN search has O(n) complexity - you must compare the query against every vector in the database. For billion-scale datasets, this is computationally prohibitive. ANN algorithms reduce complexity to O(log n) or even O(1) by using intelligent data structures that sacrifice perfect recall for speed.

**Core Trade-off**: Speed vs Accuracy. All ANN methods balance three key metrics:
- **Recall**: Percentage of true nearest neighbors found
- **Query Speed**: Time to retrieve k neighbors
- **Memory Usage**: Index size and RAM requirements

From [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) (Malkov & Yashunin, 2016):
> "We present a new approach for the approximate K-nearest neighbor search based on navigable small world graphs with controllable hierarchy (Hierarchical NSW, HNSW). The proposed solution is fully graph-based, without any need for additional search structures."

## ANN Overview: Exact vs Approximate Search

### Exact Search

**Brute Force (Flat Index)**:
- Compare query to every vector in database
- Complexity: O(n·d) where n = dataset size, d = dimensions
- Guarantees 100% recall
- Impractical for n > 1M vectors

**Example**: Searching 1B vectors @ 768 dimensions @ 10 GFLOPS:
```
1,000,000,000 × 768 = 768 billion operations
768B ops ÷ 10 GFLOPS = 76.8 seconds per query
```

### Approximate Search Benefits

ANN reduces search time from seconds to milliseconds:

| Dataset Size | Exact Search | ANN (HNSW) | Speedup |
|--------------|-------------|------------|---------|
| 1M vectors   | 100ms       | 1ms        | 100x    |
| 10M vectors  | 1s          | 2ms        | 500x    |
| 100M vectors | 10s         | 5ms        | 2000x   |
| 1B vectors   | 100s        | 10ms       | 10000x  |

From [Pinecone HNSW Guide](https://www.pinecone.io/learn/series/faiss/hnsw/):
> "HNSW produces state-of-the-art performance with super fast search speeds and fantastic recall... time and time again."

**Speed-Accuracy Trade-off Visualization**:
```
Recall (%)
100 |     Exact Search (slow)
    |    /
 95 |   * HNSW (M=64)
    |  /
 90 | * HNSW (M=32)
    |/
 85 * IVF (nprobe=50)
    |
 80 * LSH (L=20)
    |________________ Query Time (ms)
       1   10   100
```

## Section 1: HNSW Algorithm (100 lines)

### Hierarchical Navigable Small World Graphs

**HNSW** combines two powerful concepts:
1. **Skip Lists**: Multi-layer structure with exponentially decreasing vertex density
2. **Navigable Small Worlds**: Graphs with both long-range and short-range links

From [Understanding HNSW for Vector Search](https://milvus.io/blog/understand-hierarchical-navigable-small-worlds-hnsw-for-vector-search.md) (Milvus, 2025):
> "Hierarchical Navigable Small World (HNSW) is a graph-based algorithm that performs approximate nearest neighbor searches (ANN) in vector databases. HNSW creates a multi-layered graph structure where each layer is a navigable small world network."

**Key Innovation**: Start search at the top sparse layer with long-range links (fast traversal), then progressively move down to denser layers with shorter links (accurate refinement).

### Graph Construction

**Layer Assignment**: Each vector gets assigned to layers 0 through l with exponentially decaying probability:

```
P(layer = l) = exp(-l / m_L) × (1 - exp(-1 / m_L))

where m_L = 1/ln(M) for optimal performance
```

**Parameters**:
- **M**: Number of bi-directional links per vertex (typically 16-64)
- **efConstruction**: Size of dynamic candidate list during construction (typically 100-500)
- **efSearch**: Size of dynamic candidate list during search (typically 100-500)

From [HNSW Paper](https://arxiv.org/abs/1603.09320):
> "Starting search from the upper layer together with utilizing the scale separation boosts the performance compared to NSW and allows a logarithmic complexity scaling."

**Construction Process**:
```
For each new vector v:
  1. Randomly select insertion layer l
  2. Start at top layer entry point
  3. Greedy search to find closest M neighbors at each layer
  4. Add bidirectional links to M nearest neighbors
  5. Descend to next layer
  6. Repeat until reaching layer 0
```

**Graph Structure Visualization**:
```
Layer 3:  o ←——————————————————————→ o  (entry point, longest links)
           ↓                         ↓
Layer 2:  o ←———→ o ←————→ o ←——————→ o  (medium links)
           ↓       ↓        ↓         ↓
Layer 1:  o → o → o → o → o → o → o → o  (shorter links)
           ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
Layer 0:  [all vectors with shortest links]  (base layer)
```

### Search Algorithm

**Two-Phase Greedy Search**:

1. **Zoom-out phase** (upper layers): Long hops, few vertices
2. **Zoom-in phase** (lower layers): Short hops, many vertices

```python
def hnsw_search(query, entry_point, ef, K):
    current_layer = max_layer
    current_nearest = entry_point

    # Phase 1: Traverse upper layers (zoom-out)
    while current_layer > 0:
        current_nearest = greedy_search_layer(
            query, current_nearest, current_layer, ef=1
        )
        current_layer -= 1

    # Phase 2: Search layer 0 (zoom-in)
    candidates = greedy_search_layer(
        query, current_nearest, layer=0, ef=ef
    )

    return top_k(candidates, K)

def greedy_search_layer(query, entry, layer, ef):
    visited = set()
    candidates = PriorityQueue()  # max-heap by distance
    results = PriorityQueue()     # min-heap by distance

    candidates.push(entry, distance(query, entry))
    results.push(entry, distance(query, entry))

    while candidates:
        current = candidates.pop()

        if distance(query, current) > distance(query, results.top()):
            break  # Local minimum reached

        for neighbor in current.neighbors[layer]:
            if neighbor not in visited:
                visited.add(neighbor)
                dist = distance(query, neighbor)

                if dist < distance(query, results.top()) or len(results) < ef:
                    candidates.push(neighbor, dist)
                    results.push(neighbor, dist)

                    if len(results) > ef:
                        results.pop()  # Remove farthest

    return results
```

From [Pinecone HNSW Implementation](https://www.pinecone.io/learn/series/faiss/hnsw/):
> "We enter the top layer, where we find the longest links. These vertices will tend to be higher-degree vertices (with links separated across multiple layers), meaning that we, by default, start in the zoom-in phase described for NSW."

### HNSW Performance Characteristics

**Complexity**:
- Construction: O(N × log(N) × M × efConstruction)
- Search: O(log(N) × M × efSearch)
- Memory: O(N × M × avg_layers) where avg_layers ≈ 1/(1 - exp(-1/m_L))

**Parameter Tuning** (from FAISS benchmarks):

```
High Recall (95%+):     M=64, efConstruction=500, efSearch=300
Balanced (90%):         M=32, efConstruction=200, efSearch=100
Fast Search (85%):      M=16, efConstruction=100, efSearch=50
```

**Memory Usage Example** (FAISS, Sift1M dataset):
```
M=2:   0.5 GB
M=16:  1.2 GB
M=32:  2.1 GB
M=64:  3.8 GB
M=128: 4.9 GB
```

## Section 2: IVF (Inverted File Index) (80 lines)

### Clustering-Based Search

**IVF** partitions the vector space into Voronoi cells using k-means clustering, then uses an inverted index to map each cluster to its assigned vectors.

From [Inverted File Indexing (IVF) in FAISS](https://medium.com/@Jawabreh0/inverted-file-indexing-ivf-in-faiss-a-comprehensive-guide-c183fe979d20) (Jawabreh, 2024):
> "Inverted File Indexing (IVF) is an efficient indexing strategy for large-scale vector search. By organizing data into clusters and storing references in posting lists, IVF reduces the computational complexity of nearest neighbor searches."

**Core Idea**: Don't search all vectors - only search vectors in the nearest clusters.

### IVF Construction

**Step 1: Coarse Quantization (Clustering)**

Train k-means on database vectors to create centroids:

```python
# Train k-means quantizer
nlist = int(sqrt(N))  # Rule of thumb: sqrt(dataset_size)
quantizer = faiss.IndexFlatL2(d)
kmeans = faiss.Kmeans(d, nlist, niter=20, verbose=True)
kmeans.train(database_vectors)

# Add centroids to quantizer
quantizer.add(kmeans.centroids)
```

**Step 2: Assign Vectors to Clusters**

Each vector is assigned to its nearest centroid:

```python
# Assign vectors to clusters (posting lists)
_, assignments = quantizer.search(database_vectors, 1)

# Build inverted lists
inverted_lists = defaultdict(list)
for vector_id, cluster_id in enumerate(assignments):
    inverted_lists[cluster_id].append(vector_id)
```

From [FAISS IVF Documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes):
> "The feature space is partitioned into nlist cells. The database vectors are assigned to one of these cells using a quantization function, and stored in an inverted file structure formed of nlist inverted lists."

### IVF Search Algorithm

**Two-Stage Search**:

1. **Coarse search**: Find nprobe nearest centroids to query
2. **Fine search**: Compare query to vectors in selected clusters

```python
def ivf_search(query, nprobe, k):
    # Stage 1: Coarse search - find nearest centroids
    centroid_dists, centroid_ids = quantizer.search(query, nprobe)

    # Stage 2: Fine search - scan inverted lists
    candidates = []
    for cluster_id in centroid_ids[0]:
        for vector_id in inverted_lists[cluster_id]:
            vector = database[vector_id]
            dist = distance(query, vector)
            candidates.append((dist, vector_id))

    # Return top-k
    candidates.sort(key=lambda x: x[0])
    return candidates[:k]
```

**Complexity Reduction**:
```
Exact search:      O(N × d)
IVF search:        O(nlist × d + (nprobe/nlist × N) × d)
                 ≈ O(sqrt(N) × d) when nlist = sqrt(N), nprobe = sqrt(N)

Speedup: N / sqrt(N) = sqrt(N)
```

### IVF Parameter Selection

**nlist (Number of Clusters)**:

Rule of thumb from [FAISS Guidelines](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes):
```
nlist = C × sqrt(N)

where C ≈ 10 balances:
- Assignment cost: O(nlist × d)
- Scan cost: O(nprobe/nlist × N × d)

Examples:
  1M vectors:   nlist = 1,000 - 4,000
  10M vectors:  nlist = 4,000 - 16,000
  100M vectors: nlist = 16,000 - 64,000
  1B vectors:   nlist = 65,536 - 262,144
```

**nprobe (Search Width)**:

From [Similarity Search with IVF](https://towardsdatascience.com/similarity-search-knn-inverted-file-index-7cab80cc0e79):

```
nprobe controls recall vs speed:

nprobe = 1:      ~50% recall, fastest
nprobe = 10:     ~80% recall, 10x slower
nprobe = 50:     ~90% recall, 50x slower
nprobe = nlist:  100% recall, exhaustive (no speedup)
```

### IVF Variants

**IndexIVFFlat**: No compression, exact distances after coarse quantization
```python
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
```

**IndexIVFPQ**: Product quantization for compression (covered in vector-spaces/04-vector-quantization.md)
```python
index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
```

**IndexIVFScalarQuantizer**: Scalar quantization (8-bit, 6-bit, 4-bit)
```python
index = faiss.IndexIVFScalarQuantizer(quantizer, d, nlist, faiss.ScalarQuantizer.QT_8bit)
```

## Section 3: LSH (Locality Sensitive Hashing) (70 lines)

### Random Projection for Similarity

**LSH** uses random hash functions that map similar vectors to the same hash buckets with high probability, unlike traditional hash functions that minimize collisions.

From [Locality Sensitive Hashing: The Illustrated Guide](https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/) (Pinecone):
> "LSH is almost the opposite [of traditional hashing]. In LSH, we want to maximize collisions - although ideally only for similar inputs."

**Core Property**: If vectors are similar, they should hash to the same bucket:
```
If sim(v1, v2) is high  =>  P(hash(v1) == hash(v2)) is high
If sim(v1, v2) is low   =>  P(hash(v1) == hash(v2)) is low
```

### Random Projection LSH

**Hash Function**: Project vector onto random hyperplane and threshold:

```
h(v) = sign(r · v)

where r ~ N(0, I) is a random unit vector
```

For binary codes with L bits, use L random projections:

```python
def create_lsh_hash_functions(d, L):
    """Create L random projection vectors"""
    return np.random.randn(L, d)

def hash_vector(v, projections):
    """Compute L-bit binary hash code"""
    dot_products = projections @ v
    return (dot_products > 0).astype(int)

# Example
projections = create_lsh_hash_functions(d=128, L=64)
hash_code = hash_vector(query_vector, projections)
# Result: [1, 0, 1, 1, 0, ...] (64 bits)
```

From [Approximate Nearest Neighbor with LSH](https://pyimagesearch.com/2025/01/27/approximate-nearest-neighbor-with-locality-sensitive-hashing-lsh/) (PyImageSearch, 2025):
> "Locality Sensitive Hashing (LSH) is a technique used to perform approximate nearest neighbor searches in high-dimensional spaces by reducing dimensionality while preserving distance."

### LSH Index Construction

**Multi-Probe LSH** uses multiple hash tables to increase recall:

```python
class LSHIndex:
    def __init__(self, d, L, K):
        """
        d: vector dimension
        L: number of hash tables
        K: bits per hash table
        """
        self.hash_functions = [
            np.random.randn(K, d) for _ in range(L)
        ]
        self.tables = [defaultdict(list) for _ in range(L)]

    def add(self, vector, vector_id):
        """Add vector to all hash tables"""
        for i, proj in enumerate(self.hash_functions):
            hash_code = tuple(hash_vector(vector, proj))
            self.tables[i][hash_code].append(vector_id)

    def search(self, query, k):
        """Find candidates from all hash tables"""
        candidates = set()

        for i, proj in enumerate(self.hash_functions):
            hash_code = tuple(hash_vector(query, proj))
            candidates.update(self.tables[i][hash_code])

        # Rank candidates by actual distance
        results = []
        for vector_id in candidates:
            vector = database[vector_id]
            dist = distance(query, vector)
            results.append((dist, vector_id))

        results.sort(key=lambda x: x[0])
        return results[:k]
```

### LSH Parameter Tuning

**Trade-offs**:

From [LSH Parameters](https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/):

```
More bits (K) per table:
  ✓ Better precision (fewer false positives)
  ✗ Lower recall (more false negatives)
  ✗ More memory

More tables (L):
  ✓ Higher recall (more chances to find neighbors)
  ✗ Slower search (more tables to probe)
  ✗ More memory

Typical settings:
  K = 10-20 bits per table
  L = 10-50 tables
  Total: 100-1000 bits per vector
```

**Probability Analysis**:

For cosine similarity θ between vectors:
```
P(collision in 1 table) = (1 - θ/π)^K
P(collision in ≥1 of L tables) = 1 - (1 - (1 - θ/π)^K)^L
```

### LSH in FAISS

FAISS implements optimized LSH with random rotations:

```python
# Binary LSH index
nbits = 2 * d  # Typically 2x dimension
index = faiss.IndexLSH(d, nbits)

# Train with random rotation (better than random projection)
index.train(training_vectors)

# Add and search
index.add(database_vectors)
distances, indices = index.search(queries, k)
```

From [FAISS LSH Implementation](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes):
> "The algorithm is not vanilla-LSH, but a better choice. Instead a set of orthogonal projectors is used if n_bits <= d, or a tight frame if n_bits > d."

## Section 4: FAISS Index Types (80 lines)

### Index Comparison Matrix

From [FAISS Index Guidelines](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index):

| Index Type | Search Speed | Memory | Recall | Best For |
|------------|-------------|---------|---------|----------|
| **IndexFlatL2** | Slow (exact) | High | 100% | Baselines, <1M vectors |
| **IndexIVFFlat** | Medium | High | 95-99% | 1M-10M vectors |
| **IndexIVFPQ** | Fast | Low | 90-95% | 10M-1B vectors |
| **IndexHNSWFlat** | Very Fast | Very High | 95-99% | <100M, low latency |
| **IndexHNSWSQ** | Very Fast | Medium | 90-95% | <100M, balanced |
| **IndexLSH** | Fast | Medium | 80-90% | Binary codes, sketching |

### Flat Indexes (Exact Search)

**IndexFlatL2**: Brute force L2 distance
```python
index = faiss.IndexFlatL2(d)
index.add(vectors)
D, I = index.search(queries, k)  # Guarantees exact results
```

**IndexFlatIP**: Inner product (for cosine with normalized vectors)
```python
# Normalize vectors for cosine similarity
faiss.normalize_L2(vectors)
index = faiss.IndexFlatIP(d)
index.add(vectors)
```

### IVF Indexes

**IndexIVFFlat**: IVF with no compression
```python
nlist = 100  # Number of clusters
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)

index.train(training_vectors)
index.add(database_vectors)
index.nprobe = 10  # Search 10 clusters
D, I = index.search(queries, k)
```

**IndexIVFPQ**: IVF + Product Quantization
```python
nlist = 100
M = 8       # Number of subquantizers
nbits = 8   # Bits per subquantizer (256 centroids)

index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
```

From [Comparing FAISS Indexes](https://medium.com/@saidines12/comparing-different-indexes-for-rag-62ecd798d5ac) (Pola, 2024):

```
IndexFlatL2:
  - Search time: 1000ms
  - Memory: 100%
  - Recall: 100%

IndexIVFFlat (nprobe=10):
  - Search time: 100ms (10x faster)
  - Memory: 100%
  - Recall: 95%

IndexIVFPQ (M=8, nprobe=10):
  - Search time: 50ms (20x faster)
  - Memory: 12.5% (8x compression)
  - Recall: 90%
```

### HNSW Indexes

**IndexHNSWFlat**: HNSW with no compression
```python
M = 32  # Connections per vertex
index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efConstruction = 200  # Build quality
index.add(vectors)

index.hnsw.efSearch = 100  # Search quality
D, I = index.search(queries, k)
```

**IndexHNSWSQ**: HNSW + Scalar Quantization
```python
M = 32
index = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, M)
```

**IndexHNSWPQ**: HNSW + Product Quantization
```python
M = 32
M_pq = 8
index = faiss.IndexHNSWPQ(d, M_pq, M)
```

### Composite Indexes

**Index Factory** for complex configurations:

```python
# IVF with 4096 clusters + PQ with 64 bytes
index = faiss.index_factory(d, "IVF4096,PQ64")

# HNSW with 32 links + Flat
index = faiss.index_factory(d, "HNSW32,Flat")

# IVF + PQ + refinement
index = faiss.index_factory(d, "IVF1024,PQ32+16")
```

From [FAISS Index Factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory):
> "The index factory is a way to create Faiss indexes with a simple string description. It is particularly useful for composite indexes."

### Performance Benchmarks

From [ChromaDB vs Pinecone vs FAISS Benchmarks](https://pub.towardsai.net/vector-databases-performance-comparison-chromadb-vs-pinecone-vs-faiss-real-benchmarks-that-will-3eb83027c584) (Towards AI, 2025):

**1M Vectors, 768 Dimensions:**
```
IndexFlatL2:
  QPS: 10
  Recall@10: 100%
  Memory: 3.1 GB

IndexIVFFlat (nlist=1000, nprobe=10):
  QPS: 500
  Recall@10: 96%
  Memory: 3.1 GB

IndexIVFPQ (nlist=1000, M=64, nprobe=10):
  QPS: 2000
  Recall@10: 92%
  Memory: 0.4 GB

IndexHNSWFlat (M=32):
  QPS: 5000
  Recall@10: 98%
  Memory: 5.2 GB
```

## Section 5: Performance Benchmarks (40 lines)

### Throughput vs Recall

From [ANN Benchmarks](https://github.com/erikbern/ann-benchmarks) (Bernhardsson):

**SIFT1M Dataset (1M vectors, 128 dimensions, 10-NN)**:

```
Algorithm         QPS @ 90% Recall    QPS @ 95% Recall    QPS @ 99% Recall
HNSW (M=16)            15,000              10,000               5,000
HNSW (M=32)            12,000               8,000               3,500
IVF (nprobe=10)         8,000               4,000               1,000
IVF (nprobe=50)         3,000               2,000                 800
LSH (L=20)              5,000               2,000                 500

Memory:
HNSW (M=16):     850 MB
HNSW (M=32):    1200 MB
IVF:             520 MB
LSH:             340 MB
```

### Latency Distribution

**P50/P95/P99 Latencies** (DEEP1B dataset, 1B vectors):

```
IndexIVFPQ (nlist=262144, nprobe=32, M=64):
  P50: 2.1ms
  P95: 4.8ms
  P99: 12.3ms
  Memory: 68 GB
  Recall@10: 92%

IndexHNSWFlat (M=32, efSearch=64):
  P50: 1.2ms
  P95: 2.8ms
  P99: 6.5ms
  Memory: 512 GB (not practical for 1B)
  Recall@10: 96%
```

From [Indexing 1B Vectors](https://github.com/facebookresearch/faiss/wiki/Indexing-1G-vectors) (FAISS Wiki):
> "For 1 billion vectors, IndexIVFPQ is the most practical choice, offering good recall with manageable memory usage."

### Real-World Performance

**Production System Examples**:

```
Pinterest (2B+ vectors, image embeddings):
  - Algorithm: IVF + PQ
  - QPS: 10,000+ per machine
  - Latency P99: < 10ms
  - Recall@20: 94%

Spotify (billions of track/user vectors):
  - Algorithm: HNSW
  - Latency P50: < 5ms
  - Recall@50: 95%+

Google (trillion+ scale):
  - Algorithm: ScaNN (similar to IVF+PQ)
  - Custom optimizations for TPU
```

### Cost-Performance Trade-offs

**Memory vs Speed** (1M vectors @ 768 dims):

```
Configuration                 Memory    QPS    Recall@10
IndexFlatL2                   3.1 GB      10      100%
IndexIVFFlat (optimized)      3.1 GB     500       96%
IndexIVFPQ (M=64)            0.4 GB    2000       92%
IndexIVFPQ (M=32)            0.2 GB    3500       88%
IndexHNSWFlat (M=32)         5.2 GB    5000       98%
IndexHNSWSQ (M=32)           3.5 GB    4500       95%

Cost per 1M queries (AWS c5.4xlarge):
IndexFlatL2:      $50   (slow, needs many machines)
IndexIVFPQ:       $2    (fast, cheap)
IndexHNSW:        $3    (fastest, more RAM cost)
```

## Sources

**Research Papers:**
- [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) - Malkov & Yashunin, 2016 (accessed 2025-01-31)
- [Similarity Search in High Dimensions via Hashing](https://www.cs.princeton.edu/courses/archive/spring13/cos598C/Gionis.pdf) - Gionis et al., STOC 2002 (accessed 2025-01-31)

**Technical Documentation:**
- [FAISS Indexes Wiki](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) - Facebook Research (accessed 2025-01-31)
- [Guidelines to Choose an Index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index) - FAISS Wiki (accessed 2025-01-31)
- [Hierarchical Navigable Small Worlds (HNSW)](https://www.pinecone.io/learn/series/faiss/hnsw/) - Pinecone (accessed 2025-01-31)
- [Locality Sensitive Hashing: The Illustrated Guide](https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/) - Pinecone (accessed 2025-01-31)

**Technical Articles:**
- [Inverted File Indexing (IVF) in FAISS: A Comprehensive Guide](https://medium.com/@Jawabreh0/inverted-file-indexing-ivf-in-faiss-a-comprehensive-guide-c183fe979d20) - Jawabreh, Medium 2024 (accessed 2025-01-31)
- [Understanding Hierarchical Navigable Small World (HNSW)](https://zilliz.com/learn/hierarchical-navigable-small-worlds-HNSW) - Zilliz, 2024 (accessed 2025-01-31)
- [Similarity Search, Part 1: kNN & Inverted File Index](https://towardsdatascience.com/similarity-search-knn-inverted-file-index-7cab80cc0e79) - Towards Data Science (accessed 2025-01-31)
- [Comparing Different Indexes for RAG](https://medium.com/@saidines12/comparing-different-indexes-for-rag-62ecd798d5ac) - Pola, Medium 2024 (accessed 2025-01-31)

**Benchmarks & Comparisons:**
- [ChromaDB vs Pinecone vs FAISS - Real Benchmarks](https://pub.towardsai.net/vector-databases-performance-comparison-chromadb-vs-pinecone-vs-faiss-real-benchmarks-that-will-3eb83027c584) - Towards AI, 2025 (accessed 2025-01-31)
- [ANN Benchmarks](https://github.com/erikbern/ann-benchmarks) - Erik Bernhardsson (accessed 2025-01-31)

**Implementation Resources:**
- [Understanding HNSW for Vector Search](https://milvus.io/blog/understand-hierarchical-navigable-small-worlds-hnsw-for-vector-search.md) - Milvus, 2025 (accessed 2025-01-31)
- [How HNSW Algorithms Can Improve Search](https://redis.io/blog/how-hnsw-algorithms-can-improve-search/) - Redis, 2025 (accessed 2025-01-31)
- [Approximate Nearest Neighbor with LSH](https://pyimagesearch.com/2025/01/27/approximate-nearest-neighbor-with-locality-sensitive-hashing-lsh/) - PyImageSearch, 2025 (accessed 2025-01-31)
