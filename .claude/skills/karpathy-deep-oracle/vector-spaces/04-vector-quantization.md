# Vector Quantization for Embeddings

## Overview

Vector quantization is a compression technique that dramatically reduces the memory footprint and storage requirements of high-dimensional embeddings. Unlike dimensionality reduction (which reduces the number of dimensions), quantization reduces the precision of each value in the embedding, transforming continuous float32 values into discrete representations with significantly fewer bits.

For embeddings, quantization is a post-processing step applied after the vectors are generated. The primary goal is to make large-scale vector search feasible by reducing memory usage by up to 32x while maintaining reasonable retrieval accuracy.

**Why Quantization Matters:**
- High-dimensional vectors require massive memory (1M vectors at 128D float32 = several GB)
- Memory costs scale linearly with dataset size
- Search speed depends on how quickly vectors can be loaded and compared
- Production deployments often need to handle billions of vectors

From [Pinecone - Product Quantization](https://www.pinecone.io/learn/series/faiss/product-quantization/) (accessed 2025-02-02):
> Vector similarity search can require huge amounts of memory. Indexes containing 1M dense vectors (a small dataset in today's world) will often require several GBs of memory to store. Product quantization (PQ) is a popular method for dramatically compressing high-dimensional vectors to use 97% less memory.

## Quantization vs Dimensionality Reduction

It's crucial to understand the difference:

**Dimensionality Reduction:**
- Reduces D (number of dimensions): 128D → 64D
- Each value remains float32 (4 bytes)
- Example: PCA, UMAP, Matryoshka Representation Learning

**Quantization:**
- Keeps D constant: 128D → 128D
- Reduces S (scope/precision of values): float32 → int8 or binary
- Transforms continuous space into discrete symbolic representations

From [HuggingFace - Embedding Quantization](https://huggingface.co/blog/embedding-quantization) (accessed 2025-02-02):
> Quantization for embeddings refers to a post-processing step for the embeddings themselves. In particular, binary quantization refers to the conversion of the float32 values in an embedding to 1-bit values, resulting in a 32x reduction in memory and storage usage.

---

## Scalar Quantization (int8/uint8)

### How It Works

Scalar quantization maps the continuous range of float32 values (-∞ to +∞) to a discrete set of integer values, typically int8 (-128 to 127) or uint8 (0 to 255). This provides 256 distinct quantization levels.

**Process:**
1. **Calibration:** Analyze a large calibration dataset to determine min/max ranges for each embedding dimension
2. **Bucket Creation:** Divide the range into 256 equal buckets
3. **Mapping:** Map each float32 value to its nearest int8 bucket
4. **Storage:** Store the quantized int8 values plus the calibration parameters (min/max ranges)

**Mathematical Formula:**
```
quantized_value = round((value - min) / (max - min) * 255)
```

**Compression Ratio:**
- Original: 128D × 32 bits = 4,096 bits
- Quantized: 128D × 8 bits = 1,024 bits
- **4x reduction in memory**

### Implementation

From [HuggingFace - Embedding Quantization](https://huggingface.co/blog/embedding-quantization) (accessed 2025-02-02):

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
from datasets import load_dataset

# 1. Load model
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# 2. Prepare calibration dataset (crucial for quality)
corpus = load_dataset("nq_open", split="train[:1000]")["question"]
calibration_embeddings = model.encode(corpus)

# 3. Encode and quantize
embeddings = model.encode(["I am driving to the lake.", "It is a beautiful day."])
int8_embeddings = quantize_embeddings(
    embeddings,
    precision="int8",
    calibration_embeddings=calibration_embeddings,
)

# Result:
# embeddings.shape: (2, 1024), nbytes: 8192, dtype: float32
# int8_embeddings.shape: (2, 1024), nbytes: 2048, dtype: int8
```

### Calibration Dataset Importance

The calibration dataset critically influences performance because it defines the quantization buckets. Best practices:

- **Size:** Use thousands of embeddings (not just a handful)
- **Diversity:** Cover the expected distribution of your production data
- **Quality:** Representative samples from your actual use case

Without proper calibration, you'll see warnings like:
> Computing int8 quantization buckets based on 2 embeddings. int8 quantization is more stable with 'ranges' calculated from more embeddings or a 'calibration_embeddings' that can be used to calculate the buckets.

### Accuracy vs Memory Trade-off

From [HuggingFace - Embedding Quantization](https://huggingface.co/blog/embedding-quantization) (accessed 2025-02-02):

**Performance Retention on MTEB Retrieval (NDCG@10):**

| Model | float32 | int8 | Retention |
|-------|---------|------|-----------|
| mxbai-embed-large-v1 (1024D) | 54.39 | 52.79 | **97%** |
| Cohere-embed-v3 (1024D) | 55.0 | 55.0 | **100%** |
| all-MiniLM-L6-v2 (384D) | 41.66 | 37.82 | 90.79% |

With rescoring (retrieve 4x more candidates, then rescore with float32 query):
- **mxbai-embed-large-v1:** 97% → **99.3%** retention
- Rescore multiplier of 4-5 is optimal

### Speed Benefits

From [HuggingFace - Embedding Quantization](https://huggingface.co/blog/embedding-quantization) (accessed 2025-02-02):

Tested on GCP `a2-highgpu-4g` with USearch (CPU, exact search):

| Quantization | Min | Mean | Max |
|--------------|-----|------|-----|
| float32 | 1x | 1x | 1x |
| int8 | 2.99x | **3.66x** | 4.8x |

**Why int8 is Faster:**
- Smaller memory footprint = better cache utilization
- Modern CPUs have optimized int8 SIMD instructions
- Reduced memory bandwidth requirements

---

## Binary Quantization

### How It Works

Binary quantization is the most extreme form of compression, converting each float32 value to a single bit (0 or 1). This provides a 32x reduction in memory usage.

**Process:**
1. **Normalize:** Ensure embeddings are L2-normalized (unit length)
2. **Threshold:** Apply simple threshold at zero
3. **Pack:** Pack 8 bits into bytes for efficient storage

**Thresholding Function:**
```
f(x) = {
    0  if x ≤ 0
    1  if x > 0
}
```

**Compression Ratio:**
- Original: 1024D × 32 bits = 32,768 bits
- Quantized: 1024D × 1 bit = 1,024 bits = 128 bytes (int8/uint8)
- **32x reduction in memory**

From [Pinecone - Product Quantization](https://www.pinecone.io/learn/series/faiss/product-quantization/) (accessed 2025-02-02):
> In practice, it is much more common to store bits as bytes instead, so when we quantize to binary embeddings, we pack the bits into bytes using `np.packbits`. Therefore, quantizing a float32 embedding with a dimensionality of 1024 yields an int8 or uint8 embedding with a dimensionality of 128.

### Distance Computation

Binary embeddings use **Hamming Distance** instead of cosine similarity or Euclidean distance:

```
Hamming Distance = number of bit positions where embeddings differ
```

**Advantages:**
- Computed with 2 CPU cycles using XOR + POPCOUNT instructions
- Extremely fast: up to 45x faster than float32 similarity
- Hardware-accelerated on modern CPUs

### Implementation

From [HuggingFace - Embedding Quantization](https://huggingface.co/blog/embedding-quantization) (accessed 2025-02-02):

```python
from sentence_transformers import SentenceTransformer

# Option 1: Quantize during encoding
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
binary_embeddings = model.encode(
    ["I am driving to the lake.", "It is a beautiful day."],
    precision="binary",  # or "ubinary" for uint8
)

# Option 2: Quantize after encoding
from sentence_transformers.quantization import quantize_embeddings
embeddings = model.encode(["I am driving to the lake.", "It is a beautiful day."])
binary_embeddings = quantize_embeddings(embeddings, precision="binary")

# Result:
# embeddings.shape: (2, 1024), nbytes: 8192, dtype: float32
# binary_embeddings.shape: (2, 128), nbytes: 256, dtype: int8
```

### Rescoring for Accuracy

Binary quantization can lose significant accuracy (74-87% retention). The solution is **rescoring**:

From [Pinecone - Product Quantization](https://www.pinecone.io/learn/series/faiss/product-quantization/) (accessed 2025-02-02):
> [Yamada et al. (2021)](https://arxiv.org/abs/2106.00882) introduced a rescore step, which they called _rerank_, to boost the performance. They proposed that the float32 query embedding could be compared with the binary document embeddings using dot-product.

**Rescoring Process:**
1. Retrieve `rescore_multiplier × top_k` candidates using binary embeddings (fast!)
2. Compare float32 query with those binary candidates using dot-product
3. Re-rank and return final top_k results

**Example:** To get top 10 results:
- Retrieve top 40 with binary (rescore_multiplier=4)
- Rescore those 40 with float32 query
- Return top 10 after reranking

### Accuracy with Rescoring

From [HuggingFace - Embedding Quantization](https://huggingface.co/blog/embedding-quantization) (accessed 2025-02-02):

**Performance Retention on MTEB Retrieval (NDCG@10):**

| Model | float32 | binary | binary + rescore | Retention |
|-------|---------|--------|------------------|-----------|
| mxbai-embed-large-v1 (1024D) | 54.39 | 50.35 (92.5%) | 52.46 | **96.45%** |
| nomic-embed-text-v1.5 (768D) | 53.01 | - | 46.49 | 87.7% |
| Cohere-embed-v3 (1024D) | 55.0 | - | 52.3 | 94.6% |

Rescoring boosts performance from 92.5% → 96.45% for mxbai-embed-large-v1.

### Speed Benefits

From [HuggingFace - Embedding Quantization](https://huggingface.co/blog/embedding-quantization) (accessed 2025-02-02):

Tested on GCP `a2-highgpu-4g` with FAISS (CPU, exact search):

| Quantization | Min | Mean | Max |
|--------------|-----|------|-----|
| float32 | 1x | 1x | 1x |
| binary | 15.05x | **24.76x** | 45.8x |

Even with rescoring overhead, binary is dramatically faster than float32.

### When Binary Works Best

From [HuggingFace - Embedding Quantization](https://huggingface.co/blog/embedding-quantization) (accessed 2025-02-02):

**Binary quantization doesn't universally work with all embedding models:**

✓ **Works well:**
- mxbai-embed-large-v1: 96.45% retention
- all-MiniLM-L6-v2: 93.79% retention
- Cohere-embed-v3: 94.6% retention

✗ **Works poorly:**
- e5-base-v2: 74.77% retention (dimension collapse issue)

Some models suffer from **dimension collapse** where only a subspace of the latent space is used. Binary quantization further collapses this space, leading to high performance losses.

---

## Product Quantization (PQ)

### How It Works

Product quantization is a sophisticated technique that combines **vector splitting** with **clustering** to achieve high compression ratios while maintaining good accuracy.

From [Pinecone - Product Quantization](https://www.pinecone.io/learn/series/faiss/product-quantization/) (accessed 2025-02-02):

**Process:**
1. **Split:** Divide the high-dimensional vector into `m` equal subvectors
2. **Cluster:** Train separate k-means clusterers for each subspace (each with k* centroids)
3. **Assign:** Assign each subvector to its nearest centroid in that subspace
4. **Encode:** Replace centroid values with unique IDs (codebook)
5. **Store:** Store only the compact vector of IDs

**Example:**
```
Original vector: [1, 8, 3, 9, 1, 2, 9, 4, 5, 4, 6, 2]  (12D)

Step 1 - Split into m=4 subvectors:
u₁ = [1, 8, 3]
u₂ = [9, 1, 2]
u₃ = [9, 4, 5]
u₄ = [4, 6, 2]

Step 2 - Assign to nearest centroids:
u₁ → centroid[1]
u₂ → centroid[1]
u₃ → centroid[2]
u₄ → centroid[1]

Step 3 - Store IDs:
Quantized: [1, 1, 2, 1]  (4D of IDs)
```

### Memory Complexity

From [Pinecone - Product Quantization](https://www.pinecone.io/learn/series/faiss/product-quantization/) (accessed 2025-02-02):

**Comparison of memory usage:**

| Method | Complexity | Example (k=2048, m=8, D=128) |
|--------|------------|------------------------------|
| k-means | k × D | 2048 × 128 = 262,144 |
| Product Quantization | k^(1/m) × D | 2048^(1/8) × 128 = 332 |

PQ achieves **789x lower memory complexity** in this example!

**Why this matters:**
- k=2048 recommended for good quality
- Without PQ: Need dataset several times larger than 262k for training
- With PQ: Only need several multiples of 332 for training
- Training is feasible with much smaller datasets

### Compression Ratio

**Realistic Example:**
- Original: 128D × 32 bits = 4,096 bits
- PQ (m=8, nbits=8): 8 IDs × 8 bits = 64 bits
- **64x reduction in memory**

But wait, we need to store the codebook:
- Codebook size: m × k* × D* × 32 bits
- With m=8, k*=256, D*=16: 8 × 256 × 16 × 32 = 1,048,576 bits
- Amortized across millions of vectors, codebook overhead is negligible

### FAISS Implementation

From [Pinecone - Product Quantization](https://www.pinecone.io/learn/series/faiss/product-quantization/) (accessed 2025-02-02):

```python
import faiss

# Initialize IndexPQ
D = 128  # vector dimensionality
m = 8    # number of subvectors
nbits = 8  # bits per subquantizer (k* = 2**nbits = 256)

index = faiss.IndexPQ(D, m, nbits)

# Train on dataset (PQ requires training!)
index.train(xb)  # xb = training vectors

# Add vectors
index.add(xb)

# Search
distances, indices = index.search(xq, k=10)  # xq = query vectors
```

**Key Parameters:**
- `D`: Must be divisible by `m`
- `m`: Number of subquantizers (typically 8 or 16)
- `nbits`: Bits per subquantizer (typically 8, giving k*=256 centroids)
- Higher nbits = better accuracy but slower and more memory

### PQ Performance

From [Pinecone - Product Quantization](https://www.pinecone.io/learn/series/faiss/product-quantization/) (accessed 2025-02-02):

Tested on Sift1M dataset with mxbai-embed-large-v1:

| Index Type | Memory (MB) | Search Time (ms) | Recall (%) |
|------------|-------------|------------------|------------|
| FlatL2 | 256 | 8.26 | 100 |
| IndexPQ | 6.5 | 1.49 | 50 |
| IndexIVFPQ | 9.2 | 0.09 | 52 |

**IndexIVFPQ (IVF + PQ combined):**
- 96% memory reduction vs FlatL2
- 92x faster search
- 52% recall (reasonable for many use cases)

### Limitations

**Recall Trade-off:**
- PQ sacrifices accuracy for compression
- 50-60% recall is typical
- Cannot achieve very high recall (>90%)
- If high recall required, consider HNSW or other methods

**Training Requirements:**
- Requires representative training dataset
- Training can be slow with high nbits (>11)
- Training time increases exponentially with nbits

---

## Hybrid Approach: Binary + Scalar Quantization

### The Best of Both Worlds

From [HuggingFace - Embedding Quantization](https://huggingface.co/blog/embedding-quantization) (accessed 2025-02-02):

Combining binary and scalar quantization achieves extreme speed from binary embeddings with great accuracy from scalar rescoring:

**Pipeline:**
1. **Binary Index (in-memory):** Store all vectors as binary (32x compression)
2. **Scalar Index (on-disk):** Store all vectors as int8 (4x compression)
3. **Binary Search:** Retrieve top `rescore_multiplier × top_k` candidates using binary (ultra-fast!)
4. **Load from Disk:** Load those candidates from int8 index (on-the-fly)
5. **Scalar Rescore:** Rescore with float32 query vs int8 candidates
6. **Return:** Final top_k results

**Memory vs Disk:**
- Binary index: 5.2 GB in memory (41M vectors, 1024D)
- Int8 index: 47.5 GB on disk (41M vectors, 1024D)
- **Total memory: 5.2 GB (vs 200 GB for float32)**
- Total disk: 52 GB (vs 200 GB for float32)

### Real-World Example: 41M Wikipedia

From [HuggingFace - Embedding Quantization](https://huggingface.co/blog/embedding-quantization) (accessed 2025-02-02):

Demo at: https://huggingface.co/spaces/sentence-transformers/quantized-retrieval

**Setup:**
- Dataset: 41 million Wikipedia texts
- Model: mixedbread-ai/mxbai-embed-large-v1 (1024D)
- Binary index: 5.2 GB memory
- Int8 index: 47.5 GB disk (memory-mapped)

**Performance:**
- Search latency: <100ms for top 10 results
- Accuracy: ~96% of float32 retrieval
- Cost: ~$20/month (vs ~$800/month for float32)

---

## Quantization Support in Vector Databases

### Binary Quantization Support

From [HuggingFace - Embedding Quantization](https://huggingface.co/blog/embedding-quantization) (accessed 2025-02-02):

| Database | Binary Support |
|----------|----------------|
| **Faiss** | ✅ [Binary indexes](https://github.com/facebookresearch/faiss/wiki/Binary-indexes) |
| **USearch** | ✅ Full support |
| **Vespa AI** | ✅ Native support |
| **Milvus** | ✅ Binary metrics |
| **Qdrant** | ✅ [Binary Quantization](https://qdrant.tech/documentation/guides/quantization/#binary-quantization) |
| **Weaviate** | ✅ [Binary Quantization](https://weaviate.io/developers/weaviate/configuration/bq-compression) |

### Scalar (int8) Quantization Support

From [HuggingFace - Embedding Quantization](https://huggingface.co/blog/embedding-quantization) (accessed 2025-02-02):

| Database | int8 Support |
|----------|--------------|
| **Faiss** | ⚠️ Indirectly via [IndexHNSWSQ](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexHNSWSQ.html) |
| **USearch** | ✅ Full support |
| **Vespa AI** | ✅ Native int8 tensors |
| **OpenSearch** | ✅ [Scalar quantization](https://opensearch.org/docs/latest/field-types/supported-field-types/knn-vector) |
| **ElasticSearch** | ✅ [Byte-sized vectors](https://www.elastic.co/de/blog/save-space-with-byte-sized-vectors) |
| **Milvus** | ⚠️ Indirectly via [IVF_SQ8](https://milvus.io/docs/index.md) |
| **Qdrant** | ⚠️ Indirectly via [Scalar Quantization](https://qdrant.tech/documentation/guides/quantization/#scalar-quantization) |

### Product Quantization Support

From [Pinecone - Product Quantization](https://www.pinecone.io/learn/series/faiss/product-quantization/) (accessed 2025-02-02):

| Database | PQ Support |
|----------|------------|
| **Faiss** | ✅ IndexPQ, IndexIVFPQ |
| **Weaviate** | ✅ [PQ Compression](https://docs.weaviate.io/weaviate/configuration/compression/pq-compression) |
| **Milvus** | ✅ [PQ indexing](https://milvus.io/docs/index.md) |
| **OpenSearch** | ✅ [FAISS PQ integration](https://docs.opensearch.org/latest/vector-search/optimizing-storage/faiss-product-quantization/) |
| **MongoDB Atlas** | ✅ [Vector quantization](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-quantization/) |

---

## Choosing the Right Quantization Method

### Decision Matrix

| Criteria | Binary | int8 | Product Quantization |
|----------|--------|------|----------------------|
| **Memory Reduction** | 32x | 4x | 30-60x |
| **Speed Improvement** | 15-45x | 3-5x | 5-15x |
| **Accuracy Retention** | 87-96% | 97-100% | 50-60% |
| **Training Required** | No | Calibration dataset | Yes (clustering) |
| **Best For** | High-scale retrieval | Production balance | Extreme scale |

### Recommendations

**Choose Binary When:**
- ✅ Scale is massive (billions of vectors)
- ✅ Speed is critical (real-time search)
- ✅ You can tolerate 4-13% accuracy loss
- ✅ You have headroom for rescoring latency
- ✅ Your model works well with binary (test first!)

**Choose int8 When:**
- ✅ You need near-perfect accuracy (97-100%)
- ✅ You want simple implementation
- ✅ You have a good calibration dataset
- ✅ Moderate scale (millions to billions)
- ✅ Production deployment balance

**Choose Product Quantization When:**
- ✅ Extreme memory constraints
- ✅ You can sacrifice recall (50-60% acceptable)
- ✅ You have training data and compute
- ✅ You can combine with IVF for speed (IVFPQ)
- ✅ Scale is huge (billions of vectors)

**Hybrid Approach (Binary + int8):**
- ✅ Best overall balance
- ✅ Speed of binary + accuracy of int8
- ✅ Reasonable memory usage (binary in-memory, int8 on disk)
- ✅ Production-ready for most use cases

---

## Performance Summary

From [HuggingFace - Embedding Quantization](https://huggingface.co/blog/embedding-quantization) (accessed 2025-02-02):

### Memory & Storage

| Quantization | Memory Reduction | Example (250M vectors, 1024D) |
|--------------|------------------|-------------------------------|
| float32 | 1x (baseline) | 953.67 GB ($3,623/mo) |
| int8 | 4x | 238.41 GB ($905/mo) |
| binary | 32x | 29.80 GB ($113/mo) |
| PQ (m=8, nbits=8) | ~30-60x | ~20-30 GB |

### Speed

| Quantization | Min | Mean | Max |
|--------------|-----|------|-----|
| float32 | 1x | 1x | 1x |
| int8 | 2.99x | 3.66x | 4.8x |
| binary | 15.05x | 24.76x | 45.8x |
| PQ | ~5x | ~10x | ~15x |

### Accuracy (MTEB Retrieval NDCG@10)

| Quantization | Accuracy Retention |
|--------------|-------------------|
| float32 | 100% (baseline) |
| int8 | ~99.3% |
| int8 (no rescore) | ~97% |
| binary + rescore | ~96% |
| binary (no rescore) | ~92.5% |
| PQ (IndexPQ) | ~50-60% |
| PQ (IndexIVFPQ) | ~52% |

---

## Future Directions

From [HuggingFace - Embedding Quantization](https://huggingface.co/blog/embedding-quantization) (accessed 2025-02-02):

### Sub-int8 Quantization

**Potential:** 4-bit or even 2-bit scalar quantization
- int4: 128 or 64 buckets instead of 256
- **8x compression** instead of 4x
- Trade-off: Lower accuracy, needs research

### Matryoshka + Quantization

**Perpendicular optimizations:**
1. MRL: Shrink embeddings (1024D → 128D) = 8x reduction, ~2% performance loss
2. Binary: Compress values (float32 → binary) = 32x reduction, ~4% performance loss
3. **Combined:** 8 × 32 = **256x compression**, ~10% performance loss

**Example:**
- 1024D float32 → 128D binary
- 256x faster retrieval at ~10% accuracy cost
- Extremely promising for massive scale

### 3-Stage Retrieval Pipeline

**Proposed architecture:**
1. **Stage 1:** Binary search (ultra-fast, broad recall)
2. **Stage 2:** int8 rescoring (fast, high precision)
3. **Stage 3:** Cross-encoder reranking (slow but perfect)

**Benefits:**
- State-of-the-art accuracy
- Low latency (most work done in stages 1-2)
- Low memory/disk usage
- Low costs

---

## Sources

**Web Research:**
- [Pinecone - Product Quantization](https://www.pinecone.io/learn/series/faiss/product-quantization/) (accessed 2025-02-02)
- [HuggingFace - Binary and Scalar Embedding Quantization](https://huggingface.co/blog/embedding-quantization) (accessed 2025-02-02)
- [Towards Data Science - Similarity Search, Part 2: Product Quantization](https://towardsdatascience.com/similarity-search-product-quantization-b2a1a6397701/) (accessed 2025-02-02)
- [Qdrant - What is Vector Quantization?](https://qdrant.tech/articles/what-is-vector-quantization/) (accessed 2025-02-02)
- [Weaviate - Product Quantization (PQ)](https://docs.weaviate.io/weaviate/configuration/compression/pq-compression) (accessed 2025-02-02)
- [Microsoft Learn - Product Quantization in Azure Cosmos DB](https://learn.microsoft.com/en-us/azure/cosmos-db/mongodb/vcore/product-quantization) (accessed 2025-02-02)

**Additional References:**
- [Yamada et al. (2021) - Binary Embeddings with Rescoring](https://arxiv.org/abs/2106.00882)
- [Jégou et al. (2010) - Product Quantization for Nearest Neighbor Search](https://www.researchgate.net/publication/47815472_Product_Quantization_for_Nearest_Neighbor_Search)
- [FAISS Documentation - Binary Indexes](https://github.com/facebookresearch/faiss/wiki/Binary-indexes)
- [Sentence Transformers - Embedding Quantization](https://sbert.net/examples/applications/embedding-quantization/README.html)
