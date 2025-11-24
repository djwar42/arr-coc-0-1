# Vertex AI Matching Engine: Vector Search at Scale

## Overview

**Vertex AI Matching Engine** (also called **Vector Search**) is Google Cloud's managed vector similarity search service, built on Google Research's ScaNN (Scalable Nearest Neighbors) algorithm. It enables billion-scale approximate nearest neighbor (ANN) search with sub-50ms latencies, making it ideal for RAG systems, recommendation engines, visual search, and multimodal AI applications.

**Key Question**: How do you search billions of high-dimensional vectors in milliseconds without managing infrastructure?

**Answer**: Vertex AI Matching Engine provides fully-managed, horizontally-scalable vector search with built-in ScaNN optimization, streaming updates, and automatic index management.

From [Vector Search Overview](https://docs.cloud.google.com/vertex-ai/docs/vector-search/overview) (accessed 2025-11-16):
> "Vector Search is a powerful vector search engine built on groundbreaking technology developed by Google Research. Leveraging the ScaNN algorithm, Vector Search lets you build next-generation search and recommendation systems as well as generative AI applications."

## Core Concepts

### What is Vector Search?

**Vector search** finds semantically similar items by measuring distance between high-dimensional embeddings. Unlike keyword search (exact matches), vector search captures semantic meaning.

**Example Use Cases**:
- **RAG Systems**: Find relevant context documents for LLM queries
- **Visual Search**: "Find images similar to this product photo"
- **Recommendation**: "Users who liked X also liked Y" (embedding-based)
- **Anomaly Detection**: Find outliers in embedding space
- **Multimodal Search**: Text query → image results (CLIP embeddings)

**How It Works**:
1. **Embedding Generation**: Convert data (text/images/audio) to vectors (e.g., 768D CLIP embeddings)
2. **Index Creation**: Build ScaNN index optimized for ANN search
3. **Index Deployment**: Deploy index with autoscaling endpoints
4. **Query**: Send query vector → retrieve k-nearest neighbors
5. **Stream Updates**: Add/update embeddings without full rebuild

### ScaNN Algorithm

**ScaNN** (Scalable Nearest Neighbors) is Google's vector search algorithm, open-sourced in 2020.

From [ScaNN Research](https://research.google/blog/soar-new-algorithms-for-even-faster-vector-search-with-scann/) (accessed 2025-11-16):
> "SOAR is an algorithmic improvement to vector search that introduces effective and low-overhead redundancy to ScaNN, Google's vector search library, making it significantly faster."

**Key Features**:
- **Quantization**: Compress vectors (float32 → int8) for 4x memory savings
- **Anisotropic quantization**: Per-dimension scaling for better accuracy
- **Tree-based partitioning**: Hierarchical clustering for fast pruning
- **SOAR optimization**: Redundant scoring paths for speed/recall tradeoff

**ScaNN vs Other Algorithms**:

| Algorithm | Recall@10 | Latency (ms) | Memory | Use Case |
|-----------|-----------|--------------|--------|----------|
| Brute Force | 100% | 1000+ | High | <10K vectors |
| Tree-AH | 95% | 50-100 | Medium | Balanced |
| ScaNN | 95%+ | 10-50 | Low | Production (billions) |
| IVF-PQ | 90% | 20-80 | Very Low | Large scale |

From [AlloyDB ScaNN Index](https://cloud.google.com/blog/products/databases/understanding-the-scann-index-in-alloydb) (accessed 2025-11-16):
> "The ScaNN index is a tree-based quantization index for approximate nearest neighbor search. In Tree-quantization techniques, indexes learn a hierarchical structure."

## Index Creation

### Index Types

**1. Tree-AH (Tree and Asymmetric Hashing)**

**Best for**: Balanced recall/latency tradeoffs

**How it works**:
- Hierarchical k-means tree partitioning
- Asymmetric hashing for distance computation
- Query navigates tree → searches leaf partition

**Configuration**:
```python
from google.cloud import aiplatform

index_config = {
    "treeAhConfig": {
        "leafNodeEmbeddingCount": 10000,  # Embeddings per leaf
        "leafNodesToSearchPercent": 10    # Pruning aggressiveness
    },
    "dimensions": 768,
    "approximateNeighborsCount": 150,
    "distanceMeasureType": "DOT_PRODUCT_DISTANCE"
}
```

**Tuning Parameters**:
- `leafNodeEmbeddingCount`: Larger = fewer partitions = slower build, faster query
- `leafNodesToSearchPercent`: Higher = better recall, higher latency
- Typical: 10K-50K embeddings per leaf, search 5-20% of leaves

**2. Brute Force**

**Best for**: Small datasets (<10K vectors), exact nearest neighbors required

**How it works**:
- Exhaustive search (computes distance to all vectors)
- No approximation → 100% recall
- No index build time

**Configuration**:
```python
index_config = {
    "bruteForceConfig": {},
    "dimensions": 512,
    "distanceMeasureType": "COSINE_DISTANCE"
}
```

**When to use**:
- Dataset < 10,000 vectors
- Need exact results (no approximation acceptable)
- Prototyping/baseline comparisons

### Distance Metrics

**1. DOT_PRODUCT_DISTANCE**

**Formula**: `distance = -1 * (query · vector)`

**Use when**:
- Embeddings are L2-normalized (unit length)
- Equivalent to cosine similarity for normalized vectors
- Faster than cosine (no normalization overhead)

**Example** (CLIP embeddings):
```python
import numpy as np

# Embeddings from CLIP model
embedding = model.encode_image(image)
# Normalize to unit length
embedding = embedding / np.linalg.norm(embedding)

# Now dot product = cosine similarity
```

**2. COSINE_DISTANCE**

**Formula**: `distance = 1 - cosine_similarity(query, vector)`

**Use when**:
- Embeddings not pre-normalized
- Want angle-based similarity (ignores magnitude)
- Text embeddings (BERT, sentence transformers)

**3. SQUARED_L2_DISTANCE**

**Formula**: `distance = ||query - vector||²`

**Use when**:
- Euclidean distance semantically meaningful
- Clustering applications (K-means)
- Magnitude matters

**Performance Comparison**:
- DOT_PRODUCT: Fastest (single operation)
- COSINE: Slower (normalization + dot product)
- SQUARED_L2: Moderate (subtraction + sum of squares)

### Index Size and Replicas

**Replica Count**:
```python
deployed_index = index_endpoint.deploy_index(
    index=index,
    deployed_index_id="my_index",
    min_replica_count=2,
    max_replica_count=10,
    machine_type="n1-standard-16"
)
```

**Autoscaling Behavior**:
- Scales up: QPS exceeds capacity (>70% CPU utilization)
- Scales down: QPS drops (idle replicas terminated)
- Scaling time: 5-10 minutes per replica
- Minimum replicas: Always running (no cold starts)

**Machine Types**:

| Machine Type | vCPUs | Memory | Max Index Size | Cost/hour |
|--------------|-------|--------|----------------|-----------|
| n1-standard-16 | 16 | 60 GB | ~50M vectors | $0.76 |
| n1-standard-32 | 32 | 120 GB | ~100M vectors | $1.52 |
| n1-highmem-16 | 16 | 104 GB | ~80M vectors | $1.02 |
| n1-highmem-32 | 32 | 208 GB | ~150M vectors | $2.04 |

**Sizing Guidelines**:
- **Index size = (dimensions × 4 bytes × num_vectors) / compression_ratio**
- ScaNN compression: ~4x (float32 → quantized)
- Leave 20-30% memory headroom for query processing

**Example** (100M vectors, 768D):
```
Uncompressed: 768 × 4 × 100M = 307 GB
ScaNN compressed: 307 / 4 = ~77 GB
Machine: n1-highmem-32 (208 GB) → fits comfortably
```

## Embedding Generation

### Vertex AI Embeddings API

**Text Embeddings**:
```python
from vertexai.language_models import TextEmbeddingModel

model = TextEmbeddingModel.from_pretrained("text-embedding-005")

# Generate embeddings
texts = ["Machine learning", "Deep neural networks", "Vector search"]
embeddings = model.get_embeddings(texts)

for text, embedding in zip(texts, embeddings):
    vector = embedding.values  # 768-dim vector
    print(f"{text}: {vector[:5]}...")  # First 5 dimensions
```

**Model Options**:

| Model | Dimensions | Context Length | Languages | Cost per 1K chars |
|-------|------------|----------------|-----------|-------------------|
| text-embedding-005 | 768 | 2048 tokens | 100+ | $0.00001 |
| text-multilingual-embedding-002 | 768 | 2048 | 100+ | $0.00001 |
| textembedding-gecko@003 | 768 | 3072 | English | $0.00001 |

**Multimodal Embeddings**:

From [Multimodal Embeddings API](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-embeddings-api) (accessed 2025-11-16):
> "The Multimodal embeddings API generates vectors based on the input you provide, which can include a combination of image, text, and video data."

```python
from vertexai.vision_models import MultiModalEmbeddingModel

model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

# Image + text embedding
image = Image.load_from_file("product.jpg")
embeddings = model.get_embeddings(
    image=image,
    contextual_text="red sneakers size 10",
    dimension=1408  # Multimodal embedding dimension
)

image_embedding = embeddings.image_embedding
text_embedding = embeddings.text_embedding
```

**Use Cases**:
- Visual product search (image query → similar products)
- Content moderation (flag similar images to banned content)
- Video similarity (scene-level embeddings)

### Custom Embeddings

**Bring Your Own Model**:
```python
from transformers import CLIPModel, CLIPProcessor
import torch

# Load CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Generate embeddings
images = [Image.open(f"image_{i}.jpg") for i in range(100)]
inputs = processor(images=images, return_tensors="pt", padding=True)

with torch.no_grad():
    embeddings = model.get_image_features(**inputs)
    # Normalize for dot product distance
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

# Upload to Matching Engine
embeddings_list = embeddings.cpu().numpy().tolist()
```

**Embedding Best Practices**:
1. **Normalize**: Always L2-normalize for DOT_PRODUCT distance
2. **Dimension Reduction**: Consider PCA if embeddings >1024D (slower indexing)
3. **Batch Processing**: Generate embeddings in batches (GPU efficiency)
4. **Consistency**: Use same model for indexing and querying

## Index Deployment

### Creating an Index

**Step 1: Prepare Embeddings File**

Format: JSONL (newline-delimited JSON)
```json
{"id": "doc_001", "embedding": [0.12, -0.34, 0.56, ...]}
{"id": "doc_002", "embedding": [0.23, 0.11, -0.45, ...]}
```

Upload to GCS:
```bash
gsutil cp embeddings.jsonl gs://my-bucket/embeddings/
```

**Step 2: Create Index**

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

# Create index
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="product-embeddings-v1",
    contents_delta_uri="gs://my-bucket/embeddings/",
    dimensions=768,
    approximate_neighbors_count=150,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
    leaf_node_embedding_count=10000,
    leaf_nodes_to_search_percent=10,
    description="CLIP embeddings for 10M product images"
)

print(f"Index created: {index.resource_name}")
```

**Build Time**:
- 1M vectors: ~10-20 minutes
- 10M vectors: ~1-2 hours
- 100M vectors: ~4-6 hours
- 1B vectors: ~12-24 hours

**Step 3: Create Index Endpoint**

```python
# Create endpoint
endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="product-search-endpoint",
    public_endpoint_enabled=True,  # Internet-accessible
    description="Production product search endpoint"
)

print(f"Endpoint created: {endpoint.resource_name}")
```

**Step 4: Deploy Index to Endpoint**

```python
# Deploy index
deployed_index = endpoint.deploy_index(
    index=index,
    deployed_index_id="prod_v1",
    display_name="Production Product Index v1",
    machine_type="n1-standard-32",
    min_replica_count=2,
    max_replica_count=10,
    enable_access_logging=True
)

print(f"Deployment complete: {deployed_index.id}")
```

**Deployment Options**:

| Option | Value | Effect |
|--------|-------|--------|
| `public_endpoint_enabled` | True | Internet access (with auth) |
| `public_endpoint_enabled` | False | VPC-only (Private Service Connect) |
| `enable_access_logging` | True | Log all queries (Cloud Logging) |
| `automatic_resources` | Auto-scaling config | Alternative to manual replicas |

## Query API

### Basic Query

```python
# Query with single vector
query_embedding = [0.12, -0.34, 0.56, ...]  # 768D vector

response = endpoint.find_neighbors(
    deployed_index_id="prod_v1",
    queries=[query_embedding],
    num_neighbors=10
)

# Process results
for neighbor in response[0]:
    print(f"ID: {neighbor.id}, Distance: {neighbor.distance}")
```

**Response Format**:
```python
[
    [
        MatchNeighbor(id="doc_123", distance=0.92),
        MatchNeighbor(id="doc_456", distance=0.89),
        MatchNeighbor(id="doc_789", distance=0.85),
        ...
    ]
]
```

### Approximate vs Exact k-NN

**Approximate Nearest Neighbors (ANN)**:
- ScaNN/Tree-AH indexes
- Trade-off: Speed vs Recall
- Typical recall: 90-98% @ k=10
- Latency: 10-50ms for billions of vectors

**Exact Nearest Neighbors**:
- Brute force index
- 100% recall guaranteed
- Latency: Proportional to dataset size
- Only feasible for <10K vectors

**Recall Measurement**:
```python
# Ground truth (brute force)
exact_results = brute_force_search(query, k=10)

# ANN results
ann_results = scann_search(query, k=10)

# Calculate recall
hits = len(set(ann_results) & set(exact_results))
recall = hits / k  # e.g., 9/10 = 90% recall
```

**Tuning for Higher Recall**:
1. Increase `leafNodesToSearchPercent` (10% → 20%)
2. Increase `approximateNeighborsCount` (150 → 300)
3. Use more replicas (lower per-replica QPS)

**Cost vs Recall Tradeoff**:
- 90% recall: 10-20ms latency, lower cost
- 95% recall: 20-40ms latency, moderate cost
- 98% recall: 40-80ms latency, higher cost

### Batch Queries

```python
# Query multiple vectors at once
query_embeddings = [
    [0.12, -0.34, ...],  # Query 1
    [0.45, 0.23, ...],   # Query 2
    [0.67, -0.11, ...]   # Query 3
]

responses = endpoint.find_neighbors(
    deployed_index_id="prod_v1",
    queries=query_embeddings,
    num_neighbors=10
)

# responses[0] = neighbors for query 1
# responses[1] = neighbors for query 2
# responses[2] = neighbors for query 3
```

**Batch Performance**:
- Batch size 1: 15ms latency
- Batch size 10: 25ms latency (2.5ms per query)
- Batch size 100: 100ms latency (1ms per query)
- Optimal batch: 10-50 queries (balances throughput and latency)

### Metadata Filtering

**Restricts** (Allowlist filtering):
```python
response = endpoint.find_neighbors(
    deployed_index_id="prod_v1",
    queries=[query_embedding],
    num_neighbors=10,
    filter=[
        {"namespace": "category", "allow": ["shoes", "sneakers"]},
        {"namespace": "price", "allow": ["0-50", "50-100"]}
    ]
)
```

**How It Works**:
1. ScaNN finds k-nearest neighbors (unfiltered)
2. Post-filter: Remove results not matching criteria
3. Return top-k from filtered set

**Performance Impact**:
- Filtering AFTER search (not during)
- Need to retrieve more candidates: `num_neighbors=10` → fetch 100, filter → return 10
- Use `num_neighbors_override` for better recall with filters

**Limitations**:
- Filtering adds 5-15ms latency overhead
- Heavy filtering (>90% rejection) reduces recall
- Alternative: Separate indexes per category

## Stream Updates

### Incremental Index Updates

**Problem**: Rebuilding 100M vector index takes hours. How to add new embeddings without downtime?

**Solution**: Streaming updates (Preview feature)

From [Real-time AI with Vertex AI](https://cloud.google.com/blog/products/ai-machine-learning/real-time-ai-with-google-cloud-vertex-ai) (accessed 2025-11-16):
> "Starting this month, Vertex AI Matching Engine and Feature Store will support real-time Streaming Ingestion as Preview features."

**Enable Streaming Updates**:
```python
# Create index with streaming enabled
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="streaming-index",
    contents_delta_uri="gs://my-bucket/embeddings/",
    dimensions=768,
    is_complete_overwrite=False,  # Enable incremental updates
    ...
)
```

**Add Embeddings**:
```python
# Upsert new embeddings
new_embeddings = [
    {"id": "new_001", "embedding": [0.12, ...]},
    {"id": "new_002", "embedding": [0.34, ...]},
]

# Write to GCS delta location
delta_file = "gs://my-bucket/deltas/2025-01-15.jsonl"
# Upload new_embeddings as JSONL to delta_file

# Update index
index.update_embeddings(
    contents_delta_uri=delta_file
)

# Refresh takes 5-15 minutes (not hours!)
```

**Update Frequency**:
- **Batch updates**: Every 15-60 minutes
- **Latency**: 5-15 minutes for index refresh
- **Throughput**: Millions of embeddings per update

**Remove Embeddings**:
```python
# Mark embeddings for deletion
deletes = [
    {"id": "old_001"},
    {"id": "old_002"}
]

# Write deletion file
deletion_file = "gs://my-bucket/deltas/deletions.jsonl"
# Upload deletes to deletion_file

index.remove_embeddings(
    contents_delta_uri=deletion_file
)
```

**Update vs Rebuild**:

| Operation | Time (100M vectors) | Downtime | Use Case |
|-----------|---------------------|----------|----------|
| Full Rebuild | 4-6 hours | Yes (new index) | Major schema changes |
| Streaming Update | 5-15 minutes | No | Add/update embeddings |
| Remove Embeddings | 5-15 minutes | No | Delete outdated items |

**Best Practices**:
1. **Batch updates**: Accumulate changes, update hourly (not per-item)
2. **Versioning**: Keep track of delta files (enable rollback)
3. **Monitoring**: Track update latency and errors
4. **Fallback**: Maintain ability to rebuild from scratch

## Cost Analysis

### Pricing Components

**1. Index Storage**

**Formula**: `cost = index_size_GB × $0.30/GB/month`

**Example** (100M vectors, 768D, ScaNN compression):
```
Index size = (768 × 4 bytes × 100M) / 4 compression
           = 76.8 GB compressed
Monthly cost = 76.8 × $0.30 = $23.04
```

**2. Index Deployment (Replicas)**

**Formula**: `cost = machine_hours × machine_cost`

**Example** (2 replicas, n1-standard-32, 24/7):
```
Machine cost = $1.52/hour
Monthly hours = 730
Cost per replica = 730 × $1.52 = $1,109.60
Total (2 replicas) = $2,219.20/month
```

**3. Query Pricing**

**Free tier**: First 1M queries/month
**Beyond 1M**: $4.00 per 1K queries

**Example** (10M queries/month):
```
Free: 1M queries
Billable: 9M queries
Cost = (9,000 × $4.00) = $36,000/month
```

**Note**: Query pricing is the dominant cost at scale!

### Cost Optimization Strategies

**1. Right-Size Replicas**

```python
# Monitor QPS and replica utilization
from google.cloud import monitoring_v3

client = monitoring_v3.MetricServiceClient()
project_name = f"projects/{project_id}"

# Query endpoint QPS
results = client.list_time_series(
    request={
        "name": project_name,
        "filter": 'metric.type="aiplatform.googleapis.com/prediction/qps"',
        "interval": interval,
    }
)

# If avg QPS < 10 per replica → reduce replicas
# If avg QPS > 100 per replica → add replicas
```

**Target**: 20-50 QPS per replica (70% capacity headroom)

**2. Use Smaller Machine Types**

| Dataset Size | Machine Type | Monthly Cost (2 replicas) |
|--------------|--------------|---------------------------|
| 10M vectors | n1-standard-16 | $1,110 |
| 50M vectors | n1-standard-32 | $2,220 |
| 100M vectors | n1-highmem-32 | $2,980 |

**Rule**: Choose smallest machine that fits index + 30% headroom

**3. Reduce Query Volume**

**Caching**:
```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def cached_search(query_hash, k=10):
    # Only query Matching Engine if not cached
    return endpoint.find_neighbors(...)

# Hash query vector
query_hash = hash(tuple(query_embedding))
results = cached_search(query_hash, k=10)
```

**Deduplication**: Batch identical queries from multiple users

**4. Index Compression**

- ScaNN quantization: 4x compression (enabled by default)
- Dimension reduction (PCA): 768D → 384D (2x savings, slight accuracy loss)
- Product quantization: Higher compression, lower recall

**5. Multi-Tenancy**

**Single Index for Multiple Applications**:
- Use namespaces in metadata
- Filter by namespace during query
- Share infrastructure cost across teams

### Total Cost of Ownership (TCO)

**Example**: 100M product images, 10M queries/month

| Component | Calculation | Monthly Cost |
|-----------|-------------|--------------|
| Index Storage | 76.8 GB × $0.30 | $23 |
| Deployment (2 replicas) | 2 × n1-standard-32 | $2,219 |
| Queries | 9M × $4/1K | $36,000 |
| **Total** | | **$38,242/month** |

**Cost Breakdown**:
- Queries: 94% of cost
- Deployment: 6% of cost
- Storage: <1% of cost

**Key Insight**: Query cost dominates at scale. Optimize by caching, batching, and right-sizing QPS.

## arr-coc-0-1 Visual Embedding Search

### Use Case: Relevance-Aware Patch Retrieval

**Scenario**: Given a query ("find images with red cars"), retrieve the most relevant 64×64 patches from a corpus of 1M images.

**Why Matching Engine?**
- **Scale**: 1M images × 196 patches/image = 196M patch embeddings
- **Speed**: <50ms latency for top-100 patches
- **Multimodal**: CLIP embeddings enable text → image patch search

### Architecture

**Components**:
1. **Embedding Generation**: CLIP text/image encoder
2. **Index**: ScaNN tree-AH index (196M vectors, 512D)
3. **Query**: Text query → CLIP text embedding → k-NN search
4. **Reranking**: Fetch top-100 patches → ARR-COC relevance scoring → select top-64
5. **Token Allocation**: Variable LOD based on patch relevance

**Data Flow**:
```
Query: "red cars in urban scenes"
    ↓
CLIP Text Encoder → [0.23, -0.11, 0.67, ...]  (512D)
    ↓
Matching Engine (196M patch embeddings)
    ↓
Top-100 Patches (with distances)
    ↓
ARR-COC Relevance Scoring
    ↓
Top-64 Patches (sorted by relevance)
    ↓
Token Allocation: High-relevance patches get 400 tokens, low-relevance get 64 tokens
```

### Implementation

**Step 1: Generate Patch Embeddings**

```python
from transformers import CLIPModel, CLIPProcessor
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Extract 64×64 patches from 1M images
patches = extract_patches(images, patch_size=64)  # 196M patches

# Generate embeddings (batch processing)
embeddings = []
for batch in batches(patches, batch_size=256):
    inputs = processor(images=batch, return_tensors="pt")
    with torch.no_grad():
        batch_emb = model.get_image_features(**inputs)
        batch_emb = batch_emb / batch_emb.norm(dim=-1, keepdim=True)
        embeddings.append(batch_emb.cpu().numpy())

all_embeddings = np.concatenate(embeddings)  # Shape: (196M, 512)
```

**Step 2: Create and Deploy Index**

```python
# Upload embeddings to GCS
with open("patch_embeddings.jsonl", "w") as f:
    for i, emb in enumerate(all_embeddings):
        f.write(json.dumps({
            "id": f"patch_{i}",
            "embedding": emb.tolist(),
            "metadata": {
                "image_id": i // 196,
                "patch_row": (i % 196) // 14,
                "patch_col": (i % 196) % 14
            }
        }) + "\n")

# Create index
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="arr-coc-patch-embeddings",
    contents_delta_uri="gs://arr-coc/patch_embeddings/",
    dimensions=512,
    approximate_neighbors_count=200,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
    leaf_node_embedding_count=20000,
    leaf_nodes_to_search_percent=15
)

# Deploy index
endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="arr-coc-search-endpoint"
)

endpoint.deploy_index(
    index=index,
    deployed_index_id="patch_index_v1",
    machine_type="n1-highmem-32",
    min_replica_count=2,
    max_replica_count=8
)
```

**Step 3: Query with Text**

```python
# Text query → CLIP embedding
query_text = "red sports car in city street"
query_inputs = processor(text=[query_text], return_tensors="pt")

with torch.no_grad():
    query_emb = model.get_text_features(**query_inputs)
    query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)

# Query Matching Engine
response = endpoint.find_neighbors(
    deployed_index_id="patch_index_v1",
    queries=[query_emb[0].cpu().numpy().tolist()],
    num_neighbors=100
)

# Get top-100 patch IDs and distances
top_patches = [
    {"id": n.id, "distance": n.distance}
    for n in response[0]
]
```

**Step 4: ARR-COC Relevance Scoring**

```python
from arr_coc.knowing import (
    InformationScorer,
    SalienceScorer,
    ParticipationScorer
)

# Load patches
patches_data = load_patches([p["id"] for p in top_patches])

# Score with three ways of knowing
info_scorer = InformationScorer()
salience_scorer = SalienceScorer()
participation_scorer = ParticipationScorer()

relevance_scores = []
for patch in patches_data:
    # Propositional: Information content
    info_score = info_scorer.score(patch)

    # Perspectival: Visual salience
    salience_score = salience_scorer.score(patch)

    # Participatory: Query-content coupling
    participation_score = participation_scorer.score(patch, query_emb)

    # Combined relevance
    relevance = (info_score + salience_score + participation_score) / 3
    relevance_scores.append(relevance)

# Select top-64 by relevance
top_64_indices = np.argsort(relevance_scores)[-64:]
selected_patches = [patches_data[i] for i in top_64_indices]
```

**Step 5: Variable LOD Token Allocation**

```python
from arr_coc.attending import RelevanceAllocator

allocator = RelevanceAllocator(total_budget=14400)  # 64 patches × 225 avg tokens

# Allocate tokens based on relevance
allocations = allocator.allocate(
    patches=selected_patches,
    relevance_scores=[relevance_scores[i] for i in top_64_indices],
    min_tokens=64,
    max_tokens=400
)

# allocations[0] = {"patch_id": "patch_12345", "tokens": 400}  # High relevance
# allocations[63] = {"patch_id": "patch_67890", "tokens": 64}  # Low relevance
```

### Performance Metrics

**Query Latency Breakdown**:
- CLIP text encoding: 5-10ms
- Matching Engine search (196M patches): 30-50ms
- Fetch top-100 patch data: 10-20ms
- ARR-COC relevance scoring: 50-100ms
- **Total**: 95-180ms per query

**Scaling**:
- 196M patch embeddings
- 2 replicas (n1-highmem-32)
- ~50 QPS sustained throughput
- 95th percentile latency: <200ms

**Cost** (monthly):
- Index storage: 196M × 512 × 4 / 4 = 98 GB → $29.40
- Deployment: 2 × n1-highmem-32 → $2,980
- Queries (1M/month): $0 (free tier)
- **Total**: ~$3,010/month

### Comparison: Matching Engine vs FAISS

**Why Matching Engine?**

| Feature | FAISS (self-hosted) | Matching Engine |
|---------|---------------------|-----------------|
| **Scaling** | Manual sharding | Auto-scaling |
| **Availability** | DIY replication | Managed HA |
| **Updates** | Rebuild index | Streaming updates |
| **Latency** | 10-30ms | 30-50ms |
| **Cost** | Compute + DevOps | Query + deployment |
| **Ops Burden** | High | Low |

**When to use FAISS**:
- <10M vectors (single machine)
- Ultra-low latency required (<10ms)
- Cost-sensitive (DIY cheaper at small scale)

**When to use Matching Engine**:
- >10M vectors (multi-machine scale)
- Need managed infrastructure
- Streaming updates required
- Production SLA requirements

## Best Practices

### 1. Normalize Embeddings

```python
# CRITICAL for DOT_PRODUCT distance
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Verify normalization
assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)
```

**Why**: DOT_PRODUCT expects unit vectors. Non-normalized embeddings → incorrect distances.

### 2. Monitor Recall

```python
# Periodically evaluate recall
ground_truth = brute_force_knn(query_emb, k=10)
ann_results = matching_engine_knn(query_emb, k=10)

recall = len(set(ground_truth) & set(ann_results)) / k
print(f"Recall@10: {recall:.2%}")

# Alert if recall drops below 90%
if recall < 0.90:
    alert("Matching Engine recall degraded!")
```

### 3. Use Batch Queries

```python
# BAD: Sequential queries
for query in queries:
    results = endpoint.find_neighbors(queries=[query], ...)  # 100 RPCs

# GOOD: Batch query
results = endpoint.find_neighbors(queries=queries, ...)  # 1 RPC
```

**Speedup**: 10x-50x for large batches

### 4. Index Versioning

```python
# Deploy new index alongside old
endpoint.deploy_index(
    index=new_index,
    deployed_index_id="v2",  # New version
    ...
)

# Traffic split: 90% v1, 10% v2 (canary)
# Monitor metrics, gradually shift traffic

# Undeploy old version
endpoint.undeploy_index(deployed_index_id="v1")
```

### 5. Cost Monitoring

```python
# Track query volume
from google.cloud import monitoring_v3

client = monitoring_v3.MetricServiceClient()

# Alert if queries exceed budget
query_count = get_metric("aiplatform.googleapis.com/prediction/prediction_count")
if query_count > BUDGET_THRESHOLD:
    alert(f"Query budget exceeded: {query_count}")
```

### 6. Embedding Drift Detection

```python
# Monitor embedding distribution over time
new_embeddings = generate_embeddings(new_data)
old_embeddings = load_historical_embeddings()

# Compare distributions (KL divergence, Wasserstein distance)
drift_score = wasserstein_distance(old_embeddings, new_embeddings)

if drift_score > DRIFT_THRESHOLD:
    alert("Embedding distribution drift detected → rebuild index")
```

## Troubleshooting

### Issue: High Latency

**Symptoms**: Queries take >100ms

**Diagnosis**:
1. Check QPS per replica: `qps_per_replica = total_qps / replica_count`
2. Check machine CPU: `gcloud monitoring time-series list --filter="metric.type=compute.googleapis.com/instance/cpu/utilization"`
3. Check batch size: Large batches (>100 queries) increase latency

**Solutions**:
- Add replicas if QPS/replica > 100
- Reduce `num_neighbors` (fewer results = faster)
- Decrease `leafNodesToSearchPercent` (lower recall, faster)

### Issue: Low Recall

**Symptoms**: Recall@10 < 85%

**Diagnosis**:
1. Measure recall against ground truth
2. Check if embeddings normalized
3. Check distance metric (DOT_PRODUCT requires normalization)

**Solutions**:
- Increase `approximateNeighborsCount` (150 → 300)
- Increase `leafNodesToSearchPercent` (10% → 20%)
- Use COSINE distance if embeddings not normalized

### Issue: Deployment Failed

**Symptoms**: `deploy_index()` fails with "Resource exhausted"

**Diagnosis**:
- Index too large for machine type
- Check: `index_size_gb < machine_memory_gb × 0.7`

**Solutions**:
- Use larger machine type (n1-highmem-32, n1-highmem-64)
- Reduce dimensions (PCA compression)
- Shard index across multiple endpoints

### Issue: Streaming Updates Slow

**Symptoms**: `update_embeddings()` takes >30 minutes

**Diagnosis**:
- Large delta files (>10M embeddings)
- Index rebuild triggered (not incremental)

**Solutions**:
- Smaller delta batches (<1M embeddings per update)
- Verify `is_complete_overwrite=False`
- Check for schema changes (forces rebuild)

## Sources

**Google Cloud Documentation**:
- [Vector Search Overview](https://docs.cloud.google.com/vertex-ai/docs/vector-search/overview) (accessed 2025-11-16)
- [Multimodal Embeddings API](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-embeddings-api) (accessed 2025-11-16)

**Google Cloud Blog**:
- [Vertex Matching Engine: Blazing fast and massively scalable nearest neighbor search](https://cloud.google.com/blog/products/ai-machine-learning/vertex-matching-engine-blazing-fast-and-massively-scalable-nearest-neighbor-search) (accessed 2025-11-16)
- [Real-time AI with Google Cloud Vertex AI](https://cloud.google.com/blog/products/ai-machine-learning/real-time-ai-with-google-cloud-vertex-ai) (accessed 2025-11-16)
- [Understanding the ScaNN index in AlloyDB](https://cloud.google.com/blog/products/databases/understanding-the-scann-index-in-alloydb) (accessed 2025-11-16)

**Google Research**:
- [SOAR: New algorithms for even faster vector search with ScaNN](https://research.google/blog/soar-new-algorithms-for-even-faster-vector-search-with-scann/) (accessed 2025-11-16)

**Additional References**:
- [Billion-scale vector search with cohere embeddings (Vespa)](https://colab.research.google.com/github/vespa-engine/pyvespa/blob/master/docs/sphinx/source/examples/billion-scale-vector-search-with-cohere-embeddings-cloud.ipynb) (accessed 2025-11-16)
- [Building Billion-Scale Vector Search - part two (Vespa Blog)](https://medium.com/vespa/building-billion-scale-vector-search-part-two-94f0101d15dd) (accessed 2025-11-16)

**Related Knowledge**:
- [Vector Databases for VLMs](../vector-spaces/02-vector-databases-vlms.md)
- [Vector Embeddings for Vision-Language Models](../vector-spaces/00-vector-embeddings-vlms.md)

---

**Last Updated**: 2025-11-16
**Next Review**: Monitor for ScaNN v2 release, multi-region index replication, GPU-accelerated indexing
