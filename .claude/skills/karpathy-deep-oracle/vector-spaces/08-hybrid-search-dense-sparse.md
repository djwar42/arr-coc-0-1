# Hybrid Search: Combining Dense and Sparse Vectors

## Overview

Hybrid search combines multiple search algorithms to improve accuracy and relevance by leveraging the complementary strengths of different retrieval methods. The most common approach merges **sparse vectors** (keyword-based, like BM25) with **dense vectors** (semantic embeddings) to deliver search results that capture both exact term matches and contextual meaning.

**Why Hybrid Search Matters:**
- Dense vectors excel at understanding context and semantic meaning
- Sparse vectors excel at exact keyword matching and domain-specific terms
- Neither method alone captures the full spectrum of user intent
- Combining both provides more robust retrieval across diverse queries

From [Hybrid Search Explained](https://weaviate.io/blog/hybrid-search-explained) (Weaviate, accessed 2025-02-02):
> "Hybrid search queries are ideal for a search system that wants to leverage the power of semantic search capabilities but still rely on exact keyword matches. For example, the example search query 'How to catch an Alaskan Pollock' from before would lead to better results with a hybrid search query than with a common keyword search or even a semantic search query."

## Sparse Methods: Keyword-Based Retrieval

### BM25 (Best Match 25)

BM25 builds on TF-IDF by incorporating document length normalization and saturation effects. It remains the de facto standard for sparse retrieval.

**BM25 Scoring Formula:**

```
score(D,Q) = Σ IDF(qi) · [f(qi,D) · (k1 + 1)] / [f(qi,D) + k1 · (1 - b + b · |D|/avgdl)]
```

Where:
- `D` = document
- `Q` = query
- `qi` = query term i
- `f(qi,D)` = term frequency of qi in D
- `k1` = term frequency saturation parameter (typically 1.2-2.0)
- `b` = length normalization parameter (typically 0.75)
- `|D|` = document length
- `avgdl` = average document length in collection
- `IDF(qi)` = inverse document frequency of term qi

**BM25 Characteristics:**
- **Fast**: Inverted index lookup is efficient
- **Interpretable**: Scores are based on observable term statistics
- **Exact matching**: Catches specific terminology and product names
- **No training required**: Works out-of-the-box on any text corpus

**Limitations:**
- **Vocabulary mismatch problem**: Misses semantically similar terms with different wording
- **No contextual understanding**: "bank" (financial) vs "bank" (river) treated identically
- **Synonym blindness**: "car" and "automobile" have zero overlap

### BM25F (Field-Weighted BM25)

BM25F extends BM25 to support multiple text fields with different importance weights.

**Use Case:**
```python
# Document structure with weighted fields
document = {
    "title": "MacBook Pro 16-inch",      # Weight: 2.0 (high importance)
    "description": "Powerful laptop...", # Weight: 1.0 (normal)
    "reviews": "Great battery life..."   # Weight: 0.5 (lower)
}
```

From [Weaviate documentation](https://weaviate.io/blog/hybrid-search-explained) (accessed 2025-02-02):
> "BM25F allows multiple text fields per object to be given different weights in the ranking calculation. These weights are important for when fields in a document are more important than others. For example, a title may be given more weight than the abstract, since the title is sometimes more informative and concise."

### SPLADE: Learned Sparse Embeddings

SPLADE (**Sp**arse **L**exical **a**nd **E**xpansion) uses transformer models to create learnable sparse representations that overcome the vocabulary mismatch problem.

**Key Innovation: Term Expansion**

Traditional sparse methods only match exact terms. SPLADE learns to expand queries and documents with related terms:

```
Original query: "Orangutans are native to rainforests"

SPLADE expansion adds relevant terms:
- "jungle" (synonym)
- "tropical" (related concept)
- "forest" (related term)
- "habitat" (semantic connection)
- "primate" (category)
```

**SPLADE Architecture:**

From [SPLADE for Sparse Vector Search](https://www.pinecone.io/learn/splade/) (Pinecone, accessed 2025-02-02):

1. **Input**: Text tokenized with BERT tokenizer
2. **Processing**: Pass through BERT with MLM (Masked Language Modeling) head
3. **Aggregation**: Max pooling across token-level predictions
4. **Output**: 30,522-dimensional sparse vector (BERT vocab size)

```python
# SPLADE vector generation (conceptual)
tokens = tokenizer(text)
output = bert_model(**tokens)  # MLM head outputs

# Aggregate token-level vectors via max pooling
sparse_vec = torch.max(
    torch.log(1 + torch.relu(output.logits)) * attention_mask,
    dim=1
)[0]

# Result: Most values are 0, non-zero values indicate relevance
# Example: {"programmed": 2.36, "cell": 1.95, "death": 1.96, ...}
```

**SPLADE Advantages:**
- **Learned expansion**: Identifies synonyms and related terms automatically
- **Context-aware**: Same word gets different weights in different contexts
- **Interpretable**: Non-zero positions map to vocabulary tokens
- **Better than BM25**: Overcomes vocabulary mismatch while maintaining sparsity

**SPLADE Limitations:**
- **Slower than BM25**: More non-zero values (typically 100-300 vs 10-30)
- **Requires GPU**: Encoding needs transformer inference
- **Model dependency**: Needs fine-tuned SPLADE model

From [SPLADE research](https://www.pinecone.io/learn/splade/) (accessed 2025-02-02):
> "SPLADE takes all these distributions and aggregates them into a single distribution called the importance estimation. This importance estimation is the sparse vector produced by SPLADE. We can combine all these probability distributions into a single distribution that tells us the relevance of every token in the vocab to our input sentence."

## Dense Methods: Semantic Embeddings

### Sentence Transformers

Pre-trained models that encode text into fixed-size dense vectors capturing semantic meaning.

**Popular Models:**
- `all-MiniLM-L6-v2`: 384 dimensions, fast, good general performance
- `all-mpnet-base-v2`: 768 dimensions, higher quality, slower
- `BGE-base-en-v1.5`: 768 dimensions, state-of-the-art for retrieval

**Dense Vector Properties:**
- **High-dimensional**: Typically 384-1024 dimensions
- **Mostly non-zero**: Values distributed across all dimensions
- **Semantic similarity**: Cosine distance reflects meaning similarity
- **Context-aware**: Captures nuanced relationships

**Advantages:**
- **Semantic search**: Finds conceptually similar content
- **Multi-lingual**: Can match across languages
- **Handles paraphrasing**: Different wording, same meaning
- **No vocabulary gap**: Represents concepts, not just terms

**Limitations:**
- **Slower retrieval**: Approximate nearest neighbor search required
- **Less precise for exact terms**: May miss specific product codes or IDs
- **Black box**: Hard to interpret why two vectors are similar
- **Requires training data**: Domain adaptation needs labeled examples

### ColBERT: Late Interaction Model

ColBERT uses **multi-vector representations** where each token gets its own embedding, enabling fine-grained matching.

**Late Interaction Mechanism:**

From [Qdrant Hybrid Search Guide](https://qdrant.tech/articles/hybrid-search/) (accessed 2025-02-02):

```
Query: "programming languages"
├─ Token embeddings: [embed(programming), embed(languages)]

Document: "Python and JavaScript are popular languages"
├─ Token embeddings: [embed(Python), embed(and), embed(JavaScript), ...]

Late Interaction Scoring:
Max-sim between each query token and all document tokens:
- max_sim(embed(programming), document_embeds) = 0.89 (matches "Python")
- max_sim(embed(languages), document_embeds) = 0.95 (matches "languages")

Final score = sum of max-sims = 1.84
```

**ColBERT Advantages:**
- **Token-level matching**: Captures precise interactions
- **Better than single-vector**: More nuanced relevance
- **Precomputable document vectors**: Only query needs runtime encoding
- **Efficient reranking**: Faster than cross-encoders

**ColBERT Use Case:**

From [Hybrid Search Revamped](https://qdrant.tech/articles/hybrid-search/) (Qdrant, accessed 2025-02-02):
> "Use ColBERT-like models as a reranking step, after retrieving candidates with single-vector dense and/or sparse methods. This reflects the latest trends in the field, as single-vector methods are still the most efficient, but multivectors capture the nuances of the text better."

## Fusion Strategies: Combining Results

Fusion methods combine ranked results from multiple search algorithms based solely on scores, without examining document content.

### Reciprocal Rank Fusion (RRF)

The most popular fusion algorithm, used by default in Weaviate, Qdrant, and Elasticsearch.

**RRF Formula:**

```
RRF_score(d) = Σ [1 / (k + rank_i(d))]
```

Where:
- `d` = document
- `rank_i(d)` = rank of document d in result list i
- `k` = constant (typically 60)

**Example Calculation:**

From [Weaviate Hybrid Search](https://weaviate.io/blog/hybrid-search-explained) (accessed 2025-02-02):

```
Documents: A, B, C

BM25 Rankings:        Dense Rankings:
1. A                  1. B
2. B                  2. C
3. C                  3. A

RRF Scores (k=0):
- Document A: 1/1 + 1/3 = 1.33
- Document B: 1/2 + 1/1 = 1.50  ← Winner
- Document C: 1/3 + 1/2 = 0.83
```

**Why RRF Works:**

- **Rank-based**: Immune to score scale differences between methods
- **Penalizes low ranks**: 1/rank decreases rapidly
- **Simple**: No parameter tuning beyond k
- **Proven effective**: Consistently outperforms score normalization approaches

**RRF Parameters:**

```python
# Weaviate example
results = collection.query.hybrid(
    query="quantum computing applications",
    alpha=0.5,  # 0=pure BM25, 1=pure vector, 0.5=equal weight
    fusion_type="rankedFusion"  # RRF (default)
)
```

### Relative Score Fusion

Alternative fusion method that normalizes scores before combining.

**Relative Score Formula:**

```
norm_score(d, method) = score(d, method) / max_score(method)
final_score(d) = α · norm_score(d, sparse) + (1-α) · norm_score(d, dense)
```

**When to Use:**
- When score magnitudes matter (e.g., one method is known to be more reliable)
- When you want explicit control over weighting via α
- For interpretable final scores

**RRF vs Relative Score:**

From [Benham & Culpepper 2018](https://arxiv.org/abs/1811.06147) research:
- **RRF**: Better when methods have different score distributions
- **Relative Score**: Better when both methods are well-calibrated
- **In practice**: RRF more robust, less tuning required

### Why Not Linear Combination?

From [Qdrant analysis](https://qdrant.tech/articles/hybrid-search/) (accessed 2025-02-02):

**Common misconception:**
```python
# This doesn't work well!
final_score = 0.7 * vector_score + 0.3 * bm25_score
```

**Problem:** BM25 and cosine similarity scores are not linearly separable. Plotting both as coordinates in 2D space shows relevant and non-relevant documents are completely mixed—no linear formula can distinguish them.

**Better alternatives:**
1. **RRF**: Rank-based fusion
2. **Reranking**: Use cross-encoder or ColBERT on top-k results
3. **Cascaded retrieval**: Multi-stage with different methods

## Vector Database Support

### Weaviate

**Hybrid Search API:**

```python
from weaviate import Client

client = Client("http://localhost:8080")

results = client.query.get(
    "Article",
    ["title", "content"]
).with_hybrid(
    query="machine learning applications",
    alpha=0.5,  # Balance between sparse and dense
    fusion_type="rankedFusion"  # or "relativeScoreFusion"
).with_limit(10).do()
```

**Features:**
- Built-in BM25 and vector search
- RRF and Relative Score Fusion
- Named vectors support (multiple embeddings per object)
- Configurable alpha parameter for weighting

### Qdrant

**Query API (v1.10+):**

From [Qdrant Hybrid Search](https://qdrant.tech/articles/hybrid-search/) (accessed 2025-02-02):

```python
from qdrant_client import QdrantClient, models

client = QdrantClient("localhost", port=6333)

# Complex multi-stage hybrid search
results = client.query_points(
    collection_name="documents",
    prefetch=[
        # Stage 1: Retrieve with sparse vectors
        models.Prefetch(
            query=models.SparseVector(
                indices=[125, 9325, 58214],
                values=[0.164, 0.229, 0.731]
            ),
            using="sparse",
            limit=50
        ),
        # Stage 2: Retrieve with dense vectors
        models.Prefetch(
            query=[0.1, 0.2, 0.3, ...],  # Dense embedding
            using="dense",
            limit=50
        )
    ],
    # Fusion: Combine via RRF
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=10
)
```

**Qdrant Features:**
- Native sparse + dense vector support
- Multi-stage retrieval pipelines
- RRF fusion built-in
- Named vectors for multiple embeddings
- Payload filtering during search

**Advanced: Multi-stage with ColBERT Reranking:**

```python
results = client.query_points(
    collection_name="documents",
    prefetch=[
        # Prefetch candidates via hybrid search
        models.Prefetch(
            prefetch=[
                models.Prefetch(query=sparse_vec, using="sparse", limit=100),
                models.Prefetch(query=dense_vec, using="dense", limit=100)
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=25  # Reduce to top 25 via fusion
        )
    ],
    # Rerank with multi-vector (ColBERT)
    query=multi_vector_query,  # List of token embeddings
    using="colbert",
    limit=10
)
```

### Pinecone

**Hybrid Search with Sparse-Dense Indexes:**

```python
import pinecone

index = pinecone.Index("hybrid-index")

# Upsert with both dense and sparse vectors
index.upsert(vectors=[
    {
        "id": "doc1",
        "values": [0.1, 0.2, ...],  # Dense embedding
        "sparse_values": {
            "indices": [10, 50, 100],
            "values": [0.5, 0.8, 0.3]
        },
        "metadata": {"text": "..."}
    }
])

# Query with hybrid search
results = index.query(
    vector=[0.1, 0.2, ...],  # Dense query
    sparse_vector={
        "indices": [10, 50],
        "values": [0.6, 0.9]
    },
    top_k=10,
    alpha=0.5  # Balance parameter
)
```

### Elasticsearch

**Hybrid Search via RRF:**

```json
POST /my-index/_search
{
  "query": {
    "hybrid": {
      "queries": [
        {
          "match": {
            "content": "machine learning"
          }
        },
        {
          "knn": {
            "field": "embedding",
            "query_vector": [0.1, 0.2, ...],
            "k": 50
          }
        }
      ]
    }
  },
  "rank": {
    "rrf": {
      "window_size": 50,
      "rank_constant": 60
    }
  }
}
```

### PostgreSQL + pgvector

**Hybrid Search with SQL:**

```sql
-- Sparse: Full-text search with ts_rank
WITH sparse_results AS (
  SELECT id, ts_rank(tsvector_col, query) AS bm25_score
  FROM documents, plainto_tsquery('machine learning') query
  WHERE tsvector_col @@ query
  ORDER BY bm25_score DESC
  LIMIT 100
),
-- Dense: Vector similarity
dense_results AS (
  SELECT id, 1 - (embedding <=> query_vec) AS vector_score
  FROM documents
  ORDER BY embedding <=> query_vec
  LIMIT 100
)
-- Combine via RRF
SELECT
  COALESCE(s.id, d.id) as id,
  COALESCE(1.0/(60 + s.rank), 0) + COALESCE(1.0/(60 + d.rank), 0) as rrf_score
FROM (SELECT id, ROW_NUMBER() OVER() as rank FROM sparse_results) s
FULL OUTER JOIN (SELECT id, ROW_NUMBER() OVER() as rank FROM dense_results) d
  ON s.id = d.id
ORDER BY rrf_score DESC
LIMIT 10;
```

## Implementation Patterns

### Pattern 1: Simple Hybrid Search

**Architecture:**
```
User Query
    ↓
[Encode Query]
    ├─→ Dense Embedding (384-dim)
    └─→ Sparse Vector (BM25 terms)
         ↓
[Parallel Search]
    ├─→ Vector Search (top 100)
    └─→ BM25 Search (top 100)
         ↓
[Fusion: RRF]
    ↓
Top 10 Results
```

**Python Implementation:**

```python
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class SimpleHybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Build indexes
        self.doc_embeddings = self.embedder.encode(documents)
        self.bm25 = BM25Okapi([doc.split() for doc in documents])

    def search(self, query, k=10, alpha=0.5):
        # Dense search
        query_emb = self.embedder.encode([query])[0]
        dense_scores = cosine_similarity([query_emb], self.doc_embeddings)[0]
        dense_ranks = np.argsort(dense_scores)[::-1]

        # Sparse search
        sparse_scores = self.bm25.get_scores(query.split())
        sparse_ranks = np.argsort(sparse_scores)[::-1]

        # RRF fusion
        rrf_scores = {}
        for rank, doc_idx in enumerate(dense_ranks[:100]):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1/(60 + rank)
        for rank, doc_idx in enumerate(sparse_ranks[:100]):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1/(60 + rank)

        # Return top k
        top_indices = sorted(rrf_scores.keys(),
                           key=lambda x: rrf_scores[x],
                           reverse=True)[:k]
        return [self.documents[i] for i in top_indices]
```

### Pattern 2: Hybrid Search with Reranking

**Architecture:**
```
Query
  ↓
[Hybrid Retrieval] → 50 candidates
  ↓
[Reranker: Cross-encoder or ColBERT]
  ↓
Top 10 refined results
```

**Benefits:**
- **Higher precision**: Reranker examines full query-document interaction
- **Catches edge cases**: Second chance for marginally relevant documents
- **Slower but better**: Only rerank small candidate set

**Implementation:**

```python
from sentence_transformers import CrossEncoder

class HybridWithReranking:
    def __init__(self, documents):
        self.hybrid_retriever = SimpleHybridSearch(documents)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def search(self, query, k=10):
        # Stage 1: Hybrid retrieval
        candidates = self.hybrid_retriever.search(query, k=50)

        # Stage 2: Reranking
        pairs = [[query, doc] for doc in candidates]
        rerank_scores = self.reranker.predict(pairs)

        # Sort by reranker scores
        ranked = sorted(zip(candidates, rerank_scores),
                       key=lambda x: x[1],
                       reverse=True)

        return [doc for doc, score in ranked[:k]]
```

### Pattern 3: Multi-Stage Cascaded Search

**Architecture:**
```
Query
  ↓
[Stage 1: Fast Retrieval] → 1000 candidates
  ├─→ Matryoshka 128-dim (fast approximate)
  └─→ SPLADE sparse
       ↓
[Stage 2: Refined Retrieval] → 100 candidates
  ├─→ Matryoshka 384-dim (slower, better)
  └─→ BM25 on filtered set
       ↓
[Stage 3: Precision Ranking] → 10 final
  └─→ ColBERT multi-vector reranking
```

**When to Use:**
- **Large corpora** (>10M documents)
- **Latency-sensitive** applications
- **Complex queries** needing multiple perspectives

### Pattern 4: Domain-Specific Hybrid

**Customize for specialized domains:**

```python
class EcommerceHybridSearch:
    """E-commerce search with product-specific features"""

    def search(self, query, filters=None):
        # Parse query intent
        intent = self.classify_intent(query)

        if intent == "product_code":
            # Exact match dominates
            return self.exact_search(query)

        elif intent == "brand_specific":
            # High weight on sparse (brand names)
            alpha = 0.3  # 70% sparse, 30% dense

        elif intent == "conceptual":
            # High weight on dense (semantic)
            alpha = 0.8  # 80% dense, 20% sparse

        # Execute hybrid search with dynamic alpha
        results = self.hybrid_search(query, alpha=alpha)

        # Apply business logic filters
        if filters:
            results = self.apply_filters(results, filters)

        return results
```

## Benchmarking Hybrid Search

### Evaluation Metrics

**Information Retrieval Metrics:**

```python
from ranx import Qrels, Run, evaluate

# Ground truth: query -> {doc_id: relevance_score}
qrels = Qrels({
    "q1": {"doc1": 3, "doc5": 5, "doc7": 2},
    "q2": {"doc2": 4, "doc8": 3}
})

# Search results: query -> {doc_id: score}
run = Run({
    "q1": {"doc1": 0.9, "doc5": 0.8, "doc7": 0.7, "doc3": 0.6},
    "q2": {"doc2": 0.95, "doc8": 0.85, "doc4": 0.75}
})

# Evaluate
metrics = evaluate(qrels, run, ["ndcg@10", "map@10", "mrr", "precision@10"])
print(metrics)
# {'ndcg@10': 0.87, 'map@10': 0.82, 'mrr': 0.95, 'precision@10': 0.80}
```

**Key Metrics:**
- **NDCG@k**: Normalized Discounted Cumulative Gain (accounts for position)
- **MAP@k**: Mean Average Precision
- **MRR**: Mean Reciprocal Rank (first relevant result position)
- **Recall@k**: Percentage of relevant docs retrieved

### A/B Testing Strategies

**Compare hybrid vs single methods:**

```python
def compare_methods(queries, ground_truth):
    results = {
        "bm25_only": evaluate_method(bm25_search, queries),
        "vector_only": evaluate_method(vector_search, queries),
        "hybrid_rrf": evaluate_method(hybrid_search, queries),
        "hybrid_rerank": evaluate_method(hybrid_rerank_search, queries)
    }

    for method, metrics in results.items():
        print(f"{method}: NDCG@10={metrics['ndcg@10']:.3f}")
```

**Typical Results Pattern:**

From [Qdrant observations](https://qdrant.tech/articles/hybrid-search/) (accessed 2025-02-02):

```
Query: "cybersport desk"
├─ BM25 Only:     "desk" ❌ (generic desk, not gaming)
└─ Vector Only:   "gaming desk" ✅
   → Hybrid wins with vector

Query: "64.2 inch console table"
├─ BM25 Only:     "cervantez 64.2'' console table" ✅ (exact match)
└─ Vector Only:   "69.5'' console table" ❌ (wrong size)
   → Hybrid wins with BM25

Conclusion: Neither method dominates all cases
```

## Production Considerations

### Latency Optimization

**Latency Budget Allocation:**

```
Target: 100ms total latency

Breakdown:
- Dense embedding:     5ms  (batch on GPU)
- Sparse tokenization: 1ms  (CPU)
- Vector search:      30ms  (HNSW ANN)
- BM25 search:        10ms  (inverted index)
- RRF fusion:          2ms  (merge sorted lists)
- Reranking:          40ms  (cross-encoder on top 50)
- Network/overhead:   12ms
                     ------
Total:               100ms
```

**Optimization Techniques:**

1. **Batch encoding**: Process multiple queries together on GPU
2. **Quantization**: Use int8 embeddings (2-4x speedup)
3. **Index optimization**: Tune HNSW parameters (M, efSearch)
4. **Caching**: Cache frequent query embeddings
5. **Async retrieval**: Run sparse and dense searches in parallel

### Scaling Considerations

**Horizontal Scaling:**

```
Load Balancer
    ↓
┌─────────┬─────────┬─────────┐
│ Node 1  │ Node 2  │ Node 3  │  (Query processing)
└────┬────┴────┬────┴────┬────┘
     │         │         │
┌────▼─────────▼─────────▼────┐
│    Shared Vector Database    │  (Weaviate/Qdrant cluster)
└──────────────────────────────┘
```

**Sharding Strategy:**

- **By collection**: Different document types on different shards
- **By size**: Shard large collections across multiple nodes
- **Geographic**: Distribute based on user location

**Replication:**

- **Read replicas**: Scale read throughput
- **Write primary**: Single source of truth for updates
- **Consistency**: Eventual consistency acceptable for search (stale results OK for ~seconds)

### Cost Optimization

**Compute Costs:**

```
Monthly costs for 10M documents:

Embedding Generation (one-time):
- GPU hours: 10M docs × 0.01s = 28 hours
- A100 GPU: $2.50/hr × 28 = $70

Vector DB Hosting:
- Weaviate Cloud: ~$200/month (optimized index)
- Self-hosted: $150/month (EC2 + storage)

Inference:
- 1M queries/month
- Average 100ms/query
- CPU: $50/month (batch processing)
```

**Optimization Strategies:**

1. **Quantization**: 4x memory reduction → smaller instances
2. **Sparse vectors**: Store only non-zero values
3. **Dimensionality reduction**: 384-dim vs 768-dim (2x savings)
4. **Index pruning**: Remove rarely accessed documents
5. **Cold storage**: Archive old documents to cheaper storage

### Monitoring & Observability

**Key Metrics to Track:**

```python
class HybridSearchMetrics:
    def track_query(self, query, results, latency_ms):
        metrics = {
            # Latency
            "total_latency_ms": latency_ms,
            "dense_latency_ms": self.dense_time,
            "sparse_latency_ms": self.sparse_time,
            "fusion_latency_ms": self.fusion_time,

            # Quality
            "num_results": len(results),
            "dense_contribution": self.count_dense_only(results),
            "sparse_contribution": self.count_sparse_only(results),
            "overlap": self.count_overlap(results),

            # Usage
            "query_length": len(query.split()),
            "result_clicked": None,  # Track later
            "user_satisfied": None   # Track later
        }

        self.log_metrics(metrics)
```

**Alerting:**

- **Latency > P95 threshold**: Search taking too long
- **Low overlap**: Methods diverging (investigate query patterns)
- **High sparse-only ratio**: Dense embeddings may need retraining
- **Zero results**: Coverage gap in corpus

## Advanced Topics

### Query Understanding & Intent Classification

**Adaptive alpha based on query type:**

```python
def adaptive_hybrid(query):
    # Classify query intent
    if has_product_code(query):
        alpha = 0.2  # Mostly BM25
    elif has_brand_name(query):
        alpha = 0.3  # Favor BM25
    elif is_conceptual(query):
        alpha = 0.8  # Favor dense
    elif is_navigational(query):
        alpha = 0.5  # Balanced

    return hybrid_search(query, alpha=alpha)
```

### Learned Fusion Weights

**Train a small model to predict optimal alpha:**

```python
from sklearn.ensemble import GradientBoostingRegressor

class LearnedFusion:
    def __init__(self):
        self.model = GradientBoostingRegressor()

    def train(self, queries, optimal_alphas):
        features = [self.extract_features(q) for q in queries]
        self.model.fit(features, optimal_alphas)

    def extract_features(self, query):
        return [
            len(query.split()),           # Query length
            has_numbers(query),            # Contains numbers
            has_special_chars(query),      # Contains special chars
            entity_count(query),           # Named entities
            avg_word_length(query)         # Complexity
        ]

    def predict_alpha(self, query):
        features = self.extract_features(query)
        return self.model.predict([features])[0]
```

### Multi-Vector Fields

**Support different embedding types per field:**

```python
# Document with multiple vector representations
document = {
    "id": "doc1",
    "title": "Machine Learning Guide",
    "content": "This guide covers...",

    # Multiple embeddings
    "title_embedding": [0.1, 0.2, ...],      # 384-dim
    "content_embedding": [0.5, 0.6, ...],    # 384-dim
    "title_sparse": {10: 0.5, 50: 0.8},      # SPLADE
    "content_sparse": {30: 0.3, 100: 0.9}    # SPLADE
}

# Query with field-specific search
results = hybrid_search(
    query="ML tutorial",
    field_weights={
        "title": 2.0,    # Title 2x important
        "content": 1.0
    }
)
```

## Sources

**Web Research:**

- [Hybrid Search Explained](https://weaviate.io/blog/hybrid-search-explained) - Weaviate (accessed 2025-02-02): Comprehensive guide to sparse/dense fusion, BM25, and RRF algorithm
- [SPLADE for Sparse Vector Search](https://www.pinecone.io/learn/splade/) - Pinecone (accessed 2025-02-02): Detailed explanation of learned sparse embeddings and term expansion
- [Hybrid Search Revamped](https://qdrant.tech/articles/hybrid-search/) - Qdrant (accessed 2025-02-02): Query API, multi-stage retrieval, ColBERT reranking, and production patterns
- [What is ColBERT and Late Interaction](https://jina.ai/news/what-is-colbert-and-late-interaction-and-why-they-matter-in-search/) - Jina AI: Late interaction models and multi-vector representations

**Research Papers:**

- Formal, T., Piwowarski, B., & Clinchant, S. (2021). [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720). SIGIR 2021
- Formal, T., Lassance, C., Piwowarski, B., & Clinchant, S. (2021). [SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval](https://arxiv.org/abs/2109.10086)
- Khattab, O., & Zaharia, M. (2020). [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832). SIGIR 2020
- Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., & Zaharia, M. (2021). [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488). NAACL 2022
- Benham, R., & Culpepper, J. S. (2018). [Risk-Reward Trade-offs in Rank Fusion](https://arxiv.org/abs/1811.06147). ADCS 2018

**Vector Database Documentation:**

- [Weaviate Hybrid Search Documentation](https://docs.weaviate.io/weaviate/search/hybrid)
- [Qdrant Query API Documentation](https://qdrant.tech/documentation/concepts/search/#query-api)
- [Pinecone Hybrid Search Guide](https://docs.pinecone.io/docs/hybrid-search)
- [Elasticsearch Hybrid Search with RRF](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html)
- [Vertex AI Hybrid Search](https://cloud.google.com/vertex-ai/docs/vector-search/about-hybrid-search) - Google Cloud

**Community Discussions:**

- [r/Rag: How are you guys doing Hybrid Search in production?](https://www.reddit.com/r/Rag/comments/1gd1hxu/how_are_you_guys_doing_hybrid_search_in_production/) - Reddit (accessed 2025-02-02): Real-world production experiences and PostgreSQL patterns
