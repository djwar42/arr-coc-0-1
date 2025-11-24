# BATCH 2: VLM Attention Injection Research

## Core Architecture Patterns

### LLaVA Cross-Attention Layer Positions

**MLLama 3.2 (Meta's Multimodal LLaMA):**
- Cross-attention layers at: **3, 8, 13, 18, 23, 28, 33, 38**
- Pattern: Every 5 layers alternating between cross-attention and self-attention
- Total: 8 cross-attention injection points across 40 layers

**LLaVA Original:**
- Uses simple linear projection (NOT cross-attention!)
- `vision_feature_layer` parameter selects which ViT layer to extract features from
- Can provide multiple indices for multi-layer feature extraction
- Projects visual features directly into word embedding space

**LLaVA-SP Enhancement:**
- Adds 6 spatial visual tokens to original visual tokens
- Enhances spatial representation without full cross-attention

### Q-Former Architecture (BLIP-2)

**Core Components:**
- 32 learnable query vectors (typical configuration)
- Cross-attention layers inserted every G layers
- Initialized from BERT-base weights (except cross-attention = random init)

**Mechanism:**
```python
# Learnable queries interact with frozen image features
queries = nn.Parameter(torch.randn(32, hidden_dim))  # 32 learnable queries

# Cross-attention every G layers
for i, layer in enumerate(transformer_layers):
    if i % G == 0:  # Cross-attention layer
        queries = cross_attention(queries, image_features)
    queries = self_attention(queries)
```

**Training Objectives:**
1. Image-Text Contrastive (ITC)
2. Image-Text Matching (ITM)
3. Image-grounded Text Generation (ITG)

### FiLM Conditioning (Feature-wise Linear Modulation)

**Original Paper:** Perez et al. 2017 - 2936 citations!

**Core Formula:**
```
FiLM(F_i,c | γ_i,c, β_i,c) = γ_i,c · F_i,c + β_i,c
```

Where:
- F_i,c = feature activation at position i, channel c
- γ_i,c = learned scaling (from conditioning input)
- β_i,c = learned shift (from conditioning input)

**Key Insight from Distill.pub Analysis:**
- γ and β are NOT modulated in a single consistent way
- Sometimes scaling dominates, sometimes shifting
- Network learns which modulation strategy works best per layer

**FiLM-Ensemble (2022):**
- Uses FiLM for probabilistic deep learning
- γ and β vectors have dimension D_n (feature dimension at layer n)

**Implementation Pattern:**
```python
class FiLMLayer(nn.Module):
    def __init__(self, num_features, conditioning_dim):
        super().__init__()
        self.gamma_generator = nn.Linear(conditioning_dim, num_features)
        self.beta_generator = nn.Linear(conditioning_dim, num_features)

    def forward(self, features, conditioning):
        gamma = self.gamma_generator(conditioning)
        beta = self.beta_generator(conditioning)
        return gamma * features + beta
```

## Attention Injection Strategies

### Layer-wise Vision Injection (ICCV 2025)

**Key Finding:**
- Typical LVLM architecture: Visual encoder → Vision-language connector → LLM
- Performance varies significantly based on which layers receive visual injection
- Disentangled attention helps separate visual and textual processing

### Gated Cross-Attention (Nvidia Family)

- Image tiles + tile tags passed into gated cross-attention
- Captures fine image details while minimizing total model size
- Gating mechanism controls information flow

### Flamingo Perceiver Architecture

- Uses perceiver-style cross-attention
- Learnable queries attend to visual features
- More flexible than fixed position injection

## Integration with Spicy Lentil Architecture

### Where to Inject Vision Features

**Recommended Pattern:**
1. **Early layers (3-8):** Low-level visual features (edges, textures)
2. **Mid layers (13-23):** Object-level features
3. **Late layers (28-38):** Semantic/conceptual features

### FiLM for 9 Ways of Knowing

Each pathway can have its own FiLM conditioning:
```python
# Per-pathway FiLM modulation
for pathway in nine_pathways:
    gamma, beta = film_generator(cognitive_fingerprint)
    pathway_output = gamma * pathway(slot_features) + beta
```

### Q-Former Style Learned Queries

Instead of fixed queries, use learned queries for each slot:
```python
# Per-slot learned queries (like Q-Former but for object slots)
slot_queries = nn.Parameter(torch.randn(K, num_queries, hidden_dim))
```

## Performance Benchmarks

**BLIP-2:**
- 8920 citations
- State-of-the-art on multiple VQA benchmarks
- Efficient: Frozen vision and language models

**FiLM:**
- 97.7% accuracy on CLEVR (visual reasoning)
- Works across many domains (VQA, robotics, games)

**LLaVA-1.5:**
- Simple but effective linear projection
- Competitive with more complex architectures

## Key Formulas

### Cross-Attention
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### FiLM Modulation
```
output = γ * features + β
```

### Q-Former Query Update
```
q_new = CrossAttn(q, image_features) + SelfAttn(q)
```

## Implementation Recommendations

1. **Start simple:** Linear projection (LLaVA style)
2. **Add FiLM:** For query-conditioned modulation
3. **Consider Q-Former:** For learnable visual extraction
4. **Position matters:** Test different layer injection points

---

**Sources:**
- HuggingFace Transformers Documentation
- "Understanding Multimodal LLaMA 3.2 Architecture" - Medium
- "FiLM: Visual Reasoning with a General Conditioning Layer" - AAAI 2018
- "Feature-wise transformations" - Distill.pub 2018
- "BLIP-2: Bootstrapping Language-Image Pre-training" - ICML 2023
