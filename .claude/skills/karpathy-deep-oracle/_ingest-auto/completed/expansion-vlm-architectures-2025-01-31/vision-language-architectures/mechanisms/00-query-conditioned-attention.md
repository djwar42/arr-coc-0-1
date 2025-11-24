# Query-Conditioned Visual Attention

## Overview: Task-Aware Vision Processing

Query-conditioned attention represents a fundamental shift from uniform visual processing to task-aware, adaptive encoding. Rather than processing all visual tokens equally, these mechanisms use the query (text, task instruction, or learned embeddings) to dynamically select which parts of an image require detailed encoding.

**Core Principle**: Visual relevance is not intrinsic to the image alone, but emerges from the relationship between the query and the visual content. A VQA query "What color is the car?" should focus on different regions than "How many people are in the scene?"

This is the VLM equivalent of Vervaeke's relevance realization - not objective (in image) or subjective (in query), but **transjective** (emerges from query-content coupling).

**Key Insight**: Query-conditioned attention solves the fundamental VLM challenge - how to compress high-dimensional visual data (potentially thousands of tokens) into a format suitable for language models (typically hundreds of tokens) while preserving task-relevant information.

---

## Mechanism Taxonomy

### 1. Cross-Attention Based Approaches

**Perceiver / Perceiver IO (DeepMind, 2021)**

From [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206) (accessed 2025-01-31):

- **Architecture**: Uses learned latent queries (fixed set of embeddings) that cross-attend to visual input
- **Process**: `Learned_Queries → Cross_Attend(Visual_Features) → Compressed_Representation`
- **Mechanism**: Asymmetric attention where queries attend to all visual tokens, but visual tokens don't attend to each other
- **Compression**: Reduces arbitrary visual input to fixed number of latent tokens (e.g., 512 tokens regardless of image size)
- **Advantage**: Computational cost is `O(Q × V)` not `O(V²)` where Q << V

**Flamingo (DeepMind, 2022)**

From [Flamingo: a Visual Language Model for Few-Shot Learning](https://proceedings.neurips.cc/paper_files/paper/2022/file/960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf) (accessed 2025-01-31):

- **Architecture**: Gated cross-attention layers interleaved with frozen LM layers
- **Key Innovation**: Gating mechanism controls information flow from vision to language
  ```
  Tanh_Gate = tanh(W_gate @ Visual_Features)
  Output = LM_Hidden + Tanh_Gate ⊙ Cross_Attention(LM_Hidden, Visual_Features)
  ```
- **Benefits**:
  - Gradual integration of visual information into language model
  - Preserves pre-trained LM capabilities during vision-language alignment
  - Gate values can be initialized to zero, allowing stable training
- **Use Case**: Interleaved image-text sequences (video understanding, multi-image reasoning)

**Implementation Pattern**:
```python
# Flamingo-style gated cross-attention
visual_context = cross_attention(
    query=language_hidden_states,
    key_value=visual_features
)
gate = tanh(gate_projection(visual_features))
output = language_hidden_states + gate * visual_context
```

### 2. Learned Query Approaches

**Q-Former (BLIP-2, Salesforce, 2023)**

From [BLIP-2: Bootstrapping Language-Image Pre-training](https://proceedings.mlr.press/v202/li23q/li23q.pdf) (accessed 2025-01-31):

- **Architecture**: Querying Transformer with learnable query tokens
- **Dual Attention**:
  - **Self-attention**: Query tokens attend to each other (share information)
  - **Cross-attention**: Query tokens attend to frozen vision encoder output
- **Key Features**:
  - Fixed number of learned queries (e.g., 32 query tokens)
  - Queries are task-agnostic but image-aware
  - Bootstraps from frozen vision encoder + frozen LLM
- **Training Strategy**:
  1. Vision-language alignment (Q-Former learns to extract visual features)
  2. Vision-to-language generative learning (Q-Former connects to LLM)

**Advantages**:
- Decouples vision encoder from LLM (both can remain frozen)
- Learned queries become "optimal questions to ask about an image"
- Efficient: Processes any image size → fixed number of tokens

**Comparison to Perceiver**:
- **Perceiver**: Queries are purely learned (task-agnostic, image-agnostic)
- **Q-Former**: Queries interact with image via cross-attention (image-aware)
- **Q-Former**: Additional self-attention among queries (information sharing)

From [Vision Language Models - Rohit Bandaru](https://rohitbandaru.github.io/blog/Vision-Language-Models/) (accessed 2025-01-31):
> "The Q-Former is similar to the Perceiver in that it processes images with a fixed number of learned query embeddings. However, it also takes in text input and uses self-attention among queries, making it more flexible for vision-language tasks."

### 3. Dynamic Routing Approaches

**Task-Aware Dynamic Transformer (TADT)**

From [Task-Aware Dynamic Transformer for Efficient Arbitrary-Scale Image Super-Resolution](https://arxiv.org/html/2408.08736v1) (arXiv:2408.08736, accessed 2025-01-31):

- **Architecture**: Controller network predicts routing vectors based on task/input
- **Mechanism**: Selects which self-attention branches to activate per layer
- **Process**:
  ```
  Task_Query + Image_Features → TARC_Controller → Routing_Vector
  Routing_Vector → Select_Attention_Branches → Adaptive_Feature_Extraction
  ```
- **Benefits**:
  - Computational efficiency (only activate necessary branches)
  - Task-specific visual processing
  - Scales to arbitrary resolutions without retraining

**Key Concept**: The controller learns "which parts of the network are relevant for this task" rather than "which parts of the image are relevant."

---

## Benefits Analysis

### 1. Computational Efficiency

**Token Reduction**:
- Raw image patches: 1024×1024 image with 16×16 patches = 4,096 tokens
- After query-conditioned compression: 32-512 tokens (8-128× reduction)
- Quadratic attention savings: `O(4096²)` → `O(512²)` = 64× faster

**Memory Footprint**:
- Standard ViT: Must process full token sequence through all layers
- Query-conditioned: Fixed token budget regardless of input size
- Enables processing ultra-high-resolution images within LLM context windows

From [GTA: A Geometry-Aware Attention Mechanism](https://arxiv.org/abs/2310.10375) (arXiv:2310.10375, accessed 2025-01-31):
> "Geometry-aware attention encodes the geometric structure of tokens as relative transformation, enabling more efficient processing of visual data compared to uniform attention."

### 2. Task-Specific Focus

**Relevance Realization**:
- VQA "What color is the car?" → Focus on car regions, color-relevant features
- Image captioning → Broader scene understanding, object relationships
- Object detection → Boundary-precise encoding, spatial relationships

**Adaptive Resolution**:
- Important regions: High token allocation (detailed encoding)
- Background regions: Low token allocation (compressed encoding)
- Similar to human foveal vision (high acuity center, low acuity periphery)

### 3. Improved Vision-Language Alignment

**Semantic Bridging**:
- Learned queries act as "questions that vision encoders should answer"
- Cross-attention learns which visual features matter for language understanding
- Gating mechanisms preserve LM capabilities while integrating vision

**Few-Shot Learning**:
From Flamingo paper (accessed 2025-01-31):
- Query-conditioned models excel at rapid adaptation (few examples)
- Visual context can guide language generation without full fine-tuning
- Interleaved vision-text enables in-context learning for multimodal tasks

---

## Relevance Realization Parallels

Query-conditioned attention implements several Vervaekean principles:

### Transjective Knowing
- **Not Objective**: Visual importance isn't fixed in the image
- **Not Subjective**: Query alone doesn't determine relevance
- **Transjective**: Relevance emerges from query-image coupling
- Example: "What time is it?" makes clocks relevant; "What's the weather?" makes sky/windows relevant

### Opponent Processing
**Particularize ↔ Compress**:
- Query specifies what to particularize (car color → focus on car)
- Budget forces compression (32 tokens must capture whole scene)
- Cross-attention balances these tensions

**Exploit ↔ Explore**:
- Exploit: Focus on query-relevant regions (known importance)
- Explore: Maintain peripheral awareness (unexpected relevance)
- Gating learns when to exploit visual features vs. rely on language priors

### Procedural Knowing (How)
**Learned Query Optimization**:
- Q-Former learns "how to ask visual questions"
- Gating learns "when to integrate visual information"
- Routing learns "which processing paths to activate"

This is the fourth P - not propositional/perspectival/participatory, but **procedural skills** for efficient relevance realization.

---

## Implementation Patterns

### Pattern 1: Fixed Learned Queries (Perceiver-Style)

```python
class LearnedQueryAttention(nn.Module):
    def __init__(self, num_queries=512, dim=768):
        self.queries = nn.Parameter(torch.randn(num_queries, dim))
        self.cross_attention = CrossAttention(dim)

    def forward(self, visual_features):
        # visual_features: [batch, num_patches, dim]
        # queries: [num_queries, dim]

        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        compressed = self.cross_attention(
            query=queries,
            key=visual_features,
            value=visual_features
        )
        return compressed  # [batch, num_queries, dim]
```

**Characteristics**:
- Query-conditioned by learned embeddings (not explicit text)
- Fixed compression ratio
- Task-agnostic (learns general visual summarization)

### Pattern 2: Text-Conditioned Queries (Q-Former Style)

```python
class TextConditionedQueries(nn.Module):
    def __init__(self, num_queries=32, dim=768):
        self.queries = nn.Parameter(torch.randn(num_queries, dim))
        self.self_attn = SelfAttention(dim)
        self.cross_attn = CrossAttention(dim)

    def forward(self, visual_features, text_embeddings=None):
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Optional: Modulate queries with text
        if text_embeddings is not None:
            queries = queries + self.text_projection(text_embeddings)

        # Self-attention among queries (information sharing)
        queries = self.self_attn(queries)

        # Cross-attention to visual features
        output = self.cross_attn(
            query=queries,
            key=visual_features,
            value=visual_features
        )
        return output
```

**Characteristics**:
- Can incorporate text conditioning
- Self-attention allows query cooperation
- More flexible than pure learned queries

### Pattern 3: Gated Integration (Flamingo-Style)

```python
class GatedCrossAttention(nn.Module):
    def __init__(self, dim=768):
        self.cross_attn = CrossAttention(dim)
        self.gate_proj = nn.Linear(dim, dim)
        self.alpha = nn.Parameter(torch.zeros(1))  # Learnable gate scaling

    def forward(self, language_hidden, visual_features):
        # Cross-attend from language to vision
        visual_context = self.cross_attn(
            query=language_hidden,
            key=visual_features,
            value=visual_features
        )

        # Compute gates (controls information flow)
        gate = torch.tanh(self.gate_proj(visual_features.mean(dim=1)))
        gate = gate.unsqueeze(1) * self.alpha  # [batch, 1, dim]

        # Gated residual connection
        output = language_hidden + gate * visual_context
        return output
```

**Characteristics**:
- Preserves language model hidden states
- Learnable information flow (gate can start at zero)
- Gradual vision-language integration

---

## Practical Considerations

### When to Use Each Approach

**Perceiver/Learned Queries**:
- When you need consistent compression ratio
- Task-agnostic visual encoding
- Minimal architectural changes to LLM

**Q-Former/Text-Conditioned**:
- When you have explicit text queries
- Need task-specific visual encoding
- Want to keep vision encoder and LLM frozen

**Flamingo/Gated Cross-Attention**:
- Interleaved vision-text sequences
- Preserving pre-trained LM capabilities critical
- Few-shot learning scenarios

**Dynamic Routing**:
- Computational efficiency critical
- Multi-task scenarios (different tasks need different processing)
- Arbitrary input resolutions

### Training Strategies

From BLIP-2 paper (accessed 2025-01-31):

**Two-Stage Approach**:
1. **Vision-Language Alignment**: Freeze vision encoder, train Q-Former with image-text contrastive loss
2. **Generative Learning**: Freeze LLM, train Q-Former to generate captions/answers

**Benefits**:
- Leverages pre-trained models (no catastrophic forgetting)
- Efficient (only train lightweight Q-Former)
- Scalable (can swap vision encoders or LLMs)

### Debugging Query Conditioning

**Attention Visualization**:
```python
# Visualize which visual tokens each query attends to
attn_weights = cross_attention.get_attention_weights()
# attn_weights: [batch, num_queries, num_visual_tokens]

for query_idx in range(num_queries):
    visualize_attention_map(image, attn_weights[0, query_idx])
```

**Query Specialization**:
- Check if different queries focus on different aspects (objects, colors, spatial relationships)
- Analyze query similarity (too similar = redundant queries)
- Monitor gate values (are visual features being integrated?)

---

## Sources

**Research Papers**:
- [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206) - arXiv:2103.03206 (accessed 2025-01-31)
- [Flamingo: a Visual Language Model for Few-Shot Learning](https://proceedings.neurips.cc/paper_files/paper/2022/file/960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf) - NeurIPS 2022 (accessed 2025-01-31)
- [BLIP-2: Bootstrapping Language-Image Pre-training](https://proceedings.mlr.press/v202/li23q/li23q.pdf) - ICML 2023, 8,475 citations (accessed 2025-01-31)
- [Task-Aware Dynamic Transformer for Efficient Arbitrary-Scale Image Super-Resolution](https://arxiv.org/html/2408.08736v1) - arXiv:2408.08736 (accessed 2025-01-31)
- [GTA: A Geometry-Aware Attention Mechanism](https://arxiv.org/abs/2310.10375) - arXiv:2310.10375, cited by 22 (accessed 2025-01-31)
- [Vision-Language Transformer and Query Generation for Referring Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Ding_Vision-Language_Transformer_and_Query_Generation_for_Referring_Segmentation_ICCV_2021_paper.pdf) - ICCV 2021, cited by 364 (accessed 2025-01-31)
- [Matryoshka Query Transformer for Large Vision-Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/59c147c7d4fdb732daea3064eab949bf-Paper-Conference.pdf) - NeurIPS 2024, cited by 34 (accessed 2025-01-31)

**Technical Resources**:
- [Vision Language Models - Rohit Bandaru](https://rohitbandaru.github.io/blog/Vision-Language-Models/) (accessed 2025-01-31)
- [BLIP-2 - Hugging Face Documentation](https://huggingface.co/docs/transformers/en/model_doc/blip-2) (accessed 2025-01-31)
- [Design choices for Vision Language Models in 2024](https://huggingface.co/blog/gigant/vlm-design) - Hugging Face Blog (accessed 2025-01-31)
- [Understanding DeepMind's Flamingo Visual Language Models](https://medium.com/@paluchasz/understanding-flamingo-visual-language-models-bea5eeb05268) - Medium (accessed 2025-01-31)

**Additional References**:
- [Aligning Vision-Language Models with User's Gaze Attention](https://proceedings.neurips.cc/paper_files/paper/2024/file/03738e5f26967582eeb3b57eef82f1f0-Paper-Conference.pdf) - NeurIPS 2024, cited by 20 (accessed 2025-01-31)
- [EVLM: An Efficient Vision-Language Model for Visual Understanding](https://arxiv.org/html/2407.14177v1) - arXiv:2407.14177 (accessed 2025-01-31)
