# Query-Conditioned Visual Attention

## Overview - Task-Aware Vision Processing

Query-conditioned visual attention represents a fundamental shift in how vision-language models (VLMs) process images. Instead of encoding visual information uniformly, these mechanisms allow the visual encoder to be aware of and responsive to the text query, focusing computational resources on task-relevant image regions.

This approach addresses a core inefficiency in naive VLM architectures: when answering "What color is the car?", the model doesn't need equal detail about the background sky or distant trees. Query-conditioned attention enables dynamic, task-aware visual processing that adapts based on what the user is asking.

**Key principle**: Visual encoding should be query-dependent, not query-agnostic. The text query guides which visual features are extracted and how they're represented.

From [Attention Prompting on Image for Large Vision-Language Models](https://arxiv.org/abs/2409.17143) (arXiv:2409.17143, accessed 2025-01-31):
> "We propose a new prompting technique named Attention Prompting on Image, which just simply overlays a text-query-guided attention heatmap on the original input image and effectively enhances LVLM on various tasks."

From [GazeLLM: Multimodal LLMs incorporating Human Visual Attention](https://dl.acm.org/doi/10.1145/3745900.3746075) (ACM AHs '25, accessed 2025-01-31):
> "This visual attention mechanism allows humans to balance the precision of world understanding with the efficiency of processing visual input... By processing these selectively focused inputs, we demonstrate that our approach achieves comprehension equivalent to or even better than processing the entire image at full resolution, but with significantly reduced data input."

## Mechanism Taxonomy

### 1. Cross-Attention Based Approaches

**Perceiver Architecture** uses cross-attention with learned latent queries:

```
Visual Features (N tokens) → Cross-Attention ← Learned Queries (M tokens)
                                    ↓
                            Compressed Representation (M tokens)
```

- **Asymmetric processing**: M << N (typically 64-256 learned queries vs thousands of visual tokens)
- **Learned queries**: Fixed set of learnable embeddings that attend over visual features
- **General architecture**: Not text-conditioned initially, but extended for VLMs

From [Perceiver Architecture](https://scholar.google.com/scholar?q=Perceiver+architecture+cross-attention+learned+queries) (accessed 2025-01-31):
> "Cross-attention to learned latents... uses learnt queries and attention to select the salient visual features"

**Flamingo's Gated Cross-Attention** integrates visual information into frozen language models:

```
Text Tokens → Self-Attention → Gated Cross-Attention ← Visual Tokens
                                        ↓
                                Modulated Text Features
```

- **Interleaved architecture**: Cross-attention layers inserted between LM self-attention blocks
- **Gating mechanism**: Tanh gating allows gradual learning (starts near-zero, preserves pretrained LM)
- **Text-conditional**: Text queries attend over visual features from Perceiver Resampler

From [Flamingo: a Visual Language Model for Few-Shot Learning](https://proceedings.neurips.cc/paper_files/paper/2022/file/960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf) (NeurIPS 2022, accessed 2025-01-31):
> "Interleave cross-attention layers with frozen pre-trained language self-attention layers... Gated cross-attention, allowing flamingo to slowly learn to inject image information into the language model"

### 2. Learned Query Tokens (Q-Former)

BLIP-2's **Q-Former** introduces a specialized querying transformer:

```
Frozen Vision Encoder → Visual Features
                            ↓
                    Q-Former (32 queries)
                      ↓           ↓
              Self-Attention  Cross-Attention
                      ↓
              Frozen LLM Input
```

- **Dual attention**: Queries use both self-attention (among themselves) and cross-attention (to visual features)
- **Bootstrapping**: Connects frozen vision encoder to frozen LLM via learnable Q-Former
- **Fixed query count**: Typically 32 learned query embeddings per image

From [Design choices for Vision Language Models in 2024](https://huggingface.co/blog/gigant/vlm-design) (HuggingFace, accessed 2025-01-31):
> "Q-Former (BLIP-2) or the Perceiver Resampler (Flamingo) abstractors. Both are using learnt queries and attention to select the salient visual features"

**Key difference from Perceiver**: Q-Former queries are trained specifically for vision-language alignment, not just compression.

### 3. Attention Prompting on Images

Recent approaches use **query-guided attention heatmaps** overlaid on input images:

```
Text Query → CLIP Text Encoder
                ↓
        Similarity Computation
                ↓
    Attention Heatmap (spatial)
                ↓
    Multiply with Original Image
                ↓
        Query-Focused Input
```

From [Attention Prompting on Image](https://arxiv.org/abs/2409.17143) (arXiv 2024, accessed 2025-01-31):
> "We generate an attention heatmap for the input image dependent on the text query with an auxiliary model like CLIP. Then the heatmap simply multiplies the pixel values of the original image to obtain the actual input image for the LVLM."

**Results**: Improved LLaVA-1.5 by 3.8% on MM-Vet and 2.9% on LLaVA-Wild benchmarks.

### 4. Gaze-Based Selective Processing

**GazeLLM** uses human visual attention to guide image cropping:

```
First-Person Video + Eye Tracking
            ↓
    Gaze-Centered Crop (448×448)
            ↓
    10% of original pixels
            ↓
    Equivalent or better task understanding
```

From [GazeLLM](https://dl.acm.org/doi/10.1145/3745900.3746075) (ACM AHs '25, accessed 2025-01-31):
> "User evaluations demonstrated that utilizing partial video cropped around the gaze point from 1st-person vision achieves task descriptions of equal or superior quality to full-sized videos, despite using only one-tenth of the pixels."

**Key insight**: Human gaze naturally focuses on task-relevant regions. Gaze-conditioned cropping mirrors this biological attention mechanism.

## Benefits Analysis

### Computational Efficiency

**Token reduction**:
- Perceiver: N visual tokens → M query tokens (M << N)
- Q-Former: Arbitrary visual tokens → 32 fixed query tokens
- GazeLLM: 1440×1440 → 448×448 (10.3% of pixels)

**Memory savings**:
- Standard ViT: O(N²) attention cost for N visual tokens
- Query-based: O(M·N) where M is query count
- Typical reduction: 1000+ tokens → 32-256 tokens

### Task-Specific Focus

Query conditioning allows the model to:
1. **Prioritize relevant regions**: Text query "red car" focuses on automotive objects and color
2. **Ignore distractors**: Background elements receive less processing
3. **Adapt granularity**: Complex queries may require more detailed visual encoding

From [Task-Aware Visual Encoding](https://scholar.google.com/scholar?q=task-aware+visual+encoding+attention+transformers) (accessed 2025-01-31):
> "Task-aware feature composition... adjusting the density of timelines for efficient browsing of first-person videos"

### Relevance Realization Parallels

Query-conditioned attention naturally aligns with Vervaekean relevance realization:

**Participatory knowing**: Query-content coupling (shark's fitness for ocean, not shark alone or ocean alone)

**Transjective relevance**: Visual features are neither purely objective (in image) nor subjective (in query) but emerge from their relationship

**Opponent processing**: Balance between:
- Exploit (focus on query-relevant regions) ↔ Explore (maintain context awareness)
- Compress (reduce tokens) ↔ Particularize (preserve details)

## Implementation Patterns

### Pattern 1: Frozen-Frozen Bridge (BLIP-2)

```python
# Pseudocode
frozen_visual_features = vision_encoder(image)  # Don't update
query_embeddings = nn.Parameter(torch.randn(32, hidden_dim))  # Learn these

# Q-Former with dual attention
for layer in q_former_layers:
    # Self-attention among queries
    queries = self_attention(query_embeddings, query_embeddings)
    # Cross-attention to visual features
    queries = cross_attention(queries, frozen_visual_features)

llm_input = queries  # Feed to frozen LLM
```

**Advantage**: Preserve both vision and language pretraining, only train bridge.

### Pattern 2: Gated Integration (Flamingo)

```python
# Pseudocode
visual_tokens = perceiver_resampler(image_features)  # Compress to ~64 tokens

for text_token_i in text_sequence:
    # Standard LM self-attention
    hidden = self_attention(text_token_i, context)

    # Gated cross-attention to visual tokens
    visual_info = cross_attention(hidden, visual_tokens)
    gate = tanh(gate_params)  # Starts near 0
    hidden = hidden + gate * visual_info
```

**Advantage**: Gradual integration, preserves LM capabilities.

### Pattern 3: Attention Heatmap Weighting

```python
# Pseudocode using CLIP
text_features = clip_text_encoder(query)
image_features = clip_image_encoder(image)  # Spatial feature map

# Compute similarity at each spatial location
similarity_map = cosine_similarity(text_features, image_features)
attention_heatmap = softmax(similarity_map / temperature)

# Weight original image
weighted_image = image * attention_heatmap.unsqueeze(-1)

# Feed to VLM
output = vlm(weighted_image, query)
```

**Advantage**: Simple, interpretable, no architecture changes needed.

### Pattern 4: Dynamic Routing

Task-aware routing selects different visual processing paths:

```python
# Pseudocode
task_embedding = encode_task(query)

# Route to task-specific encoders
if is_ocr_task(task_embedding):
    visual_features = high_res_encoder(image)
elif is_scene_task(task_embedding):
    visual_features = wide_fov_encoder(image)
else:
    visual_features = default_encoder(image)
```

**Advantage**: Specialized processing per task type.

## Practical Considerations

### When to Use Query Conditioning

**Strong cases**:
- VQA (Visual Question Answering) - query directly determines what to look for
- Instruction-following - "Find the red car" vs "Describe the scene"
- Resource-constrained deployment - reduce tokens for faster inference

**Weak cases**:
- Open-ended captioning - no specific query to condition on
- Image classification - single label, uniform processing may suffice
- Aesthetic evaluation - requires holistic image understanding

### Training Strategies

1. **Two-stage training** (BLIP-2 approach):
   - Stage 1: Train Q-Former with vision-language alignment objectives
   - Stage 2: Train generative capabilities with frozen Q-Former

2. **Gradual unfreezing**:
   - Start with frozen vision + frozen LM
   - Train bridge (Q-Former, gated attention)
   - Optionally unfreeze LM adapter layers

3. **Multi-task learning**:
   - Train on diverse query types simultaneously
   - Ensures query-conditioning generalizes across tasks

### Common Pitfalls

**Over-compression**: Too few query tokens loses critical details
- Solution: Task-dependent query count (OCR needs more tokens than scene classification)

**Query-image mismatch**: Training on generic captions doesn't teach specific query conditioning
- Solution: Use instruction-tuning datasets with diverse question types

**Attention collapse**: All queries attend to same visual regions
- Solution: Diversity regularization, orthogonality constraints on queries

## Comparison to Standard Attention

| Mechanism | Resolution | Query Source | Adaptability |
|-----------|-----------|--------------|--------------|
| **Standard ViT** | Uniform (all patches equal) | None (self-attention) | Fixed |
| **Perceiver** | Compressed (M queries) | Learned latents | Architecture-level |
| **Q-Former** | Compressed (32 queries) | Learned + query-aware | Trained for VL tasks |
| **Flamingo** | Compressed (64 tokens) | Perceiver + gated text | Per-token gating |
| **Attention Prompting** | Spatially weighted | Text-query heatmap | Per-query dynamic |
| **GazeLLM** | Gaze-focused crop | Human visual attention | Biologically inspired |

**Key distinction**: Standard transformer attention is a *mechanism* (QKV softmax). Query-conditioned attention is a *process* that realizes relevance through query-content coupling.

## Future Directions

### Hybrid Approaches

Combine multiple conditioning strategies:
```
Central vision (high-res, query-focused)
    +
Peripheral vision (low-res, context-aware)
    +
Gated integration (task-dependent weighting)
```

From [GazeLLM Discussion](https://dl.acm.org/doi/10.1145/3745900.3746075):
> "An approach that provides both gaze-centered video and a downsampled full view, enabling comprehension of off-gaze areas at a lower resolution. This approach corresponds to the relationship between the central and peripheral fields in human vision."

### Adaptive Query Count

Dynamic query allocation based on:
- Query complexity ("describe" vs "what color is X?")
- Image complexity (cluttered scenes need more queries)
- Available compute budget

### Multi-modal Query Conditioning

Extend beyond text queries:
- Audio-conditioned visual attention (focus on sound source)
- Prior-image-conditioned attention (spot differences)
- Temporal-conditioned attention (track object through video)

## Sources

**Primary Papers**:
- [Attention Prompting on Image for Large Vision-Language Models](https://arxiv.org/abs/2409.17143) - arXiv:2409.17143 (accessed 2025-01-31)
- [GazeLLM: Multimodal LLMs incorporating Human Visual Attention](https://dl.acm.org/doi/10.1145/3745900.3746075) - ACM AHs '25 (accessed 2025-01-31)
- [Flamingo: a Visual Language Model for Few-Shot Learning](https://proceedings.neurips.cc/paper_files/paper/2022/file/960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf) - NeurIPS 2022 (accessed 2025-01-31)

**Web Research**:
- [Design choices for Vision Language Models in 2024](https://huggingface.co/blog/gigant/vlm-design) - HuggingFace (accessed 2025-01-31)
- [Vision Language Models - Rohit Bandaru](https://rohitbandaru.github.io/blog/Vision-Language-Models/) (accessed 2025-01-31)
- Task-aware visual encoding searches (Google Scholar, accessed 2025-01-31)

**Additional References**:
- Perceiver architecture papers and scholarly articles
- BLIP-2 Q-Former documentation
- Cross-attention mechanism tutorials and guides
