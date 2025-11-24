# Vision-Language Model Token Concatenation Strategies

## Overview

Vision-language models (VLMs) face a fundamental architectural decision: how to combine visual tokens from images with text tokens in a unified sequence that can be processed by a transformer. This decision affects model performance, training efficiency, inference speed, and the types of tasks the model can handle effectively.

**What is Token Concatenation in VLMs?**

Token concatenation refers to the process of merging visual tokens (extracted from images via a vision encoder) with text tokens (from a language model tokenizer) into a single sequence that can be fed into a transformer architecture. The key challenge is that visual and textual modalities have fundamentally different characteristics:

- **Visual tokens**: High-dimensional, spatially structured, typically 256-576 tokens per image (e.g., 16×16 or 24×24 patches)
- **Text tokens**: Low-dimensional, sequential, semantically discrete

The concatenation strategy determines:
1. **Where** visual and text tokens are merged (input level, intermediate layers, or via cross-attention)
2. **How** they are aligned (direct concatenation, learned projection, resampling)
3. **When** they interact (early fusion, late fusion, or continuous interaction)

**Why Concatenation Order Matters**

The order and method of concatenation directly impacts:
- **Context window utilization**: With limited context (e.g., 2048 tokens), inefficient visual token usage reduces text capacity
- **Attention efficiency**: Quadratic complexity means every visual token increases computational cost
- **Cross-modal reasoning**: The ability of the model to relate visual and textual information
- **Training stability**: Different strategies have different gradient flow characteristics
- **Task performance**: Some tasks (OCR, fine-grained VQA) need more visual tokens; others (high-level reasoning) need fewer

**Historical Context**

The evolution of token concatenation strategies reflects the broader development of VLMs:

1. **CLIP Era (2021)**: Contrastive learning with separate encoders for vision and text
2. **Early VLMs (2019-2021)**: Direct concatenation approaches (VisualBERT, LXMERT)
3. **Efficient Fusion (2022)**: Cross-attention and resampling methods (Flamingo, BLIP-2)
4. **Modern Architectures (2023-2024)**: Adaptive strategies, dynamic token allocation, native resolution support

From [Design choices for Vision Language Models in 2024](https://huggingface.co/blog/gigant/vlm-design) (HuggingFace, accessed 2025-01-31):
> "Vision and language models are the new shiny thing in the AI space, delivering mind-blowing results at a very fast pace. One thing we can learn from all these different models is the choices that were made and the results they yield."

## Early Fusion Strategies

Early fusion concatenates visual and text tokens at the input level, before or at the very beginning of the transformer processing. This is the simplest approach architecturally.

### Direct Concatenation (LLaVA Family)

**Architecture**: Visual tokens are linearly projected to match text embedding dimensions, then concatenated with text tokens as a unified sequence.

**LLaVA** (Large Language and Vision Assistant; Liu et al. 2023) pioneered this minimalist approach:
- Vision encoder: CLIP ViT-L/14 (336×336) → 576 tokens
- Projection: Simple MLP (2 layers in LLaVA-1.5)
- Concatenation: `[visual_tokens, text_tokens]` fed to frozen LLM
- Training: Two-stage (alignment pretraining, then instruction tuning)

From [Generalized Visual Language Models](https://lilianweng.github.io/posts/2022-06-09-vlm/) (Lilian Weng, 2022):
> "One straightforward approach to fuse visual information into language models is to treat images as normal text tokens and train the model on a sequence of joint representations of both text and images."

**Advantages**:
- **Simplicity**: Minimal architectural changes to pretrained LLMs
- **Flexibility**: Easy to support interleaved image-text sequences
- **No information loss**: All visual tokens are preserved
- **Training efficiency**: Only projection layer needs training initially

**Disadvantages**:
- **Context window pressure**: 576 tokens per image consumes significant context
- **Redundancy**: Adjacent image patches often contain similar information
- **Computational cost**: Quadratic attention over all visual tokens
- **Scaling challenges**: Multiple images or video frames multiply token count

**Example Token Flow** (LLaVA-1.5 with 7B LLM):
```
Input: [Image] + "What is in this image?"
↓
Vision Encoder (CLIP ViT-L/14): 336×336 → 576 visual tokens
↓
MLP Projection: 576 × 1024 (CLIP dim) → 576 × 4096 (LLaMA dim)
↓
Concatenation: [576 visual tokens, 6 text tokens]
↓
LLaMA-7B: Process 582 tokens total
↓
Output: "The image shows..."
```

**Performance Characteristics**:
- VQAv2: ~78-80% accuracy (LLaVA-1.5)
- Strong on compositional reasoning (all visual information available)
- Inference: ~1.5-2x slower than text-only due to long visual prefix

### Prefix Language Modeling (SimVLM)

**SimVLM** (Simple Visual Language Model; Wang et al. 2022) treats images as prefix sequences with bidirectional attention, while text has causal attention.

**Architecture**:
- Images: Bidirectional prefix (can attend to all image tokens)
- Text: Causal suffix (can attend to image + previous text only)
- Training: Mixed batches of image-text pairs (ALIGN) and text-only (C4)

From [Window Token Concatenation for Efficient Visual Large Language Models](https://arxiv.org/abs/2504.04024) (arXiv:2504.04024, 2025):
> "Visual Large Language Models (VLLMs) face a fundamental challenge: the large number of visual tokens significantly increases computational costs and inference time."

**Key Innovation**: PrefixLM objective combines:
- **Bidirectional vision encoding**: Rich cross-patch interactions
- **Autoregressive text generation**: Preserves language modeling capability
- **Mixed training**: Maintains language quality while learning vision

**Advantages**:
- Better vision understanding (bidirectional prefix)
- Maintains strong language generation (causal text)
- Scales to large datasets effectively

**Disadvantages**:
- Still uses all visual tokens (no compression)
- More complex training setup than pure concatenation
- Requires careful balancing of image-text vs text-only data

### Joint Training from Scratch (CM3, BeiT-3)

**CM3** (Causally-Masked Multimodal Modeling; Aghajanyan et al. 2022) and **BeiT-3** treat images as a foreign language, tokenizing them with VQVAE and processing alongside text.

**CM3 Architecture**:
- Images tokenized: 256×256 → 256 tokens (via VQVAE-GAN)
- HTML-like markup: `<img src="[256 image tokens]">`
- Causal + masked: Generate masked spans at sequence end
- Training: ~1T tokens of web data (HTML, images, text)

**BeiT-3 Architecture**:
- Unified ViT backbone for both vision and language
- Modality experts: Different FFN layers per modality
- Shared self-attention: Joint vision-language processing

**Advantages**:
- **End-to-end learning**: No dependence on pretrained components
- **True multimodal understanding**: Single model processes both modalities
- **Flexible generation**: Can generate both text and images

**Disadvantages**:
- **Massive data requirements**: ~1T tokens needed for convergence
- **Training cost**: Orders of magnitude more expensive than adaptation
- **Image quality**: VQVAE tokenization loses visual fidelity
- **Fixed resolution**: 256×256 limitation from discrete tokenization

**Performance Comparison** (from various papers):
```
LLaVA-1.5 (7B):    VQAv2 78.5%, trained on ~600K examples
SimVLM (Large):    VQAv2 77.9%, trained on ~1.8B examples
CM3 (13B):         VQAv2 ~72%, trained on ~1T tokens
```

The efficiency gap highlights the value of leveraging pretrained models rather than training from scratch.

## Late Fusion Strategies

Late fusion maintains separate processing pipelines for vision and text, merging them at higher layers or through specialized attention mechanisms. This allows each modality to develop rich representations before interaction.

### Cross-Attention Fusion (Flamingo)

**Flamingo** (Alayrac et al. 2022) pioneered efficient cross-attention fusion with frozen pretrained models.

**Architecture**:
- **Vision encoder**: CLIP ViT (frozen)
- **Perceiver Resampler**: Reduces visual tokens from 256-1024 → 64 fixed tokens
- **Gated cross-attention layers**: Interleaved between frozen LLM layers
- **Text attention**: Causal self-attention (standard autoregressive)

From [Generalized Visual Language Models](https://lilianweng.github.io/posts/2022-06-09-vlm/) (Lilian Weng, 2022):
> "To easily handle text with interleaved images, masking in Flamingo is designed such that text token only cross-attends to visual tokens corresponding to the last preceding image, largely reducing the number of visual tokens that a certain text token can see."

**Perceiver Resampler**:
```python
# Conceptual architecture
visual_features = vision_encoder(image)  # Shape: [256, 1024]
learnable_queries = nn.Parameter(torch.randn(64, 1024))  # Fixed 64 tokens

# Cross-attention: queries attend to visual features
resampled = cross_attention(
    Q=learnable_queries,    # [64, 1024]
    K=visual_features,      # [256, 1024]
    V=visual_features       # [256, 1024]
)  # Output: [64, 1024] - compressed visual representation
```

**Gated Cross-Attention**:
- Inserted between frozen LLM layers (e.g., every 4th layer)
- Tanh gating: Allows model to learn how much visual info to incorporate
- Text can attend to visual tokens from preceding images only
- Gradual fusion: Each layer can integrate visual information differently

**Interleaved Image-Text Handling**:
```
Input sequence: "Look at <image1>. Now see <image2>. What changed?"

Token attention patterns:
- "Look": attends to nothing (no preceding image)
- "at": attends to nothing
- "<image1>": N/A (visual tokens, not text)
- "Now": attends to image1 visual tokens
- "see": attends to image1 visual tokens
- "<image2>": N/A (visual tokens, not text)
- "What": attends to image2 visual tokens only (last preceding)
- "changed": attends to image2 visual tokens only
```

**Advantages**:
- **Token efficiency**: 64 fixed tokens per image vs 256-576 in LLaVA
- **Frozen LM**: Preserves language capabilities, faster training
- **Scalable**: Handles multiple images without token explosion
- **Few-shot learning**: Natural support for in-context examples with images

**Disadvantages**:
- **Fine-grained loss**: Resampling may discard important visual details
- **Training complexity**: Must learn resampler + cross-attention layers
- **Architectural coupling**: Cross-attention layers are model-specific
- **Inference overhead**: Additional forward passes through cross-attention

**Performance** (Flamingo-80B):
- VQAv2 (4-shot): 82.0%
- OKVQA (4-shot): 62.7%
- TextVQA (4-shot): 54.1%

### Query-Based Resampling (BLIP-2)

**BLIP-2** (Li et al. 2023) introduces Q-Former, a learnable query-based bottleneck that connects vision and language.

**Q-Former Architecture**:
- **Learnable queries**: 32 learned vectors (queries)
- **Dual attention**: Self-attention among queries + cross-attention to image features
- **Text conditioning**: Optional text input to guide query selection
- **Output**: 32 visual tokens that work with frozen LLM

**Three-Stage Training**:
1. **Vision-language representation learning**:
   - ITC (image-text contrastive): Align queries with text via CLIP-style loss
   - ITG (image-grounded text generation): Generate text conditioned on queries
   - ITM (image-text matching): Binary classification of matched pairs

2. **Vision-to-language generative learning**:
   - Connect Q-Former to frozen LLM via linear projection
   - Train on caption generation tasks

3. **Instruction tuning** (optional):
   - Fine-tune on instruction-following datasets

From [Design choices for Vision Language Models in 2024](https://huggingface.co/blog/gigant/vlm-design):
> "Q-Former is also conditioned on input text... Both are using learnt queries and attention to select the salient visual information for a given token budget, but Q-Former is also conditioned on input text."

**Text-Conditioned Resampling**:
```python
# Q-Former can take optional text input
text_query = "What color is the car?"
learnable_queries = nn.Parameter(torch.randn(32, 768))  # 32 learned queries

# Self-attention among queries (with text guidance)
queries_with_text = self_attention(
    torch.cat([learnable_queries, text_embedding], dim=0)
)

# Cross-attention to image features
visual_tokens = cross_attention(
    Q=queries_with_text[:32],  # Keep only query tokens
    K=image_features,
    V=image_features
)  # Output: 32 text-conditioned visual tokens
```

**Advantages**:
- **Text-aware compression**: Queries can focus on task-relevant visual features
- **Extreme efficiency**: Only 32 tokens per image
- **Modular design**: Q-Former can connect any vision encoder to any LLM
- **Strong performance**: Matches or exceeds larger models with fewer tokens

**Disadvantages**:
- **Training complexity**: Three-stage training is computationally expensive
- **Text dependency**: May not work well when text context is minimal
- **Information bottleneck**: 32 tokens may be insufficient for fine-grained tasks
- **Query learning**: Requires substantial data to learn good query patterns

**Performance** (BLIP-2 with Flan-T5-XXL):
- VQAv2: 81.9%
- OKVQA: 57.5%
- TextVQA: 61.3% (better than Flamingo on text-heavy tasks)

### Token Fusion with Adaptive Compression (TokenFusion, WiCo)

Modern approaches dynamically compress visual tokens based on spatial redundancy.

**WiCo** (Window Token Concatenation; Li et al. 2025) uses sliding window concatenation:

From [Window Token Concatenation for Efficient Visual Large Language Models](https://arxiv.org/abs/2504.04024):
> "We employ a sliding window to concatenate spatially adjacent visual tokens. However, directly concatenating these tokens may group diverse tokens into one, and thus obscure some fine details."

**WiCo Architecture**:
```
Original: 24×24 = 576 tokens per image
↓
Sliding Window (2×2): Group 4 adjacent tokens
↓
Concatenation: 576 tokens → 144 concatenated features
↓
Optional: WiCo+ decomposes in later LLM layers for fine-grained tasks
```

**Window Concatenation Process**:
```python
# Input: [576, D] visual tokens from 24×24 grid
# Window size: 2×2 (reduces by 4×)

# Reshape to spatial grid
tokens_grid = tokens.reshape(24, 24, D)

# Apply sliding window with stride 2
windows = []
for i in range(0, 24, 2):
    for j in range(0, 24, 2):
        window = tokens_grid[i:i+2, j:j+2, :]  # Shape: [2, 2, D]
        concatenated = window.flatten(0, 1)     # Shape: [4, D]
        windows.append(concatenated.mean(0))    # or concat → [4D]

# Output: [144, D] or [144, 4D] depending on strategy
```

**WiCo+ Enhancement**:
- **Early layers**: Use compressed 144 tokens (efficient processing)
- **Late layers**: Decompress back to higher resolution for fine-grained reasoning
- **Adaptive**: Model learns when to use compressed vs full resolution

**Performance** (WiCo vs LLaVA-1.5 baseline):
- VQAv2: 80.1% (WiCo+) vs 78.5% (baseline) - 1.6% improvement
- Token reduction: 576 → 144 tokens (4× reduction)
- Inference speedup: ~2.3× faster due to fewer tokens

**TokenFusion** (Wang et al. 2022) focuses on RGB-Depth fusion but the principles apply to any multimodal token fusion:
- Tokens from different modalities are fused at multiple scales
- Early layers: Fine-grained local fusion
- Late layers: Global semantic fusion
- Learnable fusion weights per layer

## Interleaved Patterns

Modern VLMs must handle complex document structures with text, images, tables, and diagrams interleaved arbitrarily. The token concatenation strategy must support this flexibility.

### Arbitrary Interleaving (MM-Interleaved, CM3)

**MM-Interleaved** (Tian et al. 2024) generates sequences with arbitrary image-text ordering:

**Architecture**:
- **Multi-modal feature synchronizer**: Aligns image and text features at multiple scales
- **Interleaved generation**: Model can output image tokens OR text tokens at each step
- **Training data**: OBELISC dataset (141M web documents with interleaved content)

**Token Sequence Examples**:
```
Pattern 1 - Text → Image:
"A beautiful sunset over the ocean. <image_start> [256 image tokens] <image_end>"

Pattern 2 - Image → Text:
"<image_start> [256 image tokens] <image_end> This photo was taken in Hawaii."

Pattern 3 - Multiple interleaved:
"First, <img1> shows the problem. Then <img2> shows the solution. Finally..."
```

From [Generalized Visual Language Models](https://lilianweng.github.io/posts/2022-06-09-vlm/):
> "Inspired by ViT and CoAtNet, SimVLM splits the image into smaller patches in a flatten 1D sequence of patches. They use the convolutional stage consisting of the first 3 blocks of ResNet to extract contextualized patches."

**Causality Preservation**:

Critical for autoregressive generation: each token can only attend to previous tokens, regardless of modality.

```python
# Attention mask for interleaved sequence
# T = text token, I = image token

Sequence: [T1, T2, I1, I2, I3, T3, T4, I4]

Attention mask (1 = can attend, 0 = cannot):
      T1  T2  I1  I2  I3  T3  T4  I4
T1  [ 1   0   0   0   0   0   0   0 ]  # Only self
T2  [ 1   1   0   0   0   0   0   0 ]  # Self + T1
I1  [ 1   1   1   0   0   0   0   0 ]  # Self + T1,T2
I2  [ 1   1   1   1   0   0   0   0 ]  # Self + T1,T2,I1
I3  [ 1   1   1   1   1   0   0   0 ]  # Self + prev
T3  [ 1   1   1   1   1   1   0   0 ]  # Self + prev
T4  [ 1   1   1   1   1   1   1   0 ]  # Self + prev
I4  [ 1   1   1   1   1   1   1   1 ]  # Self + all prev
```

**Advantages**:
- **Maximum flexibility**: Handles any interleaving pattern
- **Generative capability**: Can generate images AND text
- **Document understanding**: Natural for complex documents (papers, web pages)
- **Few-shot adaptability**: Interleaved examples in prompt

**Disadvantages**:
- **Training complexity**: Requires massive datasets with natural interleaving
- **Tokenization overhead**: Need separate tokenizers for each modality
- **Inference complexity**: Must decide when to generate image vs text tokens
- **Error propagation**: Mistakes in early tokens affect all subsequent generation

### Chameleon and Multi-Image Sequences

**Chameleon** focuses on handling multiple images in a single sequence effectively:

**Strategies for Multi-Image**:
1. **Separate image prefixes**: `[img1_tokens][text1][img2_tokens][text2]`
2. **Shared visual context**: All images encoded once, text attends to all
3. **Image-specific attention**: Text tokens attend only to relevant image(s)

**Example Multi-Image Sequence**:
```
Query: "Compare these two images. What changed?"
Image1: [576 tokens - showing 'before' state]
Text: "Compare these two images"
Image2: [576 tokens - showing 'after' state]
Text: "What changed?"

Total context: 576 + 8 + 576 + 3 = 1163 tokens
```

**Attention Optimization**:
- **Local attention**: Each image segment uses local attention (cheaper)
- **Global aggregation**: Summary tokens from each image for global reasoning
- **Sparse attention**: Text tokens attend to k-nearest image patches

**Video Understanding as Extreme Multi-Image**:
```
Video: 30 frames @ 576 tokens each = 17,280 tokens
Problem: Exceeds most context windows

Solution strategies:
1. Temporal sampling: Select key frames (e.g., 8 frames → 4,608 tokens)
2. Temporal compression: Average adjacent frames
3. Hierarchical: Low-res all frames + high-res key frames
4. Recurrent: Process frames sequentially with memory
```

## Token Embedding Alignment

A critical but often overlooked aspect: how to align visual tokens with the text token embedding space of the pretrained LM.

### Projection Strategies

**Linear Projection** (simplest):
```python
# CLIP ViT-L/14: 1024-dim
# LLaMA-7B: 4096-dim

visual_projection = nn.Linear(1024, 4096)
visual_tokens = visual_projection(clip_features)  # [N, 1024] → [N, 4096]
```

**Advantages**: Fast, few parameters, works surprisingly well
**Disadvantages**: Limited expressiveness, no non-linearity

**MLP Projection** (LLaVA-1.5):
```python
visual_projection = nn.Sequential(
    nn.Linear(1024, 4096),
    nn.GELU(),
    nn.Linear(4096, 4096)
)
```

**Advantages**: More expressive, learns better alignment
**Disadvantages**: 2× parameters, slightly slower

**Q-Former** (BLIP-2) - most sophisticated:
```python
class QFormer(nn.Module):
    def __init__(self):
        self.queries = nn.Parameter(torch.randn(32, 768))
        self.self_attn = MultiHeadAttention(768, num_heads=12)
        self.cross_attn = MultiHeadAttention(768, num_heads=12)

    def forward(self, image_features, text_features=None):
        # Self-attention among queries
        q = self.self_attn(self.queries, self.queries, self.queries)

        # Cross-attention to image
        visual_tokens = self.cross_attn(q, image_features, image_features)

        # Optional: condition on text
        if text_features is not None:
            visual_tokens = self.cross_attn(visual_tokens, text_features, text_features)

        return visual_tokens  # [32, 768]
```

**Advantages**: Text-conditional, extreme compression, learnable selection
**Disadvantages**: Complex, expensive to train, requires large datasets

### Cross-Modal Semantic Alignment

Different approaches to ensure visual and text tokens share semantic space:

**CLIP-Based Alignment** (LLaVA, Frozen):
- Vision encoder: CLIP ViT (already aligned to text via contrastive learning)
- Projection: Map CLIP space → LLM embedding space
- Assumption: CLIP alignment transfers to LLM space

**Joint Training** (BeiT-3, CM3):
- Vision and text processed by same transformer
- Shared self-attention ensures alignment through gradients
- No explicit alignment needed (learned end-to-end)

**Contrastive Alignment** (CoCa):
- Dual loss: Contrastive (like CLIP) + Generative (like LLaVA)
- Visual tokens optimized for both similarity and generation
- Best of both worlds: retrieval + generation capability

**Empirical Observations**:
1. CLIP alignment alone is insufficient - projection layer is critical
2. 2-layer MLP projection outperforms single linear layer by 2-3% on VQA
3. Q-Former achieves best compression but requires 3-stage training
4. Joint training requires 10-100× more data but achieves best alignment

## Examples and Use Cases

### LLaVA: Simple Interleaved Concatenation

**Use Case**: General-purpose visual question answering

**Token Flow**:
```
User: "What's in this image? How many people are there?"
Image: dog_park.jpg (640×480)

Processing:
1. Resize image: 640×480 → 336×336 (CLIP requirement)
2. Extract patches: 336×336 → 24×24 patches → 576 tokens
3. CLIP encoding: [576, 1024] visual features
4. MLP projection: [576, 1024] → [576, 4096]
5. Tokenize text: "What's in this image? How many people are there?" → 11 tokens
6. Concatenate: [576 visual + 11 text] = 587 tokens
7. LLaMA-7B: Process 587 tokens
8. Generate: "This image shows a dog park with several dogs playing. There are 3 people visible..."

Context usage: 587 / 4096 tokens = 14.3%
```

**Strengths**:
- All visual information preserved
- Simple architecture
- Works for fine-grained questions

**Weaknesses**:
- High token count limits multi-image support
- Slow inference (quadratic attention)

### Flamingo: Multi-Image with Cross-Attention

**Use Case**: Few-shot learning with multiple example images

**Token Flow**:
```
User (few-shot prompt):
"<img1> This is a golden retriever.
<img2> This is a poodle.
<img3> What breed is this?"

Processing:
1. Each image: 336×336 → CLIP ViT → 256 tokens
2. Perceiver resampler: 256 tokens → 64 tokens (per image)
3. Total visual: 3 images × 64 = 192 tokens
4. Text: ~30 tokens
5. Sequence: 192 visual + 30 text = 222 tokens

Attention pattern:
- "This is a golden retriever" attends to img1's 64 tokens
- "This is a poodle" attends to img2's 64 tokens
- "What breed is this?" attends to img3's 64 tokens

Context usage: 222 / 2048 = 10.8% (much more efficient!)
```

**Strengths**:
- Efficient multi-image handling
- Natural few-shot learning
- Scales to many images

**Weaknesses**:
- Resampler may lose fine details
- Complex architecture

### BLIP-2: Text-Conditioned Compression

**Use Case**: Question-specific visual attention

**Token Flow**:
```
User: "What color is the car in the bottom right?"
Image: street_scene.jpg (1024×768)

Processing:
1. Resize: 1024×768 → 384×384
2. CLIP ViT: 384×384 → 24×24 = 576 tokens
3. Q-Former with text query:
   - Input query text: "What color is the car in the bottom right?"
   - 32 learnable queries + query embedding
   - Cross-attend to image: Focus on bottom-right region
   - Output: 32 tokens emphasizing relevant region
4. Total: 32 visual + 12 text = 44 tokens
5. LLM: Generate answer from 44 tokens

Context usage: 44 / 2048 = 2.1% (extremely efficient!)
```

**Strengths**:
- Query-aware visual compression
- Minimal token usage
- Focuses on task-relevant features

**Weaknesses**:
- May miss important context
- Requires well-posed questions

### WiCo: Adaptive Window Concatenation

**Use Case**: Balance between efficiency and fine-grained understanding

**Token Flow**:
```
User: "Describe this document in detail."
Image: technical_diagram.png (1024×1024)

Processing:
1. High-res ViT: 1024×1024 → 32×32 = 1024 tokens
2. Window concatenation (2×2):
   - Spatially adjacent tokens grouped
   - 1024 tokens → 256 tokens (4× reduction)
3. Fine-tune last ViT layers:
   - Encourage similar features within windows
   - Adaptive to content
4. LLM early layers: Process 256 tokens efficiently
5. WiCo+ decomposition (optional):
   - Late LLM layers: Expand back to higher resolution
   - 256 → 512 tokens for fine-grained reasoning
6. Total: 256-512 visual + text

Context usage: 256-512 / 4096 = 6-12%
```

**Strengths**:
- Adaptive resolution
- Better than naive downsampling
- Configurable compression ratio

**Weaknesses**:
- Requires fine-tuning ViT
- Decompression adds computation

## Citations and Sources

**Source Documents:**
- None (purely web research-based knowledge file)

**Web Research:**

**Primary Papers:**
- [Window Token Concatenation for Efficient Visual Large Language Models](https://arxiv.org/abs/2504.04024) - arXiv:2504.04024 (Li et al., 2025, accessed 2025-01-31): WiCo sliding window concatenation method
- [Generalized Visual Language Models](https://lilianweng.github.io/posts/2022-06-09-vlm/) - Lilian Weng's Blog (June 2022, accessed 2025-01-31): Comprehensive overview of VLM architectures and fusion strategies

**HuggingFace Resources:**
- [Design choices for Vision Language Models in 2024](https://huggingface.co/blog/gigant/vlm-design) - HuggingFace Blog (Théo Gigant, April 2024, accessed 2025-01-31): Comparative analysis of VLM design decisions and trade-offs

**Key Papers Referenced:**
- LLaVA: Liu et al., "Visual Instruction Tuning" (2023)
- Flamingo: Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning" (2022)
- BLIP-2: Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models" (2023)
- SimVLM: Wang et al., "SimVLM: Simple Visual Language Model Pretraining with Weak Supervision" (2022)
- CM3: Aghajanyan et al., "CM3: A Causal Masked Multimodal Model of the Internet" (2022)
- BeiT-3: Wang et al., "Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks" (2022)
- VisualBERT: Li et al., "VisualBERT: A Simple and Performant Baseline for Vision and Language" (2019)
- CoCa: Yu & Wang et al., "CoCa: Contrastive Captioners are Image-Text Foundation Models" (2022)
- TokenFusion: Wang et al., "Multimodal Token Fusion for Vision Transformers" (2022)
- MM-Interleaved: Tian et al., "MM-Interleaved: Interleaved Image-Text Generative Modeling via Multi-modal Feature Synchronizer" (2024)

**Additional Resources:**
- CLIP: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021)
- ViT: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (2021)
