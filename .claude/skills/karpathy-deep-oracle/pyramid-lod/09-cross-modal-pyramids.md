# Cross-Modal Pyramids (Text-Image-Audio)

## Overview

Cross-modal pyramids extend the hierarchical level-of-detail concept beyond single modalities to align and fuse information from text, images, and audio. By constructing pyramid structures for each modality and establishing correspondences across scales, vision-language models can perform more effective multimodal reasoning. This enables query-aware processing where text queries guide visual pyramid level selection, or where audio features influence video frame resolution.

**Key Insight**: Just as image pyramids provide multi-scale visual representations, text and audio can be organized into hierarchical embeddings (word → sentence → document for text; frame → temporal window → full sequence for audio). Aligning these pyramids across modalities creates a unified multi-scale representation space for VLMs.

---

## Section 1: Hierarchical Text Embeddings (Word → Sentence → Document)

### Token-Level to Document-Level Hierarchies

Modern NLP models like BERT and GPT produce embeddings at multiple granularities:

**Token-Level Embeddings**:
- Individual word pieces or subword units
- Dimension: typically 768 (BERT-base) or 1024+ (larger models)
- Example: "The cat" → `[E_the, E_cat]` where each E is a 768-dim vector

**Sentence-Level Aggregation**:
- Pooling strategies: [CLS] token, mean pooling, max pooling
- Attention-weighted aggregation over tokens
- Specialized sentence encoders (Sentence-BERT, InstructOR)
- Captures semantic meaning of complete thoughts

**Document-Level Hierarchies**:
- Recursive aggregation: sentences → paragraphs → full document
- Hierarchical transformers with multi-level attention
- Pyramid structure: fine detail at bottom (tokens), coarse semantics at top (document summary)

From [Extracting Sentence Embeddings from Pretrained Transformers](https://arxiv.org/html/2408.08073v1) (arXiv:2408.08073, accessed 2025-01-31):
- Token aggregation methods for sentence embeddings
- Representation shaping techniques to improve embedding quality
- Hierarchical pooling to create document-level representations

**Implementation Example**:
```python
# Pseudo-code for hierarchical text pyramid
def build_text_pyramid(document):
    tokens = tokenize(document)
    token_embeds = bert_model(tokens)  # [num_tokens, 768]

    # Level 0: token embeddings
    level_0 = token_embeds

    # Level 1: sentence embeddings (aggregate every ~16 tokens)
    level_1 = [mean_pool(token_embeds[i:i+16]) for i in range(0, len(tokens), 16)]

    # Level 2: paragraph embeddings (aggregate sentences)
    level_2 = [mean_pool(level_1[i:i+4]) for i in range(0, len(level_1), 4)]

    # Level 3: document embedding (aggregate paragraphs)
    level_3 = mean_pool(level_2)

    return [level_0, level_1, level_2, level_3]
```

### Aligned Text-Image Pyramids (CLIP-Style)

**CLIP Contrastive Learning**:
- Trains text encoder and image encoder in shared embedding space
- Contrastive loss: maximize similarity for matched text-image pairs, minimize for mismatched
- Result: text embeddings and image embeddings are geometrically aligned

**Pyramid Extension to CLIP**:
From [Multimodal Alignment and Fusion: A Survey](https://arxiv.org/html/2411.17040v1) (arXiv:2411.17040, accessed 2025-01-31):
- Cross-modal alignment at multiple scales
- Word-level text ↔ patch-level image (fine scale)
- Sentence-level text ↔ region-level image (medium scale)
- Document-level text ↔ full image (coarse scale)
- Hierarchical contrastive learning across pyramid levels

**Application to VLMs**:
- Query: "Find red cars" → sentence embedding
- Image pyramid: multiple resolution levels
- Match query embedding to appropriate pyramid level(s)
- Fine details needed? Use high-res levels. Coarse semantics? Use low-res levels.

Cross-reference: See `vision-language/03-clip-contrastive-learning.md` for CLIP architecture details.

---

## Section 2: Audio Spectrograms as Frequency Pyramids

### Mel-Spectrogram Multi-Resolution Representations

From [Understanding Spectrograms](https://www.izotope.com/en/learn/understanding-spectrograms) (accessed 2025-01-31):
- **Spectrogram**: Visual representation of audio showing time, frequency, and amplitude
- Horizontal axis: time (seconds)
- Vertical axis: frequency (Hz, lowest at bottom, highest at top)
- Brightness/color: amplitude (bright = loud, dark = quiet)

**Audio Pyramid Construction**:

**Frequency Pyramids**:
- Low-resolution: 40 Mel-frequency bins (coarse spectral shape)
- Medium-resolution: 80 Mel bins (standard speech recognition)
- High-resolution: 128-256 Mel bins (music, fine-grained audio)

**Temporal Pyramids**:
- Coarse: 10ms frame rate (60fps equivalent)
- Medium: 25ms frame rate (standard speech)
- Fine: 5ms frame rate (music transients)

From [Environmental Sound Classification Using Temporal-Frequency Attention](https://pmc.ncbi.nlm.nih.gov/articles/PMC8566500/) (accessed 2025-01-31):
- Temporal-frequency attention networks (TFCNN)
- Pyramidal temporal pooling with discriminative learning
- Multi-scale feature extraction from audio spectrograms

**Speech vs. Music Pyramid Structures**:

**Speech Audio Pyramids**:
- Focus on 100-8000 Hz range (human voice frequencies)
- Coarser frequency resolution acceptable (phoneme recognition)
- Temporal detail critical (consonants, plosives)
- Example: 40 Mel bins, 10ms frames

**Music Audio Pyramids**:
- Full 20-20,000 Hz spectrum needed
- Fine frequency resolution (pitch, harmony)
- Temporal detail for rhythm and transients
- Example: 128 Mel bins, 5-10ms frames

### Wavelet Transforms for Audio Pyramids

**Alternative to STFT-based Spectrograms**:
- Wavelet transform provides multi-scale time-frequency analysis
- Natural pyramid structure: coarse wavelet scales = low frequencies, fine scales = high frequencies
- Better temporal localization for transient events (drum hits, plosives)

**Wavelet Pyramid Levels**:
```
Level 3 (coarse):   20-100 Hz    (bass, kick drum)
Level 2 (medium):   100-1000 Hz  (fundamentals, vocals)
Level 1 (fine):     1000-8000 Hz (harmonics, consonants)
Level 0 (finest):   8000-20k Hz  (sibilance, cymbals)
```

Cross-reference: See `practical-implementation/55-3d-volume-texture-spatiotemporal-vit.md` for video-audio joint pyramids.

---

## Section 3: Multi-Modal Pyramid Fusion

### Aligning Image, Text, and Audio Pyramids

From [Pyramidal Cross-Modal Transformer with Sustained Visual Guidance](https://dl.acm.org/doi/10.1145/3652583.3658005) (accessed 2025-01-31):
- Pyramidal visual guidance layer parses visual features into multi-resolution pyramid
- Cross-modal interaction between visual and semantic (text) information
- Sustained guidance across hierarchical levels
- Hybrid modal interaction layer with modal-blended attention

**Pyramid Level Correspondence**:

| **Modality** | **Coarse Level** | **Medium Level** | **Fine Level** |
|--------------|------------------|------------------|----------------|
| **Vision** | Full image (7×7 patches) | Regions (14×14) | Patches (28×28) |
| **Text** | Document summary | Sentences | Words/tokens |
| **Audio** | Full clip (5 sec) | Windows (0.5 sec) | Frames (25ms) |

**Cross-Modal Attention at Matched Scales**:
```python
# Pseudo-code for aligned pyramid fusion
def fuse_multimodal_pyramids(image_pyramid, text_pyramid, audio_pyramid):
    fused_features = []

    for level in range(num_levels):
        # Extract features at matched scale
        img_feat = image_pyramid[level]      # e.g., 14×14×512
        txt_feat = text_pyramid[level]       # e.g., sentence embeds
        aud_feat = audio_pyramid[level]      # e.g., 0.5sec windows

        # Cross-modal attention
        txt_to_img = cross_attention(query=txt_feat, key=img_feat, value=img_feat)
        aud_to_img = cross_attention(query=aud_feat, key=img_feat, value=img_feat)

        # Fusion
        fused = concatenate([txt_to_img, aud_to_img, img_feat])
        fused_features.append(fused)

    return fused_features
```

### Fusion Strategies (Early, Mid, Late) with Pyramids

From [Multimodal Alignment and Fusion: A Survey](https://arxiv.org/html/2411.17040v1) (arXiv:2411.17040, accessed 2025-01-31):
- Comprehensive review of multimodal alignment and fusion techniques
- Early fusion: combine raw features before encoding
- Mid fusion: combine intermediate representations
- Late fusion: combine high-level semantic features

**Early Fusion with Pyramids**:
- Concatenate raw inputs: RGB image + Mel spectrogram + word tokens
- Joint pyramid construction on concatenated features
- Advantage: captures low-level cross-modal correlations
- Disadvantage: computationally expensive, modalities may have different optimal scales

**Mid Fusion with Pyramids** (Recommended):
- Build separate pyramids for each modality
- Fuse at intermediate layers with cross-attention
- Advantage: modality-specific pyramid construction, flexible fusion
- Used in most modern VLMs (Flamingo, BLIP-2, LLaVA)

**Late Fusion with Pyramids**:
- Extract high-level features from each pyramid independently
- Combine via weighted sum or learned gating
- Advantage: modularity, can use pre-trained modality-specific encoders
- Disadvantage: misses low-level cross-modal interactions

### ImageBind-Style Universal Embedding Spaces

**ImageBind Approach**:
- Learn joint embedding space for 6+ modalities (image, text, audio, depth, thermal, IMU)
- Use image as "binding" modality (all others align to image)
- Emergent cross-modal alignment (audio-text pairs without direct supervision)

**Pyramid Extension to ImageBind**:
```
                    Joint Embedding Space
                           /|\
                          / | \
                         /  |  \
                        /   |   \
        Image Pyramid  Text Pyramid  Audio Pyramid
          /|\            /|\            /|\
         / | \          / | \          / | \
       L3 L2 L1       L3 L2 L1       L3 L2 L1
     (coarse → fine) (doc → word)  (clip → frame)
```

**Multi-Scale ImageBind**:
- Align pyramids at each level to shared embedding space
- Coarse levels: global semantics (object categories, document topics, audio events)
- Fine levels: local details (texture, word choice, spectral fine structure)
- Cross-modal retrieval at any scale: text query → appropriate image pyramid level

Cross-reference: See `vision-language/06-multimodal-fusion-strategies.md` for fusion techniques.

---

## Section 4: Aligned Cross-Modal LOD

### Text Query → Appropriate Image Pyramid Level

**Query-Aware Pyramid Level Selection**:

**Coarse Queries** → Low-Resolution Pyramid Levels:
- Query: "Is there a person in the image?"
- Required detail: object presence/absence (binary)
- Pyramid level: Coarse (7×7 patches, 49 tokens)
- Reasoning: global scene understanding, no fine texture needed

**Fine-Grained Queries** → High-Resolution Pyramid Levels:
- Query: "What text is written on the sign in the background?"
- Required detail: OCR, small text regions
- Pyramid level: Fine (28×28+ patches, 784+ tokens)
- Reasoning: precise spatial localization and texture detail

**Implementation Strategy**:
```python
def query_aware_pyramid_selection(text_query, image_pyramid):
    # Classify query granularity
    query_type = classify_query(text_query)

    if query_type == "global":  # "What is the scene?"
        return image_pyramid[0]  # Coarsest level
    elif query_type == "object":  # "Where is the car?"
        return image_pyramid[1:2]  # Medium levels
    elif query_type == "detailed":  # "Read the license plate"
        return image_pyramid[2:4]  # Finest levels
    else:
        return image_pyramid  # Use all levels
```

### Audio Features → Video Frame Resolution

**Audio-Driven Visual LOD**:

**Low-Frequency Audio** (e.g., bass, speech fundamental):
- Indicates slow visual motion or static scenes
- Use lower video frame rate (15 fps) and lower spatial resolution
- Example: podcast video, talking head

**High-Frequency Audio** (e.g., cymbals, sibilants):
- Indicates rapid visual motion or transients
- Use higher frame rate (30-60 fps) and higher spatial resolution
- Example: action scenes, music videos with fast cuts

**Synchronized Audio-Visual Pyramids**:
```python
def audio_driven_video_pyramid(audio_features, video_frames):
    # Analyze audio frequency content
    low_freq_energy = compute_energy(audio_features, freq_range=(20, 200))
    high_freq_energy = compute_energy(audio_features, freq_range=(2000, 8000))

    if high_freq_energy > 2 * low_freq_energy:
        # High-frequency audio → use fine video pyramid
        frame_rate = 60
        spatial_resolution = (1920, 1080)
    else:
        # Low-frequency audio → use coarse video pyramid
        frame_rate = 15
        spatial_resolution = (640, 480)

    return process_video(video_frames, frame_rate, spatial_resolution)
```

### Joint Optimization of Cross-Modal Pyramids

**Training Objective**:
Optimize pyramid structures jointly across modalities to maximize task performance.

**Loss Function**:
```
L_total = L_task + λ_align * L_alignment + λ_pyramid * L_pyramid_reg

Where:
- L_task: primary task loss (e.g., VQA accuracy)
- L_alignment: cross-modal alignment loss (contrastive)
- L_pyramid_reg: pyramid structure regularization
```

**Learnable Pyramid Depth**:
- Not all tasks require same number of pyramid levels
- Use neural architecture search (NAS) or gradient-based methods
- Search space: 2-5 pyramid levels per modality

**Dynamic Level Selection During Training**:
- Start with all levels active
- Gradually prune less-useful levels via gating
- End-to-end differentiable selection

Cross-reference: See `attending.py` in ARR-COC project for relevance-driven LOD allocation.

### Applications: Video Retrieval and Audio-Visual Learning

**Video Retrieval with Cross-Modal Pyramids**:
1. Text query: "Person running on beach at sunset"
2. Extract query embedding pyramid (word → sentence → full query)
3. Match to video pyramid levels:
   - Coarse: sunset lighting (global scene)
   - Medium: person + beach (object-level)
   - Fine: running motion (action-level)
4. Retrieve videos with highest cross-modal similarity across levels

**Audio-Visual Speech Recognition**:
- Audio pyramid: phonemes (fine) → words (medium) → sentences (coarse)
- Video pyramid: lip pixels (fine) → mouth region (medium) → face (coarse)
- Align audio and visual pyramids temporally
- Fuse at matched scales: phoneme audio ↔ lip video, word audio ↔ mouth video

**Music Video Analysis**:
- Audio: beat detection (temporal pyramid), timbre (frequency pyramid)
- Video: scene cuts (temporal pyramid), visual rhythm (spatial pyramid)
- Synchronization: align beat timestamps with scene cuts
- Semantic matching: musical mood ↔ visual aesthetics at coarse levels

---

## Sources

**Web Research** (accessed 2025-01-31):
- [Multimodal Alignment and Fusion: A Survey](https://arxiv.org/html/2411.17040v1) - arXiv:2411.17040, comprehensive review of cross-modal techniques
- [Extracting Sentence Embeddings from Pretrained Transformers](https://arxiv.org/html/2408.08073v1) - arXiv:2408.08073, hierarchical text embeddings
- [Pyramidal Cross-Modal Transformer with Sustained Visual Guidance](https://dl.acm.org/doi/10.1145/3652583.3658005) - ICMR 2024, pyramidal visual-semantic fusion
- [Environmental Sound Classification Using Temporal-Frequency Attention](https://pmc.ncbi.nlm.nih.gov/articles/PMC8566500/) - PMC8566500, audio pyramid techniques
- [Understanding Spectrograms](https://www.izotope.com/en/learn/understanding-spectrograms) - iZotope tutorial, spectrogram fundamentals

**Cross-References**:
- `vision-language/03-clip-contrastive-learning.md` - CLIP architecture and training
- `vision-language/06-multimodal-fusion-strategies.md` - Fusion techniques for VLMs
- `practical-implementation/55-3d-volume-texture-spatiotemporal-vit.md` - Video-audio joint modeling
- `../attending.py` (ARR-COC project) - Relevance-driven LOD allocation
- `../realizing.py` (ARR-COC project) - Pipeline orchestration for multi-scale processing

**Additional Reading**:
- ImageBind paper (Meta AI) - Universal embedding spaces
- CLIP paper (OpenAI) - Contrastive text-image learning
- Sentence-BERT paper - Sentence embedding techniques
- TimeSformer, ViViT papers - Video transformers with temporal pyramids
