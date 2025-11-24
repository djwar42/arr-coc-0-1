# KNOWLEDGE DROP: SAM 3 Text-Prompted Segmentation Overview

## Drop Metadata
- **Created:** 2025-11-20 21:54
- **Source:** Web research (Meta AI, Roboflow, Ultralytics)
- **Topic:** SAM 3 - Segment Anything with Concepts
- **Target File:** 18-sam3-overview-text.md

## Key Knowledge Captured

### SAM 3 Core Innovation
- **Released:** November 19, 2025 by Meta AI
- **Key Feature:** First SAM model with text prompting
- **Task:** Promptable Concept Segmentation (PCS)
- **Capability:** Detects ALL instances of a concept (not just one)
- **Parameters:** 848M (~3.4 GB)

### Open-Vocabulary Segmentation
- **270K unique concepts** (50x more than LVIS)
- 22 million entities across 17 top-level categories
- Zero-shot understanding of any noun phrase
- Presence token for discrimination between similar concepts

### Architecture Highlights
- Decoupled Detector-Tracker design
- Shared Perception Encoder for vision-language fusion
- Global presence head (checks existence before localization)
- Inherited memory attention from SAM 2 for video

### Performance Benchmarks
- **SA-Co/Gold:** 54.1 cgF1 (75-80% of human performance at 72.8)
- **LVIS:** 37.2 cgF1 (state-of-the-art open-vocabulary)
- **Speed:** ~30ms per image on H200 GPU
- **Capacity:** Handles 100+ objects per image

### Key Differences from SAM 1/2
- SAM 1/2: One mask per prompt
- SAM 3: ALL matching instances from one text prompt
- Text prompts eliminate need to click each object
- Visual exemplar prompts for similarity matching

### SA-Co Dataset
- ~5.2M high-quality images
- 52.5K videos
- >4M unique noun phrases
- ~1.4B masks

## ARR-COC Research Relevance

### Cross-Modal Attention
- Perception Encoder creates aligned embedding spaces
- Bidirectional attention between vision and language
- Model for multi-modal fusion in LLMs

### Presence Token Innovation
- Learnable token that improves discrimination
- Potential application: attention routing tokens
- Task-specific tokens for better performance

### Decoupled Architecture
- Separates recognition from localization
- Parallels routing vs computation in MoE
- Efficient model design patterns

### Training Insights
- 4-stage training (component-wise)
- When to freeze vs fine-tune
- Curriculum learning strategies

## Sources Cited

- Meta AI SAM 3 research (ai.meta.com/research/publications/sam-3)
- GitHub: github.com/facebookresearch/sam3
- HuggingFace: huggingface.co/facebook/sam3
- Roboflow technical analysis (blog.roboflow.com/what-is-sam3/)
- Ultralytics documentation (docs.ultralytics.com/models/sam-3/)
- OpenReview paper (openreview.net/forum?id=r35clVtGzw)

## File Statistics
- **Target file:** 18-sam3-overview-text.md
- **Lines:** ~720
- **Sections:** 8 (as specified)
- **ARR-COC Integration:** ~10% of content
