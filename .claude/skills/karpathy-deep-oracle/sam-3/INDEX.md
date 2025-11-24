# SAM 3 Knowledge Index

**Segment Anything Model 3 - Promptable Concept Segmentation**

Complete knowledge base for SAM 3 (November 2024), covering architecture, training, datasets, deployment, and implementation.

---

## Architecture (Workers 1-10)

| File | Description | Lines |
|------|-------------|-------|
| `KNOWLEDGE-DROP-sam3-detector-architecture-2025-11-21.md` | DETR-based text-conditioned detector | 310 |
| `KNOWLEDGE-DROP-sam3-presence-token-2025-11-21.md` | Presence token mechanism for discrimination | 280 |
| `KNOWLEDGE-DROP-sam3-decoupled-design-2025-11-21.md` | Decoupled detector-tracker design | 298 |
| `KNOWLEDGE-DROP-sam3-vision-encoder-2025-11-21.md` | Shared Perception Encoder | 195 |
| `KNOWLEDGE-DROP-sam3-open-vocabulary-prompts-2025-11-21.md` | 270K concept vocabulary system | 280 |
| `KNOWLEDGE-DROP-sam3-detector-deep-2025-11-21.md` | DETR transformer deep dive | 380 |
| `KNOWLEDGE-DROP-sam3-tracker-sam2-2025-11-21.md` | SAM 2 inherited tracker | 318 |
| `KNOWLEDGE-DROP-sam3-text-encoder-2025-11-21.md` | Text encoder integration | 285 |
| `KNOWLEDGE-DROP-sam3-geometry-exemplar-prompts-2025-11-21.md` | Geometry + exemplar prompts | 295 |
| `KNOWLEDGE-DROP-sam3-mask-decoder-2025-11-21.md` | Mask decoder modifications | 295 |

---

## Data Engine & SA-Co Dataset (Workers 11-18)

| File | Description | Lines |
|------|-------------|-------|
| `KNOWLEDGE-DROP-sam3-annotation-pipeline-2025-11-21.md` | 4-phase automatic annotation pipeline | 285 |
| `KNOWLEDGE-DROP-sam3-concept-extraction-2025-11-21.md` | NLP concept extraction | 245 |
| `KNOWLEDGE-DROP-sam3-saco-creation-2025-11-21.md` | SA-Co Gold/Silver/VEval creation | 285 |
| `KNOWLEDGE-DROP-sam3-270k-concepts-2025-11-21.md` | 270K unique concepts (50x more) | 295 |
| `KNOWLEDGE-DROP-sam3-annotation-quality-2025-11-21.md` | Quality control strategies | 280 |
| `KNOWLEDGE-DROP-sam3-data-engine-comparison-2025-11-21.md` | SAM 2 vs SAM 3 data engines | 280 |
| `KNOWLEDGE-DROP-sam3-4m-concepts-scale-2025-11-21.md` | Scaling to 4M concepts | 167 |
| `KNOWLEDGE-DROP-sam3-quality-control-2025-11-21.md` | Quality control mechanisms | 298 |

---

## Training & Performance (Workers 19-26)

| File | Description | Lines |
|------|-------------|-------|
| `KNOWLEDGE-DROP-sam3-training-pipeline-2025-11-21.md` | 4-stage training pipeline | 290 |
| `KNOWLEDGE-DROP-sam3-image-results-2025-11-21.md` | Image segmentation benchmarks | 285 |
| `KNOWLEDGE-DROP-sam3-video-results-2025-11-21.md` | Video segmentation benchmarks | 285 |
| `KNOWLEDGE-DROP-sam3-vs-sam2-2025-11-21.md` | SAM 3 vs SAM 2 comparison | 340 |
| `KNOWLEDGE-DROP-sam3-vs-competitors-2025-11-21.md` | vs OWLv2, DINO-X, Gemini 2.5 | 320 |
| `KNOWLEDGE-DROP-sam3-human-performance-2025-11-21.md` | 75-80% of human performance | 210 |
| `KNOWLEDGE-DROP-sam3-zero-shot-2025-11-21.md` | Zero-shot generalization | 285 |
| `KNOWLEDGE-DROP-sam3-performance-benchmarks-2025-11-21.md` | Latency, throughput, memory | 275 |

---

## Use Cases & Applications (Workers 27-32)

| File | Description | Lines |
|------|-------------|-------|
| `KNOWLEDGE-DROP-sam3-text-prompt-examples-2025-11-21.md` | Text prompt examples & best practices | 380 |
| `KNOWLEDGE-DROP-sam3-interactive-refinement-2025-11-21.md` | Points + text refinement | 316 |
| `KNOWLEDGE-DROP-sam3-batched-inference-2025-11-21.md` | Batch processing patterns | 320 |
| `KNOWLEDGE-DROP-sam3-agent-2025-11-21.md` | SAM 3 Agent with MLLMs | 298 |
| `KNOWLEDGE-DROP-sam3-mllm-integration-2025-11-21.md` | Integration with GPT-4V, Gemini | 380 |
| `KNOWLEDGE-DROP-sam3-production-deployment-2025-11-21.md` | Docker, API serving, scaling | 450 |

---

## SA-Co Benchmark (Workers 33-37)

| File | Description | Lines |
|------|-------------|-------|
| `KNOWLEDGE-DROP-sam3-saco-gold-structure-2025-11-21.md` | SA-Co/Gold dataset format | 268 |
| `KNOWLEDGE-DROP-sam3-saco-silver-structure-2025-11-21.md` | SA-Co/Silver dataset format | 280 |
| `KNOWLEDGE-DROP-sam3-saco-veval-structure-2025-11-21.md` | SA-Co/VEval video benchmark | 280 |
| `KNOWLEDGE-DROP-sam3-evaluation-metrics-2025-11-21.md` | cgF1, pHOTA, mAP, HOTA | 280 |
| `KNOWLEDGE-DROP-sam3-annotation-format-2025-11-21.md` | JSON annotation format | 285 |

---

## Implementation Details (Workers 38-42)

| File | Description | Lines |
|------|-------------|-------|
| `KNOWLEDGE-DROP-sam3-installation-setup-2025-11-21.md` | Python 3.12, PyTorch 2.7, CUDA 12.6 | 385 |
| `KNOWLEDGE-DROP-sam3-huggingface-integration-2025-11-21.md` | HuggingFace checkpoint access | 295 |
| `KNOWLEDGE-DROP-sam3-notebooks-walkthrough-2025-11-21.md` | 6 example notebooks | 485 |
| `KNOWLEDGE-DROP-sam3-api-reference-2025-11-21.md` | Sam3Processor, video_predictor API | 398 |
| `KNOWLEDGE-DROP-sam3-finetuning-customization-2025-11-21.md` | Fine-tuning & domain adaptation | 298 |

---

## Quick Reference

**Total Files**: 42 KNOWLEDGE-DROPs
**Total Lines**: ~12,759 lines
**Created**: 2025-11-23 via 42 ZEUS pattern

### Key Innovations in SAM 3

1. **Text Prompts**: Natural language concept prompts ("player in white")
2. **Presence Token**: Decouples recognition from localization (+9.9% improvement)
3. **Decoupled Design**: Separate detector + tracker for task-specific optimization
4. **270K Concepts**: 50x more than existing benchmarks
5. **75-80% Human Performance**: On SA-Co benchmark

### Performance Highlights

- **LVIS**: 47.0 Mask AP (+22% over previous best)
- **SA-Co/Gold**: 65.0 cgF1 (2x better than OWLv2)
- **Inference**: 30ms per image on H200 GPU
- **Video**: 30 FPS with multi-GPU parallelization

### Getting Started

1. Start with: `KNOWLEDGE-DROP-sam3-installation-setup-2025-11-21.md`
2. Architecture overview: `KNOWLEDGE-DROP-sam3-detector-architecture-2025-11-21.md`
3. Text prompts: `KNOWLEDGE-DROP-sam3-text-prompt-examples-2025-11-21.md`
4. API reference: `KNOWLEDGE-DROP-sam3-api-reference-2025-11-21.md`

---

## Cross-References

- **SAM 1**: `sam-general/` (foundation model)
- **SAM 2**: `sam-2/` (video segmentation)
- **SAM 3D**: `sam-3d/` (3D object segmentation)
- **SA-1B Dataset**: `karpathy/sa1b-dataset/` (1.1B masks)

---

**Last Updated**: 2025-11-23
**Source**: 42 ZEUS Knowledge Expansion
