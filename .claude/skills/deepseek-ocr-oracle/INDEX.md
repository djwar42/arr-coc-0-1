# DeepSeek-OCR Oracle - Master Index

**Complete navigation guide to DeepSeek-OCR vision-language model knowledge base**

---

## 📖 Overview

This oracle provides comprehensive expertise on **DeepSeek-OCR**, a vision-language model that achieves extreme efficiency through optical compression. Key innovation: 16× spatial compression using serial SAM+CLIP architecture, reducing images to 73-421 visual tokens while maintaining high OCR and document understanding performance.

**Core Knowledge Areas:**
- Architecture & Design (SAM+CLIP serial, 16× compression, MoE decoder)
- Code Implementation (entry points, processing pipeline, model layers)
- Training Pipeline (3-stage: DeepEncoder → Full VLM → Gundam)
- Practical Usage (HuggingFace, deployment, fine-tuning)
- Concepts & Philosophy (optical compression, forgetting, token budgets)
- Comparisons (vs standard VLMs, vs ARR-COC-VIS)

---

## 📂 Directory Structure

```
deepseek-ocr-oracle/
├── SKILL.md                        # Oracle entry point & metadata
├── INDEX.md                        # This file - master navigation
├── file-index.md                   # Complete codebase file reference
├── _ingest/                        # Manual document ingestion
│   └── README.md
├── _ingest-auto/                   # Automated ingestion (dynamic learning)
│   └── README.md
├── source-codebases/               # Original DeepSeek-OCR source
│   └── 00-DeepSeek-OCR/           # Full codebase
│       ├── INDEX.md               # Codebase overview
│       └── ...
├── architecture/                   # System design & components
│   ├── 00-overview.md             # Complete architecture
│   ├── 01-deepencoder.md          # SAM + CLIP serial design
│   ├── 02-compression.md          # 16× compression mechanism
│   ├── 03-projector.md            # Feature fusion & projection
│   ├── 04-moe-decoder.md          # DeepSeek-3B-MoE
│   └── 05-resolution-modes.md     # Multi-resolution support
├── code-reference/                 # Implementation details
│   ├── 00-entry-points.md         # Main execution paths
│   ├── 01-image-processing.md     # Preprocessing & tiling
│   ├── 02-sam-implementation.md   # SAM model code
│   ├── 03-clip-implementation.md  # CLIP model code
│   ├── 04-projector-code.md       # MlpProjector code
│   ├── 05-token-calculation.md    # Dynamic token budgets
│   └── 06-inference-flow.md       # Complete forward pass
├── training/                       # Training pipeline
│   ├── 00-overview.md             # 3-stage pipeline
│   ├── 01-stage1-deepencoder.md   # DeepEncoder pre-training
│   ├── 02-stage2-full-vlm.md      # Full VLM training
│   ├── 03-stage3-gundam.md        # High-res fine-tuning
│   ├── 04-data-engineering.md     # Dataset construction
│   ├── 05-optimization.md         # Training optimizations
│   └── 06-infrastructure.md       # HAI-LLM platform
├── usage/                          # Practical guides
│   ├── 00-quick-start.md          # 3-line inference
│   ├── 01-huggingface.md          # HuggingFace integration
│   ├── 02-resolution-selection.md # Choosing resolution modes
│   ├── 03-fine-tuning.md          # Custom dataset training
│   ├── 04-vllm-deployment.md      # Production deployment
│   └── 05-gradio-demo.md          # Web interface
├── concepts/                       # Core ideas & philosophy
│   ├── 00-optical-compression.md  # Core innovation
│   ├── 01-token-budgets.md        # Why 73-421 tokens?
│   ├── 02-forgetting.md           # Progressive compression
│   ├── 03-tiling-strategy.md      # Gundam mode explained
│   └── 04-design-philosophy.md    # Serial architecture rationale
└── comparisons/                    # Context & positioning
    ├── 00-vs-standard-vlms.md     # How it differs
    ├── 01-vs-arr-coc-vis.md       # Compression vs relevance
    └── 02-performance-metrics.md  # Benchmarks & results
```

---

## 🎯 Quick Navigation by Topic

### Architecture & Design
| Topic | File | Description |
|-------|------|-------------|
| **System Overview** | [architecture/00-overview.md](architecture/00-overview.md) | Complete architecture: DeepEncoder → Projector → MoE Decoder |
| **DeepEncoder** | [architecture/01-deepencoder.md](architecture/01-deepencoder.md) | SAM + CLIP serial design (380M params) |
| **Compression** | [architecture/02-compression.md](architecture/02-compression.md) | 16× spatial compression mechanism |
| **Projector** | [architecture/03-projector.md](architecture/03-projector.md) | Feature fusion & MLP projection to LLM space |
| **MoE Decoder** | [architecture/04-moe-decoder.md](architecture/04-moe-decoder.md) | DeepSeek-3B-MoE (570M active / 3B total) |
| **Resolution Modes** | [architecture/05-resolution-modes.md](architecture/05-resolution-modes.md) | Multi-resolution support (73-421 tokens) |

### Code Implementation
| Topic | File | Description |
|-------|------|-------------|
| **Entry Points** | [code-reference/00-entry-points.md](code-reference/00-entry-points.md) | Main execution paths & model loading |
| **Image Processing** | [code-reference/01-image-processing.md](code-reference/01-image-processing.md) | Preprocessing, tiling, normalization |
| **SAM Implementation** | [code-reference/02-sam-implementation.md](code-reference/02-sam-implementation.md) | SAM model code walkthrough |
| **CLIP Implementation** | [code-reference/03-clip-implementation.md](code-reference/03-clip-implementation.md) | CLIP model code walkthrough |
| **Projector Code** | [code-reference/04-projector-code.md](code-reference/04-projector-code.md) | MlpProjector implementation |
| **Token Calculation** | [code-reference/05-token-calculation.md](code-reference/05-token-calculation.md) | Dynamic token budget logic |
| **Inference Flow** | [code-reference/06-inference-flow.md](code-reference/06-inference-flow.md) | Complete forward pass |

### Training Pipeline
| Topic | File | Description |
|-------|------|-------------|
| **Training Overview** | [training/00-overview.md](training/00-overview.md) | 3-stage pipeline summary |
| **Stage 1: DeepEncoder** | [training/01-stage1-deepencoder.md](training/01-stage1-deepencoder.md) | DeepEncoder pre-training (frozen LLM) |
| **Stage 2: Full VLM** | [training/02-stage2-full-vlm.md](training/02-stage2-full-vlm.md) | End-to-end VLM training |
| **Stage 3: Gundam** | [training/03-stage3-gundam.md](training/03-stage3-gundam.md) | High-resolution fine-tuning |
| **Data Engineering** | [training/04-data-engineering.md](training/04-data-engineering.md) | Dataset construction & quality |
| **Optimization** | [training/05-optimization.md](training/05-optimization.md) | Training optimizations & tricks |
| **Infrastructure** | [training/06-infrastructure.md](training/06-infrastructure.md) | HAI-LLM platform setup |

### Practical Usage
| Topic | File | Description |
|-------|------|-------------|
| **Quick Start** | [usage/00-quick-start.md](usage/00-quick-start.md) | 3-line inference example |
| **HuggingFace** | [usage/01-huggingface.md](usage/01-huggingface.md) | HuggingFace integration |
| **Resolution Selection** | [usage/02-resolution-selection.md](usage/02-resolution-selection.md) | Choosing the right mode |
| **Fine-Tuning** | [usage/03-fine-tuning.md](usage/03-fine-tuning.md) | Custom dataset training |
| **vLLM Deployment** | [usage/04-vllm-deployment.md](usage/04-vllm-deployment.md) | Production deployment |
| **Gradio Demo** | [usage/05-gradio-demo.md](usage/05-gradio-demo.md) | Web interface setup |

### Core Concepts
| Topic | File | Description |
|-------|------|-------------|
| **Optical Compression** | [concepts/00-optical-compression.md](concepts/00-optical-compression.md) | The core innovation |
| **Token Budgets** | [concepts/01-token-budgets.md](concepts/01-token-budgets.md) | Why 73-421 tokens? |
| **Forgetting** | [concepts/02-forgetting.md](concepts/02-forgetting.md) | Progressive compression over time |
| **Tiling Strategy** | [concepts/03-tiling-strategy.md](concepts/03-tiling-strategy.md) | Gundam mode explained |
| **Design Philosophy** | [concepts/04-design-philosophy.md](concepts/04-design-philosophy.md) | Why serial architecture works |

### Comparisons & Context
| Topic | File | Description |
|-------|------|-------------|
| **vs Standard VLMs** | [comparisons/00-vs-standard-vlms.md](comparisons/00-vs-standard-vlms.md) | How DeepSeek-OCR differs |
| **vs ARR-COC-VIS** | [comparisons/01-vs-arr-coc-vis.md](comparisons/01-vs-arr-coc-vis.md) | Compression vs relevance |
| **Performance Metrics** | [comparisons/02-performance-metrics.md](comparisons/02-performance-metrics.md) | Benchmarks & results |

---

## 🔍 Key Concepts Cross-Reference

### Optical Compression (16× reduction)
- **Architecture**: [architecture/02-compression.md](architecture/02-compression.md)
- **Concept**: [concepts/00-optical-compression.md](concepts/00-optical-compression.md)
- **Code**: [code-reference/02-sam-implementation.md](code-reference/02-sam-implementation.md)

### Serial SAM+CLIP Design
- **Architecture**: [architecture/01-deepencoder.md](architecture/01-deepencoder.md)
- **SAM Code**: [code-reference/02-sam-implementation.md](code-reference/02-sam-implementation.md)
- **CLIP Code**: [code-reference/03-clip-implementation.md](code-reference/03-clip-implementation.md)
- **Philosophy**: [concepts/04-design-philosophy.md](concepts/04-design-philosophy.md)

### Token Budgets (73-421 tokens)
- **Concept**: [concepts/01-token-budgets.md](concepts/01-token-budgets.md)
- **Calculation**: [code-reference/05-token-calculation.md](code-reference/05-token-calculation.md)
- **Resolution Modes**: [architecture/05-resolution-modes.md](architecture/05-resolution-modes.md)

### 3-Stage Training Pipeline
- **Overview**: [training/00-overview.md](training/00-overview.md)
- **Stage 1**: [training/01-stage1-deepencoder.md](training/01-stage1-deepencoder.md)
- **Stage 2**: [training/02-stage2-full-vlm.md](training/02-stage2-full-vlm.md)
- **Stage 3**: [training/03-stage3-gundam.md](training/03-stage3-gundam.md)

### Gundam Mode (High-Resolution Tiling)
- **Concept**: [concepts/03-tiling-strategy.md](concepts/03-tiling-strategy.md)
- **Training**: [training/03-stage3-gundam.md](training/03-stage3-gundam.md)
- **Code**: [code-reference/01-image-processing.md](code-reference/01-image-processing.md)

### Progressive Forgetting
- **Concept**: [concepts/02-forgetting.md](concepts/02-forgetting.md)
- **Compression**: [architecture/02-compression.md](architecture/02-compression.md)
- **Training**: [training/01-stage1-deepencoder.md](training/01-stage1-deepencoder.md)

---

## 🚀 Common Use Cases → Documentation

### "I want to use DeepSeek-OCR"
1. Start: [usage/00-quick-start.md](usage/00-quick-start.md)
2. Integration: [usage/01-huggingface.md](usage/01-huggingface.md)
3. Mode Selection: [usage/02-resolution-selection.md](usage/02-resolution-selection.md)

### "I want to understand how it works"
1. Overview: [architecture/00-overview.md](architecture/00-overview.md)
2. Core Innovation: [concepts/00-optical-compression.md](concepts/00-optical-compression.md)
3. Design: [architecture/01-deepencoder.md](architecture/01-deepencoder.md)

### "I want to see the code"
1. Entry Points: [code-reference/00-entry-points.md](code-reference/00-entry-points.md)
2. Inference Flow: [code-reference/06-inference-flow.md](code-reference/06-inference-flow.md)
3. Full Codebase: [source-codebases/00-DeepSeek-OCR/INDEX.md](source-codebases/00-DeepSeek-OCR/INDEX.md)

### "I want to train/fine-tune it"
1. Overview: [training/00-overview.md](training/00-overview.md)
2. Fine-Tuning: [usage/03-fine-tuning.md](usage/03-fine-tuning.md)
3. Data: [training/04-data-engineering.md](training/04-data-engineering.md)

### "I want to deploy it"
1. vLLM: [usage/04-vllm-deployment.md](usage/04-vllm-deployment.md)
2. Demo: [usage/05-gradio-demo.md](usage/05-gradio-demo.md)
3. Optimization: [training/05-optimization.md](training/05-optimization.md)

### "How does it compare to X?"
1. Standard VLMs: [comparisons/00-vs-standard-vlms.md](comparisons/00-vs-standard-vlms.md)
2. ARR-COC-VIS: [comparisons/01-vs-arr-coc-vis.md](comparisons/01-vs-arr-coc-vis.md)
3. Benchmarks: [comparisons/02-performance-metrics.md](comparisons/02-performance-metrics.md)

---

## 📚 Source Materials

### Source Codebase
- **Location**: [source-codebases/00-DeepSeek-OCR/](source-codebases/00-DeepSeek-OCR/)
- **Index**: [source-codebases/00-DeepSeek-OCR/INDEX.md](source-codebases/00-DeepSeek-OCR/INDEX.md)
- **Description**: Complete DeepSeek-OCR codebase with vLLM integration

### Codebase Files Index
- **Location**: [file-index.md](file-index.md)
- **Description**: Complete file tree with summaries for all codebase files

---

## 🔄 Dynamic Learning

This oracle supports **SEEKING mode** - it can expand its knowledge dynamically using web research when needed.

### Manual Ingestion
Place new documents in `_ingest/` folder and instruct oracle-creator to process them.

### Automatic Ingestion
Oracle can autonomously download and integrate new research papers, documentation, and code examples during knowledge expansion. Downloaded materials temporarily stored in `_ingest-auto/` during processing.

---

## 💡 Tips for Using This Oracle

**For Architecture Questions:**
- Start with [architecture/00-overview.md](architecture/00-overview.md) for big picture
- Dive into specific components (DeepEncoder, Projector, MoE) as needed
- Check [concepts/](concepts/) folder for design rationale

**For Implementation Questions:**
- Begin with [code-reference/00-entry-points.md](code-reference/00-entry-points.md)
- Follow [code-reference/06-inference-flow.md](code-reference/06-inference-flow.md) for execution flow
- Reference actual codebase in [source-codebases/00-DeepSeek-OCR/](source-codebases/00-DeepSeek-OCR/)

**For Training Questions:**
- Read [training/00-overview.md](training/00-overview.md) first
- Understand 3-stage progression
- Check [training/04-data-engineering.md](training/04-data-engineering.md) for dataset details

**For Usage Questions:**
- Quick start: [usage/00-quick-start.md](usage/00-quick-start.md)
- Production deployment: [usage/04-vllm-deployment.md](usage/04-vllm-deployment.md)
- Fine-tuning: [usage/03-fine-tuning.md](usage/03-fine-tuning.md)

---

## 📊 Statistics

- **Total Documentation Files**: 39
- **Architecture Docs**: 6
- **Code Reference Docs**: 7
- **Training Docs**: 7
- **Usage Docs**: 6
- **Concept Docs**: 5
- **Comparison Docs**: 3
- **Source Codebases**: 1 (complete)
- **Dynamic Learning**: Enabled (SEEKING mode)

---

**Last Updated**: 2025-10-28
**Oracle Version**: 1.0
**Knowledge Status**: Comprehensive & Complete
