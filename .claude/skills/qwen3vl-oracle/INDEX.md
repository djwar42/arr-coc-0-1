# qwen3vl-oracle Index

**Quick Navigation**: [Architecture](#architecture) | [Training](#training) | [Codebase](#codebase) | [Usage](#usage) | [Concepts](#concepts) | [Examples](#examples) | [Source Codebase](#source-codebase)

**Total Documentation Files: 10 core files** covering Qwen3-VL's critical innovations

---

## Architecture

Complete system architecture and innovation documentation (4 files)

- **[00-overview.md](architecture/00-overview.md)** - Complete system architecture
  - System diagram with all components
  - Data flow: Image → ViT → DeepStack → LLM → Text
  - Key innovations summary

- **[01-positional-encoding.md](architecture/01-positional-encoding.md)** ⭐ Interleaved-MRoPE
  - **Full-frequency allocation** across time/width/height dimensions
  - 3D positional encoding structure `(temporal, height, width)`
  - Interleaving pattern: every frequency contains all dimensions
  - Evolution: Qwen2 → Qwen2.5 → Qwen3-VL
  - Implementation in `rope2d.py`
  - **Enriched with** RoPE mathematics from LearnOpenCV and arXiv

- **[02-deepstack.md](architecture/02-deepstack.md)** ⭐ Multi-layer injection
  - ViT feature extraction at **layers [6, 12, 18, 24]**
  - LLM injection at **layers [0, 8, 16, 24]**
  - Hierarchical features: fine-grained details → high-level semantics
  - Benefits for OCR, spatial reasoning, document parsing
  - ARR-COC integration considerations

- **[03-timestamp-alignment.md](architecture/03-timestamp-alignment.md)** ⭐ Video temporal encoding
  - **Timestamp tokens** `<t0>`, `<t1>`, `<t2>` ... for explicit temporal grounding
  - Decoupled temporal encoding: tokens (time) + M-RoPE (space)
  - Precise event localization with second-level indexing
  - Video grid splitting in `rope2d.py:get_rope_index_3()`
  - ARR-COC compatibility: adaptive temporal + spatial compression

---

## Training

Fine-tuning and deployment strategies (1 file)

- **[00-fine-tuning-overview.md](training/00-fine-tuning-overview.md)** ⭐ Complete training guide
  - **Training strategies**: LoRA vs Full fine-tuning vs Mixed approach
  - **Module selection**: When to train ViT, MLP projection, or LLM
  - **Resolution strategies**: Fixed vs variable resolution training
  - **Hardware requirements**:
    - Minimum: 1× A100 80GB (LoRA on 7B)
    - Recommended: 8× A100 80GB (Full fine-tuning on 7B)
    - Large-scale: 16-32× H100 80GB (30B+ models)
  - **DeepSpeed integration**: ZeRO Stage 2/3, CPU offloading
  - **Hyperparameters**: Learning rate, batch size, optimization
  - **Common issues**: OOM, NaN loss, slow convergence (with solutions)
  - **Data packing**: 30-50% training speedup
  - **LoRA merging**: Post-training adapter integration

---

## Codebase

File-by-file code documentation with line numbers (1 file)

- **[01-vision-process.md](codebase/01-vision-process.md)** - Vision preprocessing
  - **File**: `qwen-vl-utils/src/qwen_vl_utils/vision_process.py`
  - `smart_resize()` - Dynamic resolution with pixel budgets (Lines 144-169)
  - `fetch_image()` - Image loading and preprocessing (Lines 167-214)
  - `fetch_video()` - Video frame sampling and processing (Lines 477-554)
  - Complete code walkthrough with **Claude's code comments**

---

## Usage

Practical guides for using Qwen3-VL (1 file)

- **[00-quickstart.md](usage/00-quickstart.md)** - Get started in 5 minutes
  - Installation with transformers
  - Basic inference examples
  - First image/video understanding example
  - HuggingFace integration basics

---

## Concepts

Deep dives into key innovations (2 files)

- **[00-interleaved-mrope.md](concepts/00-interleaved-mrope.md)** - Core positional innovation
  - What is Interleaved-MRoPE and why it matters
  - Full-frequency positional encoding explained
  - 3D encoding structure `(temporal, height, width)`
  - Mathematical formulation with rotation matrices
  - Comparison with traditional M-RoPE approaches

- **[05-arr-coc-integration.md](concepts/05-arr-coc-integration.md)** - ARR-COC compatibility
  - How Qwen3-VL's architecture supports variable-token compression
  - Interleaved-MRoPE benefits for compressed patches
  - Timestamp tokens for adaptive temporal sampling
  - DeepStack for hierarchical relevance-aware features
  - Query-aware compression strategies

---

## Examples

Working code samples (1 file)

- **[00-basic-inference.md](examples/00-basic-inference.md)** - Simple inference examples
  - Image understanding with transformers
  - Basic setup and configuration
  - First working example

---

## Source Codebase

**Complete Qwen3-VL codebase** included in oracle:

```
source-codebases/00-Qwen3-VL/
├── qwen-vl-finetune/        # Fine-tuning implementation
│   ├── qwenvl/
│   │   ├── data/
│   │   │   ├── rope2d.py               # M-RoPE implementations (3 versions)
│   │   │   └── data_processor.py       # Dataset loading and packing
│   │   └── train/
│   │       ├── train_qwen.py           # Main training script
│   │       └── trainer.py              # Custom trainer class
│   └── scripts/                        # Training scripts (LoRA, full FT, DeepSpeed)
│
├── qwen-vl-utils/            # Utility functions for preprocessing
│   └── src/qwen_vl_utils/
│       └── vision_process.py           # Image/video preprocessing
│
├── evaluation/               # Evaluation benchmarks (MMMU, etc.)
├── cookbooks/                # Example Jupyter notebooks
└── INDEX.md                  # Codebase overview (inside codebase folder)
```

**Key files with Claude's code comments**:
- `rope2d.py` - All three M-RoPE versions with detailed technical review
- `vision_process.py` - Preprocessing pipeline walkthrough

---

## Quick Reference by Topic

**Innovations**:
- M-RoPE → [architecture/01-positional-encoding.md](architecture/01-positional-encoding.md), [concepts/00-interleaved-mrope.md](concepts/00-interleaved-mrope.md)
- DeepStack → [architecture/02-deepstack.md](architecture/02-deepstack.md)
- Timestamps → [architecture/03-timestamp-alignment.md](architecture/03-timestamp-alignment.md)

**Practical**:
- Training → [training/00-fine-tuning-overview.md](training/00-fine-tuning-overview.md)
- Inference → [usage/00-quickstart.md](usage/00-quickstart.md), [examples/00-basic-inference.md](examples/00-basic-inference.md)
- Code → [codebase/01-vision-process.md](codebase/01-vision-process.md)

**Integration**:
- ARR-COC → [concepts/05-arr-coc-integration.md](concepts/05-arr-coc-integration.md)
- Architecture → [architecture/00-overview.md](architecture/00-overview.md)

---

## Coverage Summary

**Strong Coverage (⭐ = comprehensive)**:
- ✅ **Interleaved-MRoPE** (2 files: architecture + concepts)
- ✅ **DeepStack multi-layer injection** (1 comprehensive file)
- ✅ **Timestamp alignment** (1 comprehensive file)
- ✅ **Fine-tuning strategies** (1 complete guide)
- ✅ **Vision preprocessing** (1 code walkthrough)
- ✅ **ARR-COC integration** (1 conceptual guide)

**Basic Coverage**:
- ✓ System architecture overview
- ✓ Quick start guide
- ✓ Basic inference examples

**External Enrichment**:
- 📚 RoPE mathematics from **LearnOpenCV**
- 📚 M-RoPE theory from **arXiv papers**
- 📚 Qwen AI blog on **Qwen3-VL innovations**

---

**Last Updated**: 2025-10-28
**Status**: Core documentation complete
**Total Files**: 10 focused documentation files covering critical aspects
