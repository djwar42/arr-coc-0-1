# SAM Source Code - VARIABLE ZEUS Pattern
**Target**: ALL SAM codebases (SAM 1, 2, 2.1, 3)
**GitHub Repos**:
- https://github.com/facebookresearch/segment-anything (SAM 1)
- https://github.com/facebookresearch/sam2 (SAM 2 & 2.1)
- https://github.com/facebookresearch/sam3 (SAM 3)
**Goal**: Fetch ALL source code + create KNOWLEDGE-DROPs about structure
**Workers**: VARIABLE (as many as needed for complete codebase coverage)

---

## Strategy

VARIABLE ZEUS = ADAPTIVE PARALLELISM
- Start with UNKNOWN number of files (discover during fetch)
- Each worker fetches ONE file + creates KNOWLEDGE-DROP about its structure
- Scale workers to match total file count
- Result: Complete source code in `source-codebases/` with full documentation

---

## SAM 1 Files to Fetch
**GitHub**: https://github.com/facebookresearch/segment-anything
**Target folder**: `source-codebases/sam1-segment-anything/`

### Core Architecture
- `segment_anything/__init__.py` - Package exports
- `segment_anything/build_sam.py` - Model builders (vit_h, vit_l, vit_b)
- `segment_anything/predictor.py` - SamPredictor class (image-level API)
- `segment_anything/automatic_mask_generator.py` - Automatic segmentation

### Model Architecture
- `segment_anything/modeling/sam.py` - Main SAM model class
- `segment_anything/modeling/image_encoder.py` - ViT-H image encoder
- `segment_anything/modeling/prompt_encoder.py` - Prompt encoder (points/boxes)
- `segment_anything/modeling/mask_decoder.py` - Mask decoder
- `segment_anything/modeling/transformer.py` - Transformer blocks

### Utilities
- `segment_anything/utils/transforms.py` - Image transforms
- `segment_anything/utils/amg.py` - Automatic mask generation utilities
- `segment_anything/utils/onnx.py` - ONNX export utilities

### Scripts
- `scripts/amg.py` - Command-line automatic mask generation
- `scripts/export_onnx_model.py` - ONNX export script

### Config
- `setup.py` - Package setup
- `requirements.txt` - Dependencies
- `LICENSE` - Apache 2.0 license
- `README.md` - Documentation

**Estimated files**: ~20-30 files

---

## SAM 2 Files to Fetch
**GitHub**: https://github.com/facebookresearch/sam2
**Target folder**: `source-codebases/sam2-video/`

### Core Architecture
- `sam2/__init__.py` - Package exports
- `sam2/build_sam.py` - Model builders (hiera_l, hiera_b, hiera_t, hiera_s)
- `sam2/sam2_image_predictor.py` - Image predictor API
- `sam2/sam2_video_predictor.py` - Video predictor API (streaming memory)

### Model Architecture
- `sam2/modeling/sam2_base.py` - Base SAM 2 model
- `sam2/modeling/backbones/image_encoder.py` - Hiera encoder
- `sam2/modeling/backbones/hieradet.py` - Hierarchical detector
- `sam2/modeling/memory_attention.py` - Streaming memory attention
- `sam2/modeling/memory_encoder.py` - Memory encoder
- `sam2/modeling/sam/prompt_encoder.py` - Prompt encoder
- `sam2/modeling/sam/mask_decoder.py` - Mask decoder
- `sam2/modeling/sam/transformer.py` - Transformer blocks

### Utilities
- `sam2/utils/misc.py` - Miscellaneous utilities
- `sam2/utils/transforms.py` - Image/video transforms
- `sam2/utils/amg.py` - Automatic mask generation utilities

### Training (SAM 2.1)
- `training/` - Training scripts (released in 2.1)
- `training/train.py` - Main training loop
- `training/dataset/` - Dataset loaders
- `training/configs/` - Training configs

### Config
- `setup.py` - Package setup
- `requirements.txt` - Dependencies
- `pyproject.toml` - Project config
- `LICENSE` - Apache 2.0 license
- `README.md` - Documentation

**Estimated files**: ~40-60 files

---

## SAM 3 Files to Fetch
**GitHub**: https://github.com/facebookresearch/sam3
**Target folder**: `source-codebases/sam3-concepts/`

### Core Architecture
- `sam3/__init__.py` - Package exports
- `sam3/model_builder.py` - Model builders (image + video)
- `sam3/model/sam3_image_processor.py` - Sam3Processor class (text prompts!)
- `sam3/model/sam3_video_predictor.py` - Video predictor with text

### Model Architecture
- `sam3/model/sam3_detector.py` - DETR-based detector (text-conditioned)
- `sam3/model/sam3_tracker.py` - SAM 2-style tracker
- `sam3/model/sam3_encoder.py` - Shared vision encoder
- `sam3/model/presence_token.py` - Presence token (discriminate "red" vs "white")
- `sam3/model/text_encoder.py` - Text encoder for prompts
- `sam3/model/image_encoder.py` - Vision encoder
- `sam3/model/prompt_encoder.py` - Geometry + exemplar prompts
- `sam3/model/mask_decoder.py` - Mask decoder

### Data Engine
- `sam3/data_engine/` - Data annotation pipeline (4M concepts!)
- `sam3/data_engine/auto_annotator.py` - Automatic annotation
- `sam3/data_engine/concept_extractor.py` - Extract concepts from text

### SA-Co Dataset
- `sam3/datasets/saco_gold.py` - SA-Co/Gold loader
- `sam3/datasets/saco_silver.py` - SA-Co/Silver loader
- `sam3/datasets/saco_veval.py` - SA-Co/VEval (video) loader

### Examples
- `examples/sam3_image_predictor_example.ipynb` - Image segmentation
- `examples/sam3_video_predictor_example.ipynb` - Video segmentation
- `examples/sam3_image_batched_inference.ipynb` - Batched inference
- `examples/sam3_agent.ipynb` - SAM 3 Agent (complex prompts)

### Config
- `setup.py` - Package setup
- `requirements.txt` - Dependencies
- `pyproject.toml` - Project config
- `LICENSE` - SAM License
- `README.md` - Documentation

**Estimated files**: ~60-80 files

---

## VARIABLE ZEUS Execution Plan

### Phase 1: Discovery (1 worker per repo)
- [✓] Worker 1: List all files in segment-anything repo (Completed 2025-11-21)
  - Output: KNOWLEDGE-DROP-sam1-source-structure-2025-11-21.md
  - Files discovered: ~25-30 total files, 18-20 Python files
  - Structure: segment_anything/ (13 core files), scripts/ (2), notebooks/ (3), demo/ (~15)
  - Key modules: build_sam.py, predictor.py, automatic_mask_generator.py
  - Model architecture: sam.py, image_encoder.py, prompt_encoder.py, mask_decoder.py, transformer.py
- [ ] Worker 2: List all files in sam2 repo
- [✓] Worker 3: List all files in sam3 repo (Completed 2025-11-21)
  - Output: KNOWLEDGE-DROP-sam3-source-structure-2025-11-21.md
  - Files discovered: 93 Python files in sam3/ directory
  - Structure: agent (26 files), eval (37 files), model (30+ files)
→ Output: Complete file inventory (WORKERS_NEEDED = total_files)

### Phase 2: Fetch (WORKERS_NEEDED parallel workers)
- Each worker: Fetch ONE file from GitHub + create KNOWLEDGE-DROP
- KNOWLEDGE-DROP format:
  ```markdown
  # [Filename] - [Brief description]
  **Path**: [repo]/[path/to/file]
  **Purpose**: [What this file does]
  **Key classes/functions**: [List]
  **Dependencies**: [What it imports]
  **Used by**: [What imports it]
  ```

### Phase 3: Integration (main worker)
- Main worker reviews ALL KNOWLEDGE-DROPs
- Adds claudes_code_comments to ALL Python files
- Organizes into source-codebases/ folders
- Creates INDEX.md for each codebase

---

## Expected Output Structure

```
source-codebases/
├── sam1-segment-anything/
│   ├── INDEX.md (navigation + overview)
│   ├── segment_anything/
│   │   ├── __init__.py (+ claudes comments)
│   │   ├── build_sam.py (+ claudes comments)
│   │   ├── predictor.py (+ claudes comments)
│   │   └── modeling/
│   │       ├── sam.py (+ claudes comments)
│   │       └── ...
│   └── scripts/
├── sam2-video/
│   ├── INDEX.md
│   ├── sam2/
│   │   ├── __init__.py (+ claudes comments)
│   │   ├── sam2_image_predictor.py (+ claudes comments)
│   │   ├── sam2_video_predictor.py (+ claudes comments)
│   │   └── modeling/
│   └── training/ (SAM 2.1)
└── sam3-concepts/
    ├── INDEX.md
    ├── sam3/
    │   ├── __init__.py (+ claudes comments)
    │   ├── model_builder.py (+ claudes comments)
    │   ├── model/
    │   │   ├── sam3_detector.py (+ claudes comments)
    │   │   └── sam3_tracker.py (+ claudes comments)
    │   └── data_engine/
    └── examples/
```

---

## Worker Responsibilities

### Discovery Workers (3 total)
1. **SAM 1 Discovery**: List all files in segment-anything repo
2. [✓] **SAM 2 Discovery**: List all files in sam2 repo (Completed 2025-11-21)
3. **SAM 3 Discovery**: List all files in sam3 repo

### Fetch Workers (VARIABLE count)
- **Each worker**: Fetch ONE file + create KNOWLEDGE-DROP
- **Parallel execution**: All workers run simultaneously
- **No coordination needed**: Each file is independent

### Integration Worker (1 total)
- **Main reviewer**: Read ALL KNOWLEDGE-DROPs
- **Add comments**: claudes_code_comments to ALL Python files
- **Organize**: Move to source-codebases/ folders
- **Document**: Create INDEX.md for each codebase

---

## Success Criteria

✅ **Complete source code** for SAM 1, 2, 3 in `source-codebases/`
✅ **KNOWLEDGE-DROPs** for every file (structure documented)
✅ **Claudes comments** on ALL Python files
✅ **INDEX.md** for each codebase (navigation)
✅ **No missing files** (verified against GitHub repos)

---

## Estimated Totals

- **SAM 1**: ~25 files
- **SAM 2**: ~50 files
- **SAM 3**: ~70 files
- **TOTAL**: ~145 files
- **Workers needed**: 145 + 3 (discovery) + 1 (integration) = 149 workers

**VARIABLE ZEUS scales from 3 → 149 workers dynamically!**
