# ovis-2-5-oracle Index

**Quick Navigation**: [Architecture](#architecture) | [Training](#training) | [Codebase](#codebase) | [Usage](#usage) | [Concepts](#concepts) | [References](#references) | [Examples](#examples)

**Total Documentation Files: 42** (because that's the answer to everything!)

---

## Architecture

Complete system architecture and component documentation

- [00-overview.md](architecture/00-overview.md) - Complete system architecture
  - System diagram with all components
  - Data flow: Image → NaViT → VT → VET → Qwen3 → Text

- [01-navit-vision.md](architecture/01-navit-vision.md) - Native-resolution ViT
  - SigLIP 2 backbone
  - RoPE integration for spatial awareness
  - Code: `ovis/model/vit/modeling_siglip2_navit.py`

- [02-visual-tokenizer.md](architecture/02-visual-tokenizer.md) - Visual feature extraction
  - ViT encoding
  - Visual head (Linear + LayerNorm)
  - Smart resize algorithm
  - Code: `ovis/model/modeling_ovis.py:36-189`

- [03-visual-embedding-table.md](architecture/03-visual-embedding-table.md) - Structural alignment
  - Probabilistic lookup mechanism
  - Why VET matters for alignment
  - Training dynamics
  - Code: `ovis/model/modeling_ovis.py:25-34`

- [04-qwen3-llm.md](architecture/04-qwen3-llm.md) - Language decoder
  - Qwen3 architecture
  - Why Qwen3 over Qwen2.5
  - Deep reasoning capabilities
  - Code: LLM integration in `modeling_ovis.py:206-211`

- [05-multimodal-merging.md](architecture/05-multimodal-merging.md) - Text + vision fusion
  - Embedding replacement mechanism
  - Attention mask handling
  - Label creation for training
  - Code: `merge_multimodal()` method

- [06-thinking-mode.md](architecture/06-thinking-mode.md) - Reflection system
  - Two-phase generation
  - Thinking budget
  - `<think>` tag handling
  - Self-correction mechanism
  - Code: `generate()` with `enable_thinking`

- [07-data-flow.md](architecture/07-data-flow.md) - Complete execution trace
  - Step-by-step pipeline
  - Tensor shapes at each stage
  - Timing and memory analysis

---

## Training

Five-phase progressive training curriculum

- [00-overview.md](training/00-overview.md) - Training philosophy
  - 5-phase curriculum diagram
  - Progressive capability building
  - Module training progression
  - Resolution progression

- [01-phase-p1-vet.md](training/01-phase-p1-vet.md) - VET pre-training
  - Goal: Train Visual Embedding Table
  - Modules: VT (partial), VET, Visual Head
  - Frozen: Most ViT, all LLM
  - Data: Image-caption pairs
  - Resolution: 448²-896²
  - RoPE: Disabled

- [02-phase-p2-multimodal.md](training/02-phase-p2-multimodal.md) - Core visual understanding
  - Goal: Full-parameter multimodal training
  - Modules: ALL trainable
  - Data: OCR, grounding, captions
  - Resolution: 448²-1792² (2× expansion)
  - RoPE: Activated

- [03-phase-p3-instruction.md](training/03-phase-p3-instruction.md) - Instruction tuning
  - Goal: Instruction following + deep reasoning
  - Data: Text, image, multi-image, video, thinking-style
  - Domains: QA, STEM, medical, multilingual

- [04-phases-p4-p5-rl.md](training/04-phases-p4-p5-rl.md) - RL optimization
  - P4: DPO (Direct Preference Optimization)
  - P5: GRPO (Group Relative Policy Optimization)
  - Preference alignment
  - Reasoning optimization

- [05-data-composition.md](training/05-data-composition.md) - Dataset details
  - OCR data sources (30M pages)
  - Grounding data pipeline
  - Reasoning data (CoT + thinking-style)
  - Quality filtering

- [06-infrastructure.md](training/06-infrastructure.md) - Training setup
  - Data packing mechanism
  - Hybrid parallelism (DP + TP + CP)
  - DeepSpeed ZeRO-3
  - 3-4× speedup techniques

---

## Codebase

File-by-file code documentation with line numbers

- [00-structure.md](codebase/00-structure.md) - Repository organization
  - Directory structure
  - File organization
  - Import dependencies
  - Module relationships

- [01-modeling-ovis.md](codebase/01-modeling-ovis.md) - Core model
  - File: `ovis/model/modeling_ovis.py`
  - Classes: VisualEmbedding, VisualTokenizer, Ovis
  - Key methods with line numbers
  - Usage examples

- [02-visual-tokenizer-impl.md](codebase/02-visual-tokenizer-impl.md) - Tokenizer details
  - `__init__()` implementation
  - `smart_resize()` algorithm (lines 59-98)
  - `preprocess()` pipeline
  - `forward()` complete flow

- [03-conversation-formatter.md](codebase/03-conversation-formatter.md) - Chat templates
  - File: `ovis/model/conversation_formatter.py`
  - Classes: ConversationFormatter, Qwen3ConversationFormatter
  - Chat template format
  - Label generation logic

- [04-training-loop.md](codebase/04-training-loop.md) - Training script
  - File: `ovis/train/train.py`
  - `load_model()` function
  - `load_data()` function
  - `train()` main loop
  - Module selection system

- [05-datasets.md](codebase/05-datasets.md) - Data loading
  - Files: `ovis/train/dataset/*.py`
  - CaptionDataset for P1
  - ConversationDataset for P2+
  - DataCollator batching

---

## Usage

Practical guides for using Ovis

- [00-quickstart.md](usage/00-quickstart.md) - Get started in 5 minutes
  - Installation
  - Basic inference
  - First example

- [01-huggingface-integration.md](usage/01-huggingface-integration.md) - HF Hub
  - Model loading
  - `from_pretrained()` options
  - `trust_remote_code` requirement
  - Tokenizer usage

- [02-advanced-features.md](usage/02-advanced-features.md) - Power features
  - Multi-image processing
  - Video understanding
  - Thinking mode
  - Resolution control
  - Visual grounding

- [03-fine-tuning.md](usage/03-fine-tuning.md) - Custom training
  - Data preparation
  - Dataset format
  - Training script
  - DeepSpeed config
  - Module selection

---

## Concepts

Deep dives into key ideas

- [00-structural-alignment.md](concepts/00-structural-alignment.md) - Core innovation
  - Why VET matters
  - Discrete vs continuous embeddings
  - Mathematical formulation
  - Benefits for cross-modal learning

- [01-probabilistic-vte.md](concepts/01-probabilistic-vte.md) - VTE explained
  - Probabilistic lookup mechanism
  - Math: `embedding = Σᵢ (pᵢ × eᵢ)`
  - vs discrete indexing
  - Training dynamics

- [02-native-resolution.md](concepts/02-native-resolution.md) - No fixed tiling
  - Fixed tiling problems
  - Native resolution benefits
  - Smart resize algorithm
  - Aspect ratio preservation
  - Grid encoding

- [03-thinking-reflection.md](concepts/03-thinking-reflection.md) - Deep reasoning
  - `<think>` tag format
  - Self-correction process
  - Reflection vs linear CoT
  - Training data requirements

- [04-rope-positional.md](concepts/04-rope-positional.md) - Rotary embeddings
  - RoPE in ViT blocks
  - Spatial awareness benefits
  - Implementation details
  - Why in every block

- [05-vervaeke-comparison.md](concepts/05-vervaeke-comparison.md) - ARR-COC-VIS parallels
  - Relevance realization vs attention
  - Ovis's structural alignment
  - Complementary approaches
  - Potential integration points

---

## References

Quick lookup tables

- [00-api-reference.md](references/00-api-reference.md) - Complete API
  - `Ovis.chat()` - Full signature
  - `Ovis.generate()` - Parameters
  - `Ovis.preprocess_inputs()` - Usage
  - `VisualTokenizer` methods
  - All public methods documented

- [01-model-config.md](references/01-model-config.md) - Configuration
  - OvisConfig parameters
  - visual_tokenizer_config
  - llm_config
  - visual_vocab_size
  - hidden_size
  - multimodal_max_length

---

## Examples

Copy-paste ready code

- [00-basic-inference.md](examples/00-basic-inference.md) - Hello World
  - Complete working example
  - Load model
  - Load image
  - Generate response

- [01-thinking-mode.md](examples/01-thinking-mode.md) - Deep reasoning
  - Enable thinking
  - Set thinking budget
  - Extract thinking and answer
  - Parse `<think>` tags

- [02-multi-image.md](examples/02-multi-image.md) - Multiple images
  - Load multiple images
  - Format prompt correctly
  - Grid encoding
  - Handle responses

- [03-video-processing.md](examples/03-video-processing.md) - Video input
  - Frame sampling
  - Temporal encoding
  - Video token handling
  - Best practices

- [04-fine-tuning-script.md](examples/04-fine-tuning-script.md) - Custom training
  - Complete training script
  - Dataset preparation
  - DeepSpeed config
  - Module selection
  - Best practices

---

## Search by Topic

### Native Resolution
- [architecture/01-navit-vision.md](architecture/01-navit-vision.md)
- [concepts/02-native-resolution.md](concepts/02-native-resolution.md)
- [codebase/02-visual-tokenizer-impl.md](codebase/02-visual-tokenizer-impl.md) - `smart_resize()` method

### Visual Embedding Table (VET)
- [architecture/03-visual-embedding-table.md](architecture/03-visual-embedding-table.md)
- [concepts/01-probabilistic-vte.md](concepts/01-probabilistic-vte.md)
- [training/01-phase-p1-vet.md](training/01-phase-p1-vet.md)
- [codebase/01-modeling-ovis.md](codebase/01-modeling-ovis.md) - VisualEmbedding class

### Thinking Mode
- [architecture/06-thinking-mode.md](architecture/06-thinking-mode.md)
- [concepts/03-thinking-reflection.md](concepts/03-thinking-reflection.md)
- [examples/01-thinking-mode.md](examples/01-thinking-mode.md)
- [usage/02-advanced-features.md](usage/02-advanced-features.md)

### Training Phases
- [training/01-phase-p1-vet.md](training/01-phase-p1-vet.md) - VET initialization
- [training/02-phase-p2-multimodal.md](training/02-phase-p2-multimodal.md) - Full-parameter
- [training/03-phase-p3-instruction.md](training/03-phase-p3-instruction.md) - Instruction tuning
- [training/04-phases-p4-p5-rl.md](training/04-phases-p4-p5-rl.md) - RL optimization

### Multimodal Integration
- [architecture/05-multimodal-merging.md](architecture/05-multimodal-merging.md)
- [examples/02-multi-image.md](examples/02-multi-image.md)
- [examples/03-video-processing.md](examples/03-video-processing.md)

---

## Code Location Quick Reference

### Core Model Files
```
ovis/model/
├── modeling_ovis.py           # Main model, VT, VET
├── configuration_ovis.py      # Model config
├── conversation_formatter.py  # Chat templates
└── vit/
    ├── modeling_siglip2_navit.py   # Vision transformer
    └── configuration_siglip2_navit.py
```

### Training Files
```
ovis/train/
├── train.py                   # Main training script
├── arguments.py               # Training arguments
└── dataset/
    ├── caption_dataset.py     # P1 dataset
    └── conversation_dataset.py # P2+ dataset
```

### Inference Files
```
ovis/serve/
├── infer_basic_demo.py        # Basic examples
├── infer_think_demo.py        # Thinking mode
└── web_ui.py                  # Gradio demo
```

---

**Ready to explore? Start with [architecture/00-overview.md](architecture/00-overview.md) for the big picture, or jump directly to any topic above!**
