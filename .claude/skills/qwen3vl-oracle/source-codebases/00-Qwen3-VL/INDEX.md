# Qwen3-VL Codebase Index

**Complete Qwen3-VL multimodal vision-language model implementation**

## Quick Start

- **Main README**: [README.md](README.md) - Official documentation
- **Fine-tuning**: [qwen-vl-finetune/](qwen-vl-finetune/) - Fine-tuning code
- **Utils**: [qwen-vl-utils/](qwen-vl-utils/) - Utility functions
- **Evaluation**: [evaluation/](evaluation/) - Benchmark evaluation scripts

## Directory Structure

```
00-Qwen3-VL/
├── qwen-vl-finetune/        # Fine-tuning implementation
├── qwen-vl-utils/            # Utility functions for preprocessing
├── evaluation/               # Evaluation benchmarks
├── cookbooks/                # Example notebooks and guides
├── docker/                   # Docker configurations
├── web_demo_mm.py            # Web demo interface
└── README.md                 # Official project README
```

## Key Components

- **qwen-vl-finetune/** - Custom fine-tuning with Interleaved-MRoPE
- **qwen-vl-utils/src/qwen_vl_utils/** - Image/video preprocessing
- **evaluation/** - MMMU, DocVQA, OCRBench evaluations
- **README.md** - Model usage and API documentation

## Key Features

- **Interleaved-MRoPE**: Multi-resolution positional encoding
- **DeepStack**: Multi-layer visual injection  
- **Timestamp Encoding**: Video understanding
- **Dynamic Resolution**: Native resolution processing

## Documentation Reference

For detailed documentation, see:
- [../../architecture/](../../architecture/) - Architecture documentation
- [../../codebase/](../../codebase/) - Code reference
- [../../concepts/](../../concepts/) - Key concepts explained

This is the complete Qwen3-VL codebase from QwenLM/Qwen3-VL repository.
