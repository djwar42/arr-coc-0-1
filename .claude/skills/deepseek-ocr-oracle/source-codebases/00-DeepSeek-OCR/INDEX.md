# DeepSeek-OCR Codebase Index

**Complete DeepSeek-OCR vision-language model implementation**

## Quick Start

- **Main README**: [README.md](README.md) - Official documentation
- **Core Implementation**: [DeepSeek-OCR-master/](DeepSeek-OCR-master/) - Model code
- **Paper**: [DeepSeek_OCR_paper.pdf](DeepSeek_OCR_paper.pdf) - Research paper

## Directory Structure

```
00-DeepSeek-OCR/
├── DeepSeek-OCR-master/          # Main implementation
│   ├── DeepSeek-OCR-hf/          # HuggingFace integration
│   └── DeepSeek-OCR-vllm/        # vLLM inference
├── DeepSeek_OCR_paper.pdf        # Research paper
├── README.md                      # Setup instructions
└── requirements.txt               # Dependencies
```

## Key Components

- **DeepSeek-OCR-hf/**: HuggingFace model implementation
  - Dual encoder architecture (SAM + CLIP)
  - Dynamic multi-crop processing ("Gundam mode")
  - Token compression (64-400 tokens per image)

- **DeepSeek-OCR-vllm/**: vLLM inference optimizations
  - Streaming inference
  - Grounding visualization
  - High-throughput batch processing

## Key Features

- **Optical Compression**: 64-400 tokens via resolution modes
- **Dual Encoders**: SAM (structural) + CLIP (semantic)
- **Gundam Mode**: 1024×1024 global + n×640×640 crops
- **2D Spatial Layout**: Preserves visual structure for LLM

## Documentation Reference

For detailed documentation, see:
- [../../architecture/](../../architecture/) - Architecture documentation
- [../../code-reference/](../../code-reference/) - Code reference
- [../../concepts/](../../concepts/) - Key concepts explained

This codebase includes Claude's code comments added during Phase 3 documentation (2025-10-28).
