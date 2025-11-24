# Data Composition

**Category**: Training
**Related**: [00-overview.md](00-overview.md)

## OCR Data (P2: 70%)

**Document OCR**: 30M pages
- Sources: PDF corpora, scanned documents
- Languages: 100+ languages
- Annotation: fitz extraction, MinerU, GOT-OCR2.0

**Scene OCR**: 20M images
- Sources: LAION, Wukong
- Annotation: PaddleOCR
- Task: Text extraction from natural scenes

## Grounding Data (P2: 15%)

**Object Detection**: Detection boxes + labels
**Visual Grounding**: Region descriptions
**Spatial Reasoning**: Relationship understanding

## Caption Data (P1 + P2: ~100M)

- CC12M: Conceptual Captions
- LAION subsets
- COCO Captions

## Instruction Data (P3: 200M)

**General QA**: TextVQA, DocVQA, ChartQA
**STEM**: MathVista, ScienceQA
**Medical**: Medical imaging datasets
**Multi-image**: Comparative analysis
**Video**: Temporal understanding
**Thinking-style**: CoT + reflection data

## Preference Data (P4: 10M)

Pairs of (chosen, rejected) responses
- Quality: Clear reasoning vs confused
- Style: Thinking-mode vs vanilla CoT

## RL Data (P5: 5M)

Math problems with verifiable correct answers
- Reward signal based on correctness
