#

 Computer Vision Foundation Oracle - Master Index

Comprehensive cross-referenced index of all knowledge in the Computer Vision Foundation Oracle.

## Quick Start Guide

**New to this oracle?** Start here:
1. Read [SKILL.md](SKILL.md) for overview and navigation
2. Choose your topic area below
3. Start with `00-overview.md` in that topic folder
4. Drill down into specific source documents as needed

---

## 📚 Topic Areas

### 🎯 Segmentation
**Overview**: [segmentation/00-overview.md](segmentation/00-overview.md)

**Key Concepts**:
- Segment Anything Model (SAM)
- Instance vs semantic segmentation
- Point cloud segmentation
- Weak and noisy supervision
- 3D bounding box integration

**Files**:
- [01-sam-segment-anything.md](segmentation/01-sam-segment-anything.md) - SAM architecture, applications, integration with 3D

**Primary Sources**:
- [25_SAM-segment-anything-meta-ai.md](source-documents/25_SAM-segment-anything-meta-ai.md) - Official Meta AI documentation
- [05_When 3D Bounding-Box Meets SAM.md](source-documents/05_When%203D%20Bounding-Box%20Meets%20SAM_%20Point%20Cloud%20Instance%20Segmentation%20With%20Weak-and-Noisy%20Supervision.md) - 3D point cloud segmentation

### 🧊 3D Vision
**Overview**: [3d-vision/00-overview.md](3d-vision/00-overview.md)

**Key Concepts**:
- Monocular depth estimation
- Point cloud processing
- Compositional 3D understanding
- 3D bounding boxes
- Multi-view fusion

**Primary Sources**:
- [06_23611 - 3rd Monocular Depth Estimation Challenge.md](source-documents/06_23611%20-%203rd%20Monocular%20Depth%20Estimation%20Challenge.md)
- [07_23612 - 2nd Workshop on Compositional 3D Vision.md](source-documents/07_23612%20%20%202nd%20Workshop%20on%20Compositional%203D%20Vision.md)

### 🧠 Multimodal Reasoning
**Overview**: [multimodal-reasoning/00-overview.md](multimodal-reasoning/00-overview.md)

**Key Concepts**:
- Neural algorithmic reasoning
- CLRS-30 benchmark
- Auto-regressive bottleneck
- TransNAR architecture
- Out-of-distribution generalization

**Primary Sources**:
- [26_CLRS-algorithmic-reasoning-benchmark.md](source-documents/26_CLRS-algorithmic-reasoning-benchmark.md) - Google DeepMind benchmark
- [08_23568 - Multimodal Algorithmic Reasoning Workshop.md](source-documents/08_23568%20%20%20Multimodal%20Algorithmic%20Reasoning%20Workshop.md)

### 🏗️ Foundation Models
**Overview**: [foundation-models/00-overview.md](foundation-models/00-overview.md)

**Key Concepts**:
- Vision Transformers (ViT)
- Transformer architectures
- Self-attention mechanisms
- Auto-regressive vs non-auto-regressive
- Training strategies

**Primary Sources**:
- [27_vision-transformers-vit-guide.md](source-documents/27_vision-transformers-vit-guide.md) - ViT architecture guide

### 🏆 Workshops & Challenges
**Overview**: [workshops-challenges/00-overview.md](workshops-challenges/00-overview.md)

**Key Events**:
- Multimodal Algorithmic Reasoning Workshop
- 3rd Monocular Depth Estimation Challenge
- 2nd Workshop on Compositional 3D Vision

---

## 🔑 Key Concepts Index

### Architectures
- **SAM (Segment Anything Model)**: [segmentation/01-sam-segment-anything.md](segmentation/01-sam-segment-anything.md), [source-documents/25_SAM*.md](source-documents/25_SAM-segment-anything-meta-ai.md)
- **Vision Transformer (ViT)**: [foundation-models/00-overview.md](foundation-models/00-overview.md), [source-documents/27_vision*.md](source-documents/27_vision-transformers-vit-guide.md)
- **TransNAR**: [multimodal-reasoning/00-overview.md](multimodal-reasoning/00-overview.md)
- **Graph Neural Networks**: [multimodal-reasoning/00-overview.md](multimodal-reasoning/00-overview.md)

### Techniques
- **Self-attention**: [foundation-models/00-overview.md](foundation-models/00-overview.md), [segmentation/01-sam*.md](segmentation/01-sam-segment-anything.md)
- **Multi-head attention**: [foundation-models/00-overview.md](foundation-models/00-overview.md)
- **Zero-shot transfer**: [segmentation/01-sam*.md](segmentation/01-sam-segment-anything.md), [foundation-models/00-overview.md](foundation-models/00-overview.md)
- **Weak supervision**: [segmentation/00-overview.md](segmentation/00-overview.md), [3d-vision/00-overview.md](3d-vision/00-overview.md)

### Challenges
- **Auto-regressive bottleneck**: [multimodal-reasoning/00-overview.md](multimodal-reasoning/00-overview.md), [foundation-models/00-overview.md](foundation-models/00-overview.md)
- **Out-of-distribution generalization**: [multimodal-reasoning/00-overview.md](multimodal-reasoning/00-overview.md)
- **Data scarcity**: [3d-vision/00-overview.md](3d-vision/00-overview.md)
- **Noisy annotations**: [segmentation/00-overview.md](segmentation/00-overview.md), [3d-vision/00-overview.md](3d-vision/00-overview.md)

### Applications
- **Image segmentation**: [segmentation/00-overview.md](segmentation/00-overview.md)
- **Depth estimation**: [3d-vision/00-overview.md](3d-vision/00-overview.md), [workshops-challenges/00-overview.md](workshops-challenges/00-overview.md)
- **Point cloud processing**: [segmentation/01-sam*.md](segmentation/01-sam-segment-anything.md), [3d-vision/00-overview.md](3d-vision/00-overview.md)
- **Algorithmic reasoning**: [multimodal-reasoning/00-overview.md](multimodal-reasoning/00-overview.md)

---

## 📄 Source Documents

### Supplemental Research (Web-Scraped)

**25_SAM-segment-anything-meta-ai.md**
- **Topic**: Segment Anything Model
- **Source**: Meta AI official documentation
- **Key Content**: SAM architecture, SA-1B dataset, zero-shot performance, SAM 2
- **Citations**: 15,075 (ICCV 2023)
- **Related Topics**: Segmentation, foundation models

**26_CLRS-algorithmic-reasoning-benchmark.md**
- **Topic**: Algorithmic reasoning benchmark
- **Source**: Google DeepMind GitHub and paper
- **Key Content**: CLRS-30 algorithms, dataset structure, baseline models, CLRS-Text
- **Citations**: 134 (ICML 2022)
- **Related Topics**: Multimodal reasoning, graph neural networks

**27_vision-transformers-vit-guide.md**
- **Topic**: Vision Transformer architecture
- **Source**: Comprehensive ViT guide
- **Key Content**: ViT vs CNN, architecture components, training strategies, benchmarks
- **Related Topics**: Foundation models, transformers, attention mechanisms

### Computer Vision Foundation Materials (00-24)

**Workshop Presentations**:
- 08_23568 - Multimodal Algorithmic Reasoning Workshop
- 06_23611 - 3rd Monocular Depth Estimation Challenge
- 07_23612 - 2nd Workshop on Compositional 3D Vision

**Research Papers**:
- 05_When 3D Bounding-Box Meets SAM (Point cloud segmentation)
- Multiple YouTube video transcripts from CV Foundation channel

---

## 🔍 Search by Question Type

### "How does...?"
- "How does SAM work?" → [segmentation/01-sam-segment-anything.md](segmentation/01-sam-segment-anything.md)
- "How does ViT differ from CNNs?" → [foundation-models/00-overview.md](foundation-models/00-overview.md)
- "How does self-attention work?" → [foundation-models/00-overview.md](foundation-models/00-overview.md)

### "What is...?"
- "What is SAM?" → [segmentation/01-sam-segment-anything.md](segmentation/01-sam-segment-anything.md)
- "What is CLRS?" → [source-documents/26_CLRS*.md](source-documents/26_CLRS-algorithmic-reasoning-benchmark.md)
- "What is compositional 3D vision?" → [3d-vision/00-overview.md](3d-vision/00-overview.md)

### "Why do...?"
- "Why do LLMs fail at counting?" → [multimodal-reasoning/00-overview.md](multimodal-reasoning/00-overview.md)
- "Why use transformers for vision?" → [foundation-models/00-overview.md](foundation-models/00-overview.md)
- "Why is weak supervision challenging?" → [3d-vision/00-overview.md](3d-vision/00-overview.md)

### "When to use...?"
- "When to use SAM?" → [segmentation/01-sam-segment-anything.md](segmentation/01-sam-segment-anything.md)
- "When to use ViT vs CNN?" → [foundation-models/00-overview.md](foundation-models/00-overview.md)
- "When is multimodal reasoning needed?" → [multimodal-reasoning/00-overview.md](multimodal-reasoning/00-overview.md)

---

## 📊 Benchmarks & Datasets

### Major Benchmarks
- **CLRS-30**: [source-documents/26_CLRS*.md](source-documents/26_CLRS-algorithmic-reasoning-benchmark.md), [multimodal-reasoning/00-overview.md](multimodal-reasoning/00-overview.md)
- **ImageNet**: [foundation-models/00-overview.md](foundation-models/00-overview.md), [source-documents/27_vision*.md](source-documents/27_vision-transformers-vit-guide.md)
- **COCO**: [segmentation/00-overview.md](segmentation/00-overview.md), [foundation-models/00-overview.md](foundation-models/00-overview.md)
- **ScanNet V2**: [segmentation/01-sam*.md](segmentation/01-sam-segment-anything.md)

### Datasets
- **SA-1B** (1B masks, 11M images): [source-documents/25_SAM*.md](source-documents/25_SAM-segment-anything-meta-ai.md)
- **ImageNet-21k**: [foundation-models/00-overview.md](foundation-models/00-overview.md)
- **ADE20K**: [foundation-models/00-overview.md](foundation-models/00-overview.md)

---

## 🔬 Research Papers Cross-Reference

### Highly Cited Papers
- **SAM (15,075 citations)**: [source-documents/25_SAM*.md](source-documents/25_SAM-segment-anything-meta-ai.md)
- **ViT (ICLR 2021)**: [source-documents/27_vision*.md](source-documents/27_vision-transformers-vit-guide.md)
- **CLRS (134 citations)**: [source-documents/26_CLRS*.md](source-documents/26_CLRS-algorithmic-reasoning-benchmark.md)

### Workshop Papers
- **Multimodal Algorithmic Reasoning**: [source-documents/08_23568*.md](source-documents/08_23568%20%20%20Multimodal%20Algorithmic%20Reasoning%20Workshop.md)
- **Depth Estimation Challenge**: [source-documents/06_23611*.md](source-documents/06_23611%20-%203rd%20Monocular%20Depth%20Estimation%20Challenge.md)
- **Compositional 3D Vision**: [source-documents/07_23612*.md](source-documents/07_23612%20%20%202nd%20Workshop%20on%20Compositional%203D%20Vision.md)

---

## 🎓 Learning Paths

### Beginner: Understanding Segmentation
1. [segmentation/00-overview.md](segmentation/00-overview.md) - Start here
2. [segmentation/01-sam-segment-anything.md](segmentation/01-sam-segment-anything.md) - SAM basics
3. [source-documents/25_SAM*.md](source-documents/25_SAM-segment-anything-meta-ai.md) - Official documentation

### Intermediate: Vision Transformers
1. [foundation-models/00-overview.md](foundation-models/00-overview.md) - Transformer overview
2. [source-documents/27_vision*.md](source-documents/27_vision-transformers-vit-guide.md) - ViT architecture
3. [multimodal-reasoning/00-overview.md](multimodal-reasoning/00-overview.md) - Limitations and solutions

### Advanced: Algorithmic Reasoning
1. [multimodal-reasoning/00-overview.md](multimodal-reasoning/00-overview.md) - Core concepts
2. [source-documents/26_CLRS*.md](source-documents/26_CLRS-algorithmic-reasoning-benchmark.md) - CLRS benchmark
3. [source-documents/08_23568*.md](source-documents/08_23568%20%20%20Multimodal%20Algorithmic%20Reasoning%20Workshop.md) - Full workshop

### Expert: 3D Vision Systems
1. [3d-vision/00-overview.md](3d-vision/00-overview.md) - Foundations
2. [segmentation/01-sam-segment-anything.md](segmentation/01-sam-segment-anything.md) - SAM for 3D
3. [workshops-challenges/00-overview.md](workshops-challenges/00-overview.md) - Current challenges

---

## 🔗 Related Oracles

- **deepseek-ocr-oracle**: Vision-language OCR models, optical compression
- **ovis-2-5-oracle**: Ovis 2.5 multimodal VLM, native resolution processing
- **john-vervaeke-oracle**: Relevance realization, cognitive science foundations

---

## 📈 Statistics

- **Total Source Documents**: 28
- **Topic Areas**: 5 (Segmentation, 3D Vision, Multimodal Reasoning, Foundation Models, Workshops)
- **Detailed Content Files**: 6 (5 overviews + 1 deep dive)
- **Combined Citations**: 15,000+ (SAM, ViT, CLRS papers)
- **Coverage Period**: 2020-2025
- **Primary Conferences**: CVPR, ICCV, ICML, ICLR

---

## 🛠️ Maintenance Notes

**Last Updated**: 2025-10-28
**Source Quality**: High (official papers, workshops, Meta AI, Google DeepMind)
**Comprehensiveness**: Extensive (28 sources + 5 extracted topics)
**Cross-Referencing**: Complete (all files linked)

---

**Navigation Tip**: Use your browser's search (Ctrl/Cmd+F) to find specific topics in this index, then follow the links to relevant files.
