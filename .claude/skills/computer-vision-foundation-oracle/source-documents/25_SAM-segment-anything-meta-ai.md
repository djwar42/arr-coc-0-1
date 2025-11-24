---
sourceFile: "SAM - Segment Anything Model (Meta AI)"
exportedBy: "Bright Data Web Scraper"
exportDate: "2025-10-28"
sourceURL: "https://segment-anything.com/ and https://arxiv.org/abs/2304.02643"
---

# SAM - Segment Anything Model (Meta AI)

## Overview

The Segment Anything Model (SAM) produces high-quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image.

## Paper Abstract

We introduce the Segment Anything (SA) project: a new task, model, and dataset for image segmentation. Using our efficient model in a data collection loop, we built the largest segmentation dataset to date (by far), with over 1 billion masks on 11M licensed and privacy respecting images.

The model is designed and trained to be promptable, so it can transfer zero-shot to new image distributions and tasks. We evaluate its capabilities on numerous tasks and find that its zero-shot performance is impressive -- often competitive with or even superior to prior fully supervised results.

We are releasing the Segment Anything Model (SAM) and corresponding dataset (SA-1B) of 1B masks and 11M images to foster research into foundation models for computer vision.

## Key Details

**Authors**: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick

**Publication**: Submitted on 5 Apr 2023 to arXiv

**Conference**: Published at ICCV 2023

**Citation Count**: 15,075 citations (as of scrape date)

**ArXiv ID**: [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)

**Official Website**: https://segment-anything.com

**GitHub Repository**: https://github.com/facebookresearch/segment-anything

## Dataset: SA-1B

- **Size**: 1 billion masks on 11 million images
- **Largest segmentation dataset to date**
- **Licensed and privacy-respecting images**
- **Publicly released for research**

## Model Capabilities

1. **Promptable Segmentation**: Accepts various input prompts
   - Points
   - Boxes
   - Masks
   - Text (in some variants)

2. **Zero-shot Transfer**: Works on new image distributions without retraining

3. **Competitive Performance**: Often matches or exceeds prior fully supervised results

4. **Versatile Applications**: Can segment any object in any image with a single click

## Architecture Components

Based on the transformer architecture adapted for vision:
1. **Image Encoder**: Processes the image to extract visual features
2. **Prompt Encoder**: Encodes user prompts (points, boxes, text)
3. **Mask Decoder**: Generates segmentation masks based on image and prompt embeddings

## Subsequent Development: SAM 2

**SAM 2 (2024)**: Extended to video segmentation
- **Paper**: "SAM 2: Segment Anything in Images and Videos" (arXiv:2408.00714)
- **Citation Count**: 2,248 citations
- **Capability**: Foundation model for promptable visual segmentation in both images and videos

## Key Innovations

1. **Data Collection Loop**: Efficient model used to build massive dataset
2. **Promptability**: Design allows flexible interaction paradigms
3. **Scale**: Unprecedented dataset size for segmentation
4. **Generalization**: Strong zero-shot performance across domains

## Applications Mentioned in Literature

From research and practical deployments:
- Interactive annotation tools
- Medical imaging segmentation
- Video object segmentation
- Autonomous driving scene understanding
- Augmented reality object isolation
- Point cloud segmentation (via multi-view projection)

## Related Work

- **Object Detection Integration**: Papers combining SAM with YOLO and other detectors
- **Remote Sensing**: SAM applications in satellite imagery analysis
- **3D Vision**: Using SAM for point cloud instance segmentation
- **Multimodal**: Integration with language models

## Significance

SAM represents a major advancement in computer vision foundation models:
- First truly "universal" segmentation model
- Enables zero-shot segmentation across diverse domains
- Democratizes access to powerful segmentation capabilities
- Largest publicly available segmentation dataset

## Resources

- **Official Website**: https://segment-anything.com
- **Demo**: https://segment-anything.com/demo
- **Paper (ICCV 2023)**: https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf
- **arXiv**: https://arxiv.org/abs/2304.02643
- **GitHub**: https://github.com/facebookresearch/segment-anything
- **Dataset**: Available for download at segment-anything.com

## Impact on Computer Vision Foundation

SAM has become a cornerstone of modern computer vision research, appearing in:
- Thousands of follow-up papers
- Integration with various CV pipelines
- Extension to new modalities (video, 3D)
- Benchmark for foundation model capabilities
