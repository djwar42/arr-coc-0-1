# SA-Co/Silver Dataset Structure

## Overview

SA-Co/Silver is an image benchmark for **promptable concept segmentation (PCS)** released alongside SAM 3. Unlike SA-Co/Gold which features triple annotation for high confidence, Silver provides single-annotation ground truth at larger scale across diverse domains.

**Key Distinction from Gold**: Silver has only **single ground-truth per datapoint**, which means results may have more variance and tend to **underestimate model performance** since they don't account for different valid interpretations of queries.

## Dataset Structure

### 10 Diverse Domains

SA-Co/Silver comprises **10 subsets** covering diverse visual domains:

1. **BDD100k** - Berkeley Driving Dataset (autonomous driving scenes)
2. **DROID** - Robotics manipulation dataset
3. **Ego4D** - First-person/egocentric video frames
4. **MyFoodRepo-273** - Food recognition images
5. **GeoDE** - Geographic diversity dataset
6. **iNaturalist-2017** - Wildlife and nature species
7. **National Gallery of Art** - Artwork and paintings
8. **SA-V** - Segment Anything Video frames
9. **YT-Temporal-1B** - YouTube video frames
10. **Fathomnet** - Underwater marine imagery

### Scale and Statistics

| Domain                   | # Image-NPs  | # Image-NP-Masks |
|--------------------------|--------------|------------------|
| BDD100k                  | 5,546        | 13,210           |
| DROID                    | 9,445        | 11,098           |
| Ego4D                    | 12,608       | 24,049           |
| MyFoodRepo-273           | 20,985       | 28,347           |
| GeoDE                    | 14,850       | 7,570            |
| iNaturalist-2017         | 1,439,051    | 48,899           |
| National Gallery of Art  | 22,294       | 18,991           |
| SA-V                     | 18,337       | 39,683           |
| YT-Temporal-1B           | 7,816        | 12,221           |
| Fathomnet                | 287,193      | 14,174           |

**Note**: iNaturalist-2017 has the largest number of image-NP pairs (1.4M+), while Fathomnet has 287K+ pairs, demonstrating Silver's massive scale compared to Gold.

## Quality Characteristics

### Single Annotation vs Gold's Triple Annotation

- **Gold**: Each image-NP pair has **3 independent annotations** for high inter-annotator agreement
- **Silver**: Single annotation per datapoint for **scale efficiency**

### Performance Implications

From the SA-Co/Silver README:
> "Unlike SA-Co/Gold, there is only a single ground-truth for each datapoint, which means the results may have a bit more variance and tend to underestimate model performance, since they don't account for possible different interpretations of each query."

This means:
- Models may be **wrongly penalized** for valid but different interpretations
- Results have **higher variance** compared to Gold
- Scores are generally **conservative estimates** of true performance

## Annotation Format

SA-Co/Silver uses a format derived from **COCO format** with specific fields for PCS:

### Images Entry
```json
{
  "id": 10000000,
  "file_name": "path/to/image.jpg",
  "text_input": "the large wooden table",
  "width": 1280,
  "height": 720,
  "is_instance_exhaustive": 1,
  "is_pixel_exhaustive": 1
}
```

Key fields:
- **text_input**: The noun phrase (text prompt) for this image-NP pair
- **is_instance_exhaustive**: Boolean - if 1, all instances are correctly annotated
- **is_pixel_exhaustive**: Boolean - if 1, all pixels for the concept are covered (allows crowd segments)

### Annotations Entry
```json
{
  "area": 0.173,
  "id": 1,
  "image_id": 10000000,
  "bbox": [0.037, 0.508, 0.838, 0.491],
  "segmentation": {"counts": "...", "size": [720, 1280]},
  "category_id": 1,
  "iscrowd": 0
}
```

Key fields:
- **bbox**: Bounding box in [x, y, w, h] format, **normalized by image dimensions**
- **segmentation**: Mask in **RLE (Run-Length Encoding)** format
- **iscrowd**: Boolean - if 1, segment overlaps multiple instances

### Positive vs Negative Prompts

- **Positive NPs**: Image-NP pairs that have corresponding annotations (objects present)
- **Negative NPs**: Image-NP pairs with no annotations (concept not present in image)

Negative prompts are crucial for evaluating a model's ability to recognize when a concept is absent.

## Use Cases

### Primary Use: Evaluation Benchmark

SA-Co/Silver serves as an **evaluation benchmark** for promptable concept segmentation, complementing Gold with:
- Larger scale (more diverse test cases)
- More domain coverage (10 specialized domains)
- Broader concept vocabulary

### Official Metric: cgF1

The official evaluation metric for SA-Co/Silver is **cgF1** (concept-grounded F1), which measures how well the model:
1. Recognizes concepts (classification)
2. Localizes them precisely (segmentation)

### Not for Training

SA-Co/Silver is an **evaluation-only** benchmark. Training data for SAM 3 comes from:
- **SA-Co/HQ**: 5.2M images with 4M unique concepts (high-quality human annotations)
- **SA-Co/Synthetic**: 38M phrases with 1.4B masks (model-generated)

## SAM 3 Performance on Silver

| Model       | Average cgF1 | IL_MCC | PmF1  |
|-------------|--------------|--------|-------|
| gDino-T     | 3.09         | 0.12   | 19.75 |
| OWLv2*      | 11.23        | 0.32   | 31.18 |
| OWLv2       | 8.18         | 0.23   | 32.55 |
| LLMDet-L    | 6.73         | 0.17   | 28.19 |
| Gemini 2.5  | 9.67         | 0.19   | 45.51 |
| **SAM 3**   | **49.57**    | **0.76**| **65.17** |

SAM 3 achieves **49.57 cgF1** on average across all 10 domains - approximately **4-5x better** than competitors.

### Per-Domain Performance

SAM 3's best performance:
- **GeoDE**: 70.07 cgF1
- **National Gallery of Art**: 65.80 cgF1
- **MyFoodRepo-273**: 52.96 cgF1
- **Fathomnet**: 51.53 cgF1

SAM 3's challenging domains:
- **Ego4D**: 38.64 cgF1
- **National Gallery of Art (markup)**: 38.06 cgF1

## Accessing SA-Co/Silver

### Download Sources

**HuggingFace**: [facebook/SACo-Silver](https://huggingface.co/datasets/facebook/SACo-Silver)

**Roboflow**: [universe.roboflow.com/sa-co-silver](https://universe.roboflow.com/sa-co-silver)

### Image Preparation

Most domains require preprocessing steps:
- Download raw images from original sources (BDD100k, Ego4D, etc.)
- Run preprocessing scripts provided in `sam3/scripts/eval/silver/`
- Or download pre-processed images directly from Roboflow

### Visualization

Example notebook: `saco_gold_silver_vis_example.ipynb`

### Running Evaluation

```bash
# Edit paths in eval_base.yaml first
python sam3/train/train.py -c configs/silver_image_evals/sam3_gold_image_bdd100k.yaml --use-cluster 0 --num-gpus 1
```

## Comparison: Gold vs Silver

| Aspect | SA-Co/Gold | SA-Co/Silver |
|--------|------------|--------------|
| Annotations per datapoint | 3 (triple) | 1 (single) |
| Purpose | High-confidence benchmark | Large-scale benchmark |
| Domains | 7 domains | 10 domains |
| Variance | Lower | Higher |
| Performance estimates | More accurate | Conservative |
| Scale | Smaller | Larger |

## Technical Details

### Frame Datasets

Some Silver domains (DROID, SA-V, Ego4D, YT-Temporal-1B) are **frame datasets** extracted from videos:
- Require video download and frame extraction
- Use `CONFIG_FRAMES.yaml` for unified configuration
- Scripts provided for each source

### Special Requirements

- **Ego4D**: Requires license agreement and AWS credentials
- **YT-Temporal-1B**: Requires YouTube cookies for download
- **SA-V**: Requires access request to Meta for download links

## Sources

**Primary Sources:**
- [SA-Co/Silver README](https://github.com/facebookresearch/sam3/blob/main/scripts/eval/silver/README.md) - Official documentation (accessed 2025-11-23)
- [SAM 3 GitHub Repository](https://github.com/facebookresearch/sam3) - Main repository (accessed 2025-11-23)

**Additional References:**
- [Meta AI Blog: SAM 3](https://ai.meta.com/blog/segment-anything-model-3/) - Official announcement
- [HuggingFace: SA-Co/Silver](https://huggingface.co/datasets/facebook/SACo-Silver) - Dataset hosting
- [Roboflow: SA-Co/Silver](https://universe.roboflow.com/sa-co-silver) - Alternative hosting with preprocessing

**Related KNOWLEDGE-DROPs:**
- SA-Co/Gold Dataset Structure (Worker 33)
- SA-Co/VEval Video Benchmark (Worker 35)
- Evaluation Metrics (Worker 36)
