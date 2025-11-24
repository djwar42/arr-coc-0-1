# SAM 1 Overview: The Original Foundation Model for Segmentation

**Model**: Segment Anything Model (SAM)
**Release**: April 5, 2023
**Organization**: Meta AI Research (FAIR)
**Status**: Foundation model for image segmentation
**Impact**: 15,632+ citations (as of 2025), 52.6k GitHub stars

---

## 1. Introduction & Context

### What is SAM?

The **Segment Anything Model (SAM)** represents a paradigm shift in computer vision, bringing the foundation model approach—previously successful in NLP (GPT) and vision-language (CLIP)—to the domain of image segmentation.

From [Segment Anything (arXiv:2304.02643)](https://arxiv.org/abs/2304.02643) (Kirillov et al., Meta AI, 2023):
> "We introduce the Segment Anything (SA) project: a new task, model, and dataset for image segmentation. Using our efficient model in a data collection loop, we built the largest segmentation dataset to date (by far), with over 1 billion masks on 11M licensed and privacy respecting images."

**Key Innovation**: SAM is designed to be **promptable**—users can guide segmentation through points, boxes, masks, or text, enabling zero-shot transfer to new tasks and domains without retraining.

From [SAM_STUDY_GENERAL.md](../../../PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md) lines 25-55:
> "Before SAM, segmentation models required large annotated datasets for each domain, task-specific training, and domain expertise. With SAM, zero-shot transfer to new domains, promptable interface (no retraining needed), and general-purpose foundation model capabilities became possible."

### Why SAM Matters: The Foundation Model Paradigm Shift

**Traditional Segmentation (Pre-SAM)**:
- Domain-specific models (medical, satellite, autonomous driving)
- Requires 1,000s-100,000s of annotated masks per domain
- Task-specific architectures (semantic vs instance vs panoptic)
- No generalization across domains

**SAM's Foundation Model Approach**:
- **Single unified model** for all segmentation tasks
- **Zero-shot generalization** to unseen domains
- **Promptable interface** replaces task-specific training
- **1.1 billion masks** on 11M images (SA-1B dataset)

From [GitHub: facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) (accessed 2025-11-20):
> "The Segment Anything Model (SAM) produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a dataset of 11 million images and 1.1 billion masks, and has strong zero-shot performance on a variety of segmentation tasks."

**Research Impact**: SAM's zero-shot capabilities have been validated across:
- **Medical imaging** (MedSAM, SAM-Med2D) - 2,759 citations for MedSAM alone
- **Remote sensing** (satellite imagery, agriculture) - 351 citations
- **Autonomous driving** (lane detection, pedestrian segmentation)
- **Content creation** (background removal, rotoscoping)

---

## 2. Core Contributions: Task, Model, Dataset

From [SAM_STUDY_GENERAL.md](../../../PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md) lines 69-74, the Segment Anything project introduced three fundamental components:

### 2.1 Task: Promptable Segmentation

**Definition**: Return a valid segmentation mask for any prompt in any image, where prompts can be:
- **Foreground/background points** (click to segment)
- **Bounding boxes** (box → mask)
- **Rough masks** (coarse mask → refined mask)
- **Text descriptions** (via CLIP integration)

**Why Promptable?**
- Enables **interactive segmentation** (human-in-the-loop)
- Supports **zero-shot transfer** (no task-specific training)
- Allows **ambiguity handling** (multiple valid masks per prompt)

From [arXiv:2304.02643](https://arxiv.org/abs/2304.02643) Abstract:
> "The model is designed and trained to be promptable, so it can transfer zero-shot to new image distributions and tasks."

### 2.2 Model: SAM Architecture

**Three-Component Design**:

```
Input Image (1024×1024)
    ↓
[1] ViT-H Image Encoder (MAE pre-trained)
    → 256×64×64 image embeddings
    ↓
[2] Prompt Encoder (sparse + dense)
    → Sparse: points/boxes as positional encodings
    → Dense: masks as convolutional embeddings
    ↓
[3] Mask Decoder (lightweight transformer)
    → Predicts 3 mask candidates + IoU scores
    ↓
Output Masks (multiple candidates ranked by IoU)
```

**Key Architecture Features**:
- **Heavy image encoder** (636M params) - Run once per image
- **Lightweight decoder** (<4M params) - Run per prompt (real-time)
- **Multi-mask output** - 3 candidates for ambiguous prompts
- **IoU prediction** - Self-assessment of mask quality

### 2.3 Dataset: SA-1B (1.1 Billion Masks)

**Scale**:
- **11 million images** (licensed, privacy-respecting)
- **1.1 billion masks** (100 masks/image average)
- **Class-agnostic** (no semantic labels)

**Data Collection Strategy**:
- **Three-stage data engine** (Assisted → Semi-Auto → Fully Auto)
- **Model-in-the-loop** annotation
- **Human verification** for quality control

From [SAM_STUDY_GENERAL.md](../../../PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md) lines 94-116:
> "Stage 1: Assisted-Manual (120k images) - Expert annotators, SAM assists annotation, 6× faster than manual.
> Stage 2: Semi-Automatic (180k images) - SAM generates masks, humans refine, confidence-based selection.
> Stage 3: Fully Automatic (11M images) - SAM generates all masks, human quality verification, 100 masks/image average."

**Dataset Format** (from [GitHub](https://github.com/facebookresearch/segment-anything)):

```python
{
    "image": {
        "image_id": int,
        "width": int,
        "height": int,
        "file_name": str
    },
    "annotations": [{
        "id": int,
        "segmentation": dict,  # COCO RLE format
        "bbox": [x, y, w, h],
        "area": int,
        "predicted_iou": float,
        "stability_score": float,
        "crop_box": [x, y, w, h],
        "point_coords": [[x, y]]
    }]
}
```

---

## 3. Promptable Interface: The Heart of SAM

### 3.1 Four Prompt Types

From [SAM_STUDY_GENERAL.md](../../../PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md) lines 77-82:

**1. Point Prompts** (Most Common):
```python
# Foreground point (1) or background point (0)
input_point = np.array([[500, 375]])  # (x, y)
input_label = np.array([1])  # 1=foreground
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True  # Get 3 candidates
)
```

**2. Box Prompts** (Bounding Box → Mask):
```python
input_box = np.array([100, 100, 500, 500])  # [x0, y0, x1, y1]
masks, scores, logits = predictor.predict(
    box=input_box,
    multimask_output=False  # Box is unambiguous
)
```

**3. Mask Prompts** (Coarse → Refined):
```python
# Use previous mask as input for refinement
low_res_mask = logits[np.argmax(scores), :, :]
masks, scores, logits = predictor.predict(
    mask_input=low_res_mask[None, :, :],
    multimask_output=False
)
```

**4. Text Prompts** (via CLIP Integration):
- Not native to SAM, but achievable through:
  - **Grounding DINO** (text → box) + SAM (box → mask)
  - **CLIP** (text → image regions) + SAM (regions → masks)

### 3.2 Interactive Segmentation Workflow

**Typical User Flow**:

```
1. User clicks point on object
   ↓
2. SAM predicts 3 mask candidates
   ↓
3. User selects best mask OR adds refinement points
   ↓
4. SAM updates mask based on new points
   ↓
5. Repeat until satisfied (typically 1-3 iterations)
```

**Real-World Example** (from medical imaging applications):
- **Initial click** on tumor → SAM generates 3 masks (small, medium, large)
- **Refinement click** on tumor boundary → SAM refines to precise boundary
- **Background click** outside tumor → SAM excludes background region
- **Result**: High-quality tumor segmentation in 3-5 seconds

From [Research on Medical Image Segmentation Based on SAM](https://www.mdpi.com/2306-5354/12/6/608) (Fan et al., 2025):
> "Upon its release, SAM demonstrated exceptional zero-shot generalization across 23 diverse natural image datasets, surpassing both interactive and task-specific segmentation methods."

### 3.3 Multi-Mask Output & Ambiguity Handling

**Why 3 Masks?**
- Handles **ambiguous prompts** (e.g., point on shirt → whole person? just shirt? shirt+pants?)
- Provides **hierarchical options** (part → object → object+context)
- User selects **best mask** based on intent

**IoU Prediction Head**:
- SAM predicts **mask quality** (IoU with ground truth)
- Ranks candidates by predicted IoU
- Enables **confidence-based selection**

From [arXiv:2304.02643](https://arxiv.org/abs/2304.02643):
> "The model predicts multiple masks for ambiguous prompts along with an IoU score rating each mask's quality, enabling robust handling of inherent ambiguity in segmentation."

---

## 4. Zero-Shot Generalization Capabilities

### 4.1 Domain Transfer Without Retraining

From [SAM_STUDY_GENERAL.md](../../../PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md) lines 84-87:

**Key Capability**: SAM generalizes to new domains with **zero task-specific training**.

**Domains Tested** (23 datasets in original paper):
- **Natural images** (COCO, LVIS)
- **Medical imaging** (X-rays, CT scans, MRI)
- **Satellite imagery** (buildings, roads, agriculture)
- **Underwater scenes** (marine life, coral reefs)
- **Microscopy** (cell segmentation)
- **Autonomous driving** (lanes, pedestrians, vehicles)

**Performance Highlights** (from original paper):
- **COCO**: Competitive with fully-supervised methods
- **Medical imaging**: 90%+ Dice score on several datasets
- **Remote sensing**: Outperforms domain-specific models on building detection

From [TV-SAM: Increasing Zero-Shot Segmentation Performance](https://www.sciopen.com/article/10.26599/BDMA.2024.9020058) (Jiang et al., 2024):
> "SAM exhibits powerful zero-shot segmentation capabilities, with performance often competitive with or even superior to prior fully supervised results across diverse domains."

### 4.2 No Task-Specific Training Required

**Traditional Approach**:
```
Medical Dataset (10k images) → Train UNet → Medical Segmentation Model
Satellite Dataset (5k images) → Train FCN → Satellite Segmentation Model
Driving Dataset (20k images) → Train DeepLab → Driving Segmentation Model
```

**SAM Approach**:
```
SA-1B Dataset (11M images, 1.1B masks) → Train SAM Once
    ↓
SAM → Medical Segmentation (zero-shot)
SAM → Satellite Segmentation (zero-shot)
SAM → Driving Segmentation (zero-shot)
```

**Cost Savings**:
- **No annotation** for new domains (just prompts)
- **No retraining** (same weights for all tasks)
- **No domain expertise** (promptable interface)

### 4.3 Robustness Across Image Distributions

**Challenge**: Natural images (SA-1B) → Medical images (different distribution)

**SAM's Solution**:
- **MAE pre-training** on ImageNet (robust vision encoder)
- **Class-agnostic** training (no semantic assumptions)
- **Diverse SA-1B data** (11M images, varied sources)

From [Zero-Shot Performance of SAM in 2D Medical Imaging](https://www.researchgate.net/publication/378325886) (2023):
> "SAM's zero-shot performance in medical imaging is remarkable, achieving 90%+ Dice scores on several datasets despite being trained exclusively on natural images."

---

## 5. Automatic Mask Generation

### 5.1 Generate ALL Masks in Image

From [SAM_STUDY_GENERAL.md](../../../PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md) lines 89-92:

**Use Case**: Generate every object mask in an image without any prompts.

**Algorithm**:
```python
from segment_anything import SamAutomaticMaskGenerator

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,  # Grid density
    pred_iou_thresh=0.88,  # Quality threshold
    stability_score_thresh=0.95,  # Stability threshold
    crop_n_layers=1,  # Multi-scale cropping
    crop_overlap_ratio=0.341,  # Crop overlap
    min_mask_region_area=100  # Filter small masks
)

masks = mask_generator.generate(image)
# Returns list of masks sorted by area (largest first)
```

**Output Format**:
```python
masks = [
    {
        'segmentation': binary_mask,  # (H, W) boolean array
        'area': 54032,  # Pixel count
        'bbox': [10, 20, 100, 150],  # [x, y, w, h]
        'predicted_iou': 0.92,  # SAM's confidence
        'stability_score': 0.96,  # Mask stability
        'crop_box': [0, 0, 512, 512],  # Crop used
        'point_coords': [[64, 128]]  # Grid point
    },
    # ... more masks
]
```

### 5.2 Hierarchical Segmentation

**Multi-Scale Approach**:
- **32×32 grid** of points on full image (1024 candidates)
- **Multiple crops** at different scales (multi-scale objects)
- **NMS filtering** to remove duplicates
- **Quality ranking** by predicted IoU

**Result**: 50-200 masks per image covering all objects at multiple hierarchies.

**Applications**:
- **Data annotation** (bootstrap datasets)
- **Object discovery** (find all objects)
- **Scene understanding** (hierarchical object relationships)

From [GitHub README](https://github.com/facebookresearch/segment-anything):
> "Generate masks for an entire image: The automatic mask generator can be used to produce masks for every object in an image, providing a complete hierarchical segmentation."

### 5.3 Command-Line Interface

**Quick Usage**:
```bash
python scripts/amg.py \
    --checkpoint sam_vit_h_4b8939.pth \
    --model-type vit_h \
    --input images/dog.jpg \
    --output masks/
```

**Output**:
- `masks/dog_masks.json` (all masks in COCO format)
- `masks/dog_visualization.png` (visualized masks)

---

## 6. Paper & Resources

### 6.1 Research Paper

From [SAM_STUDY_GENERAL.md](../../../PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md) lines 59-68:

**Title**: Segment Anything
**arXiv**: https://arxiv.org/abs/2304.02643
**Published**: April 5, 2023
**Conference**: ICCV 2023

**Authors**: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick (Meta AI Research, FAIR)

**Citation** (BibTeX):
```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

**Impact Metrics** (as of 2025-11-20):
- **15,632+ citations** on Google Scholar
- **700,000+ downloads** (model checkpoints)
- **52.6k GitHub stars**
- **6.2k GitHub forks**

### 6.2 Official Resources

**GitHub Repository**:
- URL: https://github.com/facebookresearch/segment-anything
- Code: Inference, training, ONNX export
- Notebooks: Interactive examples, automatic mask generation
- License: Apache 2.0

**Project Website**:
- URL: https://segment-anything.com
- Interactive demo: https://segment-anything.com/demo
- Dataset: SA-1B download (11M images, 1.1B masks)

**Model Checkpoints** (three sizes):

| Model | Params | Download | Size | Speed |
|-------|--------|----------|------|-------|
| **ViT-H** | 636M | [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) | 2.4 GB | Slow, best quality |
| **ViT-L** | 308M | [sam_vit_l_0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) | 1.2 GB | Medium |
| **ViT-B** | 91M | [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) | 375 MB | Fast, good quality |

**Installation**:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git

# Optional dependencies
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

### 6.3 Community Tools & Extensions

**Grounding DINO + SAM**:
- Text-prompted segmentation
- "segment the dog" → automatic segmentation
- URL: https://github.com/IDEA-Research/Grounded-Segment-Anything

**Label Studio + SAM**:
- Interactive annotation tool
- SAM-assisted labeling
- URL: https://labelstud.io

**SAM-GEO**:
- Geospatial segmentation
- Satellite imagery applications
- URL: https://github.com/opengeos/segment-geospatial

---

## 7. Model Impact & Community Adoption

### 7.1 Downstream Applications

**Medical Imaging** (15+ specialized models):
- **MedSAM** (Nature Communications, 2024) - 2,759 citations
- **SAM-Med2D** - Medical foundation model
- **SAM-Med3D** - 3D medical imaging

**Remote Sensing** (10+ applications):
From [The Segment Anything Model (SAM) for Remote Sensing Applications](https://www.sciencedirect.com/science/article/pii/S1569843223003643) (Osco et al., 2023, 351 citations):
> "SAM advances the application of foundation models in remote sensing, enabling building detection, road extraction, and agricultural monitoring with minimal annotation."

**Autonomous Driving**:
- Lane detection
- Pedestrian segmentation
- Vehicle segmentation

**Content Creation**:
- Background removal (Photoshop, Canva)
- Rotoscoping (video editing)
- Object isolation

**Agriculture**:
- Crop health monitoring
- Plant counting
- Weed detection

**Robotics**:
- Grasping point detection
- Scene understanding
- Object manipulation

### 7.2 Research Influence

**Foundation Model Paradigm**:
- Inspired **SAM 2** (video segmentation, 2024)
- Inspired **SAM 3** (text-prompted segmentation, November 2025)
- Influenced **Grounding DINO**, **SegGPT**, **SEEM**

**Academic Impact**:
- **15,632+ citations** (unprecedented for CV paper in 2 years)
- **23 datasets** evaluated in original paper
- **100+ follow-up papers** in medical imaging alone

From [A systematic survey and meta-analysis of SAM](https://www.sciencedirect.com/science/article/abs/pii/S092427162500334X) (Wan et al., 2025):
> "Compared with traditional segmentation methods, SAM exhibits powerful zero-shot segmentation capabilities, fundamentally changing how researchers approach segmentation tasks across domains."

### 7.3 Industry Adoption

**Meta Products**:
- Facebook content moderation
- Instagram background removal
- Reality Labs AR/VR applications

**Third-Party Integration**:
- **Adobe Photoshop** - Background removal
- **Canva** - Design tools
- **Label Studio** - Annotation platform
- **Roboflow** - Computer vision platform

**Open-Source Ecosystem**:
- **52.6k GitHub stars** (top 100 ML repos)
- **6.2k forks** (active development)
- **17 contributors** to core repo
- **100+ community extensions**

---

## 8. ARR-COC Integration: Relevance-Driven Segmentation

### 8.1 Promptable Relevance Realization

**Connection to ARR-COC-0-1**: SAM's promptable interface directly embodies relevance realization—users allocate attention through prompts (points, boxes, text) to realize which image regions are relevant to their task.

**Propositional Knowing** (What):
- **Semantic knowledge**: "Segment this tumor" (medical imaging)
- **Object categories**: "This is a dog, segment it" (object detection)
- **Spatial facts**: "Bounding box [100, 100, 500, 500] contains object"

**Example**:
```python
# Propositional: "The tumor is located at (x=250, y=180)"
tumor_point = np.array([[250, 180]])
tumor_label = np.array([1])  # Foreground

masks, scores, logits = predictor.predict(
    point_coords=tumor_point,
    point_labels=tumor_label
)
# SAM uses propositional knowledge to segment tumor region
```

### 8.2 Perspectival Knowing: Spatial Relationships

**Perspectival Knowing** (How):
- **Spatial perspective**: Where objects are relative to each other
- **Boundary perception**: How edges define object boundaries
- **Hierarchical understanding**: Part-whole relationships (shirt → person → scene)

**Example**:
```python
# Perspectival: Understanding shirt is PART OF person
# Initial point on shirt
shirt_point = np.array([[300, 200]])
masks, _, _ = predictor.predict(point_coords=shirt_point, ...)

# Multi-mask output provides hierarchical options:
# Mask 1: Just the shirt (small)
# Mask 2: Shirt + pants (medium)
# Mask 3: Whole person (large)
```

**SAM's Multi-Mask Output** = Different perspectival framings of the same click!

### 8.3 Participatory Knowing: Interactive Refinement Loop

**Participatory Knowing** (Embodied):
- **Interactive refinement**: User clicks → SAM responds → user refines → SAM updates
- **Embodied cognition**: Physical clicking embodies relevance allocation
- **Tool-mediated knowing**: SAM becomes extension of user's segmentation ability

**Interactive Loop**:
```python
# Step 1: Initial click (user participates)
initial_point = np.array([[150, 150]])
masks1, scores1, logits1 = predictor.predict(point_coords=initial_point, ...)

# Step 2: User sees result, adds refinement click (participatory feedback)
refinement_points = np.array([[150, 150], [200, 180]])  # Add boundary point
refinement_labels = np.array([1, 1])  # Both foreground

# Step 3: SAM updates based on participation
masks2, scores2, logits2 = predictor.predict(
    point_coords=refinement_points,
    point_labels=refinement_labels,
    mask_input=logits1[np.argmax(scores1), :, :],  # Use previous mask
)

# Cycle continues until user's relevance realization is satisfied
```

**Key Insight**: The interactive loop IS participatory knowing—user and SAM co-create the segmentation through embodied interaction.

### 8.4 ARR-COC-0-1 Specific Application

**Relevance-Guided Segmentation Pipeline**:

```python
# arr-coc-0-1 integration pseudocode
class RelevanceGuidedSAM:
    def __init__(self, sam_model):
        self.sam = sam_model
        self.relevance_history = []  # Track user's relevance allocation

    def segment_with_relevance(self, image, relevance_prompts):
        """
        relevance_prompts = {
            'propositional': ["tumor", "brain region"],
            'perspectival': [(x1, y1), (x2, y2)],  # Spatial clicks
            'participatory': refinement_history
        }
        """
        # Propositional → Text prompts (via Grounding DINO)
        boxes = grounding_dino.detect(image, relevance_prompts['propositional'])

        # Perspectival → Point prompts
        points = np.array(relevance_prompts['perspectival'])

        # Combine prompts
        masks, scores, logits = self.sam.predict(
            box=boxes[0],
            point_coords=points,
            multimask_output=True  # Get perspectival options
        )

        # Participatory → User selects best mask
        user_selection = self.interactive_refinement(masks, scores)

        # Record relevance realization
        self.relevance_history.append({
            'prompts': relevance_prompts,
            'selected_mask': user_selection,
            'confidence': scores[user_selection]
        })

        return masks[user_selection]
```

**ARR-COC Relevance Allocation** (10% of SAM's capability):
- **Attention allocation**: Points/boxes direct SAM's "attention" to relevant regions
- **Salience detection**: Multi-mask output reveals different salience framings
- **Relevance feedback loop**: Interactive refinement = participatory relevance realization
- **Meaning-making**: User's prompts MAKE regions meaningful (tumor, person, object)

**Research Question for ARR-COC-0-1**:
> "Can we train ARR-COC to allocate attention (prompts) to SAM in a way that mirrors human relevance realization patterns during segmentation tasks?"

**Potential Experiment**:
1. Record human segmentation sessions (prompts + refinements)
2. Train ARR-COC to predict next prompt given current mask
3. Evaluate: Does ARR-COC's prompt allocation match human relevance patterns?

---

## Sources

**Source Documents**:
- [SAM_STUDY_GENERAL.md](../../../PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md) - Lines 1-200 (comprehensive SAM overview)

**Web Research** (accessed 2025-11-20):
- [Segment Anything (arXiv:2304.02643)](https://arxiv.org/abs/2304.02643) - Kirillov et al., Meta AI, April 2023
- [GitHub: facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) - Official repository, 52.6k stars
- [Research on Medical Image Segmentation Based on SAM](https://www.mdpi.com/2306-5354/12/6/608) - Fan et al., 2025
- [TV-SAM: Increasing Zero-Shot Segmentation Performance](https://www.sciopen.com/article/10.26599/BDMA.2024.9020058) - Jiang et al., 2024
- [SAM for Remote Sensing Applications](https://www.sciencedirect.com/science/article/pii/S1569843223003643) - Osco et al., 2023 (351 citations)
- [MedSAM: Segment Anything in Medical Images](https://www.nature.com/articles/s41467-024-44824-z) - Ma et al., Nature Communications, 2024 (2,759 citations)
- [A systematic survey and meta-analysis of SAM](https://www.sciencedirect.com/science/article/abs/pii/S092427162500334X) - Wan et al., 2025

**Additional References**:
- Meta AI Project Page: https://segment-anything.com
- Interactive Demo: https://segment-anything.com/demo
- SA-1B Dataset: https://ai.facebook.com/datasets/segment-anything
