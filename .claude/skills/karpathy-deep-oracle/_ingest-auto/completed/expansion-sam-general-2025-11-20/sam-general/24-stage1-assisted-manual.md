# SAM Data Engine Stage 1: Assisted-Manual Annotation

## Overview

Stage 1 of SAM's data engine represents the foundation of the entire SA-1B dataset creation process. This "assisted-manual" annotation phase established the initial high-quality training data that would bootstrap all subsequent annotation stages. By combining human expertise with model assistance, Meta achieved a 6x speedup over traditional manual annotation while maintaining exceptional mask quality.

### Key Statistics

| Metric | Value |
|--------|-------|
| **Images Annotated** | 120,000 |
| **Masks Generated** | 4.3 million |
| **Average Masks per Image** | ~36 |
| **Annotation Time Threshold** | 30 seconds per mask |
| **Speed Improvement** | 6x faster than pure manual |

### Stage 1 Goals

**Primary Objectives:**
- Create initial high-quality training data for SAM
- Establish annotation workflows and quality standards
- Train annotators on the interactive segmentation interface
- Bootstrap model improvement for subsequent stages

**Why Start with Assisted-Manual:**
- Need human judgment for ambiguous cases
- Establishes ground truth quality baseline
- Allows iterative refinement of annotation tools
- Creates diverse, high-quality seed dataset

### The Human-AI Loop

```
Initial SAM (pre-trained on public data)
    |
    v
Annotator clicks foreground/background points
    |
    v
SAM generates mask prediction
    |
    v
Annotator refines with brush/eraser
    |
    v
Final mask saved
    |
    v
SAM retrained on accumulated masks
    |
    v
Improved SAM used for next batch
```

**Sources:**
- [From SAM to SAM 2: Exploring Improvements](https://arxiv.org/html/2408.06305v1) - Section 2.2.1
- [Segment Anything Paper](https://arxiv.org/abs/2304.02643) - Kirillov et al., 2023

---

## Annotation Tools

### Interactive Segmentation Interface

Meta developed a custom browser-based annotation tool that enabled efficient model-assisted annotation. The interface was designed to minimize annotator effort while maximizing mask quality.

**Core Tool Components:**

1. **Point Prompts**
   - Click anywhere to indicate foreground (object to segment)
   - Click to indicate background (areas to exclude)
   - Multiple points can refine ambiguous predictions

2. **Bounding Box Tool**
   - Draw rectangle around object
   - SAM generates mask within box bounds
   - Useful for well-defined objects

3. **Brush Tool**
   - Add pixels to mask
   - Pixel-precise control for edges
   - Variable brush size

4. **Eraser Tool**
   - Remove pixels from mask
   - Correct over-segmentation
   - Refine boundaries

### Tool Workflow Integration

```
1. Annotator views image
   |
2. Clicks point on object (foreground)
   |
3. SAM generates initial mask (< 1 second)
   |
4. If satisfactory -> Accept
   |
   If needs refinement:
   |-- Add more points (foreground/background)
   |-- Use brush to extend mask
   |-- Use eraser to trim mask
   |
5. Final refinement with pixel tools
   |
6. Accept and move to next object
```

### Design Principles

**Efficiency First:**
- Single-click segmentation when possible
- Real-time mask preview (< 100ms response)
- Keyboard shortcuts for common actions
- Undo/redo for quick corrections

**Quality Assurance:**
- Visual overlay showing mask boundaries
- Zoom for precise edge work
- Alpha blending for transparency control
- Side-by-side comparison views

**Ergonomic Considerations:**
- Minimized mouse movement
- Consistent tool locations
- Progressive disclosure of advanced tools
- Fatigue-reducing interface design

**Sources:**
- [Kili Technology: Deep-diving into SAM 2](https://kili-technology.com/blog/deep-diving-into-sam2-how-quality-data-propelled-meta-s-visual-segmentation-model)
- [From SAM to SAM 2](https://arxiv.org/html/2408.06305v1) - Section 2.2.1

---

## Annotator Workflow

### Standard Annotation Process

The Stage 1 workflow followed a systematic approach to ensure consistent, high-quality annotations across the 120,000 images.

**Step-by-Step Workflow:**

1. **Image Loading**
   - High-resolution image displayed
   - Images downsampled to 1,500 pixels (from avg 3,300x4,950)
   - Faces and license plates pre-blurred for privacy

2. **Object Identification**
   - Annotator scans image for segmentable objects
   - Focus on "things" (discrete objects) over "stuff" (amorphous regions)
   - Prioritize clearly visible, unambiguous objects

3. **Initial Segmentation**
   - Click foreground point on object center
   - SAM generates mask prediction
   - Evaluate mask quality visually

4. **Refinement (if needed)**
   - Add background points to exclude incorrect regions
   - Add foreground points to include missed regions
   - Use brush/eraser for fine boundaries

5. **Acceptance Criteria**
   - Mask accurately covers object boundary
   - No significant under/over-segmentation
   - Consistent with neighboring masks

6. **Proceed or Skip**
   - If mask takes > 30 seconds, skip to next object
   - Move to next image when all identifiable objects annotated

### Time Management

**30-Second Rule:**
- Critical for maintaining annotation velocity
- Prevents annotators from getting stuck on difficult cases
- Ensures diverse coverage across images
- Difficult objects captured in later semi-automatic stage

**Annotation Time Distribution:**
- Easy objects: 5-10 seconds (point + accept)
- Medium objects: 15-25 seconds (few refinements)
- Complex objects: > 30 seconds (skip)

### Quality Over Speed Balance

The workflow balanced annotation speed with quality:

- **Target:** Maximum masks while maintaining >90% IoU quality
- **Outcome:** 4.3M masks from 120K images (36 masks/image average)
- **Quality:** 94% of final SA-1B masks have IoU > 90%

**Sources:**
- [From SAM to SAM 2](https://arxiv.org/html/2408.06305v1) - Section 2.2.1
- [Segment Anything Paper](https://arxiv.org/abs/2304.02643)

---

## Model Assistance

### SAM's Role in Stage 1

The pre-trained SAM model provided real-time assistance during annotation, transforming sparse user inputs into complete segmentation masks.

**Initial Model Training:**
- Pre-trained on publicly available segmentation datasets
- General object detection capability
- Not yet optimized for interactive annotation

**Assistance Mechanisms:**

1. **Point-to-Mask Generation**
   - Single click produces complete object mask
   - Handles ambiguous prompts with multiple mask candidates
   - Predicts IoU confidence for each mask

2. **Multi-Mask Output**
   - Three mask candidates per prompt
   - Hierarchical: whole object, part, subpart
   - Annotator selects best match or refines

3. **Iterative Refinement**
   - Additional points refine initial prediction
   - Background points teach exclusion boundaries
   - Each click improves mask accuracy

### Progressive Model Improvement

**Iterative Training Loop:**

```
Week 1: Initial SAM on public data
    -> Annotations collected
    -> Retrain SAM

Week 2: Improved SAM v2
    -> Better masks, faster annotation
    -> More annotations collected
    -> Retrain SAM

Week N: SAM vN
    -> Near-human quality predictions
    -> Minimal refinement needed
```

**Improvement Metrics:**
- Annotation time decreased as model improved
- Fewer refinement clicks needed per mask
- Higher first-attempt acceptance rate

### Model Architecture for Real-Time Assistance

**Performance Requirements:**
- Image embedding: ~100ms (computed once per image)
- Mask generation: ~50ms per prompt
- Total user wait time: < 150ms

**Architecture Enabling Speed:**
- Heavy image encoder (ViT-H) runs once
- Lightweight prompt encoder + mask decoder for interactions
- Amortized computation across multiple objects per image

**Sources:**
- [Segment Anything Paper](https://arxiv.org/abs/2304.02643) - Section 4
- [From SAM to SAM 2](https://arxiv.org/html/2408.06305v1) - Section 2.2

---

## Quality Control

### Quality Assurance Process

Meta implemented multiple quality control mechanisms to ensure Stage 1 annotations met high standards.

**Automated Quality Checks:**

1. **Mask Validity**
   - Minimum mask size threshold
   - Connected component analysis
   - Boundary smoothness checks

2. **Overlap Detection**
   - Identify overlapping masks
   - Flag potential duplicate annotations
   - Ensure spatial consistency

3. **Coverage Analysis**
   - Track objects per image distribution
   - Identify under-annotated images
   - Balance dataset diversity

**Human Quality Review:**

1. **Random Sampling**
   - Regular quality audits on random subset
   - Expert review of annotation accuracy
   - Feedback loop to annotators

2. **Edge Case Review**
   - Manual inspection of flagged annotations
   - Review of skipped difficult objects
   - Quality comparison across annotators

3. **Inter-Annotator Agreement**
   - Multiple annotators on same images
   - IoU comparison between annotations
   - Calibration sessions for consistency

### Quality Metrics

**Primary Metrics:**
- **IoU (Intersection over Union):** Target > 90%
- **Boundary Accuracy:** Pixel-level precision on edges
- **Completeness:** All visible objects captured

**Achieved Quality:**
- 94% of masks have IoU > 90%
- 99.1% of final SA-1B masks pass quality threshold
- Consistent quality across image types

**Sources:**
- [Segment Anything Paper](https://arxiv.org/abs/2304.02643) - Section 5
- [From SAM to SAM 2](https://arxiv.org/html/2408.06305v1)

---

## Data Statistics

### Stage 1 Output Summary

| Category | Value |
|----------|-------|
| Total Images | 120,000 |
| Total Masks | 4.3 million |
| Average Masks/Image | 35.8 |
| Median Masks/Image | ~30 |
| Max Masks/Image | 100+ |

### Annotation Efficiency

**Time Metrics:**
- Average time per mask: < 15 seconds
- Comparison to manual annotation: 6x speedup
- Target throughput achieved: ~36 masks/image

**6x Speedup Breakdown:**
- Pure manual polygon annotation: ~90 seconds/mask
- SAM-assisted annotation: ~14 seconds/mask
- Speedup factor: 90/14 ≈ 6.4x

### Mask Properties

**Size Distribution:**
- Small objects (< 1% image area): 30%
- Medium objects (1-10%): 45%
- Large objects (> 10%): 25%

**Object Categories:**
- Everyday objects (furniture, vehicles)
- Natural objects (plants, animals)
- Man-made structures
- Body parts and accessories
- Food items

### Image Properties

**Original Resolution:**
- Average: 3,300 x 4,950 pixels
- High quality for detailed annotation

**Working Resolution:**
- Downsampled to 1,500 pixels on longest side
- Maintains sufficient detail for accurate annotation
- Reduces bandwidth and processing requirements

### Quality Distribution

| Quality Level | Percentage |
|---------------|------------|
| IoU > 95% | ~70% |
| IoU 90-95% | ~24% |
| IoU < 90% | ~6% |

### Comparison to Later Stages

| Stage | Images | Masks | Masks/Image |
|-------|--------|-------|-------------|
| Stage 1 (Assisted-Manual) | 120K | 4.3M | 36 |
| Stage 2 (Semi-Automatic) | 180K | 5.9M | 33 |
| Stage 3 (Fully Automatic) | 11M | 1.1B | 100 |

**Sources:**
- [Segment Anything Paper](https://arxiv.org/abs/2304.02643) - Section 5
- [From SAM to SAM 2](https://arxiv.org/html/2408.06305v1) - Section 2.2

---

## ARR-COC Integration

### Relevance to ARR-COC Training

Stage 1's assisted-manual approach offers valuable lessons for custom model training in ARR-COC:

**Key Takeaways:**

1. **Human-in-the-Loop Efficiency**
   - Model assistance dramatically speeds annotation
   - Quality maintained through human oversight
   - Iterative improvement creates virtuous cycle

2. **Quality-First Foundation**
   - Initial high-quality data bootstraps entire pipeline
   - 6x speedup achieved without sacrificing accuracy
   - Foundation for scaling to billions of masks

3. **Progressive Model Improvement**
   - Retrain on collected data periodically
   - Each iteration improves assistance quality
   - Feedback loop accelerates data collection

### Implementation Patterns

**For Custom Dataset Creation:**

```python
# Assisted annotation workflow
class AssistedAnnotator:
    def __init__(self, base_model):
        self.model = base_model
        self.collected_masks = []

    def annotate_image(self, image):
        # Model provides initial predictions
        predictions = self.model.predict(image)

        # Human refines predictions
        refined_masks = human_refinement(predictions)

        # Collect for retraining
        self.collected_masks.extend(refined_masks)

    def retrain_model(self):
        # Periodic retraining on collected data
        self.model.finetune(self.collected_masks)
        self.collected_masks = []  # Reset for next cycle
```

**Quality Control Integration:**

```python
# Quality validation pipeline
def validate_mask(mask, image):
    checks = [
        check_minimum_size(mask),
        check_boundary_smoothness(mask),
        check_overlap_with_existing(mask),
        check_iou_threshold(mask, threshold=0.9)
    ]
    return all(checks)
```

### Training Recommendations

**For ARR-COC Projects:**

1. **Start with Assisted Annotation**
   - Use pre-trained model (SAM, SAM 2) for assistance
   - Collect domain-specific refinements
   - Build custom training set iteratively

2. **Implement 30-Second Rule**
   - Skip difficult cases for efficiency
   - Capture in later automated stages
   - Maintain annotation velocity

3. **Quality Checkpoints**
   - Regular IoU validation
   - Random human review samples
   - Inter-annotator calibration

4. **Progressive Scaling**
   - Stage 1: Assisted-manual (highest quality)
   - Stage 2: Semi-automatic (balance)
   - Stage 3: Fully automatic (maximum scale)

**Sources:**
- [Segment Anything Paper](https://arxiv.org/abs/2304.02643)
- [SAM GitHub Repository](https://github.com/facebookresearch/segment-anything)

---

## Summary

Stage 1's assisted-manual annotation phase demonstrated that combining human expertise with model assistance can achieve dramatic efficiency gains (6x speedup) while maintaining exceptional quality (94% IoU > 90%). The 4.3 million masks generated from 120,000 images provided the foundation for SAM's entire training pipeline, enabling the subsequent semi-automatic and fully automatic stages that would scale to 1.1 billion masks.

**Key Innovations:**
- Interactive annotation interface with real-time model feedback
- 30-second skip rule for maintaining velocity
- Iterative model retraining during annotation
- Quality-first approach with systematic validation

**Impact:**
- Established annotation workflows for billion-scale datasets
- Proved human-AI collaboration in data annotation
- Created foundation for promptable segmentation models
- Influenced subsequent data engine designs (SAM 2, SAM 3)

---

**Document Information:**
- **Created:** 2025-11-20
- **Part of:** SAM General Knowledge Expansion
- **Related:** 25-stage2-semi-automatic.md, 26-stage3-fully-automatic.md
