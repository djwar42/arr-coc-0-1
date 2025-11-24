# SAM Three-Stage Data Engine

## Section 1: Data Engine Overview

### The Core Problem: Data Scarcity at Scale

The Segment Anything Model faced a fundamental challenge that no prior segmentation model had encountered: the internet lacks web-scale segmentation data. Unlike image classification (ImageNet) or object detection (COCO), there were no billion-scale mask datasets available. This necessitated a completely novel approach to data collection.

From [SAM Study General](../source-documents/SAM_STUDY_GENERAL.md):
- Goal: Create 1.1 billion masks from 11 million images
- Challenge: No existing web-scale segmentation datasets
- Solution: Model-in-the-loop data engine

### Data Engine Philosophy

The data engine represents a paradigm shift in dataset creation - using the model being trained to assist in its own training data creation. This creates a positive feedback loop:

```
Better Model → Better Annotations → Better Data → Better Model
     ↑                                                  ↓
     └──────────────────────────────────────────────────┘
```

### Three-Stage Progressive Automation

The data engine implements progressive automation across three stages:

| Stage | Images | Automation Level | Human Role | Speed |
|-------|--------|------------------|------------|-------|
| Stage 1: Assisted-Manual | 120K | Low (model assists) | Primary annotator | ~38 sec/mask |
| Stage 2: Semi-Automatic | 180K | Medium (model proposes) | Refine & complete | Faster |
| Stage 3: Fully Automatic | 11M | High (model generates) | Quality verification | ~4.5 sec/mask |

### Key Innovation: Model-in-the-Loop

From [Encord SAM Explanation](https://encord.com/blog/segment-anything-model-explained/):

Traditional annotation pipelines:
- Humans create all annotations manually
- Model is trained on completed dataset
- No feedback loop during data creation

SAM's data engine:
- Model assists annotation from the start
- Model improves as data grows
- Re-trained model improves annotation quality
- Continuous iteration between model and data

### Scale Achievement

The three-stage approach achieved unprecedented scale:
- **Total masks**: 1.1 billion
- **Total images**: 11 million
- **Average masks/image**: ~100
- **Annotation speedup**: 8x faster (Stage 3 vs Stage 1)

From [Meta Universe of AI](https://medium.com/universe-of-ai/building-better-data-engines-metas-3-stage-data-engine-sam-d7714ab5e9c5):
> "Meta has almost >2.9 billion active users across its platforms. The world's population is around 8 billion. That is a massive scale. And with that scale comes different use cases."

This context explains why Meta needed such massive annotation capability - to train models that serve billions of users across diverse applications.

---

## Section 2: Stage 1 - Assisted-Manual (120K Images)

### Overview

Stage 1 establishes the foundational high-quality dataset through human expertise assisted by an early SAM model. The goal is accuracy over speed - "Let's get this right before we make this fast."

From [SAM Study General](../source-documents/SAM_STUDY_GENERAL.md):
```
Human Annotator
    ↓ Clicks points
SAM (pre-trained on public data)
    ↓ Generates mask
Human
    ↓ Refines if needed
Final Mask
```

### Annotation Process

**Initial Model Training:**
SAM is first pre-trained on publicly available segmentation datasets before Stage 1 begins. This gives annotators a useful tool from day one.

**Browser-Based Interactive Tool:**
- Professional annotators use custom browser-based interface
- Click foreground/background points on image
- SAM generates mask prediction in real-time
- Annotator refines mask edges if needed
- No pixel-by-pixel drawing required

**Point Prompting Workflow:**
```python
# Conceptual workflow
1. Annotator clicks point on object → [250, 300]
2. SAM generates initial mask → ~85% accurate
3. Annotator clicks refinement points → [280, 320], [220, 290]
4. SAM updates mask → ~98% accurate
5. Annotator approves or makes minor edits
6. Mask saved with metadata
```

### Speed and Efficiency

From [Medium Article](https://medium.com/universe-of-ai/building-better-data-engines-metas-3-stage-data-engine-sam-d7714ab5e9c5):
- Average annotation time: **~38 seconds per mask**
- This is time-consuming but critical for quality
- Establishes "gold standard" for subsequent stages

**Why So Slow?**
- Multiple annotators verify same data
- Disagreements are flagged for review
- Quality takes precedence over speed
- Creates high-quality training signal

### Annotator Infrastructure

**Multiple Annotators Per Image:**
- Same data annotated by multiple humans
- Disagreements flagged for resolution
- Consensus algorithms determine final labels

**Annotator Support Systems:**
From [Medium Article](https://medium.com/universe-of-ai/building-better-data-engines-metas-3-stage-data-engine-sam-d7714ab5e9c5):
- **Monitoring**: Logs for each datapoint, timestamps, performance metrics
- **Annotation Systems**: Visual tools for annotate, commit, revert, comment, review
- **Version Control**: Annotators can revert to older changes
- **Annotator Details**: Past performance, accuracy, expertise tracking
- **Task Management**: Dynamic assignment based on expertise and load balancing

### Model Improvement During Stage 1

As more Stage 1 data is collected:
- SAM is periodically re-trained on accumulated masks
- Model predictions become more accurate
- Annotators need fewer refinement clicks
- Efficiency improves over time

**Key Insight**: The model learns from high-quality human corrections, not just final masks. This is supervised learning with human feedback at its finest.

### Quality Metrics

**Accuracy Goals:**
- Pixel-level precision for mask boundaries
- Correct object identity (not merging objects)
- Complete coverage of object extent

**What Gets Annotated:**
- All visible objects in each image
- Hierarchical segmentation (parts within objects)
- Both prominent and fine-grained objects

### Stage 1 Output

- **120,000 images** fully annotated
- Multiple masks per image (all visible objects)
- High-quality "gold standard" dataset
- Foundation for training improved SAM for Stage 2

---

## Section 3: Stage 2 - Semi-Automatic (180K Images)

### Overview

Stage 2 shifts the human role from primary annotator to refiner and completer. SAM (re-trained on Stage 1 data) now proposes confident masks automatically, and humans focus on what the model missed.

From [SAM Study General](../source-documents/SAM_STUDY_GENERAL.md):
```
SAM (re-trained on Stage 1 data)
    ↓ Generates confident masks automatically
Human
    ↓ Annotates remaining objects
Final Masks (100+ per image)
```

### Confidence-Based Selection

**Automatic Mask Generation:**
- SAM generates masks for entire image
- Each mask has confidence score (predicted IoU)
- High-confidence masks are auto-accepted
- Low-confidence areas flagged for human review

**Confidence Threshold Strategy:**
```python
# Conceptual threshold logic
for mask in generated_masks:
    if mask.predicted_iou > HIGH_THRESHOLD:
        auto_accept(mask)
    elif mask.predicted_iou > MEDIUM_THRESHOLD:
        flag_for_review(mask)
    else:
        discard(mask)
```

### Human Focus Shift

**Before (Stage 1):** Humans annotate everything
**After (Stage 2):** Humans complete the gaps

**Annotator Tasks in Stage 2:**
1. Review auto-generated masks for quality
2. Annotate objects SAM missed
3. Split merged objects
4. Refine ambiguous boundaries
5. Add fine-grained masks (small objects)

### Increasing Mask Diversity

A key goal of Stage 2 is **diversity enhancement**. Stage 1 masks might be biased toward:
- Common objects (people, cars, animals)
- Large, prominent objects
- Simple shapes

Stage 2 humans specifically target:
- Rare objects the model missed
- Fine-grained parts (door handles, buttons)
- Unusual object configurations
- Domain-specific content

### Conflict Resolution

From [Medium Article](https://medium.com/universe-of-ai/building-better-data-engines-metas-3-stage-data-engine-sam-d7714ab5e9c5):

When multiple annotators disagree:

**Consensus Algorithms:**
- Majority voting for simple disagreements
- Weighted voting based on annotator accuracy history
- Complex cases escalated to manual review

**Gold Standard Comparison:**
> "This output is then passed for quality validation by comparing the annotation to a 'gold standard'. This gold standard is a small dataset of highly accurate annotations."

This reveals annotator performance issues and systematic errors.

### Semi-Automatic Results

**Image Coverage:**
- 180,000 images annotated
- Mix of automatic + manual annotations
- ~100 masks per image average

**Efficiency Gains:**
- Faster than Stage 1 (exact speed not published)
- Annotators leverage model capabilities
- Focus on hard cases vs routine annotation

**Quality Maintenance:**
- Human verification prevents error propagation
- Diversity injection overcomes model biases
- Gold standard comparisons catch systematic issues

### Model Re-Training for Stage 3

Stage 2 data is combined with Stage 1:
- Total: 300K images (120K + 180K)
- Millions of masks
- SAM re-trained for Stage 3 fully automatic mode

**Key Improvement:** The model now has:
- High-quality boundaries from Stage 1
- Diverse objects from Stage 2
- Confidence calibration from verification

---

## Section 4: Stage 3 - Fully Automatic (11M Images)

### Overview

Stage 3 is where the data engine achieves web scale. SAM generates all masks without human prompting, producing approximately 100 masks per image across 11 million images.

From [SAM Study General](../source-documents/SAM_STUDY_GENERAL.md):
```
SAM (re-trained on Stage 1+2)
    ↓ Generates all masks (no human in loop)
Human Quality Check
    ↓ Random sampling validation
Final Dataset: SA-1B
```

### Automatic Mask Generation Pipeline

**Grid-Based Prompting:**
SAM is prompted with a regular grid of foreground points across each image:
```python
# Conceptual automatic generation
def generate_all_masks(image):
    masks = []

    # Grid of prompt points
    for x in range(0, image.width, GRID_SPACING):
        for y in range(0, image.height, GRID_SPACING):
            point = (x, y)
            candidate_masks = sam.predict(point)
            masks.extend(candidate_masks)

    # Post-processing
    masks = apply_nms(masks)  # Non-maximal suppression
    masks = filter_by_quality(masks)

    return masks
```

### Ambiguity-Aware Predictions

For ambiguous prompts, SAM generates multiple valid masks:
- Whole object vs part
- Foreground vs background interpretation
- Overlapping objects

**Multi-Mask Output:**
Each point prompt produces 3 mask candidates:
- Subpart level
- Part level
- Whole object level

### Non-Maximal Suppression

To avoid duplicate masks:
```python
def apply_nms(masks, iou_threshold=0.7):
    # Sort by predicted IoU score
    masks = sorted(masks, key=lambda m: m.score, reverse=True)

    final_masks = []
    for mask in masks:
        # Check overlap with already accepted masks
        overlap = any(iou(mask, accepted) > iou_threshold
                      for accepted in final_masks)
        if not overlap:
            final_masks.append(mask)

    return final_masks
```

### Quality Filtering

**Stability Score:**
Each mask is evaluated for stability - how much the mask changes with small perturbations to the input prompt. Stable masks are more reliable.

**Predicted IoU:**
The model predicts its own mask quality. Low predicted IoU masks are filtered out.

**Minimum Area:**
Very small masks (likely noise) are removed.

### Scale Achievement

**Raw Numbers:**
- 11 million images
- ~100 masks per image (average)
- 1.1 billion total masks

**Speed:**
From [Medium Article](https://medium.com/universe-of-ai/building-better-data-engines-metas-3-stage-data-engine-sam-d7714ab5e9c5):
> "Deployed systems annotate with a speed of around **4.5 seconds per mask**, which is around **8x faster**" than Stage 1.

### Human Quality Verification

**Random Sampling:**
- Statistical sampling of generated masks
- Human reviewers validate quality
- Issues flagged for investigation

**What Humans Check:**
- Boundary precision
- Object identity correctness
- Coverage completeness
- Obvious errors

**Quality Results:**
From [Encord](https://encord.com/blog/segment-anything-model-explained/):
> "The quality of the segmentation masks is rigorously evaluated, with automatic masks deemed high quality and effective for training models, leading to the decision to include automatically generated masks exclusively in SA-1B."

### Dataset Characteristics

**Image Sources:**
- Licensed from data providers
- Privacy-preserving (faces and license plates blurred)
- Geographic diversity (all continents)
- Subject diversity (scenes, objects, places)

**Mask Properties:**
- Class-agnostic (no semantic labels)
- Hierarchical (parts within wholes)
- Dense coverage (most pixels covered)
- High boundary quality

---

## Section 5: Model-in-the-Loop Training

### The Iterative Improvement Cycle

The data engine's power comes from continuous iteration between model training and data collection:

```
Stage 1: Model v0 (public data) → Assists annotation
         ↓
         Humans create high-quality masks
         ↓
Stage 2: Model v1 (Stage 1 data) → Proposes masks
         ↓
         Humans complete and diversify
         ↓
Stage 3: Model v2 (Stage 1+2 data) → Generates all masks
         ↓
         1.1B masks dataset
         ↓
Final:   Model v3 (SA-1B) → Production SAM
```

### What the Model Learns

**Stage 1 → Stage 2:**
- Accurate boundary prediction
- Response to point prompts
- Multiple valid mask outputs
- Confidence estimation

**Stage 2 → Stage 3:**
- Diverse object types
- Fine-grained segmentation
- Rare object categories
- Complex scene understanding

**Final Training:**
- Web-scale generalization
- Zero-shot transfer ability
- Robustness across domains

### Training Details

From [SAM Study General](../source-documents/SAM_STUDY_GENERAL.md):
- **Pre-train** on synthetic data
- **Fine-tune** on SA-1B
- **Promptable training objective**

**Loss Functions:**
- Focal loss for mask prediction
- Dice loss for boundary quality
- IoU prediction loss for confidence calibration

### Why Model-in-the-Loop Works

**Traditional Approach Problems:**
- Annotators are bottleneck
- Model never sees annotation process
- No feedback during data creation
- Quality varies across annotators

**Model-in-the-Loop Benefits:**
- Model reduces annotator burden
- Model learns from corrections
- Annotation quality improves with model
- Scalable to billions of masks

### Avoiding Error Propagation

A key concern: model errors could propagate into training data, causing model to reinforce mistakes.

**Mitigation Strategies:**
1. Human verification at each stage
2. Confidence thresholds filter uncertain predictions
3. Multiple annotators catch individual errors
4. Gold standard comparisons detect systematic issues
5. Diversity injection overcomes model biases

---

## Section 6: Quality Evolution Across Stages

### Accuracy vs Coverage Trade-off

Each stage optimizes for different priorities:

| Stage | Priority | Accuracy | Coverage | Diversity |
|-------|----------|----------|----------|-----------|
| 1 | Accuracy | Very High | Low | Limited |
| 2 | Coverage | High | Medium | Growing |
| 3 | Scale | Good | Very High | Excellent |

### Mask Quality Metrics

**Boundary Precision:**
- How closely mask edges follow object boundaries
- Measured against human annotations
- SAM achieves near-human quality

**Predicted IoU:**
- Model's estimate of mask quality
- Calibrated against actual IoU
- Enables automatic quality filtering

**Stability Score:**
- Consistency under prompt perturbation
- High stability = reliable mask
- Used for filtering unreliable outputs

### Quality Comparison: Automatic vs Human

From the SAM paper, automatic masks from Stage 3 were compared to:
- Professional human annotations
- Other segmentation models (ViTDet)

Results showed automatic masks were:
- Comparable to human quality
- Better than prior automatic methods
- Suitable as sole data source for training

### Geographic and Demographic Quality

From [Encord](https://encord.com/blog/segment-anything-model-explained/):

SA-1B dataset diversity analysis:
> "SA-1B has a substantially higher percentage of images from Europe, Asia, and Oceania, as well as middle-income countries, compared to other open-source datasets."

**Coverage by Region:**
- All geographic regions included
- At least 28 million masks per region
- Africa has 10x more masks than previous largest dataset

### Quality Assurance Systems

From [Medium Article](https://medium.com/universe-of-ai/building-better-data-engines-metas-3-stage-data-engine-sam-d7714ab5e9c5):

**Monitoring:**
- Real-time dashboards
- Performance tracking per annotator
- Speed and accuracy metrics

**Version Control:**
- Annotations have version history
- Changes can be reverted
- Audit trail for quality issues

**Gold Standard Testing:**
- Small high-quality reference set
- Regular comparison to production annotations
- Detects drift and systematic errors

### Lessons for Quality at Scale

1. **Start with quality, scale later**: Stage 1's careful groundwork enables Stage 3's automation
2. **Use the model to help**: Model assistance makes human annotation faster and more consistent
3. **Iterate continuously**: Don't wait for perfect data - improve incrementally
4. **Verify at every stage**: Human checks prevent error propagation
5. **Embrace diversity**: Actively seek what the model misses

---

## Section 7: Technical Implementation Details

### Annotation Interface Architecture

**Browser-Based Tool:**
```
User Interface (Browser)
    ↓
WebSocket Connection
    ↓
SAM Server (GPU)
    ↓
Real-time Mask Prediction
    ↓
Overlay on Image
```

**Latency Requirements:**
- Mask generation: <50ms
- Interactive, real-time feel
- No perceivable delay for annotators

### Grid Prompting in Stage 3

**Grid Configuration:**
```python
# Automatic mask generation settings
class AutoMaskConfig:
    points_per_side = 32  # 32x32 grid = 1024 points
    pred_iou_thresh = 0.88
    stability_score_thresh = 0.95
    crop_n_layers = 1
    crop_n_points_downscale_factor = 2
    min_mask_region_area = 100
```

**Multi-Scale Processing:**
- Process at multiple crop scales
- Combine masks across scales
- Capture both large and small objects

### Quality Score Computation

**Predicted IoU:**
- Separate prediction head in SAM
- Trained to estimate actual mask quality
- Used for ranking and filtering

**Stability Score:**
```python
def stability_score(mask, logits, threshold_offset=1.0):
    """
    Compute stability as IoU between masks at different thresholds.
    High stability = mask doesn't change much with threshold.
    """
    high_thresh_mask = logits > (threshold + threshold_offset)
    low_thresh_mask = logits > (threshold - threshold_offset)

    intersection = (high_thresh_mask & low_thresh_mask).sum()
    union = (high_thresh_mask | low_thresh_mask).sum()

    return intersection / union
```

### Data Pipeline

**Image Processing:**
```
Raw Images (various sizes)
    ↓
Resize to 1024x1024
    ↓
Normalize (ImageNet stats)
    ↓
Image Encoder (once per image)
    ↓
Cached Embedding
```

**Mask Generation:**
```
Cached Embedding + Grid Points
    ↓
Prompt Encoder + Mask Decoder
    ↓
Raw Mask Logits
    ↓
Post-Processing (NMS, filtering)
    ↓
Final Masks with Metadata
```

### Storage and Management

**Mask Storage Format:**
```python
{
    'segmentation': binary_mask,  # H x W
    'area': int,
    'bbox': [x, y, w, h],
    'predicted_iou': float,
    'stability_score': float,
    'crop_box': [x, y, w, h],  # If from cropped region
    'point_coords': [[x, y], ...]  # Prompt points
}
```

**Dataset Size:**
- Images: Downscaled to ~1500px on longest edge
- Masks: Run-length encoded for compression
- Total SA-1B size: ~5TB

### Compute Infrastructure

**Stage 3 Requirements:**
- GPU cluster for parallel mask generation
- 11 million images x ~100 masks each
- Estimated: thousands of GPU-hours

**Training Final Model:**
- Multiple training runs with different seeds
- A100 GPUs (likely hundreds)
- Days of training time

---

## Section 8: ARR-COC Integration Patterns

### Applicable Data Engine Principles

SAM's three-stage data engine offers powerful patterns for ARR-COC training data creation and model improvement.

### Pattern 1: Progressive Automation for ARR Data

**Stage 1 - Assisted Manual (Character Recognition)**
```python
# ARR Implementation Concept
class AssistedAnnotation:
    """
    Use current ARR model to assist manual character annotation.
    """
    def annotate_page(self, image, model):
        # Model proposes character boxes
        proposals = model.detect_characters(image)

        # Human verifies and corrects
        verified = human_review(proposals)

        # High-quality ground truth
        return verified
```

**Application:**
- Bootstrap new character sets with model assistance
- Correct OCR errors with visual model proposals
- Build reading order training data efficiently

### Pattern 2: Model-in-the-Loop Training

**Continuous Improvement Cycle:**
```python
class ARRDataEngine:
    def iteration_cycle(self):
        # Stage 1: Current model assists annotation
        new_data = self.collect_assisted_annotations()

        # Stage 2: Retrain model
        improved_model = self.train_on_new_data(new_data)

        # Stage 3: Use improved model for next round
        self.model = improved_model

        # Better model → Better annotations → Repeat
```

**Benefits for ARR:**
- Faster annotation of new comic styles
- Improved reading order prediction
- Better handling of unusual layouts

### Pattern 3: Quality-Speed Trade-off

**Different Phases Need Different Priorities:**

| ARR Development Phase | SAM Equivalent | Priority |
|----------------------|----------------|----------|
| New character set | Stage 1 | Accuracy (few, perfect samples) |
| Expanding coverage | Stage 2 | Coverage (more samples, good quality) |
| Production scaling | Stage 3 | Scale (many samples, quality filtering) |

### Pattern 4: Confidence-Based Automation

**Auto-Accept High Confidence:**
```python
class ARRAutoAnnotation:
    def process_panel(self, panel, model):
        predictions = model.predict(panel)

        for pred in predictions:
            if pred.confidence > HIGH_THRESHOLD:
                # Auto-accept: no human needed
                self.accept_annotation(pred)
            elif pred.confidence > MEDIUM_THRESHOLD:
                # Human verification
                self.flag_for_review(pred)
            else:
                # Discard or full manual annotation
                self.manual_annotate(panel)
```

### Pattern 5: Diversity Injection

**Active Learning for ARR:**
```python
class DiversitySampler:
    """
    Specifically sample cases the model struggles with.
    """
    def select_for_annotation(self, dataset, model):
        # Find low-confidence predictions
        uncertain = [x for x in dataset if model.predict(x).confidence < 0.7]

        # Find underrepresented styles/layouts
        rare = [x for x in dataset if x.style in RARE_STYLES]

        # Combine for diverse training
        return uncertain + rare
```

**Application:**
- Target unusual panel layouts
- Focus on rare character styles
- Sample challenging reading orders

### Integration Architecture

**ARR Data Engine Pipeline:**
```
Raw Comic Pages
    ↓
Current Model Proposals
    ↓
Human Review Interface
    ↓
Quality-Filtered Annotations
    ↓
Model Retraining
    ↓
Improved Model
    ↓
(Cycle Repeats)
```

### Key Takeaways for ARR-COC

1. **Start with quality**: Build small, perfect datasets before scaling
2. **Use the model**: Let current model assist annotation
3. **Iterate fast**: Don't wait for perfect data - improve continuously
4. **Filter by confidence**: Auto-accept high confidence, review medium
5. **Inject diversity**: Actively target what the model struggles with
6. **Monitor quality**: Track annotation accuracy over time

The SAM data engine demonstrates that massive scale is achievable through careful iteration between human expertise and model capability - a pattern directly applicable to building better ARR training datasets.

---

## Sources

**Source Documents:**
- [SAM_STUDY_GENERAL.md](../../PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md) - Primary research document with data engine details

**Web Research:**
- [Meta's 3 Stage Data Engine, Segment Anything Model](https://medium.com/universe-of-ai/building-better-data-engines-metas-3-stage-data-engine-sam-d7714ab5e9c5) - Kusha Sahu, Universe of AI (accessed 2025-11-20)
- [Segment Anything Model (SAM) Explained](https://encord.com/blog/segment-anything-model-explained/) - Encord Blog (accessed 2025-11-20)
- [Segment Anything Paper](https://arxiv.org/abs/2304.02643) - Kirillov et al., arXiv:2304.02643 (April 2023)

**Additional References:**
- [GitHub: facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
- [SA-1B Dataset](https://ai.meta.com/datasets/segment-anything/)
- [CVF Open Access - SAM Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf)
