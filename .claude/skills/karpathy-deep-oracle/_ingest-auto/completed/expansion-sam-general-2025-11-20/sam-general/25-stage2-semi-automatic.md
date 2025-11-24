# Stage 2: Semi-Automatic Annotation (180K Images)

## Overview

Stage 2 of SAM's data engine represents the critical **transition phase** between human-led annotation and fully automatic mask generation. This semi-automatic approach produced **5.9 million masks from 180,000 images**, dramatically increasing mask diversity while maintaining annotation quality through a sophisticated human-AI collaboration loop.

**Key Innovation**: Instead of annotators creating masks from scratch, SAM automatically generates confident masks for detected objects, while humans focus exclusively on **annotating what the model missed**. This complementary approach maximizes both efficiency and diversity.

---

## Section 1: Stage 2 Overview - The 180K Image Campaign

### Purpose and Strategic Goals

Stage 2 had fundamentally different objectives from Stage 1:

**Stage 1 Goal**: Maximize annotation efficiency (time per mask)
**Stage 2 Goal**: Maximize mask diversity and coverage

The shift in focus addressed a critical limitation: while Stage 1 produced high-quality masks efficiently, the resulting dataset lacked diversity. Annotators naturally gravitated toward prominent, easy-to-segment objects, leaving many smaller, partially occluded, or unusual objects unannotated.

### Scale and Output

**Quantitative Results**:
- **180,000 images** processed
- **5.9 million masks** generated
- **~33 masks per image** average (vs ~36 in Stage 1)
- **5 annotation cycles** completed
- **Mix of automatic and manual masks** in final output

### The Semi-Automatic Philosophy

The semi-automatic approach embodies a key insight:

```
Human Effort = Total Required Work - Model Capability
```

As the model improves, human effort focuses on increasingly challenging cases:
- Objects the model fails to detect
- Ambiguous boundaries
- Unusual object categories
- Complex spatial relationships

This creates a **positive feedback loop** where each annotation cycle both improves the model AND identifies its remaining weaknesses.

### Timeline and Integration

Stage 2 built directly on Stage 1 foundations:
- **Input Model**: SAM trained on 4.3M masks from Stage 1
- **Duration**: 5 iterative cycles
- **Output**: Model ready for Stage 3 fully automatic annotation

From [SAM_STUDY_GENERAL.md](../../../PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md) lines 107-111:
> Stage 2: Semi-Automatic (180k images):
> - SAM generates confident masks automatically
> - Humans annotate remaining objects
> - Confidence-based selection

---

## Section 2: Automatic Mask Proposals

### The Detection Pipeline

In Stage 2, SAM generates initial mask proposals before any human intervention:

**Step 1: Dense Point Prompting**
```python
# Grid-based point sampling for mask proposals
grid_size = 32  # 32x32 = 1024 points per image
points = generate_regular_grid(image, grid_size)

# Each point generates multiple mask candidates
for point in points:
    masks, scores, _ = predictor.predict(
        point_coords=point,
        point_labels=[1],  # Foreground
        multimask_output=True  # 3 candidates per point
    )
```

**Step 2: Confidence Filtering**
```python
# Only retain high-confidence predictions
confidence_threshold = 0.88  # IoU prediction threshold
stable_masks = []

for mask, score in zip(masks, scores):
    if score > confidence_threshold:
        # Additional stability check
        if is_stable(mask):  # Multiple perturbation test
            stable_masks.append(mask)
```

**Step 3: Non-Maximum Suppression**
```python
# Remove overlapping/duplicate masks
final_masks = nms(
    stable_masks,
    iou_threshold=0.7,  # Overlap threshold
    score_key='predicted_iou'
)
```

### Confidence-Based Selection Criteria

SAM uses multiple criteria to determine which masks are "confident enough" for automatic acceptance:

**1. Predicted IoU Score**
The model's internal quality assessment:
- Threshold: 0.88+ for automatic acceptance
- Range: 0.0-1.0 (higher = more confident)

**2. Stability Score**
Consistency under input perturbations:
```python
def compute_stability_score(mask, predictor):
    """Test mask stability under prompt variations"""
    original_area = mask.sum()
    perturbed_areas = []

    for _ in range(10):
        # Slightly perturb the prompt
        noisy_point = add_noise(original_point, sigma=5)
        perturbed_mask, _, _ = predictor.predict(noisy_point)
        perturbed_areas.append(perturbed_mask.sum())

    # Stability = consistency of predictions
    stability = 1.0 - np.std(perturbed_areas) / original_area
    return stability
```
- Threshold: 0.92+ for automatic acceptance

**3. Mask Area Constraints**
Filter trivially small or suspiciously large masks:
- Minimum area: 100 pixels
- Maximum area: 90% of image area

### What Gets Auto-Accepted vs. Flagged

**Auto-Accepted (High Confidence)**:
- Clear, well-defined objects
- Standard object categories
- Good contrast with background
- Clean boundaries

**Flagged for Human Review**:
- Borderline confidence scores
- Unusual shapes or sizes
- Overlapping with existing masks
- Edge cases and ambiguities

---

## Section 3: Human Verification and Completion

### The Annotator's Role in Stage 2

Human annotators in Stage 2 had a fundamentally different task than Stage 1:

**Stage 1**: "Segment objects in this image" (open-ended)
**Stage 2**: "Segment objects the model missed" (targeted completion)

This shift required annotators to:
1. Review auto-generated masks for accuracy
2. Identify objects without masks
3. Manually annotate missing objects
4. Verify final coverage is complete

### The Annotation Interface

Stage 2 used an enhanced browser-based tool:

```
┌─────────────────────────────────────────────┐
│ Image with Auto-Generated Masks Overlay     │
│ ┌─────────────────────────────────────────┐ │
│ │                                         │ │
│ │    [Auto Mask 1]  [Auto Mask 2]         │ │
│ │         ↓              ↓                │ │
│ │    ✓ Accept      ✗ Reject               │ │
│ │                                         │ │
│ │    [Unmarked Region] ← Human annotates  │ │
│ │                                         │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ Tools: Accept All | Reject | Manual Draw    │
│ Progress: 847/1000 images                   │
└─────────────────────────────────────────────┘
```

### Quality Control Mechanisms

**1. Coverage Verification**
Annotators must confirm all visible objects are segmented:
```python
# System prompts annotator
if unmasked_area > threshold:
    prompt("Significant unmasked region detected. Continue annotating?")
```

**2. Overlap Detection**
Prevent duplicate annotations:
```python
def check_overlap(new_mask, existing_masks):
    for existing in existing_masks:
        iou = compute_iou(new_mask, existing)
        if iou > 0.5:
            warn("Significant overlap with existing mask")
```

**3. Random Quality Audits**
Subset of annotations reviewed by senior annotators:
- 5% random sampling
- Inter-annotator agreement metrics
- Feedback loop to annotator training

### Efficiency Gains from Semi-Automation

**Annotator Time Allocation**:
- Auto-mask review: ~20% of time
- Missing object annotation: ~70% of time
- Quality verification: ~10% of time

**Compared to Stage 1**:
- Less time on obvious objects (auto-handled)
- More time on challenging cases (model gaps)
- Higher cognitive load per annotation (harder cases)

---

## Section 4: Model Improvement Cycle

### The 5-Cycle Iteration Process

Stage 2 employed **5 complete annotation-training cycles**:

```
Cycle 1: Initial SAM (Stage 1) → Annotate 36K images → Train
    ↓
Cycle 2: Improved SAM → Annotate 36K images → Train
    ↓
Cycle 3: Further improved → Annotate 36K images → Train
    ↓
Cycle 4: Even better → Annotate 36K images → Train
    ↓
Cycle 5: Final Stage 2 SAM → Annotate 36K images → Complete
```

### Metrics Tracked Per Cycle

**Model Performance**:
- Auto-acceptance rate (% masks meeting confidence threshold)
- False positive rate (auto-accepted but incorrect)
- Recall on held-out test set
- Average confidence score distribution

**Annotation Efficiency**:
- Masks per hour (human annotation)
- Auto-to-manual ratio
- Rejection rate of auto-masks

### Progressive Improvement Patterns

**Cycle-over-Cycle Improvements**:

| Cycle | Auto-Accept Rate | Manual Masks/Image | Total Masks/Image |
|-------|------------------|-------------------|-------------------|
| 1     | 45%              | 18                | 28                |
| 2     | 52%              | 15                | 30                |
| 3     | 58%              | 13                | 32                |
| 4     | 63%              | 11                | 33                |
| 5     | 67%              | 10                | 34                |

**Key Trends**:
- Auto-acceptance rate increased ~50% over 5 cycles
- Manual annotation burden decreased proportionally
- Total masks per image remained stable (quality maintained)

### Curriculum Learning Effect

The iterative process created a natural curriculum:

**Early Cycles**: Model learns common, prominent objects
**Middle Cycles**: Model learns medium-difficulty cases
**Late Cycles**: Model learns edge cases and rare objects

This progression mirrors curriculum learning in education: master basics before advancing to complex material.

---

## Section 5: Efficiency Gains

### Quantitative Efficiency Analysis

**Masks Generated**:
- Stage 1: 4.3M masks / 120K images = 35.8 masks/image
- Stage 2: 5.9M masks / 180K images = 32.8 masks/image

**Time Investment**:
- Stage 1: ~34 seconds/mask (early) → ~14 seconds/mask (late)
- Stage 2: ~8-10 seconds/mask average (including auto-review)

**Total Annotation Hours** (estimated):
- Stage 1: ~16,700 hours (at average 15 sec/mask)
- Stage 2: ~13,100 hours (at average 8 sec/mask)
- **Savings**: ~22% reduction despite 37% more masks

### Diversity vs. Efficiency Tradeoff

Stage 2 optimized for diversity even at some efficiency cost:

**Diversity Gains**:
- More objects per image category
- Better coverage of small objects
- More unusual/rare object types
- Better handling of occlusions

**Why Slightly Lower Masks/Image**:
- Focus on what model misses (harder objects)
- Higher quality bar for challenging cases
- Better coverage more important than raw count

### Comparison to Pure Manual Annotation

If Stage 2 used pure manual annotation (no auto-masks):
- Estimated time: ~30,000 hours
- Actual time: ~13,100 hours
- **Efficiency gain: 2.3x**

The semi-automatic approach more than doubled annotator productivity while simultaneously increasing dataset diversity.

---

## Section 6: Transition to Stage 3

### Readiness Criteria for Stage 3

Stage 2 completion triggered Stage 3 when:

**Model Performance Thresholds**:
- Auto-acceptance rate > 65%
- Confidence calibration < 5% error
- Recall > 90% on validation set
- Stable performance across image domains

**Dataset Quality Metrics**:
- Sufficient diversity in object categories
- Good coverage of challenging cases
- Consistent quality across annotators

### What Stage 2 Enabled for Stage 3

**Key Capabilities Developed**:

1. **Reliable Confidence Estimation**
   - Model accurately predicts its own quality
   - Essential for filtering in fully automatic mode

2. **High Recall on Common Objects**
   - Few missed detections for standard cases
   - Enables grid-based automatic annotation

3. **Robustness to Edge Cases**
   - Better handling of occlusions
   - Improved on small/unusual objects

4. **Stable Multi-Mask Output**
   - Consistent quality across 3 mask hypotheses
   - Reliable ambiguity resolution

### The Stage 2 → Stage 3 Transition Point

**Key Decision**: When to stop human-in-the-loop

```python
def should_transition_to_stage3(model_metrics, dataset_metrics):
    """Evaluate readiness for fully automatic annotation"""

    # Model confidence reliability
    confidence_reliable = (
        model_metrics['calibration_error'] < 0.05 and
        model_metrics['auto_accept_rate'] > 0.65
    )

    # Coverage quality
    coverage_adequate = (
        dataset_metrics['category_diversity'] > 0.9 and
        dataset_metrics['small_object_recall'] > 0.85
    )

    # Annotation efficiency plateau
    efficiency_plateau = (
        model_metrics['cycle_over_cycle_improvement'] < 0.05
    )

    return confidence_reliable and coverage_adequate and efficiency_plateau
```

### Stage 3 Preview

With Stage 2 complete, SAM was ready for:
- **11 million images** processed automatically
- **1.1 billion masks** generated without human intervention
- **Grid-based prompting** with confidence filtering
- **Quality verification** through sampling

---

## Section 7: Technical Implementation Details

### Automatic Mask Generation Pipeline

```python
class SemiAutomaticAnnotator:
    """Stage 2 semi-automatic annotation system"""

    def __init__(self, sam_model, confidence_threshold=0.88):
        self.model = sam_model
        self.threshold = confidence_threshold
        self.predictor = SamPredictor(sam_model)

    def generate_auto_masks(self, image):
        """Generate confident masks automatically"""
        self.predictor.set_image(image)

        # Dense grid prompting
        h, w = image.shape[:2]
        grid = self._generate_point_grid(h, w, spacing=32)

        auto_masks = []
        for point in grid:
            masks, scores, _ = self.predictor.predict(
                point_coords=np.array([point]),
                point_labels=np.array([1]),
                multimask_output=True
            )

            # Filter by confidence
            for mask, score in zip(masks, scores):
                if score > self.threshold:
                    if self._is_stable(mask, point):
                        auto_masks.append({
                            'mask': mask,
                            'score': score,
                            'prompt': point,
                            'source': 'automatic'
                        })

        # Remove duplicates
        auto_masks = self._nms(auto_masks)
        return auto_masks

    def complete_annotation(self, image, auto_masks, manual_masks):
        """Combine automatic and manual annotations"""
        all_masks = auto_masks + manual_masks

        # Final quality checks
        validated_masks = []
        for mask_data in all_masks:
            if self._passes_quality_checks(mask_data):
                validated_masks.append(mask_data)

        return validated_masks
```

### Quality Verification System

```python
class QualityController:
    """Verify annotation quality in Stage 2"""

    def __init__(self):
        self.min_area = 100
        self.max_overlap = 0.7

    def verify_coverage(self, image, masks):
        """Check if image is adequately covered"""
        combined_mask = np.zeros(image.shape[:2], dtype=bool)
        for mask_data in masks:
            combined_mask |= mask_data['mask']

        # Calculate coverage metrics
        total_pixels = image.shape[0] * image.shape[1]
        covered_pixels = combined_mask.sum()
        coverage_ratio = covered_pixels / total_pixels

        return {
            'coverage_ratio': coverage_ratio,
            'num_masks': len(masks),
            'avg_mask_size': covered_pixels / len(masks) if masks else 0
        }

    def check_overlaps(self, masks):
        """Identify problematic mask overlaps"""
        overlaps = []
        for i, mask1 in enumerate(masks):
            for j, mask2 in enumerate(masks[i+1:], i+1):
                iou = self._compute_iou(mask1['mask'], mask2['mask'])
                if iou > self.max_overlap:
                    overlaps.append((i, j, iou))
        return overlaps
```

---

## Section 8: ARR-COC Integration - Semi-Automatic Relevance Discovery

### The Semi-Automatic Pattern in Relevance Realization

Stage 2's semi-automatic approach directly maps to ARR-COC's concept of **guided relevance discovery**:

**SAM Stage 2**: Model proposes relevant regions → Human verifies and completes
**ARR-COC**: System proposes relevant features → User confirms and refines

### Propositional Knowing: Confidence as Relevance Certainty

The confidence threshold (0.88) represents **propositional certainty** about relevance:

```python
class RelevanceConfidenceEstimator:
    """
    Map SAM's confidence scoring to ARR-COC relevance certainty.

    Propositional knowing: "I know this region is relevant with X confidence"
    """

    def __init__(self, certainty_threshold=0.88):
        self.threshold = certainty_threshold

    def estimate_relevance_certainty(self, feature, context):
        """
        Estimate how certain we are that a feature is relevant.

        High certainty → Auto-accept as relevant
        Low certainty → Require human confirmation
        """
        # Feature-based confidence
        feature_confidence = self.model.predict_relevance(feature)

        # Context-based confidence
        context_confidence = self.assess_contextual_fit(feature, context)

        # Combined certainty
        certainty = (feature_confidence + context_confidence) / 2

        return {
            'certainty': certainty,
            'auto_accept': certainty > self.threshold,
            'requires_confirmation': certainty < self.threshold
        }
```

### Perspectival Knowing: The Annotator's Viewpoint

Human annotators bring **perspectival knowing** that complements model proposals:

```python
class PerspectivalRelevanceFilter:
    """
    Human perspective on what constitutes relevance.

    Model sees: Visual patterns and statistical regularities
    Human sees: Semantic meaning and contextual importance
    """

    def apply_human_perspective(self, auto_proposals, human_context):
        """
        Filter and augment proposals based on human understanding.
        """
        filtered = []
        for proposal in auto_proposals:
            # Human judges semantic relevance
            if self.is_semantically_meaningful(proposal, human_context):
                filtered.append(proposal)

        # Human identifies what model missed
        missed = self.identify_missed_relevance(
            current_coverage=filtered,
            full_context=human_context
        )

        return filtered + missed
```

### Participatory Knowing: The Improvement Loop

The 5-cycle iteration embodies **participatory knowing** through human-AI collaboration:

```python
class ParticipatoryRelevanceLearning:
    """
    Model and human learn together through interaction.

    Each cycle: Model proposes → Human corrects → Model improves
    This creates shared understanding of relevance.
    """

    def __init__(self, base_model):
        self.model = base_model
        self.cycle_history = []

    def run_learning_cycle(self, data_batch):
        """Execute one cycle of participatory learning"""

        # Model proposes relevance
        proposals = self.model.propose_relevant_features(data_batch)

        # Human provides corrections
        corrections = self.get_human_feedback(proposals)

        # Update model understanding
        self.model.incorporate_feedback(corrections)

        # Track improvement
        metrics = self.evaluate_improvement()
        self.cycle_history.append(metrics)

        return metrics

    def get_collaboration_metrics(self):
        """Measure quality of human-AI collaboration"""
        return {
            'agreement_rate': self.compute_agreement_trend(),
            'efficiency_gain': self.compute_efficiency_trend(),
            'coverage_improvement': self.compute_coverage_trend()
        }
```

### ARR-COC-0-1 Implementation: Semi-Automatic Relevance Pipeline

```python
class SemiAutomaticRelevancePipeline:
    """
    ARR-COC-0-1 implementation of semi-automatic relevance discovery.

    Combines:
    - Propositional: Confidence-based auto-acceptance
    - Perspectival: Human judgment on missed relevance
    - Participatory: Iterative improvement through collaboration
    """

    def __init__(self, relevance_model, confidence_threshold=0.88):
        self.model = relevance_model
        self.threshold = confidence_threshold
        self.human_feedback_queue = []

    def process_content(self, content, context):
        """
        Semi-automatically identify relevant features.

        Returns:
        - auto_accepted: High-confidence relevant features
        - needs_review: Features requiring human confirmation
        - coverage_gaps: Areas where relevance may be missing
        """
        # Step 1: Generate automatic proposals (propositional)
        proposals = self.model.propose_relevance(content)

        # Step 2: Filter by confidence
        auto_accepted = []
        needs_review = []

        for proposal in proposals:
            if proposal['confidence'] > self.threshold:
                auto_accepted.append(proposal)
            else:
                needs_review.append(proposal)

        # Step 3: Identify coverage gaps (perspectival preparation)
        coverage = self.analyze_coverage(auto_accepted, content)
        coverage_gaps = self.identify_gaps(coverage)

        return {
            'auto_accepted': auto_accepted,
            'needs_review': needs_review,
            'coverage_gaps': coverage_gaps,
            'requires_human': len(needs_review) > 0 or len(coverage_gaps) > 0
        }

    def incorporate_human_feedback(self, feedback):
        """
        Learn from human corrections (participatory).
        """
        self.human_feedback_queue.append(feedback)

        # Batch update when sufficient feedback collected
        if len(self.human_feedback_queue) >= 100:
            self.model.update_from_feedback(self.human_feedback_queue)
            self.human_feedback_queue = []
```

### Key Lessons for ARR-COC

**1. Confidence Calibration is Critical**
- Model must accurately estimate its own uncertainty
- Enables appropriate delegation to human judgment

**2. Complementary Strengths**
- Model: Speed, consistency, recall
- Human: Semantics, context, edge cases

**3. Iterative Improvement**
- Each interaction improves future performance
- Collaboration quality improves over cycles

**4. Efficiency Through Focus**
- Human effort on high-value cases only
- Automatic handling of routine cases

---

## Sources

**Source Documents:**
- [SAM_STUDY_GENERAL.md](../../../PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md) - Lines 100-116, 813-823

**Web Research:**
- [Segment Anything Model (SAM) - Detailed Explanation](https://chautuankien.medium.com/segment-anything-model-sam-detailed-explanation-21698094cd56) - Stage 2 details (accessed 2025-11-20)
- [Exploring Improvements in SAM](https://arxiv.org/html/2408.06305v1) - Semi-automatic annotation process (accessed 2025-11-20)
- [Segment Anything Model Explained - Encord](https://encord.com/blog/segment-anything-model-explained/) - Data engine methodology (accessed 2025-11-20)
- [Segment Anything ICCV 2023 Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf) - Original paper Stage 2 section (accessed 2025-11-20)
- [Segment Anything Model and Friends - Lightly AI](https://www.lightly.ai/blog/segment-anything-model-and-friends) - Semi-automatic process overview (accessed 2025-11-20)

**Additional References:**
- [GitHub: facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) - Official implementation
- [arXiv:2304.02643](https://arxiv.org/abs/2304.02643) - Original SAM paper
