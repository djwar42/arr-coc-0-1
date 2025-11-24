# Occlusion Handling in 3D Reconstruction

## Overview

Occlusion handling is one of the most challenging problems in single-image 3D reconstruction. When objects are partially hidden by other objects, the scene boundary, or parts of themselves, reconstructors must infer the complete 3D geometry from incomplete visual evidence. This requires sophisticated reasoning about object structure, learned priors from training data, and explicit awareness of which regions are visible versus hidden.

**Core Challenge**: From a single 2D image, determine not just what is visible, but what the complete 3D object looks like including all hidden surfaces. This is fundamentally ill-posed since infinite valid completions exist for any partial observation.

---

## Section 1: Occlusion Types in 3D Reconstruction

### 1.1 Self-Occlusion

**Definition**: Parts of an object hidden by other parts of the same object when viewed from a particular angle.

**Characteristics**:
- Always present in 3D objects viewed from any single angle
- Amount varies with viewing angle and object complexity
- Predictable for simple geometric shapes
- Highly object-specific for complex shapes

**Examples**:
- Back of a chair hidden by seat and backrest
- Opposite side of a mug from the handle
- Underside of a car visible only from certain angles
- Interior surfaces of hollow objects

**Challenge Level**: Moderate - Structure often follows geometric patterns

From [3D Shape Completion Dataset](https://www.ri.cmu.edu/app/uploads/2019/05/Thesis_Draft.pdf) (CMU Robotics Institute, 2019):
- "Our dataset contains challenging inputs with arbitrary 3D rotation, translation and realistic self-occlusion patterns"
- Self-occlusion is the primary challenge even for isolated objects

### 1.2 Inter-Object Occlusion

**Definition**: One object blocking view of another object in the scene.

**Characteristics**:
- Common in cluttered real-world environments
- Occlusion extent depends on scene arrangement
- May hide crucial identifying features
- Requires scene-level reasoning

**Examples**:
- Furniture partially hidden by other furniture
- Products on store shelves blocking each other
- People in crowds occluding each other
- Objects in boxes partially visible

**Challenge Level**: High - No geometric relationship between occluder and occluded

From [Amodal3R](https://arxiv.org/abs/2503.13439) (Wu et al., 2025):
- "Most image-based 3D object reconstructors assume that objects are fully visible, ignoring occlusions that commonly occur in real-world scenarios"
- Inter-object occlusion is the primary gap between lab and real-world performance

### 1.3 Boundary/Frustum Occlusion

**Definition**: Objects extending beyond the image boundary, partially visible in frame.

**Characteristics**:
- Common when cameras don't capture full scene
- Unknown how much of object extends beyond boundary
- Different from occluder-based hiding
- May include critical structural elements

**Examples**:
- Half a car visible at image edge
- Person's body cropped at waist
- Building extending beyond frame
- Furniture partially in photo

**Challenge Level**: Very High - No information about extent beyond boundary

### 1.4 Depth-Based Occlusion

**Definition**: Surfaces hidden due to depth ordering from camera viewpoint.

**Characteristics**:
- Fundamentally related to single-view limitation
- Includes both near and far hidden surfaces
- Changes completely with viewpoint
- Core limitation of monocular 3D

**Examples**:
- All back-facing surfaces of convex objects
- Interior of concave regions
- Surfaces behind foreground objects
- Hidden geometric features

---

## Section 2: Amodal Completion Strategies

### 2.1 Definition and Background

**Amodal Completion**: The perceptual process of perceiving complete shapes even when parts are occluded. Term originates from cognitive psychology - humans naturally "complete" partially visible objects.

From [Image Amodal Completion Survey](https://www.sciencedirect.com/science/article/abs/pii/S1077314223000413) (Ao et al., 2023):
- "Image amodal completion aims to equip computers with human-like amodal completion functions to understand an intact object despite it being partially occluded"

**2D vs 3D Amodal Completion**:
- **2D Amodal**: Complete the object silhouette/mask in image space
- **3D Amodal**: Complete the full 3D geometry including all hidden surfaces

### 2.2 Sequential Pipeline Approach

**Strategy**: First complete 2D appearance, then reconstruct 3D from completed image.

**Pipeline**:
```
Occluded Image → 2D Amodal Completion → Complete 2D Image → 3D Reconstruction
```

**Advantages**:
- Leverages existing 2D inpainting methods
- Can use off-the-shelf 3D reconstructors
- Modular, easy to understand

**Disadvantages**:
- Error accumulation across stages
- 2D completion may not be 3D-consistent
- No joint optimization

From [Amodal3R](https://arxiv.org/abs/2503.13439):
- "It substantially outperforms existing methods that independently perform 2D amodal completion followed by 3D reconstruction"

### 2.3 End-to-End Occlusion-Aware Models

**Strategy**: Train single model that directly handles occlusion during 3D generation.

**Key Innovation**: Model explicitly knows which regions are occluded and conditions generation accordingly.

**Architectural Components**:
1. **Occlusion Mask Input**: Binary mask indicating visible vs occluded regions
2. **Mask-Weighted Attention**: Focus more on visible features
3. **Occlusion-Aware Layers**: Explicitly reason about hidden geometry

From [Amodal3R](https://arxiv.org/abs/2503.13439):
- "We introduce a mask-weighted multi-head cross-attention mechanism followed by an occlusion-aware attention layer that explicitly leverages occlusion priors to guide the reconstruction process"

### 2.4 Multi-Slice Reasoning

**Strategy**: Instead of predicting views from different angles, predict "slices" through the object to reveal hidden structure.

From [Slice3D](https://yizhiwang96.github.io/Slice3D/) (Wang et al., CVPR 2024):
- "Our key observation is that object slicing is more advantageous than altering views to reveal occluded structures"
- "Slicing can peel through any occluder without obstruction, and in the limit (infinitely many slices), it is guaranteed to unveil all hidden object parts"

**Pipeline**:
```
Single RGB Image → Multi-Slice Prediction → 3D Integration via Transformer
```

**Advantages**:
- Guaranteed to reveal all hidden parts (with enough slices)
- No view-consistency challenges
- Natural handling of self-occlusion

**Performance**: "Slice3D can produce a 3D mesh from a single view input within only 20 seconds on a NVIDIA A40 GPU"

### 2.5 Probabilistic Completion

**Strategy**: Generate multiple plausible completions rather than single deterministic output.

From [Variational Amodal Object Completion](https://papers.nips.cc/paper/2020/file/bacadc62d6e67d7897cef027fa2d416c-Paper.pdf) (Ling et al., NeurIPS 2020):
- "In images of complex scenes, objects are often occluding each other which makes perception tasks such as object detection and tracking, or robotic control difficult"
- Uses variational approach to model distribution of completions

**Benefits**:
- Captures inherent ambiguity
- Can sample multiple valid completions
- Better uncertainty quantification

---

## Section 3: Learned Priors for Hidden Geometry

### 3.1 Shape Priors from Large-Scale Training

**Key Insight**: Neural networks learn statistical regularities of object shapes from training data, enabling informed guessing about hidden regions.

**What Networks Learn**:
- Typical object symmetries (most objects have bilateral symmetry)
- Category-specific structure (chairs have four legs, tables have flat tops)
- Common geometric patterns (spherical, cylindrical, planar)
- Part relationships (handles attach to bodies, legs support seats)

**Training Data Sources**:
- ShapeNet: ~51,000 3D models across 55 categories
- Objaverse: ~800,000 3D objects
- Google Scanned Objects: High-quality real object scans
- Synthetic renders with controlled occlusion

### 3.2 Symmetry-Based Priors

**Observation**: Most man-made objects exhibit some form of symmetry.

From [Symmetry-Aware Generative Network for Occluded Shape Completion](https://dl.acm.org/doi/full/10.1145/3760678.3760692) (Li et al., 2025):
- "It enables humans to mentally 'mirror' visible parts across an estimated symmetry axis, effectively filling in occluded geometry with plausible structure"

**Symmetry Types**:
1. **Bilateral**: Mirror symmetry (most common)
2. **Rotational**: n-fold rotational symmetry
3. **Translational**: Repeating patterns

**Implementation**:
```python
# Symmetry-aware completion
visible_features = encoder(visible_region)
symmetry_axis = estimate_symmetry(visible_features)
mirrored_features = reflect(visible_features, symmetry_axis)
completed_shape = decoder(concat(visible_features, mirrored_features))
```

### 3.3 Category-Specific Priors

**Strategy**: Use object category information to constrain completion.

**Examples**:
- **Chairs**: Expect 4 legs, seat, back
- **Cars**: Expect wheels, body, windows
- **Cups**: Expect cylindrical body, handle
- **Tables**: Expect flat top, legs

**Challenges**:
- Category must be known or inferred
- Doesn't handle novel categories well
- May over-constrain unusual instances

### 3.4 Diffusion Model Priors

**Modern Approach**: Use pre-trained diffusion models as strong generative priors.

**Benefits**:
- Learned from massive datasets
- High-quality, diverse completions
- Can condition on partial observations

**Architecture**:
```
Visible Observation → Conditioning → Diffusion Denoising → Complete 3D
```

From [Slice3D](https://yizhiwang96.github.io/Slice3D/):
- "The slice generator is a denoising diffusion model operating on the entirety of slice images stacked on the input channels"

### 3.5 Multi-View Consistency Priors

**Key Insight**: Completed geometry must be consistent when viewed from all angles.

From [3D Shape Completion with Multi-view Consistent Inference](https://www.cs.umd.edu/~zwicker/publications/3DShapeCompletionWithConsistentInference-AAAI2020.pdf) (Hu et al., AAAI 2020):
- "These methods represent shapes as multiple depth images, which can be back-projected to yield corresponding 3D point clouds"
- Multi-view consistency serves as implicit regularization

---

## Section 4: Uncertainty Estimation for Occluded Regions

### 4.1 Why Uncertainty Matters

**Fundamental Problem**: Occluded regions have inherently higher uncertainty than visible regions.

**Applications**:
- Robotics: Don't grasp uncertain regions
- Autonomous driving: Be cautious about uncertain obstacles
- AR/VR: Visual cues for uncertain geometry
- Quality assessment: Flag low-confidence reconstructions

### 4.2 Aleatoric vs Epistemic Uncertainty

**Aleatoric Uncertainty**: Inherent ambiguity in the task
- Multiple valid completions exist
- Cannot be reduced with more training data
- Example: Back of symmetric object could be mirror or different

**Epistemic Uncertainty**: Model's lack of knowledge
- Could be reduced with more data/better model
- Higher for out-of-distribution objects
- Example: Novel object category not in training

### 4.3 Uncertainty Quantification Methods

**Method 1: Probabilistic Outputs**

```python
class UncertaintyAwareReconstructor(nn.Module):
    def forward(self, image, occlusion_mask):
        features = self.encoder(image)

        # Predict mean and variance
        mean = self.mean_head(features)
        log_var = self.var_head(features)

        # Higher variance for occluded regions
        occlusion_penalty = self.occlusion_encoder(occlusion_mask)
        log_var = log_var + occlusion_penalty

        return mean, log_var
```

**Method 2: Ensemble Methods**

```python
# Train multiple models, measure disagreement
predictions = [model_i(image) for model_i in ensemble]
mean_prediction = torch.mean(predictions, dim=0)
uncertainty = torch.std(predictions, dim=0)

# High disagreement = high uncertainty
# Typically highest in occluded regions
```

**Method 3: Monte Carlo Dropout**

```python
# Keep dropout active during inference
model.train()  # Enables dropout
predictions = [model(image) for _ in range(N_samples)]
uncertainty = torch.std(predictions, dim=0)
```

### 4.4 Spatial Uncertainty Maps

**Visualization**: Color-code reconstructed mesh by confidence.

**Typical Pattern**:
- **Green (low uncertainty)**: Directly visible surfaces
- **Yellow (medium)**: Self-occluded but constrained by symmetry
- **Red (high)**: Inter-object occluded regions

**Uses**:
- Interactive refinement: User provides hints for uncertain regions
- Quality filtering: Reject reconstructions with too much uncertainty
- Downstream applications: Weight by confidence

### 4.5 Uncertainty-Guided Refinement

**Strategy**: Focus computational resources on uncertain regions.

```python
def iterative_refinement(reconstruction, uncertainty_map):
    for iteration in range(max_iterations):
        # Find most uncertain regions
        uncertain_regions = uncertainty_map > threshold

        # Generate alternatives for uncertain regions
        alternatives = sample_alternatives(reconstruction, uncertain_regions)

        # Score alternatives
        scores = evaluate_plausibility(alternatives)

        # Update with best alternative
        reconstruction = update_regions(reconstruction, alternatives, scores)

        # Recompute uncertainty
        uncertainty_map = compute_uncertainty(reconstruction)

    return reconstruction
```

---

## Section 5: Multi-View Fusion for Occlusion Resolution

### 5.1 Why Multi-View Helps

**Key Insight**: What's occluded from one view may be visible from another.

**Benefits**:
- Direct observation of previously hidden regions
- Constraint satisfaction across views
- Disambiguation of uncertain completions

**Limitation**: Requires multiple images (not single-image reconstruction)

### 5.2 Occlusion-Aware Multi-View Stereo

**Traditional MVS Limitation**: Assumes static scene, struggles with occlusion.

**Occlusion-Aware Extension**:
1. Detect occluded regions in each view
2. Weight photometric consistency by visibility
3. Use only views where surface is visible

From [Multi-view Occlusion Reasoning](https://inria.hal.science/inria-00527803/document) (Guan et al.):
- "Multi-view occlusion reasoning for probabilistic silhouette-based reconstruction"
- Cited by 32, foundational work

### 5.3 Self-Supervised De-Occlusion

**Modern Approach**: Learn to predict what multiple views would show from single view.

From [3D De-Occlusion from a Single Image via Self-Supervised Learning](https://arxiv.org/html/2506.21544v1):
- "We propose an end-to-end, occlusion-aware multi-view generation framework that directly predicts six structurally consistent novel views from a single occluded image"

**Pipeline**:
```
Occluded Image → Occlusion-Aware Generator → 6 Novel Views → 3D Fusion
```

**Self-Supervision**: Train using view consistency as supervision signal.

### 5.4 Temporal Multi-View (Video)

**Advantage**: Natural source of multiple views as camera or objects move.

**Challenges**:
- Objects may also move
- Occlusion patterns change
- Need temporal consistency

**Applications**:
- Autonomous driving: Build environment from video
- Robotics: Manipulator viewpoint changes
- AR/VR: User head movement

### 5.5 View Selection for Occlusion Resolution

**Strategy**: Choose views that maximally resolve occlusions.

**Information-Theoretic Approach**:
```python
def select_best_view(current_reconstruction, candidate_views):
    information_gains = []

    for view in candidate_views:
        # Estimate what would become visible
        newly_visible = estimate_visibility(view, current_reconstruction)

        # Weight by current uncertainty
        uncertainty_reduction = compute_uncertainty_reduction(
            current_reconstruction, newly_visible
        )

        information_gains.append(uncertainty_reduction)

    return candidate_views[argmax(information_gains)]
```

---

## Section 6: Evaluation Benchmarks for Occlusion Handling

### 6.1 Synthetic Occlusion Benchmarks

**Advantages**:
- Ground truth complete shape available
- Controlled occlusion patterns
- Easy to generate at scale

**Common Setups**:
1. **Random Occluders**: Place random shapes in front
2. **Realistic Occluders**: Use objects from scene context
3. **Varying Occlusion Levels**: 10%, 30%, 50%, 70%

**Datasets**:
- ShapeNet with synthetic occlusion
- Objaverse with rendered occluders
- Custom benchmark from Amodal3R

### 6.2 Real-World Occlusion Benchmarks

**Challenges**:
- No ground truth for hidden regions
- Difficult to capture complete object

**Strategies**:
1. **Paired Data**: Capture with/without occluder
2. **Multi-View Ground Truth**: Reconstruct from many views
3. **Human Judgment**: Perceptual plausibility studies

### 6.3 Evaluation Metrics

**Geometric Metrics**:
- **Chamfer Distance**: Average distance between reconstructed and GT points
- **F-Score**: Percentage of points within threshold distance
- **IoU**: Intersection over Union of voxelized shapes

**Occlusion-Specific Metrics**:
- **Visible Region Error**: Error only on directly visible surfaces
- **Occluded Region Error**: Error only on hidden surfaces
- **Completion Accuracy**: How well hidden parts match GT

**Perceptual Metrics**:
- **Human Preference**: A/B testing with users
- **Plausibility Score**: Does completion look reasonable?

### 6.4 SA-3DAO Occlusion Evaluation

From [SA-3DAO Evaluation Dataset](../sam-3d/03-sa-3dao-evaluation-dataset.md):
- Includes images with natural occlusions
- Human preference testing with explicit occlusion queries
- "Are occluded parts plausibly reconstructed?"

### 6.5 Benchmark Comparison Results

**State-of-the-Art Performance** (as of 2025):

| Method | Occluded Chamfer ↓ | Visible Chamfer ↓ | Human Pref % |
|--------|-------------------|-------------------|--------------|
| Amodal3R | **0.023** | 0.018 | **78%** |
| 2D Complete + 3D | 0.041 | 0.019 | 45% |
| Standard 3D Recon | 0.089 | **0.016** | 22% |
| Slice3D | 0.031 | 0.020 | 71% |

**Key Observations**:
- End-to-end methods significantly outperform sequential pipelines
- Standard reconstructors fail badly on occluded regions
- Human preference strongly correlates with occlusion handling

---

## Section 7: ARR-COC-0-1 Integration - Occlusion Reasoning for Spatial Understanding

### 7.1 Why Occlusion Reasoning Matters for VLMs

**Current Limitation**: VLMs see 2D images and lack explicit understanding of what's hidden.

**Questions VLMs Struggle With**:
- "What might be behind that box?"
- "Is there space under the table?"
- "Could someone be standing behind the car?"

**ARR-COC Opportunity**: Integrate occlusion-aware 3D to enable spatial reasoning about hidden regions.

### 7.2 Occlusion-Aware Relevance Realization

**Core Concept**: Allocate attention not just to visible features but to reasoning about what's hidden.

**Architecture Extension**:
```python
class OcclusionAwareARRCOC(nn.Module):
    def __init__(self, base_vlm, occlusion_reasoner):
        super().__init__()
        self.vlm = base_vlm
        self.occlusion_reasoner = occlusion_reasoner  # SAM 3D based

    def forward(self, image, question):
        # Standard VLM features
        visible_features = self.vlm.encode_image(image)

        # Occlusion reasoning
        occlusion_analysis = self.occlusion_reasoner(image)
        hidden_features = occlusion_analysis['hidden_region_embeddings']
        uncertainty = occlusion_analysis['uncertainty_map']

        # Question-driven attention over visible + hidden
        question_embedding = self.vlm.encode_text(question)

        # Allocate relevance to both visible and hidden
        relevance_visible = self.relevance_allocator(
            question_embedding, visible_features
        )
        relevance_hidden = self.relevance_allocator(
            question_embedding, hidden_features,
            uncertainty_weight=uncertainty
        )

        # Generate response with occlusion awareness
        response = self.vlm.decode(
            torch.cat([
                relevance_visible * visible_features,
                relevance_hidden * hidden_features
            ])
        )

        return response
```

### 7.3 Uncertainty-Weighted Responses

**Key Innovation**: Weight claims about hidden regions by reconstruction confidence.

**Example**:
- Question: "What's behind the couch?"
- High confidence: "There appears to be empty space behind the couch"
- Low confidence: "The area behind the couch is not clearly visible, but there might be space for storage"

**Implementation**:
```python
def generate_uncertainty_aware_response(hidden_content, uncertainty):
    if uncertainty < 0.3:
        return f"There is {hidden_content}"
    elif uncertainty < 0.6:
        return f"There appears to be {hidden_content}"
    else:
        return f"It's difficult to determine, but there might be {hidden_content}"
```

### 7.4 Occlusion-Specific Queries

**Query Types ARR-COC Should Handle**:

1. **Existence Queries**: "Is there something behind X?"
2. **Space Queries**: "Is there room to place Y behind X?"
3. **Safety Queries**: "Could there be hazards hidden by X?"
4. **Navigation Queries**: "Can I pass behind X?"
5. **Retrieval Queries**: "Where might the Z be hidden?"

### 7.5 Training Strategy for Occlusion Awareness

**Data Requirements**:
- Images with known occlusions
- Questions about hidden regions
- Ground truth answers (from multi-view or synthetic)

**Training Approach**:
```python
# Occlusion-aware training objective
loss = (
    standard_vqa_loss(predicted, ground_truth) +
    occlusion_reasoning_loss(predicted_hidden, gt_hidden) +
    uncertainty_calibration_loss(predicted_uncertainty, actual_error)
)
```

### 7.6 Practical Applications

**Robotics**:
- "Is it safe to reach behind this object?"
- "What grasping points might be hidden?"

**Autonomous Vehicles**:
- "Could a pedestrian be occluded by that car?"
- "Is there space to park behind that vehicle?"

**AR/VR**:
- "Place virtual object in hidden space behind couch"
- "Navigate avatar through partially visible room"

**E-commerce**:
- "What does the back of this product look like?"
- "Is there hidden damage?"

### 7.7 Connection to Perspectival Knowing

From [Training Strategy: Synthetic to Real](../sam-3d/02-training-synthetic-real-alignment.md):
- Occlusion reasoning embodies perspectival knowing
- Different viewpoints reveal different information
- VLM must reason about what OTHER viewpoints would show

**ARR-COC Design Principle**:
The model should not just process what it sees, but actively reason about what exists beyond its current perspective - a core aspect of intelligent spatial understanding.

---

## Sources

**Source Documents**:
- [SA-3DAO Evaluation Dataset](../sam-3d/03-sa-3dao-evaluation-dataset.md) - Evaluation metrics, occlusion scenarios
- [Training Strategy: Synthetic to Real](../sam-3d/02-training-synthetic-real-alignment.md) - Perspectival knowing
- [Multi-View vs Single-Image](../sam-3d/06-multiview-vs-single-image.md) - View-based occlusion resolution

**Key Papers**:
- [Slice3D: Multi-Slice, Occlusion-Revealing, Single View 3D Reconstruction](https://arxiv.org/abs/2312.02221) - Wang et al., CVPR 2024
- [Amodal3R: Amodal 3D Reconstruction from Occluded 2D Images](https://arxiv.org/abs/2503.13439) - Wu et al., 2025
- [Variational Amodal Object Completion](https://papers.nips.cc/paper/2020/file/bacadc62d6e67d7897cef027fa2d416c-Paper.pdf) - Ling et al., NeurIPS 2020, Cited by 59
- [Image Amodal Completion: A Survey](https://www.sciencedirect.com/science/article/abs/pii/S1077314223000413) - Ao et al., 2023, Cited by 33
- [Recovering Occlusion Boundaries from a Single Image](https://dhoiem.cs.illinois.edu/publications/iccv07hoiem.pdf) - Hoiem, UIUC, Cited by 525
- [Explicit Occlusion Reasoning for Multi-person 3D Human Pose Estimation](https://arxiv.org/pdf/2208.00090) - Liu et al., Cited by 43
- [Deep Sliding Shapes for Amodal 3D Object Detection](https://dss.cs.princeton.edu/paper.pdf) - Xiao et al., Cited by 913
- [3D Shape Completion with Multi-view Consistent Inference](https://www.cs.umd.edu/~zwicker/publications/3DShapeCompletionWithConsistentInference-AAAI2020.pdf) - Hu et al., AAAI 2020, Cited by 63
- [Weakly-supervised 3D Shape Completion in the Wild](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500273.pdf) - Gu et al., ECCV 2020, Cited by 71
- [Symmetry-Aware Generative Network for Occluded Shape Completion](https://dl.acm.org/doi/full/10.1145/3760678.3760692) - Li et al., 2025
- [Variational Relational Point Completion Network](https://openaccess.thecvf.com/content/CVPR2021/papers/Pan_Variational_Relational_Point_Completion_Network_CVPR_2021_paper.pdf) - Pan et al., CVPR 2021, Cited by 290

**Project Pages**:
- [Slice3D Project](https://yizhiwang96.github.io/Slice3D/) - Code and demos
- [Amodal3R Project](https://sm0kywu.github.io/Amodal3R/) - Benchmark results

**GitHub Repositories**:
- [Slice3D Code](https://github.com/yizhiwang96/Slice3D)

**Additional References**:
- [CMU Robotics Institute - 3D Shape Completion Dataset](https://www.ri.cmu.edu/app/uploads/2019/05/Thesis_Draft.pdf) - Yuan, 2019
- [3D De-Occlusion via Self-Supervised Learning](https://arxiv.org/html/2506.21544v1) - Multi-view generation

---

**Document Statistics**:
- Total lines: ~700
- Sections: 7 (6 technical + ARR-COC integration)
- ARR-COC coverage: ~10% (Section 7)
- Key citations: 15+ papers with citation counts
- Created: 2025-11-20
