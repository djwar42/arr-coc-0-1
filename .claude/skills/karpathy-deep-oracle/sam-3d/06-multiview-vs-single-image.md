# Multi-View vs Single-Image 3D Reconstruction

**A comprehensive comparison of reconstruction paradigms for understanding 3D from 2D images**

---

## Table of Contents

1. [Overview: The Fundamental Dichotomy](#section-1-overview)
2. [Multi-View Reconstruction Methods](#section-2-multi-view-methods)
3. [Single-Image Reconstruction Methods](#section-3-single-image-methods)
4. [Comparison: Accuracy, Speed, Use Cases](#section-4-comparison)
5. [When to Use Each Approach](#section-5-when-to-use)
6. [Hybrid Methods](#section-6-hybrid-methods)
7. [Current State-of-the-Art](#section-7-sota)
8. [ARR-COC-0-1: Single-Image vs Multi-View for VLM Spatial Understanding](#section-8-arr-coc)

---

## Section 1: Overview - The Fundamental Dichotomy {#section-1-overview}

### The Core Question

How do we recover 3D structure from 2D images? This question has two fundamentally different answers based on the available input:

**Multi-View Reconstruction**: Given multiple images from different viewpoints, triangulate 3D geometry using geometric constraints.

**Single-Image Reconstruction**: Given only one image, infer 3D structure using learned priors about the visual world.

### Why This Distinction Matters

From [A Survey of 3D Reconstruction: The Evolution from Multi-View to Single-View](https://pmc.ncbi.nlm.nih.gov/articles/PMC12473764/) (Liu et al., 2025):

The distinction represents fundamentally different information sources:
- **Multi-view**: Exploits geometric redundancy (same point seen from multiple angles)
- **Single-view**: Exploits statistical redundancy (learned patterns from training data)

### Historical Context

**Traditional Computer Vision (1980s-2010s)**: Multi-view methods dominated
- Structure from Motion (SfM)
- Simultaneous Localization and Mapping (SLAM)
- Multi-View Stereo (MVS)

**Deep Learning Era (2014-present)**: Single-image methods emerged
- Learned depth estimation
- Neural shape generation
- Diffusion-based 3D reconstruction

**Current State (2024-2025)**: Both paradigms have advanced dramatically
- Multi-view: NeRF, 3D Gaussian Splatting
- Single-image: SAM 3D Objects, Zero123, diffusion shortcuts

---

## Section 2: Multi-View Reconstruction Methods {#section-2-multi-view-methods}

### 2.1 Structure from Motion (SfM)

**Definition**: Recover 3D structure and camera poses simultaneously from a sequence of 2D images.

From [Multi-view 3D reconstruction based on deep learning](https://www.sciencedirect.com/science/article/abs/pii/S0925231224003242) (Wu et al., 2024):

**Pipeline**:
1. **Feature Detection**: Find distinctive points (SIFT, ORB, SuperPoint)
2. **Feature Matching**: Establish correspondences across images
3. **Geometric Verification**: Filter matches using epipolar constraints
4. **Bundle Adjustment**: Jointly optimize camera poses and 3D points

**Key Algorithms**:
- **COLMAP**: Gold standard for academic research (cited by 10,000+ papers)
- **OpenMVG**: Modular open-source implementation
- **VGGSfM**: Learning-based SfM combining deep features with classical geometry

**Strengths**:
- Mathematically principled (projective geometry)
- Metric scale recovery (with known baseline or calibration)
- Highly accurate for textured scenes

**Limitations**:
- Fails on textureless surfaces
- Requires sufficient viewpoint diversity
- Computationally expensive for large image collections

### 2.2 SLAM (Simultaneous Localization and Mapping)

**Definition**: Build a map of the environment while simultaneously tracking the camera position in real-time.

**Key Differences from SfM**:
- Real-time operation (30+ FPS)
- Incremental processing (online)
- Loop closure detection

**Modern SLAM Systems**:
- **ORB-SLAM3**: Visual-inertial SLAM with multi-map support
- **DROID-SLAM**: Deep learning for dense SLAM
- **Gaussian Splatting SLAM**: Real-time 3DGS-based SLAM

### 2.3 Neural Radiance Fields (NeRF)

From [NeRF: Neural Radiance Field in 3D Vision](https://arxiv.org/html/2210.00379v6) (arXiv, 2025 update):

**Core Idea**: Represent a scene as a continuous 5D function:
```
F(x, y, z, theta, phi) -> (RGB, density)
```

Where (x, y, z) is 3D position and (theta, phi) is viewing direction.

**Training Process**:
1. Given: Multiple posed images
2. Learn: MLP that predicts color and density for any 3D point
3. Render: Use volumetric ray marching to synthesize novel views

**Advantages**:
- Implicit continuous representation (no discrete voxels)
- View-dependent effects (specular reflections)
- State-of-the-art novel view synthesis

**Limitations**:
- Slow training (hours per scene)
- Slow rendering (seconds per image)
- Requires accurate camera poses

**Key Variants** (from [A Critical Analysis of NeRF-Based 3D Reconstruction](https://www.mdpi.com/2072-4292/15/14/3585), Remondino et al., 2023):
- **Instant-NGP**: Hash encoding for 100x faster training
- **Mip-NeRF**: Anti-aliased rendering for varying distances
- **NeuS**: Improved surface reconstruction via SDF

### 2.4 3D Gaussian Splatting (3DGS)

From [3D Gaussian Splatting vs NeRF: The End Game of 3D Reconstruction](https://pyimagesearch.com/2024/12/09/3d-gaussian-splatting-vs-nerf-the-end-game-of-3d-reconstruction/) (PyImageSearch, 2024):

**Core Idea**: Represent scene as millions of 3D Gaussian primitives, each with:
- Position (x, y, z)
- Covariance (orientation and scale)
- Opacity (alpha)
- Spherical harmonics (view-dependent color)

**Key Innovation**: Explicit representation enables real-time rendering via rasterization.

**Comparison with NeRF** (from [comparative analysis of NeRF, Gaussian Splatting, and SfM](https://isprs-archives.copernicus.org/articles/XLVIII-2-W8-2024/93/2024/), Clini et al., 2024):

| Aspect | NeRF | 3D Gaussian Splatting |
|--------|------|----------------------|
| Representation | Implicit (MLP) | Explicit (Gaussians) |
| Training | Hours | Minutes |
| Rendering | Seconds | Real-time (100+ FPS) |
| Memory | Low | High |
| Editability | Difficult | Easy |

**Why 3DGS is Often Preferred**:
- Real-time rendering enables interactive applications
- Explicit representation enables editing and physics simulation
- Faster optimization for time-sensitive applications

### 2.5 Multi-View Stereo (MVS)

**Definition**: Dense 3D reconstruction from multiple calibrated images.

**Pipeline**:
1. SfM provides sparse 3D points and camera poses
2. MVS densifies the reconstruction
3. Output: Dense point cloud or mesh

**Modern Deep MVS**:
- **MVSNet**: Cost volume with 3D CNN
- **CasMVSNet**: Coarse-to-fine cascade
- **PatchMatch**: Efficient random search for depth

---

## Section 3: Single-Image Reconstruction Methods {#section-3-single-image-methods}

### 3.1 The Ill-Posed Problem

Single-image 3D reconstruction is mathematically ill-posed: infinitely many 3D scenes can project to the same 2D image.

From [Few-Shot Generalization for Single-Image 3D Reconstruction via Priors](https://arxiv.org/abs/1909.01205) (Wallace et al., 2019):

"We reframe single-view 3D reconstruction as learnt, category agnostic refinement of a provided, category-specific prior."

**Solution**: Learn statistical priors from large datasets to constrain the solution space.

### 3.2 Monocular Depth Estimation

**Definition**: Predict per-pixel depth from a single RGB image.

**Evolution**:
1. **Classical methods**: Shape-from-shading, defocus cues
2. **Deep learning**: Encoder-decoder CNNs
3. **Foundation models**: Depth Anything, MiDaS, ZoeDepth

**Depth Anything** (from [Unleashing the Power of Large-Scale Unlabeled Data](https://arxiv.org/abs/2401.10891), Yang et al., 2024):

Key innovations:
- **Training data**: 1.5M labeled + 62M unlabeled images
- **Self-training**: Use teacher model to generate pseudo-labels
- **Strong augmentations**: Color jitter, spatial perturbations
- **Feature alignment**: DINOv2 encoder provides robust features

**Depth Anything V2** achieves:
- Best zero-shot generalization
- Fine-grained detail preservation
- Robust handling of transparent/reflective surfaces

**Depth Anything 3 (DA3)** (announced 2025):
- Unified depth-ray representation
- Multi-view depth estimation
- Camera pose estimation

**Note**: Monocular depth is relative (up to scale and shift), not metric unless trained on specific datasets.

### 3.3 Learned 3D Shape Generation

**Approaches** (from [Single image 3D object reconstruction based on deep learning](https://dl.acm.org/doi/10.1007/s11042-020-09722-8), Fu et al., 2021):

**1. Voxel-based**:
- 3D-R2N2, Pix2Vox
- Regular grid representation
- Limited resolution (memory-intensive)

**2. Point cloud-based**:
- PointOutNet, FoldingNet
- Unstructured set of 3D points
- No topology/surface information

**3. Mesh-based**:
- Pixel2Mesh, AtlasNet
- Vertices, edges, faces
- Topology constraints (genus)

**4. Implicit functions**:
- Occupancy Networks, DeepSDF
- Continuous representation
- Resolution-independent

### 3.4 Diffusion-Based 3D Generation

From [Learning View Priors for Single-View 3D Reconstruction](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kato_Learning_View_Priors_for_Single-View_3D_Reconstruction_CVPR_2019_paper.pdf) (Kato et al., 2019) and later diffusion work:

**Modern Approaches**:
- **Zero-1-to-3**: Condition on single image, generate novel views
- **DreamFusion**: Text-to-3D via score distillation sampling (SDS)
- **Wonder3D**: Multi-view normal and color diffusion

**SAM 3D Objects** (from source document):
Key innovations:
- **Diffusion shortcuts**: Near real-time generation
- **Transformer decoder**: Multi-step refinement
- **Training**: ~1M images, 3.14M meshes

### 3.5 Learning Priors from Data

**Types of Learned Priors** (from [A brief overview of single-view 3D reconstruction based on deep learning](https://frl.nyu.edu/a-brief-overview-of-single-view-3d-reconstruction-based-on-deep-learning/)):

1. **Shape priors**: Expected 3D shapes for object categories
2. **Texture priors**: Typical texture patterns on surfaces
3. **Geometric priors**: Vanishing points, ground plane, gravity
4. **Semantic priors**: Object parts, articulation constraints

**Few-Shot Learning**: Adapt to new categories with minimal examples by leveraging category-agnostic priors.

---

## Section 4: Comparison - Accuracy, Speed, Use Cases {#section-4-comparison}

### 4.1 Accuracy Comparison

From [Recent Developments in Image-Based 3D Reconstruction](https://www.mdpi.com/2079-9292/14/15/3032) (Rodriguez-Lira et al., 2025):

**Multi-View Methods**:
- **Geometric accuracy**: High (millimeter-level for controlled setups)
- **Completeness**: Depends on viewpoint coverage
- **Metric scale**: Recoverable with calibration or known baseline

**Single-Image Methods**:
- **Geometric accuracy**: Moderate (learned statistical average)
- **Completeness**: Always complete (hallucinate unseen regions)
- **Metric scale**: Generally unavailable (relative depth only)

### 4.2 Speed Comparison

| Method | Training | Inference | Real-time? |
|--------|----------|-----------|------------|
| COLMAP SfM | N/A | Minutes-hours | No |
| NeRF | 2-12 hours | 10-60 seconds | No |
| 3D Gaussian Splatting | 5-30 minutes | Real-time | Yes |
| Depth Anything V2 | Pre-trained | ~100ms | Yes |
| SAM 3D Objects | Pre-trained | Near real-time | Almost |

### 4.3 Use Case Comparison

**Multi-View Methods Excel At**:
- Cultural heritage documentation
- Industrial metrology
- Autonomous driving (lidar + camera)
- VR/AR environment capture
- Film and visual effects

**Single-Image Methods Excel At**:
- E-commerce product visualization
- Social media AR filters
- Quick prototyping
- Content creation from reference images
- Accessibility (no special capture)

### 4.4 Quality vs Convenience Trade-off

From [Single view generalizable 3D reconstruction based on 3D priors](https://www.nature.com/articles/s41598-025-03200-7) (Fang et al., 2025):

"Monocular 3D reconstruction refers to the process of constructing a three-dimensional scene from a single image, enabling the synthesis of new views."

**Trade-off Spectrum**:
```
High Accuracy                              High Convenience
    |                                           |
    v                                           v
Multi-view    Multi-view     Single-image    Single-image
(controlled)   (casual)       (domain)        (general)
```

---

## Section 5: When to Use Each Approach {#section-5-when-to-use}

### 5.1 Use Multi-View When

**1. Accuracy is Critical**:
- Medical imaging
- Engineering inspection
- Scientific measurement

**2. Multiple Views are Available**:
- Video capture
- Multi-camera rigs
- Robot exploration

**3. Scene is Static**:
- Architecture
- Products
- Landscapes

**4. Novel View Synthesis is Needed**:
- Virtual tours
- VR experiences
- Video stabilization

### 5.2 Use Single-Image When

**1. Only One Image Exists**:
- Historical photos
- User-uploaded images
- Artwork/illustrations

**2. Speed is Priority**:
- Real-time applications
- Interactive tools
- Mobile deployment

**3. Convenience Matters**:
- Consumer applications
- Casual users
- Quick iterations

**4. Complete Output Needed**:
- Must see back of objects
- Scene generation
- Creative applications

### 5.3 Decision Framework

```
Q1: Do you have multiple images?
    |
    +-- No --> Single-image method
    |
    +-- Yes --> Q2: Is real-time needed?
                    |
                    +-- Yes --> 3D Gaussian Splatting
                    |
                    +-- No --> Q3: Need view-dependent effects?
                                    |
                                    +-- Yes --> NeRF
                                    |
                                    +-- No --> SfM + MVS
```

### 5.4 Application-Specific Recommendations

**E-Commerce**:
- Primary: Single-image (SAM 3D Objects)
- Reason: One product photo available, need fast turnaround

**Robotics**:
- Primary: Multi-view (SLAM, NeRF)
- Augment: Single-image depth for quick perception
- Reason: Need metric accuracy for manipulation

**Gaming/VFX**:
- Primary: Multi-view (3DGS, photogrammetry)
- Reason: Need high-quality assets, time for capture

**Social Media**:
- Primary: Single-image
- Reason: User convenience, instant results

---

## Section 6: Hybrid Methods {#section-6-hybrid-methods}

### 6.1 Why Hybrid?

From [Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](https://arxiv.org/html/2503.14483v1) (Guo et al., 2025):

"We develop a new pipeline, named Murre, for multi-view geometry reconstruction of 3D scenes based on SfM-guided monocular depth estimation."

**Motivation**: Combine strengths of both paradigms
- Multi-view: Geometric accuracy, consistency
- Single-image: Speed, completeness, dense estimation

### 6.2 Murre: SfM-Guided Monocular Depth

From [Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_Multi-view_Reconstruction_via_SfM-guided_Monocular_Depth_Estimation_CVPR_2025_paper.pdf) (CVPR 2025):

**Architecture**:
1. Run SfM to get sparse 3D points
2. Use sparse points to guide monocular depth network
3. Achieve dense, metrically accurate depth

**Benefits**:
- Dense depth where SfM would be sparse
- Accurate scale from multi-view geometry
- Robust to textureless regions

### 6.3 SfM-TTR: Test-Time Refinement

From [SfM-TTR: Using Structure From Motion for Test-Time Refinement of Single-View Depth](https://openaccess.thecvf.com/content/CVPR2023/papers/Izquierdo_SfM-TTR_Using_Structure_From_Motion_for_Test-Time_Refinement_of_Single-View_CVPR_2023_paper.pdf) (Izquierdo et al., CVPR 2023):

**Approach**:
1. Get initial depth from monocular network
2. Run SfM on video sequence
3. Align and refine monocular depth with SfM

**Key Insight**: Self-supervised refinement without ground truth depth.

### 6.4 Monocular Surface Priors for SfM

From [Monocular Surface Priors for Robust Structure-from-Motion](https://demuc.de/papers/pataki2025mpsfm.pdf) (Pataki et al., 2025):

"While SfM can generally estimate 3D only for points observed in multiple views, we leverage single-view observations with depth."

**Innovation**: Use monocular depth to:
- Initialize 3D point depths
- Regularize bundle adjustment
- Fill in sparse regions

### 6.5 Diffusion + Multi-View Consistency

**Modern Hybrid Approaches**:
- Generate multi-view images from single image
- Ensure consistency across views
- Run NeRF/3DGS on generated views

**Examples**:
- **Zero123++**: Multi-view consistent image generation
- **SyncDreamer**: Synchronized multi-view diffusion
- **ImageDream**: Multi-view images for 3D generation

### 6.6 Deep Two-View SfM

From [Deep Two-View Structure-from-Motion Revisited](https://arxiv.org/abs/2104.00556) (Wang et al., 2021):

"Our method outperforms all state-of-the-art two-view SfM methods by a clear margin on KITTI depth."

**Approach**: Replace traditional SfM components with learned ones:
- Feature matching: Deep correspondence network
- Essential matrix: Direct regression
- Depth estimation: Joint optimization

---

## Section 7: Current State-of-the-Art (2024-2025) {#section-7-sota}

### 7.1 Multi-View SOTA

**Scene Reconstruction**:
- **3D Gaussian Splatting** (Kerbl et al., 2023): Real-time novel view synthesis
- **Nerfstudio** (Tancik et al., 2023): Modular NeRF framework
- **DUSt3R** (Wang et al., 2024): Dense stereo from any images

From [Comparative Assessment of Neural Radiance Fields and 3D Gaussian Splatting](https://www.mdpi.com/1424-8220/25/10/2995) (Atik et al., 2025):

"This paper explores the performance of NeRF and 3DGS methods in generating point clouds from UAV images."

**Key Finding**: 3DGS outperforms NeRF in computational efficiency while maintaining comparable quality.

**Human Reconstruction**:
- **SMPL-X**: Full-body parametric model
- **NeuMan**: Neural human radiance fields
- **3D Gaussian Splatting for Humans**: Real-time human rendering

### 7.2 Single-Image SOTA

**Depth Estimation**:
- **Depth Anything V2** (2024): Foundation model for monocular depth
- **Metric3D** (2023): Metric depth estimation
- **UniDepth** (2024): Universal monocular depth

**Object Reconstruction**:
- **SAM 3D Objects** (Meta, 2025): Near real-time 3D from single image
- **One-2-3-45** (2023): Single-image to 3D in 45 seconds
- **Wonder3D** (2023): Multi-view diffusion for 3D

**Human Reconstruction**:
- **SAM 3D Body** (Meta, 2025): Promptable human mesh recovery
- **HMR 2.0** (2023): State-of-the-art HMR
- **4D-Humans** (2023): Video-based human reconstruction

### 7.3 Emerging Trends

**1. Foundation Models for 3D**:
- Large-scale pre-training
- Zero-shot generalization
- Multi-task learning (depth, normals, segmentation)

**2. Feed-Forward 3D Generation**:
- No per-scene optimization
- Single forward pass
- Real-time inference

**3. Unified Representations**:
- Combining explicit (mesh) and implicit (NeRF)
- Differentiable rendering everywhere
- End-to-end learning

**4. Video Understanding**:
- Temporal consistency
- Dynamic scenes
- 4D reconstruction

### 7.4 Benchmarks and Evaluation

**Common Benchmarks**:
- **ScanNet**: Indoor RGB-D scenes
- **NYUv2**: Indoor depth estimation
- **KITTI**: Outdoor driving
- **ShapeNet**: 3D object models
- **CO3D**: Common Objects in 3D

**Evaluation Metrics**:
- **Depth**: AbsRel, SqRel, RMSE, delta < 1.25
- **3D**: Chamfer Distance, F-Score
- **Novel Views**: PSNR, SSIM, LPIPS

---

## Section 8: ARR-COC-0-1 - Single-Image vs Multi-View for VLM Spatial Understanding {#section-8-arr-coc}

### 8.1 Why This Matters for ARR-COC

ARR-COC-0-1 aims to build Vision-Language Models (VLMs) with genuine spatial understanding. The choice between single-image and multi-view 3D reconstruction directly impacts:

1. **Spatial relevance realization**: How the model allocates attention to spatial relationships
2. **Perspectival knowing**: Understanding that objects exist in 3D space
3. **Participatory knowledge**: Grounding language in physical reality

### 8.2 Single-Image 3D for VLM Perception

**Advantages for ARR-COC**:

**1. Real-Time Spatial Understanding**:
- User provides one image, expects immediate response
- No time for multi-view capture
- Depth Anything V2 provides instant relative depth

**2. Learned Priors as Common Sense**:
- Networks learn "chairs have 4 legs" from data
- This matches human commonsense reasoning
- VLM can leverage same priors for language grounding

**3. Scalability**:
- Internet images are mostly single-view
- Training data is abundant
- No need for expensive multi-view datasets

**Implementation Strategy**:
```python
# ARR-COC spatial perception module
def get_spatial_features(image):
    # 1. Depth estimation (relative)
    depth = depth_anything_v2(image)

    # 2. Surface normals (geometry)
    normals = normal_predictor(image)

    # 3. Segmentation (objects)
    masks = sam_2(image)

    # 4. Fuse into spatial tokens
    spatial_tokens = fuse_3d_features(depth, normals, masks)

    return spatial_tokens
```

### 8.3 Multi-View 3D for Embodied Applications

**When ARR-COC Needs Multi-View**:

**1. Robotics Tasks**:
- Metric accuracy for manipulation
- Multiple camera views available
- Need physical interaction reasoning

**2. Navigation**:
- Building spatial maps
- Loop closure and relocalization
- Metric scale essential

**3. Scene Understanding**:
- Complete 3D scene graphs
- Object relationships in space
- View-independent representations

**Implementation Strategy**:
```python
# ARR-COC embodied perception module
def get_scene_representation(video_frames, camera_poses):
    # 1. Run 3D Gaussian Splatting
    scene_gaussians = train_3dgs(video_frames, camera_poses)

    # 2. Extract semantic features
    semantic_gaussians = add_semantics(scene_gaussians, sam_2)

    # 3. Build scene graph
    scene_graph = extract_relationships(semantic_gaussians)

    return scene_graph
```

### 8.4 Hybrid Approach for ARR-COC

**Recommended Architecture**:

**Layer 1: Single-Image Foundation**
- Always run: Depth Anything V2, SAM 2, DINO features
- Provides immediate spatial understanding
- Works for any image input

**Layer 2: Multi-View Enhancement**
- If video available: SfM for metric scale
- If multiple views: 3DGS for consistent representation
- Optional but higher quality

**Layer 3: Spatial Language Grounding**
- Map 3D features to language tokens
- Enable spatial queries ("what's behind the chair?")
- Support spatial instructions ("move left of the table")

### 8.5 Trade-offs for VLM Design

**Single-Image Priority** (recommended for ARR-COC v1):
- Pros: Fast, scalable, works with any image
- Cons: Relative depth, hallucinated occluded regions
- Use case: General image understanding

**Multi-View Priority** (for ARR-COC v2+):
- Pros: Metric accuracy, consistent geometry
- Cons: Requires video/multi-view input, slower
- Use case: Embodied AI, robotics

### 8.6 3D Token Design for VLMs

**Key Question**: How to represent 3D information for transformer attention?

**Option 1: Depth-Aware Image Tokens**
- Augment 2D patches with depth value
- Simple, minimal architecture change
- Limited 3D reasoning

**Option 2: Point Cloud Tokens**
- Represent scene as 3D points with features
- PointBERT-style encoding
- Rich 3D structure

**Option 3: Triplane Tokens**
- Compress 3D into three orthogonal planes
- Balance efficiency and expressiveness
- Used in SAM 3D Objects

**ARR-COC Recommendation**:
- Start with depth-aware tokens (Option 1)
- Evolve to triplane (Option 3) for complex scenes
- Reserve point clouds (Option 2) for embodied tasks

### 8.7 Spatial Query Types

**Enabled by Single-Image 3D**:
- "What's in the foreground?"
- "Which object is closer?"
- "Describe the depth layout"

**Enabled by Multi-View 3D**:
- "What's the distance between X and Y?"
- "Navigate around the obstacle"
- "How would this look from above?"

**ARR-COC Goal**: Support both through adaptive 3D representation.

### 8.8 Future Directions

**1. Unified 3D Foundation Model**:
- Single model for depth, normals, reconstruction
- Multi-task pre-training
- Zero-shot 3D understanding

**2. Language-Conditioned 3D Generation**:
- Text-to-3D for VLM imagination
- Spatial grounding for language models
- 3D reasoning chains

**3. Video Temporal Consistency**:
- Track objects in 3D across frames
- Build 4D representations
- Enable video spatial QA

---

## Summary

### Key Takeaways

1. **Multi-view** excels at geometric accuracy but requires multiple images
2. **Single-image** provides convenience and speed but relies on learned priors
3. **Hybrid methods** combine strengths of both paradigms
4. **3D Gaussian Splatting** is emerging as preferred multi-view method
5. **Depth Anything** represents SOTA for single-image depth
6. **SAM 3D Objects** achieves near real-time single-image 3D

### For ARR-COC-0-1

- **Start with single-image**: Works with any input, fast, scalable
- **Enhance with multi-view**: When video/multiple images available
- **Design 3D tokens carefully**: Bridge between 3D and language
- **Support spatial queries**: Enable true 3D understanding in VLMs

---

## Sources

**Source Documents**:
- SAM_STUDY_3D.md - Core SAM 3D Objects and Body documentation

**Web Research (accessed 2025-11-20)**:

**Survey Papers**:
- [A Survey of 3D Reconstruction: The Evolution from Multi-View to Single-View](https://pmc.ncbi.nlm.nih.gov/articles/PMC12473764/) - Liu et al., 2025
- [Multi-view 3D reconstruction based on deep learning](https://www.sciencedirect.com/science/article/abs/pii/S0925231224003242) - Wu et al., 2024
- [Recent Developments in Image-Based 3D Reconstruction](https://www.mdpi.com/2079-9292/14/15/3032) - Rodriguez-Lira et al., 2025

**Multi-View Methods**:
- [A Critical Analysis of NeRF-Based 3D Reconstruction](https://www.mdpi.com/2072-4292/15/14/3585) - Remondino et al., 2023
- [3D Gaussian Splatting vs NeRF: The End Game](https://pyimagesearch.com/2024/12/09/3d-gaussian-splatting-vs-nerf-the-end-game-of-3d-reconstruction/) - PyImageSearch, 2024
- [Comparative analysis of NeRF, Gaussian Splatting, and SfM](https://isprs-archives.copernicus.org/articles/XLVIII-2-W8-2024/93/2024/) - Clini et al., 2024

**Single-Image Methods**:
- [Few-Shot Generalization for Single-Image 3D Reconstruction via Priors](https://arxiv.org/abs/1909.01205) - Wallace et al., 2019
- [Learning View Priors for Single-View 3D Reconstruction](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kato_Learning_View_Priors_for_Single-View_3D_Reconstruction_CVPR_2019_paper.pdf) - Kato et al., 2019
- [Single image 3D object reconstruction based on deep learning](https://dl.acm.org/doi/10.1007/s11042-020-09722-8) - Fu et al., 2021

**Depth Estimation**:
- [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](https://arxiv.org/abs/2401.10891) - Yang et al., 2024
- [Depth Anything Official Page](https://depth-anything.github.io/)
- [Depth Anything V2 GitHub](https://github.com/DepthAnything/Depth-Anything-V2)

**Hybrid Methods**:
- [Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](https://arxiv.org/html/2503.14483v1) - Guo et al., CVPR 2025
- [SfM-TTR: Using Structure From Motion for Test-Time Refinement](https://openaccess.thecvf.com/content/CVPR2023/papers/Izquierdo_SfM-TTR_Using_Structure_From_Motion_for_Test-Time_Refinement_of_Single-View_CVPR_2023_paper.pdf) - Izquierdo et al., CVPR 2023
- [Deep Two-View Structure-from-Motion Revisited](https://arxiv.org/abs/2104.00556) - Wang et al., 2021

**NeRF and 3DGS**:
- [NeRF: Neural Radiance Field in 3D Vision](https://arxiv.org/html/2210.00379v6) - arXiv survey
- [Matt Tancik's NeRF Page](https://www.matthewtancik.com/nerf)
- [Comparative Assessment of NeRF and 3DGS](https://www.mdpi.com/1424-8220/25/10/2995) - Atik et al., 2025

**Additional References**:
- [Single view generalizable 3D reconstruction based on 3D priors](https://www.nature.com/articles/s41598-025-03200-7) - Fang et al., 2025
- [Brief overview of single-view 3D reconstruction](https://frl.nyu.edu/a-brief-overview-of-single-view-3d-reconstruction-based-on-deep-learning/) - NYU Future Reality Lab
- [Monocular Surface Priors for Robust SfM](https://demuc.de/papers/pataki2025mpsfm.pdf) - Pataki et al., 2025
