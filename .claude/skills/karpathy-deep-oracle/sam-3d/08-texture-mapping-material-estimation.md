# Texture Mapping & Material Estimation for 3D Reconstruction

**Category**: 3D Reconstruction Architecture
**Audience**: ML engineers, 3D graphics researchers, VLM developers
**Prerequisites**: Basic understanding of 3D graphics, deep learning fundamentals
**Last Updated**: 2025-11-20

---

## Table of Contents

1. [Texture Mapping Fundamentals](#section-1-texture-mapping-fundamentals)
2. [Single-Image Texture Extraction Challenges](#section-2-single-image-texture-extraction-challenges)
3. [PBR Material Estimation](#section-3-pbr-material-estimation)
4. [Neural UV Unwrapping Methods](#section-4-neural-uv-unwrapping-methods)
5. [Texture Completion for Occluded Regions](#section-5-texture-completion-for-occluded-regions)
6. [Quality Metrics and Evaluation](#section-6-quality-metrics-and-evaluation)
7. [ARR-COC-0-1: Texture Features for Visual Grounding](#section-7-arr-coc-0-1-texture-features-for-visual-grounding)

---

## Section 1: Texture Mapping Fundamentals

### 1.1 Overview of Texture Mapping

Texture mapping is the process of applying 2D images (textures) to 3D geometry surfaces, creating the visual appearance of detail without increasing geometric complexity. This technique is fundamental to modern 3D graphics and has become a critical component of learning-based 3D reconstruction systems.

From [SAM 3D Study](../source-documents/SAM_STUDY_3D.md):
- SAM 3D Objects produces **textured outputs** from single images
- Achieves **state-of-the-art 3D mesh generation** with detailed surface appearance

### 1.2 UV Coordinate Systems

**UV Coordinates** are the standard parameterization for mapping 2D textures onto 3D surfaces:

```
UV Space: [0, 1] x [0, 1] 2D domain
    U = horizontal axis
    V = vertical axis

3D Surface → UV Mapping → 2D Texture
    (x, y, z) → (u, v) → RGB/Material values
```

**Key Properties:**
- **Bijective mapping**: Each 3D surface point maps to unique (u, v) coordinate
- **Continuity preservation**: Adjacent surface regions remain adjacent in UV space
- **Distortion minimization**: Area and angle preservation in unwrapping

From [ArtUV: Artist-style UV Unwrapping](https://arxiv.org/html/2509.20710v1) (arXiv:2509.20710, accessed 2025-11-20):
- Professional UV mapping divides mesh into logical charts
- Seam placement affects texture quality and visibility
- End-to-end neural methods now achieve artist-quality results

### 1.3 Texture Representation Types

**1. Explicit Textures:**
```python
# Standard 2D texture maps
albedo_map: Tensor[H, W, 3]    # Base color/diffuse
normal_map: Tensor[H, W, 3]    # Surface normals
roughness_map: Tensor[H, W, 1] # Surface roughness
metallic_map: Tensor[H, W, 1]  # Metallic property
```

**2. Neural Textures:**
```python
# Learnable feature maps
neural_texture: Tensor[H, W, C]  # C >> 3 channels
# Decoded via neural network during rendering
```

From [Deferred Neural Rendering](https://niessnerlab.org/papers/2019/11neuralrendering/) (Thies et al., cited by 1647):
- Neural textures store learned features instead of explicit colors
- Network decodes features into final appearance
- Enables view-dependent effects and lighting adaptation

**3. Implicit Neural Representations:**
```python
# Continuous coordinate-based functions
def texture_field(uv: Tensor[N, 2]) -> Tensor[N, 3]:
    # MLP maps (u, v) to RGB
    return mlp(uv)
```

### 1.4 Projection Methods

**Planar Projection:**
- Projects texture along single axis
- Best for flat surfaces
- Simple but causes distortion on curved surfaces

**Cylindrical/Spherical Projection:**
- Wraps texture around axis
- Good for objects with rotational symmetry
- Common for human body texturing

**UV Unwrapping:**
- Cuts mesh into flat pieces (charts)
- Minimizes distortion
- Required for complex geometry

**Tri-planar Projection:**
```python
def triplanar_sample(pos: Tensor[N, 3], normal: Tensor[N, 3],
                     texture: Tensor[H, W, C]) -> Tensor[N, C]:
    # Blend projections based on normal direction
    weights = abs(normal) ** blend_sharpness
    weights = weights / weights.sum(dim=-1, keepdim=True)

    xy_sample = sample(texture, pos[..., :2])  # Z projection
    xz_sample = sample(texture, pos[..., [0, 2]])  # Y projection
    yz_sample = sample(texture, pos[..., 1:])  # X projection

    return (weights[..., 2:3] * xy_sample +
            weights[..., 1:2] * xz_sample +
            weights[..., 0:1] * yz_sample)
```

---

## Section 2: Single-Image Texture Extraction Challenges

### 2.1 The Ill-Posed Nature of Single-View Texture

Extracting complete textures from a single image is fundamentally **ill-posed**:

**Observable Challenges:**
1. **Partial visibility**: Only front-facing surfaces visible
2. **Self-occlusion**: Object parts hidden by other parts
3. **Lighting entanglement**: Observed color = albedo * lighting
4. **Scale ambiguity**: Texture resolution unknown

From [3D Reconstruction from Single RGB Image](https://av.dfki.de/publications/3d-reconstruction-from-a-single-rgb-image-using-deep-learning-a-review/) (DFKI Review):
- Single-view 3D is a challenging ill-posed optimization problem
- Recent deep learning methods leverage learned priors
- Reviews depth maps, surface normals, point clouds, and meshes

### 2.2 Lighting and Shadow Removal

**The Albedo Disentanglement Problem:**
```
Observed_Color = Albedo * Shading * Specular + Ambient
```

To recover true material appearance, systems must:
1. Estimate illumination environment
2. Decompose shading from albedo
3. Remove specular highlights
4. Handle ambient occlusion

From [SuperMat](https://arxiv.org/html/2411.17515v1) (arXiv:2411.17515, accessed 2025-11-20):
- Single-step framework for material decomposition
- Achieves high-quality albedo with shadow removal
- End-to-end training with re-render loss improves results
- "SuperMat demonstrates superior ability in removing highlights and shadows, accurately capturing the intrinsic color"

**Learning-Based Decomposition:**
```python
class MaterialDecomposer(nn.Module):
    def forward(self, image: Tensor[B, 3, H, W]) -> Dict[str, Tensor]:
        features = self.encoder(image)

        # Parallel expert branches for different materials
        albedo = self.albedo_branch(features)    # [B, 3, H, W]
        roughness = self.roughness_branch(features)  # [B, 1, H, W]
        metallic = self.metallic_branch(features)    # [B, 1, H, W]

        return {'albedo': albedo, 'roughness': roughness,
                'metallic': metallic}
```

### 2.3 Viewpoint-Dependent Effects

**View-Dependent Phenomena:**
- Specular highlights change with viewing angle
- Fresnel effects at grazing angles
- Subsurface scattering in translucent materials
- Anisotropic reflections

**Handling in Single-View:**
```python
# Model view-dependent effects explicitly
def view_conditioned_texture(uv, view_dir, normal):
    base_color = sample_albedo(uv)

    # View-dependent correction
    fresnel = fresnel_schlick(dot(view_dir, normal))
    specular_tint = predict_specular(uv, view_dir)

    return base_color * (1 - fresnel) + specular_tint * fresnel
```

### 2.4 Resolution and Detail Recovery

**Super-Resolution for Textures:**
- Input images limited by camera resolution
- Need high-res textures for close-up rendering
- Learn to hallucinate plausible details

From [Ultra-high Resolution Facial Texture Reconstruction](https://www.sciopen.com/article/10.26599/CVM.2025.9450488) (SciOpen, 2025):
- UV-IDM maps facial regions to UV space
- Latent diffusion model completes texture
- Achieves ultra-high resolution from limited input

**Detail Synthesis Pipeline:**
```python
def enhance_texture(low_res_texture: Tensor,
                    reference_patches: List[Tensor]) -> Tensor:
    # 1. Upsample base texture
    upsampled = bilinear_upsample(low_res_texture, scale=4)

    # 2. Match similar patches from reference
    matched_details = patch_matching(upsampled, reference_patches)

    # 3. Blend details
    enhanced = blend_with_poisson(upsampled, matched_details)

    return enhanced
```

---

## Section 3: PBR Material Estimation

### 3.1 Physically-Based Rendering Materials

**PBR Material Model Components:**

From [SuperMat](https://arxiv.org/html/2411.17515v1):
The Cook-Torrance BRDF defines how light interacts with surfaces:

```
L(p, omega) = albedo * (1 - metallic) * integral[diffuse_irradiance]
            + integral[DFG / (4 * dot(omega, n) * dot(omega_i, n)) * L_i]
```

Where:
- **D** = Normal Distribution Function (roughness-dependent)
- **F** = Fresnel term (metallic-dependent)
- **G** = Geometry/shadowing term

**Standard PBR Maps:**

| Map | Channels | Range | Description |
|-----|----------|-------|-------------|
| Albedo | RGB | [0, 1] | Base color without lighting |
| Metallic | 1 | [0, 1] | 0=dielectric, 1=metal |
| Roughness | 1 | [0, 1] | 0=mirror, 1=matte |
| Normal | RGB | [-1, 1] | Surface orientation |
| AO | 1 | [0, 1] | Ambient occlusion |

### 3.2 Single-Image Material Estimation Methods

**Evolution of Approaches:**

**1. Optimization-Based Methods:**
From [SuperMat Related Works](https://arxiv.org/html/2411.17515v1):
- Use differentiable rendering
- Minimize difference between rendered and target images
- Often produce blurred boundaries and physically implausible parameters

**2. Data-Driven Methods:**
- Leverage large diffusion models
- Reframe as conditional image generation
- Recent advances: RGB-X, IntrinsicAnything, StableMaterial

**3. Single-Step Diffusion (State-of-the-Art):**
From [SuperMat](https://arxiv.org/html/2411.17515v1):
```python
class SuperMat(nn.Module):
    """Single-step material decomposition model."""

    def __init__(self):
        self.shared_encoder = UNetEncoder()  # Shared backbone
        # Structural expert branches
        self.albedo_branch = ExpertBranch()  # Last UpBlock + conv_out
        self.rm_branch = ExpertBranch()      # Roughness + Metallic

    def forward(self, image, noise=None):
        # Single-step inference (vs 30-50 steps for regular diffusion)
        features = self.shared_encoder(image)

        albedo = self.albedo_branch(features)
        rm = self.rm_branch(features)

        return albedo, rm
```

**Key Innovation - End-to-End Training with Re-Render Loss:**
```python
def re_render_loss(pred_materials, gt_materials, geometry, lighting):
    """Compute loss by rendering with predicted materials."""

    pred_render = pbr_render(
        albedo=pred_materials['albedo'],
        metallic=pred_materials['metallic'],
        roughness=pred_materials['roughness'],
        normal=geometry['normal'],
        position=geometry['position'],
        camera=geometry['camera'],
        environment_map=lighting
    )

    gt_render = pbr_render(
        albedo=gt_materials['albedo'],
        metallic=gt_materials['metallic'],
        roughness=gt_materials['roughness'],
        normal=geometry['normal'],
        position=geometry['position'],
        camera=geometry['camera'],
        environment_map=lighting
    )

    return perceptual_loss(pred_render, gt_render)
```

### 3.3 Performance Benchmarks

From [SuperMat Experiments](https://arxiv.org/html/2411.17515v1):

**Albedo Estimation:**
| Method | PSNR | SSIM | LPIPS | Time(s) |
|--------|------|------|-------|---------|
| Derender3D | 22.26 | 0.872 | 0.162 | 0.16 |
| IID | 22.87 | 0.888 | 0.130 | 1.45 |
| RGB-X | 20.12 | 0.884 | 0.141 | 3.32 |
| StableMaterial | 23.90 | 0.905 | 0.103 | 0.53 |
| **SuperMat** | **27.50** | **0.924** | **0.085** | **0.07** |

**Key Achievement**: SuperMat achieves SOTA quality at **millisecond speeds** (0.07s vs seconds for diffusion methods).

### 3.4 Handling Metallic and Roughness

**Challenge**: These properties are particularly difficult to estimate:
- No direct visual correspondence (unlike albedo)
- Highly dependent on viewing angle and lighting
- Often show more variation than albedo

From [SuperMat Results](https://arxiv.org/html/2411.17515v1):
- "SuperMat outperforms the others with well-defined boundaries between metallic and non-metallic regions"
- "Achieves consistency within material types while preserving high-frequency details"
- "Results closely align with physical realism"

**Material Property Ranges:**
```python
def validate_pbr_materials(materials: Dict[str, Tensor]) -> bool:
    """Ensure physically plausible material values."""

    albedo = materials['albedo']
    roughness = materials['roughness']
    metallic = materials['metallic']

    # Energy conservation: metals have low diffuse
    metal_mask = metallic > 0.5
    metal_albedo = albedo[metal_mask]

    # Metallic surfaces should have high luminance albedo
    # (represents specular color, not diffuse)
    if metal_albedo.mean() < 0.5:
        return False  # Implausible

    # Non-metals should have reasonable diffuse
    dielectric_albedo = albedo[~metal_mask]
    if dielectric_albedo.max() > 0.95:
        return False  # Conservation violation

    return True
```

---

## Section 4: Neural UV Unwrapping Methods

### 4.1 Traditional vs Neural UV Unwrapping

**Traditional Methods:**
- Least Squares Conformal Maps (LSCM)
- Angle-Based Flattening (ABF)
- Requires manual seam placement
- Optimizes for distortion minimization

From [ResearchGate: LSCM for Automatic Texture Atlas](https://www.researchgate.net/publication/220720636):
- UV mapping is bijective between 3D surface and 2D texture domain
- Generated via UV unwrapping task
- Quality depends on chart segmentation and parameterization

**Neural UV Methods:**

From [ArtUV: Artist-style UV Unwrapping](https://arxiv.org/html/2509.20710v1):
- Fully automated, end-to-end UV unwrapping
- Simulates professional UV mapping process
- Divides mesh into logical charts automatically

From [Nuvo: Neural UV Mapping](https://dl.acm.org/doi/10.1007/978-3-031-72933-1_2) (ECCV 2024, cited by 20):
- Designed for geometry from 3D reconstruction/generation
- Handles "unruly" representations (noisy, incomplete)
- Learns UV mapping jointly with texture

### 4.2 Automated Chart Segmentation

**Neural Seam Prediction:**

From [SeamCrafter](https://arxiv.org/html/2509.20725v1):
- "Mesh seams play pivotal role in partitioning 3D surfaces for UV parametrization"
- "Poorly placed seams often result in visible artifacts"
- Neural network predicts optimal seam placement

```python
class SeamPredictor(nn.Module):
    """Predict optimal UV seams for mesh unwrapping."""

    def __init__(self):
        self.edge_encoder = EdgeFeatureEncoder()
        self.gnn = GraphNeuralNetwork()
        self.seam_classifier = MLP(hidden_dim, 1)

    def forward(self, mesh: MeshData) -> Tensor:
        # Encode edge features
        edge_features = self.edge_encoder(mesh.edges)

        # Message passing on mesh graph
        node_features = self.gnn(mesh, edge_features)

        # Classify each edge as seam or not
        seam_probs = self.seam_classifier(edge_features)

        return seam_probs
```

### 4.3 Distortion-Aware Learning

**Objectives for Neural UV:**
1. **Angle preservation** (conformal)
2. **Area preservation** (equiareal)
3. **Boundary straightness**
4. **Chart connectivity**

```python
def uv_distortion_loss(mesh_3d: Tensor, mesh_uv: Tensor,
                       faces: Tensor) -> Tensor:
    """Compute UV mapping distortion metrics."""

    # Get triangle vertices
    v3d = mesh_3d[faces]  # [F, 3, 3]
    v2d = mesh_uv[faces]  # [F, 3, 2]

    # Compute Jacobian of mapping
    J = compute_jacobian(v3d, v2d)  # [F, 2, 2]

    # Singular values measure distortion
    s1, s2 = svd_2x2(J)

    # Conformal distortion: s1/s2 should be 1
    conformal_loss = ((s1 - s2) / (s1 + s2 + eps)).pow(2).mean()

    # Area distortion: s1*s2 should be constant
    area_3d = triangle_area_3d(v3d)
    area_2d = triangle_area_2d(v2d)
    area_ratio = area_2d / (area_3d + eps)
    area_loss = (area_ratio - area_ratio.mean()).pow(2).mean()

    return conformal_loss + area_loss
```

### 4.4 Atlas Packing and Optimization

**Chart Packing Problem:**
- Fit all UV charts into [0, 1]^2 texture space
- Maximize texture utilization
- Maintain gutter space between charts

From [NeuTex: Neural Texture Mapping](https://openaccess.thecvf.com/content/CVPR2021/papers/Xiang_NeuTex_Neural_Texture_Mapping_for_Volumetric_Neural_Rendering_CVPR_2021_paper.pdf) (CVPR 2021, cited by 129):
- "NeuTex represents geometry as 3D volume but appearance as 2D neural texture"
- "Automatically discovered texture UV mapping"
- Jointly learns optimal UV parameterization

**Neural Atlas Optimization:**
```python
class NeuralAtlas(nn.Module):
    """Learn optimal UV atlas jointly with texture."""

    def __init__(self):
        self.uv_network = UVPredictionNetwork()
        self.texture_field = NeuralTextureField()

    def forward(self, points_3d: Tensor, normals: Tensor) -> Tensor:
        # Predict UV coordinates
        uv = self.uv_network(points_3d, normals)

        # Sample neural texture
        colors = self.texture_field(uv)

        return colors, uv

    def loss(self, rendered, target, uv):
        render_loss = (rendered - target).pow(2).mean()

        # Regularize UV mapping
        uv_loss = self.compute_uv_regularization(uv)

        return render_loss + 0.1 * uv_loss
```

---

## Section 5: Texture Completion for Occluded Regions

### 5.1 The Inpainting Challenge

**Problem Statement:**
- Single view only sees partial surface
- Need to hallucinate plausible appearance for hidden regions
- Must maintain consistency with visible parts

From [AI-based Large Scale 3D Terrain Completion](https://ict.usc.edu/research/projects/ai-based-large-scale-3d-terrain-completion/) (USC ICT):
- "Using inpainted depth and texture, produces completed mesh"
- "Integrates smoothly with original 3D reconstructed models"

### 5.2 Diffusion-Based Completion

**UV Space Inpainting:**

From [SuperMat UV Refinement](https://arxiv.org/html/2411.17515v1):
```python
class UVRefinementNetwork(nn.Module):
    """Complete and refine partial UV textures."""

    def __init__(self):
        # Expanded input channels for conditions
        self.conv_in = nn.Conv2d(8, 320, 3, 1, 1)  # RGB + mask + position
        self.unet = UNet()  # Stable Diffusion backbone

    def forward(self, partial_uv: Tensor, mask: Tensor,
                position_map: Tensor) -> Tensor:
        # Concatenate conditions
        x = torch.cat([partial_uv, mask, position_map], dim=1)
        x = self.conv_in(x)

        # Single-step refinement
        refined = self.unet(x)

        return refined
```

**Multi-View Aggregation Before Completion:**
```python
def aggregate_multiview_texture(views: List[ViewData], mesh: MeshData) -> Tensor:
    """Combine textures from multiple views via backprojection."""

    # Initialize UV accumulation buffers
    color_sum = torch.zeros(H, W, 3)
    weight_sum = torch.zeros(H, W, 1)

    for view in views:
        # Backproject view to UV space
        uv_colors = backproject_to_uv(view.image, view.camera, mesh)
        uv_mask = compute_visibility_mask(view.camera, mesh)

        # Weighted accumulation
        color_sum += uv_colors * uv_mask
        weight_sum += uv_mask

    # Average with epsilon for stability
    partial_uv = color_sum / (weight_sum + eps)

    return partial_uv, weight_sum > 0
```

### 5.3 Symmetry and Prior Exploitation

**Leveraging Object Priors:**
- Many objects have bilateral symmetry
- Category-specific texture priors (e.g., faces, cars)
- Material continuity expectations

```python
def symmetry_aware_completion(partial_texture: Tensor,
                               symmetry_axis: int = 0) -> Tensor:
    """Complete texture using symmetry prior."""

    # Flip along symmetry axis
    flipped = torch.flip(partial_texture, dims=[symmetry_axis])

    # Mask of missing regions
    missing = partial_texture.isnan() | (partial_texture == 0)

    # Fill missing with flipped values where available
    completed = partial_texture.clone()
    completed[missing] = flipped[missing]

    # Blend at boundaries
    blend_region = dilate(missing, kernel_size=5) & ~missing
    completed[blend_region] = 0.5 * (partial_texture[blend_region] +
                                      flipped[blend_region])

    return completed
```

### 5.4 Consistency Enforcement

**Multi-View Consistency:**
- Completed texture must look correct from all angles
- Rendered views should match observed views
- Gradual transitions between observed and hallucinated

From [SuperMat Pipeline](https://arxiv.org/html/2411.17515v1):
- "UV refinement network addresses maintaining consistency across viewpoints"
- "Inpaints missing areas and eliminates discrepancies between views"
- Complete pipeline: ~3 seconds per 3D object

**Consistency Loss:**
```python
def multiview_consistency_loss(texture: Tensor, mesh: MeshData,
                                cameras: List[Camera],
                                target_views: List[Tensor]) -> Tensor:
    """Ensure texture renders consistently across views."""

    total_loss = 0
    for camera, target in zip(cameras, target_views):
        rendered = render(mesh, texture, camera)

        # Only penalize visible regions
        visibility_mask = compute_visibility(mesh, camera)

        loss = (rendered - target).abs() * visibility_mask
        total_loss += loss.sum() / visibility_mask.sum()

    return total_loss / len(cameras)
```

---

## Section 6: Quality Metrics and Evaluation

### 6.1 Image Quality Metrics

**Standard Metrics for Texture Evaluation:**

| Metric | Range | Best | Measures |
|--------|-------|------|----------|
| PSNR | 0-inf | Higher | Peak signal-to-noise ratio |
| SSIM | 0-1 | 1.0 | Structural similarity |
| LPIPS | 0-1 | 0.0 | Perceptual similarity |

From [SuperMat Evaluation](https://arxiv.org/html/2411.17515v1):
```python
def evaluate_material_decomposition(pred: Dict, gt: Dict) -> Dict:
    """Compute standard quality metrics."""

    metrics = {}

    for material in ['albedo', 'roughness', 'metallic']:
        p, g = pred[material], gt[material]

        # Peak Signal-to-Noise Ratio
        mse = ((p - g) ** 2).mean()
        metrics[f'{material}_psnr'] = 10 * torch.log10(1.0 / mse)

        # Structural Similarity
        metrics[f'{material}_ssim'] = compute_ssim(p, g)

        # Learned Perceptual Similarity
        metrics[f'{material}_lpips'] = lpips_model(p, g)

    return metrics
```

### 6.2 Rendering-Based Evaluation

**Re-Rendering Metrics:**
- Render with predicted materials under novel lighting
- Compare to ground truth rendering
- Tests material interactions, not just individual maps

From [SuperMat](https://arxiv.org/html/2411.17515v1):
- "Relighting" column evaluates joint performance
- Uses ground truth normals and positions
- Randomly sampled environment maps (716 total)

```python
def relighting_evaluation(pred_materials: Dict, gt_materials: Dict,
                          geometry: Dict, num_lights: int = 10) -> Dict:
    """Evaluate materials under novel lighting conditions."""

    metrics = {'psnr': [], 'ssim': [], 'lpips': []}

    for _ in range(num_lights):
        # Random environment map
        env_map = sample_environment_map()

        # Render both
        pred_render = pbr_render(pred_materials, geometry, env_map)
        gt_render = pbr_render(gt_materials, geometry, env_map)

        # Compute metrics
        metrics['psnr'].append(compute_psnr(pred_render, gt_render))
        metrics['ssim'].append(compute_ssim(pred_render, gt_render))
        metrics['lpips'].append(compute_lpips(pred_render, gt_render))

    return {k: np.mean(v) for k, v in metrics.items()}
```

### 6.3 User Studies and Perceptual Quality

**Human Preference Testing:**

From [SAM 3D Study](../source-documents/SAM_STUDY_3D.md):
- SAM 3D Objects achieves **5:1 win rate** in human preference
- Compares against leading 3D reconstruction methods
- Tests overall quality including texture/material

**Perceptual Quality Factors:**
1. **Shadow removal**: Clean albedo without baked lighting
2. **Material plausibility**: Physically reasonable values
3. **Detail preservation**: High-frequency texture maintained
4. **Boundary clarity**: Sharp transitions between materials

### 6.4 Geometric and UV Quality

**UV Mapping Quality Metrics:**

```python
def evaluate_uv_quality(mesh_3d: Tensor, mesh_uv: Tensor,
                        faces: Tensor) -> Dict:
    """Evaluate UV mapping quality."""

    metrics = {}

    # Angle distortion (conformal error)
    angles_3d = compute_triangle_angles(mesh_3d[faces])
    angles_uv = compute_triangle_angles_2d(mesh_uv[faces])
    metrics['angle_distortion'] = (angles_3d - angles_uv).abs().mean()

    # Area distortion
    areas_3d = triangle_area_3d(mesh_3d[faces])
    areas_uv = triangle_area_2d(mesh_uv[faces])
    normalized_3d = areas_3d / areas_3d.sum()
    normalized_uv = areas_uv / areas_uv.sum()
    metrics['area_distortion'] = (normalized_3d - normalized_uv).abs().mean()

    # Texture utilization
    metrics['utilization'] = compute_atlas_utilization(mesh_uv, faces)

    return metrics
```

---

## Section 7: ARR-COC-0-1: Texture Features for Visual Grounding

### 7.1 Texture as Relevance Signal

For ARR-COC-0-1's visual grounding and relevance realization system, texture and material information provides crucial semantic signals beyond pure geometry.

**Texture-Based Relevance Cues:**
1. **Material identification**: Metal vs. fabric vs. wood affects object function
2. **State recognition**: Clean vs. dirty, new vs. worn
3. **Part segmentation**: Different materials indicate different components
4. **Functional affordances**: Glossy = slippery, rough = grippable

### 7.2 Material-Aware Attention

**Integrating Material Features into VLM:**

```python
class MaterialAwareAttention(nn.Module):
    """Attention mechanism that incorporates material properties."""

    def __init__(self, d_model: int):
        super().__init__()
        self.material_encoder = nn.Sequential(
            nn.Linear(5, d_model // 4),  # RGB + roughness + metallic
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model)
        )
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads=8)

    def forward(self, visual_tokens: Tensor,
                material_features: Tensor,
                text_query: Tensor) -> Tensor:
        # Encode material properties
        material_tokens = self.material_encoder(material_features)

        # Concatenate with visual features
        enhanced_visual = visual_tokens + material_tokens

        # Cross-attention with text
        attended, weights = self.multihead_attn(
            query=text_query,
            key=enhanced_visual,
            value=enhanced_visual
        )

        return attended, weights
```

### 7.3 Texture-Guided Object Grounding

**Grounding with Material Cues:**
- "The shiny metal object" -> attend to high metallic regions
- "The rough wooden surface" -> attend to high roughness + wood color
- "The transparent glass container" -> special handling for translucent

```python
def material_guided_grounding(image: Tensor,
                               material_maps: Dict[str, Tensor],
                               text_query: str) -> Tensor:
    """Ground objects using material-based cues."""

    # Extract material attributes from query
    material_attrs = parse_material_attributes(text_query)
    # e.g., {'metallic': 'high', 'roughness': 'low', 'color': 'silver'}

    # Build material likelihood map
    likelihood = torch.ones_like(material_maps['albedo'][..., 0])

    if 'metallic' in material_attrs:
        target = 1.0 if material_attrs['metallic'] == 'high' else 0.0
        likelihood *= gaussian(material_maps['metallic'], target, sigma=0.2)

    if 'roughness' in material_attrs:
        target = 1.0 if material_attrs['roughness'] == 'high' else 0.0
        likelihood *= gaussian(material_maps['roughness'], target, sigma=0.2)

    if 'color' in material_attrs:
        target_rgb = color_name_to_rgb(material_attrs['color'])
        color_dist = (material_maps['albedo'] - target_rgb).norm(dim=-1)
        likelihood *= torch.exp(-color_dist / 0.3)

    return likelihood
```

### 7.4 Spatial Reasoning with Materials

**3D Material-Aware Scene Understanding:**
```python
class MaterialAwareSceneGraph(nn.Module):
    """Build scene graph with material-aware relationships."""

    def compute_relationships(self, objects: List[Object3D],
                               materials: List[MaterialProps]) -> List[Relation]:
        relations = []

        for i, (obj_i, mat_i) in enumerate(zip(objects, materials)):
            for j, (obj_j, mat_j) in enumerate(zip(objects, materials)):
                if i >= j:
                    continue

                # Spatial relationships
                spatial = compute_spatial_relation(obj_i.bbox, obj_j.bbox)

                # Material-based relationships
                if mat_i.metallic > 0.5 and mat_j.metallic < 0.5:
                    relations.append(Relation(i, j, 'conducts_to'))

                if mat_i.roughness < 0.3 and mat_j.roughness > 0.7:
                    relations.append(Relation(i, j, 'smoother_than'))

                relations.append(Relation(i, j, spatial))

        return relations
```

### 7.5 Applications for ARR-COC

**Texture in Relevance Allocation:**
1. **Attention allocation**: Prioritize textured regions with semantic content
2. **Quality assessment**: Skip over textureless/uniform regions
3. **Uncertainty estimation**: Low-confidence materials need more tokens
4. **Cross-modal grounding**: Match text descriptions to material properties

**Integration Points:**
- Pre-computed material maps as additional encoder input
- Material-conditioned cross-attention layers
- Texture-based spatial relationship reasoning
- Material consistency loss for generated descriptions

---

## Sources

**Source Documents:**
- [SAM_STUDY_3D.md](../source-documents/SAM_STUDY_3D.md) - Meta's SAM 3D announcement and capabilities

**Web Research:**

**PBR Material Estimation:**
- [SuperMat: Physically Consistent PBR Material Estimation](https://arxiv.org/html/2411.17515v1) - arXiv:2411.17515 (accessed 2025-11-20)
- [Neural LightRig: Object Normal and Material Estimation](https://openaccess.thecvf.com/content/CVPR2025/papers/He_Neural_LightRig_Unlocking_Accurate_Object_Normal_and_Material_Estimation_with_CVPR_2025_paper.pdf) - CVPR 2025
- [HumanMaterial: Human Material Estimation](https://arxiv.org/html/2507.18385v1) - arXiv:2507.18385
- [PBR-Net: Imitating Physically Based Rendering](http://www.liushuaicheng.org/TIP/PBR-Net/PBRNet-TIP.pdf) - TIP 2020

**Neural UV and Texture Mapping:**
- [ArtUV: Artist-style UV Unwrapping](https://arxiv.org/html/2509.20710v1) - arXiv:2509.20710 (accessed 2025-11-20)
- [Nuvo: Neural UV Mapping for Unruly 3D](https://dl.acm.org/doi/10.1007/978-3-031-72933-1_2) - ECCV 2024, cited by 20
- [SeamCrafter: Enhancing Mesh Seam Generation](https://arxiv.org/html/2509.20725v1) - arXiv:2509.20725
- [NeuTex: Neural Texture Mapping](https://openaccess.thecvf.com/content/CVPR2021/papers/Xiang_NeuTex_Neural_Texture_Mapping_for_Volumetric_Neural_Rendering_CVPR_2021_paper.pdf) - CVPR 2021, cited by 129

**Texture Synthesis and Completion:**
- [TEGLO: High Fidelity Canonical Texture Mapping](https://openaccess.thecvf.com/content/WACV2024/papers/Vinod_TEGLO_High_Fidelity_Canonical_Texture_Mapping_From_Single-View_Images_WACV_2024_paper.pdf) - WACV 2024, cited by 7
- [Deferred Neural Rendering](https://niessnerlab.org/papers/2019/11neuralrendering/) - Thies et al., cited by 1647
- [Learning a Neural 3D Texture Space](https://openaccess.thecvf.com/content_CVPR_2020/papers/Henzler_Learning_a_Neural_3D_Texture_Space_From_2D_Exemplars_CVPR_2020_paper.pdf) - CVPR 2020, cited by 108

**3D Reconstruction Surveys:**
- [3D Reconstruction from Single RGB Image](https://av.dfki.de/publications/3d-reconstruction-from-a-single-rgb-image-using-deep-learning-a-review/) - DFKI Review
- [AI-based 3D Terrain Completion](https://ict.usc.edu/research/projects/ai-based-large-scale-3d-terrain-completion/) - USC ICT

**Facial Texture:**
- [Ultra-high Resolution Facial Texture Reconstruction](https://www.sciopen.com/article/10.26599/CVM.2025.9450488) - CVM 2025

---

## Summary

Texture mapping and material estimation are critical components of modern 3D reconstruction systems, bridging the gap between geometric reconstruction and photorealistic rendering. Key advances include:

1. **Single-step diffusion models** (SuperMat) achieve SOTA material decomposition at millisecond speeds
2. **Neural UV unwrapping** automates professional-quality atlas generation
3. **End-to-end training with re-render loss** significantly improves material quality
4. **Multi-view aggregation with UV refinement** completes textures for unseen regions

For ARR-COC-0-1, texture and material features provide rich semantic signals for visual grounding, enabling material-aware attention and more precise object localization based on surface properties. The integration of PBR material estimation into VLM pipelines opens new possibilities for understanding physical properties from visual input.
