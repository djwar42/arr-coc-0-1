# BATCH 8: 3D Mesh Features Research

## PointNet Architecture

**Paper:** Qi et al. 2016 - **22,222 citations!**

### Core Innovation

Direct processing of point clouds without converting to voxels or meshes.

### Architecture

```python
class PointNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Per-point MLP
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 1024), nn.ReLU()
        )

        # Global feature aggregation
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, points):
        # points: [B, N, 3]
        # Per-point features
        features = self.mlp1(points)  # [B, N, 1024]

        # Global max pooling (permutation invariant!)
        global_feature = self.max_pool(features.transpose(1, 2))
        global_feature = global_feature.squeeze(-1)  # [B, 1024]

        # Classification
        return self.classifier(global_feature)
```

### Key Properties

1. **Permutation invariance:** Max pooling over points
2. **Input transform:** T-Net learns canonical orientation
3. **Feature transform:** Regularizes feature space

### T-Net (Spatial Transformer)

```python
class TNet(nn.Module):
    """Learns transformation matrix for point cloud alignment"""
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(k, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 1024), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, k * k)
        )

        # Initialize to identity
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data = torch.eye(k).flatten()

    def forward(self, x):
        features = self.mlp(x).max(dim=1)[0]
        transform = self.fc(features).view(-1, self.k, self.k)
        return transform
```

## PointNet++ (Hierarchical)

Adds local neighborhood processing:

```python
# Set abstraction layer
def set_abstraction(points, features, num_samples, radius, mlp):
    # Sample centroids
    centroids = farthest_point_sample(points, num_samples)

    # Group neighbors
    neighbors = ball_query(points, centroids, radius)

    # Apply MLP to each group
    local_features = mlp(neighbors)

    # Max pool within group
    aggregated = local_features.max(dim=-2)[0]

    return centroids, aggregated
```

## Mesh Curvature Computation

### Discrete Curvature

For meshes, curvature is computed per vertex:

**Gaussian Curvature (K):**
```
K_v = (2π - Σ θᵢ) / A_v
```
Where θᵢ = angles around vertex, A_v = vertex area

**Mean Curvature (H):**
```
H_v = (1/4A_v) Σ |eᵢ| (θ_left + θ_right - π)
```

### Implementation

```python
def compute_curvature(vertices, faces):
    """Compute Gaussian and mean curvature per vertex"""
    num_vertices = len(vertices)
    gaussian = torch.zeros(num_vertices)
    mean = torch.zeros(num_vertices)

    for face in faces:
        v0, v1, v2 = vertices[face]

        # Compute angles
        angles = compute_triangle_angles(v0, v1, v2)

        # Accumulate angle deficit
        for i, angle in enumerate(angles):
            gaussian[face[i]] += 2 * np.pi - angle

    # Normalize by vertex area
    vertex_areas = compute_vertex_areas(vertices, faces)
    gaussian = gaussian / vertex_areas

    return gaussian, mean
```

## Genus Computation (Topology)

### Euler Characteristic

```
χ = V - E + F = 2 - 2g
```

Where:
- V = vertices
- E = edges
- F = faces
- g = genus (number of holes)

### Computing Genus

```python
def compute_genus(vertices, edges, faces):
    V = len(vertices)
    E = len(edges)
    F = len(faces)

    euler = V - E + F
    genus = (2 - euler) // 2

    return genus
```

### Why Genus Matters

- **g = 0:** Sphere-like (ball, cube)
- **g = 1:** Torus (donut, coffee cup)
- **g = 2:** Double torus
- **Higher g:** More complex topology

## 3D Shape Descriptors

### Shape Features

```python
def compute_shape_descriptors(mesh):
    features = {}

    # Geometric
    features['surface_area'] = compute_surface_area(mesh)
    features['volume'] = compute_volume(mesh)
    features['compactness'] = 36 * np.pi * features['volume']**2 / features['surface_area']**3

    # Curvature
    K, H = compute_curvature(mesh.vertices, mesh.faces)
    features['mean_gaussian_curvature'] = K.mean()
    features['mean_mean_curvature'] = H.mean()
    features['curvature_variance'] = K.var()

    # Topology
    features['genus'] = compute_genus(mesh.vertices, mesh.edges, mesh.faces)

    # Normals
    normals = compute_vertex_normals(mesh)
    features['normal_variance'] = normals.var()

    return features
```

### Deep Shape Descriptors

```python
class ShapeEncoder(nn.Module):
    """Learn shape descriptors from 3D data"""
    def __init__(self, feature_dim=256):
        self.pointnet = PointNet(feature_dim)
        self.curvature_encoder = nn.Linear(2, 32)  # K, H per point
        self.fusion = nn.Linear(feature_dim + 32, feature_dim)

    def forward(self, points, curvatures):
        # Point cloud features
        point_features = self.pointnet(points)

        # Curvature features
        curv_features = self.curvature_encoder(curvatures).max(dim=1)[0]

        # Fuse
        combined = torch.cat([point_features, curv_features], dim=-1)
        return self.fusion(combined)
```

## Integration with Spicy Lentil

### Mesh Features for Object Slots

Each object slot can have 3D understanding:

```python
class SlotWith3DFeatures(nn.Module):
    def __init__(self, slot_dim, mesh_dim):
        self.slot_encoder = nn.Linear(slot_dim, mesh_dim)
        self.mesh_features = nn.Linear(mesh_dim, slot_dim)

    def forward(self, slot, predicted_mesh):
        # Extract mesh features
        curvature = compute_curvature(predicted_mesh)
        genus = compute_genus(predicted_mesh)
        topology_features = torch.tensor([genus, curvature.mean()])

        # Combine with slot
        mesh_enc = self.mesh_features(topology_features)
        return slot + mesh_enc
```

### 3D Spatial Pathways

The perspectival pathway benefits from 3D understanding:

```python
class PerspectivalWith3D(nn.Module):
    def forward(self, slot, mesh_reconstruction):
        # View-dependent features
        views = render_multiple_views(mesh_reconstruction)

        # Combine 2D views with 3D structure
        view_features = self.view_encoder(views)
        mesh_features = self.mesh_encoder(mesh_reconstruction)

        # Perspectival knowing = understanding viewpoint dependence
        return self.fuse(view_features, mesh_features)
```

### Genus as Complexity Measure

Topological complexity can determine processing depth:

```python
# More topologically complex objects need more processing
genus = compute_genus(object_mesh)
processing_depth = base_depth + genus * depth_per_genus
```

## Performance Numbers

### PointNet Results

| Task | Dataset | Accuracy |
|------|---------|----------|
| Classification | ModelNet40 | 89.2% |
| Part Seg | ShapeNet | 83.7% IoU |
| Semantic Seg | S3DIS | 78.6% |

### PointNet++ Improvements

- Classification: +2% over PointNet
- Segmentation: +3-4% IoU
- Better on fine-grained geometry

## Key Formulas

### Permutation Invariant Aggregation
```
f({x_1, ..., x_n}) = γ(MAX_{i=1..n}(h(x_i)))
```

### Gaussian Curvature
```
K = κ_1 × κ_2
```
Where κ_1, κ_2 are principal curvatures

### Mean Curvature
```
H = (κ_1 + κ_2) / 2
```

---

**Sources:**
- "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation" - CVPR 2017
- "PointNet++: Deep Hierarchical Feature Learning on Point Sets" - NeurIPS 2017
- "Deep Learning on Point Clouds and Its Application: A Survey" - Sensors 2019
- Discrete Differential Geometry textbooks
