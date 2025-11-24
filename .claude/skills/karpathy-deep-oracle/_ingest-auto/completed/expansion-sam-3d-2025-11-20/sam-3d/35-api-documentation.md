# API Documentation & Integration Guide

**Using SAM 3D in production: Python API, REST endpoints, and SDK integration**

---

## 1. Python API Overview

**Installation:**
```bash
pip install sam-3d
```

**Basic Usage:**
```python
from sam3d import SAM3DPredictor

# Initialize model
predictor = SAM3DPredictor.from_pretrained("facebook/sam-3d-base")

# Load image
image = load_image("photo.jpg")

# Generate 3D mesh
mesh = predictor.predict(image)

# Export
mesh.export("output.obj")
```

---

## 2. Model API

**SAM3DPredictor:**
- `predict(image)` → Mesh3D
- `predict_batch(images)` → List[Mesh3D]
- `predict_with_prompts(image, prompts)` → Mesh3D

**Parameters:**
- `image`: PIL.Image or np.ndarray (H×W×3)
- `prompts`: Optional[Dict] (points, boxes, masks)

**Returns:**
- `Mesh3D` object (vertices, faces, textures)

---

## 3. REST API

**Endpoint:**
```
POST https://api.meta.ai/sam3d/v1/predict
```

**Request:**
```json
{
  "image": "base64_encoded_image",
  "format": "obj",
  "options": {
    "quality": "high",
    "texture": true
  }
}
```

**Response:**
```json
{
  "mesh_url": "https://cdn.../mesh.obj",
  "vertices": 12543,
  "faces": 25086,
  "latency_ms": 145
}
```

---

## 4. Mesh3D Object

**Attributes:**
- `vertices`: np.ndarray (N×3)
- `faces`: np.ndarray (M×3)
- `textures`: Optional[np.ndarray] (N×3)

**Methods:**
- `export(path, format='obj')`
- `render(view='front')`
- `simplify(target_faces=10000)`

---

## 5. Batch Processing

**Efficient Multi-Image:**
```python
images = [load_image(f) for f in image_paths]
meshes = predictor.predict_batch(images, batch_size=4)
```

**Batching Benefits:**
- 2.5× throughput vs sequential
- Shared preprocessing

---

## 6. ARR-COC-0-1 Integration (10%)

**API for Production Deployment:**

Clean API enables:
- Easy integration into VLM pipelines
- Batch processing for training data
- REST endpoints for cloud services

---

**Sources:**
- SAM 3D GitHub repository
- API documentation
- REST endpoint specifications
