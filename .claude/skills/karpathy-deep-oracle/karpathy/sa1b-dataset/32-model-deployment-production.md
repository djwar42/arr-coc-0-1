# SAM Deployment to Production

## Overview

Deploy SAM for production with **ONNX export, model serving APIs, containerization, and edge optimization**.

## Deployment Workflow

**1. Model Export**
```python
# Export to ONNX for cross-platform deployment
import torch.onnx

torch.onnx.export(
    sam,
    (example_input,),
    "sam_vit_h.onnx",
    opset_version=17,
    input_names=['image'],
    output_names=['masks', 'iou_predictions']
)
```

**2. API Server (FastAPI)**
```python
from fastapi import FastAPI, File
import numpy as np

app = FastAPI()
sam = load_sam_model("sam_vit_h_4b8939.pth")

@app.post("/segment")
async def segment(image: File, prompt: dict):
    img = preprocess(image)
    masks = sam.predict(img, prompt)
    return {"masks": masks.tolist()}
```

**3. Containerization**
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
COPY sam_model.py /app/
COPY sam_vit_h_4b8939.pth /models/
RUN pip install segment-anything fastapi uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
```

**4. Edge Deployment (TensorRT/CoreML)**
- TensorRT: NVIDIA edge devices (Jetson)
- CoreML: iOS/macOS deployment
- Model size: ViT-B ~350MB, ViT-H ~2.4GB

## Production Considerations

**Latency targets:**
- Cloud: <500ms end-to-end
- Edge: <1s (with caching)
- Real-time: <100ms (requires ViT-B + quantization)

**Scaling:**
- Horizontal: Multiple replica pods
- Vertical: GPU scaling (T4/A10 for inference)
- Caching: Redis for embeddings

## ARR-COC Application

**Document grounding API:**
```python
@app.post("/relevance_grounding")
async def ground(document: File, query: str):
    # Load document image
    img = load_doc_image(document)

    # Cache embedding
    cache_key = hash(document)
    if cache_key in embedding_cache:
        embedding = embedding_cache[cache_key]
    else:
        embedding = sam.image_encoder(img)
        embedding_cache[cache_key] = embedding

    # Ground query to spatial regions
    prompt = text_to_spatial_prompt(query)
    masks = sam.mask_decoder(embedding, prompt)

    return {"relevance_masks": masks}
```

**Sources**: SAM deployment examples, FastAPI docs, Docker best practices
