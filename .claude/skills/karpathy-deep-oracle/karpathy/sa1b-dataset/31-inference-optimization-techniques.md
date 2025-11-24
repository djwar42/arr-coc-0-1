# Inference Optimization for SAM

## Overview

SAM inference can be optimized for speed via **image embedding caching, prompt batching, quantization, and TorchScript compilation**.

## Key Optimizations

**1. Image Embedding Caching**
- Encode image once, reuse for multiple prompts
- ViT forward pass: ~100-200ms
- Mask decoder: ~10-20ms per prompt
- **10-20× speedup** for multi-prompt queries

```python
# Cache image embeddings
with torch.no_grad():
    image_embedding = sam.image_encoder(image)  # Once

# Fast prompt-based inference
for prompt in prompts:
    masks = sam.mask_decoder(image_embedding, prompt)  # Fast
```

**2. Prompt Batching**
- Process multiple prompts simultaneously
- Batch size 8-32 typical

**3. Quantization (INT8/FP16)**
- FP16: 2× speedup, minimal quality loss
- INT8: 4× speedup, ~1-2% IoU degradation

**4. TorchScript Compilation**
- Export to TorchScript for deployment
- 10-30% faster inference

## Inference Modes

**Automatic mode** (everything): Segment entire image
**Point mode**: Click-based segmentation
**Box mode**: Bounding box prompt
**Mask mode**: Refine existing mask

## ARR-COC Application

**Batch spatial grounding for documents:**
```python
# Cache document image once
doc_embedding = sam.image_encoder(document_image)

# Ground multiple relevance queries efficiently
for query in relevance_queries:
    prompt = query_to_spatial_prompt(query)
    relevance_mask = sam.mask_decoder(doc_embedding, prompt)
```

**Sources**: SAM GitHub repo, Inference optimization guides
