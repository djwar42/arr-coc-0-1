# CLIP Vision Encoder: Deep Dive into Contrastive Vision-Language Pre-training

## Overview

CLIP (Contrastive Language-Image Pre-training) represents a paradigm shift in vision-language models by learning visual representations through natural language supervision at scale. Unlike traditional supervised learning on fixed label sets (like ImageNet's 1000 classes), CLIP learns to align images with arbitrary text descriptions through contrastive learning, enabling zero-shot transfer to downstream tasks.

From [CLIP: Connecting text and images](https://openai.com/index/clip/) (OpenAI, January 2021):
- Trained on 400M image-text pairs from the internet
- Achieves competitive zero-shot performance on 30+ datasets
- Uses InfoNCE contrastive loss to align vision and text embeddings
- Vision encoder based on Vision Transformer (ViT) or ResNet architectures

From [Demystifying CLIP Data](https://arxiv.org/abs/2309.16671) (Xu et al., 2023):
- **MetaCLIP** demonstrates that CLIP's success lies primarily in data curation, not model architecture
- Metadata-curated training on CommonCrawl achieves 70.8% ImageNet zero-shot (vs CLIP's 68.3% on ViT-B)
- Scaling to 1B samples with same training budget reaches 72.4% accuracy
- Data distribution over metadata concepts is key to performance

## CLIP Architecture: Two-Encoder Contrastive Learning

### Dual Encoder Design

CLIP consists of two separate encoders trained jointly:

**Vision Encoder Options:**
1. **ResNet-50/101**: Modified with attention pooling instead of average pooling
2. **Vision Transformer (ViT)**: ViT-B/32, ViT-B/16, ViT-L/14, ViT-H/14 variants
   - Patch size in model name (e.g., /14 = 14×14 pixel patches)
   - Layer normalization applied to combined patch and position embeddings

**Text Encoder:**
- Transformer with causal masking (63M-parameter, 12-layer, 512-wide)
- Max sequence length: 76 tokens
- [BOS] and [EOS] tokens added to text
- Feature extracted from [EOS] token embedding

**Joint Embedding Space:**
- Both encoders project to same dimensionality (e.g., 512-d for ViT-B/32)
- L2 normalization applied to all embeddings
- Learned temperature parameter τ scales logits for contrastive loss

### Contrastive Pre-training Objective

CLIP uses **InfoNCE loss** (contrastive learning):

```
Given batch of N (image, text) pairs:
- Compute N × N similarity matrix: S[i,j] = cos(image_i, text_j) / τ
- Image-to-text loss: -log(exp(S[i,i]) / Σ_j exp(S[i,j]))
- Text-to-image loss: -log(exp(S[i,i]) / Σ_i exp(S[i,i]))
- Final loss: average of both directions
```

**Key properties:**
- Symmetric loss encourages bidirectional alignment
- Large batch sizes (32,768 in CLIP) provide more negative examples
- Temperature τ learned during training (controls sharpness of distribution)
- Each positive pair contrasted against 2N-1 negative examples per batch

From [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip):
- OpenCLIP reproduces and extends CLIP training
- Supports multiple data sources with resampling
- Provides pre-trained models: ViT-B/32, ViT-L/14, ViT-H/14, ViT-bigG/14
- Achieves 80.1% ImageNet zero-shot with ViT-bigG/14 on LAION-2B

## Vision Transformer Variants for CLIP

### ViT-B/32 (Base Model, 32×32 patches)

**Architecture:**
- 12 transformer layers, 768 hidden dim, 12 attention heads
- Input: 224×224 image → 49 patches (7×7 grid of 32×32 patches)
- Parameters: ~86M for vision encoder
- Embedding dim: 512 for CLIP joint space

**Performance (OpenCLIP on LAION-400M):**
- ImageNet zero-shot: ~63% accuracy
- Fast inference: ~140 images/sec on V100

### ViT-B/16 (Base Model, 16×16 patches)

**Architecture:**
- 12 transformer layers, 768 hidden dim
- Input: 224×224 → 196 patches (14×14 grid)
- 4× more patches than ViT-B/32 → more detailed spatial info
- Parameters: ~86M vision encoder

**Performance:**
- ImageNet zero-shot: 68-73% (depending on training data)
- DataComp-1B trained: 73.5% zero-shot accuracy

### ViT-L/14 (Large Model, 14×14 patches)

**Architecture:**
- 24 transformer layers, 1024 hidden dim, 16 attention heads
- Input: 224×224 → 256 patches (16×16 grid of 14×14 patches)
- Parameters: ~304M for vision encoder
- Most commonly used CLIP variant

**Performance:**
- Original CLIP (OpenAI, WIT dataset): 75.5% ImageNet zero-shot
- OpenCLIP (LAION-2B, 32B samples): 75.3% accuracy
- DataComp-1B (13B samples): 79.2% accuracy
- Training time: ~2 weeks on 256 GPUs for large-scale runs

### ViT-H/14 (Huge Model, 14×14 patches)

**Architecture:**
- 32 transformer layers, 1280 hidden dim, 16 attention heads
- Input: 224×224 → 256 patches
- Parameters: ~632M for vision encoder

**Performance:**
- LAION-2B trained (32B samples): 78.0% ImageNet zero-shot
- DFN-5B trained (39B samples): 83.4% at 224px, 84.4% at 378px

### ViT-bigG/14 (Giant Model)

**Architecture:**
- Largest CLIP ViT variant
- Parameters: ~1.8B+ for vision encoder
- Requires significant compute (34B+ training samples)

**Performance:**
- LAION-2B: 80.1% ImageNet zero-shot (CLIP's best reported)
- PE-Core-bigG-14-448: 85.4% with MetaCLIP-5.4B data

## OpenCLIP: Open Source CLIP Implementation

### Training Efficiency Improvements

From [OpenCLIP GitHub](https://github.com/mlfoundations/open_clip):

**1. Contrastive Loss Optimization:**
- 4-10× more efficient than image-to-caption generation (VirTex approach)
- Symmetric InfoNCE loss provides rich gradient signal
- Large batch training (32k+) crucial for performance

**2. Vision Transformer Adoption:**
- 3× compute efficiency gain over ResNet backbones
- Better scaling properties with model size
- Native patch-based processing aligns with contrastive learning

**3. Mixed Precision Training:**
- Automatic Mixed Precision (AMP) reduces memory, speeds training
- FP16 gradients with FP32 master weights
- Enables larger batch sizes per GPU

**4. Distributed Training Features:**
- `--gather-with-grad`: Gathers embeddings across GPUs while maintaining gradients
- `--local-loss`: Computes loss locally per GPU (reduces communication)
- Combined: O(n) memory complexity instead of O(n²) for logit matrix
- Scales to 1024+ GPUs without performance degradation

### OpenCLIP Pre-trained Models

**State-of-the-art zero-shot ImageNet accuracy:**

| Model | Training Data | Resolution | Samples Seen | ImageNet Acc |
|-------|---------------|------------|--------------|--------------|
| ViT-B-32 | LAION-2B | 256px | 13B | ~63% |
| ViT-B-16 | DataComp-1B | 224px | 13B | 73.5% |
| ViT-L-14 | LAION-2B | 224px | 32B | 75.3% |
| ViT-L-14 | DataComp-1B | 224px | 13B | 79.2% |
| ViT-H-14 | LAION-2B | 224px | 32B | 78.0% |
| ViT-bigG-14 | LAION-2B | 224px | 34B | 80.1% |

**Model availability:** [HuggingFace Hub - OpenCLIP library](https://huggingface.co/models?library=open_clip)

### Training Data: From WIT to LAION to DataComp

**Original CLIP (OpenAI):**
- 400M image-text pairs from WebImageText (WIT)
- Curated from internet, filtering details not disclosed
- Training compute: not publicly reported

**LAION-400M/2B/5B:**
- Largest public image-text datasets
- CommonCrawl web scraping + CLIP filtering
- LAION-2B: 2.3 billion English image-text pairs
- LAION-5B: adds multilingual data

**DataComp-1B:**
- 1.4B image-text pairs
- Rigorous curation methodology
- Better quality-to-quantity ratio than LAION
- Achieves higher accuracy with fewer samples

**MetaCLIP approach:**
- Metadata-driven curation (balances concept distribution)
- Uses CLIP's concept vocabulary from WIT
- Achieves 70.8% with 400M pairs, 72.4% with 1B pairs
- Demonstrates data > architecture for CLIP performance

## Frozen vs Fine-tunable CLIP Encoders in VLMs

### Frozen CLIP Vision Encoder (Most Common)

**Advantages:**
- Preserves powerful zero-shot capabilities
- Reduces training compute and memory
- Prevents overfitting to downstream task
- Enables rapid prototyping of VLM architectures

**Usage in VLMs:**
- LLaVA: Frozen ViT-L/14 CLIP + MLP projector + trainable LLM
- OpenFlamingo: Frozen CLIP + Perceiver Resampler + trainable LM
- BLIP-2: Frozen CLIP image encoder + trainable Q-Former + frozen LLM

From [F-VLM: Open-vocabulary object detection upon frozen vision and language models](https://research.google/blog/f-vlm-open-vocabulary-object-detection-upon-frozen-vision-and-language-models/):
- F-VLM keeps both CLIP vision AND text encoders frozen
- Only trains lightweight detection head
- Shows strong scaling with frozen model capacity
- Generalizes better to open-vocabulary scenarios

### Fine-tuning CLIP (Less Common, Task-Specific)

**When to fine-tune:**
- Domain-specific tasks (medical imaging, satellite imagery)
- Fine-grained classification (species, car models)
- Tasks with distribution shift from web data

**Fine-tuning strategies:**
- **Full fine-tuning**: Update all parameters (most expensive)
- **Partial unfreezing**: Last N layers trainable (e.g., last 10 layers)
- **LoRA/Adapter layers**: Parameter-efficient fine-tuning
- **Prompt tuning**: Learnable prompt tokens (CoOp, CoCoOp)

From [Understanding Fine-tuning CLIP for Open-vocabulary Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2025/papers/Peng_Understanding_Fine-tuning_CLIP_for_Open-vocabulary_Semantic_Segmentation_in_Hyperbolic_Space_CVPR_2025_paper.pdf):
- Joint fine-tuning of image + text encoders significantly improves performance
- Freezing text encoder preserves embeddings but limits adaptation
- Hyperbolic space fine-tuning enhances hierarchical concept learning

**Trade-offs:**
- Fine-tuning improves task accuracy but reduces zero-shot generalization
- Frozen encoders maintain robustness to distribution shift
- Hybrid approaches: freeze early layers, fine-tune later layers

## Tensor Parallelism for Large CLIP Vision Encoders

### Column and Row Parallelism for ViT

From [distributed-training/02-megatron-lm-tensor-parallelism.md](../distributed-training/02-megatron-lm-tensor-parallelism.md):

**Self-Attention Tensor Parallelism:**
```
# ViT self-attention with tensor parallel
Q, K, V projection: Column parallel (split attention heads across GPUs)
Output projection: Row parallel (reduce across GPUs)

For ViT-L/14 (16 heads):
- GPU 0: heads 0-7
- GPU 1: heads 8-15
- Each GPU computes local attention, reduce at output
```

**MLP Tensor Parallelism:**
```
# ViT MLP block
Linear1 (expand): Column parallel (split hidden dimension)
GELU activation: Local (no communication)
Linear2 (project): Row parallel (all-reduce gradients)
```

**Memory savings for ViT-H/14 (632M params):**
- 2-way tensor parallel: ~2.5GB per GPU (from ~4.7GB single GPU)
- 4-way tensor parallel: ~1.3GB per GPU
- Enables training larger ViT variants (bigG-14) on consumer GPUs

### Pipeline Parallelism for CLIP Dual Encoders

From [distributed-training/01-deepspeed-pipeline-parallelism.md](../distributed-training/01-deepspeed-pipeline-parallelism.md):

**Stage assignment for CLIP:**
```
Vision Encoder (ViT-L/14, 24 layers):
- Stage 0 (GPU 0-1): Patch embedding + layers 0-7
- Stage 1 (GPU 2-3): Layers 8-15
- Stage 2 (GPU 4-5): Layers 16-23 + projection

Text Encoder (12 layers):
- Stage 3 (GPU 6): All text encoder layers

Contrastive Loss:
- Stage 4 (GPU 7): Gather embeddings, compute InfoNCE
```

**Microbatching for pipeline efficiency:**
- Split batch of 256 into 8 microbatches of 32
- Reduce pipeline bubble (idle time between stages)
- 1F1B schedule: alternates forward and backward passes

## Triton Inference Server for CLIP Deployment

From [inference-optimization/02-triton-inference-server.md](../inference-optimization/02-triton-inference-server.md):

### Multi-Model CLIP Ensemble

**Ensemble pipeline for VLM:**
```
Model 1: CLIP Vision Encoder (ViT-L/14)
  Input: batch of images (224×224×3)
  Output: image embeddings (batch_size × 768)

Model 2: CLIP Text Encoder
  Input: batch of text tokens (batch_size × 77)
  Output: text embeddings (batch_size × 768)

Model 3: Similarity Scoring
  Input: image_emb, text_emb
  Output: similarity scores (batch_size × num_classes)
```

**Triton optimization:**
- Dynamic batching: Aggregate requests for throughput
- Model versioning: A/B test different CLIP checkpoints
- GPU scheduling: Pin vision encoder to specific GPU
- TensorRT backend: Optimize ViT attention kernels

### Zero-Shot Classification Serving

```python
# Triton ensemble config for CLIP zero-shot
{
  "name": "clip_zero_shot_classifier",
  "platform": "ensemble",
  "input": [
    {"name": "image", "data_type": "FP32", "dims": [3, 224, 224]},
    {"name": "class_names", "data_type": "STRING", "dims": [-1]}
  ],
  "output": [
    {"name": "probabilities", "data_type": "FP32", "dims": [-1]}
  ],
  "ensemble_scheduling": {
    "step": [
      {
        "model_name": "clip_vision_encoder",
        "input_map": {"input": "image"},
        "output_map": {"output": "image_features"}
      },
      {
        "model_name": "clip_text_encoder",
        "input_map": {"input": "class_names"},
        "output_map": {"output": "text_features"}
      },
      {
        "model_name": "cosine_similarity",
        "input_map": {
          "image_features": "image_features",
          "text_features": "text_features"
        },
        "output_map": {"output": "probabilities"}
      }
    ]
  }
}
```

**Performance characteristics:**
- Batch size 1 (online): ~8-15ms latency (ViT-L/14 on A100)
- Batch size 32 (offline): ~3000 images/sec throughput
- Text encoder amortized: Encode class names once, reuse

## Intel oneAPI Optimization for CLIP

From [alternative-hardware/02-intel-oneapi-ml.md](../alternative-hardware/02-intel-oneapi-ml.md):

### IPEX (Intel Extension for PyTorch) CLIP Optimization

**Operator fusion for ViT:**
```python
import intel_extension_for_pytorch as ipex

# Optimize CLIP vision encoder for Arc GPUs
model = open_clip.create_model('ViT-B-32', pretrained='laion400m_e32')
model = model.to('xpu')  # Intel XPU device
model = ipex.optimize(model)

# IPEX fuses operations:
# - LayerNorm + Linear (attention projection)
# - GELU + Linear (MLP blocks)
# - Patch embedding convolution + reshape
```

**Performance on Intel Arc A770:**
- ViT-B/32: ~45-60 images/sec (vs ~30 images/sec unfused)
- ViT-L/14: ~12-18 images/sec
- Mixed precision (BF16): 1.5-2× speedup
- Memory bandwidth optimization reduces VRAM to 6-8GB

### Intel Data Center GPU Max (Ponte Vecchio)

**Multi-tile CLIP inference:**
```
Arc GPU Max 1550 (2 tiles):
- Tile 0: ViT layers 0-11
- Tile 1: ViT layers 12-23
- XeLink fabric: 512 GB/s inter-tile bandwidth

Batch 128 performance:
- Single tile: ~850 images/sec
- Dual tile: ~1450 images/sec (1.7× scaling)
```

## ARR-COC-0-1: CLIP Features for Relevance Scoring

### Integration with Adaptive Relevance Realization

From [RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/](../../../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/):

**CLIP as Feature Extractor for Knowing Module:**

```python
# arr_coc/texture.py - 13-channel texture array generation
# CLIP provides semantic visual features for relevance computation

import open_clip

class TextureArrayGenerator:
    def __init__(self):
        # Load frozen CLIP vision encoder
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='laion2b_s32b_b82k'
        )
        self.clip_model.eval()

    def extract_clip_features(self, image_patches):
        """Extract CLIP embeddings for patch-level relevance scoring"""
        with torch.no_grad():
            # Preprocess patches to CLIP input format
            clip_inputs = torch.stack([self.preprocess(p) for p in image_patches])

            # Get patch embeddings from CLIP vision encoder
            patch_features = self.clip_model.encode_image(clip_inputs)
            patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)

        return patch_features  # (num_patches, 768) for ViT-L
```

**Query-Aware Relevance with CLIP Text Encoder:**

```python
# arr_coc/knowing.py - Participatory knowing (query-content coupling)

class ParticipatoryScorer:
    """Measure relevance through query-content alignment"""

    def __init__(self, clip_model):
        self.clip_model = clip_model
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')

    def score(self, patch_features, query_text):
        """Compute CLIP similarity for query-aware relevance"""
        # Encode query text
        query_tokens = self.tokenizer([query_text])
        query_features = self.clip_model.encode_text(query_tokens)
        query_features = query_features / query_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity: higher = more relevant
        relevance_scores = (patch_features @ query_features.T).squeeze()

        return relevance_scores  # (num_patches,)
```

**Relevance-Driven Token Allocation:**

ARR-COC uses CLIP features to implement transjective relevance:
- **Propositional knowing**: CLIP captures statistical co-occurrence patterns from 400M+ image-text pairs
- **Perspectival knowing**: CLIP embeddings represent semantic salience landscape
- **Participatory knowing**: CLIP text-image similarity quantifies query-content coupling

**Variable LOD based on CLIP relevance:**
- High CLIP similarity → Allocate 200-400 tokens (detailed processing)
- Medium CLIP similarity → Allocate 100-200 tokens (moderate detail)
- Low CLIP similarity → Allocate 64-100 tokens (minimal processing)

**Why CLIP for ARR-COC:**
1. **Zero-shot generalization**: No fine-tuning needed for new domains
2. **Semantic alignment**: Text-image joint embedding space enables query-driven relevance
3. **Robust features**: Pre-trained on 400M pairs → generalizes to diverse visual concepts
4. **Frozen encoder**: Preserves zero-shot capabilities while enabling relevance computation

## Training CLIP from Scratch

### Data Preparation

From [OpenCLIP documentation](https://github.com/mlfoundations/open_clip):

**WebDataset format (.tar shards):**
```
training_data/
├── shard_00000.tar
│   ├── 00000.jpg
│   ├── 00000.txt
│   ├── 00001.jpg
│   ├── 00001.txt
│   └── ...
├── shard_00001.tar
└── ...
```

**CSV format (for smaller datasets):**
```csv
filepath,caption
/data/images/cat.jpg,"a photo of a cat"
/data/images/dog.jpg,"a dog playing in the park"
```

### Training Command (Single Node, 4 GPUs)

```bash
cd open_clip/src
torchrun --nproc_per_node 4 -m open_clip_train.main \
    --train-data '/data/laion400m/shards-{00000..41455}.tar' \
    --train-num-samples 400000000 \
    --dataset-type webdataset \
    --batch-size 320 \
    --precision amp \
    --workers 4 \
    --warmup 2000 \
    --epochs 32 \
    --lr 5e-4 \
    --model ViT-B-32 \
    --gather-with-grad \
    --local-loss \
    --imagenet-val /data/imagenet/val
```

**Key hyperparameters:**
- Batch size: 320 per GPU → 1280 effective (4 GPUs)
- Learning rate: 5e-4 with cosine decay
- Warmup steps: 2000 (gradual learning rate increase)
- Precision: Automatic Mixed Precision (AMP)
- Loss flags: `--gather-with-grad --local-loss` for memory efficiency

### Multi-Node SLURM Training (32 nodes × 4 GPUs)

```bash
#!/bin/bash
#SBATCH --nodes=32
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4

srun --cpu_bind=v --accel-bind=gn python -u src/open_clip_train/main.py \
    --save-frequency 1 \
    --train-data="/data/LAION-2B/{00000..41455}.tar" \
    --train-num-samples 2300000000 \
    --warmup 10000 \
    --batch-size=256 \
    --epochs=32 \
    --workers=8 \
    --model ViT-L-14 \
    --local-loss \
    --gather-with-grad
```

**Scaling to 128 GPUs:**
- Effective batch size: 256 × 128 = 32,768
- Training time: ~2 weeks for LAION-2B (32B samples seen)
- Checkpoint frequency: Every epoch (~72B samples / 32k batch = ~2.2M steps)

### Key Training Techniques

**1. Gradient Accumulation:**
```bash
--accum-freq 4  # Simulate 4× larger batch size
# Effective batch = batch-size × accum-freq × num_gpus
```

**2. Patch Dropout (2-3× speedup):**
```bash
--force-patch-dropout 0.5  # Drop 50% of visual tokens during training
# Final fine-tuning: --force-patch-dropout 0.0
```

**3. Multiple Data Sources:**
```bash
--train-data "CC12M/shards.tar::LAION-400M/shards.tar" \
--train-data-upsampling-factors "1::10"  # Upsample LAION 10×
```

## Zero-Shot Classification with CLIP

### Inference Pattern

```python
import torch
from PIL import Image
import open_clip

# Load model
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14', pretrained='laion2b_s32b_b82k'
)
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-L-14')

# Prepare image
image = preprocess(Image.open("photo.jpg")).unsqueeze(0)

# Define classes via text prompts
class_names = ["a dog", "a cat", "a bird"]
text = tokenizer(class_names)

# Compute embeddings
with torch.no_grad(), torch.autocast("cuda"):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # Normalize
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Probabilities:", similarity.cpu().numpy())
# Output: [[0.92, 0.05, 0.03]] (92% dog)
```

### Prompt Engineering for Better Zero-Shot

**Template ensembling:**
```python
templates = [
    "a photo of a {}",
    "a picture of a {}",
    "an image of a {}",
    "{} in the wild",
    "a rendering of a {}"
]

# Ensemble over templates
text_features_list = []
for template in templates:
    prompts = [template.format(cls) for cls in class_names]
    text_tokens = tokenizer(prompts)
    text_features_list.append(model.encode_text(text_tokens))

# Average text features across templates
text_features = torch.stack(text_features_list).mean(dim=0)
text_features /= text_features.norm(dim=-1, keepdim=True)
```

**Performance impact:**
- Single template: 68.3% ImageNet accuracy (CLIP ViT-L/14)
- 80 templates ensembled: 76.2% ImageNet accuracy (+7.9%)

## Limitations and Failure Modes

### Systematic Weaknesses

From [CLIP: Connecting text and images](https://openai.com/index/clip/):

**1. Abstract/Counting Tasks:**
- Counting objects: Only slightly better than random guessing
- Spatial reasoning: Struggles with "nearest", "farthest", "above", "below"
- MNIST handwritten digits: 88% accuracy (vs 99.75% human)

**2. Fine-Grained Classification:**
- Car models (Stanford Cars): Moderate performance
- Aircraft variants (FGVC Aircraft): Poor discrimination
- Flower species (Oxford Flowers): Needs fine-tuning

**3. Out-of-Distribution Generalization:**
- Trained on web photos → poor on handwritten, sketches, abstract art
- Domain shift larger than typical ImageNet → target dataset

**4. Prompt Sensitivity:**
- "a photo of a dog" vs "dog" can yield different results
- Requires prompt engineering for optimal performance
- No guaranteed best prompt for arbitrary tasks

### Computational Limitations

**Training cost:**
- ViT-L/14 on 400M pairs: ~2 weeks on 256 V100 GPUs
- Estimated ~$50k-100k compute cost for full CLIP training
- Data curation: Significant engineering effort

**Inference cost:**
- ViT-L/14: ~15ms per image (single A100)
- Text encoding amortized for classification (encode once, reuse)
- Large models (ViT-H, ViT-G) require high-end GPUs

## State-of-the-Art CLIP Variants (2024-2025)

### SigLIP (Sigmoid Loss for Language-Image Pre-training)

From web research ([SigLIP paper](https://arxiv.org/abs/2303.15343)):
- Replaces softmax with sigmoid loss (pairwise binary classification)
- Better batch efficiency: Scales to larger batch sizes
- **ViT-SO400M-14-SigLIP-384**: 83.1% ImageNet zero-shot (trained on WebLI)
- **ViT-gopt-16-SigLIP2-384**: 85.0% ImageNet zero-shot (multilingual WebLI)

### DFN (Depth-aware Feature Network)

From web research ([DFN paper](https://arxiv.org/abs/2309.17425)):
- 5B parameter training dataset (DFN-5B)
- **ViT-L-14-DFN**: 82.2% ImageNet zero-shot (39B samples)
- **ViT-H-14-378-quickgelu-DFN**: 84.4% ImageNet zero-shot at 378px resolution

### PE (Position Encoding) Core

From web research ([PE paper](https://arxiv.org/abs/2504.13181)):
- Advanced position encoding for vision transformers
- **PE-Core-L-14-336**: 83.5% ImageNet zero-shot (MetaCLIP-5.4B data)
- **PE-Core-bigG-14-448**: 85.4% ImageNet zero-shot (current SOTA, 448px resolution)

## Sources

**Source Documents:**
- [distributed-training/02-megatron-lm-tensor-parallelism.md](../distributed-training/02-megatron-lm-tensor-parallelism.md)
- [inference-optimization/02-triton-inference-server.md](../inference-optimization/02-triton-inference-server.md)
- [alternative-hardware/02-intel-oneapi-ml.md](../alternative-hardware/02-intel-oneapi-ml.md)

**Web Research:**
- [OpenCLIP GitHub Repository](https://github.com/mlfoundations/open_clip) (accessed 2025-01-16)
- [CLIP: Connecting text and images - OpenAI](https://openai.com/index/clip/) (accessed 2025-01-16)
- [Demystifying CLIP Data - arXiv:2309.16671](https://arxiv.org/abs/2309.16671) (accessed 2025-01-16)
- [OpenCLIP HuggingFace Models](https://huggingface.co/models?library=open_clip) (accessed 2025-01-16)
- [F-VLM: Open-vocabulary object detection - Google Research](https://research.google/blog/f-vlm-open-vocabulary-object-detection-upon-frozen-vision-and-language-models/) (accessed 2025-01-16)

**ARR-COC-0-1 Implementation:**
- [arr_coc/texture.py](../../../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/texture.py)
- [arr_coc/knowing.py](../../../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/knowing.py)

**Additional References:**
- SigLIP: Sigmoid Loss for Language-Image Pre-training - [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)
- DFN: Learning Visual Representations from Paired Images and Text - [arXiv:2309.17425](https://arxiv.org/abs/2309.17425)
- Understanding Fine-tuning CLIP for Open-vocabulary Semantic Segmentation - [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Peng_Understanding_Fine-tuning_CLIP_for_Open-vocabulary_Semantic_Segmentation_in_Hyperbolic_Space_CVPR_2025_paper.pdf)
