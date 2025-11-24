# Late Fusion VLM Architectures: Project and Concatenate

**Vision-language models that process vision and language separately, then merge via simple projection layers**

---

## Overview

Late fusion represents the **simplest and most widely adopted** VLM architecture pattern: encode visual features independently, project them into the language model's embedding space via a learnable transformation (typically MLP), then concatenate with text tokens for joint processing by the LLM.

**Core principle**: Leverage powerful pretrained components (vision encoder + LLM) with minimal architectural coupling. The fusion happens "late" — after separate encoding but before LLM processing.

From [Design choices for Vision Language Models in 2024](https://huggingface.co/blog/gigant/vlm-design) (HuggingFace, accessed 2025-11-14):
> "The authors call this mapping the '_projection_', and it is trained on image/caption pairs while keeping the vision and language models frozen."

**Advantages of late fusion**:
- **Simplicity**: Minimal new parameters (just projection layer)
- **Modularity**: Swap vision encoders or LLMs independently
- **Training efficiency**: Freeze pretrained components, train only projection
- **No information loss**: All visual tokens preserved (unlike Q-Former compression)

**Disadvantages**:
- **Token explosion**: High visual token counts (576-9,792 tokens)
- **Context pressure**: Visual tokens consume LLM context budget
- **No cross-modal interaction during encoding**: Vision and language don't "talk" until LLM

---

## Section 1: Late Fusion Principles

### 1.1 The Late Fusion Pipeline

**Standard late fusion workflow**:
```
Input: Image + Text Query
    ↓
[Vision Encoder] → Visual Features [N, D_vision]
    ↓
[Projection Layer] → Projected Features [N, D_llm]
    ↓
[Concatenation] → [Visual Tokens | Text Tokens]
    ↓
[LLM] → Generated Response
```

**Key characteristics**:
1. **Independent encoding**: Vision encoder processes image without text context
2. **Dimension alignment**: Projection maps vision features → LLM embedding space
3. **Token concatenation**: Visual + text tokens form unified sequence
4. **Joint processing**: LLM attends over both modalities simultaneously

From [Understanding Vision-Language Models](https://medium.com/@pietrobolcato/understanding-vision-language-models-vlms-a-practical-guide-8da18e9f0e0c) (Medium, accessed 2025-11-14):
> "This is typically done by a smaller module called the Projection or Fusion Layer: Usually implemented as a small Transformer block or a simple MLP."

### 1.2 Why "Late" Fusion?

**Comparison with fusion timing**:

| Fusion Type | When Fusion Occurs | Example |
|-------------|-------------------|---------|
| **Early fusion** | Before/during encoding | VisualBERT (joint encoder) |
| **Mid fusion** | Between encoders | BLIP-2 Q-Former |
| **Late fusion** | After encoding, before LLM | LLaVA, Qwen-VL |

From [Rohit Bandaru - Vision Language Models](https://rohitbandaru.github.io/blog/Vision-Language-Models/) (accessed 2025-11-14):
> "_Late fusion_ wouldn't be effective for VLMs since the LLM needs access to visual data earlier in the pipeline to generate appropriate text..."

**Note**: Despite this quote's skepticism, late fusion **dominates** modern VLMs (LLaVA family, Qwen-VL, InternVL) due to simplicity and strong empirical results.

### 1.3 Historical Context

**Evolution of late fusion**:

**2023 - LLaVA breakthrough**: Demonstrated that simple 2-layer MLP projection achieves competitive performance with minimal training.

**2024 - Scaling success**: LLaVA-1.5, LLaVA-NeXT, Qwen-VL prove late fusion scales to 13B-72B parameters.

**2025 - Dominant pattern**: Most open-source VLMs adopt late fusion (InternVL, DeepSeek-VL, Yi-VL).

**Why late fusion won**:
- Pretrained LLMs already excel at multimodal reasoning (text includes descriptions of visual concepts)
- Simple projection layer learns strong vision-language alignment with <1% of total parameters
- Engineering simplicity enables rapid iteration and experimentation

---

## Section 2: LLaVA Projector Architecture

### 2.1 LLaVA: The Late Fusion Pioneer

**LLaVA** (Large Language and Vision Assistant, Liu et al. 2023) established the late fusion paradigm:

**Architecture**:
- **Vision encoder**: CLIP ViT-L/14-336 (frozen)
- **Projection**: 2-layer MLP (trainable)
- **LLM**: Vicuna-7B/13B (initially frozen, later fine-tuned)

From [LLaVA Architecture: From Frozen ViT to Fine-Tuned LLM](https://learnopencv.com/llava-training-a-visual-assistant/) (LearnOpenCV, accessed 2025-11-14):
> "A model that _bridges vision and language_ through a tightly coupled architecture built for practical reasoning across modalities."

**LLaVA-1.0 (NeurIPS 2023)**:
```python
# Original LLaVA projection: Single linear layer
class LLaVAProjector(nn.Module):
    def __init__(self, vision_dim=1024, llm_dim=4096):
        super().__init__()
        self.projection = nn.Linear(vision_dim, llm_dim)

    def forward(self, vision_features):
        # vision_features: [B, N_patches, 1024] from CLIP ViT-L/14
        # Output: [B, N_patches, 4096] for Vicuna-7B
        return self.projection(vision_features)
```

**Token flow**:
- CLIP ViT-L/14-336: 336×336 image → 24×24 = 576 tokens
- Each token: 1024-dim CLIP features
- Projection: 1024 → 4096 (Vicuna embedding dim)
- Total visual tokens: 576 × 4096 dims

### 2.2 LLaVA-1.5: The MLP Upgrade

From [Talk To Your Image — A Step-by-Step LLaVa-1.5](https://www.datadriveninvestor.com/2023/10/24/talk-to-your-image-a-step-by-step-llava-1-5/) (DataDrivenInvestor, accessed 2025-11-14):
> "The transition from a linear _projection_ to a two-layer _MLP_ significantly enhances _LLaVA_-1.5's multimodal capabilities."

**LLaVA-1.5 improvement**: Replace single linear layer with 2-layer MLP + GeLU activation.

```python
class LLaVA15Projector(nn.Module):
    """
    LLaVA-1.5 MLP projector (2-layer with GeLU).

    Key improvement: Non-linearity enables richer vision-language alignment.
    """
    def __init__(self, vision_dim=1024, llm_dim=4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, vision_features):
        # vision_features: [B, 576, 1024]
        # Output: [B, 576, 4096]
        return self.mlp(vision_features)
```

**Why 2-layer MLP matters**:
1. **Non-linearity**: GeLU enables complex feature transformations
2. **Capacity**: 2× parameters vs single linear (1024×4096 + 4096×4096)
3. **Empirical gains**: +2-3% on VQA benchmarks (78.5% vs 76.2% for linear)

**Parameter count**:
- Linear projection: 1024 × 4096 = 4.2M parameters
- MLP projection: (1024 × 4096) + (4096 × 4096) = 20.9M parameters
- Still <0.3% of total model parameters (Vicuna-7B = 7B params)

### 2.3 LLaVA Projection Variants

From [Understanding the Multi-modal Projector in LLaVA](https://medium.com/@mlshark/understanding-the-multi-modal-projector-in-llava-d1bc89debbd5) (Medium, accessed 2025-11-14):
> "In the code, there are at least three types of _projectors_: linear, identity, and multi-linear layers with GeLU such as mlp4x-gelu, mlp5x-gelu."

**LLaVA projector options** (from codebase):

```python
class MultiModalProjector(nn.Module):
    """
    Flexible projector supporting multiple architectures.

    Types:
    - linear: Single linear layer (LLaVA-1.0)
    - mlp2x-gelu: 2-layer MLP (LLaVA-1.5)
    - mlp4x-gelu: 4-layer MLP (experimental)
    - identity: No projection (for debugging)
    """
    def __init__(self, projector_type, vision_dim, llm_dim):
        super().__init__()

        if projector_type == 'linear':
            self.layers = nn.Linear(vision_dim, llm_dim)

        elif projector_type == 'mlp2x-gelu':
            self.layers = nn.Sequential(
                nn.Linear(vision_dim, llm_dim),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim)
            )

        elif projector_type == 'mlp4x-gelu':
            # Deeper projection (rarely used - overfitting risk)
            hidden_dim = llm_dim
            self.layers = nn.Sequential(
                nn.Linear(vision_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, llm_dim)
            )

        elif projector_type == 'identity':
            # For debugging when vision_dim == llm_dim
            self.layers = nn.Identity()

    def forward(self, x):
        return self.layers(x)
```

**Empirical findings**:
- **mlp2x-gelu** is sweet spot (LLaVA-1.5 default)
- Deeper MLPs (4x, 5x) show marginal gains, risk overfitting
- Linear projection still competitive for smaller models (<3B params)

---

## Section 3: Image Slicing and Grid Tokenization

### 3.1 High-Resolution Challenge

**Problem**: CLIP ViT-L/14-336 trained on 336×336 images. High-res images (1024×1024+) lose detail when downsampled.

From [LLaVA Image Grid Slicing Strategy](../karpathy/vision-language-architectures/implementations/06-llava-image-slicing.md) (accessed 2025-11-14):
> "LLaVA pioneered a pragmatic approach to high-resolution image processing: instead of increasing the vision encoder's native resolution, split high-resolution images into a grid of patches."

**Solution**: Grid slicing — partition high-res image into 336×336 patches, process each independently, concatenate features.

### 3.2 Grid Slicing Strategy

**Basic grid partition**:
```python
def slice_image_grid(image, grid_size=(2, 2), patch_size=336):
    """
    Slice high-resolution image into grid of patches.

    Args:
        image: [3, H, W] input image
        grid_size: (rows, cols) grid dimensions
        patch_size: Size of each patch (336 for CLIP)

    Returns:
        patches: List of [3, patch_size, patch_size] tensors
    """
    C, H, W = image.shape
    rows, cols = grid_size

    patches = []
    patch_h = H // rows
    patch_w = W // cols

    for i in range(rows):
        for j in range(cols):
            y_start = i * patch_h
            x_start = j * patch_w
            y_end = y_start + patch_size
            x_end = x_start + patch_size

            # Extract and resize patch to 336×336
            patch = image[:, y_start:y_end, x_start:x_end]
            if patch.shape[1:] != (patch_size, patch_size):
                patch = F.interpolate(
                    patch.unsqueeze(0),
                    size=(patch_size, patch_size),
                    mode='bilinear'
                ).squeeze(0)

            patches.append(patch)

    return patches
```

**Token count explosion**:

| Grid Size | Image Resolution | Patches | Tokens (no global) | Tokens (+ global) |
|-----------|------------------|---------|-------------------|-------------------|
| 1×1 | 336×336 | 1 | 576 | 576 |
| 2×2 | 672×672 | 4 | 2,304 | 2,880 |
| 3×3 | 1008×1008 | 9 | 5,184 | 5,760 |
| 4×4 | 1344×1344 | 16 | 9,216 | 9,792 |

From [LLaVA-NeXT Blog](https://llava-vl.github.io/blog/2024-01-30-llava-next/) (accessed 2025-11-14):
> "LLaVA-NeXT can now process 4× more pixels than LLaVA-1.5, effectively handling images up to 672×672 with 2×2 grid (2,880 tokens) or 1008×1008 with 3×3 grid (5,760 tokens)."

### 3.3 Dynamic Resolution and Adaptive Grids

**LLaVA-UHD: Aspect ratio awareness**

From [Beyond LLaVA-HD: Diving into High-Resolution Large Multimodal Models](https://arxiv.org/abs/2406.08487) (accessed 2025-11-14):
> "Initially, we explore various grid options for slicing images, similar to LLaVA-Next, but with finer granularity. We investigate how different grid configurations (2×2, 3×3, 4×4) affect performance across various image resolutions."

**Adaptive grid selection**:
```python
def adaptive_grid_selection(image_height, image_width, base_patch=336):
    """
    Select grid size based on image aspect ratio and dimensions.

    Strategy:
    - Maintain aspect ratio (minimize padding)
    - Optimize for minimal wasted patches
    - Respect max token budget
    """
    aspect_ratio = image_width / image_height

    # Calculate patches needed along each dimension
    patches_h = (image_height + base_patch - 1) // base_patch
    patches_w = (image_width + base_patch - 1) // base_patch

    # Clamp to reasonable limits (4×4 max)
    patches_h = min(patches_h, 4)
    patches_w = min(patches_w, 4)

    return (patches_h, patches_w)

# Examples:
adaptive_grid_selection(1080, 1920)  # → (4, 6) for 16:9 video
adaptive_grid_selection(800, 800)    # → (3, 3) for square
adaptive_grid_selection(1200, 600)   # → (4, 2) for panorama
```

**Token budget management**:
- **Small images (<672px)**: 1×1 grid (576 tokens)
- **Medium images (672-1008px)**: 2×2 grid (2,880 tokens)
- **Large images (1008-1344px)**: 3×3 grid (5,760 tokens)
- **XL images (>1344px)**: 4×4 grid (9,792 tokens) or task-specific compression

---

## Section 4: Token Concatenation Strategies

### 4.1 Concatenation Order and Patterns

From [Vision-Language Model Token Concatenation Strategies](../karpathy/vision-language/00-token-concatenation-strategies.md) (accessed 2025-11-14):
> "The order and method of concatenation directly impacts context window utilization, attention efficiency, cross-modal reasoning, and task performance."

**Standard concatenation patterns**:

**Pattern 1: Prefix (most common)**
```python
# Visual tokens first, then text
sequence = [visual_tokens_1, visual_tokens_2, ..., text_tokens]
# Total: N_visual + N_text tokens

# Example (LLaVA):
# [576 visual] + [10 text] = 586 tokens
```

**Pattern 2: Interleaved (multi-image)**
```python
# Alternate images and text
sequence = [visual_1, text_1, visual_2, text_2, ...]

# Example (multi-image reasoning):
# [img1: 576] + ["Compare this"] + [img2: 576] + ["to this"]
# Total: 1,152 visual + ~5 text = 1,157 tokens
```

**Pattern 3: Suffix (rare)**
```python
# Text first, then visual
sequence = [text_tokens, visual_tokens]

# Use case: Pre-specified questions before image
```

### 4.2 Multi-Image Concatenation

**Handling multiple images in single sequence**:

```python
def process_multi_image_sequence(images, texts, vision_encoder, projector):
    """
    Process interleaved image-text sequence.

    Args:
        images: List of image tensors
        texts: List of text strings (len = len(images) or len(images)+1)
        vision_encoder: CLIP ViT
        projector: MLP projection layer

    Returns:
        tokens: Concatenated visual + text token sequence
    """
    tokens = []

    for i, image in enumerate(images):
        # Process image
        with torch.no_grad():
            visual_features = vision_encoder(image.unsqueeze(0))
        projected = projector(visual_features)  # [1, 576, 4096]

        tokens.append(projected.squeeze(0))  # Add visual tokens

        # Add text if available
        if i < len(texts):
            text_tokens = tokenize_text(texts[i])
            tokens.append(text_tokens)

    # Concatenate all tokens
    return torch.cat(tokens, dim=0)  # [N_total, 4096]
```

**Token budget example** (multi-image VQA):
```
Query: "What changed between these three images?"
Image 1: 672×672 → 2,304 tokens (2×2 grid)
Text: "What changed"
Image 2: 672×672 → 2,304 tokens
Text: "between these"
Image 3: 672×672 → 2,304 tokens
Text: "three images?"

Total: (2,304 × 3) + (~10 text) = 6,922 tokens
```

### 4.3 Position Encoding for Visual Tokens

**Challenge**: Visual tokens lack inherent position information after projection.

**Solution 1: Learnable position embeddings**
```python
class VisualPositionEmbedding(nn.Module):
    """
    Add learnable 2D position embeddings to grid patches.
    """
    def __init__(self, grid_size=(2, 2), hidden_dim=4096):
        super().__init__()
        rows, cols = grid_size
        # Learnable embedding for each patch position
        self.pos_embed = nn.Parameter(
            torch.randn(rows * cols, hidden_dim) * 0.02
        )

    def forward(self, visual_tokens):
        # visual_tokens: [B, N_patches × 576, 4096]
        # Add position info (broadcast across patch tokens)
        return visual_tokens + self.pos_embed.unsqueeze(1)
```

**Solution 2: No explicit position encoding**
- Many models (LLaVA) omit visual position embeddings
- Rely on CLIP's internal position encoding (preserved through projection)
- LLM learns spatial relationships from attention patterns

---

## Section 5: Tensor Parallelism for Large ViT + LLM

**Influenced by**: [Megatron-LM Tensor Parallelism](../karpathy/distributed-training/02-megatron-lm-tensor-parallelism.md)

### 5.1 Why Tensor Parallelism for Late Fusion

**Problem**: Large VLMs don't fit on single GPU:
- Vision encoder: ViT-H/14 (632M params) or EVA-CLIP-8B (8.7B params)
- LLM: LLaMA-70B (70B params) or Qwen-72B (72B params)
- Total: 72.7B parameters ≈ 145GB FP16 (exceeds A100 80GB)

From [Megatron-LM Paper](https://arxiv.org/abs/1909.08053) (Shoeybi et al., 2019):
> "We introduce tensor parallelism to split the computation of a single transformer layer across multiple GPUs, achieving near-linear scaling."

**Tensor parallel strategy for late fusion VLM**:
1. **Column-parallel vision encoder**: Split ViT attention heads across GPUs
2. **Row-parallel projection**: Split MLP projection weights by rows
3. **Column-parallel LLM**: Split LLM FFN and attention across GPUs

### 5.2 Column-Parallel Vision Encoder

**Split CLIP ViT attention heads** across GPUs:

```python
# Standard ViT attention (single GPU):
# 32 heads, head_dim = 64, hidden = 2048
Q = X @ W_Q  # [B, N, 2048] @ [2048, 2048]
K = X @ W_K
V = X @ W_V

# Tensor parallel (TP=4):
# GPU 0: heads 0-7   (512 dims)
# GPU 1: heads 8-15  (512 dims)
# GPU 2: heads 16-23 (512 dims)
# GPU 3: heads 24-31 (512 dims)

# Each GPU computes subset of heads independently
Q_local = X @ W_Q_shard  # [B, N, 2048] @ [2048, 512]
# No communication needed during forward!
```

**Benefits**:
- Each GPU holds 1/4 of vision encoder weights
- Parallel computation (no communication during Q/K/V projection)
- Single all-reduce after attention output projection

### 5.3 Row-Parallel Projection Layer

**Split MLP projection by rows**:

```python
# MLP projection: [N, 1024] → [N, 4096]
# TP=4: Each GPU has 1/4 of output dimensions

# GPU 0: W_proj[0:1024, :]      outputs dims 0-1023
# GPU 1: W_proj[1024:2048, :]   outputs dims 1024-2047
# GPU 2: W_proj[2048:3072, :]   outputs dims 2048-3071
# GPU 3: W_proj[3072:4096, :]   outputs dims 3072-4095

# Forward:
Y_local = vision_features @ W_proj_shard  # [576, 1024] @ [1024, 1024]
# All-reduce to sum partial results
Y = AllReduce(Y_local)  # [576, 4096]
```

**Communication**: Single all-reduce per MLP layer (efficient with NVLink).

### 5.4 Complete Tensor Parallel VLM Pipeline

```python
class TensorParallelVLM(nn.Module):
    """
    Tensor parallel late fusion VLM.

    Architecture:
    - Column-parallel ViT
    - Row-parallel projection
    - Column-parallel LLM
    """
    def __init__(self, tp_size=4):
        super().__init__()
        self.tp_size = tp_size
        self.rank = torch.distributed.get_rank()

        # Column-parallel vision encoder
        self.vision_encoder = ColumnParallelViT(
            num_heads=32,
            heads_per_rank=32 // tp_size  # 8 heads per GPU
        )

        # Row-parallel projection
        self.projector = RowParallelMLP(
            in_features=1024,
            out_features=4096,
            tp_size=tp_size
        )

        # Column-parallel LLM
        self.llm = ColumnParallelLLM(
            hidden_size=4096,
            num_layers=32,
            tp_size=tp_size
        )

    def forward(self, images, text_tokens):
        # Vision encoding (column-parallel, no communication)
        visual_features = self.vision_encoder(images)  # [B, 576, 1024]

        # Projection (row-parallel, 1 all-reduce)
        visual_tokens = self.projector(visual_features)  # [B, 576, 4096]

        # Concatenate visual + text
        inputs = torch.cat([visual_tokens, text_tokens], dim=1)

        # LLM generation (column-parallel)
        outputs = self.llm(inputs)

        return outputs
```

**Memory savings** (TP=4 for LLaVA-70B):
- Single GPU: ~145GB FP16 (exceeds A100 80GB)
- TP=4: ~36GB per GPU (fits on A100 40GB)

---

## Section 6: Triton Serving for Multi-Model VLM Pipeline

**Influenced by**: [Triton Inference Server](../karpathy/inference-optimization/02-triton-inference-server.md)

### 6.1 VLM as Ensemble Pipeline

Late fusion VLMs are natural **model ensembles**:
1. Vision encoder (CLIP/SigLIP)
2. Projection layer (MLP)
3. Language model (LLaMA/Vicuna/Qwen)

From [NVIDIA Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html) (accessed 2025-11-13):
> "Triton Inference Server is an open source inference serving software that streamlines AI inferencing."

**Triton ensemble advantages**:
- **Independent scaling**: Scale vision encoder separately from LLM
- **Mixed precision**: FP16 vision encoder, INT4 quantized LLM
- **Dynamic batching**: Batch vision encoding separately from text generation

### 6.2 Triton Ensemble Configuration

**Model repository structure**:
```
models/
├── clip_vision_encoder/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
├── mlp_projector/
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan  # TensorRT optimized
└── llm_generator/
    ├── config.pbtxt
    └── 1/
        └── model.plan  # vLLM backend
```

**Ensemble config** (`vlm_pipeline/config.pbtxt`):
```protobuf
name: "vlm_pipeline"
platform: "ensemble"
max_batch_size: 8

input [
  {
    name: "IMAGE"
    data_type: TYPE_FP32
    dims: [ 3, 336, 336 ]
  },
  {
    name: "TEXT_TOKENS"
    data_type: TYPE_INT64
    dims: [ -1 ]  # Variable length
  }
]

output [
  {
    name: "GENERATED_TEXT"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "clip_vision_encoder"
      model_version: -1
      input_map {
        key: "input_image"
        value: "IMAGE"
      }
      output_map {
        key: "visual_features"
        value: "vision_features_internal"
      }
    },
    {
      model_name: "mlp_projector"
      model_version: -1
      input_map {
        key: "vision_features"
        value: "vision_features_internal"
      }
      output_map {
        key: "projected_features"
        value: "visual_tokens_internal"
      }
    },
    {
      model_name: "llm_generator"
      model_version: -1
      input_map {
        key: "visual_tokens"
        value: "visual_tokens_internal"
      }
      input_map {
        key: "text_tokens"
        value: "TEXT_TOKENS"
      }
      output_map {
        key: "output_text"
        value: "GENERATED_TEXT"
      }
    }
  ]
}
```

### 6.3 Dynamic Batching for Vision Encoding

**Vision encoder batching** (faster than LLM generation):

```protobuf
# clip_vision_encoder/config.pbtxt
name: "clip_vision_encoder"
backend: "onnxruntime"
max_batch_size: 32

dynamic_batching {
  preferred_batch_size: [ 8, 16 ]
  max_queue_delay_microseconds: 1000  # Wait 1ms for batch
}

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0, 1 ]  # Replicate on 2 GPUs
  }
]
```

**Performance impact**:
- Without batching: 32 requests → 32 forward passes (slow)
- With batching (batch=16): 32 requests → 2 forward passes (2× faster)
- Vision encoding is compute-bound → batching critical

---

## Section 7: Kubeflow Pipelines for VLM Training Workflows

**Influenced by**: [Kubeflow ML Pipelines](../karpathy/orchestration/01-kubeflow-ml-pipelines.md)

### 7.1 Late Fusion Training Pipeline

**Multi-stage training** (LLaVA approach):
1. **Stage 1 - Projection pretraining**: Train projection layer only (vision + LLM frozen)
2. **Stage 2 - Instruction tuning**: Fine-tune projection + LLM (vision frozen)
3. **Stage 3 - High-res fine-tuning**: Train with grid slicing (optional)

From [Implement MLOps with Kubeflow Pipelines](https://developers.redhat.com/articles/2024/01/25/implement-mlops-kubeflow-pipelines) (Red Hat Developer, 2024):
> "Kubeflow Pipelines is an open source platform for implementing MLOps, providing a framework for building, deploying, and managing machine learning workflows."

**Kubeflow pipeline definition**:
```python
import kfp
from kfp import dsl

@dsl.pipeline(
    name='llava-training-pipeline',
    description='Multi-stage late fusion VLM training'
)
def llava_training_pipeline(
    dataset_path: str,
    vision_encoder_model: str = 'openai/clip-vit-large-patch14-336',
    llm_model: str = 'lmsys/vicuna-7b-v1.5',
    learning_rate_stage1: float = 1e-3,
    learning_rate_stage2: float = 2e-5
):
    # Stage 1: Projection pretraining
    pretrain_projection = train_projection_op(
        dataset=dataset_path,
        vision_model=vision_encoder_model,
        llm_model=llm_model,
        lr=learning_rate_stage1,
        epochs=1,
        batch_size=256
    )

    # Stage 2: Instruction tuning
    instruction_tune = instruction_tuning_op(
        pretrained_projection=pretrain_projection.outputs['checkpoint'],
        instruction_data=dataset_path + '/instruction_data',
        lr=learning_rate_stage2,
        epochs=3,
        batch_size=128
    ).after(pretrain_projection)

    # Stage 3: Evaluation
    evaluate = evaluate_vlm_op(
        model_checkpoint=instruction_tune.outputs['checkpoint'],
        eval_datasets=['vqav2', 'textvqa', 'gqa']
    ).after(instruction_tune)

    return evaluate
```

### 7.2 Distributed Training Component

**PyTorchJob for multi-GPU training**:

```yaml
# Stage 1: Projection pretraining component
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: llava-projection-pretrain
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: nvcr.io/nvidia/pytorch:24.01-py3
            command:
            - python
            - /workspace/train_projection.py
            args:
            - --vision-model=openai/clip-vit-large-patch14-336
            - --llm-model=lmsys/vicuna-7b-v1.5
            - --batch-size=256
            - --learning-rate=1e-3
            - --freeze-vision
            - --freeze-llm
            resources:
              limits:
                nvidia.com/gpu: 1
    Worker:
      replicas: 7  # 8 GPUs total (1 master + 7 workers)
      template:
        spec:
          containers:
          - name: pytorch
            image: nvcr.io/nvidia/pytorch:24.01-py3
            command:
            - python
            - /workspace/train_projection.py
            args:
            - --vision-model=openai/clip-vit-large-patch14-336
            - --llm-model=lmsys/vicuna-7b-v1.5
            - --batch-size=256
            - --learning-rate=1e-3
            resources:
              limits:
                nvidia.com/gpu: 1
```

**Training efficiency**:
- **Stage 1**: 1 epoch on CC3M (3M images) ≈ 8 GPU-hours (A100)
- **Stage 2**: 3 epochs on instruction data (665K samples) ≈ 16 GPU-hours
- **Total training**: <24 GPU-hours for LLaVA-7B (minimal compared to LLM pretraining)

---

## Section 8: ARR-COC-0-1: Relevance-Driven Token Selection Before LLM

**ARR-COC-0-1 integration**: Late fusion with **intelligent token pruning** before LLM concatenation.

### 8.1 The Relevance Realization Challenge for Late Fusion

**Standard late fusion problem**: All visual tokens concatenated → LLM must process irrelevant patches.

From [ARR-COC-VIS README](../../../../README.md) (accessed 2025-11-14):
> "ARR-COC-VIS combines cutting-edge vision models with cognitive science principles to achieve **intelligent, query-aware visual compression** (7-10× token reduction) while maintaining task performance."

**ARR-COC approach**: Apply **Vervaekean relevance realization** before token concatenation:

```
Standard Late Fusion:
Image → ViT → [All 576 tokens] → Projection → LLM

ARR-COC Late Fusion:
Image → ViT → Relevance Realization → [Top-K relevant tokens] → Projection → LLM
                    ↓
            Query-aware selection
            (64-400 tokens per patch)
```

### 8.2 Transjective Relevance for Token Selection

**Three ways of knowing for token selection**:

```python
class RelevanceAwareProjection(nn.Module):
    """
    ARR-COC-0-1: Late fusion with relevance-driven token selection.

    Process:
    1. Vision encoding (standard ViT)
    2. Relevance scoring (3 ways of knowing)
    3. Token selection (top-K most relevant)
    4. Projection to LLM space
    5. Concatenation with text
    """
    def __init__(self, vision_dim=1024, llm_dim=4096, token_budget=256):
        super().__init__()
        self.token_budget = token_budget

        # Relevance scorers (Vervaekean)
        self.shannon_scorer = InformationScorer(vision_dim)  # Propositional
        self.jung_scorer = SymbolicScorer(vision_dim)        # Perspectival
        self.vervaeke_scorer = CouplingScorer(vision_dim)    # Participatory

        # Opponent processing
        self.tension_balancer = TensionBalancer()

        # Standard projection
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, vision_features, query_embedding):
        """
        Args:
            vision_features: [B, 576, 1024] from ViT
            query_embedding: [B, D_text] query representation

        Returns:
            visual_tokens: [B, token_budget, 4096] selected + projected
        """
        # Score relevance (3 ways of knowing)
        info_scores = self.shannon_scorer(vision_features)      # [B, 576]
        symbolic_scores = self.jung_scorer(vision_features)     # [B, 576]
        coupling_scores = self.vervaeke_scorer(
            vision_features, query_embedding
        )  # [B, 576]

        # Balance tensions (opponent processing)
        relevance_map = self.tension_balancer(
            info_scores, symbolic_scores, coupling_scores
        )  # [B, 576]

        # Select top-K tokens
        top_k_indices = torch.topk(
            relevance_map, k=self.token_budget, dim=1
        ).indices  # [B, token_budget]

        # Gather selected tokens
        selected_features = torch.gather(
            vision_features,
            dim=1,
            index=top_k_indices.unsqueeze(-1).expand(-1, -1, vision_features.size(-1))
        )  # [B, token_budget, 1024]

        # Project to LLM space
        visual_tokens = self.projector(selected_features)  # [B, token_budget, 4096]

        return visual_tokens
```

**Token reduction example**:
- Standard LLaVA: 576 tokens (2×2 grid: 2,304 tokens)
- ARR-COC selection: 256 tokens (4.5× reduction)
- Context savings: 1,728 tokens for other uses (longer conversations, more images)

### 8.3 Adaptive Token Budgets

**Variable LOD allocation** per relevance:

```python
def adaptive_token_budget(relevance_scores, base_budget=256, alpha=0.5):
    """
    Allocate variable token budgets based on image complexity.

    High relevance images → more tokens
    Low relevance images → fewer tokens

    Args:
        relevance_scores: [B, 576] relevance map
        base_budget: Average tokens per image
        alpha: Adaptation strength (0 = fixed, 1 = full adaptation)

    Returns:
        token_budgets: [B] per-image token counts
    """
    # Calculate image complexity (mean relevance)
    complexity = relevance_scores.mean(dim=1)  # [B]

    # Normalize complexity to [0.5, 1.5] range
    complexity_normalized = 0.5 + complexity / complexity.max()

    # Allocate budgets
    token_budgets = (base_budget * (1 - alpha) +
                     base_budget * complexity_normalized * alpha)

    # Clamp to reasonable range [64, 400]
    return torch.clamp(token_budgets, 64, 400).int()

# Example:
# Simple image (low relevance): 128 tokens
# Complex image (high relevance): 400 tokens
# Average: 256 tokens (same total budget)
```

**Benefits**:
1. **Efficient context usage**: Simple images use fewer tokens
2. **Preserved detail**: Complex images get more tokens
3. **Query-aware**: Token allocation adapts to question difficulty

---

## Comparison with Alternative Architectures

### Late Fusion vs Mid Fusion vs Early Fusion

| Aspect | Late Fusion (LLaVA) | Mid Fusion (BLIP-2) | Early Fusion (VisualBERT) |
|--------|---------------------|---------------------|---------------------------|
| **When fusion occurs** | After separate encoding | Between encoders (Q-Former) | During joint encoding |
| **Visual token count** | High (576-9,792) | Low (32 queries) | Medium (256-576) |
| **Training complexity** | Simple (2-stage) | Complex (3-stage) | Moderate (joint) |
| **Modularity** | Excellent | Good | Poor |
| **Information preservation** | Complete | Lossy (compression) | Complete |
| **Inference speed** | Slow (many tokens) | Fast (few tokens) | Moderate |
| **Fine-grained tasks** | Excellent | Limited | Good |
| **Best for** | General VQA, OCR | Efficient inference | Multi-modal pretraining |

**Performance comparison** (VQAv2 accuracy):
- LLaVA-1.5 (7B, late fusion): 78.5%
- BLIP-2 (Flan-T5-XXL, mid fusion): 81.9%
- VisualBERT (110M, early fusion): ~70%

**Token efficiency**:
- Late fusion: 576-9,792 visual tokens
- Mid fusion: 32 visual tokens (18× compression)
- Early fusion: 256-576 visual tokens

---

## Key Takeaways

### 1. Simplicity Wins

Late fusion's minimal architecture (vision encoder + projection + LLM) enables rapid experimentation and deployment. LLaVA proved that 2-layer MLP projection is sufficient for strong performance.

### 2. Token Management is Critical

Grid slicing enables high-resolution processing but creates token explosion (9,792 tokens for 4×4 grid). Production systems must carefully balance resolution vs context budget.

### 3. Modularity Enables Innovation

Late fusion's clear separation (vision | projection | LLM) allows independent improvements:
- Swap CLIP → SigLIP for better alignment
- Upgrade LLaMA-7B → Qwen-14B for stronger reasoning
- Add relevance realization (ARR-COC) for efficiency

### 4. Training Efficiency

Minimal trainable parameters (projection layer) enables fast iteration:
- Stage 1 (projection): <1% of total params, trained in hours
- Stage 2 (instruction tuning): Full model, but starting from strong pretrained components

### 5. Production Deployment

Late fusion maps naturally to distributed serving:
- Tensor parallelism: Shard large ViT + LLM across GPUs
- Triton ensembles: Scale vision encoding independently from generation
- Kubeflow pipelines: Orchestrate multi-stage training workflows

### 6. Relevance Realization Enhancement

ARR-COC-0-1 demonstrates late fusion's extensibility: insert intelligent token selection before concatenation for 4-10× context savings without sacrificing task performance.

---

## When to Use Late Fusion

**✅ Use late fusion when:**
- Leveraging pretrained vision encoders + LLMs (minimal new training)
- Need modular architecture (swap components independently)
- Fine-grained visual understanding required (OCR, detailed VQA)
- Rapid prototyping and experimentation
- Sufficient LLM context window (8k+ tokens)

**❌ Avoid late fusion when:**
- Strict token budget (<2k tokens)
- Real-time inference critical (prefer mid fusion compression)
- Training from scratch (consider joint training)
- Extremely resource-constrained deployment

---

## Sources

**Source Documents:**
- [LLaVA Image Grid Slicing Strategy](../karpathy/vision-language-architectures/implementations/06-llava-image-slicing.md) - Grid slicing implementation details
- [Vision-Language Model Token Concatenation Strategies](../karpathy/vision-language/00-token-concatenation-strategies.md) - Concatenation patterns and multi-image handling
- [Megatron-LM Tensor Parallelism](../karpathy/distributed-training/02-megatron-lm-tensor-parallelism.md) - Column/row parallel patterns for large models
- [Triton Inference Server](../karpathy/inference-optimization/02-triton-inference-server.md) - Multi-model ensemble serving
- [Kubeflow ML Pipelines](../karpathy/orchestration/01-kubeflow-ml-pipelines.md) - Distributed training workflows
- [ARR-COC-VIS README](../../../../README.md) - Relevance realization integration

**Web Research:**

**Primary Papers:**
- [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) - NeurIPS 2023, original late fusion architecture
- [Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)](https://arxiv.org/abs/2310.03744) - MLP projection upgrade
- [Beyond LLaVA-HD: Diving into High-Resolution Large Multimodal Models](https://arxiv.org/abs/2406.08487) - Grid slicing analysis (accessed 2025-11-14)
- [Window Token Concatenation for Efficient Visual Large Language Models (WiCo)](https://arxiv.org/abs/2504.04024) - Token reduction via sliding window (accessed 2025-11-14)

**HuggingFace Resources:**
- [Design choices for Vision Language Models in 2024](https://huggingface.co/blog/gigant/vlm-design) - Comparative VLM architecture analysis (accessed 2025-11-14)

**Technical Blogs:**
- [LLaVA Architecture: From Frozen ViT to Fine-Tuned LLM](https://learnopencv.com/llava-training-a-visual-assistant/) - LearnOpenCV tutorial (accessed 2025-11-14)
- [Understanding the Multi-modal Projector in LLaVA](https://medium.com/@mlshark/understanding-the-multi-modal-projector-in-llava-d1bc89debbd5) - Medium deep dive (accessed 2025-11-14)
- [Talk To Your Image — A Step-by-Step LLaVa-1.5](https://www.datadriveninvestor.com/2023/10/24/talk-to-your-image-a-step-by-step-llava-1-5/) - DataDrivenInvestor guide (accessed 2025-11-14)
- [Vision Language Models](https://rohitbandaru.github.io/blog/Vision-Language-Models/) - Rohit Bandaru blog (accessed 2025-11-14)
- [Understanding Vision-Language Models (VLMs)](https://medium.com/@pietrobolcato/understanding-vision-language-models-vlms-a-practical-guide-8da18e9f0e0c) - Medium practical guide (accessed 2025-11-14)

**GitHub Resources:**
- [LLaVA GitHub Repository](https://github.com/haotian-liu/LLaVA) - Official implementation (accessed 2025-11-14)

**Additional References:**
- [LLaVA-NeXT Blog](https://llava-vl.github.io/blog/2024-01-30-llava-next/) - Multi-image and high-resolution upgrades (accessed 2025-11-14)
- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053) - Tensor parallelism fundamentals (Shoeybi et al., 2019)
- [NVIDIA Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html) - Production serving (accessed 2025-11-13)
- [Implement MLOps with Kubeflow Pipelines](https://developers.redhat.com/articles/2024/01/25/implement-mlops-kubeflow-pipelines) - Red Hat Developer (2024)
