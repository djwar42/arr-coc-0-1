# SAM 2 Memory Attention Module

## Section 1: Memory Attention Overview

### Purpose and Role

The Memory Attention module is the **key architectural innovation** that enables SAM 2 to extend from image segmentation to video segmentation. It serves as the bridge between:
- **Spatial understanding**: Current frame features from the image encoder
- **Temporal context**: Historical information stored in the memory bank

From [SAM 2 Paper](https://arxiv.org/abs/2408.00714) (arXiv:2408.00714, August 2024):
> "The role of memory attention is to condition the current frame features on the past frames features and predictions as well as on any new prompts."

### Core Functionality

**What Memory Attention Does:**
1. **Conditions current frame** on historical context
2. **Integrates past predictions** with current frame features
3. **Incorporates prompt information** from previously prompted frames
4. **Produces contextualized embeddings** for the mask decoder

### Position in SAM 2 Pipeline

```
Video Frame
    ↓
Image Encoder (Hiera)
    ↓
Per-frame embeddings (unconditional)
    ↓
╔═════════════════════════════════════════════════════════════
║ MEMORY ATTENTION MODULE
║ ├─ Self-attention on current frame features
║ ├─ Cross-attention to memory bank
║ │   ├─ Recent frame memories (N frames)
║ │   ├─ Prompted frame memories (M frames)
║ │   └─ Object pointers (high-level semantic tokens)
║ └─ MLP projection
╚═════════════════════════════════════════════════════════════
    ↓
Memory-conditioned embeddings
    ↓
Mask Decoder (with prompts)
    ↓
Output Masks + IoU Scores + Occlusion Scores
```

### Why Memory Attention Matters

**Without Memory Attention (SAM 1):**
- Each frame processed independently
- No temporal context
- Cannot track objects across frames
- Cannot handle occlusions or re-appearances

**With Memory Attention (SAM 2):**
- Frames are temporally connected
- Object appearance is tracked over time
- Occlusions can be handled (object "remembered")
- Re-appearances are recognized
- Consistent segmentation across video

### Architecture Overview

From [Towards Data Science Analysis](https://towardsdatascience.com/segment-anything-2-what-is-the-secret-sauce-a-deep-learners-guide-1c43dd07a6f8/) (August 2024):

The Memory Attention block consists of **L transformer blocks** (default L=4):
1. **Self-attention**: Current frame features attend to themselves
2. **Cross-attention**: Current features attend to memory bank contents
3. **MLP**: Final projection and mixing

```python
class MemoryAttention(nn.Module):
    def __init__(self, num_layers=4, hidden_dim=256, num_heads=8):
        self.layers = nn.ModuleList([
            MemoryAttentionLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, frame_features, memory_bank):
        x = frame_features
        for layer in self.layers:
            x = layer(x, memory_bank)
        return x
```

### Streaming Architecture

SAM 2 uses a **streaming approach** for real-time video processing:

1. **Process frames sequentially** as they arrive
2. **Memory bank is updated** after each frame
3. **FIFO queue** maintains recent memories
4. **Constant memory footprint** regardless of video length

This enables:
- Real-time processing (44 FPS on A100)
- Arbitrarily long videos
- Interactive refinement during processing

---

## Section 2: Cross-Attention Mechanism

### Cross-Attention Fundamentals

Cross-attention allows the current frame to "query" information from the memory bank. This is the core mechanism for incorporating temporal context.

**Standard Cross-Attention Formula:**
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

Where:
- Q (Query): Current frame features
- K (Key): Memory bank features
- V (Value): Memory bank features
- d_k: Key dimension for scaling
```

### Memory Attention Cross-Attention Design

From the SAM 2 paper architecture details:

```python
class MemoryAttentionLayer(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8):
        # Self-attention on current frame
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )

        # Cross-attention to memory
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x, memory):
        # Self-attention
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # Cross-attention to memory
        x = x + self.cross_attn(self.norm2(x), memory, memory)[0]

        # MLP
        x = x + self.mlp(self.norm3(x))

        return x
```

### Positional Encoding in Cross-Attention

SAM 2 uses multiple positional encoding strategies:

**1. Absolute Positional Embeddings:**
- Standard sinusoidal encoding
- Applied to all tokens

**2. 2D Rotary Positional Embeddings (RoPE):**
- Used in self-attention and cross-attention layers
- Captures relative spatial relationships
- Better for modeling 2D image structure

From the paper:
> "In addition to sinusoidal absolute positional embeddings, we use 2d spatial Rotary Positional Embedding (RoPE) in self-attention and cross-attention layers."

**Important:** Object pointer tokens are **excluded from RoPE** as they don't have specific spatial correspondence.

### Cross-Attention to Different Memory Types

The cross-attention attends to **three types of memory simultaneously**:

**1. Spatial Memory Features (Recent Frames):**
- Shape: `(N, H*W, C)` where N is number of memories
- Contains spatial feature maps from recent frames
- Provides appearance and location context

**2. Spatial Memory Features (Prompted Frames):**
- Shape: `(M, H*W, C)` where M is number of prompted frames
- Contains feature maps from user-prompted frames
- Preserves prompt information throughout video

**3. Object Pointers:**
- Shape: `(num_frames, num_tokens, C)`
- High-level semantic tokens (not spatial)
- Capture object identity across frames
- Based on mask decoder output tokens

### Efficient Attention Implementation

SAM 2 benefits from efficient attention implementations:

```python
# FlashAttention-2 compatible design
# Removing RPB from image encoder enables this optimization

from flash_attn import flash_attn_func

def efficient_cross_attention(q, k, v, softmax_scale=None):
    """
    Flash Attention implementation for cross-attention

    Args:
        q: Query tensor (batch, seqlen_q, heads, head_dim)
        k: Key tensor (batch, seqlen_k, heads, head_dim)
        v: Value tensor (batch, seqlen_k, heads, head_dim)

    Returns:
        Attention output (batch, seqlen_q, heads, head_dim)
    """
    return flash_attn_func(q, k, v, softmax_scale=softmax_scale)
```

This provides significant speedup at high resolutions (1024x1024).

---

## Section 3: Memory-to-Frame Attention

### How Current Frame Queries Memory

The memory-to-frame attention is the process where current frame features extract relevant information from stored memories.

**Query Generation:**
```python
# Current frame features become queries
current_frame_features = image_encoder(current_frame)  # (B, H*W, C)

# Project to query space
Q = query_projection(current_frame_features)  # (B, H*W, C)
```

**Key-Value Generation from Memory:**
```python
# Memory bank contains:
# - Recent frame memories: (N, H*W, C)
# - Prompted frame memories: (M, H*W, C)
# - Object pointers: (num_frames, num_tokens, C)

# Concatenate all memory sources
all_memories = concatenate([
    recent_memories.reshape(-1, C),      # N*H*W tokens
    prompted_memories.reshape(-1, C),    # M*H*W tokens
    object_pointers.reshape(-1, C)       # num_frames*num_tokens tokens
])

K = key_projection(all_memories)
V = value_projection(all_memories)
```

### Attention Weight Computation

The attention weights determine how much each current frame position attends to each memory position:

```python
def compute_memory_attention(Q, K, V, positional_encodings):
    """
    Compute memory-to-frame attention

    Args:
        Q: Queries from current frame (B, H*W, C)
        K: Keys from memory (B, num_memories, C)
        V: Values from memory (B, num_memories, C)
        positional_encodings: RoPE encodings

    Returns:
        Attended features (B, H*W, C)
    """
    # Apply RoPE to Q and K
    Q_rope = apply_rope_2d(Q, positional_encodings)
    K_rope = apply_rope_2d(K, positional_encodings)

    # Compute attention scores
    d_k = Q.shape[-1]
    scores = torch.matmul(Q_rope, K_rope.transpose(-2, -1)) / math.sqrt(d_k)

    # Softmax normalization
    attn_weights = F.softmax(scores, dim=-1)

    # Weighted sum of values
    output = torch.matmul(attn_weights, V)

    return output, attn_weights
```

### Temporal Position Embedding

SAM 2 embeds temporal information into memories:

**Recent Frames (with temporal encoding):**
```python
# Embed temporal position for recent N frames
for i, memory in enumerate(recent_memories):
    temporal_position = i  # 0 = most recent, N-1 = oldest
    temporal_encoding = get_temporal_encoding(temporal_position)
    memory = memory + temporal_encoding
```

**Prompted Frames (without temporal encoding):**
```python
# No temporal encoding for prompted frames
# Rationale: Training signal is sparser, harder to generalize
# to different temporal ranges at inference
```

From the paper:
> "We embed temporal position information into the memories of N recent frames, allowing the model to represent short-term object motion, but not into those of prompted frames."

### Information Flow

The memory-to-frame attention enables rich information flow:

```
Memory Bank                          Current Frame
┌─────────────────┐                 ┌─────────────┐
│ Recent Frame 1  │ ──┐             │             │
│ Recent Frame 2  │ ──┤             │  Position   │
│ ...             │ ──┼─ Cross ───→ │  (x, y)     │
│ Recent Frame N  │ ──┤   Attn      │             │
│ Prompted Frame 1│ ──┤             │  Receives   │
│ ...             │ ──┤             │  context    │
│ Prompted Frame M│ ──┤             │  from all   │
│ Object Pointer 1│ ──┤             │  memories   │
│ ...             │ ──┘             │             │
└─────────────────┘                 └─────────────┘
```

Each spatial position in the current frame can attend to:
- All spatial positions in all stored memories
- All object pointers
- Weighted by relevance (learned attention)

---

## Section 4: Frame-to-Memory Update

### Memory Encoder

After mask prediction, the memory encoder creates a new memory to add to the bank:

```python
class MemoryEncoder(nn.Module):
    """
    Creates memory from mask prediction and frame features
    """
    def __init__(self, hidden_dim=64):
        # Downsample mask
        self.mask_downsample = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Fusion convolutions
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )

    def forward(self, predicted_mask, frame_embedding):
        """
        Args:
            predicted_mask: Output mask from decoder (B, 1, H, W)
            frame_embedding: Unconditional frame features (B, C, H', W')

        Returns:
            memory: Fused memory features (B, C_mem, H', W')
        """
        # Downsample mask to match frame embedding resolution
        mask_embedding = self.mask_downsample(predicted_mask)

        # Concatenate with frame embedding
        combined = torch.cat([frame_embedding, mask_embedding], dim=1)

        # Fuse information
        memory = self.fusion(combined)

        return memory
```

### Memory Bank Update Strategy

The memory bank uses a **FIFO (First-In-First-Out) queue**:

```python
class MemoryBank:
    def __init__(self, max_recent=6, max_prompted=8):
        self.max_recent = max_recent
        self.max_prompted = max_prompted

        # FIFO queues
        self.recent_memories = deque(maxlen=max_recent)
        self.prompted_memories = deque(maxlen=max_prompted)
        self.object_pointers = []

    def update(self, memory, object_pointer, is_prompted=False):
        """
        Add new memory to bank

        Args:
            memory: Spatial memory features
            object_pointer: High-level semantic token
            is_prompted: Whether this frame was prompted
        """
        if is_prompted:
            # Prompted frames go to separate queue
            self.prompted_memories.append(memory)
        else:
            # Recent frames in FIFO queue
            self.recent_memories.append(memory)

        # Always store object pointer
        self.object_pointers.append(object_pointer)

    def get_all_memories(self):
        """Get concatenated memories for cross-attention"""
        return {
            'recent': list(self.recent_memories),
            'prompted': list(self.prompted_memories),
            'pointers': self.object_pointers
        }
```

### Object Pointer Extraction

Object pointers are extracted from mask decoder output tokens:

```python
# In mask decoder forward pass
def forward(self, image_emb, prompt_emb):
    # ... decoder transformer blocks ...

    # Extract mask tokens for predictions
    mask_tokens = tokens[:num_mask_outputs]

    # Use best mask's token as object pointer
    best_mask_idx = iou_scores.argmax()
    object_pointer = mask_tokens[best_mask_idx]

    return masks, iou_scores, occlusion_scores, object_pointer
```

The object pointer captures high-level semantic information about the segmented object, enabling identity tracking across frames.

### Memory Compression

SAM 2 compresses memory to 64 dimensions (from 256):

```python
# Memory features are projected to smaller dimension
memory_features = projection_layer(memory_features)  # 256 → 64

# Object pointers are split into multiple tokens
# 256-dim pointer → 4 tokens of 64-dim each
object_pointer_tokens = object_pointer.reshape(4, 64)
```

This reduces storage requirements by 4x while maintaining performance.

---

## Section 5: Attention Weights and Visualization

### Attention Weight Properties

The attention weights reveal how SAM 2 uses memory:

**Weight Distribution Patterns:**
1. **Spatial correspondence**: High weights for similar positions
2. **Appearance matching**: High weights for similar features
3. **Recency bias**: Often higher weights for recent frames
4. **Prompt importance**: High weights for prompted frames

### Computing and Analyzing Weights

```python
def analyze_attention_weights(attn_weights, memory_info):
    """
    Analyze attention patterns

    Args:
        attn_weights: (B, H*W, num_memory_tokens)
        memory_info: Dict with memory metadata

    Returns:
        Analysis results
    """
    # Separate weights by memory type
    recent_end = memory_info['num_recent'] * memory_info['spatial_size']
    prompted_end = recent_end + memory_info['num_prompted'] * memory_info['spatial_size']

    recent_weights = attn_weights[:, :, :recent_end]
    prompted_weights = attn_weights[:, :, recent_end:prompted_end]
    pointer_weights = attn_weights[:, :, prompted_end:]

    # Compute statistics
    analysis = {
        'recent_attention': recent_weights.mean().item(),
        'prompted_attention': prompted_weights.mean().item(),
        'pointer_attention': pointer_weights.mean().item(),
        'max_weight': attn_weights.max().item(),
        'entropy': compute_entropy(attn_weights)
    }

    return analysis
```

### Attention Visualization

```python
def visualize_memory_attention(attn_weights, current_frame, memories):
    """
    Visualize which memory positions attend to current frame
    """
    import matplotlib.pyplot as plt

    # Average attention across heads
    avg_weights = attn_weights.mean(dim=1)  # (B, H*W, num_mem)

    # Reshape to spatial
    H, W = int(math.sqrt(avg_weights.shape[1])), int(math.sqrt(avg_weights.shape[1]))

    fig, axes = plt.subplots(2, len(memories) + 1, figsize=(15, 6))

    # Current frame
    axes[0, 0].imshow(current_frame)
    axes[0, 0].set_title('Current Frame')

    # Memory frames and attention maps
    for i, memory_frame in enumerate(memories):
        axes[0, i+1].imshow(memory_frame)
        axes[0, i+1].set_title(f'Memory {i}')

        # Attention map for this memory
        mem_weights = avg_weights[:, :, i*H*W:(i+1)*H*W]
        attn_map = mem_weights.sum(dim=-1).reshape(H, W)
        axes[1, i+1].imshow(attn_map, cmap='hot')
        axes[1, i+1].set_title(f'Attention to Memory {i}')

    plt.tight_layout()
    return fig
```

### Ablation Results on Attention Design

From paper ablations (Table 12):

| Object Pointers | GRU Memory | MOSE dev | SA-V val | LVOSv2 val |
|-----------------|------------|----------|----------|------------|
| No              | No         | 73.1     | 64.5     | 67.0       |
| No              | Yes        | 72.3     | 65.3     | 68.9       |
| **Yes**         | **No**     | **73.0** | **68.3** | **71.6**   |

**Key findings:**
- Object pointers significantly improve SA-V and long-term (LVOSv2) performance
- GRU memory does not provide improvement over direct storage
- Simpler design (no GRU) is preferred

---

## Section 6: Implementation Details

### Complete Memory Attention Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SAM2MemoryAttention(nn.Module):
    """
    Complete Memory Attention module for SAM 2

    Conditions current frame features on:
    - Recent frame memories
    - Prompted frame memories
    - Object pointers
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        memory_dim: int = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Transformer layers
        self.layers = nn.ModuleList([
            MemoryAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Memory projection (64 → 256 for attention)
        self.memory_proj = nn.Linear(memory_dim, hidden_dim)

        # Final layer norm
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        frame_features: torch.Tensor,
        memory_bank: dict,
        pos_encoding: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            frame_features: Current frame (B, H*W, C)
            memory_bank: Dict containing:
                - recent: List of (H*W, C_mem) tensors
                - prompted: List of (H*W, C_mem) tensors
                - pointers: List of (num_tokens, C_mem) tensors
            pos_encoding: Positional encodings

        Returns:
            Conditioned features (B, H*W, C)
        """
        # Project and concatenate all memories
        memories = []

        for mem in memory_bank.get('recent', []):
            memories.append(self.memory_proj(mem))

        for mem in memory_bank.get('prompted', []):
            memories.append(self.memory_proj(mem))

        for ptr in memory_bank.get('pointers', []):
            memories.append(self.memory_proj(ptr))

        if memories:
            memory_tokens = torch.cat(memories, dim=0)  # (num_tokens, C)
            memory_tokens = memory_tokens.unsqueeze(0).expand(
                frame_features.shape[0], -1, -1
            )  # (B, num_tokens, C)
        else:
            # No memory - behave like SAM 1
            memory_tokens = None

        # Process through transformer layers
        x = frame_features
        for layer in self.layers:
            x = layer(x, memory_tokens, pos_encoding)

        return self.norm(x)


class MemoryAttentionBlock(nn.Module):
    """Single transformer block for memory attention"""

    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Cross-attention to memory
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x, memory, pos_encoding=None):
        # Self-attention with residual
        x_norm = self.norm1(x)
        x = x + self.self_attn(x_norm, x_norm, x_norm)[0]

        # Cross-attention to memory (if available)
        if memory is not None:
            x_norm = self.norm2(x)
            x = x + self.cross_attn(x_norm, memory, memory)[0]

        # MLP with residual
        x = x + self.mlp(self.norm3(x))

        return x
```

### Integration with SAM 2 Pipeline

```python
class SAM2VideoPredictor:
    """
    Video segmentation using SAM 2 with memory attention
    """

    def __init__(self, model_config):
        self.image_encoder = HieraEncoder(model_config)
        self.memory_attention = SAM2MemoryAttention(model_config)
        self.prompt_encoder = PromptEncoder(model_config)
        self.mask_decoder = MaskDecoder(model_config)
        self.memory_encoder = MemoryEncoder(model_config)
        self.memory_bank = MemoryBank(
            max_recent=model_config.num_memories,
            max_prompted=model_config.num_prompted
        )

    def process_frame(self, frame, prompts=None):
        """Process single frame with memory context"""

        # 1. Encode frame (runs once, cached)
        frame_embedding = self.image_encoder(frame)

        # 2. Apply memory attention
        conditioned_embedding = self.memory_attention(
            frame_embedding,
            self.memory_bank.get_all_memories()
        )

        # 3. Encode prompts (if any)
        if prompts is not None:
            prompt_embedding = self.prompt_encoder(prompts)
        else:
            prompt_embedding = None

        # 4. Decode mask
        masks, iou_scores, occlusion, obj_ptr = self.mask_decoder(
            conditioned_embedding,
            prompt_embedding
        )

        # 5. Create and store memory
        memory = self.memory_encoder(masks, frame_embedding)
        self.memory_bank.update(
            memory, obj_ptr,
            is_prompted=(prompts is not None)
        )

        return masks, iou_scores, occlusion
```

---

## Section 7: ARR-COC Integration Opportunities

### Memory Attention for Visual Feature Alignment

The memory attention mechanism offers patterns applicable to ARR-COC's multi-modal learning:

**Cross-Modal Memory Bank:**
```python
class MultiModalMemoryAttention(nn.Module):
    """
    Apply SAM 2 memory attention patterns to
    cross-modal feature alignment
    """

    def __init__(self, hidden_dim=256):
        super().__init__()

        # Separate memory banks per modality
        self.visual_memory = MemoryBank(max_size=8)
        self.text_memory = MemoryBank(max_size=4)

        # Cross-modal attention
        self.cross_modal_attn = SAM2MemoryAttention(
            hidden_dim=hidden_dim,
            num_layers=2
        )

    def forward(self, current_features, modality='visual'):
        """
        Condition current features on multi-modal memory
        """
        # Gather memories from both modalities
        memory_bank = {
            'recent': self.visual_memory.get_recent(),
            'prompted': self.text_memory.get_all(),  # Text as "prompts"
            'pointers': self.get_cross_modal_pointers()
        }

        return self.cross_modal_attn(current_features, memory_bank)
```

### Temporal Consistency in Sequence Processing

Apply memory attention for maintaining consistency in sequential generation:

```python
class TemporalConsistencyModule(nn.Module):
    """
    Ensure temporal consistency in autoregressive generation
    using SAM 2 memory patterns
    """

    def __init__(self, hidden_dim=256, context_length=8):
        super().__init__()
        self.memory_attention = SAM2MemoryAttention(
            hidden_dim=hidden_dim,
            num_layers=4
        )
        self.context_memory = deque(maxlen=context_length)

    def forward(self, current_token_features):
        """
        Condition current generation on recent context
        """
        # Build memory from recent context
        memory_bank = {
            'recent': list(self.context_memory),
            'prompted': [],
            'pointers': []
        }

        # Apply memory attention
        conditioned = self.memory_attention(
            current_token_features,
            memory_bank
        )

        # Update context memory
        self.context_memory.append(current_token_features)

        return conditioned
```

### Key Patterns for Transfer

**1. Streaming Architecture:**
- Process sequences frame-by-frame/token-by-token
- Maintain fixed-size memory bank (FIFO)
- Constant memory regardless of sequence length

**2. Object Pointers:**
- High-level semantic tokens for identity
- Enable tracking across time/modalities
- Lightweight (64-dim) but information-rich

**3. Separate Memory Types:**
- Recent context (temporal locality)
- Important/prompted items (user guidance)
- Identity tokens (semantic consistency)

**4. Temporal Encoding:**
- Only for short-term context
- Avoids generalization issues
- Lets attention learn temporal relationships

### Performance Considerations

From SAM 2 benchmarks:
- Memory attention adds ~15% overhead over no-memory baseline
- Scales linearly with memory size
- FlashAttention-2 provides significant speedup
- 64-dim memory compression reduces 4x storage

**Recommendations for ARR-COC:**
- Use small memory dimensions (64-128)
- Limit memory bank size (6-8 entries)
- Implement efficient attention kernels
- Consider memory compression for long sequences

---

## Sources

**Primary Sources:**
- [SAM 2 Paper](https://arxiv.org/abs/2408.00714) - arXiv:2408.00714 (August 2024)
- [SAM 2 GitHub](https://github.com/facebookresearch/sam2) - Official implementation

**Technical Analysis:**
- [Towards Data Science Guide](https://towardsdatascience.com/segment-anything-2-what-is-the-secret-sauce-a-deep-learners-guide-1c43dd07a6f8/) - Avishek Biswas (August 2024)
- [Emergent Mind SAM 2 Analysis](https://www.emergentmind.com/topics/segment-anything-model-2-sam-2)

**Related Papers:**
- [Restricted Memory Banks (RMem)](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_RMem_Restricted_Memory_Banks_Improve_Video_Object_Segmentation_CVPR_2024_paper.pdf) - CVPR 2024
- [Temporal Memory Attention for Video Segmentation](https://www.researchgate.net/publication/354690713_Temporal_Memory_Attention_for_Video_Semantic_Segmentation)

**Additional References:**
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) - Efficient attention implementation
- [RoPE Positional Encoding](https://arxiv.org/abs/2104.09864) - Rotary Position Embedding
- [Hiera Architecture](https://arxiv.org/abs/2306.00989) - Hierarchical Vision Transformer

---

**Last Updated:** 2025-11-20
**Knowledge File:** SAM 2 Memory Attention Module
**Part of:** SAM General Knowledge Expansion
