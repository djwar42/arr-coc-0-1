# BATCH 4: Learned Channel Weighting - SENet & FiLM

## Squeeze-and-Excitation Networks (SENet)

**Paper:** Hu et al. 2017 - **46,418 citations!**
**Won:** ImageNet 2017

### Core Architecture

The SE block recalibrates channel-wise features:

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global spatial information
        y = self.squeeze(x).view(b, c)
        # Excitation: Channel attention weights
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale: Recalibrate channels
        return x * y.expand_as(x)
```

### The SE Block Formula

**Squeeze (Global Information):**
```
z_c = F_sq(u_c) = (1/H×W) Σᵢ Σⱼ u_c(i,j)
```

**Excitation (Channel Weights):**
```
s = F_ex(z, W) = σ(W₂ · δ(W₁ · z))
```
Where:
- δ = ReLU
- σ = Sigmoid
- W₁ ∈ ℝ^(C/r × C) (reduction)
- W₂ ∈ ℝ^(C × C/r) (expansion)

**Scale (Recalibration):**
```
x̃_c = F_scale(u_c, s_c) = s_c · u_c
```

### Key Design Choices

1. **Reduction ratio (r):** Typically 16
   - Too small: Not enough compression
   - Too large: Loses representational power

2. **Where to place:** After each block
   - ResNet: After residual add
   - Inception: After inception module

3. **Overhead:** ~2.5% extra parameters, ~10% extra compute

## FiLM (Feature-wise Linear Modulation)

**Paper:** Perez et al. 2017 - **2,936 citations**

### Core Mechanism

```
FiLM(F_i,c | γ_i,c, β_i,c) = γ_i,c · F_i,c + β_i,c
```

Unlike SENet's scalar attention, FiLM uses:
- **γ (gamma):** Learned scaling factors
- **β (beta):** Learned shift factors
- Both generated from conditioning input (e.g., question)

### Implementation

```python
class FiLMLayer(nn.Module):
    def __init__(self, num_features, conditioning_dim):
        super().__init__()
        self.gamma = nn.Linear(conditioning_dim, num_features)
        self.beta = nn.Linear(conditioning_dim, num_features)

    def forward(self, features, conditioning):
        # Generate modulation parameters
        gamma = self.gamma(conditioning)  # Scale
        beta = self.beta(conditioning)    # Shift

        # Apply FiLM transformation
        return gamma.unsqueeze(-1).unsqueeze(-1) * features + \
               beta.unsqueeze(-1).unsqueeze(-1)
```

### FiLM vs SENet

| Aspect | SENet | FiLM |
|--------|-------|------|
| Conditioning | Self (from features) | External (query, text) |
| Operation | Multiply only | Multiply + Add |
| Parameters | Per channel | Per channel, per condition |
| Use case | Image classification | VQA, multimodal |

## Channel Attention Variants

### ECA-Net (Efficient Channel Attention)

- Avoids dimensionality reduction
- Uses 1D convolution instead of FC layers
- More efficient: k=3 kernel captures local cross-channel interaction

```python
class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y).expand_as(x)
```

### CBAM (Convolutional Block Attention Module)

Combines channel AND spatial attention:
```python
def cbam(x):
    x = channel_attention(x) * x  # SENet-style
    x = spatial_attention(x) * x  # Position-aware
    return x
```

## Query-Conditioned Feature Selection

### For VQA/VLM

Different questions need different visual features:
- "What color is the car?" → Color channels
- "How many people?" → Object detection channels
- "Is it raining?" → Weather/texture channels

```python
class QueryConditionedSE(nn.Module):
    def __init__(self, visual_dim, query_dim, reduction=16):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, visual_dim)
        self.se = nn.Sequential(
            nn.Linear(visual_dim * 2, visual_dim // reduction),
            nn.ReLU(),
            nn.Linear(visual_dim // reduction, visual_dim),
            nn.Sigmoid()
        )

    def forward(self, visual_features, query):
        # Project query
        q = self.query_proj(query)

        # Combine visual + query
        combined = torch.cat([visual_features.mean(-1).mean(-1), q], dim=-1)

        # Generate attention weights
        weights = self.se(combined)

        return visual_features * weights.unsqueeze(-1).unsqueeze(-1)
```

## Integration with Spicy Lentil

### Channel Weighting in 9 Pathways

Each of the 9 ways of knowing can have channel attention:

```python
class PathwayWithChannelAttention(nn.Module):
    def __init__(self, hidden_dim):
        self.pathway = nn.Linear(hidden_dim, hidden_dim)
        self.se_block = SEBlock(hidden_dim)

    def forward(self, x):
        out = self.pathway(x)
        return self.se_block(out)
```

### FiLM for Cognitive Fingerprint Conditioning

The user's cognitive fingerprint can modulate all channels:

```python
# Generate FiLM parameters from cognitive fingerprint
gamma, beta = film_generator(cognitive_fingerprint)

# Apply to all 9 pathways
for pathway in nine_pathways:
    pathway_output = gamma[pathway_id] * pathway(slot) + beta[pathway_id]
```

### Lundquist-Aware Channel Entropy

Ensure channel attention doesn't collapse:
```python
# Check channel attention entropy
attention_entropy = -sum(weights * log(weights))
if attention_entropy < 0.2734 * log(num_channels):
    # Add entropy regularization
    loss += entropy_penalty
```

## Performance Results

### SENet on ImageNet

| Model | Top-1 (%) | Top-5 (%) | Params |
|-------|-----------|-----------|--------|
| ResNet-50 | 75.2 | 92.1 | 25.6M |
| SE-ResNet-50 | **77.6** | **93.6** | 28.1M |
| ResNet-152 | 77.8 | 93.5 | 60.2M |
| SE-ResNet-152 | **79.3** | **94.5** | 66.8M |

### FiLM on CLEVR

- **97.7%** accuracy on visual reasoning
- Generalizes to novel compositions
- Works without bounding boxes

## Key Takeaways

1. **Channel attention is cheap:** ~2-10% overhead for significant gains
2. **External conditioning (FiLM) enables multimodal:** Query → Feature modulation
3. **Combine spatial + channel:** CBAM shows this helps
4. **Entropy matters:** Don't let attention collapse (27.34%!)

---

**Sources:**
- "Squeeze-and-Excitation Networks" - CVPR 2018
- "FiLM: Visual Reasoning with a General Conditioning Layer" - AAAI 2018
- "ECA-Net: Efficient Channel Attention" - CVPR 2020
- "CBAM: Convolutional Block Attention Module" - ECCV 2018
