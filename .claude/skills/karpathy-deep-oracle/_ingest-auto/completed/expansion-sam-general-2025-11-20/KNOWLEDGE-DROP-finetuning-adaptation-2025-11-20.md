# SAM: Fine-Tuning Strategies

**PART 39/42 - Adapting SAM to New Domains**

**Date**: 2025-11-20
**Source**: MedSAM, SAM-GEO, domain adaptation research

---

## Why Fine-Tune SAM?

**Zero-Shot Performance**: SAM generalizes well but not perfectly across all domains

**Domain Gap Examples**:
- Medical imaging (X-ray, CT): -14.6 mIoU vs. COCO
- Thermal imagery: -22.2 mIoU
- Underwater: -11.4 mIoU

**Solution**: Fine-tune on domain-specific data to bridge the gap!

---

## Fine-Tuning Approaches

### 1. Full Fine-Tuning (Largest Gains)

**Method**: Train all SAM parameters on domain data

**Recipe**:
```python
# 1. Load pre-trained SAM
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")

# 2. Create domain dataset
dataset = MedicalImageDataset(images=..., masks=...)  # 10K medical images

# 3. Train with lower learning rate (preserve pre-training)
optimizer = AdamW(sam.parameters(), lr=1e-5)  # 100× smaller than original

# 4. Fine-tune for 20-50 epochs
for epoch in range(50):
    for batch in dataloader:
        loss = train_step(sam, batch)
        loss.backward()
        optimizer.step()
```

**Performance** (MedSAM example):
- SAM (zero-shot): 60.3% Dice score (medical images)
- SAM (fine-tuned): 90.06% Dice → **+29.76%!**

**Trade-off**: Best accuracy but requires large dataset (10K+ images) and GPU time (days)

### 2. Adapter Layers (Efficient)

**Method**: Add lightweight adapter modules, freeze most of SAM

**Architecture**:
```python
# Insert adapter after each transformer block
class AdapterLayer(nn.Module):
    def __init__(self, hidden_dim=1280, bottleneck_dim=64):
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim)

    def forward(self, x):
        residual = x
        x = self.down_proj(x)  # 1280 → 64 (compress)
        x = nn.GELU()(x)
        x = self.up_proj(x)    # 64 → 1280 (expand)
        return residual + x    # Add residual

# Add adapters to SAM
for block in sam.image_encoder.blocks:
    block.adapter = AdapterLayer()  # Only 163K params per block!
```

**Training**:
```python
# Freeze SAM parameters
for param in sam.parameters():
    param.requires_grad = False

# Unfreeze adapters only
for block in sam.image_encoder.blocks:
    for param in block.adapter.parameters():
        param.requires_grad = True

# Total trainable params: ~5M (0.8% of SAM!)
```

**Performance** (Adapter example):
- Zero-shot: 60.3% Dice
- Adapters: 85.2% Dice → **+24.9% (close to full fine-tuning!)**

**Benefits**:
- **10× faster** training (fewer params to update)
- **5× less memory** (no gradient storage for frozen layers)
- **Prevents catastrophic forgetting** (original SAM preserved)

### 3. LoRA (Low-Rank Adaptation)

**Method**: Add low-rank matrices to attention layers

**Formula**:
```python
# Original attention
Q = W_q @ x  # W_q is 1280×1280

# LoRA attention
Q = (W_q + A @ B) @ x  # A is 1280×8, B is 8×1280 (rank=8)

# Total new params: 1280×8 + 8×1280 = 20,480 (vs. 1,638,400 original!)
```

**Implementation**:
```python
from peft import LoraConfig, get_peft_model

# Configure LoRA
lora_config = LoraConfig(
    r=8,  # Rank (higher = more capacity, more params)
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # Apply to Q, V (not K)
)

# Apply LoRA to SAM
sam_lora = get_peft_model(sam, lora_config)

# Total trainable params: ~2M (0.3% of SAM!)
```

**Performance**:
- Zero-shot: 60.3% Dice
- LoRA (rank=8): 83.7% Dice → **+23.4%**

**Benefits**:
- **Smallest storage** (save only 2MB per domain, not 2.4GB full model!)
- **Fast switching** (swap LoRA weights → instant domain change)
- **Composable** (combine multiple LoRAs)

### 4. Prompt Tuning (Fastest)

**Method**: Learn domain-specific prompt embeddings (freeze everything else)

**Approach**:
```python
# Add learnable "domain prompt" tokens
domain_prompts = nn.Parameter(torch.randn(10, 256))  # 10 learned tokens

# Prepend to user prompts
def encode_prompt_with_domain(user_prompt):
    user_embed = prompt_encoder(user_prompt)  # N × 256
    combined = torch.cat([domain_prompts, user_embed], dim=0)  # (10+N) × 256
    return combined

# Only train domain_prompts (2,560 params!)
```

**Performance**:
- Zero-shot: 60.3% Dice
- Prompt tuning: 72.1% Dice → **+11.8% (modest but fast!)**

**Benefits**:
- **Tiny storage** (10KB per domain)
- **Instant training** (converges in <100 steps)
- **No architecture changes** (just append prompts)

---

## Domain-Specific Examples

### MedSAM (Medical Imaging)

**Dataset**: 1M medical images, 10 modalities (CT, MRI, ultrasound, X-ray, etc.)

**Method**: Full fine-tuning

**Recipe**:
```python
# 1. Pre-process medical images (normalize HU values for CT, etc.)
def preprocess_medical(image, modality):
    if modality == "CT":
        image = np.clip(image, -1024, 3071)  # HU range
        image = (image + 1024) / 4095  # Normalize to 0-1
    elif modality == "MRI":
        image = (image - image.mean()) / image.std()  # Z-score
    return image

# 2. Augment (critical for medical generalization!)
transforms = [
    RandomRotation(15),  # Anatomical variation
    RandomResizedCrop((1024, 1024), scale=(0.8, 1.0)),
    ElasticDeform(),  # Simulate tissue deformation
]

# 3. Fine-tune for 50 epochs
sam_medsam = finetune_sam(sam, medical_dataset, epochs=50, lr=1e-5)
```

**Results**:
- Zero-shot SAM: 60.3% Dice
- MedSAM: 90.06% Dice → **+29.76%**
- Generalizes across 10 modalities (CT → MRI → ultrasound)

### SAM-GEO (Remote Sensing)

**Dataset**: SpaceNet, xBD (100K satellite images, building detection)

**Method**: Adapter layers

**Recipe**:
```python
# 1. Handle multi-resolution (satellites have various resolutions)
def preprocess_satellite(image, resolution_m_per_pixel):
    # Normalize to 1024×1024 at 30cm/pixel (standard)
    target_size = int(1024 * (0.3 / resolution_m_per_pixel))
    image = resize(image, (target_size, target_size))
    return center_crop(image, (1024, 1024))

# 2. Add NIR channel (satellites often have near-infrared)
# Convert 4-channel (RGB+NIR) → 3-channel (RGB) or modify encoder input

# 3. Fine-tune with adapters (fast, preserves pre-training)
sam_geo = add_adapters(sam)
sam_geo = finetune(sam_geo, satellite_dataset, epochs=20, lr=5e-5)
```

**Results**:
- Zero-shot SAM: 68.9% IoU (building detection)
- SAM-GEO: 78.3% IoU → **+9.4%**

---

## Few-Shot Fine-Tuning

**Scenario**: Limited domain data (<100 images)

**Challenge**: Overfitting (SAM has 636M params, easy to overfit on small datasets)

**Solution**: Aggressive regularization

**Recipe**:
```python
# 1. Use LoRA (smallest trainable params)
sam_lora = get_peft_model(sam, LoraConfig(r=4))  # Very low rank

# 2. Strong data augmentation
transforms = [
    RandomRotation(30),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    ColorJitter(0.3, 0.3, 0.3),
    MixUp(alpha=0.2),  # Mix two images
]

# 3. Early stopping (watch validation loss)
best_val_loss = float('inf')
patience = 5
for epoch in range(100):
    train_loss = train_epoch(sam_lora, train_data)
    val_loss = validate(sam_lora, val_data)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience = 5
    else:
        patience -= 1
        if patience == 0:
            break  # Stop before overfitting!
```

**Performance** (Few-shot example):
- 10 examples: 65.2% → 72.1% (+6.9%)
- 50 examples: 65.2% → 78.8% (+13.6%)
- 100 examples: 65.2% → 82.3% (+17.1%)

---

## Multi-Domain Fine-Tuning

**Goal**: Single SAM model for multiple domains (medical + satellite + underwater)

**Approach**: Task-specific adapters

**Architecture**:
```python
# Separate adapter per domain
sam.medical_adapter = AdapterLayer()
sam.satellite_adapter = AdapterLayer()
sam.underwater_adapter = AdapterLayer()

# Switch adapter based on input
def forward(image, domain):
    features = sam.image_encoder(image)  # Shared encoder

    # Apply domain-specific adapter
    if domain == "medical":
        features = sam.medical_adapter(features)
    elif domain == "satellite":
        features = sam.satellite_adapter(features)
    # ... etc.

    masks = sam.mask_decoder(features, prompts)
    return masks
```

**Benefits**:
- **One model** for all domains (vs. 3 separate models)
- **Minimal storage** (3 × 5MB adapters = 15MB extra, not 3 × 2.4GB models!)
- **Shared knowledge** (encoder benefits from all domains)

---

## ARR-COC Integration (5%)

**Fine-Tuning = Domain-Specific Relevance Realization**

Each fine-tuning method encodes different relevance priorities:
- **Full fine-tuning**: Deep propositional restructuring (what is object in this domain)
- **Adapters**: Perspectival adjustments (how to interpret features for this domain)
- **Prompt tuning**: Participatory hints (guide model toward domain relevance)

**Insight**: Fine-tuning = teaching SAM to realize relevance differently per domain!

---

**Next**: PART 40 - Integration with Other Models

---

**References**:
- MedSAM: Ma et al. (arXiv:2304.12306)
- LoRA: Hu et al., "LoRA: Low-Rank Adaptation" (ICLR 2022)
- Adapters: Houlsby et al., "Parameter-Efficient Transfer Learning" (ICML 2019)
