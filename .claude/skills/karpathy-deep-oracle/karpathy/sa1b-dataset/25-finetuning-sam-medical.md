# Fine-tuning SAM for Domain-Specific Tasks (Medical, Satellite)

## Overview

Fine-tuning SAM on domain-specific data (medical imaging, satellite imagery, etc.) adapts the foundation model to specialized visual domains. **Only fine-tune the mask decoder** while keeping the image encoder frozen for efficient adaptation.

**Key principle**: Freeze ViT encoder, train mask decoder
- ViT learned general features on SA-1B
- Mask decoder learns domain-specific segmentation
- 100-1000× faster than training from scratch

## Fine-tuning Strategy

**What to freeze:**
- Image encoder (ViT-H/L/B) ← Frozen, pretrained on SA-1B
- Prompt encoder ← Usually frozen

**What to train:**
- Mask decoder ← Fine-tuned on domain data
- (Optional) IoU prediction head

**Why this works:**
- ViT features are general (edges, textures, shapes)
- Mask decoder learns domain-specific "what is a valid mask?"
- Transfer learning with minimal compute

## Fine-tuning Code

```python
import torch
from segment_anything import sam_model_registry

def setup_finetuning(
    model_type: str = "vit_h",
    checkpoint: str = "sam_vit_h_4b8939.pth",
):
    """
    Prepare SAM for fine-tuning.
    """
    # Load pretrained SAM
    sam = sam_model_registry[model_type](checkpoint=checkpoint)

    # Freeze image encoder
    for param in sam.image_encoder.parameters():
        param.requires_grad = False

    # Freeze prompt encoder
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False

    # Unfreeze mask decoder (this is what we fine-tune)
    for param in sam.mask_decoder.parameters():
        param.requires_grad = True

    return sam

# Fine-tuning training loop
def finetune_sam(
    sam,
    train_dataloader,
    num_epochs: int = 10,
    lr: float = 1e-4,
):
    """Fine-tune SAM mask decoder on domain-specific data."""

    optimizer = torch.optim.AdamW(
        [p for p in sam.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()

    sam.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            images = batch['images']  # (B, 3, 1024, 1024)
            masks_gt = batch['masks']  # (B, 1, 256, 256)
            prompts = batch['prompts']  # Point or box prompts

            # Forward pass
            with torch.no_grad():
                image_embeddings = sam.image_encoder(images)  # Frozen

            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=prompts.get('points'),
                boxes=prompts.get('boxes'),
                masks=None
            )

            # Trainable mask decoder
            low_res_masks, iou_predictions = sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            # Upsample predictions
            masks_pred = torch.nn.functional.interpolate(
                low_res_masks,
                size=(256, 256),
                mode='bilinear',
                align_corners=False
            )

            # Compute loss (only mask decoder gets gradients)
            loss = loss_fn(masks_pred, masks_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return sam
```

## Medical Imaging Example

**Use case**: Segment tumors in CT scans

```python
# 1. Load pretrained SAM
sam = setup_finetuning(
    model_type="vit_h",
    checkpoint="sam_vit_h_4b8939.pth"
)

# 2. Prepare medical dataset
from torch.utils.data import DataLoader
medical_dataset = CTScanDataset(
    data_dir="/path/to/ct_scans",
    annotations="/path/to/tumor_masks"
)
train_loader = DataLoader(medical_dataset, batch_size=4)

# 3. Fine-tune on medical data
sam_medical = finetune_sam(
    sam,
    train_loader,
    num_epochs=20,
    lr=1e-4
)

# 4. Save fine-tuned model
torch.save(sam_medical.state_dict(), "sam_medical_tumors.pth")
```

## Satellite Imagery Example

**Use case**: Segment buildings/roads from aerial photos

```python
# Fine-tune for satellite imagery
sam_satellite = setup_finetuning(model_type="vit_l")

satellite_dataset = SatelliteDataset(
    image_dir="/path/to/aerial_photos",
    mask_dir="/path/to/building_masks"
)

sam_satellite = finetune_sam(
    sam_satellite,
    DataLoader(satellite_dataset, batch_size=8),
    num_epochs=15
)
```

## Performance Gains

**Compared to training from scratch:**
- **100-1000× faster**: Fine-tune in hours vs days
- **10-100× less data**: Works with 1k-10k images vs 1M+
- **Better generalization**: ViT features transfer well

**Typical results:**
- Medical imaging: 85-90% Dice score with 5k images
- Satellite imagery: 80-85% IoU with 10k images
- Natural images (novel categories): 75-80% accuracy with 1k images

## ARR-COC Application

**Fine-tune SAM for document layout segmentation:**
```python
# Use SAM to segment document regions (text blocks, images, tables)
sam_documents = setup_finetuning()

# Fine-tune on document layout dataset
doc_dataset = DocumentLayoutDataset(annotations=arr_coc_layouts)
sam_documents = finetune_sam(sam_documents, doc_dataset)

# Use for spatial grounding in relevance training
spatial_masks = sam_documents.predict(document_image, prompt=query)
```

**Sources**:
- [Fine-tuning SAM (GitHub)](https://github.com/mazurowski-lab/finetune-SAM)
- SAM Paper Section 6.2 (Zero-shot Transfer)
- Medical Imaging AI Survey (2024)
