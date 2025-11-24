# Datasets

**Category**: Codebase
**Files**: `ovis/train/dataset/`

## CaptionDataset

**File**: `caption_dataset.py`
**Used in**: Phase P1

**Format**:
```json
{
  "image": "path/to/image.jpg",
  "caption": "A cat sitting on a mat"
}
```

## ConversationDataset

**File**: `conversation_dataset.py`
**Used in**: Phase P2+

**Format**:
```json
{
  "image": "path/to/image.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nQuestion?"},
    {"from": "gpt", "value": "Answer"}
  ]
}
```

## MultimodalDataset

**File**: `multimodal_dataset.py`

Data collator that:
- Batches samples
- Pads sequences
- Creates attention masks
- Handles images

## Datainfo JSON

References datasets:

```json
{
  "dataset_name": {
    "meta_file": "path/to/data.json",
    "storage_type": "hybrid",
    "data_format": "conversation",
    "image_dir": "path/to/images/"
  }
}
```
