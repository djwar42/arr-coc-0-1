# Fine-Tuning Ovis

**Category**: Usage
**Related**: [../training/00-overview.md](../training/00-overview.md)

## Data Format

```json
[
  {
    "image": "path/to/image.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nQuestion?"},
      {"from": "gpt", "value": "Answer"}
    ]
  }
]
```

## Training Script

```bash
# Using provided training script
bash scripts/run_ovis2_5_sft.sh
```

## Configuration

Edit training config to specify:
- Dataset paths
- Learning rate
- Batch size
- Output directory

## Module Selection

Can fine-tune specific modules:
- All: Full model fine-tuning
- LLM only: Vision frozen
- Custom: Specify modules

See [../training/00-overview.md](../training/00-overview.md) for details.
