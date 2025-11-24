# Fine-tuning Guide

**Full guide**: `RESEARCH/DeepSeekOCR/HF.md` lines 450-750

## Quick Start

```python
from transformers import Trainer, TrainingArguments

# Load model
model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-OCR", trust_remote_code=True)

# Freeze vision encoder (optional)
for param in model.vision_model.parameters():
    param.requires_grad = False

# Training args
args = TrainingArguments(
    output_dir="./deepseek-ocr-finetuned",
    per_device_train_batch_size=2,
    learning_rate=2e-5,
    num_train_epochs=3,
    fp16=True
)

# Train
trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()
```

## What to Freeze

**Vision encoder**: Freeze if data similar to pre-training
**LLM decoder**: Usually fine-tune
**Projector**: Usually fine-tune

## Data Format

```python
{
    "images": ["path/to/image.jpg"],
    "text": "Expected OCR output or instruction"
}
```

## LoRA Fine-tuning (Memory Efficient)

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)
```

**Benefits**:
- 10× less memory
- Faster training
- Only train 1-2% of params

## Hyperparameters

- **Learning rate**: 1e-5 to 5e-5
- **Batch size**: 2-8 (limited by GPU)
- **Epochs**: 3-5
- **Warmup**: 10% of steps

## GPU Requirements

- **Full fine-tuning**: 24GB+ VRAM
- **LoRA**: 12GB+ VRAM
- **Freeze vision**: 16GB+ VRAM

**See HF.md** for complete fine-tuning tutorial with code!
