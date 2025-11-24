# HuggingFace Integration

**Category**: Usage
**Related**: [00-quickstart.md](00-quickstart.md)

## Loading from Hub

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "AIDC-AI/Ovis2.5-9B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True  # Required!
).cuda()
```

## Why trust_remote_code?

Ovis uses custom modeling code not in transformers library.
Setting `trust_remote_code=True` allows loading this code from Hub.

## Model Variants

- `AIDC-AI/Ovis2.5-2B`
- `AIDC-AI/Ovis2.5-9B`

## Configuration

Access model config:
```python
config = model.config
print(config.visual_vocab_size)  # 16384
print(config.hidden_size)        # 3584
```
