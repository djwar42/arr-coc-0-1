# Fine-Tuning Script Example

**Category**: Examples
**Related**: [../usage/03-fine-tuning.md](../usage/03-fine-tuning.md)

## Minimal Training Script

```python
from ovis.train.train import train
from ovis.train.arguments import ModelArguments, TrainingArguments

model_args = ModelArguments(
    ovis_pretrained_path="AIDC-AI/Ovis2.5-9B"
)

training_args = TrainingArguments(
    data_type="conversation",
    train_modules="all",  # or "llm" for LLM-only
    datainfo_path="path/to/datainfo.json",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    output_dir="./output",
    deepspeed="path/to/deepspeed_config.json"
)

train(model_args, training_args)
```

## Run via Script

```bash
bash scripts/run_ovis2_5_sft.sh
```
