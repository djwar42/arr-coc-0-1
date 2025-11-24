# Training Loop

**Category**: Codebase
**File**: `ovis/train/train.py`

## Main Function: train()

```python
def train(model_args, training_args):
    """
    Main training loop

    Steps:
    1. Load model
    2. Load data
    3. Setup trainer
    4. Train
    5. Save checkpoint
    """
```

## Key Functions

### load_model()

Loads Ovis model with specified modules trainable/frozen.

```python
model = load_model(
    pretrained_path="AIDC-AI/Ovis2.5-9B",
    train_modules="all"  # or specific modules
)
```

### load_data()

Loads training dataset based on phase.

```python
dataset = load_data(
    data_type="conversation",  # or "caption"
    datainfo_path="path/to/datainfo.json"
)
```

### Module Selection

Can train specific modules:
- `"all"`: Full model
- `"llm"`: LLM only
- `"vte"`: VET only
- Custom: `"vte:1e-4|llm:5e-6"` (with learning rates)
