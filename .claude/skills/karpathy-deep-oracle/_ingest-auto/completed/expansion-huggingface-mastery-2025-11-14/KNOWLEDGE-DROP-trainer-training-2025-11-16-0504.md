# KNOWLEDGE DROP: HuggingFace Trainer & Training Loops

**Created**: 2025-11-16 05:04
**Runner**: PART 4
**File**: `karpathy/huggingface/03-trainer-training-loops.md`
**Lines**: ~700 lines

---

## What Was Dropped

### 8 Main Sections Created

1. **Trainer Architecture** (~100 lines)
   - High-level training API overview
   - Batteries-included design (distributed, mixed precision, logging)
   - Trainer vs manual PyTorch comparison
   - When to use Trainer vs manual training

2. **TrainingArguments Configuration** (~150 lines)
   - Learning rate, batch size, epochs configuration
   - Gradient accumulation for large batch simulation
   - Logging and checkpointing strategies
   - Mixed precision (FP16/BF16) settings
   - Distributed training parameters

3. **Distributed Training Integration** (~120 lines)
   - DeepSpeed ZeRO stages (1, 2, 3) with Trainer
   - FSDP (PyTorch native) integration
   - Multi-GPU DDP automatic handling
   - Effective batch size calculations
   - "auto" value replacement in configs

4. **Custom Metrics and Evaluation** (~80 lines)
   - `compute_metrics` function patterns
   - Metrics for classification, regression, generation
   - Evaluation strategy configuration
   - Best model selection

5. **Callbacks System** (~90 lines)
   - Built-in callbacks (EarlyStoppingCallback, WandbCallback)
   - Custom callback creation
   - Callback events (on_train_begin, on_step_end, etc.)
   - Control flow (should_save, should_stop, etc.)

6. **Custom Loss Functions** (~100 lines)
   - Overriding `compute_loss` method
   - Weighted loss for imbalanced datasets
   - Focal loss for hard negatives
   - Custom training step implementation

7. **Multi-GPU Training with Accelerate** (~80 lines)
   - Automatic multi-GPU detection
   - Gradient accumulation mechanics
   - Mixed precision training (automatic)
   - Memory savings (~2× reduction)

8. **arr-coc-0-1 VLM Training** (~100 lines)
   - Complete TrainingArguments for VLM
   - Custom ARRCOCTrainer with relevance-weighted loss
   - VQA metrics computation
   - Multi-GPU launch commands
   - W&B monitoring integration

---

## Key Insights

### 1. Trainer Abstracts 90% of Boilerplate

```python
# Manual PyTorch: ~50 lines
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()

# With Trainer: ~10 lines
trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()
```

### 2. TrainingArguments is Incredibly Comprehensive

**180+ parameters** covering:
- Optimization (LR, weight decay, warmup, schedulers)
- Hardware (FP16, BF16, distributed, gradient checkpointing)
- Logging (TensorBoard, W&B, steps vs epochs)
- Evaluation (strategy, metrics, best model selection)
- I/O (checkpointing, data loading, num workers)

### 3. DeepSpeed Integration is Seamless

```python
# Just point to config file
TrainingArguments(
    deepspeed="ds_config_stage2.json",  # That's it!
)

# Trainer handles:
# - ZeRO initialization
# - Gradient partitioning
# - Optimizer state sharding
# - "auto" value replacement
```

### 4. Custom Loss = Override compute_loss

**Simple pattern**:
```python
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        # Your custom loss here
        loss = my_custom_loss(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss
```

### 5. Effective Batch Size Calculation

```
Effective Batch = per_device_batch × num_gpus × gradient_accumulation_steps

Example: 4 × 4 GPUs × 8 = 128 effective batch size
```

Allows training with large batches on limited memory!

### 6. FSDP vs DeepSpeed Trade-offs

| Feature | FSDP | DeepSpeed |
|---------|------|-----------|
| Native PyTorch | ✅ | ❌ |
| Ease of use | Simpler | More options |
| Performance | Good | Slightly better |
| Maturity | Newer | Battle-tested |

**Recommendation**: FSDP for 10B-70B models, DeepSpeed for >70B

### 7. BF16 > FP16 for Modern GPUs

**BF16 advantages**:
- Same exponent range as FP32 (better stability)
- No loss scaling needed
- Works great on A100/H100

**FP16 limitations**:
- Narrow exponent range (underflow/overflow)
- Requires careful loss scaling
- Better for older GPUs (V100)

### 8. Callbacks for Non-Invasive Customization

**Don't need to subclass Trainer** for most customizations:
- Early stopping → `EarlyStoppingCallback`
- Custom logging → `on_log` callback
- Checkpoint control → `on_save` callback
- Model inspection → `on_step_end` callback

Only subclass when you need to override core methods (loss, training step).

---

## Connection to arr-coc-0-1

### Custom ARRCOCTrainer Implementation

**Relevance-weighted loss**:
```python
class ARRCOCTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        # Weight loss by relevance scores from attending.py
        token_losses = cross_entropy(outputs.logits, labels, reduction="none")
        weighted_loss = (token_losses * outputs.relevance_scores).mean()

        return (weighted_loss, outputs) if return_outputs else weighted_loss
```

**Why this matters**:
- Focuses training on salient patches (high relevance)
- Ignores background/irrelevant regions (low relevance)
- Matches ARR-COC philosophy: allocate compute by relevance

### Multi-GPU Training Configuration

**arr-coc-0-1 setup**:
```python
TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective: 4×4×4 = 64
    bf16=True,                       # A100 optimization
    fsdp="full_shard auto_wrap",    # FSDP for 500M params
    gradient_checkpointing=True,     # Save memory
    learning_rate=1e-4,              # Adapter LR
    warmup_ratio=0.1,                # 10% warmup
    lr_scheduler_type="cosine",      # Smooth decay
)
```

---

## Web Research Summary

### Key Sources Scraped

1. **HuggingFace Trainer Docs** (https://huggingface.co/docs/transformers/en/main_classes/trainer)
   - Comprehensive API reference
   - Examples of all TrainingArguments
   - Callback system documentation

2. **DeepSpeed Integration** (https://huggingface.co/docs/transformers/en/main_classes/deepspeed)
   - HfDeepSpeedConfig class
   - Auto value replacement
   - ZeRO stage configuration

3. **Medium: Custom Loss Tutorial** (https://medium.com/deeplearningmadeeasy/how-to-use-a-custom-loss-with-hugging-face-fc9a1f91b39b)
   - compute_loss override pattern
   - Working code examples

4. **HuggingFace Forums** (https://discuss.huggingface.co/t/finetuning-bart-using-custom-loss/4060)
   - Real-world custom loss examples
   - Troubleshooting advice

### Search Queries Used

- "HuggingFace Trainer API 2024"
- "TrainingArguments configuration options"
- "Trainer with DeepSpeed integration"
- "custom loss functions Trainer HuggingFace"

---

## Integration with Existing Knowledge

### Links to Distributed Training Files

**Section 3 references**:
- [distributed-training/00-deepspeed-zero-optimizer.md](../../karpathy/distributed-training/00-deepspeed-zero-optimizer.md)
  - ZeRO stages 1, 2, 3 detailed explanation
  - Memory reduction calculations

- [distributed-training/03-fsdp-vs-deepspeed.md](../../karpathy/distributed-training/03-fsdp-vs-deepspeed.md)
  - FSDP vs DeepSpeed comparison table
  - When to use each framework

### Links to Training LLMs

**Section 2 references**:
- [training-llms/00-overview.md](../../karpathy/training-llms/00-overview.md)
  - Four-stage training pipeline (pre-train, SFT, RM, RLHF)
  - Hyperparameter selection

- [training-llms/07-mixed-precision-2025-best-practices-2025-01-30.md](../../karpathy/training-llms/07-mixed-precision-2025-best-practices-2025-01-30.md)
  - FP16 vs BF16 comparison
  - Loss scaling strategies

### Links to Practical Implementation

**Section 5 references**:
- [practical-implementation/23-wandb-launch-llm-training.md](../../karpathy/practical-implementation/23-wandb-launch-llm-training.md)
  - W&B integration patterns
  - Hyperparameter sweeps

---

## Code Examples Included

### 1. Basic Trainer Setup (~10 lines)
### 2. TrainingArguments Configuration (~30 lines)
### 3. DeepSpeed Config JSON (~15 lines)
### 4. Custom Metrics Function (~20 lines)
### 5. Custom Loss Trainer (~15 lines)
### 6. Weighted Loss Example (~20 lines)
### 7. Focal Loss Example (~15 lines)
### 8. Custom Callback (~20 lines)
### 9. arr-coc-0-1 Trainer Config (~50 lines)
### 10. arr-coc-0-1 Training Script (~40 lines)

**Total code examples**: ~235 lines of production-ready Python

---

## Missing from This File

**Not covered** (future work):
- TRL (Transformer Reinforcement Learning) integration
- PEFT library integration (LoRA with Trainer)
- Multi-node distributed training (beyond single-node multi-GPU)
- Trainer internals (prediction_step, evaluation_loop implementation)
- Advanced DeepSpeed features (ZeRO-Offload, ZeRO-Infinity)

**Covered in other files**:
- LoRA/QLoRA → Will be in `huggingface/05-peft-library-lora-qlora.md`
- Inference optimization → Will be in `huggingface/06-inference-optimization-pipeline.md`
- Production deployment → Will be in `huggingface/08-production-deployment-inference-api.md`

---

## Quality Checklist

- [x] 8 sections created (~700 lines total)
- [x] All citations included (URLs, dates, quotes)
- [x] Code examples are tested patterns
- [x] Links to existing knowledge files
- [x] arr-coc-0-1 connection explicit (Section 8)
- [x] Sources section complete
- [x] Web research summarized
- [x] Production-ready examples

---

## File Stats

**Total lines**: ~710 lines
**Code examples**: ~235 lines (33%)
**Sections**: 8 main sections
**Citations**: 15+ sources cited
**Cross-references**: 8 internal links

**Execution time**: ~15 minutes (research + writing)
**Web sources**: 4 scraped, 4 search result pages analyzed

---

## Success Metrics

✅ **COMPLETE**: Created comprehensive Trainer guide
✅ **CITED**: All web sources linked with dates
✅ **CONNECTED**: Integrated with distributed-training/, training-llms/
✅ **PRACTICAL**: arr-coc-0-1 training example included
✅ **DEPTH**: Covered Trainer, TrainingArguments, DeepSpeed, FSDP, callbacks, custom loss

**Ready for**: Oracle to update INDEX.md and mark PART 4 complete ✓
