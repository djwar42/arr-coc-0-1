# Training Overview: 5-Phase Curriculum

**Category**: Training
**Related**: [01-phase-p1-vet.md](01-phase-p1-vet.md), [../architecture/00-overview.md](../architecture/00-overview.md)
**Code**: `ovis/train/train.py`

## Philosophy

Ovis 2.5 uses a **progressive 5-phase training curriculum** that builds capabilities incrementally:

```
P1: VET         → Initialize visual vocabulary
P2: Multimodal  → Core visual understanding
P3: Instruction → Task following + reasoning
P4: DPO         → Preference alignment
P5: GRPO        → Reasoning optimization
```

**Key Insight**: Each phase builds on the previous, enabling stable training without catastrophic forgetting.

## Training Diagram

```
┌───────────────────────────────────────────────────────────────┐
│ Phase P1: VET Pre-training (Caption Prediction)               │
├───────────────────────────────────────────────────────────────┤
│ Trainable: VT (partial), VET, Visual Head                     │
│ Frozen: Most ViT, ALL LLM                                     │
│ Data: Image-caption pairs                                     │
│ Resolution: 448²-896²                                         │
│ RoPE: Disabled                                                │
│ Goal: Learn visual vocabulary                                 │
└───────────────────────┬───────────────────────────────────────┘
                        ↓
┌───────────────────────────────────────────────────────────────┐
│ Phase P2: Multimodal Pre-training (OCR + Grounding)          │
├───────────────────────────────────────────────────────────────┤
│ Trainable: ALL MODULES (full-parameter)                      │
│ Frozen: Nothing                                               │
│ Data: OCR, grounding, captions                                │
│ Resolution: 448²-1792² (2× expansion)                         │
│ RoPE: Activated                                               │
│ Goal: Core visual understanding                               │
└───────────────────────┬───────────────────────────────────────┘
                        ↓
┌───────────────────────────────────────────────────────────────┐
│ Phase P3: Instruction Tuning (Task Following)                │
├───────────────────────────────────────────────────────────────┤
│ Trainable: ALL MODULES                                        │
│ Frozen: Nothing                                               │
│ Data: Text, image, multi-image, video, thinking-style        │
│ Domains: QA, STEM, medical, multilingual                     │
│ Goal: Instruction following + deep reasoning                  │
└───────────────────────┬───────────────────────────────────────┘
                        ↓
┌───────────────────────────────────────────────────────────────┐
│ Phase P4: DPO (Preference Alignment)                         │
├───────────────────────────────────────────────────────────────┤
│ Trainable: ALL MODULES                                        │
│ Method: Direct Preference Optimization                        │
│ Data: Preference pairs (CoT vs thinking-style)                │
│ Loss: DPO + auxiliary NLL                                     │
│ Goal: Align with human preferences                            │
└───────────────────────┬───────────────────────────────────────┘
                        ↓
┌───────────────────────────────────────────────────────────────┐
│ Phase P5: GRPO (Reasoning Optimization)                      │
├───────────────────────────────────────────────────────────────┤
│ Trainable: LLM ONLY (vision frozen)                          │
│ Method: Group Relative Policy Optimization                    │
│ Data: Math problems, verifiable rewards                       │
│ Goal: Optimize reasoning capabilities                         │
└───────────────────────────────────────────────────────────────┘
```

## Phase Details

### Phase P1: VET Pre-training

**Purpose**: Initialize Visual Embedding Table with meaningful visual vocabulary

**Training Setup**:
```yaml
modules:
  trainable:
    - visual_tokenizer (partial ViT layers)
    - visual_embedding_table (VET)
    - visual_head (projection)
  frozen:
    - most_vit_layers
    - entire_llm

data:
  type: image-caption pairs
  size: ~100M examples
  format: (image, caption)

resolution:
  min: 448×448
  max: 896×896
  rope: disabled  # Activate in P2

epochs: 1
learning_rate: 5e-5
batch_size: 1280
```

**What Happens**:
- VET learns to map visual features to discrete embeddings
- Visual head learns to generate probability distributions
- LLM remains frozen (pretrained knowledge preserved)

**Output**: Model with initialized VET ready for multimodal learning

### Phase P2: Multimodal Pre-training

**Purpose**: Core visual understanding (OCR, grounding, spatial reasoning)

**Training Setup**:
```yaml
modules:
  trainable: ALL (full-parameter training)

data:
  composition:
    - OCR: 70% (document + scene text)
    - Grounding: 15% (object detection, spatial)
    - Captions: 15% (general descriptions)
  size: ~500M examples

resolution:
  min: 448×448
  max: 1792×1792  # 2× expansion from P1
  rope: enabled    # Activate for spatial awareness

epochs: 1
learning_rate: 3e-5
batch_size: 640
```

**What Happens**:
- ViT fine-tunes on visual tasks
- VET adapts to diverse visual content
- LLM learns multimodal understanding
- RoPE enables better spatial awareness

**Output**: Strong base multimodal model

### Phase P3: Instruction Tuning

**Purpose**: Task following, deep reasoning, diverse domains

**Training Setup**:
```yaml
modules:
  trainable: ALL

data:
  modalities:
    - Text-only: 10%
    - Single image: 50%
    - Multi-image: 20%
    - Video: 10%
    - Thinking-style: 10%

  domains:
    - General QA
    - STEM (math, physics, coding)
    - Medical imaging
    - Document understanding
    - Multilingual OCR

  size: ~200M examples

resolution: Same as P2 (448²-1792²)

epochs: 1
learning_rate: 2e-5
batch_size: 320
```

**What Happens**:
- Model learns to follow instructions
- Develops domain expertise
- Thinking mode capabilities emerge
- Multi-image/video understanding

**Output**: Instruction-tuned model ready for deployment

### Phase P4: DPO (Direct Preference Optimization)

**Purpose**: Align with human preferences on reasoning quality

**Training Setup**:
```yaml
modules:
  trainable: ALL

method: DPO (Direct Preference Optimization)

data:
  format: (prompt, chosen_response, rejected_response)
  preferences:
    - Clear reasoning vs confused reasoning
    - Thinking-style vs vanilla CoT
    - Correct answer vs wrong answer
    - Self-correction vs hasty answer

  size: ~10M preference pairs

loss:
  primary: DPO loss
  auxiliary: NLL loss (maintain generation quality)

epochs: 1
learning_rate: 1e-5
batch_size: 128
```

**What Happens**:
- Model learns to prefer better reasoning
- Values self-correction
- Aligns with human judgment
- Maintains generation fluency

**Output**: Preference-aligned model

### Phase P5: GRPO (Group Relative Policy Optimization)

**Purpose**: Optimize reasoning through reinforcement learning

**Training Setup**:
```yaml
modules:
  trainable: LLM ONLY
  frozen: Vision (ViT, VT, VET)  # Vision frozen!

method: GRPO (Group Relative Policy Optimization)

data:
  type: Math problems with verifiable answers
  reward:
    - Correct answer: +1.0
    - Correct reasoning: +0.5
    - Self-correction: +0.3
    - Logical consistency: +0.2

  size: ~5M math problems

epochs: 1
learning_rate: 5e-6
batch_size: 64
```

**What Happens**:
- Model optimizes for verifiable correctness
- Learns better reasoning strategies
- Self-correction improves
- Vision frozen (stable visual understanding)

**Output**: Final Ovis 2.5 model with optimized reasoning

## Module Training Progression

| Phase | ViT | VT | VET | Visual Head | LLM |
|-------|-----|----|----|-------------|-----|
| **P1** | Mostly frozen | Partial | ✅ Train | ✅ Train | ❌ Frozen |
| **P2** | ✅ Train | ✅ Train | ✅ Train | ✅ Train | ✅ Train |
| **P3** | ✅ Train | ✅ Train | ✅ Train | ✅ Train | ✅ Train |
| **P4** | ✅ Train | ✅ Train | ✅ Train | ✅ Train | ✅ Train |
| **P5** | ❌ Frozen | ❌ Frozen | ❌ Frozen | ❌ Frozen | ✅ Train |

## Resolution Progression

```
P1: 448²-896²    → Learn basic visual features
P2: 448²-1792²   → 2× expansion, high-res understanding
P3: 448²-1792²   → Maintain high-res capability
P4: 448²-1792²   → Consistent with P3
P5: 448²-1792²   → Consistent (vision frozen)
```

**Why Progressive?**
- P1: Lower res = faster VET initialization
- P2: Full res = better OCR and spatial understanding
- P3-P5: Maintain capability

## Data Composition

### Total Training Data: ~815M Examples

```
P1: 100M   (12%) - Captions
P2: 500M   (61%) - OCR + Grounding
P3: 200M   (25%) - Instructions
P4: 10M    (1%)  - Preferences
P5: 5M     (1%)  - Math RL
```

### Data Quality

**P1-P2**: Web-scale, automatic annotation
- Sources: LAION, CC12M, YFCC100M
- Quality: Filtered for coherence

**P3**: High-quality human annotations
- Sources: Academic datasets, expert labeling
- Quality: Curated, diverse

**P4-P5**: Expert-labeled preferences
- Sources: Human preference ratings
- Quality: Gold standard

## Infrastructure

### Hardware Requirements

**Training**:
- 20-40 nodes × 8× A100-80GB GPUs
- Total: 160-320 GPUs
- Time: 2-3 weeks for full curriculum

**Key Optimizations**:
- DeepSpeed ZeRO-3 (memory efficiency)
- Flash Attention 2 (2-3× speedup)
- Mixed precision (bfloat16)
- Gradient checkpointing
- Data packing (3-4× throughput)

### Training Speed

| Phase | Steps | Time (320 A100s) |
|-------|-------|------------------|
| P1 | ~80K | 2-3 days |
| P2 | ~400K | 10-12 days |
| P3 | ~160K | 4-5 days |
| P4 | ~8K | 12-16 hours |
| P5 | ~4K | 6-8 hours |
| **Total** | ~652K | **~18-21 days** |

## Learning Rate Schedules

### P1-P3: Cosine with Warmup
```
Warmup: 5% of steps
Peak LR: Phase-specific
Decay: Cosine to 10% of peak
Min LR: 1e-7
```

### P4: Constant with Short Warmup
```
Warmup: 100 steps
LR: 1e-5 (constant)
```

### P5: Step Decay
```
Initial: 5e-6
Decay: 0.5× every 1K steps
Min: 1e-7
```

## Validation Strategy

### Benchmarks (Every 1K Steps)

**P1-P2**:
- COCO Captions (CIDEr)
- TextVQA (accuracy)
- DocVQA (ANLS)

**P3**:
- All P2 benchmarks +
- MMBench (general understanding)
- MathVista (reasoning)
- MMMU (multimodal understanding)

**P4-P5**:
- Focus on reasoning metrics
- Human preference ratings (sampled)
- Math problem accuracy

### Early Stopping

**Criteria**: No improvement for 3 consecutive validations
**Action**: Save checkpoint, continue to next phase

## Ablation Studies

### Without Phase P1

**Result**: 5-7% lower performance on all benchmarks
**Conclusion**: VET initialization critical

### Without Phase P4

**Result**: 3-4% lower on preference metrics
**Conclusion**: DPO improves alignment quality

### Without Phase P5

**Result**: 8-10% lower on math reasoning
**Conclusion**: GRPO essential for STEM tasks

## Related Topics

- [01-phase-p1-vet.md](01-phase-p1-vet.md) - VET initialization details
- [02-phase-p2-multimodal.md](02-phase-p2-multimodal.md) - Core training
- [03-phase-p3-instruction.md](03-phase-p3-instruction.md) - Instruction tuning
- [04-phases-p4-p5-rl.md](04-phases-p4-p5-rl.md) - RL optimization
- [05-data-composition.md](05-data-composition.md) - Dataset details
- [06-infrastructure.md](06-infrastructure.md) - Technical setup

## Code References

**Training Script**: `ovis/train/train.py`
**Arguments**: `ovis/train/arguments.py`
**Datasets**: `ovis/train/dataset/`
**Config**: `ovis/model/configuration_ovis.py`
