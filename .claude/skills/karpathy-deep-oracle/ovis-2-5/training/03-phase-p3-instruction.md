# Phase P3: Instruction Tuning

**Category**: Training
**Related**: [00-overview.md](00-overview.md)

## Goal

Task following, deep reasoning, diverse domains.

## Data

**Modalities**:
- Text-only: 10%
- Single image: 50%
- Multi-image: 20%
- Video: 10%
- Thinking-style: 10%

**Domains**: QA, STEM, medical, document understanding, multilingual

**Size**: ~200M examples

## Training

```yaml
epochs: 1
learning_rate: 2e-5
batch_size: 320
```

## Output

Instruction-tuned model with thinking mode capabilities.
