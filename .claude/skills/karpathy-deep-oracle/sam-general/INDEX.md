# SAM General - Index

**Segment Anything Model (SAM) knowledge: foundation models for promptable segmentation**

**Total Files**: 5 (3,403 lines)
**Coverage**: SAM 1 foundations, promptable interface, zero-shot generalization
**ARR-COC Integration**: 10% per file (relevance-guided segmentation)

---

## Files

| File | Description | Lines | Keywords |
|------|-------------|-------|----------|
| `00-sam1-overview-foundation.md` | SAM 1 overview: foundation model, core contributions, impact | 659 | foundation model, promptable, zero-shot, SA-1B |
| `01-promptable-interface.md` | Prompt types: points, boxes, masks, text; interactive workflows | 1,095 | point prompt, box prompt, mask prompt, IoU |
| `02-zero-shot-generalization.md` | Domain transfer: 23 datasets, medical imaging, remote sensing | 542 | zero-shot, generalization, MedSAM, transfer |
| `08-prompt-encoder.md` | Prompt encoder architecture, embedding types, positional encoding | 796 | prompt encoder, embeddings, sparse/dense |
| `INDEX.md` | This file - navigation and overview | 111 | navigation, index |

---

## Quick Start

1. **New to SAM?** Start with `00-sam1-overview-foundation.md`
2. **Understanding prompts?** See `01-promptable-interface.md`
3. **Domain transfer?** Check `02-zero-shot-generalization.md`

---

## Topics Covered

### SAM 1 Foundations
- Foundation model paradigm shift
- Task-Model-Dataset contributions
- 1.1B masks, 11M images (SA-1B)
- 15k+ citations, 52k GitHub stars

### Promptable Interface
- Point prompts (foreground/background clicks)
- Box prompts (bounding box to mask)
- Mask prompts (rough to refined)
- Multi-prompt combinations
- Interactive refinement workflows

### Zero-Shot Generalization
- 23 dataset benchmark
- Medical imaging (CT, MRI, X-ray)
- Remote sensing (satellite, aerial)
- No fine-tuning required

### ARR-COC Integration (10% per file)
- Promptable relevance realization
- Propositional knowing (semantic labels)
- Perspectival knowing (spatial relationships)
- Participatory knowing (interactive refinement)

---

## Cross-References

**Related Folders**:
- `karpathy/vision-language-architectures/` - ViT encoders
- `deepseek/06-DeepSeek-OCR/` - SAM+CLIP serial design
- `ovis-2-5/` - Visual Embedding Table (VET)

**Source Study**:
- `PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md` (1,543 lines)

---

## Expansion Status

**Completed**: 5/42 PARTs (11.9%)
**Remaining**: 39 PARTs (SAM 2, SAM 3, training, applications, benchmarks)
**Source Expansion**: `_ingest-auto/completed/expansion-sam-general-2025-11-20/`

---

**Last Updated**: 2025-11-20
