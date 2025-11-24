# Ablation Studies: SAM Architecture Design

## Overview

SAM paper conducts ablation studies on **ViT size, prompt types, multi-mask output, and IoU prediction**. Key findings inform architecture choices.

## ViT Encoder Size

**Model variants:**
| Model | Params | ImageNet | SA-1B IoU |
|-------|--------|----------|-----------|
| ViT-B | 91M | 84.7% | 86.2% |
| ViT-L | 308M | 85.9% | 88.1% |
| ViT-H | 632M | 86.8% | 89.4% |

**Finding**: ViT-H provides best quality, ViT-B sufficient for many tasks

## Prompt Type Effectiveness

**Single prompt IoU:**
- Point: 82.3%
- Box: 91.2%
- Mask: 94.1%

**Finding**: Box prompts most effective for single-prompt queries

## Multi-Mask Output

**Without multi-mask (single output):**
- IoU: 84.1%
- Handles ambiguity poorly

**With multi-mask (3 outputs):**
- IoU: 89.4%
- Ranks masks by predicted IoU
- User selects best mask

**Finding**: Multi-mask critical for ambiguous prompts

## IoU Prediction Head

**Without IoU prediction:**
- Can't rank multiple masks
- No confidence estimation

**With IoU prediction:**
- Accurate ranking (correlation 0.94)
- Auto-select best mask
- Confidence-based filtering

**Finding**: IoU head essential for usability

## Mask Decoder Depth

**Decoder variants:**
- 1 layer: 86.8% IoU
- 2 layers: 89.4% IoU ✓ (SAM uses this)
- 4 layers: 89.6% IoU (minimal gain)

**Finding**: 2-layer decoder optimal (speed vs quality)

## MAE Pre-training Impact

**Training from scratch:**
- 100k iterations: 78.2% IoU
- Requires 5-10× more data

**With MAE pre-training:**
- 90k iterations: 89.4% IoU
- Converges faster, better generalization

**Finding**: MAE pre-training critical for efficiency

## ARR-COC Implications

**Architecture decisions for VLM spatial grounding:**
- Use pre-trained vision encoder (ViT-L or ViT-H)
- Multi-output for query ambiguity
- Confidence scoring (IoU-style prediction)
- Lightweight decoder for speed

**Sources**: SAM Paper Section 6 (Ablations), Appendix C
