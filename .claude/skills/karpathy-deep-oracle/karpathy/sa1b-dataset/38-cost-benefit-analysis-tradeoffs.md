# Cost-Benefit Analysis: Training vs Fine-tuning SAM

## Overview

**Training from scratch**: $70k-100k, 68 hours on 256 A100s, full SA-1B (11M images)
**Fine-tuning**: $100-500, 2-10 hours on 1-8 GPUs, 1k-10k images

## Training from Scratch

**Costs:**
- Compute: 256× A100 × 68 hours = 17,408 GPU-hours
- Cloud (AWS p4d): ~$70k-100k total
- Storage: ~10TB ($50-100/month)
- Time: 3 days

**Benefits:**
- Full model control
- Custom architecture
- Domain-specific from ground up

**When to use:**
- Building foundation model
- Novel architecture
- Unlimited budget

## Fine-tuning Pre-trained SAM

**Costs:**
- Compute: 1-8× GPUs × 2-10 hours = 2-80 GPU-hours
- Cloud (1× A100): ~$100-500 total
- Storage: ~1-100GB ($1-10/month)
- Time: Hours to 1 day

**Benefits:**
- 100-1000× cheaper
- 10-100× less data needed (1k-10k vs 11M images)
- Better generalization (ViT transfer)
- Faster iteration

**When to use:**
- Domain adaptation (medical, satellite)
- Limited budget
- Quick prototyping

## Cost Comparison Table

| Approach | Compute Cost | Data Needed | Time | Quality |
|----------|--------------|-------------|------|---------|
| Scratch | $70k-100k | 11M images | 68 hours (256 GPUs) | 100% |
| Fine-tune | $100-500 | 1k-10k images | 2-10 hours (1-8 GPUs) | 85-95% |
| Zero-shot | $0 | 0 images | 0 hours | 70-80% |

## Model Size Tradeoffs

**ViT-H (632M params):**
- Best quality (89.4% IoU)
- Slowest inference (~200ms)
- Largest checkpoint (2.4GB)

**ViT-L (308M params):**
- Good quality (88.1% IoU)
- Fast inference (~100ms)
- Medium checkpoint (1.2GB)

**ViT-B (91M params):**
- Decent quality (86.2% IoU)
- Fastest inference (~50ms)
- Small checkpoint (350MB)

## ARR-COC Recommendations

**Phase 1 (Prototyping):**
- Fine-tune ViT-B on 1k document layouts
- Cost: ~$100, 2-4 hours on 1 GPU
- Quality: 80-85% for spatial grounding

**Phase 2 (Production):**
- Fine-tune ViT-L on 10k documents
- Cost: ~$500, 10 hours on 8 GPUs
- Quality: 85-90% for relevance regions

**Phase 3 (Research):**
- Consider training from scratch if:
  - Novel architecture needed
  - Multi-modal integration (text + vision)
  - Unlimited research budget

**Recommendation**: Fine-tune ViT-L (best cost/quality balance)

**Sources**: SAM Paper, Cloud GPU pricing (AWS/GCP), ML engineering best practices
