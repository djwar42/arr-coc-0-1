# Model Architectures - Overview

**Category**: Core architectural innovations in DeepSeek models
**Source Documents**: V3, V2, MoE, MLA papers
**Date Created**: 2025-10-28

---

## 🎯 What This Category Covers

The architectural foundations that make DeepSeek models efficient and powerful:
- **Multi-Head Latent Attention (MLA)** - 93% KV cache reduction
- **DeepSeekMoE** - Fine-grained experts with shared experts
- **Multi-Token Prediction** - Predicting multiple future tokens
- **Auxiliary-Loss-Free Balancing** - Load balancing without performance hit

---

## 📊 Core Architectures

### 1. Multi-Head Latent Attention (MLA)
**What**: Compresses KV cache into low-dimensional latent space
**Why**: Inference is memory-bound at long context
**Impact**: 93.3% KV cache reduction, 5.76× faster inference

**Source**: DeepSeek-V2 (introduced), V3 (continued)
**Details**: [01-multi-head-latent-attention.md](01-multi-head-latent-attention.md)

### 2. DeepSeekMoE (Mixture of Experts)
**What**: Fine-grained experts (m×N) + shared experts (Ks)
**Why**: Better expert specialization, less redundancy
**Impact**: 40-70% compute savings vs dense models

**Source**: DeepSeekMoE paper, V2, V3
**Details**: [02-deepseek-moe.md](02-deepseek-moe.md)

### 3. Multi-Token Prediction
**What**: Train to predict multiple future tokens (not just next token)
**Why**: Improves benchmark performance, enables speculative decoding
**Impact**: Better predictions, faster inference potential

**Source**: DeepSeek-V3
**Details**: [03-multi-token-prediction.md](03-multi-token-prediction.md)

### 4. Auxiliary-Loss-Free Load Balancing
**What**: Load balance MoE without auxiliary loss
**Why**: Auxiliary loss hurts performance to force balance
**Impact**: Better performance + balanced experts

**Source**: DeepSeek-V3
**Details**: [04-aux-loss-free-balancing.md](04-aux-loss-free-balancing.md)

---

## 🔄 Architecture Evolution

**V2 → V3 Changes**:
| Feature | V2 | V3 |
|---------|----|----|
| **MLA** | ✓ Introduced | ✓ Continued |
| **MoE** | Fine-grained + shared | Same design |
| **Load Balancing** | Auxiliary loss | Aux-loss-free |
| **Training Objective** | Next token | Multi-token |
| **Scale** | 236B total, 21B active | 671B total, 37B active |

**Key insight**: V3 refined V2's architecture, didn't rebuild it.

---

## 💡 Design Principles (Karpathy's Take)

**On MLA**:
- Memory is the bottleneck for long context
- Learned compression beats hand-crafted
- Small compute overhead worth huge memory savings

**On MoE**:
- Fine-grained > coarse-grained experts
- Shared experts handle common knowledge elegantly
- Specialization emerges naturally with right design

**On Aux-Loss-Free Balancing**:
- Don't force balance if it hurts performance
- Let routing learn naturally
- V3 proves you can have both (balance + performance)

**On Multi-Token Prediction**:
- Simple idea, works well
- Not revolutionary but helpful
- Bonus: enables speculative decoding

---

## 🔗 Cross-References

**Connects to Training Efficiency**:
- MoE enables sparse computation
- MLA reduces memory footprint
- Together: economical scaling

**Connects to Codebases**:
- `deepseek/codebases/09-FlashMLA/` - MLA implementation
- `deepseek/codebases/03-DeepSeek-MoE/` - MoE architecture
- `deepseek/codebases/05-DeepSeek-V3/` - Full V3 architecture

**Connects to Source Documents**:
- [V3 Technical Report](../../source-documents/01-deepseek-v3-technical-report/00-STUDY.md)
- [V2 Technical Report](../../source-documents/02-deepseek-v2-technical-report/00-STUDY.md)
- [DeepSeekMoE Paper](../../source-documents/03-deepseek-moe-paper/00-STUDY.md)
- [MLA Explained](../../source-documents/06-mla-explained/00-STUDY.md)

---

**Last Updated**: 2025-10-28
**Status**: Active - core category for DeepSeek architecture understanding
