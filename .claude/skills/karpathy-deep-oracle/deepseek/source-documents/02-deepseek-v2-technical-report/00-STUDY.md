# DeepSeek-V2 Technical Report - Study

**Source**: arXiv (DeepSeek-V2_ A Strong, Economical, and Efficient Mixture-of-Experts Language Model - arXiv.md)
**Date Processed**: 2025-10-28
**Category**: DeepSeek Models (Core Architecture)

---

## 📝 TL;DR

DeepSeek-V2 is where **MLA** and **DeepSeekMoE** were introduced - the two core architectures that V3 built on.

**236B total params, 21B active** (vs V3's 671B total, 37B active)

Key innovations:
- **Multi-head Latent Attention (MLA)**: 93.3% KV cache reduction
- **DeepSeekMoE**: Fine-grained experts, 42.5% training cost savings
- **5.76× throughput** improvement over dense DeepSeek 67B

This is the foundation. V3 is basically "V2 but bigger + FP8 + aux-loss-free balancing + multi-token prediction".

---

## 🎯 Core Innovations

### Multi-Head Latent Attention (MLA)
**Problem**: Standard attention has massive KV cache (stores keys and values for all tokens)

**Solution**: Low-rank compression of KV cache into latent vectors

**Results**:
- **93.3% KV cache reduction**
- Enables efficient inference even at long context lengths
- No significant performance loss

**How it works**:
- Compresses K and V into shared latent representation
- Decompresses during attention computation
- Much smaller memory footprint

**Why it matters**: Makes inference economical for large models.

### DeepSeekMoE (Fine-Grained MoE)
**Problem**: Standard MoE has routing challenges and load imbalance

**Solution**: Fine-grained experts with better routing

**Features**:
- **Smaller experts, more of them** (vs large coarse-grained experts)
- Better expert specialization
- Improved load balancing

**Results**:
- **42.5% training cost savings** vs dense 67B model
- Maintains or improves performance
- Enables economical scaling

**Why it matters**: MoE that actually works at scale.

---

## 📊 Performance vs Cost

| Metric | DeepSeek 67B (Dense) | DeepSeek-V2 (MoE) | Improvement |
|--------|---------------------|-------------------|-------------|
| **Training Cost** | Higher baseline | -42.5% | Massive savings |
| **KV Cache** | Higher baseline | -93.3% | Way smaller |
| **Max Throughput** | 1× baseline | 5.76× | Much faster |
| **Parameters** | 67B total, 67B active | 236B total, 21B active | Sparse wins |

**Bottom line**: V2 proves that sparse models (MoE) + efficient attention (MLA) = economical scaling.

---

## 🔧 Training Details

**Pre-training**:
- **8.1T tokens** (high-quality, multi-source)
- **128K context length**
- Supervised fine-tuning (SFT) + Reinforcement learning (RL)

**Performance**:
- Top-tier among open-source models
- Only 21B activated params but competitive with much larger dense models

---

## 💡 Why V2 Matters for Understanding V3

**V2 introduced the architecture**:
- MLA ← still used in V3
- DeepSeekMoE ← refined in V3 (aux-loss-free balancing)

**V3 added the engineering**:
- FP8 mixed precision
- DualPipe for better pipeline parallelism
- Auxiliary-loss-free load balancing (V2 used auxiliary loss)
- Multi-token prediction objective

**Evolution**: V2 = architecture, V3 = architecture + engineering

---

## 🔗 Connections

**Connects to V3**:
- V3 uses same MLA and MoE core
- V3 scales up (671B vs 236B)
- V3 removes auxiliary loss for balancing
- V3 adds FP8 and DualPipe

**Connects to Codebases**:
- `deepseek/codebases/09-FlashMLA/` - MLA implementation
- `deepseek/codebases/03-DeepSeek-MoE/` - MoE architecture
- `deepseek/codebases/05-DeepSeek-V3/` - V3 evolution

**Connects to Knowledge Categories**:
- Model architectures (MLA, MoE)
- Training efficiency (sparse computation)
- Inference optimization (KV cache compression)

---

## 📚 Key Sections to Deep Dive Later

**Section 2: Architecture**:
- 2.1: Multi-Head Latent Attention details
- 2.2: DeepSeekMoE architecture
- 2.3: Load balancing strategies (auxiliary loss based)

**Comparison to V3**:
- V2: Auxiliary loss for load balancing
- V3: Auxiliary-loss-free balancing (performance improvement)

---

**Last Updated**: 2025-10-28
**Status**: Core study complete
**Note**: V2 is the foundation, V3 is the refinement
