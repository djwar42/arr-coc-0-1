# DeepSeek Source Document Categorization

**Total Documents**: 94
**Source**: `_ingest/DEEPSEEK-NEW-KARPATHY-IS-DEEPSSEK-EXPERKT/`

---

## 📊 Document Categories

### Category 1: DeepSeek Models (Core Architecture)
**Focus**: Official technical reports, model releases, architecture explanations

**Documents** (~15):
- DeepSeek-V3 Technical Report (arXiv, Studylib, VITALab)
- DeepSeek-V2 Technical Report
- DeepSeek-R1 papers (arXiv, PMC-NIH)
- DeepSeek-V3.2-Exp announcements
- DeepSeek-OCR paper
- DeepSeekMoE paper
- deepseek-ai GitHub repositories
- Hugging Face model pages

**Priority**: **HIGH** - Core knowledge

---

### Category 2: Multi-Head Latent Attention (MLA)
**Focus**: MLA mechanism, memory efficiency, implementations

**Documents** (~12):
- "A Gentle Introduction to Multi-Head Latent Attention"
- "Decoding Multi-Head Latent Attention (Part 1)"
- "DeepSeek-V3 Explained, Part 1: Understanding Multi-Head Latent Attention"
- "Hardware-Centric Analysis of DeepSeek's Multi-Head Latent Attention"
- "From GQA to MLA for a Better and More Memory-Efficient Attention Computation"
- "Towards Economical Inference: Enabling DeepSeek's MLA in Any Transformer"
- "TransMLA: Multi-Head Latent Attention Is All You Need"
- "Multi-head Temporal Latent Attention"
- "Enhancing DeepSeek models with MLA and FP8 optimizations in vLLM"

**Priority**: **HIGH** - Core technology

---

### Category 3: Mixture of Experts (MoE)
**Focus**: MoE architecture, load balancing, expert specialization

**Documents** (~10):
- "A Survey on Mixture of Experts in Large Language Models"
- "DeepSeekMoE: Towards Ultimate Expert Specialization"
- "AUXILIARY-LOSS-FREE LOAD BALANCING STRATEGY FOR MIXTURE-OF-EXPERTS"
- "DENSEMIXER: IMPROVING MOE POST-TRAINING VIA PRECISE ROUTER GRADIENT"
- "Dynamic Expert Specialization: Towards Catastrophic Forgetting-Free"
- "KTransformers: CPU/GPU Hybrid Inference for MoE Models"
- "DeepSeek's Low Inference Cost Explained: MoE & Strategy"

**Priority**: **HIGH** - Core architecture

---

### Category 4: FP8 & Quantization
**Focus**: 8-bit floating point, quantization techniques, training efficiency

**Documents** (~15):
- "FP8-LM: Training FP8 Large Language Models"
- "Fine-grained FP8" (Hugging Face)
- "33% faster LLM inference with FP8 quantization"
- "FP8: Efficient model inference with 8-bit floating point numbers"
- "8-bit numerical formats for deep neural networks"
- "Enhancing DeepSeek models with MLA and FP8 optimizations in vLLM"
- "Training and inference of large language models using 8-bit floating point"
- "Novel adaptive quantization methodology for 8-bit floating-point DNN training"
- "SHIFTED AND SQUEEZED 8-BIT FLOATING POINT FORMAT"
- "FP8 Quantization in NeMo RL"
- "NVFP4 for Efficient and Accurate Low-Precision Inference"
- "Optimizing LLMs for Performance and Accuracy with Post-Training Quantization"
- "Quantization" (Hugging Face)

**Priority**: **HIGH** - Core efficiency technique

---

### Category 5: ESFT (Expert Specialized Fine-Tuning)
**Focus**: Efficient fine-tuning for MoE models

**Documents** (~7):
- "deepseek-ai/ESFT: Expert Specialized Fine-Tuning" (GitHub)
- "Let the Expert Stick to His Last" (arXiv, ACL, Hugging Face - 3 versions)
- "ExpertWeave: Efficiently Serving Expert-Specialized Fine-Tuned Adapters"
- "DeepSeek AI Researchers Propose ESFT to Reduce Memory by up to 90%"
- "Comprehensive Guide to LLM Fine-Tuning"
- "Supervised Fine Tuning From Scratch"

**Priority**: **MEDIUM** - Important for practical use

---

### Category 6: GRPO & Reinforcement Learning
**Focus**: Group Relative Policy Optimization, RL for reasoning

**Documents** (~8):
- "GRPO (Group Relative Policy Optimization) explanation compared to PPO"
- "Theory Behind GRPO"
- "The Illustrated GRPO: A Detailed and Pedagogical Explanation"
- "DeepSeek R1: Understanding GRPO and Multi-Stage Training"
- "LLM Optimization: Optimizing AI with GRPO, PPO, and DPO"
- "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning"
- "No Supervision, No Problem: Pure RL Improves Mathematical Reasoning"

**Priority**: **MEDIUM** - R1-specific technique

---

### Category 7: Multi-Token Prediction
**Focus**: Predicting multiple tokens, improved planning

**Documents** (~7):
- "Better & Faster Large Language Models via Multi-token Prediction"
- "Accelerating SGLang with Multiple Token Prediction"
- "Understanding and Enhancing the Planning Capability via Multi-Token Prediction"
- "CodeFill: Multi-token Code Completion"
- "Multi-Token Prediction (MTP) — Megatron Core"
- "PARALLEL TOKEN GENERATION FOR LANGUAGE MODELS"
- "THINKING INTO THE FUTURE: LATENT LOOKAHEAD TRAINING"

**Priority**: **MEDIUM** - Advanced technique

---

### Category 8: Sparse Attention
**Focus**: Sparse attention mechanisms, long context efficiency

**Documents** (~5):
- "What Is DeepSeek Sparse Attention (DSA)"
- "Sparse Attention Mechanisms in Large Language Models"
- "SparseServe: Unlocking Parallelism for Dynamic Sparse Attention"
- "DeepSeek V3.2-Exp Cuts Long-Context Costs with DSA"
- "DeepSeek-V3.2-Exp Launches With Sparse Attention"

**Priority**: **MEDIUM** - V3.2-Exp specific

---

### Category 9: Hardware & Optimization
**Focus**: GEMM, CUDA, hardware-specific optimizations

**Documents** (~8):
- "DeepGEMM"
- "Welcome to CUTLASS"
- "EdgeProfiler: A Fast Profiling Framework for Lightweight LLMs"
- "Hardware-Centric Analysis of DeepSeek's MLA"
- "The Big LLM Architecture Comparison"
- "KTransformers: CPU/GPU Hybrid Inference"

**Priority**: **LOW** - Implementation details

---

### Category 10: Industry Analysis & Applications
**Focus**: Business impact, adoption, comparisons, use cases

**Documents** (~7):
- "DeepSeek Day Two: Focus Turns to Enterprise AI Adoption"
- "DeepSeek-V3 AI: 671B Model Costs $5.5M to Train"
- "DeepSeek - Where there is a will, there is a way"
- "DeepSeek is open-access and the next AI disrupter for radiology"
- "DeepSeek's Artificial Democratisation and Real Implications"
- "Why China Is On a Pace to Win the AI Race"
- "MICHAEL CEMBALEST • JP MORGAN"
- "Comparative Analysis of OpenAI GPT-4o and DeepSeek R1"
- "Exploiting DeepSeek-R1: Breaking Down Chain of Thought Security"

**Priority**: **LOW** - Context, not technical depth

---

## 📋 Processing Strategy

### Phase 1: HIGH Priority (Core Knowledge) - ~52 docs
1. **DeepSeek Models** (15 docs) - Start here
2. **MLA** (12 docs)
3. **MoE** (10 docs)
4. **FP8 & Quantization** (15 docs)

### Phase 2: MEDIUM Priority (Advanced Techniques) - ~27 docs
5. **ESFT** (7 docs)
6. **GRPO & RL** (8 docs)
7. **Multi-Token Prediction** (7 docs)
8. **Sparse Attention** (5 docs)

### Phase 3: LOW Priority (Context & Implementation) - ~15 docs
9. **Hardware & Optimization** (8 docs)
10. **Industry Analysis** (7 docs)

---

## 📝 Document Processing Template

For each document:
1. Create numbered folder: `{category}-{number}-{short-name}/`
2. Copy original document
3. Create `00-STUDY.md` with:
   - **Summary** - Key points (3-5 bullets)
   - **Core Concepts** - Main ideas explained
   - **Technical Details** - Algorithms, architectures, metrics
   - **Code/Examples** - If present
   - **Related Documents** - Cross-references
   - **Knowledge Category Mapping** - Which categories this feeds into

---

**Last Updated**: 2025-10-28
**Status**: Categorization complete, ready for processing
