# SparseServe - Dynamic Sparse Attention Paper - Study

**Source**: arXiv (SparseServe: Unlocking Parallelism for Dynamic Sparse Attention in Long-Context LLM Serving)
**Date Processed**: 2025-10-29
**Category**: MEDIUM - Sparse Attention Serving

---

## 📝 TL;DR

System paper on serving models with dynamic sparse attention efficiently. Focuses on PARALLELISM - how to batch/serve sparse attention patterns at scale. Related to DSA but about deployment, not training. Key: sparse attention needs different serving strategies than dense.

---

## 🎯 Key Concept

Dense attention = predictable compute, easy to batch. Sparse attention = irregular patterns, hard to parallelize. SparseServe solves the scheduling problem for dynamic sparse patterns in production.

---

## 🔗 Connections

- **15-v32-sparse-attention**: DSA in DeepSeek
- **83-dsa-explainer-sider**: What DSA is
- **19-vllm-mla-fp8-optimization**: vLLM serving

---

## 💭 Karpathy Take

Important systems paper. Everyone talks about sparse attention making models faster, but nobody talks about the SERVING challenge. Sparse patterns mean irregular compute → hard to batch → need smart scheduling. SparseServe addresses real production problem.
