# Sparse Attention Survey - Study

**Source**: Clausius Scientific Press (Sparse Attention Mechanisms in Large Language Models: Applications, Classification, Performance Analysis, and Optimization)
**Date Processed**: 2025-10-29
**Category**: MEDIUM - Survey Paper

---

## 📝 TL;DR

Survey paper categorizing sparse attention methods: fixed patterns (windows, blocks), learned patterns (routing), hybrid approaches. Performance analysis across methods. DeepSeek's DSA fits in "learned/dynamic" category. Good reference for landscape of sparse attention research.

---

## 🎯 Categories

- **Fixed**: Window, block, strided attention
- **Learned**: Dynamic routing, content-based selection
- **Hybrid**: Combine fixed structure + learned selection

DSA = learned (content-aware lightning indexer)

---

## 🔗 Connections

- **15-v32-sparse-attention**: DSA implementation
- **83-dsa-explainer-sider**: DSA specifics

---

## 💭 Karpathy Take

Survey papers are great for context. Shows DSA isn't first sparse attention (we've had fixed patterns for years), but it's in the newer "learned/dynamic" category that adapts to content. Survey probably written before V3.2-Exp, so DSA might not be included.
