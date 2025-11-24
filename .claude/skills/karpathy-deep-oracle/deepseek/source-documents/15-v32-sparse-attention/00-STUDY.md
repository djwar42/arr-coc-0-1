# V3.2 Sparse Attention - Study

**Source**: Sirius Digitals (DeepSeek V3.2-Exp Cuts Long-Context Costs with DSA)
**Date Processed**: 2025-10-28
**Category**: Model Architectures (Sparse Attention)

---

## 📝 Summary

DeepSeek-V3.2-Exp introduces **DeepSeek Sparse Attention (DSA)** for long-context efficiency.

**Key innovation**: Sparse attention patterns for contexts beyond 128K tokens
**Benefit**: 50% cost reduction for long-context inference
**Trade-off**: Maintains benchmark parity (no quality loss)

**Status**: Experimental (V3.2-Exp), may integrate into production models

---

## 🎯 What is DSA?

**Problem**: Standard attention is O(n²) for sequence length n
**At 128K+ tokens**: Becomes bottleneck even with MLA

**DSA Solution**:
- Selective attention (not all tokens attend to all tokens)
- Learned sparsity patterns
- Maintains critical information flow

**Result**: Lower compute without hurting performance

---

## 🔗 Cross-References

- [V3 Technical Report](../01-deepseek-v3-technical-report/00-STUDY.md) - Base V3 architecture
- [MLA Explained](../06-mla-explained/00-STUDY.md) - Complementary to MLA
- Future: V3.2 may integrate DSA into production

---

**Last Updated**: 2025-10-28
**Status**: Experimental feature tracking
