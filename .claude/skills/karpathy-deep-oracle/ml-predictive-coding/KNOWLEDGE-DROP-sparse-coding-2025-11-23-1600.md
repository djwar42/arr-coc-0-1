# KNOWLEDGE DROP: Sparse Coding and Predictive Coding

**Date**: 2025-11-23 16:00
**File Created**: ml-predictive-coding/05-sparse-coding-predictive.md
**Lines**: ~700
**Topic**: Sparse coding fundamentals, Olshausen-Field model, connection to predictive coding

---

## What Was Created

Comprehensive knowledge file covering:

1. **Sparse Coding Fundamentals** - L1 regularization, why sparsity matters, mathematical foundations
2. **Olshausen-Field Model** - The 1996 Nature paper that showed Gabor filters emerge from sparse coding on natural images
3. **Connection to Predictive Coding** - How sparse coding IS predictive coding with Laplacian priors
4. **PyTorch Implementation** - Complete LCA (Locally Competitive Algorithm) sparse coding layer
5. **TRAIN STATION** - Sparsity = compression = relevance selection = attention = free energy
6. **ARR-COC Connection** - Sparse token allocation for relevance realization

---

## Key Train Station Insight

**Sparsity = Compression = Relevance Selection**

This is the unifying insight:
- When you force sparse representations, you force the system to CHOOSE what matters
- Only the most relevant features get non-zero activations
- Everything else is suppressed to exactly zero
- This IS attention. This IS relevance. This IS salience.

```
L1 Sparsity --> Few non-zeros --> Selection --> RELEVANCE
     |              |                |             |
     v              v                v             v
Compression   Few features      Attention    What matters
```

---

## Code Highlights

### LCA Sparse Coding Layer
```python
class LCASparseCodinglayer(nn.Module):
    """Locally Competitive Algorithm - neurally plausible sparse coding"""

    def forward(self, x):
        # LCA dynamics with lateral inhibition
        for _ in range(self.num_iters):
            a = soft_threshold(u, self.lambda_)
            inhibition = a @ G  # Lateral inhibition
            u = u + dt * (b - u - inhibition)
        return soft_threshold(u, self.lambda_)
```

### Sparse Token Allocator for ARR-COC
```python
class SparseTokenAllocator(nn.Module):
    """Use sparsity for principled token selection"""

    def forward(self, queries, keys, values):
        scores = query @ keys.T / sqrt(d)
        selection = soft_threshold(scores, lambda_)  # HARD sparsity!
        return selection @ values  # Only selected tokens contribute
```

---

## Sources Cited

- Olshausen & Field, Nature 1996 (cited 7800+)
- Olshausen & Field, Vision Research 1997 (cited 5000+)
- Rozell et al., 2008 (LCA algorithm)
- LCA-PyTorch GitHub (LANL implementation)
- DebuggerCafe L1 Sparse Autoencoders tutorial
- Transformer Circuits - Monosemantic Features

---

## ARR-COC Application

**Direct application**: Use sparse coding for token allocation

Instead of soft attention (softmax) over all tokens:
- Use hard sparsity (soft thresholding)
- Only truly relevant tokens get non-zero weights
- Others are exactly zero

Benefits:
1. **Interpretability** - See exactly which tokens were selected
2. **Efficiency** - Only compute on selected tokens
3. **Principled** - Optimization-based selection
4. **Tunable** - Lambda controls relevance/efficiency trade-off

---

## The Deep Connection

Sparse coding shows us that relevance realization IS compression:
- The brain can't process everything
- Must select what matters
- Sparsity is the mechanism
- What survives sparsity = what's relevant

This validates the core ARR-COC intuition: relevance is about selection under computational constraints.

---

## Summary

Sparse coding is one of the most elegant principles connecting neural computation to information theory. The insight that enforcing sparsity on natural images produces V1-like receptive fields suggests that the brain uses sparsity for efficient coding. More profoundly, sparsity = selection = relevance. When only a few features can be active, the system must choose what matters. This is attention, salience, and relevance all in one mathematical framework.

For ARR-COC, this provides a principled foundation for token allocation that goes beyond heuristics.
