# KNOWLEDGE DROP: Morphogenetic Field Learning

**Date**: 2025-11-23 16:00
**Source**: Web research on neural fields, SIREN, NeRF, implicit representations
**File Created**: ml-morphogenesis/04-morphogenetic-field-learning.md
**Lines**: ~700

---

## Key Insights Acquired

### Neural Fields = Morphogenetic Fields

The core insight is that neural fields (coordinate-based networks) are computationally equivalent to morphogenetic fields in biology:

| Biological | Neural |
|------------|--------|
| Morphogen concentration | Network output value |
| Cell position | Input coordinates |
| Diffusion gradient | Spatial derivative |
| Cell fate threshold | Activation threshold |

### SIREN is Critical for Gradients

From Sitzmann et al. (cited 3691 times):
- ReLU networks have piecewise constant first derivatives
- ReLU networks have ZERO second derivatives
- SIREN uses sin() activation: f(x) = sin(omega_0 * Wx + b)
- SIREN derivatives are also sinusoids - smooth everywhere!
- 10+ dB higher PSNR than ReLU baselines

This matters because morphogenetic guidance requires accurate gradients!

### The TRAIN STATION Unification

```
Neural Field = NeRF = SDF = Morphogenetic Field
           \    |    /        |
            \   |   /         |
             SAME THING:
        Implicit continuous function
        Gradient-based optimization
        Coordinate-based queries
```

### Positional Encoding

From Tancik et al. (Fourier Features):
```python
gamma(p) = [sin(2^0 * pi * p), cos(2^0 * pi * p), ...]
```

This lifts low-D coordinates to high-D space, enabling high-frequency learning.

---

## Code Patterns Provided

1. **SineLayer**: SIREN activation with proper initialization
2. **SIREN**: Complete network with gradient computation
3. **PositionalEncoding**: Fourier features for NeRF
4. **NeuralRadianceField**: Full NeRF with volume rendering
5. **MorphogeneticField**: SIREN-based morphogen concentration
6. **ReactionDiffusionField**: Spatiotemporal morphogenesis
7. **ImplicitRelevanceField**: ARR-COC relevance as neural field

---

## ARR-COC Connection (10%)

Model relevance as an implicit field over (token, concept) space:
- Continuous relevance landscape (smooth interpolation)
- Gradient-based token allocation (morphogenetic guidance)
- Implicit compression (relevance in weights)
- Morphogenetic interpretation (tokens "develop" relevance)

---

## Sources

- SIREN paper (NeurIPS 2020, 3691 citations)
- NeRF project page
- Neural Fields FAQ (Brown University)
- Awesome Implicit Representations (GitHub)
- Multiple PyTorch implementations

---

## Integration Notes

This file connects to:
- ml-topology/* (loss landscapes as neural fields)
- ml-predictive-coding/* (prediction = implicit representation)
- ml-active-inference/* (free energy = field energy)
- ml-morphogenesis/00-03 (neural cellular automata, bioelectric, GNN, self-organizing)
