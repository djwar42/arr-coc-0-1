# KNOWLEDGE DROP: Self-Organizing Neural Networks

**Date**: 2025-11-23 17:30
**Source**: Web research on self-organizing maps, neural gas, growing neural gas
**File Created**: ml-morphogenesis/03-self-organizing-nn.md

---

## What Was Added

Comprehensive guide to self-organizing neural networks (~700 lines) covering:

1. **Self-Organizing Maps (Kohonen Maps)**
   - Algorithm and learning rules
   - Topology preservation
   - Grid variants (rectangular, hexagonal, toroidal)

2. **Growing Neural Gas (GNG)**
   - Incremental topology learning
   - Node insertion and edge aging
   - Continuous learning capability

3. **Neural Gas Algorithm**
   - Rank-based learning
   - Comparison with SOM

4. **Variants and Extensions**
   - Incremental GNG
   - Growing When Required
   - Plastic Neural Gas

5. **PyTorch Implementations**
   - Complete SOM class (~200 lines)
   - Complete GNG class (~200 lines)
   - GPU acceleration support

6. **TRAIN STATION: Self-Organization = Emergence = Autopoiesis**
   - Deep unification with morphogenesis
   - Connection to biological development
   - SOM as morphogenetic field
   - GNG as autopoiesis

7. **ARR-COC-0-1 Connection**
   - RelevanceSOM concept
   - Self-organizing relevance maps
   - Growing relevance networks

---

## Key Technical Content

### SOM Learning Rule
```
w_ij(t+1) = w_ij(t) + alpha(t) * h_ij(t) * (x - w_ij(t))
```

### GNG Core Operations
- Find nearest and second-nearest nodes
- Adapt winner and neighbors
- Age and prune edges
- Insert nodes based on error

### Performance Benchmarks
- GPU speedup: 100x over NumPy implementations
- 16,000 samples, 300 features: 12 seconds (GPU)

---

## TRAIN STATION Insight

**Self-organization = emergence = autopoiesis**

All share the same pattern:
- Local interactions produce global order
- No central controller
- Feedback loops stabilize patterns
- Topological constraints shape outcomes

The SOM grid IS a morphogenetic field:
- Neurons = cells
- Weights = cell states
- Neighborhood function = morphogen diffusion

GNG IS autopoietic:
- Network produces itself
- Nodes create nodes
- Structure maintains structure

---

## Sources

- [torchsom arXiv paper](https://arxiv.org/html/2510.11147v1)
- [Fritzke GNG paper](https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/fritzke95.pdf)
- [Wikipedia: Neural gas](https://en.wikipedia.org/wiki/Neural_gas)
- [TorchSOM GitHub](https://github.com/michelin/TorchSOM)

---

## Usage

```python
from ml_morphogenesis.self_organizing import SelfOrganizingMap, GrowingNeuralGas

# Train SOM
som = SelfOrganizingMap(input_dim=4, map_size=(10, 10))
som.fit(data, n_epochs=50)

# Train GNG
gng = GrowingNeuralGas(input_dim=2, max_nodes=50)
gng.fit(data, n_passes=5)
```

---

## Integration Points

- Connect to morphogenesis concepts (NCA, bioelectric)
- Link to predictive coding (self-organization in hierarchies)
- Bridge to active inference (autopoiesis = self-evidencing)
