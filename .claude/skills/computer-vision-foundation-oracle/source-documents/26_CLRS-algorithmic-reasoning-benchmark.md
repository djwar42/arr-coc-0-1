---
sourceFile: "CLRS Algorithmic Reasoning Benchmark (Google DeepMind)"
exportedBy: "Bright Data Web Scraper"
exportDate: "2025-10-28"
sourceURL: "https://github.com/google-deepmind/clrs and https://arxiv.org/abs/2205.15659"
---

# CLRS Algorithmic Reasoning Benchmark

## Overview

The CLRS Algorithmic Reasoning Benchmark consolidates and extends previous work toward evaluation of algorithmic reasoning by providing a suite of implementations of classical algorithms from the "Introduction to Algorithms" textbook (CLRS).

## Key Details

**Authors**: Petar Veličković, Adrià Puigdomènech Badia, David Budden, Razvan Pascanu, Andrea Banino, Misha Dashevskiy, Raia Hadsell, Charles Blundell

**Organization**: Google DeepMind

**Publication**: ICML 2022 (arXiv:2205.15659)

**Citation Count**: 134 citations

**GitHub**: https://github.com/google-deepmind/clrs

**Stars**: 507 stars, 109 forks

## CLRS-30 Benchmark

30 classical algorithms across multiple categories:

### Sorting (4 algorithms)
- Insertion sort
- Bubble sort
- Heapsort (Williams, 1964)
- Quicksort (Hoare, 1962)

### Searching (3 algorithms)
- Minimum
- Binary search
- Quickselect (Hoare, 1961)

### Divide and Conquer (1 algorithm)
- Maximum subarray (Kadane's variant) (Bentley, 1984)

### Greedy (2 algorithms)
- Activity selection (Gavril, 1972)
- Task scheduling (Lawler, 1985)

### Dynamic Programming (3 algorithms)
- Matrix chain multiplication
- Longest common subsequence
- Optimal binary search tree (Aho et al., 1974)

### Graphs (12 algorithms)
- Depth-first search (Moore, 1959)
- Breadth-first search (Moore, 1959)
- Topological sorting (Knuth, 1973)
- Articulation points
- Bridges
- Kosaraju's strongly connected components (Aho et al., 1974)
- Kruskal's minimum spanning tree (Kruskal, 1956)
- Prim's minimum spanning tree (Prim, 1957)
- Bellman-Ford single-source shortest paths (Bellman, 1958)
- Dijkstra's single-source shortest paths (Dijkstra et al., 1959)
- DAG single-source shortest paths
- Floyd-Warshall all-pairs shortest-paths (Floyd, 1962)

### Strings (2 algorithms)
- Naïve string matching
- Knuth-Morris-Pratt (KMP) string matcher (Knuth et al., 1977)

### Geometry (2 algorithms)
- Segment intersection
- Graham scan convex hull (Graham, 1972)
- Jarvis' march convex hull (Jarvis, 1973)

## Dataset Structure

**Training Trajectories**: 1,000 per algorithm (problem size 16)
**Evaluation Trajectories**: 32 × multiplier (problem size 16)
**Test Trajectories**: 32 × multiplier (problem size 64)

The "multiplier" varies by algorithm to compensate for evaluation signal paucity.

## Algorithms as Graphs

CLRS represents all algorithms using graph structures:
- **Nodes**: Objects being manipulated
- **Edges**: Relations between objects
- **Ordering**: Imposed through predecessor links for arrays/trees

## Key Features

1. **Procedural Generation**: Generate unlimited training data
2. **Trajectory Hints**: Expose internal algorithm state
3. **Out-of-Distribution Testing**: Evaluate generalization to larger problem sizes
4. **Standardized Evaluation**: Fair comparison across methods

## CLRS-Text (2024 Extension)

**Paper**: "The CLRS-Text Algorithmic Reasoning Language Benchmark" (arXiv:2406.04229)

**Authors**: Larisa Markeeva, Sean McLeish, Borja Ibarz, and others

**Innovation**: Text-based variant suitable for language models
- Textual representations of algorithmic traces
- Enables evaluation of LLM algorithmic reasoning
- Same 30 algorithms as CLRS-30
- Procedural generation of text traces

## Provided Baseline Processors

JAX implementations of GNN processors:
- Deep Sets (Zaheer et al., NIPS 2017)
- End-to-End Memory Networks (Sukhbaatar et al., NIPS 2015)
- Graph Attention Networks (Veličković et al., ICLR 2018)
- Graph Attention Networks v2 (Brody et al., ICLR 2022)
- Message-Passing Neural Networks (Gilmer et al., ICML 2017)
- Pointer Graph Networks (Veličković et al., NeurIPS 2020)

## Installation

```bash
pip install dm-clrs
# or
pip install git+https://github.com/google-deepmind/clrs.git
```

## Usage Example

```python
train_ds, num_samples, spec = clrs.create_dataset(
    folder='/tmp/CLRS30',
    algorithm='bfs',
    split='train',
    batch_size=32)

for i, feedback in enumerate(train_ds.as_numpy_iterator()):
    if i == 0:
        model.init(feedback.features, initial_seed)
    loss = model.feedback(rng_key, feedback)
```

## Data Structure

```python
Feedback = collections.namedtuple('Feedback', ['features', 'outputs'])
Features = collections.namedtuple('Features', ['inputs', 'hints', 'lengths'])
```

- **inputs**: Input data to the algorithm
- **hints**: Internal state trajectory (with time dimension)
- **lengths**: True length of each trajectory (for masking)
- **outputs**: Ground truth for evaluation

## Significance

CLRS provides:
1. **Unified evaluation**: Standard benchmark for algorithmic reasoning
2. **Reproducibility**: Public dataset and baseline code
3. **Extensibility**: Easy to add new algorithms
4. **Research focus**: Identifies key challenges in neural algorithmic reasoning

## Key Research Findings

From the multimodal reasoning workshop:
- LLMs struggle with simple algorithmic tasks despite massive scale
- Gemini 1.5: Only 37% accuracy on 3×3 multiplication
- 60% accuracy drop when problem size exceeds training distribution
- Auto-regressive bottleneck limits reasoning capability

## Related Extensions

- **SALSA-CLRS**: Sparse and scalable variant
- **TransNAR**: Transformer + GNN hybrid architecture
- **Distillation approaches**: From multimodal to unimodal models

## Resources

- **GitHub**: https://github.com/google-deepmind/clrs
- **Paper**: https://arxiv.org/abs/2205.15659
- **CLRS-Text Paper**: https://arxiv.org/abs/2406.04229
- **Dataset Download**: https://storage.googleapis.com/dm-clrs/CLRS30_v1.0.0.tar.gz
- **License**: Apache-2.0

## Citation

```bibtex
@article{deepmind2022clrs,
  title={The CLRS Algorithmic Reasoning Benchmark},
  author={Petar Veličković and Adrià Puigdomènech Badia and
    David Budden and Razvan Pascanu and Andrea Banino and Misha Dashevskiy and
    Raia Hadsell and Charles Blundell},
  journal={arXiv preprint arXiv:2205.15659},
  year={2022}
}
```
