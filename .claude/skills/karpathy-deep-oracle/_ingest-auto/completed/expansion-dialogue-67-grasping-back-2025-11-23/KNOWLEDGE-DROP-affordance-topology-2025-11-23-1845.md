# KNOWLEDGE DROP: Affordance Space Topology

**Created**: 2025-11-23 18:45
**Runner**: PART 22
**File Created**: gibson-affordances/02-affordance-space-topology.md
**Lines**: ~700

---

## Summary

Created comprehensive documentation on affordance space topology - the mathematical structure of action possibilities in Gibson's ecological psychology framework.

## Key Concepts Covered

### Core Framework
- Affordance space as continuous manifold of action possibilities
- Metric, topology, and dynamics structure
- Landscape metaphor: valleys (good actions), peaks (avoided), saddle points (decisions)

### Major Research Synthesized

1. **Pezzulo & Cisek (2016)** - "Navigating the Affordance Landscape"
   - Affordance competition framework
   - Hierarchical selection
   - Prediction-driven behavior

2. **Scholz et al. (2022)** - Neural affordance maps
   - End-to-end learning of affordance codes
   - Active inference navigation
   - Zero-shot generalization

3. **Cisek (2007)** - Affordance competition hypothesis
   - Neural implementation in dorsal stream
   - Parallel action specification
   - Competition in fronto-parietal cortex

4. **Bennett et al. (2024)** - Uncontrolled manifold analysis
   - Task-relevant vs redundant dimensions
   - Dynamical systems perspective

### Mathematical Formalizations
- Dynamical systems: dx/dt = f(x, a, e)
- Landscape function: L(a) = C(a) - V(a) + U(a)
- Expected free energy for navigation
- Topological data analysis (persistent homology, Betti numbers)

### Navigation Principles
- Gradient descent on affordance landscape
- Active inference: minimize expected free energy
- Hierarchical planning: coarse to fine
- Feedback control with continuous monitoring

## ARR-COC-0-1 Integration (Section 8)

### Relevance Realization as Affordance Navigation
- Cognitive affordances (what can be thought/attended)
- Information environment instead of physical
- Attentional allocations as actions
- Understanding as outcome

### Key Mappings
| Affordance Theory | ARR-COC-0-1 |
|-------------------|-------------|
| Affordance space | Relevance space |
| Action selection | Token weighting |
| Navigation | Inference |
| Attractors | Interpretations |

### Novel Contributions
- Relevance landscape formulation: R(a) = Info_Gain - Cost - Uncertainty
- Active inference for relevance selection
- Topological properties of relevance space
- Temporal thickness in relevance navigation
- Zero-shot generalization through local relevance codes

## Sources Cited

**Primary**:
- Pezzulo & Cisek (2016) - Trends in Cognitive Sciences
- Scholz et al. (2022) - Frontiers in Neurorobotics
- Cisek (2007) - Phil Trans Royal Society B
- Bennett et al. (2024) - PLOS ONE
- Bruineberg & Rietveld (2014) - Frontiers Human Neuroscience
- Butler (2024) - Phenomenology and Cognitive Sciences
- Kiverstein et al. (2021) - Synthese

**Foundational**:
- Gibson (1979) - The Ecological Approach to Visual Perception
- Tolman (1948) - Cognitive maps
- Friston (2010) - Free energy principle

## Quality Assessment

- **Depth**: High - covers mathematical formalizations, neural implementations, computational models
- **Breadth**: Good - spans ecological psychology, motor control, cognitive science, AI
- **ARR-COC Integration**: Strong - detailed mapping of affordance concepts to relevance realization
- **Sources**: Well-cited with access dates and DOIs

## Connections to Other Oracle Content

- Links to: friston/ (active inference, free energy)
- Links to: gibson-affordances/00-ecological-psychology.md (foundational affordance theory)
- Links to: gibson-affordances/01-direct-perception.md (direct perception)
- Links to: topology-4d/ (manifolds, saddle points)
- Links to: cognitive-mastery/ (attention, salience)

---

**Status**: COMPLETE
**Next**: Mark checkbox in ingestion.md
