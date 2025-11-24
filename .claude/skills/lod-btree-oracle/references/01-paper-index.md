# Research Paper Index

**Complete bibliography of source materials for lod-btree-oracle**

---

## Source Materials Location

**Primary Sources**: `RESEARCH/Video Game LOD and BTree/NotebookLM_Sources/`
**Count**: 20 research papers
**Supplementary**: Web research via Bright Data (2024-2025)

---

## Core BSP/Spatial Partitioning

### Binary space partitioning - Wikipedia
**Topics**: BSP tree construction, spatial hierarchies, history
**Key Contributions**: Foundational BSP concepts, algorithm overview
**Use For**: Understanding BSP basics, historical context

### Athena3D
**Topics**: 3D spatial partitioning implementation
**Key Contributions**: Practical BSP application
**Use For**: Implementation details

**Supplementary Web**:
- [Valve Developer Wiki: Binary Space Partitioning](https://developer.valvesoftware.com/wiki/Binary_space_partitioning)
- "How Doom Used BSP Trees" (twobithistory.org)

---

## Level of Detail Systems

### Progressive Buffers: View-dependent Geometry and Texture LOD Rendering
**Topics**: View-dependent LOD, progressive meshes, smooth transitions
**Key Contributions**: Complete LOD pipeline for geometry and textures
**Use For**: Understanding modern LOD techniques

### Integrating Occlusion Culling with Parallel LOD for Rendering Complex 3D Environments on GPU
**Topics**: GPU-based LOD, occlusion culling integration, parallel processing
**Key Contributions**: Combining LOD with visibility determination
**Use For**: Real-time rendering optimization

### LOD Map - A Visual Interface for Navigating Multiresolution Volume Visualization
**Topics**: LOD navigation, volume rendering, user interfaces
**Key Contributions**: Interactive LOD control for scientific visualization
**Use For**: User-driven LOD selection

---

## Perceptual Rendering & Visual Attention

### Attention and Visual Memory in Visualization and Computer Graphics
**Authors**: Christopher G. Healey, James T. Enns
**Topics**: Preattentive processing, visual salience, change blindness, inattentional blindness
**Key Contributions**: Comprehensive survey of visual perception research for graphics
**Use For**: Understanding human visual system, perceptual foundations

### Gaze-aware Displays and Interaction - SURREAL TEAM
**Authors**: Katerina Mania, Ann McNamara, Andreas Polychronakis
**Source**: SIGGRAPH 2021 Course Notes
**Topics**: Eye tracking, gaze-contingent displays, foveated rendering, VR interaction
**Key Contributions**: Modern gaze-aware rendering techniques
**Use For**: Foveated rendering implementation, VR optimization

### Managing Level of Detail through Head-Tracked Peripheral Degradation: A Model and Resulting Design Principles
**Topics**: Head tracking, peripheral vision degradation, LOD design principles
**Key Contributions**: Biologically-grounded LOD allocation
**Use For**: Perceptual LOD strategies

### Visual Attention in 3D Video Games
**Topics**: Player attention, game rendering optimization, immersion
**Key Contributions**: Domain-specific visual attention analysis
**Use For**: Game development applications

### Focus Guided Light Field Saliency Estimation
**Topics**: Salience detection, light field rendering, attention modeling
**Key Contributions**: Computational salience models
**Use For**: Automatic salience-based LOD

### Accentuating focus maps via partial schematization
**Topics**: Focus+context visualization, selective detail, abstraction
**Key Contributions**: Partial schematization techniques
**Use For**: Artistic LOD control

### An Interactive Perceptual Rendering Pipeline using Contrast and Spatial Masking
**Topics**: Perceptual rendering, masking effects, quality metrics
**Key Contributions**: Perceptual optimization pipeline
**Use For**: Perceptually-guided rendering

**Supplementary Web**:
- [Towards Attention-aware Foveated Rendering](https://www.computationalimaging.org/publications/attention-aware/) (Stanford Computational Imaging)
- Meta's Fixed Foveated Rendering Documentation

---

## Terrain & Geometry

### Fast, Realistic Terrain Synthesis
**Topics**: Procedural generation, heightfield synthesis, erosion simulation
**Key Contributions**: Efficient terrain generation algorithms
**Use For**: Terrain creation, procedural content

### Watertight Incremental Heightfield Tessellation
**Topics**: Real-time tessellation, crack prevention, GPU implementation
**Key Contributions**: Watertight adaptive tessellation
**Use For**: Terrain rendering, GPU tessellation

### Opportunities with Slippy Maps for Terrain Visualization in Virtual and Augmented Reality
**Topics**: Tile-based rendering, streaming, geographic visualization
**Key Contributions**: Web mapping techniques for VR/AR
**Use For**: Large-scale terrain systems

---

## VR/AR Applications

### Virtual Reality Annual International Symposium, March 1-5, 1997, Albuquerque, New Mexico
**Topics**: VR rendering, stereoscopic displays, latency optimization
**Key Contributions**: Historical VR techniques still relevant today
**Use For**: VR-specific LOD considerations

---

## Urban & Semantic Visualization

### Applying level-of-detail and perceptive effects to 3D urban semantics visualization
**Topics**: City modeling, semantic LOD, urban planning
**Key Contributions**: Domain-specific LOD for urban environments
**Use For**: Semantic LOD, architectural visualization

---

## Personalization & Recommendation

### Exploiting LOD-Based Similarity Personalization Strategies for Recommender Systems
**Topics**: User modeling, personalized rendering, adaptive quality
**Key Contributions**: LOD for recommendation systems
**Use For**: Personalized LOD strategies

### Analysis and Design of a Personalized Recommendation System Based on a Dynamic User Interest Model
**Topics**: User preference modeling, dynamic adaptation
**Key Contributions**: User-aware system design
**Use For**: Adaptive LOD allocation

---

## Database & OLAP

### QUERY, ANALYSIS, AND VISUALIZATION OF MULTIDIMENSIONAL DATABASES
**Topics**: OLAP cubes, query optimization, multidimensional visualization
**Key Contributions**: LOD for database visualization
**Use For**: Query-aware LOD, data exploration

---

## Supplementary Modern Research (2024-2025)

### LODGE: Level-of-Detail Large-Scale Gaussian Splatting (2025)
**Source**: arXiv:2505.23158
**Topics**: 3D Gaussian splatting, real-time rendering, large-scale scenes
**Key Contributions**: Modern neural rendering with LOD
**Use For**: Cutting-edge LOD techniques

### Fast Rendering of Parametric Objects on Modern GPUs (2024)
**Authors**: J. Unterguggenberger et al.
**Topics**: GPU tessellation, parametric surfaces, adaptive detail
**Key Contributions**: Hardware tessellation pipeline
**Use For**: Modern GPU LOD implementation

### SimLOD: Simultaneous LOD Generation and Rendering for Point Clouds (2024)
**Topics**: Point cloud rendering, octree LOD, incremental generation
**Key Contributions**: Real-time LOD construction
**Use For**: Point cloud applications

---

## Key Papers by Topic

### Must-Read for BSP Understanding
1. Binary space partitioning (Wikipedia)
2. Valve Developer Wiki: BSP
3. "How Doom Used BSP Trees"

### Must-Read for LOD Fundamentals
1. Progressive Buffers (view-dependent LOD)
2. Integrating Occlusion Culling with Parallel LOD
3. Fast, Realistic Terrain Synthesis

### Must-Read for Perceptual Rendering
1. Attention and Visual Memory in Visualization
2. Gaze-aware Displays and Interaction
3. Managing LOD through Head-Tracked Peripheral Degradation

### Must-Read for Modern GPU Techniques
1. Fast Rendering of Parametric Objects on Modern GPUs (2024)
2. SimLOD (2024)
3. LODGE (2025)

---

## Citation Format

When referencing papers in oracle documentation:

```
**Source**: [Paper Title] (Author, Year)
**Location**: RESEARCH/Video Game LOD and BTree/NotebookLM_Sources/[filename].md
**Key Finding**: [Specific contribution]
```

---

**Last Updated**: 2025-10-28
**Total Papers**: 20 core + 3 supplementary modern research
