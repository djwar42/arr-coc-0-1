# lod-btree-oracle: Master Index

**Complete documentation index for LOD and BSP/BTree systems**

## 📖 Quick Navigation

- [SKILL.md](SKILL.md) - Oracle entry point and usage guide
- [concepts/](#concepts-8-files) - Core principles (8 files) **+1 Texture Viz**
- [techniques/](#techniques-20-files) - Rendering methods (20 files) **+5 Texture Viz**
- [algorithms/](#algorithms-7-files) - Implementation strategies (7 files)
- [applications/](#applications-9-files) - Real-world use cases (9 files) **+4 Texture Viz**
- [integration/](#integration-6-files) - System combinations (6 files) **+2 Phase 3**
- [optimization/](#optimization-2-files) - Performance optimization (2 files) **NEW - Phase 3**
- [theory/](#theory-1-file) - Theoretical foundations (1 file) **NEW - Dialogues 20-21**
- [references/](#references-2-files) - Quick lookups (2 files)

**Total: ~56 documentation files (+15 from 2025-01-30, +3 from Session 3, +1 from Dialogues 20-21, +2 from Phase 3, +10 from Texture Visualization 2025-10-31)**

---

## Concepts (8 files)

Core principles and foundational ideas about LOD and BSP systems:

### 00-lod-fundamentals.md
**What**: Level of Detail fundamentals
**Topics**: Discrete vs continuous LOD, view-dependent rendering, performance optimization
**Sources**: Progressive Buffers, Fast Terrain Synthesis, LOD Map papers

### 01-bsp-btree-basics.md
**What**: Binary Space Partitioning and BTree structures
**Topics**: BSP tree construction, spatial hierarchies, subdivision strategies
**Sources**: Binary Space Partitioning (Wikipedia), Athena3D

### 02-visual-perception.md
**What**: Attention and visual memory in computer graphics
**Topics**: Preattentive processing, visual salience, change blindness, inattentional blindness
**Sources**: Attention and Visual Memory in Visualization, Focus Guided Light Field Saliency

### 03-transjective-relevance.md
**What**: Perceptual vs objective LOD selection
**Topics**: Gaze-content coupling, user-aware rendering, transjective optimization
**Sources**: Gaze-aware Displays, Managing LOD through Head-Tracked Peripheral Degradation

### 04-perceptual-masking.md
**What**: Contrast and spatial masking for rendering optimization
**Topics**: Perceptual rendering pipeline, masking effects, quality metrics
**Sources**: Interactive Perceptual Rendering Pipeline using Contrast and Spatial Masking

### 05-focus-schematization.md
**What**: Partial schematization techniques
**Topics**: Focus maps, accentuation strategies, visual abstraction
**Sources**: Accentuating focus maps via partial schematization

### 07-multimodal-token-theory-2025-01-30.md (NEW - Session 3)
**What**: VAR and Laplacian pyramid theory for VLM token allocation
**Topics**: Visual Autoregressive models, discrete latent diffusion, frequency partitioning, coarse-to-fine generation
**Sources**: arXiv 2510.02826 (Hong & Belkadi Oct 2025), VAR original paper, VQ-VAE
**Length**: 440 lines
**Cross-references**: Image pyramids, progressive compression VLMs (PVC, FastVLM)

### 00-vervaekean-texture-visualization-2025-10-31.md (NEW - Texture Viz Expansion)
**What**: Vervaeke's relevance realization framework applied to texture visualization
**Topics**: Four ways of knowing (propositional, perspectival, participatory, procedural), multi-perspective displays, opponent processing in visualization design, ARR-COC philosophical grounding
**Sources**: ARR-COC Platonic Dialogues (Parts 8, 29, 46), Vervaeke's epistemology, game engine material inspectors
**Length**: 845 lines
**Cross-references**: ARR-COC 13-channel textures, Unity/Unreal material editors, interactive visualization

---

## Techniques (28 files)

Specific rendering and optimization methods:

### 00-foveated-rendering.md
**What**: Gaze-aware rendering strategies (overview)
**Topics**: Eye tracking, foveated displays, resolution gradation, VR optimization
**Sources**: Gaze-aware Displays and Interaction, VR Annual Symposium

### 00-foveated-rendering-01-logpolar-mapping-2025-01-30.md
**What**: Log-polar transforms for foveated vision (Level 1 deep-dive)
**Topics**: Mathematical foundations, VLM applications, cortical magnification mapping
**Sources**: 2024-2025 research on retinotopic rendering, Schwartz complex log model

### 00-foveated-rendering-02-biological-foundations-2025-01-30.md
**What**: Biological vision foundations (Level 2 deep-dive)
**Topics**: Retinal sampling (150K cones/mm²), V1 cortical magnification, retinotopic mapping
**Sources**: Nature 2025, computational neuroscience, All-TNNs emergent properties

### 00-foveated-rendering-03-vlm-token-allocation-2025-01-30.md
**What**: VLM token allocation strategies (overview)
**Topics**: Homunculus Protocol, Chain-of-Focus, query-aware compression, tiered budgets
**Sources**: arXiv May 2025, V* benchmark, 2024-2025 VLM research

### 00-foveated-rendering-03-01-token-merging-pruning-2025-01-30.md
**What**: Token merging and pruning (Level 1 deep-dive)
**Topics**: AIM hybrid approach (72% reduction), SparseVLM (75% training-free), HiRED (85% 3-stage)
**Sources**: arXiv 2412.03248, arXiv 2410.04417, AAAI 2025

### 00-foveated-rendering-03-02-progressive-compression-2025-01-30.md
**What**: Progressive multi-scale compression (Level 2 deep-dive)
**Topics**: PVC (99% video reduction), FastVLM (2.7× speedup), Token Compensator
**Sources**: CVPR 2025, Apple ML Research, difficulty-aware budgeting

### 00-foveated-rendering-03-03-dynamic-reduction-2025-01-30.md
**What**: Dynamic token reduction during generation (Level 3 deep-dive)
**Topics**: DyRate (runtime adaptation), attention-guided dropping, layer-adaptive compression
**Sources**: arXiv 2501.14204, HiRED (AAAI 2025), generation-aware pruning

### 00-foveated-rendering-03-04-training-free-methods-2025-01-30.md
**What**: Training-free token reduction (Level 4 deep-dive)
**Topics**: SparseVLM (114 cit), VScan (2-stage), DToMA (IJCAI 2025), PruneVid (ACL 2025, 92.6% reduction)
**Sources**: Plug-and-play deployment, zero-shot compression, video optimization

### 00-foveated-rendering-03-05-advanced-compression-2025-01-30.md
**What**: Advanced VLM token compression methods (Level 5 deep-dive)
**Topics**: VIST slow-fast bio-inspired (2.3× compression), PuMer text-informed pruning (53 cit), ACT-IN-LLM adaptive policy (+6.2%), LookupViT bidirectional cross-attention (ECCV 2024)
**Sources**: arXiv 2502.00791, arXiv 2305.17530, OpenReview 2024, arXiv 2407.12753, COLING 2025 survey

### 01-peripheral-degradation.md
**What**: Head-tracked LOD management
**Topics**: Peripheral vision degradation, head-tracking, design principles
**Sources**: Managing Level of Detail through Head-Tracked Peripheral Degradation

### 02-occlusion-culling.md
**What**: Integrating culling with parallel LOD
**Topics**: GPU occlusion queries, visibility determination, parallel processing
**Sources**: Integrating Occlusion Culling with Parallel LOD for Rendering Complex 3D Environments on GPU

### 03-progressive-buffers.md
**What**: View-dependent geometry and texture LOD rendering
**Topics**: Progressive meshes, texture pyramids, smooth transitions
**Sources**: Progressive Buffers: View-dependent Geometry and Texture LOD Rendering

### 04-focus-maps.md
**What**: Accentuating focus via partial schematization
**Topics**: Focus+context visualization, selective detail, abstraction levels
**Sources**: Accentuating focus maps via partial schematization

### 05-vortex-spiral-sampling-2025-01-30.md
**What**: Vortex and spiral sampling patterns
**Topics**: Golden spiral, Fibonacci patterns, multi-vortex systems, phyllotaxis
**Sources**: 2024-2025 research on spiral sampling, bio-inspired vision

### 06-game-engine-lod-systems-2025-01-30.md (EXPANDED - Session 3)
**What**: Game engine LOD techniques (Nanite, texture streaming)
**Topics**: Unreal Nanite virtualized geometry, cluster DAGs, texture mipmaps, Hi-Z occlusion culling, UE5.6 Nanite Foliage, Fast Geometry Streaming Plugin
**Sources**: arXiv 2507.08142 (UE5 review), Tom Looman UE5.6 Performance Highlights (July 2025), Epic State of Unreal 2025, The Witcher 4 Tech Demo
**Length**: 484 lines (+217 from Session 3)
**New content**: UE5.6 specific optimizations (60FPS PS5 w/ RT), Nanite foliage chunk culling, Fast Geometry Streaming (Epic+CDPR collab), Renderer parallelization, GPU Profiler 2.0

### 09-cluster-based-cascade-filtering-2025-01-30.md
**What**: Semantic cluster-based cascade for VLM patch sampling ("clust frust" technique)
**Topics**: SAM segmentation (Kirillov et al. 2023, 1B masks), cluster metadata channels 12-14, two-stage cascade (50 clusters → 500 patches), 7.4× patch reduction, query-aware semantic filtering, 64× combined speedup with embeddings
**Sources**: Platonic Dialogue 27 (lines 157-275), Kirillov et al. (2023, Meta AI, arXiv:2304.02643 - SAM), SLIC superpixels, frustum culling analogy
**Length**: 905 lines
**Cross-references**: CLIP embeddings (query-aware sampling), metadata texture arrays (channels 12-14), GPU texture primitives (mipmap scanning)

### 09-cluster-based-cascade-filtering-appendix-frustum-2025-01-30.md
**What**: Frustum culling analogy for cluster filtering ("clust frust" name origin)
**Topics**: View frustum culling (graphics pipeline), cluster frustum (semantic culling), spatial culling techniques, BVH/octree filtering, unified culling framework
**Sources**: Graphics literature on frustum culling, spatial data structures, GPU occlusion culling
**Length**: 283 lines
**Cross-references**: Cluster-based cascade filtering (main technique), occlusion culling integration, BSP construction

---

## Algorithms (6 files)

Implementation strategies and computational methods:

### 00-bsp-construction.md
**What**: Building BSP trees
**Topics**: Polygon splitting, convexity, tree balancing, construction heuristics
**Sources**: Binary space partitioning (Wikipedia)

### 01-lod-selection.md
**What**: Dynamic LOD level calculation
**Topics**: Distance metrics, screen-space error, hysteresis, popping prevention
**Sources**: Progressive Buffers, Integrating Occlusion Culling with Parallel LOD

### 02-terrain-synthesis.md
**What**: Fast, realistic terrain generation
**Topics**: Heightfield synthesis, procedural generation, noise functions, erosion
**Sources**: Fast, Realistic Terrain Synthesis

### 03-heightfield-tessellation.md
**What**: Watertight incremental tessellation
**Topics**: Real-time tessellation, crack prevention, GPU implementation
**Sources**: Watertight Incremental Heightfield Tessellation

### 04-personalized-lod.md
**What**: User-preference based LOD strategies
**Topics**: User modeling, personalized rendering, adaptive quality
**Sources**: Exploiting LOD-Based Similarity Personalization Strategies, Analysis and Design of Personalized Recommendation System

### 05-adaptive-subdivision-2025-01-30.md
**What**: Adaptive spatial subdivision algorithms
**Topics**: BSP trees, quadtrees, octrees, K-d trees, RoPE 2D encoding for vision transformers
**Sources**: 2024-2025 spatial partitioning research, vision transformer position encoding

### 06-image-pyramid-multiscale-2025-01-30.md (EXPANDED - Session 3)
**What**: Image pyramid multi-scale processing (Gaussian, Laplacian, Steerable)
**Topics**: Gaussian/Laplacian pyramids, **steerable pyramids** (Simoncelli & Freeman 1995), orientation-selective decomposition, polar-separable filters, LapCAT (Nature 2025), pyramid-based loss functions, VLM training with LapLoss
**Sources**: arXiv 2503.05974 (LapLoss 2025), Simoncelli & Freeman 1995 (1697 cit), Medium tutorial (Isaac Berrios Jan 2024), Nature s41598-025-94464-6 (LapCAT), CVPR 2025 (PVC, FastVLM)
**Length**: 837 lines (+336 from Session 3)
**New content**: Steerable pyramids deep-dive (mathematical formulation, decomposition algorithm, orientation-aware token allocation), LapCAT architecture (encoder-decoder with pyramid loss), anisotropic token budgets for VLMs

### 00-unity-material-inspector-2025-10-31.md (NEW - Texture Viz Expansion)
**What**: Unity material inspector and texture debugging
**Topics**: Channel-packed texture display (URP MASK maps), material ball preview (3D sphere rotation), mipmap visualization, normal map false color (XYZ → RGB), Rendering Debugger channel inspection
**Sources**: Unity official documentation, community tutorials, Shader Graph workflows
**Length**: 320 lines
**Cross-references**: ARR-COC 13-channel packing (4 RGBA textures), false color composites, LOD token budgets

### 00-unreal-material-editor-2025-10-31.md (NEW - Texture Viz Expansion)
**What**: Unreal Engine 5 material editor visualization
**Topics**: Node-based texture flow, live preview viewport, Texture Graph (UE5.4+), shader complexity view, channel extraction nodes
**Sources**: Epic Games official docs, ArtStation tutorials, Unreal Engine forums
**Length**: 664 lines
**Cross-references**: ARR-COC relevance pipeline visualization, node-based knowing → balancing → attending → realizing

### 00-webgl-shader-debugging-2025-10-31.md (NEW - Texture Viz Expansion)
**What**: WebGL shader debugging tools and workflows
**Topics**: SpectorJS (frame capture, multi-channel texture inspection), Chrome DevTools WebGL profiling, printf-style shader debugging (output to pixels)
**Sources**: Real-Time Rendering blog, BabylonJS Spector.js repo, MDN WebGL documentation
**Length**: 612 lines
**Cross-references**: Three.js/Babylon.js debugging, ARR-COC 13-channel WebGL verification

### 00-texture-atlas-visualization-2025-10-31.md (NEW - Texture Viz Expansion)
**What**: Texture atlas visualization and sprite sheet inspection
**Topics**: Grid overlay for atlas regions, interactive viewers (TexturePacker, Leshy), click-to-inspect patterns, UV coordinate mapping
**Sources**: TexturePacker documentation, Unity Sprite Atlas inspector, game dev tools
**Length**: 310 lines
**Cross-references**: ARR-COC 32×32 patch grid as texture atlas, interactive patch inspector UI

### 00-interactive-channel-compositing-2025-10-31.md (NEW - Texture Viz Expansion)
**What**: Interactive UI patterns for channel selection and compositing
**Topics**: Channel selector interfaces (Photoshop Channels Panel, Substance 2D Viewport, Blender Compositor), real-time preview, split-view comparisons, WebGL GPU-accelerated rendering
**Sources**: Adobe Photoshop/Substance Designer/Blender documentation, web-based image editors
**Length**: 1,072 lines
**Cross-references**: ARR-COC 13-channel compositor design, false color mode dropdown, RGB mapping controls

### 00-advanced-shader-2025-10-31.md (NEW - Advanced Texture Expansion)
**What**: Advanced WebGL shader programming for multi-channel compositing
**Topics**: Multi-pass rendering pipelines, compute shaders for texture processing, deferred rendering, GPU-accelerated channel operations, ARR-COC 13-channel parallel processing
**Sources**: WebGL 2.0 specification, Real-Time Rendering tutorials, GPU Gems, Khronos documentation
**Length**: 439 lines
**Cross-references**: ARR-COC texture compositor GPU acceleration, WebGL2 advanced features, deferred rendering for multi-channel visualization

### 00-webgl2-advanced-2025-10-31.md (NEW - Advanced Texture Expansion)
**What**: WebGL 2.0 advanced features for multi-channel rendering
**Topics**: Multiple Render Targets (MRT), Uniform Buffer Objects (UBO), instanced rendering, 3D textures, volume rendering, ARR-COC parallel 13-channel processing
**Sources**: WebGL 2.0 API docs, MDN tutorials, graphics programming guides
**Length**: 421 lines
**Cross-references**: ARR-COC multi-channel rendering optimizations, advanced shader programming, GPU streaming

### 00-gpu-debugging-2025-10-31.md (NEW - Advanced Texture Expansion)
**What**: GPU debugging tools for WebGL and texture pipelines
**Topics**: RenderDoc WebGL capture, NVIDIA NSight Graphics, SpectorJS integration, GPU profiling, shader debugging workflows, ARR-COC pipeline profiling
**Sources**: RenderDoc documentation, NVIDIA developer docs, graphics debugging best practices
**Length**: 342 lines
**Cross-references**: WebGL shader debugging, Three.js/Babylon.js profiling, ARR-COC performance analysis

### 00-texture-sampling-filtering-2025-10-31.md (NEW - 3D Rendering Expansion)
**What**: Texture sampling and filtering techniques for 3D rendering
**Topics**: Mipmapping (automatic LOD selection), anisotropic filtering, texture coordinates (UV wrapping, clamping), bilinear/trilinear filtering, texture atlas sampling, ARR-COC multi-channel sampling patterns
**Sources**: WebGPU Unleashed (Shi Yan), Bart Wronski frequency analysis, GPU Gems 2, Mitchell & Netravali filtering
**Length**: 566 lines
**Cross-references**: ARR-COC 32×32 grid sampling, relevance-aware filtering, texture atlas visualization

### 00-procedural-texture-generation-2025-10-31.md (NEW - 3D Rendering Expansion)
**What**: Procedural texture generation with GPU shaders
**Topics**: Fragment shader generation, noise functions (Perlin, Simplex, Worley/cellular), GPU optimization (texture-based permutation), fBm (Fractional Brownian Motion), domain warping, ARR-COC query-aware texture synthesis
**Sources**: The Book of Shaders (Patricio Gonzalez Vivo), GPU Gems 2 Chapter 26 (Simon Green), Ken Perlin papers, Steven Worley Cellular Texture Basis Function
**Length**: 886 lines
**Cross-references**: ARR-COC dynamic texture generation, relevance-driven detail scaling, procedural pattern synthesis

### 00-texture-mapping-techniques-2025-10-31.md (NEW - 3D Rendering Expansion)
**What**: Advanced texture mapping techniques for 3D geometry
**Topics**: UV mapping (unwrapping, transformations), triplanar mapping (world-space projection), projective textures (decals, shadows), cube mapping (environment maps), spherical mapping, ARR-COC 32×32 grid UV strategies
**Sources**: Catlike Coding tutorials, Ben Golus triplanar normals, Adobe Substance 3D Painter, Real-Time Rendering Resources
**Length**: 564 lines
**Cross-references**: ARR-COC variable LOD patches (64-400 tokens), query-aware texture sampling, adaptive UV mapping

### 00-shader-pipeline-fundamentals-2025-10-31.md (NEW - Shader Programming Expansion)
**What**: Shader pipeline stages and GLSL fundamentals
**Topics**: Vertex/tessellation/geometry/fragment/compute shaders, data flow (attributes, varyings, uniforms), shader compilation and linking, built-in variables, ARR-COC multi-pass 13-channel processing
**Sources**: LearnOpenGL, WebGL Fundamentals, Stanford CS248A, AMD GPUOpen, Reddit r/GraphicsProgramming
**Length**: 1,054 lines
**Cross-references**: ARR-COC shader architecture, advanced shader techniques, multi-pass rendering

### 00-advanced-shader-techniques-2025-10-31.md (NEW - Shader Programming Expansion)
**What**: Advanced shader programming techniques
**Topics**: Vertex displacement (heightmap, procedural), parallax mapping (steep parallax, offset limiting), screen-space effects (SSAO, SSR, bloom), geometry shaders, tessellation (adaptive, edge-based), ARR-COC relevance-based detail
**Sources**: LearnOpenGL Advanced Lighting, Catlike Coding, Inigo Quilez techniques, OpenGL SuperBible
**Length**: 882 lines
**Cross-references**: ARR-COC advanced rendering, shader pipeline fundamentals, procedural texture generation

---

## Applications (12 files)

Real-world use cases and domain-specific implementations:

### 00-video-games.md
**What**: Visual attention in 3D video games
**Topics**: Game rendering, player attention, performance optimization, immersion
**Sources**: Visual Attention in 3D Video Games

### 01-vr-ar.md
**What**: Virtual and augmented reality LOD
**Topics**: VR rendering, AR occlusion, stereoscopic LOD, latency optimization
**Sources**: Virtual Reality Annual International Symposium, Gaze-aware Displays

### 02-terrain-visualization.md
**What**: Slippy maps and terrain systems
**Topics**: Tile-based rendering, streaming, geographic LOD, web mapping
**Sources**: Opportunities with Slippy Maps for Terrain Visualization in VR/AR, Fast Realistic Terrain Synthesis

### 03-multiresolution-viz.md
**What**: Navigating multiresolution volume data
**Topics**: Volume rendering, LOD navigation, visual interfaces, data exploration
**Sources**: LOD Map - A Visual Interface for Navigating Multiresolution Volume Visualization

### 04-urban-semantics.md
**What**: 3D urban visualization with LOD
**Topics**: City modeling, semantic LOD, perceptive effects, urban planning
**Sources**: Applying level-of-detail and perceptive effects to 3D urban semantics visualization

### 00-threejs-texture-display-2025-10-31.md (NEW - Texture Viz Expansion)
**What**: Three.js texture visualization and WebGL display
**Topics**: DataTexture for multi-channel data, ShaderMaterial for custom compositing, interactive 3D texture viewer (OrbitControls), false color GLSL shaders, heatmap rendering
**Sources**: Three.js official documentation, WebGL tutorials, GitHub examples
**Length**: 465 lines
**Cross-references**: ARR-COC 13-channel 3D viewer, GPU-accelerated channel compositing, interactive patch inspection

### 00-gradio-3d-integration-2025-10-31.md (NEW - Texture Viz Expansion)
**What**: Gradio integration with 3D visualizations
**Topics**: Custom HTML components (gr.HTML with Three.js in head), data transfer patterns (NumPy → JavaScript via base64), bidirectional events, Gradio custom components API
**Sources**: Gradio official guides, community examples, Gradio-Lite
**Length**: 669 lines
**Cross-references**: ARR-COC Gradio + Three.js blueprint, Python backend ↔ JavaScript frontend, Phase 2 interactive viewer

### 00-babylonjs-texture-tools-2025-10-31.md (NEW - Texture Viz Expansion)
**What**: Babylon.js texture inspector and debugging tools
**Topics**: Built-in Inspector tool (scene.debugLayer.show()), channel isolation (toggle RGBA), Material Playground live editing, Spector.js WebGL frame capture
**Sources**: Babylon.js official documentation, Medium tutorials, community comparisons
**Length**: 482 lines
**Cross-references**: Three.js vs Babylon.js decision matrix, ARR-COC development debugging (Babylon) vs production deployment (Three.js)

### 00-arr-coc-texture-viewer-implementation-2025-10-31.md (NEW - Texture Viz Expansion)
**What**: ARR-COC texture viewer implementation roadmap
**Topics**: Phase 1 (Enhanced Gradio Microscope - 1-2 days), Phase 2 (Three.js Interactive 3D Viewer - 3-5 days), Phase 3 (Advanced Debugging Tools - 5-7 days), technology stack recommendations, complete code examples
**Sources**: Synthesis of all texture visualization research (Unity/Unreal/Three.js/Gradio/WebGL/Vervaeke)
**Length**: 2,279 lines
**Cross-references**: ARR-COC existing microscope code, Vervaekean four ways of knowing, practical implementation checklist

### 00-scientific-viz-patterns-2025-10-31.md (NEW - Advanced Texture Expansion)
**What**: Scientific visualization patterns for multi-component field data
**Topics**: ParaView multi-component display, VTK pipeline architecture, VisIt tensor fields, multi-variate data visualization, ARR-COC as 13-component perceptual field
**Sources**: ParaView documentation, VTK user guides, scientific visualization textbooks
**Length**: 404 lines
**Cross-references**: ARR-COC 13-channel perceptual data, multi-variate field visualization, advanced shader compositing

### 00-foveation-interactive-2025-10-31.md (NEW - Advanced Texture Expansion)
**What**: Computational foveation for interactive visualization
**Topics**: Foveation principles, gaze-contingent rendering, eccentricity-based LOD, semantic gaze (query-aware), ARR-COC query as visual attention
**Sources**: Foveated rendering research, gaze-contingent displays, perceptual rendering
**Length**: 329 lines
**Cross-references**: ARR-COC query-aware relevance, biological foveation foundations, log-polar mapping

### 00-vlm-viz-sota-2025-10-31.md (NEW - Advanced Texture Expansion)
**What**: Vision-Language Model visualization state-of-the-art (2024-2025)
**Topics**: Mechanistic interpretability (Probing, Activation Patching, Logit Lens, SAEs, Automated Explanation), VLM-Lens toolkit, attention heatmaps, token attribution, ARR-COC comparison framework
**Sources**: arXiv 2024-2025 VLM interpretability papers, VLM-Lens, SAVIS, A-VL
**Length**: 340 lines
**Cross-references**: ARR-COC relevance visualization, VLM attention patterns, mechanistic interpretability methods

---

## Integration (7 files)

Combining LOD with other systems:

### 00-lod-with-culling.md
**What**: Parallel LOD and occlusion culling
**Topics**: GPU pipeline integration, visibility + detail optimization, performance
**Sources**: Integrating Occlusion Culling with Parallel LOD

### 01-gaze-tracking.md
**What**: Eye tracking for LOD optimization
**Topics**: Foveated rendering, gaze prediction, latency compensation
**Sources**: Gaze-aware Displays and Interaction, Managing LOD through Head-Tracked Peripheral Degradation

### 02-multidimensional-queries.md
**What**: Query-aware LOD in databases
**Topics**: OLAP cubes, query optimization, visualization of multidimensional data
**Sources**: QUERY, ANALYSIS, AND VISUALIZATION OF MULTIDIMENSIONAL DATABASES

### 05-implementation-resources-2025-01-30.md
**What**: VLM token compression implementation resources
**Topics**: AIM (ICCV 2025, 7× FLOPs reduction), GitHub awesome lists, production deployment guides, integration patterns, benchmark comparison
**Sources**: LaVi-Lab/AIM repository, Awesome-Token-Compress, Awesome-Token-Merge-for-MLLMs

### 07-metadata-texture-arrays-2025-01-30.md
**What**: Metadata storage in GPU texture arrays for VLM acceleration
**Topics**: 40-channel architecture (position, clusters, CLIP embeddings, temporal cache), spatial locality (5× fewer cache misses), 280× video speedup, co-located data structures
**Sources**: Platonic Dialogue 27, GPU texture array specs (OpenGL/CUDA 2024, validated via Bright Data), PCA compression for embeddings
**Length**: 1,094 lines
**Cross-references**: Multi-channel perceptual filters, temporal coherence, spatial locality optimization, CLIP embeddings in textures, cluster filtering

### 07-clip-embeddings-in-textures-2025-01-30.md
**What**: PCA-compressed CLIP embeddings stored in texture arrays for hardware-accelerated query relevance
**Topics**: 768D → 16D PCA compression (48× reduction), texture channels 18-33, dense CLIP features, multi-query amortization (110× speedup), embedding warping for video, training methodology
**Sources**: Platonic Dialogue 27 (lines 372-495), May et al. (2019, NIH, 29 cit - PCA for embeddings), Zhang et al. (2024, ACL, 9 cit - 768D→300D), PCA-RAG (arXiv 2025), Milvus documentation
**Length**: 971 lines
**Cross-references**: Metadata texture arrays (channels 18-33), cluster filtering (query-aware sampling), spatial locality optimization

### 08-temporal-coherence-video-vlm-2025-01-30.md (Phase 3) **NEW**
**What**: Temporal coherence for video VLMs using optical flow and motion compensation
**Topics**: WAFT optical flow (8× faster than RAFT), backward warping (0.05ms), validity masking, StreamingVLM architecture, 280× video speedup, event-based cameras, neuromorphic SNNs (10,000× power efficiency)
**Sources**: WAFT (arXiv 2506.21526), StreamFlow (NeurIPS 2024, 44% speedup), StreamingVLM (arXiv 2510.09608), Intel Loihi neuromorphic (100× energy reduction)
**Length**: 1,014 lines
**Cross-references**: Metadata texture arrays (layers 15-18 temporal storage), multi-channel motion detection, biological vision

---

## Optimization (4 files)

Performance optimization techniques:

### 00-gpu-streaming-2025-10-31.md (NEW - Advanced Texture Expansion)
**What**: GPU texture streaming and virtual texturing
**Topics**: Virtual texturing (sparse texture residency), BC6H/BC7 compression for multi-channel, streaming optimization, ARR-COC 1024-patch streaming strategies
**Sources**: GPU architecture documentation, virtual texturing papers, texture compression specs
**Length**: 378 lines
**Cross-references**: ARR-COC patch streaming, spatial locality cache, advanced shader multi-pass

### 00-shader-optimization-2025-10-31.md (NEW - Shader Programming Expansion)
**What**: Shader optimization and performance profiling
**Topics**: Instruction count optimization (ALU vs memory), branching costs (dynamic vs compile-time), precision qualifiers (lowp/mediump/highp), mobile optimization (bandwidth, fillrate), texture sampling optimization, ARR-COC 13-channel efficient processing
**Sources**: ARM Mali GPU optimization guides, PowerVR optimization, Adreno best practices, WebGL performance tips
**Length**: 957 lines
**Cross-references**: ARR-COC shader performance, shader pipeline fundamentals, GPU profiling tools

### 01-spatial-locality-cache-2025-01-30.md (Phase 2) ✅ **COMPLETED**
**What**: GPU cache optimization via spatial locality in memory layout
**Topics**: L1/L2/texture cache hierarchy, co-located vs scattered data (5× cache miss reduction), texture memory path, batch processing patterns, memory bandwidth utilization (45% → 78%)
**Sources**: NVIDIA CUDA documentation (validated via Bright Data), GPU architecture specs, cache performance analysis, texture memory optimization
**Length**: 1,105 lines (Phase 2 - DELIVERED)
**Cross-references**: Metadata texture arrays (co-located storage), temporal coherence (cache-friendly temporal data)

### 02-query-aware-adaptive-channels-2025-01-30.md (Phase 3) **NEW**
**What**: Query-aware dynamic channel selection for multi-channel perceptual processing
**Topics**: Query classification (BERT-based, 88% accuracy), adaptive channel routing (threshold/top-K/budget-based), 1.91× speedup, confidence-based expansion, learned vs hand-crafted filters, RL for channel selection
**Sources**: DeBiFormer (arXiv 2410.08582), BilevelPruning (CVPR 2024, 18 cit), Adaptive Sparse Transformer (CVPR 2024, 100 cit), FlexPrefill (32 cit)
**Length**: 1,021 lines
**Cross-references**: Multi-channel perceptual filters (9-channel architecture), temporal coherence (query-type routing for video), metadata texture arrays (channel storage)

---

## Theory (1 file)

Theoretical foundations and biological grounding:

###03-biological-grounding-vlms.md
**What**: Biological grounding for VLM token allocation with cortical magnification M(e)
**Topics**: Cortical magnification M(e) = M₀/(e+e₀), query-driven fixation, foveal-peripheral trade-off, Vervaeke's relevance realization for vision, human-VLM alignment validation, biological vs engineering approaches
**Sources**: Platonic Dialogues 20-21, Daniel & Whitteridge 1961, Foveated Retinotopy (arXiv 2402.15480)
**Key Contribution**: First biologically-grounded VLM token allocation with explicit neuroscience formulas, extends Foveated Retinotopy to VLMs with query-awareness

---

## References (2 files)

Quick lookups and practical guides:

### 00-glossary.md
**What**: Key terms and definitions
**Topics**: LOD, BSP, foveation, culling, tessellation, perceptual rendering

### 01-paper-index.md
**What**: Source research papers
**Topics**: Complete bibliography with abstracts and key contributions

---

## Topic Cross-Reference

### By Research Area

**Perceptual Rendering:**
- concepts/02-visual-perception.md
- concepts/03-transjective-relevance.md
- concepts/04-perceptual-masking.md
- techniques/00-foveated-rendering.md
- techniques/01-peripheral-degradation.md

**Spatial Data Structures:**
- concepts/01-bsp-btree-basics.md
- algorithms/00-bsp-construction.md

**Terrain & Geometry:**
- algorithms/02-terrain-synthesis.md
- algorithms/03-heightfield-tessellation.md
- applications/02-terrain-visualization.md

**User-Aware Systems:**
- concepts/03-transjective-relevance.md
- algorithms/04-personalized-lod.md
- integration/01-gaze-tracking.md

**Real-Time Graphics:**
- techniques/02-occlusion-culling.md
- techniques/03-progressive-buffers.md
- integration/00-lod-with-culling.md

### By Application Domain

**Video Games:**
- applications/00-video-games.md
- techniques/02-occlusion-culling.md
- algorithms/01-lod-selection.md

**VR/AR:**
- applications/01-vr-ar.md
- techniques/00-foveated-rendering.md
- integration/01-gaze-tracking.md

**Visualization:**
- applications/03-multiresolution-viz.md
- applications/04-urban-semantics.md
- integration/02-multidimensional-queries.md

**VLM Research Landscape (2024-2025):**
- theory/03-biological-grounding-vlms.md (biological foundations)
- techniques/00-foveated-rendering.md (VLM applications section)
- algorithms/06-image-pyramid-multiscale-2025-01-30.md (VLM pyramid methods)
- **Key papers**: PyramidDrop (ICLR 2025, 90 cit), DPN-LLaVA (March 2025), FastVLM (Apple, July 2025), Foveated Retinotopy (Oct 2025)
- **Our differentiation**: Biological cortical magnification + query-driven fixation + Vervaeke's relevance realization

---

## Source Materials

All documentation derived from 20+ research papers in:
`RESEARCH/Video Game LOD and BTree/NotebookLM_Sources/`

Plus supplementary web research via Bright Data for:
- Modern GPU techniques
- Real-time rendering advances
- VR/AR best practices
- Game engine implementations

---

## How to Use This Index

1. **Browse by category** - Navigate to concepts/techniques/algorithms/applications
2. **Search by topic** - Use cross-reference sections
3. **Follow connections** - Each file links to related docs
4. **Start broad** - Begin with concepts/, drill down to specifics

---

**Last Updated**: 2025-10-28
**Files**: 26 documentation files + INDEX.md + SKILL.md
