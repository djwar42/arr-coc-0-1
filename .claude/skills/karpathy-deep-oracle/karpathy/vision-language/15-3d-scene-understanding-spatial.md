# 3D Scene Understanding and Spatial Reasoning in VLMs

## Overview

3D scene understanding represents one of the most challenging frontiers for vision-language models, requiring systems to not only recognize objects but also comprehend their spatial relationships, geometric properties, and physical configurations in three-dimensional space. While VLMs have achieved remarkable success in 2D image understanding, extending these capabilities to 3D spatial reasoning remains an active area of research with critical implications for embodied AI, robotics, AR/VR, and autonomous navigation.

The challenge lies in bridging the gap between 2D visual inputs (images, videos) and the 3D spatial knowledge required for real-world interaction. Recent research has explored multiple approaches: leveraging 3D representations like point clouds and scene graphs, training on large-scale spatial reasoning datasets, and developing architectures that explicitly model geometric relationships.

## 3D Representations for VLMs

### Point Cloud Processing

Point clouds provide an explicit 3D geometric representation that preserves spatial structure while remaining computationally tractable. Recent VLMs have integrated point cloud encoders to process 3D data directly:

**PointLLM** (ECCV 2024) pioneered empowering large language models to understand point clouds through a specialized encoder-decoder architecture. The system uses a point cloud encoder to extract geometric features, which are then projected into the LLM's token space through learned adapters. This enables the model to reason about 3D object properties, spatial configurations, and scene layouts directly from point cloud data.

From [PointLLM: Empowering Large Language Models to Understand Point Clouds](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03601.pdf) (ECCV 2024):
- Point cloud tokenization through geometric feature extraction
- Cross-modal alignment between 3D geometric features and language embeddings
- Multi-task learning for 3D object recognition, spatial relationships, and scene understanding

**MiniGPT-3D** (ACM MM 2024) achieves efficient 3D-LLM training in just 27 hours on a single RTX 3090 GPU while reaching multiple state-of-the-art results. The key innovation is efficient point cloud feature extraction combined with lightweight adapters that align 3D representations with pre-trained language models without requiring massive computational resources.

From [Efficiently Aligning 3D Point Clouds with Large Language Models](https://dl.acm.org/doi/10.1145/3664647.3681257) (ACM MM 2024, accessed 2025-01-31):
- Efficient 3D feature extraction using sparse convolutions
- Lightweight cross-modal alignment with frozen LLM backbone
- Training efficiency through strategic data sampling and augmentation

### Volumetric and Multi-View Representations

**Avi: Action from Volumetric Inference** introduces a novel 3D Vision-Language-Action (VLA) architecture that reframes robotic action generation as 3D perception. The system constructs volumetric representations from multi-view images, enabling dense spatial reasoning for manipulation tasks.

From [Avi: A 3D Vision-Language Action Model Architecture](https://arxiv.org/abs/2510.21746) (arXiv 2025, accessed 2025-01-31):
- Volumetric scene representation from RGB-D or multi-view images
- 3D spatial reasoning through volumetric transformers
- Direct action prediction grounded in 3D geometric understanding

**3D-VLA** (ICML 2024) builds on a 3D-based large language model with action tokens for embodied environments. The system maintains a world model in 3D space, enabling persistent spatial reasoning across multiple time steps.

From [3D-VLA: A 3D Vision-Language-Action Generative World Model](https://proceedings.mlr.press/v235/zhen24a.html) (ICML 2024, accessed 2025-01-31):
- 3D world model representation for temporal consistency
- Action tokens for embodied interaction
- Generative modeling of 3D scene dynamics

### Scene Graphs and Structured Representations

Scene graphs provide a structured, interpretable representation of 3D scenes by explicitly encoding objects and their relationships. This representation bridges vision and language naturally through graph-structured knowledge.

**SceneGPT** (arXiv 2024) demonstrates that pre-trained LLMs can perform 3D spatial reasoning without explicit 3D supervision by leveraging 3D scene graphs. The system constructs scene graphs encoding objects and spatial relationships, then uses in-context learning to adapt LLMs for spatial reasoning tasks.

From [SceneGPT: A Language Model for 3D Scene Understanding](https://arxiv.org/abs/2408.06926) (arXiv 2024, accessed 2025-01-31):
- 3D scene graph construction from point clouds or RGB-D data
- Object-centric representation with explicit spatial relationships
- Zero-shot spatial reasoning through in-context learning
- Qualitative evaluation on object semantics, physical properties, and scene-level spatial understanding

**SpatialRGPT** (NeurIPS 2024) enhances VLM spatial reasoning by constructing 3D scene graphs and integrating relative depth information. The system generates spatial relationship descriptions from scene graphs, enabling more accurate geometric reasoning.

From [SpatialRGPT: Grounded Spatial Reasoning in Vision-Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/f38cb4cf9a5eaa92b3cfa481832719c6-Paper-Conference.pdf) (NeurIPS 2024, accessed 2025-01-31):
- 3D scene graph construction with spatial relationship annotations
- Depth plugin for relative distance estimation
- Conversion of 3D scene graphs into textual descriptions for VLM training
- Improved performance on spatial reasoning benchmarks through explicit geometric grounding

### Depth Maps and Multi-View Fusion

**SD-VLM** (arXiv 2024) introduces spatial measuring and understanding with depth-enhanced VLMs. The system bridges 3D scene graphs with 2D image annotations by projecting spatial information from 3D to 2D image planes using depth maps.

From [SD-VLM: Spatial Measuring and Understanding with Depth](https://arxiv.org/html/2509.17664v1) (arXiv 2024, accessed 2025-01-31):
- Depth-aware spatial annotation transfer from 3D to 2D
- Multi-view consistency through geometric constraints
- Object distance estimation and size comparison in metric space

**GPT4Scene** (arXiv 2025) explores a purely vision-based solution for 3D spatial understanding inspired by human perception. The system constructs Bird's Eye View (BEV) images from video sequences and establishes global-local correspondence through consistent object marking.

From [GPT4Scene: Understand 3D Scenes from Videos with Vision-Language Models](https://arxiv.org/abs/2501.01428) (arXiv 2025, accessed 2025-01-31):
- BEV image construction from video for global scene context
- Consistent object ID marking across frames and BEV
- Zero-shot improvement over GPT-4o on 3D understanding tasks
- Training on 165K video annotations achieves state-of-the-art on all 3D understanding benchmarks
- Emergent ability to understand 3D scenes even without explicit BEV prompting after training

## Spatial Reasoning Mechanisms

### Quantitative Spatial Understanding

VLMs traditionally struggle with quantitative spatial reasoning—estimating metric distances, comparing sizes, or understanding precise geometric relationships. Recent research addresses this limitation through specialized training.

**SpatialVLM** (CVPR 2024) develops an automatic 3D spatial VQA data generation framework that scales to 2 billion VQA examples on 10 million real-world images. By training VLMs on internet-scale spatial reasoning data in metric space, the system significantly enhances both qualitative and quantitative spatial reasoning.

From [SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities](https://arxiv.org/abs/2401.12168) (CVPR 2024, accessed 2025-01-31):
- Automatic 3D spatial VQA data generation from internet images
- Training on 2 billion spatial reasoning examples
- Metric space reasoning: distance estimation, size comparison, spatial relationships
- Chain-of-thought spatial reasoning capabilities
- Novel downstream applications in robotics through quantitative estimation

The system addresses VLMs' limited spatial reasoning by providing training data that explicitly annotates 3D spatial relationships in calibrated metric space, enabling models to learn quantitative geometric understanding.

### Qualitative Spatial Relations

Beyond metric measurements, VLMs must understand qualitative spatial relationships: "above," "behind," "next to," "inside," etc. These relationships are crucial for natural language interaction and scene understanding.

**Spatial Relationship Modeling** research (WACV 2024) demonstrates that explicitly modeling spatial relations improves vision-and-language reasoning. The approach constructs spatial relationship graphs that encode directional and topological relationships between objects.

From [Improving Vision-and-Language Reasoning via Spatial Relations Modeling](https://openaccess.thecvf.com/content/WACV2024/papers/Yang_Improving_Vision-and-Language_Reasoning_via_Spatial_Relations_Modeling_WACV_2024_paper.pdf) (WACV 2024):
- Explicit spatial relationship graphs for object pairs
- Directional relationships: above, below, left, right, front, behind
- Topological relationships: inside, contains, overlaps, adjacent
- Improved performance on spatial reasoning VQA benchmarks

**Top-View Spatial Reasoning** (EMNLP 2024) explores VLMs' capability to understand spatial relations from top-view perspectives. This viewpoint is particularly important for navigation, robotics, and scene planning tasks.

From [Vision-Language Models as Top-View Spatial Reasoners](https://aclanthology.org/2024.emnlp-main.106.pdf) (EMNLP 2024, accessed 2025-01-31):
- Top-view spatial reasoning for navigation and planning
- Bird's eye view understanding for layout comprehension
- Spatial relationship extraction from overhead perspectives

### Geometric Reasoning and Visual Grounding

**3D Visual Grounding** enables VLMs to locate specific objects in 3D scenes based on natural language descriptions. This requires understanding not just object semantics but also their spatial context within the scene.

**SeeGround** (Semantic Scholar 2024) introduces a zero-shot 3D visual grounding framework leveraging 2D VLMs trained on large-scale 2D data. The system projects 3D scenes into multiple 2D views, applies 2D grounding, and fuses results back into 3D space.

From [A VLM Agent for Zero-Shot 3D Visual Grounding](https://www.semanticscholar.org/paper/f1e19216db7b90b793ce33d8d8dfe225bef18d89) (Semantic Scholar 2024, accessed 2025-01-31):
- Zero-shot 3D grounding using 2D VLM capabilities
- Multi-view projection and 3D consistency checking
- Language-guided object localization in 3D scenes

**Geometric Imagination for Spatial Reasoning** (arXiv 2024) proposes that VLMs should develop internal 3D geometric representations even when processing 2D images. The approach trains models to imagine 3D structure and use it for spatial reasoning.

From [Think with 3D: Geometric Imagination Grounded Spatial Reasoning](https://arxiv.org/html/2510.18632v1) (arXiv 2024, accessed 2025-01-31):
- Learning to generate 3D latent representations from 2D images
- Alignment with 3D foundation models (e.g., VGGT)
- Geometric imagination as intermediate reasoning step
- Improved performance on spatial reasoning benchmarks

## Scene Understanding Architectures

### Vision-Language-Action Models for 3D

Recent architectures explicitly integrate 3D spatial understanding with vision-language models for embodied AI applications. These systems bridge perception, reasoning, and action in 3D space.

**FALCON** (alphaXiv 2024) introduces a framework grounding Vision-Language-Action (VLA) models in robust 3D spatial understanding. The system uses spatial awareness modules to maintain geometric consistency during manipulation planning.

From [Grounding Vision-Language-Action Model in Spatial](https://www.alphaxiv.org/overview/2510.17439v1) (alphaXiv 2024, accessed 2025-01-31):
- 3D spatial grounding for robotic manipulation
- Geometric consistency checking during action prediction
- Integration of spatial reasoning with action models

### Large-Scale 3D Scene Datasets

Training effective 3D VLMs requires large-scale datasets with comprehensive spatial annotations. Recent dataset efforts provide the foundation for scaling 3D scene understanding.

**SceneVerse** (ECCV 2024) introduces the first million-scale 3D vision-language dataset, encompassing 68K 3D indoor scenes with 2.5M vision-language pairs. This scale enables pre-training VLMs with diverse 3D spatial knowledge.

From [SceneVerse: Scaling 3D Vision-Language Learning for...](https://eccv.ecva.net/virtual/2024/poster/748) (ECCV 2024):
- 68K 3D indoor scenes from multiple datasets
- 2.5M vision-language annotation pairs
- Diverse spatial reasoning tasks and annotations
- Benchmark for evaluating 3D VLM capabilities

**LSceneLLM** (CVPR 2025) enhances large 3D scene understanding using adaptive visual preferences. The system learns to allocate attention based on query relevance, similar to human visual attention patterns.

From [LSceneLLM: Enhancing Large 3D Scene Understanding Using Adaptive Visual Preferences](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhi_LSceneLLM_Enhancing_Large_3D_Scene_Understanding_Using_Adaptive_Visual_Preferences_CVPR_2025_paper.pdf) (CVPR 2025):
- Adaptive attention allocation for 3D scenes
- Query-dependent visual preference learning
- Efficient processing of large-scale 3D environments

### Multi-Modal Integration

**Video-Based 3D Understanding** leverages temporal information to build 3D scene representations from video sequences. This approach mimics human perception, which constructs 3D understanding through motion and multiple viewpoints.

GPT4Scene (discussed earlier) demonstrates that video-based approaches can achieve state-of-the-art 3D understanding without explicit 3D inputs. The temporal consistency across frames provides implicit 3D cues that VLMs can learn to exploit.

**RGB-D Integration** combines color images with depth sensors (e.g., LiDAR, structured light) to provide explicit 3D geometric information. This approach is common in robotics and AR/VR applications where depth sensors are available.

From [Large Language Models and 3D Vision for Intelligent](https://www.mdpi.com/1424-8220/25/20/6394) (MDPI Sensors 2025):
- RGB-D fusion for 3D scene reconstruction
- Depth-guided spatial reasoning
- Applications in robotics, autonomous vehicles, and AR/VR

## Applications and Benchmarks

### Robotics and Embodied AI

3D spatial understanding is critical for robotic manipulation, navigation, and interaction. VLMs with spatial reasoning enable robots to understand natural language instructions grounded in physical 3D space.

**Robotic Manipulation Applications:**
- Object grasping and placement based on spatial descriptions
- Tool use requiring understanding of geometric constraints
- Multi-step tasks involving spatial planning
- Human-robot collaboration with shared spatial understanding

**Navigation Applications:**
- Natural language navigation commands ("go to the kitchen")
- Spatial relationship understanding ("the door next to the window")
- Path planning through complex 3D environments
- Dynamic obstacle avoidance with spatial prediction

### AR/VR and Spatial Computing

Augmented and virtual reality applications require precise 3D spatial understanding to place virtual objects correctly and enable natural interaction.

**AR/VR Use Cases:**
- Virtual object placement grounded in real-world geometry
- Spatial anchoring and persistence across sessions
- Natural language commands for scene manipulation
- Multi-user spatial collaboration

### Evaluation Datasets and Metrics

**ScanRefer** provides a benchmark for 3D visual grounding in real-world scans, requiring models to locate objects based on natural language descriptions within 3D scenes.

**ReferIt3D** extends referring expression comprehension to 3D, testing models' ability to identify objects based on spatial relationships and properties in 3D point clouds.

**Spatial-MM** (EMNLP 2024) introduces a comprehensive VQA dataset specifically designed to evaluate spatial understanding and reasoning capabilities of LMMs across diverse spatial relationship types.

From [An Empirical Analysis on Spatial Reasoning Capabilities of...](https://aclanthology.org/2024.emnlp-main.1195/) (EMNLP 2024, accessed 2025-01-31):
- Comprehensive spatial reasoning benchmark
- Multiple spatial relationship categories
- Quantitative and qualitative reasoning tasks
- Evaluation protocol for VLM spatial capabilities

### Open Challenges and Future Directions

**Lexicon3D** (NeurIPS 2024) provides the first comprehensive evaluation of 3D scene understanding with visual foundation models, revealing significant challenges:

From [Probing Visual Foundation Models for Complex 3D Scene...](https://proceedings.neurips.cc/paper_files/paper/2024/file/8c67fc501a50977947c5bebbc39ca8f6-Paper-Conference.pdf) (NeurIPS 2024):
- Systematic evaluation of foundation models on 3D tasks
- Performance gaps between 2D and 3D understanding
- Challenges in complex spatial reasoning
- Need for specialized 3D architectures and training

**Remaining Challenges:**

1. **Scale and Efficiency**: Processing large 3D scenes remains computationally expensive
2. **Temporal Consistency**: Maintaining coherent 3D understanding across video sequences
3. **Metric Precision**: Achieving human-level accuracy in quantitative spatial estimation
4. **Dynamic Scenes**: Understanding moving objects and changing spatial relationships
5. **Occlusion Handling**: Reasoning about hidden or partially visible objects
6. **Cross-Modal Alignment**: Better bridging language semantics with 3D geometry

**Future Research Directions:**

- **Self-Supervised 3D Learning**: Leveraging unlabeled 3D data at scale
- **Physics-Informed Spatial Reasoning**: Incorporating physical constraints and dynamics
- **Interactive 3D Learning**: Learning from embodied interaction and manipulation
- **Multi-Modal Fusion**: Better integration of RGB, depth, LiDAR, and language
- **Efficient 3D Architectures**: Reducing computational cost while maintaining accuracy

## Sources

**Research Papers:**

- [GPT4Scene: Understand 3D Scenes from Videos with Vision-Language Models](https://arxiv.org/abs/2501.01428) - arXiv:2501.01428 (accessed 2025-01-31)
- [SceneGPT: A Language Model for 3D Scene Understanding](https://arxiv.org/abs/2408.06926) - arXiv:2408.06926 (accessed 2025-01-31)
- [SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities](https://arxiv.org/abs/2401.12168) - arXiv:2401.12168, CVPR 2024 (accessed 2025-01-31)
- [SpatialRGPT: Grounded Spatial Reasoning in Vision-Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/f38cb4cf9a5eaa92b3cfa481832719c6-Paper-Conference.pdf) - NeurIPS 2024 (accessed 2025-01-31)
- [SD-VLM: Spatial Measuring and Understanding with Depth](https://arxiv.org/html/2509.17664v1) - arXiv:2509.17664 (accessed 2025-01-31)
- [Avi: A 3D Vision-Language Action Model Architecture](https://arxiv.org/abs/2510.21746) - arXiv:2510.21746 (accessed 2025-01-31)
- [3D-VLA: A 3D Vision-Language-Action Generative World Model](https://proceedings.mlr.press/v235/zhen24a.html) - ICML 2024 (accessed 2025-01-31)
- [PointLLM: Empowering Large Language Models to Understand Point Clouds](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03601.pdf) - ECCV 2024
- [Efficiently Aligning 3D Point Clouds with Large Language Models](https://dl.acm.org/doi/10.1145/3664647.3681257) - ACM MM 2024 (accessed 2025-01-31)
- [LSceneLLM: Enhancing Large 3D Scene Understanding Using Adaptive Visual Preferences](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhi_LSceneLLM_Enhancing_Large_3D_Scene_Understanding_Using_Adaptive_Visual_Preferences_CVPR_2025_paper.pdf) - CVPR 2025
- [SceneVerse: Scaling 3D Vision-Language Learning for...](https://eccv.ecva.net/virtual/2024/poster/748) - ECCV 2024
- [Vision-Language Models as Top-View Spatial Reasoners](https://aclanthology.org/2024.emnlp-main.106.pdf) - EMNLP 2024 (accessed 2025-01-31)
- [Improving Vision-and-Language Reasoning via Spatial Relations Modeling](https://openaccess.thecvf.com/content/WACV2024/papers/Yang_Improving_Vision-and-Language_Reasoning_via_Spatial_Relations_Modeling_WACV_2024_paper.pdf) - WACV 2024
- [Think with 3D: Geometric Imagination Grounded Spatial Reasoning](https://arxiv.org/html/2510.18632v1) - arXiv:2510.18632 (accessed 2025-01-31)
- [Grounding Vision-Language-Action Model in Spatial](https://www.alphaxiv.org/overview/2510.17439v1) - alphaXiv (accessed 2025-01-31)
- [A VLM Agent for Zero-Shot 3D Visual Grounding](https://www.semanticscholar.org/paper/f1e19216db7b90b793ce33d8d8dfe225bef18d89) - Semantic Scholar 2024 (accessed 2025-01-31)
- [An Empirical Analysis on Spatial Reasoning Capabilities of...](https://aclanthology.org/2024.emnlp-main.1195/) - EMNLP 2024 (accessed 2025-01-31)
- [Probing Visual Foundation Models for Complex 3D Scene...](https://proceedings.neurips.cc/paper_files/paper/2024/file/8c67fc501a50977947c5bebbc39ca8f6-Paper-Conference.pdf) - NeurIPS 2024
- [Large Language Models and 3D Vision for Intelligent](https://www.mdpi.com/1424-8220/25/20/6394) - MDPI Sensors 2025

**Additional References:**

- [GitHub: 3D-PointCloud Papers and Datasets](https://github.com/zhulf0804/3D-PointCloud)
- [GitHub: Awesome-3D-Visual-Grounding](https://github.com/liudaizong/Awesome-3D-Visual-Grounding)
- [SpatialVLM Project Website](https://spatial-vlm.github.io/)
- [GPT4Scene Project Website](https://gpt4scene.github.io/)
