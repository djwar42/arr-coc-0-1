# Glossary: LOD and BSP/BTree Key Terms

**Quick reference for essential terminology in Level of Detail and Binary Space Partitioning systems**

---

## Core Concepts

### Level of Detail (LOD)
A technique that reduces the number of GPU operations by rendering distant or less important objects with lower geometric/texture complexity. Can be discrete (predefined levels) or continuous (smooth transitions).

### Binary Space Partitioning (BSP)
A method of recursively subdividing space into two convex sets using hyperplanes, creating a BSP tree structure. Used for efficient spatial queries, collision detection, and rendering optimization.

### BTree (B-Tree)
A self-balancing tree data structure that maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time. Widely used in databases and file systems.

---

## LOD Terminology

### Discrete LOD
Predefined levels of geometric complexity (e.g., LOD0, LOD1, LOD2) with abrupt transitions between levels.

### Continuous LOD
Smooth, progressive transitions between detail levels without visible "popping" artifacts.

### View-Dependent LOD
LOD selection based on camera distance, viewing angle, screen-space projection, or user attention.

### Progressive Meshes
A representation allowing smooth LOD transitions by incrementally adding or removing vertices/triangles.

### Popping
Visual artifact when switching between discrete LOD levels, where geometry suddenly changes.

### Hysteresis
Technique to prevent LOD oscillation by using different thresholds for increasing vs decreasing detail levels.

### Screen-Space Error
Metric measuring how much geometric simplification affects the object's appearance on screen.

---

## BSP/Spatial Partitioning

### BSP Tree
Binary tree where each node represents a partitioning plane dividing space into two half-spaces (front and back).

### Leaf Node
Terminal node in BSP tree containing actual spatial contents (empty space, solid geometry, etc.).

### Partition Plane (Splitting Plane)
Hyperplane dividing space in a BSP node. In 2D: line. In 3D: plane.

### Convex Set
Region where any line segment between two points in the region lies entirely within the region.

### Axial Plane
Partition aligned with coordinate axes (horizontal, vertical in 2D; walls/floors in 3D). Reduces polygon splitting.

### Portal
Opening between adjacent BSP leafs, used for visibility determination and rendering optimization.

### Potentially Visible Set (PVS)
Set of leafs/regions potentially visible from a given location, precomputed for fast runtime queries.

---

## Perceptual Rendering

### Foveated Rendering
Rendering technique that provides highest detail in the foveal region (center of gaze) and progressively lower detail in peripheral vision.

### Gaze-Aware Display
System that tracks user's eye position and adjusts rendering based on gaze direction.

### Fixed Foveated Rendering (FFR)
Foveated rendering assuming fixed gaze direction (typically screen center).

### Eye-Tracked Foveated Rendering (ETFR)
Dynamic foveated rendering using real-time eye tracking data.

### Peripheral Degradation
Reducing rendering quality in areas outside central vision, matching human visual system limitations.

### Eccentricity
Angular distance from foveal center, used to determine LOD allocation in foveated rendering.

### Saccade
Rapid eye movement between fixation points. Rendering can be reduced during saccades.

### Visual Attention
Cognitive process of selectively concentrating on salient visual information while ignoring other perceivable information.

### Preattentive Processing
Rapid, automatic visual processing occurring before focused attention, used to guide LOD allocation.

### Change Blindness
Failure to detect visual changes when they occur during eye movements or brief occlusions.

---

## Rendering Techniques

### Occlusion Culling
Not rendering objects obscured by other geometry, reducing GPU workload.

### View Frustum Culling
Excluding objects outside the camera's viewing volume from rendering pipeline.

### Back-Face Culling
Not rendering polygon faces pointing away from camera.

### Tessellation
Subdividing geometric primitives into smaller pieces, typically done on GPU.

### Progressive Buffers
View-dependent geometry and texture LOD rendering with smooth transitions.

### Slippy Maps
Tile-based rendering system for geographic/terrain data, supporting seamless zooming and panning.

---

## Terrain-Specific

### Heightfield
2D grid where each point has a height value, defining terrain surface.

### Watertight Tessellation
Tessellation ensuring no gaps (cracks) between adjacent patches.

### Incremental Tessellation
Real-time adaptive tessellation based on view parameters.

### Terrain Synthesis
Procedural generation of realistic terrain using noise functions, erosion simulation, etc.

### Tile Streaming
Loading terrain tiles on-demand based on camera position.

---

## Personalization & Adaptation

### Personalized LOD
LOD strategies adapted to individual user preferences, hardware capabilities, or usage patterns.

### User Modeling
Creating profiles of user behavior/preferences to inform rendering decisions.

### Adaptive Quality
Dynamic adjustment of rendering quality based on performance constraints and user requirements.

### Transjective Relevance
Relevance emerging from relationship between viewer (agent) and content (arena), not purely objective or subjective.

---

## Technical Metrics

### Triangle Count
Number of triangles in a mesh, primary metric for geometric complexity.

### Draw Calls
Number of rendering commands sent to GPU per frame.

### Fill Rate
Rate at which GPU can write pixels to framebuffer.

### Vertex Transform Rate
Rate at which GPU can process vertices.

### Frame Time
Time to render single frame, inverse of frame rate.

### LOD Bias
User-adjustable parameter shifting LOD selection toward higher or lower detail.

---

## CSG (Constructive Solid Geometry)

### Brush
Basic convex solid primitive used in BSP-based level editors.

### CSG Operation
Boolean operation (union, subtraction, intersection) combining brushes.

### Convex Brush
Brush with no concave (inward-pointing) features.

### Clipping Hull
Simplified collision geometry created by expanding brush boundaries outward.

### Detail Brush
Brush excluded from BSP visibility calculations to reduce compile time and improve performance.

---

## VR/AR Specific

### Stereoscopic Rendering
Rendering separate images for left and right eyes to create 3D depth perception.

### Latency
Delay between user input (head movement, gaze shift) and visual response.

### Motion-to-Photon Latency
Time from physical motion to corresponding pixels appearing on display.

### Reprojection
Technique warping previous frame based on head movement to reduce latency artifacts.

### Foveation Gain
Performance improvement from foveated rendering, typically 2-4x GPU savings.

---

## Database/Query Systems

### OLAP (Online Analytical Processing)
Multidimensional data analysis requiring LOD for visualization of large datasets.

### Data Cube
Multidimensional array of values, often requiring LOD navigation.

### Query-Aware LOD
LOD selection informed by user queries, emphasizing relevant data dimensions.

---

## Related to ARR-COC-VIS

### Relevance Realization
Process of dynamically determining what information is relevant in a given context (Vervaeke framework).

### Opponent Processing
Balancing tensions between competing priorities (e.g., compression vs particularization).

### Propositional Knowing
Knowledge as information content (knowing THAT).

### Perspectival Knowing
Knowledge through salience landscapes (knowing WHAT IT'S LIKE).

### Participatory Knowing
Knowledge through agent-arena coupling (knowing BY BEING).

### Procedural Knowing
Knowledge as learned skills (knowing HOW).

---

## Cross-References

For detailed explanations, see:
- **concepts/** - Foundational principles
- **techniques/** - Implementation methods
- **algorithms/** - Computational approaches
- **applications/** - Domain-specific uses
- **integration/** - System combinations

---

**Source**: Compiled from research papers in `RESEARCH/Video Game LOD and BTree/NotebookLM_Sources/` and supplementary web research.
