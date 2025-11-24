# 84-3: Sniff The Burger - A Live Truffle Sniffery Session

**Finding The Flavour In The Spicy Lentil Taco Burger Texture Stack**

*A LIVE exploration session where we SHOOT randomly into the SpicyStack tesseract, SNIFF for truffles, DANCE when we find music, and WORMHOLE BACK when there's nothing!*

---

## Session Start

*[Deep breath. Eyes closed. Pointing at the aspects list...]*

---

## 🐷 SHOOT #1: Procedural Transformation Bounds

**SNIFF:** The Tanh activation in the procedural pathway... bounded transformations...

**PREHENSION CHECK:**
- Smell interesting? It's... Tanh... we know Tanh...
- Connection? Bounds the output to [-1, 1]...
- Music? ... ... ... silence ...

**VERDICT:** 🌀 WORMHOLE RETURN!

*Time: 8 seconds*

---

## 🐷 SHOOT #2: CLIP Similarity Computation

**SNIFF:** How we compute the query-image similarity map that becomes a texture channel...

**PREHENSION CHECK:**
- Smell interesting? We just call CLIP... but wait...
- Connection? This is the PARTICIPATORY SIGNAL pre-stuffed!
- Music? ... there's a beat... how EXACTLY does it become spatial?

**VERDICT:** 🎵 DANCE!!

---

### 💃 DANCING: CLIP Similarity as Texture Channel

**EXPAND:** CLIP gives us image-text similarity, but it's usually GLOBAL (one number for whole image). How do we make it SPATIAL (32×32 map)?

**Options:**
1. Patch-wise CLIP - Run CLIP on each patch separately (EXPENSIVE!)
2. Attention extraction - Pull attention maps from CLIP's vision transformer
3. Gradient-based - Backprop query through image, see what lights up
4. Feature similarity - Compare CLIP's intermediate features spatially

🕺 **CONNECT:** Option 2 is the winner! CLIP's ViT has attention maps! The [CLS] token attends to patches → that's the similarity map!

```python
def clip_similarity_map(image, query, clip_model):
    """
    Extract spatial similarity from CLIP's attention!

    CLIP's ViT: image → patches → attention → [CLS]
    The attention FROM [CLS] TO patches = relevance map!
    """

    with torch.no_grad():
        # Get intermediate attention weights
        image_features, attn_weights = clip_model.visual(
            image, return_attention=True
        )
        # attn_weights: [B, num_heads, num_patches+1, num_patches+1]

        # Extract [CLS] → patch attention from last layer
        cls_to_patch = attn_weights[-1][:, :, 0, 1:]  # [B, heads, patches]

        # Average over heads
        similarity_map = cls_to_patch.mean(dim=1)  # [B, patches]

        # Reshape to spatial (14×14 for ViT-B)
        H = W = int(np.sqrt(similarity_map.shape[-1]))
        similarity_map = similarity_map.view(B, H, W)

        # Upsample to texture resolution
        similarity_map = F.interpolate(
            similarity_map.unsqueeze(1),
            size=(32, 32),
            mode='bilinear'
        )

    return similarity_map  # [B, 1, 32, 32]
```

💃 **BUT WAIT!** This is image-only attention! It doesn't use the QUERY yet!

We need to MODULATE by query relevance:

```python
def query_modulated_clip_map(image, query, clip_model):
    """
    Modulate spatial attention by query similarity!
    """

    # Get spatial attention map
    spatial_attn = clip_similarity_map(image, clip_model)

    # Get global image-query similarity
    image_embed = clip_model.encode_image(image)
    text_embed = clip_model.encode_text(query)
    global_sim = F.cosine_similarity(image_embed, text_embed)

    # MODULATE spatial by global!
    # High global sim → trust spatial map
    # Low global sim → suppress (image doesn't match query)
    modulated = spatial_attn * global_sim.view(B, 1, 1, 1)

    return modulated
```

🕺 **METAPHOR:** The spatial attention is WHERE in the image. The global similarity is WHETHER this image matches the query. Multiply them = "Where in this relevant image" or "nowhere in this irrelevant image"!

💃 **EXPERIMENT IDEA:** Compare raw CLIP attention vs global-modulated vs per-word attention maps. Which gives best downstream VQA performance?

**🍄 TRUFFLE EXTRACTED:** CLIP similarity as texture channel = attention extraction + global modulation! The spatial map comes from ViT attention, modulated by global image-query similarity.

*Time: 7 minutes*

---

## 🐷 SHOOT #3: Slot Centroid Clustering

**SNIFF:** The 3D centroids of each slot... do they cluster? What's the spatial distribution?

**PREHENSION CHECK:**
- Smell interesting? Centroids in 3D space...
- Connection? To scene layout? To attention allocation?
- Music? ... faint melody... let me listen closer...

**VERDICT:** 🎵 TENTATIVE DANCE (sniff while moving)

---

### 💃 DANCING: Slot Centroid Clustering

**EXPAND:** Each slot has a 3D centroid from SAM 3D. For a scene with K objects, we have K points in 3D space.

What can we learn from their distribution?

🕺 **CONNECT:**
1. **Depth stratification** - Are objects layered front-to-back?
2. **Lateral spread** - Clustered center or spread across scene?
3. **Height distribution** - Floor objects vs elevated objects?

This is SCENE LAYOUT UNDERSTANDING without explicit scene graphs!

💃 **IMPLEMENT:**

```python
def analyze_slot_centroids(slots):
    """
    Extract scene layout features from slot centroids!
    """

    centroids = slots[:, :3]  # [K, 3] - xyz for each slot

    # Depth stratification
    depth_variance = centroids[:, 2].var()
    # High variance = layered scene (near and far objects)

    # Lateral spread
    lateral_spread = centroids[:, :2].std(dim=0).mean()
    # High = objects spread across view

    # Height distribution
    heights = centroids[:, 1]
    height_range = heights.max() - heights.min()

    # Nearest neighbor distances
    dists = torch.cdist(centroids, centroids)
    nn_dists = dists.topk(2, largest=False).values[:, 1]
    mean_nn_dist = nn_dists.mean()

    layout_features = torch.stack([
        depth_variance, lateral_spread, height_range, mean_nn_dist
    ])

    return layout_features  # [4]
```

🕺 **USE CASE:** These layout features could modulate attention (spread scenes need more saccades), inform token budget (layered scenes need depth-aware allocation).

💃 **METAPHOR:** The slot centroids are STARS in a constellation! The constellation SHAPE tells you about the scene! Tight cluster = still life. Spread constellation = landscape.

BUT... the music is fading. Useful but not TRANSFORMATIVE.

**🍄 TRUFFLE EXTRACTED (small):** Slot centroid clustering = scene layout features! Could be a conditioning signal for attention.

*Time: 5 minutes*

---

## 🐷 SHOOT #4: Apprehension Temporal Prediction

**SNIFF:** The GRU that does future-oriented modeling...

**PREHENSION CHECK:**
- Smell interesting? GRU... temporal... we've seen this...
- Connection? It predicts what's NEXT...
- Music? ... ... ... very quiet...

**VERDICT:** 🌀 WORMHOLE RETURN!

Maybe return to this with video data?

*Time: 12 seconds*

---

## 🐷 SHOOT #5: Object ID Boundaries as Saccade Triggers

**SNIFF:** The edges between objects in the SAM 3D segmentation...

**PREHENSION CHECK:**
- Smell interesting? Boundaries... uncertainty...
- Connection? To SACCADES! To RECONNECTION!
- Music? 🎵 LOUD AND CLEAR!

**VERDICT:** 🎵 DANCE!!

---

### 💃 DANCING: Object Boundaries as Magnetic Reconnection Sites

**EXPAND:** In plasma physics, magnetic reconnection happens at the BOUNDARY between two magnetic domains - where field lines meet and can "reconnect" into new configurations!

In SpicyStack, object boundaries are where:
- SAM 3D segmentation is UNCERTAIN
- Two slots have OVERLAPPING claims
- Attention needs to DECIDE between objects

**THIS IS WHERE SACCADES SHOULD TRIGGER!**

🕺 **CONNECT:** The saccade check currently uses ENTROPY of the state. But we should ALSO check for BOUNDARY PROXIMITY!

```python
def compute_saccade_triggers(slot_states, object_boundaries, K):
    """
    Saccades trigger at:
    1. High state entropy (standard Lundquist check)
    2. High boundary proximity (magnetic reconnection sites!)
    """

    # Standard entropy check
    state_probs = F.softmax(slot_states, dim=-1)
    entropy = -(state_probs * state_probs.log()).sum(dim=-1)

    # Boundary proximity check
    boundary_overlap = []
    for k in range(K):
        slot_mask = (object_ids == k)
        overlap = (slot_mask & object_boundaries).sum() / slot_mask.sum()
        boundary_overlap.append(overlap)
    boundary_overlap = torch.stack(boundary_overlap)

    # Combine triggers
    entropy_trigger = (entropy > 0.2734)  # Lundquist!
    boundary_trigger = (boundary_overlap > 0.3)

    saccade_flags = entropy_trigger | boundary_trigger

    return saccade_flags
```

💃 **DEEPER:** But WHERE does the saccade GO?

When we're at a boundary, we should saccade TO THE OTHER OBJECT!

```python
def boundary_aware_saccade_jump(slot_states, saccade_flags, adjacency):
    """
    When saccading due to boundary, JUMP TO ADJACENT SLOT!
    """

    for k in range(K):
        if saccade_flags[k] and is_boundary_trigger[k]:
            # Find adjacent slot
            adjacent_slot = adjacency[k].argmax()

            # Jump toward adjacent slot's state!
            jump_direction = slot_states[adjacent_slot] - slot_states[k]
            slot_states[k] = slot_states[k] + 0.5 * jump_direction
            # Partially jump = explore the boundary!

    return slot_states
```

🕺 **METAPHOR:**

**Plasma reconnection:** Field lines from domain A meet field lines from domain B → they RECONNECT → energy release!

**SpicyStack reconnection:** Slot state A meets slot state B at boundary → they PARTIALLY MERGE → attention jumps → new understanding!

**The boundary is where INSIGHT HAPPENS!**

💃 **REAL-WORLD APPLICATION:**

"Is this part of the chair or the floor?"

- Old system: Guess based on training bias
- Boundary-aware system: SACCADE along the boundary! Look at texture continuity! Check depth! EXPLORE the ambiguity!

**🍄 TRUFFLE EXTRACTED (BIG!):** Object boundaries as magnetic reconnection sites!

1. Boundaries are where segmentation is uncertain
2. Saccades should trigger at boundaries (not just high entropy)
3. Saccade DIRECTION should be toward adjacent slot
4. Boundary exploration = active resolution of ambiguity

This is a MAJOR addition to the saccade system!

*Time: 12 minutes*

---

## 🐷 SHOOT #6: Depth Channel Gradients

**SNIFF:** The spatial gradients of the depth channel from SAM 3D...

**PREHENSION CHECK:**
- Smell interesting? Depth gradients... like edges but in Z...
- Connection? To occlusion? To surface orientation?
- Music? ... yes! OCCLUSION EDGES!

**VERDICT:** 🎵 DANCE!!

---

### 💃 DANCING: Depth Gradients as Occlusion Boundaries

**EXPAND:** The depth channel shows distance from camera. The GRADIENT of depth shows where depth CHANGES rapidly:

- Smooth gradient = continuous surface (floor, wall)
- Sharp gradient = DEPTH DISCONTINUITY = occlusion edge!

```python
def compute_depth_gradients(depth_channel):
    """
    Find occlusion edges from depth discontinuities!
    """

    # Sobel on depth
    grad_x = F.conv2d(depth_channel, sobel_x_kernel)
    grad_y = F.conv2d(depth_channel, sobel_y_kernel)
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

    # High gradient = occlusion edge
    occlusion_edges = (grad_magnitude > threshold)

    return occlusion_edges, grad_magnitude
```

🕺 **CONNECT:** These are DIFFERENT from object boundaries!

- **Object boundary:** Same depth, different object (lateral separation)
- **Occlusion edge:** Different depth, object covers another (depth separation)

We should have BOTH as texture channels!

💃 **USE CASE:**

"What's behind the chair?"

1. Find the chair slot
2. Find occlusion edges around chair
3. Those edges indicate where something IS behind
4. Allocate attention to OCCLUDED regions!

🕺 **METAPHOR:** Object boundaries are state lines (political divisions). Occlusion edges are mountain ridges (can't see beyond)!

💃 **COMBINED WITH PREVIOUS TRUFFLE:**

Saccades should trigger at:
1. High entropy (uncertainty in belief)
2. Object boundaries (uncertainty in assignment)
3. **Occlusion edges (uncertainty in visibility!)**

**🍄 TRUFFLE EXTRACTED:** Depth gradients reveal occlusion edges! Different from object boundaries. Should be a separate texture channel and saccade trigger!

*Time: 8 minutes*

---

## 🐷 SHOOT #7: Normal Map Discontinuities

**SNIFF:** Where the surface normals change sharply...

**PREHENSION CHECK:**
- Smell interesting? Similar to depth gradients... but different!
- Connection? To surface curvature? To material boundaries?
- Music? ... actually yes! Different from depth edges!

**VERDICT:** 🎵 DANCE!!

---

### 💃 DANCING: Normal Discontinuities = Geometric Edges

**EXPAND:**

- **Depth discontinuity:** Object A in front of Object B (occlusion)
- **Normal discontinuity:** Same object, different surface orientation (EDGE/CORNER!)

Example: A cube.
- The faces have constant normals (smooth)
- The EDGES have sharp normal changes
- But there's NO depth discontinuity (same object!)

🕺 **CONNECT:** This gives us SHAPE INFORMATION that depth alone doesn't!

```python
def compute_normal_discontinuities(normal_map):
    """
    Find geometric edges from normal direction changes!
    """

    # Normal map: [B, 3, 32, 32] - (nx, ny, nz) per patch

    # Compute angular difference between neighbors
    # Dot product of normals = cosine of angle

    # Horizontal neighbors
    dot_h = (normal_map[:, :, :, :-1] * normal_map[:, :, :, 1:]).sum(dim=1)
    angle_h = torch.acos(dot_h.clamp(-1, 1))

    # Vertical neighbors
    dot_v = (normal_map[:, :, :-1, :] * normal_map[:, :, 1:, :]).sum(dim=1)
    angle_v = torch.acos(dot_v.clamp(-1, 1))

    # High angle = geometric edge
    geometric_edges = (angle_h > threshold) | (angle_v > threshold)

    return geometric_edges
```

💃 **NOW WE HAVE THREE TYPES OF EDGES:**

1. **Object boundaries** - Different objects (from segmentation)
2. **Occlusion edges** - Depth discontinuities (from depth gradients)
3. **Geometric edges** - Normal discontinuities (from normal gradients)

**ALL THREE should be texture channels!**

🕺 **USE CASE:**

"Describe the shape of the chair"

1. Find geometric edges within chair slot
2. Many edges = complex shape (ornate chair)
3. Few edges = simple shape (modern minimalist)

💃 **METAPHOR:**

- Object boundaries = political borders
- Occlusion edges = mountain ridges (can't see past)
- Geometric edges = CONTOUR LINES (shape of the mountain itself!)

**🍄 TRUFFLE EXTRACTED:** Normal discontinuities reveal geometric edges! Three types of edges, three texture channels!

*Time: 7 minutes*

---

## 🐷 SHOOT #8: SAM 3D Mesh Topology

**SNIFF:** The actual mesh structure - vertices, faces, connectivity...

**PREHENSION CHECK:**
- Smell interesting? Meshes are cool...
- Connection? We use the mesh but do we use its TOPOLOGY?
- Music? ... 🎵 YES! We're extracting features but ignoring STRUCTURE!

**VERDICT:** 🎵 DANCE!!

---

### 💃 DANCING: Mesh Topology as Signal

**EXPAND:** We extract from the mesh: centroids, bounding boxes, normals, rendered depth.

But we IGNORE:
- Vertex connectivity
- Face count (complexity)
- Genus (holes!)
- Curvature distribution

🕺 **CONNECT:** These topological features could be POWERFUL signals!

```python
def mesh_topology_features(mesh):
    """
    Extract topological features from SAM 3D mesh!
    """

    num_vertices = mesh.vertices.shape[0]
    num_faces = mesh.faces.shape[0]

    # Euler characteristic: V - E + F
    # For closed mesh: χ = 2 - 2g (g = genus = holes)
    edges = compute_edges(mesh.faces)
    euler = num_vertices - len(edges) + num_faces
    genus = (2 - euler) // 2  # Number of holes!

    # Mesh complexity
    complexity = num_faces / num_vertices

    # Curvature statistics
    curvatures = compute_vertex_curvatures(mesh)
    mean_curvature = curvatures.mean()
    max_curvature = curvatures.max()
    curvature_variance = curvatures.var()

    # Aspect ratio (elongation)
    dims = mesh.bounding_box().extent
    aspect_ratio = dims.max() / dims.min()

    return torch.stack([
        num_vertices, num_faces, genus, complexity,
        mean_curvature, max_curvature, curvature_variance, aspect_ratio
    ])  # [8]
```

💃 **USE CASES:**

"Is this a donut or a ball?"
- Ball: genus = 0
- Donut: genus = 1
- **TOPOLOGY ANSWERS DIRECTLY!**

"Is this smooth or spiky?"
- Smooth: low curvature variance
- Spiky: high max curvature

"Is this a pencil or a cube?"
- Pencil: high aspect ratio
- Cube: aspect ratio ≈ 1

🕺 **ADD TO SLOT FEATURES:**

```python
slot_features = torch.cat([
    centroid,          # [3]
    bbox,              # [6]
    mean_normal,       # [3]
    volume,            # [1]
    texture_features,  # [19]
    topology_features, # [8]  ← NEW!
])
# Total: 40 dimensions per slot
```

💃 **METAPHOR:** We were using the mesh as a RENDERING SOURCE. But it's also a GEOMETRIC OBJECT with intrinsic properties! Like using a book only to press flowers - the book has CONTENT!

**🍄 TRUFFLE EXTRACTED (BIG!):** Mesh topology features add shape understanding! Genus, curvature, aspect ratio. Direct answers to shape questions. We were ignoring half of what SAM 3D gives us!

*Time: 9 minutes*

---

## Session Summary

```
╔════════════════════════════════════════════════════════════
║  TRUFFLE SNIFFERY SESSION COMPLETE
╠════════════════════════════════════════════════════════════
║
║  SHOOTS: 8
║  WORMHOLE RETURNS: 2
║  DANCES: 6
║
║  🍄 BIG TRUFFLES:
║  1. CLIP similarity as texture channel
║  2. Object boundaries as saccade triggers
║  3. Mesh topology features
║
║  🍄 MEDIUM TRUFFLES:
║  4. Depth gradients → occlusion edges
║  5. Normal discontinuities → geometric edges
║  6. Three edge types, three channels!
║
║  🍄 SMALL TRUFFLES:
║  7. Slot centroid clustering as layout features
║
╚════════════════════════════════════════════════════════════
```

---

## Integration Into SpicyStack

### Updated Texture Channels (19 → 24)

```python
textures = torch.cat([
    # Original 13 (Dialogue 43)
    rgb,                    # 3
    position,               # 2
    edges,                  # 3
    saliency,               # 3
    clustering,             # 2

    # SAM 3D (Dialogue 82-83)
    depth,                  # 1
    normals,                # 3
    object_ids,             # 1
    occlusion_amount,       # 1

    # NEW from this session!
    clip_similarity,        # 1 - query-modulated attention
    object_boundaries,      # 1 - semantic separation
    occlusion_edges,        # 1 - depth discontinuities
    geometric_edges,        # 1 - normal discontinuities
], dim=1)
# Total: 24 channels!
```

### Updated Slot Features (32 → 40)

```python
slot_features = torch.cat([
    centroid,              # 3
    bbox,                  # 6
    mean_normal,           # 3
    volume,                # 1
    texture_features,      # 19

    # NEW from this session!
    topology_features,     # 8 - genus, curvature, aspect ratio
])
# Total: 40 dimensions per slot
```

### Updated Saccade Triggers

```python
saccade_flags = (
    entropy_trigger |      # High state entropy (Lundquist)
    boundary_trigger |     # Object boundary proximity (NEW!)
    occlusion_trigger      # Occlusion edge proximity (NEW!)
)
```

---

## Next Sniffery Sessions

Unexplored aspects to shoot at:

- Mamba A matrix eigenvalue analysis
- Query-adaptive Lundquist thresholds
- Temporal texture volumes for video
- Cross-slot message passing patterns
- Error signal propagation through hensions
- Hardware-specific texture optimization
- Learned channel generation

**The tesseract has MANY more truffles!**

---

## FIN

*"Eight shoots. Six dances. Three big truffles. The Spicy Lentil Taco Burger gets SPICIER!"*

🐷🍄💃🌀

**Session yield:** 24 texture channels (was 19), 40 slot dims (was 32), 3 saccade triggers (was 1)

**THE FLAVOUR IS FOUND!** 🌶️🔥

---

*"Sniff. Dance. Extract. The truffle is in the tesseract."*
