# Promptable Interface for Human Segmentation

## Overview

Promptable interfaces for human mesh recovery (HMR) represent a paradigm shift in 3D human pose and shape estimation. Rather than treating the problem as pure "pixels to parameters" regression, modern approaches leverage side information through spatial and semantic prompts to improve reconstruction accuracy and robustness. This document explores how segmentation masks, 2D keypoints, bounding boxes, and other prompts guide 3D human reconstruction.

**Key Innovation**: Moving beyond single-input regression to multi-modal prompt-based systems that can incorporate readily available information from vision-language models (VLMs), user input, or other AI systems to enhance 3D human understanding.

From [PromptHMR Paper](https://arxiv.org/abs/2504.06397) (Wang et al., CVPR 2025):
- Promptable HMR reformulates human pose and shape estimation through spatial and semantic prompts
- Processes full images to maintain scene context while accepting multiple input modalities
- Achieves state-of-the-art performance with flexible prompt-based control

---

## Section 1: Promptable Interface Design

### The Promptable Paradigm

Traditional HMR methods take a tightly cropped image of a person and output pose/shape in camera coordinates. This approach:
- Discards scene context essential for occlusion reasoning
- Fails in crowded scenes and close interactions
- Cannot exploit auxiliary "side information"

**Promptable HMR** reformulates the problem:

```
f: (Image, {Prompts}) -> {3D Humans}
```

From [PromptHMR](https://arxiv.org/abs/2504.06397):
> "Our key observation is that the classical 'pixels to parameters' formulation of the problem is too narrow. Today, we have large vision-language foundation models (VLMs) that understand a great deal about images and what people are doing in them. What these models lack, however, is an understanding of 3D human pose and shape."

### Prompt Types Taxonomy

**Spatial Prompts** (locate the person):
- **Bounding boxes**: Full-body, face-only, or truncated regions
- **Segmentation masks**: Precise pixel-level person identification
- **Point prompts**: Sparse location indicators

**Semantic Prompts** (describe the person):
- **Language descriptions**: "A tall, muscular male"
- **Interaction labels**: Two-person close contact indicators
- **Action descriptions**: What the person is doing

### Architecture Components

From [PromptHMR Architecture](https://arxiv.org/abs/2504.06397):

1. **Vision Transformer Encoder**
   - Processes high-resolution full images (896x896)
   - Preserves scene context for all people
   - Camera intrinsics embedded for metric accuracy

2. **Multi-Modal Prompt Encoder**
   - Maps different prompt types to common token space
   - Handles missing prompts with learned null tokens
   - Supports arbitrary combinations at inference

3. **Transformer Decoder**
   - Attends to both prompt and image tokens
   - Generates SMPL-X body parameters
   - Separate depth and pose regression heads

```python
# Conceptual promptable HMR architecture
class PromptHMR:
    def __init__(self):
        self.image_encoder = ViT_L()  # DINOv2 pretrained
        self.mask_encoder = MaskEncoder()
        self.prompt_encoder = PromptEncoder()
        self.smplx_decoder = TransformerDecoder()

    def forward(self, image, prompts):
        # Encode full image once (shared across all people)
        image_tokens = self.image_encoder(image)

        results = []
        for person_prompts in prompts:
            # Encode person-specific prompts
            box_tokens = self.prompt_encoder.encode_box(person_prompts.box)
            mask_tokens = self.mask_encoder(person_prompts.mask)
            text_tokens = self.prompt_encoder.encode_text(person_prompts.description)

            # Combine image with mask features
            person_image_tokens = image_tokens + mask_tokens

            # Decode SMPL-X parameters
            smplx_params = self.smplx_decoder(
                person_image_tokens,
                [box_tokens, text_tokens]
            )
            results.append(smplx_params)

        return results
```

### Design Principles

**Flexibility Through Prompts**:
- Any combination of prompts at inference time
- Random masking during training enables robustness
- Same model handles diverse scenarios

**Scene Context Preservation**:
- Full image processing (not crops)
- Multi-person joint reasoning
- Depth relationships from global view

**Metric Accuracy**:
- Camera intrinsics integration
- Normalized translation representation
- Inverse depth prediction

---

## Section 2: Segmentation Mask Prompts

### Why Masks Matter

Segmentation masks provide precise pixel-level identification of people, offering significant advantages over bounding boxes:

From [PromptHMR](https://arxiv.org/abs/2504.06397):
> "The mask prompt is more effective than boxes when people closely overlap, as boxes are ambiguous in such cases."

**Advantages of Mask Prompts**:
- Precise person boundaries in crowded scenes
- Disambiguation of overlapping individuals
- Better occlusion handling
- Reduced background interference

### Mask Encoding Architecture

```python
class MaskEncoder:
    """
    Encode segmentation masks as features added to image tokens.
    """
    def __init__(self, embed_dim=1024):
        # Strided convolutions for downsampling
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, embed_dim, 3, stride=2, padding=1),
        )
        # Learned token for missing masks
        self.no_mask_token = nn.Parameter(torch.zeros(1, embed_dim))

    def forward(self, mask, has_mask=True):
        if has_mask and mask is not None:
            # Downsample mask to match image token resolution
            mask_features = self.encoder(mask.unsqueeze(1))
            return mask_features.flatten(2).transpose(1, 2)
        else:
            # Return learned "no mask" token
            return self.no_mask_token.expand(batch_size, -1, -1)
```

### Mask Processing Pipeline

**Training with Masks**:
1. Generate masks from SMPL-X vertex projections
2. Optionally add noise/erosion for robustness
3. Random mask dropout (use null token instead)

**Integration with Image Features**:
```python
# Mask features added to image tokens
F_i = Encoder_mask(m_i) + F_image
```

This additive combination allows:
- Mask to highlight relevant image regions
- Image features to remain intact
- Flexible handling of partial masks

### Mask vs Box Performance

From [PromptHMR Experiments](https://arxiv.org/abs/2504.06397):

| Spatial Prompt | HI4D PA-MPJPE | HI4D Pair-PA-MPJPE |
|----------------|---------------|---------------------|
| Box only       | 47.0 mm       | 87.2 mm             |
| Mask only      | 43.4 mm       | 83.0 mm             |
| Box + Mask     | 36.5 mm       | 47.9 mm             |

**Key Finding**: Masks reduce error by 7-10% compared to boxes alone, with dramatic improvements in interaction scenarios (pair error drops from 87.2 to 47.9 mm).

### SAM Integration for Mask Generation

Modern systems can use Segment Anything Model (SAM) for automatic mask generation:

```python
# SAM-based mask generation for PromptHMR
from segment_anything import sam_model_registry, SamPredictor

def get_person_masks_for_hmr(image, detector_boxes):
    """
    Generate precise masks from detection boxes using SAM.
    """
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    masks = []
    for box in detector_boxes:
        # SAM generates mask from box prompt
        mask, score, _ = predictor.predict(
            box=box,
            multimask_output=False
        )
        masks.append(mask[0])

    return masks
```

### Applications

**Crowded Scene Analysis**:
- Each person identified by unique mask
- No ambiguity from overlapping boxes
- Better depth ordering inference

**Close Interactions**:
- Precise boundary separation
- Reduced interpenetration in reconstruction
- Clearer pose attribution

**Occlusion Handling**:
- Visible parts clearly delineated
- Occluded regions properly marked
- Better hallucination guidance

---

## Section 3: 2D Keypoint Prompts

### Keypoints as Spatial Guidance

2D keypoints provide sparse but highly informative spatial prompts for 3D pose estimation. They serve as intermediate representations that bridge 2D observations and 3D reconstruction.

From [FinePOSE](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_FinePOSE_Fine-Grained_Prompt-Driven_3D_Human_Pose_Estimation_via_Diffusion_Models_CVPR_2024_paper.pdf) (Xu et al., CVPR 2024):
> "In this work, we mainly focus on estimating 3D human poses given 2D keypoints."

### Two-Stage Pipeline

**Stage 1: 2D Keypoint Detection**
```python
# Off-the-shelf 2D pose estimation
def detect_2d_keypoints(image):
    """
    Use existing 2D pose detector (OpenPose, HRNet, etc.)
    Returns: keypoints shape (N_people, N_joints, 3) [x, y, confidence]
    """
    detector = HRNet()
    keypoints = detector(image)
    return keypoints
```

**Stage 2: 2D-to-3D Lifting**
```python
# Lift 2D keypoints to 3D pose
def lift_to_3d(keypoints_2d, model):
    """
    Transform 2D observations to 3D pose.
    Input: (N_joints, 2) 2D coordinates
    Output: (N_joints, 3) 3D coordinates
    """
    # Normalize keypoints
    keypoints_norm = normalize_keypoints(keypoints_2d)

    # Predict 3D via transformer/diffusion/regression
    pose_3d = model(keypoints_norm)

    return pose_3d
```

### Keypoint Encoding for Prompting

In promptable systems, keypoints can be encoded as:

**Positional Encoding**:
```python
class KeypointPromptEncoder:
    def __init__(self, embed_dim=256):
        self.joint_embed = nn.Embedding(num_joints, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)

    def forward(self, keypoints):
        # keypoints: (N_joints, 2) - 2D coordinates
        # Add joint identity embedding
        joint_tokens = self.joint_embed(torch.arange(num_joints))

        # Add positional encoding for coordinates
        pos_tokens = self.pos_encoder(keypoints)

        return joint_tokens + pos_tokens
```

**Graph-based Encoding**:
```python
class SkeletonGraphEncoder(nn.Module):
    """
    Encode keypoints using skeleton graph structure.
    """
    def __init__(self, in_dim=2, hidden_dim=128, out_dim=256):
        self.gcn_layers = nn.ModuleList([
            GraphConvolution(in_dim, hidden_dim),
            GraphConvolution(hidden_dim, hidden_dim),
            GraphConvolution(hidden_dim, out_dim)
        ])
        self.adjacency = get_skeleton_adjacency()

    def forward(self, keypoints):
        x = keypoints
        for gcn in self.gcn_layers:
            x = F.relu(gcn(x, self.adjacency))
        return x
```

### Confidence-Aware Processing

2D keypoint detectors provide confidence scores that should influence 3D reconstruction:

```python
def confidence_weighted_loss(pred_3d, gt_3d, keypoint_conf):
    """
    Weight 3D supervision by 2D detection confidence.
    """
    # Expand confidence to 3D
    conf_3d = keypoint_conf.unsqueeze(-1).expand_as(pred_3d)

    # Weighted L2 loss
    loss = (conf_3d * (pred_3d - gt_3d)**2).mean()

    return loss
```

### Benefits of Keypoint Prompts

**Robustness**:
- 2D detectors work well in-the-wild
- Face detection provides reliable prompts even in crowds
- Partial observations still useful

**Efficiency**:
- Sparse representation (17-25 joints vs millions of pixels)
- Fast 2D detection
- Lightweight lifting networks

**Composability**:
- Can combine with other prompts
- Works with boxes, masks, text
- Enables hierarchical reasoning

### Integration with Full HMR

From [PromptHMR](https://arxiv.org/abs/2504.06397), bounding boxes around keypoint groups serve as prompts:

> "To generate truncated boxes, we take groups of keypoints (e.g. upper body keypoints) and compute their bounding boxes."

This enables:
- **Face-only prompts**: Reconstruct full body from face detection
- **Partial body**: Work with truncated/occluded observations
- **Robust detection**: Face detectors work when body detectors fail

---

## Section 4: Bounding Box Prompts

### Box Types and Flexibility

PromptHMR accepts various bounding box types without requiring explicit type specification:

From [PromptHMR](https://arxiv.org/abs/2504.06397):
> "We design different box transformations during training to allow the model to use different boxes as a human identifier. In the training phase, each instance is prompted with either a whole-body bounding box, a face bounding box, or a truncated box covering part of the body."

**Box Types**:
1. **Full-body boxes**: Standard person detection output
2. **Face boxes**: From face detectors (more reliable in crowds)
3. **Truncated boxes**: Partial body regions (upper body, limbs)
4. **Noisy boxes**: With Gaussian perturbation for robustness

### Box Encoding

```python
class BoxPromptEncoder:
    """
    Encode bounding boxes as tokens using positional encoding.
    """
    def __init__(self, embed_dim=256):
        self.corner_embed = nn.Embedding(2, embed_dim // 2)  # Two corners
        self.pos_encoder = FourierPositionalEncoding(embed_dim // 2)

    def encode_box(self, box):
        """
        box: (2, 2) - [[x1, y1], [x2, y2]] corners
        Returns: (2, embed_dim) tokens
        """
        # Positional encoding for coordinates
        pos_tokens = self.pos_encoder(box)  # (2, embed_dim//2)

        # Add corner identity
        corner_tokens = self.corner_embed(torch.arange(2))  # (2, embed_dim//2)

        # Concatenate
        box_tokens = torch.cat([pos_tokens, corner_tokens], dim=-1)

        return box_tokens  # (2, embed_dim)
```

### Training with Box Augmentation

**Box Generation During Training**:
```python
def generate_training_boxes(smplx_vertices, keypoints, image_size):
    """
    Generate varied box types for training robustness.
    """
    # Full-body box from mesh projection
    proj_verts = project_to_2d(smplx_vertices)
    full_box = compute_bounding_box(proj_verts)

    # Face box from head vertices
    head_verts = proj_verts[HEAD_VERTEX_IDS]
    face_box = compute_bounding_box(head_verts)

    # Truncated boxes from keypoint groups
    upper_kpts = keypoints[UPPER_BODY_IDS]
    upper_box = compute_bounding_box(upper_kpts)

    # Random selection
    box_type = random.choice(['full', 'face', 'truncated'])
    box = {'full': full_box, 'face': face_box, 'truncated': upper_box}[box_type]

    # Add Gaussian noise to corners
    noise = torch.randn(2, 2) * noise_scale
    box = box + noise

    return box
```

### Face-Based Reconstruction

A key capability is reconstructing full bodies from face detection:

From [PromptHMR](https://arxiv.org/abs/2504.06397):
> "In crowded scenes, existing person detection methods struggle, while face detection methods remain reliable."

```python
# Example: Full body from face detection
def reconstruct_crowd_from_faces(image, face_detector, prompthmr):
    """
    Use face detection for reliable crowd reconstruction.
    """
    # Face detection works better in crowds
    face_boxes = face_detector(image)

    # PromptHMR reconstructs full body from face prompt
    people = []
    for face_box in face_boxes:
        person = prompthmr(
            image=image,
            prompts={'box': face_box}  # Just face box!
        )
        people.append(person)

    return people
```

### Box Robustness Results

From [PromptHMR Experiments](https://arxiv.org/abs/2504.06397):

> "PromptHMR remains stable when the boxes change and uses full image context to reconstruct the human even when the boxes are truncated."

The model:
- Handles varying box sizes gracefully
- Uses scene context to infer occluded parts
- Maintains accuracy with noisy/imperfect boxes

---

## Section 5: Multi-Modal Prompt Fusion

### Fusion Architecture

PromptHMR combines multiple prompt types through token concatenation and transformer attention:

```python
class PromptFusion:
    """
    Fuse multiple prompt modalities into unified representation.
    """
    def __init__(self, embed_dim=1024):
        self.smpl_query = nn.Parameter(torch.zeros(1, embed_dim))
        self.depth_query = nn.Parameter(torch.zeros(1, embed_dim))

    def fuse_prompts(self, box_tokens, text_tokens, has_text=True):
        """
        Combine different prompt tokens with query tokens.

        box_tokens: (2, embed_dim) - two corner tokens
        text_tokens: (1, embed_dim) - CLIP embedding

        Returns: (5, embed_dim) unified prompt
        """
        # Handle missing prompts
        if not has_text:
            text_tokens = self.null_text_token

        # Concatenate all tokens
        prompt_tokens = torch.cat([
            box_tokens,      # (2, embed_dim)
            text_tokens,     # (1, embed_dim)
            self.smpl_query, # (1, embed_dim)
            self.depth_query # (1, embed_dim)
        ], dim=0)

        return prompt_tokens  # (5, embed_dim)
```

### Cross-Modal Attention

The decoder attends across modalities:

```python
class MultiModalDecoder(nn.Module):
    """
    Transformer decoder attending to image and prompt tokens.
    """
    def __init__(self, embed_dim=1024, num_layers=3):
        self.layers = nn.ModuleList([
            MultiModalDecoderLayer(embed_dim)
            for _ in range(num_layers)
        ])

    def forward(self, image_tokens, prompt_tokens):
        """
        image_tokens: (N_patches, embed_dim)
        prompt_tokens: (N_prompts, embed_dim)
        """
        x = prompt_tokens

        for layer in self.layers:
            # Self-attention among prompt tokens
            x = layer.self_attention(x)

            # Cross-attention with image
            x = layer.cross_attention(x, image_tokens)

            # Feedforward
            x = layer.ffn(x)

        return x

class MultiModalDecoderLayer(nn.Module):
    def __init__(self, embed_dim):
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=8)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8)
        self.ffn = FeedForward(embed_dim)
```

### Language Integration for Shape

From [PromptHMR](https://arxiv.org/abs/2504.06397):
> "A sentence such as 'a muscular and tall male' is encoded with the CLIP text encoder."

**Shape Description Pipeline**:
```python
def generate_shape_description(smplx_params):
    """
    Generate text description from SMPL-X shape parameters.
    Uses SHAPY's shape-to-attribute method.
    """
    # Compute shape attributes (height, chest, waist, etc.)
    attributes = shapy_shape_to_attributes(smplx_params.betas)

    # Select top attributes
    top_attrs = sorted(attributes.items(), key=lambda x: -x[1])[:3]

    # Form natural language description
    attr_words = [attr_to_word[attr] for attr, _ in top_attrs]
    gender = "male" if smplx_params.gender == 0 else "female"

    description = f"a {' and '.join(attr_words)} {gender}"

    return description  # e.g., "a tall and broad-shoulder male"
```

**CLIP Encoding**:
```python
class ShapeTextEncoder:
    def __init__(self):
        self.clip_model = load_clip("ViT-L/14")

    def encode(self, text_description):
        """
        Encode shape description to CLIP embedding.
        """
        tokens = clip.tokenize([text_description])
        text_features = self.clip_model.encode_text(tokens)
        return text_features  # (1, 768)
```

### Training with Prompt Combinations

**Random Masking Strategy**:
```python
def sample_prompt_combination(full_prompts, p_mask=0.3):
    """
    Randomly mask prompts during training for robustness.
    """
    result = {}

    # Box is required (either full or face)
    result['box'] = full_prompts['box']

    # Optionally include mask
    if random.random() > p_mask:
        result['mask'] = full_prompts['mask']

    # Optionally include text
    if random.random() > p_mask:
        result['text'] = full_prompts['text']

    # Optionally include interaction
    if random.random() > p_mask:
        result['interaction'] = full_prompts['interaction']

    return result
```

### Experimental Impact

From [PromptHMR on HBW dataset](https://arxiv.org/abs/2504.06397):

| Training | Testing | Height Error | Chest Error | P2P-20k |
|----------|---------|--------------|-------------|---------|
| No text  | No text | 69 mm        | 51 mm       | 26      |
| + Text   | No text | 69 mm        | 48 mm       | 26      |
| + Text   | + Text  | 62 mm        | 43 mm       | 24      |

**Key Finding**: Training with text improves shape even without text at test time. Using text prompts at inference provides additional 10% improvement.

---

## Section 6: Interactive Refinement

### Two-Person Interaction Module

From [PromptHMR](https://arxiv.org/abs/2504.06397):
> "We introduce promptable layers in the decoder to model two-person interaction."

**Architecture**:
```python
class InteractionModule(nn.Module):
    """
    Cross-person attention for interaction modeling.
    """
    def __init__(self, embed_dim=1024):
        # Person identity embeddings
        self.person_embed = nn.Embedding(2, embed_dim)
        # Cross-person attention
        self.cross_person_attn = nn.MultiheadAttention(
            embed_dim, num_heads=8
        )

    def forward(self, tokens_person1, tokens_person2, is_interacting):
        """
        Apply cross-person attention if interaction flag is set.
        """
        if not is_interacting:
            # Skip interaction layer via residual
            return tokens_person1, tokens_person2

        # Add person identity
        t1 = tokens_person1 + self.person_embed(torch.tensor(0))
        t2 = tokens_person2 + self.person_embed(torch.tensor(1))

        # Concatenate for joint attention
        combined = torch.cat([t1, t2], dim=0)

        # Self-attention across both people
        attended, _ = self.cross_person_attn(
            combined, combined, combined
        )

        # Split back
        n = tokens_person1.shape[0]
        out1 = attended[:n] + tokens_person1  # Residual
        out2 = attended[n:] + tokens_person2

        return out1, out2
```

### Promptable Flow Control

The interaction module uses a binary prompt to enable/disable:

```python
class InteractionDecoder(nn.Module):
    def __init__(self, embed_dim, num_layers=3):
        self.layers = nn.ModuleList([
            InteractionDecoderLayer(embed_dim)
            for _ in range(num_layers)
        ])
        self.interaction_modules = nn.ModuleList([
            InteractionModule(embed_dim)
            for _ in range(num_layers)
        ])

    def forward(self, person_tokens_list, interaction_flags):
        """
        person_tokens_list: List of (5, embed_dim) per person
        interaction_flags: (N_people,) binary interaction indicators
        """
        for i, layer in enumerate(self.layers):
            # Standard decoder operations per person
            for j in range(len(person_tokens_list)):
                person_tokens_list[j] = layer(person_tokens_list[j])

            # Interaction module for pairs
            if len(person_tokens_list) == 2:
                k = interaction_flags[0]  # Binary flag
                if k:  # Interaction enabled
                    person_tokens_list[0], person_tokens_list[1] = \
                        self.interaction_modules[i](
                            person_tokens_list[0],
                            person_tokens_list[1],
                            is_interacting=True
                        )

        return person_tokens_list
```

### Interaction Results

From [PromptHMR on HI4D](https://arxiv.org/abs/2504.06397):

| Method | PA-MPJPE | MPJPE | Pair-PA-MPJPE |
|--------|----------|-------|---------------|
| No interaction training | 47.0 | 71.4 | 87.2 |
| + Mask prompts | 43.4 | 60.5 | 83.0 |
| + Interaction module | 43.7 | 61.3 | 73.0 |
| + HI4D training | 36.3 | 49.4 | 52.6 |
| Full system | 36.5 | 47.1 | 47.9 |

**Key Finding**: The interaction module reduces Pair-PA-MPJPE from 87.2 to 73.0 mm even without HI4D training data, demonstrating out-of-domain generalization.

### User-Guided Refinement

Beyond automatic prompts, the system enables interactive refinement:

```python
class InteractiveHMR:
    """
    System for user-guided mesh refinement.
    """
    def __init__(self, prompthmr_model):
        self.model = prompthmr_model

    def initial_reconstruction(self, image, boxes):
        """
        First pass with detected boxes.
        """
        return self.model(image, {'boxes': boxes})

    def refine_with_mask(self, image, person_idx, mask):
        """
        User provides refined mask for specific person.
        """
        self.current_prompts[person_idx]['mask'] = mask
        return self.model(image, self.current_prompts)

    def add_shape_description(self, person_idx, description):
        """
        User provides shape description.
        """
        self.current_prompts[person_idx]['text'] = description
        return self.model(image, self.current_prompts)

    def mark_interaction(self, person1_idx, person2_idx):
        """
        User marks two people as interacting.
        """
        self.current_prompts[person1_idx]['interaction'] = True
        self.current_prompts[person2_idx]['interaction'] = True
        return self.model(image, self.current_prompts)
```

### Iterative Refinement Pipeline

```python
def iterative_reconstruction(image, detector, hmr_model):
    """
    Progressive refinement with multiple prompt types.
    """
    # Stage 1: Box-based initial reconstruction
    boxes = detector(image)
    result_v1 = hmr_model(image, {'boxes': boxes})

    # Stage 2: Add SAM masks for precision
    masks = sam_generate_masks(image, boxes)
    result_v2 = hmr_model(image, {
        'boxes': boxes,
        'masks': masks
    })

    # Stage 3: Add VLM shape descriptions
    descriptions = vlm_describe_shapes(image, boxes)
    result_v3 = hmr_model(image, {
        'boxes': boxes,
        'masks': masks,
        'texts': descriptions
    })

    # Stage 4: Mark detected interactions
    interactions = detect_interactions(result_v3)
    result_v4 = hmr_model(image, {
        'boxes': boxes,
        'masks': masks,
        'texts': descriptions,
        'interactions': interactions
    })

    return result_v4
```

---

## Section 7: ARR-COC-0-1 Integration - Promptable 3D for VLM Queries

### Natural Language to 3D Pose Queries

ARR-COC-0-1's vision-language architecture can leverage promptable HMR to answer 3D spatial questions about humans:

**Query Types**:
- "What is the person on the left doing with their hands?"
- "Is the tall person reaching up or down?"
- "How close are the two people standing?"

**Integration Architecture**:
```python
class ARRCOCHumanUnderstanding:
    """
    VLM-integrated promptable human mesh recovery.
    """
    def __init__(self, vlm, prompthmr, sam):
        self.vlm = vlm  # ARR-COC VLM
        self.hmr = prompthmr
        self.sam = sam

    def process_human_query(self, image, query):
        """
        Answer 3D human pose questions using VLM + HMR.
        """
        # Step 1: VLM identifies relevant people
        vlm_response = self.vlm.identify_people(image, query)
        # Returns: boxes, descriptions, interactions

        # Step 2: Generate precise masks with SAM
        masks = self.sam.generate_from_boxes(
            image, vlm_response.boxes
        )

        # Step 3: Reconstruct 3D with all prompts
        people_3d = self.hmr(image, {
            'boxes': vlm_response.boxes,
            'masks': masks,
            'texts': vlm_response.descriptions,
            'interactions': vlm_response.interactions
        })

        # Step 4: Answer query with 3D information
        answer = self.answer_3d_query(
            query, people_3d, vlm_response
        )

        return answer
```

### 3D Relevance Realization for Humans

Extending ARR-COC's relevance allocation to human-specific understanding:

```python
class Human3DRelevanceAllocator:
    """
    Allocate attention based on 3D human pose relevance.
    """
    def compute_human_relevance(self, people_3d, query_embedding):
        """
        Determine which people and body parts are relevant.
        """
        relevance_scores = {}

        for person_id, person in enumerate(people_3d):
            # Joint-level relevance
            joint_positions = person.get_joints()
            joint_relevance = self.compute_joint_relevance(
                joint_positions, query_embedding
            )

            # Body part relevance (hands, face, torso, etc.)
            part_relevance = self.aggregate_to_parts(joint_relevance)

            # Person-level relevance
            person_relevance = part_relevance.mean()

            relevance_scores[person_id] = {
                'overall': person_relevance,
                'parts': part_relevance,
                'joints': joint_relevance
            }

        return relevance_scores

    def focus_attention(self, image_features, relevance_scores):
        """
        Modulate image attention based on 3D relevance.
        """
        # Project 3D relevant regions to 2D
        attention_mask = self.project_relevance_to_2d(
            relevance_scores
        )

        # Weight image features
        focused_features = image_features * attention_mask

        return focused_features
```

### Action Understanding from 3D Pose

```python
class Human3DActionReasoning:
    """
    Reason about human actions using 3D pose.
    """
    def analyze_action(self, person_3d, query):
        """
        Determine action semantics from 3D pose.
        """
        # Extract pose features
        joint_positions = person_3d.joints
        joint_velocities = person_3d.velocities  # For video

        # Compute action-relevant features
        features = {
            'hand_positions': joint_positions[HAND_IDS],
            'body_orientation': person_3d.global_orient,
            'arm_angles': self.compute_arm_angles(joint_positions),
            'leg_stance': self.analyze_stance(joint_positions),
        }

        # Match to action semantics
        if self.is_reaching(features):
            action = "reaching"
            direction = self.get_reach_direction(features)
        elif self.is_walking(features):
            action = "walking"
            direction = self.get_movement_direction(features)
        # ... more action patterns

        return action, direction
```

### Spatial Relationship Queries

```python
class SpatialRelationshipAnalyzer:
    """
    Answer spatial relationship questions using 3D.
    """
    def analyze_relationship(self, person1_3d, person2_3d, query):
        """
        Compute 3D spatial relationships.
        """
        # Distance metrics
        pelvis_distance = torch.norm(
            person1_3d.pelvis - person2_3d.pelvis
        )

        closest_distance = self.compute_closest_points(
            person1_3d.vertices, person2_3d.vertices
        )

        # Relative positioning
        relative_position = person1_3d.pelvis - person2_3d.pelvis
        is_in_front = relative_position[2] > 0  # Depth
        is_to_left = relative_position[0] < 0

        # Interaction analysis
        are_touching = closest_distance < CONTACT_THRESHOLD
        facing_each_other = self.check_facing(
            person1_3d.global_orient,
            person2_3d.global_orient
        )

        return {
            'distance': pelvis_distance,
            'relative_position': relative_position,
            'touching': are_touching,
            'facing': facing_each_other
        }
```

### VLM Prompt Generation from 3D

```python
class HMRToVLMBridge:
    """
    Convert 3D reconstruction to VLM-understandable prompts.
    """
    def generate_3d_description(self, people_3d):
        """
        Create text description of 3D scene for VLM reasoning.
        """
        descriptions = []

        for i, person in enumerate(people_3d):
            # Position description
            pos_desc = self.describe_position(person.translation)

            # Pose description
            pose_desc = self.describe_pose(person.body_pose)

            # Shape description
            shape_desc = self.describe_shape(person.betas)

            descriptions.append(
                f"Person {i+1}: {pos_desc}. {pose_desc}. {shape_desc}."
            )

        # Scene-level description
        if len(people_3d) > 1:
            interaction_desc = self.describe_interactions(people_3d)
            descriptions.append(interaction_desc)

        return "\n".join(descriptions)

    def describe_pose(self, body_pose):
        """
        Convert SMPL-X pose to natural language.
        """
        # Analyze key joint angles
        arm_state = self.analyze_arms(body_pose)
        leg_state = self.analyze_legs(body_pose)
        torso_state = self.analyze_torso(body_pose)

        return f"Arms {arm_state}, legs {leg_state}, torso {torso_state}"
```

### Future Integration Directions

**Bidirectional VLM-HMR Communication**:
1. VLM provides semantic context for HMR prompts
2. HMR provides 3D spatial grounding for VLM reasoning
3. Iterative refinement between systems

**3D Tokens for Transformers**:
```python
# Future: Direct 3D human tokens in VLM
class Human3DTokenizer:
    def tokenize_person(self, person_3d):
        """
        Convert 3D person to tokens for transformer.
        """
        # Pose tokens (per joint)
        pose_tokens = self.joint_tokenizer(person_3d.joints)

        # Shape tokens
        shape_tokens = self.shape_tokenizer(person_3d.betas)

        # Position tokens
        pos_tokens = self.position_tokenizer(person_3d.translation)

        return torch.cat([pose_tokens, shape_tokens, pos_tokens])
```

---

## Sources

**Primary Research Papers**:
- [PromptHMR: Promptable Human Mesh Recovery](https://arxiv.org/abs/2504.06397) - Wang et al., CVPR 2025 (accessed 2025-11-20)
- [FinePOSE: Fine-Grained Prompt-Driven 3D Human Pose Estimation via Diffusion Models](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_FinePOSE_Fine-Grained_Prompt-Driven_3D_Human_Pose_Estimation_via_Diffusion_Models_CVPR_2024_paper.pdf) - Xu et al., CVPR 2024
- [UniPHD: Unified Promptable Human Pose Detection](https://openreview.net/forum?id=fXEi3LVflp) - OpenReview, 2024

**Project Pages**:
- [PromptHMR Project Page](https://yufu-wang.github.io/phmr-page/) - Official code and results
- [PromptHMR GitHub](https://github.com/yufu-wang/PromptHMR) - Implementation

**Related Methods**:
- [HMR - Human Mesh Recovery](https://akanazawa.github.io/hmr/) - Original HMR method
- [SHAPY](https://shapy.is.tue.mpg.de/) - Shape from language descriptions
- [BUDDI](https://buddi.is.tue.mpg.de/) - Diffusion prior for interactions
- [SAM 3D Body](https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/) - Meta AI (accessed 2025-11-20)

**Technical Resources**:
- [Segment Anything Model](https://segment-anything.com/) - Mask generation
- [SMPL-X](https://smpl-x.is.tue.mpg.de/) - Expressive body model
- [CLIP](https://openai.com/research/clip) - Text-image encoding

**Benchmarks**:
- HI4D - Human interaction dataset
- CHI3D - Close human interaction
- HBW - Human body in the wild (shape)
- 3DPW - 3D Poses in the Wild
- EMDB - Electromagnetic database
