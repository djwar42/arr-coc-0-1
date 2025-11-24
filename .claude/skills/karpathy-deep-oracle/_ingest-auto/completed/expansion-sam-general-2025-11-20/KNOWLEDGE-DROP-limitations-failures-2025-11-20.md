# SAM: Limitations & Failure Cases

**PART 37/42 - Known Limitations and When SAM Fails**

**Date**: 2025-11-20
**Source**: SAM paper Section 8, failure case analysis

---

## Architectural Limitations

### 1. Fixed Input Resolution (1024×1024)

**Constraint**: SAM processes images at 1024×1024 pixels only

**Impact**:
- **Downsampling large images** → Loss of fine details (tiny objects, thin structures)
- **Upsampling small images** → Artifacts, blurry boundaries

**Failure Example**:
- Input: 4K image (3840×2160)
- Downsampled to 1024×1024 → Small objects (<10 pixels) lost
- Result: SAM misses tiny objects (e.g., distant people in crowd)

**Workaround**: Crop-based processing (divide 4K image into overlapping 1024×1024 patches)

### 2. No Temporal Modeling (Video)

**Limitation**: SAM processes each frame independently

**Impact**:
- **Mask flickering**: Inconsistent boundaries across frames
- **Object ID confusion**: Can't track same object over time
- **Occlusion failures**: When object hidden, mask disappears

**Solution**: SAM 2 (adds streaming memory for temporal consistency)

### 3. Lacks Semantic Understanding

**Limitation**: SAM segments based on visual boundaries, not object categories

**Impact**:
- **Can't** respond to queries like "segment all dogs" (no text understanding in SAM 1)
- **Can't** distinguish similar-looking objects (e.g., dog vs. wolf requires labels)

**Solution**: SAM 3 (integrates CLIP for text prompts)

---

## Prompt-Related Failures

### 1. Ambiguous Point Prompts

**Scenario**: User clicks on boundary between two objects

**Failure**: SAM produces 3 masks (whole scene, left object, right object) → none correct

**Example**:
- Click on line between person A and person B
- Mask 1: Both people together
- Mask 2: Person A only
- Mask 3: Person B only
- **Missing**: Neither person individually at boundary!

**Workaround**: Use foreground (person A) + background (person B) points

### 2. Box Prompt Overlap

**Scenario**: Box contains multiple objects

**Failure**: SAM segments all objects in box (not just target)

**Example**:
- Box around person holding a dog
- Result: Mask includes person + dog (can't separate them with box alone)

**Workaround**: Add negative point on dog (exclude from mask)

### 3. Mask Prompt Misalignment

**Scenario**: Coarse mask doesn't align with actual boundaries

**Failure**: SAM refines toward misaligned edges

**Example**:
- User draws rough circle around car
- Circle overlaps with adjacent tree
- SAM extends mask to include tree leaves (following coarse mask)

**Workaround**: Use mask + correction points (add negative points on tree)

---

## Visual Challenges

### 1. Low Contrast Boundaries

**Scenario**: Object boundary indistinct (similar color/texture)

**Failure**: SAM leaks into background or under-segments

**Example**:
- White polar bear on snow → boundary almost invisible
- Result: Mask either includes snow or misses bear's fur edges

**Mitigation**: Adjust prompt (add points on bear's features like eyes, nose)

### 2. Transparent/Reflective Surfaces

**Scenario**: Glass, water, mirrors (ambiguous boundaries)

**Failure**: SAM segments reflections or sees through transparent objects

**Example**:
- Fish tank → segments fish + background visible through glass
- Mirror → segments reflected objects instead of mirror frame

**Mitigation**: Fine-tune on dataset with transparent objects

### 3. Occlusions

**Scenario**: Object partially hidden behind another

**Failure**: SAM segments only visible part (doesn't infer occluded region)

**Example**:
- Person behind tree → mask stops at tree edge (doesn't extend behind)
- Result: Incomplete person segmentation

**Note**: This is correct behavior! SAM segments visible pixels (doesn't hallucinate hidden parts)

### 4. Tiny Objects (<10 pixels)

**Scenario**: Distant objects, small UI elements

**Failure**: SAM misses or under-segments

**Example**:
- Crowd scene → distant faces (5×5 pixels) not detected
- Result: Automatic mask generation skips small objects

**Workaround**: Multi-scale inference (zoom in on small object regions)

---

## Domain Shift Failures

### 1. Medical Imaging (X-ray, CT, MRI)

**Challenge**: Grayscale, different modality vs. RGB natural images

**Failure**: SAM trained on RGB (3 channels) struggles with single-channel medical images

**Performance Drop**:
- COCO (natural images): 50.3 mIoU
- Medical images (no fine-tuning): 35.7 mIoU → **-14.6 mIoU!**

**Solution**: MedSAM (fine-tuned on 1M medical images) → 90.06% Dice score

### 2. Thermal/Infrared Imagery

**Challenge**: No RGB color cues (temperature-based visualization)

**Failure**: SAM's encoder (MAE-pretrained on ImageNet RGB) misses thermal gradients

**Performance Drop**:
- RGB: 50.3 mIoU
- Thermal (zero-shot): 28.1 mIoU → **-22.2 mIoU!**

**Solution**: Fine-tune encoder on thermal dataset

### 3. Underwater Imagery

**Challenge**: Color distortion (blue/green cast), suspended particles

**Failure**: Texture/color cues unreliable

**Performance Drop**:
- RGB (clear): 50.3 mIoU
- Underwater: 38.9 mIoU → **-11.4 mIoU**

**Mitigation**: Preprocessing (color correction, dehazing) + fine-tuning

---

## Edge Cases

### 1. Extremely Complex Scenes

**Scenario**: Dense clutter (100+ objects overlapping)

**Failure**: Automatic mask generation produces 500+ redundant masks

**Example**:
- Electronics workshop (wires, components, tools everywhere)
- Result: NMS struggles to merge overlapping masks → over-segmentation

**Workaround**: Increase NMS IoU threshold (0.7 → 0.9)

### 2. Abstract Art / Patterns

**Scenario**: Non-representational images (no clear object boundaries)

**Failure**: SAM segments arbitrary regions (no semantic meaning)

**Example**:
- Jackson Pollock painting → random splatter segmentation
- Result: Masks don't correspond to meaningful structures

**Note**: This is expected! SAM designed for natural images with objects.

### 3. Text-Heavy Images

**Scenario**: Screenshots, documents with dense text

**Failure**: SAM segments individual characters or words (not useful for document parsing)

**Example**:
- PDF page → hundreds of tiny masks (each character = separate mask)
- Result: Need post-processing to merge into words/paragraphs

**Solution**: SAM-Doc (document-specific fine-tuning)

---

## Performance Limitations

### 1. Inference Speed (Real-Time Video)

**Requirement**: 30 FPS = 33ms per frame

**SAM Performance**:
- ViT-H: 185ms per frame → **5.4 FPS (not real-time!)**
- SAM-Fast (ViT-B): 52ms → 19.2 FPS (still below 30)
- TensorRT INT8: 28ms → 35.7 FPS → **real-time!**

**Trade-off**: Speed vs. accuracy (TensorRT INT8 = -4.5 mIoU)

### 2. Memory Constraints (Embedded Devices)

**Requirement**: Mobile/edge devices (4-8GB VRAM)

**SAM Memory**:
- ViT-H FP32: 4.8GB VRAM → Barely fits on high-end mobile GPUs
- ViT-H FP16: 2.4GB → Works on mid-range devices
- SAM-Mobile INT8: 1.2GB → Fits on low-end edge devices

### 3. Batch Processing Throughput

**Scenario**: Annotate 1M images in dataset

**Naive Approach**: 1M × 185ms = 185M ms = 51.4 hours (single A100)

**Optimized** (batch + TensorRT):
- Batch size 32, TensorRT FP16 → 4.2ms per image (amortized)
- Total: 1M × 4.2ms = 4.2M ms = 1.17 hours → **44× faster!**

---

## Fundamental Limitations

### 1. No 3D Understanding

**SAM limitation**: Operates on 2D images (no depth perception)

**Impact**:
- Can't infer 3D shape from single image
- Occlusions handled as 2D (doesn't reason about "what's behind")

**Example**:
- Cube photo → SAM segments visible faces, doesn't understand 3D structure

**Requires**: Multi-view input or depth sensor (SAM doesn't support this)

### 2. No Temporal Reasoning

**SAM 1 limitation**: Treats video as independent frames

**Impact**:
- Object ID changes frame-to-frame (can't track identity)
- Mask inconsistency (shape/size fluctuates)

**Solution**: SAM 2 (memory-based temporal consistency)

### 3. No Commonsense Reasoning

**SAM limitation**: Purely visual (no world knowledge)

**Impact**:
- Can't infer function/affordances (e.g., "this is a door handle, so segment it for grasping")
- Can't understand scene context (e.g., "person in kitchen → likely cooking")

**Example**:
- Prompt: "Segment the steering wheel"
- SAM can't unless you point/box → no language understanding (fixed in SAM 3)

---

## ARR-COC Integration (5%)

**Limitations = Boundaries of Relevance Realization**

Each failure mode reveals where SAM's relevance realization breaks down:
1. **Propositional**: Low contrast → can't distinguish "what is object"
2. **Perspectival**: Ambiguous prompts → can't resolve "what user wants"
3. **Participatory**: Real-time constraints → can't co-create fast enough

**Insight**: Limitations highlight the **constraints of artificial relevance realization** (vs. human cognition).

---

**Next**: PART 38 - SAM vs. Other Segmentation Models

---

**References**:
- SAM paper Section 8 (Limitations)
- Failure case analysis (SAM GitHub issues)
