# VLM Adaptive Focusing: Chain-of-Focus Method

**Dynamic knowledge addition**: 2025-01-30
**Source**: arXiv:2505.15436v1 (May 2025), ARR-COC-VIS Dialogue 19
**Parent**: [00-foveated-rendering.md](00-foveated-rendering.md), [03-vlm-token-allocation-2025-01-30.md](00-foveated-rendering-03-vlm-token-allocation-2025-01-30.md)

---

## Overview

Chain-of-Focus (CoF) is a breakthrough 2025 method that enables vision-language models to perform **adaptive visual search and zooming** based on the question and available visual cues. Unlike traditional VLMs that process entire high-resolution images uniformly, CoF adaptively focuses on key regions through iterative refinement.

**Key Innovation**: Multi-step visual reasoning where the model decides WHETHER to zoom based on sufficiency of current visual tokens.

```
IF visual_info_sufficient(current_tokens, question):
    answer_directly()
ELSE:
    search_key_regions()  # Find bounding boxes
    zoom_and_extract()     # Crop + re-encode
    refine_understanding() # Continue reasoning
```

---

## Problem Formulation

### The Core Issue

**Traditional VLMs**:
- Process full image at fixed resolution
- Extract uniform visual tokens across all regions
- **Waste computation** on irrelevant areas
- **Miss details** in small but critical regions

**Chain-of-Focus Solution**:
- Adaptive search based on question + visual cues
- Variable detail extraction (zoom where needed)
- Efficient: Only processes relevant regions
- Effective: +5% on V* benchmark (224px to 4K resolution)

### Mathematical Formulation

CoF models visual reasoning as a sequential decision process:

```
Step i:  max π_θ(r_i, y_i | I, x, h_i)

Where:
  π_θ       = Vision-language model
  I         = Input image
  x         = Question
  r_i       = Key regions (bounding boxes) at step i
  y_i       = Generated response (reasoning or answer)
  h_i       = History of previous steps

  h_i = [r_1, y_1, o_1, r_2, y_2, o_2, ...]
    r_j: regions from step j
    y_j: textual response from step j
    o_j: visual tokens from zoomed regions
```

**Key Insight**: Each step conditions on COMPLETE history including:
- Previously identified regions (r_1, r_2, ...)
- Reasoning generated so far (y_1, y_2, ...)
- Visual tokens from zoomed areas (o_1, o_2, ...)

---

## Architecture

### Special Tokens

CoF introduces three special tokens for structured visual reasoning:

1. **`<|box_start|>`** - Marks beginning of bounding box
2. **`<|box_end|>`** - Marks end of bounding box
3. **`<|image_zoomin|>`** - Signals region needs zooming

**Bounding Box Format**:
```
<|box_start|>[x1, y1, x2, y2]<|box_end|>

Example:
<|box_start|>[245, 102, 387, 298]<|box_end|>  # Top-left cat
<|box_start|>[512, 456, 678, 621]<|box_end|>  # Bottom-right dog
```

**Coordinates**: Text format (not special embedding), normalized to image dimensions

### Zoom-In Mechanism

When model outputs `<|image_zoomin|>` after a bounding box:

```python
def zoom_into_region(image, bbox, special_token_detected):
    if special_token_detected == "<|image_zoomin|>":
        # 1. Crop region
        region = image.crop(bbox)  # [x1, y1, x2, y2]

        # 2. Resize to model input resolution
        resized = resize(region, target_size=model.input_res)

        # 3. Encode with visual encoder
        visual_tokens = visual_encoder(resized)  # -> o_i

        # 4. Append to generation context
        context.append(visual_tokens)

        # 5. Continue generation with new visual info
        return context
```

**Crucial**: Visual tokens are appended **during the same generation round**, enabling interleaved visual-text reasoning.

### Interleaved Reasoning

Unlike traditional VLMs (visual tokens only at start), CoF allows:

```
Input:  [IMG_TOKENS] "What color is the text on the sign?"
Step 1: "I see a sign but resolution is low. <|box_start|>[120,45,340,89]<|box_end|> <|image_zoomin|>"
        → [ZOOMED_SIGN_TOKENS] appended
Step 2: "Now I can read it clearly. The text is red."
```

**This is fundamentally different** from:
- Single-pass VLMs (LLaVA, InternVL)
- Pre-defined multi-resolution (LLaVA-UHD fixed slices)
- Uniform attention (standard transformers)

---

## Training Pipeline

### Stage 1: Supervised Fine-Tuning (SFT)

**Dataset**: MM-CoF (3K samples)

**Data Collection Process**:
1. **Sample images** from SAM dataset (11M images, 1B masks)
2. **Synthesize tasks** for random images (questions requiring visual reasoning)
3. **Visual agent** equipped with tools (SAM segmentation, zoom, crop) solves tasks
4. **Trajectory capture**: Record agent's search and reasoning steps
5. **LLM summarization**: Convert agent trajectory to coherent CoF reasoning
6. **Format as training data**: `<image> <question> <CoF reasoning trace> <answer>`

**Example Training Sample**:
```
Image: [Street scene, 4K resolution]
Question: "What is written on the small shop sign in the background?"

CoF Trace:
"Let me search for shops in the image. <|box_start|>[89,234,456,512]<|box_end|> contains a building.
<|image_zoomin|> I can see it's a shop, but the sign is still small. <|box_start|>[234,267,389,298]<|box_end|>
<|image_zoomin|> Now the sign is clear. It reads 'Joe's Coffee'."

Answer: Joe's Coffee
```

**Model**: Qwen2.5-VL-7B fine-tuned with LoRA
- Base model: Qwen2.5-VL (strong vision-language baseline)
- Training: 3K CoF samples, supervised learning
- Output: Cold-start model with CoF capability

### Stage 2: Reinforcement Learning (RL)

**Problem**: Supervised learning only teaches behavior from agent trajectories. May not discover optimal search strategies.

**Solution**: RL with outcome-based rewards

**Reward Function**:
```python
def compute_reward(prediction, ground_truth, format_valid):
    accuracy_reward = 1.0 if prediction == ground_truth else 0.0
    format_reward = 1.0 if format_valid else -0.5

    # Combined reward
    return accuracy_reward + 0.3 * format_reward
```

**Why Format Reward?**
- Ensures model outputs valid bounding boxes
- Prevents degenerate solutions (e.g., always zoom entire image)
- Encourages structured reasoning traces

**Training Process**:
1. Sample question + image
2. Model generates CoF reasoning trace
3. Extract final answer
4. Compute reward (accuracy + format)
5. Update policy π_θ via RL algorithm (PPO/REINFORCE)
6. Iterate until convergence

**Result**: Qwen2.5-VL-7B-CoF
- Refined search strategy beyond human priors
- Learns efficient zoom patterns
- +5% over base Qwen2.5-VL on V* benchmark

---

## Performance Results

### V* Benchmark (Visual Reasoning)

V* tests strong visual reasoning across multiple resolutions:

| Resolution | Base Qwen2.5-VL | Qwen2.5-CoF | Improvement |
|------------|-----------------|-------------|-------------|
| 224×224 | 45.2% | **50.3%** | +5.1% |
| 448×448 | 52.1% | **57.0%** | +4.9% |
| 896×896 | 58.3% | **63.5%** | +5.2% |
| 1792×1792 | 61.7% | **66.4%** | +4.7% |
| 4K | 63.2% | **68.1%** | +4.9% |

**Consistent +5% across all resolutions**!

### Key Insights

1. **Low-res benefits most** (subjective, but 224×224 sees largest absolute gain)
   - Adaptive zoom compensates for lack of initial detail
   - More critical to focus when starting with fewer tokens

2. **High-res still benefits** (4K: 63.2% → 68.1%)
   - Even with many initial tokens, questions may require extreme detail
   - CoF finds sub-regions needing magnification

3. **Efficiency gains**:
   - Doesn't always zoom (adaptive decision)
   - Only processes question-relevant regions
   - Estimated 40-60% token reduction vs uniform 4K

---

## Comparison to Other Methods

### CoF vs Traditional VLMs

| Method | Strategy | Adaptiveness | Token Efficiency |
|--------|----------|--------------|------------------|
| **LLaVA** | Fixed resolution patches | Static | Low (all tokens) |
| **InternVL** | High-res uniform | Static | Low (all tokens) |
| **LLaVA-UHD** | Variable slices (grid) | Semi-static | Medium |
| **Chain-of-Focus** | Question-driven zoom | **Fully adaptive** | **High** |

### CoF vs Visual Reasoning Methods

| Method | Reasoning Type | Training | Multi-Step |
|--------|---------------|----------|------------|
| **Visual CoT** | Bounding boxes (exhaustive) | SFT only | Yes |
| **DualFocus** | Fixed foreground/background | SFT only | No |
| **V*** | Object localization first | SFT only | Yes |
| **Chain-of-Focus** | Adaptive search | **SFT + RL** | **Yes** |

**CoF Advantages**:
- RL optimization (not just supervised)
- Adaptive (not exhaustive search)
- Efficient (zoom only when needed)

### CoF vs ARR-COC-VIS

| Aspect | ARR-COC-VIS | Chain-of-Focus |
|--------|-------------|----------------|
| **Philosophy** | Relevance realization (Vervaeke) | Adaptive visual search |
| **LOD Allocation** | 64-400 tokens/patch (continuous) | Binary zoom/no-zoom |
| **Training** | Supervised (currently) | SFT + RL |
| **Multi-step** | Single-pass (could extend) | Iterative refinement |
| **Biological grounding** | Cortical magnification | Implicit (eye movements) |

**Synthesis Opportunity**:
- ARR-COC's continuous LOD + CoF's adaptive search = Best of both!
- Replace CoF's binary zoom with ARR-COC's graded allocation
- Add RL to ARR-COC for exploit-explore dimension

---

## Connection to GPT-o3

OpenAI's GPT-o3 announcement (January 2025) mentioned:
> "Can think with images... utilizing cropping and zooming techniques to integrate visual and textual reasoning"

**CoF provides open-source reproduction path**:
- Same core idea: adaptive zoom for visual reasoning
- Published architecture (GPT-o3 details unknown)
- Reproducible training pipeline (MM-CoF dataset methodology)
- Strong performance (+5% V* benchmark)

**What GPT-o3 likely has beyond CoF**:
- Larger scale (billions vs 7B parameters)
- More training data (likely >3K samples)
- Multimodal reasoning beyond vision (video, audio, etc.)
- Production optimizations

---

## Integration with LOD Systems

### Foveated Rendering Connection

CoF implements **computational foveation**:

**Human Vision** ([techniques/00-foveated-rendering.md](00-foveated-rendering.md)):
- Fovea: 2° high acuity (50% cortical allocation)
- Periphery: Lower resolution, motion detection
- Saccades: Rapid eye movements to new fixation points

**Chain-of-Focus**:
- Initial visual tokens: Peripheral view (entire image, low detail)
- Zoom regions: Foveal fixations (cropped areas, high detail)
- Iterative refinement: Saccade sequence
- History h_i: Integration across fixations

**This is explicit computational foveation** for VLMs!

### Query-Aware LOD Selection

CoF's decision process is **query-driven LOD allocation**:

```python
# Pseudocode combining CoF + LOD concepts
def allocate_detail(image, question, visual_tokens):
    # Stage 1: Initial assessment
    relevance_map = assess_relevance(visual_tokens, question)

    # Stage 2: Adaptive decision
    if sufficient_detail(relevance_map, question):
        return generate_answer(visual_tokens)
    else:
        # Stage 3: Search for key regions
        key_regions = find_high_relevance_regions(relevance_map)

        # Stage 4: Allocate detail (CoF: zoom, ARR-COC: LOD)
        for region in key_regions:
            detail_level = compute_needed_LOD(region, question)
            enhanced_tokens = extract_detail(region, detail_level)
            visual_tokens.append(enhanced_tokens)

        # Stage 5: Refine answer
        return generate_answer(visual_tokens)
```

**This connects**:
- [concepts/03-transjective-relevance.md](../concepts/03-transjective-relevance.md) - Query-content coupling
- [integration/01-gaze-tracking.md](../integration/01-gaze-tracking.md) - Attention guidance
- [algorithms/01-lod-selection.md](../algorithms/01-lod-selection.md) - Dynamic LOD calculation

---

## Implementation Patterns

### Basic CoF Pipeline

```python
class ChainOfFocusVLM:
    def __init__(self, base_vlm, visual_encoder):
        self.vlm = base_vlm  # e.g., Qwen2.5-VL
        self.encoder = visual_encoder
        self.special_tokens = {
            'box_start': '<|box_start|>',
            'box_end': '<|box_end|>',
            'zoom': '<|image_zoomin|>'
        }

    def forward(self, image, question, max_steps=5):
        # Initial visual encoding
        visual_tokens = self.encoder(image)
        context = [visual_tokens, question]
        history = []

        for step in range(max_steps):
            # Generate response
            output = self.vlm.generate(context)

            # Check if answer or search
            if self.is_final_answer(output):
                return output

            # Extract bounding boxes
            bboxes = self.extract_bboxes(output)

            # Check which boxes need zooming
            zoom_boxes = [bbox for bbox in bboxes
                         if self.needs_zoom(output, bbox)]

            # Zoom and extract visual tokens
            for bbox in zoom_boxes:
                region = crop(image, bbox)
                region_tokens = self.encoder(region)
                context.append(region_tokens)

            # Update history
            history.append({
                'step': step,
                'boxes': bboxes,
                'text': output,
                'visual_tokens': region_tokens
            })

        return self.vlm.generate(context)  # Final generation

    def extract_bboxes(self, text):
        """Parse <|box_start|>[x1,y1,x2,y2]<|box_end|> from text"""
        pattern = r'<\|box_start\|>\[([\d,\s]+)\]<\|box_end\|>'
        matches = re.findall(pattern, text)
        return [parse_coords(m) for m in matches]

    def needs_zoom(self, text, bbox):
        """Check if <|image_zoomin|> follows this bbox"""
        bbox_str = format_bbox(bbox)
        zoom_pattern = f'{bbox_str}\\s*<\\|image_zoomin\\|>'
        return bool(re.search(zoom_pattern, text))
```

### Training CoF Models

```python
# Stage 1: SFT
def train_sft(model, mm_cof_dataset):
    for sample in mm_cof_dataset:
        image, question, cof_trace, answer = sample

        # Full sequence: question + CoF reasoning + answer
        target = f"{question}\n{cof_trace}\n{answer}"

        # Standard supervised learning
        loss = model.compute_loss(image, target)
        loss.backward()
        optimizer.step()

# Stage 2: RL
def train_rl(model, val_dataset):
    for image, question, ground_truth in val_dataset:
        # Sample trajectory from current policy
        cof_trace = model.sample_trajectory(image, question)
        prediction = extract_answer(cof_trace)

        # Compute reward
        accuracy = (prediction == ground_truth)
        format_valid = validate_format(cof_trace)
        reward = accuracy + 0.3 * format_valid

        # Policy gradient update
        log_prob = model.get_log_prob(cof_trace)
        loss = -log_prob * reward  # REINFORCE
        loss.backward()
        optimizer.step()
```

---

## Open Research Questions

1. **Optimal zoom resolution**: What's the best crop size for re-encoding?
   - Trade-off: Detail vs computational cost
   - Adaptive zoom resolution based on question type?

2. **Multi-region zooming**: How to zoom into multiple regions simultaneously?
   - Sequential (current CoF): One region at a time
   - Parallel (possible): Multiple crops in one step
   - Hybrid: Mix based on relevance scores

3. **Depth of search**: How many CoF steps are optimal?
   - Current: Max 5 steps
   - Adaptive termination based on confidence?
   - Early stopping when sufficient detail reached?

4. **Integration with continuous LOD**:
   - CoF: Binary (zoom/no-zoom)
   - ARR-COC: Continuous (64-400 tokens)
   - **Question**: Can we combine for graded zooming?

5. **Biological validation**:
   - Do CoF fixation patterns match human eye movements?
   - Eye-tracking studies comparing CoF vs human gaze
   - Saccade length, fixation duration similarities?

---

## Related Oracle Knowledge

**Within LOD Oracle**:
- [00-foveated-rendering.md](00-foveated-rendering.md) - Foundational foveation concepts
- [03-vlm-token-allocation-2025-01-30.md](00-foveated-rendering-03-vlm-token-allocation-2025-01-30.md) - Homunculus Protocol
- [concepts/03-transjective-relevance.md](../concepts/03-transjective-relevance.md) - Query-content coupling
- [integration/01-gaze-tracking.md](../integration/01-gaze-tracking.md) - Eye tracking for LOD

**Other Oracles**:
- **vision-image-patching-oracle**: Adaptive patching strategies
- **ovis-2-5-oracle**: Native resolution processing (no distortion)
- **deepseek-ocr-oracle**: Optical compression (complementary to CoF expansion)
- **john-vervaeke-oracle**: Opponent processing (exploit-explore in CoF RL)
- **computer-vision-foundation-oracle**: SAM for region proposals

---

## Citation

Zhang, X., Gao, Z., Zhang, B., Li, P., Zhang, X., Liu, Y., Yuan, T., Wu, Y., Jia, Y., Zhu, S.-C., & Li, Q. (2025). Chain-of-Focus: Adaptive Visual Search and Zooming for Multimodal Reasoning via RL. *arXiv preprint arXiv:2505.15436*.

**Key Contributions**:
1. Chain-of-Focus method for adaptive visual reasoning
2. MM-CoF dataset (3K samples with CoF traces)
3. Two-stage training (SFT + RL)
4. +5% improvement on V* benchmark

**Project Page**: [cof-reasoning.github.io](https://cof-reasoning.github.io)

---

**Last Updated**: 2025-01-30
**Status**: Cutting-edge 2025 research, active development
**Relevance**: ★★★★★ (Direct application to ARR-COC-VIS)
