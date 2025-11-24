# Gradio HF Testing Thoughts - ARR-COC MVP Validation

**Quick braindump of testing/validation ideas for HuggingFace Space**

---

## 🎯 Core Goal

Build interactive validation interface that lets users (and us) SEE if relevance realization actually works.

Not about accuracy metrics. About **qualitative validation**: "Does this make sense?"

---

## 📊 Gradio Components We Need

### 1. The Main Interface

```python
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            # INPUT
            image_input = gr.Image(type="pil", label="Upload Image")
            query_input = gr.Textbox(label="Query", placeholder="Where is the cat?")

            # Quick test queries
            gr.Examples([
                ["examples/cat.jpg", "Where is the cat?"],
                ["examples/street.jpg", "How many people?"],
                ["examples/sign.jpg", "What does the sign say?"],
            ], inputs=[image_input, query_input])

            submit_btn = gr.Button("Compute Relevance", variant="primary")

        with gr.Column():
            # OUTPUT
            with gr.Tabs():
                with gr.TabItem("Homunculus"):
                    homunculus_viz = gr.Image(label="Selected Patches (Top-200)")

                with gr.TabItem("Heatmaps"):
                    with gr.Row():
                        info_map = gr.Image(label="Information (Propositional)")
                        persp_map = gr.Image(label="Salience (Perspectival)")
                    with gr.Row():
                        partic_map = gr.Image(label="Query Coupling (Participatory)")
                        final_map = gr.Image(label="Balanced Relevance")

                with gr.TabItem("Weights"):
                    weights_viz = gr.Image(label="Learned Balancer Weights")
                    weights_data = gr.JSON(label="Per-Patch Tensions")

                with gr.TabItem("Comparison"):
                    with gr.Row():
                        arr_coc_viz = gr.Image(label="ARR-COC (Query-Aware)")
                        baseline_viz = gr.Image(label="Baseline (Uniform)")
```

---

## 🧪 Testing Modes

### Mode 1: Single Image Deep Dive
- Upload one image
- Try multiple queries
- See how allocation changes
- **Validates**: Query-awareness (H1)

### Mode 2: Gallery View
- Show 4-6 pre-loaded examples
- Each with canonical query
- Click to zoom/interact
- **Validates**: Diversity of behavior

### Mode 3: A/B Comparison
- Two side-by-side visualizations
- User votes: "Which makes more sense?"
- Collect preferences
- **Validates**: Qualitative superiority

### Mode 4: Ablation Explorer
- Checkboxes to enable/disable:
  - [ ] Information scorer
  - [ ] Perspectival scorer
  - [ ] Participatory scorer
  - [ ] Adaptive balancing
- See how each component contributes
- **Validates**: Component importance (H2, H3)

### Mode 5: K-Slider Debug
- Slider: K = 50, 100, 200, 400, 800
- Real-time visualization update
- See when K is "enough"
- **Validates**: Token efficiency (H4)

---

## 🎨 Visualization Ideas

### The Homunculus (Most Important!)

```python
def draw_homunculus(image, selected_indices, grid_size=32):
    """
    Overlay selected/rejected patches on original image.

    Green boxes: Selected (top-K)
    Red dim overlay: Rejected
    """
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw, 'RGBA')

    patch_h = image.height // grid_size
    patch_w = image.width // grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            x0, y0 = j * patch_w, i * patch_h
            x1, y1 = x0 + patch_w, y0 + patch_h

            if idx in selected_indices:
                # Selected: Green border
                draw.rectangle([x0, y0, x1, y1], outline='lime', width=2)
            else:
                # Rejected: Red semi-transparent overlay
                draw.rectangle([x0, y0, x1, y1], fill=(255, 0, 0, 80))

    return img_draw
```

### Heatmap Visualization

```python
def draw_heatmap(scores, image_shape):
    """
    Convert 32x32 relevance scores to smooth heatmap overlay.
    """
    # Resize scores to image size (bilinear interpolation)
    heatmap = cv2.resize(scores, (image_shape[1], image_shape[0]))

    # Normalize to 0-1
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # Apply colormap (hot)
    heatmap_colored = plt.cm.hot(heatmap)[:, :, :3]  # RGB

    return (heatmap_colored * 255).astype(np.uint8)
```

### Weight Visualization

```python
def visualize_tensions(tensions, grid_size=32):
    """
    Show learned weights as RGB channels.

    R: Compress/Particularize tension
    G: Exploit/Explore tension
    B: Focus/Diversify tension
    """
    # tensions: [32, 32, 3]
    # Normalize each channel to 0-1
    normalized = (tensions - tensions.min(axis=(0,1))) / (tensions.max(axis=(0,1)) - tensions.min(axis=(0,1)))

    return (normalized * 255).astype(np.uint8)
```

---

## 🔬 Quantitative Metrics to Display

Even though it's qualitative, show some numbers:

```python
def compute_metrics(selected_indices, scores):
    """Display alongside visualizations."""
    return {
        "num_selected": len(selected_indices),
        "avg_relevance": scores[selected_indices].mean(),
        "max_relevance": scores.max(),
        "selectivity": scores.std(),  # Higher = more concentrated
        "coverage": f"{len(selected_indices) / len(scores) * 100:.1f}%"
    }
```

Display in sidebar:
```
✓ Selected: 200 / 1024 patches (19.5%)
✓ Avg Relevance: 0.73
✓ Selectivity: 0.42 (concentrated)
```

---

## 💾 Data Collection Ideas

### User Feedback
```python
with gr.Row():
    feedback = gr.Radio(
        ["Makes sense ✓", "Unclear ~", "Wrong ✗"],
        label="Does this allocation make sense for the query?"
    )
    submit_feedback = gr.Button("Submit Feedback")

# Log to CSV or database
def log_feedback(image_id, query, allocation, user_rating):
    # Append to feedback.csv
    ...
```

### Export for Analysis
```python
export_btn = gr.Button("Export Results (JSON)")

def export_results(image, query, scores, selected_indices):
    return {
        "query": query,
        "image_hash": hash(image.tobytes()),
        "scores": scores.tolist(),
        "selected": selected_indices.tolist(),
        "timestamp": datetime.now().isoformat()
    }
```

---

## 🚀 Deployment Tiers

### Tier 1: MVP Space (CPU-only, public)
- **Purpose**: Demo the concept
- **Features**:
  - Homunculus visualization
  - Basic heatmap
  - Example queries
- **Limitations**:
  - No real VLM inference (just relevance viz)
  - CPU-friendly operations only
  - Mock participatory scorer (no learned weights)

### Tier 2: Dev Space (GPU, private)
- **Purpose**: Internal testing with real model
- **Features**:
  - Full pipeline (Qwen + ARR-COC)
  - All visualizations
  - Checkpoint comparison
- **Limitations**:
  - Slow (no optimization)
  - Private access only

### Tier 3: Research Space (A10G, private)
- **Purpose**: Serious validation
- **Features**:
  - Fast inference
  - Batch processing
  - Data collection
  - Statistical analysis
- **Cost**: $1-3/hour

---

## 🎓 Educational Features

Make it explain itself:

```python
with gr.Accordion("What am I seeing?", open=False):
    gr.Markdown("""
    **Homunculus**: Shows which patches the system selected (green) vs rejected (red).
    Query-aware systems should select patches relevant to your question!

    **Information**: High-entropy regions (edges, text, complexity).
    **Salience**: Visually prominent regions (contrast, edges).
    **Query Coupling**: Regions that match the query semantics.

    **Balanced**: Adaptive combination of all three ways of knowing.
    """)
```

---

## 📋 Testing Workflow

### Day 1: Build Visualizations
- Implement homunculus
- Implement heatmaps
- Test with mock data

### Day 2: Integrate ARR-COC
- Connect texture.py, knowing.py
- Real relevance computation
- No VLM yet (just visualization)

### Day 3: Add Interactions
- Query input
- Example gallery
- Ablation toggles

### Day 4: Deploy & Test
- Push to HF Space
- Test on diverse images
- Collect initial feedback

### Day 5: Iterate
- Fix bugs
- Improve visualizations
- Add requested features

---

## 🐛 Debug Features

Hidden debug panel (for development):

```python
with gr.Accordion("🔧 Debug", open=False):
    show_raw_scores = gr.Checkbox(label="Show raw score arrays")
    show_timings = gr.Checkbox(label="Show computation times")
    force_cpu = gr.Checkbox(label="Force CPU (test Space compatibility)")

    debug_output = gr.JSON(label="Debug Info")
```

---

## 🎯 Success Criteria

After 1 week of testing, we should know:

**✅ Success if**:
- Users can clearly see query-aware behavior
- Homunculus shows semantic selectivity
- Three ways contribute meaningfully
- System feels responsive (<5s per query)

**⚠️ Needs work if**:
- Visualizations are confusing
- No clear difference from baseline
- Too slow for interaction
- Users report "doesn't make sense"

**❌ Pivot if**:
- Relevance scores are random
- No query-awareness visible
- Homunculus looks like noise
- Fundamental hypothesis rejected

---

## 💡 Future Ideas (Post-MVP)

- [ ] Video input (temporal relevance)
- [ ] Attention flow animation
- [ ] 3D relevance visualization
- [ ] Multi-query comparison
- [ ] Batch processing UI
- [ ] Automated benchmark suite
- [ ] Integration with VQAv2 eval
- [ ] Active learning: "Label this confusing case"

---

## 🔗 Integration with spacecheck.py

Test deployment automatically:

```bash
# Deploy new version
git push hf main

# Wait for build
sleep 120

# Check if it works
python spacecheck.py NorthHead/arr-coc-0-1

# If running, open browser
if [ $? -eq 0 ]; then
    open https://huggingface.co/spaces/NorthHead/arr-coc-0-1
fi
```

---

## 📝 Documentation for Users

Add to Space README:

```markdown
## How to Use

1. Upload an image OR select an example
2. Enter a query like "Where is the cat?"
3. Click "Compute Relevance"
4. Explore the visualizations:
   - **Homunculus**: Which patches were selected?
   - **Heatmaps**: Why were they selected?
   - **Weights**: How did the system balance the three ways?

## What You're Seeing

This is a research prototype testing **Vervaekean Relevance Realization**
for vision-language models. The system learns to allocate visual tokens
based on query relevance, using three complementary "ways of knowing":

- Propositional (information content)
- Perspectival (visual salience)
- Participatory (query-content coupling)

Your feedback helps us validate whether this approach works!
```

---

## 🎬 Demo Script

For showing to colleagues:

```
# Example 1: Object localization
Image: Street scene with car and people
Query: "Where is the red car?"
Expected: Homunculus highlights car region

# Example 2: Counting
Image: Group photo
Query: "How many people are wearing hats?"
Expected: Homunculus highlights all heads

# Example 3: Spatial reasoning
Image: Kitchen
Query: "What is on the left side?"
Expected: Homunculus concentrates on left half

# Example 4: OCR
Image: Street sign
Query: "What does the sign say?"
Expected: Homunculus focuses on sign text

# Example 5: Ablation
Same image/query, toggle off "Participatory"
Expected: Less query-aware, more uniform
```

---

**Bottom Line**: Build the interface that lets us SEE relevance realization in action.

Qualitative validation first, quantitative later.

If we can't see it working in Gradio, it's not going to work at scale.

Let's build it! 🚀
