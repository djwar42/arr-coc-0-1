# HFApp - HuggingFace Gradio Demo

**ARR-COC 0.1 Interactive Visualization Interface**

Deployed at: https://huggingface.co/spaces/NorthHead/arr-coc-0-1

---

## What This Is

Interactive Gradio interface for exploring ARR-COC (Adaptive Relevance Realization - Contexts Optical Compression). Visualize how the system processes images through Vervaekean relevance realization.

---

## Running Locally

```bash
# From project root
python HFApp/app.py

# Opens at http://localhost:7860
```

---

## Interface Tabs

### 🎯 Demo
Main interface - upload image, enter query, see relevance realization in action.

### 🔬 Ablations
Component testing - isolate and test individual ARR_COC modules.

### 📈 Analysis
Deep dive on checkpoints - analyze trained model behavior.

### ⚖️ Compare
A/B testing - compare different checkpoints side-by-side.

---

## ARR_COC Components Used

The app imports and visualizes these core modules from `ARR_COC/`:

```python
from ARR_COC import (
    # Texture Processing
    generate_texture_array,      # 13-channel texture extraction

    # Three Ways of Knowing
    information_score,           # Shannon entropy (propositional)
    perspectival_score,          # Edge/gradient salience
    ParticipatoryScorer,         # Query-relevance coupling

    # Balance & Allocation
    AdaptiveTensionBalancer,     # Opponent processing
    TokenAllocator,              # K=200 patch selection

    # Full Integration
    ARRCOCQwen                   # ARR-COC + Qwen3-VL model
)
```

### Pipeline Flow

```
Image + Query
    ↓
[TEXTURE] → 13 channels (RGB, LAB, Sobel, spatial, eccentricity)
    ↓
[KNOWING] → 3 scorers (information, perspectival, participatory)
    ↓
[BALANCING] → Opponent processing (compress ↔ particularize)
    ↓
[ATTENDING] → Token allocation (K=200 patches selected)
    ↓
Visualization (homunculus, heatmaps, metrics)
```

---

## Microscope Visualizations

The `microscope/` folder contains visualization tools:

| Component | Purpose |
|-----------|---------|
| `0-homunculus/` | Patch selection overlay (green=selected, red=rejected) |
| `1-heatmaps/` | Relevance score heatmaps with colormaps |
| `2-textures/` | 13-channel inspection, false color, semantic groups |
| `3-three-ways/` | Vervaekean scorer breakdown |
| `4-comparison/` | Side-by-side model comparison |
| `5-metrics/` | Summary statistics |

See `microscope/README.md` for complete documentation.

---

## Training with CLI

To train and deploy new model checkpoints, use the CLI tools:

```bash
# From project root

# 1. Setup infrastructure (one-time)
python CLI/cli.py setup

# 2. Launch training job
python CLI/cli.py launch

# 3. Monitor progress
python CLI/cli.py monitor

# 4. Teardown (when done)
python CLI/cli.py teardown
```

### Interactive TUI

```bash
python CLI/tui.py
```

Full-screen terminal UI with:
- Real-time job monitoring
- W&B run tracking
- Infrastructure status
- One-key navigation

---

## File Structure

```
HFApp/
├── app.py              # Main Gradio interface (4 tabs)
├── art.py              # 3D spinner + ASCII art header
├── assets/
│   └── prompts/        # Example prompt images
└── microscope/         # Visualization toolkit
    ├── 0-homunculus/   # Patch selection viz
    ├── 1-heatmaps/     # Relevance overlays
    ├── 2-textures/     # 13-channel inspection
    ├── 3-three-ways/   # Scorer breakdown
    ├── 4-comparison/   # A/B testing
    └── 5-metrics/      # Summary stats
```

---

## HuggingFace Spaces Deployment

The app is configured for HF Spaces via root `README.md` frontmatter:

```yaml
app_file: HFApp/app.py
sdk: gradio
sdk_version: 5.49.1
```

Push to HuggingFace to deploy automatically.

---

## Dependencies

Key packages (see `requirements.txt`):
- `gradio` - Web interface
- `torch`, `torchvision` - Deep learning
- `transformers` - Qwen3-VL
- `matplotlib` - Visualizations
- `opencv-python` - Image processing
- `spaces` - HF Spaces GPU decorator

---

## Development Philosophy

**Gradio as Development Microscope** (Part 39)

The interface isn't just a demo - it's the primary development tool. Build visualizations FIRST (before training) to see what the system is doing. Fast feedback loop for same-day iteration.

**Vervaekean Framework**

Visualize the three ways of knowing:
- **Propositional** (information) - What patterns exist?
- **Perspectival** (salience) - What stands out?
- **Participatory** (relevance) - What matters for this query?

---

## Related

- `ARR_COC/` - Core modules being visualized
- `CLI/` - Training infrastructure commands
- `ARR_COC/Training/` - Training script and data
- `Stack/` - Docker images for cloud training

---

*"The microscope is operational. Time to look through it and see what relevance realization looks like."* 🔬
