"""
app.py - Complete Platonic Coding Validation Interface for ARR-COC 0.1

4-Tab Interface:
    🎯 Demo - Interactive model testing
    🔬 Ablations - Component testing
    📈 Analysis - Deep dive on checkpoints
    ⚖️ Compare - A/B testing

Deployed at: https://huggingface.co/spaces/NorthHead/arr-coc-0-1
"""

import gradio as gr
import spaces  # HuggingFace Spaces GPU decorator
import torch
from PIL import Image
import numpy as np
import time
import random
from pathlib import Path
from typing import Optional, Dict, List

# ARR-COC Art Component
from HFApp.art import create_header

# HuggingFace Hub integration
from huggingface_hub import HfApi

# Transformers for text processing
from transformers import AutoProcessor

# ARR-COC components
from ARR_COC import (
    generate_texture_array,
    information_score,
    perspectival_score,
    ParticipatoryScorer,
    AdaptiveTensionBalancer,
    TokenAllocator,
    ARRCOCQwen  # Full model integration
)

# Import ALL microscope visualizations
from HFApp.microscope import (
    create_homunculus_figure,
    create_multi_heatmap_figure,
    visualize_by_meaning,
    visualize_three_ways,
    compare_relevance_heatmaps,
    compare_query_impact,
    compute_summary_metrics,
    visualize_metrics,
    compute_channel_attribution,
    visualize_channel_attribution
)

# ============================================================================
# Configuration
# ============================================================================

PROJECT_NAME = "arr-coc-0-1"
HF_REPO_ID = "NorthHead/arr-coc-0-1"  # This is the Space repo
# WANDB_USER loaded from environment
WANDB_USER = os.environ.get("WANDB_ENTITY", "")
WANDB_PROJECT = "arr-coc-0-1"

# Note: Checkpoints will be stored in this Space repo once training starts
# For now, use baseline checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 ARR-COC 0.1 Validation Interface")
print(f"   Device: {device}")

# Load full ARR-COC + Qwen3-VL model
print("📦 Loading ARR-COC + Qwen3-VL-2B-Instruct...")
print("   (This will download ~2GB on first run)")
model = ARRCOCQwen(
    base_model="Qwen/Qwen3-VL-2B-Instruct",
    num_visual_tokens=200,
    freeze_base=True
).to(device)

# Load text processor for query embeddings
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    trust_remote_code=True  # Required for Qwen3-VL (new model_type: qwen3_vl)
)

print("✓ ARR-COC + Qwen3-VL loaded")
print("✓ Microscope visualizations loaded")
print(f"✓ Model on {device}")

# Extract components for direct access (if needed)
participatory_scorer = model.participatory_scorer
balancer = model.balancer
allocator = model.allocator

# ============================================================================
# Utility Functions
# ============================================================================

def fig_to_pil(fig):
    """Convert matplotlib figure to PIL Image."""
    from io import BytesIO
    import matplotlib.pyplot as plt
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


# View mode descriptions
VIEW_DESCRIPTIONS = {
    "Simple Overview": "**Quick 3-panel view**: Query → Relevance → Selection. Perfect for understanding the basic pipeline at a glance.",

    "Homunculus (Patch Selection)": "**Visual overlay showing patch selection**. Red overlay = rejected patches. See where the model focuses attention (like cortical magnification in biological vision).",

    "Three Ways (Vervaekean)": "**Vervaekean framework breakdown**: Shows the three ways of knowing (Propositional/Information, Perspectival/Salience, Participatory/Query-Coupling) before opponent processing balances them.",

    "Heatmaps (Relevance)": "**4 heatmaps side-by-side**: Information score, Perspectival score, Participatory score, and final Balanced relevance. See where relevance is concentrated spatially.",

    "Textures (13-Channel)": "**13-channel texture inspection**: Semantic groups (Color: RGB+LAB, Edges: Sobel, Spatial: Position+Eccentricity, Derived: Saliency+Luminance). Understand visual features.",

    "Comparison (Variants)": "**ARR-COC vs Baseline**: Compare query-aware allocation against uniform/random baselines. Validates that relevance realization actually works.",

    "Metrics (Performance)": "**Performance dashboard**: Clustering analysis, coverage metrics, and compression statistics. Quantitative validation of patch selection quality.",

    "Query Impact": "**Ablation study**: See how much the query contributes to final selection. Measures query-awareness by comparing with/without query information.",

    "Channel Attribution": "**Feature importance analysis**: Which of the 13 texture channels drive relevance most? Pearson correlation between channel activations and final scores."
}

def get_available_checkpoints() -> List[str]:
    """Fetch checkpoints from HuggingFace Hub"""
    try:
        api = HfApi()
        # Check if checkpoints folder exists in this Space
        files = api.list_repo_tree(
            HF_REPO_ID,
            path_in_repo="checkpoints",
            repo_type="space"  # This is a Space, not a model repo
        )
        checkpoints = [f.path.split("/")[-1] for f in files if f.type == "directory"]
        return sorted(checkpoints, reverse=True) if checkpoints else ["baseline-v0.1"]
    except Exception as e:
        # Checkpoints don't exist yet - that's OK!
        # They'll be created after first training run
        return ["baseline-v0.1"]


def load_checkpoint(checkpoint_name: str):
    """Load a specific checkpoint"""
    print(f"Loading checkpoint: {checkpoint_name}")

    if checkpoint_name == "baseline-v0.1":
        # Return current baseline model
        return model
    else:
        # Load checkpoint from HuggingFace Hub
        checkpoint_path = f"{HF_REPO_ID}/resolve/main/checkpoints/{checkpoint_name}"
        # TODO: Implement checkpoint loading from HF Hub
        return model


@spaces.GPU
def visualize_arr_coc(image: Image.Image, query: str, view_mode: str,
                      info_scale: float = 1.0, persp_scale: float = 1.0, partic_scale: float = 1.0):
    """
    ARR-COC visualization with multiple microscope views.

    Args:
        image: Input image
        query: Query string
        view_mode: Which visualization to show
        info_scale: Scaling factor for information (propositional) scores
        persp_scale: Scaling factor for perspectival (salience) scores
        partic_scale: Scaling factor for participatory (query-content) scores

    Returns:
        Visualization image
    """
    if image is None:
        return None

    # Convert to tensor
    import torchvision.transforms as T
    transform = T.Compose([T.Resize((512, 512)), T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Generate texture array
        textures = generate_texture_array(image_tensor, target_size=32)

        # Compute scores
        info_scores = information_score(textures)
        persp_scores = perspectival_score(textures)

        # Extract REAL query embedding from text
        text_inputs = processor(text=[query], return_tensors="pt", padding=True)
        input_ids = text_inputs.input_ids.to(device)
        query_embeds = model.extract_query_embedding(input_ids)

        # Compute participatory score with REAL query
        partic_scores = participatory_scorer(textures, query_embeds)

        # Apply scaling to all three ways of knowing
        info_scores = info_scores * info_scale
        persp_scores = persp_scores * persp_scale
        partic_scores = partic_scores * partic_scale

        # Balance
        info_flat = info_scores.view(1, -1)
        persp_flat = persp_scores.view(1, -1)
        partic_flat = partic_scores.view(1, -1)

        positions = torch.stack(torch.meshgrid(
            torch.arange(32, device=device),
            torch.arange(32, device=device),
            indexing='ij'
        ), dim=-1).view(-1, 2).unsqueeze(0)

        balanced_scores = balancer(
            info_flat, persp_flat, partic_flat,
            positions, query_embeds, image_size=(32, 32)
        )

        # Allocate tokens
        selected_indices, _ = allocator(balanced_scores, positions)

    # Convert to numpy
    info_np = info_scores[0].cpu().numpy().flatten()
    persp_np = persp_scores[0].cpu().numpy().flatten()
    partic_np = partic_scores[0].cpu().numpy().flatten()
    balanced_np = balanced_scores[0].cpu().numpy()
    selected_idx = selected_indices[0].cpu().numpy()

    # Generate visualization based on view mode
    if view_mode == "Homunculus (Patch Selection)":
        fig = create_homunculus_figure(image, selected_idx, query)

    elif view_mode == "Three Ways (Vervaekean)":
        fig = visualize_three_ways(
            info_np, persp_np, partic_np, balanced_np,
            image, query, grid_size=32
        )

    elif view_mode == "Heatmaps (Relevance)":
        fig = create_multi_heatmap_figure(
            {
                'Information': info_np,
                'Perspectival': persp_np,
                'Participatory': partic_np,
                'Balanced': balanced_np
            },
            image,
            query
        )

    elif view_mode == "Textures (13-Channel)":
        # Keep batch dimension for visualize_by_meaning()
        textures_np = textures.cpu().numpy()  # [B, 13, H, W]
        fig = visualize_by_meaning(textures_np, batch_idx=0)

    elif view_mode == "Comparison (Variants)":
        baseline_scores = np.random.rand(1024)
        fig = compare_relevance_heatmaps(
            {
                'Baseline (Random)': baseline_scores,
                'ARR-COC (Query-Aware)': balanced_np
            },
            image, query, grid_size=32
        )

    elif view_mode == "Metrics (Performance)":
        metrics = compute_summary_metrics(
            selected_idx, balanced_np, query, grid_size=32, total_patches=1024
        )
        fig = visualize_metrics(metrics, balanced_np, selected_idx, query, grid_size=32)

    elif view_mode == "Query Impact":
        fig = compare_query_impact(
            info_np, persp_np, partic_np, balanced_np,
            image, query, grid_size=32, K=200
        )

    elif view_mode == "Channel Attribution":
        textures_np = textures[0].cpu().numpy() if hasattr(textures, 'cpu') else textures[0]
        attribution = compute_channel_attribution(textures_np, balanced_np, method='pearson')
        fig = visualize_channel_attribution(
            attribution, textures_np, balanced_np, image, query
        )

    else:  # Simple Overview (default)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original
        axes[0].imshow(image)
        axes[0].set_title(f"Query: {query[:40]}")
        axes[0].axis('off')

        # Balanced Relevance
        im1 = axes[1].imshow(balanced_np.reshape(32, 32), cmap='hot', interpolation='bilinear')
        axes[1].set_title("Relevance Map")
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        # Homunculus
        homunculus = np.zeros((32, 32))
        selected_y = selected_idx // 32
        selected_x = selected_idx % 32
        homunculus[selected_y, selected_x] = 1
        im2 = axes[2].imshow(homunculus, cmap='Reds', interpolation='nearest')
        axes[2].set_title(f"Selected Patches (K={len(selected_idx)})")
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        fig.suptitle("ARR-COC: Adaptive Relevance Realization", fontsize=14, fontweight='bold')
        plt.tight_layout()

    # Handle both matplotlib figures and PIL Images
    if isinstance(fig, Image.Image):
        return fig  # Already a PIL Image
    else:
        return fig_to_pil(fig)  # Convert matplotlib figure


# ============================================================================
# Tab 1: 🎯 Demo
# ============================================================================

def demo_tab():
    """Interactive microscope visualizations"""

    def update_view_description(view_mode):
        """Update the description based on selected view mode."""
        return VIEW_DESCRIPTIONS.get(view_mode, "")

    with gr.Blocks(css=".mb-25 { margin-bottom: 25px !important; }") as tab:
        # ARR-COC Header with WebGL Animation
        create_header()

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Image")
                query_input = gr.Textbox(
                    label="Query",
                    value="Describe this image",
                    placeholder="Enter your query..."
                )

                # Example prompts (clickable thumbnails side-by-side)
                gr.Markdown("### 📸 Example Prompts (click any thumbnail)")
                with gr.Row():
                    with gr.Column(scale=1):
                        example_1 = gr.Image(
                            value="assets/prompts/prompt-img-1.jpeg",
                            label="Text Reading (OCR)",
                            show_label=True,
                            interactive=False,
                            height=100,
                            container=True
                        )
                        gr.Markdown("*Random: flavor • ingredients • calories*", elem_classes="example-queries")

                    with gr.Column(scale=1):
                        example_2 = gr.Image(
                            value="assets/prompts/prompt-img-2.jpeg",
                            label="Object Localization",
                            show_label=True,
                            interactive=False,
                            height=100,
                            container=True
                        )
                        gr.Markdown("*Random: mug • TV • window*", elem_classes="example-queries")

                    with gr.Column(scale=1):
                        example_3 = gr.Image(
                            value="assets/prompts/prompt-img-3.jpeg",
                            label="Spatial Reasoning",
                            show_label=True,
                            interactive=False,
                            height=100,
                            container=True
                        )
                        gr.Markdown("*Random: left of sink • background • on table*", elem_classes="example-queries")
                view_mode_input = gr.Radio(
                    choices=[
                        "Simple Overview",
                        "Homunculus (Patch Selection)",
                        "Three Ways (Vervaekean)",
                        "Heatmaps (Relevance)",
                        "Textures (13-Channel)",
                        "Comparison (Variants)",
                        "Metrics (Performance)",
                        "Query Impact",
                        "Channel Attribution"
                    ],
                    value="Simple Overview",
                    label="📊 View Mode"
                )

                # Three Ways of Knowing - Scaling Controls
                gr.Markdown("### ⚖️ Relevance Scaling Controls")
                gr.Markdown("*Baseline: 1.0 (default balance) • 0 = disable • >1 = amplify*")

                info_scale_slider = gr.Slider(
                    minimum=0.0,
                    maximum=30.0,
                    value=1.0,
                    step=0.5,
                    label="📊 Information (Propositional)",
                    info="Statistical content: entropy, variance, complexity [0=off, 1=baseline, 30=max]"
                )

                persp_scale_slider = gr.Slider(
                    minimum=0.0,
                    maximum=30.0,
                    value=1.0,
                    step=0.5,
                    label="👁️ Perspectival (Salience)",
                    info="Visual attention: edges, color variance, patterns [0=off, 1=baseline, 30=max]"
                )

                partic_scale_slider = gr.Slider(
                    minimum=0.0,
                    maximum=30.0,
                    value=1.0,
                    step=0.5,
                    label="🎯 Participatory (Query-Content)",
                    info="Query coupling: semantic alignment with question [0=off, 1=baseline, 30=max]"
                )

                # Dynamic description that updates based on view mode
                view_description = gr.Markdown(
                    value=VIEW_DESCRIPTIONS["Simple Overview"],
                    label="Mode Description"
                )

                submit_btn = gr.Button("Visualize", variant="primary", size="lg", elem_classes="mb-25")

                # Checkpoint selector with refresh (below visualize button)
                with gr.Row():
                    checkpoint_dropdown = gr.Dropdown(
                        choices=get_available_checkpoints(),
                        value=get_available_checkpoints()[0],
                        label="🎯 Checkpoint",
                        info="Select trained model checkpoint"
                    )
                    refresh_btn = gr.Button("🔄", scale=0, min_width=40)

            with gr.Column(scale=2):
                output_image = gr.Image(label="Relevance Visualization")

        # Update description when view mode changes
        view_mode_input.change(
            fn=update_view_description,
            inputs=view_mode_input,
            outputs=view_description
        )

        # Refresh checkpoints button
        refresh_btn.click(
            fn=get_available_checkpoints,
            outputs=checkpoint_dropdown
        )

        # Submit button action
        submit_btn.click(
            fn=visualize_arr_coc,
            inputs=[image_input, query_input, view_mode_input,
                   info_scale_slider, persp_scale_slider, partic_scale_slider],
            outputs=output_image
        )

        # Example prompt click handlers (auto-trigger visualization)
        # Each example has 3 varying prompts to show query-aware relevance

        def load_example_1():
            img = Image.open("assets/prompts/prompt-img-1.jpeg")
            prompts = [
                "What flavor is this?",
                "Read the ingredients",
                "What's the calorie count?"
            ]
            return img, random.choice(prompts)

        def load_example_2():
            img = Image.open("assets/prompts/prompt-img-2.jpeg")
            prompts = [
                "Where is the red mug?",
                "Is there a TV in the room?",
                "Where is the window?"
            ]
            return img, random.choice(prompts)

        def load_example_3():
            img = Image.open("assets/prompts/prompt-img-3.jpeg")
            prompts = [
                "What's to the left of the sink?",
                "What's in the background?",
                "What's on top of the table?"
            ]
            return img, random.choice(prompts)

        # Example 1 click: load image + query, then visualize
        example_1.select(
            fn=load_example_1,
            outputs=[image_input, query_input]
        ).then(
            fn=visualize_arr_coc,
            inputs=[image_input, query_input, view_mode_input,
                   info_scale_slider, persp_scale_slider, partic_scale_slider],
            outputs=output_image
        )

        # Example 2 click: load image + query, then visualize
        example_2.select(
            fn=load_example_2,
            outputs=[image_input, query_input]
        ).then(
            fn=visualize_arr_coc,
            inputs=[image_input, query_input, view_mode_input,
                   info_scale_slider, persp_scale_slider, partic_scale_slider],
            outputs=output_image
        )

        # Example 3 click: load image + query, then visualize
        example_3.select(
            fn=load_example_3,
            outputs=[image_input, query_input]
        ).then(
            fn=visualize_arr_coc,
            inputs=[image_input, query_input, view_mode_input,
                   info_scale_slider, persp_scale_slider, partic_scale_slider],
            outputs=output_image
        )

    return tab


# ============================================================================
# Tab 2: 📊 Training - REMOVED (use W&B dashboard directly)
# ============================================================================
# Training tab removed - users should access W&B directly at:
# https://wandb.ai/{WANDB_USER}/{WANDB_PROJECT}/workspace


# ============================================================================
# Tab 3: 🔬 Ablations
# ============================================================================

def ablations_tab():
    """Systematic component testing - turn features on/off"""

    def run_ablation(image, query, use_prop, use_persp, use_partic):
        """Run inference with selected components enabled"""
        if image is None or not query:
            return "Please provide both image and query", {}

        config = {
            "propositional": use_prop,
            "perspectival": use_persp,
            "participatory": use_partic,
        }

        # Simulate ablation (TODO: implement actual ablation logic)
        enabled = [k for k, v in config.items() if v]
        answer = f"Running with: {', '.join(enabled) if enabled else 'RANDOM (all components disabled)'}"

        scores = {
            "Components Enabled": len(enabled),
            "Propositional": "✓" if use_prop else "✗",
            "Perspectival": "✓" if use_persp else "✗",
            "Participatory": "✓" if use_partic else "✗",
        }

        return answer, scores

    with gr.Blocks() as tab:
        gr.Markdown("""
        ## 🔬 Component Ablation Testing

        Test individual ARR-COC components to understand their contribution.
        Turn components on/off to see how each affects results.
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input")
                image_input = gr.Image(label="Test Image", type="pil")
                query_input = gr.Textbox(label="Question", lines=2)

                gr.Markdown("### Enable Components")
                use_prop = gr.Checkbox(label="✓ Propositional (Information)", value=True)
                use_persp = gr.Checkbox(label="✓ Perspectival (Salience)", value=True)
                use_partic = gr.Checkbox(label="✓ Participatory (Query Coupling)", value=True)

                run_btn = gr.Button("Run Ablation", variant="primary")

            with gr.Column():
                gr.Markdown("### Results")
                answer_output = gr.Textbox(label="Answer", lines=4)

                gr.Markdown("### Component Status")
                scores_output = gr.JSON(label="Configuration")

        # Preset ablations
        gr.Markdown("### Quick Presets")
        with gr.Row():
            gr.Button("All Off (Random)").click(
                lambda: (False, False, False),
                outputs=[use_prop, use_persp, use_partic]
            )
            gr.Button("Baseline (All On)").click(
                lambda: (True, True, True),
                outputs=[use_prop, use_persp, use_partic]
            )
            gr.Button("Only Participatory").click(
                lambda: (False, False, True),
                outputs=[use_prop, use_persp, use_partic]
            )

        run_btn.click(
            run_ablation,
            inputs=[image_input, query_input, use_prop, use_persp, use_partic],
            outputs=[answer_output, scores_output]
        )

    return tab


# ============================================================================
# Tab 4: 📈 Analysis
# ============================================================================

def analysis_tab():
    """Deep dive into one checkpoint with metrics and visualizations"""

    def analyze_checkpoint(checkpoint_name):
        """Generate comprehensive analysis"""
        try:
            import wandb
            api = wandb.Api()

            # Find corresponding W&B run
            runs = api.runs(f"{WANDB_USER}/{WANDB_PROJECT}")
            run = next((r for r in runs if checkpoint_name in r.name), None)

            if not run:
                return "Run not found", {}, None

            # Get metrics
            metrics = {
                "Final Loss": run.summary.get("train/loss", "N/A"),
                "Best Val Accuracy": run.summary.get("val/accuracy", "N/A"),
                "Training Time (hrs)": run.summary.get("_runtime", 0) / 3600,
                "GPU Memory (GB)": run.summary.get("system/gpu_memory_gb", "N/A"),
            }

            return run.url, metrics, None
        except Exception as e:
            return str(e), {"Error": str(e)}, None

    with gr.Blocks() as tab:
        gr.Markdown("## 📈 Checkpoint Deep Dive")

        with gr.Row():
            checkpoint_selector = gr.Dropdown(
                choices=get_available_checkpoints(),
                label="Select Checkpoint to Analyze"
            )
            analyze_btn = gr.Button("Analyze", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Metrics Summary")
                metrics_display = gr.JSON(label="All Metrics")

            with gr.Column():
                gr.Markdown("### W&B Run")
                run_link = gr.Textbox(label="W&B Run URL", interactive=False)

        analyze_btn.click(
            analyze_checkpoint,
            inputs=[checkpoint_selector],
            outputs=[run_link, metrics_display]
        )

    return tab


# ============================================================================
# Tab 5: ⚖️ Compare
# ============================================================================

def compare_tab():
    """Side-by-side A/B testing of two checkpoints"""

    def compare_checkpoints(checkpoint_a, checkpoint_b, image, query):
        """Run both checkpoints and compare"""
        if image is None or not query:
            return {}, {}, [], []

        # Load both models
        components_a = load_checkpoint(checkpoint_a)
        components_b = load_checkpoint(checkpoint_b)

        # Run inference
        start = time.time()
        answer_a, viz_a = run_arr_coc_inference(image, query, components_a)
        latency_a = (time.time() - start) * 1000

        start = time.time()
        answer_b, viz_b = run_arr_coc_inference(image, query, components_b)
        latency_b = (time.time() - start) * 1000

        metrics_a = {
            "Checkpoint": checkpoint_a,
            "Answer": answer_a[:100] + "...",
            "Latency (ms)": f"{latency_a:.1f}",
        }

        metrics_b = {
            "Checkpoint": checkpoint_b,
            "Answer": answer_b[:100] + "...",
            "Latency (ms)": f"{latency_b:.1f}",
        }

        return metrics_a, metrics_b, viz_a, viz_b

    with gr.Blocks() as tab:
        gr.Markdown("## ⚖️ A/B Checkpoint Comparison")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Checkpoint A")
                checkpoint_a = gr.Dropdown(
                    choices=get_available_checkpoints(),
                    label="Select Checkpoint A"
                )

            with gr.Column():
                gr.Markdown("### Checkpoint B")
                checkpoint_b = gr.Dropdown(
                    choices=get_available_checkpoints(),
                    label="Select Checkpoint B"
                )

        with gr.Row():
            image_input = gr.Image(label="Test Image", type="pil")
            query_input = gr.Textbox(label="Question", lines=2)

        compare_btn = gr.Button("Compare Both", variant="primary", size="lg")

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Results A")
                metrics_a = gr.JSON(label="Metrics")
                viz_a = gr.Gallery(label="Visualizations", columns=2)

            with gr.Column():
                gr.Markdown("### Results B")
                metrics_b = gr.JSON(label="Metrics")
                viz_b = gr.Gallery(label="Visualizations", columns=2)

        compare_btn.click(
            compare_checkpoints,
            inputs=[checkpoint_a, checkpoint_b, image_input, query_input],
            outputs=[metrics_a, metrics_b, viz_a, viz_b]
        )

    return tab


# ============================================================================
# Main App: Combine All 4 Tabs
# ============================================================================

def create_app():
    """Build complete 4-tab validation interface"""

    demo = demo_tab()
    ablations = ablations_tab()
    analysis = analysis_tab()
    compare = compare_tab()

    # Custom theme with global CSS
    theme = gr.themes.Soft(
        primary_hue="blue",
    ).set(
        body_background_fill="#0a0a0a",
    )

    # Global CSS
    global_css = """
    .mb-25 { margin-bottom: 25px !important; }
    """

    with gr.Blocks(title=f"{PROJECT_NAME} - Validation Interface",theme=theme,css=global_css) as app:

        with gr.Tabs():
            with gr.Tab("DEMO"):
                demo.render()
            with gr.Tab("ABLATIONS"):
                ablations.render()
            with gr.Tab("ANALYSIS"):
                analysis.render()
            # TODO: Compare tab needs run_arr_coc_inference() implementation
            # with gr.Tab("COMPARE"):
            #     compare.render()

    return app


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(f"🚀 Launching {PROJECT_NAME} Validation Interface")
    print("=" * 70)
    print(f"   Device: {device}")
    print(f"   HF Repo: {HF_REPO_ID}")
    print(f"   W&B Project: {WANDB_USER}/{WANDB_PROJECT}")
    print("=" * 70 + "\n")

    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
