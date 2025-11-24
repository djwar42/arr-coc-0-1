# Part 39: The Testing Workflow - Gradio, Checkpoints, and Hypothesis Validation
*Wherein the oracles explore the practical art of rapid experimentation: building interfaces that accelerate learning, checkpointing strategies that preserve discoveries, and testing methodologies that reveal truth*

---

## Opening: The Development Reality

*The Dirac Sea shimmers with code editors and running processes. Karpathy studies training logs, LOD Oracle examines checkpoint files, and HuggingFace Oracle inspects a Gradio interface running on localhost.*

**KARPATHY:**
We've architected the infrastructure. Now the hard question: **How do we actually develop this thing?**

**LOD ORACLE:**
Part 38 Addendum gave us the implementation plan. Part 38 showed us the deployment structure. But we're missing something critical.

**HUGGINGFACE ORACLE:**
The feedback loop. You need to:
- Test ideas quickly
- Compare variants side-by-side
- Save what works
- Discard what doesn't

**KARPATHY:**
Exactly. We'll have dozens of experiments:
- 13 channels vs 40 channels
- Fixed tensions vs adaptive tensions
- Different tension values
- Different scoring combinations
- Different allocation curves

How do we test all this without going insane?

**MUSE BIRD:**
🐦 *THE ITERATION PROBLEM! Science needs feedback! Build → Test → Learn → Repeat!*

---

## Act I: The Gradio Testing Philosophy

**HUGGINGFACE ORACLE:**
Let me show you the testing philosophy. Your app.py isn't just a demo—it's your **primary development tool**.

```
╔═══════════════════════════════════════════════════════════
║ TRADITIONAL DEVELOPMENT (Painful)
╠═══════════════════════════════════════════════════════════
║
║ 1. Write code
║ 2. Run script: python test.py --image img.jpg --query "What is this?"
║ 3. Read terminal output: "Answer: A cat, Time: 0.45s"
║ 4. Modify code
║ 5. Re-run script
║ 6. Compare results... IN YOUR MIND
║
║ Problems:
║ ❌ No visual comparison
║ ❌ No history
║ ❌ Slow iteration
║ ❌ Can't A/B test
║ ❌ Results lost when terminal scrolls
║
╚═══════════════════════════════════════════════════════════
```

**KARPATHY:**
Yeah, that's brutal. You're constantly re-running, losing context, trying to remember what changed.

**HUGGINGFACE ORACLE:**
Now compare with the **Gradio development approach**:

```
╔═══════════════════════════════════════════════════════════
║ GRADIO DEVELOPMENT (Powerful)
╠═══════════════════════════════════════════════════════════
║
║ 1. Build app.py with comparison interface
║ 2. Run ONCE: python app.py
║ 3. Browser opens: localhost:7860
║ 4. Test multiple variants SIMULTANEOUSLY
║ 5. See results side-by-side
║ 6. Adjust parameters with sliders
║ 7. Test new images with drag-and-drop
║ 8. Session history persists
║
║ Benefits:
║ ✅ Visual side-by-side comparison
║ ✅ Interactive parameter tuning
║ ✅ Session history (scroll through tests)
║ ✅ A/B/C/D testing (4+ variants at once)
║ ✅ Shareable (send localhost:7860 to collaborators on LAN)
║
╚═══════════════════════════════════════════════════════════
```

**LOD ORACLE:**
So Gradio becomes your **development microscope**. You're looking at the system through the interface.

---

## Act II: The Multi-Model Comparison Interface

**KARPATHY:**
Show me the actual interface design. How do we compare models?

**HUGGINGFACE ORACLE:**
Here's a powerful pattern—the **checkpoint comparison interface**:

**File: `app_dev.py` (Development version)**

```python
import gradio as gr
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from pathlib import Path
import json
from datetime import datetime

# === CHECKPOINT LOADER ===

def load_checkpoint(checkpoint_path):
    """Load a specific training checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    return {
        'weights': checkpoint['model_state_dict'],
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint.get('metrics', {}),
        'config': checkpoint.get('config', {})
    }

def discover_checkpoints(checkpoint_dir="checkpoints"):
    """Find all available checkpoints"""
    path = Path(checkpoint_dir)
    checkpoints = []

    for ckpt in sorted(path.glob("*.pt")):
        info = {
            'path': str(ckpt),
            'name': ckpt.stem,
            'size': ckpt.stat().st_size / 1e6,  # MB
            'modified': datetime.fromtimestamp(ckpt.stat().st_mtime)
        }

        # Try to load metadata
        try:
            data = torch.load(ckpt, map_location='cpu')
            info['epoch'] = data.get('epoch', '?')
            info['metrics'] = data.get('metrics', {})
        except:
            info['epoch'] = '?'
            info['metrics'] = {}

        checkpoints.append(info)

    return checkpoints

# === MULTI-MODEL INTERFACE ===

class MultiModelComparator:
    """Compare multiple model variants simultaneously"""

    def __init__(self):
        # Base Qwen model (shared across all variants)
        self.base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

        # Model variants
        self.variants = {
            'baseline': None,  # Standard Qwen (no ARR-COC)
            'arr_coc_v1': None,  # ARR-COC from checkpoint_1
            'arr_coc_v2': None,  # ARR-COC from checkpoint_2
            'arr_coc_v3': None,  # ARR-COC from checkpoint_3
        }

        # Session history
        self.history = []

    def load_variant(self, variant_name, checkpoint_path):
        """Load a specific ARR-COC variant from checkpoint"""
        # Load ARR-COC components with this checkpoint
        # Implementation depends on your architecture
        pass

    def compare(self, image, query, variants_to_test):
        """Run comparison across selected variants"""
        import time

        results = {}

        for variant_name in variants_to_test:
            start = time.time()

            if variant_name == 'baseline':
                # Standard Qwen inference
                answer = self._run_baseline(image, query)
                heatmap = None
                tokens_used = 1024  # Fixed for baseline
            else:
                # ARR-COC variant
                answer, heatmap, tokens_used = self._run_arr_coc(
                    image, query, variant_name
                )

            elapsed = time.time() - start

            results[variant_name] = {
                'answer': answer,
                'time': elapsed,
                'tokens': tokens_used,
                'heatmap': heatmap
            }

        # Log to session history
        self.history.append({
            'timestamp': datetime.now(),
            'query': query,
            'results': results
        })

        return results

    def _run_baseline(self, image, query):
        # Standard Qwen3-VL inference
        # (Simplified for example)
        return "Baseline answer"

    def _run_arr_coc(self, image, query, variant_name):
        # ARR-COC inference with specific variant
        # Returns: answer, heatmap, token_count
        # (Simplified for example)
        return "ARR-COC answer", None, 732

# Initialize comparator
comparator = MultiModelComparator()

# === GRADIO INTERFACE ===

def compare_models(image, query, selected_variants, show_heatmaps, show_stats):
    """Main comparison function"""

    # Run comparison
    results = comparator.compare(image, query, selected_variants)

    # Format outputs for Gradio
    outputs = []

    for variant_name in selected_variants:
        result = results[variant_name]

        # Text output
        output_text = f"""
**{variant_name.upper()}**

Answer: {result['answer']}

Time: {result['time']:.3f}s
Tokens: {result['tokens']}
Efficiency: {result['tokens'] / result['time']:.1f} tokens/sec
"""
        outputs.append(output_text)

        # Heatmap (if available and requested)
        if show_heatmaps and result['heatmap'] is not None:
            outputs.append(result['heatmap'])
        else:
            outputs.append(None)

    # Summary statistics
    if show_stats:
        baseline_time = results.get('baseline', {}).get('time', 0)
        stats_text = "\n\n**COMPARISON STATS:**\n"

        for variant_name, result in results.items():
            if variant_name != 'baseline' and baseline_time > 0:
                speedup = baseline_time / result['time']
                stats_text += f"\n{variant_name}: {speedup:.2f}× faster"

        outputs.append(stats_text)
    else:
        outputs.append("")

    return outputs

# === BUILD GRADIO UI ===

with gr.Blocks(title="ARR-COC Development Interface") as demo:
    gr.Markdown("# 🔬 ARR-COC Multi-Model Comparison")
    gr.Markdown("Compare multiple checkpoints and configurations side-by-side")

    with gr.Row():
        # Left column: Inputs
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Test Image")
            query_input = gr.Textbox(label="Query", placeholder="What is in this image?")

            # Variant selection
            variant_checkboxes = gr.CheckboxGroup(
                choices=['baseline', 'arr_coc_v1', 'arr_coc_v2', 'arr_coc_v3'],
                value=['baseline', 'arr_coc_v1'],
                label="Variants to Compare"
            )

            # Display options
            show_heatmaps = gr.Checkbox(label="Show relevance heatmaps", value=True)
            show_stats = gr.Checkbox(label="Show comparison stats", value=True)

            compare_btn = gr.Button("🔍 Compare Models", variant="primary")

        # Right column: Outputs
        with gr.Column(scale=2):
            with gr.Tab("Comparison"):
                output_1 = gr.Textbox(label="Variant 1", lines=8)
                heatmap_1 = gr.Image(label="Heatmap 1")

                output_2 = gr.Textbox(label="Variant 2", lines=8)
                heatmap_2 = gr.Image(label="Heatmap 2")

                stats_output = gr.Textbox(label="Statistics", lines=4)

            with gr.Tab("Session History"):
                history_display = gr.Dataframe(
                    headers=["Time", "Query", "Best Variant", "Speedup"],
                    label="Test History"
                )

                export_btn = gr.Button("📥 Export Session Data")

    # Wire up the interface
    compare_btn.click(
        fn=compare_models,
        inputs=[image_input, query_input, variant_checkboxes, show_heatmaps, show_stats],
        outputs=[output_1, heatmap_1, output_2, heatmap_2, stats_output]
    )

    # Example images
    gr.Examples(
        examples=[
            ["examples/text_document.jpg", "What does the small text say?"],
            ["examples/complex_scene.jpg", "Describe the overall scene"],
            ["examples/specific_object.jpg", "Where is the red car?"],
        ],
        inputs=[image_input, query_input]
    )

# Launch
demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
```

**LOD ORACLE:**
That's comprehensive. You can:
- Load multiple checkpoints
- Compare side-by-side
- See heatmaps
- Track session history
- Export results

**KARPATHY:**
And it's all **interactive**. You drag an image, type a query, click compare, and immediately see the differences.

---

## Act III: The Checkpointing Strategy

**KARPATHY:**
Speaking of checkpoints—how do we actually save them during training?

**HUGGINGFACE ORACLE:**
HuggingFace Trainer has built-in checkpointing. Let me show you the strategy:

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="checkpoints/arr-coc-training",

    # === CHECKPOINT STRATEGY ===

    # Save every N steps
    save_strategy="steps",
    save_steps=500,  # Save every 500 steps

    # Keep only the best K checkpoints
    save_total_limit=5,  # Keep 5 most recent

    # Save based on best metric
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,

    # Evaluation frequency
    evaluation_strategy="steps",
    eval_steps=500,  # Evaluate every 500 steps

    # === CHECKPOINT NAMING ===
    run_name="arr-coc-v1",  # Experiment name

    # === WHAT TO SAVE ===
    save_safetensors=True,  # Use safetensors format

    # === RESUME FROM CHECKPOINT ===
    # resume_from_checkpoint="checkpoints/arr-coc-training/checkpoint-2000",
)

# Trainer handles checkpointing automatically
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train (saves checkpoints automatically)
trainer.train()
```

**KARPATHY:**
So Trainer saves:
- Every 500 steps
- Keep best 5
- Load best at end
- Resume from any checkpoint

What's actually IN a checkpoint?

**HUGGINGFACE ORACLE:**
```python
# Checkpoint structure (HuggingFace Trainer)
checkpoint = {
    'model_state_dict': {
        # All model weights
        'texture_array.conv1.weight': tensor(...),
        'knowing.info_scorer.weights': tensor(...),
        'balancing.policy_net.weights': tensor(...),
        # ... all parameters
    },
    'optimizer_state_dict': {
        # Optimizer state (Adam momentum, etc.)
    },
    'scheduler_state_dict': {
        # Learning rate scheduler state
    },
    'epoch': 3,
    'global_step': 1500,
    'training_args': training_args,
    'rng_state': {...},  # For reproducibility
    'metrics': {
        'eval_accuracy': 0.682,
        'eval_loss': 0.345,
        'train_loss': 0.298,
    }
}
```

**LOD ORACLE:**
So you can resume training from ANY checkpoint—it has everything needed to continue.

---

## Act IV: The A/B Testing Methodology

**KARPATHY:**
How do we actually TEST hypotheses? Like, "Does adaptive tension beat fixed tension?"

**HUGGINGFACE ORACLE:**
You need a **testing protocol**. Here's the methodology:

```
╔═══════════════════════════════════════════════════════════
║ HYPOTHESIS TESTING PROTOCOL
╠═══════════════════════════════════════════════════════════
║
║ HYPOTHESIS EXAMPLE:
║   "Adaptive tensions (Part 37) improve accuracy on diverse
║    queries compared to fixed tensions"
║
║ ─────────────────────────────────────────────────────────
║ STEP 1: Define Variants
║ ─────────────────────────────────────────────────────────
║
║ Variant A: Fixed tensions
║   • compress=0.65 (always)
║   • exploit=0.55 (always)
║   • focus=0.70 (always)
║
║ Variant B: Adaptive tensions
║   • compress = policy_net(context)
║   • exploit = policy_net(context)
║   • focus = policy_net(context)
║
║ ─────────────────────────────────────────────────────────
║ STEP 2: Create Test Dataset
║ ─────────────────────────────────────────────────────────
║
║ Diverse queries dataset:
║   • 50 specific queries ("Where is the red car?")
║   • 50 vague queries ("Describe the scene")
║   • 50 detail queries ("What does the small text say?")
║   • 50 counting queries ("How many people?")
║
║ Total: 200 test cases
║
║ ─────────────────────────────────────────────────────────
║ STEP 3: Run Comparison in Gradio
║ ─────────────────────────────────────────────────────────
║
║ Load both checkpoints:
║   • checkpoint_fixed.pt → Variant A
║   • checkpoint_adaptive.pt → Variant B
║
║ Run app_dev.py:
║   • Test all 200 cases
║   • Side-by-side comparison
║   • Manual inspection + automated metrics
║
║ ─────────────────────────────────────────────────────────
║ STEP 4: Collect Metrics
║ ─────────────────────────────────────────────────────────
║
║ For each test case, record:
║   • Accuracy (correct answer? 0/1)
║   • Time (inference speed)
║   • Tokens used (efficiency)
║   • Subjective quality (0-5 scale, your rating)
║
║ Export to CSV:
║   query_type, variant, accuracy, time, tokens, quality
║
║ ─────────────────────────────────────────────────────────
║ STEP 5: Analyze Results
║ ─────────────────────────────────────────────────────────
║
║ Statistical analysis:
║
║ import pandas as pd
║
║ df = pd.read_csv('comparison_results.csv')
║
║ # Group by variant
║ summary = df.groupby(['query_type', 'variant']).agg({
║     'accuracy': 'mean',
║     'time': 'mean',
║     'tokens': 'mean'
║ })
║
║ print(summary)
║
║ # Result:
║ #                      accuracy  time   tokens
║ # query_type  variant
║ # specific    fixed    0.68     0.045  732
║ #             adaptive 0.74     0.042  689
║ # vague       fixed    0.65     0.048  823
║ #             adaptive 0.69     0.044  791
║ # detail      fixed    0.62     0.052  912
║ #             adaptive 0.71     0.048  845
║
║ # CONCLUSION: Adaptive wins across ALL query types!
║ #   • +6% accuracy (specific)
║ #   • +4% accuracy (vague)
║ #   • +9% accuracy (detail) ← BIGGEST WIN
║ #   • 8% faster on average
║ #   • 7% fewer tokens
║
║ ─────────────────────────────────────────────────────────
║ STEP 6: Document & Decide
║ ─────────────────────────────────────────────────────────
║
║ Create report:
║   • RESEARCH/experiments/01-adaptive-vs-fixed.md
║   • Include: hypothesis, methodology, results, conclusion
║   • Save comparison_results.csv
║   • Screenshot key comparisons from Gradio
║
║ Decision:
║   ✅ Adaptive tensions confirmed superior
║   → Use checkpoint_adaptive.pt as new baseline
║   → Archive checkpoint_fixed.pt
║
╚═══════════════════════════════════════════════════════════
```

**KARPATHY:**
So the workflow is:
1. **Hypothesis** → "Adaptive beats fixed"
2. **Variants** → Train both
3. **Test** → Run in Gradio interface
4. **Metrics** → Collect automated + manual data
5. **Analyze** → Statistical comparison
6. **Decide** → Keep winner, archive loser

**LOD ORACLE:**
And Gradio is the **testing instrument**. You're not running blind scripts—you're SEEING the differences.

---

## Act V: The Rapid Iteration Loop

**KARPATHY:**
What does a typical development day look like?

**HUGGINGFACE ORACLE:**
Here's the ideal iteration loop:

```
╔═══════════════════════════════════════════════════════════
║ DAILY DEVELOPMENT LOOP
╠═══════════════════════════════════════════════════════════
║
║ MORNING: Train New Variant
║ ─────────────────────────────────────────────────────────
║ 09:00 - Have an idea: "What if we add saliency weighting?"
║
║ 09:15 - Modify balancing.py:
║         • Add saliency multiplier to balanced_scores
║
║ 09:30 - Start training:
║         python train.py --experiment saliency-weighted \
║                         --epochs 3 \
║                         --checkpoint-every 500
║
║ 10:00 - Training running (monitor logs)
║ 12:00 - Training complete, 3 checkpoints saved
║
║ AFTERNOON: Test & Compare
║ ─────────────────────────────────────────────────────────
║ 13:00 - Launch Gradio interface:
║         python app_dev.py
║
║ 13:05 - Load checkpoints:
║         • Baseline (no saliency)
║         • Saliency-weighted (new)
║
║ 13:10 - Test 20 diverse images
║         • Drag and drop in browser
║         • Type queries
║         • Compare side-by-side
║         • See heatmaps
║
║ 14:00 - Observations:
║         ✅ Saliency helps on complex scenes (+8% accuracy)
║         ❌ Saliency hurts on text queries (-3% accuracy)
║         → MIXED RESULTS
║
║ EVENING: Refine & Document
║ ─────────────────────────────────────────────────────────
║ 15:00 - Hypothesis refinement:
║         "Saliency should be CONDITIONAL"
║         → High weight for scene queries
║         → Low weight for text queries
║
║ 15:30 - Modify balancing.py:
║         • Add query-type detector
║         • Adaptive saliency weight
║
║ 16:00 - Quick training run (1 epoch, fast check)
║
║ 16:30 - Test in Gradio
║         ✅ Conditional saliency wins on BOTH!
║
║ 17:00 - Document:
║         • Git commit: "Add conditional saliency weighting"
║         • Save experiment notes
║         • Export comparison CSV
║
║ 17:30 - End of day
║         Tomorrow: Test on larger dataset, ablation study
║
╚═══════════════════════════════════════════════════════════
```

**KARPATHY:**
So you're iterating **within a single day**:
- Morning: train
- Afternoon: test in Gradio
- Evening: refine and document

**LOD ORACLE:**
Fast feedback loop. You KNOW by end of day if idea worked.

---

## Act VI: The Checkpoint Management

**KARPATHY:**
We're going to accumulate a LOT of checkpoints. How do we manage them?

**HUGGINGFACE ORACLE:**
You need a **checkpoint naming convention** and **metadata tracking**:

```python
# === CHECKPOINT NAMING CONVENTION ===

# Format: {experiment}_{date}_{step}_{metric}.pt

# Examples:
checkpoints/
├── baseline_2025-01-30_step-1000_acc-0.678.pt
├── adaptive-tensions_2025-01-30_step-1500_acc-0.698.pt
├── saliency-weighted_2025-01-31_step-1000_acc-0.684.pt
├── conditional-saliency_2025-01-31_step-1500_acc-0.712.pt  ← BEST!
└── 40-channel-texture_2025-02-01_step-2000_acc-0.705.pt

# === METADATA TRACKING ===

# checkpoints/metadata.json
{
  "conditional-saliency_2025-01-31_step-1500_acc-0.712.pt": {
    "experiment": "conditional-saliency",
    "date": "2025-01-31",
    "step": 1500,
    "epoch": 3,
    "metrics": {
      "eval_accuracy": 0.712,
      "eval_loss": 0.298,
      "speedup": 1.25
    },
    "config": {
      "texture_channels": 13,
      "adaptive_tensions": true,
      "saliency_weighting": "conditional"
    },
    "notes": "Best so far! Adaptive saliency based on query type.",
    "hypothesis": "Conditional saliency improves both scene and text queries",
    "result": "CONFIRMED - +5% across all query types",
    "keep": true  # Mark for preservation
  },
  "saliency-weighted_2025-01-31_step-1000_acc-0.684.pt": {
    "experiment": "saliency-weighted",
    "result": "MIXED - good for scenes, bad for text",
    "keep": false,  # Archive/delete
    "superseded_by": "conditional-saliency_2025-01-31_step-1500_acc-0.712.pt"
  }
}

# === CHECKPOINT CLEANUP SCRIPT ===

def cleanup_checkpoints(keep_best_n=5, keep_marked=True):
    """Remove old checkpoints, keeping only the best"""
    import json
    from pathlib import Path

    # Load metadata
    with open('checkpoints/metadata.json') as f:
        metadata = json.load(f)

    # Find checkpoints to keep
    keep_files = set()

    # Keep explicitly marked
    if keep_marked:
        for ckpt, info in metadata.items():
            if info.get('keep', False):
                keep_files.add(ckpt)

    # Keep top N by accuracy
    sorted_ckpts = sorted(
        metadata.items(),
        key=lambda x: x[1]['metrics'].get('eval_accuracy', 0),
        reverse=True
    )
    for ckpt, _ in sorted_ckpts[:keep_best_n]:
        keep_files.add(ckpt)

    # Remove others
    ckpt_dir = Path('checkpoints')
    for ckpt_file in ckpt_dir.glob('*.pt'):
        if ckpt_file.name not in keep_files:
            print(f"🗑️  Removing {ckpt_file.name}")
            ckpt_file.unlink()

    print(f"✅ Kept {len(keep_files)} checkpoints")
```

**KARPATHY:**
So we have:
- **Naming convention** with metrics in filename
- **Metadata JSON** tracking experiments
- **Cleanup script** to remove old checkpoints

**LOD ORACLE:**
And you can see the progression:
- baseline → 0.678
- adaptive → 0.698 (+2%)
- conditional saliency → 0.712 (+3.4% from baseline!)

---

## Act VII: The Ablation Study Pattern

**KARPATHY:**
How do we do ablation studies? Like, "Which components actually matter?"

**HUGGINGFACE ORACLE:**
Ablation studies in Gradio are powerful. Here's the pattern:

```python
# === ABLATION STUDY: Which scorers matter? ===

def ablation_study_interface():
    """Test removing components one at a time"""

    with gr.Blocks() as demo:
        gr.Markdown("# 🔬 Ablation Study: Three Ways of Knowing")

        # Test configuration
        with gr.Row():
            use_info = gr.Checkbox(label="Propositional (info)", value=True)
            use_persp = gr.Checkbox(label="Perspectival (saliency)", value=True)
            use_partic = gr.Checkbox(label="Participatory (query)", value=True)

        image_input = gr.Image(type="pil")
        query_input = gr.Textbox()
        test_btn = gr.Button("Test Configuration")

        # Results
        with gr.Row():
            with gr.Column():
                gr.Markdown("### All Three Scorers")
                result_all = gr.Textbox(label="Result", lines=6)
                metrics_all = gr.JSON(label="Metrics")

            with gr.Column():
                gr.Markdown("### Current Configuration")
                result_ablated = gr.Textbox(label="Result", lines=6)
                metrics_ablated = gr.JSON(label="Metrics")

        comparison = gr.Textbox(label="Impact Analysis", lines=4)

        def run_ablation(image, query, use_info, use_persp, use_partic):
            # Run with all scorers
            result_full = run_model(image, query,
                                    info=True, persp=True, partic=True)

            # Run with selected scorers
            result_partial = run_model(image, query,
                                       info=use_info, persp=use_persp, partic=use_partic)

            # Compare
            accuracy_drop = result_full['accuracy'] - result_partial['accuracy']

            analysis = f"""
Removed scorers impact:
• Accuracy drop: {accuracy_drop:.1%}
• Speed change: {result_partial['time'] / result_full['time']:.2f}×

Conclusion: {"MINOR" if accuracy_drop < 0.02 else "SIGNIFICANT"} impact
"""

            return (
                result_full['answer'], result_full['metrics'],
                result_partial['answer'], result_partial['metrics'],
                analysis
            )

        test_btn.click(
            fn=run_ablation,
            inputs=[image_input, query_input, use_info, use_persp, use_partic],
            outputs=[result_all, metrics_all, result_ablated, metrics_ablated, comparison]
        )

    return demo

# Launch ablation study
demo = ablation_study_interface()
demo.launch()
```

**KARPATHY:**
So you can toggle components on/off and immediately see the impact:
- Remove Propositional → accuracy drops 8%
- Remove Perspectival → accuracy drops 12%
- Remove Participatory → accuracy drops 15% (BIGGEST!)

**LOD ORACLE:**
Visual, interactive ablation. You DISCOVER which components matter by playing with toggles.

---

## Act VIII: The Export & Reproducibility

**KARPATHY:**
How do we make results reproducible? For papers, for collaborators?

**HUGGINGFACE ORACLE:**
Export everything from Gradio sessions:

```python
# === SESSION EXPORT ===

class SessionLogger:
    """Log all comparisons for reproducibility"""

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = []

    def log_comparison(self, image_path, query, results):
        """Log a single comparison"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'image': image_path,
            'query': query,
            'variants': results
        }
        self.results.append(entry)

    def export(self, format='csv'):
        """Export session data"""
        if format == 'csv':
            # Flatten to CSV
            rows = []
            for entry in self.results:
                for variant, result in entry['variants'].items():
                    rows.append({
                        'timestamp': entry['timestamp'],
                        'image': entry['image'],
                        'query': entry['query'],
                        'variant': variant,
                        'answer': result['answer'],
                        'time': result['time'],
                        'tokens': result['tokens'],
                        'accuracy': result.get('accuracy', None)
                    })

            df = pd.DataFrame(rows)
            filename = f"results/{self.experiment_name}_{self.session_id}.csv"
            df.to_csv(filename, index=False)
            return filename

        elif format == 'json':
            # Full JSON export
            filename = f"results/{self.experiment_name}_{self.session_id}.json"
            with open(filename, 'w') as f:
                json.dump({
                    'experiment': self.experiment_name,
                    'session_id': self.session_id,
                    'results': self.results
                }, f, indent=2)
            return filename

# Add to Gradio interface
session_logger = SessionLogger("adaptive-vs-fixed")

def compare_and_log(image, query, variants):
    results = comparator.compare(image, query, variants)

    # Log to session
    session_logger.log_comparison(image.filename, query, results)

    return format_results(results)

# Export button
def export_session():
    csv_file = session_logger.export(format='csv')
    json_file = session_logger.export(format='json')
    return f"Exported:\n{csv_file}\n{json_file}"

export_btn.click(fn=export_session, outputs=export_status)
```

**LOD ORACLE:**
So every test session generates:
- CSV for statistical analysis
- JSON for full reproducibility
- Timestamped, with image paths and queries

**KARPATHY:**
Anyone can load that CSV and reproduce your analysis. Or re-run with same images/queries.

---

## Act IX: The Continuous Validation

**KARPATHY:**
As we train, how do we know we're not breaking things?

**HUGGINGFACE ORACLE:**
You need a **validation dashboard** that runs automatically:

```python
# === CONTINUOUS VALIDATION ===

# File: validate_checkpoint.py

def validate_checkpoint(checkpoint_path, test_suite="standard"):
    """Run full validation on a checkpoint"""

    # Load checkpoint
    model = load_checkpoint(checkpoint_path)

    # Run test suite
    if test_suite == "standard":
        test_cases = load_test_suite("test_suites/standard_100.json")
    elif test_suite == "comprehensive":
        test_cases = load_test_suite("test_suites/comprehensive_500.json")

    results = []
    for test in test_cases:
        result = model.generate(test['image'], test['query'])

        # Check correctness
        correct = evaluate_answer(result['answer'], test['expected'])

        results.append({
            'test_id': test['id'],
            'query_type': test['type'],
            'correct': correct,
            'time': result['time'],
            'tokens': result['tokens']
        })

    # Compute metrics
    metrics = {
        'accuracy': sum(r['correct'] for r in results) / len(results),
        'avg_time': sum(r['time'] for r in results) / len(results),
        'avg_tokens': sum(r['tokens'] for r in results) / len(results),
        'by_query_type': {}
    }

    # Breakdown by query type
    for qtype in ['specific', 'vague', 'detail', 'counting']:
        subset = [r for r in results if r['query_type'] == qtype]
        if subset:
            metrics['by_query_type'][qtype] = {
                'accuracy': sum(r['correct'] for r in subset) / len(subset),
                'count': len(subset)
            }

    return metrics

# Hook into training
from transformers import TrainerCallback

class ValidationCallback(TrainerCallback):
    """Run validation after each checkpoint"""

    def on_save(self, args, state, control, **kwargs):
        """Called whenever a checkpoint is saved"""
        checkpoint_path = f"{args.output_dir}/checkpoint-{state.global_step}"

        print(f"\n🔍 Validating {checkpoint_path}...")
        metrics = validate_checkpoint(checkpoint_path, test_suite="standard")

        print(f"✅ Accuracy: {metrics['accuracy']:.1%}")
        print(f"⚡ Avg time: {metrics['avg_time']:.3f}s")
        print(f"🎯 Avg tokens: {metrics['avg_tokens']:.0f}")

        # Log to W&B / TensorBoard / etc.
        if wandb.run is not None:
            wandb.log({
                "val/accuracy": metrics['accuracy'],
                "val/time": metrics['avg_time'],
                "val/tokens": metrics['avg_tokens'],
            }, step=state.global_step)

        # Save validation report
        with open(f"{checkpoint_path}/validation_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

# Add callback to trainer
trainer = Trainer(
    ...,
    callbacks=[ValidationCallback()]
)
```

**KARPATHY:**
So after EVERY checkpoint save:
- Run 100 test cases automatically
- Compute metrics
- Log to tracking system
- Save validation report

**LOD ORACLE:**
You catch regressions immediately. If accuracy drops, you know which checkpoint introduced the problem.

---

## Closing: The Complete Testing Workflow

**SOCRATES:**
*Materializing from the Dirac Sea*

You've architected the testing methodology. Let me synthesize:

```
╔═══════════════════════════════════════════════════════════
║ THE COMPLETE TESTING WORKFLOW
╠═══════════════════════════════════════════════════════════
║
║ 1. DEVELOPMENT INTERFACE (Gradio)
║    • Multi-model comparison
║    • Side-by-side results
║    • Interactive parameter tuning
║    • Heatmap visualization
║    • Session history tracking
║
║ 2. CHECKPOINT MANAGEMENT
║    • Naming convention with metrics
║    • Metadata tracking (JSON)
║    • Keep best N checkpoints
║    • Resume training from any point
║
║ 3. HYPOTHESIS TESTING
║    • Define variants
║    • Create test dataset
║    • Run comparison in Gradio
║    • Collect metrics (auto + manual)
║    • Statistical analysis
║    • Document & decide
║
║ 4. RAPID ITERATION
║    • Morning: train variant
║    • Afternoon: test in Gradio
║    • Evening: refine & document
║    • Fast feedback loop (same day)
║
║ 5. ABLATION STUDIES
║    • Toggle components on/off
║    • Interactive impact measurement
║    • Discover essential components
║
║ 6. EXPORT & REPRODUCIBILITY
║    • CSV for analysis
║    • JSON for full reproducibility
║    • Timestamped sessions
║
║ 7. CONTINUOUS VALIDATION
║    • Auto-validate every checkpoint
║    • Catch regressions early
║    • Track metrics over time
║
╚═══════════════════════════════════════════════════════════
```

**This is how you turn philosophy into working code.**

**KARPATHY:**
We've designed a system where:
- Testing is **visual** (Gradio interface)
- Checkpoints are **managed** (naming + metadata)
- Hypotheses are **validated** (A/B testing protocol)
- Iterations are **fast** (same-day feedback)
- Results are **reproducible** (export all data)

**LOD ORACLE:**
And it all centers on the **Gradio interface** as your primary development tool. Not an afterthought—a **microscope for your system**.

**HUGGINGFACE ORACLE:**
Plus HuggingFace Trainer handles checkpointing automatically. You focus on experiments, not infrastructure.

**MUSE BIRD:**
🐦 *FROM LOCALHOST TO INSIGHTS! Test fast, learn fast, build fast! Science at the speed of iteration!*

---

## Epilogue: The Next Steps

**KARPATHY:**
We now have three complete documents:
- **Part 38 Addendum:** Implementation plan (code structure)
- **Part 38 Main:** Infrastructure architecture (HuggingFace)
- **Part 39:** Testing workflow (Gradio + checkpoints)

**LOD ORACLE:**
What remains?

**HUGGINGFACE ORACLE:**
Just **implementation**. The design is complete. Time to code.

**KARPATHY:**
Here's the immediate action plan:

```
╔═══════════════════════════════════════════════════════════
║ IMMEDIATE NEXT STEPS
╠═══════════════════════════════════════════════════════════
║
║ 1. ✅ Part 38 Addendum written
║ 2. ✅ Part 38 infrastructure designed
║ 3. ✅ Part 39 testing workflow designed
║ 4. ⭕ Build app_dev.py (multi-model Gradio)
║ 5. ⭕ Implement texture_array.py (13 channels MVP)
║ 6. ⭕ Test texture generation in Gradio
║ 7. ⭕ Implement knowing.py
║ 8. ⭕ Test scorers in Gradio
║ 9. ⭕ Implement balancing.py
║ 10. ⭕ Compare fixed vs adaptive in Gradio
║ 11. ⭕ Implement attending.py
║ 12. ⭕ Full pipeline test in Gradio
║ 13. ⭕ Deploy to HuggingFace Space (MVP)
║
╚═══════════════════════════════════════════════════════════
```

**LOD ORACLE:**
37 dialogues of philosophy. 2 dialogues of implementation planning. 1 dialogue of testing methodology.

**Now we build.**

**KARPATHY:**
The theory is sound. The architecture is clear. The testing is designed.

**SOCRATES:**
And what have we learned about development?

**KARPATHY:**
That **testing is discovery**. You don't know what works until you SEE it. Gradio makes the invisible visible.

**LOD ORACLE:**
And that **checkpoints are checkpoints**. Not just for recovery—for **experimentation**. Load any state, compare any variants.

**HUGGINGFACE ORACLE:**
And that **infrastructure matters**. HuggingFace gives you: model hosting, GPU compute, dataset storage, collaboration tools. For free.

**MUSE BIRD:**
🐦 *40 DIALOGUES COMPLETE! From neurons to knowing! From attention to relevance! From theory to... IMPLEMENTATION!*

---

    ∿◇∿
   From plans
  To processes
 Testing reveals
Truth through iteration

*The Dirac Sea shimmers with running code, Gradio interfaces, and checkpoint files. The oracles fade, leaving behind a complete methodology for turning relevance realization into reality.*

**FIN**
