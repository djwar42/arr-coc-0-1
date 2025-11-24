# Validation for Platonic Coding Codebases

**The Karpathy Way: Simple, Honest, Reproducible**

---

## Philosophy

> "Don't tell me it works. Show me the loss curve going down." — Every ML engineer ever

We're not building production systems. We're exploring ideas through Platonic Dialogues and proving they work (or don't).

**Validation should be:**
1. **Fast** — Get signal in minutes, not days
2. **Visual** — See what's happening, don't just read numbers
3. **Honest** — If it's broken, say it's broken
4. **Reproducible** — Anyone can run it and get the same results

---

## The Validation Stack

```
╔═══════════════════════════════════════════════════════════
║ SIMPLE VALIDATION ARCHITECTURE
╠═══════════════════════════════════════════════════════════
║
║ 🧪 SMOKE TEST (5 min)
║     ├── Does it forward pass? (test_shapes.py)
║     ├── Does it backward pass? (test_gradients.py)
║     └── Can we overfit 1 batch? (smoke_test.py)
║
║ 🏃 QUICK VALIDATION (30 min)
║     ├── Train on 100 examples (VQAv2 subset)
║     ├── Log to W&B (live loss curves)
║     └── Check: loss going down? accuracy going up?
║
║ 📊 W&B DASHBOARD
║     ├── Live training metrics (loss, acc, lr)
║     ├── System metrics (GPU util, memory)
║     └── Shareable link → embed in Gradio Space
║
║ 🎨 GRADIO SPACE (The Interface)
║     ├── Tab 1: Demo (try the model)
║     ├── Tab 2: Training (W&B dashboard embed)
║     └── Tab 3: Ablations (compare checkpoints)
║
╚═══════════════════════════════════════════════════════════
```

---

## Step-by-Step: Validating ARR-COC

### **Step 0: Smoke Tests (5 minutes)**

**Goal:** Prove the code doesn't crash.

```python
# tests/test_smoke.py
def test_forward_pass():
    """Can we run inference without crashing?"""
    model = ARRCOCQwen.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    image = torch.rand(1, 3, 448, 448)
    input_ids = torch.randint(0, 1000, (1, 10))

    outputs = model(pixel_values=image, input_ids=input_ids)

    assert outputs.logits.shape == (1, 10, model.vocab_size)
    print("✓ Forward pass works")

def test_backward_pass():
    """Can we compute gradients?"""
    model = ARRCOCQwen.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Random batch
    image = torch.rand(1, 3, 448, 448)
    input_ids = torch.randint(0, 1000, (1, 10))
    labels = torch.randint(0, 1000, (1, 10))

    # Forward + backward
    outputs = model(pixel_values=image, input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    print(f"✓ Backward pass works (loss = {loss.item():.4f})")

def test_overfit_one_batch():
    """Can we overfit a single batch? (The litmus test)"""
    model = ARRCOCQwen.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # One batch (repeat 100 times)
    image = torch.rand(1, 3, 448, 448)
    input_ids = torch.randint(0, 1000, (1, 10))
    labels = input_ids.clone()

    losses = []
    for i in range(100):
        outputs = model(pixel_values=image, input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

        if i % 20 == 0:
            print(f"Step {i}: loss = {loss.item():.4f}")

    # Loss should drop significantly
    assert losses[-1] < losses[0] * 0.5, "Model isn't learning!"
    print(f"✓ Overfitting works (loss: {losses[0]:.4f} → {losses[-1]:.4f})")
```

**Run:**
```bash
pytest tests/test_smoke.py -v
```

**Expected:**
```
✓ Forward pass works
✓ Backward pass works
✓ Overfitting works (loss: 8.2341 → 0.0234)
```

**If any test fails → fix before proceeding.**

---

### **Step 1: Quick Training Run (30 minutes)**

**Goal:** Train on 100 examples, see if loss goes down.

```python
# training/quick_validation.py
import wandb
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

# Load 100 examples from VQAv2
dataset = load_dataset("HuggingFaceM4/VQAv2", split="train[:100]")

# Setup W&B
wandb.init(
    project="arr-coc-validation",
    name="quick-run-100",
    config={
        "model": "arr-coc-0-1",
        "dataset": "VQAv2",
        "num_examples": 100,
        "epochs": 10
    }
)

# Train
args = TrainingArguments(
    output_dir="./checkpoints",
    report_to="wandb",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    logging_steps=5,
    save_steps=50
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset
)

trainer.train()

# Check final metrics
final_loss = trainer.state.log_history[-1]['loss']
print(f"\n📊 Final loss: {final_loss:.4f}")

if final_loss < 2.0:
    print("✅ PASS: Model is learning!")
else:
    print("❌ FAIL: Loss too high, something's wrong")

wandb.finish()
```

**Run:**
```bash
python training/quick_validation.py
```

**Watch W&B dashboard live:**
- Loss should go down
- GPU utilization should be >80%
- No NaN values

---

### **Step 2: What to Track in W&B**

**Essential Metrics:**
```python
wandb.log({
    # Training
    "train/loss": loss,
    "train/perplexity": torch.exp(loss),
    "train/learning_rate": optimizer.param_groups[0]['lr'],

    # Validation (if you have eval set)
    "val/loss": val_loss,
    "val/accuracy": val_acc,

    # ARR-COC specific
    "arr_coc/avg_relevance": avg_relevance_score,
    "arr_coc/tokens_selected": num_selected_tokens,
    "arr_coc/compression_ratio": 1024 / num_selected_tokens,

    # System
    "system/gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
})
```

**Config to save:**
```python
wandb.config.update({
    "model_name": "arr-coc-0-1",
    "base_model": "Qwen2-VL-2B-Instruct",
    "num_visual_tokens": 200,
    "dataset": "VQAv2",
    "learning_rate": 1e-5,
    "batch_size": 4,
    "gradient_accumulation": 4,
})
```

---

### **Step 3: The Gradio Interface**

**5-Tab Design for Complete Validation:**

```
╔═══════════════════════════════════════════════════════════
║ COMPLETE PLATONIC CODING VALIDATION INTERFACE
╠═══════════════════════════════════════════════════════════
║
║ 🎯 DEMO - Try the model interactively
║ 📊 TRAINING - Monitor all W&B runs
║ 🔬 ABLATIONS - Systematic component testing
║ 📈 ANALYSIS - Deep dive on one checkpoint
║ ⚖️ COMPARE - A/B test two checkpoints
║
╚═══════════════════════════════════════════════════════════
```

#### **Tab 1: 🎯 Demo**
*Interactive model testing with checkpoint selection*

```python
def demo_tab():
    """Let users test any checkpoint interactively"""

    def get_available_checkpoints():
        """Fetch checkpoints from HuggingFace Hub"""
        from huggingface_hub import HfApi
        api = HfApi()

        # List all checkpoints in the repo
        files = api.list_repo_tree("newsofpeace2/arr-coc-0-1", path_in_repo="checkpoints")
        checkpoints = [f.path.split("/")[-1] for f in files if f.type == "directory"]
        return sorted(checkpoints, reverse=True)  # Latest first

    def run_inference(checkpoint_name, image, query):
        """Load checkpoint and run inference"""
        # Load selected checkpoint
        model = load_checkpoint(checkpoint_name)

        # Run inference
        answer = model.generate(image, query)

        # Generate visualizations
        viz = generate_visualizations(model, image, query)

        return answer, viz

    with gr.Blocks() as tab:
        gr.Markdown("## Try ARR-COC on Your Images")

        with gr.Row():
            checkpoint_dropdown = gr.Dropdown(
                choices=get_available_checkpoints(),
                label="Select Checkpoint",
                value=get_available_checkpoints()[0],
                info="Pick which trained model to test"
            )
            refresh_btn = gr.Button("🔄 Refresh Checkpoints")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload Image", type="pil")
                query_input = gr.Textbox(
                    label="Ask a Question",
                    placeholder="What is in this image?"
                )
                submit_btn = gr.Button("Submit", variant="primary")

            with gr.Column():
                answer_output = gr.Textbox(label="Answer", lines=3)
                viz_output = gr.Gallery(
                    label="ARR-COC Visualizations",
                    columns=3,
                    height="auto"
                )

        # Examples
        gr.Examples(
            examples=[
                ["examples/cat.jpg", "What is the cat doing?"],
                ["examples/street.jpg", "How many cars are visible?"],
                ["examples/people.jpg", "Describe the scene"],
            ],
            inputs=[image_input, query_input]
        )

        # Event handlers
        submit_btn.click(
            run_inference,
            inputs=[checkpoint_dropdown, image_input, query_input],
            outputs=[answer_output, viz_output]
        )

        refresh_btn.click(
            get_available_checkpoints,
            outputs=[checkpoint_dropdown]
        )

    return tab
```

#### **Tab 2: 📊 Training**
*Live W&B dashboard showing all training runs*

```python
def training_tab():
    """Embed W&B workspace showing all runs"""

    PROJECT_NAME = "arr-coc-0-1"
    WANDB_USER = "newsofpeace2"
    workspace_url = f"https://wandb.ai/{WANDB_USER}/{PROJECT_NAME}/workspace"

    with gr.Blocks() as tab:
        gr.Markdown(f"""
        ## Training Dashboard

        All training runs for `{PROJECT_NAME}` are tracked here.

        **Features:**
        - Compare runs side-by-side
        - Filter by tags (baseline, experiment, ablation)
        - Live training progress
        - Export data for analysis

        [Open in new tab ↗]({workspace_url})
        """)

        # Quick stats
        def get_quick_stats():
            """Fetch quick stats from W&B"""
            import wandb
            api = wandb.Api()
            runs = api.runs(f"{WANDB_USER}/{PROJECT_NAME}")

            stats = {
                "Total Runs": len(runs),
                "Running": len([r for r in runs if r.state == "running"]),
                "Finished": len([r for r in runs if r.state == "finished"]),
                "Failed": len([r for r in runs if r.state == "failed"]),
            }
            return stats

        with gr.Row():
            stats_display = gr.JSON(label="Quick Stats", value=get_quick_stats())
            refresh_stats = gr.Button("🔄 Refresh Stats")

        refresh_stats.click(get_quick_stats, outputs=[stats_display])

        # Embedded W&B workspace
        gr.HTML(f"""
            <iframe
                src="{workspace_url}?jupyter=true"
                style="width: 100%; height: 75vh; border: 1px solid #ddd; border-radius: 8px;">
            </iframe>
        """)

    return tab
```

#### **Tab 3: 🔬 Ablations**
*Systematic component testing - turn features on/off*

```python
def ablations_tab():
    """Test individual components of ARR-COC"""

    def run_ablation(image, query, use_propositional, use_perspectival, use_participatory):
        """Run inference with selected components enabled"""

        config = {
            "propositional": use_propositional,
            "perspectival": use_perspectival,
            "participatory": use_participatory,
        }

        # Run with ablated model
        model = load_ablated_model(config)
        answer = model.generate(image, query)

        # Get component scores
        scores = model.get_component_scores()

        return answer, scores

    with gr.Blocks() as tab:
        gr.Markdown("""
        ## Component Ablation Testing

        Test individual ARR-COC components to understand their contribution.
        Turn components on/off to see how each affects results.
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input")
                image_input = gr.Image(label="Test Image", type="pil")
                query_input = gr.Textbox(label="Question")

                gr.Markdown("### Enable Components")
                use_prop = gr.Checkbox(label="Propositional (Information)", value=True)
                use_persp = gr.Checkbox(label="Perspectival (Salience)", value=True)
                use_partic = gr.Checkbox(label="Participatory (Query Coupling)", value=True)

                run_btn = gr.Button("Run Ablation", variant="primary")

            with gr.Column():
                gr.Markdown("### Results")
                answer_output = gr.Textbox(label="Answer", lines=3)

                gr.Markdown("### Component Contributions")
                scores_output = gr.JSON(label="Scores")

        # Preset ablations
        gr.Markdown("### Common Ablation Tests")
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
```

#### **Tab 4: 📈 Analysis**
*Deep dive into one checkpoint with metrics and visualizations*

```python
def analysis_tab():
    """Deep analysis of a single checkpoint"""

    def analyze_checkpoint(checkpoint_name):
        """Generate comprehensive analysis"""
        import wandb
        api = wandb.Api()

        # Find corresponding W&B run
        runs = api.runs(f"newsofpeace2/arr-coc-0-1")
        run = next((r for r in runs if checkpoint_name in r.name), None)

        if not run:
            return "Run not found", {}, None

        # Get all metrics
        metrics = {
            "Final Loss": run.summary.get("train/loss", "N/A"),
            "Best Val Accuracy": run.summary.get("val/accuracy", "N/A"),
            "Tokens Selected": run.summary.get("arr_coc/tokens_selected", "N/A"),
            "Compression Ratio": run.summary.get("arr_coc/compression_ratio", "N/A"),
            "Training Time (hrs)": run.summary.get("_runtime", 0) / 3600,
            "GPU Memory (GB)": run.summary.get("system/gpu_memory_gb", "N/A"),
        }

        # Get training history
        history = run.scan_history(keys=["train/loss", "val/accuracy"])

        # Create plots
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        losses = [h.get("train/loss") for h in history if h.get("train/loss")]
        accs = [h.get("val/accuracy") for h in history if h.get("val/accuracy")]

        ax1.plot(losses)
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")

        ax2.plot(accs)
        ax2.set_title("Validation Accuracy")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Accuracy")

        return run.url, metrics, fig

    with gr.Blocks() as tab:
        gr.Markdown("## Checkpoint Deep Dive")

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
                gr.Markdown("### Training Curves")
                plots_display = gr.Plot(label="Loss & Accuracy")

        gr.Markdown("### W&B Run Details")
        run_link = gr.Textbox(label="W&B Run URL", interactive=False)

        analyze_btn.click(
            analyze_checkpoint,
            inputs=[checkpoint_selector],
            outputs=[run_link, metrics_display, plots_display]
        )

    return tab
```

#### **Tab 5: ⚖️ Compare**
*Side-by-side A/B testing of two checkpoints*

```python
def compare_tab():
    """Compare two checkpoints on the same input"""

    def compare_checkpoints(checkpoint_a, checkpoint_b, image, query):
        """Run both checkpoints and compare"""
        import time

        # Load both models
        model_a = load_checkpoint(checkpoint_a)
        model_b = load_checkpoint(checkpoint_b)

        # Run inference with timing
        start = time.time()
        answer_a = model_a.generate(image, query)
        latency_a = (time.time() - start) * 1000

        start = time.time()
        answer_b = model_b.generate(image, query)
        latency_b = (time.time() - start) * 1000

        # Get metrics
        metrics_a = {
            "Answer": answer_a,
            "Latency (ms)": f"{latency_a:.1f}",
            "Tokens Used": model_a.last_num_tokens,
            "Avg Relevance": f"{model_a.last_avg_relevance:.3f}",
        }

        metrics_b = {
            "Answer": answer_b,
            "Latency (ms)": f"{latency_b:.1f}",
            "Tokens Used": model_b.last_num_tokens,
            "Avg Relevance": f"{model_b.last_avg_relevance:.3f}",
        }

        # Generate visualizations
        viz_a = generate_visualizations(model_a, image, query)
        viz_b = generate_visualizations(model_b, image, query)

        return metrics_a, metrics_b, viz_a, viz_b

    with gr.Blocks() as tab:
        gr.Markdown("## A/B Checkpoint Comparison")

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
```

#### **Main App: Combine All 5 Tabs**

```python
# app.py - Complete Platonic Coding Validation Interface
import gradio as gr

# Project configuration
PROJECT_NAME = "arr-coc-0-1"
WANDB_USER = "newsofpeace2"

# Build all tabs
demo = demo_tab()
training = training_tab()
ablations = ablations_tab()
analysis = analysis_tab()
compare = compare_tab()

# Combine into 5-tab interface
app = gr.TabbedInterface(
    [demo, training, ablations, analysis, compare],
    ["🎯 Demo", "📊 Training", "🔬 Ablations", "📈 Analysis", "⚖️ Compare"],
    title=f"{PROJECT_NAME} - Validation Interface",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
```

---

## What Success Looks Like

### **Smoke Tests**
```
✓ Forward pass works
✓ Backward pass works
✓ Overfitting works (loss: 8.23 → 0.02)
```

### **Quick Validation (100 examples, 10 epochs)**
```
📊 W&B Dashboard shows:
  - Loss: 6.2 → 2.1 (going down ✓)
  - Perplexity: 492 → 8.2 (going down ✓)
  - GPU util: 85% (good ✓)
  - No NaN values (good ✓)
```

### **Gradio Space**
- Tab 1: Model answers questions (sometimes correctly)
- Tab 2: W&B dashboard shows training progress
- Tab 3: Can compare baseline vs ARR-COC side-by-side

---

## What Failure Looks Like

### **Red Flags:**
- ❌ Loss not decreasing after 50 steps
- ❌ NaN values in loss
- ❌ GPU utilization <20% (something's blocking)
- ❌ Can't overfit a single batch
- ❌ Model outputs gibberish every time

### **Response:**
1. Check gradients: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
2. Check data: Print batch, verify it makes sense
3. Check learning rate: Try 10x higher, 10x lower
4. Check architecture: Did participatory scorer break something?

**Don't pretend it works if it doesn't.** Fix or pivot.

---

## The Honest Validation Report

After running validation, write a short report:

```markdown
# ARR-COC Validation Results

**Date:** 2025-01-31
**Model:** arr-coc-0-1
**Test:** 100 examples, 10 epochs

## Smoke Tests
✓ Forward pass: PASS
✓ Backward pass: PASS
✓ Overfit 1 batch: PASS (loss 8.23 → 0.02)

## Quick Training
✓ Loss decreased: 6.2 → 2.1
✓ No NaN values
✓ GPU utilization: 85%

## Qualitative Results
- Model answers simple questions correctly (5/10)
- Model hallucinates on complex questions (still a problem)
- ARR-COC selects reasonable patches (visual inspection)

## Known Issues
- Accuracy lower than baseline (expected for v0.1)
- Participatory scorer may need more training
- 200 tokens might not be enough (try 400 in v0.2)

## Verdict
✅ ARR-COC is trainable and shows learning signal.
Not production-ready, but good enough for research demo.

## Next Steps
1. Train on full VQAv2 (not just 100 examples)
2. Add LOD token expansion (64-400 variable budgets)
3. Compare to baseline quantitatively
```

**Be honest. Don't oversell.**

---

## Validation Checklist for Every Platonic Coding Repo

Before calling a codebase "validated":

- [ ] **Smoke tests pass** (forward, backward, overfit 1 batch)
- [ ] **Quick training shows learning** (loss goes down on 100 examples)
- [ ] **W&B dashboard exists** (shareable link)
- [ ] **Gradio Space deployed** (can demo + see training)
- [ ] **Honest report written** (what works, what doesn't)
- [ ] **Reproducible** (anyone can clone + run smoke tests)

**Don't skip steps.** If smoke tests fail, fix them first.

---

## Files to Add to Every Platonic Coding Repo

```
arr-coc-0-1/
├── tests/
│   ├── test_smoke.py          # Forward, backward, overfit tests
│   └── test_shapes.py          # Shape validation
├── training/
│   ├── quick_validation.py     # 100 examples, 10 epochs
│   └── train.py                # Full training script
├── app.py                      # 3-tab Gradio interface
├── VALIDATION.md               # Honest validation report
└── requirements.txt            # wandb, pytest, gradio, ...
```

---

## Philosophy: Iterate Fast, Fail Fast

> "The goal isn't to build something perfect. The goal is to find out if the idea works, as quickly as possible." — Karpathy (probably)

**Good validation:**
- Runs in <1 hour
- Shows clear signal (works or doesn't)
- Anyone can reproduce

**Bad validation:**
- Takes days to get results
- Results are ambiguous
- Only you can run it

**Keep it simple. Keep it honest. Keep it reproducible.**

---

## TL;DR

```
1. Write smoke tests (5 min)
2. Run quick validation (30 min)
3. Log to W&B (2 lines of code)
4. Build Gradio interface (3 tabs)
5. Write honest report (what works, what doesn't)
```

**If it's broken, say it's broken. Then fix it.**

That's the Karpathy way ¯\_(ツ)_/¯
