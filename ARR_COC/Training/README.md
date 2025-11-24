# ARR-COC Cloud Training

Beautiful terminal UI for training vision-language models on Vertex AI with W&B Launch.

---

## Quick Start

```bash
# 1. Copy config template
cp .training.starter .training
vim .training  # Fill in YOUR_USERNAME for HF_HUB_REPO_ID

# 2. Launch TUI
cd training/
python tui.py  # or: python cli.py (CLI mode)

# 3. Setup infrastructure (press '3' - one-time, ~5 min)
#    Auto-creates: GCS bucket, Artifact Registry, Service Account, W&B Queue
#    Auto-requests: GPU quota (T4/L4/A100)

# 4. Launch training (press '2')
#    Builds Docker → Submits to queue → Vertex AI runs on spot GPU

# 5. Monitor (press '1')
#    Real-time logs + metrics → https://wandb.ai/[user]/arr-coc-0-1
```

---

## The Flow

```
╔═══════════════════════════════════════════════════════
║ TRAINING WORKFLOW
╠═══════════════════════════════════════════════════════
║  1. Setup (press '3' - one-time)
║     ├─ Auto-detect W&B entity/project
║     ├─ Configure GCP region
║     ├─ Create: Buckets, Registry, Service Account, Queue
║     └─ Request GPU quota (T4/L4/A100)
║
║  2. Configure .training
║     └─ Model size, batch, learning rate, tokens
║
║  3. Launch (press '2')
║     ├─ Build Docker image (Cloud Build)
║     ├─ Submit to W&B Launch queue
║     └─ Vertex AI provisions spot GPU
║
║  4. Monitor (press '1')
║     ├─ Real-time logs + metrics
║     └─ W&B dashboard: wandb.ai/[user]/arr-coc-0-1
║
║  5. Results
║     ├─ Checkpoints → GCS bucket
║     └─ Auto-upload → HuggingFace Hub
╚═══════════════════════════════════════════════════════
```

---

## Files

**Interfaces:**
- `tui.py` - Main Textual TUI (home, launch, monitor, setup, infra, teardown)
- `cli.py` - CLI mode (same features, no TUI)

**Training:**
- `train.py` - Training script (ARRCOCQwen + Qwen3-VL-2B)
- `quick_validation.py` - Local validation (100 samples, 10 epochs)

**Config:**
- `.training.starter` - Template (committed)
- `.training` - Your config (gitignored)

**CLI structure:**
```
cli/
├── home/       - Navigation hub
├── launch/     - Submit jobs
├── monitor/    - Real-time tracking
├── setup/      - One-time infrastructure
├── infra/      - Status & quotas
├── teardown/   - Cleanup resources
└── shared/     - Helpers (WandBHelper, SetupHelper, callbacks)
```

---

## Configuration

**First time setup:**
```bash
cp .training.starter .training
vim .training  # Fill in YOUR_USERNAME
```

**Edit `.training` to change:**
- Model size (2B/4B/8B/30B)
- Batch size and accumulation
- Learning rate and epochs
- Number of visual tokens
- GPU type (A100/H100/L4/T4)

**Then resubmit:**
```bash
python cli.py launch  # Uses new config
```

---

## Costs (Spot Instances)

| Duration | Samples | GPU | Cost |
|----------|---------|-----|------|
| 5 min | 10 | 1x A100 | $0.09 |
| 12 hours | Full VQAv2 | 1x A100 | $13.20 |
| 3.5 hours | Full VQAv2 | 4x A100 | $7.70 |

**Spot instances save 70%** ($1.10/hr vs $3.67/hr for A100)

---

## What Gets Trained

**Trainable components (~2M params):**
- ParticipatoryScorer (~1.3M params)
- AdaptiveTensionBalancer (~700K params)

**Frozen (base model):**
- Qwen3-VL-2B (entire base model)

**Why this works:**
- ARR-COC components learn query-aware relevance
- Base model already knows vision + language
- Fast iteration, low cost

---

## Troubleshooting

**Missing prerequisites:**
```bash
# Start the interactive TUI
python tui.py

# Go to Setup page (press '3')
# The page will show you exactly what's missing:
#   - W&B authentication
#   - HuggingFace authentication
#   - GCP project configuration
#   - Service account credentials

# Run the interactive setup to fix everything
```

**Job fails to start:**
- Check W&B dashboard for error logs
- Verify Docker image built successfully
- Check GCP quotas (GPU availability)

**Training crashes:**
- Check W&B logs: `python cli.py monitor`
- Common: OOM → reduce batch size in `.training`
- Spot preemption → training auto-resumes from checkpoint

---

## Philosophy

**Simple, hackable, practical** (the Karpathy way):
- ✅ One config file (`.training`)
- ✅ One command to launch (`python cli.py launch`)
- ✅ Real-time monitoring (`python cli.py monitor`)
- ✅ Spot instances (70% cost savings)
- ✅ Auto-resume on preemption

**No complex orchestration, no YAML hell, just Python + Docker + W&B Launch.**

---

**Happy training!** ¯\_(ツ)_/¯
