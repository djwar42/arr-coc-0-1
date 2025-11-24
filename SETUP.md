# ARR-COC-0-1 Setup Guide

**5-minute setup with ASCII flowchart!**

```
╔════════════════════════════════════════════════════════════════════════════
║  🚀 ARR-COC-0-1 SETUP FLOWCHART - Complete 5-Minute Guide
╠════════════════════════════════════════════════════════════════════════════
║
║  📋 Prerequisites: GCP account, W&B account, gcloud CLI, Python 3.8+
║
╠════════════════════════════════════════════════════════════════════════════
║  STEP 1 │ GCP PROJECT + BILLING + AUTH
╠════════════════════════════════════════════════════════════════════════════
║
║  ┌─ Create Project ─────────────────────────────────────────────────────
║  │  🌐 https://console.cloud.google.com
║  │  • Click project dropdown → "New Project"
║  │  • Name: arr-coc-training (or your choice)
║  │  • 📝 SAVE YOUR_PROJECT_ID: (e.g., stable-granite-432)
║  │  • Click "CREATE"
║  │
║  ├─ Note About "environment tag" Warning ─────────────────────────────
║  │  When you run `gcloud config set project` below, you'll see:
║  │  "INFORMATION: Project has no 'environment' tag set..."
║  │
║  │  ✅ Ignore this warning - it's completely fine!
║  │  Tags require GCP Organizations (most users don't have this).
║  │  This warning has ZERO impact on setup, launch, or training.
║  │
║  ├─ 🚨 ENABLE BILLING (CRITICAL!) ─────────────────────────────────────
║  │  🌐 https://console.cloud.google.com/billing
║  │  • Link project to billing account
║  │  • ⚠️  WITHOUT BILLING: HTTP 403 errors, everything fails!
║  │  • 💰 Estimated cost: $20-50/month (GPU training)
║  │
║  ├─ Authenticate GCloud ───────────────────────────────────────────────
║  │  💻 gcloud auth login
║  │     → Opens browser for Google login
║  │
║  │  💻 gcloud config set project YOUR_PROJECT_ID
║  │     → Sets your active GCP project
║  │     → If you see: "INFORMATION: Project has no 'environment' tag"
║  │        This means you skipped the tag step (no organization or optional)
║  │     → ✅ Safe to ignore - zero impact on functionality!
║  │
║  │  💻 gcloud auth application-default login
║  │     → Creates ADC (Application Default Credentials)
║  │     → Required for Python scripts to access GCP
║  │
║  └─ Verify Billing ────────────────────────────────────────────────────
║     💻 gcloud billing projects describe YOUR_PROJECT_ID
║     ✅ Should show: billingEnabled: true
║     ❌ If false: Re-enable billing, wait 2-3 min
║
╠════════════════════════════════════════════════════════════════════════════
║  STEP 2 │ AUTHENTICATE W&B + HUGGINGFACE
╠════════════════════════════════════════════════════════════════════════════
║
║  ┌─ Weights & Biases (Required) ────────────────────────────────────────
║  │  💻 wandb login
║  │     → Get API key from: https://wandb.ai/authorize
║  │     → Paste when prompted
║  │
║  └─ HuggingFace (Optional) ────────────────────────────────────────────
║     💻 huggingface-cli login
║     → For model checkpoint hosting on HF Hub
║
╠════════════════════════════════════════════════════════════════════════════
║  STEP 2.5 │ CONFIGURE TRAINING SETTINGS (1 min)
╠════════════════════════════════════════════════════════════════════════════
║
║  Edit: ARR_COC/Training/.training
║
║  Update these 2 lines with YOUR values:
║
║  GCP_PROJECT_ID="your-project-id-here"  ← Replace with YOUR_PROJECT_ID from Step 1
║
║  WANDB_ENTITY="your-wandb-username"     ← Replace with your W&B username/team
║
║  💾 Save the file!
║
╠════════════════════════════════════════════════════════════════════════════
║  STEP 3 │ RUN SETUP (Creates Infrastructure - 2-5 min)
╠════════════════════════════════════════════════════════════════════════════
║
║  💻 cd /path/to/arr-coc-0-1
║  💻 python CLI/cli.py setup
║
║  ┌─ What Gets Created ──────────────────────────────────────────────────
║  │
║  │  🗄️  Artifact Registry (deletable)
║  │     • arr-coc-registry (us-central1)
║  │     • Stores: training image, launcher image
║  │     • Deleted on teardown
║  │
║  │  🗄️  Artifact Registry (persistent - NEVER deleted)
║  │     • arr-coc-registry-persistent (us-central1)
║  │     • Stores: PyTorch base image (~15GB)
║  │     • Reused across projects (saves rebuild time!)
║  │
║  │  🔑 Service Account
║  │     • arr-coc-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com
║  │     • IAM Roles: Vertex AI, Cloud Build, Storage, Artifact Registry
║  │     • Key file: ~/.gcp-keys/arr-coc-sa.json (auto-created!)
║  │
║  │  📊 W&B Launch Queue
║  │     • vertex-ai-queue
║  │     • Job submission endpoint for Vertex AI
║  │
║  │  📊 W&B Project
║  │     • entity/arr-coc-0-1
║  │     • Training run tracking and metrics
║  │
║  │  🤗 HuggingFace Repo (if configured)
║  │     • user/arr-coc-0-1
║  │     • Model checkpoint hosting
║  │
║  └─ What's Created Later (On-Demand) ──────────────────────────────────
║     • GCS buckets (regional, when ZEUS picks GPU region)
║     • Worker pools (first launch creates)
║     • Docker images (first launch builds: ~30 min total)
║
╠════════════════════════════════════════════════════════════════════════════
║  STEP 4 │ VERIFY (Check Everything Works)
╠════════════════════════════════════════════════════════════════════════════
║
║  💻 python CLI/cli.py infrastructure
║
║  ┌─ Expected Output ─────────────────────────────────────────────────────
║  │
║  │  GCP Infrastructure:
║  │    ✅ Billing: Enabled (billingEnabled: true)
║  │    ✅ Registries: arr-coc-registry + persistent
║  │    ✅ Service Account: arr-coc-sa@...
║  │    ✅ GCS Buckets: 0 (created on-demand)
║  │
║  │  W&B Infrastructure:
║  │    ✅ Queue: vertex-ai-queue
║  │    ✅ Project: entity/arr-coc-0-1
║  │
║  │  Local:
║  │    ✅ Key file: ~/.gcp-keys/arr-coc-sa.json
║  │
║  └─ Common Issues ──────────────────────────────────────────────────────
║     ❌ "Billing: Disabled"
║        → Enable at: console.cloud.google.com/billing
║        → Wait 2-3 min, re-run setup
║     ❌ "Permission denied"
║        → Re-run: gcloud auth login
║        → Re-run: gcloud auth application-default login
║     ❌ "API not enabled"
║        → Auto-enabled by setup, wait 30 sec and retry
║
╠════════════════════════════════════════════════════════════════════════════
║  🎉 READY TO LAUNCH!
╠════════════════════════════════════════════════════════════════════════════
║
║  💻 python CLI/cli.py launch
║     → First launch builds Docker images (~30 min)
║     → Subsequent launches: ~2 min (images cached!)
║
║  💻 python CLI/cli.py monitor
║     → Watch training progress in real-time
║     → See Vertex AI jobs, W&B runs, GPU usage
║
║  🌐 https://wandb.ai/{entity}/arr-coc-0-1
║     → View training metrics, loss curves, system stats
║
╠════════════════════════════════════════════════════════════════════════════
║  🔑 KEY FILE INFO: ~/.gcp-keys/arr-coc-sa.json
╠════════════════════════════════════════════════════════════════════════════
║
║  What: Service account private key (JSON)
║  Why: Needed for teardown operations
║  Security: ⚠️  DO NOT commit to git! (contains private key)
║  Created: Automatically by setup
║  Used by: python CLI/cli.py teardown
║
║  If missing: Re-run setup to recreate
║
╚════════════════════════════════════════════════════════════════════════════
```
