# Vertex AI Agent Setup

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  YOUR MACHINE (Local)                                       │
│  ┌──────────────┐                                           │
│  │ Launch Screen│ ─── Submit Job ───┐                       │
│  └──────────────┘                    │                       │
│  ┌──────────────┐                    │                       │
│  │Monitor Screen│ ─── View Status ───┤                       │
│  └──────────────┘                    │                       │
└────────────────────────────────────────│────────────────────┘
                                        │
                                        ↓
┌─────────────────────────────────────────────────────────────┐
│  W&B CLOUD                                                  │
│  ┌──────────────────────────────────┐                       │
│  │  Launch Queue                    │                       │
│  │  vertex-ai-queue                 │                       │
│  │                                  │                       │
│  │  ┌────────┐ ┌────────┐          │                       │
│  │  │ Job 1  │ │ Job 2  │          │                       │
│  │  │pending │ │pending │          │                       │
│  │  └────────┘ └────────┘          │                       │
│  └──────────────────────────────────┘                       │
└────────────────────────────────────────│────────────────────┘
                                        │
                                        │ Agent polls queue
                                        │
                                        ↓
┌─────────────────────────────────────────────────────────────┐
│  GOOGLE CLOUD PLATFORM                                      │
│  ┌──────────────────────────────────┐                       │
│  │  W&B Launch Agent                │                       │
│  │  (Cloud Run or Compute Engine)   │                       │
│  │                                  │                       │
│  │  • Polls queue every 30s         │                       │
│  │  • Picks up jobs                 │                       │
│  │  • Submits to Vertex AI          │                       │
│  └──────────────────────────────────┘                       │
│                    │                                         │
│                    ↓                                         │
│  ┌──────────────────────────────────┐                       │
│  │  Vertex AI Training              │                       │
│  │                                  │                       │
│  │  • Builds Docker image           │                       │
│  │  • Provisions GPU (L4/A100)      │                       │
│  │  • Runs training job             │                       │
│  │  • Logs to W&B                   │                       │
│  └──────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

All of these should already be set up if you ran `training/setup.sh`:

```bash
# Check setup status
python trainer.py setup
```

Required:
- ✓ GCP project configured
- ✓ W&B Launch queue created (vertex-ai-queue)
- ✓ Service account with permissions
- ✓ GCS bucket for artifacts
- ✓ Artifact Registry repository

### Deploy Agent

```bash
cd training/
./deploy-agent.sh
```

**Choose deployment option:**

**Option 1: Cloud Run (Recommended)** ✓
- Always-on service
- Auto-scales (min: 1, max: 1 instance)
- Managed by Google
- $5-10/month

**Option 2: Compute Engine**
- VM-based agent
- More control
- $30-50/month

**Option 3: Manual**
- View instructions only
- Deploy yourself

### Test the Setup

1. **Submit a test job:**
   ```bash
   python trainer.py launch
   # Click "Submit (s)"
   ```

2. **Monitor progress:**
   ```bash
   python trainer.py monitor
   ```

3. **Expected flow:**
   ```
   Job State Timeline:
   ├─ pending  (in queue, waiting for agent)
   ├─ leased   (agent picked it up)
   ├─ running  (training on Vertex AI)
   └─ finished (training complete)
   ```

## Agent Management

### View Agent Status

**From CLI:**
```bash
# Cloud Run logs
gcloud run logs read wandb-launch-agent --region=us-central1

# Compute Engine (if used)
gcloud compute ssh wandb-launch-agent --zone=us-central1-a
journalctl -u wandb-agent -f
```

**From Monitor Screen:**
```bash
python trainer.py monitor
# Shows agent status at top:
# ● Agent Running · Queue: vertex-ai-queue
```

### Stop/Start Agent

**Cloud Run:**
```bash
# Stop (scale to 0)
gcloud run services update wandb-launch-agent \
  --region=us-central1 \
  --min-instances=0

# Start (scale to 1)
gcloud run services update wandb-launch-agent \
  --region=us-central1 \
  --min-instances=1
```

**Compute Engine:**
```bash
# Stop VM
gcloud compute instances stop wandb-launch-agent --zone=us-central1-a

# Start VM
gcloud compute instances start wandb-launch-agent --zone=us-central1-a
```

### Update Agent

**Cloud Run:**
```bash
gcloud run services update wandb-launch-agent \
  --image=wandb/launch-agent:latest \
  --region=us-central1
```

## Troubleshooting

### Jobs stuck in "pending" state

**Cause:** Agent not running or can't access queue

**Fix:**
```bash
# Check agent logs
gcloud run logs read wandb-launch-agent --region=us-central1

# Verify agent can access queue
wandb launch-agent --dry-run -q vertex-ai-queue -e newsofpeace2
```

### Jobs fail with "PERMISSION_DENIED"

**Cause:** Service account lacks permissions

**Fix:**
```bash
# Re-run setup
cd training/
./setup.sh
```

### Jobs stuck in "leased" state

**Cause:** Agent picked up job but Vertex AI failed to start

**Fix:**
```bash
# Check Vertex AI quotas
gcloud compute project-info describe --project=YOUR_PROJECT

# Check GPU availability
gcloud compute accelerator-types list --filter="zone:us-central1"
```

## Cost Estimates

**Agent (Cloud Run):**
- Always-on agent: ~$5-10/month
- Negligible CPU usage (just polls queue)

**Training (Vertex AI):**
- L4 GPU: ~$0.90/hour
- A100 GPU: ~$3.67/hour
- Preemptible: 60-80% discount

**Storage:**
- GCS bucket: ~$0.02/GB/month
- Artifact Registry: Free tier (0.5GB)

**Example monthly cost:**
- Agent: $10
- 10 hours training (L4): $9
- Storage (50GB): $1
- **Total: ~$20/month**

## Configuration

All settings in `.training` file:

```bash
# GCP Settings
GCP_PROJECT_ID=your-project
GCP_REGION=us-central1
GCS_BUCKET_NAME=arr-coc-0-1-data

# W&B Settings
WANDB_ENTITY=newsofpeace2
WANDB_PROJECT=arr-coc-0-1
WANDB_LAUNCH_QUEUE_NAME=vertex-ai-queue

# Training Settings
BASE_MODEL=Qwen/Qwen3-VL-2B-Instruct
NUM_EPOCHS=1
BATCH_SIZE=2
LEARNING_RATE=1e-4
```

## Next Steps

1. Deploy agent: `./training/deploy-agent.sh`
2. Submit test job: `python trainer.py launch`
3. Monitor progress: `python trainer.py monitor`
4. Check W&B dashboard: https://wandb.ai/newsofpeace2/arr-coc-0-1

---

**Need help?** Check agent logs or re-run `./training/setup.sh`
