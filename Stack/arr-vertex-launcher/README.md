# arr-vertex-launcher Image

W&B Launch agent that submits training jobs to Vertex AI.

## What's Inside

**Base Image:**
- `python:3.10-slim` (lightweight)

**Installed Packages:**
- `wandb[launch]` - W&B Launch agent
- `google-cloud-aiplatform` - Vertex AI SDK

**Custom Components:**
- `entrypoint-wrapper.sh` - Fatal error detection wrapper
- `wandb_vertex_patch.py` - Spot instance support patch
- `sitecustomize.py` - Auto-applies patches on import

## Purpose

- Runs as Cloud Run Job
- Polls W&B queue for training jobs
- Submits jobs to Vertex AI Custom Jobs
- Monitors job status
- Detects fatal errors (quota, permissions, 500/503)

## 4-Tier Architecture

```
arr-pytorch-base - PyTorch from source
    ↓ FROM
arr-ml-stack - ML packages
    ↓ FROM
arr-trainer - Training code
    ↓ FROM
arr-vertex-launcher (this!) - Launch agent

W&B Queue
    ↓
arr-vertex-launcher (Cloud Run)
    ↓
Vertex AI Custom Job
    ↓
arr-trainer (actual training)
```

## Building

**Automatic (via launch):**
```bash
python CLI/cli.py launch
# Detects hash change → triggers Cloud Build
```

**Manual:**
```bash
gcloud builds submit \
  --config Stack/arr-vertex-launcher/cloudbuild.yaml \
  --substitutions=_IMAGE_TAG=manual-test .
```

## Hash-Based Rebuilding

**Hash includes:**
- `Stack/arr-vertex-launcher/Dockerfile`
- `Stack/arr-vertex-launcher/entrypoint-wrapper.sh`
- `Stack/arr-vertex-launcher/wandb_vertex_patch.py`
- `Stack/arr-vertex-launcher/sitecustomize.py`

See: `.image-manifest` for complete list

**Files must be git committed for hash detection!**

## Deployment

**Where it runs:**
- Cloud Run Jobs (from MECHA winner region)
- CPU-only (no GPU needed)
- Timeout: Configurable (default 3600s)

**Entry point:**
```bash
/entrypoint-wrapper.sh wandb launch-agent \
  --queue vertex-ai-queue \
  --entity {entity}
```

## Fatal Error Detection

The `entrypoint-wrapper.sh` detects:
- Quota exceeded errors
- Permission denied errors
- 500/503 GCP service errors
- Image pull failures
- Python exceptions

When fatal error detected:
1. Shows last 50 lines of context
2. Kills agent process
3. Exits with error code

## Spot Instance Support

The `wandb_vertex_patch.py` enables:
- Spot/preemptible VMs (60-91% cost savings)
- Applied automatically via `sitecustomize.py`

## Image Tag

```
us-central1-docker.pkg.dev/{PROJECT}/arr-coc-registry/arr-vertex-launcher:{hash}
```

Hash is last 7 chars of SHA256 of tracked files.

## Build Context

Uses `Dockerfile.dockerignore` for per-image isolation.

**What gets INCLUDED:**
- `Stack/arr-vertex-launcher/*` files only

**What gets EXCLUDED:**
- `ARR_COC/` (not needed - launcher just submits jobs)
- `ARR_COC/Training/` (not needed)
- `CLI/` (not needed)
- All other Stack images

## Files

- `Dockerfile` - Minimal agent build
- `Dockerfile.dockerignore` - Per-image build exclusions
- `.image-manifest` - Files that trigger rebuilds
- `entrypoint-wrapper.sh` - Fatal error detection
- `wandb_vertex_patch.py` - Spot instance support
- `sitecustomize.py` - Auto-patch on import
- `launch-rebuilt.log` - Build history
- `README.md` - This file

## Troubleshooting

**Agent not picking up jobs:**
- Check W&B queue name matches
- Verify entity/project correct
- Check Cloud Run logs

**Quota exceeded:**
- Check `python CLI/cli.py monitor --vertex-runner`
- Look for quota error in Note column
- Request quota increase in GCP Console

**Permission denied:**
- Verify service account has Vertex AI User role
- Check IAM bindings in GCP Console

**Image pull failures:**
- Ensure arr-trainer image exists in registry
- Check image hash matches what launcher expects

**500/503 errors:**
- GCP service issue
- Retry usually works
- Check GCP status page
