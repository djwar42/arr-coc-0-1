# arr-trainer Image

Docker image for ARR-COC model training on Vertex AI.

## What's Inside

**Base Image:**
- `arr-ml-stack` (our custom ML stack on PyTorch base)
- Includes: PyTorch 2.4, CUDA, cuDNN, Python 3.10, ML packages

**Installed via pip install -e .:**
- `ARR_COC` - ARR-COC model package (installed as editable)

**Copied Code:**
- `ARR_COC/` - ARR-COC model package
- `ARR_COC/Training/` - Training scripts and configs

**Environment:**
- `PYTHONUNBUFFERED=1` - Immediate output
- `CUDA_VISIBLE_DEVICES=0` - GPU selection

## Hash-Based Rebuilding

This image uses **content hashing via .image-manifest** to detect changes.

**Hash includes:**
1. `Stack/arr-trainer/Dockerfile` - Build instructions
2. `Stack/arr-trainer/pyproject.toml` - Package config
3. `Stack/arr-ml-stack/Dockerfile` - Base image
4. `ARR_COC/**/*.py` - Model code
5. `ARR_COC/Training/train.py` - Training script

**When ANY of these change → automatic rebuild!**

See: `.image-manifest` for complete list

## Building

**Automatic (recommended):**
```bash
python CLI/cli.py launch
# Detects hash change → triggers Cloud Build automatically
```

**Manual (for testing):**
```bash
# Cloud Build (from project root)
gcloud builds submit \
  --config Stack/arr-trainer/.cloudbuild.yaml \
  --substitutions=_IMAGE_TAG=manual-test .

# Local (Mac requires platform flag)
docker build --platform linux/amd64 \
  -f Stack/arr-trainer/Dockerfile \
  -t arr-trainer:local .
```

## Image Tag Format

Images are tagged with content hash:
```
us-central1-docker.pkg.dev/{PROJECT}/arr-coc-registry/arr-trainer:{hash}
```

Example: `arr-trainer:fb83ac1`

- Hash is last 7 chars of SHA256
- Guarantees image matches code exactly
- No ambiguity like `:latest`

## Deployment

**Where it runs:**
- Vertex AI Custom Jobs (GCP)
- Machine type: From MECHA winner region
- GPU: L4, T4, A100, or H100 (based on availability)

**Entry point:**
```bash
accelerate launch ARR_COC/Training/train.py
```

## Troubleshooting

**ModuleNotFoundError: No module named 'ARR_COC'**
- Check `pip install -e .` succeeded in Dockerfile
- Verify pyproject.toml has correct package name

**Image build fails:**
- Check Cloud Build logs in GCP Console
- Verify Dockerfile COPY paths are correct
- Ensure Artifact Registry exists

**Hash doesn't change when code changes:**
- Verify file is listed in `.image-manifest`
- Remember: Files must be git committed for hash detection!
- See CLAUDE.md section on .image-manifest

## Build Context Exclusions

This image has its own `Dockerfile.dockerignore` (per-image isolation).

**What gets INCLUDED:**
- `ARR_COC/` - Model package
- `ARR_COC/Training/` - Training code
- `Stack/arr-trainer/pyproject.toml` - Package config

**What gets EXCLUDED:**
- `CLI/` - Local CLI tools
- `HFApp/` - Gradio app
- `Stack/` (other images)
- `.git/`, `__pycache__/`, `wandb/`, etc.

See: `Dockerfile.dockerignore` for complete list

## Files

- `Dockerfile` - Image build instructions
- `Dockerfile.dockerignore` - Per-image build exclusions
- `.image-manifest` - Files that trigger rebuilds
- `pyproject.toml` - Python package configuration
- `README.md` - This file
