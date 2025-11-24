# arr-ml-stack Image

ML packages layer built on our custom PyTorch base.

## What's Inside

**Base Image:**
- `arr-pytorch-base:latest` (PyTorch 2.6.0 from source, no conda!)

**Installed Packages:**
- `accelerate>=1.4.0` - Multi-GPU training
- `wandb[launch]>=0.19.0` - Experiment tracking + Launch agent
- `transformers>=4.53.0` - HuggingFace models
- `huggingface-hub>=0.27.0` - Model uploads
- `datasets>=2.21.0` - Dataset loading
- `google-cloud-storage>=2.19.0` - GCS integration
- `google-cloud-aiplatform>=1.75.0` - Vertex AI SDK
- `peft>=0.14.0` - Parameter-efficient fine-tuning
- `kornia>=0.7.4` - Computer vision ops
- `pillow>=11.3.0` - Image processing
- `safetensors>=0.4.7` - Safe tensor serialization

**Security Packages (CVE fixes):**
- `pip>=25.3`, `setuptools>=70.0`
- `urllib3>=2.5.0`, `certifi>=2024.12.14`
- `h11>=0.16.0`, `jinja2>=3.1.6`
- `h2>=4.3`, `brotli>=1.2.0`, `lief>=0.15`

## 4-Tier Architecture

```
arr-pytorch-base - PyTorch from source, no conda
    ↓ FROM
arr-ml-stack (this!) - ML packages (transformers, wandb, etc.)
    ↓ FROM
arr-trainer - ARR-COC training code
    ↓ FROM
arr-vertex-launcher - W&B Launch agent
```

## Security Features

**Ubuntu Security Patches:**
- CVE-2025-5245, CVE-2025-5244 (binutils)
- 12-15 additional LOW severity CVEs

**Build Tools Removed (post-install):**
- binutils, gcc-11, gcc-12
- Eliminates 8 CVEs, saves ~250MB

**Test Files Removed:**
- Python test directories with vulnerable setuptools
- Mitigates CVE-2025-47273, CVE-2024-6345, CVE-2022-40897

## Building

**Automatic (via launch):**
```bash
python CLI/cli.py launch
# Detects hash change → triggers Cloud Build
```

**Manual:**
```bash
gcloud builds submit \
  --config Stack/arr-ml-stack/cloudbuild.yaml \
  --substitutions=_IMAGE_TAG=manual-test .
```

## Hash-Based Rebuilding

**Hash includes:**
- `Stack/arr-ml-stack/Dockerfile`
- `Stack/arr-pytorch-base/Dockerfile` (base changes → rebuild)

**When Dockerfile changes → automatic rebuild**

See: `.image-manifest` for complete list

## CHONK Progress Markers

Build progress tracked with unique gems:
- 20% - System tools installed (Quartz)
- 35% - Security patches applied (Jade)
- 65% - ML packages loaded (Amethyst, Pearl)
- 85% - Build tools purged (Opal)
- 95% - Test files cleansed (Moonstone)
- 100% - Stack complete! Triple Harmonic

## Image Tag

```
us-central1-docker.pkg.dev/{PROJECT}/arr-coc-registry/arr-ml-stack:{hash}
```

Hash is last 7 chars of SHA256 of tracked files.

## Verification

After build, verify packages:
```bash
docker run arr-ml-stack:latest python -c "
import torch; print(f'PyTorch: {torch.__version__}')
import transformers; print(f'Transformers: {transformers.__version__}')
import wandb; print(f'W&B: {wandb.__version__}')
import accelerate; print(f'Accelerate: {accelerate.__version__}')
"
```

## GPU Support

Inherited from arr-pytorch-base:
- T4 (sm_75)
- L4 (sm_89)
- A100 (sm_80)
- H100 (sm_90)

## Files

- `Dockerfile` - Package installation + security hardening
- `Dockerfile.dockerignore` - Build exclusions
- `Dockerfile.google-vertex-ai-ARCHIVED` - Old approach (deprecated)
- `README.md` - This file

## Troubleshooting

**Package version conflicts:**
- Check requirements are pinned correctly
- Look for pip resolver warnings in build logs

**CVE scan failures:**
- Update pinned versions to latest secure
- Run security patches with `apt-get upgrade`

**Import errors:**
- Verify base image has correct PYTHONPATH
- Check package was actually installed (grep build logs)
