# arr-pytorch-base Image

Custom PyTorch 2.6.0 built from source with full GPU architecture support.

## What's Inside

**Base Image:**
- `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` (builder)
- `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` (runtime)

**Built from Source:**
- PyTorch 2.6.0
- torchvision 0.20.0
- torchaudio 2.6.0

**GPU Architecture Support:**
- sm_75: T4 (Turing)
- sm_80: A100 (Ampere)
- sm_86: A6000, RTX 3090
- sm_89: L4 (Ada Lovelace) ← **Missing from pip wheels!**
- sm_90: H100 (Hopper)

## Why Build from Source?

**Problem with pip wheels:**
- Missing sm_89 (L4 GPU) support
- Conda baggage in official pytorch/pytorch images

**Our solution:**
- Build with `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"`
- No conda! Single Python 3.10 environment
- ~4-5 fewer CVEs (no bundled wheels)
- ~800MB-1.2GB smaller

## 4-Tier Architecture

```
arr-pytorch-base (this!) - PyTorch from source, no conda
    ↓ FROM
arr-ml-stack - ML packages (transformers, wandb, etc.)
    ↓ FROM
arr-trainer - ARR-COC training code
    ↓ FROM
arr-vertex-launcher - W&B Launch agent
```

## Building

**Build Time:**
- ~10-15 minutes on 176-vCPU (Cloud Build c3-standard-176)
- ~4+ hours on 4-vCPU

**This image is built ONCE and cached in Artifact Registry!**

**Automatic (via launch):**
```bash
python CLI/cli.py launch
# Only rebuilds if Dockerfile changes
```

**Manual (Cloud Build):**
```bash
gcloud builds submit \
  --config Stack/arr-pytorch-base/cloudbuild.yaml \
  --substitutions=_MAX_JOBS=176 .
```

## Image Tag

```
us-central1-docker.pkg.dev/{PROJECT}/arr-coc-registry-persistent/arr-pytorch-base:latest
```

This image lives in the **persistent** registry (not rebuilt daily).

## Hash-Based Rebuilding

**Hash includes:**
- `Stack/arr-pytorch-base/Dockerfile`

**When Dockerfile changes → full rebuild (4+ hours)**

## CHONK Progress Markers

Build progress tracked with gem harmonics:
- 10% - Git clone complete (Sapphire)
- 15% - CMake patches applied (Diamond, Aquamarine)
- 70% - PyTorch built! First Harmonic (Ruby, Emerald, Topaz)
- 85% - torchvision built! Double Harmonic
- 100% - torchaudio built! Triple Harmonic

## Key Features

**Two-stage build:**
1. **Builder stage**: Full build tools, compile everything
2. **Runtime stage**: Minimal image, just Python + PyTorch

**ccache enabled:**
- C++ compilation cache (10-100× speedup on rebuilds)
- Persisted in `/ccache` mount

**CMake patches:**
- All `third_party/CMakeLists.txt` patched to VERSION 3.5
- Fixes Ubuntu 22.04 CMake 3.22 compatibility

## Verification

After build, verify GPU architectures:
```bash
docker run arr-pytorch-base:latest python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'Arch list: {torch.cuda.get_arch_list()}')
"
```

Should show: `['sm_75', 'sm_80', 'sm_86', 'sm_89', 'sm_90']`

## Troubleshooting

**Build timeout:**
- Increase MAX_JOBS (default=4)
- Use c3-standard-176 for 10-15 min builds

**CMake version errors:**
- Check CMake patch ran (find sed output in logs)
- Verify protobuf patched: `cat /opt/pytorch/third_party/protobuf/cmake/CMakeLists.txt`

**Import errors in runtime:**
- Ensure cuDNN runtime base image used
- Check PYTHONPATH set correctly

## Files

- `Dockerfile` - Two-stage build (builder + runtime)
- `Dockerfile.dockerignore` - Build exclusions
- `CHANGES.md` - Detailed lessons learned
- `README.md` - This file

## Lessons Learned

See `CHANGES.md` for detailed debugging journey including:
- CUPTI investigation (6 build attempts!)
- Staging bucket retry logic
- 3 best debugging methods for Docker builds
