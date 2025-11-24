# Stack - Docker Images

**Location**: `Stack/`

Four Docker images for ARR-COC training infrastructure, all built automatically via CLI.

---

## 🏗️ Image Hierarchy

```
arr-pytorch-base     → PyTorch compiled from source (2-4 hours)
    ↓
arr-ml-stack         → ML dependencies (10-15 min)
    ↓
arr-trainer          → Training container for Vertex AI

arr-vertex-launcher  → W&B Launch agent for Cloud Run
```

---

## 📦 **arr-pytorch-base/** - PyTorch Base

**Purpose**: PyTorch 2.5 compiled from source with CUDA
**Build time**: 2-4 hours (only when Dockerfile changes)

---

## 📦 **arr-ml-stack/** - ML Stack

**Purpose**: ML dependencies on top of PyTorch base
**Build time**: 10-15 minutes

---

## 📦 **arr-trainer/** - Training Image

**Runs on**: Vertex AI GPU
**Purpose**: Execute ARR-COC training
**Build time**: 10-15 minutes

**Contains**:
- PyTorch 2.5 + CUDA (from arr-ml-stack)
- ARR-COC training code
- Python dependencies

---

## 🚀 **arr-vertex-launcher/** - Launcher Image

**Runs on**: Cloud Run (ephemeral)
**Purpose**: Submit training jobs to Vertex AI

**Contains**:
- W&B Launch Agent
- gcloud CLI (Vertex AI submission)
- Spot instance patching

---

## 🔨 **Building Images**

**ALL images are built automatically via CLI - no manual docker commands needed!**

```bash
# From project root
python CLI/cli.py launch
```

The CLI:
1. Detects Dockerfile/code changes via hash system
2. Rebuilds only what's needed (parent → child order)
3. Pushes to Artifact Registry automatically
4. Updates Cloud Run job if launcher changes

**Hash detection**: Each image has `.image-manifest` listing files that trigger rebuild.

---

## 🔄 **When Images Rebuild**

| Image | Rebuilds When |
|-------|---------------|
| arr-pytorch-base | `Stack/arr-pytorch-base/Dockerfile` changes |
| arr-ml-stack | Parent changes OR `Stack/arr-ml-stack/Dockerfile` changes |
| arr-trainer | Parent changes OR `Stack/arr-trainer/Dockerfile` OR training code changes |
| arr-vertex-launcher | `Stack/arr-vertex-launcher/*` changes |

**See**: `.image-manifest` in each folder for complete file list.

---

## ⚠️ Important Notes

- **NEVER build manually** - always use `python CLI/cli.py launch`
- Hash system requires **git commit** before changes are detected
- Parent image rebuild triggers all child rebuilds
- Launcher image is separate from training stack (different base)
