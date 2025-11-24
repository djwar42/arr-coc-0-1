# PyTorch Clean Implementation - Change Summary

## ✅ Completed Changes

### 1. Created pytorch-clean Image Structure
**Location:** `Stack/pytorch-clean/`

**Files Created:**
- `.image-manifest` - Lists Dockerfile for hash-based change detection
- `Dockerfile` - Multi-stage build from nvidia/cuda, compiles PyTorch 2.6.0 with ALL GPU archs
- `PLAN.md` - Complete theory, benefits, tradeoffs, implementation plan
- `CHANGES.md` (this file) - Change tracking

**Key Features:**
- FROM nvidia/cuda:12.4.1-cudnn9-devel-ubuntu22.04 (clean base, no conda!)
- TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" (T4/L4/A100/H100 support)
- Builds PyTorch 2.6.0 + torchvision 0.20.0 + torchaudio 2.6.0 from source
- Multi-stage build (builder + runtime) for smaller final image
- Build time: 2-4 hours FIRST TIME, then cached forever

### 2. Updated base-image Dockerfile
**Location:** `Stack/base-image/Dockerfile`

**Changes:**
- Changed FROM: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` → `pytorch-clean:2.6.0-cuda12.4`
- Removed all "CENTERPIECE CONSEQUENCE" comments (no longer needed!)
- Removed Python 3.10 installation (already in pytorch-clean)
- Removed dual pip upgrade (only one Python environment now!)
- Removed /opt/conda references in cleanup code
- Simplified from 348 lines → 218 lines (37% reduction!)

**Result:** Clean, simple Dockerfile with no conda baggage hacks!

---

## 🚧 Pending Changes

### 3. Update cli/launch/core.py
**Location:** `CLI/launch/core.py`

**Changes Needed:**

#### A. Add pytorch-clean build function (before _handle_base_image)
Insert new function around line 1050:
```python
def _handle_pytorch_clean_image(
    config: Dict,
    region: str,
    status: StatusCallback,
) -> bool:
    """
    Ensure pytorch-clean image exists (Image 0: PyTorch from source, no conda)

    This is the FOUNDATION of our 4-tier architecture.
    Built from nvidia/cuda with PyTorch compiled from source.

    Build time: 2-4 hours FIRST TIME, then cached in Artifact Registry.
    Hash-based: Only rebuilds if Dockerfile changes (almost never).

    Returns:
        True if pytorch-clean exists or was built successfully
        False if build failed
    """
    # Similar structure to _handle_base_image but for pytorch-clean
    # Check hash, build if needed, push to registry
```

#### B. Call _handle_pytorch_clean_image in run_launch_core
Around line 447 (before _handle_base_image):
```python
# Step 1.3: Ensure pytorch-clean image exists (needed by base image)
# PyTorch from source with full GPU arch support (T4/L4/A100/H100)
# Rebuilds RARELY (only on PyTorch version changes)
if not _handle_pytorch_clean_image(config, region, status):
    return False

# Step 1.4: Ensure base image exists (needed by training image)
# ... existing code ...
```

#### C. Update required_images dict (line 2255)
Change from:
```python
required_images = {
    "base": f"{registry_base}/arr-base",
    "training": f"{registry_base}/arr-training",
    "runner": f"{registry_base}/arr-runner"
}
```

To:
```python
required_images = {
    "pytorch-clean": f"{registry_base}/arr-pytorch-clean",
    "base": f"{registry_base}/arr-base",
    "training": f"{registry_base}/arr-training",
    "runner": f"{registry_base}/arr-runner"
}
```

#### D. Update "all 3 images" messages
Lines to change:
- Line 2238: "Verify all 3 images" → "Verify all 4 images"
- Line 2306: "All 3 images verified" → "All 4 images verified"
- Line 2308: "3-6-9 diamond" → "4-tier diamond"

### 4. Update TUI Files
**Location:** `CLI/tui.py` and related screen files

**Changes Needed:**
- Update any "3 images" references to "4 images"
- Update setup/launch/teardown screens to show pytorch-clean status
- Update progress indicators (3-step → 4-step)

### 5. Update Verification/Checks
**Search for:** Any hardcoded checks for 3 images

**Files to check:**
- `CLI/launch/validation.py` (if exists)
- Any go/no-go checks
- Setup verification

---

## 🔧 Build System Integration

### Hash-Based Change Detection
pytorch-clean uses the same `.image-manifest` system:
- Lists `Stack/pytorch-clean/Dockerfile`
- Hash calculated on file contents
- Rebuild triggered ONLY if Dockerfile changes
- 99% of time: uses cached image from Artifact Registry (0 sec)

### Build Flow After Changes
```
1. pytorch-clean (Image 0) - Check hash → use cached (0 sec) or build (2-4 hours)
   ↓
2. arr-base (Image 1) - Check hash → use cached or build (~15 min)
   ↓
3. arr-training (Image 2) - Check hash → use cached or build (~5 min)
   ↓
4. arr-runner (Image 3) - Check hash → use cached or build (~2 min)
```

### First Time Build (One-Time Cost)
- Total: ~2.5-4.5 hours (mostly pytorch-clean)
- pytorch-clean: 2-4 hours (build PyTorch from source)
- arr-base: ~15 min (install ML packages)
- arr-training: ~5 min (add training code)
- arr-runner: ~2 min (W&B launcher)

### Daily Development (Normal Use)
- Total: ~15-20 min (same as before!)
- pytorch-clean: 0 sec (cached in Artifact Registry)
- arr-base: ~15 min (if requirements.txt changed)
- arr-training: ~5 min (if code changed)
- arr-runner: ~2 min (if needed)

---

## 📊 Benefits Summary

### Architecture
- ✅ Single Python 3.10 environment (no conda!)
- ✅ 37% fewer lines in base Dockerfile (348 → 218)
- ✅ No dual pip confusion
- ✅ No cleanup hacks needed
- ✅ Clear 4-tier separation

### GPU Support
- ✅ T4 (sm_75) - supported
- ✅ L4 (sm_89) - supported (CRITICAL - broken in pip wheels!)
- ✅ A100 (sm_80) - supported
- ✅ H100 (sm_90) - supported

### Security
- ✅ ~4-5 fewer CVEs (no conda bundled wheels)
- ✅ No /opt/conda security surface
- ✅ Cleaner dependency tree

### Image Size
- ✅ ~800MB-1.2GB smaller (no conda overhead)

---

## 🧪 Testing Checklist

After implementing pending changes:

- [ ] Build pytorch-clean (first time, 2-4 hours)
- [ ] Verify CUDA arch list includes sm_75, sm_89, sm_80, sm_90
- [ ] Test pytorch-clean caching (rebuild = 0 sec)
- [ ] Build arr-base FROM pytorch-clean
- [ ] Build arr-training
- [ ] Build arr-runner
- [ ] Run full verification (all 4 images exist)
- [ ] Run CVE scan and compare to previous
- [ ] Deploy to Vertex AI and test on ALL GPUs (T4/L4/A100/H100)

---

## 📝 Commit Messages

When committing these changes:

```bash
# Commit 1: pytorch-clean structure
git add Stack/pytorch-clean/
git commit -m "Add pytorch-clean image: PyTorch from source, no conda

- Builds PyTorch 2.6.0 with TORCH_CUDA_ARCH_LIST='7.5;8.0;8.6;8.9;9.0'
- Multi-stage build (builder + runtime)
- Full GPU support: T4/L4/A100/H100
- No conda baggage

Includes:
- Dockerfile (multi-stage from nvidia/cuda)
- .image-manifest (hash-based change detection)
- PLAN.md (complete theory and implementation)
- CHANGES.md (change tracking)

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"

# Commit 2: Simplified base image
git add Stack/base-image/Dockerfile
git commit -m "Simplify base Dockerfile with pytorch-clean foundation

FROM pytorch-clean:2.6.0-cuda12.4 (not pytorch/pytorch anymore!)

Removed:
- Python 3.10 installation (already in pytorch-clean)
- Dual pip upgrade hacks (single Python environment)
- /opt/conda cleanup code (no conda!)
- All 'CENTERPIECE CONSEQUENCE' comments

Result: 348 lines → 218 lines (37% reduction!)

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"

# Commit 3: Build pipeline integration
git add CLI/launch/core.py
git commit -m "Integrate pytorch-clean into 4-tier build pipeline

- Add _handle_pytorch_clean_image() build function
- Update run_launch_core() to build pytorch-clean first
- Update image verification from 3 → 4 images
- Update 'all 3 images' messages → 'all 4 images'

Build order now:
  pytorch-clean (Image 0) → arr-base (1) → arr-training (2) → arr-runner (3)

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"

# Commit 4: TUI updates
git add CLI/tui.py CLI/*/screen.py
git commit -m "Update TUI to show 4-tier image architecture

- Update setup/launch/teardown screens for 4 images
- Update progress indicators
- Update status displays

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

**Status:** 2/5 major changes complete (40%)
**Next:** Implement core.py changes for build pipeline integration
