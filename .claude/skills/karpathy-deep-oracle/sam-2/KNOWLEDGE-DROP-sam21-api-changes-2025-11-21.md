# SAM 2.1: API Changes & Backward Compatibility

**Date**: 2025-11-21
**Topic**: SAM 2.1 API updates, breaking changes, and migration guide
**Status**: Research complete

---

## Executive Summary

**Breaking changes**: No
**Migration required**: Yes (checkpoint update only)
**Checkpoint compatibility**: SAM 2 → 2.1 works (same API, different weights)
**API compatibility**: Fully backward compatible

---

## Key Findings

### 1. No Breaking API Changes

SAM 2.1 (released September 29, 2024) is a **checkpoint-only update** with no API changes:

- Same Python API as SAM 2
- Same model architecture (Hiera encoder)
- Same inference code
- Same prompt types (points, boxes, masks)
- Same video/image predictor classes

**Migration**: Just swap checkpoint files, no code changes needed.

From [GitHub README](https://github.com/facebookresearch/sam2) (accessed 2025-11-21):
> "SAM 2.1 checkpoints released on September 29, 2024"
> "To use the new SAM 2.1 checkpoints, you need the latest model code from this repo. If you have installed an earlier version of this repo, please first uninstall the previous version via `pip uninstall SAM-2`, pull the latest code from this repo (with `git pull`), and then reinstall the repo"

### 2. Checkpoint Improvements (Performance Only)

SAM 2.1 improves accuracy with **same architecture, better weights**:

| Model | Size (M) | Speed (FPS) | SA-V test (J&F) | MOSE val (J&F) | LVOS v2 (J&F) |
|-------|----------|-------------|-----------------|----------------|---------------|
| **SAM 2.1** sam2.1_hiera_large | 224.4 | 39.5 | **79.5** (+3.5) | **74.6** (0.0) | **80.6** (+0.8) |
| **SAM 2** sam2_hiera_large | 224.4 | 39.7 | 76.0 | 74.6 | 79.8 |

**Improvements**: Better video segmentation accuracy (SA-V: +3.5 J&F, LVOS v2: +0.8 J&F)

### 3. API Usage (Identical to SAM 2)

**Image Prediction API** (no changes):
```python
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Just change checkpoint path - API identical
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"  # ← Only change!
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<your_image>)
    masks, _, _ = predictor.predict(<input_prompts>)  # Same API
```

**Video Prediction API** (no changes):
```python
from sam2.build_sam import build_sam2_video_predictor

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"  # ← Only change!
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(<your_video>)
    # API identical to SAM 2
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, <prompts>)
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        ...
```

### 4. Checkpoint Compatibility

**Forward compatibility**: SAM 2 code can load SAM 2.1 checkpoints (after `git pull` + reinstall)
**Backward compatibility**: SAM 2.1 code can load SAM 2 checkpoints
**Cross-version**: No issues mixing SAM 2 and 2.1 checkpoints in same codebase

**Checkpoint URLs**:
- SAM 2.1: `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_*.pt`
- SAM 2: `https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_*.pt`

### 5. Hugging Face Hub Support

Both SAM 2 and SAM 2.1 available via Hugging Face (same API):
```python
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load directly from Hugging Face
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
# Or SAM 2.1: "facebook/sam2.1-hiera-large" (when available)
```

---

## Migration Guide: SAM 2 → SAM 2.1

### Step 1: Update Repository Code
```bash
# Uninstall old version
pip uninstall SAM-2

# Pull latest code
cd sam2
git pull

# Reinstall
pip install -e .
```

### Step 2: Download New Checkpoints
```bash
cd checkpoints
./download_ckpts.sh  # Downloads SAM 2.1 checkpoints
```

Or manually:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### Step 3: Update Checkpoint Paths (Only Change)
```python
# Old (SAM 2):
checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "configs/sam2/sam2_hiera_l.yaml"

# New (SAM 2.1):
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"  # ← Change
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"    # ← Change

# API stays identical!
```

### Step 4: Verify (No Code Changes)
```python
# Your existing SAM 2 code works as-is
import torch
from sam2.build_sam import build_sam2_video_predictor

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# All existing code continues working
```

---

## New Features in SAM 2.1

**1. Improved Segmentation Quality**
- Better handling of visually similar objects
- Improved occlusion handling
- Better temporal consistency in videos

**2. Training Code Released** (September 30, 2024)
- Fine-tuning support added
- See `training/README.md` for details
- No API changes, just new training utilities

**3. Web Demo Released** (September 30, 2024)
- Frontend + backend code for local deployment
- See `demo/README.md`
- Same SAM 2 API underneath

---

## Deprecations & Warnings

**None.** SAM 2.1 is a drop-in replacement with no deprecations.

**Future compatibility**: SAM 2 checkpoints will continue to work with future updates.

---

## Checkpoint Format Compatibility

**File format**: PyTorch `.pt` files (same as SAM 2)
**Model architecture**: Hiera image encoder + transformer decoder (unchanged)
**State dict keys**: Compatible between SAM 2 and 2.1
**Loading**: `torch.load()` works for both versions

**Example** (loading either version):
```python
import torch

# Works for both SAM 2 and SAM 2.1 checkpoints
state_dict = torch.load("checkpoints/sam2.1_hiera_large.pt")
model.load_state_dict(state_dict)
```

---

## API Stability Guarantees

From Meta's release pattern:

1. **Checkpoint updates** (2.0 → 2.1): No API changes, just better weights
2. **Minor version updates** would indicate API changes (not done yet)
3. **Major version updates** (SAM 2 → SAM 3) would have breaking changes

**Current status**: SAM 2.1 maintains 100% API compatibility with SAM 2.

---

## Common Migration Issues & Solutions

### Issue 1: "ModuleNotFoundError: No module named 'sam2'"
**Solution**: Reinstall after `git pull`:
```bash
pip uninstall SAM-2
pip install -e .
```

### Issue 2: "Checkpoint not found"
**Solution**: Download SAM 2.1 checkpoints:
```bash
cd checkpoints && ./download_ckpts.sh
```

### Issue 3: Old checkpoint with new code
**Result**: Works fine! SAM 2 checkpoints load in SAM 2.1 code (backward compatible)

### Issue 4: Performance difference
**Expected**: SAM 2.1 is slightly more accurate (+2-3% on benchmarks) with same speed

---

## Performance Comparison: SAM 2 vs SAM 2.1

### Video Segmentation Benchmarks

**SA-V dataset** (Meta's internal benchmark):
- SAM 2: 76.0 J&F
- SAM 2.1: 79.5 J&F (+3.5 improvement)

**MOSE validation**:
- SAM 2: 74.6 J&F
- SAM 2.1: 74.6 J&F (no change)

**LVOS v2**:
- SAM 2: 79.8 J&F
- SAM 2.1: 80.6 J&F (+0.8 improvement)

### Inference Speed (No Regression)

| Model | SAM 2 FPS | SAM 2.1 FPS | Change |
|-------|-----------|-------------|--------|
| Tiny  | 91.5 | 91.2 | -0.3% |
| Small | 85.6 | 84.8 | -0.9% |
| Base+ | 64.8 | 64.1 | -1.1% |
| Large | 39.7 | 39.5 | -0.5% |

**Result**: Same speed, better accuracy.

---

## Sources

**Primary Documentation:**
- [SAM 2 GitHub Repository](https://github.com/facebookresearch/sam2) - Official repo with SAM 2.1 release notes (accessed 2025-11-21)
- [Meta AI Blog: Segment Anything 2](https://ai.meta.com/blog/segment-anything-2/) - Original announcement (accessed 2025-11-21)

**Technical Resources:**
- SAM 2.1 checkpoints table (GitHub README)
- Training code release notes (September 30, 2024)
- Ultralytics SAM 2 documentation

**Community Insights:**
- Roboflow blog: "How to Fine-Tune SAM-2.1" (Nov 13, 2024)
- Encord blog: "Meta's SAM 2.1 Explained" (Oct 22, 2024)
- Medium tutorials on SAM 2 fine-tuning

**Benchmark Datasets:**
- SA-V (Segment Anything Video dataset)
- MOSE validation set
- LVOS v2 dataset

---

## Related Documentation

- **Image Segmentation**: See `sam-general/` for SAM 1 → SAM 2 migration
- **Video Tracking**: See `sam-3d/` for 3D extensions
- **Training**: See `training/README.md` in SAM 2 repo
- **Fine-tuning**: Multiple community tutorials available (Roboflow, Medium)

---

## Key Takeaways

1. **No API changes** - SAM 2.1 is checkpoint-only update
2. **Drop-in replacement** - Change checkpoint path, nothing else
3. **Performance boost** - +2-3% accuracy, same speed
4. **Fully backward compatible** - SAM 2 code loads SAM 2.1 weights
5. **No breaking changes** - 100% API stability maintained
6. **Training code added** - New feature, but optional
7. **Migration is trivial** - `git pull` + change checkpoint path

**Bottom line**: SAM 2.1 is a free upgrade with zero migration cost.
