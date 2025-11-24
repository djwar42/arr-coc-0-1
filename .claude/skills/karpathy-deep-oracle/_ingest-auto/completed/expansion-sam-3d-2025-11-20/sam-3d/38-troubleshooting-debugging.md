# Troubleshooting & Debugging Guide

**Common issues and solutions when using SAM 3D**

---

## 1. Installation Issues

**Problem: CUDA not found**
```
RuntimeError: CUDA not available
```

**Solution:**
- Install CUDA toolkit (11.8+)
- Verify: `nvidia-smi`
- Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

---

## 2. Inference Errors

**Problem: Out of memory (OOM)**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce batch size: `batch_size=1`
- Use FP16: `predictor.half()`
- Smaller input: Resize image to 512×512

**Problem: Slow inference**

**Solutions:**
- Use GPU (not CPU)
- Enable TensorRT: `predictor.enable_tensorrt()`
- Batch multiple images: `predict_batch()`

---

## 3. Mesh Quality Issues

**Problem: Holes in mesh**

**Causes:**
- Extreme occlusion (>80%)
- Transparent objects

**Solutions:**
- Multi-view input (if available)
- Manual mesh repair (Blender, MeshLab)

**Problem: Interpenetration (body parts overlap)**

**Solution:**
- Post-process with penetration loss
- Use physics-based refinement

---

## 4. Pose Estimation Errors

**Problem: Left/right arm flip**

**Cause:** Depth ambiguity

**Solution:**
- Temporal smoothing (if video)
- Multi-view constraints
- Manual correction

**Problem: Unnatural joint angles**

**Solution:**
- Enable pose prior: `use_pose_prior=True`
- Stronger biomechanical constraints

---

## 5. Debugging Tips

**Visualize Intermediate Outputs:**
```python
# 2D keypoints
predictor.visualize_keypoints(image)

# Depth map
depth = predictor.predict_depth(image)

# Mesh rendering
mesh.render(view='front')
```

**Check Input:**
- Image size: Should be >256×256
- Format: RGB (not BGR)
- Range: [0, 255] or [0, 1]

---

## 6. ARR-COC-0-1 Integration (10%)

**Debugging 3D Grounding:**

When ARR-COC spatial reasoning fails:
1. Visualize SAM 3D mesh (check quality)
2. Verify 3D coordinates (not NaN)
3. Check depth consistency (plausible ranges)

---

**Sources:**
- SAM 3D GitHub issues
- Common error messages
- Debugging best practices
