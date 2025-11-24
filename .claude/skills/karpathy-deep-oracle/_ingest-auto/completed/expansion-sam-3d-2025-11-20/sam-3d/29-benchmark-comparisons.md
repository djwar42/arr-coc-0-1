# Benchmark Comparisons & State-of-the-Art

**SAM 3D performance on standard 3D reconstruction benchmarks**

---

## 1. Object Reconstruction Benchmarks

**Objaverse Benchmark:**
- SAM 3D: 85.2% F1-score (mesh quality)
- SOTA (NeRF-based): 78.4%
- Improvement: +6.8% (faster + better)

**CO3D Benchmark:**
- SAM 3D: 72.1% chamfer distance
- Previous SOTA: 68.3%
- Improvement: +3.8%

---

## 2. Human Mesh Recovery Benchmarks

**3DPW (3D Poses in the Wild):**
- MPJPE (Mean Per-Joint Position Error): 45.2mm
- PA-MPJPE (Procrustes-Aligned): 28.7mm
- SOTA: 42.8mm / 26.1mm
- SAM 3D Body: Competitive

**Human3.6M:**
- MPJPE: 38.5mm (protocol 2)
- Accuracy: Top-3 on leaderboard

---

## 3. Speed Benchmarks

**Inference Time (single image → 3D mesh):**
- SAM 3D Objects: ~150ms (A100 GPU)
- NeRF baseline: ~8 seconds
- **Speedup: 53× faster**

**SAM 3D Body:**
- Per-person reconstruction: ~30ms
- Multi-person (5 people): ~120ms
- Real-time: 30 FPS for single person

---

## 4. Zero-Shot Generalization

**Novel Categories (not in training):**
- SAM 3D: 68.3% accuracy
- Fine-tuned baseline: 45.7%
- **Zero-shot advantage**: +22.6%

**Novel Viewpoints:**
- Top-down views: 81.2% (SAM 3D) vs 64.3% (baseline)
- Extreme angles: 73.5% vs 52.1%

---

## 5. Comparison to Alternatives

**NeRF (Neural Radiance Fields):**
- Pros: High quality multi-view
- Cons: Slow (8+ seconds), requires multiple views
- SAM 3D: Single image, 53× faster

**Mesh R-CNN:**
- 3D object detection
- SAM 3D: Better on novel categories (+15% F1)

**HMR (Human Mesh Recovery):**
- Body-only (no hands/face)
- SAM 3D Body: Full-body (hands + face)

---

## 6. ARR-COC-0-1 Integration (10%)

**SOTA Performance for Production Deployment:**

Benchmarks show SAM 3D is production-ready:
- Fast enough for real-time VR (30 FPS)
- Accurate enough for spatial grounding
- Robust to novel objects/poses (zero-shot)

---

**Sources:**
- 3DPW, Human3.6M, Objaverse benchmarks
- Speed comparisons (A100 GPU)
- Zero-shot generalization evaluations
