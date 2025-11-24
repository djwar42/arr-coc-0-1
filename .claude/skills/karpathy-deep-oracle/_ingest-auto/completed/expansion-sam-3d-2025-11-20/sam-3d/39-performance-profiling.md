# Performance Profiling & Optimization

**Measuring and optimizing SAM 3D inference performance**

---

## 1. Profiling Tools

**PyTorch Profiler:**
```python
from torch.profiler import profile

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    mesh = predictor.predict(image)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**NVIDIA Nsight:**
- Kernel-level profiling
- Memory bandwidth analysis
- Tensor Core utilization

---

## 2. Bottleneck Analysis

**Common Bottlenecks:**

1. **Image Preprocessing:** 5ms
   - Resize, normalize, to_tensor
   - Optimize: Precompute batches

2. **Model Inference:** 28ms
   - Vision encoder: 15ms
   - Mesh decoder: 13ms
   - Optimize: TensorRT, FP16

3. **Post-Processing:** 2ms
   - Mesh simplification
   - Optimize: Vectorize operations

---

## 3. Optimization Strategies

**FP16 (Half Precision):**
- 2× faster
- <1% accuracy loss
- Enable: `predictor.half()`

**Quantization (INT8):**
- 4× faster
- 2-3% accuracy loss
- Requires calibration

**Kernel Fusion:**
- Combine ops (resize + normalize)
- Reduce memory transfers

---

## 4. Memory Profiling

**GPU Memory Usage:**
```python
import torch

print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

**Typical Usage:**
- Model weights: 1.2 GB (FP32)
- Activations: 0.5 GB (batch=1)
- **Total: 1.7 GB**

**Optimization:**
- FP16: 0.6 GB weights
- Gradient checkpointing: 0.3 GB activations

---

## 5. Batching Performance

**Throughput vs Latency:**

| Batch Size | Latency (ms) | Throughput (FPS) |
|------------|--------------|------------------|
| 1          | 30           | 33               |
| 4          | 60           | 66 (4/0.06)      |
| 8          | 100          | 80 (8/0.1)       |

**Trade-off:**
- Larger batch: Higher throughput, higher latency
- Choose based on use case (real-time vs offline)

---

## 6. ARR-COC-0-1 Integration (10%)

**Profiling in Production:**

Monitor SAM 3D in ARR-COC pipeline:
- P50/P95/P99 latency
- Throughput under load
- Memory usage spikes

---

**Sources:**
- PyTorch profiling guide
- NVIDIA optimization best practices
- Performance benchmarking methodology
