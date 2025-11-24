# KNOWLEDGE DROP: TPU Programming Fundamentals

**Runner**: PART 16
**Timestamp**: 2025-11-13 23:45
**Status**: ✓ COMPLETE

---

## Knowledge File Created

**File**: `karpathy/alternative-hardware/03-tpu-programming-fundamentals.md`
**Size**: 559 lines (42,864 bytes)
**Topic**: Google TPU architecture, programming frameworks (JAX, PyTorch XLA), and optimization strategies

---

## Knowledge Acquired

### TPU Architecture Fundamentals

**Core insight**: TPUs are specialized matrix multiplication machines
- TensorCore = MXU (matrix unit) + VPU (vector unit) + VMEM (fast memory)
- Systolic array: 128x128 ALUs (v4/v5) or 256x256 (v6e Trillium)
- Memory hierarchy: VMEM (fastest, 22x HBM) → HBM → ICI → PCIe → DCN

**Performance numbers** (TPU v5p example):
- FLOPs: 4.59e14 bf16 FLOPs/s per chip (9.18e14 int8)
- HBM: 96GB capacity, 2.8TB/s bandwidth
- ICI: 180 GB/s bidirectional between nearest neighbors

### TPU Generations Compared

| Model | Year | HBM | FLOPs (bf16) | Key Feature |
|-------|------|-----|--------------|-------------|
| v3 | 2018 | 32GB | 1.4e14 | 2D torus |
| v4p | 2021 | 32GB | 2.75e14 | 3D torus |
| v5p | 2023 | 96GB | 4.59e14 | 3× HBM vs v4 |
| v5e | 2023 | 16GB | 1.97e14 | Inference-optimized |
| v6e | 2025 | 32GB | 9.20e14 | 256x256 systolic array |

**v5p improvements**: 2× FLOPs and 3× HBM vs v4 (from Google Cloud blog, Dec 2023)

### Programming Frameworks

**JAX on TPU** (primary framework):
- NumPy-like interface with automatic XLA compilation
- `pmap` for data parallelism, `pjit` for model parallelism
- Native TPU support (XLA is the TPU compiler)
- Free access: Colab, TPU Research Cloud

**PyTorch XLA**:
- PyTorch → XLA IR → TPU code
- Lazy execution with graph compilation
- Critical: `xm.mark_step()` to trigger compilation
- Multi-core: `xmp.spawn()` for distributed training

**XLA compiler**:
- Operator fusion, layout optimization, constant folding
- First run: compilation overhead (seconds)
- Subsequent runs: cached compiled code
- Shape changes trigger recompilation

### Networking and Scaling

**ICI (Inter-Chip Interconnect)**:
- Direct chip-to-chip links (not through host)
- v3/v5e/v6e: 4 nearest neighbors (2D torus)
- v4/v5p: 6 nearest neighbors (3D torus)
- Latency: ~1μs per hop

**Pod topologies**:
- v4: up to 16×16×16 = 4,096 chips
- v5p: up to 16×20×28 = 8,960 chips (4 exaflops!)
- v5e/v6e: up to 16×16 = 256 chips (2D only)

**Bandwidth hierarchy** (v5p example):
- HBM: 2.5 TB/s (fastest)
- ICI: 180 GB/s (inter-chip)
- PCIe: 16 GB/s (host connection)
- DCN: 6.25 GB/s (data center network)

### Performance Optimization

**Arithmetic intensity** = critical for performance:
- Required AI (v5e): ~240 FLOPs/byte to be compute-bound from HBM
- Required AI (VMEM): ~10-20 FLOPs/byte (22× higher bandwidth)
- Matrix multiply AI: approximately equals batch size (for large matrices)

**Optimization strategies**:
1. **Batch size**: Increase until compute-bound (B > 240 from HBM, B > 11 from VMEM)
2. **Precision**: Use bf16 (2× faster than fp32), int8 (2× faster than bf16)
3. **Dimensions**: Pad to multiples of 128 (256 for v6e) to fill systolic array
4. **VMEM prefetching**: Load next weights while current operation runs
5. **Static shapes**: Avoid recompilation overhead

**Example calculation** (from JAX Scaling Book):
- Matrix `int8[16384, 4096] @ int8[B, 4096]` on TPU v5e
- Compute-bound when B > 271 (from HBM)
- Compute-bound when B > 11 (from VMEM)

### TPU vs GPU

**TPU advantages**:
- 2-5× faster matrix multiplication
- 1.2-1.7× better performance per watt (v4 vs A100)
- Higher memory bandwidth per FLOP
- Direct chip interconnects enable massive scaling
- Often cheaper for equivalent throughput

**GPU advantages**:
- More flexible for diverse workloads
- Mature ecosystem and tooling
- Better debugging/profiling tools
- CUDA-specific optimizations

**When to use TPUs**:
- Large transformer training (LLMs, ViTs)
- Matrix-multiplication-heavy workloads
- Need massive scale (1000+ accelerators)
- Cost-sensitive inference

### Systolic Array Architecture

**How it works**:
- 128×128 grid (16,384 ALUs) or 256×256 for v6e
- Weights flow down, activations flow from left
- Diagonal loading pattern maximizes overlap
- Pipelined execution after initial bubble

**Performance**: One `bf16[8,128] @ bf16[128,128]` multiplication every 8 cycles

**Key requirement**: Matrix dimensions ≥ 128 (256 for v6e) to fully utilize

### Code Examples

**JAX data parallelism**:
```python
@pmap
def train_step(state, batch):
    loss, grads = value_and_grad(loss_fn)(state.params, batch)
    return state.apply_gradients(grads=grads), loss
```

**PyTorch XLA training**:
```python
for batch in para_loader.per_device_loader(device):
    loss = model(data)
    loss.backward()
    xm.optimizer_step(optimizer)
    xm.mark_step()  # CRITICAL: trigger compilation
```

### Common Pitfalls

1. **Dynamic shapes**: Triggers recompilation every time
2. **Missing mark_step()**: XLA graph grows indefinitely (PyTorch)
3. **Small batches**: Memory-bound instead of compute-bound
4. **Irregular dimensions**: Wastes systolic array capacity
5. **Excessive host transfers**: PCIe is 100× slower than HBM

### Real-World Use Cases

**GPT-2 training** (from Google Developers Blog, Aug 2025):
- JAX/Flax implementation on TPU v5e pod
- Free access through Colab TPU runtime
- Full pre-training example with code

**Large-scale training** (from Google Cloud Blog, Jul 2021):
- ResNet-50 on v3-32 TPU Pod (32 cores)
- Multi-host PyTorch XLA coordination
- Data streaming from Google Cloud Storage

---

## Sources Used

**Official Documentation** (9 sources):
- Google Cloud TPU documentation (intro, architecture, system specs)
- PyTorch/XLA master documentation
- JAX documentation and quickstart
- Cloud TPU tutorials

**Technical Deep Dives** (3 sources):
- JAX ML Scaling Book: "How to Think About TPUs" (comprehensive architecture)
- JAX ML Scaling Book: "Programming TPUs in JAX"
- Pallas custom kernel documentation

**Tutorials and Guides** (6 sources):
- Google Developers Blog: GPT-2 training with JAX (Aug 2025)
- Medium: PyTorch XLA parallel training (Abhishek Swain)
- Google Cloud: PyTorch XLA performance profiling
- Google Cloud: Scaling workloads with PyTorch/XLA (Jul 2021)
- YouTube: Easy TPU Development (Nodematic, Feb 2025)
- YouTube: Free TPU access setup (Mashaan Alshammari, Nov 2024)

**Comparisons and Analysis** (5 sources):
- Medium ByteBridge: GPU and TPU comparative analysis
- CloudOptimo: TPU vs GPU 2025 comparison (Apr 2025)
- SkyMod: Inside Google's TPU architecture (Aug 2025)
- Google Cloud: Introducing TPU v5p (Dec 2023)
- SemiAnalysis: TPUv5e cost analysis (Sep 2023)

**Additional** (2 sources):
- Wikipedia: Tensor Processing Unit overview
- PyTorch Developer Day 2020: PyTorch/XLA internals video

**Total**: 25 sources cited with URLs and access dates

---

## Knowledge Gaps Filled

### Before PART 16
- Had Vertex AI TPU management knowledge (from EXPANSION 7)
- Had GCP spot instance TPU pricing
- Had general TPU mentions in INDEX.md
- **Missing**: TPU architecture details, programming specifics, JAX/PyTorch XLA

### After PART 16
- ✓ Complete TPU architecture (TensorCore, MXU, VPU, VMEM, systolic arrays)
- ✓ TPU generations comparison (v3 → v6e specifications)
- ✓ JAX programming patterns (pmap, pjit, compilation)
- ✓ PyTorch XLA programming (mark_step, multi-core, profiling)
- ✓ XLA compiler operation and optimization
- ✓ Networking topologies (ICI, DCN, pod slices)
- ✓ Performance optimization strategies (batch sizing, precision, VMEM)
- ✓ TPU vs GPU trade-offs
- ✓ Real-world examples (GPT-2, large-scale training)

### Connection to Existing Knowledge

**Complements GCP knowledge**:
- Existing: Vertex AI TPU provisioning, spot instances, pricing
- New: How to actually program and optimize for TPUs

**Complements distributed training**:
- Existing: DeepSpeed, Megatron-LM, FSDP (PARTS 1-4)
- New: TPU-specific parallelism with ICI networking

**Complements inference optimization**:
- Existing: TensorRT, vLLM, torch.compile (PARTS 5-8)
- New: TPU inference with v5e, cost-efficiency analysis

**Enables alternative hardware section**:
- Sets foundation for AMD ROCm, Apple Metal, Intel oneAPI comparisons
- Provides baseline for "when to use what" decision trees

---

## Impact on karpathy-deep-oracle

**New capabilities**:
1. Can guide TPU programming (JAX and PyTorch XLA)
2. Can explain TPU architecture trade-offs vs GPUs
3. Can optimize workloads for TPU characteristics
4. Can troubleshoot TPU performance issues
5. Can recommend when TPUs make sense

**arr-coc-0-1 relevance**:
- Current: Uses CUDA/GPU (A100 on Vertex AI)
- Future: Could explore TPU deployment for:
  - Inference at scale (v5e cost optimization)
  - Large-scale training experiments (v5p pods)
  - Comparison studies (TPU vs GPU for VLM workloads)

**Knowledge completeness**:
- Alternative hardware: 1/4 complete (TPU done, AMD/Apple/Intel remain)
- Overall EXPANSION 21: 25% complete

---

## Next Steps

**Immediate** (remaining EXPANSION 21):
- PART 13: AMD ROCm ML programming
- PART 14: Apple Metal ML programming
- PART 15: Intel oneAPI ML programming

**Future connections**:
- Cross-reference with distributed training (how TPU ICI compares to NVLink)
- Cross-reference with inference (TPU v5e vs TensorRT/vLLM)
- Add TPU-specific optimization examples to practical implementation

---

**PART 16 complete** ✓

Created: karpathy/alternative-hardware/03-tpu-programming-fundamentals.md (559 lines)
Cited: 25 sources (official docs, tutorials, technical deep dives, comparisons)
Checkbox marked: [✓] in ingestion.md
