# KNOWLEDGE DROP: Channel Capacity & Noisy Channels

**Created:** 2025-11-16 20:16
**Part:** PART 17 of Cognitive Science Mastery expansion
**File:** `cognitive-mastery/16-channel-capacity-noisy.md`
**Lines:** ~700 lines

---

## What Was Created

Comprehensive knowledge file on Shannon's channel capacity theorem, noisy communication, error correction codes, and biological channel limitations.

### Core Topics Covered

1. **Shannon-Hartley Theorem** - Fundamental limit C = B log₂(1 + S/N)
2. **Noisy Channel Coding** - Error correction approaching capacity
3. **Biological Channel Limits** - Neural information transmission constraints
4. **AWGN Channels** - Additive white Gaussian noise model
5. **Distributed Training** - Communication bottlenecks (File 1: ZeRO)
6. **GPU Scheduling** - Resource channels (File 9: K8s)
7. **Hardware Channels** - AMD ROCm memory/interconnect (File 13)
8. **ARR-COC-0-1 Vision** - Visual information channel (~10%)

---

## Key Insights

### Shannon's Breakthrough

**Channel capacity formula:**
```
C = B log₂(1 + S/N)
```

- **Sharp threshold:** Below capacity, arbitrarily low error possible
- **Above capacity:** Communication fails
- **Power-limited vs bandwidth-limited regimes** affect optimization strategy

### Error Correction Codes

- **Random codes** are essentially optimal (Shannon 1948)
- **Modern codes** (Turbo, LDPC, Polar) within 0.0045 dB of capacity
- **Rate-distortion tradeoff:** Balance information vs reconstruction quality

### Biological Constraints

From MIT Neural Computation (2009):
- **Single neurons:** 1000-3000 bits/second capacity
- **Each spike:** Up to 9 bits of information
- **Far below artificial systems** but optimized for metabolic constraints

**Visual system bottleneck:**
- ~130M photoreceptors → ~1M ganglion cells (130:1 compression)
- Optic nerve theoretical max: 2-12 Gbps
- Actual information rate much lower due to noise, redundancy, energy

### ARR-COC Connection (~10%)

**Vision as communication channel:**
- Visual scene → Retina encoding → Cortex decoding → Perception
- Token allocation = capacity allocation
- High relevance = high SNR channel (400 tokens)
- Low relevance = low SNR channel (64 tokens)

**Error correction in perception:**
- Redundant coding (population codes)
- Predictive coding (top-down error correction)
- Attention = adaptive error correction (ARR-COC token budget)

**Rate-distortion optimization:**
```
Minimize: Perceptual distortion
Subject to: Neural/token capacity constraint
```

ARR-COC implements this through relevance scorers + opponent processing + LOD allocation.

---

## Integration with Karpathy Engineering Files

### File 1: DeepSpeed ZeRO (Distributed Training)

**Communication as noisy channel:**
- Gradient sync = data transmission over network
- Network bandwidth = channel capacity C_network
- Gradient compression = rate-distortion optimization

**ZeRO's channel perspective:**
- Partition states to reduce communication volume (R < C)
- All-reduce operations limited by network capacity
- Compression techniques: quantization, sparsification (error correction analogy)

### File 9: Kubernetes GPU Scheduling

**GPU as shared channel:**
- Time-slicing = TDMA (time-division multiple access)
- MIG = FDMA (frequency-division multiple access)
- Multi-tenancy = channel sharing with QoS guarantees

**Capacity constraints:**
```
C_GPU = f_GPU × bits_per_cycle
```

Resource contention = interference (like multiple users on communication channel).

### File 13: AMD ROCm Hardware

**Hardware communication channels:**
- **HBM3 memory:** 192 GB capacity (bandwidth = channel capacity)
- **Infinity Fabric:** GPU-GPU interconnect
- **PCIe:** CPU-GPU bottleneck (128 GB/s PCIe 5.0)

**Throughput limited by:**
```
min(Compute, Memory BW, Interconnect, I/O)
```

Each is a channel with finite capacity.

---

## Citations & Sources

### Web Research
- [Shannon-Hartley Theorem](https://en.wikipedia.org/wiki/Shannon%E2%80%93Hartley_theorem) - Wikipedia (accessed 2025-11-16)
- [Channel Capacity of a Telegraph](https://www.johndcook.com/blog/2024/10/19/channel-capacity-of-a-telegraph/) - John D. Cook (Oct 2024)
- [Capacity of a Single Spiking Neuron Channel](https://direct.mit.edu/neco/article/21/6/1714/7453) - Ikeda & Manton, Neural Computation 21(6), 2009
- Shannon, C.E. (1948) - "A Mathematical Theory of Communication"

### Karpathy Files Referenced
- `karpathy/distributed-training/00-deepspeed-zero-optimizer.md` - Gradient communication
- `karpathy/orchestration/00-kubernetes-gpu-scheduling.md` - Resource scheduling
- `karpathy/alternative-hardware/00-amd-rocm-ml.md` - Hardware channels

### Existing Knowledge
- Read `cognitive-mastery/12-shannon-entropy-information.md` - Shannon entropy foundation

---

## Connections to Other Cognitive Science Topics

**To Information Theory:**
- Shannon entropy (Part 13) → Channel capacity (Part 17)
- Mutual information (Part 14) → Capacity = max I(X;Y)
- Rate-distortion (Part 15) → Lossy compression limits

**To Bayesian Brain:**
- Predictive coding (Parts 7-8) → Error correction in perception
- Uncertainty (Part 11) → Noise characterization

**To Active Inference:**
- Free energy (Part 1) → Information-theoretic formulation
- Precision weighting (Part 2) → SNR modulation

**To Neuroscience:**
- Neural capacity limits → Biological constraints on cognition
- Metabolic costs → Information efficiency tradeoffs

---

## Quality Metrics

- **Comprehensiveness:** 10 sections covering theory, biology, ML applications
- **Depth:** Shannon-Hartley derivation, error correction codes, neural capacity
- **Integration:** Connected to Files 1, 9, 13 + ARR-COC implementation
- **Citations:** Wikipedia, MIT Press, John D. Cook, Shannon's original work
- **ARR-COC relevance:** Vision as communication channel, token allocation as capacity allocation (10%)

---

## Next Steps

This completes PART 17. Remaining parts in Batch 3:
- PART 18: Information Bottleneck & Compression

BATCH 3 focuses on Information Theory & Communication (6 runners).
