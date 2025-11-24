# Channel Capacity & Noisy Channels

Deep dive into Shannon's channel capacity theorem, the fundamental limits of reliable communication over noisy channels, error correction codes, and biological constraints on neural information transmission.

---

## Section 1: Shannon-Hartley Theorem - The Fundamental Limit

### 1.1 Channel Capacity Definition

The **Shannon-Hartley theorem** establishes the maximum rate at which information can be transmitted over a communication channel with specified bandwidth in the presence of noise:

```
C = B log₂(1 + S/N)
```

**Where:**
- C = channel capacity in bits per second
- B = bandwidth of the channel in hertz
- S = average received signal power (watts)
- N = average noise power (watts)
- S/N = signal-to-noise ratio (linear, not dB)

From [Shannon-Hartley Theorem](https://en.wikipedia.org/wiki/Shannon%E2%80%93Hartley_theorem) (Wikipedia, accessed 2025-11-16):
> "The theorem establishes Shannon's channel capacity for such a communication link, a bound on the maximum amount of error-free information per time unit that can be transmitted with a specified bandwidth in the presence of the noise interference."

**Key insight:** This is a **theoretical maximum** - you can get arbitrarily close but never exceed it.

### 1.2 Noisy Channel Coding Theorem

Claude Shannon's 1948 breakthrough showed that:

**If R < C (rate below capacity):**
- There exists a coding technique allowing arbitrarily small error probability
- Reliable communication is theoretically possible

**If R > C (rate above capacity):**
- Error probability increases without bound
- No useful information can be transmitted

**Critical implication:** The capacity is a sharp threshold, not a gradual degradation.

### 1.3 Power-Limited vs Bandwidth-Limited Regimes

**Bandwidth-Limited (high SNR):**
When S/N ≫ 1:

```
C ≈ B log₂(S/N) ≈ 0.332 × B × SNR_dB
```

- Capacity grows logarithmically with power
- Nearly linear with bandwidth
- Typical in modern wireless (good signal quality)

**Power-Limited (low SNR):**
When S/N ≪ 1:

```
C ≈ 1.44 × B × (S/N) ≈ 1.44 × S/N₀
```

- Capacity linear in power
- Independent of bandwidth (if noise is white)
- Typical in deep space communication, GPS signals

From [Shannon-Hartley Theorem](https://en.wikipedia.org/wiki/Shannon%E2%80%93Hartley_theorem):
> "GPS ranging signals have SNR of -30 dB (S/N = 10⁻³), yielding capacity ≈ 1443 bit/s with 1 MHz bandwidth. Navigation message sends 50 bit/s (well below capacity)."

### 1.4 Comparison to Hartley's Law

Before Shannon, Hartley (1928) proposed:

```
R ≤ 2B log₂(M)
```

where M = number of distinguishable pulse levels.

**Shannon's contribution:** Connected M to SNR via:

```
M = √(1 + S/N)
```

This means error-correction coding (not just more voltage levels) achieves capacity.

---

## Section 2: Error Correction Codes - Approaching Capacity

### 2.1 The Need for Redundancy

**Why error correction?**
- Noise corrupts transmitted symbols
- Direct transmission unreliable above certain error rates
- Redundancy allows detection and correction

**Basic principle:** Add controlled redundancy to enable error recovery.

**Example:** Repeat code
```
Data: 1 0 1
Transmitted: 1 1 1  0 0 0  1 1 1
Received:    1 0 1  0 0 0  1 1 1  (one error)
Decoded:     1      0      1      (majority vote)
```

Rate = 1/3 (very inefficient), but can correct single errors per block.

### 2.2 Shannon's Random Coding Proof

Shannon proved capacity is achievable through **random code construction:**

1. **Random codebook:** Generate 2^(nR) random codewords of length n
2. **Maximum likelihood decoding:** Find closest codeword to received signal
3. **As n → ∞:** Error probability → 0 if R < C

**Key insight:** Random codes are essentially optimal - structured codes can match but not significantly exceed random performance.

### 2.3 Modern Error Correction Codes

**Block Codes:**
- Reed-Solomon: Used in CDs, DVDs, QR codes
- BCH codes: Satellite communications
- Polar codes: 5G wireless (proved to achieve capacity)

**Convolutional Codes:**
- Viterbi decoding
- Used in GSM, satellite communications

**Turbo Codes (1993):**
- First practical codes approaching Shannon limit
- Within 0.5 dB of capacity
- Used in 3G/4G cellular

**LDPC Codes:**
- Low-Density Parity-Check codes
- Performance within 0.0045 dB of capacity
- Used in Wi-Fi, 5G, satellite TV

From [Error Correction Codes and Channel Capacity](https://www.sciencedirect.com/topics/computer-science/shannon-capacity) (accessed 2025-11-16):
> "Shannon capacity is defined as the theoretical maximum data rate that can be achieved on a communication channel with additive white Gaussian noise."

### 2.4 Rate-Distortion Tradeoff

For lossy compression (perception as compression):

```
R(D) = min I(X;Y)
```

where minimum is over all channels with distortion ≤ D.

**Implications for ARR-COC:**
- Visual compression is lossy by necessity
- Must balance information rate with reconstruction quality
- Token budget optimization is a rate-distortion problem

---

## Section 3: Biological Channel Limitations

### 3.1 Neural Channel Capacity

From [Capacity of a Single Spiking Neuron Channel](https://direct.mit.edu/neco/article/21/6/1714/7453/Capacity-of-a-Single-Spiking-Neuron-Channel) (Ikeda & Manton, 2009):

**Temporal coding model:**
- Information encoded in precise spike timing
- Interspike interval τ ~ Gamma distribution
- Capacity depends on timing precision

**Rate coding model:**
- Information encoded in firing rate over fixed window
- Simpler but lower capacity than temporal coding

**Numerical results:**
- Each spike can carry **up to 9 bits of information**
- Typical neural channels: **1000-3000 bits per second**
- Much lower than artificial communication systems

**Why so limited?**
- Metabolic constraints (energy cost of spikes)
- Noise from stochastic ion channels
- Refractory periods
- Physical constraints on axon diameter and conduction velocity

### 3.2 Sensory System Bottlenecks

**Visual system capacity constraints:**

From biological literature:
- Optic nerve: ~1 million axons
- Peak firing rate: ~200-400 Hz per neuron
- Assuming 2-3 bits/spike → ~2-12 Gbps theoretical maximum
- Actual information rate much lower due to:
  - Redundancy in neural code
  - Noise and uncertainty
  - Metabolic constraints

**Retinal information compression:**
- ~130 million photoreceptors
- Compress to ~1 million ganglion cells
- Massive compression ratio ~130:1 before signal even reaches brain

**Foveal vs peripheral channels:**
- Fovea: High-capacity, high-precision
- Periphery: Lower capacity, coarser coding
- This motivates variable LOD in ARR-COC

### 3.3 Metabolic Constraints on Information

**Energy cost of neural communication:**
- ATP required for:
  - Na⁺/K⁺ pump (restore gradients)
  - Vesicle recycling
  - Neurotransmitter synthesis
- Estimated 10⁸ ATP per action potential

**Information efficiency:**
```
η = bits transmitted / ATP consumed
```

**Brain operates near thermodynamic limits** for efficiency given biological constraints.

From [Biological Channel Capacity Neural Limitations](https://scholar.google.com/scholar?q=biological+channel+capacity+neural+limitations) search results:
> "Physical principles for scalable neural recording show brain operates near fundamental thermodynamic limits for information processing given metabolic constraints."

---

## Section 4: Additive White Gaussian Noise (AWGN)

### 4.1 AWGN Channel Model

**Properties:**
- Noise is **additive:** Y = X + N
- **White:** Equal power at all frequencies within bandwidth
- **Gaussian:** N ~ N(0, σ²) at each time point

**Why Gaussian assumption?**
- Central Limit Theorem: Sum of many independent noise sources
- Mathematically tractable
- Worst-case noise for given power (maximizes entropy)

**Channel model:**
```
Received signal = Transmitted signal + Gaussian noise
Y(t) = X(t) + N(t)
```

### 4.2 Colored Noise (Frequency-Dependent)

For non-white noise with power spectral density N(f):

```
C = ∫₀^B log₂(1 + S(f)/N(f)) df
```

**Water-filling algorithm:**
- Allocate more power to frequencies with better SNR
- Optimal power allocation: S(f) = max(0, λ - N(f))
- Like pouring water into uneven container

**Practical example:** DSL uses water-filling across frequency bins to maximize throughput.

### 4.3 Fading Channels

**Rayleigh fading:** Received power varies randomly
- Typical in wireless with multipath
- Capacity is random variable
- Ergodic capacity: Average over fading states

**Outage capacity:** Probability channel supports given rate
- Important for real-time applications
- Trade rate for reliability

---

## Section 5: Distributed Training Implications (File 1: ZeRO)

From [karpathy/distributed-training/00-deepspeed-zero-optimizer.md](../karpathy/distributed-training/00-deepspeed-zero-optimizer.md):

**Communication as noisy channel:**
- Gradient synchronization across GPUs
- Network bandwidth = channel capacity
- Compression ≈ error correction coding

**ZeRO's channel perspective:**
- Partition optimizer states across GPUs
- Reduce redundant communication
- Trade computation for communication (like coding overhead)

**All-reduce operations:**
```
Capacity constraint: C_network = B_network log₂(1 + SNR_network)
```

- Limited by network bandwidth B_network
- Packet loss ≈ channel errors
- TCP retransmission = error correction

**Gradient compression techniques:**
- Quantization: Reduce bits per value
- Sparsification: Send only significant gradients
- Error feedback: Accumulate quantization errors
- These are rate-distortion optimization problems

**Optimal strategy:**
```
Minimize: Communication volume
Subject to: Convergence guarantees (distortion constraint)
```

This is exactly the rate-distortion formulation from Section 2.4.

---

## Section 6: K8s GPU Scheduling & Resource Channels (File 9)

From [karpathy/orchestration/00-kubernetes-gpu-scheduling.md](../karpathy/orchestration/00-kubernetes-gpu-scheduling.md):

**GPU as shared channel:**
- Multiple pods competing for GPU time
- Scheduling ≈ time-division multiplexing
- Capacity shared among users

**Channel capacity analogy:**
```
C_GPU = f_GPU × bits_per_cycle
```

- f_GPU: Clock frequency (bandwidth analog)
- Utilization efficiency affects effective capacity

**Multi-tenancy as channel sharing:**
- Time-slicing: Sequential access (TDMA)
- MIG (Multi-Instance GPU): Spatial partitioning (FDMA)
- Quality of Service ≈ guaranteed capacity allocation

**Resource contention = interference:**
- Memory bandwidth contention
- PCIe bus saturation
- Like multiple users on same communication channel

**Optimal scheduling:**
- Maximize aggregate throughput (total capacity utilization)
- Fair allocation (capacity per user)
- Trade-off similar to multiple access channels

---

## Section 7: AMD ROCm Hardware Channels (File 13)

From [karpathy/alternative-hardware/00-amd-rocm-ml.md](../karpathy/alternative-hardware/00-amd-rocm-ml.md):

**Hardware communication channels:**

**1. HBM3 Memory Channel:**
- MI300X: 192 GB HBM3 (vs H100: 80 GB)
- Bandwidth = channel capacity for memory access
- Higher capacity → support more concurrent operations

**2. Infinity Fabric (AMD interconnect):**
- GPU-to-GPU communication channel
- Capacity determines multi-GPU training efficiency
- Analogous to network channel in distributed training

**3. PCIe Channel:**
- CPU ↔ GPU communication bottleneck
- PCIe 5.0: 128 GB/s (bidirectional)
- Limited capacity affects:
  - Data loading (preprocessing bottleneck)
  - Model inference (input/output transfer)
  - Multi-node training (inter-node communication)

**Channel capacity considerations:**
```
Training throughput ≤ min(
    Compute capacity,
    Memory bandwidth capacity,
    Interconnect capacity,
    I/O capacity
)
```

**ROCm optimization strategies:**
- Overlap computation with communication (pipeline parallelism)
- Reduce communication volume (gradient compression)
- Efficient use of limited channels

---

## Section 8: ARR-COC-0-1 - Visual Information Channel (~10%)

### 8.1 Vision as Noisy Communication Channel

**ARR-COC framework as communication system:**

```
Visual Scene → Retina/Encoding → Visual Cortex/Decoding → Perception
     ↓              ↓                    ↓                    ↓
  Source      Encoder/Channel        Decoder            Destination
```

**Channel characteristics:**
- **Bandwidth:** Optic nerve (~1 million fibers)
- **Noise:** Photoreceptor noise, neural variability
- **Capacity:** Limited by biological constraints

**ARR-COC as optimal encoder:**
- Maximize mutual information I(Scene; Tokens)
- Subject to token budget constraint (64-400 tokens)
- This is exactly Shannon's channel coding problem

### 8.2 Token Allocation as Capacity Allocation

**Variable LOD = adaptive channel capacity:**

High relevance regions:
```
C_fovea = B_fovea log₂(1 + SNR_fovea)  (high capacity)
```

Low relevance regions:
```
C_periphery = B_periphery log₂(1 + SNR_periphery)  (low capacity)
```

**Propositional knowing (Shannon entropy):**
- H(patch) = information content
- Allocate more tokens to high-entropy regions
- Like water-filling in colored noise channels

**Query-aware allocation:**
- Query modulates effective SNR
- Relevant patches get higher "signal power"
- Optimal allocation: maximize I(Scene; Tokens | Query)

### 8.3 Error Correction in Visual Perception

**Biological error correction mechanisms:**

**1. Redundant coding:**
- Multiple neurons encode same feature
- Population codes more robust than single neuron
- Like repetition codes in communication

**2. Predictive coding:**
- Top-down predictions ≈ error correction
- Prediction errors drive learning
- Hierarchical error correction cascade

**3. Attention as error correction:**
- Focus resources on important signals
- Like allocating more parity bits to critical data
- ARR-COC implements this via token budget

**Rate-distortion in perception:**
```
Minimize: Distortion(perceived, actual)
Subject to: Neural capacity constraint
```

ARR-COC solves this through:
- Relevance scorers identify important regions
- Opponent processing balances compression vs detail
- Token allocation optimizes information/cost ratio

### 8.4 Biological Plausibility

**ARR-COC channel model matches biology:**

**Foveal vision:**
- High-capacity channel (dense photoreceptors, detailed encoding)
- 400 tokens for foveal patch ≈ high SNR communication

**Peripheral vision:**
- Low-capacity channel (sparse receptors, coarse encoding)
- 64 tokens for peripheral patch ≈ low SNR communication

**Saccades as channel switching:**
- Eye movements redirect high-capacity channel
- Like beam steering in directional antennas
- Maximizes information acquisition under capacity constraints

**Evidence from neuroscience:**
From [Capacity of a Single Spiking Neuron Channel](https://direct.mit.edu/neco/article/21/6/1714/7453):
> "Single neurons transmit 1000-3000 bits/second, far below artificial systems but optimized for biological constraints."

ARR-COC's 64-400 token range provides 6-9 bits per patch assuming ~1-2 bits per token dimension, matching neural capacity constraints.

---

## Section 9: Practical Implications for ML Systems

### 9.1 Communication Bottlenecks in Distributed Training

**Channel capacity limits:**
- Network bandwidth constraints
- All-reduce operations scale with model size
- Gradient synchronization overhead

**Optimization strategies:**
- Gradient compression (reduce R to match C)
- Asynchronous updates (relax synchronization)
- Mixed precision (fewer bits = more information per channel use)

### 9.2 Inference Serving Constraints

**Latency ≈ 1/Capacity:**
- Higher capacity → lower latency
- Throughput ≈ Capacity
- QoS guarantees require reserved capacity

**Batching as channel multiplexing:**
- Process multiple requests simultaneously
- Amortize overhead across batch
- Like TDMA in communication systems

### 9.3 Model Compression as Rate-Distortion

**Pruning:**
- Remove weights → reduce channel usage
- Accuracy degradation ≈ distortion
- Find optimal rate-distortion point

**Quantization:**
- Fewer bits per weight
- Direct analogy to channel coding with limited alphabet

**Knowledge distillation:**
- Transfer information through limited channel (teacher → student)
- Maximize mutual information under capacity constraint

---

## Section 10: Advanced Topics

### 10.1 Multiple Access Channels

**Several senders, one receiver:**
- Capacity region (achievable rate pairs)
- Time-division, frequency-division, code-division
- Analogy: Multi-task learning sharing network capacity

### 10.2 Broadcast Channels

**One sender, multiple receivers:**
- Different receivers may have different SNR
- Superposition coding
- Analogy: Model serving multiple clients with varying requirements

### 10.3 Relay Channels

**Sender → Relay → Receiver:**
- Relay can amplify or decode-and-forward
- Improves effective SNR
- Analogy: Hierarchical neural processing (V1 → V2 → V4 → IT)

---

## Sources

**Source Documents:**
- [karpathy/distributed-training/00-deepspeed-zero-optimizer.md](../karpathy/distributed-training/00-deepspeed-zero-optimizer.md)
- [karpathy/orchestration/00-kubernetes-gpu-scheduling.md](../karpathy/orchestration/00-kubernetes-gpu-scheduling.md)
- [karpathy/alternative-hardware/00-amd-rocm-ml.md](../karpathy/alternative-hardware/00-amd-rocm-ml.md)

**Web Research:**
- [Shannon-Hartley Theorem](https://en.wikipedia.org/wiki/Shannon%E2%80%93Hartley_theorem) - Wikipedia (accessed 2025-11-16)
- [Channel Capacity of a Telegraph](https://www.johndcook.com/blog/2024/10/19/channel-capacity-of-a-telegraph/) - John D. Cook (October 19, 2024)
- [Capacity of a Single Spiking Neuron Channel](https://direct.mit.edu/neco/article/21/6/1714/7453/Capacity-of-a-Single-Spiking-Neuron-Channel) - Ikeda & Manton, Neural Computation (2009)
- [Shannon Capacity](https://www.sciencedirect.com/topics/computer-science/shannon-capacity) - ScienceDirect (accessed 2025-11-16)
- Google Scholar searches: "biological channel capacity neural limitations" (accessed 2025-11-16)

**Additional References:**
- Shannon, C.E. (1948) - "A Mathematical Theory of Communication" - Bell System Technical Journal
- Noisy Channel Coding Theorem - Information Theory fundamentals
- Error Correction Codes - Turbo codes, LDPC codes achieving near-capacity performance
