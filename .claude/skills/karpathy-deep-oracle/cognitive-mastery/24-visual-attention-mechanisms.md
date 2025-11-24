# Visual Attention Mechanisms

## Overview

Visual attention mechanisms determine how the brain selectively processes relevant visual information while filtering distractions. This knowledge synthesizes neuroscience research on attention types, neural substrates, and computational principles with distributed ML system design for implementing attention-like mechanisms in vision models.

From existing knowledge in [cognitive-foundations/03-attention-resource-allocation.md](../cognitive-foundations/03-attention-resource-allocation.md) and [cognitive-mastery/01-precision-attention-resource.md](cognitive-mastery/01-precision-attention-resource.md), we know attention functions as precision-weighted resource allocation. This file extends that foundation with specific visual attention mechanisms.

---

## Section 1: Covert vs Overt Attention

### Definitions

**Overt Attention**: Attention shifts accompanied by eye movements (saccades). Physically reorients foveal vision to bring stimuli to high-resolution processing.

**Covert Attention**: Attention shifts without eye movements. Mental spotlight moves while gaze remains fixed, enhancing processing at attended locations.

### Neural Mechanisms

From [Differences between overt, covert and natural attention shifts to emotional faces](https://www.sciencedirect.com/science/article/pii/S0306452224004615) (Pasqualette & Kulke, 2024):

- **Overt attention** generates greater neural activation due to saccade execution involving motor planning (frontal eye fields, superior colliculus)
- **Covert attention** requires additional inhibition of saccadic responses - must suppress eye movement while shifting mental focus
- **Shared processes**: Both rely on overlapping cortical regions (parietal cortex, FEF) for attention control
- **Key difference**: Covert attention = shared attention processes + saccade inhibition

### Behavioral Evidence

From [Neural Differences between Covert and Overt Attention](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2016.00592/full) (Kulke et al., 2016):

- Covert attention faster for brief cues (no saccade latency ~200ms)
- Overt attention superior for sustained tracking and peripheral targets
- Hybrid strategies common: covert precedes overt (plan while shifting mental focus)

### Natural Attention (Uninstructed)

Pasqualette & Kulke (2024) found that **natural attention shifts** (freely chosen, not instructed) differ from both:

- Earlier emotional processing (Early Posterior Negativity starts sooner)
- Different saccade latencies than instructed overt
- Stronger emotional modulation of neural responses
- **Implication**: Lab studies with rigid instructions may miss ecological attention dynamics

---

## Section 2: Attention Capture

### Bottom-Up Capture

**Salience-Driven Capture**: Conspicuous stimuli (high contrast, motion, unique features) automatically attract attention even when task-irrelevant.

From [Signal Suppression 2.0: Account of Attentional Capture](https://pmc.ncbi.nlm.nih.gov/articles/PMC12313171/) (Gaspelin, 2025):

- **Signal suppression mechanism**: Active suppression of salient distractors' signals
- Not just ignoring - proactive dampening of distractor-evoked activity
- Requires top-down control to override bottom-up salience

### Feature-Based Capture

From [Attentional Capture and Control](https://www.annualreviews.org/content/journals/10.1146/annurev-psych-011624-025340) (Theeuwes, 2024):

- **Feature integration theory**: Attention binds features (color, orientation, motion) into unified objects
- Salient feature singletons capture attention automatically in parallel search
- Conjunction search (multiple features) requires serial attention deployment

### Debate: Can We Resist Capture?

Theeuwes (2024) reviews long-standing debate:

- **Bottom-up view**: Salient stimuli always capture (stimulus-driven)
- **Top-down view**: Capture depends on attentional set (goal-directed filtering)
- **Resolution**: Both mechanisms interact - signal suppression allows top-down goals to overcome bottom-up salience, but requires resources

---

## Section 3: Feature vs Spatial Attention

### Spatial Attention

**Definition**: Attention to a location in visual space, enhancing all features present at that location.

From [Neuronal Effects of Spatial and Feature Attention](https://pmc.ncbi.nlm.nih.gov/articles/PMC6616284/) (Ni et al., 2019):

- Broadly improves perception at specific location (location-based gain)
- Implemented via **normalization mechanism**: attended location receives divisive normalization suppression from smaller pool
- Neural signature: Enhanced activity in V1-V4 for stimuli at attended location, regardless of features

### Feature-Based Attention

**Definition**: Attention to a visual feature (color, orientation, motion direction) across entire visual field.

From [Feature-based attention in visual cortex](https://www.sciencedirect.com/science/article/pii/S0166223606000877) (Maunsell & Treue, 2006):

- Modulates sensory responses **globally** - not restricted to single location
- Attending to "red" enhances red object processing everywhere in visual field
- Neural mechanism: Gain modulation in feature-selective neurons (e.g., V4 color cells)

### Interaction Between Spatial and Feature Attention

From [Differences and Similarities for Spatial and Feature-Based Attention](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00283/full) (Galashan et al., 2017):

- **Spatial attention**: Greater activity in posterior parietal cortex (where processing)
- **Feature-based attention**: Activates areas specialized for that feature (e.g., MT+ for motion)
- **Shared control network**: Frontal eye fields (FEF) and intraparietal sulcus (IPS) control both types
- **Real-world usage**: Typically combined - attend to red car at intersection (feature + space)

From [Spatial and feature-selective attention interact](https://www.biorxiv.org/content/10.1101/2024.12.03.626620v1.full-text) (bioRxiv, 2024):

- Spatial attention + feature attention = **multiplicative interaction**, not additive
- Combined attention creates "attentional zoom": narrow spatial focus on attended feature
- Neural implementation: Spatial resampling depends on both location AND feature

---

## Section 4: Posner Cueing Paradigm

### Classic Paradigm Design

The **Posner cueing task** (Posner, 1980) measures covert attention by separating cue location from target detection.

**Procedure**:
1. Central fixation
2. **Cue**: Peripheral flash (80% valid, 20% invalid)
3. **Target**: Appears at cued or uncued location
4. Measure reaction time (RT)

### Key Findings

From [Measuring attention using the Posner cuing paradigm](https://pmc.ncbi.nlm.nih.gov/articles/PMC3656349/) (Hayward & Ristic, 2013):

- **Valid trials** (target at cued location): Faster RT (~20-50ms benefit)
- **Invalid trials** (target at uncued location): Slower RT (~20-50ms cost)
- **Cue validity effect**: RT(invalid) - RT(valid) = attention benefit magnitude

### Spatial Validity and Attention Orienting

From [The footprints of visual attention in the Posner cueing paradigm](https://jov.arvojournals.org/article.aspx?articleid=2121543) (Eckstein et al., 2002):

- **Attention benefit** NOT just from enhanced processing at cued location
- Also involves **uncertainty reduction**: Cue narrows spatial uncertainty, improving decision efficiency
- **External noise matters**: Attention effects larger in high-noise environments (filters noise)

### Endogenous vs Exogenous Cueing

From [Vision: Paying attention](https://elifesciences.org/articles/99560) (Poth, 2024):

- **Endogenous (voluntary)**: Central arrow cue, ~300ms to deploy, sustained over time
- **Exogenous (reflexive)**: Peripheral flash, <100ms onset, transient (~300ms duration)
- Both can direct covert and overt attention, but different time courses

---

## Section 5: Attention Bottleneck

### Limited Capacity

Visual attention is fundamentally **capacity-limited**:

- Cannot process all visual input simultaneously at high resolution
- Selective mechanism prioritizes task-relevant stimuli
- **Bottleneck location**: Debate between early (sensory) vs late (post-perceptual) selection

### Attentional Blink

**Phenomenon**: When detecting two targets in rapid succession (100-500ms apart), second target often missed.

- Demonstrates temporal bottleneck in attention
- First target "consumes" attentional resources, leaving insufficient capacity for second
- Recovery time ~500ms

### Change Blindness and Inattentional Blindness

Covered in detail in [cognitive-mastery/28-change-inattentional-blindness.md](cognitive-mastery/28-change-inattentional-blindness.md), these phenomena demonstrate:

- Attention required for conscious awareness of visual changes
- Unattended stimuli can go completely unnoticed even when visible
- **Bottleneck implication**: Without attention, no awareness

---

## Section 6: Computational Implementation with Distributed Training (File 1: ZeRO Optimizer)

From [distributed-training/00-deepspeed-zero-optimizer.md](../distributed-training/00-deepspeed-zero-optimizer.md):

### Multi-GPU Attention Mechanisms

**Challenge**: Visual attention models (e.g., Vision Transformers) have massive attention matrices (N² complexity for N patches).

**ZeRO Solution**:
- **ZeRO-1**: Partition optimizer states across GPUs → each GPU computes attention for subset of heads
- **ZeRO-2**: Partition gradients → backprop through attention distributed
- **ZeRO-3**: Partition parameters → attention weight matrices sharded

**Attention-Specific Optimization**:
```python
# Pseudo-code for distributed multi-head attention
# Each GPU handles H/N heads (H = total heads, N = GPUs)

attention_output = []
for head_idx in my_gpu_heads:
    Q, K, V = get_head_params(head_idx)  # Fetched from sharded storage
    attn = compute_attention(Q, K, V)     # Local computation
    attention_output.append(attn)

# All-gather to reconstruct full attention output
full_attention = all_gather(attention_output)
```

**Memory Benefit**: For ViT-Huge (632M params), ZeRO-3 reduces per-GPU memory from 80GB to ~10GB.

### Covert vs Overt Analogy

- **Covert attention in models**: Weight attention without changing input patch order (attention reweighting)
- **Overt attention in models**: Dynamic patch selection before attention (selective input)
- Distributed systems enable both by partitioning attention computation

---

## Section 7: Real-Time Inference Optimization (File 5: TensorRT Fundamentals)

From [inference-optimization/00-tensorrt-fundamentals.md](../inference-optimization/00-tensorrt-fundamentals.md):

### Optimizing Attention for Low-Latency Deployment

**TensorRT Attention Optimizations**:

1. **Flash Attention Kernels**: Fused attention computation (QKV multiply + softmax + weighted sum) in single kernel
   - Reduces memory bandwidth bottleneck
   - 2-4x speedup for large sequence lengths

2. **INT8 Quantization for Attention**:
   - Quantize Q, K, V matrices to INT8
   - Maintains accuracy with calibration (attention scores relatively robust)
   - 4x memory reduction, 2-3x throughput increase

3. **Multi-Head Attention Fusion**:
   - TensorRT fuses head splitting, attention computation, and concatenation
   - Single kernel launch instead of multiple

**Real-Time Visual Attention**:
```python
# TensorRT optimized attention for 384 patches
# Input: [B, 384, 768] features from vision encoder
# Output: [B, 384, 768] attended features

# Without optimization: ~45ms on A100
# With TensorRT + Flash Attention: ~8ms on A100
# Enables 60 FPS visual attention for robotics/AR
```

### Covert Attention = Cached Attention

In deployment, **covert attention** analogy:
- Cache K, V matrices for static parts of scene
- Only recompute Q for moving/changing regions
- Mimics biological covert attention (shift focus without reprocessing whole field)

---

## Section 8: Kubernetes GPU Scheduling for Attention Experiments (File 9: K8s GPU Scheduling)

From [orchestration/00-kubernetes-gpu-scheduling.md](../orchestration/00-kubernetes-gpu-scheduling.md):

### Orchestrating Attention Research Workloads

**Use Case**: Running Posner cueing experiments with computational models across parameter sweeps.

**K8s Pattern for Attention Studies**:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: posner-cueing-simulation
spec:
  parallelism: 8  # 8 attention conditions in parallel
  template:
    spec:
      containers:
      - name: attention-model
        image: attention-research:latest
        resources:
          limits:
            nvidia.com/gpu: 1  # Each condition gets 1 GPU
        env:
        - name: CUE_VALIDITY
          value: "0.8"  # 80% valid trials
        - name: ATTENTION_TYPE
          value: "covert"  # Or "overt", "natural"
```

**Benefit**: Parallel exploration of attention parameter space
- Different cue validities (50%, 80%, 100%)
- Different SOA (stimulus onset asynchrony) timings
- Covert vs overt vs mixed strategies

**GPU Scheduling**:
- Each attention experiment isolated on separate GPU
- Time-slicing for small models (multiple experiments per GPU)
- Results aggregated via distributed storage (NFS/S3)

---

## Section 9: ARR-COC-0-1 - Relevance Allocation as Visual Attention (10%)

### Biological Inspiration from Visual Attention

ARR-COC-0-1 implements **query-driven relevance realization** that parallels biological visual attention mechanisms:

**Covert Attention Analogy**:
- No patch reordering (like no eye movements)
- Attention = token budget allocation based on relevance scores
- High-relevance patches get 400 tokens (foveal equivalent)
- Low-relevance patches get 64 tokens (peripheral equivalent)

**Feature-Based Attention**:
- Query: "Find red objects" → boosts relevance of red patches globally
- Implemented via participatory knowing scorer (query-content cross-attention)
- Mimics feature-based attention's global modulation

### Posner Cueing Analogy

ARR-COC-0-1's two-stage allocation:

1. **Cue stage**: Query provides "cue" about what's relevant
   - Valid cue: Query matches image content → high relevance scores
   - Invalid cue: Query mismatches → uniform low scores

2. **Target stage**: Patch features are "targets"
   - Attended patches (high budget) processed deeply
   - Unattended patches (low budget) processed coarsely

**Validity Effect**:
- Query-aligned patches: Fast, accurate extraction (like valid trials)
- Query-misaligned patches: Slow, lossy extraction (like invalid trials)

### Spatial + Feature Attention Integration

From [arr-coc-0-1/arr_coc/knowing.py](../../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/arr_coc/knowing.py):

```python
# Three ways of knowing = three attention mechanisms

# Propositional (information content) = spatial attention to informative regions
propositional_score = shannon_entropy(patch)

# Perspectival (salience) = feature-based attention to salient features
perspectival_score = jungian_archetype_response(patch)

# Participatory (query coupling) = combined spatial + feature attention
participatory_score = cross_attention(query, patch)

# Final relevance = multiplicative interaction (like real attention)
relevance = propositional * perspectival * participatory
```

**Multiplicative Interaction**: Matches finding from Galashan et al. (2017) and bioRxiv (2024) that spatial + feature attention interact multiplicatively, not additively.

### Attention Bottleneck = Token Budget

**Biological bottleneck**: Limited neural resources, can't process all stimuli at high resolution.

**ARR-COC bottleneck**: Limited token budget (K=200 total patches), can't allocate 400 tokens to all patches.

**Solution (both)**:
- Prioritize relevant stimuli
- Variable resolution (foveal detail + peripheral gist)
- Dynamic reallocation based on task demands

### Training Attention Mechanisms

From [arr-coc-0-1/training/](../../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/training/):

**Quality Adapter** learns attention-like policies:
- Input: Relevance scores (propositional, perspectival, participatory)
- Output: Token budget per patch
- Trained end-to-end on downstream VQA task

**Analogy to biological learning**:
- Reinforcement learning of attention deployment
- Feedback: Task performance (VQA accuracy)
- Policy: Where to allocate "attentional resources" (tokens)

**Distributed Training (ZeRO-3)**:
- Quality adapter parameters sharded across GPUs
- Enables training large attention policies (>100M params)
- Each GPU computes attention for subset of patches

---

## Sources

### Source Documents
- [cognitive-foundations/03-attention-resource-allocation.md](../cognitive-foundations/03-attention-resource-allocation.md) - Existing attention knowledge base
- [cognitive-mastery/01-precision-attention-resource.md](cognitive-mastery/01-precision-attention-resource.md) - Precision-weighted attention framework
- [distributed-training/00-deepspeed-zero-optimizer.md](../distributed-training/00-deepspeed-zero-optimizer.md) - ZeRO optimizer for distributed attention
- [inference-optimization/00-tensorrt-fundamentals.md](../inference-optimization/00-tensorrt-fundamentals.md) - TensorRT attention kernels
- [orchestration/00-kubernetes-gpu-scheduling.md](../orchestration/00-kubernetes-gpu-scheduling.md) - K8s for attention experiments

### Web Research (accessed 2025-11-16)

**Covert vs Overt Attention**:
- [Differences between overt, covert and natural attention shifts to emotional faces](https://www.sciencedirect.com/science/article/pii/S0306452224004615) - Pasqualette & Kulke, Neuroscience 2024
- [Neural Differences between Covert and Overt Attention](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2016.00592/full) - Kulke et al., Frontiers 2016

**Attention Capture**:
- [Signal Suppression 2.0: Account of Attentional Capture](https://pmc.ncbi.nlm.nih.gov/articles/PMC12313171/) - Gaspelin, 2025
- [Attentional Capture and Control](https://www.annualreviews.org/content/journals/10.1146/annurev-psych-011624-025340) - Theeuwes, Annual Review of Psychology 2024

**Feature vs Spatial Attention**:
- [Neuronal Effects of Spatial and Feature Attention](https://pmc.ncbi.nlm.nih.gov/articles/PMC6616284/) - Ni et al., 2019
- [Feature-based attention in visual cortex](https://www.sciencedirect.com/science/article/pii/S0166223606000877) - Maunsell & Treue, Trends in Neurosciences 2006
- [Differences and Similarities for Spatial and Feature-Based Attention](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00283/full) - Galashan et al., 2017
- [Spatial and feature-selective attention interact](https://www.biorxiv.org/content/10.1101/2024.12.03.626620v1.full-text) - bioRxiv 2024

**Posner Cueing**:
- [Measuring attention using the Posner cuing paradigm](https://pmc.ncbi.nlm.nih.gov/articles/PMC3656349/) - Hayward & Ristic, 2013
- [The footprints of visual attention in the Posner cueing paradigm](https://jov.arvojournals.org/article.aspx?articleid=2121543) - Eckstein et al., Journal of Vision 2002
- [Vision: Paying attention](https://elifesciences.org/articles/99560) - Poth, eLife 2024
- [Posner cueing task](https://en.wikipedia.org/wiki/Posner_cueing_task) - Wikipedia

### Additional References
- [PsyToolkit: Spatial Cueing](https://www.psytoolkit.org/experiment-library/cueing.html)
- [Testable: Posner cueing task template](https://www.testable.org/experiment-guides/attention/posner-cueing-task)
