# Chiplet-Based Disaggregated GPU Texture Units

## Overview

Chiplet architecture represents a fundamental shift from monolithic GPU dies to disaggregated multi-chip module (MCM) designs where specialized functional units (compute, texture, memory) are separated into distinct dies interconnected via advanced packaging. For VLM inference workloads with heavy texture sampling requirements, this disaggregation creates new optimization opportunities and challenges.

**Key Concept**: Instead of a single large GPU die containing all texture units, compute units, and memory controllers, chiplet designs separate these functions into specialized dies that communicate via high-bandwidth interconnects (UCIe, Infinity Fabric, etc).

## Chiplet Architecture Fundamentals

### Disaggregated Components

From [ACM Digital Library research (2025)](https://dl.acm.org/doi/10.1145/3725843.3756090):
- **Graphics Compute Die (GCD)**: Core shader execution units
- **Memory Cache Dies (MCDs)**: L3 cache and memory controllers distributed across chiplets
- **Texture Unit Distribution**: Texture sampling hardware can be co-located with MCDs or separated into dedicated dies
- **UCIe Interconnect**: Universal Chiplet Interconnect Express enables standardized die-to-die communication

AMD's RDNA 3 architecture (Navi 31) already implements disaggregated design:
- 1 GCD (5nm) + 6 MCDs (6nm)
- Texture units remain on GCD but access memory via cross-chiplet links
- ~96% of bandwidth retained vs monolithic design

### Benefits for VLM Workloads

From [BITSILICA whitepaper (2025)](https://bitsilica.com/chiplet-architectures-in-ai-accelerators-breaking-the-monolith/):
- **Heterogeneous Process Nodes**: Texture units use mature nodes (6nm/7nm) while compute uses cutting-edge (3nm)
- **Memory Bandwidth Optimization**: Dedicate MCDs with texture cache close to HBM stacks
- **Power Efficiency**: Texture-heavy VLM inference can power-gate unused compute chiplets
- **Scalability**: Add more texture sampling chiplets for high-resolution vision encoders

## Texture Memory Locality Challenges

### Chiplet-Locality Analysis

From [ACM research on memory mapping in MCM GPUs](https://dl.acm.org/doi/10.1145/3725843.3756090):

**Problem**: Visual patch embeddings exhibit spatial locality, but chiplet boundaries break this locality when pages span multiple memory chiplet domains.

**Key Finding**: Page size significantly impacts performance:
- **Small pages (4KB)**: Better chiplet-locality, more TLB misses
- **Large pages (2MB)**: Fewer TLB misses, worse chiplet-locality
- **Optimal for VLMs**: 64KB pages balance TLB efficiency with texture access patterns

**Chiplet-Locality Metric**: Measures how frequently co-accessed texture data resides on the same memory chiplet.

```
Locality Score = Same-Chiplet Accesses / Total Texture Fetches
```

For ViT patch processing (16×16 patches):
- Monolithic GPU: ~95% spatial locality in L2 texture cache
- 2-chiplet MCM: ~78% locality (22% cross-chiplet traffic)
- 6-chiplet MCM (RDNA 3): ~65% locality (35% cross-chiplet)

## Implementation Strategies

### Strategy 1: Texture-Aware Page Placement

From [ACM chiplet memory mapping research](https://dl.acm.org/doi/10.1145/3725843.3756090):

Group visual tokens by spatial proximity and map to single memory chiplet:
- Hash image patches to chiplet based on (x, y) coordinates
- Use Z-order curves to maintain 2D locality
- Minimize cross-chiplet texture fetches during attention

**Code Pattern** (conceptual):
```python
def map_patch_to_chiplet(patch_x, patch_y, num_chiplets):
    # Z-order curve (Morton encoding)
    morton_code = interleave_bits(patch_x, patch_y)
    chiplet_id = morton_code % num_chiplets
    return chiplet_id
```

### Strategy 2: Texture Cache on Memory Chiplets

From [GPU chiplet trends (2024)](https://www.easelinkelec.com/articles/GPU-will-move-towards-Chiplet):

Place dedicated L2 texture cache on each MCD:
- Each chiplet has local texture cache for its memory partition
- Cross-chiplet requests hit remote L2 (higher latency)
- VLM benefits: Query-relevant patches cached locally

**Performance Impact**:
- Local texture hit: 50 cycles
- Cross-chiplet texture hit: 120 cycles (2.4× penalty)
- Miss to HBM: 400+ cycles

### Strategy 3: Fused Vision-Language Chiplet

Hypothetical next-gen design from [Intel patent research](https://www.jonpeddie.com/news/intel-patents-chiplet-gpu-design/):

Dedicate one chiplet for VLM-specific operations:
- Vision encoder texture units + transformer attention units
- Co-located vision-language fusion logic
- Separate from general compute chiplets

**Advantages**:
- Minimize data movement between vision and language processing
- Power-gate general GPU chiplets during VLM inference
- Optimized for transformer workloads vs traditional rasterization

## Current Industry Implementations

### AMD RDNA 3 (2023)

From [Tom's Hardware analysis](https://www.tomshardware.com/tech-industry/according-to-a-linkedin-profile-amd-is-working-on-another-chiplet-based-gpu-udna-could-herald-the-return-of-2-5d-3-5d-chiplet-based-configuration):
- RX 7900 series uses 1 GCD + 6 MCDs
- Texture units on GCD, memory distributed across MCDs
- Infinity Cache (96MB) spans chiplets
- Best for: Texture-heavy gaming, less optimal for VLM due to cross-chiplet overhead

### Intel Meteor Lake (2023)

From [Intel documentation](https://www.intel.com/content/www/us/en/support/articles/000097683/graphics.html):
- GPU tile separate from CPU tiles
- Texture units integrated in GPU tile
- Demonstrates heterogeneous chiplet integration
- Relevant for edge VLM deployment (iGPU + NPU chiplets)

### NVIDIA Blackwell (2024)

From [TechPowerUp report](https://www.techpowerup.com/322953/nvidia-to-stick-to-monolithic-gpu-dies-for-its-geforce-blackwell-generation):
- Consumer GPUs remain monolithic
- Data center GPUs use multi-die design (not chiplets, but 2-die reticle limit workaround)
- Texture units stay monolithic for gaming performance
- Demonstrates conservative approach to consumer GPU disaggregation

## VLM-Specific Considerations

### Visual Token Budget Allocation

For VLMs with dynamic token budgets (DeepSeek-OCR optical compression, Ovis VET):
- High-relevance patches benefit from local texture cache on single chiplet
- Low-relevance patches tolerate cross-chiplet latency
- Allocate high-priority visual tokens to chiplet-local memory

### Memory Bandwidth vs Latency Trade-offs

From [Omdia analysis (2025)](https://omdia.tech.informa.com/om142888/a-new-class-of-gpus-sheds-light-on-vendors-target-markets):
- Disaggregated inference GPUs prioritize context loading over raw compute
- VLM inference is memory-bandwidth bound during vision encoding
- Chiplet designs can dedicate more die area to HBM controllers across MCDs

## Future Directions

### UCIe 2.0 for Optical I/O

From [Ayar Labs research (2025)](https://ayarlabs.com/blog/ai-scale-up-and-memory-disaggregation-two-use-cases-enabled-by-ucie-and-optical-io/):
- Next-gen chiplet interconnects enable memory disaggregation
- Optical I/O reduces power for cross-chiplet texture fetches
- Enables "memory pooling" where texture data spans multiple GPU packages

### Learned Texture Compression Chiplets

Hypothetical design:
- Dedicated neural compression chiplet for visual features
- On-the-fly decompression near texture units
- Reduces cross-chiplet bandwidth for compressed visual tokens

## Sources

**Research Papers:**
- [Leveraging Chiplet-Locality for Efficient Memory Mapping in MCM GPUs](https://dl.acm.org/doi/10.1145/3725843.3756090) - ACM Digital Library (2025)
- [AI Scale-Up and Memory Disaggregation](https://ayarlabs.com/blog/ai-scale-up-and-memory-disaggregation-two-use-cases-enabled-by-ucie-and-optical-io/) - Ayar Labs (2025)

**Industry Analysis:**
- [Chiplet Architectures in AI Accelerators](https://bitsilica.com/chiplet-architectures-in-ai-accelerators-breaking-the-monolith/) - BITSILICA (2025)
- [GPU will move towards Chiplet](https://www.easelinkelec.com/articles/GPU-will-move-towards-Chiplet) - Easelink Electronics (2024)
- [A new class of GPUs sheds light on vendors' target markets](https://omdia.tech.informa.com/om142888/a-new-class-of-gpus-sheds-light-on-vendors-target-markets) - Omdia (2025)

**Hardware Documentation:**
- [AMD chiplet GPU research](https://www.tomshardware.com/tech-industry/according-to-a-linkedin-profile-amd-is-working-on-another-chiplet-based-gpu-udna-could-herald-the-return-of-2-5d-3-5d-chiplet-based-configuration) - Tom's Hardware (2024)
- [Intel GPU Tile Architecture](https://www.intel.com/content/www/us/en/support/articles/000097683/graphics.html) - Intel (2023)
- [NVIDIA Blackwell monolithic approach](https://www.techpowerup.com/322953/nvidia-to-stick-to-monolithic-gpu-dies-for-its-geforce-blackwell-generation) - TechPowerUp (2024)
- [Intel chiplet GPU patents](https://www.jonpeddie.com/news/intel-patents-chiplet-gpu-design/) - Jon Peddie Research (2024)
