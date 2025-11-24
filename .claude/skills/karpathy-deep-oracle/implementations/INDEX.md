# GPU Hardware Implementations - Index

**Total**: 20 files covering GPU hardware acceleration for VLMs

## Overview

Hardware-specific implementations and optimizations for vision-language models:
- Custom silicon and accelerators
- Texture memory optimization
- Multi-GPU coherency
- Neuromorphic and alternative architectures

## Files

### Texture & Memory (55-64 series)
| File | Description |
|------|-------------|
| `55-3d-volume-texture-spatiotemporal-vit.md` | 3D texture for video ViT |
| `56-mipmap-pyramid-hierarchical-vlm.md` | Mipmap LOD for VLMs |
| `57-retinal-chip-neuromorphic-foveated.md` | Neuromorphic foveation |
| `58-hbm3-texture-streaming-vlm.md` | HBM3 bandwidth optimization |
| `59-chiplet-disaggregated-texture-units.md` | Chiplet GPU architectures |
| `60-photonic-interconnects-texture-memory.md` | Optical interconnects |
| `61-cuda-texture-memory-vit.md` | CUDA texture optimization |
| `62-multi-gpu-texture-coherency-federated.md` | Multi-GPU coherency |
| `63-mixed-reality-passthrough-texture-encoding.md` | MR/AR encoding |
| `64-learned-texture-codec-neural-compression.md` | Neural codecs |

### GPU Hardware Comparison
| File | Description |
|------|-------------|
| `amd-rdna-vs-nvidia-tensor-cores.md` | AMD vs NVIDIA comparison |
| `apple-neural-engine-unified-architecture.md` | Apple ANE, M-series |
| `fpga-texture-filtering-custom-silicon.md` | FPGA accelerators |
| `mesh-shaders-vit-acceleration.md` | Mesh shader pipeline |
| `mobile-gpu-texture-int8-quantization.md` | Mobile GPU INT8 |
| `neuromorphic-event-cameras-sparse-encoding.md` | Event-based vision |
| `rt-cores-nerf-vision-encoding.md` | RT cores for NeRF |
| `variable-rate-shading-foveated-vlm.md` | VRS foveation |
| `vulkan-sparse-texture-vlm-memory.md` | Vulkan sparse textures |
| `webgpu-compute-shaders-browser-vlm.md` | WebGPU browser VLMs |

## Cross-References

- CUDA knowledge: `../cuda/`
- GPU profiling: `../karpathy/practical-implementation/`
- GCP GPUs: `../gcp-gpu/`
