# Mixed Reality Passthrough: Real-Time Texture Encoding for VR Headsets

## Overview

Mixed reality (MR) passthrough enables VR headsets to display real-world camera feeds in real-time, creating seamless blends of physical and virtual environments. The challenge: encoding RGB camera streams with sub-20ms latency while maintaining visual fidelity on mobile ARM SoCs with passive cooling and 4-10W thermal envelopes.

**Key Constraint**: Latency budget of 11ms per frame for 90Hz refresh rate VR applications - requiring hardware-accelerated texture pipelines and efficient GPU encoding.

From [Native Mixed Reality Compositing on Meta Quest 3](https://arxiv.org/html/2509.18929v1) (arXiv:2509.18929v1, accessed 2025-11-01):
- Meta Quest 3 (Snapdragon XR2 Gen 2) achieves 720p30 MR compositing at 70-80% GPU utilization
- Thermal throttling limits sustained operation to 5-10 minutes without advanced cooling
- Next-gen SoCs (Snapdragon 8 Gen 3) provide 34% headroom, enabling 1080p60 MR capture

## MR Passthrough Architecture

### Camera Access Pipeline

**Passthrough Camera API** (Meta Quest 3, 2025):
- Access to front-facing RGB cameras (1080p+ resolution)
- Metadata includes: focal length, image center, camera position relative to headset center
- Real-time feed streaming with minimal preprocessing overhead

**Resource Allocation**:
- Passthrough rendering: 10-15% GPU (hardware-accelerated)
- Avatar/scene rendering: 30-35% GPU
- ML segmentation (person/background): 20-25% CPU
- 720p compositing + encoding: 25-30% CPU/GPU
- **Total**: 70-80% combined load on XR2 Gen 2

### Texture-Based Encoding

**Why Texture Pipelines?**
GPU texture units provide fixed-function hardware for:
- Bilinear/trilinear filtering (free interpolation)
- Mipmap generation (multi-resolution support)
- Format conversion (RGB → YUV420 for H.264/HEVC)
- Memory bandwidth optimization via tiled layouts

**Encoding Power Draw**: 2.5-3W for 720p30 H.264 encoding - significant thermal load on 4-6W typical TDP devices.

## Real-Time Latency Requirements

### VR Timing Constraints

**90Hz Refresh Rate** → 11ms rendering budget per eye:
- Camera capture: 2-3ms
- ML segmentation (optional): 3-4ms
- Compositing: 2-3ms
- GPU rendering: 3-4ms
- **Total**: ~11-14ms (tight!)

**Asynchronous Pipeline Strategy**:
1. Offload ML segmentation to DSP/NPU (parallel to GPU)
2. Pipeline texture uploads with rendering
3. Use double-buffering for camera frames
4. Leverage hardware video encoders (minimal CPU overhead)

From [Meta AI Research](https://ai.meta.com/blog/using-integrated-ml-to-deliver-low-latency-mobile-vr-graphics/):
- Offload neural network inference to specialized processors (DSP, NPU)
- Pipeline ML execution with GPU rendering
- Achieves 11ms budget for 90Hz VR with ML-driven super-resolution

## Foveated Rendering + Compression

### Variable Resolution Encoding

**Biological Inspiration**: Human foveal vision has 10× higher acuity in center 2° than periphery.

**Implementation**:
- Center region: Full 1080p encoding (high bitrate)
- Mid-periphery: 720p (medium bitrate)
- Far periphery: 480p (low bitrate)
- **Bandwidth savings**: 40-60% vs uniform resolution

**GPU Texture Sampling**:
- Use mipmaps for multi-resolution pyramid
- Hardware anisotropic filtering smooths transitions
- Log-polar coordinate transforms for biological accuracy

### Compression Trade-offs

| Resolution | FPS | Bitrate | Latency | GPU Load | Thermal |
|-----------|-----|---------|---------|----------|---------|
| 720p30    | 30  | 15 Mbps | 12ms    | 70-80%   | 5-10 min |
| 1080p30   | 30  | 25 Mbps | 18ms    | 90-95%   | 2-5 min  |
| 720p60    | 60  | 30 Mbps | 14ms    | 85-90%   | 3-8 min  |

**Next-Gen SoCs** (Snapdragon 8 Gen 3, MediaTek Dimensity 9300):
- 1.5-2× CPU/GPU performance vs XR2 Gen 2
- 50-65% combined load for 720p30 (34% headroom)
- Sustained 1080p60 MR capture feasible with active cooling

## VLM Integration for Scene Understanding

### On-Device ML Segmentation

**Unity Sentis + FastSAM**:
- Real-time person segmentation at ~5 FPS on mobile hardware
- YOLOv8-seg backbone for fast CNN inference
- Achieves 99.03% accuracy for background removal

**Purpose**: Separate real-world subjects from background for compositing with virtual environments.

**Resource Cost**:
- CPU: 20-25% (XR2 Gen 2)
- Memory: 6-7 GB RAM utilization (out of 8 GB total)
- Inference time: 0.64s per frame (suitable for 30 FPS capture)

### Compositing Pipeline

1. **Camera capture** → RGB texture (1080p)
2. **ML segmentation** → Alpha mask (person vs background)
3. **Virtual scene render** → RGBA texture (game content)
4. **Composite** → Blend real + virtual using alpha mask
5. **Encode** → H.264/HEVC stream for recording/streaming

**Thermal Management**:
- Adaptive quality controls (reduce resolution when thermal limits approached)
- Frame rate throttling (90Hz → 72Hz → 60Hz)
- Dynamic scene complexity reduction (LOD adjustments)

## Practical Implementation

### Hardware Requirements

**Minimum** (Meta Quest 3):
- Snapdragon XR2 Gen 2 (8-core ARM, Adreno 740 GPU)
- 8 GB LPDDR5 RAM (5.75 GB developer accessible)
- Passive cooling (4-6W typical, 10W peak TDP)
- **Result**: 720p30 MR for 5-10 minute sessions

**Recommended** (Next-Gen XR):
- Snapdragon 8 Gen 3 or equivalent (34% headroom)
- 12 GB LPDDR5X RAM (76.8 GB/s bandwidth)
- Enhanced thermal design (vapor chamber, active cooling)
- **Result**: 1080p60 MR for extended sessions (30+ minutes)

### API Access

**Meta Passthrough Camera API** (Experimental, 2025):
```cpp
// Unity C# example
var cameraAccess = OVRPlugin.GetPassthroughCameraAccess();
cameraAccess.RequestCameraStream(OVRPlugin.CameraStream.RGB);
Texture2D cameraTexture = cameraAccess.GetCameraTexture();
// Texture now available for GPU compositing
```

**Key Features**:
- Low-latency texture streaming
- Automatic lens distortion correction
- Stereoscopic camera support (dual RGB feeds)
- Integration with Unity/Unreal rendering pipelines

## Sources

**Web Research**:
- [Native Mixed Reality Compositing on Meta Quest 3](https://arxiv.org/html/2509.18929v1) - arXiv:2509.18929v1 (accessed 2025-11-01)
- [Meta AI: Low-Latency Mobile VR Graphics](https://ai.meta.com/blog/using-integrated-ml-to-deliver-low-latency-mobile-vr-graphics/) (accessed 2025-11-01)

**Related Topics**:
- See implementations/56-mipmap-pyramid-hierarchical-vlm.md for multi-resolution texture encoding
- See implementations/57-retinal-chip-neuromorphic-foveated.md for hardware-accelerated foveation
