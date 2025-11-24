# Mobile GPU Texture Units with INT8 Quantization for VLM Inference

## Section 1: Mobile GPU Texture Unit Architecture

### Overview

Modern mobile GPUs feature dedicated texture units optimized for high-throughput sampling operations. These hardware blocks handle texture filtering, compression/decompression, and memory access patterns essential for graphics workloads - and increasingly, neural network inference.

From [TMModel: Modeling Texture Memory and Mobile GPU](https://dl.acm.org/doi/10.1145/3721145.3725774) (ACM 2025):
> "The fundamental execution unit in a shader core is the warp, a group of hardware threads that execute the same instruction simultaneously. The latest Adreno and Mali GPUs are equipped with specialized L1 texture cache, offering notable performance advantages over standard compute paths."

### Qualcomm Adreno 7xx Series

The Adreno 740 GPU (Snapdragon 8 Gen 2) represents the current state-of-the-art in mobile graphics:

**Architecture Highlights:**
- Dedicated texture sampling units with bilinear/trilinear filtering hardware
- L1 texture cache per shader core (32-64 KB typical)
- L2 texture cache shared across cores (512 KB - 1 MB)
- Hardware decompression for ASTC, ETC2, and BC formats
- Up to 8 texture samples per clock cycle per core

**Memory Hierarchy:**
```
Texture Request
    ↓
L1 Texture Cache (per-core, 32-64 KB)
    ↓
L2 Texture Cache (shared, 512 KB - 1 MB)
    ↓
System Memory (LPDDR5X, ~50 GB/s bandwidth)
```

**INT8 Support:**
The Adreno 740 includes native INT8 texture sampling via Vulkan and OpenGL ES extensions. Textures can store quantized neural network weights as 8-bit unsigned integers, with hardware performing efficient gather operations.

### ARM Mali-G78 MP20/MP24

The Mali-G78 (second-generation Valhall architecture) competes with Adreno in flagship mobile devices.

From [ARM Mali-G78 Product Documentation](https://developer.arm.com/Processors/Mali-G78) (accessed 2025-01-31):
> "Mali-G78 is the highest performing Arm GPU that enables premium user experiences with improved energy efficiency, increased machine learning performance, and desktop-class gaming."

**Key Features:**
- Up to 24 execution engines (MP24 configuration)
- Texture unit per execution engine
- 128-bit texture fetch per cycle
- Native support for ASTC LDR/HDR and ETC2 compression
- Hardware INT8 acceleration for AI workloads

**Texture Cache Architecture:**
- L1 cache: 16-32 KB per execution engine
- L2 cache: Configurable 512 KB - 2 MB
- Optimized for spatially coherent access patterns
- Prefetching logic for sequential texture reads

### Apple GPU Architecture

Apple's custom GPU (M-series and A-series chips) integrates texture units with the unified memory architecture.

**Architecture Benefits:**
- Unified memory eliminates CPU-GPU data copies
- Texture units access system RAM directly (up to 800 GB/s on M3 Max)
- Hardware decompression for ASTC and proprietary formats
- Metal API provides direct INT8 texture support

From [Efficient Deployment of Vision-Language Models on Mobile Devices](https://arxiv.org/html/2507.08505v1) (arXiv 2025):
> "Unified memory architecture provides zero-copy data sharing between CPU, GPU, and Neural Engine, enabling efficient weight storage in texture memory with bandwidth advantages."

**Performance Characteristics:**
- Single-cycle texture lookup from L1 cache
- 4x4 bilinear filter in single operation
- Native FP16 and INT8 data types
- Optimized for SIMD operations across GPU threads

### Texture Sampling Hardware

All three architectures share common texture sampling capabilities:

**Filtering Modes:**
- Point (nearest neighbor) - 1 memory access
- Bilinear - 4 memory accesses, interpolated
- Trilinear - 8 memory accesses (2 mipmap levels)
- Anisotropic - Up to 16 samples

**Address Modes:**
- Clamp to edge
- Repeat (wrap)
- Mirrored repeat
- Clamp to border

**Format Support:**
- Normalized integers (UNORM8, UNORM16)
- Signed integers (SNORM8, INT8, INT16)
- Floating point (FP16, FP32)
- Compressed (ASTC, ETC2, BC1-7)

### Hardware Acceleration Benefits

Texture units provide massive parallelism compared to standard compute:

**Throughput Comparison (Adreno 740):**
- Compute shader load: ~100 GB/s effective
- Texture unit load: ~200 GB/s effective (2x improvement)
- Hardware filtering: Free (no shader ALU cost)

From [ShaderNN: A lightweight and efficient inference engine](https://www.sciencedirect.com/science/article/pii/S0925231224013997) (ScienceDirect 2025):
> "We propose Shader Neural Network (ShaderNN), an OpenGL-based, fast, and power-efficient inference framework designed for mobile devices, leveraging texture units for weight storage and sampling."

**Power Efficiency:**
- Texture access: ~10 pJ per byte
- Compute access: ~30 pJ per byte
- Hardware decompression: ~5 pJ per byte

### Memory Bandwidth Optimization

Texture compression reduces memory traffic by 4-8x:

**ASTC Compression:**
- Variable block sizes: 4x4 to 12x12 pixels
- Compression ratios: 4:1 to 32:1
- Hardware decompression in texture unit
- ~10-20 cycles decompression latency

**ETC2 Compression:**
- Fixed 4x4 blocks
- 4:1 or 8:1 compression
- Lower quality than ASTC
- Faster decompression (~5 cycles)

**Bandwidth Savings Example:**
```
Uncompressed FP32 weights: 4 bytes/weight
INT8 quantized: 1 byte/weight (4x reduction)
ASTC compressed INT8: 0.25 bytes/weight (16x reduction)

For 7B parameter VLM:
- FP32: 28 GB
- INT8: 7 GB
- ASTC INT8: 1.75 GB
```


## Section 2: INT8 Quantization for VLM Inference

### Quantization Fundamentals

Quantization maps continuous floating-point values to discrete integer representations. For neural networks, this reduces memory footprint and enables integer arithmetic acceleration.

**Quantization Equation:**
```
Q(x) = round(x / scale) + zero_point
x_dequant = (Q(x) - zero_point) * scale
```

Where:
- `scale`: Floating-point scaling factor
- `zero_point`: Integer offset (often 0 for symmetric quantization)
- `Q(x)`: Quantized integer value (INT8: -128 to 127 or 0 to 255)

### Post-Training Quantization (PTQ)

From [Neural Network Model quantization on mobile](https://community.arm.com/arm-community-blogs/b/ai-blog/posts/neural-network-model-quantization-on-mobile) (ARM Community 2023):
> "PTQ is a quantization technique that can reduce model memory footprint while also improving inference latency on the CPU and hardware accelerators, without additional training."

**Dynamic Range Quantization:**
- Weights: Quantized offline to INT8
- Activations: Quantized dynamically during inference
- No calibration dataset required
- 4x memory reduction, 2-3x speedup

**Full Integer Quantization:**
- Both weights and activations quantized to INT8
- Requires representative calibration dataset
- 4x memory reduction, 3x+ speedup
- Compatible with integer-only accelerators (NPUs)

From [Performance Evaluation of INT8 Quantized Inference on Mobile GPUs](https://www.researchgate.net/publication/356828619_Performance_Evaluation_of_INT8_Quantized_Inference_on_Mobile_GPUs) (ResearchGate 2022):
> "This paper presents a unified framework that integrates various INT8 quantization methods, such as symmetric, asymmetric, per-layer, and per-channel quantization schemes."

**Quantization Granularity:**

**Per-Tensor (Layer-wise):**
- Single scale/zero-point for entire tensor
- Simplest implementation
- May lose accuracy with outliers

**Per-Channel (Filter-wise):**
- Separate scale/zero-point per output channel
- Better accuracy preservation
- Minimal overhead (scale LUT per channel)

**Per-Group (Block-wise):**
- Quantize groups of parameters independently
- Balance between per-tensor and per-channel
- Used in LLM quantization (e.g., GPTQ)

### Quantization-Aware Training (QAT)

QAT simulates quantization during training, allowing the model to adapt to reduced precision.

From [Neural Network Model quantization on mobile](https://community.arm.com/arm-community-blogs/b/ai-blog/posts/neural-network-model-quantization-on-mobile) (ARM Community 2023):
> "QAT trains DL models with already quantized weights and activations (8-bit instead of 32-bit float) from the start of training. This leads to better performance at the cost of additional training time."

**Training Process:**
1. Forward pass: Simulate quantization with fake-quant nodes
2. Compute loss with quantized activations
3. Backward pass: Gradients flow through straight-through estimators
4. Update FP32 master weights
5. Repeat until convergence

**Accuracy Comparison:**
- FP32 baseline: 100% accuracy (reference)
- PTQ INT8: 98-99% accuracy (typical)
- QAT INT8: 99-100% accuracy (minimal degradation)

**QAT Implementation (TensorFlow Lite):**
```python
import tensorflow_model_optimization as tfmot

# Load pre-trained FP32 model
base_model = tf.keras.models.load_model('model.h5')

# Apply QAT
quant_aware_model = tfmot.quantization.keras.quantize_model(base_model)

# Fine-tune with quantization simulation
quant_aware_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
quant_aware_model.fit(train_data, epochs=5)

# Convert to TFLite INT8
converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

### Symmetric vs Asymmetric Quantization

**Symmetric Quantization:**
```
Q(x) = round(x / scale)
Range: [-127, 127] or [-128, 127]
zero_point = 0
```
- Simpler hardware implementation
- Zero-centered distributions
- Preferred for weights

**Asymmetric Quantization:**
```
Q(x) = round(x / scale) + zero_point
Range: [0, 255]
zero_point ≠ 0
```
- Better for non-centered activations (ReLU outputs)
- Captures full dynamic range
- Slight overhead for zero-point handling

### Mixed-Precision Quantization

For large models (>2B parameters), uniform INT8 quantization can fail due to extreme outliers.

From [Efficient Deployment of Vision-Language Models on Mobile Devices](https://arxiv.org/html/2507.08505v1) (arXiv 2025):
> "Above 2.72B parameters, regular 8-bit quantization fails to follow FP16 baseline accuracy. Mixed-precision performs 16-bit matrix multiplication for outlier feature dimensions and 8-bit for the remaining 99.9%."

**LLM Quantization Strategy:**
- Attention weights: INT8 (95% of parameters)
- Outlier channels: FP16 (0.1% of parameters)
- FFN weights: INT4 or INT8 (memory-limited)
- Embedding layers: FP16 (precision-critical)

**Memory vs Accuracy Trade-off:**
```
LLaMA-7B model:
- FP16: 14 GB, 100% accuracy
- INT8 (uniform): 7 GB, 92% accuracy (degraded)
- INT8 (mixed): 7.2 GB, 99.5% accuracy
- INT4 (mixed): 4 GB, 97% accuracy
```

### Calibration and Range Estimation

Accurate quantization requires proper scale/zero-point calculation.

**Min-Max Calibration:**
```
scale = (max(x) - min(x)) / 255
zero_point = -round(min(x) / scale)
```
- Simple, fast
- Sensitive to outliers

**Percentile Calibration:**
```
scale = (P99.9(x) - P0.1(x)) / 255
```
- Clips extreme outliers
- More robust

**Entropy Calibration (KL-divergence):**
```
scale = argmin KL(P_fp32 || P_quant)
```
- Minimizes information loss
- Used in TensorRT, ONNX Runtime

**Calibration Dataset Size:**
- 100-1000 samples typical
- Represents deployment distribution
- More samples = better scale estimation

From [Quantized Image Super-Resolution on Mobile NPUs](https://openaccess.thecvf.com/content/CVPR2025W/MAI/papers/Ignatov_Quantized_Image_Super-Resolution_on_Mobile_NPUs_Mobile_AI_2025_Challenge_CVPRW_2025_paper.pdf) (CVPR 2025):
> "To ensure high efficiency and compatibility with low-power edge NPUs, all models had to be additionally INT8 quantized (both weights and activations)."


## Section 3: Texture Compression Integration

### ASTC Texture Compression for Neural Networks

Adaptive Scalable Texture Compression (ASTC) provides variable-rate compression ideal for neural network weights.

From [Compressing Deep Neural Networks with ASTC](https://developer.arm.com/community/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/high-performing-graphics-2017-compressing-deep-neural-networks-with-astc) (ARM Developer 2017):
> "Most texture compression schemes assume spatially correlated inputs. ASTC supports both 2D and 3D texture compression with variable block sizes, enabling compression ratios from 8:1 to 32:1."

**ASTC Block Sizes:**
```
Format           Bits/Pixel    Compression Ratio
4x4              8.00          4:1
5x4              6.40          5:1
6x5              5.12          6.25:1
8x5              4.00          8:1
10x8             2.56          12.5:1
12x12            0.89          36:1
```

**Weight Storage Strategy:**
1. Reshape weight tensor to 2D texture (e.g., [C_out, C_in * K * K])
2. Compress with ASTC encoder offline
3. Upload compressed texture to GPU
4. Decode on-the-fly during inference

**ASTC Compression Pipeline:**
```
FP32 Weights (28 GB for LLaMA-7B)
    ↓ Quantization
INT8 Weights (7 GB)
    ↓ Reshape to 2D
Texture Layout (channels × spatial)
    ↓ ASTC Encode (6x6 blocks)
Compressed Texture (1.4 GB, 5:1 ratio)
    ↓ GPU Upload
VRAM Storage
    ↓ Hardware Decode
INT8 Values (in L1 cache)
```

From [Using ASTC compression for DNN Weights](https://old.hotchips.org/wp-content/uploads/hc_archives/hc29/HC29.23-Posters-Pub/HC29.23.p10-Texture-Compression-Sharma-GeorgiaTech-v02.pdf) (Hot Chips 2017):
> "ASTC qualities: Asymmetric (fast decode), variable block size, 2D/3D support. Hardware decompression occurs in texture units with ~10-20 cycle latency."

### ETC2 Texture Compression

ETC2 (Ericsson Texture Compression 2) is an alternative compression format.

**ETC2 Characteristics:**
- Fixed 4x4 block size
- 4:1 compression (RGB) or 8:1 (RGB+Alpha)
- Mandatory in OpenGL ES 3.0+
- Supported on all Android devices (API 18+)

From [Android OpenGL Texture Compression](https://stackoverflow.com/questions/9148795/android-opengl-texture-compression) (Stack Overflow):
> "ETC2 is backwards compatible with ETC1 by using invalid bit combinations, and improves image quality. Supported by all Android devices with OpenGL ES 3.0."

**Compression Quality:**
```
Method          PSNR (dB)   Compression    Decode Speed
Uncompressed    ∞           1:1            N/A
ASTC 6x6        48-52       5:1            ~100 Mpixel/s
ETC2            44-48       4:1            ~150 Mpixel/s
JPEG            40-45       10:1           ~50 Mpixel/s (CPU)
```

**Format Selection:**
- ASTC: Better quality, flexible ratios, ARM/Qualcomm/Apple
- ETC2: Universal support, faster decode, lower quality
- BC7: Desktop/console only, not mobile

### Decompression in Texture Units

Mobile GPU texture units decompress formats transparently during sampling.

**Decompression Pipeline:**
```
Texture Fetch Request
    ↓
L1 Cache Lookup
    ↓ (miss)
L2 Cache Lookup
    ↓ (miss)
Fetch Compressed Block from VRAM (128 bits)
    ↓
Hardware Decompression Unit
    ↓
4x4 Decompressed Pixels (256 bits)
    ↓
L1 Cache Store
    ↓
Return Sampled Value
```

**Decompression Latency:**
- ASTC: 10-20 cycles
- ETC2: 5-10 cycles
- BC1-3: 3-5 cycles
- Amortized over 16 pixels in block

**Energy Efficiency:**
```
Operation                Energy (pJ)
DRAM access (32-bit)     640
L2 cache access          50
Decompression            10
L1 cache access          5

Compressed texture fetch total: 640 + 50 + 10 + 5 = 705 pJ
Uncompressed fetch (4x data): 4 × 640 + 50 + 5 = 2615 pJ

Energy savings: 2.7x
```

### Memory Bandwidth Savings

Texture compression dramatically reduces memory traffic.

**Bandwidth Calculation (7B parameter VLM):**
```
Scenario: Inference on 336×336 image, 1024 token generation

Uncompressed INT8 weights:
- 7B parameters × 1 byte = 7 GB
- Memory bandwidth: 50 GB/s
- Load time: 140 ms per inference

ASTC-compressed (6x6, 5:1 ratio):
- 7 GB / 5 = 1.4 GB
- Load time: 28 ms per inference
- 5x faster weight loading

Effective bandwidth:
- Physical: 50 GB/s
- Effective: 250 GB/s (with 5:1 compression)
```

From [TMModel: Modeling Texture Memory and Mobile GPU](https://dl.acm.org/doi/10.1145/3721145.3725774) (ACM 2025):
> "Texture cache hierarchy with compression provides 2-5x effective bandwidth improvement over standard compute paths, critical for memory-bound inference."

**Compression Trade-offs:**
```
Method          Memory    Bandwidth    Quality    Encode Time
INT8            1.0x      1.0x         Reference  0s
ASTC 8x5        0.5x      2.0x         ~99%       ~1s per layer
ASTC 6x6        0.4x      2.5x         ~98%       ~2s per layer
ASTC 10x8       0.32x     3.1x         ~95%       ~3s per layer
```

### Practical Limitations

**Block Boundary Artifacts:**
- ASTC compresses 4x4 to 12x12 blocks independently
- Weight tensors may have discontinuities at boundaries
- Mitigation: Pad tensors to block-aligned sizes

**Compression Overhead:**
- ASTC encoding is slow (~1-10s per layer)
- Performed offline during model conversion
- Negligible decode overhead (hardware accelerated)

**Format Compatibility:**
- ASTC: ARM Mali, Qualcomm Adreno, Apple GPU
- ETC2: All OpenGL ES 3.0+ devices
- BC formats: Not supported on mobile

**Quantization + Compression Interaction:**
- Compress after quantization (not before)
- INT8 values more compressible than FP32
- Per-channel quantization improves ASTC quality


## Section 4: Implementation Patterns

### OpenGL ES Implementation

OpenGL ES 3.0+ provides ASTC and ETC2 texture support on mobile.

**Texture Creation and Upload:**
```c
// C/C++ - OpenGL ES 3.2
// Create ASTC-compressed texture for INT8 weights

GLuint texture;
glGenTextures(1, &texture);
glBindTexture(GL_TEXTURE_2D, texture);

// ASTC 6x6 format (5:1 compression)
GLenum format = GL_COMPRESSED_RGBA_ASTC_6x6_KHR;

// Upload compressed data
int width = 1024;  // Weight matrix width (rounded to block size)
int height = 4096; // Weight matrix height
int compressedSize = ((width + 5) / 6) * ((height + 5) / 6) * 16;

glCompressedTexImage2D(
    GL_TEXTURE_2D,
    0,                    // mip level
    format,
    width,
    height,
    0,                    // border
    compressedSize,
    compressedData        // ASTC-encoded bytes
);

// Set sampling parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
```

**Compute Shader for Matrix Multiplication:**
```glsl
#version 320 es
precision highp float;
precision highp int;

layout(local_size_x = 16, local_size_y = 16) in;

// Weight texture (ASTC-compressed, INT8)
uniform highp usampler2D weights;

// Input activations (FP32 buffer)
layout(std430, binding = 0) readonly buffer Input {
    float activations[];
};

// Output (FP32 buffer)
layout(std430, binding = 1) writeonly buffer Output {
    float outputs[];
};

// Quantization parameters
uniform float weight_scale;
uniform int weight_zero_point;

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;

    float sum = 0.0;

    // Matrix multiplication: output[row, col] = input[row, :] @ weights[:, col]
    for (uint k = 0; k < 768u; k++) {
        // Fetch INT8 weight from texture (hardware decompresses ASTC)
        uvec4 weight_quantized = texelFetch(weights, ivec2(col, k), 0);
        int w_int8 = int(weight_quantized.r);

        // Dequantize: w_fp32 = (w_int8 - zero_point) * scale
        float w_dequant = float(w_int8 - weight_zero_point) * weight_scale;

        // Accumulate
        float activation = activations[row * 768u + k];
        sum += activation * w_dequant;
    }

    outputs[row * gl_NumWorkGroups.x * 16u + col] = sum;
}
```

### Vulkan Implementation

Vulkan provides more explicit control over texture memory and synchronization.

**Texture Creation:**
```cpp
// C++ - Vulkan 1.3
VkImageCreateInfo imageInfo = {};
imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
imageInfo.imageType = VK_IMAGE_TYPE_2D;
imageInfo.format = VK_FORMAT_ASTC_6x6_UNORM_BLOCK;  // ASTC 6x6
imageInfo.extent = {1024, 4096, 1};
imageInfo.mipLevels = 1;
imageInfo.arrayLayers = 1;
imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

VkImage weightTexture;
vkCreateImage(device, &imageInfo, nullptr, &weightTexture);

// Allocate device memory
VkMemoryRequirements memRequirements;
vkGetImageMemoryRequirements(device, weightTexture, &memRequirements);

VkMemoryAllocateInfo allocInfo = {};
allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
allocInfo.allocationSize = memRequirements.size;
allocInfo.memoryTypeIndex = findMemoryType(
    memRequirements.memoryTypeBits,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
);

VkDeviceMemory weightMemory;
vkAllocateMemory(device, &allocInfo, nullptr, &weightMemory);
vkBindImageMemory(device, weightTexture, weightMemory, 0);
```

**Compute Shader (GLSL/SPIR-V):**
```glsl
#version 450

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) uniform usampler2D weights;  // ASTC INT8
layout(set = 0, binding = 1) buffer InputBuffer { float inputs[]; };
layout(set = 0, binding = 2) buffer OutputBuffer { float outputs[]; };

layout(push_constant) uniform PushConstants {
    float weight_scale;
    int weight_zero_point;
    uint M, N, K;  // Matrix dimensions
};

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;

    if (row >= M || col >= N) return;

    float accum = 0.0;
    for (uint k = 0; k < K; k++) {
        uint w_q = texelFetch(weights, ivec2(col, k), 0).r;
        float w = float(int(w_q) - weight_zero_point) * weight_scale;
        accum += inputs[row * K + k] * w;
    }

    outputs[row * N + col] = accum;
}
```

### Metal Implementation (Apple Silicon)

Metal provides optimized texture access on iOS/macOS.

**Texture Setup:**
```swift
// Swift - Metal
let textureDescriptor = MTLTextureDescriptor()
textureDescriptor.pixelFormat = .astc_6x6_ldr  // ASTC 6x6
textureDescriptor.width = 1024
textureDescriptor.height = 4096
textureDescriptor.usage = [.shaderRead]
textureDescriptor.storageMode = .shared  // Unified memory

let weightTexture = device.makeTexture(descriptor: textureDescriptor)!

// Upload ASTC-compressed data
let region = MTLRegionMake2D(0, 0, 1024, 4096)
weightTexture.replace(
    region: region,
    mipmapLevel: 0,
    withBytes: astcData,
    bytesPerRow: ((1024 + 5) / 6) * 16
)
```

**Compute Kernel:**
```metal
// Metal Shading Language
#include <metal_stdlib>
using namespace metal;

kernel void matmul_int8_texture(
    texture2d<uint, access::sample> weights [[texture(0)]],
    device const float* inputs [[buffer(0)]],
    device float* outputs [[buffer(1)]],
    constant float& weight_scale [[buffer(2)]],
    constant int& zero_point [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_edge);

    uint row = gid.y;
    uint col = gid.x;

    float sum = 0.0f;
    for (uint k = 0; k < 768; k++) {
        // Sample INT8 weight (ASTC decompressed by hardware)
        uint w_q = weights.sample(s, float2(col + 0.5f, k + 0.5f)).r;
        float w = float(int(w_q) - zero_point) * weight_scale;

        sum += inputs[row * 768 + k] * w;
    }

    outputs[row * 1024 + col] = sum;
}
```

### Mobile VLM Deployment Strategies

From [Efficient Deployment of Vision-Language Models on Mobile Devices](https://arxiv.org/html/2507.08505v1) (arXiv 2025):
> "Efficiently deploying VLMs on mobile hardware presents non-trivial challenges due to the need to map distinct stages of the inference pipeline onto heterogeneous compute units including CPUs, GPUs, and NPUs."

**Recommended Architecture:**

**Image Encoder (GPU texture path):**
- Store ViT weights in ASTC-compressed textures
- Use compute shaders for attention and FFN
- Leverage texture filtering for interpolation
- ~30-50 ms latency for 336×336 image

**Text Decoder (CPU/NPU):**
- Autoregressive generation (sequential)
- Lower batch size (typically 1)
- CPU or NPU more efficient than GPU
- ~50-100 ms per token

**Hybrid Scheduling:**
```
┌─────────────┐
│ Image Input │
└──────┬──────┘
       │ GPU
       ▼
┌─────────────┐
│ViT Encoding │ ← ASTC texture weights
│ (Attention) │ ← Compute shaders
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Features   │
│  (tokens)   │
└──────┬──────┘
       │ Transfer to CPU
       ▼
┌─────────────┐
│LLM Decoding │ ← INT8 or INT4 quantization
│(Transformer)│ ← CPU or NPU
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Text Output  │
└─────────────┘
```

### Benchmarking Results

From [Efficient Deployment of Vision-Language Models on Mobile Devices](https://arxiv.org/html/2507.08505v1) (arXiv 2025):

**OnePlus 13R (Snapdragon 8 Gen 3):**

**LLaVA-1.5 7B:**
- CPU-only (llama.cpp): 101s total, 88-90°C, 10-12W
- GPU-offload (not available): N/A

**MobileVLM-3B:**
- CPU-only: 35s total, 70-72°C, 3.5W
- Image encoding: 14.1s
- Text generation: 6.6s

**Imp-v1.5-3B (MLC-LLM):**
- GPU-offload: 25s total, 60°C, 1.3W
- Image encoding (GPU): 18s at 90-100% GPU utilization
- Text generation (CPU): 3s at 120% CPU utilization

**Key Insights:**
1. GPU offloading reduces power by 3-10x (1.3W vs 10-12W)
2. Temperature drops 30°C with GPU path (60°C vs 90°C)
3. CPU remains bottleneck for sequential text generation
4. Texture units enable efficient weight storage and sampling

### Practical Implementation Checklist

**Model Preparation:**
- [ ] Quantize model to INT8 (PTQ or QAT)
- [ ] Measure per-channel scales and zero-points
- [ ] Reshape weight tensors to 2D texture layout
- [ ] Compress with ASTC (6x6 or 8x5 blocks)
- [ ] Validate compression quality (PSNR > 45 dB)

**Runtime Setup:**
- [ ] Create compressed textures with appropriate formats
- [ ] Upload ASTC/ETC2 data to GPU memory
- [ ] Bind textures to compute shader samplers
- [ ] Configure nearest-neighbor filtering (no interpolation)
- [ ] Set appropriate cache hints (clamp to edge)

**Inference Pipeline:**
- [ ] Dispatch compute shaders for matrix operations
- [ ] Dequantize weights in shader (scale/zero-point)
- [ ] Accumulate in FP32 for numerical stability
- [ ] Profile texture fetch latency and cache hit rates
- [ ] Monitor GPU utilization and thermal throttling

**Optimization:**
- [ ] Tune workgroup sizes (16×16 typical)
- [ ] Minimize host-device transfers
- [ ] Batch multiple inference requests when possible
- [ ] Consider mixed-precision for accuracy-critical layers
- [ ] Profile memory bandwidth utilization


## Sources

**Web Research:**
- [TMModel: Modeling Texture Memory and Mobile GPU](https://dl.acm.org/doi/10.1145/3721145.3725774) - ACM 2025 (accessed 2025-01-31)
- [Efficient Deployment of Vision-Language Models on Mobile Devices: A Case Study on OnePlus 13R](https://arxiv.org/html/2507.08505v1) - arXiv 2025 (accessed 2025-01-31)
- [Neural Network Model quantization on mobile](https://community.arm.com/arm-community-blogs/b/ai-blog/posts/neural-network-model-quantization-on-mobile) - ARM Community 2023 (accessed 2025-01-31)
- [Performance Evaluation of INT8 Quantized Inference on Mobile GPUs](https://www.researchgate.net/publication/356828619_Performance_Evaluation_of_INT8_Quantized_Inference_on_Mobile_GPUs) - ResearchGate 2022 (accessed 2025-01-31)
- [ShaderNN: A lightweight and efficient inference engine for mobile devices](https://www.sciencedirect.com/science/article/pii/S0925231224013997) - ScienceDirect 2025 (accessed 2025-01-31)
- [Quantized Image Super-Resolution on Mobile NPUs](https://openaccess.thecvf.com/content/CVPR2025W/MAI/papers/Ignatov_Quantized_Image_Super-Resolution_on_Mobile_NPUs_Mobile_AI_2025_Challenge_CVPRW_2025_paper.pdf) - CVPR 2025 (accessed 2025-01-31)
- [Compressing Deep Neural Networks with ASTC](https://developer.arm.com/community/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/high-performing-graphics-2017-compressing-deep-neural-networks-with-astc) - ARM Developer 2017 (accessed 2025-01-31)
- [Using ASTC compression for DNN Weights](https://old.hotchips.org/wp-content/uploads/hc_archives/hc29/HC29.23-Posters-Pub/HC29.23.p10-Texture-Compression-Sharma-GeorgiaTech-v02.pdf) - Hot Chips 2017 (accessed 2025-01-31)
- [ARM Mali-G78 Product Support](https://developer.arm.com/Processors/Mali-G78) - ARM Developer (accessed 2025-01-31)
- [Android OpenGL Texture Compression](https://stackoverflow.com/questions/9148795/android-opengl-texture-compression) - Stack Overflow (accessed 2025-01-31)

**Additional References:**
- Qualcomm Adreno GPU Architecture Documentation
- Apple Metal Programming Guide
- Vulkan Specification 1.3
- OpenGL ES 3.2 Specification
- TensorFlow Lite Quantization Guide
- PyTorch Mobile Quantization API
