# Mobile VLM Deployment

## Overview

Deploying Vision-Language Models (VLMs) on mobile devices presents unique challenges due to computational constraints, memory limitations, and power budgets. However, mobile deployment offers critical advantages: enhanced privacy through on-device inference, low-latency interaction, offline operation, and support for real-time applications. This guide covers efficient mobile VLM architectures, model conversion frameworks, and platform-specific optimization strategies.

**Key Mobile VLM Challenges:**
- Limited compute capacity (CPU/GPU/NPU heterogeneity)
- Restricted memory bandwidth (typically 4-12 GB RAM)
- Tight thermal budgets (sustained operation at 60-95°C)
- Power constraints (battery life considerations)
- Model size restrictions (disk and memory footprint)

**Mobile VLM Advantages:**
- Privacy preservation (sensitive data stays on-device)
- Low latency (no network round-trip)
- Offline capability (no connectivity required)
- Reduced server costs (edge inference)

From [Efficient Deployment of Vision-Language Models on Mobile Devices](https://arxiv.org/html/2507.08505v1) (arXiv:2507.08505, accessed 2025-02-02):
- Study evaluates llama.cpp, MLC-Imp, and mllm frameworks on OnePlus 13R
- CPU overutilization (600-800% of 8 cores) dominates current deployments
- GPU/NPU underutilization represents major optimization opportunity
- Framework-level decisions impact performance as much as model architecture

## Mobile VLM Challenges

### Hardware Constraints

**Compute Heterogeneity:**
Mobile devices feature heterogeneous compute units with different characteristics:
- **CPU**: Sequential processing, GEMV-heavy workloads (token generation)
- **GPU**: Parallel processing, GEMM-heavy workloads (attention, vision encoding)
- **NPU/Neural Engine**: Specialized AI accelerators (often underutilized)

**Memory Limitations:**
- Typical mobile devices: 4-12 GB unified memory
- Model + KV cache + activations must fit in RAM
- Memory bandwidth constraints affect inference speed

**Thermal Constraints:**
- Sustained operation triggers thermal throttling at 70-95°C
- CPU-only inference can reach 90-95°C (10-12W power draw)
- GPU-offloaded inference maintains 60°C (1.3W power draw)

From [Efficient Deployment arXiv paper](https://arxiv.org/html/2507.08505v1) (accessed 2025-02-02):
- LLaVA-1.5 7B CPU-only: 90-95°C, 10-12W power consumption
- MobileVLM-3B CPU-only: 70-72°C, ~3.5W power consumption
- Imp-v1.5-3B GPU-accelerated: 60°C, 1.3W power consumption

### Pipeline Stage Characteristics

VLM inference consists of distinct stages with different computational profiles:

**Image Encoding:**
- GEMM-dominated (matrix multiplications in vision transformer)
- Benefits from GPU/NPU acceleration
- Typically 2-18 seconds depending on model size

**Feature Fusion/Projection:**
- Lightweight linear projections
- Maps vision features to LLM embedding space
- Minimal computational overhead

**Prompt Evaluation:**
- Parallel attention over entire context
- GEMM-heavy (batch matrix multiplications)
- GPU/NPU accelerated (when available)

**Token Generation:**
- Sequential autoregressive decoding
- GEMV-dominated (low parallelism)
- Often CPU-bound due to limited data reuse

From [MobileVLM GitHub](https://github.com/Meituan-AutoML/MobileVLM) (accessed 2025-02-02):
- MobileVLM achieves 21.5 tokens/sec on Qualcomm Snapdragon 888 CPU
- 65.3 tokens/sec on NVIDIA Jetson Orin GPU
- Lightweight Downsample Projector (LDP) reduces vision-to-language mapping overhead

## MobileVLM Architecture

MobileVLM is designed explicitly for mobile deployment, combining efficient components at every stage.

### Core Design Principles

From [MobileVLM GitHub](https://github.com/Meituan-AutoML/MobileVLM) (accessed 2025-02-02):

**MobileVLM V1 (December 2023):**
- 1.4B and 2.7B parameter variants
- Custom MobileLLaMA language models (trained from scratch)
- CLIP-based vision encoder (pre-trained)
- Lightweight Downsample Projector (LDP)
- Achieves on-par performance with 7B models

**MobileVLM V2 (February 2024):**
- Improved LDPv2 projector architecture
- Enhanced training scheme for mobile VLMs
- High-quality dataset curation
- 1.7B achieves comparable performance to 3B models
- 3B model outperforms many 7B+ VLMs

### Architecture Components

**Vision Encoder:**
- Pre-trained CLIP ViT-L/14@336px
- Frozen during VLM training (reduces memory)
- 336×336 input resolution
- Outputs grid of vision tokens

**Lightweight Downsample Projector (LDPv2):**
- Mobile-optimized projection module
- Reduces vision token count while preserving information
- Faster than standard MLPs
- Lower memory footprint

**MobileLLaMA Language Model:**
- Custom-trained small language models (1.4B, 2.7B parameters)
- Optimized for mobile inference
- Supports efficient quantization (INT8, INT4)

### Training Strategy

**Stage I: Pre-training (3-5 hours on 8×A100):**
- Frozen vision encoder ❄️
- Learnable LDPv2 projector 🔥
- Learnable LLM 🔥
- Dataset: ShareGPT4V-PT (1.2M image-text pairs)

**Stage II: Multi-task Fine-tuning (9-12 hours on 8×A100):**
- Frozen vision encoder ❄️
- Learnable LDPv2 projector 🔥
- Learnable LLM 🔥
- Dataset: MobileVLM_V2_FT_Mix2M (diverse multimodal tasks)

### Performance Benchmarks

From [MobileVLM GitHub](https://github.com/Meituan-AutoML/MobileVLM) (accessed 2025-02-02):

| Model | LLM | GQA | SQA | VQA-T | POPE | MME | MMB | Avg |
|-------|-----|-----|-----|-------|------|-----|-----|-----|
| MobileVLM V1-1.7B | MobileLLaMA 1.4B | 56.1 | 57.3 | 41.5 | 84.5 | 1196 | 53.2 | 58.7 |
| MobileVLM V2-1.7B | MobileLLaMA 1.4B | **59.3** | **66.7** | **52.1** | **84.3** | **1303** | **57.7** | **64.2** |
| MobileVLM V1-3B | MobileLLaMA 2.7B | 59.0 | 61.2 | 47.5 | 84.9 | 1289 | 59.6 | 62.8 |
| MobileVLM V2-3B | MobileLLaMA 2.7B | **61.1** | **70.0** | **57.5** | **84.7** | **1441** | **63.2** | **68.1** |
| MobileVLM V2-7B | Vicuna-7B | **62.6** | **74.8** | **62.3** | **85.3** | **1561** | **69.2** | **72.1** |

**Key Insight**: MobileVLM V2-1.7B matches or exceeds performance of many 3B models, demonstrating that architecture + training quality can compensate for parameter count.

## TinyLLaVA Architecture

TinyLLaVA focuses on efficient vision-language understanding through model distillation and architectural optimization.

From [arXiv paper on mobile VLM deployment](https://arxiv.org/html/2507.08505v1) and web research (accessed 2025-02-02):

### Design Philosophy

**Distillation-Based Approach:**
- Knowledge distillation from LLaVA-1.5 7B
- Maintains accuracy while reducing parameters
- Targets 3B parameter range

**Architecture Choices:**
- Phi-2 as language model backbone (2.7B parameters)
- SigLIP as vision encoder (more efficient than CLIP)
- Linear projection layer (simpler than LDP)

**Training Efficiency:**
- Leverages high-quality instruction tuning data
- Benefits from teacher model's learned representations
- Faster convergence than training from scratch

### Comparison with MobileVLM

**TinyLLaVA Strengths:**
- Simpler architecture (easier to understand/modify)
- Proven distillation pipeline
- Good accuracy-efficiency trade-off

**MobileVLM Strengths:**
- Custom-designed for mobile hardware
- More sophisticated projection (LDPv2)
- Better performance at same parameter count
- Explicit mobile optimization (memory, speed, power)

## Model Conversion for Mobile

Converting standard VLMs to mobile-friendly formats requires careful optimization across multiple dimensions.

### CoreML (iOS/macOS)

From [ONNX Runtime mobile deployment guide](https://onnxruntime.ai/docs/tutorials/mobile/) (accessed 2025-02-02):

**CoreML Overview:**
- Apple's framework for on-device machine learning
- Optimizes for Apple Silicon (CPU, GPU, Neural Engine)
- Supports models in `.mlmodel` or `.mlpackage` format

**Conversion Pipeline:**
```
PyTorch/TensorFlow Model
    ↓
ONNX Format (intermediate)
    ↓
CoreML Tools conversion
    ↓
.mlpackage (Core ML format)
    ↓
Xcode integration
```

**CoreML Execution Providers:**
- **CPU**: General purpose, consistent performance
- **GPU**: Parallel matrix operations, GEMM workloads
- **Neural Engine (ANE)**:
  - Specialized AI accelerator on A11+ and M1+ chips
  - Optimized for INT8/FP16 operations
  - Automatic scheduling by Core ML runtime
  - Limited operator support (not all ops accelerated)

From [Apple Machine Learning Research](https://machinelearning.apple.com/research/core-ml-on-device-llama) (accessed 2025-02-02):
- Llama 3.1 deployment on Apple Silicon achieves real-time performance
- Block-wise INT4 quantization for memory efficiency
- KV cache as model I/O for efficient autoregressive decoding
- Neural Engine utilization for compatible operators

**CoreML Conversion Best Practices:**
```python
import coremltools as ct

# Convert PyTorch model to Core ML
model = torch.load("mobilevlm.pth")
traced_model = torch.jit.trace(model, example_input)

# Convert to Core ML with optimizations
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(shape=(1, 3, 336, 336))],
    compute_units=ct.ComputeUnit.ALL,  # CPU + GPU + Neural Engine
    minimum_deployment_target=ct.target.iOS16,
)

# Save as mlpackage
mlmodel.save("MobileVLM.mlpackage")
```

**CoreML Quantization:**
- `ct.compression.quantize_weights()` for INT8/INT4 weights
- Mixed precision: FP16 activations, INT8 weights
- Palettization for further compression

### ONNX Runtime Mobile

From [ONNX Runtime mobile guide](https://onnxruntime.ai/docs/tutorials/mobile/) (accessed 2025-02-02):

**ONNX Runtime Mobile Overview:**
- Cross-platform inference engine (Android, iOS)
- Supports multiple execution providers
- Optimized operator kernels for mobile CPUs
- Extensible with custom operators

**Supported Execution Providers:**
- **CPU**: Default, universal compatibility
- **XNNPACK**: Optimized CPU kernels (highly recommended)
- **NNAPI** (Android): Android Neural Networks API
- **CoreML** (iOS): Apple's ML framework
- **QNN** (Qualcomm): Qualcomm AI Engine

**ONNX Conversion Pipeline:**
```
PyTorch/TensorFlow Model
    ↓
Export to ONNX format
    ↓
ONNX Optimizer (graph optimizations)
    ↓
Quantization (optional, INT8/INT4)
    ↓
ORT format conversion (optional, reduced size)
    ↓
Mobile deployment
```

**ONNX Runtime Mobile Packages:**
- **onnxruntime-android**: Java/C/C++ APIs for Android
- **onnxruntime-objc**: Objective-C APIs for iOS
- **onnxruntime-c**: C API (cross-platform)

**Android Deployment Example:**
```java
// Load ONNX model with NNAPI execution provider
OrtEnvironment env = OrtEnvironment.getEnvironment();
OrtSession.SessionOptions options = new OrtSession.SessionOptions();
options.addNnapi();  // Use Android NNAPI
options.setIntraOpNumThreads(4);

OrtSession session = env.createSession("mobilevlm.onnx", options);

// Run inference
OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputData);
OrtSession.Result result = session.run(Collections.singletonMap("input", inputTensor));
```

**iOS Deployment Example:**
```objective-c
// Load ONNX model with CoreML execution provider
NSError *error = nil;
ORTEnv *env = [[ORTEnv alloc] initWithLoggingLevel:ORTLoggingLevelWarning error:&error];

ORTSessionOptions *options = [[ORTSessionOptions alloc] initWithError:&error];
[options appendCoreMLExecutionProviderWithOptions:@{} error:&error];

ORTSession *session = [[ORTSession alloc] initWithEnv:env
                                             modelPath:@"mobilevlm.onnx"
                                          sessionOptions:options
                                                   error:&error];
```

### TensorFlow Lite

**TFLite Overview:**
- Google's mobile ML framework
- Optimized for Android (GPU delegate available)
- Supports quantization and pruning
- `.tflite` model format

**Conversion Process:**
```python
import tensorflow as tf

# Convert Keras/TensorFlow model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimization options
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # FP16 quantization

# Convert
tflite_model = converter.convert()

# Save
with open('mobilevlm.tflite', 'wb') as f:
    f.write(tflite_model)
```

**TFLite Delegates:**
- **GPU Delegate**: OpenGL-based acceleration
- **NNAPI Delegate**: Android Neural Networks API
- **Hexagon Delegate**: Qualcomm DSP acceleration

### Model Format Comparison

| Format | Platform | Size Reduction | Execution Providers | Ease of Use |
|--------|----------|----------------|---------------------|-------------|
| CoreML (.mlpackage) | iOS/macOS | Good (FP16, INT8) | CPU, GPU, ANE | Excellent (Xcode) |
| ONNX (.onnx) | Cross-platform | Excellent (INT8, INT4) | CPU, XNNPACK, NNAPI, CoreML, QNN | Good (flexible) |
| TFLite (.tflite) | Android-focused | Excellent (INT8) | CPU, GPU, NNAPI, Hexagon | Good (Android Studio) |
| ORT format (.ort) | Cross-platform | Excellent (custom build) | CPU, XNNPACK, NNAPI | Advanced (minimal build) |

## Platform-Specific Optimization

### iOS Optimization (Apple Neural Engine)

From [Apple Machine Learning Research - FastVLM](https://machinelearning.apple.com/research/fast-vision-language-models) (accessed 2025-02-02):

**Apple Neural Engine (ANE) Architecture:**
- Introduced with A11 Bionic (iPhone 8/X)
- Dedicated AI accelerator for ML workloads
- Optimized for INT8/FP16 operations
- Automatic scheduling by Core ML runtime

**FastVLM (Apple's Mobile VLM):**
- Hybrid vision encoder architecture
- 85× faster than baseline models
- 3× smaller model size
- Optimized for Apple Silicon (M1/M2/M3, A14+)
- Runs smoothly on MacBook Pro and iPhone

**Apple Silicon Optimization Strategies:**

**1. Neural Engine Utilization:**
- Prefer INT8 operations (ANE native support)
- Use FP16 for activations (GPU/ANE hybrid)
- Avoid unsupported ops (falls back to CPU)

**2. Unified Memory Architecture:**
- Zero-copy tensor sharing between CPU/GPU/ANE
- Reduce memory bandwidth usage
- Optimize tensor layouts for each processor

**3. Core ML Compute Units:**
```swift
let config = MLModelConfiguration()
config.computeUnits = .all  // CPU + GPU + Neural Engine
// or
config.computeUnits = .cpuAndGPU  // Exclude Neural Engine
// or
config.computeUnits = .cpuAndNeuralEngine  // Exclude GPU
```

**4. Model Compilation:**
- Core ML automatically compiles models for target hardware
- Ahead-of-time compilation reduces first-inference latency
- Operator fusion and graph optimizations

From [Core ML Llama deployment guide](https://machinelearning.apple.com/research/core-ml-on-device-llama) (accessed 2025-02-02):

**Llama 3.1 Optimization Techniques:**
- **Block-wise INT4 quantization**: Reduces memory 4×
- **KV cache management**: Stateful model I/O for efficiency
- **Operator fusion**: Combine multiple ops into single kernel
- **Palettization**: Further weight compression beyond quantization

**Performance Targets for Real-Time VLMs:**
- Vision encoding: <200ms (user expectation for "instant")
- Token generation: >10 tokens/sec (readable text streaming)
- Total latency: <2 seconds for first response

### Android Optimization (Qualcomm AI Engine)

From [Qualcomm AI Engine documentation](https://www.qualcomm.com/processors/ai-engine) (accessed 2025-02-02):

**Qualcomm AI Engine Architecture:**
- Hexagon NPU (Neural Processing Unit)
- Dedicated AI accelerator in Snapdragon SoCs
- Optimized for INT8 operations
- Vector processing for parallel AI workloads

**Snapdragon Optimization Strategies:**

**1. QNN (Qualcomm Neural Network) SDK:**
- Native SDK for Hexagon NPU
- INT8 quantization support
- Model compiler for NPU-specific optimizations
- Runtime scheduling across CPU/GPU/NPU

**2. NNAPI (Android Neural Networks API):**
- Android's standard ML API
- Abstracts hardware acceleration
- Supports Qualcomm, MediaTek, Samsung NPUs
- Driver-level optimizations

**3. Heterogeneous Compute:**
From [Efficient VLM Deployment paper](https://arxiv.org/html/2507.08505v1) (accessed 2025-02-02):
- **CPU**: Token generation (sequential GEMV)
- **GPU (Adreno)**: Vision encoding, attention (parallel GEMM)
- **NPU (Hexagon)**: Quantized inference (INT8 ops)

**Observed Performance on Snapdragon 8 Gen 2:**
- LLaVA-1.5 7B (CPU-only): 90-95°C, 10-12W, 101s latency
- MobileVLM-3B (CPU-only): 70-72°C, 3.5W, 35s latency
- Imp-v1.5-3B (GPU-accelerated): 60°C, 1.3W, 25s latency

**Key Findings:**
- CPU overutilization (600-800% of 8 cores) in most frameworks
- GPU/NPU significantly underutilized
- Framework-level optimization critical for efficiency

**Android Deployment Best Practices:**

**1. Use XNNPACK for CPU inference:**
```cpp
OrtSessionOptions options;
OrtSessionOptionsAppendExecutionProvider_Xnnpack(&options);
```

**2. Enable NNAPI for NPU acceleration:**
```java
SessionOptions options = new SessionOptions();
options.addNnapi();
options.setNnapiAcceleratorName("qti-default");  // Qualcomm Hexagon
```

**3. Thermal management:**
- Monitor device temperature
- Throttle inference rate if overheating
- Use lower precision (INT8) to reduce power

**4. Memory management:**
- Pre-allocate tensors to avoid runtime allocation
- Reuse KV cache across requests
- Release unused memory promptly

### Cross-Platform Deployment with llama.cpp

From [llama.cpp MobileVLM support](https://github.com/ggerganov/llama.cpp/blob/master/examples/llava/MobileVLM-README.md) (accessed 2025-02-02):

**llama.cpp Mobile Support:**
- Officially supports MobileVLM since January 2024
- Pure C/C++ implementation (no Python dependencies)
- Runs on iOS, Android, desktop platforms
- Quantization support (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0)

**Deployment Steps:**

**1. Convert model to GGUF format:**
```bash
python convert-mobilevlm-to-gguf.py \
    --model-path MobileVLM-1.7B \
    --output mobilevlm-1.7b.gguf
```

**2. Quantize model (optional but recommended):**
```bash
./quantize mobilevlm-1.7b.gguf mobilevlm-1.7b-q4_0.gguf Q4_0
```

**3. Build for mobile:**
```bash
# iOS
cmake -B build-ios -DCMAKE_TOOLCHAIN_FILE=ios.toolchain.cmake
cmake --build build-ios

# Android
cmake -B build-android -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake
cmake --build build-android
```

**Performance Characteristics:**
- MobileVLM-1.7B Q4_0: ~21.5 tokens/sec on Snapdragon 888 (CPU)
- MobileVLM-3B Q4_0: ~15 tokens/sec on Snapdragon 888 (CPU)
- Memory usage: ~2-4 GB depending on quantization

## Quantization and Compression

Quantization is essential for mobile VLM deployment, reducing model size and memory bandwidth requirements.

### Quantization Strategies

**Post-Training Quantization (PTQ):**
- No retraining required
- Calibration dataset for activation ranges
- INT8 (8-bit) or INT4 (4-bit) weights
- Minimal accuracy loss for VLMs (<2% typical)

**Quantization-Aware Training (QAT):**
- Simulate quantization during training
- Better accuracy preservation
- Longer training time
- Recommended for aggressive quantization (INT4, INT2)

### Quantization Formats

**INT8 Quantization:**
- 4× memory reduction (FP32 → INT8)
- Widely supported on mobile hardware
- Minimal accuracy degradation
- ~1-2% accuracy loss typical

**INT4 Quantization:**
- 8× memory reduction (FP32 → INT4)
- Requires careful calibration
- ~3-5% accuracy loss typical
- Block-wise quantization improves quality

**Mixed Precision:**
- Critical layers (early vision encoder) in FP16
- Middle layers in INT8
- Final projection in INT8
- Balances accuracy and efficiency

### KV Cache Compression

From [Apple Core ML Llama guide](https://machinelearning.apple.com/research/core-ml-on-device-llama) (accessed 2025-02-02):

**KV Cache Challenges:**
- Grows linearly with sequence length
- Dominates memory for long contexts
- Memory bandwidth bottleneck

**Optimization Techniques:**

**1. Quantized KV Cache:**
- Store keys/values in INT8 or FP16
- Dequantize on-the-fly during attention
- ~40% memory reduction (FP32 → FP16)

**2. Grouped-Query Attention (GQA):**
- Share KV cache across multiple query heads
- Reduces cache size proportionally
- Minimal accuracy impact

**3. Sliding Window Attention:**
- Limit attention to recent tokens
- Discard old KV cache entries
- Suitable for streaming inference

**4. KV Cache Pruning:**
- Remove less important cache entries
- Based on attention scores or heuristics
- Maintains critical context

## Mobile VLM Deployment Frameworks

### Framework Comparison

From [Efficient VLM Deployment study](https://arxiv.org/html/2507.08505v1) (accessed 2025-02-02):

| Framework | Platform Support | GPU/NPU Support | Quantization | Ease of Use | Performance |
|-----------|------------------|-----------------|--------------|-------------|-------------|
| **llama.cpp** | iOS, Android, Desktop | Limited (Metal for iOS) | Excellent (GGUF) | Good | CPU-optimized |
| **MLC-Imp** | iOS, Android | Excellent (GPU-first) | Good (INT4/INT8) | Moderate | GPU-optimized |
| **mllm** | Android | Good (NNAPI, XNNPACK) | Good | Moderate | Balanced |
| **ONNX Runtime** | iOS, Android | Excellent (many EPs) | Excellent | Good | Flexible |
| **Core ML** | iOS/macOS only | Excellent (ANE) | Good | Excellent | iOS-optimized |

### llama.cpp

**Strengths:**
- Pure C/C++ (no runtime dependencies)
- Excellent quantization support (GGUF format)
- Cross-platform (write once, run anywhere)
- Active community, frequent updates

**Limitations:**
- Primarily CPU-focused (limited GPU utilization)
- Manual memory management
- Less aggressive optimizations than platform-specific solutions

**Best For:**
- Cross-platform deployment
- CPU-constrained devices
- Applications requiring maximum compatibility

### MLC-Imp

**Strengths:**
- GPU-first design (offloads vision + attention)
- Excellent performance on GPU-capable devices
- TVM-based compilation for hardware-specific optimization
- Supports multiple backends (Metal, Vulkan, OpenCL)

**Limitations:**
- Complex build process
- Requires GPU for best performance
- Less mature than llama.cpp

**Best For:**
- High-end devices with capable GPUs
- Applications requiring lowest latency
- Developers comfortable with TVM ecosystem

From [MLC-Imp GitHub](https://github.com/MILVLG/mlc-imp) (accessed 2025-02-02):
- Imp-v1.5-3B achieves 60°C, 1.3W on GPU-accelerated inference
- Significant performance advantage over CPU-only approaches

### mllm

**Strengths:**
- Balanced CPU/GPU utilization
- XNNPACK integration for CPU
- NNAPI support for Android NPU
- Lightweight runtime

**Limitations:**
- Less documentation than alternatives
- Android-focused (limited iOS support)
- Smaller community

**Best For:**
- Android deployments
- Applications needing NPU acceleration
- Developers wanting simple API

### ONNX Runtime Mobile

**Strengths:**
- Most flexible (many execution providers)
- Excellent quantization tooling
- Microsoft backing (enterprise support)
- Comprehensive documentation

**Limitations:**
- Larger binary size than alternatives
- Requires understanding of execution providers
- Configuration complexity

**Best For:**
- Production deployments
- Applications needing multiple EPs
- Teams with ML engineering expertise

## Optimization Techniques Summary

### Model-Level Optimizations

**1. Architecture Selection:**
- Choose mobile-optimized models (MobileVLM, TinyLLaVA)
- Prefer efficient vision encoders (SigLIP vs CLIP)
- Use lightweight projection layers (LDP vs MLP)

**2. Quantization:**
- Post-training quantization for existing models
- Quantization-aware training for best accuracy
- Mixed precision for critical layers

**3. Pruning:**
- Structured pruning (remove entire attention heads)
- Unstructured pruning (remove individual weights)
- Magnitude-based or gradient-based selection

**4. Knowledge Distillation:**
- Distill from larger teacher models
- Preserve accuracy while reducing parameters
- Combine with quantization for maximum compression

### Runtime Optimizations

**1. Operator Fusion:**
- Combine multiple operations into single kernels
- Reduce memory access overhead
- Framework-level optimization (TVM, Core ML)

**2. Execution Provider Selection:**
- Profile different EPs for your workload
- CPU for GEMV-heavy stages (token generation)
- GPU/NPU for GEMM-heavy stages (attention, vision)

**3. Memory Management:**
- Pre-allocate tensors
- Reuse buffers across inference runs
- KV cache optimization (quantization, pruning)

**4. Thermal Management:**
- Monitor device temperature
- Throttle inference rate if needed
- Prefer GPU over CPU for sustained workloads (lower power)

### Framework-Level Optimizations

From [Efficient VLM Deployment study](https://arxiv.org/html/2507.08505v1) (accessed 2025-02-02):

**Key Insights:**
- Framework choice impacts performance as much as model size
- CPU overutilization (600-800%) is main bottleneck
- GPU/NPU underutilization represents major opportunity
- Careful scheduling across heterogeneous compute units critical

**Best Practices:**
- Offload vision encoding to GPU (2-18s → <1s potential)
- Use XNNPACK for CPU token generation
- Enable NPU for quantized models (Hexagon, ANE)
- Profile actual hardware utilization (don't trust defaults)

## Practical Deployment Workflow

### Step 1: Model Selection and Conversion

```bash
# Choose appropriate model for target device
# - High-end phones: MobileVLM-3B or Imp-v1.5-3B
# - Mid-range phones: MobileVLM-1.7B
# - Low-end phones: TinyLLaVA-1.5B

# Convert PyTorch to ONNX
python convert_to_onnx.py \
    --model MobileVLM-3B \
    --output mobilevlm-3b.onnx

# Optimize ONNX graph
python -m onnxruntime.transformers.optimizer \
    --input mobilevlm-3b.onnx \
    --output mobilevlm-3b-opt.onnx

# Quantize to INT8
python -m onnxruntime.quantization.quantize_dynamic \
    --model_input mobilevlm-3b-opt.onnx \
    --model_output mobilevlm-3b-int8.onnx \
    --per_channel
```

### Step 2: Platform-Specific Preparation

**iOS (Core ML):**
```python
import coremltools as ct

# Convert to Core ML
mlmodel = ct.convert(
    "mobilevlm-3b.onnx",
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.iOS16,
)

# Quantize weights
mlmodel = ct.compression.quantize_weights(mlmodel, nbits=8)

# Save
mlmodel.save("MobileVLM.mlpackage")
```

**Android (ONNX Runtime):**
```gradle
// Add ONNX Runtime dependency
dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.18.0'
    // or with NNAPI support
    implementation 'com.microsoft.onnxruntime:onnxruntime-android-nnapi:1.18.0'
}
```

### Step 3: Integration and Optimization

**Measure Baseline Performance:**
```python
import time
import psutil

# Measure latency
start = time.time()
output = model.run(input_data)
latency = time.time() - start

# Measure memory
memory_mb = psutil.Process().memory_info().rss / 1024 / 1024

# Measure CPU/GPU utilization
cpu_percent = psutil.cpu_percent(interval=1)
# (platform-specific GPU monitoring)

print(f"Latency: {latency:.2f}s")
print(f"Memory: {memory_mb:.0f} MB")
print(f"CPU: {cpu_percent:.1f}%")
```

**Optimize Based on Bottlenecks:**
- High latency → Try GPU/NPU acceleration
- High memory → Aggressive quantization (INT4)
- High CPU usage → Offload to GPU
- Thermal throttling → Reduce inference rate

### Step 4: Production Deployment

**Monitoring:**
- Track inference latency (p50, p95, p99)
- Monitor device temperature
- Measure battery drain
- Log framework errors/fallbacks

**Graceful Degradation:**
- Fallback to CPU if GPU unavailable
- Reduce image resolution under thermal stress
- Queue requests if device overheating
- Skip frames in video processing

## Case Study: Efficient VLM Deployment on Snapdragon

From [Efficient VLM Deployment on Mobile Devices](https://arxiv.org/html/2507.08505v1) (arXiv:2507.08505, accessed 2025-02-02):

**Experimental Setup:**
- Device: OnePlus 13R (Snapdragon 8 Gen 2)
- Models: LLaVA-1.5 7B, MobileVLM-3B, Imp-v1.5-3B
- Frameworks: llama.cpp, MLC-Imp, mllm

**Key Findings:**

**CPU-Only Deployments (llama.cpp + LLaVA-1.5 7B):**
- CPU utilization: 600-800% (of 8 cores)
- Temperature: 90-95°C (thermal throttling)
- Power consumption: 10-12W
- Latency: 101 seconds end-to-end
- GPU/NPU: 0% utilization (completely idle)

**CPU-Only Mobile Model (llama.cpp + MobileVLM-3B):**
- CPU utilization: 500-800%
- Temperature: 70-72°C (improved)
- Power consumption: ~3.5W
- Latency: 35 seconds end-to-end
- Memory: ~32% of 16GB (~5.1 GB)

**GPU-Accelerated (MLC-Imp + Imp-v1.5-3B):**
- CPU utilization: 120% (single core + background)
- GPU utilization: 90-100% (fully utilized)
- Temperature: 60°C (comfortable)
- Power consumption: 1.3W (8× lower than CPU-only)
- Latency: 25 seconds end-to-end
- Battery impact: ~2 days vs ~8 hours (one query/minute)

**Latency Breakdown:**

| Model | Framework | Image Encode | Prompt Eval | Token Gen | Total |
|-------|-----------|--------------|-------------|-----------|-------|
| LLaVA-1.5 7B | llama.cpp | 2.4s | 70.4s | 11.1s | 82.2s |
| LLaVA-1.5 7B | mllm | 0.04s | 78.9s | 90.2s | 173.7s |
| MobileVLM-3B | llama.cpp | 3.1s | 15.3s | 6.6s | 22.8s |
| Imp-v1.5-3B | MLC-Imp | 18.0s | 2.0s | 1.0s | 25.0s |

**Critical Observations:**
1. Framework-level differences massive (82s vs 174s for same model)
2. GPU offloading reduces temperature by 30°C
3. Power consumption varies by 8× (1.3W vs 10W)
4. CPU bottleneck prevents GPU/NPU utilization in most cases
5. Proper scheduling critical for mobile deployment

## Future Directions

### Emerging Techniques

**1. Speculative Decoding:**
- Draft tokens with small model
- Verify with main model
- Potentially 2-3× speedup for token generation

**2. Dynamic Resolution:**
- Adjust image resolution based on complexity
- High-res for detailed images, low-res for simple scenes
- Save computation without sacrificing quality

**3. Mixture of Experts (MoE):**
- Activate subset of parameters per token
- Reduce computation while maintaining capacity
- Challenging on mobile (memory bandwidth)

**4. Neural Architecture Search (NAS):**
- Hardware-aware model design
- Optimize for specific mobile platforms
- Automated architecture discovery

### Hardware Advancements

**Next-Generation Mobile NPUs:**
- Apple M4/A18: Enhanced Neural Engine
- Qualcomm Snapdragon 8 Gen 4: Improved Hexagon
- MediaTek Dimensity 9400: Advanced APU

**Unified Memory Optimizations:**
- Better CPU/GPU/NPU memory sharing
- Reduced copy overhead
- Improved bandwidth utilization

## Sources

**Research Papers:**
- [Efficient Deployment of Vision-Language Models on Mobile Devices: A Case Study on OnePlus 13R](https://arxiv.org/html/2507.08505v1) - arXiv:2507.08505 (accessed 2025-02-02)

**GitHub Repositories:**
- [MobileVLM: Strong and Open Vision Language Assistant for Mobile Devices](https://github.com/Meituan-AutoML/MobileVLM) (accessed 2025-02-02)
- [llama.cpp MobileVLM Support](https://github.com/ggerganov/llama.cpp/blob/master/examples/llava/MobileVLM-README.md) (accessed 2025-02-02)
- [MLC-Imp: MLC Implementation for Imp models](https://github.com/MILVLG/mlc-imp) (accessed 2025-02-02)

**Official Documentation:**
- [ONNX Runtime Mobile Deployment Guide](https://onnxruntime.ai/docs/tutorials/mobile/) (accessed 2025-02-02)
- [Apple Machine Learning Research - FastVLM](https://machinelearning.apple.com/research/fast-vision-language-models) (accessed 2025-02-02)
- [Apple Machine Learning Research - Core ML Llama Deployment](https://machinelearning.apple.com/research/core-ml-on-device-llama) (accessed 2025-02-02)
- [Qualcomm AI Engine Documentation](https://www.qualcomm.com/processors/ai-engine) (accessed 2025-02-02)

**Additional References:**
- Web research on iOS/Android VLM deployment (accessed 2025-02-02)
- Web research on Apple Neural Engine and Qualcomm AI Engine (accessed 2025-02-02)
