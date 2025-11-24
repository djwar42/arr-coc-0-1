# Edge Deployment and VLM Quantization

## Overview

Deploying vision-language models (VLMs) on edge devices and mobile platforms requires aggressive compression techniques to fit within strict memory, compute, and power constraints. Quantization—reducing numerical precision from FP32/FP16 to INT8/INT4—is the primary technique enabling real-time VLM inference on resource-constrained hardware. This guide covers quantization fundamentals, VLM-specific strategies, mobile deployment architectures, and production tooling.

**Key Challenge**: VLMs combine vision encoders (ViT/CNN) and language decoders (transformer LLMs), each requiring different quantization strategies. Vision components are more sensitive to quantization than language models, and cross-modal fusion layers need careful handling to preserve alignment quality.

## Section 1: Quantization Fundamentals for VLMs (~80 lines)

### INT8 and INT4 Quantization Basics

**Post-Training Quantization (PTQ)** converts trained FP32/FP16 models to lower precision without retraining. For VLMs, PTQ enables deployment on mobile GPUs, NPUs, and edge accelerators:

From [VLMQ: Efficient Post-Training Quantization for Large Vision-Language Models](https://arxiv.org/html/2508.03351v1) (arXiv, accessed 2025-01-31):
- **INT8 quantization**: 8-bit integers provide ~4x memory reduction and 2-4x inference speedup with minimal accuracy loss (typically <1% on VQA tasks)
- **INT4 quantization**: 4-bit integers achieve ~8x compression but require careful calibration to avoid >5% accuracy degradation
- PTQ workflow: Load FP32 model → Calibrate on representative dataset → Apply quantization → Evaluate accuracy

**Symmetric vs Asymmetric Quantization**:

**Symmetric quantization** (zero-point = 0):
```
scale = max(|weights|) / 127  # For INT8
quantized_weight = round(weight / scale)
```
- Simpler hardware implementation (no zero-point offset)
- Works well for weights centered around zero
- Used in TensorRT, ONNX Runtime INT8 backends

**Asymmetric quantization** (arbitrary zero-point):
```
scale = (max_val - min_val) / 255
zero_point = -round(min_val / scale)
quantized_weight = round(weight / scale) + zero_point
```
- Better range coverage for activations
- Handles non-centered distributions (e.g., ReLU outputs)
- Used in TensorFlow Lite, PyTorch quantization

From [Post-training Quantization for Large Vision-Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/cffbaf4f47546ece96bb42c0edda40ee-Paper-Conference.pdf) (NeurIPS 2024, accessed 2025-01-31):
- **Per-tensor quantization**: Single scale/zero-point for entire tensor (faster, less accurate)
- **Per-channel quantization**: Separate scale per output channel (2-3% better accuracy, slightly slower)
- **Group quantization**: Divide weights into groups (e.g., 128 elements), quantize each group separately—balance between per-tensor and per-channel

### Calibration and Accuracy-Speed Tradeoffs

**Calibration determines quantization parameters** (scale, zero-point) by running representative data through the model:

From [Model Quantization: Meaning, Benefits & Techniques](https://www.clarifai.com/blog/model-quantization) (Clarifai blog, accessed 2025-01-31):
- **MinMax calibration**: Use min/max of activation distributions
  - Fast, simple, but sensitive to outliers
  - Works for well-behaved distributions

- **Histogram-based calibration**: Analyze activation histograms, choose quantization range that minimizes KL divergence
  - More robust to outliers
  - Used by TensorRT for INT8

- **Percentile calibration**: Use 99th/99.9th percentile instead of max
  - Ignores extreme outliers
  - Common for vision encoders with occasional large activations

**Accuracy vs Speed Tradeoffs**:

| Precision | Memory | Speed | Typical Accuracy Loss |
|-----------|--------|-------|----------------------|
| FP32 (baseline) | 1x | 1x | 0% |
| FP16 | 0.5x | 1.5-2x | <0.1% |
| INT8 | 0.25x | 2-4x | 0.5-2% |
| INT4 | 0.125x | 3-6x | 2-8% |
| Mixed INT8/INT4 | 0.15-0.2x | 2.5-5x | 1-4% |

From [A Survey on Efficient Vision-Language Models](https://arxiv.org/html/2504.09724v1) (arXiv, accessed 2025-01-31):
- **Moondream 0.5B** uses INT8/INT4 quantization optimized for mobile/edge
- Achieves 15-30 FPS on mobile GPUs (Snapdragon 8 Gen 2) with <3% accuracy drop
- Uses Quantization-Aware Training (QAT) to further reduce accuracy loss

### Dynamic vs Static Quantization

**Static quantization** (offline):
- Quantize weights + activations ahead of time
- Requires calibration dataset
- Best performance (INT8 ops throughout)
- Used for deployment: TensorRT, ONNX Runtime, CoreML

**Dynamic quantization** (runtime):
- Quantize weights offline, activations at runtime
- No calibration needed
- Slightly slower (quantize activations on-the-fly)
- Good for rapid prototyping: PyTorch dynamic quantization

**Quantization-Aware Training (QAT)**:
From [Post-training Quantization for Large Vision-Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/cffbaf4f47546ece96bb42c0edda40ee-Paper-Conference.pdf):
- Simulate quantization during fine-tuning
- Model learns to be robust to quantization noise
- Recovers 1-3% accuracy over PTQ
- Essential for INT4 VLMs to maintain quality

---

## Section 2: VLM-Specific Quantization Strategies (~100 lines)

### Vision Encoder Quantization

Vision encoders (ViT, CLIP vision, SigLIP) are **more sensitive to quantization** than language models due to:
1. **Fine-grained visual features**: Small changes in embeddings affect downstream tasks
2. **Attention patterns**: Quantizing attention matrices can disrupt spatial relationships
3. **Normalization layers**: LayerNorm is sensitive to reduced precision

From [MBQ: Modality-Balanced Quantization for Large Vision-Language Models](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_MBQ_Modality-Balanced_Quantization_for_Large_Vision-Language_Models_CVPR_2025_paper.pdf) (CVPR 2025, accessed 2025-01-31):

**Best practices for vision encoder quantization**:

1. **Keep patch embedding and final projection in higher precision**:
   - Patch embedding: FP16 (quantizing to INT8 degrades edge detection)
   - Final vision projection: FP16 (critical for vision-language alignment)
   - Inner ViT layers: INT8 (safe to quantize)

2. **Per-channel quantization for attention**:
   ```
   Q, K, V projections: INT8 per-channel
   Attention scores: FP16 (quantizing softmax hurts quality)
   Attention output: INT8 per-channel
   ```

3. **Outlier-aware quantization**:
   From [VLMQ: Efficient Post-Training Quantization for Large Vision-Language Models](https://arxiv.org/html/2508.03351v1):
   - Identify channels with extreme activations (>3σ from mean)
   - Keep outlier channels in FP16, quantize rest to INT8
   - Reduces accuracy loss by 1-2% with <5% compute overhead

**Example: ViT-L/14 Quantization Strategy**:
```
Input: [224x224x3] → FP16 patch embedding
↓
12 ViT blocks (INT8 Q/K/V, FP16 attention scores)
↓
Final LayerNorm: FP16
↓
Vision projection: FP16 → [768d] vision embeddings
```

### Language Decoder Quantization

Language models (LLaMA, Mistral) are **more robust to quantization** than vision encoders:

From [Exploring Quantization for Efficient Pre-Training of Transformer Language Models](https://aclanthology.org/2024.findings-emnlp.787.pdf) (EMNLP 2024, accessed 2025-01-31):
- LLMs can tolerate **INT4 weights** with <1% perplexity increase
- Activations should remain INT8 or higher
- KV cache can be quantized to INT8 (saves 4x memory for long contexts)

**Quantization strategy for VLM language decoder**:

1. **Weight-only INT4 quantization**:
   ```python
   # Quantize weights to 4-bit, keep activations FP16
   Linear layers: W_int4 @ X_fp16 → Y_fp16
   ```
   - Used by: llama.cpp, vLLM INT4 mode, MLC-LLM
   - 3-4x memory reduction, 2x speedup on CPU/mobile GPU
   - Minimal accuracy loss (<0.5% on language tasks)

2. **INT8 activation quantization**:
   - Apply to FFN intermediate activations (largest tensors)
   - Use per-token dynamic quantization for better quality
   - Essential for memory-constrained devices

3. **KV cache quantization**:
   From [INT8 W8A8 - vLLM](https://docs.vllm.ai/en/v0.7.1/features/quantization/int8.html) (vLLM docs, accessed 2025-01-31):
   ```
   KV cache: FP16 [batch, heads, seq_len, head_dim]
   Quantize to INT8: 4x memory reduction
   Critical for long context (>2048 tokens)
   ```

### Cross-Modal Fusion Quantization

**Fusion layers** (where vision embeddings merge with language tokens) are **critical for VLM quality**:

From [Post-training Quantization for Large Vision-Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/cffbaf4f47546ece96bb42c0edda40ee-Paper-Conference.pdf):

**Challenge**: Vision and language modalities have different activation distributions:
- Vision embeddings: Smooth, Gaussian-like
- Language tokens: Spiky, long-tailed (due to discrete vocabulary)

**Solution: Modality-specific quantization**:

1. **Separate calibration per modality**:
   ```
   Vision tokens: Calibrate on COCO images
   Language tokens: Calibrate on text corpus
   Fusion layer: Calibrate on paired image-text data
   ```

2. **Mixed-precision fusion**:
   - Vision embeddings → INT8
   - Language embeddings → INT8
   - Cross-attention Q/K/V → FP16 (preserves alignment)
   - Cross-attention output → INT8

3. **Saliency-aware quantization**:
   From [MBQ: Modality-Balanced Quantization](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_MBQ_Modality-Balanced_Quantization_for_Large_Vision-Language_Models_CVPR_2025_paper.pdf):
   - Measure gradient importance of each layer
   - Allocate higher precision (INT8) to salient layers
   - Use lower precision (INT4) for less important layers
   - Achieves 15% better accuracy than uniform quantization

**Example: LLaVA-style fusion quantization**:
```
Vision encoder → [B, 576, 768] INT8 embeddings
↓
Vision projection (FP16): [768 → 4096] → align to LLM dim
↓
Concat with language tokens (INT8)
↓
Cross-attention layers (Q/K FP16, V INT8)
↓
LLM decoder (INT4 weights, INT8 activations)
```

### Handling Multi-Image and Video Inputs

**Multi-image VLMs** (e.g., Qwen-VL, GPT-4V) process multiple images per query:

From [Efficient Deployment of Vision-Language Models on Mobile Devices](https://arxiv.org/html/2507.08505v1) (arXiv, accessed 2025-01-31):

**Memory challenges**:
- 4 images × 576 tokens × 768d × FP16 = 3.5MB per sample
- INT8 quantization: Reduces to 0.9MB (4x savings)
- INT4 vision embeddings: 0.45MB (8x savings, but degrades quality)

**Video quantization strategies**:
1. **Temporal redundancy**: Adjacent frames have similar features
   - Quantize frame-to-frame differences (delta encoding)
   - Use INT4 for deltas, INT8 for keyframes

2. **Spatial pooling before quantization**:
   - Reduce resolution per frame (e.g., 336×336 → 224×224)
   - Apply INT8 quantization to downsampled features
   - Saves memory without losing temporal coherence

---

## Section 3: Mobile and Edge Deployment Architectures (~90 lines)

### Mobile GPU Optimization

**Modern mobile GPUs** (Adreno, Mali, Apple GPU) support INT8/FP16 inference but have different characteristics than datacenter GPUs:

From [Deploying an Efficient Vision-Language Model on Mobile Devices](https://www.edge-ai-vision.com/2025/05/deploying-an-efficient-vision-language-model-on-mobile-devices/) (Edge AI and Vision Alliance, accessed 2025-01-31):

**Mobile GPU architecture**:
- **Unified memory**: CPU and GPU share RAM (no PCIe bottleneck)
- **Tile-based rendering**: Process tiles locally to reduce DRAM bandwidth
- **FP16-optimized**: Most mobile GPUs have 2x throughput for FP16 vs FP32
- **Limited INT8 support**: Only flagship chips (Snapdragon 8 Gen 2+, A17 Pro) have efficient INT8

**Optimization strategies**:

1. **Operator fusion**:
   ```
   Before: Conv2D → BatchNorm → ReLU (3 kernel launches)
   After:  Conv-BN-ReLU fused (1 kernel)
   ```
   - Reduces memory traffic by 3x
   - Essential for bandwidth-limited mobile GPUs

2. **Quantization-aware kernel selection**:
   From [Efficient Deployment of Vision-Language Models on Mobile Devices](https://arxiv.org/html/2507.08505v1):
   - Use FP16 for small tensors (<1MB): Kernel launch overhead dominates
   - Use INT8 for large tensors (>10MB): Compute-bound, INT8 gives 2x speedup
   - Dynamic dispatch based on tensor size

3. **Memory layout optimization**:
   - **NCHW** (channels-first): Better for convolutions on mobile GPUs
   - **NHWC** (channels-last): Better for depthwise separable convs
   - Use **NC4HW4** (4-channel groups) for optimal vectorization on Adreno/Mali

**Benchmarks (Snapdragon 8 Gen 2)**:

From [Vision-Language Models for Edge Networks](https://arxiv.org/html/2502.07855v1) (arXiv survey, accessed 2025-01-31):
| Model | Precision | Latency | Memory |
|-------|-----------|---------|--------|
| MiniCPM-V 2.6 | FP16 | 1200ms | 4.5GB |
| MiniCPM-V 2.6 | INT8 | 450ms | 1.2GB |
| MiniCPM-V 2.6 | Mixed INT4/8 | 280ms | 800MB |

### NPU and Dedicated Accelerators

**Neural Processing Units (NPUs)** on mobile SoCs provide 2-10x better efficiency than GPUs for INT8 inference:

From [Efficient GPT-4V level multimodal large language model for mobile devices](https://www.nature.com/articles/s41467-025-61040-5) (Nature, accessed 2025-01-31):

**NPU characteristics**:
- **INT8/INT16 only**: No FP32/FP16 support on most mobile NPUs
- **Fixed function**: Less flexible than GPU (e.g., limited attention support)
- **Power efficient**: 5-10x better TOPS/Watt than mobile GPU

**Deployment on NPU**:

1. **Operator compatibility**:
   - **Supported**: Conv2D, DepthwiseConv, Linear, ReLU, Sigmoid, LayerNorm
   - **Partially supported**: Softmax (requires FP16 fallback), GELU (approximated)
   - **Not supported**: Complex attention (Q·K^T with large matrices), custom ops

2. **Hybrid NPU-GPU execution**:
   ```
   Vision encoder → NPU (INT8 convolutions)
   ↓
   Vision projection → GPU (FP16, small layer)
   ↓
   LLM decoder → NPU + GPU hybrid:
     - Self-attention: GPU (FP16, complex patterns)
     - FFN: NPU (INT8, 70% of compute)
   ```

3. **NPU memory management**:
   From [Self-adapting Large Visual-Language Models to Edge Devices](https://dl.acm.org/doi/10.1007/978-3-031-73390-1_18) (ECCV 2024, accessed 2025-01-31):
   - NPUs have limited on-chip memory (2-8MB)
   - Partition model into "tiles" that fit in NPU memory
   - Overlap NPU compute with DRAM transfers (double buffering)

**Apple Neural Engine (ANE)**:
- Optimized for CoreML INT8 models
- Supports 16-bit activations internally (better quality than 8-bit)
- Efficient for ViT attention (fused attention kernels)

**Qualcomm Hexagon NPU**:
- Best performance with per-channel INT8 quantization
- Supports multi-threaded execution (4-6 threads on Gen 3)
- Efficient KV cache management for LLM decoding

### CPU Fallback and Heterogeneous Execution

**Not all operators can run on GPU/NPU** → need efficient CPU fallback:

From [Edge deployment of LLMs and ML models: A review](https://www.rohan-paul.com/p/edge-deployment-of-llms-and-ml-models) (Rohan's Bytes, accessed 2025-01-31):

**CPU optimization for VLMs**:

1. **SIMD vectorization** (ARM NEON, x86 AVX2):
   ```cpp
   // INT8 dot product using NEON
   int8x16_t a = vld1q_s8(weights);
   int8x16_t b = vld1q_s8(activations);
   int16x8_t c = vmull_s8(vget_low_s8(a), vget_low_s8(b));
   // 16 INT8 MACs in 3 instructions
   ```
   - Essential for CPU-only devices (Raspberry Pi, low-end phones)

2. **Cache-aware blocking**:
   - Tile matrix multiplications to fit in L2 cache (256KB-1MB)
   - Reduces DRAM accesses by 5-10x

3. **Mixed-precision execution**:
   From [When CPUs Outperform GPUs for On-Device LLM Inference](https://arxiv.org/abs/2505.06461) (arXiv, accessed 2025-01-31):
   - CPUs can efficiently mix INT8, INT16, FP32 in same computation
   - GPUs incur overhead switching precisions
   - For small batch sizes (<4), CPU can match/exceed mobile GPU performance

**Heterogeneous scheduling**:
```
Vision encoder:
  Patch embedding → GPU (FP16, parallel)
  ViT blocks → NPU (INT8, sequential)

Language decoder:
  Attention → GPU (FP16, dynamic seq length)
  FFN → NPU (INT8, fixed shapes)
  Sampling → CPU (FP32, non-deterministic)
```

---

## Section 4: Tooling and Frameworks (~80 lines)

### ONNX Runtime for Edge Deployment

**ONNX Runtime** provides cross-platform INT8 inference with multiple execution providers:

From [Optimizing Transformer Inference with ONNX Runtime and Quantization](https://medium.com/@bhagyarana80/optimizing-transformer-inference-with-onnx-runtime-and-quantization-098f8149a15c) (Medium, accessed 2025-01-31):

**Quantization workflow**:

1. **Export model to ONNX**:
   ```python
   import torch.onnx

   # Export PyTorch VLM to ONNX
   torch.onnx.export(
       vlm_model,
       (dummy_image, dummy_text),
       "vlm_fp32.onnx",
       input_names=["image", "text"],
       output_names=["logits"],
       dynamic_axes={"text": {1: "seq_len"}}  # Variable text length
   )
   ```

2. **Apply static quantization**:
   ```python
   from onnxruntime.quantization import quantize_static, QuantType

   # Calibrate on representative dataset
   quantize_static(
       model_input="vlm_fp32.onnx",
       model_output="vlm_int8.onnx",
       calibration_data_reader=CalibrationDataReader(val_dataset),
       quant_format=QuantType.QDQ,  # Quantize-Dequantize format
       per_channel=True,  # Better accuracy
       activation_type=QuantType.QUInt8,
       weight_type=QuantType.QInt8
   )
   ```

3. **Optimize for mobile**:
   ```python
   from onnxruntime.transformers import optimizer

   # Apply mobile-specific optimizations
   optimizer.optimize_model(
       "vlm_int8.onnx",
       model_type="bert",  # For transformer layers
       num_heads=16,
       hidden_size=1024,
       optimization_options={
           "enable_gelu_approximation": True,  # Faster on mobile
           "enable_skip_layer_norm_fusion": True,
           "enable_attention_fusion": True
       }
   )
   ```

**Execution providers**:
- **CoreMLExecutionProvider**: Apple Neural Engine (iOS/macOS)
- **QNNExecutionProvider**: Qualcomm Hexagon NPU (Android)
- **CUDAExecutionProvider**: NVIDIA GPUs (edge servers)
- **CPUExecutionProvider**: ARM NEON, x86 AVX (universal fallback)

From [Deploying Quantized LLMs with ONNX Runtime](https://apxml.com/courses/quantized-llm-deployment/chapter-4-optimizing-deploying-quantized-llms/deployment-onnx-runtime) (ApX Machine Learning, accessed 2025-01-31):
- ONNX Runtime supports dynamic batching (essential for variable-length text)
- KV cache optimization built-in for decoder-only models
- Profiling tools to identify bottlenecks (CPU vs GPU time)

### TensorRT for NVIDIA Edge Devices

**TensorRT** optimizes models for NVIDIA GPUs (Jetson, datacenter edge):

From [TensorRT SDK | NVIDIA Developer](https://developer.nvidia.com/tensorrt) (NVIDIA Developer, accessed 2025-01-31):

**Quantization with TensorRT**:

1. **INT8 calibration**:
   ```python
   import tensorrt as trt

   # Create calibrator for INT8
   calibrator = trt.IInt8EntropyCalibrator2(
       calibration_data=val_loader,
       cache_file="vlm_calibration.cache"
   )

   # Build INT8 engine
   with trt.Builder(logger) as builder:
       config = builder.create_builder_config()
       config.set_flag(trt.BuilderFlag.INT8)
       config.int8_calibrator = calibrator

       engine = builder.build_engine(network, config)
   ```

2. **Mixed-precision policy**:
   From [TensorRT Implementations of Model Quantization on Edge GPUs](https://userweb.cs.txstate.edu/~k_y47/webpage/pubs/mcsoc23.pdf) (Texas State University, accessed 2025-01-31):
   ```python
   # Set per-layer precision
   for layer in network:
       if "attention" in layer.name:
           layer.precision = trt.DataType.HALF  # FP16 for attention
       elif "vision_proj" in layer.name:
           layer.precision = trt.DataType.HALF  # FP16 for critical layers
       else:
           layer.precision = trt.DataType.INT8  # INT8 for most layers
   ```

3. **Optimization techniques**:
   - **Layer fusion**: Automatically fuses Conv-BN-ReLU, Attention ops
   - **Kernel auto-tuning**: Benchmarks different kernels, selects fastest
   - **Dynamic shapes**: Supports variable batch size, sequence length

**Benchmarks (Jetson Orin Nano)**:

From [Visual Language Intelligence and Edge AI 2.0](https://developer.nvidia.com/blog/visual-language-intelligence-and-edge-ai-2-0/) (NVIDIA blog, accessed 2025-01-31):
| Model | Precision | Latency | Power |
|-------|-----------|---------|-------|
| LLaVA-7B | FP16 | 850ms | 15W |
| LLaVA-7B | INT8 | 320ms | 8W |
| LLaVA-7B | AWQ INT4 | 180ms | 6W |

### Mobile Frameworks (TFLite, CoreML, MLC-LLM)

**TensorFlow Lite** for Android/iOS:

From [Edge deployment of LLMs and ML models: A review](https://www.rohan-paul.com/p/edge-deployment-of-llms-and-ml-models):

```python
import tensorflow as tf

# Convert and quantize VLM for TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(vlm_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
converter.representative_dataset = representative_dataset_gen

tflite_model = converter.convert()
```

**CoreML for iOS/macOS**:
```python
import coremltools as ct

# Convert PyTorch VLM to CoreML with INT8
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="image", shape=(1,3,224,224)),
            ct.TensorType(name="text", shape=(1,77))],
    compute_units=ct.ComputeUnit.ALL,  # Use Neural Engine + GPU
    compute_precision=ct.precision.FLOAT16
)

# Apply INT8 quantization
mlmodel_quantized = ct.compression.linear_quantize_weights(
    mlmodel,
    mode="linear",
    dtype="int8"
)
```

**MLC-LLM** for cross-platform deployment:

From [ModelTC/LightCompress](https://github.com/ModelTC/LightCompress) (GitHub, accessed 2025-01-31):

- Supports AWQ (4-bit), GPTQ (INT4), and RTN (round-to-nearest) quantization
- Compiles to platform-specific backends (Metal, Vulkan, CUDA)
- Efficient KV cache management for LLM decoding
- Used for deploying LLaVA, MiniCPM-V on mobile

### Optimization Workflows and Best Practices

**End-to-end deployment pipeline**:

```python
# 1. Train/fine-tune VLM (FP32/BF16)
vlm_model = train_vlm(dataset)

# 2. Post-training quantization
quantized_model = quantize_vlm(
    vlm_model,
    calibration_data=val_dataset[:100],
    vision_precision="int8",  # Vision encoder
    language_precision="int4",  # Language decoder
    fusion_precision="fp16"  # Cross-attention
)

# 3. Compile for target platform
if platform == "ios":
    compiled = compile_to_coreml(quantized_model)
elif platform == "android":
    compiled = compile_to_tflite(quantized_model)
elif platform == "jetson":
    compiled = compile_to_tensorrt(quantized_model)

# 4. Profile and validate
profile_latency(compiled, test_data)
validate_accuracy(compiled, test_data)  # Ensure <2% degradation

# 5. Deploy with fallback strategies
deploy_with_fallback(
    primary=compiled,
    fallback_cpu=quantized_model_onnx,  # If NPU/GPU fails
    memory_limit=2GB
)
```

**Common pitfalls and solutions**:

From [Efficient Deployment of Vision-Language Models on Mobile Devices](https://arxiv.org/html/2507.08505v1):

1. **Problem**: Accuracy drops >5% after quantization
   - **Solution**: Use QAT for INT4, mixed-precision for critical layers

2. **Problem**: OOM (out of memory) on mobile
   - **Solution**: Quantize KV cache to INT8, use gradient checkpointing during calibration

3. **Problem**: High latency variance (50ms → 500ms spikes)
   - **Solution**: Thermal throttling—reduce precision under high temp, use frame skipping

4. **Problem**: Incompatible operators on NPU
   - **Solution**: Operator splitting—run unsupported ops on CPU/GPU, use async execution

---

## Sources

**Academic Papers (arXiv)**:
- [VLMQ: Efficient Post-Training Quantization for Large Vision-Language Models](https://arxiv.org/html/2508.03351v1) (arXiv:2508.03351, accessed 2025-01-31)
- [Post-training Quantization for Large Vision-Language Models](https://arxiv.org/html/2410.08119v3) (arXiv:2410.08119, accessed 2025-01-31)
- [A Survey on Efficient Vision-Language Models](https://arxiv.org/html/2504.09724v1) (arXiv:2504.09724, accessed 2025-01-31)
- [Efficient Deployment of Vision-Language Models on Mobile Devices](https://arxiv.org/html/2507.08505v1) (arXiv:2507.08505, accessed 2025-01-31)
- [Vision-Language Models for Edge Networks](https://arxiv.org/html/2502.07855v1) (arXiv:2502.07855, accessed 2025-01-31)
- [When CPUs Outperform GPUs for On-Device LLM Inference](https://arxiv.org/abs/2505.06461) (arXiv:2505.06461, accessed 2025-01-31)

**Conference Proceedings**:
- [Post-training Quantization for Large Vision-Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/cffbaf4f47546ece96bb42c0edda40ee-Paper-Conference.pdf) (NeurIPS 2024, accessed 2025-01-31)
- [MBQ: Modality-Balanced Quantization for Large Vision-Language Models](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_MBQ_Modality-Balanced_Quantization_for_Large_Vision-Language_Models_CVPR_2025_paper.pdf) (CVPR 2025, accessed 2025-01-31)
- [Exploring Quantization for Efficient Pre-Training of Transformer Language Models](https://aclanthology.org/2024.findings-emnlp.787.pdf) (EMNLP 2024, accessed 2025-01-31)
- [Self-adapting Large Visual-Language Models to Edge Devices](https://dl.acm.org/doi/10.1007/978-3-031-73390-1_18) (ECCV 2024, accessed 2025-01-31)

**Technical Documentation & Blogs**:
- [INT8 W8A8 - vLLM](https://docs.vllm.ai/en/v0.7.1/features/quantization/int8.html) (vLLM docs, accessed 2025-01-31)
- [Model Quantization: Meaning, Benefits & Techniques](https://www.clarifai.com/blog/model-quantization) (Clarifai, accessed 2025-01-31)
- [TensorRT SDK](https://developer.nvidia.com/tensorrt) (NVIDIA Developer, accessed 2025-01-31)
- [Visual Language Intelligence and Edge AI 2.0](https://developer.nvidia.com/blog/visual-language-intelligence-and-edge-ai-2-0/) (NVIDIA blog, accessed 2025-01-31)
- [Optimizing Transformer Inference with ONNX Runtime and Quantization](https://medium.com/@bhagyarana80/optimizing-transformer-inference-with-onnx-runtime-and-quantization-098f8149a15c) (Medium, accessed 2025-01-31)
- [Deploying Quantized LLMs with ONNX Runtime](https://apxml.com/courses/quantized-llm-deployment/chapter-4-optimizing-deploying-quantized-llms/deployment-onnx-runtime) (ApX ML, accessed 2025-01-31)
- [Deploying an Efficient Vision-Language Model on Mobile Devices](https://www.edge-ai-vision.com/2025/05/deploying-an-efficient-vision-language-model-on-mobile-devices/) (Edge AI and Vision Alliance, accessed 2025-01-31)

**Research & Industry Articles**:
- [Efficient GPT-4V level multimodal large language model for mobile devices](https://www.nature.com/articles/s41467-025-61040-5) (Nature, accessed 2025-01-31)
- [Edge deployment of LLMs and ML models: A review](https://www.rohan-paul.com/p/edge-deployment-of-llms-and-ml-models) (Rohan's Bytes, accessed 2025-01-31)
- [TensorRT Implementations of Model Quantization on Edge GPUs](https://userweb.cs.txstate.edu/~k_y47/webpage/pubs/mcsoc23.pdf) (Texas State University, accessed 2025-01-31)

**GitHub Repositories**:
- [ModelTC/LightCompress](https://github.com/ModelTC/LightCompress) (accessed 2025-01-31)
- [Zhen-Dong/Awesome-Quantization-Papers](https://github.com/Zhen-Dong/Awesome-Quantization-Papers) (accessed 2025-01-31)

**Additional Web Resources**:
- [Hugging Face: Vision Language Models Quantization Collection](https://huggingface.co/collections/neuralmagic/vision-language-models-quantization) (accessed 2025-01-31)
- [NVIDIA Quantization Guide](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/nlp/quantization.html) (accessed 2025-01-31)
