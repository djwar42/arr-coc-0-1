# Edge Deployment: TensorRT, CoreML, and Mobile Optimization

**Comprehensive guide to deploying optimized AI models on edge devices including NVIDIA Jetson, Apple iOS/macOS, Android, and embedded systems**

From [NVIDIA TensorRT-LLM for Jetson](https://forums.developer.nvidia.com/t/tensorrt-llm-for-jetson/313227) (accessed 2025-11-14):
> "TensorRT-LLM is a high-performance LLM inference library with advanced quantization, attention kernels, and paged KV caching."

From [Apple Machine Learning FastVLM Research](https://machinelearning.apple.com/research/fast-vision-language-models) (accessed 2025-11-14):
> "FastVLM delivers accurate, fast, and efficient visual query processing, making it suitable for powering real-time applications on-device."

---

## Overview

Edge AI deployment brings inference to the data source, enabling low-latency responses (<50ms), privacy-preserving processing, and reduced cloud costs. This guide covers platform-specific optimizations for TensorRT on Jetson, CoreML on Apple devices, ONNX Runtime on Android, and cross-platform deployment strategies.

**Key edge deployment requirements:**
- **Ultra-low latency**: <50ms for real-time applications (autonomous vehicles, robotics)
- **Power constraints**: 5-30W typical power budget for edge devices
- **Model compression**: 4-8× reduction via quantization (INT8, INT4, FP16)
- **Memory efficiency**: Limited RAM/VRAM (4-64GB typical)
- **Thermal management**: Sustained performance without throttling

**Performance targets:**
- Vision models: 30-60 FPS for real-time video processing
- VLMs: 10-30 tokens/second for on-device chat
- Object detection: <100ms inference for YOLO-class models
- Image classification: <10ms for efficient architectures

---

## Section 1: TensorRT on NVIDIA Jetson Edge

### Jetson Platform Overview

From [NVIDIA Jetson Download Center](https://developer.nvidia.com/embedded/downloads) (accessed 2025-11-14):

**Jetson Orin Family (2024-2025 edge platforms):**

| Model | GPU | CUDA Cores | Tensor Cores | Memory | TDP | Use Case |
|-------|-----|------------|--------------|--------|-----|----------|
| **Jetson Orin Nano** | 1024-core Ampere | 1024 | 32 | 4-8GB | 7-15W | Entry-level edge AI |
| **Jetson Orin NX** | 1024-core Ampere | 1024 | 32 | 8-16GB | 10-25W | Robotics, drones |
| **Jetson AGX Orin** | 2048-core Ampere | 2048 | 64 | 32-64GB | 15-60W | Autonomous systems |
| **Jetson AGX Orin Industrial** | 2048-core Ampere | 2048 | 64 | 64GB | 60W | Industrial automation |

**Key features:**
- **Ampere architecture**: Same GPU architecture as A100 datacenter GPUs
- **INT8 Tensor Cores**: Hardware acceleration for quantized models
- **NVDEC/NVENC**: Hardware video encode/decode (H.264, H.265, AV1)
- **Unified memory**: CPU and GPU share memory (zero-copy operations)
- **JetPack SDK**: Complete development environment with TensorRT, CUDA, cuDNN

### TensorRT-LLM on Jetson

From [Running LLMs with TensorRT-LLM on NVIDIA Jetson AGX Orin](https://www.hackster.io/shahizat/running-llms-with-tensorrt-llm-on-nvidia-jetson-agx-orin-34372f) (accessed 2025-11-14):

**Deploying Large Language Models on Jetson:**

TensorRT-LLM brings datacenter-class LLM optimizations to edge devices through:
- **Paged KV caching**: Efficient memory management for long contexts
- **FlashAttention**: Optimized attention kernels for Ampere
- **INT8/FP8 quantization**: 4× memory reduction with minimal accuracy loss
- **Multi-stream execution**: Concurrent request processing

**Example: Llama 3.2 3B on Jetson Orin Nano (8GB):**

```bash
# Install TensorRT-LLM (JetPack 6.0+)
sudo apt update
sudo apt install tensorrt-llm python3-pip

# Clone TensorRT-LLM repository
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM

# Convert Llama 3.2 3B to TensorRT engine
python examples/llama/convert_checkpoint.py \
    --model_dir /path/to/llama-3.2-3b \
    --output_dir ./llama-3.2-3b-trt \
    --dtype float16 \
    --use_weight_only \
    --weight_only_precision int8

# Build TensorRT engine
trtllm-build \
    --checkpoint_dir ./llama-3.2-3b-trt \
    --output_dir ./llama-3.2-3b-engine \
    --gemm_plugin float16 \
    --max_batch_size 4 \
    --max_input_len 2048 \
    --max_output_len 512

# Run inference
python run.py \
    --engine_dir ./llama-3.2-3b-engine \
    --tokenizer_dir /path/to/llama-3.2-3b \
    --max_output_len 128
```

**Performance metrics (Jetson AGX Orin 64GB):**
- **Llama 3.2 1B INT8**: 45-50 tokens/second
- **Llama 3.2 3B INT8**: 25-30 tokens/second
- **Llama 3.2 11B VLM FP16**: 8-12 tokens/second (vision + text)
- **First token latency**: 100-300ms depending on context length

### Vision Model Deployment on Jetson

From [YOLO11 Jetson Orin Nano: Super Fast Edge AI](https://www.ultralytics.com/blog/ultralytics-yolo11-on-nvidia-jetson-orin-nano-super-fast-and-efficient) (accessed 2025-11-14):

**Deploying YOLO11 with TensorRT:**

```python
import tensorrt as trt
from ultralytics import YOLO

# Export YOLO11 to ONNX
model = YOLO('yolo11n.pt')  # Nano model
model.export(format='onnx', simplify=True)

# Build TensorRT engine
def build_engine(onnx_file, engine_file):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    # Enable FP16 for 2× speedup
    config.set_flag(trt.BuilderFlag.FP16)

    # Set max workspace size (4GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_file, 'rb') as f:
        parser.parse(f.read())

    # Build and save engine
    engine = builder.build_serialized_network(network, config)
    with open(engine_file, 'wb') as f:
        f.write(engine)

build_engine('yolo11n.onnx', 'yolo11n_fp16.engine')

# Run inference with TensorRT
model = YOLO('yolo11n_fp16.engine')
results = model('image.jpg')
```

**YOLO11 performance on Jetson Orin Nano (FP16):**
- **YOLOv11n**: 60-80 FPS @ 640×640 resolution
- **YOLOv11s**: 40-50 FPS @ 640×640
- **YOLOv11m**: 25-30 FPS @ 640×640
- **Power consumption**: 7-12W depending on model size

**Optimization techniques:**
1. **Dynamic shapes**: Pre-allocate multiple engine sizes (640, 1280, 1920)
2. **Batch processing**: Process 4-8 frames simultaneously for throughput
3. **Stream multiplexing**: Use CUDA streams for concurrent video inputs
4. **Zero-copy**: Leverage unified memory for direct camera access

### INT8 Quantization for Edge

**PTQ (Post-Training Quantization) workflow:**

```python
import tensorrt as trt

# Calibration dataset
def load_calibration_data():
    # Load representative images
    import glob
    images = []
    for img_path in glob.glob('calibration_data/*.jpg')[:1000]:
        img = preprocess_image(img_path)
        images.append(img)
    return images

# Create INT8 calibrator
class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data, cache_file='calibration.cache'):
        super().__init__()
        self.data = calibration_data
        self.cache_file = cache_file
        self.batch_size = 1
        self.current_index = 0

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index < len(self.data):
            batch = self.data[self.current_index]
            self.current_index += 1
            return [batch]
        return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)

# Build INT8 engine
calibration_data = load_calibration_data()
calibrator = EntropyCalibrator(calibration_data)

config.set_flag(trt.BuilderFlag.INT8)
config.int8_calibrator = calibrator

# Expect 3-4× speedup over FP16 with <1% accuracy loss
```

**INT8 performance gains (Jetson AGX Orin):**
- **ResNet-50**: 450 FPS (INT8) vs 180 FPS (FP16) = 2.5× speedup
- **EfficientNet-B0**: 380 FPS (INT8) vs 150 FPS (FP16) = 2.5× speedup
- **MobileNetV3**: 820 FPS (INT8) vs 350 FPS (FP16) = 2.3× speedup
- **Power reduction**: 15-20% lower power draw at same FPS

---

## Section 2: Apple CoreML Deployment

### Apple Silicon for ML

From [Apple M4 specifications](https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/) and [M4 Pro/Max announcement](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/) (accessed 2025-11-14):

**Apple Silicon Neural Engine (2024-2025):**

| Chip | Neural Engine | TOPS | GPU Cores | Unified Memory | Use Case |
|------|---------------|------|-----------|----------------|----------|
| **M4** | 16-core | 38 TOPS | 10 | 16-32GB | MacBook Air, iPad Pro |
| **M4 Pro** | 16-core | 38 TOPS | 20 | 24-64GB | MacBook Pro 14" |
| **M4 Max** | 16-core | 38 TOPS | 40 | 36-128GB | MacBook Pro 16", Mac Studio |
| **A18 Pro** (iPhone 16 Pro) | 16-core | 35 TOPS | 6 | 8GB | iPhone flagship |

**Key advantages:**
- **60× Neural Engine improvement** since M1 (11 TOPS → 38 TOPS)
- **Unified memory architecture**: Zero-copy between CPU, GPU, Neural Engine
- **Power efficiency**: <5W Neural Engine operation vs 15-30W GPU
- **On-device processing**: Complete privacy, no cloud dependency

### FastVLM: Apple's On-Device Vision-Language Model

From [Apple Machine Learning FastVLM Research](https://machinelearning.apple.com/research/fast-vision-language-models) (accessed 2025-11-14):

Apple's FastVLM represents breakthrough on-device multimodal AI:

**FastVLM architecture innovations:**
1. **Token-level early exit**: Dynamically skip layers based on confidence
2. **Adaptive visual token reduction**: Compress visual embeddings 3-4×
3. **Efficient cross-attention**: Reduce vision-text fusion cost by 2×
4. **Neural Engine optimization**: Maximize INT8 operations on NPU

**FastVLM performance metrics:**
- **85× faster** than baseline VLM implementations
- **3.4× smaller** model size through pruning and quantization
- **<100ms latency** for image understanding queries
- **Runs on iPhone 15 Pro+**: Full on-device execution

**FastVLM deployment (CoreML conversion):**

```python
import coremltools as ct
from transformers import AutoModel, AutoProcessor

# Load FastVLM model (hypothetical - Apple hasn't released weights)
model = AutoModel.from_pretrained('apple/fastvlm-base')
processor = AutoProcessor.from_pretrained('apple/fastvlm-base')

# Trace model with example inputs
image_input = torch.randn(1, 3, 384, 384)
text_input = processor.tokenizer("Describe this image", return_tensors='pt')

traced_model = torch.jit.trace(
    model,
    (image_input, text_input['input_ids'])
)

# Convert to CoreML with Neural Engine optimization
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.ImageType(name='image', shape=(1, 3, 384, 384)),
        ct.TensorType(name='input_ids', shape=(1, 77))
    ],
    compute_units=ct.ComputeUnit.ALL,  # CPU, GPU, Neural Engine
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS17
)

# Optimize for Neural Engine
mlmodel = ct.optimize.coreml.OpQuantizer(
    mlmodel,
    nbits=8,  # INT8 quantization
    mode='linear'
).quantize()

mlmodel.save('FastVLM.mlpackage')
```

### CoreML Model Optimization

From [Deploy machine learning and AI models on-device with Core ML](https://developer.apple.com/la/videos/play/wwdc2024/10161/) (WWDC 2024, accessed 2025-11-14):

**CoreML optimization workflow:**

**1. Quantization for Neural Engine**

```python
import coremltools as ct

# Load existing CoreML model
model = ct.models.MLModel('model.mlpackage')

# Quantize to INT8 for Neural Engine
quantized_model = ct.optimize.coreml.palettize_weights(
    model,
    nbits=8,
    mode='linear',  # or 'kmeans' for better accuracy
    granularity='per_channel'
)

# Further optimize for specific device
from coremltools.optimize.coreml import OptimizationConfig

config = OptimizationConfig(
    activation_type='float16',
    weight_type='int8'
)

optimized_model = ct.optimize.coreml.linear_quantize_weights(
    model,
    config=config
)

optimized_model.save('model_int8.mlpackage')
```

**2. Flexible Shapes for Dynamic Input**

```python
# Define flexible input shapes
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.ImageType(
            name='image',
            shape=ct.EnumeratedShapes(
                shapes=[
                    (1, 3, 224, 224),
                    (1, 3, 384, 384),
                    (1, 3, 512, 512)
                ]
            )
        )
    ],
    compute_units=ct.ComputeUnit.ALL
)
```

**3. Multi-Function Models**

```python
# Package multiple operations in single model
@torch.jit.script
class MultiTaskModel(torch.nn.Module):
    def __init__(self, encoder, detector, classifier):
        super().__init__()
        self.encoder = encoder
        self.detector = detector
        self.classifier = classifier

    def encode(self, x):
        return self.encoder(x)

    def detect(self, x):
        features = self.encoder(x)
        return self.detector(features)

    def classify(self, x):
        features = self.encoder(x)
        return self.classifier(features)

model = MultiTaskModel(encoder, detector, classifier)

# Convert with multiple functions
mlmodel = ct.convert(
    model,
    inputs=[ct.ImageType(shape=(1, 3, 224, 224))],
    outputs=[
        ct.TensorType(name='encoded_features'),
        ct.TensorType(name='detections'),
        ct.TensorType(name='class_probabilities')
    ],
    compute_units=ct.ComputeUnit.ALL
)
```

### iOS Deployment Best Practices

**Swift integration for real-time inference:**

```swift
import CoreML
import Vision

class VisionProcessor {
    private var model: VNCoreMLModel?

    init() {
        // Load CoreML model
        guard let modelURL = Bundle.main.url(forResource: "YOLOv8", withExtension: "mlmodelc") else {
            fatalError("Model not found")
        }

        do {
            let mlModel = try MLModel(contentsOf: modelURL)
            self.model = try VNCoreMLModel(for: mlModel)
        } catch {
            print("Failed to load model: \(error)")
        }
    }

    func processImage(_ image: CVPixelBuffer, completion: @escaping ([VNRecognizedObjectObservation]) -> Void) {
        guard let model = model else { return }

        let request = VNCoreMLRequest(model: model) { request, error in
            guard let results = request.results as? [VNRecognizedObjectObservation] else {
                return
            }
            completion(results)
        }

        // Configure for low latency
        request.imageCropAndScaleOption = .scaleFit
        request.usesCPUOnly = false  // Use Neural Engine

        let handler = VNImageRequestHandler(cvPixelBuffer: image, options: [:])

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
            } catch {
                print("Inference error: \(error)")
            }
        }
    }
}
```

**Performance optimization:**
- **Batch size 1**: Neural Engine optimized for single-image inference
- **Pre-warmed model**: Load model at app startup, not on-demand
- **Buffer reuse**: Recycle CVPixelBuffers to reduce allocations
- **Background threading**: Run inference on dedicated queue

**Measured performance (iPhone 16 Pro, A18 Pro):**
- **MobileNetV3**: 2-3ms latency on Neural Engine
- **EfficientNet-B0**: 5-7ms latency
- **YOLOv8n**: 12-15ms latency (object detection)
- **Vision Transformer**: 20-25ms latency (image classification)

---

## Section 3: Android ONNX Runtime Mobile

### ONNX Runtime Mobile Architecture

From [ONNX Runtime Mobile documentation](https://onnxruntime.ai/docs/get-started/with-mobile.html) (accessed 2025-11-14):

**ORT Mobile optimizations:**
- **Reduced binary size**: 2-5 MB (vs 20-30 MB desktop)
- **Hardware acceleration**: NNAPI, GPU (OpenCL/Vulkan), CPU (XNNPACK)
- **Quantization support**: INT8, UINT8, float16
- **Graph optimizations**: Layer fusion, constant folding, dead code elimination

**Supported Android hardware:**
- **Qualcomm Snapdragon**: Hexagon DSP, Adreno GPU, Kryo CPU
- **Samsung Exynos**: Mali GPU, ARM CPU
- **Google Tensor**: Mali GPU, TPU (limited support)
- **MediaTek Dimensity**: Mali GPU, APU (AI Processing Unit)

### ONNX Model Deployment on Android

**1. Model Conversion and Optimization**

```python
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Export PyTorch to ONNX
model = YourModel()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    export_params=True,
    opset_version=14,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# Quantize to INT8 for mobile
quantize_dynamic(
    'model.onnx',
    'model_int8.onnx',
    weight_type=QuantType.QUInt8
)

# Optimize graph
from onnxruntime.transformers.optimizer import optimize_model

optimized_model = optimize_model(
    'model_int8.onnx',
    model_type='bert',  # or 'gpt2', 'vit', etc.
    num_heads=12,
    hidden_size=768,
    optimization_options=None
)

optimized_model.save_model_to_file('model_optimized.onnx')
```

**2. Android Integration (Kotlin)**

```kotlin
import ai.onnxruntime.*

class ModelInference(context: Context) {
    private val ortEnv = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        // Load model from assets
        val modelBytes = context.assets.open("model_optimized.onnx").readBytes()

        // Create session with NNAPI execution provider
        val sessionOptions = OrtSession.SessionOptions().apply {
            addNnapi()  // Use Android Neural Networks API
            setIntraOpNumThreads(4)
            setGraphOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }

        session = ortEnv.createSession(modelBytes, sessionOptions)
    }

    fun runInference(inputData: FloatArray, shape: LongArray): FloatArray {
        // Create input tensor
        val inputTensor = OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(inputData),
            shape
        )

        // Run inference
        val results = session.run(mapOf("input" to inputTensor))

        // Extract output
        val outputTensor = results[0].value as FloatArray

        results.close()
        inputTensor.close()

        return outputTensor
    }

    fun close() {
        session.close()
        ortEnv.close()
    }
}
```

**3. GPU Acceleration (OpenCL/Vulkan)**

```kotlin
// Enable GPU execution provider
val sessionOptions = OrtSession.SessionOptions().apply {
    // Try GPU first, fallback to CPU
    addConfigEntry("providers", "['OpenCLExecutionProvider', 'CPUExecutionProvider']")
    setGraphOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

    // Set GPU tuning parameters
    addConfigEntry("tunable_op_enabled", "1")
    addConfigEntry("tunable_op_tuning_enable", "1")
}
```

### Android-Specific Optimizations

From [Cross-Platform Optimization of ONNX Models for Mobile and Edge Deployment](https://www.researchgate.net/publication/392623112_Cross-Platform_Optimization_of_ONNX_Models_for_Mobile_and_Edge_Deployment) (accessed 2025-11-14):

**Graph optimization levels:**

| Level | Optimizations | Impact |
|-------|---------------|--------|
| **Basic** | Constant folding, redundant node elimination | 5-10% speedup |
| **Extended** | Layer fusion (Conv+BN+ReLU), transpose optimization | 20-30% speedup |
| **Layout** | NHWC↔NCHW conversion, memory layout optimization | 10-15% speedup (GPU) |

**NNAPI delegate configuration:**

```kotlin
class NNAPIConfig {
    companion object {
        fun getOptimalConfig(device: String): OrtSession.SessionOptions {
            val options = OrtSession.SessionOptions()

            when {
                device.contains("Snapdragon") -> {
                    // Qualcomm: Prefer Hexagon DSP
                    options.addConfigEntry("nnapi_accelerator_name", "hexagon-dsp")
                    options.addConfigEntry("nnapi_execution_preference", "sustained_speed")
                }
                device.contains("Exynos") -> {
                    // Samsung: Use GPU (Mali)
                    options.addConfigEntry("nnapi_accelerator_name", "mali-gpu")
                    options.addConfigEntry("nnapi_execution_preference", "low_power")
                }
                device.contains("Tensor") -> {
                    // Google: Use default (TPU when available)
                    options.addNnapi()
                    options.addConfigEntry("nnapi_execution_preference", "fast_single_answer")
                }
                else -> {
                    // Generic: CPU fallback
                    options.setIntraOpNumThreads(Runtime.getRuntime().availableProcessors())
                }
            }

            return options
        }
    }
}
```

**Performance benchmarks (Snapdragon 8 Gen 3):**
- **MobileNetV2 INT8 (NNAPI)**: 3-5ms latency
- **EfficientNet-B0 INT8 (NNAPI)**: 8-12ms latency
- **YOLO-NAS-S INT8 (GPU)**: 25-30ms latency
- **BERT-Base INT8 (Hexagon DSP)**: 45-60ms latency

---

## Section 4: Cross-Platform Deployment Strategies

### Model Format Comparison

| Format | Platforms | Hardware Acceleration | Quantization | Graph Optimization |
|--------|-----------|----------------------|--------------|-------------------|
| **TensorRT** | Linux (Jetson, x86) | NVIDIA GPUs | INT8, INT4, FP16, FP8 | Excellent |
| **CoreML** | iOS, macOS | Neural Engine, GPU, CPU | INT8, float16 | Good |
| **ONNX Runtime** | Android, iOS, Windows, Linux | NNAPI, GPU, NPU, CPU | INT8, UINT8, float16 | Excellent |
| **TFLite** | Android, iOS, embedded | NNAPI, GPU, EdgeTPU | INT8, float16 | Good |
| **OpenVINO** | x86, ARM (limited) | Intel CPUs, iGPU, VPU | INT8, FP16 | Excellent |

### Universal Deployment Workflow

**1. Training → ONNX Export → Platform-Specific Conversion**

```python
# Step 1: Train in PyTorch
model = train_model()

# Step 2: Export to ONNX (intermediate format)
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    opset_version=14,
    do_constant_folding=True
)

# Step 3: Optimize ONNX graph
from onnxruntime.transformers.optimizer import optimize_model

optimized = optimize_model(
    'model.onnx',
    model_type='bert',
    optimization_options={
        'enable_gelu_approximation': True,
        'enable_attention_fusion': True,
        'enable_skip_layer_norm_fusion': True
    }
)

optimized.save_model_to_file('model_optimized.onnx')

# Step 4a: Convert to TensorRT for Jetson
import tensorrt as trt
# ... TensorRT build process (see Section 1)

# Step 4b: Convert to CoreML for iOS
import coremltools as ct
mlmodel = ct.convert('model_optimized.onnx')
mlmodel.save('model.mlpackage')

# Step 4c: Use ONNX Runtime directly on Android
# No conversion needed - deploy model_optimized.onnx
```

### Latency Optimization Techniques

From [Edge vs Cloud AI: Key Differences, Benefits & Hybrid Future](https://www.clarifai.com/blog/edge-vs-cloud-ai) and [ML-driven latency optimization for mobile edge computing](https://www.sciencedirect.com/science/article/pii/S2215016125004388) (accessed 2025-11-14):

**Achieving <50ms latency:**

**1. Model Architecture Selection**
- **MobileNetV3**: 5-10ms (224×224 image classification)
- **EfficientNet-Lite**: 10-15ms (image classification)
- **YOLOv8n**: 15-25ms (object detection)
- **FastViT**: 8-12ms (vision transformer)

**2. Resolution/Quality Tradeoffs**
```python
# Dynamic resolution based on latency budget
class AdaptiveResolution:
    def __init__(self, model, target_latency_ms=50):
        self.model = model
        self.target_latency = target_latency_ms
        self.resolution_options = [224, 320, 416, 512]

    def infer_adaptive(self, image):
        for resolution in self.resolution_options:
            resized = resize_image(image, resolution)

            start = time.time()
            result = self.model(resized)
            latency = (time.time() - start) * 1000

            if latency <= self.target_latency:
                return result, resolution, latency

        # If all resolutions too slow, use smallest
        return self.model(resize_image(image, 224)), 224, latency
```

**3. Batch Processing for Throughput**
```python
# Process video frames in batches
def batch_inference(frames, model, batch_size=4):
    results = []

    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        batch_tensor = torch.stack([preprocess(f) for f in batch])

        with torch.no_grad():
            outputs = model(batch_tensor)

        results.extend(outputs)

    return results

# Throughput: 4× batch = 2.5-3× speedup vs single-frame
```

**4. Hardware-Specific Tuning**
- **Tensor Core utilization**: Pad dimensions to multiples of 8 (FP16) or 16 (INT8)
- **Memory alignment**: Ensure tensors aligned to 128-byte boundaries
- **Kernel fusion**: Combine operations to reduce memory bandwidth

### Edge AI System Architecture

From [Edge AI Infrastructure: Deploying GPUs Closer to Data Sources](https://introl.com/blog/edge-ai-infrastructure-deploying-gpus-data-sources) (accessed 2025-11-14):

**Hybrid edge-cloud architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Cloud Layer (Training)                    │
│  - Model training (H100, A100)                              │
│  - Hyperparameter tuning                                    │
│  - Data aggregation                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓ ↑
                    Model Deployment / Telemetry
                            ↓ ↑
┌─────────────────────────────────────────────────────────────┐
│                   Edge Gateway (Coordination)                │
│  - Model distribution                                       │
│  - Load balancing                                           │
│  - Aggregated inference (Jetson AGX Orin)                  │
└─────────────────────────────────────────────────────────────┘
                            ↓ ↑
                    Inference Requests / Results
                            ↓ ↑
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Edge Device │  │  Edge Device │  │  Edge Device │
│  (Jetson     │  │  (iPhone     │  │  (Android    │
│   Orin Nano) │  │   16 Pro)    │  │   Phone)     │
│              │  │              │  │              │
│  - Real-time │  │  - On-device │  │  - Local     │
│    inference │  │    privacy   │  │    inference │
│  - <50ms     │  │  - <100ms    │  │  - <100ms    │
│    latency   │  │    latency   │  │    latency   │
└──────────────┘  └──────────────┘  └──────────────┘
```

**Benefits of edge deployment:**
- **95% lower latency**: 2-50ms edge vs 100-500ms cloud (from edge AI infrastructure study)
- **82% bandwidth savings**: Process locally, send only metadata/alerts to cloud
- **Privacy preservation**: Sensitive data never leaves device
- **Offline operation**: No network dependency for critical functions

---

## Section 5: ARR-COC Edge Deployment

### ARR-COC Mobile VLM Feasibility

**Challenges for deploying ARR-COC on edge:**

1. **13-channel texture array**: 13× memory vs RGB
   - RGB image (224×224×3): 602KB
   - ARR-COC texture (224×224×13): 2.6MB
   - **Solution**: Compute textures on-the-fly, stream to model

2. **Three relevance scorers**: 3× model evaluations
   - Propositional (entropy): Lightweight, <1ms
   - Perspectival (Jungian): ResNet-based, 5-10ms
   - Participatory (cross-attention): Query-dependent, 10-15ms
   - **Total overhead**: 15-25ms additional latency

3. **Variable LOD token allocation**: Dynamic processing
   - High-relevance patches: 400 tokens (expensive)
   - Low-relevance patches: 64 tokens (cheap)
   - **Optimization**: Pre-build engines for each LOD level

### Mobile ARR-COC Architecture

**Lightweight ARR-COC for edge deployment:**

```python
class EdgeARRCOC:
    def __init__(self, platform='jetson'):
        # Texture extraction (optimized for edge)
        if platform == 'jetson':
            self.texture_extractor = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(
                open('texture_fp16.engine', 'rb').read()
            )
        elif platform == 'ios':
            self.texture_extractor = ct.models.MLModel('TextureExtractor.mlpackage')
        elif platform == 'android':
            self.texture_extractor = ort.InferenceSession(
                'texture_int8.onnx',
                providers=['NNAPIExecutionProvider', 'CPUExecutionProvider']
            )

        # Simplified relevance scoring (single unified model)
        self.relevance_scorer = load_edge_scorer(platform)

        # Qwen3-VL quantized for edge
        self.vlm = load_quantized_vlm(platform, bits=4)  # INT4 for memory

    def process_image(self, image, query, latency_budget_ms=100):
        start = time.time()

        # 1. Extract texture array (10-15ms)
        textures = self.texture_extractor(image)

        # 2. Fast relevance scoring (5-10ms)
        relevance_map = self.relevance_scorer(textures, query)

        # 3. Adaptive token allocation
        elapsed = (time.time() - start) * 1000
        remaining_budget = latency_budget_ms - elapsed

        # Allocate tokens based on remaining latency budget
        token_budgets = self.adaptive_allocation(
            relevance_map,
            remaining_budget_ms=remaining_budget
        )

        # 4. VLM inference with variable LOD (60-80ms)
        result = self.vlm.generate(
            textures,
            query,
            token_budgets=token_budgets
        )

        total_latency = (time.time() - start) * 1000
        return result, total_latency

    def adaptive_allocation(self, relevance_map, remaining_budget_ms):
        # Allocate more tokens if latency budget allows
        if remaining_budget_ms > 80:
            # Generous allocation
            return {
                'high': 400,
                'medium': 200,
                'low': 64
            }
        elif remaining_budget_ms > 50:
            # Balanced allocation
            return {
                'high': 200,
                'medium': 100,
                'low': 32
            }
        else:
            # Aggressive compression
            return {
                'high': 100,
                'medium': 50,
                'low': 16
            }
```

### Platform-Specific ARR-COC Deployment

**1. Jetson AGX Orin (64GB) - Full ARR-COC**

```python
# Deploy full ARR-COC with all three scorers
# Hardware: 2048 CUDA cores, 64 Tensor Cores, 64GB unified memory

class JetsonARRCOC:
    def __init__(self):
        # TensorRT engines for all components
        self.texture_engine = load_trt_engine('texture_fp16.engine')
        self.propositional_engine = load_trt_engine('entropy_int8.engine')
        self.perspectival_engine = load_trt_engine('jungian_fp16.engine')
        self.participatory_engine = load_trt_engine('cross_attn_fp16.engine')
        self.qwen3_vlm_engine = load_trt_engine('qwen3_vl_11b_int4.engine')

    def process(self, image, query):
        # Concurrent execution using CUDA streams
        stream1 = cuda.Stream()
        stream2 = cuda.Stream()
        stream3 = cuda.Stream()

        # Extract textures
        textures = self.texture_engine.infer(image)

        # Parallel relevance scoring
        with stream1:
            prop_score = self.propositional_engine.infer(textures)
        with stream2:
            persp_score = self.perspectival_engine.infer(textures)
        with stream3:
            part_score = self.participatory_engine.infer(textures, query)

        # Synchronize streams
        cuda.synchronize()

        # Combine scores
        relevance = (prop_score + persp_score + part_score) / 3

        # VLM inference with dynamic LOD
        result = self.qwen3_vlm_engine.infer(textures, query, relevance)

        return result

# Expected performance: 80-120ms total latency
```

**2. iPhone 16 Pro (A18 Pro) - Simplified ARR-COC**

```swift
// Simplified ARR-COC for iOS using CoreML
class iOSARRCOC {
    let textureExtractor: VNCoreMLModel
    let relevanceScorer: VNCoreMLModel  // Unified scorer
    let compactVLM: VNCoreMLModel  // Qwen3-VL 3B INT8

    init() {
        // Load CoreML models
        self.textureExtractor = try! VNCoreMLModel(for: TextureExtractor().model)
        self.relevanceScorer = try! VNCoreMLModel(for: UnifiedRelevanceScorer().model)
        self.compactVLM = try! VNCoreMLModel(for: CompactVLM().model)
    }

    func process(image: CVPixelBuffer, query: String) async -> String {
        // 1. Texture extraction on Neural Engine (8-12ms)
        let textures = await runInference(textureExtractor, input: image)

        // 2. Relevance scoring on Neural Engine (5-8ms)
        let relevance = await runInference(relevanceScorer, inputs: [textures, query])

        // 3. Compact VLM on GPU (60-80ms for 3B model)
        let result = await runInference(compactVLM, inputs: [textures, query, relevance])

        return result
    }
}

// Expected performance: 70-100ms total latency
// Power consumption: 3-5W (sustainable for long sessions)
```

**3. Android (Snapdragon 8 Gen 3) - Cloud-Offload Hybrid**

```kotlin
// Hybrid edge-cloud ARR-COC for Android
class AndroidARRCOC(context: Context) {
    private val edgeTexture = loadONNXModel(context, "texture_int8.onnx", NNAPI)
    private val edgeRelevance = loadONNXModel(context, "relevance_int8.onnx", NNAPI)
    private val cloudVLM = CloudVLMClient()  // Fallback to cloud API

    suspend fun process(image: Bitmap, query: String, mode: ProcessingMode): String {
        // Always compute textures and relevance locally (privacy)
        val textures = edgeTexture.infer(image)
        val relevance = edgeRelevance.infer(textures, query)

        return when (mode) {
            ProcessingMode.EDGE_ONLY -> {
                // Run lightweight VLM on-device (Qwen3 1B)
                val localVLM = loadONNXModel(context, "qwen3_1b_int4.onnx", GPU)
                localVLM.infer(textures, query, relevance)
            }
            ProcessingMode.CLOUD_OFFLOAD -> {
                // Send compressed features to cloud (not raw image)
                val compressed = compressFeatures(textures, relevance)
                cloudVLM.infer(compressed, query)
            }
            ProcessingMode.ADAPTIVE -> {
                // Check network latency and battery level
                if (isLowLatencyNetwork() && batteryLevel > 20%) {
                    cloudVLM.infer(compressFeatures(textures, relevance), query)
                } else {
                    val localVLM = loadONNXModel(context, "qwen3_1b_int4.onnx", GPU)
                    localVLM.infer(textures, query, relevance)
                }
            }
        }
    }
}

// Expected performance:
// - Edge-only: 120-150ms (limited by 1B model quality)
// - Cloud-offload: 200-300ms (network latency + datacenter inference)
// - Adaptive: Best of both based on conditions
```

### Power Consumption Analysis

**Edge device power budgets:**

| Device | Idle Power | Inference Power | Sustained Power | Thermal Limit |
|--------|------------|-----------------|-----------------|---------------|
| **Jetson Orin Nano** | 2W | 7-12W | 15W | 25W (throttles) |
| **Jetson AGX Orin** | 5W | 20-40W | 60W | 75W (throttles) |
| **iPhone 16 Pro** | 0.5W | 3-5W | 8W | 12W (throttles) |
| **Snapdragon 8 Gen 3** | 1W | 4-8W | 12W | 15W (throttles) |

**ARR-COC power optimization strategies:**

1. **Adaptive frame rate**: Process every 2nd or 3rd frame when battery low
2. **Resolution scaling**: Lower resolution when power constrained
3. **Thermal throttling awareness**: Monitor device temperature, reduce load
4. **Batch processing**: Process multiple frames in burst, then idle (more efficient)

---

## Deployment Checklist

### Model Preparation
- [ ] **Export to ONNX**: Intermediate format for platform flexibility
- [ ] **Quantize model**: INT8 for edge, INT4 for extreme compression
- [ ] **Optimize graph**: Layer fusion, constant folding, dead code elimination
- [ ] **Test accuracy**: Ensure <1-2% degradation from quantization
- [ ] **Measure baseline performance**: Latency and throughput on target hardware

### Platform-Specific Conversion
- [ ] **TensorRT (Jetson)**: Build engine with FP16/INT8, optimize profiles
- [ ] **CoreML (iOS)**: Convert with Neural Engine support, quantize weights
- [ ] **ONNX Runtime (Android)**: Configure NNAPI/GPU delegates
- [ ] **Dynamic shapes**: Support variable input sizes if needed
- [ ] **Batch processing**: Configure optimal batch size for throughput

### Deployment Testing
- [ ] **Latency profiling**: Measure P50, P95, P99 latencies
- [ ] **Power consumption**: Monitor power draw under sustained load
- [ ] **Thermal behavior**: Test for throttling under extended operation
- [ ] **Memory usage**: Check peak RAM/VRAM consumption
- [ ] **Error handling**: Fallback strategies for out-of-memory, timeout

### Production Hardening
- [ ] **Model versioning**: Track model versions for updates
- [ ] **A/B testing**: Compare new model against baseline in production
- [ ] **Monitoring**: Log inference latencies, error rates, power consumption
- [ ] **Over-the-air updates**: Infrastructure for deploying model updates
- [ ] **Graceful degradation**: Fallback to simpler model if resources constrained

---

## Sources

**Web Research (accessed 2025-11-14):**

1. [NVIDIA TensorRT-LLM for Jetson](https://forums.developer.nvidia.com/t/tensorrt-llm-for-jetson/313227) - TensorRT-LLM on Jetson discussion
2. [Apple Machine Learning FastVLM Research](https://machinelearning.apple.com/research/fast-vision-language-models) - FastVLM on-device VLM
3. [Apple M4 chip announcement](https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/) - M4 Neural Engine specifications
4. [Apple M4 Pro/Max announcement](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/) - M4 Pro/Max specifications
5. [NVIDIA Jetson Download Center](https://developer.nvidia.com/embedded/downloads) - Jetson platform specifications
6. [Running LLMs with TensorRT-LLM on NVIDIA Jetson AGX Orin](https://www.hackster.io/shahizat/running-llms-with-tensorrt-llm-on-nvidia-jetson-agx-orin-34372f) - Jetson LLM deployment guide
7. [YOLO11 Jetson Orin Nano: Super Fast Edge AI](https://www.ultralytics.com/blog/ultralytics-yolo11-on-nvidia-jetson-orin-nano-super-fast-and-efficient) - YOLO deployment on Jetson
8. [Deploying Accelerated Llama 3.2 from the Edge to the Cloud](https://www.edge-ai-vision.com/2024/10/deploying-accelerated-llama-3-2-from-the-edge-to-the-cloud/) - Edge AI deployment strategies
9. [Deploy machine learning and AI models on-device with Core ML](https://developer.apple.com/la/videos/play/wwdc2024/10161/) - WWDC 2024 CoreML session
10. [ONNX Runtime Blogs](https://onnxruntime.ai/blogs) - ONNX Runtime announcements and optimizations
11. [ONNX Runtime Mobile documentation](https://onnxruntime.ai/docs/get-started/with-mobile.html) - Mobile deployment guide
12. [Cross-Platform Optimization of ONNX Models for Mobile and Edge Deployment](https://www.researchgate.net/publication/392623112_Cross-Platform_Optimization_of_ONNX_Models_for_Mobile_and_Edge_Deployment) - Academic research on ONNX optimization
13. [Edge vs Cloud AI: Key Differences, Benefits & Hybrid Future](https://www.clarifai.com/blog/edge-vs-cloud-ai) - Edge AI architecture overview
14. [ML-driven latency optimization for mobile edge computing](https://www.sciencedirect.com/science/article/pii/S2215016125004388) - Latency optimization research
15. [Edge AI Infrastructure: Deploying GPUs Closer to Data Sources](https://introl.com/blog/edge-ai-infrastructure-deploying-gpus-data-sources) - Edge AI infrastructure guide

**Related Knowledge:**
- [karpathy/inference-optimization/00-tensorrt-fundamentals.md](../inference-optimization/00-tensorrt-fundamentals.md) - TensorRT core concepts
- [karpathy/inference-optimization/01-tensorrt-vlm-deployment.md](../inference-optimization/01-tensorrt-vlm-deployment.md) - VLM-specific TensorRT optimizations
- [karpathy/alternative-hardware/01-apple-metal-ml.md](../alternative-hardware/01-apple-metal-ml.md) - Apple Silicon ML capabilities
- [karpathy/alternative-hardware/02-intel-oneapi-ml.md](../alternative-hardware/02-intel-oneapi-ml.md) - Intel oneAPI for edge devices
