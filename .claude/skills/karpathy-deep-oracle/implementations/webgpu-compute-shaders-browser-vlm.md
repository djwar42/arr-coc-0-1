# WebGPU Compute Shaders for Browser-Based VLM Inference

## Overview

WebGPU enables high-performance GPU compute directly in web browsers, making it possible to run vision-language models (VLMs) client-side without server infrastructure. By leveraging compute shaders and texture sampling capabilities, developers can deploy privacy-preserving, zero-latency VLM inference that scales to billions of users with minimal operational costs.

**Key advantages of browser-based VLM inference:**
- **Privacy by default**: User data never leaves the device
- **Zero server costs**: Computation happens on client GPUs
- **Universal deployment**: Single codebase runs across Windows, macOS, Linux, Android
- **Instant availability**: No installation, updates via web standards

From [WebGPU Inference: LLMs That Run in Your Browser](https://medium.com/@bhagyarana80/webgpu-inference-llms-that-run-in-your-browser-6251d27a0565) (accessed 2025-01-31):
> "With WebGPU, you can ship a local LLM that runs inside a tab — no servers, no API keys, and, let's be real, no data leaving the device."

## Section 1: WebGPU Compute Pipeline Architecture

### Compute Shader Fundamentals

WebGPU compute shaders execute parallel workloads on GPU compute units, organized into workgroups that share local memory. Unlike graphics pipelines, compute shaders have no fixed function stages—they're pure general-purpose GPU programming.

**Compute pipeline creation:**
```javascript
const computePipeline = device.createComputePipeline({
  layout: 'auto',
  compute: {
    module: device.createShaderModule({
      code: wgslShaderCode
    }),
    entryPoint: 'main'
  }
});
```

**WGSL shader structure (Web GPU Shading Language):**
```wgsl
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var inputTexture: texture_2d<f32>;
@group(0) @binding(3) var textureSampler: sampler;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= arrayLength(&input)) { return; }

  // Compute operations here
  output[idx] = input[idx] * 2.0;
}
```

**Key compute pipeline components:**
- **Storage buffers**: Read/write GPU memory for tensor data
- **Texture samplers**: Hardware-accelerated interpolation for vision features
- **Workgroup size**: Threads per workgroup (typically 64-256)
- **Bind groups**: Resource binding for shader access

From [WebAssembly and WebGPU enhancements for faster Web AI](https://developer.chrome.com/blog/io24-webassembly-webgpu-1) (Chrome Developers, accessed 2025-01-31):
> "ML workloads push tensors through a graph of computational nodes. Tensors are very large data structures, performing computation on models which can have billions of weights."

### Texture Sampling in Compute Shaders

WebGPU compute shaders can sample textures using hardware texture units, enabling efficient bilinear/trilinear filtering for vision encoding:

```wgsl
@group(0) @binding(0) var visionFeatures: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;

@compute @workgroup_size(16, 16)
fn processImagePatch(@builtin(global_invocation_id) gid: vec3<u32>) {
  let texSize = textureDimensions(visionFeatures);
  let uv = vec2<f32>(gid.xy) / vec2<f32>(texSize);

  // Hardware-accelerated bilinear interpolation
  let sampledValue = textureSample(visionFeatures, linearSampler, uv);

  // Process sampled features...
}
```

**Texture sampling advantages for VLMs:**
- **Hardware acceleration**: Native texture unit filtering (no manual interpolation)
- **Memory efficiency**: Compressed texture formats (BC7, ASTC)
- **Cache optimization**: Texture caches are optimized for 2D spatial locality
- **Quality control**: Automatic mipmap generation for multi-scale features

**Sampler configuration:**
```javascript
const sampler = device.createSampler({
  magFilter: 'linear',      // Magnification filter
  minFilter: 'linear',      // Minification filter
  mipmapFilter: 'linear',   // Between mipmap levels
  addressModeU: 'clamp-to-edge',
  addressModeV: 'clamp-to-edge'
});
```

### Storage Buffer vs Texture Memory

**Storage buffers** (best for transformers):
- Direct float32/float16 access
- No format restrictions
- Optimal for attention mechanisms, MLPs
- Max size: ~2GB per buffer

**Texture memory** (best for vision encoders):
- Hardware filtering (bilinear, trilinear)
- Compressed formats save bandwidth
- 2D/3D spatial locality optimization
- Limited to 16K × 16K texels

**Hybrid approach for VLMs:**
```javascript
// Vision encoder: Use textures for image patches
const imageTexture = device.createTexture({
  size: [768, 768],
  format: 'rgba16float',
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
});

// Language decoder: Use storage buffers for token embeddings
const tokenBuffer = device.createBuffer({
  size: 4096 * 768 * 4, // 4096 tokens × 768 dim × 4 bytes
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
});
```

From [Using WebGPU - ONNX Runtime](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html) (accessed 2025-01-31):
> "By default, a model's inputs and outputs are tensors that hold data in CPU memory. When you run a session with WebGPU EP, the data is copied to GPU memory, and the results are copied back to CPU memory."

## Section 2: Browser-Based VLM Deployment Architecture

### Model Quantization for Web

Browser VLM inference requires aggressive quantization to fit within memory constraints and achieve acceptable performance:

**Quantization levels:**
- **FP16 (16-bit float)**: 2× memory reduction, minimal accuracy loss
- **INT8 (8-bit integer)**: 4× memory reduction, <1% accuracy degradation
- **INT4 (4-bit integer)**: 8× memory reduction, requires careful calibration
- **Mixed precision**: FP16 for critical layers, INT8 for bulk weights

**Example model sizes (7B parameter VLM):**
- FP32: 28 GB (impractical for web)
- FP16: 14 GB (borderline)
- INT8: 7 GB (workable)
- INT4: 3.5 GB (optimal for web)

**ONNX Runtime Web quantization workflow:**
```javascript
// Load quantized model (INT8 weights, FP16 activations)
const session = await ort.InferenceSession.create('vlm-q4f16.onnx', {
  executionProviders: [{
    name: 'webgpu',
    preferredLayout: 'NCHW'  // Optimize for texture memory
  }]
});
```

From [WebGPU Inference: LLMs That Run in Your Browser](https://medium.com/@bhagyarana80/webgpu-inference-llms-that-run-in-your-browser-6251d27a0565) (accessed 2025-01-31):
> "Choose a compact, browser-friendly bundle (e.g., 3B–7B, quantized)"

### Progressive Loading Strategies

Large VLM models require progressive loading to avoid blocking the browser:

**1. Chunked model loading:**
```javascript
async function loadModelProgressive(modelUrl, onProgress) {
  const response = await fetch(modelUrl);
  const reader = response.body.getReader();
  const contentLength = +response.headers.get('Content-Length');

  let receivedLength = 0;
  let chunks = [];

  while(true) {
    const {done, value} = await reader.read();
    if (done) break;

    chunks.push(value);
    receivedLength += value.length;
    onProgress(receivedLength / contentLength);
  }

  const blob = new Blob(chunks);
  return await ort.InferenceSession.create(blob);
}
```

**2. Model sharding (split across multiple files):**
```javascript
// Load vision encoder first (needed immediately)
const visionEncoder = await ort.InferenceSession.create('vision-encoder.onnx');

// Stream language decoder in background
loadModelProgressive('language-decoder.onnx', (progress) => {
  console.log(`Loading decoder: ${(progress * 100).toFixed(1)}%`);
}).then(decoder => {
  languageDecoder = decoder;
});
```

**3. IndexedDB caching:**
```javascript
async function loadCachedModel(modelUrl) {
  const db = await openDB('vlm-models', 1, {
    upgrade(db) {
      db.createObjectStore('models');
    }
  });

  // Check cache first
  let modelData = await db.get('models', modelUrl);
  if (!modelData) {
    // Download and cache
    const response = await fetch(modelUrl);
    modelData = await response.arrayBuffer();
    await db.put('models', modelData, modelUrl);
  }

  return await ort.InferenceSession.create(modelData);
}
```

### Memory Management Constraints

Browser memory limits vary by platform:
- **Desktop Chrome**: 4-8 GB JavaScript heap + 2 GB WebGPU buffers
- **Mobile Chrome**: 2-4 GB total (shared between CPU/GPU)
- **Safari**: More restrictive, ~2 GB WebGPU limit

**Memory optimization techniques:**
```javascript
// 1. Dispose intermediate tensors explicitly
const intermediate = await session.run({input: inputTensor});
const result = await processOutput(intermediate.output);
intermediate.output.dispose();  // Free GPU memory immediately

// 2. Use GPU tensors to avoid CPU copies
const gpuInputTensor = ort.Tensor.fromGpuBuffer(gpuBuffer, {
  dataType: 'float16',
  dims: [1, 3, 224, 224]
});

// 3. Enable graph capture for static shapes (reduces overhead)
const session = await ort.InferenceSession.create(modelUrl, {
  executionProviders: [{
    name: 'webgpu',
    enableGraphCapture: true  // Compile ahead of time
  }]
});
```

From [Using WebGPU - ONNX Runtime](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html) (accessed 2025-01-31):
> "Call `tensor.dispose()` explicitly to destroy the underlying GPU buffer when it is no longer needed."

### KV Cache Management for Autoregressive Decoding

Vision-language models generate text autoregressively (one token at a time), requiring careful KV cache management:

**Problem**: Each forward pass recomputes attention for all previous tokens
**Solution**: Cache key/value tensors and reuse them

```javascript
class VLMInferenceEngine {
  constructor(session) {
    this.session = session;
    this.kvCache = new Map();  // Store GPU tensors
  }

  async generateToken(imageFeatures, previousTokens, pastKV) {
    // First token: no KV cache
    if (!pastKV) {
      const feeds = {
        'image_features': imageFeatures,
        'input_ids': previousTokens
      };
      const outputs = await this.session.run(feeds);

      // Store KV cache on GPU (no CPU copy!)
      return {
        nextToken: outputs.logits,
        pastKV: {
          key: outputs.past_key,    // Keep on GPU
          value: outputs.past_value
        }
      };
    }

    // Subsequent tokens: reuse KV cache
    const feeds = {
      'image_features': imageFeatures,
      'input_ids': previousTokens.slice(-1),  // Only last token
      'past_key': pastKV.key,
      'past_value': pastKV.value
    };

    const outputs = await this.session.run(feeds);

    // Dispose old KV cache (GPU memory)
    pastKV.key.dispose();
    pastKV.value.dispose();

    return {
      nextToken: outputs.logits,
      pastKV: {
        key: outputs.present_key,
        value: outputs.present_value
      }
    };
  }

  async generate(imageFeatures, maxTokens = 100) {
    let tokens = [START_TOKEN];
    let pastKV = null;

    for (let i = 0; i < maxTokens; i++) {
      const result = await this.generateToken(
        imageFeatures,
        new ort.Tensor('int64', tokens, [1, tokens.length]),
        pastKV
      );

      const nextToken = argmax(result.nextToken);
      if (nextToken === END_TOKEN) break;

      tokens.push(nextToken);
      pastKV = result.pastKV;
    }

    // Clean up final KV cache
    if (pastKV) {
      pastKV.key.dispose();
      pastKV.value.dispose();
    }

    return tokens;
  }
}
```

## Section 3: Framework Ecosystem and Integration

### ONNX Runtime Web with WebGPU

ONNX Runtime Web provides production-ready WebGPU support with optimized operators:

**Basic setup:**
```javascript
import * as ort from 'onnxruntime-web/webgpu';

// Configure WebGPU backend
ort.env.wasm.numThreads = 1;  // WebGPU handles parallelism
ort.env.wasm.simd = true;
ort.env.webgpu.powerPreference = 'high-performance';

const session = await ort.InferenceSession.create('model.onnx', {
  executionProviders: ['webgpu'],
  graphOptimizationLevel: 'all'
});

// Run inference
const feeds = {
  'input': new ort.Tensor('float32', imageData, [1, 3, 224, 224])
};
const results = await session.run(feeds);
```

**Advanced WebGPU configuration:**
```javascript
// Access underlying WebGPU device
const device = ort.env.webgpu.device;

// Create custom GPU buffer for zero-copy input
const inputBuffer = device.createBuffer({
  size: 1 * 3 * 224 * 224 * 4,  // NCHW × 4 bytes
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  mappedAtCreation: true
});

// Write image data directly to GPU
const mapping = inputBuffer.getMappedRange();
new Float32Array(mapping).set(imageData);
inputBuffer.unmap();

// Create tensor from GPU buffer (no CPU copy!)
const gpuTensor = ort.Tensor.fromGpuBuffer(inputBuffer, {
  dataType: 'float32',
  dims: [1, 3, 224, 224]
});

// Inference with GPU tensor
const results = await session.run({ 'input': gpuTensor });

// Output stays on GPU for downstream processing
const outputGpuTensor = results.output;  // Still on GPU!
```

From [Using WebGPU - ONNX Runtime](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html) (accessed 2025-01-31):
> "To use WebGPU EP, you just need to make 2 small changes: Update your import statement... and specify 'webgpu' EP explicitly in session options"

### TensorFlow.js WebGPU Backend

TensorFlow.js offers WebGPU support with a high-level API:

```javascript
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';

// Set WebGPU backend
await tf.setBackend('webgpu');
await tf.ready();

// Load model (TF.js format)
const model = await tf.loadGraphModel('model.json');

// Inference with automatic GPU execution
const imageTensor = tf.browser.fromPixels(imageElement)
  .expandDims(0)
  .div(255.0);

const predictions = model.predict(imageTensor);
const output = await predictions.data();  // Copy to CPU

// Clean up GPU memory
imageTensor.dispose();
predictions.dispose();
```

**WebGPU-specific optimizations:**
```javascript
// Enable memory growth (avoid OOM on large models)
tf.env().set('WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE', 1);

// Force float16 precision (2× faster, half memory)
tf.env().set('WEBGPU_USE_FLOAT16', true);

// Custom WebGPU kernel (advanced)
tf.registerKernel({
  kernelName: 'MyCustomOp',
  backendName: 'webgpu',
  kernelFunc: ({inputs, backend, attrs}) => {
    const webgpuBackend = backend;
    const device = webgpuBackend.device;

    // Write custom WGSL shader...
    const program = device.createComputePipeline({...});

    // Execute on GPU...
    return outputTensor;
  }
});
```

From [Machine Learning in the Browser: JS Just Got Smarter](https://medium.com/@ThinkingLoop/machine-learning-in-the-browser-js-just-got-smarter-af16b7d7af42) (accessed 2025-01-31):
> "With TensorFlow.js, ONNX Runtime, WebAssembly, and WebGPU, the gap between 'frontend developer' and 'AI engineer' is shrinking fast."

### Transformers.js with WebGPU

Hugging Face Transformers.js brings transformer models to the browser:

```javascript
import { pipeline, env } from '@xenova/transformers';

// Use WebGPU device (via ONNX Runtime Web)
env.backends.onnx.wasm.proxy = false;

// Load vision-language model
const captioner = await pipeline(
  'image-to-text',
  'Xenova/vit-gpt2-image-captioning',
  { device: 'webgpu' }
);

// Generate caption from image
const result = await captioner('https://example.com/image.jpg', {
  max_length: 50,
  num_beams: 4
});

console.log(result[0].generated_text);
// Output: "a cat sitting on a wooden table next to a laptop"
```

**Custom VLM inference with Transformers.js:**
```javascript
import { AutoTokenizer, AutoModelForVision2Seq } from '@xenova/transformers';

// Load tokenizer and model separately
const tokenizer = await AutoTokenizer.from_pretrained('llava-1.5-7b');
const model = await AutoModelForVision2Seq.from_pretrained('llava-1.5-7b', {
  device: 'webgpu',
  dtype: 'fp16'  // Use float16 for efficiency
});

// Prepare inputs
const imageInput = await prepareImageTensor(imageUrl);
const textInput = tokenizer("Describe this image:", {return_tensors: 'pt'});

// Run VLM inference
const outputs = await model.generate({
  pixel_values: imageInput,
  input_ids: textInput.input_ids,
  max_length: 200,
  temperature: 0.7
});

// Decode output tokens
const caption = tokenizer.decode(outputs[0], {skip_special_tokens: true});
```

### WebLLM for Browser-Native LLMs

WebLLM (using TVM compilation) offers optimized LLM inference with WebGPU:

```javascript
import { CreateWebWorkerMLCEngine } from "@mlc-ai/web-llm";

// Initialize in web worker (keeps UI responsive)
const engine = await CreateWebWorkerMLCEngine(
  new Worker('./worker.js', {type: 'module'}),
  'Llama-3-8B-Instruct-q4f16',  // Quantized model
  {
    initProgressCallback: (progress) => {
      console.log(`Loading: ${progress.progress.toFixed(1)}%`);
    }
  }
);

// OpenAI-compatible streaming API
const stream = await engine.chat.completions.create({
  messages: [
    { role: "user", content: "Describe this image" },
    { role: "user", content: imageBase64, type: "image_url" }
  ],
  stream: true,
  temperature: 0.8
});

// Stream tokens as they're generated
for await (const chunk of stream) {
  const token = chunk.choices[0]?.delta?.content;
  if (token) updateUI(token);
}
```

From [WebAssembly and WebGPU enhancements for faster Web AI](https://developer.chrome.com/blog/io24-webassembly-webgpu-1) (Chrome Developers, accessed 2025-01-31):
> "Today, application developers and researchers build models using frameworks, models execute in the browser using a runtime like Tensorflow.js or ONNX Runtime Web"

## Section 4: Performance Optimization and Real-World Deployment

### Benchmarking Browser VLM Performance

**Key metrics for browser VLM inference:**
- **First Token Latency (FTL)**: Time to first generated token (critical for UX)
- **Tokens per Second (TPS)**: Throughput for longer generations
- **Model Load Time**: Initial startup cost (cached vs uncached)
- **Memory Footprint**: Peak GPU + CPU memory usage

**Example benchmark results (7B VLM, INT4 quantized):**
```
Device: MacBook Pro M1 Max (32-core GPU)
Browser: Chrome 120 with WebGPU

Model Load Time: 8.2s (first run), 1.1s (cached)
First Token Latency: 340ms
Tokens/Second: 18.5 TPS
Peak GPU Memory: 4.2 GB
Peak CPU Memory: 1.8 GB

Device: Windows Desktop (RTX 4070)
Browser: Chrome 120 with WebGPU

Model Load Time: 12.1s (first run), 1.5s (cached)
First Token Latency: 280ms
Tokens/Second: 24.3 TPS
Peak GPU Memory: 5.1 GB
Peak CPU Memory: 2.1 GB
```

**Profiling WebGPU performance:**
```javascript
// Enable timestamp queries
const timestampQuerySet = device.createQuerySet({
  type: 'timestamp',
  count: 2
});

// Measure GPU execution time
const commandEncoder = device.createCommandEncoder();
commandEncoder.writeTimestamp(timestampQuerySet, 0);

// ... run compute passes ...

commandEncoder.writeTimestamp(timestampQuerySet, 1);

// Read back timing data
const resolveBuffer = device.createBuffer({
  size: 16,
  usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
});

commandEncoder.resolveQuerySet(timestampQuerySet, 0, 2, resolveBuffer, 0);
device.queue.submit([commandEncoder.finish()]);

// Get elapsed time in nanoseconds
const timingData = await resolveBuffer.mapAsync(GPUMapMode.READ);
const timestamps = new BigUint64Array(resolveBuffer.getMappedRange());
const elapsedNs = Number(timestamps[1] - timestamps[0]);
console.log(`GPU time: ${(elapsedNs / 1e6).toFixed(2)}ms`);
```

### Production Deployment Patterns

**1. Progressive Web App (PWA) with offline support:**
```javascript
// Service worker for model caching
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open('vlm-models-v1').then((cache) => {
      return cache.addAll([
        '/models/vision-encoder.onnx',
        '/models/language-decoder.onnx',
        '/models/tokenizer.json'
      ]);
    })
  );
});

self.addEventListener('fetch', (event) => {
  if (event.request.url.includes('/models/')) {
    event.respondWith(
      caches.match(event.request).then((response) => {
        return response || fetch(event.request);
      })
    );
  }
});
```

**2. CDN-hosted models with version control:**
```javascript
const MODEL_VERSION = 'v2.1.0';
const CDN_BASE = 'https://cdn.example.com/models';

async function loadModel() {
  const modelUrl = `${CDN_BASE}/${MODEL_VERSION}/vlm-q4f16.onnx`;

  // Check cache version
  const cachedVersion = localStorage.getItem('model-version');
  if (cachedVersion !== MODEL_VERSION) {
    // Clear old cache
    const cache = await caches.open('vlm-models');
    await cache.delete(modelUrl);
    localStorage.setItem('model-version', MODEL_VERSION);
  }

  return await loadCachedModel(modelUrl);
}
```

**3. Adaptive quality based on device capabilities:**
```javascript
async function selectOptimalModel() {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  // Detect GPU capabilities
  const limits = device.limits;
  const maxBufferSize = limits.maxStorageBufferBindingSize;

  // Select model size based on memory
  if (maxBufferSize >= 8 * 1024 ** 3) {
    return 'vlm-13b-q4.onnx';  // 13B model for high-end
  } else if (maxBufferSize >= 4 * 1024 ** 3) {
    return 'vlm-7b-q4.onnx';   // 7B model for mid-range
  } else {
    return 'vlm-3b-q8.onnx';   // 3B model for low-end
  }
}
```

### Error Handling and Fallbacks

**Graceful degradation strategy:**
```javascript
async function initVLMInference() {
  // 1. Try WebGPU (best performance)
  if ('gpu' in navigator) {
    try {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance'
      });
      if (adapter) {
        const session = await ort.InferenceSession.create(modelUrl, {
          executionProviders: ['webgpu']
        });
        return { backend: 'webgpu', session };
      }
    } catch (e) {
      console.warn('WebGPU failed, falling back to WebAssembly:', e);
    }
  }

  // 2. Fallback to WASM (CPU, slower but universal)
  const session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ['wasm'],
    executionProviderOptions: {
      wasm: {
        numThreads: navigator.hardwareConcurrency || 4
      }
    }
  });
  return { backend: 'wasm', session };
}

// User-facing error messages
async function handleInferenceError(error) {
  if (error.message.includes('out of memory')) {
    return {
      error: 'GPU_OOM',
      message: 'Image is too large. Try a smaller image.',
      suggestion: 'Reduce image resolution to 1024×1024 or lower.'
    };
  } else if (error.message.includes('device lost')) {
    return {
      error: 'GPU_LOST',
      message: 'GPU connection lost. Please refresh the page.',
      suggestion: 'This can happen if another app is using the GPU.'
    };
  }
  // Generic error
  return {
    error: 'INFERENCE_FAILED',
    message: 'Unable to process image. Please try again.',
    suggestion: error.message
  };
}
```

### Security Considerations

**Content Security Policy (CSP) for WebGPU:**
```html
<meta http-equiv="Content-Security-Policy"
      content="default-src 'self';
               script-src 'self' 'wasm-unsafe-eval';
               worker-src 'self' blob:;">
```

**Preventing malicious model injection:**
```javascript
// Verify model integrity with SHA-256 hash
async function loadVerifiedModel(url, expectedHash) {
  const response = await fetch(url);
  const buffer = await response.arrayBuffer();

  // Compute hash
  const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
  const hashHex = Array.from(new Uint8Array(hashBuffer))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');

  if (hashHex !== expectedHash) {
    throw new Error('Model integrity check failed!');
  }

  return await ort.InferenceSession.create(buffer);
}
```

**Limiting resource usage (prevent DoS):**
```javascript
// Timeout mechanism
async function runInferenceWithTimeout(session, feeds, timeoutMs = 30000) {
  const timeoutPromise = new Promise((_, reject) => {
    setTimeout(() => reject(new Error('Inference timeout')), timeoutMs);
  });

  const inferencePromise = session.run(feeds);

  return await Promise.race([inferencePromise, timeoutPromise]);
}

// Rate limiting
class RateLimiter {
  constructor(maxRequests, windowMs) {
    this.maxRequests = maxRequests;
    this.windowMs = windowMs;
    this.requests = [];
  }

  async checkLimit() {
    const now = Date.now();
    this.requests = this.requests.filter(t => now - t < this.windowMs);

    if (this.requests.length >= this.maxRequests) {
      throw new Error('Rate limit exceeded. Please wait.');
    }

    this.requests.push(now);
  }
}

const limiter = new RateLimiter(10, 60000);  // 10 requests per minute
```

From [WebAssembly and WebGPU enhancements for faster Web AI](https://developer.chrome.com/blog/io24-webassembly-webgpu-1) (Chrome Developers, accessed 2025-01-31):
> "Special purpose compute on the GPU or accelerators can offer performance that is orders of magnitude higher, especially for larger models and on high-end devices."

## Sources

**Web Research:**
- [WebAssembly and WebGPU enhancements for faster Web AI, part 1](https://developer.chrome.com/blog/io24-webassembly-webgpu-1) - Chrome for Developers (accessed 2025-01-31)
- [WebGPU Inference: LLMs That Run in Your Browser](https://medium.com/@bhagyarana80/webgpu-inference-llms-that-run-in-your-browser-6251d27a0565) - Medium (accessed 2025-01-31)
- [Using WebGPU - ONNX Runtime](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html) - ONNX Runtime Documentation (accessed 2025-01-31)
- [Machine Learning in the Browser: JS Just Got Smarter](https://medium.com/@ThinkingLoop/machine-learning-in-the-browser-js-just-got-smarter-af16b7d7af42) - Medium (accessed 2025-01-31)

**Additional References:**
- [WebGPU Specification](https://www.w3.org/TR/webgpu/) - W3C Web Standard
- [WebGPU Fundamentals](https://webgpufundamentals.org/) - WebGPU Learning Resource
- [ONNX Runtime Web GitHub](https://github.com/microsoft/onnxruntime/tree/main/js/web) - Source Code
- [TensorFlow.js WebGPU Backend](https://www.tensorflow.org/js) - TensorFlow.js Documentation
- [Transformers.js](https://huggingface.co/docs/transformers.js) - Hugging Face Documentation
- [WebLLM](https://github.com/mlc-ai/web-llm) - MLC AI Project
