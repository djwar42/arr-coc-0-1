# DirectX 12 Texture Operations for Neural Rendering

**Date**: 2025-01-31  
**Focus**: D3D12 API for VLM inference and neural rendering pipelines

---

## Overview

DirectX 12 provides low-level GPU control for texture operations, compute shaders, and neural network inference via DirectML. Useful for Windows-based VLM deployment, game engine integration, and real-time applications.

---

## DirectX 12 Texture API (~80 lines)

### Descriptor Heaps

**Descriptor**: GPU handle to resource (texture, buffer, sampler)

```cpp
// Create descriptor heap
D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
heapDesc.NumDescriptors = 1000;  // Max descriptors
heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;  // Shader resources
heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

ID3D12DescriptorHeap* descHeap;
device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&descHeap));
```

**VLM use**: Store visual token textures in descriptor heap

### Bindless Textures

**Traditional**: Bind textures one-by-one (slow for many textures)
**Bindless**: Access all textures via index (fast)

```hlsl
// Bindless texture array
Texture2D textures[] : register(t0, space0);  // Unbounded array

// Compute shader
[numthreads(8, 8, 1)]
void LoadPatches(uint3 dtid : SV_DispatchThreadID) {
    uint patchIdx = dtid.x;
    uint imageIdx = dtid.y;

    // Access texture by index (bindless!)
    float4 patch = textures[imageIdx].Load(patchCoords[patchIdx]);
}
```

**Benefit**: Batch processing multiple images without rebinding

### Resource Barriers

**Purpose**: Synchronize GPU resource access

```cpp
// Transition texture for compute shader
D3D12_RESOURCE_BARRIER barrier = {};
barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
barrier.Transition.pResource = visionTexture;
barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

commandList->ResourceBarrier(1, &barrier);
```

---

## Neural Rendering with D3D12 (~110 lines)

### Compute Shaders for Inference

```hlsl
// ViT patch embedding in compute shader
RWStructuredBuffer<float> patchEmbeddings : register(u0);
Texture2D<float3> inputImage : register(t0);
cbuffer Constants : register(b0) {
    uint patchSize;
    uint numPatches;
};

[numthreads(16, 16, 1)]
void ExtractPatches(uint3 dtid : SV_DispatchThreadID) {
    uint patchX = dtid.x;
    uint patchY = dtid.y;
    uint patchIdx = patchY * numPatches + patchX;

    // Load patch pixels
    float3 avgColor = 0;
    for (uint y = 0; y < patchSize; y++) {
        for (uint x = 0; x < patchSize; x++) {
            avgColor += inputImage.Load(int3(patchX * patchSize + x, patchY * patchSize + y, 0));
        }
    }
    avgColor /= (patchSize * patchSize);

    // Store embedding
    patchEmbeddings[patchIdx * 3 + 0] = avgColor.r;
    patchEmbeddings[patchIdx * 3 + 1] = avgColor.g;
    patchEmbeddings[patchIdx * 3 + 2] = avgColor.b;
}
```

### DirectML Integration

**DirectML**: Microsoft's API for GPU-accelerated ML inference

```cpp
// Create DirectML device
IDMLDevice* dmlDevice;
DMLCreateDevice(d3d12Device, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&dmlDevice));

// Load ONNX model (ViT encoder)
IDMLCompiledOperator* vitOperator;
dmlDevice->CompileGraph(graphDesc, DML_EXECUTION_FLAG_NONE, IID_PPV_ARGS(&vitOperator));

// Inference
dmlDevice->DispatchGraph(vitOperator, inputTensor, outputTensor);
```

**VLM pipeline**:
1. D3D12 compute: Extract patches
2. DirectML: ViT encoder
3. DirectML: Language model
4. D3D12: Render output

### Texture Readback for Loss Computation

**Training**: Need to read GPU results back to CPU

```cpp
// Create readback resource
D3D12_RESOURCE_DESC readbackDesc = {};
readbackDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
readbackDesc.Width = textureSize;
readbackDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

ID3D12Resource* readbackBuffer;
device->CreateCommittedResource(
    &heapProps,
    D3D12_HEAP_FLAG_NONE,
    &readbackDesc,
    D3D12_RESOURCE_STATE_COPY_DEST,
    nullptr,
    IID_PPV_ARGS(&readbackBuffer)
);

// Copy GPU texture to readback buffer
commandList->CopyResource(readbackBuffer, gpuTexture);

// Map to CPU
void* cpuData;
readbackBuffer->Map(0, nullptr, &cpuData);
// ... compute loss using cpuData ...
readbackBuffer->Unmap(0, nullptr);
```

---

## VLM-Specific Patterns (~90 lines)

### Multi-Model Inference Pipelines

**Use case**: Ensemble of VLMs for robustness

```cpp
// Pipeline: VLM1 + VLM2 + VLM3 → Voting
for (int modelIdx = 0; modelIdx < 3; modelIdx++) {
    // Bind model-specific resources
    descHeap->SetDescriptorHeap(modelDescHeaps[modelIdx]);

    // Dispatch VLM inference
    dmlDevice->DispatchGraph(vlmModels[modelIdx], inputTokens, outputs[modelIdx]);
}

// Voting compute shader
commandList->Dispatch(numQueries, 1, 1);  // Combine 3 outputs
```

### Vision-Language Fusion in Compute Shaders

```hlsl
// Cross-attention between vision and text tokens
StructuredBuffer<float> visionTokens : register(t0);  // [196, 768]
StructuredBuffer<float> textTokens : register(t1);    // [77, 768]
RWStructuredBuffer<float> fusedTokens : register(u0); // [273, 768]

[numthreads(64, 1, 1)]
void FuseTokens(uint dtid : SV_DispatchThreadID) {
    uint tokenIdx = dtid.x;

    if (tokenIdx < 196) {
        // Vision token
        fusedTokens[tokenIdx] = visionTokens[tokenIdx];
    } else {
        // Text token
        fusedTokens[tokenIdx] = textTokens[tokenIdx - 196];
    }

    // Cross-attention (simplified)
    float attnWeight = dot(fusedTokens[tokenIdx], allTokens[0]);
    fusedTokens[tokenIdx] *= attnWeight;
}
```

### Cross-Device Synchronization

**Multi-GPU inference**:
```cpp
// Device 0: Process vision
ID3D12CommandQueue* queueGPU0;
device0->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queueGPU0));
queueGPU0->ExecuteCommandLists(1, visionCommandLists);

// Device 1: Process language
ID3D12CommandQueue* queueGPU1;
device1->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queueGPU1));
queueGPU1->ExecuteCommandLists(1, languageCommandLists);

// Synchronize
ID3D12Fence* fence;
device0->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence));
queueGPU0->Signal(fence, 1);
queueGPU1->Wait(fence, 1);  // GPU1 waits for GPU0
```

---

## Performance Optimization (~60 lines)

### Command List Recording

**Bundle commonly-used commands**:
```cpp
// Record bundle once
ID3D12GraphicsCommandList* bundle;
device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_BUNDLE, bundleAllocator, nullptr, IID_PPV_ARGS(&bundle));

bundle->SetDescriptorHeaps(1, &descHeap);
bundle->SetComputeRootSignature(rootSignature);
bundle->Dispatch(numPatches, 1, 1);
bundle->Close();

// Execute many times (fast)
for (int frame = 0; frame < 1000; frame++) {
    commandList->ExecuteBundle(bundle);
}
```

### GPU Timeline Optimization

**Overlap compute and copy**:
```
Timeline:
|--- Compute Queue: VLM Inference ---|
         |--- Copy Queue: Texture Upload ---|
```

**Implementation**:
```cpp
// Compute queue
computeQueue->ExecuteCommandLists(1, inferenceCommands);

// Copy queue (parallel)
copyQueue->ExecuteCommandLists(1, uploadCommands);

// Synchronize before next frame
computeQueue->Wait(copyFence, 1);
```

### Memory Aliasing

**Share GPU memory between resources**:
```cpp
// Heap for aliased resources
D3D12_HEAP_DESC heapDesc = {};
heapDesc.SizeInBytes = 256 * 1024 * 1024;  // 256 MB
ID3D12Heap* aliasHeap;
device->CreateHeap(&heapDesc, IID_PPV_ARGS(&aliasHeap));

// Resource 1: Vision encoder output
device->CreatePlacedResource(aliasHeap, 0, &desc1, ...);

// Resource 2: Language model input (reuses same memory!)
device->CreatePlacedResource(aliasHeap, 0, &desc2, ...);
```

**Benefit**: 2× memory savings (sequential pipeline stages)

---

## Cross-References

- `06-cuda-texture-memory-vit.md` - CUDA equivalent
- DirectML docs - microsoft.com/directml

---

## Summary

DirectX 12 for VLM inference:
- **Compute shaders**: GPU-accelerated patch extraction, fusion
- **DirectML**: ONNX model deployment
- **Bindless textures**: Batch processing
- **Multi-GPU**: Cross-device synchronization

**Use cases**: Windows VLM apps, game engine integration, real-time rendering
