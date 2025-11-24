# Knowledge Expansion: Multi-GPU, Inference, Orchestration, Alternative Hardware

**Date**: 2025-11-13
**Status**: Ready for parallel execution (16 runners)
**Target**: Fill knowledge gaps - distributed training, inference optimization, orchestration, alternative hardware

---

## EXPANSION 18: Multi-GPU Distributed Frameworks

### PART 1: Create karpathy/distributed-training/00-deepspeed-zero-optimizer.md (~500 lines)

- [✓] PART 1: Create karpathy/distributed-training/00-deepspeed-zero-optimizer.md (FILE EXISTS - 22.6 KB)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md for distributed training mentions
- [ ] Grep for "DeepSpeed" AND "ZeRO" in karpathy/ folder
- [ ] Read any existing distributed-*.md files
- [ ] Identify gaps: What's NOT covered about DeepSpeed ZeRO?

**Step 1: Web Research - DeepSpeed ZeRO**
- [ ] Search: "DeepSpeed ZeRO optimizer tutorial 2024"
- [ ] Search: "ZeRO-1 ZeRO-2 ZeRO-3 comparison"
- [ ] Search: "DeepSpeed memory optimization techniques"
- [ ] Scrape top 3 results for detailed content

**Step 2: Web Research - ZeRO Implementation**
- [ ] Search: "site:github.com DeepSpeed ZeRO PyTorch example"
- [ ] Search: "ZeRO optimizer partitioning strategy"
- [ ] Scrape DeepSpeed GitHub docs/tutorials

**Step 3: Create Knowledge File**
- [ ] Create karpathy/distributed-training/00-deepspeed-zero-optimizer.md
- [ ] Section 1: ZeRO Overview - what is it, why use it (~100 lines)
- [ ] Section 2: ZeRO-1, ZeRO-2, ZeRO-3 comparison (~150 lines)
- [ ] Section 3: Memory optimization breakdown (~100 lines)
- [ ] Section 4: PyTorch integration & code examples (~100 lines)
- [ ] Section 5: arr-coc-0-1 use cases (~50 lines)
- [ ] All sections must cite web sources

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-deepspeed-zero-2025-11-13-[TIME].md
- [ ] Include: Runner (PART 1), Timestamp, Status (✓ COMPLETE)
- [ ] List knowledge file created with line count
- [ ] List sources used (arXiv, GitHub, tutorials)
- [ ] Describe how this fills distributed training gaps

---

### PART 2: Create karpathy/distributed-training/01-deepspeed-pipeline-parallelism.md (~450 lines)

- [✓] PART 2: Create karpathy/distributed-training/01-deepspeed-pipeline-parallelism.md (Completed 2025-11-13 22:59)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md for pipeline parallelism mentions
- [ ] Grep for "pipeline" AND "parallelism" in karpathy/
- [ ] Check what we already know about DualPipe (DeepSeek)
- [ ] Identify gaps: DeepSpeed-specific pipeline patterns

**Step 1: Web Research - Pipeline Parallelism**
- [ ] Search: "DeepSpeed pipeline parallelism tutorial"
- [ ] Search: "GPipe vs PipeDream vs DeepSpeed comparison"
- [ ] Search: "pipeline parallelism batch splitting"
- [ ] Scrape top 3 results

**Step 2: Web Research - DeepSpeed Implementation**
- [ ] Search: "site:github.com DeepSpeed pipeline example"
- [ ] Search: "DeepSpeed Megatron integration"
- [ ] Scrape GitHub examples

**Step 3: Create Knowledge File**
- [ ] Create karpathy/distributed-training/01-deepspeed-pipeline-parallelism.md
- [ ] Section 1: Pipeline parallelism fundamentals (~80 lines)
- [ ] Section 2: DeepSpeed pipeline implementation (~120 lines)
- [ ] Section 3: Micro-batching strategies (~100 lines)
- [ ] Section 4: Comparison with other frameworks (~100 lines)
- [ ] Section 5: VLM-specific patterns (~50 lines)
- [ ] Cite all web sources

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-pipeline-parallelism-2025-11-13-[TIME].md

---

### PART 3: Create karpathy/distributed-training/02-megatron-lm-tensor-parallelism.md (~500 lines)

- [✓] PART 3: Create karpathy/distributed-training/02-megatron-lm-tensor-parallelism.md (Completed 2025-11-13 22:57)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md for Megatron mentions
- [ ] Grep for "Megatron" OR "tensor parallelism" in karpathy/
- [ ] Check existing NVIDIA-related files
- [ ] Identify gaps: Megatron-LM specifics

**Step 1: Web Research - Megatron-LM**
- [ ] Search: "Megatron-LM tensor parallelism explained"
- [ ] Search: "Megatron-LM vs DeepSpeed comparison"
- [ ] Search: "NVIDIA Megatron-LM tutorial 2024"
- [ ] Scrape top 3 results

**Step 2: Web Research - Tensor Parallelism**
- [ ] Search: "site:github.com NVIDIA Megatron-LM examples"
- [ ] Search: "tensor parallelism vs data parallelism"
- [ ] Scrape NVIDIA docs

**Step 3: Create Knowledge File**
- [ ] Create karpathy/distributed-training/02-megatron-lm-tensor-parallelism.md
- [ ] Section 1: Tensor parallelism fundamentals (~100 lines)
- [ ] Section 2: Megatron-LM architecture (~150 lines)
- [ ] Section 3: Tensor slicing strategies (~100 lines)
- [ ] Section 4: Communication patterns (~100 lines)
- [ ] Section 5: Multi-GPU VLM training (~50 lines)
- [ ] Cite all sources

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-megatron-lm-2025-11-13-[TIME].md

---

### PART 4: Create karpathy/distributed-training/03-fsdp-vs-deepspeed.md (~400 lines)

- [✓] PART 4: Create karpathy/distributed-training/03-fsdp-vs-deepspeed.md (Completed 2025-11-13 22:58)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md for FSDP mentions
- [ ] Grep for "FSDP" in karpathy/ (we have some mentions)
- [ ] Check what we know about distributed frameworks
- [ ] Identify gaps: Direct FSDP vs DeepSpeed comparison

**Step 1: Web Research - FSDP**
- [ ] Search: "PyTorch FSDP vs DeepSpeed ZeRO"
- [ ] Search: "FSDP2 improvements 2024"
- [ ] Search: "when to use FSDP vs DeepSpeed"
- [ ] Scrape top 3 results

**Step 2: Web Research - Benchmarks**
- [ ] Search: "FSDP DeepSpeed performance comparison"
- [ ] Search: "site:github.com FSDP example PyTorch"
- [ ] Scrape benchmark data

**Step 3: Create Knowledge File**
- [ ] Create karpathy/distributed-training/03-fsdp-vs-deepspeed.md
- [ ] Section 1: FSDP overview (~80 lines)
- [ ] Section 2: DeepSpeed overview (~80 lines)
- [ ] Section 3: Feature comparison table (~100 lines)
- [ ] Section 4: Performance benchmarks (~80 lines)
- [ ] Section 5: When to use which (~60 lines)
- [ ] Cite all sources

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-fsdp-vs-deepspeed-2025-11-13-[TIME].md

---

## EXPANSION 19: Production Inference Optimization

### PART 5: Create karpathy/inference-optimization/00-tensorrt-fundamentals.md (~500 lines)

- [✓] PART 5: Create karpathy/inference-optimization/00-tensorrt-fundamentals.md (Completed 2025-11-13)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md for TensorRT mentions
- [ ] Grep for "TensorRT" OR "inference optimization" in karpathy/
- [ ] Check existing inference-related files
- [ ] Identify gaps: TensorRT deep dive missing

**Step 1: Web Research - TensorRT Basics**
- [ ] Search: "TensorRT tutorial 2024 PyTorch"
- [ ] Search: "TensorRT optimization techniques"
- [ ] Search: "TensorRT vs ONNX Runtime comparison"
- [ ] Scrape top 3 results

**Step 2: Web Research - TensorRT Implementation**
- [ ] Search: "site:github.com TensorRT PyTorch example"
- [ ] Search: "TensorRT graph optimization patterns"
- [ ] Scrape NVIDIA TensorRT docs

**Step 3: Create Knowledge File**
- [ ] Create karpathy/inference-optimization/00-tensorrt-fundamentals.md
- [ ] Section 1: TensorRT overview (~100 lines)
- [ ] Section 2: Graph optimization strategies (~150 lines)
- [ ] Section 3: Kernel fusion patterns (~100 lines)
- [ ] Section 4: PyTorch to TensorRT workflow (~100 lines)
- [ ] Section 5: Performance benchmarks (~50 lines)
- [ ] Cite all sources

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-tensorrt-fundamentals-2025-11-13-[TIME].md

---

### PART 6: Create karpathy/inference-optimization/01-tensorrt-vlm-deployment.md (~450 lines)

- [✓] PART 6: Create karpathy/inference-optimization/01-tensorrt-vlm-deployment.md (FILE EXISTS - 35.7 KB)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md for VLM deployment mentions
- [ ] Grep for "VLM" AND "deployment" in karpathy/
- [ ] Check existing VLM inference files
- [ ] Identify gaps: TensorRT for VLMs specifically

**Step 1: Web Research - VLM TensorRT**
- [ ] Search: "TensorRT vision language model deployment"
- [ ] Search: "TensorRT multimodal model optimization"
- [ ] Search: "CLIP TensorRT optimization"
- [ ] Scrape top 3 results

**Step 2: Web Research - Practical Examples**
- [ ] Search: "site:github.com TensorRT VLM example"
- [ ] Search: "TensorRT dynamic shapes multimodal"
- [ ] Scrape examples

**Step 3: Create Knowledge File**
- [ ] Create karpathy/inference-optimization/01-tensorrt-vlm-deployment.md
- [ ] Section 1: VLM inference challenges (~80 lines)
- [ ] Section 2: TensorRT for vision encoders (~120 lines)
- [ ] Section 3: TensorRT for language decoders (~120 lines)
- [ ] Section 4: Dynamic batching strategies (~80 lines)
- [ ] Section 5: arr-coc-0-1 deployment (~50 lines)
- [ ] Cite all sources

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-tensorrt-vlm-2025-11-13-[TIME].md

---

### PART 7: Create karpathy/inference-optimization/02-triton-inference-server.md (~500 lines)

- [✓] PART 7: Create karpathy/inference-optimization/02-triton-inference-server.md (FILE EXISTS - 29.9 KB)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md for Triton mentions
- [ ] Grep for "Triton" in karpathy/ (we have vLLM mentions)
- [ ] Check model serving knowledge
- [ ] Identify gaps: Triton Inference Server specifics

**Step 1: Web Research - Triton Server**
- [ ] Search: "NVIDIA Triton Inference Server tutorial"
- [ ] Search: "Triton dynamic batching explained"
- [ ] Search: "Triton vs TorchServe comparison"
- [ ] Scrape top 3 results

**Step 2: Web Research - Production Patterns**
- [ ] Search: "site:github.com Triton server example"
- [ ] Search: "Triton ensemble models multimodal"
- [ ] Scrape NVIDIA docs

**Step 3: Create Knowledge File**
- [ ] Create karpathy/inference-optimization/02-triton-inference-server.md
- [ ] Section 1: Triton architecture (~100 lines)
- [ ] Section 2: Dynamic batching (~120 lines)
- [ ] Section 3: Model ensemble patterns (~120 lines)
- [ ] Section 4: Deployment workflows (~100 lines)
- [ ] Section 5: VLM serving patterns (~60 lines)
- [ ] Cite all sources

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-triton-server-2025-11-13-[TIME].md

---

### PART 8: Create karpathy/inference-optimization/03-torch-compile-aot-inductor.md (~400 lines)

- [✓] PART 8: Create karpathy/inference-optimization/03-torch-compile-aot-inductor.md (Completed 2025-11-13)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md for torch.compile mentions
- [ ] Grep for "torch.compile" OR "inductor" in karpathy/
- [ ] Check existing PyTorch JIT files
- [ ] Identify gaps: AOT Inductor deep dive

**Step 1: Web Research - torch.compile**
- [ ] Search: "PyTorch torch.compile tutorial 2024"
- [ ] Search: "TorchDynamo vs TorchScript comparison"
- [ ] Search: "AOT Inductor deployment patterns"
- [ ] Scrape top 3 results

**Step 2: Web Research - Implementation**
- [ ] Search: "site:github.com torch.compile example"
- [ ] Search: "torch.compile performance benchmarks"
- [ ] Scrape PyTorch docs

**Step 3: Create Knowledge File**
- [ ] Create karpathy/inference-optimization/03-torch-compile-aot-inductor.md
- [ ] Section 1: torch.compile overview (~80 lines)
- [ ] Section 2: TorchDynamo capture (~100 lines)
- [ ] Section 3: TorchInductor optimization (~100 lines)
- [ ] Section 4: AOT compilation workflow (~80 lines)
- [ ] Section 5: VLM optimization cases (~40 lines)
- [ ] Cite all sources

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-torch-compile-2025-11-13-[TIME].md

---

## EXPANSION 20: Container Orchestration for ML

### PART 9: Create karpathy/orchestration/00-kubernetes-gpu-scheduling.md (~500 lines)

- [✓] PART 9: Create karpathy/orchestration/00-kubernetes-gpu-scheduling.md (Completed 2025-11-13 22:58)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md for Kubernetes mentions
- [ ] Grep for "Kubernetes" OR "k8s" in karpathy/
- [ ] Check existing container orchestration files
- [ ] Identify gaps: K8s GPU scheduling specifics

**Step 1: Web Research - K8s GPU**
- [ ] Search: "Kubernetes GPU scheduling tutorial"
- [ ] Search: "NVIDIA GPU Operator Kubernetes"
- [ ] Search: "Kubernetes node affinity GPU workloads"
- [ ] Scrape top 3 results

**Step 2: Web Research - ML Patterns**
- [ ] Search: "site:github.com Kubernetes GPU example ML"
- [ ] Search: "Kubernetes resource quotas GPU"
- [ ] Scrape K8s docs

**Step 3: Create Knowledge File**
- [ ] Create karpathy/orchestration/00-kubernetes-gpu-scheduling.md
- [ ] Section 1: K8s GPU fundamentals (~100 lines)
- [ ] Section 2: GPU Operator installation (~100 lines)
- [ ] Section 3: Resource quotas & limits (~120 lines)
- [ ] Section 4: Node affinity patterns (~100 lines)
- [ ] Section 5: Multi-GPU training jobs (~80 lines)
- [ ] Cite all sources

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-k8s-gpu-2025-11-13-[TIME].md

---

### PART 10: Create karpathy/orchestration/01-kubeflow-ml-pipelines.md (~450 lines)

- [✓] PART 10: Create karpathy/orchestration/01-kubeflow-ml-pipelines.md (FILE EXISTS - 32.7 KB)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md for Kubeflow mentions
- [ ] Grep for "Kubeflow" OR "MLOps" in karpathy/
- [ ] Check existing pipeline files
- [ ] Identify gaps: Kubeflow specifics

**Step 1: Web Research - Kubeflow**
- [ ] Search: "Kubeflow Pipelines tutorial 2024"
- [ ] Search: "Kubeflow vs MLflow comparison"
- [ ] Search: "Kubeflow training operators"
- [ ] Scrape top 3 results

**Step 2: Web Research - Implementation**
- [ ] Search: "site:github.com Kubeflow pipeline example"
- [ ] Search: "Kubeflow distributed training PyTorch"
- [ ] Scrape Kubeflow docs

**Step 3: Create Knowledge File**
- [ ] Create karpathy/orchestration/01-kubeflow-ml-pipelines.md
- [ ] Section 1: Kubeflow overview (~80 lines)
- [ ] Section 2: Pipeline components (~120 lines)
- [ ] Section 3: Training operators (TFJob, PyTorchJob) (~120 lines)
- [ ] Section 4: Hyperparameter tuning (Katib) (~80 lines)
- [ ] Section 5: VLM training pipelines (~50 lines)
- [ ] Cite all sources

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-kubeflow-2025-11-13-[TIME].md

---

### PART 11: Create karpathy/orchestration/02-ray-distributed-ml.md (~500 lines)

- [✓] PART 11: Create karpathy/orchestration/02-ray-distributed-ml.md (Completed 2025-11-13)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md for Ray mentions
- [ ] Grep for "Ray" in karpathy/
- [ ] Check distributed computing files
- [ ] Identify gaps: Ray for ML specifics

**Step 1: Web Research - Ray**
- [ ] Search: "Ray distributed ML tutorial"
- [ ] Search: "Ray Train PyTorch example"
- [ ] Search: "Ray vs Dask comparison ML"
- [ ] Scrape top 3 results

**Step 2: Web Research - Ray AIR**
- [ ] Search: "site:github.com Ray AIR example"
- [ ] Search: "Ray Tune hyperparameter optimization"
- [ ] Scrape Ray docs

**Step 3: Create Knowledge File**
- [ ] Create karpathy/orchestration/02-ray-distributed-ml.md
- [ ] Section 1: Ray fundamentals (~100 lines)
- [ ] Section 2: Ray Train for distributed training (~150 lines)
- [ ] Section 3: Ray Tune for HPO (~120 lines)
- [ ] Section 4: Ray Serve for inference (~80 lines)
- [ ] Section 5: VLM workflows with Ray (~50 lines)
- [ ] Cite all sources

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-ray-ml-2025-11-13-[TIME].md

---

### PART 12: Create karpathy/orchestration/03-ml-workload-patterns-k8s.md (~400 lines)

- [✓] PART 12: Create karpathy/orchestration/03-ml-workload-patterns-k8s.md (Completed 2025-11-13 23:15)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md for workload patterns
- [ ] Grep for "batch" AND "job" in karpathy/
- [ ] Check existing K8s files
- [ ] Identify gaps: ML-specific patterns on K8s

**Step 1: Web Research - ML Workloads**
- [ ] Search: "Kubernetes ML workload patterns"
- [ ] Search: "K8s CronJob training automation"
- [ ] Search: "Kubernetes gang scheduling ML"
- [ ] Scrape top 3 results

**Step 2: Web Research - Best Practices**
- [ ] Search: "site:github.com Kubernetes ML job example"
- [ ] Search: "Volcano scheduler Kubernetes ML"
- [ ] Scrape K8s ML patterns

**Step 3: Create Knowledge File**
- [ ] Create karpathy/orchestration/03-ml-workload-patterns-k8s.md
- [ ] Section 1: Batch jobs vs CronJobs (~80 lines)
- [ ] Section 2: Gang scheduling for multi-GPU (~100 lines)
- [ ] Section 3: Job queues & priorities (~100 lines)
- [ ] Section 4: Resource management patterns (~80 lines)
- [ ] Section 5: arr-coc-0-1 on K8s (~40 lines)
- [ ] Cite all sources

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-ml-workloads-k8s-2025-11-13-[TIME].md

---

## EXPANSION 21: Alternative Hardware for ML

### PART 13: Create karpathy/alternative-hardware/00-amd-rocm-ml.md (~500 lines)

- [✓] PART 13: Create karpathy/alternative-hardware/00-amd-rocm-ml.md (Completed 2025-11-14 02:20)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md for AMD/ROCm mentions
- [ ] Grep for "AMD" OR "ROCm" in karpathy/
- [ ] Check existing GPU architecture files
- [ ] Identify gaps: ROCm for ML specifics

**Step 1: Web Research - ROCm**
- [ ] Search: "AMD ROCm PyTorch tutorial 2024"
- [ ] Search: "ROCm vs CUDA comparison ML"
- [ ] Search: "MI300X GPU machine learning"
- [ ] Scrape top 3 results

**Step 2: Web Research - Implementation**
- [ ] Search: "site:github.com ROCm PyTorch example"
- [ ] Search: "ROCm installation Ubuntu 22.04"
- [ ] Scrape AMD ROCm docs

**Step 3: Create Knowledge File**
- [ ] Create karpathy/alternative-hardware/00-amd-rocm-ml.md
- [ ] Section 1: ROCm overview (~100 lines)
- [ ] Section 2: PyTorch on ROCm (~150 lines)
- [ ] Section 3: MI300X architecture (~100 lines)
- [ ] Section 4: CUDA to ROCm porting (~100 lines)
- [ ] Section 5: Performance comparison (~50 lines)
- [ ] Cite all sources

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-amd-rocm-2025-11-13-[TIME].md

---

### PART 14: Create karpathy/alternative-hardware/01-apple-metal-ml.md (~450 lines)

- [✓] PART 14: Create karpathy/alternative-hardware/01-apple-metal-ml.md (Completed 2025-11-14 03:05)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md for Apple/Metal mentions
- [ ] Grep for "Apple" OR "Metal" OR "M-series" in karpathy/
- [ ] Check existing hardware files
- [ ] Identify gaps: Metal for ML deep dive

**Step 1: Web Research - Metal**
- [ ] Search: "Apple Metal machine learning tutorial"
- [ ] Search: "PyTorch MPS backend M4"
- [ ] Search: "CoreML vs Metal Performance Shaders"
- [ ] Scrape top 3 results

**Step 2: Web Research - Implementation**
- [ ] Search: "site:github.com Metal ML example"
- [ ] Search: "M4 Neural Engine specifications"
- [ ] Scrape Apple ML docs

**Step 3: Create Knowledge File**
- [ ] Create karpathy/alternative-hardware/01-apple-metal-ml.md
- [ ] Section 1: Metal overview (~80 lines)
- [ ] Section 2: PyTorch MPS backend (~120 lines)
- [ ] Section 3: CoreML integration (~120 lines)
- [ ] Section 4: M-series performance (~80 lines)
- [ ] Section 5: VLM on Apple Silicon (~50 lines)
- [ ] Cite all sources

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-apple-metal-2025-11-13-[TIME].md

---

### PART 15: Create karpathy/alternative-hardware/02-intel-oneapi-ml.md (~500 lines)

- [✓] PART 15: Create karpathy/alternative-hardware/02-intel-oneapi-ml.md (Completed 2025-11-14 02:22)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md for Intel mentions
- [ ] Grep for "Intel" OR "oneAPI" in karpathy/
- [ ] Check existing hardware files
- [ ] Identify gaps: Intel oneAPI for ML

**Step 1: Web Research - oneAPI**
- [ ] Search: "Intel oneAPI machine learning tutorial"
- [ ] Search: "Intel Data Center GPU Max ML"
- [ ] Search: "oneAPI vs CUDA comparison"
- [ ] Scrape top 3 results

**Step 2: Web Research - Implementation**
- [ ] Search: "site:github.com oneAPI PyTorch example"
- [ ] Search: "Intel Extension for PyTorch"
- [ ] Scrape Intel oneAPI docs

**Step 3: Create Knowledge File**
- [ ] Create karpathy/alternative-hardware/02-intel-oneapi-ml.md
- [ ] Section 1: oneAPI overview (~100 lines)
- [ ] Section 2: Intel Extension for PyTorch (~150 lines)
- [ ] Section 3: Data Center GPU Max (~100 lines)
- [ ] Section 4: SYCL programming model (~100 lines)
- [ ] Section 5: Performance benchmarks (~50 lines)
- [ ] Cite all sources

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-intel-oneapi-2025-11-13-[TIME].md

---

### PART 16: Create karpathy/alternative-hardware/03-tpu-programming-fundamentals.md (~400 lines)

- [✓] PART 16: Create karpathy/alternative-hardware/03-tpu-programming-fundamentals.md (Completed 2025-11-13 23:45)

**Step 0: Check Existing Knowledge**
- [ ] Read INDEX.md for TPU mentions
- [ ] Grep for "TPU" in karpathy/
- [ ] Check existing GCP files (we have Vertex AI knowledge)
- [ ] Identify gaps: TPU programming specifics

**Step 1: Web Research - TPU**
- [ ] Search: "Google TPU programming tutorial 2024"
- [ ] Search: "JAX on TPU example"
- [ ] Search: "TPU v4 vs v5 comparison"
- [ ] Scrape top 3 results

**Step 2: Web Research - Implementation**
- [ ] Search: "site:github.com TPU JAX example"
- [ ] Search: "PyTorch XLA TPU tutorial"
- [ ] Scrape Google Cloud TPU docs

**Step 3: Create Knowledge File**
- [ ] Create karpathy/alternative-hardware/03-tpu-programming-fundamentals.md
- [ ] Section 1: TPU architecture (~80 lines)
- [ ] Section 2: JAX on TPU (~100 lines)
- [ ] Section 3: PyTorch XLA on TPU (~100 lines)
- [ ] Section 4: TPU pod slices (~80 lines)
- [ ] Section 5: When to use TPUs (~40 lines)
- [ ] Cite all sources

**Step 4: Create KNOWLEDGE DROP**
- [ ] Create KNOWLEDGE-DROP-tpu-programming-2025-11-13-[TIME].md

---

## Summary

**Total PARTs**: 16
**Expected Files**: 16 knowledge files
**Expected Lines**: ~7,600 lines
**Web Research**: Required for all PARTs (Bright Data)
**Knowledge Gaps Filled**: Multi-GPU distributed, inference optimization, orchestration, alternative hardware

**All PARTs follow format**:
1. Check existing knowledge (avoid duplication)
2. Web research (search queries specified)
3. Create knowledge file with citations
4. Create KNOWLEDGE DROP summary

**All runners execute in parallel** - maximum speed!
