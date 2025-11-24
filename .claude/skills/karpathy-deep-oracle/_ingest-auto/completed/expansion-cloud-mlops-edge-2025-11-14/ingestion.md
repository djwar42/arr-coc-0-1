# Knowledge Expansion: Cloud Platforms + MLOps + Edge (4 runners)

**Date**: 2025-11-14
**Target**: AWS SageMaker, Azure ML, MLOps Production, Edge Deployment
**Strategy**: Apply our 16 files to new platforms & deployment scenarios

---

## PART 1: AWS SageMaker Advanced (~700 lines)

- [✓] PART 1: Create karpathy/aws-sagemaker/00-distributed-inference-optimization.md (Completed 2025-11-14 03:00)

**Step 0: Check Existing Knowledge**
- [✓] Read distributed-training/00-deepspeed-zero-optimizer.md
- [✓] Read distributed-training/03-fsdp-vs-deepspeed.md
- [✓] Read inference-optimization/00-tensorrt-fundamentals.md
- [✓] Read inference-optimization/02-triton-inference-server.md

**Step 1: Web Research**
- [✓] Search: "AWS SageMaker distributed training PyTorch FSDP 2024"
- [✓] Search: "SageMaker model parallelism library DeepSpeed"
- [✓] Search: "SageMaker Inference TensorRT optimization"
- [✓] Search: "SageMaker multi-model endpoints Triton"

**Step 2: Create Knowledge File**
- [✓] Section 1: SageMaker distributed training (FSDP, DeepSpeed, model parallel library)
- [✓] Section 2: SageMaker Inference optimization (TensorRT, multi-model endpoints, serverless)
- [✓] Section 3: Cost optimization (Spot instances, Savings Plans, inference autoscaling)
- [✓] Section 4: arr-coc-0-1 on SageMaker (training + serving)
- [✓] **CITE**: distributed-training/00, 03; inference-optimization/00, 02

**Step 3: Create KNOWLEDGE DROP**
- [✓] Create KNOWLEDGE-DROP-aws-sagemaker-2025-11-14-0300.md

---

## PART 2: Azure ML Integration (~700 lines)

- [✓] PART 2: Create karpathy/azure-ml/00-distributed-training-aks-serving.md (Completed 2025-11-14 03:00)

**Step 0: Check Existing Knowledge**
- [✓] Read distributed-training/02-megatron-lm-tensor-parallelism.md
- [✓] Read inference-optimization/02-triton-inference-server.md
- [✓] Read orchestration/00-kubernetes-gpu-scheduling.md
- [✓] Read orchestration/01-kubeflow-ml-pipelines.md

**Step 1: Web Research**
- [✓] Search: "Azure ML distributed training PyTorch 2024"
- [✓] Search: "Azure AKS GPU scheduling ML workloads"
- [✓] Search: "Azure ML managed endpoints Triton"
- [✓] Search: "Azure ML Kubeflow integration"

**Step 2: Create Knowledge File**
- [✓] Section 1: Azure ML Compute distributed training (AmlCompute, AKS clusters)
- [✓] Section 2: AKS ML workloads (GPU scheduling, Kubeflow on AKS)
- [✓] Section 3: Azure ML endpoints (managed, Triton integration, autoscaling)
- [✓] Section 4: Cost optimization (Spot VMs, reserved capacity)
- [✓] **CITE**: distributed-training/02; inference-optimization/02; orchestration/00, 01

**Step 3: Create KNOWLEDGE DROP**
- [✓] Create KNOWLEDGE-DROP-azure-ml-2025-11-14-0300.md

---

## PART 3: MLOps Production Patterns (~800 lines)

- [✓] PART 3: Create karpathy/mlops-production/00-monitoring-cicd-cost-optimization.md (Completed 2025-11-14 03:01)

**Step 0: Check Existing Knowledge**
- [✓] Read orchestration/03-ml-workload-patterns-k8s.md
- [✓] Read all 4 distributed-training files (cost understanding)
- [✓] Read all 4 inference-optimization files (serving costs)
- [✓] Read all 4 alternative-hardware files (hardware cost comparison)

**Step 1: Web Research**
- [✓] Search: "MLOps production monitoring best practices 2024"
- [✓] Search: "data drift detection Evidently AI 2024"
- [✓] Search: "ML CI/CD GitHub Actions Vertex AI SageMaker"
- [✓] Search: "multi-cloud ML cost optimization"

**Step 2: Create Knowledge File**
- [✓] Section 1: Production monitoring (drift detection, model performance, Prometheus/Grafana)
- [✓] Section 2: CI/CD automation (GitHub Actions, automated retraining, deployment gates)
- [✓] Section 3: Multi-cloud cost optimization (GCP/AWS/Azure comparison, spot instances, reserved)
- [✓] Section 4: Observability (W&B, MLflow, model registry, lineage tracking)
- [✓] **CITE**: All 16 files for cost analysis across training/inference/hardware

**Step 3: Create KNOWLEDGE DROP**
- [✓] Create KNOWLEDGE-DROP-mlops-production-2025-11-14-0301.md

---

## PART 4: Edge & Mobile Deployment (~700 lines)

- [✓] PART 4: Create karpathy/edge-deployment/00-tensorrt-coreml-mobile-optimization.md (Completed 2025-11-14 03:01)

**Step 0: Check Existing Knowledge**
- [✓] Read inference-optimization/00-tensorrt-fundamentals.md
- [✓] Read inference-optimization/01-tensorrt-vlm-deployment.md
- [✓] Read alternative-hardware/01-apple-metal-ml.md
- [✓] Read alternative-hardware/02-intel-oneapi-ml.md (for Arc GPUs)

**Step 1: Web Research**
- [✓] Search: "TensorRT Jetson edge deployment 2024"
- [✓] Search: "CoreML iOS VLM deployment"
- [✓] Search: "ONNX Runtime Mobile Android optimization"
- [✓] Search: "edge AI inference optimization latency <50ms"

**Step 2: Create Knowledge File**
- [✓] Section 1: TensorRT on edge (Jetson Orin, INT8 quantization, <50ms latency)
- [✓] Section 2: Apple CoreML deployment (M4 iPad, iPhone, Neural Engine optimization)
- [✓] Section 3: Android optimization (NNAPI, GPU delegates, Snapdragon NPU)
- [✓] Section 4: Cross-platform (ONNX Runtime, TFLite, model compression)
- [✓] Section 5: arr-coc-0-1 edge deployment (mobile VLM feasibility, power constraints)
- [✓] **CITE**: inference-optimization/00, 01; alternative-hardware/01, 02

**Step 3: Create KNOWLEDGE DROP**
- [✓] Create KNOWLEDGE-DROP-edge-deployment-2025-11-14-0301.md

---

## Summary

**Total**: 4 runners (~2,900 lines)
**New folders**: aws-sagemaker/, azure-ml/, mlops-production/, edge-deployment/
**Integration**: References 16 influential files across all 4 PARTs
**Parallel execution**: Launch all 4 simultaneously
