# GCP GPU & TPU - Index

**Total**: 15+ files covering GPU/TPU usage on Google Cloud

## Overview

Google Cloud Platform GPU and TPU knowledge:
- Driver and CUDA management
- Multi-GPU/TPU training
- GKE GPU clusters
- Performance optimization

## Files

### GPU Fundamentals
| File | Description |
|------|-------------|
| `02-nvidia-driver-cuda-management.md` | Driver setup, CUDA versions |
| `03-storage-optimization-gpu-training.md` | Storage for training |
| `04-multi-gpu-training-patterns.md` | DDP, FSDP patterns |
| `05-multi-node-distributed-training.md` | Multi-node training |

### GKE & Kubernetes
| File | Description |
|------|-------------|
| `09-gke-training-operators-kubeflow.md` | Kubeflow, operators |
| `11-gke-autopilot-ml-optimization.md` | GKE Autopilot for ML |

### TPU
| File | Description |
|------|-------------|
| `12-cloud-tpu-architecture-programming.md` | TPU architecture |
| `13-tpu-multi-host-distributed.md` | Multi-host TPU |
| `14-tpu-performance-optimization.md` | TPU optimization |
| `15-gpu-vs-tpu-decision-framework.md` | GPU vs TPU selection |

### Operations & Production
| File | Description |
|------|-------------|
| `17-gpu-monitoring-observability.md` | Monitoring GPUs |
| `18-gpu-quotas-governance-policies.md` | Quotas, governance |
| `19-gpu-benchmarking-performance-testing.md` | Benchmarking |
| `21-gpu-cicd-automation-pipelines.md` | CI/CD for GPU |
| `22-gpu-security-compliance.md` | Security |

## Cross-References

- Vertex AI: `../karpathy/practical-implementation/30-vertex-ai-fundamentals.md`
- Hardware implementations: `../implementations/`
- GCP general: `../gcloud-*/`
