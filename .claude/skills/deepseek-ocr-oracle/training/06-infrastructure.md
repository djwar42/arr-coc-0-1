# Training Infrastructure

**See**: `RESEARCH/DeepSeekOCR/TRAINING.md` lines 1000-1200

## Hardware

**20 nodes × 8 A100-40G = 160 GPUs**

Per node:
- 8× A100-40GB GPUs
- 1TB RAM
- NVLink for fast inter-GPU
- InfiniBand for inter-node

## HAI-LLM Platform

DeepSeek's custom training platform:
- Distributed training orchestration
- Fault tolerance (auto-restart)
- Monitoring dashboards
- Resource management

## Storage

**Data**: 50TB NVMe SSD
- Fast random access
- Parallel data loading

**Checkpoints**: 10TB
- Save every 5000 steps
- Keep best 3 by validation loss

## Network

InfiniBand (200 Gbps):
- Fast gradient synchronization
- All-reduce operations
- Pipeline stage communication

## Cost Estimate

- **Compute**: $2/hour per A100
- **Duration**: 17 days × 24 hours = 408 hours
- **GPUs**: 160
- **Total**: 408 × 160 × $2 = **~$130k**

Plus data storage, engineering → **~$260k total**

**See TRAINING.md** for complete infrastructure setup!
