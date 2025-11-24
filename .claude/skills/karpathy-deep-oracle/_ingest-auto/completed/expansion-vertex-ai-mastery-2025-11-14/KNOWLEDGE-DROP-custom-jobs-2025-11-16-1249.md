# KNOWLEDGE DROP: Vertex AI Custom Jobs Advanced

**Date**: 2025-11-16 12:49
**PART**: 1 of 24
**Batch**: 1 (Core Infrastructure Part 1)
**File Created**: `gcp-vertex/00-custom-jobs-advanced.md`

---

## What Was Created

**New File**: `gcp-vertex/00-custom-jobs-advanced.md` (701 lines)

**Content Sections**:
1. WorkerPoolSpec Architecture (120 lines) - Chief worker, additional workers, parameter servers, Reduction Server
2. Network Configuration (140 lines) - VPC, VPC Peering, Shared VPC, Private Service Connect
3. Persistent Disk Attachment (100 lines) - Disk types, checkpoint strategy, GCS syncing
4. Preemptible Worker Handling (120 lines) - Auto-restart, gang scheduling, recovery patterns
5. Environment Variables (100 lines) - TF_CONFIG, MASTER_ADDR, RANK, WORLD_SIZE, custom vars
6. arr-coc-0-1 Multi-Worker Example (120 lines) - 4-node training configuration, DeepSpeed ZeRO-2

---

## Sources Used

**Existing Knowledge Files Read**:
- `karpathy/distributed-training/00-deepspeed-zero-optimizer.md` (ZeRO memory optimization)
- `karpathy/distributed-training/01-deepspeed-pipeline-parallelism.md` (Pipeline patterns)
- `karpathy/distributed-training/03-fsdp-vs-deepspeed.md` (Framework comparison)
- `karpathy/vertex-ai-production/00-multi-gpu-distributed-training.md` (Vertex AI distributed training)

**Web Research Performed**:
- "Vertex AI Custom Jobs WorkerPoolSpec 2024"
- "Vertex AI multi-worker distributed training"
- "Vertex AI preemptible workers checkpoint resume"
- "Vertex AI VPC network configuration"

**Official Documentation Referenced**:
- Vertex AI Distributed Training (cloud.google.com)
- CustomJobSpec API Reference (cloud.google.com)
- VPC Network Peering (cloud.google.com)
- Private Service Connect (cloud.google.com)
- Google Codelabs: Multi-Worker Training

---

## Key Topics Covered

### WorkerPoolSpec Architecture
- Worker pool 0 (chief) vs worker pool 1+ (additional workers)
- Parameter servers for async training (legacy pattern)
- Reduction Server for 2× bandwidth improvement on multi-node
- 4 worker pools max per Custom Job

### Network Configuration
- Custom VPC vs Default VPC
- VPC Peering setup (private connectivity)
- Shared VPC for multi-project organizations
- Private Service Connect (modern alternative to VPC Peering)
- IP range planning (/16 for peering, /28 for PSC)

### Persistent Disk & Checkpointing
- Disk types: pd-standard, pd-balanced, pd-ssd, pd-extreme
- Automatic mounting at /mnt/disks/training_data
- Checkpoint save frequency (500 steps local, 1000 steps GCS)
- Cross-region durability via Cloud Storage sync

### Preemptible Workers
- 60-80% cost savings
- restartJobOnWorkerRestart: true for automatic recovery
- Gang scheduling (all workers restart together)
- Checkpoint resume from persistent disk
- ZeRO-2 recommended for faster recovery

### Environment Variables
- CLUSTER_SPEC, TF_CONFIG (TensorFlow-specific)
- WORLD_SIZE, RANK, LOCAL_RANK (PyTorch/DeepSpeed)
- MASTER_ADDR, MASTER_PORT (NCCL communication)
- Custom environment variables via containerSpec.env

### arr-coc-0-1 Implementation
- 4 nodes × 8 A100 = 32 GPUs configuration
- DeepSpeed ZeRO-2 for 95% throughput, 40% memory reduction
- Model: Qwen3-VL-2B + 13-channel texture arrays + relevance scorers
- 8 GB per GPU with ZeRO-2 (29 GB without distribution)
- Persistent disk checkpointing + GCS sync

---

## Statistics

- **Total Lines**: 701
- **Sections**: 6
- **Code Examples**: 15
- **Configuration Examples**: 8 (YAML/JSON/Python)
- **Source Citations**: 14 (4 existing knowledge files + 10 web sources)
- **Cross-References**: 8 specific line citations from existing knowledge

---

## Quality Checklist

- ✅ All sections cite sources
- ✅ Web research links included with access dates
- ✅ Source documents referenced with line numbers
- ✅ Code examples are complete and runnable
- ✅ arr-coc-0-1 specific implementation included
- ✅ Cross-references to distributed-training knowledge files
- ✅ Technical accuracy verified against official docs
- ✅ Practical examples (not just theory)

---

## Next Steps

**For Oracle** (after batch 1 complete):
- Update INDEX.md with new gcp-vertex/ folder
- Verify cross-references are correct
- Check consistency with vertex-ai-production/ folder

**For Future Expansions**:
- PART 2: Pipelines & Kubeflow integration
- PART 3: Training-to-serving automation
- PART 4: Batch prediction & Feature Store
