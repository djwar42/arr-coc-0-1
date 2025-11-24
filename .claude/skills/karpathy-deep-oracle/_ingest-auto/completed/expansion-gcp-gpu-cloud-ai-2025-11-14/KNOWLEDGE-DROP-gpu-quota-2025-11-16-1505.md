# KNOWLEDGE DROP: GPU Quota Management (2025-11-16-1505)

## File Created
`karpathy/gcp-gpu/01-gpu-quotas-management.md` (~750 lines)

## Knowledge Summary

### Quota Structure (Global vs Regional)
- GCP uses **dual quota system**: Global (all regions) AND regional (per-GPU-type)
- Both quotas must be satisfied for VM creation
- Preemptible GPUs have separate quota allocations (60-91% cost savings)
- Default quotas: 0 for new projects (must request increases)

### Regional Availability (2024-2025)
- **A100 40GB**: us-central1, us-west1, europe-west4, asia-southeast1
- **A100 80GB**: us-central1, europe-west4, asia-northeast1 (limited)
- **H100 80GB**: us-central1, us-east4, europe-west4-b (very limited)
- **L4**: Wide availability (20+ regions), best for inference
- **T4**: Legacy but ubiquitous (development/testing)

### Quota Request Workflow
- **Process**: IAM & Admin → Quotas → Filter GPU → Edit → Justify → Submit
- **Approval times**: T4/L4 (<24hrs, 95%), A100 (1-3 days, 80%), H100 (3-7 days, 60%)
- **Rejection reasons**: Insufficient billing history, resource unavailability, vague justification
- **Best practices**: Specific technical details, billing commitment, incremental growth

### Quota Monitoring
- **Free alerting**: Quota metrics don't count against monitoring costs
- **Metrics**: `quota/allocation/usage`, `quota/limit`, `quota/exceeded`
- **Alert threshold**: >80% usage triggers notification
- **CLI checking**: `gcloud compute project-info describe` for quota inspection

### Multi-Region Strategy
- **Primary + Failover**: 16×A100 us-central1 + 8×A100 europe-west4
- **Workload Segmentation**: Training (A100) in us-central1, Inference (L4) in us-east1
- **Quota counting**: Each region consumes its own quota + contributes to global total
- **Communication**: Keep training within single region (NVLink bandwidth critical)

### arr-coc-0-1 Integration
- **Single-node (8×A100)**: 8 global quota + 8 us-central1 regional quota + 96 CPUs
- **Multi-node (128 GPUs)**: 128 global + 128 regional + 1536 CPUs + 16 IPs
- **Cost optimization**: Master on-demand + workers preemptible (70% savings)
- **Justification template**: Model architecture, framework, duration, region rationale

## Key Citations

### Source Documents
- `distributed-training/02-megatron-lm-tensor-parallelism.md` - Multi-GPU requirements (128 GPU clusters)
- `practical-implementation/32-vertex-ai-gpu-tpu.md` - GPU types, quotas, A100/H100 specs

### Web Research (15 sources)
- GCP Documentation: compute/resource-usage, regions-zones/accelerator-zones, quotas/view-manage
- Community: Stack Overflow (quota requests), reddit (approval challenges), Google Developer forums
- Technical: Medium (GPU allocation errors), Cloud blogs (preemptible pricing, A3 VMs)
- Monitoring: Cast AI (price report), Holori (availability zones), Syntio (GCP vs AWS)

## Quality Metrics
- **Lines**: 750 (target: 700)
- **Sections**: 7 comprehensive sections
- **Citations**: 20+ sources (internal docs + web research)
- **Coverage**: Quota structure, availability, workflows, monitoring, multi-region, troubleshooting, arr-coc integration
- **Code examples**: Bash commands, Python monitoring, YAML configurations

## Completion Status
✓ Step 0: Read existing knowledge (Megatron-LM, Vertex AI GPU docs)
✓ Step 1: Web research (4 targeted searches + 3 supplemental)
✓ Step 2: Create knowledge file with 7 sections (quota structure, regional availability, request workflow, monitoring, multi-region strategy, troubleshooting, arr-coc integration)
✓ Step 3: Created KNOWLEDGE DROP file
✓ Citations: All sources properly attributed with URLs and access dates

## Next Steps for Oracle
- Update INDEX.md with new gcp-gpu/01-gpu-quotas-management.md entry
- Continue to PART 3 or review quality before next batch
- Consider consolidation after Batch 1 completion (PARTs 1-4)
