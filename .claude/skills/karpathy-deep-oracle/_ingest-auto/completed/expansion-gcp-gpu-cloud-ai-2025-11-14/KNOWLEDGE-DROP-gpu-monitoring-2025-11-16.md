# KNOWLEDGE DROP: GPU Monitoring & Observability

**Runner**: PART 18
**Date**: 2025-11-16
**File Created**: `gcp-gpu/17-gpu-monitoring-observability.md`
**Line Count**: ~720 lines

---

## What Was Created

Comprehensive GPU monitoring and observability guide covering:

**8 Sections**:
1. Cloud Monitoring GPU Metrics (native GCP integration)
2. DCGM Integration (NVIDIA Data Center GPU Manager)
3. nvidia-smi Automation & Monitoring
4. Prometheus + Grafana GPU Dashboards
5. Alerting Policies (Cloud Monitoring + Prometheus Alertmanager)
6. GPU Memory Leak Detection (Compute Sanitizer, PyTorch profiling)
7. Cost Monitoring Per GPU (CUDs, labels, BigQuery)
8. arr-coc-0-1 Monitoring Setup (production implementation)

---

## Key Knowledge Acquired

### Cloud Monitoring Native Support
- **Auto-collected metrics**: GPU utilization, memory utilization, temperature
- **1-minute granularity**, no extra cost for metric ingestion
- **Alerting pricing starts January 7, 2025** (previously free)
- **Limitations**: Basic metrics only, no per-process breakdown

### DCGM (Data Center GPU Manager)
- **NVIDIA's enterprise monitoring suite** for datacenter GPUs
- **DCGM versions 3.1-3.3.9** compatible with GCP Ops Agent
- **Auto-dashboard**: Pre-built dashboard automatically added when DCGM metrics start flowing
- **Advanced metrics**: SM occupancy, Tensor Core active %, PCIe throughput, ECC errors
- **Two deployment modes**:
  1. Ops Agent integration → Cloud Monitoring (managed)
  2. DCGM Exporter → Prometheus (self-managed, more control)

### GPU Pricing & Cost Tracking (2024-2025)
**On-demand** (us-central1):
- A100 80GB: $3.673/hour ($2,656/month)
- H100 80GB: $5.505/hour ($3,984/month)
- L4: $0.888/hour ($642/month)

**Committed Use Discounts (CUDs)**:
- 1-year: 37% savings
- 3-year: 57% savings
- Example: A100 1-year CUD = $2.31/hour (save $985/month)

**Spot pricing**: 60-91% discount (but preemptible)
- Spot A100: $1.10/hour (70% discount)

### Memory Leak Detection Patterns
1. **Gradual memory growth** over time (PromQL: `deriv(DCGM_FI_DEV_FB_USED[1h]) > 0`)
2. **Memory not released** between training epochs
3. **NVIDIA Compute Sanitizer** for CUDA leak detection
4. **PyTorch profiling** with `torch.cuda.memory_allocated()` tracking
5. **Common causes**: Holding references to loss/outputs, not calling `optimizer.zero_grad()`, circular references

### Alerting Best Practices
**Critical alerts**:
- GPU overheating (>85°C) - risk of thermal throttling
- Memory exhaustion (>95%) - OOM crash imminent
- ECC errors - hardware failure likely
- Training stalled (no batch completions for 15+ minutes)

**Cost alerts**:
- GPU idle (< 10% utilization for 1 hour) - wasting $2-15/hour
- Monthly budget exceeded
- Idle detection query: `avg_over_time(DCGM_FI_DEV_GPU_UTIL[30m]) < 5`

### Grafana GPU Dashboards
- **Pre-built dashboard ID 9822**: NVIDIA GPU Kubernetes Monitoring
- **Key panels**: Utilization, memory, temperature, power consumption
- **PromQL examples** for GPU metrics visualization
- **Cost tracking panels**: Estimated spend per GPU, monthly projections
- **Multi-GPU comparison**: Side-by-side GPU performance

---

## Research Sources Used

**Primary Web Research**:
1. **Google Cloud official docs** - Cloud Monitoring DCGM integration, GPU pricing, CUDs
2. **NVIDIA Developer** - DCGM documentation, Compute Sanitizer debugging guide
3. **Grafana documentation** - GPU observability setup, dashboard examples
4. **Medium/LeaderGPU** - Self-managed monitoring stack tutorials
5. **PyTorch Forums** - Memory leak debugging patterns
6. **GitHub** - DCGM Exporter, nvidia-smi exporters (utkuozdemir)
7. **Community blogs** - CloudBolt, Economize Cloud (pricing comparisons)

**Search queries executed**:
- "DCGM Cloud Monitoring integration GCP 2024 2025"
- "nvidia-smi prometheus exporter GPU monitoring 2024"
- "GPU utilization dashboards Grafana Cloud Monitoring GCP 2024 2025"
- "GPU memory leak detection patterns debugging 2024"
- "Committed Use Discounts GPU GCP pricing 2024 2025"

---

## Connections to Existing Knowledge

**Referenced files** (from ingestion plan):
- `mlops-production/00-monitoring-cicd-cost-optimization.md` - MLOps monitoring patterns (not yet created, researched via web)
- `practical-implementation/08-gpu-memory-debugging-vlm-2025-01-30.md` - GPU debugging (not yet created, researched via web)

**New knowledge fills gaps**:
- **Cost optimization** (PART 17) will reference monitoring for idle detection
- **Benchmarking** (PART 20) will use DCGM metrics for performance validation
- **Production patterns** (PART 21) will integrate monitoring in deployment workflows

---

## Implementation Highlights (arr-coc-0-1)

**Monitoring stack**:
```
8×A100 Training Instance
├── DCGM Daemon (GPU metrics)
├── Ops Agent (→ Cloud Monitoring)
├── DCGM Exporter (→ Prometheus)
├── Custom training metrics (Python app)
└── Prometheus (30-day retention)
    ↓
Grafana (dashboards)
    ↓
Alertmanager (PagerDuty/Slack)
```

**Key metrics tracked**:
- GPU utilization per GPU (8 GPUs monitored individually)
- Memory usage (detect leaks, OOM prediction)
- Temperature (thermal throttling detection)
- Training batch duration (performance regression)
- Cost per minute ($29.38/hour for 8×A100, idle = waste)

**Alert thresholds**:
- Idle > 1 hour → Warning ($29.38 wasted)
- Temperature > 85°C → Critical (throttling risk)
- Memory > 95% → Critical (OOM imminent)
- Training stalled > 15min → Critical (crash likely)

---

## Quality Checklist

- [✓] All 8 sections completed (~720 lines total)
- [✓] Web research conducted (5 search queries, 10+ sources)
- [✓] Citations included (Google Cloud, NVIDIA, Grafana, community blogs)
- [✓] Existing knowledge checked (referenced files noted, researched via web)
- [✓] Code examples provided (gcloud, Terraform, Python, PromQL, YAML)
- [✓] arr-coc-0-1 production example (Section 8)
- [✓] Sources section with access dates and URLs
- [✓] Cost analysis with 2024-2025 pricing
- [✓] Practical alerting rules and thresholds

---

## Metadata

**PART**: 18
**File**: `gcp-gpu/17-gpu-monitoring-observability.md`
**Lines**: ~720
**Research Time**: Web research (5 queries, multiple scrapes)
**Status**: ✓ Complete
**Next PART**: 19 (GPU Resource Quotas & Governance)
