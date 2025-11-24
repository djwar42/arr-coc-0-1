# KNOWLEDGE DROP: Vertex AI Logging & Debugging

**Runner**: PART 12 Worker
**Date**: 2025-11-16 13:28
**Status**: SUCCESS ✓
**Files Created**: 1
**Total Lines**: ~700

---

## Summary

Created comprehensive Vertex AI logging, debugging, and troubleshooting guide covering Cloud Logging filters, common error patterns, Cloud Trace latency analysis, Cloud Profiler performance profiling, and cost spike investigation with detailed billing exports.

---

## Files Created

### gcp-vertex/11-logging-debugging-troubleshooting.md (~700 lines)

**Section 1: Cloud Logging Filters for Vertex AI (~120 lines)**
- Vertex AI resource types (aiplatform.googleapis.com/CustomJob, Endpoint, Pipeline)
- Time-based filtering and job lifecycle events
- Advanced log queries (failed jobs, GPU utilization, slow data loading)
- Log export to BigQuery with aggregation examples

**Section 2: Log Severity Levels & Structured Logging (~100 lines)**
- Cloud Logging severity hierarchy (DEBUG → CRITICAL → EMERGENCY)
- Structured JSON logging best practices for training scripts
- Filter logs by severity (ERROR, WARNING, INFO)
- Custom severity thresholds and retention policies

**Section 3: Common Error Patterns (~150 lines)**
- Out of Memory (OOM): Exit code 137, CUDA OOM, diagnosis and solutions
- Quota Exceeded: GPU quota limits, request quota increase workflow
- Permission Denied: IAM roles required, debugging decision tree
- Network Timeout: VPC configuration, Private Google Access, retry logic

**Section 4: Cloud Trace for Request Latency (~100 lines)**
- Cloud Trace overview for Vertex AI endpoints
- Enable tracing for prediction requests
- Analyze latency patterns (preprocessing, inference, postprocessing bottlenecks)
- Custom trace spans for training workflows

**Section 5: Cloud Profiler (CPU, Memory, Heap) (~100 lines)**
- Enable Cloud Profiler in training scripts
- Flame graph analysis (CPU time, call stack depth)
- GPU profiling with TensorBoard Profiler
- Profiling best practices (sampling, epoch-based profiling)

**Section 6: Cost Spike Investigation (~120 lines)**
- Enable detailed billing export to BigQuery
- Vertex AI cost analysis queries (daily spending, top SKUs)
- Identify cost spikes (day-over-day changes >50%)
- Resource-level cost attribution (job_id, endpoint_id, custom labels)
- Root cause analysis for common cost spike patterns

**Section 7: arr-coc-0-1 Debugging Workflows (~110 lines)**
- arr-coc-0-1 training job debugging (texture preprocessing OOM, LOD allocation)
- Structured logging for ARR-COC metrics (relevance scores, tensions, token budgets)
- Inference latency analysis (knowing → balancing → attending → realizing)
- Cloud Build monitoring for PyTorch compilation (2-4 hour builds)
- Cost tracking by W&B run_id with anomaly detection

---

## Key Insights

### Cloud Logging Best Practices
- Use `resource.type="aiplatform.googleapis.com/CustomJob"` for training job logs
- Export structured logs to BigQuery for advanced querying
- Filter by severity: `severity >= ERROR` for production monitoring
- Stream logs in real-time: `gcloud logging tail ...`

### Common Error Patterns
1. **OOM (Exit 137)**: Gradient checkpointing, batch size reduction, FP16 training
2. **Quota Exceeded (429)**: Request quota increase, use multiple regions
3. **Permission Denied (403)**: Grant roles/storage.objectAdmin, roles/aiplatform.user
4. **Network Timeout**: Enable Private Google Access, implement retry logic

### Cost Investigation Workflow
1. Query daily costs by SKU from BigQuery billing export
2. Identify day-over-day spikes >50% change
3. Drill down to resource-level costs (job_id, endpoint_id)
4. Common causes: Long-running jobs, idle endpoints, over-provisioned serving
5. Set up budget alerts at 50%, 75%, 90%, 100% thresholds

### arr-coc-0-1 Specific Debugging
- Log ARR-COC metrics: relevance scores, opponent tensions, token budgets
- Trace inference latency: texture preprocessing often bottleneck (13 channels)
- Monitor Cloud Build for PyTorch compilation: 2-4 hours, check for timeouts
- Track costs by W&B run_id using custom labels

---

## Citations

### Source Documents
- [practical-implementation/36-vertex-ai-debugging.md](../karpathy/practical-implementation/36-vertex-ai-debugging.md) - Cloud Logging basics, common issues, interactive shell, Cloud Profiler

### Web Research
- [Vertex AI Audit Logging](https://docs.cloud.google.com/vertex-ai/docs/general/audit-logging) (accessed 2025-11-16)
- [Vertex AI Troubleshooting](https://cloud.google.com/vertex-ai/docs/general/troubleshooting) (accessed 2025-11-16)
- [Cloud Trace Quickstart](https://docs.cloud.google.com/trace/docs/trace-app-latency) (accessed 2025-11-16)
- [Cloud Profiler Overview](https://docs.cloud.google.com/profiler/docs/about-profiler) (accessed 2025-11-16)
- [Detailed Billing Export](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery-tables/detailed-usage) (accessed 2025-11-16)

### Search Queries
- "Cloud Logging Vertex AI filters resource.type aiplatform 2024"
- "Vertex AI common errors troubleshooting OOM quota permission 2024"
- "Cloud Trace request latency analysis GCP 2024"
- "Cloud Profiler GPU CPU memory profiling GCP 2024"
- "GCP detailed billing export cost spike investigation BigQuery 2024"
- "Vertex AI cost analysis detailed billing SKU investigation 2024"

---

## Quality Checklist

- [✓] All 7 sections completed (~700 lines total)
- [✓] Cloud Logging filters with practical examples
- [✓] Common error patterns with solutions (OOM, quota, permissions, network)
- [✓] Cloud Trace latency analysis workflow
- [✓] Cloud Profiler CPU/GPU/memory profiling
- [✓] Cost spike investigation with BigQuery queries
- [✓] arr-coc-0-1 debugging workflows
- [✓] Source documents cited (practical-implementation/36)
- [✓] Web research links included (accessed 2025-11-16)
- [✓] Code examples use proper syntax highlighting
- [✓] Production-ready debugging patterns

---

## Statistics

- **Total lines**: ~700
- **Sections**: 7
- **Code examples**: 40+
- **BigQuery queries**: 12
- **Source documents**: 1
- **Web sources**: 6
- **Key topics**: Cloud Logging, error patterns, Cloud Trace, Cloud Profiler, cost investigation, arr-coc-0-1 debugging

---

**PART 12 COMPLETE** ✓
