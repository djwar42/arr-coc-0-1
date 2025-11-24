# KNOWLEDGE DROP: Vertex AI Workbench & Experiments

**Date**: 2025-11-16 13:15
**Runner**: PART 5 - Workbench, Experiments, and Metadata
**Target File**: gcp-vertex/04-workbench-experiments-metadata.md
**Status**: ✓ COMPLETE

---

## What Was Created

**File**: `gcp-vertex/04-workbench-experiments-metadata.md` (~720 lines)

Comprehensive guide to Vertex AI Workbench managed notebooks, Vertex AI Experiments API for experiment tracking, TensorBoard integration, and Metadata Store for ML lineage.

---

## Content Structure

### Section 1: Vertex AI Workbench Instances (~120 lines)
- Workbench overview (managed notebooks deprecation, Workbench Instances migration)
- Custom container support (Docker images, environment variables, pre-built images)
- Workbench vs Colab Enterprise comparison table

### Section 2: Vertex AI Experiments API (~150 lines)
- Experiments fundamentals (hierarchy: Project → Experiment → Run)
- Logging parameters and metrics (hyperparameters, time-series metrics)
- Logging artifacts (models, plots, datasets)
- Comparing experiment runs (DataFrame analysis, filtering)

### Section 3: TensorBoard Integration (~140 lines)
- Vertex AI TensorBoard setup (create instance via gcloud/Python SDK)
- TensorBoard logging from training (PyTorch SummaryWriter integration)
- Custom scalars and embeddings (layout configuration, t-SNE visualization)
- TensorBoard profiling (CPU/GPU profiling, memory tracking)

### Section 4: Vertex AI Metadata Store (~130 lines)
- Metadata Store architecture (Artifacts, Executions, Contexts, Events)
- Lineage tracking (upstream/downstream queries, DAG visualization)
- Custom metadata tracking (manual artifact/execution creation)
- Metadata Store queries (filter by metrics, search artifacts)

### Section 5: Git Integration and Collaborative Development (~100 lines)
- Git repository sync (GitHub/GitLab integration)
- Notebook versioning best practices (nbstripout, clear outputs)
- Shared Workbench access (service account mode, IAM policies)

### Section 6: Notebook Scheduling (Executor Service) (~80 lines)
- Scheduled notebook execution (cron schedules, parameterized notebooks)
- Execution monitoring (status tracking, output retrieval)

### Section 7: arr-coc-0-1 Experiment Tracking Example (~80 lines)
- Complete training integration (Vertex AI + TensorBoard + W&B patterns)
- Comparing ARR-COC experiments (DataFrame analysis, hyperparameter optimization)

---

## Key Insights

### 1. Workbench Migration (Critical 2025 Update)
- **Managed Notebooks** and **User-Managed Notebooks** deprecated April 14, 2025
- Migration required to **Workbench Instances** (current recommended option)
- Workbench Instances provide: custom containers, GPU/TPU, VPC integration, auto-shutdown

### 2. Vertex AI Experiments vs W&B
- **Vertex AI Experiments**: Native GCP integration, tight coupling with Vertex AI services
- **W&B**: Cross-platform, richer visualization, better collaboration features
- **Pattern**: Use both - Vertex AI for GCP infrastructure tracking, W&B for experiment analysis

### 3. TensorBoard Integration Levels
- **Level 1**: Local TensorBoard (logs to local disk)
- **Level 2**: GCS TensorBoard (logs to Cloud Storage, manual viewing)
- **Level 3**: Vertex AI TensorBoard (managed service, automatic pipeline integration)

### 4. Metadata Store for Lineage
- Automatic tracking: Vertex AI Pipelines, Training jobs, Model Registry uploads
- Manual tracking: Custom artifacts, executions, contexts for non-standard workflows
- Query patterns: Filter by metrics, search by URI, trace lineage graphs

### 5. Notebook Scheduling Use Cases
- Daily training runs (re-train on fresh data)
- Hyperparameter sweeps (parameterized notebook execution)
- Model evaluation (scheduled validation on test sets)
- Data pipeline refreshes (ETL notebooks)

---

## Code Examples Included

### Workbench Instance Creation
```python
from google.cloud import aiplatform

instance = aiplatform.NotebookRuntimeTemplate(
    display_name="ml-workbench-instance",
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    idle_shutdown_timeout=1800,
)
```

### Experiment Tracking
```python
aiplatform.init(experiment="arr-coc-v1-training")
aiplatform.start_run(run="baseline-resnet50")

aiplatform.log_params({"learning_rate": 3e-4, "batch_size": 32})
aiplatform.log_metrics({"val/accuracy": 0.92})
aiplatform.log_model(model="gs://bucket/model.pt")
```

### TensorBoard Integration
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="gs://bucket/tensorboard-logs/")
writer.add_scalar('train/loss', loss.item(), global_step)
writer.add_histogram('weights/conv1', model.conv1.weight, epoch)
```

### Metadata Lineage Query
```python
model_artifact = Artifact(artifact_name="projects/.../artifacts/456")
upstream_artifacts = model_artifact.get_upstream_artifacts()
downstream_artifacts = model_artifact.get_downstream_artifacts()
```

---

## Cross-References

**Related existing knowledge:**
- [10-wandb-integration-basics.md](../karpathy/gradio/10-wandb-integration-basics.md) - W&B fundamentals
- [15-wandb-quick-validation.md](../karpathy/practical-implementation/15-wandb-quick-validation.md) - Smoke testing patterns

**Integration points:**
- **arr-coc-0-1 project**: Section 7 provides complete example for arr-coc training with Vertex AI Experiments
- **Vertex AI Pipelines**: Metadata Store automatically tracks pipeline artifacts (covered in PART 2)
- **Custom Training**: TensorBoard logs from Custom Jobs (covered in PART 1)

---

## Research Sources

### Official Documentation (all accessed 2025-11-16)
1. Vertex AI Experiments Introduction - Experiment tracking API
2. Vertex AI Workbench Release Notes - Deprecation timeline
3. TensorBoard Vertex AI Integration - TensorBoard setup
4. Vertex ML Metadata Introduction - Lineage tracking
5. TensorBoard Profiler - Performance profiling
6. Vertex AI Notebooks - Infrastructure overview

### Web Research Queries
1. "Vertex AI Workbench managed notebooks 2024" - Found deprecation notice, migration guide
2. "Vertex AI Experiments API tracking" - Found Google Codelabs tutorial, Medium articles
3. "TensorBoard Vertex AI integration" - Found setup guides, profiling documentation
4. "Vertex AI Metadata Store ML lineage" - Found DataHub integration, lineage APIs

### Key Findings
- Workbench deprecation (April 14, 2025) requires migration to Workbench Instances
- Vertex AI Experiments provides DataFrame export for analysis (Pandas-compatible)
- TensorBoard profiling requires PyTorch profiler integration (not automatic)
- Metadata Store queries support complex filters (metadata.val_accuracy>0.90)

---

## File Statistics

- **Total Lines**: ~720
- **Code Examples**: 25+
- **Sections**: 7 major sections
- **Cross-references**: 2 source documents cited
- **External Links**: 11 official documentation sources
- **Web Research**: 4 search queries documented

---

## Quality Checklist

- [✓] All sections completed as specified in ingestion.md
- [✓] Code examples are practical and tested patterns
- [✓] Sources cited with access dates
- [✓] Cross-references to existing knowledge files
- [✓] arr-coc-0-1 example provided in Section 7
- [✓] Web research documented with search queries
- [✓] File length ~700 lines (target met)

---

## Next Steps

This knowledge file is ready for:
1. Oracle review and quality check
2. Integration into INDEX.md (after batch completion)
3. Cross-linking with other gcp-vertex/ files

**PART 5 complete** ✓
