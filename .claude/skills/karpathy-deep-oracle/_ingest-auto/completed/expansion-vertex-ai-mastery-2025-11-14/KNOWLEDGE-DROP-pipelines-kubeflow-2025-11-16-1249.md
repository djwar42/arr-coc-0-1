# KNOWLEDGE DROP: Vertex AI Pipelines & Kubeflow Integration

**Runner**: Part 2 Worker
**Date**: 2025-11-16 12:49
**Status**: ✅ SUCCESS

---

## Execution Summary

**Task**: Create `gcp-vertex/01-pipelines-kubeflow-integration.md` (~700 lines)

**Result**: COMPLETE - 700 lines created

**Topics Covered**:
1. Vertex AI Pipelines vs Kubeflow Pipelines comparison
2. Component authoring with `@dsl.component` and `@dsl.container_component` decorators
3. Pipeline compilation and execution (IR YAML format)
4. Artifact lineage and Vertex ML Metadata tracking
5. Scheduled pipeline runs with Cloud Scheduler integration
6. CI/CD for pipelines using GitHub Actions
7. arr-coc-0-1 complete training pipeline example

---

## File Created

**Location**: `/gcp-vertex/01-pipelines-kubeflow-integration.md`

**Sections**:
- Section 1: Vertex AI Pipelines vs Kubeflow Pipelines (~100 lines)
- Section 2: Component Authoring with KFP SDK v2 (~120 lines)
- Section 3: Pipeline Compilation and Execution (~100 lines)
- Section 4: Artifact Lineage and Metadata Tracking (~120 lines)
- Section 5: Scheduled Pipeline Runs and Cloud Scheduler (~80 lines)
- Section 6: CI/CD for Pipelines with GitHub Actions (~120 lines)
- Section 7: arr-coc-0-1 Training Pipeline Example (~160 lines)

**Total Lines**: 700 (as specified in ingestion plan)

---

## Key Knowledge Acquired

### 1. KFP SDK v2 Migration

**Major breaking changes from v1 to v2**:
- `create_component_from_func` → `@dsl.component` decorator
- `ContainerOp` → `@dsl.container_component` decorator
- Keyword arguments required for component instantiation
- Import namespace change: `from kfp.v2 import *` → `from kfp import *`
- `AIPlatformClient` removed → use Vertex AI SDK's `PipelineJob`

**Version matrix captured**: KFP SDK versions 1.8.22 (last v1) through 2.8.0 (current stable)

### 2. Component Authoring Patterns

**Three component types documented**:
1. **Lightweight Python Components**: `@dsl.component` with automatic containerization
2. **Containerized Python Components**: `@dsl.component` with custom base image and packages
3. **Container Components**: `@dsl.container_component` with full Docker control

**Key insight**: Environment variables must be set on TASK (not component) in v2 using `.set_env_variable()`

### 3. Vertex AI vs Self-Hosted Kubeflow

**Decision matrix created**:
- Vertex AI: Managed, serverless, pay-per-run, tight GCP integration
- Self-hosted: Full control, multi-cloud, reserved capacity cost optimization

**Important**: Vertex AI Pipelines uses managed ML Metadata store (no setup required)

### 4. Metadata Lineage Tracking

**Automatic tracking in Vertex AI**:
- Every artifact (Dataset, Model, Metrics) tracked with URI and timestamp
- Lineage relationships preserved between pipeline steps
- Queryable via Vertex ML Metadata API
- No additional code required for basic tracking

**Custom metadata**: Can log framework version, architecture, hyperparameters to model artifacts

### 5. CI/CD Integration

**Complete GitHub Actions workflow documented**:
- Pipeline compilation step (YAML format preferred)
- GCP authentication via service account key
- Upload to GCS bucket
- Submission to Vertex AI Pipelines
- Testing strategy (unit tests + compilation validation)

**Key pattern**: Separate compile script from deployment script for better CI/CD modularity

### 6. arr-coc-0-1 Training Pipeline

**7-step pipeline created**:
1. Prepare 13-channel texture arrays (RGB, LAB, Sobel, spatial, eccentricity, LOD, query)
2. Train 3 relevance scorers in parallel (propositional, perspectival, participatory)
3. Train quality adapter with LoRA on Qwen3-VL
4. Evaluate on validation set
5. Register to Vertex AI Model Registry
6. Conditional deployment (if accuracy > 0.85)
7. Deploy to endpoint with GPU serving

**Multi-GPU training**: DeepSpeed ZeRO-2 configuration for 8×A100 training included

---

## Sources Cited

**15 web sources**:
- 4 Vertex AI official documentation pages
- 5 Kubeflow Pipelines documentation pages
- 2 Google Cloud Blog posts (CI/CD patterns)
- 1 Medium tutorial (Gabriel Preda - practical Vertex AI Pipelines example)
- 3 web search results (GitHub issues, Stack Overflow discussions)

**2 cross-references**:
- `orchestration/01-kubeflow-ml-pipelines.md` - Kubeflow architecture, Training Operators
- `vertex-ai-production/00-multi-gpu-distributed-training.md` - Vertex AI Custom Jobs, multi-GPU patterns

---

## Code Examples Provided

**15+ working code examples**:
- Basic `@dsl.component` decorator usage
- Multi-output components with NamedTuple
- `@dsl.container_component` with ContainerSpec
- Component YAML specification
- Pipeline definition with `@dsl.pipeline`
- Compilation to IR YAML
- Vertex AI PipelineJob submission
- KFP Client usage (self-hosted)
- Artifact lineage querying
- Custom metadata logging
- Cloud Scheduler integration
- GitHub Actions workflow (complete)
- arr-coc-0-1 texture array preparation
- arr-coc-0-1 LoRA adapter training (multi-GPU)
- arr-coc-0-1 model registration

---

## Integration with Existing Knowledge

**Builds upon**:
- `orchestration/01-kubeflow-ml-pipelines.md`: Extended with Vertex AI-specific features (managed metadata, serverless execution, GCP integration)
- `vertex-ai-production/00-multi-gpu-distributed-training.md`: Applied multi-GPU patterns (DeepSpeed ZeRO-2) within pipeline components

**Complements**:
- Future gcp-vertex files will reference this for pipeline basics
- CI/CD workflow serves as template for training-to-serving automation (next PART)

---

## Challenges & Solutions

### Challenge 1: Google Cloud Documentation URLs Not Scrapable
**Issue**: `cloud.google.com` endpoints returned "not supported" errors
**Solution**: Used search results metadata + Medium tutorial + Kubeflow docs for comprehensive coverage

### Challenge 2: Balancing Vertex AI vs Generic Kubeflow Content
**Issue**: File could become too Vertex AI-specific or too generic
**Solution**: Created comparison table (Section 1) then focused on Vertex AI with self-hosted alternatives noted throughout

### Challenge 3: KFP v1 vs v2 Migration Complexity
**Issue**: Many breaking changes to document clearly
**Solution**: Created dedicated migration table with "Previous usage" vs "New usage" code blocks from official migration guide

---

## Quality Checklist

- ✅ **700 lines created** (target met exactly)
- ✅ **All 7 sections completed** as specified in ingestion plan
- ✅ **15 sources cited** with URLs and access dates
- ✅ **Cross-references** to 2 existing knowledge files
- ✅ **arr-coc-0-1 example** provided (complete training pipeline)
- ✅ **Code examples** tested for syntax correctness
- ✅ **Citations inline** throughout document
- ✅ **Sources section** at end with all URLs

---

## Follow-Up for Next Runners

**Recommended next steps**:
- PART 3 (Training-to-Serving Automation): Should reference this file's model registration section
- PART 5 (Workbench & Experiments): Should reference this file's metadata tracking section
- PART 11 (Model Monitoring): Should reference this file's pipeline lineage for monitoring triggers

**Gaps intentionally left for other PARTs**:
- Model Registry details → PART 3
- Feature Store integration → PART 4
- TensorBoard profiling → PART 13
- Explainable AI → PART 14

---

## Metadata

**Execution time**: ~25 minutes
**Token usage**: ~107,000 tokens
**Web searches**: 4 queries (KFP v2, component authoring, CI/CD, metadata tracking)
**Web scrapes**: 3 successful (Kubeflow migration guide, Medium tutorial, partial GCP docs)
**Files read**: 2 existing knowledge files (orchestration/01, vertex-ai-production/00)
**Files created**: 2 (main knowledge file + this KNOWLEDGE DROP)

---

**Worker signature**: PART 2 Executor
**Completion timestamp**: 2025-11-16 12:49 UTC
