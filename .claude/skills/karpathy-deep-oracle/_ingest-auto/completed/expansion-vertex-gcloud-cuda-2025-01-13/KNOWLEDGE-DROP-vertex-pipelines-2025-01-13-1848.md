# KNOWLEDGE DROP: Vertex AI Pipelines + Kubeflow

**Runner**: PART 1
**Timestamp**: 2025-01-13 18:48
**Status**: ✓ COMPLETE

---

## Knowledge File Created

**File**: `karpathy/practical-implementation/65-vertex-ai-pipelines-kubeflow.md`
**Lines**: 422 lines
**Sections**: 4 comprehensive sections

### Content Coverage

1. **Architecture & Core Concepts** (100 lines)
   - Vertex AI Pipelines vs Kubeflow Pipelines differences
   - Component-based architecture
   - Pipeline execution model (DAG, artifact passing, pipeline root)

2. **Python SDK & Component Development** (120 lines)
   - Kubeflow Pipelines SDK v2 installation and setup
   - Component decorator and parameters
   - Input/Output artifact types (Dataset, Model, Metrics, Artifact)
   - Container-based components

3. **Pipeline Orchestration** (100 lines)
   - Pipeline definition with @pipeline decorator
   - Pipeline compilation to JSON
   - Pipeline execution via Vertex AI SDK
   - Parallel execution and conditional logic

4. **ARR-COC Production Integration** (80 lines)
   - Multi-stage VLM training pipelines
   - Hyperparameter sweeps with parallel execution
   - Model Registry integration
   - Production deployment automation with canary strategy

---

## Web Research Sources Used

**Google Cloud Official Documentation:**
1. [Vertex AI Pipelines Introduction](https://docs.cloud.google.com/vertex-ai/docs/pipelines/introduction)
   - Serverless pipeline orchestration overview
   - Integration with Vertex AI services

2. [Build a pipeline](https://docs.cloud.google.com/vertex-ai/docs/pipelines/build-pipeline)
   - Component design patterns
   - Pipeline definition best practices

3. [Migrate from Kubeflow Pipelines to Vertex AI Pipelines](https://docs.cloud.google.com/vertex-ai/docs/pipelines/migrate-kfp)
   - Key differences in storage (PV vs GCS)
   - Migration strategies

**Technical Articles:**
4. [Building machine learning pipelines with Vertex AI and KubeFlow in GCP](https://medium.com/@gabi.preda/building-machine-learning-pipelines-with-vertex-ai-and-kubeflow-in-gcp-2214442ba62d)
   - Gabriel Preda, Medium, December 2024
   - Complete Fashion MNIST pipeline example
   - Model Registry integration patterns
   - Custom base image strategy

5. [All you need to know to get started with Vertex AI Pipelines](https://www.artefact.com/blog/all-you-need-to-know-to-get-started-with-vertex-ai-pipelines/)
   - Artefact engineering team practical guide
   - Parallel execution patterns
   - Conditional deployment strategies
   - Production best practices

6. [Kubeflow vs Vertex AI Pipelines](https://traceroute.net/2025/09/25/kubeflow-vs-vertex-ai-pipelines/)
   - Comparative analysis
   - Cost and maintenance considerations

**GitHub Resources:**
7. [googleapis/python-aiplatform](https://github.com/googleapis/python-aiplatform)
   - Official Vertex AI Python SDK
   - Code examples and API reference

---

## Knowledge Gaps Filled

**Before**: Existing Vertex AI files (30-37) mentioned Pipelines briefly but lacked deep-dive coverage on:
- Component development patterns (function-based vs container-based)
- Pipeline orchestration with KFP SDK v2
- Artifact passing and DAG execution model
- Parallel execution and conditional logic
- Production deployment patterns

**After**: Comprehensive 422-line guide covering:
- ✓ Complete Kubeflow Pipelines SDK v2 usage
- ✓ Component decorator patterns (@component, @pipeline)
- ✓ Typed artifacts (Dataset, Model, Metrics, Artifact)
- ✓ DAG-based execution with automatic parallelization
- ✓ Conditional deployment with dsl.Condition
- ✓ Model Registry integration
- ✓ ARR-COC-specific multi-stage VLM training pipelines
- ✓ Hyperparameter sweeps with parallel execution
- ✓ Production deployment automation (canary, A/B testing)

**Cross-References**:
- Links to existing Vertex AI fundamentals (30-vertex-ai-fundamentals.md)
- Links to complete examples (37-vertex-ai-complete-examples.md)
- Links to CI/CD integration (gcloud-cicd/00-pipeline-integration.md)

---

## Key Insights Captured

1. **Vertex AI Pipelines = Managed KFP**: Serverless execution of Kubeflow Pipelines, no cluster management required

2. **Component Reusability**: Function-based components with @component decorator enable fast iteration without container rebuilds

3. **Custom Base Images**: Build custom base image with all dependencies, use as foundation for all components (best practice from Artefact team)

4. **Parallel Execution is Simple**: Writing a for loop automatically parallelizes component execution

5. **Conditional Logic**: dsl.Condition enables smart deployment decisions (e.g., only deploy if accuracy > 0.90)

6. **ARR-COC Integration**: Multi-stage pipelines for training relevance scorers, quality adapter, and integrated VLM model

---

## Citations Quality Check

✓ All claims cite specific sources with URLs
✓ Web links include access dates (2025-01-13)
✓ GitHub links included for code references
✓ Cross-references to existing knowledge files
✓ "Sources" section at end with full bibliography

---

**Knowledge Drop Complete**: Vertex AI Pipelines + Kubeflow orchestration patterns now fully documented for production ML workflows.
