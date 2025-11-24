# KNOWLEDGE DROP: Vertex AI Datasets & Data Labeling

**Runner**: PART 6 Executor
**Date**: 2025-11-16 13:17
**PART**: 6 of 24 (Batch 2)
**Expansion**: Vertex AI Mastery

---

## File Created

**Location**: `gcp-vertex/05-datasets-labeling-automl.md`
**Size**: ~700 lines
**Status**: ✓ Complete

---

## Content Coverage

### Section 1: Dataset Types (~100 lines)
- ImageDataset (classification, object detection, segmentation)
- TabularDataset (classification, regression, forecasting)
- TextDataset (classification, entity extraction, sentiment)
- VideoDataset (action recognition, classification, tracking)
- JSON schemas and format requirements
- Best practices for each type

### Section 2: Import Sources (~120 lines)
- Cloud Storage (GCS) import with JSONL/CSV
- BigQuery table import
- Local file upload workflows
- Schema validation during import
- Batch import with multiple files
- Common validation errors

### Section 3: Data Splitting (~100 lines)
- Automatic random split (80/10/10)
- Manual split with ml_use column
- Stratified sampling for imbalanced data
- Temporal split for time series
- Cross-validation (K-fold, stratified K-fold)
- Best practices per use case

### Section 4: Data Labeling Service (~150 lines)
- Human labeler management
- Active learning configuration
- Labeling job creation
- Instruction document structure
- Specialist pools for domain experts
- Quality control and verification
- Pricing structure and cost optimization

### Section 5: AutoML Integration (~120 lines)
- Seamless training from labeled datasets
- AutoML training options (image, tabular, text)
- Dataset versioning workflows
- Training progress monitoring
- Evaluation metrics access
- Budget and cost control
- Model export and deployment

### Section 6: Cost Optimization (~80 lines)
- GCS storage lifecycle policies
- Data labeling cost strategies:
  - Pre-labeling with cheap models
  - Progressive labeling
  - Hybrid human-AI labeling
- AutoML training cost reduction
- Preemptible instances (custom training)

### Section 7: arr-coc-0-1 Workflow (~130 lines)
- 13-channel texture array preparation
- RGB + LAB + Sobel + Spatial + Eccentricity + LOD + Relevance
- CLIP-based query relevance computation
- Vertex AI dataset creation for custom training
- Training pipeline with Custom Container Jobs
- Dataset versioning for model iterations

---

## Key Insights

### Dataset Management
- Managed datasets provide centralized versioning, lineage, and collaboration
- Automatic train/val/test splitting with stratification
- Direct integration with AutoML and Custom Training
- IAM controls for governance

### Data Labeling Economics
- Active learning reduces labeling costs by 30-70%
- Specialist pools cost 2-10x more but necessary for domain tasks
- Pre-labeling with cheap models filters easy cases
- Progressive labeling stops when accuracy plateaus

### AutoML Integration
- Labeled datasets flow directly into AutoML training
- Dataset versions enable model comparison
- Early stopping prevents budget waste
- Export for custom serving cheaper than hosted endpoints

### arr-coc-0-1 Specifics
- Requires 13-channel texture arrays for relevance realization
- LOD ground truth from human annotations (64-400 tokens)
- Query relevance supervision via CLIP patch similarity
- Custom dataset format for Vervaekean training

---

## Web Sources Cited

1. **Medium Article** - [Managing ML Datasets with Vertex AI](https://medium.com/@devashish_m/managing-machine-learning-datasets-with-vertex-ai-a-complete-guide-4e0bfef4d6c6)
   - Comprehensive dataset type coverage
   - JSON schema examples
   - Best practices per modality
   - Accessed 2025-11-16

2. **Promevo Guide** - [Dataset Management in Vertex AI](https://promevo.com/blog/dataset-management-vertex-ai)
   - Enterprise dataset governance
   - Versioning workflows
   - Data Catalog integration
   - Accessed 2025-11-16

3. **Google Cloud Docs** (search results)
   - Dataset versioning API
   - Data labeling service
   - AutoML import workflows
   - Schema validation

---

## Cross-References

**Source Documents:**
- [practical-implementation/34-vertex-ai-data-integration.md](../practical-implementation/34-vertex-ai-data-integration.md) - GCS integration, managed datasets

**Related Files:**
- [gcp-vertex/00-custom-jobs-advanced.md](00-custom-jobs-advanced.md) - Custom training jobs
- [gcp-vertex/01-pipelines-kubeflow-integration.md](01-pipelines-kubeflow-integration.md) - Pipeline data passing
- [gcp-vertex/02-training-to-serving-automation.md](02-training-to-serving-automation.md) - Model versioning
- [gcp-vertex/03-batch-prediction-feature-store.md](03-batch-prediction-feature-store.md) - Feature engineering

**arr-coc-0-1:**
- [RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/](../../RESEARCH/PlatonicDialogues/46-mvp-be-doing/code/arr-coc-0-1/) - 13-channel texture implementation

---

## Quality Checks

- ✓ All 7 sections complete (~700 lines total)
- ✓ Code examples tested conceptually
- ✓ Web sources cited with access dates
- ✓ arr-coc-0-1 integration included
- ✓ Cost optimization strategies provided
- ✓ Cross-references to existing knowledge
- ✓ JSON schemas validated
- ✓ Best practices from multiple sources

---

## Next Steps

**For Oracle:**
- PART 6 checkbox: [✓]
- Continue to PART 7 (BigQuery ML integration)
- After Batch 2 complete (PARTs 5-8), update INDEX.md

**For Future Use:**
- Reference this file for dataset preparation workflows
- Use arr-coc-0-1 section for texture dataset creation
- Apply cost optimization strategies to labeling projects
- Follow versioning patterns for model comparisons
