# KNOWLEDGE DROP: BigQuery ML + Vertex AI Integration

**File**: gcp-vertex/06-bigquery-ml-vertex-integration.md
**Created**: 2025-11-16 13:17
**Lines**: ~700
**Status**: Complete ✓

## What Was Created

Comprehensive guide covering BigQuery ML and Vertex AI integration:

### 8 Major Sections:

1. **CREATE MODEL in BigQuery** (~100 lines)
   - XGBoost, DNN, AutoML model types
   - TRANSFORM clause for preprocessing
   - Hyperparameter tuning with num_trials
   - Model versioning and Vertex AI registration

2. **ML.PREDICT for Batch Inference** (~100 lines)
   - SQL-based prediction at scale
   - Performance optimization (partitioning, clustering)
   - ML.EXPLAIN_PREDICT for Shapley values
   - Cost monitoring ($5/TB on-demand)

3. **EXPORT MODEL to GCS** (~80 lines)
   - TensorFlow SavedModel format
   - ONNX and XGBoost Booster exports
   - TRANSFORM clause preprocessing in exports
   - Verify exported models locally

4. **Import to Vertex AI Model Registry** (~80 lines)
   - Auto-registration during CREATE MODEL
   - Upload exported SavedModel to Vertex AI
   - Model versioning with aliases (prod, staging, v1)
   - Unified model governance

5. **Vertex AI Batch Prediction from BigQuery** (~100 lines)
   - BigQuery tables as input/output
   - Automatic parallelization across workers
   - Monitor job status and completion stats
   - Cost optimization (machine types, spot VMs)

6. **Federated Queries** (~80 lines)
   - Cloud SQL (PostgreSQL/MySQL) integration
   - Google Sheets as external tables
   - Filter pushdown optimization
   - Materialize federated results for caching

7. **Cost Optimization** (~120 lines)
   - On-demand vs slot reservations comparison
   - BigQuery Editions (Standard/Enterprise/Enterprise Plus)
   - Break-even analysis (400 TB/month threshold)
   - Autoscaling configuration
   - Partition/cluster for 99.7% cost reduction

8. **arr-coc-0-1 Preprocessing Example** (~100 lines)
   - Qwen-VL embeddings in BigQuery
   - XGBoost patch relevance scoring
   - Dynamic token allocation (64-400 tokens)
   - Cloud SQL federated integration
   - Complete workflow: embeddings → training → batch prediction → export

## Key Technical Details

**BigQuery ML Model Types:**
- Linear/Logistic Regression
- XGBoost (BOOSTED_TREE_CLASSIFIER/REGRESSOR)
- Deep Neural Networks (DNN_CLASSIFIER/REGRESSOR)
- AutoML Tables (automatic architecture search)
- K-Means, Matrix Factorization, ARIMA

**Pricing Models:**
- On-demand: $5/TB processed, 2,000 shared slots
- Slot reservations: $2,000/month baseline (100 slots)
- BigQuery Editions: $0.04-$0.10 per slot-hour
- Break-even: ~400 TB/month processed

**Integration Patterns:**
- BigQuery ML → EXPORT → Vertex AI Model Registry
- BigQuery ML → auto-register → Vertex AI (no export)
- Vertex AI Batch Prediction with BigQuery I/O
- Federated queries (Cloud SQL, Sheets) → BigQuery ML

**arr-coc-0-1 Use Case:**
- Train relevance scorer on Qwen-VL embeddings
- Aggregate patch features (spatial, edge density)
- XGBoost regression for token allocation
- Batch prediction for 64-400 token range
- Export to Cloud SQL for integration

## Sources Cited

**Google Cloud Docs (11 references):**
- BigQuery ML Introduction
- BigQuery Pricing
- Inference Overview
- Export Models
- Manage Models in Vertex AI
- Vertex AI Batch Prediction
- Cloud SQL Federated Queries

**Source Documents (1 reference):**
- gcloud-data/00-storage-bigquery-ml-data.md

**Web Research (4 references):**
- OWOX BI BigQuery ML Guide (May 2025)
- BigQuery ML Inference Engine (March 2023)
- Economize Cloud CUDs Guide (July 2025)
- RTB House Capacity Pricing (March 2025)

## Code Examples

**7 Complete SQL Examples:**
1. CREATE MODEL with XGBoost and hyperparameter tuning
2. DNN with TRANSFORM clause preprocessing
3. ML.PREDICT with partition pruning
4. EXPORT MODEL to TensorFlow SavedModel
5. Federated query to Cloud SQL
6. Cost monitoring from INFORMATION_SCHEMA
7. arr-coc-0-1 patch relevance scoring pipeline

**3 Python Examples:**
1. Upload exported SavedModel to Vertex AI
2. Batch prediction job with BigQuery I/O
3. Monitor batch job status and costs

## Quality Metrics

- **Completeness**: 8/8 sections (100%)
- **Citations**: 16 sources properly cited
- **Code examples**: 10 complete, runnable examples
- **arr-coc-0-1 integration**: Full workflow example
- **Production-ready**: Cost optimization, monitoring, best practices

## Next Steps (Not Done in This PART)

- Oracle will consolidate all KNOWLEDGE DROP files
- Oracle will update INDEX.md with new file
- Oracle will update SKILL.md if needed
- Oracle will move to completed/
- Oracle will git commit
