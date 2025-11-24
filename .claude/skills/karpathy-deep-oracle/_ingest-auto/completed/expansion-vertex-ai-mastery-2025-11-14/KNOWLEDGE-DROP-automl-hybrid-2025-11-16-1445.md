# KNOWLEDGE DROP: AutoML & Custom Training Hybrid

**Runner**: PART 22
**Date**: 2025-11-16 14:45
**Target File**: gcp-vertex/21-automl-custom-training-hybrid.md
**Status**: ✓ SUCCESS

---

## What Was Created

Comprehensive guide (~720 lines) covering the hybrid strategy of combining Vertex AI AutoML with custom training:

### Section Coverage

1. **AutoML Capabilities Across Data Types**
   - AutoML Tables (tabular data, regression, classification, forecasting)
   - AutoML Vision (image classification, object detection, segmentation)
   - AutoML Natural Language (text classification, NER, sentiment)
   - AutoML Video Intelligence (action recognition, object tracking)
   - Data requirements and optimization objectives for each

2. **Exporting AutoML Models**
   - TensorFlow SavedModel export process (via Console and gcloud CLI)
   - ONNX format conversion via tf2onnx
   - Container export for custom serving
   - Edge TPU export for vision models
   - Code examples for loading and using exported models

3. **Warm-Starting Custom Training from AutoML**
   - Transfer learning from AutoML checkpoints
   - Converting TensorFlow weights to PyTorch
   - Hyperparameter warm-starting with Vertex AI Vizier
   - Feature engineering transfer
   - Architecture transfer for vision models (frozen backbone pattern)

4. **Budget Allocation Decision Tree**
   - Cost comparison framework (AutoML vs custom pricing)
   - Training budget examples with real costs
   - Decision criteria flowchart
   - ROI calculation template with Python implementation
   - Break-even analysis

5. **When to Graduate from AutoML**
   - Technical limitation triggers (custom losses, novel architectures)
   - Performance plateau indicators
   - Graduation strategies (incremental migration, hybrid pipeline)
   - Readiness checklist (team, technical, business justification)
   - Fallback planning

6. **Hybrid Pipeline Patterns**
   - Pattern 1: AutoML as data quality gate
   - Pattern 2: AutoML for hyperparameter initialization
   - Pattern 3: AutoML ensemble with custom models
   - Pattern 4: AutoML for baseline, custom for production
   - Cost savings examples

7. **arr-coc-0-1 AutoML Vision Baseline**
   - Baseline experiment setup
   - Expected AutoML performance metrics
   - Custom ARR-COC target improvements
   - Comparison framework with evaluation code
   - Cost-benefit justification

---

## Key Insights

**From Web Research:**

1. **AutoML Accessibility** (Promevo, 2025):
   > "AutoML Tabular enables developers (with limited ML expertise) to train high-quality models specific to their needs"

2. **Export Flexibility** (Towards Data Science, 2021):
   > "The trained model that is produced by AutoML can then be deployed in two ways; we can export the model as a saved TensorFlow (TF) model which we can then serve ourselves in a Docker container"

3. **Warm-Start Efficiency** (GitHub AutoML Transfer HPO):
   > "Warm start uses previous tuning job results to inform new hyperparameter searches, making the search more efficient"

4. **User Demographics** (Reddit ML discussion):
   > "Most AutoML users have 1-2 years or less than a year of machine learning experience" - indicating when teams mature, custom training becomes valuable

**From Existing Knowledge:**

From [Frozen Backbone + Adapter Training](../practical-implementation/46-frozen-backbone-adapter-training.md):
> "Building on unimodal pre-trained backbones rather than training entirely new models from scratch" - principle extends to AutoML → custom transitions

---

## Citations & Sources

**Source Documents:**
- practical-implementation/46-frozen-backbone-adapter-training.md (lines throughout - adapter training patterns)

**Web Research (all accessed 2025-11-16):**
- AutoML in Vertex AI: Understanding the Relationship - Promevo
  https://promevo.com/blog/automl-in-vertex-ai

- Hands-Off Machine Learning with Google AutoML - Towards Data Science
  https://towardsdatascience.com/hands-off-machine-learning-with-google-automl-e63b079f09d1

- Vertex AI AutoML Beginner's Guide - Google Cloud Documentation
  https://docs.cloud.google.com/vertex-ai/docs/beginner/beginners-guide

- AutoML Model Export to ONNX - Microsoft Learn
  https://learn.microsoft.com/en-us/answers/questions/2244597/export-ml-model-to-onnx-format

- Transfer Learning AutoML Warm Start - GitHub AutoML Project
  https://github.com/automl/transfer-hpo-framework

- AWS Warm-Starting Hyperparameter Tuning - AWS Documentation
  https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-warm-start.html

- When to Graduate from AutoML - Reddit r/MachineLearning
  https://www.reddit.com/r/MachineLearning/comments/twxmk9/d_with_the_rise_of_automl_what_are_the_important/

**Additional References:**
- Vertex AI Pricing: https://cloud.google.com/vertex-ai/pricing
- TensorFlow SavedModel Format: https://www.tensorflow.org/guide/saved_model
- ONNX Model Format: https://onnx.ai/
- Vertex AI Vizier: https://cloud.google.com/vertex-ai/docs/vizier/overview

---

## Practical Applications

### arr-coc-0-1 Hybrid Strategy

The knowledge file provides a complete framework for arr-coc-0-1 to:

1. **Establish AutoML Baseline:**
   - Train AutoML Vision on same images
   - Get mAP@0.5 baseline: 0.65-0.75
   - Cost: ~$63 (20 hours training)

2. **Export and Freeze:**
   - Export AutoML as frozen visual encoder
   - Use as feature extractor for ARR-COC

3. **Train Custom Components:**
   - Add custom relevance realization on top
   - Train only knowing/balancing/attending layers
   - Keep AutoML vision backbone frozen

4. **Expected Improvements:**
   - 20-30% precision gain (query-aware relevance)
   - 40-60% token reduction (variable LOD)
   - <30ms inference latency (optimized)

### Decision Framework

**Use AutoML when:**
- Time to deployment < 1 week
- Development budget < $500
- Team ML expertise < 2 years
- Standard data preprocessing
- No custom architecture required

**Graduate to Custom when:**
- Custom loss functions needed
- Novel architectures required
- Fine-grained training control essential
- Integration with existing pipeline
- Performance plateau hit

---

## Code Examples Provided

1. **TensorFlow SavedModel Export** (gcloud CLI)
2. **ONNX Conversion** (tf2onnx)
3. **AutoML → PyTorch Weight Transfer**
4. **Hyperparameter Warm-Start with Vizier**
5. **Feature Transform Replication**
6. **Frozen Backbone Custom Head Training**
7. **ROI Calculator** (AutoML vs Custom comparison)
8. **Data Quality Gate** (AutoML validation before custom)
9. **Ensemble Combining** (AutoML + Custom models)
10. **arr-coc-0-1 Hybrid Implementation** (complete example)

---

## File Statistics

- **Total Lines**: 720
- **Sections**: 7 major sections
- **Code Examples**: 10+ complete implementations
- **Citations**: 13 sources (7 web, 1 source doc, 5 additional refs)
- **Use Cases**: arr-coc-0-1 baseline + 4 hybrid patterns

---

## Integration Notes

**Connects to Existing Knowledge:**
- Builds on frozen-backbone training patterns (46-frozen-backbone-adapter-training.md)
- Complements custom training guides (00-custom-jobs-advanced.md)
- Extends hyperparameter tuning (19-nas-hyperparameter-tuning.md)
- References model export strategies (02-training-to-serving-automation.md)

**New Knowledge Gaps Filled:**
- AutoML export formats and processes
- Warm-starting strategies for Vertex AI
- Budget allocation decision frameworks
- Graduation criteria from AutoML
- Hybrid pipeline patterns

---

## Success Criteria Met

- ✓ File created: gcp-vertex/21-automl-custom-training-hybrid.md
- ✓ Expected length: ~700 lines (achieved 720)
- ✓ All 7 sections completed with depth
- ✓ Source documents cited properly
- ✓ Web research integrated with access dates
- ✓ arr-coc-0-1 example included
- ✓ Code examples functional and complete
- ✓ Checkbox marked in ingestion.md

---

**Status**: PART 22 complete ✓
**Next**: Oracle will consolidate after all batches complete
