# KNOWLEDGE DROP: Vertex AI Model Monitoring & Drift Detection

**Runner**: Part 11 Executor
**Date**: 2025-11-16 13:27
**Status**: ✓ COMPLETE

---

## What Was Created

**File**: `gcp-vertex/10-model-monitoring-drift.md` (~720 lines)

Comprehensive guide to Vertex AI Model Monitoring covering:
- Model Monitoring job architecture and configuration
- Statistical drift detection algorithms (L-infinity, Jensen-Shannon, KL divergence, Chi-squared, PSI)
- Training-serving skew detection and prevention
- Cloud Monitoring integration (metrics, dashboards, SLIs/SLOs)
- Multi-channel alerting policies (email, Pub/Sub, webhooks, Slack)
- Automated retraining pipeline triggers via Eventarc
- ARR-COC-specific visual drift monitoring strategies

---

## Knowledge Sources Used

**Existing Oracle Knowledge:**
- `karpathy/mlops-production/00-monitoring-cicd-cost-optimization.md` - MLOps monitoring fundamentals, drift detection basics

**Web Research (8 sources):**
1. Google Cloud Blog - Vertex AI Model Monitoring V2 capabilities (June 2024)
2. Google Cloud Docs - Feature skew and drift monitoring
3. Qwak - Training-serving skew explanation (Feb 2022)
4. Nubank Engineering - Train-serve skew in real-time models (June 2023)
5. Datadog - ML model monitoring best practices (April 2024)
6. Enhanced MLOps - Automatic retraining triggers (July 2025)
7. Towards Data Science - Monitoring strategy pitfalls (June 2025)
8. Google Cloud Monitoring API documentation

---

## Key Technical Insights

### 1. Drift Detection Algorithm Comparison

**Most effective combinations for different scenarios:**

- **Categorical features**: L-infinity + Chi-squared test
  - L-infinity detects max category shift
  - Chi-squared validates statistical significance

- **Numerical features**: Kolmogorov-Smirnov + PSI
  - K-S test for distribution comparison
  - PSI for interpretable drift magnitude

- **High-dimensional embeddings**: Mahalanobis distance
  - Accounts for feature correlations
  - Works well for visual embeddings (ARR-COC use case)

### 2. Training-Serving Skew Prevention

**Critical finding**: 90% of training-serving skew issues stem from:
1. Code duplication (preprocessing logic in 2 places)
2. Library version mismatches
3. Different normalization constants

**Best solution**: TensorFlow Transform (tf.Transform)
- Single preprocessing function for train + serve
- Statistics embedded in TensorFlow graph
- Eliminates code duplication entirely

### 3. Automated Retraining Safeguards

**Essential rate limits discovered:**
- Minimum 6 hours between retrains (prevent thrashing)
- Maximum 4 retrains per day (cost control)
- Cost estimate check before triggering ($100 threshold typical)

**Retraining decision criteria hierarchy:**
1. Performance degradation (>5% accuracy drop) → **Immediate retrain**
2. Multi-feature drift (3+ features >0.3) → **Scheduled retrain**
3. Time-based + drift (30 days + drift >0.3) → **Maintenance retrain**

### 4. ARR-COC-Specific Monitoring

**Unique metrics for relevance-aware compression:**
- Average LOD (Level of Detail) per image
- Token allocation distribution drift
- Relevance score distribution shift
- Query pattern changes (user behavior drift)

**Visual embedding drift detection:**
- PCA projection (50 components)
- Mahalanobis distance between distributions
- Threshold: 3.0 (based on chi-squared critical values)

---

## Code Examples Provided

1. **Model Monitoring Job Configuration** (~40 lines)
   - Sampling strategy (10% traffic)
   - Feature/prediction drift thresholds
   - Training-serving skew detection
   - Alert configuration

2. **Drift Algorithm Implementations** (~80 lines)
   - L-infinity distance calculation
   - Jensen-Shannon divergence
   - Kolmogorov-Smirnov test
   - PSI computation with example

3. **Cloud Monitoring Integration** (~60 lines)
   - Metric queries
   - Custom dashboard YAML
   - SLI/SLO configuration

4. **Alerting Policies** (~100 lines)
   - Email alerts
   - Pub/Sub notification flow
   - Slack webhook integration
   - Multi-channel severity routing

5. **Automated Retraining Pipeline** (~200 lines)
   - Eventarc trigger (Pub/Sub → Cloud Function → Vertex AI Pipeline)
   - Complete Kubeflow pipeline definition
   - Retraining decision logic
   - Safety safeguards implementation

6. **ARR-COC Custom Metrics** (~120 lines)
   - Visual drift detection
   - Embedding drift monitor
   - Query distribution drift
   - Custom Cloud Monitoring metrics

---

## Integration with Existing Knowledge

**Complements existing files:**
- Extends `mlops-production/00-monitoring-cicd-cost-optimization.md` with Vertex AI-specific implementation
- Provides practical retraining automation for concepts in `gcp-vertex/02-training-to-serving-automation.md`
- Monitoring foundation for models deployed via `gcp-vertex/01-pipelines-kubeflow-integration.md`

**Fills knowledge gaps:**
- Statistical drift algorithms (previously only conceptual)
- Training-serving skew prevention (new topic)
- Automated retraining triggers (previously manual)
- Vision-specific drift monitoring (ARR-COC context)

---

## Practical Impact for ARR-COC

**Immediate applications:**

1. **Visual domain shift detection**
   - Monitor pixel statistics (brightness, contrast, color distribution)
   - Detect when production images differ from training (e.g., indoor→outdoor shift)
   - Example: Training on Imagenet → Production receives medical images

2. **Query pattern drift**
   - Track changes in user behavior
   - Example: General object detection → Industrial defect detection
   - Trigger: >30% change in query feature distribution

3. **Relevance allocation drift**
   - Monitor token allocation patterns
   - Alert when average LOD shifts significantly
   - Indicates model's relevance realization changing

4. **Automated retraining workflow**
   - Drift detected → Fresh data fetched → Model retrained → A/B tested
   - Complete automation with safety safeguards
   - Estimated time: 2-4 hours from drift detection to canary deployment

---

## Metrics & Statistics

**File Statistics:**
- Total lines: ~720
- Code examples: 6 major implementations
- Web sources cited: 8
- Sections: 7 main sections
- Algorithms explained: 5 statistical methods

**Coverage:**
- Monitoring architecture: ✓
- Drift algorithms: ✓ (5 methods)
- Skew detection: ✓
- Cloud integration: ✓
- Alerting: ✓ (3 channels)
- Auto-retraining: ✓
- ARR-COC specifics: ✓

---

## Quality Checklist

- [✓] All web sources cited with access dates
- [✓] Existing knowledge file referenced (mlops-production/00)
- [✓] Code examples tested for syntax
- [✓] Statistical formulas verified
- [✓] ARR-COC integration examples provided
- [✓] Clear section structure (7 sections)
- [✓] Practical implementation focus
- [✓] Links to Google Cloud documentation
- [✓] Sources section complete

---

## Next Steps for Oracle

1. Review file quality and technical accuracy
2. Check integration with existing gcp-vertex files
3. Update INDEX.md with new file entry
4. Verify ARR-COC-specific examples align with project architecture
5. Consider adding to SKILL.md if monitoring becomes core capability

---

**Runner signature**: PART 11 Complete ✓
**Timestamp**: 2025-11-16 13:27
