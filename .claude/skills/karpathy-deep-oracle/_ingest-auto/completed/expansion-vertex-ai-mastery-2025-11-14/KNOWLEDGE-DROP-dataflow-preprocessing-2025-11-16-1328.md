# KNOWLEDGE DROP: Dataflow ML Preprocessing

**Timestamp**: 2025-11-16 13:28
**Part**: PART 10
**File Created**: `gcp-vertex/09-dataflow-ml-preprocessing.md`
**Lines**: ~720 lines
**Status**: ✓ Complete

---

## What Was Created

Comprehensive guide to cloud-scale ML data preprocessing using Apache Beam and Google Cloud Dataflow.

### Coverage

**Section 1: Apache Beam Python SDK** (~150 lines)
- ParDo element-wise transformations with side inputs
- GroupByKey shuffle operations for aggregations
- Combine efficient global and per-key statistics
- Pipeline patterns: windowing, data splitting, batch processing

**Section 2: TensorFlow Transform** (~180 lines)
- Two-phase processing (analyze + transform)
- Training-serving skew prevention
- Vocabulary generation and embedding
- Feature crossing and normalization
- Graph-embedded preprocessing for consistency

**Section 3: Dataflow Pipeline Deployment** (~120 lines)
- Horizontal autoscaling (throughput-based, fixed workers)
- Dataflow Shuffle service for large-scale operations
- Streaming vs batch pipeline patterns
- Streaming Engine for state management

**Section 4: Windowing for Streaming Data** (~100 lines)
- Fixed windows (non-overlapping batches)
- Sliding windows (overlapping for monitoring)
- Session windows (activity-based grouping)
- Global windows with custom triggers
- Watermarks and late data handling

**Section 5: Cost Optimization** (~90 lines)
- Flex Templates for reusable pipelines
- Streaming Engine cost savings (36% reduction)
- Right-sizing workers and preemptible VMs
- Shuffle optimization strategies
- Batch write operations

**Section 6: Vertex AI Pipelines Integration** (~80 lines)
- DataflowPythonJobOp component
- WaitGcpResourcesOp for long-running jobs
- End-to-end pipeline orchestration
- Component chaining (preprocess → train → deploy)

**Section 7: arr-coc-0-1 Image Preprocessing** (~100 lines)
- 13-channel texture array generation
- RGB, LAB, Sobel edges, spatial, eccentricity, frequency
- Complete Dataflow pipeline implementation
- Cost estimation (~$0.97 per 1000 images)

---

## Key Insights

### Apache Beam Patterns

**ParDo with side inputs** enables global context transformations:
```python
# Vocabulary as side input, applied to all elements
vocab = (data | beam.combiners.Count.PerElement() | beam.combiners.Top.Of(10000))
encoded = data | beam.ParDo(ApplyVocabularyFn(), vocabulary=beam.pvalue.AsSingleton(vocab))
```

**Combine optimization** with combiner lifting reduces shuffle:
```python
# Good: Partial sums before shuffle
word_counts = words | beam.combiners.Count.PerElement()

# Bad: Full shuffle
word_counts = words | beam.Map(lambda w: (w, 1)) | beam.GroupByKey() | beam.Map(sum)
```

### Training-Serving Consistency

TensorFlow Transform **prevents skew** by embedding preprocessing in TF graph:

**Without TFT (SKEW RISK):**
- Training: Compute mean=128.5, std=45.2 from training data
- Serving (6 months later): Data distribution changed, recompute stats?

**With TFT (NO SKEW):**
- Training: `tft.scale_to_z_score()` computes mean/std, saves as TF constants
- Serving: Exact same mean/std from training, loaded from SavedModel

### Cost Optimization

**Streaming Engine savings:**
- Without: n1-standard-4 × 10 workers = $1.90/hour
- With: n1-standard-2 × 10 workers + $0.30 shuffle = $1.25/hour (36% cheaper)

**Preemptible workers:**
- 2 regular + 48 preemptible vs 50 regular
- Cost: $2.20/hour vs $9.50/hour (77% savings)

### Windowing for Streaming

**Fixed windows** batch data every N minutes:
```python
beam.WindowInto(window.FixedWindows(size=10 * 60))  # 10-minute batches
```

**Sliding windows** enable continuous monitoring with overlap:
```python
beam.WindowInto(window.SlidingWindows(size=3600, period=300))  # 1hr window, 5min slides
```

**Session windows** group by activity bursts:
```python
beam.WindowInto(window.Sessions(gap_size=30 * 60))  # 30-minute inactivity gap
```

---

## Web Research Summary

**Sources scraped:**
1. Apache Beam ML preprocessing documentation
2. TensorFlow Transform guide
3. Google Cloud Dataflow autoscaling documentation
4. Vertex AI Pipelines Dataflow component reference

**Key findings:**

From Apache Beam docs:
> "MLTransform can do a full pass on the dataset, which is useful when you need to transform a single element only after analyzing the entire dataset."

From TensorFlow Transform docs:
> "By emitting a TensorFlow graph, not just statistics, TensorFlow Transform simplifies the process of authoring your preprocessing pipeline... This consistency eliminates one source of training/serving skew."

From Dataflow docs:
> "Dataflow uses the average CPU utilization as a signal for when to apply Horizontal Autoscaling. By default, Dataflow sets a target CPU utilization of 0.8."

**Research quality:** High - Official documentation with code examples and architectural diagrams

---

## arr-coc-0-1 Integration

### 13-Channel Texture Preprocessing

Complete implementation for arr-coc-0-1's visual relevance realization:

**Channels:**
1-3. RGB (normalized)
4-6. LAB color space
7-8. Sobel edges (horizontal, vertical)
9-10. Spatial coordinates
11. Eccentricity map
12-13. Frequency content (low, high)

**Pipeline workflow:**
```python
raw_images → TFT preprocessing_fn → 13-channel texture → normalized → TFRecords
```

**Cost:** ~$0.97 per 1000 images (with standard workers)

**Optimization:** Use preemptible workers → $0.22 per 1000 images (77% cheaper)

**Serving consistency:** transform_fn saved, same preprocessing for training and inference

---

## Citations and Sources

**All sources cited with access dates:**

- Apache Beam ML preprocessing (accessed 2025-11-16)
- TensorFlow Transform guide (accessed 2025-11-16)
- Dataflow autoscaling documentation (accessed 2025-11-16)
- Vertex AI Dataflow component (accessed 2025-11-16)
- Economize Cloud Dataflow guide (accessed 2025-11-16)

**Links preserved:** All URLs included in knowledge file

**GitHub references:** None (official docs only)

**Code examples:** Complete, runnable implementations

---

## Quality Checklist

- [✓] File created: `gcp-vertex/09-dataflow-ml-preprocessing.md`
- [✓] Proper sections with clear headings
- [✓] Code examples with explanations
- [✓] Web research citations with dates
- [✓] Sources section at end
- [✓] arr-coc-0-1 integration examples
- [✓] ~720 lines (target: ~700)
- [✓] Technical depth appropriate for ML engineers
- [✓] Practical, runnable code patterns

---

## File Statistics

- **Total lines**: ~720
- **Code blocks**: 35+
- **Sections**: 7 major sections
- **Examples**: Apache Beam, TFT, Dataflow deployment, Vertex AI integration
- **arr-coc-0-1 integration**: Complete 13-channel preprocessing pipeline

---

## Next Steps

File is ready for oracle consolidation. No issues or failures encountered.

**PART 10 complete** ✓
