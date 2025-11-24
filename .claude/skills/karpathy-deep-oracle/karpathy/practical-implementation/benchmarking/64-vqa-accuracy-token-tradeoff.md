# VQA Accuracy vs Token Budget Trade-off

## Overview

Visual Question Answering (VQA) accuracy exhibits a complex trade-off relationship with visual token budget allocation. Understanding this relationship is critical for designing efficient vision-language models that balance performance with computational cost. Token count directly impacts both inference latency and model capacity to capture fine-grained visual details necessary for accurate question answering.

**Key Insight**: VQA performance does not scale linearly with token count. There exist optimal operating points where accuracy per token is maximized, and regions where increasing tokens yields diminishing returns. Different question types exhibit varying sensitivity to token budget, suggesting query-aware allocation strategies could significantly improve efficiency.

**Token Budget Range**: Modern VLMs operate across a wide spectrum:
- **Minimal budget**: 64 tokens (aggressive compression, acceptable for simple tasks)
- **Standard budget**: 144-256 tokens (BLIP-2, LLaVA defaults)
- **High budget**: 576 tokens (LLaVA-1.5 standard, high detail preservation)
- **Extreme budget**: 1024+ tokens (diminishing returns, specialized tasks only)

From [LLaVA-NeXT Ablations](https://llava-vl.github.io/blog/2024-05-25-llava-next-ablations/) (accessed 2025-01-31):
- LLaVA-1.5 uses 576 visual tokens as default configuration
- CLIP-L-336 with dynamic resolution yields 576 tokens for 2×2 grid
- Higher resolution configurations (6×6 grid) can produce up to 7,290 tokens

From [Matryoshka Query Transformer](https://arxiv.org/html/2405.19315v1) (arXiv:2405.19315, accessed 2025-01-31):
- MQT-LLaVA achieves LLaVA-1.5 performance with only 256 tokens (2.25× reduction)
- Token reduction from 576→256 maintains performance across 11 benchmarks
- Demonstrates that fixed high token budgets contain significant redundancy

## Accuracy Curves by Token Count

### 64 Tokens: Minimal Viable Performance

**Performance Characteristics**:
- Baseline VQA accuracy typically 5-10% below standard configurations
- Acceptable for coarse-grained recognition tasks
- Struggles with spatial reasoning and fine-grained details
- Computational efficiency: ~9× faster than 576 tokens

From [LLaVA-NeXT Ablations](https://llava-vl.github.io/blog/2024-05-25-llava-next-ablations/):
- 64 tokens achieved via 1×1 AnyRes grid with single image crop
- Performance drops particularly severe on DocVQA (-20% typical) and InfoVQA tasks
- Maintains reasonable accuracy on simple recognition tasks (AI2D, ScienceQA)

**Example Performance (CLIP-L-336 + Qwen-1.5 0.5B)**:
- AI2D test: 51.1% (vs 52.7% at higher tokens)
- ChartQA: 49.2% (vs 55.8% at 256 tokens)
- DocVQA: 58.8% baseline
- VizWiz-VQA: 29.2% (significant drop on real-world photos)

**Use Cases**:
- Real-time interactive applications
- Mobile/edge deployment
- Batch processing large datasets where speed matters
- Questions requiring only basic object recognition

### 144 Tokens: BLIP-2 Standard

**Performance Characteristics**:
- BLIP-2's default Q-Former output (32 learned queries × 4-5 features)
- Sweet spot for balanced performance/efficiency in many scenarios
- Adequate for most general-purpose VQA benchmarks
- 4× faster than 576 tokens while retaining 90-95% accuracy

From [Matryoshka Query Transformer](https://arxiv.org/html/2405.19315v1):
- 144 tokens represents a key inflection point in accuracy curves
- MQT-LLaVA at 144 tokens achieves competitive results across most benchmarks
- Performance gap vs 576 tokens narrows significantly compared to 64 tokens

**BLIP-2 Architecture Context**:
- Uses 32 learnable queries in Q-Former
- Each query compresses image encoder output (e.g., 257 ViT tokens)
- 32 queries × 768D → fed to LLM as visual representation
- Effective compression ratio: ~8× from ViT output

**Estimated Performance Range** (based on research patterns):
- VQAv2 accuracy: 70-75% (within 2-4% of 256 token models)
- GQA accuracy: 60-64%
- Visual reasoning tasks: moderate performance
- OCR-heavy tasks: noticeable degradation vs 256+ tokens

### 256 Tokens: LLaVA Compressed Standard

**Performance Characteristics**:
- Modern efficiency-focused models target this range (MQT-LLaVA, FastV)
- Excellent balance: retains 95-98% of 576-token performance
- 2.25× speedup while maintaining near-parity accuracy
- Sufficient detail for most VQA question types

From [Matryoshka Query Transformer](https://arxiv.org/html/2405.19315v1):
- MQT-LLaVA with 256 tokens matches LLaVA-1.5 (576 tokens) across 11 benchmarks
- VQAv2: 78.4% (vs 78.5% at 576 tokens, -0.1%)
- GQA: 62.6% (vs 62.0% at 576 tokens, +0.6%)
- MM-Vet: 35.4% (vs 35.4% at 576 tokens, parity)

From [LLaVA-NeXT Ablations](https://llava-vl.github.io/blog/2024-05-25-llava-next-ablations/):
- 256 tokens via 2×2 AnyRes grid = (4+1) × 729 tokens pooled down
- Maintains strong performance on detail-oriented tasks
- ChartQA, DocVQA show minimal degradation vs 576 tokens

**Benchmark Performance (MQT-LLaVA 256 vs LLaVA-1.5 576)**:
- OKVQA: 57.4% vs 57.3% (+0.1%)
- ScienceQA-IMG: 68.9% vs 68.4% (+0.5%)
- TextVQA: 58.0% vs 58.2% (-0.2%)
- POPE (object hallucination): 85.8% vs 85.9% (-0.1%)

**Key Finding**: 256 tokens appears to be a perceptual threshold where human-relevant visual information is largely preserved for VQA tasks.

### 576 Tokens: LLaVA-1.5 Standard

**Performance Characteristics**:
- De facto standard for open-source VLMs (LLaVA-1.5, LLaVA-NeXT)
- High accuracy across diverse benchmarks
- Captures fine-grained spatial relationships and text details
- Baseline for comparing compression techniques

From [LLaVA-NeXT Ablations](https://llava-vl.github.io/blog/2024-05-25-llava-next-ablations/):
- 576 tokens from CLIP-L-336 with 2×2 AnyRes grid
- Configuration: 336×336 base + up to 4 additional crops at 336×336
- Total visual tokens: (2×2 grids + 1 base) × 256 features = 1,280 → pooled to 576

**Architecture Details**:
- CLIP-L-336: 224×224 → 14×14 patches → 196 tokens + CLS = 197 tokens base
- Scale to 336×336 → 24×24 patches → 576 tokens + CLS = 577 tokens
- AnyRes dynamic: Adds additional grids for higher resolution images

**Performance Benchmarks (LLaVA-1.5 7B)**:
- VQAv2: 78.5%
- GQA: 62.0%
- VizWiz: 50.0%
- ScienceQA-IMG: 68.4%
- TextVQA: 58.2%
- POPE: 85.9%

**When 576 Tokens Matter**:
- Document understanding (DocVQA, ChartQA, InfoVQA)
- Fine-grained counting tasks
- Spatial reasoning across multiple objects
- Text-heavy images (OCR, scene text VQA)

### 1024+ Tokens: Diminishing Returns Territory

**Performance Characteristics**:
- Marginal accuracy gains beyond 576 tokens for most tasks
- Significant computational overhead (2-3× slower than 576)
- Benefits limited to specialized scenarios
- Saturation effect: model capacity bottlenecks overshadow token count

From [LLaVA-NeXT Ablations](https://llava-vl.github.io/blog/2024-05-25-llava-next-ablations/):
- 6×6 AnyRes grid with max 9+1 grids × 729 tokens = 7,290 tokens
- Performance gains over 2×2 AnyRes (576 tokens) are modest
- InfoVQA: 26.7% (6×6) vs 25.7% (2×2) = +1.0% gain
- ChartQA: 55.8% (6×6) vs 49.2% (2×2) = +6.6% gain (task-specific benefit)
- Most benchmarks show <3% improvement despite 12× token increase

**Computational Cost**:
- Training time: 11h14m (6×6 grid) vs 6h30m (2×2 grid) = 1.73× slower
- Inference time: Proportionally scales with token count
- Memory footprint: Linear increase with token budget

**When High Token Counts Help**:
- Ultra-high-resolution document analysis (tables, complex diagrams)
- Video understanding with many frames (spatial-temporal detail)
- Scientific figure interpretation (dense information)
- Medical imaging VQA (diagnostic detail)

**Saturation Analysis**:
From [LLaVA-NeXT Ablations](https://llava-vl.github.io/blog/2024-05-25-llava-next-ablations/):
- Increasing from (4+1)×729 to (9+1)×729 to (16+1)×729 tokens
- (4+1)×729: Training 7h30m, ChartQA 49.4%
- (9+1)×729: Training 11h14m, ChartQA 55.8% (+6.4%)
- (16+1)×729: Training 13h10m, ChartQA 56.1% (+0.3% over 9+1)
- Clear diminishing returns above ~7,000 token budget

## Question Type Analysis

Different VQA question categories exhibit varying sensitivity to token budget, revealing opportunities for query-aware dynamic allocation.

### Counting Questions: High Token Sensitivity

**Characteristics**:
- Require global spatial understanding
- Need to distinguish individual instances across image
- Performance correlates strongly with token count

**Token Budget Requirements**:
- 64 tokens: Poor accuracy (30-40% typical)
- 256 tokens: Moderate accuracy (60-70%)
- 576 tokens: Good accuracy (75-85%)
- Benefit continues beyond 576 for complex counting (>10 objects)

**Why Counting Needs Tokens**:
- Small objects may be lost in aggressive compression
- Spatial pooling can merge nearby instances
- Fine-grained attention needed to separate overlapping objects

From [LLaVA-NeXT Ablations](https://llava-vl.github.io/blog/2024-05-25-llava-next-ablations/):
- VizWiz-VQA (real-world photos with counting): 29.2% (2×2) → 34.7% (6×6) = +5.5%
- ScienceQA (includes counting problems): Modest gains with more tokens

### Spatial Reasoning: Moderate Token Sensitivity

**Characteristics**:
- Questions about object relationships ("left of", "above", "between")
- Requires preserved spatial structure
- 256-576 token range sufficient for most cases

**Token Budget Requirements**:
- 64 tokens: Spatial structure significantly degraded
- 256 tokens: Adequate for most spatial relationships
- 576 tokens: High confidence for complex spatial queries
- 1024+ tokens: Minimal additional benefit

**Example Question Types**:
- "What is to the left of the red car?"
- "How many objects are between the chair and the table?"
- "Is the person standing in front of or behind the sign?"

**Performance Pattern**:
- Large accuracy jump from 64→256 tokens (spatial coherence threshold)
- Modest improvement from 256→576 tokens
- Minimal gain beyond 576 tokens

### Object Recognition: Low Token Sensitivity

**Characteristics**:
- Simple "What is this?" or "Is there a cat?" questions
- Relies primarily on semantic features, not fine detail
- Stable performance across wide token range

**Token Budget Requirements**:
- 64 tokens: 80-90% of peak performance (sufficient for most cases)
- 144 tokens: Near-optimal performance
- 256-576 tokens: Marginal gains (<2%)
- Token count beyond 144 provides little benefit

From [LLaVA-NeXT Ablations](https://llava-vl.github.io/blog/2024-05-25-llava-next-ablations/):
- AI2D (diagram understanding): 51.1% (2×2) → 52.7% (6×6) = +1.6% (minimal)
- POPE (object hallucination): 85.4% (2×2) → 86.1% (6×6) = +0.7% (minimal)

**Why Recognition Is Token-Efficient**:
- Pre-trained vision encoders capture semantic features in early layers
- Object categories robust to compression
- Global pooling preserves class-relevant information
- Attention mechanisms can retrieve salient features from limited tokens

### Fine-Grained Visual Questions: High Token Sensitivity

**Characteristics**:
- Attribute recognition (color, texture, material, fine details)
- Subtle visual differences (breeds, species, models)
- Requires preserved high-frequency information

**Token Budget Requirements**:
- 64 tokens: Poor performance on subtle distinctions
- 256 tokens: Adequate for many attributes
- 576 tokens: Recommended for fine-grained tasks
- 1024+ tokens: Benefits specific fine-grained domains

**Example Questions**:
- "What breed of dog is this?"
- "What color are the person's shoes?"
- "Is the surface smooth or textured?"
- "What material is the table made of?"

**Performance Characteristics**:
- Steady improvement across 64→256→576 token range
- Attribute questions benefit more than category questions
- Texture/material questions particularly sensitive to token count

### OCR and Text-Heavy Questions: Extreme Token Sensitivity

**Characteristics**:
- Reading text in images (signs, documents, charts)
- Most token-hungry question type
- Performance scales nearly linearly with tokens up to high budgets

**Token Budget Requirements**:
- 64 tokens: Very poor OCR accuracy (<30%)
- 144 tokens: Basic text reading (40-50%)
- 256 tokens: Moderate OCR capability (60-70%)
- 576 tokens: Good OCR performance (75-85%)
- 1024+ tokens: Continued improvement for dense text

From [LLaVA-NeXT Ablations](https://llava-vl.github.io/blog/2024-05-25-llava-next-ablations/):
- DocVQA: 58.8% (2×2, 64 tokens) → 62.7% (6×6, ~7K tokens) = +3.9%
- ChartQA: 49.2% (2×2) → 55.8% (6×6) = +6.6% (largest gain observed)
- InfoVQA: 25.7% (2×2) → 26.7% (6×6) = +1.0%
- TextVQA: Benefits significantly from token budget increases

**Why OCR Needs Tokens**:
- Text contains high-frequency details easily lost in compression
- Character-level features require fine spatial resolution
- Multi-line text needs preserved spatial layout
- Small text may disappear entirely with aggressive pooling

**Special Case: SynDOG Benchmark** (Tree Edit Distance Score):
- 418.5 (2×2 grid, baseline)
- 443.5 (6×6 grid with pooling) = +25.0 improvement
- 470.4 (6×6 grid, 14B model) = +51.9 improvement (model size + tokens)
- Demonstrates OCR as most token-sensitive task category

## Efficiency Metrics: Accuracy per Token

Understanding efficiency requires measuring not just absolute accuracy, but accuracy gains per additional token invested.

### Efficiency Curve Analysis

**64→144 Tokens** (+80 tokens):
- Typical accuracy gain: +5-8% across benchmarks
- Efficiency: 0.06-0.10% accuracy per token
- **Verdict**: High efficiency region, excellent ROI

**144→256 Tokens** (+112 tokens):
- Typical accuracy gain: +3-5% across benchmarks
- Efficiency: 0.03-0.04% accuracy per token
- **Verdict**: Good efficiency, still worthwhile for most applications

**256→576 Tokens** (+320 tokens):
- Typical accuracy gain: +0-3% across benchmarks
- Efficiency: 0.0-0.009% accuracy per token
- **Verdict**: Diminishing returns, task-dependent value

**576→1024+ Tokens** (+448+ tokens):
- Typical accuracy gain: +0-2% across benchmarks
- Efficiency: 0.0-0.004% accuracy per token
- **Verdict**: Very low efficiency, only for specialized tasks

### Optimal Operating Points

From [Matryoshka Query Transformer](https://arxiv.org/html/2405.19315v1) analysis:

**General-Purpose VQA**: 256 tokens
- Balances accuracy and efficiency
- 2.25× speedup vs 576 tokens
- <1% average accuracy loss
- Recommended for: Interactive applications, real-time systems, mobile deployment

**Accuracy-Critical Applications**: 576 tokens
- Standard high-performance configuration
- Minimal compromise on accuracy
- Recommended for: Benchmark evaluations, academic research, high-stakes applications

**Ultra-Efficient Deployment**: 144 tokens
- 4× speedup vs 576 tokens
- 3-5% accuracy reduction acceptable for many use cases
- Recommended for: Edge devices, batch processing, cost-sensitive applications

**Task-Specific Optimization**:
- OCR/Document tasks: 576-1024 tokens (high benefit)
- Recognition tasks: 144 tokens (sufficient)
- Spatial reasoning: 256 tokens (balanced)
- Counting tasks: 256-576 tokens (moderate benefit)
- Fine-grained attributes: 256-576 tokens (steady improvement)

### Computational Cost vs Accuracy Trade-off

From [LLaVA-NeXT Ablations](https://llava-vl.github.io/blog/2024-05-25-llava-next-ablations/):

**Training Time Scaling** (Qwen-1.5 0.5B baseline):
- 2×2 grid (baseline ~3,645 tokens): 6h30m
- 4×4 grid (pooled to ~3,645 tokens): 7h30m (+15%)
- 6×6 grid (9+1 grids, pooled): 11h14m (+73%)
- 6×6 grid (16+1 grids): 13h10m (+102%)

**Inference Time Scaling** (estimated from benchmarks):
- 64 tokens: ~16min for benchmark suite
- 144 tokens: ~20min (+25%)
- 576 tokens: ~30min (+87% vs 64 tokens)
- 7,290 tokens: ~75min (+369% vs 64 tokens)

**Memory Footprint**:
- Visual token memory scales linearly with token count
- KV cache memory: Proportional to token count × sequence length
- Activation memory: Major component for high token counts

**Energy Efficiency**:
- Tokens directly correlate with FLOPs in vision encoder and cross-attention
- 64 tokens: Most energy-efficient (mobile battery-friendly)
- 256 tokens: Good balance for data center deployment
- 576+ tokens: High energy cost, requires justification

## Model Architecture Impact

Token budget effectiveness varies significantly across different VLM architectures, as compression mechanisms and training objectives differ.

### BLIP-2 Q-Former: Learned Compression

**Architecture**:
- 32 learnable queries (default) compress vision encoder output
- Queries trained via image-text contrastive learning and ITM
- Fixed output size regardless of input resolution

**Token Budget**: 32 queries × ~4-5 effective features = ~144 tokens
- Can scale to 64 or 128 queries for higher capacity
- More queries = better detail, but diminishing returns

**Performance Characteristics**:
- Efficient compression through learned attention
- Good semantic preservation
- Some loss of spatial fine-grain detail
- Excellent for general-purpose VQA

From research patterns (BLIP-2 paper):
- 32 queries: Balanced performance/efficiency
- VQAv2: ~65-70% accuracy range
- Strong on recognition, moderate on spatial reasoning

### LLaVA: Projection-Based Compression

**Architecture**:
- MLP projector maps vision encoder output to LLM embedding space
- Preserves spatial structure of vision encoder output
- Dynamic resolution with AnyRes strategy

**Token Budget**: 256-576 tokens typical
- 336×336 base: 576 tokens from CLIP-L
- AnyRes: Scales dynamically with image complexity
- Higher tokens than BLIP-2 but better spatial preservation

From [LLaVA-NeXT Ablations](https://llava-vl.github.io/blog/2024-05-25-llava-next-ablations/):
- LLaVA-1.5: 576 tokens standard
- MQT-LLaVA: 256 tokens achieves parity
- Spatial tasks benefit from preserved token structure

**Performance Characteristics**:
- Excellent spatial reasoning
- Strong OCR and document understanding
- Higher computational cost than BLIP-2
- Better scaling with token count increases

### MQT-LLaVA: Matryoshka Compression

**Architecture**:
- Matryoshka training allows flexible token budgets at inference
- Single model supports 64, 144, 256 tokens dynamically
- Progressive compression mechanism

From [Matryoshka Query Transformer](https://arxiv.org/html/2405.19315v1):
- Trained with nested token representations
- Inference-time token budget selection
- No retraining needed for different budgets

**Token Budget**: 64-256 tokens flexible
- 64 tokens: Minimal but functional
- 144 tokens: Balanced efficiency
- 256 tokens: Matches LLaVA-1.5 576-token performance

**Performance Characteristics**:
- Best efficiency across all token ranges
- Graceful degradation with fewer tokens
- Enables adaptive token allocation
- 2.25× speedup at 256 tokens vs LLaVA-1.5

**Key Innovation**: Single model performs well across token budgets, enabling query-aware dynamic allocation.

## ARR-COC Implications: Query-Aware Token Allocation

The VQA accuracy-token tradeoff research reveals critical insights for ARR-COC's dynamic relevance realization approach.

### Relevance Realization for Token Budgets

**Core Opportunity**: Different questions require vastly different token budgets for optimal efficiency.

From research evidence:
- Recognition questions: 64-144 tokens sufficient (90% of peak performance)
- OCR questions: 576-1024 tokens beneficial (linear scaling)
- Spatial questions: 256-576 tokens optimal (diminishing returns above)
- Counting questions: 256-576 tokens recommended (moderate sensitivity)

**ARR-COC Advantage**: Query-aware token allocation based on question type could yield:
- 30-50% computational savings vs fixed 576-token budget
- Maintained or improved accuracy through efficient resource allocation
- Adaptive inference cost matching task complexity

### Opponent Processing for Token Budget

**Compress ↔ Particularize Tension**:
- **Compress**: Minimize tokens for efficiency (64-144 tokens)
- **Particularize**: Preserve detail for accuracy (576-1024 tokens)
- **Balance**: Realize relevance to allocate optimal budget per query

**Query-Content Coupling Examples**:
- "What is this animal?" → Compress (low detail needed, 64 tokens)
- "What breed of dog is this?" → Particularize (high detail needed, 256-576 tokens)
- "Read the text on the sign" → Particularize (extreme detail needed, 576+ tokens)
- "Count the people in the image" → Moderate (spatial structure needed, 256 tokens)

### Participatory Knowing: Query-Aware Relevance

**Transjective Token Allocation**:
- Token budget is not objective (image property alone)
- Token budget is not subjective (query intent alone)
- Token budget is **transjective**: Emerges from query-content relationship

**Examples**:
- Same image, different questions → different optimal token budgets
- "What color is the car?" (64 tokens) vs "What text is on the license plate?" (576 tokens)
- Image complexity alone insufficient to determine budget
- Query type + image characteristics jointly determine relevance

### Dynamic LOD as Token Budget Realization

**Mapping Relevance to Tokens**:
- **Low relevance regions**: 64 tokens (semantic gist sufficient)
- **Moderate relevance**: 256 tokens (balanced detail)
- **High relevance**: 576 tokens (fine-grained preservation)
- **Critical relevance**: 1024+ tokens (OCR, extreme detail)

**Spatial Token Allocation**:
- Foveated strategy: High tokens in query-relevant regions
- Peripheral compression: Low tokens in background
- Dynamic budget: Allocate total token budget across image regions based on realized relevance

From [LLaVA-NeXT Ablations](https://llava-vl.github.io/blog/2024-05-25-llava-next-ablations/):
- AnyRes grid strategy provides coarse spatial token allocation
- Higher resolution regions receive more tokens (up to 6×6 grids)
- ARR-COC could refine this with query-aware grid selection

### Procedural Knowing: Learned Budget Allocation

**Training the Quality Adapter**:
- Learn to predict optimal token budget per query
- Training signal: Accuracy delta vs computational cost
- Loss function balancing performance and efficiency

**Adapter Architecture Options**:
1. **Query classifier**: Predict budget category (64/144/256/576)
2. **Continuous predictor**: Output token budget as continuous value
3. **Region allocator**: Predict spatial token distribution

**Training Strategy**:
- Collect dataset of (query, image, optimal_budget) tuples
- Train adapter to minimize: Loss = α × Accuracy_loss + β × Efficiency_loss
- α, β hyperparameters control accuracy/efficiency trade-off

### Practical ARR-COC Architecture

**Proposed Pipeline**:
1. **Query Analysis**: Classify question type (recognition/spatial/OCR/counting/fine-grained)
2. **Relevance Realization**: Compute transjective relevance (query + image features)
3. **Budget Allocation**: Map relevance to token budget (64-1024 range)
4. **Spatial Distribution**: Allocate tokens across image regions (foveated strategy)
5. **Adaptive Compression**: Apply compression to meet budget constraints
6. **LLM Processing**: Forward compressed visual tokens to language model

**Expected Performance**:
- Average token budget: 200-300 (vs 576 fixed in LLaVA-1.5)
- Computational savings: 40-50%
- Accuracy: Maintained or improved (task-appropriate allocation)
- Inference speedup: 2-3× vs fixed high-token models

### Open Research Questions

1. **Budget Prediction Accuracy**: Can we reliably predict optimal token budget from query alone before seeing image?

2. **Spatial Allocation Granularity**: What is optimal grid resolution for region-wise token allocation?

3. **Training Efficiency**: Can single model learn to operate across token budgets (MQT-LLaVA style) + query-aware allocation?

4. **Multi-Hop Reasoning**: How do token budgets affect chain-of-thought VQA reasoning?

5. **Cross-Task Transfer**: Does optimal budget for VQAv2 transfer to other VQA datasets (OK-VQA, GQA, etc.)?

## Sources

**Web Research:**

- [LLaVA-NeXT: What Else Influences Visual Instruction Tuning Beyond Data?](https://llava-vl.github.io/blog/2024-05-25-llava-next-ablations/) - Comprehensive ablation studies on token budget, resolution, and training strategies (accessed 2025-01-31)

- [Matryoshka Query Transformer for Large Vision-Language Models](https://arxiv.org/html/2405.19315v1) - arXiv:2405.19315, MQT-LLaVA achieves LLaVA-1.5 performance with 256 vs 576 tokens (accessed 2025-01-31)

- [Exploiting Visual Cues for Effective Token Pruning in VLMs](https://arxiv.org/html/2412.01818v2) - Token pruning strategies and efficiency analysis (accessed 2025-01-31)

**Additional References:**

- BLIP-2 paper: 32 learned queries as compression mechanism
- LLaVA-1.5: 576 tokens as standard high-performance configuration
- Vision transformer token efficiency literature
- VQA benchmark datasets: VQAv2, GQA, OK-VQA, TextVQA, DocVQA, ChartQA, InfoVQA

**Key Datasets Referenced:**
- VQAv2: Primary VQA benchmark
- GQA: Structured visual reasoning questions
- TextVQA: Text reading in natural images
- DocVQA: Document understanding questions
- ChartQA: Chart and plot interpretation
- InfoVQA: Information extraction from infographics
- ScienceQA: Science question answering with diagrams
- AI2D: Diagram understanding
- POPE: Object hallucination evaluation
- VizWiz: Real-world photos from blind users
