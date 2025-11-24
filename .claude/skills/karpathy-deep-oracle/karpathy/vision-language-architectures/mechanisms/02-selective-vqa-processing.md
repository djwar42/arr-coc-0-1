# Selective VQA Processing: Question-Guided Vision

## Overview - Question-Guided Vision (60 lines)

Visual Question Answering (VQA) systems don't process images uniformly - they selectively attend to regions relevant to the question. This selective processing is fundamental to VQA's ability to answer diverse questions about the same image efficiently.

**Core Principle**: The question guides which parts of the image receive computational focus. Asking "What color is the car?" focuses on different regions than "How many people are in the photo?"

**Why Selective Processing Matters**:
- **Efficiency**: Don't waste computation on irrelevant image regions
- **Accuracy**: Focus resources on question-relevant features
- **Interpretability**: Attention maps show what the model "looked at"
- **Reliability**: Knowing when to abstain from answering improves trust

**Three Levels of Selection**:

1. **Spatial Selection**: Which image regions to attend to
2. **Feature Selection**: Which visual attributes matter (color vs shape vs count)
3. **Confidence Selection**: Whether to answer at all (selective prediction)

**Historical Context**:

Early VQA models (2015-2016) processed entire images uniformly, leading to:
- Wasted computation on background regions
- Poor performance on questions requiring fine-grained focus
- No mechanism to handle unanswerable questions

Modern VQA (2018-2025) uses question-conditioned attention to:
- Dynamically allocate visual processing based on question type
- Attend to multiple regions for multi-hop reasoning
- Learn when to abstain from answering (selective prediction)

**Key Innovation**: The question isn't just input for the final answer - it actively shapes how visual features are extracted and combined throughout the entire visual processing pipeline.

**Relationship to Other VLM Concepts**:
- Related to query-conditioned attention (mechanisms/00-query-conditioned-attention.md)
- Implements task-driven encoding principles (mechanisms/01-task-driven-encoding.md)
- Foundation for multi-pass transformers (mechanisms/03-multi-pass-transformers.md)

From [Question Type Guided Attention in Visual Question Answering](https://arxiv.org/abs/1804.02088) (Shi et al., 2018):
> "Visual Question Answering (VQA) requires integration of feature maps with drastically different structures and focus of the correct regions."

From [Improving Selective Visual Question Answering by Learning from Your Peers](https://arxiv.org/abs/2306.08751) (Dancette et al., 2023, CVPR):
> "The option to abstain, also called Selective Prediction, is highly relevant when deploying systems to users who must trust the system's output."

## Selection Mechanisms (90 lines)

### 1. Spatial Attention

**Bottom-Up + Top-Down Attention**:

The foundational approach combines two complementary pathways:

- **Bottom-up**: Object detector (Faster R-CNN) extracts salient regions
  - Produces ~36-100 region proposals per image
  - Each region has visual features (2048-d from ResNet)
  - Pre-attentive: doesn't know the question yet

- **Top-down**: Question guides which regions receive focus
  - Computes attention scores: `score = softmax(W * [visual_feature, question_embedding])`
  - Weights regions by relevance to question
  - Question-aware: "What color?" focuses on object regions, "How many?" scans globally

**Implementation Pattern**:
```
visual_regions = FasterRCNN(image)  # [N, 2048] for N regions
question_emb = LSTM(question)        # [1, 512]

# Compute attention
attention_scores = softmax(
    Linear([visual_regions, question_emb.expand(N)])  # [N,]
)

# Weighted visual features
attended_visual = sum(attention_scores * visual_regions)  # [2048]

# Answer prediction
answer = Classifier([attended_visual, question_emb])
```

**Question Type Guided Attention (QTA)**:

From [Shi et al., 2018](https://arxiv.org/abs/1804.02088):
- Uses question type (counting, color, activity, etc.) to balance bottom-up vs top-down features
- Different question types need different visual processing strategies:
  - **Counting**: Global attention, scan entire image
  - **Color**: Fine-grained attention on specific objects
  - **Activity**: Broader context around people
  - **Object recognition**: Tight focus on object regions

**Performance Gains** (from TDIUC dataset):
- +5% accuracy on Activity Recognition, Utility, Counting
- +3% overall accuracy on state-of-art MCB model
- Systematic improvements across 12 question type categories

### 2. Feature-Level Selection

Beyond spatial location, VQA models select which visual attributes to emphasize:

**Multi-Modal Feature Alignment**:
- **Visual features**: Color, texture, shape, spatial relationships
- **Linguistic features**: Question type, key nouns, verbs, modifiers
- **Alignment**: Match visual attributes to question requirements

Example: "What color is the car?"
- Emphasize color channels in visual features
- De-emphasize shape/texture information
- Focus on object-level features (not scene-level)

**Co-Attention Mechanisms**:

From [Hierarchical Question-Image Co-Attention](https://proceedings.neurips.cc/paper/2016/file/9dcb88e0137649590b755372b040afad-Paper.pdf) (Lu et al., NeurIPS 2016):
- Jointly attend to both question words and image regions
- Question attention: Which words matter most?
- Image attention: Which regions matter most?
- Co-reasoning: Iteratively refine both attentions together

**Three levels of co-attention**:
1. **Word-level**: Attend to specific question words ("red", "car")
2. **Phrase-level**: Attend to question phrases ("red car", "on the left")
3. **Question-level**: Overall question intent guides image attention

### 3. Confidence-Based Selection (Selective Prediction)

**The Abstention Problem**:

Real-world VQA systems face:
- Out-of-distribution images (not seen during training)
- Ambiguous or unanswerable questions
- Image quality issues (blur, occlusion)

**Solution: Selective Prediction**

From [Learning from Your Peers](https://arxiv.org/abs/2306.08751) (Dancette et al., CVPR 2023):

**Problem**: VQA models confidently answer even when they shouldn't
- Softmax confidences are poorly calibrated
- Models answer <5% of questions at 1% error rate when faced with 10% OOD examples

**Learning from Your Peers (LYP) Approach**:
1. Train multiple models on different data subsets
2. Use peer predictions as targets for learning selection functions
3. Learn which examples are easy/hard to generalize

**Performance Results**:
- **In-distribution**: 32.92% coverage at 1% risk (C@1%) - doubles previous best of 15.79%
- **Mixed ID/OOD**: 25.38% C@1% vs <5% for softmax-based abstention
- Works across different architectures and scales

**Key Insight**: Examples where peer models disagree are likely hard/OOD - the model should abstain.

## VQA-Specific Patterns (70 lines)

### Multi-Hop Reasoning

Complex questions require attending to multiple image regions sequentially:

**Example**: "What is the person to the left of the red car doing?"

**Processing Steps**:
1. **Hop 1**: Locate red car → attend to car regions
2. **Hop 2**: Find person to the left → spatial reasoning from car location
3. **Hop 3**: Identify activity → attend to person's pose/context
4. **Synthesis**: Combine information across hops

From recent research on [Multi-Hop Graph Reasoning](https://dl.acm.org/doi/10.1145/3724125) (2025):
- Constructs knowledge graphs from visual scene
- Performs multi-hop traversal guided by question
- Integrates external knowledge (KB-VQA)

**Spatial Reasoning Patterns**:

From [ReasonVQA benchmark](https://arxiv.org/html/2507.16403v1) (2025):
- Questions requiring structural knowledge and multi-hop reasoning
- Spatial relationships: "to the left of", "behind", "between"
- Compositional reasoning: combine multiple visual cues

### Question-Image Alignment

**Grounding Linguistic Concepts to Visual Objects**:

From [Guiding Visual Question Answering With Attention Priors](https://openaccess.thecvf.com/content/WACV2023/papers/Le_Guiding_Visual_Question_Answering_With_Attention_Priors_WACV_2023_paper.pdf) (Le et al., WACV 2023):

**Grounding-based Attention Prior (GAP)**:
- Extract linguistic-vision associations from query-image pairs
- Use grounding annotations to guide attention during training
- Improve attention alignment without expensive manual annotations

**Two-Way Co-Attention**:
- Question → Image: What visual regions does each question word activate?
- Image → Question: What question words are relevant to each image region?
- Bidirectional refinement improves alignment

**Attention Visualization**:

From research on [attention maps visualization](https://www.researchgate.net/figure/Attention-maps-visualized-across-question-types-Image-attention-seems-mostly-plausible_fig1_357385234):
- Image attention seems mostly plausible across models
- Different question types show distinct attention patterns
- Visualization helps debug model reasoning

### Selective Residual Learning

From [Selective Residual Learning for Visual Question Answering](https://www.sciencedirect.com/science/article/abs/pii/S0925231220304859) (Hong et al., 2020):

**SelRes Module**: Self-attention-based VQA learning
- Encourages model to learn more important question-relevant features
- Selective residual connections emphasize useful pathways
- Improves compositional reasoning

### Dynamic Attention Mechanisms

Modern VQA uses multiple attention strategies:

1. **Channel Attention**: Which feature channels are relevant?
2. **Spatial Attention**: Which image locations matter?
3. **Temporal Attention**: For video VQA - which frames/moments?

From [Dynamic Attention Networks](https://www.nature.com/articles/s41598-022-21149-9) (Miao et al., 2022):
- Combines channel and spatial attention
- Applied to memory networks for VQA
- First application of multi-attention to memory networks

## Visualization & Analysis (50 lines)

### Interpreting Attention Maps

**What Attention Maps Reveal**:

From studies on [VQA attention visualization](https://dl.acm.org/doi/10.1145/3689236.3691498) (2024):
- Which image regions the model focuses on
- Whether attention aligns with human intuition
- Failure modes: spurious correlations, language bias

**Common Patterns**:
- **Counting questions**: Distributed attention across similar objects
- **Color questions**: Tight focus on specific objects
- **Activity questions**: Broader attention including context
- **Location questions**: Spatial scanning patterns

### Failure Modes & Debugging

**Language Prior Problem**:

VQA models often exploit dataset biases:
- "What sport is this?" → "Tennis" (most common in dataset)
- Attention may not even look at the image properly
- Solution: Selective processing that verifies visual evidence

From [R-VQA: A robust visual question answering model](https://www.sciencedirect.com/science/article/abs/pii/S0950705124014618) (Chowdhury et al., 2025):
- Dataset designed to address language prior issues
- Forces models to use visual information, not just question patterns
- Improves compositional reasoning

**Attention Quality Metrics**:
- **Alignment with ground-truth** (when available): How well does attention match human annotations?
- **Consistency across similar questions**: Do similar questions produce similar attention?
- **Plausibility**: Does attention focus on relevant objects?

### Human Attention Comparison

From [Insights from eye-tracking and the human attention filter](https://www.sciencedirect.com/science/article/pii/S2667305325001048) (Rekanar et al., 2025):
- VQA models serve critical role in interpreting visual data
- Eye-tracking studies reveal how humans selectively process images for VQA
- Gap between human attention and model attention patterns

**Key Findings**:
- Humans use rapid serial visual search for counting
- Humans leverage semantic knowledge more than models
- Models may focus on spurious correlations invisible to humans

## Sources

**Foundational Papers:**
- [Question Type Guided Attention in Visual Question Answering](https://arxiv.org/abs/1804.02088) - Shi et al., ECCV 2018
- [Improving Selective Visual Question Answering by Learning from Your Peers](https://arxiv.org/abs/2306.08751) - Dancette et al., CVPR 2023 (accessed 2025-01-31)
- [Hierarchical Question-Image Co-Attention for Visual Question Answering](https://proceedings.neurips.cc/paper/2016/file/9dcb88e0137649590b755372b040afad-Paper.pdf) - Lu et al., NeurIPS 2016

**Recent Advances (2023-2025):**
- [Guiding Visual Question Answering With Attention Priors](https://openaccess.thecvf.com/content/WACV2023/papers/Le_Guiding_Visual_Question_Answering_With_Attention_Priors_WACV_2023_paper.pdf) - Le et al., WACV 2023
- [ReasonVQA: A Multi-hop Reasoning Benchmark](https://arxiv.org/html/2507.16403v1) - Tran et al., ICCV 2025
- [R-VQA: A robust visual question answering model](https://www.sciencedirect.com/science/article/abs/pii/S0950705124014618) - Chowdhury et al., 2025
- [Multi-Hop Graph Reasoning Network for Knowledge-based VQA](https://dl.acm.org/doi/10.1145/3724125) - 2025

**Attention Mechanisms:**
- [Selective Residual Learning for Visual Question Answering](https://www.sciencedirect.com/science/article/abs/pii/S0925231220304859) - Hong et al., Neurocomputing 2020
- [Dynamic Attention Networks for VQA](https://www.nature.com/articles/s41598-022-21149-9) - Miao et al., Scientific Reports 2022
- [Visual Question Answering: Attention Mechanism, Datasets and Challenges](https://dl.acm.org/doi/10.1145/3689236.3691498) - Tang et al., 2024 (accessed 2025-01-31)

**Human Studies:**
- [Insights from eye-tracking and the human attention filter](https://www.sciencedirect.com/science/article/pii/S2667305325001048) - Rekanar et al., 2025

**Visualization Research:**
- [Attention maps visualized across question types](https://www.researchgate.net/figure/Attention-maps-visualized-across-question-types-Image-attention-seems-mostly-plausible_fig1_357385234) - ResearchGate analysis

**Additional References:**
- [Co-attention Mechanisms Survey](https://link.springer.com/article/10.1007/s11042-025-21049-w) - Khan et al., 2025
- [Multi-Modal Alignment for VQA](https://www.mdpi.com/2079-9292/11/11/1778) - Xia et al., Electronics 2022
- [Dual Self-Attention with Co-Attention Networks](https://www.sciencedirect.com/science/article/abs/pii/S0031320321001436) - Liu et al., Pattern Recognition 2021
