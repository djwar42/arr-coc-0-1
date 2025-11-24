# Visual Reasoning & Question Answering in Vision-Language Models

## Overview

Visual reasoning is the ability to understand and answer questions about images by combining visual perception with logical inference. Unlike simple object recognition, visual reasoning requires VLMs to understand relationships between objects, reason about spatial configurations, perform multi-step logical deductions, and chain evidence across multiple image regions.

**Core Challenge**: Visual reasoning demands compositional understanding - breaking complex questions into sub-problems, gathering relevant visual evidence, and integrating information to produce coherent answers.

**Key Insight**: Different question types require vastly different reasoning strategies and computational resources. Counting questions need global spatial understanding, spatial reasoning requires preserved spatial structure, while simple recognition can succeed with minimal tokens.

From [Karpathy practical-implementation/50-vqav2-training-protocols.md](../karpathy/practical-implementation/50-vqav2-training-protocols.md):
- VQAv2 dataset: 443,757 training questions, 10 human answers per question
- Open-ended task with soft label encoding (min(count/3, 1.0) scoring)
- Addresses visual priming bias through complementary image pairs

From [Karpathy practical-implementation/64-vqa-accuracy-token-tradeoff.md](../karpathy/practical-implementation/benchmarking/64-vqa-accuracy-token-tradeoff.md):
- Token budget directly impacts reasoning capability
- Recognition questions: 64-144 tokens sufficient (90% performance)
- Counting/spatial questions: 256-576 tokens optimal
- OCR questions: 576-1024 tokens beneficial (linear scaling)

## VQA Task Taxonomy

### 1. Counting Questions

**Definition**: Questions requiring enumeration of objects or instances in an image.

**Examples**:
- "How many people are in the image?"
- "How many red cars can you see?"
- "Count the number of birds on the tree"

**Reasoning Requirements**:
- Global spatial understanding across entire image
- Instance segmentation (distinguish individual objects)
- Attention to small or partially occluded objects
- Grouping of similar objects without double-counting

From [Karpathy practical-implementation/benchmarking/64-vqa-accuracy-token-tradeoff.md](../karpathy/practical-implementation/benchmarking/64-vqa-accuracy-token-tradeoff.md):
- High token sensitivity: 64 tokens → 30-40% accuracy, 256 tokens → 60-70%, 576 tokens → 75-85%
- Small objects lost in aggressive compression
- Spatial pooling can merge nearby instances
- LLaVA-NeXT: VizWiz-VQA counting improved from 29.2% (2×2 grid) to 34.7% (6×6 grid)

**Common Failure Modes**:
- Undercounting due to occlusion
- Overcounting when objects partially visible
- Confusion with similar objects in background
- Difficulty with dense crowds or overlapping objects

### 2. Spatial Reasoning Questions

**Definition**: Questions about relative positions and spatial relationships between objects.

**Spatial Relations Types**:

**Directional Relations**:
- Left/right: "What is to the left of the car?"
- Above/below: "What is above the table?"
- Front/behind: "What is in front of the building?"

**Topological Relations**:
- Inside/outside: "What is inside the box?"
- On/off: "What is on the table?"
- Touching/separate: "Is the cup touching the book?"

**Distance Relations**:
- Near/far: "What is closest to the person?"
- Between: "What is between the chair and door?"

From web research ([SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_SpatialVLM_Endowing_Vision-Language_Models_with_Spatial_Reasoning_Capabilities_CVPR_2024_paper.pdf), accessed 2025-11-16):
- VLMs struggle with basic spatial relations despite strong recognition
- Synthetic data generation improves spatial reasoning capabilities
- Grounded spatial reasoning requires 3D understanding of 2D images
- SpatialVLM: Fine-tuning on spatial VQA improves directional accuracy

From web research ([Grounded Spatial Reasoning in Vision-Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/f38cb4cf9a5eaa92b3cfa481832719c6-Paper-Conference.pdf), accessed 2025-11-16):
- Spatial relations: left, right, above, below, behind, front, wide, thin, tall, short, big, small
- Metric relations: direction, distance, angle measurements
- Grounding spatial language to visual coordinates improves accuracy

From [Karpathy vision-language-architectures/implementations/09-query-conditioned-attention.md](../karpathy/vision-language-architectures/implementations/09-query-conditioned-attention.md):
- Query-conditioned attention enables spatial question focus
- Stacked attention: first layer broad focus, second layer refines
- Attention visualization reveals model's spatial reasoning process

**Token Budget Requirements**:
From [Karpathy practical-implementation/benchmarking/64-vqa-accuracy-token-tradeoff.md](../karpathy/practical-implementation/benchmarking/64-vqa-accuracy-token-tradeoff.md):
- 64 tokens: Spatial structure significantly degraded
- 256 tokens: Adequate for most spatial relationships
- 576 tokens: High confidence for complex spatial queries
- Minimal additional benefit beyond 576 tokens

### 3. Compositional Reasoning Questions

**Definition**: Questions requiring understanding of multiple attributes, objects, and their relationships simultaneously.

**Compositional Types**:

**Object + Attribute**:
- "What color is the large dog?" (object + size + color)
- "Is the red car parked or moving?" (object + color + state)

**Object + Object + Relation**:
- "Is the cat near the dog?" (two objects + spatial relation)
- "Which is taller, the tree or the building?" (two objects + comparison)

**Multi-Attribute Combinations**:
- "Find the small red ball on the green table" (size + color + object + spatial + surface)

From web research ([A Survey on Compositional Visual Reasoning](https://arxiv.org/html/2508.17298v1), accessed 2025-11-16):
- Compositional reasoning: human-like ability to break complex scenes into components
- Taxonomy: objects, attributes, relations as fundamental building blocks
- Modern VLMs struggle with systematic compositional generalization
- Failure modes: attribute binding errors, relation confusion

From web research ([NLI Improves Compositionality in Vision-Language Models](https://arxiv.org/abs/2410.22315), accessed 2025-11-16):
- VLMs often struggle to relate objects, attributes, and spatial configurations
- Natural Language Inference (NLI) training improves compositional reasoning
- Dense captions with explicit attribute-object-relation annotations help

**Benchmarks**:
- **CLEVR**: Synthetic images with perfect ground truth for compositional questions
- **GQA**: Real images with compositional question structure (scene graphs)
- **ConMe**: Rethinking compositional reasoning evaluation

### 4. Fine-Grained Visual Questions

**Definition**: Questions about specific visual details, subtle distinctions, or low-level attributes.

**Examples**:
- "What breed of dog is this?" (fine-grained classification)
- "What color are the person's shoes?" (specific attribute)
- "Is the surface smooth or textured?" (material/texture)
- "What's written on the sign?" (fine detail)

**Characteristics**:
- Requires high visual resolution
- Needs preserved high-frequency information
- Benefits from focused attention on relevant regions
- Token-hungry compared to coarse recognition

From [Karpathy practical-implementation/benchmarking/64-vqa-accuracy-token-tradeoff.md](../karpathy/practical-implementation/benchmarking/64-vqa-accuracy-token-tradeoff.md):
- Token sensitivity: steady improvement from 64 → 256 → 576 tokens
- Attribute questions benefit more than category questions
- Texture/material questions particularly sensitive to token count
- 256-576 tokens recommended for fine-grained tasks

### 5. OCR and Text-Heavy Questions

**Definition**: Questions requiring reading and understanding text in images.

**Examples**:
- "What does the sign say?"
- "What is the license plate number?"
- "Read the text on the document"
- "What time is shown on the clock?"

**Extreme Token Sensitivity**:
From [Karpathy practical-implementation/benchmarking/64-vqa-accuracy-token-tradeoff.md](../karpathy/practical-implementation/benchmarking/64-vqa-accuracy-token-tradeoff.md):
- 64 tokens: Very poor OCR accuracy (<30%)
- 256 tokens: Moderate OCR capability (60-70%)
- 576 tokens: Good OCR performance (75-85%)
- 1024+ tokens: Continued improvement for dense text
- LLaVA-NeXT: ChartQA improved from 49.2% (2×2 grid) to 55.8% (6×6 grid) - largest gain observed

**Why OCR Needs Tokens**:
- Text contains high-frequency details easily lost in compression
- Character-level features require fine spatial resolution
- Multi-line text needs preserved spatial layout
- Small text may disappear entirely with aggressive pooling

## Multi-Hop Reasoning

**Definition**: Questions requiring chaining multiple reasoning steps, where the answer to one sub-question informs the next.

**Examples**:
- "What color is the shirt of the person holding the umbrella?" (identify person → check umbrella → examine shirt)
- "How many red objects are near the largest animal?" (find largest animal → locate nearby objects → count red ones)

From web research ([II-MMR: Identifying and Improving Multi-modal Multi-hop Reasoning](https://aclanthology.org/2024.findings-acl.636.pdf), accessed 2025-11-16):
- Multi-hop reasoning: measuring number of reasoning steps required
- Current VQA benchmarks have varying reasoning complexity
- Error propagation: mistakes in early steps compound through reasoning chain
- Attention visualization reveals reasoning paths

**Reasoning Chain Structure**:

**Two-Hop Example**: "What is the color of the object to the left of the red ball?"
1. **Step 1**: Locate the red ball
2. **Step 2**: Identify object to its left
3. **Step 3**: Determine color of that object

**Three-Hop Example**: "Is the person wearing the blue hat also holding something?"
1. **Step 1**: Find person with blue hat
2. **Step 2**: Check person's hands
3. **Step 3**: Identify if holding object
4. **Step 4**: Answer yes/no

From web research ([Improving Vision-Language Models through Focus-Centric Reasoning](https://arxiv.org/html/2504.20199v1), accessed 2025-11-16):
- Focus-Centric Visual Chain paradigm: question decomposition + stepwise reasoning
- Decompose complex questions into simpler sub-questions
- Chain-of-thought prompting for VLMs improves multi-hop accuracy
- Progressive focusing: each step narrows attention to relevant regions

**Challenges**:
- **Error Propagation**: Mistakes in early steps corrupt later reasoning
- **Context Tracking**: Maintaining intermediate results across steps
- **Attention Drift**: Losing focus on relevant regions during long chains
- **Computational Cost**: Each hop requires additional processing

## Attention Visualization for VQA

**Purpose**: Understanding where models "look" when answering questions reveals reasoning process and failure modes.

### Attention Map Techniques

From [Karpathy vision-language-architectures/implementations/09-query-conditioned-attention.md](../karpathy/vision-language-architectures/implementations/09-query-conditioned-attention.md):

**Query-Conditioned Attention**:
- Text query guides which visual features receive attention
- Softmax over patch tokens weighted by query similarity
- Visualization reveals query-aware visual representations

**Stacked Attention**:
- Multiple attention layers for iterative refinement
- Layer 1: Broad focus (exploration)
- Layer 2: Refined focus (exploitation)
- Demonstrates multi-step reasoning visually

**Attention Rollout**:
- Aggregates attention across multiple layers
- Reveals effective receptive field for final answer
- Useful for debugging unexpected model behavior

### Common Attention Patterns

**Counting Questions**:
- Attention spreads across all instances of target object
- Dense activation on counted objects
- Background suppression

**Spatial Questions**:
- Strong activation on both referenced objects
- Attention highlights spatial relationship region
- Directional bias toward query-specified direction

**Attribute Questions**:
- Focused attention on single object
- Zoom-in pattern on relevant image region
- Fine-grained activation within object boundaries

## VQA Benchmarks

### VQAv2

From [Karpathy practical-implementation/50-vqav2-training-protocols.md](../karpathy/practical-implementation/50-vqav2-training-protocols.md):

**Dataset Characteristics**:
- 82,783 training images (MS COCO)
- 443,757 training questions
- 10 human answers per question (soft label encoding)
- Answer types: yes/no (38%), number (12%), other (50%)

**Evaluation Metric**:
```
VQA Accuracy = min(# humans that said answer / 3, 1.0)
```
- At least 3 humans must agree for 100% accuracy
- Partial credit for 1-2 agreements (33%, 67%)
- Accounts for answer ambiguity and subjectivity

**Bias Mitigation**:
- Complementary image pairs with different answers
- Forces model to read both image AND question
- Reduces language priors
- Blind model accuracy < 50% (image-only baseline)

### GQA (Scene Graph Question Answering)

**Characteristics**:
- Structured visual reasoning questions
- Questions derived from scene graphs
- Compositional consistency evaluation
- 1.7M questions on 113K images

**Question Structure**:
- Objects connected by relationships in scene graph
- Questions test specific graph traversals
- Compositional splits test systematic generalization
- Balances different reasoning types

### CLEVR (Compositional Language and Elementary Visual Reasoning)

**Characteristics**:
- Synthetic images with geometric primitives
- Perfect ground truth for compositional questions
- Systematic evaluation of reasoning capabilities
- Questions require multi-step logical inference

**Reasoning Types Tested**:
- Counting: "How many red cubes are there?"
- Comparison: "Is the sphere larger than the cube?"
- Spatial: "What is to the left of the red object?"
- Existence: "Is there a blue metal cylinder?"

### TextVQA / DocVQA / ChartQA

**Text-Focused Benchmarks**:
- **TextVQA**: Reading text in natural images (signs, labels)
- **DocVQA**: Document understanding questions (forms, receipts)
- **ChartQA**: Chart and plot interpretation (bar graphs, line charts)
- **InfoVQA**: Information extraction from infographics

From [Karpathy practical-implementation/benchmarking/64-vqa-accuracy-token-tradeoff.md](../karpathy/practical-implementation/benchmarking/64-vqa-accuracy-token-tradeoff.md):
- These benchmarks show largest gains from increased token budgets
- DocVQA: 58.8% (64 tokens) → 62.7% (7K tokens) = +3.9%
- ChartQA: 49.2% → 55.8% = +6.6% (largest improvement)
- OCR capability scales nearly linearly with tokens

## Common Failure Modes

### 1. Language Prior Bias

**Problem**: Model relies on statistical correlations in language rather than visual evidence.

**Example**:
- Question: "What sport is this?" on image of tennis
- Model answers "tennis" correctly but...
- Same answer given for baseball image if "racket" mentioned in question

**Mitigation**:
- VQAv2's complementary image pairs
- Adversarial examples with misleading language
- Training on visually grounded captions

### 2. Attribute Binding Errors

**Problem**: Correctly identifying objects and attributes but binding them incorrectly.

**Example**:
- Image: red car and blue truck
- Question: "What color is the car?"
- Model: "blue" (bound color from wrong object)

From web research ([Compositional reasoning challenges](https://arxiv.org/html/2508.17298v1), accessed 2025-11-16):
- Systematic failure in compositional VLMs
- Need explicit attribute-object relationship modeling
- Attention visualization reveals incorrect binding patterns

### 3. Spatial Relation Confusion

**Problem**: Confusing similar spatial relations (left vs right, above vs below).

From web research ([SpatialVLM](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_SpatialVLM_Endowing_Vision-Language_Models_with_Spatial_Reasoning_Capabilities_CVPR_2024_paper.pdf), accessed 2025-11-16):
- VLMs significantly underperform on spatial reasoning
- "left" and "right" particularly confusing (mirror symmetry)
- "above" and "below" more reliable (gravity bias)

**Mitigation**:
- Synthetic spatial data augmentation
- Explicit spatial coordinate prediction
- Egocentric perspective grounding

### 4. Counting Errors

**Systematic Biases**:
- Undercounting due to occlusion
- Overcounting when objects partially visible
- Confusion in dense scenes
- "Subitizing limit": accurate up to ~4 objects, then approximate

**Mitigation Strategies**:
- Instance segmentation pre-processing
- Attention to small objects (high token budget)
- Explicit counting head in architecture

### 5. Hallucination (Object Presence)

**POPE Benchmark** (Polling-based Object Probing Evaluation):
- Tests whether VLMs hallucinate object presence
- Negative questions: "Is there a zebra?" when no zebra present
- VLMs tend toward positive answers (yes-bias)

From [Karpathy practical-implementation/benchmarking/64-vqa-accuracy-token-tradeoff.md](../karpathy/practical-implementation/benchmarking/64-vqa-accuracy-token-tradeoff.md):
- LLaVA-1.5 POPE: 85.9% (576 tokens)
- Minimal degradation with fewer tokens (token-efficient task)
- Hallucination more related to training than token budget

## ARR-COC-0-1: Relevance Realization for Visual Reasoning

**Query-Aware Token Allocation**: Different VQA question types demand vastly different computational resources.

### Relevance Mapping to Question Types

From [Karpathy practical-implementation/benchmarking/64-vqa-accuracy-token-tradeoff.md](../karpathy/practical-implementation/benchmarking/64-vqa-accuracy-token-tradeoff.md):

**Low Relevance Regions (64 tokens)**:
- Simple recognition: "What is this animal?"
- Yes/no questions with obvious answers
- Background regions for object-specific questions

**Moderate Relevance (256 tokens)**:
- Spatial reasoning: "What is to the left of the car?"
- Basic counting: "How many people?"
- Simple attribute questions: "What color is the shirt?"

**High Relevance (576 tokens)**:
- Fine-grained classification: "What breed of dog?"
- Complex spatial: "What is between the chair and door?"
- Multi-object counting: "How many red cars?"

**Critical Relevance (1024+ tokens)**:
- OCR tasks: "What text is on the sign?"
- Dense document understanding
- Chart/table interpretation

### Transjective Token Allocation

**Core Principle**: Token budget emerges from query-content relationship, not image properties alone.

**Examples**:
- **Same Image, Different Questions**:
  - "What color is the car?" → 64 tokens (low detail needed)
  - "What text is on the license plate?" → 576+ tokens (OCR needed)

**Opponent Processing for VQA**:
- **Compress ↔ Particularize**: Balance efficiency vs detail based on question type
- **Exploit ↔ Explore**: Focus on relevant regions vs scan entire image
- **Focus ↔ Diversify**: Zoom into detail vs maintain global context

### Participatory Knowing in VQA

From [Karpathy vision-language-architectures/implementations/09-query-conditioned-attention.md](../karpathy/vision-language-architectures/implementations/09-query-conditioned-attention.md):

**Query-Visual Coupling**:
- Query embedding guides attention to relevant patches
- Attention weights realize relevance dynamically
- Multi-step reasoning through stacked attention (explore → exploit)

**ARR-COC Advantage**:
- Query-aware token allocation BEFORE LLM processing
- Hard allocation (variable tokens) vs soft weighting (attention)
- Computational savings: 30-50% vs fixed high-token budgets

### Expected Performance

**Token Budget Prediction**:
1. **Question Classification**: Recognize question type from text
   - Recognition → 64-144 tokens
   - Spatial → 256 tokens
   - Counting → 256-576 tokens
   - OCR → 576-1024 tokens

2. **Spatial Token Distribution**: Foveated strategy
   - High tokens in query-relevant regions
   - Low tokens in background
   - Dynamic allocation based on realized relevance

3. **Adaptive Compression**:
   - 64-400 tokens per patch based on relevance
   - Average budget: 200-300 tokens (vs 576 fixed)
   - Computational savings: 40-50%
   - Accuracy: Maintained or improved (task-appropriate allocation)

## Training Strategies for Visual Reasoning

### 1. Multi-Task Learning

**Approach**: Train on multiple VQA task types simultaneously.

**Benefits**:
- Shared visual encoder learns generalizable features
- Task-specific heads specialize in reasoning types
- Cross-task transfer improves sample efficiency

**Implementation**:
```python
# Multi-task loss
loss = alpha * counting_loss +
       beta * spatial_loss +
       gamma * recognition_loss +
       delta * ocr_loss

# Task weighting based on difficulty/importance
# alpha, beta, gamma, delta learned or hand-tuned
```

### 2. Curriculum Learning

**Progressive Difficulty**:
1. Start with simple recognition questions
2. Progress to single-hop reasoning
3. Advance to multi-hop compositional questions
4. Final phase: complex spatial + counting combined

**Rationale**: Build reasoning capabilities incrementally rather than learning all simultaneously.

### 3. Chain-of-Thought Training

From web research ([Multi-hop reasoning VLMs](https://arxiv.org/html/2504.20199v1), accessed 2025-11-16):

**Approach**: Train model to generate intermediate reasoning steps.

**Example**:
- Question: "What color is the shirt of the person holding the umbrella?"
- Chain-of-thought: "First, I identify the person holding the umbrella. That person is in the center of the image. Next, I examine their shirt. The shirt is blue."
- Answer: "blue"

**Benefits**:
- Interpretable reasoning process
- Easier debugging of failures
- Improved multi-hop accuracy
- Reduces error propagation

### 4. Attention Supervision

From [Karpathy vision-language-architectures/implementations/09-query-conditioned-attention.md](../karpathy/vision-language-architectures/implementations/09-query-conditioned-attention.md):

**Approach**: Use human attention maps to guide model attention.

**Loss Function**:
```python
total_loss = task_loss + lambda_attn * kl_div(pred_attention, human_attention)
```

**Human Attention Data**:
- Eye-tracking studies on VQA
- Manual annotations of relevant regions
- Attention maps from interpretable models

## Evaluation Metrics

### Task-Specific Metrics

**VQAv2 Accuracy**:
- Min(human_agreement/3, 1.0)
- Partial credit for consensus
- Separate reporting by answer type (yes/no, number, other)

**GQA Accuracy**:
- Exact match with ground truth
- Compositional consistency: answer shouldn't change under equivalent rephrasing
- Binary accuracy, grounding accuracy (correct region)

**CLEVR Accuracy**:
- Exact match (synthetic has single correct answer)
- Per-question-type breakdown
- Compositional generalization test sets

### Attention Quality Metrics

**Attention Accuracy**:
- IoU between predicted attention and human gaze heatmap
- Pointing game: does max attention fall within correct object?

**Relevance Correlation**:
- Spearman correlation between attention weights and relevance scores
- Human annotations of "relevant regions" for question

### Reasoning Path Analysis

From web research ([II-MMR analysis](https://aclanthology.org/2024.findings-acl.636.pdf), accessed 2025-11-16):

**Reasoning Steps Measurement**:
- Count number of hops in reasoning chain
- Analyze error propagation across steps
- Identify where reasoning breaks down

**Metrics**:
- Single-hop accuracy: Direct visual → answer
- Two-hop accuracy: Visual → intermediate → answer
- Three-hop+ accuracy: Multi-step reasoning chains

## Sources

**Source Documents:**
- [Karpathy practical-implementation/50-vqav2-training-protocols.md](../karpathy/practical-implementation/50-vqav2-training-protocols.md) - VQAv2 dataset structure, training protocols, evaluation metrics
- [Karpathy practical-implementation/benchmarking/64-vqa-accuracy-token-tradeoff.md](../karpathy/practical-implementation/benchmarking/64-vqa-accuracy-token-tradeoff.md) - Token budget analysis for different question types, performance curves
- [Karpathy vision-language-architectures/implementations/09-query-conditioned-attention.md](../karpathy/vision-language-architectures/implementations/09-query-conditioned-attention.md) - Query-conditioned attention mechanisms, stacked attention for multi-step reasoning

**Web Research:**
- [A Survey on Compositional Visual Reasoning](https://arxiv.org/html/2508.17298v1) - arXiv:2508.17298 (accessed 2025-11-16)
  - Compositional reasoning taxonomy, failure modes in VLMs

- [SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_SpatialVLM_Endowing_Vision-Language_Models_with_Spatial_Reasoning_Capabilities_CVPR_2024_paper.pdf) - CVPR 2024 (accessed 2025-11-16)
  - Spatial relation types, training strategies for spatial reasoning

- [Grounded Spatial Reasoning in Vision-Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/f38cb4cf9a5eaa92b3cfa481832719c6-Paper-Conference.pdf) - NeurIPS 2024 (accessed 2025-11-16)
  - Metric spatial relations, grounding spatial language

- [II-MMR: Identifying and Improving Multi-modal Multi-hop Reasoning](https://aclanthology.org/2024.findings-acl.636.pdf) - ACL 2024 Findings (accessed 2025-11-16)
  - Multi-hop reasoning path analysis, error propagation

- [Improving Vision-Language Models through Focus-Centric Reasoning](https://arxiv.org/html/2504.20199v1) - arXiv:2504.20199 (accessed 2025-11-16)
  - Focus-Centric Visual Chain, question decomposition

- [NLI Improves Compositionality in Vision-Language Models](https://arxiv.org/abs/2410.22315) - arXiv:2410.22315 (accessed 2025-11-16)
  - Natural Language Inference training for compositional reasoning

**Additional References:**
- VQAv2 benchmark: https://visualqa.org
- GQA benchmark: https://cs.stanford.edu/people/dorarad/gqa/
- CLEVR benchmark: https://cs.stanford.edu/people/jcjohns/clevr/
- TextVQA, DocVQA, ChartQA benchmarks
- POPE (Polling-based Object Probing Evaluation) for hallucination
