# SAM 3: Segment Anything with Concepts - Text-Prompted Segmentation

## Section 1: SAM 3 Introduction

### Overview and Release

SAM 3 (Segment Anything Model 3) was released on November 19, 2025 by Meta AI as the third model in the Segment Anything series. This release represents a significant paradigm shift from the previous SAM models by introducing text-based prompting for segmentation tasks.

**Key Innovation:**
- First SAM model to accept natural language text prompts
- Performs **Promptable Concept Segmentation (PCS)**
- Detects, segments, and tracks ALL instances of a concept
- Unified model for images AND videos

### Core Capabilities

Unlike SAM 1 and SAM 2, which predicted a single object per prompt, SAM 3:
- Performs **open-vocabulary instance detection**
- Returns unique masks and IDs for all matching objects simultaneously
- Transforms SAM from a geometric segmentation tool into a concept-level vision foundation model

**Example Usage:**
```python
# Text prompt to find all instances
"shipping container" → masks for ALL shipping containers in image
"yellow school bus" → all yellow school buses
"striped cat" → all striped cats
```

### Model Specifications

**Architecture:**
- 848M total parameters (~3.4 GB)
- Decoupled Detector-Tracker design
- Shared Perception Encoder
- Novel Presence Token for discrimination

**Performance:**
- ~30 ms per image on H200 GPU
- Handles 100+ objects per image
- Real-time video processing capability
- Server-scale model (not edge-deployable)

### Evolution from Previous SAM Models

| Feature | SAM 1 | SAM 2 | SAM 3 |
|---------|-------|-------|-------|
| Release | April 2023 | August 2024 | November 2025 |
| Prompts | Points, boxes, masks | Points, boxes, masks | Text, visual exemplars + all previous |
| Output | Single mask per prompt | Single mask per prompt | ALL matching instances |
| Media | Images | Images + Video | Images + Video |
| Focus | Geometry | Temporal consistency | Semantic understanding |

### Licensing

SAM 3 is available under a custom license developed by Meta. The model is released for public use subject to terms and conditions on the model weights download page.

### Official Resources

- **Paper:** "SAM 3: Segment Anything with Concepts"
- **GitHub:** https://github.com/facebookresearch/sam3
- **HuggingFace:** https://huggingface.co/facebook/sam3
- **Demo:** https://ai.meta.com/sam3
- **OpenReview:** https://openreview.net/forum?id=r35clVtGzw

---

## Section 2: Open-Vocabulary Segmentation

### What is Open-Vocabulary Segmentation?

Open-vocabulary segmentation allows models to segment objects from any text description, not just predefined categories. SAM 3 excels at this task by understanding 270K unique concepts - 50x more than existing benchmarks like LVIS.

**Key Difference from Closed-Vocabulary:**
- **Closed-vocabulary:** Can only identify trained categories (e.g., 80 COCO classes)
- **Open-vocabulary:** Understands any noun phrase you provide

### How SAM 3 Achieves Open-Vocabulary Understanding

**Architecture Components:**
1. **Perception Encoder:** Aligns visual features with text embeddings
2. **Text Encoder:** Processes natural language prompts
3. **Global Presence Head:** Determines if concept exists before localization
4. **Detection Head:** Localizes all instances once presence confirmed

**The Decoupling Innovation:**
SAM 3 separates recognition (WHAT) from localization (WHERE):
- First checks: "Is there a yellow school bus in this image?"
- Then locates: "Where are all the yellow school buses?"

This decoupling significantly improves accuracy on unseen concepts and hard negatives.

### Types of Prompts Supported

**1. Text Prompts:**
```python
# Simple noun phrases
"car"
"shipping container"
"solar panel"

# Descriptive phrases
"yellow school bus"
"striped cat"
"person wearing red shirt"
```

**2. Visual Exemplar Prompts:**
```python
# Point to an example object
exemplar_box = [100, 100, 200, 200]  # BBox of example
# SAM 3 finds all visually similar objects
```

**3. Traditional Visual Prompts:**
- Points (positive/negative clicks)
- Bounding boxes
- Rough masks for refinement

### Exhaustive Instance Detection

A critical capability of SAM 3 is **exhaustive detection** - finding ALL instances of a concept, not just the most prominent one:

**Example: "Find all trailers"**
- Previous models: Return 1 mask
- SAM 3: Returns masks for every trailer visible

This is essential for:
- Counting applications (inventory, crowds)
- Comprehensive scene understanding
- Complete data annotation

### Concept Vocabulary

SAM 3 was trained on the SA-Co dataset with:
- **270K unique concepts** (50x LVIS vocabulary)
- **22 million entities**
- **17 top-level categories**
- **72 sub-categories**

Categories include:
- Common objects (vehicles, furniture, animals)
- Specialized domains (medical equipment, industrial machinery)
- Fine-grained distinctions (breeds, models, variations)
- Abstract concepts (signs, symbols, text)

### Zero-Shot Generalization

SAM 3 demonstrates strong zero-shot performance:
- Works on concepts never seen during training
- Generalizes across visual domains
- Handles novel combinations ("blue fire hydrant")

**Benchmark Results:**
- **SA-Co/Gold:** 54.1 cgF1 (vs human 72.8 - 75-80% of human performance)
- **LVIS:** 37.2 cgF1 (state-of-the-art open-vocabulary)

---

## Section 3: Text Encoder Integration

### Vision-Language Architecture

SAM 3 introduces tight integration between visual and language understanding through its unified architecture:

```
Text Prompt / Visual Exemplar
    ↓
Prompt Encoder (Text + Visual)
    ↓
Perception Encoder (Joint Embedding Space)
    ↓
Vision-Language Fusion
    ↓
Detection/Tracking Output
```

### Text Encoder Design

The text encoder processes natural language into embeddings that align with visual features:

**Key Properties:**
- Processes short noun phrases efficiently
- Creates embeddings compatible with visual space
- Supports compositionality (adjective + noun)
- Handles synonyms and variations

**Processing Flow:**
```python
def encode_text_prompt(text: str):
    # Tokenize and embed
    tokens = tokenizer(text)
    embeddings = text_encoder(tokens)

    # Add presence token for discrimination
    presence_enhanced = embeddings + presence_token

    return presence_enhanced
```

### Perception Encoder (PE)

The Perception Encoder is the core innovation enabling vision-language fusion:

**Functionality:**
1. Receives image features from backbone
2. Receives text/exemplar embeddings from prompt encoder
3. Creates joint embedding space
4. Enables cross-modal attention

**Shared Between:**
- Detector (finds instances in single frame)
- Tracker (propagates across video frames)

This sharing ensures consistent understanding across detection and tracking.

### Presence Token Innovation

A novel component in SAM 3 is the **learnable presence token**:

```python
class SAM3Detector(nn.Module):
    def __init__(self):
        # Novel presence token
        self.presence_token = nn.Embedding(1, 512)

    def forward(self, image, text_prompt):
        text_features = self.text_encoder(text_prompt)

        # Add presence token for discrimination
        text_features = text_features + self.presence_token.weight

        # ... rest of forward pass
```

**Purpose:**
- Improves discrimination between similar concepts
- Distinguishes "red apple" from "green apple"
- Reduces false positives on hard negatives
- Enables semantic understanding beyond visual similarity

### Cross-Modal Attention

SAM 3 uses cross-attention to fuse visual and text information:

```python
# Query: Image features
# Key/Value: Text embeddings
fused_features = cross_attention(
    query=image_features,
    key=text_features,
    value=text_features
)

# Bidirectional attention for rich fusion
```

This enables the model to:
- Attend to relevant image regions based on text
- Ground language concepts in visual space
- Handle ambiguous queries through context

### Prompt-Conditioned Detection

The detection head receives prompt-conditioned features:

**Flow:**
1. Perception Encoder fuses image + text
2. Global presence head predicts: concept present?
3. If present, detection head localizes all instances
4. Each instance gets unique mask and ID

**Benefits:**
- Avoids wasted computation on absent concepts
- Reduces false positives significantly
- Enables efficient multi-concept queries

---

## Section 4: 270K Concept Detection

### SA-Co Dataset Overview

SAM 3 was trained on the SA-Co (Segment Anything with Concepts) dataset, the largest concept-segmentation corpus to date:

**Dataset Statistics:**
- ~5.2M high-quality images
- 52.5K videos
- >4M unique noun phrases
- ~1.4B masks
- 270K unique concepts

### Concept Taxonomy

The SA-Co ontology is organized hierarchically:

**22 Million Entities across:**
- 17 top-level categories
- 72 sub-categories

**Example Categories:**
- **Animals:** 50+ species with breed distinctions
- **Vehicles:** Cars, trucks, boats, aircraft by type
- **Buildings:** Residential, commercial, industrial
- **Food:** Fruits, vegetables, prepared dishes
- **Furniture:** Indoor/outdoor, by room type
- **Plants:** Trees, flowers, crops
- **Tools:** Hand tools, power tools, kitchen tools

### Data Engine Process

Meta built a four-phase data engine combining humans, SAM models, and LLMs:

**Phase 1-3: Image Data**
```
AI Annotators → Propose candidate noun phrases
       ↓
SAM Models → Generate candidate masks
       ↓
AI Verifiers (Llama 3.2) → Assess quality & exhaustivity
       ↓
Human Review → Concentrated on failure cases
       ↓
Re-train SAM → Improve model
```

**Phase 4: Video Data**
- Extended Phase 1-3 pipeline to videos
- Added temporal consistency verification
- Ensured tracking stability across frames

**Efficiency Gains:**
- 2x throughput compared to human-only pipelines
- AI handles routine cases
- Humans focus on edge cases

### Exhaustivity Verification

A key challenge was ensuring exhaustive detection (finding ALL instances):

**AI Verifiers Check:**
1. Are all instances annotated?
2. Are masks high quality?
3. Is the concept label correct?
4. Are there any false positives?

**Human Verification:**
- Random sampling validation
- Error case review
- Quality control on novel concepts

### Concept Coverage vs Existing Benchmarks

| Benchmark | Concepts | Coverage |
|-----------|----------|----------|
| COCO | 80 | Common objects |
| LVIS | 1,203 | Long-tail objects |
| OpenImages | 600 | Web images |
| **SA-Co** | **270,000** | **Universal** |

SAM 3's vocabulary is:
- 50x larger than LVIS
- 450x larger than COCO
- Covers long-tail and specialized domains

### Fine-Grained Understanding

SA-Co enables distinctions that previous datasets couldn't support:

**Examples:**
- Not just "dog" but "golden retriever puppy"
- Not just "car" but "red convertible sports car"
- Not just "plant" but "potted succulent"

This fine-grained understanding enables:
- More precise queries
- Better user experience
- Practical applications in specialized domains

### Evaluation Benchmarks

SAM 3 is evaluated on three SA-Co splits:

**SA-Co/Gold:**
- High-quality image annotations
- Human-verified ground truth
- Primary accuracy benchmark

**SA-Co/Silver:**
- Larger image dataset
- AI-generated with quality filtering
- Scale evaluation

**SA-Co/vEval:**
- Video evaluation set
- Temporal consistency testing
- Tracking performance

---

## Section 5: Comparison to SAM 1 and SAM 2

### Architecture Evolution

**SAM 1 (April 2023):**
- ViT-H image encoder (636M params)
- Simple prompt encoder
- Lightweight mask decoder
- No text understanding

**SAM 2 (August 2024):**
- Hiera image encoder (6x faster)
- Memory attention for video
- Streaming memory bank
- Still no text understanding

**SAM 3 (November 2025):**
- Perception Encoder (shared)
- Decoupled detector-tracker
- Text encoder integration
- 848M total parameters

### Prompt Capabilities

| Prompt Type | SAM 1 | SAM 2 | SAM 3 |
|-------------|-------|-------|-------|
| Points | Yes | Yes | Yes |
| Boxes | Yes | Yes | Yes |
| Masks | Yes | Yes | Yes |
| **Text** | No | No | **Yes** |
| **Visual Exemplars** | No | No | **Yes** |

### Output Differences

**SAM 1 & 2:**
- One mask per prompt
- User must click each object separately
- Multiple clicks for multiple instances

**SAM 3:**
- ALL matching instances from one prompt
- Returns unique IDs per instance
- Exhaustive detection

**Example: "Find all cars"**
- SAM 1/2: Click car 1, get mask 1. Click car 2, get mask 2. Repeat...
- SAM 3: Type "car", get masks for ALL cars

### Video Processing

**SAM 2:**
- Memory attention for temporal consistency
- Propagates masks across frames
- Real-time (44 FPS on A100)

**SAM 3:**
- Inherits SAM 2's memory architecture
- Adds concept-based detection per frame
- Matching and update stage for consistency
- Handles occlusion and re-appearance

### Performance Benchmarks

**Speed:**
| Model | Speed | Hardware |
|-------|-------|----------|
| SAM 1 | ~110ms | A100 |
| SAM 2 | ~18ms (6x faster) | A100 |
| SAM 3 | ~30ms | H200 |

**Accuracy (Instance Segmentation):**
- SAM 3 achieves state-of-the-art on LVIS and SA-Co
- Better few-shot adaptation than previous models
- Strong visual generalization with task-specific data

### When to Use Each Model

**Use SAM 1 when:**
- Simple single-object segmentation
- Edge deployment needed
- Memory constrained environments

**Use SAM 2 when:**
- Video segmentation required
- Speed is critical
- Interactive annotation tasks

**Use SAM 3 when:**
- Text prompts desired
- Need ALL instances of a concept
- Open-vocabulary detection
- Comprehensive scene labeling
- Training smaller models with auto-labels

### Model Sizes

| Model | Params | Size | Deployment |
|-------|--------|------|------------|
| SAM 1 ViT-B | 91M | 375 MB | Edge possible |
| SAM 1 ViT-H | 636M | 2.4 GB | Server |
| SAM 2 Large | ~900M | 900 MB | Server |
| **SAM 3** | 848M | ~3.4 GB | Server only |

---

## Section 6: Applications

### Data Annotation and Labeling

SAM 3 revolutionizes dataset creation:

**Traditional Workflow:**
1. Click each object individually
2. Draw mask for each
3. Repeat hundreds of times
4. Very time-consuming

**SAM 3 Workflow:**
1. Type concept name
2. Get ALL instances automatically
3. Review and refine
4. Massively faster

**Use Case: Inventory Counting**
```python
# Annotate warehouse images
processor.set_text_prompt("cardboard box")
# Returns masks for all boxes - could be 50+ per image
```

### Training Smaller Models

SAM 3 enables efficient model distillation:

**Process:**
1. Use SAM 3 to auto-label large dataset
2. Train smaller model (YOLOv8, RF-DETR) on labels
3. Deploy small model at edge
4. Get real-time inference

**Benefits:**
- SAM 3's accuracy for labeling
- Small model's speed for deployment
- Best of both worlds

### Industrial Inspection

**Applications:**
- Defect detection on assembly lines
- Component counting in manufacturing
- Quality control verification

**Example:**
```python
# Find all screws that need inspection
processor.set_text_prompt("screw")
# Returns all screws for automated analysis
```

### Medical Imaging

SAM 3's open-vocabulary capabilities extend to medical domains:

**Use Cases:**
- Cell counting in microscopy
- Tumor detection assistance
- Organ segmentation
- Instrument tracking in surgery videos

**Note:** Medical applications require domain-specific fine-tuning for production use.

### Autonomous Vehicles

**Applications:**
- Pedestrian detection and tracking
- Vehicle segmentation
- Road sign identification
- Lane marking detection

**Video Capability:** Track concepts through driving footage with temporal consistency.

### Agriculture

**Use Cases:**
- Crop counting and health monitoring
- Plant species identification
- Pest and disease detection
- Yield estimation from aerial imagery

**Example:**
```python
# Aerial farm image analysis
processor.set_text_prompt("tomato plant")
plant_masks = output["masks"]
print(f"Detected {len(plant_masks)} tomato plants")
```

### Satellite and Aerial Imagery

**Applications:**
- Building detection and counting
- Road network extraction
- Forest monitoring
- Disaster damage assessment

**Example:**
```python
# Urban analysis
processor.set_text_prompt("building")
# Returns all building footprints
```

### Content Creation

**Use Cases:**
- Background removal at scale
- Video rotoscoping (extract foreground)
- AR/VR asset creation
- Automated photo editing

**Example:**
```python
# Remove all backgrounds in product catalog
for image in product_images:
    processor.set_text_prompt("product")
    mask = output["masks"][0]
    # Create transparent background version
```

### Robotics

**Applications:**
- Object grasping and manipulation
- Scene understanding
- Human-robot interaction
- Navigation and obstacle detection

### Retail and Inventory

**Applications:**
- Shelf monitoring
- Stock counting
- Product recognition
- Customer behavior analysis

---

## Section 7: Code Examples and Usage

### Installation

```bash
# Requirements
# Python 3.12+
# PyTorch 2.7+
# CUDA 12.6+

# Create environment
conda create -n sam3 python=3.12
conda activate sam3

# Install PyTorch
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Clone and install SAM 3
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

### Image Segmentation with Text

```python
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load model
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Load image
image = Image.open("warehouse.jpg")
inference_state = processor.set_image(image)

# Text prompt to find all instances
output = processor.set_text_prompt(
    state=inference_state,
    prompt="cardboard box"
)

masks = output["masks"]     # All masks matching concept
boxes = output["boxes"]     # Bounding boxes
scores = output["scores"]   # Confidence scores

print(f"Found {len(masks)} cardboard boxes")
```

### Video Segmentation with Text

```python
from sam3.model_builder import build_sam3_video_predictor

# Load video predictor
video_predictor = build_sam3_video_predictor()

# Start video session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path="traffic.mp4"
    )
)

session_id = response["session_id"]

# Add text prompt for frame 0
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text="pedestrian"
    )
)

# Propagate through video
# Automatically tracks all pedestrians through all frames
```

### Visual Exemplar Prompting

```python
# Use visual example instead of text
# Click on one object, find all similar

output = processor.set_visual_prompt(
    state=inference_state,
    exemplar_box=[100, 100, 200, 200]  # BBox of example object
)

# Returns all visually similar objects
similar_masks = output["masks"]
```

### Interactive Refinement

```python
# After initial text prompt, refine with clicks

# Positive click to add region
processor.add_point(
    state=inference_state,
    point=[300, 400],  # (x, y)
    label=1  # 1 = foreground
)

# Negative click to remove region
processor.add_point(
    state=inference_state,
    point=[500, 200],
    label=0  # 0 = background
)

# Get refined masks
refined_output = processor.predict(inference_state)
```

### Batched Inference

```python
# Process multiple images efficiently
batch_results = processor.batch_inference(
    images=[img1, img2, img3],
    prompts=["red car", "blue car", "truck"]
)

for i, result in enumerate(batch_results):
    print(f"Image {i}: Found {len(result['masks'])} objects")
```

### SAM 3 Agent for Complex Queries

```python
from sam3.agent import SAM3Agent

agent = SAM3Agent(model)

# Complex natural language query
result = agent.process_query(
    image=image,
    query="Find all animals that are not cats or dogs"
)

# Agent breaks down query:
# 1. Detect all animals
# 2. Filter out cats
# 3. Filter out dogs
# 4. Return remaining masks
```

---

## Section 8: ARR-COC Integration

### Relevance to Attention Research

SAM 3's architecture provides valuable insights for attention mechanism research in ARR-COC:

**Cross-Modal Attention:**
- SAM 3's vision-language fusion demonstrates effective cross-modal attention
- The Perception Encoder creates aligned embedding spaces
- Bidirectional attention between modalities

**Presence Token Innovation:**
- Novel learnable token that improves discrimination
- Similar concept could enhance attention routing in language models
- Demonstrates value of task-specific tokens

### Architectural Patterns for ARR-COC

**Decoupled Architecture:**
SAM 3's separation of detection (recognition) from tracking (localization) parallels:
- Separating routing from computation in MoE
- Decoupling attention from FFN in transformers
- Potential for more efficient model designs

**Memory Mechanisms:**
SAM 3's streaming memory from SAM 2 is relevant for:
- Long-context attention in language models
- Efficient KV-cache management
- Temporal modeling patterns

### Training Insights

**Multi-Stage Training:**
SAM 3 uses 4-stage training:
1. Perception Encoder pre-training
2. Detector pre-training
3. Detector fine-tuning
4. Tracker training (frozen backbone)

**Relevance:**
- Curriculum learning strategies
- Component-wise training
- When to freeze vs fine-tune

### Data Engine Lessons

SAM 3's data engine demonstrates:
- AI-in-the-loop annotation efficiency
- Quality verification pipelines
- Human-AI collaboration patterns

**Applicable to:**
- Training data curation
- Quality filtering
- Scaling annotation efficiently

### Fine-Tuning Applications

SAM 3's few-shot adaptation capabilities suggest:
- Foundation models can efficiently adapt
- Task-specific fine-tuning valuable
- Visual representations transfer well

**Roboflow100-VL Results:**
SAM 3 surpasses leading methods in few-shot and full fine-tuning scenarios, demonstrating strong visual generalization with task-specific training data.

### Potential Research Directions

**Vision-Language Models:**
- SAM 3's approach to concept understanding
- Grounding language in visual space
- Multi-modal attention patterns

**Efficient Detection:**
- Presence prediction before localization
- Avoiding wasted computation
- Conditional computation strategies

**Temporal Modeling:**
- Memory attention for sequences
- Maintaining identity across time
- Streaming processing patterns

### Practical Integration

**Dataset Creation:**
- Use SAM 3 for efficient labeling of vision datasets
- Create training data for downstream tasks
- Quality baseline from foundation model

**Model Distillation:**
- Train smaller models on SAM 3 labels
- Deploy efficient models with SAM 3 accuracy
- Edge deployment from server model

**Benchmarking:**
- SA-Co as evaluation benchmark
- 270K concepts for comprehensive testing
- Video and image evaluation

---

## Sources

### Primary Sources

**Meta AI Official:**
- [SAM 3 Research Publication](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) (accessed 2025-11-20)
- [SAM 3 GitHub Repository](https://github.com/facebookresearch/sam3) (accessed 2025-11-20)
- [SAM 3 HuggingFace](https://huggingface.co/facebook/sam3) (accessed 2025-11-20)
- [SAM 3 Demo](https://ai.meta.com/sam3) (accessed 2025-11-20)

**Documentation:**
- [Ultralytics SAM 3 Docs](https://docs.ultralytics.com/models/sam-3/) (accessed 2025-11-20)
- [OpenReview: SAM 3](https://openreview.net/forum?id=r35clVtGzw) (accessed 2025-11-20)

### Research Sources

**Technical Analysis:**
- [Roboflow: What is SAM 3?](https://blog.roboflow.com/what-is-sam3/) (accessed 2025-11-20)
- [MarkTechPost: Meta AI Releases SAM 3](https://www.marktechpost.com/2025/11/20/meta-ai-releases-segment-anything-model-3-sam-3-for-promptable-concept-segmentation-in-images-and-videos/) (accessed 2025-11-20)

**Related Work:**
- [Open-Vocabulary SAM (ECCV 2024)](https://arxiv.org/html/2401.02955v2) - arXiv:2401.02955 (accessed 2025-11-20)
- [Perception Encoder Paper](https://arxiv.org/abs/2504.13181) - arXiv:2504.13181 (accessed 2025-11-20)

### Source Document

- [SAM_STUDY_GENERAL.md](../../../PLAN-MD-FILES/november/20th/SAM_STUDY_GENERAL.md) - Comprehensive SAM research study (lines 356-560, 747-791)

### Datasets

**SA-Co Benchmarks:**
- [SA-Co/Gold Dataset](https://universe.roboflow.com/sa-co-gold)
- [SA-Co/Silver Dataset](https://universe.roboflow.com/sa-co-silver)
- [SA-Co/vEval Dataset](https://universe.roboflow.com/sa-co-veval)
