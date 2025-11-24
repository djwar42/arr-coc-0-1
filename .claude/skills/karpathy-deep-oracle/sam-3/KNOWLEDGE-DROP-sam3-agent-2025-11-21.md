# SAM 3 Agent: Complex Text Prompting with MLLMs

## Overview

SAM 3 Agent is an agentic system that combines SAM 3 with a Multimodal Large Language Model (MLLM) to handle complex text queries that go beyond simple noun phrases. While SAM 3 excels at direct concept prompts like "yellow school bus" or "player in white", the SAM 3 Agent can interpret and segment based on complex spatial, relational, and contextual queries.

## What is SAM 3 Agent?

SAM 3 Agent uses SAM 3 as a **tool** for an MLLM, enabling segmentation of complex text queries such as:
- "the leftmost child wearing blue vest"
- "people sitting down but not holding anything"
- "the person controlling and guiding a horse"

The MLLM performs reasoning about spatial relationships, attributes, and context, then decomposes the query into simpler prompts that SAM 3 can execute.

## Architecture & Workflow

### System Components

1. **MLLM (Multimodal LLM)**: Performs reasoning and query decomposition
   - Examples: Qwen3-VL-8B-Thinking, GPT-4V, Gemini
   - Receives the image + complex text query
   - Outputs simplified prompts for SAM 3

2. **SAM 3 Model**: Executes segmentation
   - Receives decomposed prompts from MLLM
   - Returns masks, bounding boxes, and confidence scores

3. **Agent Orchestration**: Coordinates the pipeline
   - `client_llm.py`: Sends requests to MLLM
   - `client_sam3.py`: Calls SAM 3 service
   - `inference.py`: Runs single image inference

### Processing Flow

```
User Query: "the leftmost child wearing blue vest"
           |
           v
    ┌─────────────┐
    │    MLLM     │  Reasoning: Identify children, find ones
    │  (Thinking) │  with blue vest, select leftmost
    └─────────────┘
           |
           v
    Decomposed prompts:
    1. "child wearing blue vest"
    2. Spatial filtering: select leftmost
           |
           v
    ┌─────────────┐
    │   SAM 3     │  Execute segmentation
    │  Processor  │  Return masks for all matching
    └─────────────┘
           |
           v
    Post-processing: Apply spatial constraints
           |
           v
    Final mask: leftmost child with blue vest
```

## Multi-Step Reasoning Capabilities

### Types of Complex Queries

**1. Spatial Reasoning**
- "the leftmost person"
- "objects in the top-right corner"
- "the car closest to the building"

**2. Attribute Combinations**
- "tall person wearing red hat"
- "small dog with brown fur"
- "striped cat sitting down"

**3. Relational Queries**
- "person holding a phone"
- "child riding a bicycle"
- "dog next to the tree"

**4. Negation & Exclusion**
- "people NOT wearing glasses"
- "cars except the red ones"
- "animals that are not sleeping"

**5. Counting & Selection**
- "all three dogs"
- "the second person from the left"
- "every player on the field"

### Reasoning Process

The MLLM uses chain-of-thought reasoning to:

1. **Parse the query**: Identify objects, attributes, spatial terms, relations
2. **Plan execution**: Determine what SAM 3 calls are needed
3. **Execute sequentially**: Make SAM 3 calls and combine results
4. **Apply constraints**: Filter results based on spatial/logical conditions
5. **Return final masks**: Output the segmentation that satisfies all conditions

## Example Notebook Walkthrough

### File: `examples/sam3_agent.ipynb`

The SAM 3 Agent notebook demonstrates the full pipeline.

### Setup Steps

**1. Environment Configuration**
```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
torch.inference_mode().__enter__()
```

**2. Build SAM 3 Model**
```python
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
model = build_sam3_image_model(bpe_path=bpe_path)
processor = Sam3Processor(model, confidence_threshold=0.5)
```

**3. Configure MLLM**
```python
LLM_CONFIGS = {
    "qwen3_vl_8b_thinking": {
        "provider": "vllm",
        "model": "Qwen/Qwen3-VL-8B-Thinking",
    },
    # Can also use external APIs (Gemini, GPT-4V)
}

model = "qwen3_vl_8b_thinking"
LLM_API_KEY = "DUMMY_API_KEY"  # For vLLM local server
```

**4. vLLM Server Setup** (if using local MLLM)
```bash
# Create separate conda environment
conda create -n vllm python=3.12
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128

# Start server
vllm serve Qwen/Qwen3-VL-8B-Thinking \
    --tensor-parallel-size 4 \
    --allowed-local-media-path / \
    --enforce-eager \
    --port 8001
```

**5. Run Agent Inference**
```python
from functools import partial
from sam3.agent.client_llm import send_generate_request as send_generate_request_orig
from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
from sam3.agent.inference import run_single_image_inference

# Prepare inputs
image = "assets/images/test_image.jpg"
prompt = "the leftmost child wearing blue vest"

# Configure clients
send_generate_request = partial(
    send_generate_request_orig,
    server_url=LLM_SERVER_URL,
    model=llm_config["model"],
    api_key=llm_config["api_key"]
)
call_sam_service = partial(
    call_sam_service_orig,
    sam3_processor=processor
)

# Run inference
output_image_path = run_single_image_inference(
    image, prompt, llm_config,
    send_generate_request, call_sam_service,
    debug=True, output_dir="agent_output"
)
```

## Agentic Behavior Characteristics

### What Makes It "Agentic"

1. **Tool Use**: MLLM uses SAM 3 as a tool, not just a pipeline component
2. **Reasoning**: Chain-of-thought decomposition of complex queries
3. **Multi-Step Execution**: Multiple SAM 3 calls may be orchestrated
4. **Dynamic Planning**: Query determines execution strategy

### Comparison: SAM 3 vs SAM 3 Agent

| Feature | SAM 3 | SAM 3 Agent |
|---------|-------|-------------|
| Input | Simple noun phrase | Complex natural language |
| Reasoning | None (direct matching) | Multi-step reasoning |
| Spatial queries | Limited | Full support |
| Relational queries | No | Yes |
| Negation | No | Yes |
| Components | Single model | MLLM + SAM 3 |

### Example Complex Queries (from paper)

From the SAM 3 paper, the Agent can handle queries like:
- "people sitting down but not holding anything"
- "the leftmost child wearing blue vest"

These require:
1. Finding all people
2. Filtering by pose (sitting)
3. Checking for held objects
4. Applying negation (not holding)

## Performance Considerations

### Latency

- **SAM 3 alone**: ~30ms per image on H200
- **SAM 3 Agent**: Adds MLLM inference time
  - vLLM local: ~100-500ms depending on model
  - External API: Network latency + inference

### Resource Requirements

- **SAM 3**: ~3.4 GB (848M parameters)
- **MLLM**: Additional GPU memory
  - Qwen3-VL-8B: ~16-20 GB
  - Smaller MLLMs available

### Optimization Strategies

1. **Batch queries**: Group similar complex queries
2. **Cache MLLM reasoning**: Reuse decomposition patterns
3. **Smaller MLLMs**: Use 7B-8B models for faster inference
4. **Quantization**: INT8/INT4 quantization for MLLM

## Use Cases

### 1. Advanced Data Annotation
- Complex selection criteria for labeling
- "Label all people wearing safety equipment except helmets"

### 2. Intelligent Video Analysis
- Track specific entities based on complex descriptions
- "Follow the person who just picked up the red bag"

### 3. Robotics & Embodied AI
- Natural language object specification
- "Grab the tool closest to the workbench edge"

### 4. Interactive Image Editing
- Complex region selection
- "Select the background except for the people"

### 5. Accessibility Tools
- Describe and locate objects for visually impaired users
- "Find all exit signs on this floor"

## Integration Patterns

### As Part of Larger Systems

SAM 3 Agent can be integrated with:
- **Vision-Language Pipelines**: GPT-4V, Gemini, Claude
- **Robotic Systems**: Action planning based on segmentation
- **Document Analysis**: Locate specific elements in documents
- **Video Understanding**: Temporal reasoning + segmentation

### API Integration Example

```python
# Integration with external MLLM API
LLM_CONFIGS = {
    "gemini": {
        "provider": "external",
        "model": "gemini-2.5-pro",
        "base_url": "https://generativelanguage.googleapis.com/v1",
    },
    "gpt4v": {
        "provider": "external",
        "model": "gpt-4-vision-preview",
        "base_url": "https://api.openai.com/v1",
    }
}
```

## Limitations

1. **Latency**: Additional MLLM inference adds delay
2. **Cost**: External API calls add operational costs
3. **Complexity**: More failure points than SAM 3 alone
4. **MLLM Errors**: Reasoning errors propagate to segmentation
5. **GPU Requirements**: Need capacity for both models

## Future Directions

1. **Tighter Integration**: End-to-end training of MLLM + SAM 3
2. **Smaller MLLMs**: Purpose-built reasoning models for segmentation
3. **Multi-turn Refinement**: Interactive correction of agent errors
4. **Video Agent**: Temporal reasoning + tracking
5. **3D Extension**: Complex queries in 3D scenes

## Sources

**GitHub Repository:**
- [SAM 3 Agent Notebook](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_agent.ipynb) - Official implementation

**Research Paper:**
- [SAM 3: Segment Anything with Concepts](https://openreview.net/pdf/9cb68221311aa4d88167b66c0c84ef569e37122f.pdf) - Section on SAM 3 Agent (OpenReview, accessed 2025-11-23)

**Official Announcements:**
- [Meta AI Blog: Introducing SAM 3](https://ai.meta.com/blog/segment-anything-model-3/) - SAM 3 Agent description (accessed 2025-11-23)

**Technical Resources:**
- [Roboflow: What is SAM 3?](https://blog.roboflow.com/what-is-sam3/) - Overview and examples (accessed 2025-11-23)
- [SAM 3 GitHub README](https://github.com/facebookresearch/sam3/blob/main/README.md) - Example notebooks list

**Additional References:**
- Qwen3-VL-8B-Thinking model for local MLLM inference
- vLLM for serving MLLMs efficiently
