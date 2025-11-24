# Basic Inference Examples

**Category**: Examples
**Related**: [../usage/00-quickstart.md](../usage/00-quickstart.md), [01-thinking-mode.md](01-thinking-mode.md)
**Code**: Complete working examples

## Simple Image Description

```python
from transformers import AutoModelForCausalLM
from PIL import Image
import torch

# Load model once
model = AutoModelForCausalLM.from_pretrained(
    "AIDC-AI/Ovis2.5-2B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).cuda()

# Load image
image = Image.open("vacation_photo.jpg")

# Generate description
response, _, _ = model.chat(
    prompt="<image>\nDescribe this image.",
    images=[image]
)

print(response)
# Output: "The image shows a beach scene with..."
```

## Visual Question Answering

```python
# Same model, different questions
questions = [
    "How many people are in this image?",
    "What color is the car?",
    "Is it daytime or nighttime?",
    "What's the weather like?",
    "Are there any animals visible?"
]

image = Image.open("street_scene.jpg")

for question in questions:
    response, _, _ = model.chat(
        prompt=f"<image>\n{question}",
        images=[image]
    )
    print(f"Q: {question}")
    print(f"A: {response}\n")

# Output:
# Q: How many people are in this image?
# A: There are 3 people visible in the image.
#
# Q: What color is the car?
# A: The car is red.
# ...
```

## Document OCR

```python
# Extract text from document
document = Image.open("invoice.jpg")

response, _, _ = model.chat(
    prompt="<image>\nExtract all text from this document.",
    images=[document],
    max_new_tokens=2048  # Longer for full document
)

print(response)
# Output: Full text extraction with layout preserved
```

## Table Conversion

```python
# Convert table to markdown
table_image = Image.open("data_table.jpg")

response, _, _ = model.chat(
    prompt="<image>\nConvert this table to markdown format.",
    images=[table_image]
)

print(response)
# Output:
# | Column 1 | Column 2 | Column 3 |
# |----------|----------|----------|
# | Value 1  | Value 2  | Value 3  |
# ...
```

## Chart Analysis

```python
# Analyze chart trends
chart = Image.open("sales_chart.jpg")

response, _, _ = model.chat(
    prompt="<image>\nAnalyze the trends shown in this chart.",
    images=[chart]
)

print(response)
# Output: "The chart shows a clear upward trend..."
```

## Code Screenshot Understanding

```python
# Understand code from screenshot
code_screenshot = Image.open("python_code.png")

response, _, _ = model.chat(
    prompt="<image>\nExplain what this code does.",
    images=[code_screenshot]
)

print(response)
# Output: "This Python code defines a function that..."
```

## Multi-Turn Conversation

```python
# Build conversation history
image = Image.open("complex_scene.jpg")
history = []

# Turn 1
response, history, _ = model.chat(
    prompt="<image>\nWhat's the main subject of this image?",
    images=[image],
    # history=[]  # Empty for first turn
)
print(f"Turn 1: {response}")

# Turn 2 (with history)
response, history, _ = model.chat(
    prompt="Can you tell me more about its color?",
    images=[],  # Image already in history
    history=history  # Pass previous history
)
print(f"Turn 2: {response}")

# Turn 3
response, history, _ = model.chat(
    prompt="Where is it located in the image?",
    images=[],
    history=history
)
print(f"Turn 3: {response}")
```

## Batch Processing

```python
import glob
from pathlib import Path

# Process all images in directory
image_files = glob.glob("images/*.jpg")

results = []
for img_path in image_files:
    image = Image.open(img_path)

    response, _, _ = model.chat(
        prompt="<image>\nDescribe this image briefly.",
        images=[image],
        max_new_tokens=256  # Keep it brief
    )

    results.append({
        'file': Path(img_path).name,
        'description': response
    })

# Save results
import json
with open("descriptions.json", "w") as f:
    json.dump(results, f, indent=2)
```

## Comparing Two Images (Separate Queries)

```python
# Load two images
image1 = Image.open("before.jpg")
image2 = Image.open("after.jpg")

# Describe first
desc1, _, _ = model.chat(
    prompt="<image>\nDescribe this image.",
    images=[image1]
)

# Describe second
desc2, _, _ = model.chat(
    prompt="<image>\nDescribe this image.",
    images=[image2]
)

print("Before:", desc1)
print("After:", desc2)
print("\nChanges: Compare the descriptions manually")
```

## Detailed Analysis with Specific Questions

```python
# Structured analysis
image = Image.open("product_photo.jpg")

questions = {
    "category": "What category of product is this?",
    "color": "What are the main colors?",
    "condition": "What condition is it in?",
    "text": "Is there any text visible?",
    "background": "What's in the background?"
}

analysis = {}
for key, question in questions.items():
    response, _, _ = model.chat(
        prompt=f"<image>\n{question}",
        images=[image]
    )
    analysis[key] = response

# Structured output
import json
print(json.dumps(analysis, indent=2))
```

## Error Handling

```python
def safe_inference(image_path, prompt):
    """
    Robust inference with error handling
    """
    try:
        # Load image
        image = Image.open(image_path)

        # Validate image
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Generate
        response, _, _ = model.chat(
            prompt=f"<image>\n{prompt}",
            images=[image],
            max_new_tokens=1024
        )

        return {
            'success': True,
            'response': response,
            'error': None
        }

    except FileNotFoundError:
        return {
            'success': False,
            'response': None,
            'error': f"Image not found: {image_path}"
        }
    except Exception as e:
        return {
            'success': False,
            'response': None,
            'error': f"Error: {str(e)}"
        }

# Usage
result = safe_inference("image.jpg", "Describe this.")
if result['success']:
    print(result['response'])
else:
    print(f"Error: {result['error']}")
```

## Temperature Control

```python
image = Image.open("ambiguous_scene.jpg")

# Greedy (deterministic)
response_greedy, _, _ = model.chat(
    prompt="<image>\nWhat do you see?",
    images=[image],
    temperature=0.0,  # Always same output
    do_sample=False
)

# Sampling (creative)
response_creative, _, _ = model.chat(
    prompt="<image>\nWhat do you see?",
    images=[image],
    temperature=0.9,  # More varied output
    top_p=0.95,
    do_sample=True
)

print("Greedy:", response_greedy)
print("Creative:", response_creative)
```

## Speed Optimization

```python
# Fast inference for simple queries
image = Image.open("simple_scene.jpg")

response, _, _ = model.chat(
    prompt="<image>\nWhat's the main object?",
    images=[image],
    max_new_tokens=50,      # Limit output length
    temperature=0.0,        # Greedy (faster than sampling)
    do_sample=False
)

print(response)
# Fast, concise answer
```

## Complete Production Example

```python
import torch
from transformers import AutoModelForCausalLM
from PIL import Image
from pathlib import Path
import json
import time

class OvisInference:
    def __init__(self, model_name="AIDC-AI/Ovis2.5-2B"):
        """Initialize Ovis model"""
        print(f"Loading {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).cuda()
        print("Model loaded!")

    def describe_image(self, image_path, max_tokens=512):
        """Generate image description"""
        image = Image.open(image_path).convert("RGB")

        start = time.time()
        response, _, _ = self.model.chat(
            prompt="<image>\nDescribe this image in detail.",
            images=[image],
            max_new_tokens=max_tokens
        )
        elapsed = time.time() - start

        return {
            'description': response,
            'tokens': len(response.split()),
            'time': f"{elapsed:.2f}s"
        }

    def answer_question(self, image_path, question):
        """Answer specific question about image"""
        image = Image.open(image_path).convert("RGB")

        response, _, _ = self.model.chat(
            prompt=f"<image>\n{question}",
            images=[image]
        )

        return response

    def batch_process(self, image_dir, output_file):
        """Process directory of images"""
        results = []

        for img_path in Path(image_dir).glob("*.jpg"):
            print(f"Processing {img_path.name}...")

            result = self.describe_image(str(img_path))
            result['file'] = img_path.name

            results.append(result)

        # Save results
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Processed {len(results)} images → {output_file}")

# Usage
if __name__ == "__main__":
    ovis = OvisInference()

    # Single image
    result = ovis.describe_image("test.jpg")
    print(json.dumps(result, indent=2))

    # Question answering
    answer = ovis.answer_question("test.jpg", "What color is the car?")
    print(answer)

    # Batch processing
    ovis.batch_process("images/", "results.json")
```

## Related Topics

- [../usage/00-quickstart.md](../usage/00-quickstart.md) - Getting started
- [01-thinking-mode.md](01-thinking-mode.md) - Deep reasoning examples
- [02-multi-image.md](02-multi-image.md) - Multiple images
- [../usage/02-advanced-features.md](../usage/02-advanced-features.md) - Advanced usage

## Code References

**Model**: `ovis/model/modeling_ovis.py` - `Ovis` class
**Chat Method**: `ovis/model/modeling_ovis.py` - `chat()` method
**Demo Script**: `ovis/serve/infer_basic_demo.py` - Official examples
