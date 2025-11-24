# Multi-Image Example

**Category**: Examples
**Related**: [../usage/02-advanced-features.md](../usage/02-advanced-features.md)

## Compare Multiple Images

```python
from transformers import AutoModelForCausalLM
from PIL import Image

model = AutoModelForCausalLM.from_pretrained(
    "AIDC-AI/Ovis2.5-9B",
    trust_remote_code=True
).cuda()

# Load multiple images
images = [
    Image.open("before.jpg"),
    Image.open("after.jpg")
]

response, _, _ = model.chat(
    prompt="<image>\n<image>\nWhat changed between these images?",
    images=images
)

print(response)
```

## Analyze Image Sequence

```python
images = [Image.open(f"step_{i}.jpg") for i in range(5)]

response, _, _ = model.chat(
    prompt="<image>\n<image>\n<image>\n<image>\n<image>\nDescribe the progression.",
    images=images
)
```
