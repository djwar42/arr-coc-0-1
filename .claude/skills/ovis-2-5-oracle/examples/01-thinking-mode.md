# Thinking Mode Example

**Category**: Examples
**Related**: [../architecture/06-thinking-mode.md](../architecture/06-thinking-mode.md)

## Basic Usage

```python
from transformers import AutoModelForCausalLM
from PIL import Image
import torch

model = AutoModelForCausalLM.from_pretrained(
    "AIDC-AI/Ovis2.5-9B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).cuda()

image = Image.open("math_problem.jpg")

response, _, _ = model.chat(
    prompt="<image>\nSolve this problem step by step.",
    images=[image],
    max_new_tokens=3072,
    enable_thinking=True,
    thinking_budget=2048
)

print(response)
# Output includes <think>...</think> tags
```

## Parsing Output

```python
import re

def parse_thinking(response):
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        return thinking, answer
    return None, response

thinking, answer = parse_thinking(response)
print("Thinking:", thinking)
print("\nAnswer:", answer)
```
