# Basic Inference Examples

**Category**: Examples
**Related**: [usage/00-quickstart.md](../usage/00-quickstart.md)

## Hello World - Single Image

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

# Load model and processor
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# Prepare messages
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]

# Process
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print(output_text)
```

**Output Example**:
```
The image shows a serene coastal landscape at sunset. A rocky cliff juts out into calm blue waters,
with a small white building perched on top. The sky displays warm orange and pink hues from the
setting sun, creating a peaceful atmosphere.
```

## Local Image File

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///home/user/photos/vacation.jpg"},
            {"type": "text", "text": "What's in this photo?"},
        ],
    }
]

inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

print(output)
```

## Multi-Turn Conversation

```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/chart.png"},
            {"type": "text", "text": "What does this chart show?"},
        ],
    },
    {
        "role": "assistant",
        "content": "This chart shows monthly sales data from January to December, with a peak in July at 450,000 units.",
    },
    {
        "role": "user",
        "content": "What's the trend from July to December?",
    },
]

inputs = processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

print(output)
```

**Output Example**:
```
The trend from July to December shows a steady decline, dropping from 450,000 units in July
to approximately 200,000 units by December, indicating a seasonal decrease in sales during
the latter half of the year.
```

## Batch Processing

```python
# Set padding side for batch generation
processor.tokenizer.padding_side = 'left'

# Prepare multiple message sets
messages1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "image1.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

messages2 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "image2.jpg"},
            {"type": "text", "text": "What objects are in this image?"},
        ],
    }
]

# Combine for batch processing
messages = [messages1, messages2]

# Process batch
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    padding=True  # Required for batch
)
inputs = inputs.to(model.device)

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
outputs = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

for i, output in enumerate(outputs):
    print(f"Output {i+1}: {output}")
```

## OCR Example

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/receipt.jpg"},
            {"type": "text", "text": "Extract all the text from this receipt."},
        ],
    }
]

inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=512)  # Longer for OCR
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

print(output)
```

**Output Example**:
```
GROCERY STORE
123 Main Street
Date: 2025-10-27

Items:
- Milk        $3.99
- Bread       $2.49
- Eggs        $4.29
- Apples      $5.99

Subtotal:    $16.76
Tax:          $1.34
Total:       $18.10

Thank you for shopping!
```

## Related Topics

- [01-multi-image.md](01-multi-image.md) - Multiple image handling
- [02-video-understanding.md](02-video-understanding.md) - Video processing
- [03-grounding.md](03-grounding.md) - Object localization
- [usage/00-quickstart.md](../usage/00-quickstart.md) - Installation and setup
