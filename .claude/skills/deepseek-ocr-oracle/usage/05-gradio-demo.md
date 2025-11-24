# Gradio Web Interface

**Full code**: `RESEARCH/DeepSeekOCR/HF.md` lines 950-1100

## Quick Demo

```python
import gradio as gr
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-OCR")

def ocr(image, mode, task):
    mode_configs = {
        'Tiny (~73 tokens)': (512, 512, False),
        'Small (~111 tokens)': (640, 640, False),
        'Base (~273 tokens)': (1024, 1024, False),
        'Large (~421 tokens)': (1280, 1280, False),
    }
    base_size, image_size, crop = mode_configs[mode]

    result = model.infer(tokenizer, prompt=f"<image>\n{task}", images=[image],
                        base_size=base_size, image_size=image_size, crop_mode=crop)
    return result[0]

demo = gr.Interface(
    fn=ocr,
    inputs=[
        gr.Image(type='pil', label='Upload Document'),
        gr.Dropdown(['Tiny (~73 tokens)', 'Small (~111 tokens)', 'Base (~273 tokens)', 'Large (~421 tokens)'],
                   value='Base (~273 tokens)', label='Mode'),
        gr.Dropdown(['Free OCR', 'Chart Parsing', 'Convert to markdown'], label='Task')
    ],
    outputs=gr.Textbox(label='Result', lines=20)
)

demo.launch()
```

## Deploy to HuggingFace Spaces

```bash
# Create Space on HF
# Upload app.py with code above
# Add requirements.txt:
transformers
torch
pillow
gradio
```

**See HF.md** for complete Gradio setup with examples!
