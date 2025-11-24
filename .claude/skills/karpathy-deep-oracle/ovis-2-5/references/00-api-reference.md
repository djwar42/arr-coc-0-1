# API Reference

**Category**: References

## Ovis.chat()

```python
response, history, generation_info = model.chat(
    prompt: str,              # Text with <image> tokens
    images: List[PIL.Image],  # List of images
    videos: List = None,      # Optional video frames
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    enable_thinking: bool = False,
    thinking_budget: int = 2048,
    history: List = None      # Conversation history
)
```

## Ovis.generate()

Lower-level generation method.

```python
outputs = model.generate(
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    grid_thws: torch.Tensor,
    **generation_kwargs
)
```

## VisualTokenizer.preprocess()

```python
processed = tokenizer.preprocess(
    image: PIL.Image = None,
    video: List[PIL.Image] = None,
    min_pixels: int = 448*448,
    max_pixels: int = 1792*1792
)
```
