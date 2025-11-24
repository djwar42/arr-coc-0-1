# Advanced Features

**Category**: Usage
**Related**: [00-quickstart.md](00-quickstart.md)

## Multi-Image

```python
images = [Image.open(f"img{i}.jpg") for i in range(3)]

response, _, _ = model.chat(
    prompt="<image>\n<image>\n<image>\nCompare these images.",
    images=images
)
```

## Video Processing

```python
# Sample frames from video
frames = sample_video_frames("video.mp4", num_frames=8)

response, _, _ = model.chat(
    prompt="<video>\nDescribe this video.",
    videos=[frames]
)
```

## Thinking Mode

```python
response, _, _ = model.chat(
    prompt="<image>\nSolve this math problem.",
    images=[image],
    enable_thinking=True,
    thinking_budget=2048
)
```

## Resolution Control

Resolution handled automatically by smart_resize within 448²-1792² range.
