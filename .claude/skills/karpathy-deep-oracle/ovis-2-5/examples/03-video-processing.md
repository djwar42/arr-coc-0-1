# Video Processing Example

**Category**: Examples
**Related**: [../usage/02-advanced-features.md](../usage/02-advanced-features.md)

## Sample Video Frames

```python
from moviepy.editor import VideoFileClip
import numpy as np

def sample_frames(video_path, num_frames=8):
    """Sample evenly-spaced frames from video"""
    clip = VideoFileClip(video_path)
    duration = clip.duration
    times = np.linspace(0, duration, num_frames)

    frames = []
    for t in times:
        frame = clip.get_frame(t)
        frames.append(Image.fromarray(frame))

    clip.close()
    return frames

# Use with Ovis
frames = sample_frames("video.mp4", num_frames=8)

response, _, _ = model.chat(
    prompt="<video>\nDescribe what happens in this video.",
    videos=[frames]
)
```
