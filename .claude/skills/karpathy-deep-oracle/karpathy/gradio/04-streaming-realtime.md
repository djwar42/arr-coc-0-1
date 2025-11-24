# Gradio Streaming & Real-Time Communication (2025)

**Current as of**: January 2025
**Target**: Gradio 5.x with FastRTC integration

This guide covers modern streaming patterns in Gradio, including bi-directional streaming (inputs and outputs), real-time audio/video communication via FastRTC, and low-latency features introduced in 2025.

---

## Section 1: Streaming Inputs (Webcam, Audio)

**Source**: [gradio.app/guides/streaming-inputs](https://www.gradio.app/guides/streaming-inputs)

Gradio allows streaming data FROM users into your event handlers in real-time. Currently supported: `gr.Image` (webcam) and `gr.Audio` (microphone) via the `.stream()` event.

### Basic Streaming Input Pattern

```python
import gradio as gr
import numpy as np

def process_frame(frame):
    # Process each frame/audio chunk as it arrives
    return frame

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Input", sources="webcam")
        with gr.Column():
            output_img = gr.Image(label="Output")

    # stream() event with key parameters
    input_img.stream(
        lambda s: s,
        input_img,
        output_img,
        time_limit=15,        # Max processing time (seconds)
        stream_every=0.1,     # Capture frequency (seconds)
        concurrency_limit=30  # Max concurrent users
    )

demo.launch()
```

### Key Stream Event Parameters

**`time_limit`** (seconds):
- Maximum time server spends processing the stream
- Prevents users from hogging the queue
- Only counts processing time, not queue wait time
- Orange progress bar shows remaining time
- User automatically rejoins queue when time expires

**`stream_every`** (seconds):
- Frequency of input capture and server transmission
- Lower values (0.1s) → real-time effect for object detection
- Higher values (1-2s) → more context for speech transcription
- Tradeoff: responsiveness vs computational efficiency

**`concurrency_limit`**:
- Controls maximum parallel stream sessions
- Important for GPU-intensive applications
- Works with Gradio's queue system

### Real-Time Image Processing Example

```python
import gradio as gr
import numpy as np
import cv2

def transform_cv2(frame, transform):
    if transform == "cartoon":
        # Prepare color
        img_color = cv2.pyrDown(cv2.pyrDown(frame))
        for _ in range(6):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
        img_color = cv2.pyrUp(cv2.pyrUp(img_color))

        # Prepare edges
        img_edges = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        img_edges = cv2.adaptiveThreshold(
            cv2.medianBlur(img_edges, 7),
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9, 2
        )
        img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

        # Combine
        return cv2.bitwise_and(img_color, img_edges)

    elif transform == "edges":
        return cv2.cvtColor(cv2.Canny(frame, 100, 200), cv2.COLOR_GRAY2BGR)

    else:  # flip
        return np.flipud(frame)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            transform = gr.Dropdown(
                choices=["cartoon", "edges", "flip"],
                value="flip",
                label="Transformation"
            )
            input_img = gr.Image(sources=["webcam"], type="numpy")
        with gr.Column():
            output_img = gr.Image(streaming=True)

    input_img.stream(
        transform_cv2,
        [input_img, transform],
        [output_img],
        time_limit=30,
        stream_every=0.1,
        concurrency_limit=30
    )

demo.launch()
```

**Key insight**: Input values (like `transform` dropdown) can be changed DURING streaming and take effect immediately. This differs from traditional Gradio events where inputs are fixed at event start.

### Unified Image Demos (Single Component)

For some streaming apps, you don't need separate input/output components. Specify the input component as the output:

```python
css = """.my-group {max-width: 500px !important; max-height: 500px !important;}
        .my-column {display: flex !important; justify-content: center !important;
                    align-items: center !important};"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_classes=["my-column"]):
        with gr.Group(elem_classes=["my-group"]):
            transform = gr.Dropdown(
                choices=["cartoon", "edges", "flip"],
                value="flip",
                label="Transformation"
            )
            input_img = gr.Image(
                sources=["webcam"],
                type="numpy",
                streaming=True  # Enable streaming output on input component
            )

    # Input component is also the output
    input_img.stream(
        transform_cv2,
        [input_img, transform],
        [input_img],  # Same component!
        time_limit=30,
        stream_every=0.1
    )
```

### Keeping Track of Past Inputs with State

Streaming functions should be stateless (current input → current output), but use `gr.State()` to maintain history:

```python
def transcribe_handler(current_audio, state, transcript):
    # Transcribe with history context
    next_text = transcribe(current_audio, history=state)

    # Update state
    state.append(current_audio)
    state = state[-3:]  # Keep last 3 audio chunks

    return state, transcript + next_text

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            mic = gr.Audio(sources="microphone")
            state = gr.State(value=[])
        with gr.Column():
            transcript = gr.Textbox(label="Transcript")

    mic.stream(
        transcribe_handler,
        [mic, state, transcript],
        [state, transcript],
        time_limit=10,
        stream_every=1
    )
```

---

## Section 2: Streaming Outputs (Text, Video)

**Source**: [gradio.app/guides/streaming-outputs](https://www.gradio.app/guides/streaming-outputs)

Gradio supports streaming a sequence of outputs using Python **generators** (functions with `yield` instead of `return`).

### Basic Generator Pattern

```python
def my_generator(x):
    for i in range(x):
        yield i

# Use like any other function
demo = gr.Interface(my_generator, inputs="slider", outputs="number")
```

### Image Generation Streaming

```python
import gradio as gr
import numpy as np
import time

def fake_diffusion(steps):
    rng = np.random.default_rng()
    for i in range(steps):
        time.sleep(1)
        # Yield intermediate noise images
        image = rng.random(size=(600, 600, 3))
        yield image

    # Yield final image
    image = np.ones((1000, 1000, 3), np.uint8)
    image[:] = [255, 124, 0]
    yield image

demo = gr.Interface(
    fake_diffusion,
    inputs=gr.Slider(1, 10, 3, step=1),
    outputs="image"
)

demo.launch()
```

### Streaming Media: Audio and Video

For audio/video streaming, set `streaming=True` and `autoplay=True`:

**Audio Streaming:**
```python
import gradio as gr
from time import sleep

def keep_repeating(audio_file):
    for _ in range(10):
        sleep(0.5)
        yield audio_file

gr.Interface(
    keep_repeating,
    gr.Audio(sources=["microphone"], type="filepath"),
    gr.Audio(streaming=True, autoplay=True)
).launch()
```

**Video Streaming:**
```python
def keep_repeating(video_file):
    for _ in range(10):
        sleep(0.5)
        yield video_file

gr.Interface(
    keep_repeating,
    gr.Video(sources=["webcam"], format="mp4"),
    gr.Video(streaming=True, autoplay=True)
).launch()
```

**Requirements for smooth playback**:
- Audio: `.mp3`, `.wav` files or `bytes` sequences
- Video: `.mp4` files or `.ts` files with h.264 codec
- Chunks should be consistent length and >1 second for smooth playback
- Setting `streaming=True` enables efficient base64 conversion on server

---

## Section 3: FastRTC Real-Time Communication

**Source**: [github.com/gradio-app/fastrtc](https://github.com/gradio-app/fastrtc)

FastRTC is Gradio's library for building real-time audio/video applications using WebRTC and WebSockets. It enables sub-second latency for interactive AI applications.

### What is FastRTC?

FastRTC turns any Python function into a real-time audio/video stream. Key features:

- 🗣️ **Automatic Voice Detection** with built-in turn-taking (ReplyOnPause)
- 💻 **Automatic UI** via `.ui.launch()` method (Gradio-based)
- 🔌 **Automatic WebRTC Support** via `.mount(app)` on FastAPI
- ⚡️ **WebSocket Support** as alternative to WebRTC
- 📞 **Telephone Support** via `.fastphone()` method (free temporary numbers)
- 🤖 **Customizable Backend** - mount on FastAPI for production

### Installation

```bash
pip install fastrtc

# Optional: For pause detection and TTS
pip install "fastrtc[vad, tts]"
```

### Basic Audio Echo Example

```python
from fastrtc import Stream, ReplyOnPause
import numpy as np

def echo(audio: tuple[int, np.ndarray]):
    # Function receives audio until user pauses
    # Yield audio chunks back
    yield audio

stream = Stream(
    handler=ReplyOnPause(echo),
    modality="audio",
    mode="send-receive"
)
```

### LLM Voice Chat Pattern

```python
from fastrtc import (
    ReplyOnPause, Stream,
    audio_to_bytes, aggregate_bytes_to_16bit
)
import numpy as np
from groq import Groq
import anthropic
from elevenlabs import ElevenLabs

groq_client = Groq()
claude_client = anthropic.Anthropic()
tts_client = ElevenLabs()

def response(audio: tuple[int, np.ndarray]):
    # 1. Transcribe with Whisper
    prompt = groq_client.audio.transcriptions.create(
        file=("audio.mp3", audio_to_bytes(audio)),
        model="whisper-large-v3-turbo",
        response_format="verbose_json"
    ).text

    # 2. Get LLM response
    response = claude_client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    response_text = " ".join(
        block.text for block in response.content
        if getattr(block, "type", None) == "text"
    )

    # 3. Stream TTS audio
    iterator = tts_client.text_to_speech.convert_as_stream(
        text=response_text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="pcm_24000"
    )

    # 4. Yield audio chunks
    for chunk in aggregate_bytes_to_16bit(iterator):
        audio_array = np.frombuffer(chunk, dtype=np.int16).reshape(1, -1)
        yield (24000, audio_array)

stream = Stream(
    modality="audio",
    mode="send-receive",
    handler=ReplyOnPause(response)
)
```

### Webcam Stream Example

```python
from fastrtc import Stream
import numpy as np

def flip_vertically(image):
    return np.flip(image, axis=0)

stream = Stream(
    handler=flip_vertically,
    modality="video",
    mode="send-receive"
)
```

### Running FastRTC Applications

**Option 1: Gradio UI**
```python
stream.ui.launch()
```

**Option 2: Telephone (Audio Only)**
```python
stream.fastphone()  # Get free temporary phone number
```

**Option 3: FastAPI Integration**
```python
from fastapi import FastAPI

app = FastAPI()
stream.mount(app)

# Optional: Serve custom frontend
@app.get("/")
async def _():
    return HTMLResponse(content=open("index.html").read())

# uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## Section 4: Low-Latency Streaming Features (2025)

**Source**: [huggingface.co/blog/why-gradio-stands-out](https://huggingface.co/blog/why-gradio-stands-out)

Gradio 5.0+ introduces several features specifically designed for low-latency, production-grade streaming applications.

### High-Performance Streaming Architecture

**1. Simple Developer Experience**
- Python generators with `yield` statements
- No manual thread management or polling required
- Automatic WebSocket/SSE handling

**2. Token-by-Token Text Streaming**
```python
def stream_text(prompt):
    for token in llm.generate_stream(prompt):
        yield token
```

**3. Step-by-Step Image Generation**
```python
def stream_diffusion(prompt, steps):
    for step in range(steps):
        image = diffusion_model.step(prompt, step)
        yield image
```

**4. HTTP Live Streaming (HLS) Protocol**
- Smooth audio/video streaming
- Automatic chunking and buffering
- Mobile-compatible

**5. WebRTC/WebSocket via FastRTC**
- Sub-second latency for real-time applications
- Bi-directional audio/video
- Built-in voice activity detection

### Server-Side Rendering (SSR) Benefits for Streaming

Gradio 5.0's SSR eliminates loading spinners and reduces initial page load:

- Pre-rendered UI on server → immediate interaction
- Faster time-to-first-stream-chunk
- Better SEO for public demos
- Automatically enabled on HuggingFace Spaces

### Queue Management for Streaming

Gradio's built-in queue handles streaming-specific challenges:

- Long-running stream sessions don't block other users
- Real-time queue position updates via Server-Sent Events
- Configurable `time_limit` for stream events
- Separate concurrency pools via `concurrency_id`

**Example: Mixed queue configuration**
```python
with gr.Blocks() as demo:
    # Fast non-streaming endpoint
    quick_btn.click(
        quick_task,
        inputs=[...],
        outputs=[...],
        concurrency_limit=10
    )

    # Streaming endpoint with time limit
    webcam.stream(
        stream_task,
        inputs=[...],
        outputs=[...],
        time_limit=60,
        concurrency_limit=5
    )
```

### Mobile Responsive Design (2025)

- All Gradio streaming components automatically mobile-responsive
- Touch-optimized controls for webcam/microphone
- Adaptive video quality based on connection
- PWA support for native-like mobile experience

### Best Practices for Production Streaming

1. **Set appropriate `time_limit`**: Prevent queue hogging
2. **Tune `stream_every`**: Balance latency vs server load
3. **Use `concurrency_limit`**: Protect GPU resources
4. **Enable `streaming=True` on output components**: Efficient encoding
5. **Set `autoplay=True` for media**: Immediate playback
6. **Monitor queue depth**: Use Gradio Analytics or custom logging
7. **Test mobile compatibility**: Significant portion of users on mobile
8. **Use FastRTC for <500ms latency requirements**: WebRTC outperforms HTTP
9. **Implement graceful degradation**: Fallback for unsupported browsers
10. **Consider edge deployment**: Reduce geographic latency

---

## Summary

Gradio's 2025 streaming capabilities provide:

- **Bi-directional streaming**: Inputs (webcam, mic) AND outputs (text, image, video, audio)
- **FastRTC integration**: Real-time WebRTC/WebSocket communication in pure Python
- **Automatic queue management**: Handle concurrent users without manual orchestration
- **Production-ready**: SSR, mobile support, security hardening
- **Developer-friendly**: Python generators, no JavaScript required

These features position Gradio as a complete framework for real-time AI applications, from quick prototypes to production deployments.
