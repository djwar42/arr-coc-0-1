# Gradio + FastAPI Integration Patterns

**Current as of**: January 2025
**Target**: Gradio 5.x + FastAPI 0.100+

This guide covers comprehensive patterns for integrating Gradio with FastAPI, enabling you to build production-ready ML applications with both UI and API capabilities.

---

## Section 1: Mounting Gradio in FastAPI

**Source**: [gradio.app/guides/fastapi-app-with-the-gradio-client](https://www.gradio.app/guides/fastapi-app-with-the-gradio-client)

The most common integration pattern: use `gradio_client` to connect a FastAPI backend to an existing Gradio app (either local or hosted on Hugging Face Spaces).

### Use Case: Video Processing with Gradio Client

Building a web app that uses a Gradio Space as the ML backend:

**Prerequisites:**
```bash
pip install gradio_client fastapi uvicorn
```

**Step 1: Connect to Gradio Space**

```python
from gradio_client import Client, handle_file

# Connect to public Space
client = Client("abidlabs/music-separation")

# Or duplicate for private use (no queue)
client = Client.duplicate("abidlabs/music-separation", hf_token=YOUR_HF_TOKEN)

def acapellify(audio_path):
    result = client.predict(handle_file(audio_path), api_name="/predict")
    return result[0]  # Returns [vocals, instrumental]
```

**Step 2: Wrap in Video Processing**

```python
import subprocess
import os

def process_video(video_path):
    # Extract audio from video
    old_audio = os.path.basename(video_path).split(".")[0] + ".m4a"
    subprocess.run([
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-acodec', 'copy', old_audio
    ])

    # Process audio through Gradio Space
    new_audio = acapellify(old_audio)

    # Recombine with video
    new_video = f"acap_{video_path}"
    subprocess.call([
        'ffmpeg', '-y', '-i', video_path, '-i', new_audio,
        '-map', '0:v', '-map', '1:a',
        '-c:v', 'copy', '-c:a', 'aac',
        '-strict', 'experimental',
        f"static/{new_video}"
    ])
    return new_video
```

**Step 3: Create FastAPI Backend**

```python
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

videos = []

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "home.html", {"request": request, "videos": videos}
    )

@app.post("/uploadvideo/")
async def upload_video(video: UploadFile = File(...)):
    video_path = video.filename
    with open(video_path, "wb+") as fp:
        fp.write(video.file.read())

    new_video = process_video(video.filename)
    videos.append(new_video)
    return RedirectResponse(url='/', status_code=303)
```

**Step 4: Create Frontend Template**

Create `templates/home.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Video Gallery</title>
    <style>
        body { font-family: sans-serif; margin: 0; padding: 0; background-color: #f5f5f5; }
        h1 { text-align: center; margin-top: 30px; }
        .gallery { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; padding: 20px; }
        .video { border: 2px solid #ccc; box-shadow: 0px 0px 10px rgba(0,0,0,0.2);
                 border-radius: 5px; overflow: hidden; width: 300px; }
        .video video { width: 100%; height: 200px; }
        .upload-btn { background-color: #3498db; color: #fff; padding: 10px 20px;
                      border: none; border-radius: 5px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>Video Gallery</h1>
    {% if videos %}
    <div class="gallery">
        {% for video in videos %}
        <div class="video">
            <video controls>
                <source src="{{ url_for('static', path=video) }}" type="video/mp4">
            </video>
            <p>{{ video }}</p>
        </div>
        {% endfor %}
    </div>
    {% else %}
    <p>No videos uploaded yet.</p>
    {% endif %}

    <form action="/uploadvideo/" method="post" enctype="multipart/form-data">
        <input type="file" name="video" id="video-upload">
        <button type="submit" class="upload-btn">Upload</button>
    </form>
</body>
</html>
```

**Step 5: Run the App**

```bash
uvicorn main:app --reload
```

### Key Advantages of This Pattern

1. **Reuse existing Gradio Spaces**: No need to reimplement ML models
2. **Separation of concerns**: Gradio handles ML, FastAPI handles web logic
3. **Easy scaling**: Duplicate private Spaces to avoid queues
4. **File handling built-in**: `handle_file()` manages file uploads automatically

---

## Section 2: Using gradio_client from FastAPI

**Source**: [gradio.app/guides/fastapi-app-with-the-gradio-client](https://www.gradio.app/guides/fastapi-app-with-the-gradio-client)

Advanced patterns for programmatic interaction with Gradio apps from FastAPI backends.

### Pattern: LLM Backend Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gradio_client import Client
import asyncio

app = FastAPI()

# Connect to LLM Gradio Space
llm_client = Client("huggingface-projects/llama-2-7b-chat")

class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful assistant"
    max_tokens: int = 512

class ChatResponse(BaseModel):
    response: str
    tokens_used: int

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Call Gradio Space API
        result = llm_client.predict(
            request.message,
            request.system_prompt,
            request.max_tokens,
            api_name="/chat"
        )

        return ChatResponse(
            response=result[0],
            tokens_used=len(result[0].split())  # Approximate
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Pattern: Batch Processing with Queue

```python
from gradio_client import Client
from fastapi import BackgroundTasks
from typing import List
import uuid

app = FastAPI()
client = Client("stabilityai/stable-diffusion-xl")

# Store job results
jobs = {}

class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    result: Optional[str] = None

@app.post("/api/generate")
async def generate_image(prompt: str, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = JobStatus(job_id=job_id, status="pending")

    # Process in background
    background_tasks.add_task(process_image, job_id, prompt)

    return {"job_id": job_id}

def process_image(job_id: str, prompt: str):
    jobs[job_id].status = "processing"
    try:
        result = client.predict(prompt, api_name="/generate")
        jobs[job_id].status = "completed"
        jobs[job_id].result = result
    except Exception as e:
        jobs[job_id].status = "failed"
        jobs[job_id].result = str(e)

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]
```

### Pattern: Streaming Responses

```python
from fastapi.responses import StreamingResponse
import json

@app.post("/api/stream-chat")
async def stream_chat(message: str):
    async def generate():
        # Connect to streaming Gradio app
        client = Client("your-streaming-gradio-space")

        for chunk in client.predict(message, api_name="/chat_stream"):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

---

## Section 3: Multiple Gradio Apps with FastAPI

**Source**: [medium.com/@artistwhocode (LLM integration patterns)](https://medium.com/@artistwhocode/build-an-interactive-gradio-app-for-python-llms-and-fastapi-microservices-in-less-than-2-minutes-4cf8bc885b16)

Running multiple Gradio interfaces within a single FastAPI application.

### Pattern: Microservices Architecture

```python
from fastapi import FastAPI
import gradio as gr

app = FastAPI()

# Define multiple Gradio apps
def image_classifier(image):
    # Image classification logic
    return "cat", 0.95

def text_summarizer(text):
    # Text summarization logic
    return text[:100] + "..."

# Create Gradio interfaces
image_app = gr.Interface(
    fn=image_classifier,
    inputs=gr.Image(),
    outputs=[gr.Label(), gr.Number()]
)

text_app = gr.Interface(
    fn=text_summarizer,
    inputs=gr.Textbox(),
    outputs=gr.Textbox()
)

# Mount at different paths
app = gr.mount_gradio_app(app, image_app, path="/classify")
app = gr.mount_gradio_app(app, text_app, path="/summarize")

# Add custom FastAPI routes
@app.get("/")
async def root():
    return {
        "services": {
            "image_classification": "/classify",
            "text_summarization": "/summarize"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### Pattern: Shared Resources Between Apps

```python
from fastapi import FastAPI, Depends
import gradio as gr
import torch

# Shared model loader
class ModelCache:
    def __init__(self):
        self.models = {}

    def get_model(self, model_name: str):
        if model_name not in self.models:
            self.models[model_name] = torch.load(f"models/{model_name}.pth")
        return self.models[model_name]

cache = ModelCache()

def get_cache():
    return cache

# App 1: Uses shared cache
def classify_image(image):
    model = cache.get_model("resnet50")
    return model(image)

# App 2: Uses same cache
def detect_objects(image):
    model = cache.get_model("yolov8")
    return model(image)

app = FastAPI()

classify_ui = gr.Interface(classify_image, "image", "label")
detect_ui = gr.Interface(detect_objects, "image", "json")

app = gr.mount_gradio_app(app, classify_ui, path="/classify")
app = gr.mount_gradio_app(app, detect_ui, path="/detect")

# FastAPI endpoint using shared cache
@app.get("/models")
async def list_models(cache: ModelCache = Depends(get_cache)):
    return {"loaded_models": list(cache.models.keys())}
```

### Pattern: Dynamic Gradio App Loading

```python
from fastapi import FastAPI
import gradio as gr
from typing import Dict

app = FastAPI()
gradio_apps: Dict[str, gr.Blocks] = {}

def create_chat_app(agent_name: str):
    with gr.Blocks() as demo:
        gr.Markdown(f"# Chat with {agent_name}")
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        msg.submit(lambda x: f"{agent_name} says: {x}", msg, chatbot)
    return demo

@app.post("/api/create-agent/{agent_name}")
async def create_agent(agent_name: str):
    if agent_name in gradio_apps:
        return {"error": "Agent already exists"}

    # Create and mount new Gradio app
    new_app = create_chat_app(agent_name)
    app = gr.mount_gradio_app(app, new_app, path=f"/agents/{agent_name}")
    gradio_apps[agent_name] = new_app

    return {"agent": agent_name, "url": f"/agents/{agent_name}"}

@app.get("/api/agents")
async def list_agents():
    return {"agents": list(gradio_apps.keys())}
```

---

## Section 4: Backend Integration Patterns

**Source**: [medium.com/@artistwhocode](https://medium.com/@artistwhocode/build-an-interactive-gradio-app-for-python-llms-and-fastapi-microservices-in-less-than-2-minutes-4cf8bc885b16)

Real-world patterns for integrating Gradio with complex backend systems.

### Pattern: Session Management

```python
from fastapi import FastAPI, HTTPException
import gradio as gr
import requests
from typing import Dict

app = FastAPI()
BASE_URL = 'http://0.0.0.0:8080'
AGENT_ID = 'your-agent-id'
HEADERS = {'Content-Type': 'application/json'}

# Session management functions
def create_session():
    url = f"{BASE_URL}/api/v1/agents/{AGENT_ID}/sessions"
    response = requests.post(url, headers=HEADERS)
    if response.status_code == 201:
        return response.json().get("id")
    return None

def send_message(session_id, message):
    url = f"{BASE_URL}/api/v1/sessions/{session_id}/messages"
    payload = {'message': message}
    response = requests.post(url, headers=HEADERS, json=payload)
    return response.json()

def accept_action(session_id):
    url = f"{BASE_URL}/api/v1/sessions/{session_id}/actions/accept"
    response = requests.post(url, headers=HEADERS)
    return response.json()

def reject_action(session_id, reason):
    url = f"{BASE_URL}/api/v1/sessions/{session_id}/actions/reject"
    payload = {'reason': reason}
    response = requests.post(url, headers=HEADERS, json=payload)
    return response.json()

# Gradio interface with session state
with gr.Blocks() as demo:
    gr.Markdown("# Chat with LLM via FastAPI")
    chatbot = gr.Chatbot()
    state = gr.State()  # Stores session_id

    with gr.Row():
        user_input = gr.Textbox(
            show_label=False,
            placeholder="Type your message here..."
        )
        submit_button = gr.Button("Send")

    with gr.Row(visible=False) as action_buttons:
        accept_button = gr.Button("Accept")
        reject_button = gr.Button("Reject")

    def handle_message(user_message, chat_history, state):
        # Create session if doesn't exist
        if "session_id" not in state:
            session_id = create_session()
            state["session_id"] = session_id

        session_id = state.get("session_id")
        response_data = send_message(session_id, user_message)
        bot_reply = response_data.get("response", "No response")
        chat_history.append((user_message, bot_reply))

        # Show action buttons if confirmation needed
        if response_data.get("state") == "WAITING_FOR_CONFIRMATION":
            return chat_history, state, gr.update(visible=True)
        return chat_history, state, gr.update(visible=False)

    submit_button.click(
        handle_message,
        inputs=[user_input, chatbot, state],
        outputs=[chatbot, state, action_buttons]
    )

    def handle_accept(chat_history, state):
        session_id = state.get("session_id")
        response = accept_action(session_id)
        chat_history.append(("Action accepted", response.get("response")))
        return chat_history, state, gr.update(visible=False)

    def handle_reject(chat_history, state):
        session_id = state.get("session_id")
        response = reject_action(session_id, "User rejected")
        chat_history.append(("Action rejected", response.get("response")))
        return chat_history, state, gr.update(visible=False)

    accept_button.click(
        handle_accept,
        inputs=[chatbot, state],
        outputs=[chatbot, state, action_buttons]
    )
    reject_button.click(
        handle_reject,
        inputs=[chatbot, state],
        outputs=[chatbot, state, action_buttons]
    )

# Mount Gradio on FastAPI
app = gr.mount_gradio_app(app, demo, path="/chat")
```

### Pattern: Database Integration

```python
from fastapi import FastAPI
import gradio as gr
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database setup
DATABASE_URL = "sqlite:///./gradio_app.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    user_message = Column(Text)
    bot_response = Column(Text)
    session_id = Column(String)

Base.metadata.create_all(bind=engine)

def save_to_db(session_id: str, user_msg: str, bot_msg: str):
    db = SessionLocal()
    conversation = Conversation(
        session_id=session_id,
        user_message=user_msg,
        bot_response=bot_msg
    )
    db.add(conversation)
    db.commit()
    db.close()

def chat_with_db(message, history, session_state):
    response = f"Echo: {message}"

    # Save to database
    session_id = session_state.get("id", "default")
    save_to_db(session_id, message, response)

    history.append((message, response))
    return history, session_state

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    state = gr.State({"id": "user123"})

    msg.submit(chat_with_db, [msg, chatbot, state], [chatbot, state])

app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/")

# Add API endpoint to query history
@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    db = SessionLocal()
    conversations = db.query(Conversation).filter(
        Conversation.session_id == session_id
    ).all()
    db.close()
    return [
        {"user": c.user_message, "bot": c.bot_response}
        for c in conversations
    ]
```

### Pattern: Authentication Integration

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import gradio as gr
import jwt

app = FastAPI()
security = HTTPBearer()
SECRET_KEY = "your-secret-key"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload["user_id"]
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

# Protected Gradio app
def protected_function(message):
    return f"Authorized response: {message}"

with gr.Blocks() as demo:
    gr.Markdown("# Protected Chat Interface")
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    msg.submit(protected_function, msg, chatbot)

# Mount with authentication middleware
@app.middleware("http")
async def auth_middleware(request, call_next):
    if request.url.path.startswith("/gradio"):
        # Check for auth token in cookies or headers
        token = request.cookies.get("auth_token")
        if not token:
            return JSONResponse(
                status_code=401,
                content={"detail": "Not authenticated"}
            )
    return await call_next(request)

app = gr.mount_gradio_app(app, demo, path="/gradio")

# Login endpoint
@app.post("/api/login")
async def login(username: str, password: str):
    # Verify credentials (simplified)
    if username == "admin" and password == "secret":
        token = jwt.encode({"user_id": username}, SECRET_KEY, algorithm="HS256")
        return {"token": token}
    raise HTTPException(status_code=401, detail="Invalid credentials")
```

---

## Best Practices

1. **Use `gr.mount_gradio_app()`**: Official method for mounting Gradio on FastAPI
2. **Separate concerns**: FastAPI for business logic, Gradio for UI
3. **State management**: Use `gr.State()` for session data
4. **Error handling**: Wrap Gradio calls in try-except for production
5. **Resource sharing**: Use FastAPI dependencies for shared resources
6. **Authentication**: Implement at FastAPI level, not Gradio level
7. **Database integration**: Use SQLAlchemy with FastAPI patterns
8. **API design**: Expose both Gradio UI and REST endpoints
9. **Background tasks**: Use FastAPI's `BackgroundTasks` for long operations
10. **Testing**: Test FastAPI endpoints separately from Gradio UI

---

## Summary

Gradio + FastAPI integration enables:

- **Hybrid apps**: Both UI (Gradio) and API (FastAPI) from single codebase
- **Microservices**: Multiple Gradio apps in one FastAPI server
- **Production features**: Authentication, database, session management
- **Flexible architecture**: Gradio client can connect to remote Gradio Spaces
- **Scalability**: FastAPI's async capabilities + Gradio's queue system

This combination provides the best of both worlds: Gradio's rapid UI development and FastAPI's production-ready web framework.
