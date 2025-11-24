# Gradio Production Security & Deployment (2025)

**Category**: Practical Implementation
**Related**: [15-gradio-performance-optimization.md](15-gradio-performance-optimization.md), [14-gradio-fastapi-integration-patterns.md](14-gradio-fastapi-integration-patterns.md)

## Overview

Production deployment of Gradio applications requires comprehensive security measures, authentication systems, and deployment strategies. This guide covers built-in authentication, JWT-based authorization, SSO integration, and production deployment patterns for Gradio 5.x (2025).

---

## Built-in Authentication

### Basic Authentication

Gradio provides simple username/password authentication out of the box.

**Single User Authentication:**

```python
import gradio as gr

def predict(input_text):
    return f"Processed: {input_text}"

demo = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text"
)

# Single user authentication
demo.launch(auth=("admin", "password123"))
```

**Multiple User Authentication:**

```python
# List of (username, password) tuples
demo.launch(auth=[
    ("alice", "password1"),
    ("bob", "password2"),
    ("charlie", "password3")
])
```

**Function-Based Authentication:**

```python
def authenticate(username, password):
    """Custom authentication logic"""
    valid_users = {
        "admin": "admin123",
        "user": "user456"
    }
    return valid_users.get(username) == password

demo.launch(auth=authenticate)
```

### Authentication UI Features

From [gradio.app/guides/sharing-your-app](https://www.gradio.app/guides/sharing-your-app):

- Automatic login page generation
- Session persistence across page reloads
- Logout functionality
- Password masking in input fields

**Limitations of Built-in Auth:**

- No role-based access control (RBAC)
- No session management capabilities
- Passwords stored in plain text in code
- Not suitable for multi-tenant applications

---

## JWT & Advanced Authorization

For production applications requiring role-based access, JWT (JSON Web Tokens) provide stateless authentication with enhanced security.

### Gradio-Session Architecture

From [medium.com/@marek.gmyrek/gradio-from-prototype-to-production](https://medium.com/@marek.gmyrek/gradio-from-prototype-to-production-secure-scalable-gradio-apps-for-data-scientists-739cebaf669b):

**Core Components:**

1. **Authentication Layer (FastAPI)**
   - JWT token generation on login
   - Token validation middleware
   - Role-based access control

2. **Session Management**
   - External session store (in-memory, Redis, SQL)
   - Session ID encoded in JWT
   - Pluggable backend architecture

3. **Middleware Integration**
   - JWT extraction from cookies/headers
   - Session injection into Gradio handlers
   - Request/response logging

### Implementation Example

**Project Structure:**

```
gradio_app/
├── core/
│   ├── logging.py          # Loguru logging setup
│   └── session.py          # Session helpers
├── middleware/
│   ├── auth.py            # JWT validation
│   ├── session.py         # Session injection
│   └── logging.py         # Request logging
├── routes/
│   ├── login.py           # Login/logout endpoints
│   └── health.py          # Health checks
├── services/
│   ├── auth.py            # JWT creation/verification
│   ├── csrf.py            # CSRF protection
│   └── database.py        # User database
├── session/
│   ├── session_store.py   # Protocol interface
│   ├── inmemory_session.py
│   ├── redis_session.py
│   └── database_session.py
├── ui/
│   ├── gradio_app.py      # Main Gradio interface
│   └── navbar.py          # Navigation components
└── main.py                # App entry point
```

**JWT Authentication Service:**

```python
# services/auth.py
import jwt
from datetime import datetime, timedelta
from typing import Optional

class AuthService:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm

    def create_token(
        self,
        user_id: str,
        session_id: str,
        roles: list[str],
        expires_in_hours: int = 24
    ) -> str:
        """Create JWT token with user claims"""
        expiration = datetime.utcnow() + timedelta(hours=expires_in_hours)

        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "roles": roles,
            "exp": expiration,
            "iat": datetime.utcnow()
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Optional[dict]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            print("Token has expired")
            return None
        except jwt.InvalidTokenError:
            print("Invalid token")
            return None
```

**Session Store Interface:**

```python
# session/session_store.py
from typing import Protocol, Any, Optional

class SessionStore(Protocol):
    """Protocol for session storage backends"""

    def get(self, session_id: str) -> Optional[dict[str, Any]]:
        """Retrieve session data"""
        ...

    def set(self, session_id: str, data: dict[str, Any]) -> None:
        """Store session data"""
        ...

    def delete(self, session_id: str) -> None:
        """Remove session"""
        ...

    def exists(self, session_id: str) -> bool:
        """Check if session exists"""
        ...
```

**In-Memory Session Implementation:**

```python
# session/inmemory_session.py
from typing import Any, Optional
from threading import Lock

class InMemorySessionStore:
    def __init__(self):
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def get(self, session_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            return self._sessions.get(session_id, {}).copy()

    def set(self, session_id: str, data: dict[str, Any]) -> None:
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].update(data)
            else:
                self._sessions[session_id] = data.copy()

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions
```

**Authentication Middleware:**

```python
# middleware/auth.py
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, auth_service, session_store):
        super().__init__(app)
        self.auth_service = auth_service
        self.session_store = session_store

    async def dispatch(self, request: Request, call_next):
        # Skip auth for public routes
        if request.url.path in ["/login", "/health", "/static"]:
            return await call_next(request)

        # Extract JWT from cookie
        token = request.cookies.get("access_token")

        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated"
            )

        # Verify token
        payload = self.auth_service.verify_token(token)

        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )

        # Retrieve session
        session_id = payload["session_id"]
        session_data = self.session_store.get(session_id)

        # Inject into request state
        request.state.user_id = payload["user_id"]
        request.state.roles = payload["roles"]
        request.state.session = session_data

        return await call_next(request)
```

**Gradio Integration:**

```python
# main.py
from fastapi import FastAPI
from gradio import mount_gradio_app
import gradio as gr

app = FastAPI()

# Initialize services
auth_service = AuthService(secret_key="your-secret-key")
session_store = InMemorySessionStore()

# Add middleware
app.add_middleware(
    AuthMiddleware,
    auth_service=auth_service,
    session_store=session_store
)

# Gradio app with session access
def process_with_session(text, request: gr.Request):
    """Access session data in Gradio handlers"""
    user_id = request.state.user_id
    session = request.state.session

    return f"User {user_id}: {text}"

with gr.Blocks() as demo:
    input_box = gr.Textbox(label="Input")
    output_box = gr.Textbox(label="Output")
    btn = gr.Button("Process")

    btn.click(process_with_session, inputs=input_box, outputs=output_box)

# Mount Gradio with auth dependency
app = mount_gradio_app(app, demo, path="/")
```

### Benefits of JWT Architecture

From [medium.com/@marek.gmyrek](https://medium.com/@marek.gmyrek/gradio-from-prototype-to-production-secure-scalable-gradio-apps-for-data-scientists-739cebaf669b):

- **Stateless scaling**: No server-side session affinity required
- **Pluggable backends**: Swap in-memory for Redis/SQL in production
- **Role-based access**: Fine-grained permissions per user
- **Multi-tenant support**: Isolate sessions per user/organization
- **Cloud-native**: Compatible with containerized deployments

---

## SSO Integration with OIDC

Single Sign-On enables enterprise authentication using existing identity providers like Okta, Azure AD, or Google Workspace.

### OIDC with Descope

From [descope.com/blog/post/auth-sso-gradio](https://www.descope.com/blog/post/auth-sso-gradio):

**Setup Process:**

1. **Configure Descope Tenant**
   - Create tenant in Descope dashboard
   - Add email domain for automatic routing
   - Enable SSO authentication method

2. **Set Up OIDC App in Identity Provider**
   - Create OIDC application (e.g., in Okta)
   - Configure callback URL: `https://api.descope.com/v1/oauth/callback`
   - Obtain client ID and secret

3. **Configure Descope SSO**
   - Select OIDC protocol
   - Add SSO domains for email routing
   - Enter IdP OAuth endpoints
   - Configure client credentials

**Okta Integration Example:**

```python
# Configuration from Okta well-known endpoint
# https://<YOUR-OKTA-INSTANCE>.okta.com/.well-known/openid-configuration

okta_config = {
    "provider_name": "Okta",
    "client_id": "your-client-id",
    "client_secret": "your-client-secret",
    "scope": "openid profile email",
    "grant_type": "authorization_code",
    "authorization_endpoint": "https://your-okta.com/oauth2/v1/authorize",
    "token_endpoint": "https://your-okta.com/oauth2/v1/token",
    "userinfo_endpoint": "https://your-okta.com/oauth2/v1/userinfo",
    "jwks_uri": "https://your-okta.com/oauth2/v1/keys"
}
```

**FastAPI + Descope + Gradio:**

```python
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth
import gradio as gr

app = FastAPI()

# OAuth configuration
oauth = OAuth()
oauth.register(
    name='descope',
    client_id='your-descope-client-id',
    client_secret='your-descope-client-secret',
    authorize_url='https://api.descope.com/oauth/authorize',
    access_token_url='https://api.descope.com/oauth/token',
    userinfo_endpoint='https://api.descope.com/oauth/userinfo',
    client_kwargs={'scope': 'openid profile email descope.claims'}
)

@app.get('/login')
async def login(request: Request):
    """Redirect to Descope login"""
    redirect_uri = request.url_for('auth_callback')
    return await oauth.descope.authorize_redirect(request, redirect_uri)

@app.get('/auth/callback')
async def auth_callback(request: Request):
    """Handle OAuth callback"""
    token = await oauth.descope.authorize_access_token(request)
    user = token.get('userinfo')

    # Store in session
    request.session['user'] = dict(user)
    request.session['tenants'] = user.get('tenants', {})

    return RedirectResponse(url='/')

@app.get('/logout')
async def logout(request: Request):
    """Clear session"""
    request.session.clear()
    return RedirectResponse(url='/login')

# Gradio app with role check
def check_admin_role(request: gr.Request):
    """Check if user has admin role"""
    user = request.session.get('user', {})
    tenants = user.get('tenants', {})

    roles = set()
    for tenant_data in tenants.values():
        roles.update(tenant_data.get('roles', []))

    return 'Tenant Admin' in roles

def protected_function(input_text, request: gr.Request):
    """Only accessible to authenticated users"""
    if not check_admin_role(request):
        raise gr.Error("Unauthorized: Admin access required")

    return f"Admin processed: {input_text}"

with gr.Blocks() as demo:
    gr.Markdown("# Protected Admin Dashboard")
    input_box = gr.Textbox(label="Input")
    output_box = gr.Textbox(label="Output")
    btn = gr.Button("Process")

    btn.click(protected_function, inputs=input_box, outputs=output_box)

app = mount_gradio_app(app, demo, path="/", auth_dependency=check_admin_role)
```

### SSO Flow

From [descope.com/blog](https://www.descope.com/blog/post/auth-sso-gradio):

1. User enters email in Descope flow
2. Email domain triggers SSO configuration
3. Redirect to identity provider (Okta)
4. User authenticates with IdP credentials
5. IdP redirects back with authorization code
6. Descope exchanges code for JWT token
7. Token contains user claims and roles
8. Application validates token and grants access

---

## Production Deployment Patterns

### Hugging Face Spaces

From [shafiqulai.github.io/blogs/blog_5.html](https://shafiqulai.github.io/blogs/blog_5.html):

**Deployment Workflow:**

```bash
# 1. Create project structure
gradio_app/
├── app.py              # Main Gradio app (required)
├── requirements.txt    # Dependencies
├── README.md          # Space documentation
├── .env.example       # Environment template
└── [other files]      # Supporting code

# 2. Configure requirements.txt
gradio==5.49.14
transformers==4.53.0
torch==2.5.0
```

**app.py Requirements:**

- Must be named `app.py` (Spaces entry point)
- Should be in project root
- Launch demo at end: `demo.launch()`

**Environment Variables:**

```python
# Access secrets securely
import os

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")

# Configure in Spaces: Settings → Variables and Secrets
```

**Git Deployment:**

```bash
# Method 1: Git push
git lfs install
git clone https://huggingface.co/spaces/username/space-name
cd space-name
# Add files
git add .
git commit -m "Initial deployment"
git push

# Method 2: Web interface
# Files → + Contribute → Upload files
```

**Hardware Selection:**

From [shafiqulai.github.io](https://shafiqulai.github.io/blogs/blog_5.html):

- **CPU Basic** (2 vCPUs, 16GB RAM): Free tier
- **GPU T4** (16GB VRAM): $0.60/hour
- **GPU A10G** (24GB VRAM): $3.15/hour
- **GPU A100** (40GB VRAM): $4.13/hour

**Space Configuration (README.md):**

```yaml
---
title: My Gradio App
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.14
app_file: app.py
pinned: false
license: mit
short_description: AI-powered application
---
```

### AWS/Azure Deployment

From [medium.com/@marek.gmyrek](https://medium.com/@marek.gmyrek/gradio-from-prototype-to-production-secure-scalable-gradio-apps-for-data-scientists-739cebaf669b):

**Docker Containerization:**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 7860

# Run application
CMD ["python", "main.py"]
```

**AWS ECS Deployment:**

```bash
# Build and push image
docker build -t gradio-app .
docker tag gradio-app:latest <account>.dkr.ecr.region.amazonaws.com/gradio-app:latest
docker push <account>.dkr.ecr.region.amazonaws.com/gradio-app:latest

# Create ECS task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Deploy service
aws ecs create-service \
  --cluster production-cluster \
  --service-name gradio-app \
  --task-definition gradio-app \
  --desired-count 2 \
  --launch-type FARGATE
```

**Load Balancer Configuration:**

```json
{
  "Type": "application",
  "Scheme": "internet-facing",
  "IpAddressType": "ipv4",
  "Subnets": ["subnet-xxx", "subnet-yyy"],
  "SecurityGroups": ["sg-xxx"],
  "Tags": [{"Key": "Name", "Value": "gradio-alb"}]
}
```

**Azure Container Instances:**

```bash
# Deploy to Azure
az container create \
  --resource-group gradio-rg \
  --name gradio-app \
  --image <registry>.azurecr.io/gradio-app:latest \
  --cpu 2 \
  --memory 4 \
  --port 7860 \
  --dns-name-label gradio-app \
  --environment-variables \
    API_KEY=secure-key \
    DATABASE_URL=connection-string
```

---

## Security Best Practices

### Environment Variables

From [gradio.app/guides/environment-variables](https://www.gradio.app/guides/environment-variables):

**Security-Related Variables:**

```bash
# File access control
GRADIO_ALLOWED_PATHS=/data,/models,/uploads

# Server-side rendering
GRADIO_SSR_MODE=1  # Enable in production

# Share server
GRADIO_SHARE_SERVER_ADDRESS=https://your-frp-server.com

# Temporary file handling
GRADIO_TEMP_DIR=/secure/tmp

# Analytics (disable in production)
GRADIO_ANALYTICS_ENABLED=0
```

### File Upload Security

**Validate uploads:**

```python
import gradio as gr
import os

def secure_file_handler(file):
    """Validate and process uploaded files"""

    # Check file size (10MB limit)
    max_size = 10 * 1024 * 1024
    file_size = os.path.getsize(file.name)

    if file_size > max_size:
        raise gr.Error("File too large (max 10MB)")

    # Check file type
    allowed_extensions = {'.jpg', '.png', '.pdf', '.txt'}
    _, ext = os.path.splitext(file.name)

    if ext.lower() not in allowed_extensions:
        raise gr.Error(f"File type {ext} not allowed")

    # Sanitize filename
    safe_name = os.path.basename(file.name)

    # Process in allowed directory
    if not file.name.startswith(os.environ['GRADIO_ALLOWED_PATHS']):
        raise gr.Error("Invalid file path")

    return f"Processed: {safe_name}"

demo = gr.Interface(
    fn=secure_file_handler,
    inputs=gr.File(label="Upload"),
    outputs="text"
)
```

### Input Sanitization

```python
import re
from html import escape

def sanitize_input(text: str) -> str:
    """Sanitize user input"""

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Escape special characters
    text = escape(text)

    # Limit length
    max_length = 1000
    text = text[:max_length]

    return text

def safe_handler(user_input):
    """Process sanitized input"""
    clean_input = sanitize_input(user_input)
    return f"Processed: {clean_input}"
```

### Rate Limiting

```python
from functools import wraps
from time import time
from collections import defaultdict

# Simple in-memory rate limiter
request_times = defaultdict(list)

def rate_limit(max_calls: int, time_window: int):
    """Rate limit decorator"""
    def decorator(func):
        @wraps(func)
        def wrapper(request: gr.Request, *args, **kwargs):
            user_id = request.session.get('user_id', request.client.host)
            current_time = time()

            # Clean old requests
            request_times[user_id] = [
                t for t in request_times[user_id]
                if current_time - t < time_window
            ]

            # Check limit
            if len(request_times[user_id]) >= max_calls:
                raise gr.Error(
                    f"Rate limit exceeded. Max {max_calls} requests per {time_window}s"
                )

            # Record request
            request_times[user_id].append(current_time)

            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(max_calls=10, time_window=60)
def protected_endpoint(text):
    """Limited to 10 calls per minute"""
    return f"Processed: {text}"
```

---

## Summary

Production Gradio applications require:

**Authentication:**
- Built-in auth for simple use cases
- JWT + FastAPI for role-based access
- SSO integration for enterprise deployments

**Session Management:**
- External session stores (Redis, SQL)
- Pluggable backend architecture
- Stateless scaling support

**Deployment:**
- Hugging Face Spaces for quick hosting
- Docker + AWS/Azure for enterprise
- Environment-based configuration

**Security:**
- File upload validation
- Input sanitization
- Rate limiting
- Environment variable secrets

**Key Principles:**
- Never hardcode credentials
- Use HTTPS in production
- Implement proper logging
- Regular security audits
- Defense in depth approach

These patterns enable secure, scalable production deployments while maintaining Gradio's simplicity for rapid development.
