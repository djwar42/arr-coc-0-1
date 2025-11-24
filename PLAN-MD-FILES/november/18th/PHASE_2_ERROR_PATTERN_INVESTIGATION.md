# 🔍 PHASE 2: Error Pattern Investigation

**Complete breakdown of what we're matching and WHERE it comes from!**

---

## 🎯 **KEY INSIGHT**

**We have TWO types of patterns:**
1. **OUR OWN OUTPUT** - Messages WE print from `entrypoint-wrapper.sh`
2. **EXTERNAL ERRORS** - Real errors from GCP/W&B/Python

**When we see "🚨 FATAL ERROR DETECTED" in logs → That's US detecting an error and printing it!**

---

## 📋 **COMPLETE PATTERN BREAKDOWN**

### **🚨 Priority 1: Wrapper Bailout Detection**

#### **Pattern: "🚨 FATAL ERROR DETECTED"**

**✅ MATCHING OUR OWN STUFF HERE!**
- **Source**: `training/images/arr-vertex-launcher/entrypoint-wrapper.sh`
- **Lines**: 144, 153, 162, 170, 180, 192, 202, 212, 228, 238
- **What it is**: Our wrapper script prints this when it detects a fatal error!
- **Purpose**: Fast bailout marker - we print this, then kill the agent

**Example from wrapper:**
```bash
if tail -100 "$LOG_FILE" | grep -q "Machine type.*is not supported"; then
    echo "🚨 FATAL ERROR DETECTED: Machine type not supported!"  # ← WE print this!
    show_error_context "Machine type.*is not supported" ...
    echo "❌ Killing agent (PID: $AGENT_PID) - this error will not self-resolve"  # ← WE print this!
    kill "$AGENT_PID" 2>/dev/null || true
    exit 1
fi
```

**When monitoring code sees "🚨 FATAL ERROR DETECTED" → It's finding OUR bailout message!**

---

#### **Pattern: "❌ Killing agent"**

**✅ MATCHING OUR OWN STUFF HERE!**
- **Source**: `training/images/arr-vertex-launcher/entrypoint-wrapper.sh`
- **Lines**: 146, 155, 164, 173, 182, 193, 203, 213, 230, 240
- **What it is**: We print this right before killing the W&B agent
- **Purpose**: Shows WHY we're killing the agent (quota, permissions, etc.)

**Example:**
```bash
echo "❌ Killing agent (PID: $AGENT_PID) - quota limit reached"  # ← WE print this!
```

**When monitoring code sees "❌ Killing agent" → It's finding OUR shutdown message!**

---

### **📍 Priority 2: Underlying Error Patterns** (What the wrapper is DETECTING)

These are the REAL errors from GCP/W&B that TRIGGER our bailout!

---

#### **1. Machine Type Error**

**Pattern**: `"Machine type.*is not supported"`

**❌ NOT OUR OUTPUT - This is a GCP error!**
- **Source**: Google Cloud Platform API responses
- **When**: GPU/machine type incompatibility
- **Example from GCP**:
```
InvalidArgument: 400 Machine type 'n2-standard-4' is not supported for GPU 'NVIDIA_TESLA_T4' in zone 'us-central1-a'
```

**Flow:**
1. W&B Launch tries to start Vertex AI job
2. GCP API returns error: "Machine type 'n2-standard-4' is not supported..."
3. W&B agent logs this error to `/tmp/wandb-agent.log`
4. Our wrapper sees this error in logs
5. Our wrapper prints "🚨 FATAL ERROR DETECTED: Machine type not supported!"
6. Our wrapper prints "❌ Killing agent..."
7. Our wrapper kills agent and exits

**What we're extracting for the table:** The GCP error message (not our wrapper message!)

---

#### **2. Invalid Argument (400)**

**Pattern**: `"InvalidArgument: 400"`

**❌ NOT OUR OUTPUT - This is a GCP error!**
- **Source**: Google Cloud Platform API
- **When**: Bad request parameters (invalid config)
- **Example**:
```
InvalidArgument: 400 Invalid resource specification
```

**Wrapper detects this** (line 152):
```bash
if tail -100 "$LOG_FILE" | grep -q "InvalidArgument: 400"; then
    echo "🚨 FATAL ERROR DETECTED: Invalid argument (400)!"
    ...
```

---

#### **3. Permission Denied (403)**

**Pattern**: `"PermissionDenied: 403"`

**❌ NOT OUR OUTPUT - This is a GCP error!**
- **Source**: Google Cloud IAM
- **When**: Missing service account permissions
- **Example**:
```
PermissionDenied: 403 Permission 'compute.instances.create' denied
```

**Wrapper detects this** (line 161):
```bash
if tail -100 "$LOG_FILE" | grep -q "PermissionDenied: 403"; then
    echo "🚨 FATAL ERROR DETECTED: Permission denied (403)!"
    ...
```

---

#### **4. Not Found (404)**

**Pattern**: `"NotFound: 404"`

**❌ NOT OUR OUTPUT - This is a GCP error!**
- **Source**: Google Cloud APIs
- **When**: Resource doesn't exist (bucket, network, etc.)
- **Example**:
```
NotFound: 404 The specified bucket does not exist
```

**Wrapper detects this** (line 170):
```bash
if tail -100 "$LOG_FILE" | grep -q "NotFound: 404"; then
    echo "🚨 FATAL ERROR DETECTED: Resource not found (404)!"
    ...
```

---

#### **5. Quota Exceeded**

**Pattern**: `"QuotaExceeded|ResourceExhausted"`

**❌ NOT OUR OUTPUT - This is a GCP error!**
- **Source**: Google Cloud Quota System
- **When**: Project quota limits reached (GPUs, CPUs, etc.)
- **Example**:
```
QuotaExceeded: Quota 'NVIDIA_L4_GPUS' exceeded. Limit: 0 in region us-west2
```

**Wrapper detects this** (line 179):
```bash
if tail -100 "$LOG_FILE" | grep -q "QuotaExceeded\|ResourceExhausted"; then
    echo "🚨 FATAL ERROR DETECTED: Quota exceeded!"
    ...
```

---

#### **6. HTTP Errors (5xx)**

**Pattern**: `any(pattern in ctx_line for pattern in ['503', 'ServiceUnavailable', '500', 'Internal Error', 'Internal error'])`

**❌ NOT OUR OUTPUT - This is a GCP error!**
- **Source**: Google Cloud APIs (service unavailable, internal errors)
- **When**: GCP backend issues
- **Example**:
```
HttpError: <HttpError 503 when requesting ... returned "Service Unavailable">
```

**Wrapper detects HTTP errors** (line 210):
```bash
if tail -50 "$LOG_FILE" | grep -qE "HttpError: <HttpError [45][0-9]{2}"; then
    echo "🚨 FATAL ERROR DETECTED: HTTP error from GCP API!"
    ...
```

---

#### **7. HTTP Error Codes**

**Pattern**: `'HttpError' in ctx_line and any(code in ctx_line for code in ['400', '401', '403', '404', '429', '500', '502', '503'])`

**❌ NOT OUR OUTPUT - This is a GCP error!**
- **Source**: GCP API structured error responses
- **When**: Various API failures
- **Example**:
```
HttpError: <HttpError 429 when requesting ... returned "Too Many Requests">
```

---

#### **8. Image Pull Errors**

**Pattern**: `'ImagePullBackOff' in ctx_line or 'ErrImagePull' in ctx_line`

**❌ NOT OUR OUTPUT - This is a Kubernetes/GCP error!**
- **Source**: Kubernetes / GCP Container Registry
- **When**: Cannot pull Docker image from Artifact Registry
- **Example**:
```
ImagePullBackOff: Failed to pull image 'us-west2-docker.pkg.dev/project/repo/image:tag'
```

**Wrapper detects this** (line 237):
```bash
if tail -100 "$LOG_FILE" | grep -qE "ImagePullBackOff|ErrImagePull|Failed to pull image"; then
    echo "🚨 FATAL ERROR DETECTED: Container image pull failure!"
    ...
```

---

#### **9. Python Exceptions**

**Pattern**: `'Traceback' in ctx_line or 'Exception:' in ctx_line or 'Error:' in ctx_line`

**❌ NOT OUR OUTPUT - This is Python!**
- **Source**: Python exceptions from W&B agent or training code
- **When**: Unhandled exceptions
- **Example**:
```
Traceback (most recent call last):
  File "...", line 123, in foo
ValueError: Invalid input
```

**Wrapper detects this** (line 200):
```bash
if tail -50 "$LOG_FILE" | grep -A3 "Traceback (most recent call last)" | grep -q "Error:\|Exception:"; then
    echo "🚨 FATAL ERROR DETECTED: Unhandled Python exception!"
    ...
```

---

### **🎯 Priority 3: W&B Agent Errors**

#### **Pattern: "wandb: ERROR"**

**❌ NOT OUR OUTPUT - This is from W&B Launch agent!**
- **Source**: W&B Launch agent log output
- **When**: Agent encounters errors (internal W&B errors)
- **Example**:
```
wandb: ERROR Machine type 'g2-standard-4' is not supported for GPU 'NVIDIA_L4'
```

**Why this exists:**
- W&B Launch agent prefixes its error logs with "wandb: ERROR"
- We search for this to extract W&B-specific errors
- Often contains the SAME errors as GCP (just re-logged by W&B)

---

### **ℹ️ Priority 4: Info Message Filtering**

#### **Pattern: Emoji Indicators**

**Pattern**: `any(info_indicator in line for info_indicator in ['⏱️', '⏳', 'ℹ️', '🔍'])`

**✅ MATCHING OUR OWN STUFF HERE!**
- **Source**: `training/images/arr-vertex-launcher/entrypoint-wrapper.sh`
- **What it is**: Status/info emojis we print for monitoring messages
- **Purpose**: Filter out false positives (these lines aren't errors!)

**Examples from wrapper:**
```bash
echo "⏱️  Idle timeout: ${IDLE_TIMEOUT_MINUTES} minutes"  # Line 35
echo "⏳ Monitoring for fatal errors and idle timeout..."  # Line 51
```

**When we skip lines with ⏱️/⏳/ℹ️/🔍 → We're avoiding OUR OWN info messages!**

---

#### **Pattern: Monitoring Keywords**

**Pattern**: `any(info_pattern in line.lower() for info_pattern in ['monitoring for', 'checking for', 'watching for', 'looking for'])`

**✅ MATCHING OUR OWN STUFF HERE!**
- **Source**: `training/images/arr-vertex-launcher/entrypoint-wrapper.sh`
- **What it is**: Our status messages about what we're monitoring
- **Purpose**: Filter out false positives

**Example from wrapper:**
```bash
echo "⏳ Monitoring for fatal errors and idle timeout..."  # Line 51 ← OUR message!
```

**When we skip "monitoring for" → We're avoiding OUR OWN status messages!**

---

## 🎯 **SUMMARY: What's Ours vs Theirs**

| Pattern | Type | Source |
|---------|------|--------|
| 🚨 FATAL ERROR DETECTED | **OUR OUTPUT** | entrypoint-wrapper.sh |
| ❌ Killing agent | **OUR OUTPUT** | entrypoint-wrapper.sh |
| ⏱️ ⏳ ℹ️ 🔍 (emojis) | **OUR OUTPUT** | entrypoint-wrapper.sh |
| "monitoring for" | **OUR OUTPUT** | entrypoint-wrapper.sh |
| Machine type.*is not supported | **GCP ERROR** | Google Cloud API |
| InvalidArgument: 400 | **GCP ERROR** | Google Cloud API |
| PermissionDenied: 403 | **GCP ERROR** | Google Cloud IAM |
| NotFound: 404 | **GCP ERROR** | Google Cloud API |
| QuotaExceeded | **GCP ERROR** | Google Cloud Quota |
| ResourceExhausted | **GCP ERROR** | Google Cloud Quota |
| HttpError [45]xx | **GCP ERROR** | Google Cloud API |
| ImagePullBackOff | **K8S ERROR** | Kubernetes/GCP |
| ErrImagePull | **K8S ERROR** | Kubernetes/GCP |
| wandb: ERROR | **W&B OUTPUT** | W&B Launch agent |
| Traceback | **PYTHON ERROR** | Python exceptions |
| Exception: / Error: | **PYTHON ERROR** | Python exceptions |

---

## 🚀 **THE FLOW**

**When a FAILED execution happens:**

1. **Real error occurs** (GCP/W&B/Python)
   ```
   QuotaExceeded: Quota 'NVIDIA_L4_GPUS' exceeded. Limit: 0
   ```

2. **W&B agent logs it**
   ```
   wandb: ERROR QuotaExceeded: Quota 'NVIDIA_L4_GPUS' exceeded...
   ```

3. **Our wrapper detects it**
   ```bash
   if tail -100 "$LOG_FILE" | grep -q "QuotaExceeded"; then
   ```

4. **Our wrapper prints bailout message**
   ```
   🚨 FATAL ERROR DETECTED: Quota exceeded!
   ❌ Killing agent (PID: 1234) - quota limit reached
   ```

5. **Monitoring code finds BOTH**
   - OUR message: "🚨 FATAL ERROR DETECTED"
   - REAL error: "QuotaExceeded: Quota 'NVIDIA_L4_GPUS' exceeded..."

6. **We extract the REAL error (not our message!)**
   ```python
   # We want: "QuotaExceeded: Quota 'NVIDIA_L4_GPUS' exceeded..."
   # NOT: "🚨 FATAL ERROR DETECTED: Quota exceeded!"
   ```

---

## ✅ **WHAT TO SHOW IN THE TABLE**

**For each FAILED execution, show the REAL underlying error:**

```
✅ GOOD: "❌ Machine type 'n2-standard-4' is not supported for GPU 'NVIDIA_TESLA_T4'"
❌ BAD:  "🚨 FATAL ERROR DETECTED: Machine type not supported!"

✅ GOOD: "❌ QuotaExceeded: Quota 'NVIDIA_L4_GPUS' exceeded. Limit: 0 in us-west2"
❌ BAD:  "🚨 FATAL ERROR DETECTED: Quota exceeded!"

✅ GOOD: "❌ PermissionDenied: 403 Permission 'compute.instances.create' denied"
❌ BAD:  "🚨 FATAL ERROR DETECTED: Permission denied (403)!"
```

**Why?** The REAL error has actionable information (which quota, which permission, which machine type)!

---

## 🔧 **IMPLEMENTATION NOTES**

**When parsing logs:**

1. **See "🚨 FATAL ERROR DETECTED"?** → Look 20-80 lines around it for the REAL error
2. **Extract the underlying GCP/W&B/Python error** → Not our wrapper message!
3. **Strip "wandb: ERROR" prefix** → Shows cleaner message in table
4. **Filter out info lines** (⏱️, ⏳, "monitoring for") → Avoid false positives

**Example extraction:**
```python
# In context around "🚨 FATAL ERROR DETECTED"
for ctx_line in bailout_lines:
    if 'Machine type' in ctx_line and 'is not supported' in ctx_line:
        # Found the REAL GCP error!
        if 'wandb: ERROR' in ctx_line:
            # Strip W&B prefix, return clean GCP error
            return ctx_line.split('wandb: ERROR')[-1].strip()
        else:
            return ctx_line.strip()
```

---

**END OF INVESTIGATION** - All patterns documented! 🎯
