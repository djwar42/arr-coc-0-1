# Cloud Build Debugging - Log Streaming, Failure Analysis, and Performance Profiling

## Overview

Debugging Cloud Build issues requires a multi-layered approach: real-time log streaming for immediate feedback, systematic failure analysis for root cause identification, and performance profiling for optimization. This guide covers production-grade debugging strategies used in ML training pipelines and large-scale deployments.

**Key Debugging Capabilities:**
- Real-time log streaming via `gcloud builds log`
- Build failure analysis with Cloud Logging integration
- Performance profiling and bottleneck identification
- Common error patterns and solutions
- Automated debugging workflows

## Section 1: Log Streaming and Real-Time Debugging

### gcloud builds log Command

**Basic Log Streaming:**
```bash
# Stream logs for ongoing build
gcloud builds log BUILD_ID --stream

# Stream with polling interval
gcloud builds log BUILD_ID --stream --poll-interval=1s

# View completed build logs
gcloud builds log BUILD_ID

# View logs for specific region
gcloud builds log BUILD_ID --region=us-west2
```

**Advanced Log Filtering:**
```bash
# Stream logs with verbosity levels
gcloud builds log BUILD_ID --stream --verbosity=debug

# Filter by build step
gcloud builds log BUILD_ID | grep "STEP 3"

# Show only errors
gcloud builds log BUILD_ID | grep -i "error\|failed"

# Follow logs in real-time with context
gcloud builds log BUILD_ID --stream | tee build.log
```

**Common Streaming Patterns:**

From [gcloud builds log documentation](https://cloud.google.com/sdk/gcloud/reference/builds/log) (accessed 2025-02-03):

```bash
# Pattern 1: Immediate feedback during submission
gcloud builds submit --config=cloudbuild.yaml . 2>&1 | tee -a build.log

# Pattern 2: Parallel submission + log streaming
BUILD_ID=$(gcloud builds submit --config=cloudbuild.yaml . --format="value(id)")
gcloud builds log $BUILD_ID --stream

# Pattern 3: Background build with log monitoring
gcloud builds submit --config=cloudbuild.yaml . --async
# Get BUILD_ID from output, then:
gcloud builds log $BUILD_ID --stream --poll-interval=1s
```

### Real-Time Log Streaming in Cloud Console

**Logs Explorer Streaming:**

From [Cloud Logging streaming documentation](https://docs.cloud.google.com/logging/docs/view/streaming-live-tailing) (accessed 2025-02-03):

1. **Enable Streaming Mode:**
   - Navigate to Cloud Logging → Logs Explorer
   - Click "Stream logs" button (top right)
   - Logs appear in real-time as Cloud Logging ingests them

2. **Filter for Cloud Build:**
   ```
   resource.type="build"
   resource.labels.build_id="YOUR_BUILD_ID"
   ```

3. **Advanced Filters:**
   ```
   resource.type="build"
   severity>="ERROR"
   timestamp>="2025-02-03T00:00:00Z"
   ```

**Streaming Limitations:**
- Streaming stops when you interact with scroll bar
- Click "Restart streaming" to resume
- Use "Live tail" for continuous monitoring without interaction

### Programmatic Log Streaming

**Python Example (Node.js Library Pattern):**

From [Stack Overflow discussion](https://stackoverflow.com/questions/74278775/is-it-possible-to-stream-cloud-build-logs-with-the-node-js-library) (accessed 2025-02-03):

```python
from google.cloud import cloudbuild_v1
import time

def stream_build_logs(project_id, build_id):
    """Stream Cloud Build logs programmatically."""
    client = cloudbuild_v1.CloudBuildClient()

    build_name = f"projects/{project_id}/locations/global/builds/{build_id}"

    # Poll build status and fetch logs
    while True:
        build = client.get_build(name=build_name)

        # Fetch log URL
        if build.log_url:
            print(f"Log URL: {build.log_url}")

        # Check build status
        if build.status in [
            cloudbuild_v1.Build.Status.SUCCESS,
            cloudbuild_v1.Build.Status.FAILURE,
            cloudbuild_v1.Build.Status.TIMEOUT,
            cloudbuild_v1.Build.Status.CANCELLED
        ]:
            print(f"Build finished with status: {build.status.name}")
            break

        time.sleep(2)  # Poll every 2 seconds

# Usage
stream_build_logs("my-project", "abc-123-def")
```

**subprocess.Popen Pattern for CLI Streaming:**

Critical pattern from arr-coc-0-1 project (2025-11-13):

```python
import subprocess

def stream_cloud_build(cmd):
    """Stream CloudBuild output line-by-line (no silent waiting)."""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1,  # Line buffered - immediate output
    )

    # Stream output as it happens
    for line in process.stdout:
        line = line.rstrip()
        if line:
            print(f"[CloudBuild] {line}")

    returncode = process.wait(timeout=25200)  # 7 hours max

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)

# Example usage
stream_cloud_build([
    "gcloud", "builds", "submit",
    "--config=cloudbuild.yaml",
    "--timeout=30m",
    "."
])
```

**Why This Matters:**
- ❌ `subprocess.run(capture_output=True)` is SILENT until completion
- ✅ `subprocess.Popen()` + line-by-line streaming shows progress
- For 2-4 hour PyTorch builds: **NO MORE SILENT WAITING**

## Section 2: Build Failure Analysis

### Common Cloud Build Errors

From [Cloud Build troubleshooting documentation](https://docs.cloud.google.com/build/docs/troubleshooting) (accessed 2025-02-03):

#### Error 1: Missing Service Agent Permissions

**Symptom:**
```
ERROR: (gcloud.builds.submit) PERMISSION_DENIED:
The caller does not have permission
```

**Root Cause:** Cloud Build service agent lacks required IAM permissions.

**Solution:**
```bash
# Grant Cloud Build Service Account role
PROJECT_ID=$(gcloud config get-value project)
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
  --role="roles/cloudbuild.builds.builder"
```

#### Error 2: Build Timeout

**Symptom:**
```
ERROR: build step 3 "gcr.io/cloud-builders/docker" timed out
status: TIMEOUT
```

**Root Cause:** Default timeout (10 minutes) too short for large builds.

**Solution:**
```yaml
# cloudbuild.yaml
timeout: 3600s  # 1 hour

steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/image', '.']
    timeout: 1800s  # Step-level timeout (30 min)
```

**Command-line override:**
```bash
gcloud builds submit --timeout=3600s --config=cloudbuild.yaml .
```

#### Error 3: Docker Layer Caching Issues

**Symptom:**
```
ERROR: failed to initialize analyzer: getting previous image:
Get history for image: no such object
```

**Root Cause:** Kaniko or Docker builder cannot access previous image layers.

From [Stack Overflow discussion](https://stackoverflow.com/questions/77181996/gcloud-build-failure-error-failed-to-initialize-analyzer-no-such-object) (accessed 2025-02-03):

**Solution:**
```yaml
steps:
  - name: 'gcr.io/kaniko-project/executor:latest'
    args:
      - '--destination=gcr.io/$PROJECT_ID/image:latest'
      - '--cache=true'
      - '--cache-ttl=24h'
      # Explicitly specify cache repo
      - '--cache-repo=gcr.io/$PROJECT_ID/cache'
```

#### Error 4: Worker Pool Internet Access (NO_PUBLIC_EGRESS)

**Symptom:**
```
Err: Could not connect to archive.ubuntu.com:80, connection timed out
Err: Could not connect to security.ubuntu.com:80, connection timed out
E: Unable to locate package python3.10
```

**Root Cause:** Worker pool configured with `NO_PUBLIC_EGRESS` blocks internet access.

From arr-coc-0-1 project CLAUDE.md (2025-11-13):

**Solution:**
```bash
# Update worker pool to allow public egress
gcloud builds worker-pools update pytorch-mecha-pool \
  --region=us-west2 \
  --config-from-file=worker_pool_public_egress.yaml
```

**worker_pool_public_egress.yaml:**
```yaml
privatePoolV1Config:
  networkConfig:
    egressOption: PUBLIC_EGRESS  # Enable internet access
  workerConfig:
    machineType: c3-standard-176
    diskSizeGb: 100
```

**Verification:**
```bash
gcloud builds worker-pools describe pytorch-mecha-pool \
  --region=us-west2 \
  --format="value(privatePoolV1Config.networkConfig.egressOption)"
# Should output: PUBLIC_EGRESS
```

#### Error 5: Logs Not Streaming to CLI

**Symptom:** `gcloud builds submit` succeeds but logs don't appear.

From [GitHub issue discussion](https://github.com/GoogleCloudPlatform/cloud-builders/issues/562) (accessed 2025-02-03):

**Root Cause:** When `logsBucket` is specified in `cloudbuild.yaml`, logs aren't streamed to CLI.

**Solution:**
```yaml
# Option 1: Remove logsBucket for CLI streaming
# cloudbuild.yaml
# logsBucket: 'gs://my-logs-bucket'  # Comment out

# Option 2: Use --suppress-logs flag
# Command line
gcloud builds submit --suppress-logs --config=cloudbuild.yaml .
```

### Systematic Failure Analysis Workflow

**Step-by-Step Debugging Process:**

```bash
# 1. Get recent builds
gcloud builds list --limit=10 --region=us-west2

# 2. Identify failed build
BUILD_ID="abc-123-def"

# 3. Get build details
gcloud builds describe $BUILD_ID --region=us-west2 --format=yaml

# 4. Check build status
gcloud builds describe $BUILD_ID \
  --format="value(status,statusDetail,timeout,timing)"

# 5. Retrieve full logs
gcloud builds log $BUILD_ID > build_failure.log

# 6. Extract errors
grep -i "error\|fail\|exception" build_failure.log | head -50

# 7. Check build timing
gcloud builds describe $BUILD_ID \
  --format="value(steps[].name,steps[].status,steps[].timing)"

# 8. Analyze which step failed
gcloud builds log $BUILD_ID | grep -A 10 "FAILURE"
```

### Error Reporting Best Practices

From arr-coc-0-1 training/cli/launch/core.py (2025-11-13):

**Full Error Display (No Truncation):**

```python
def report_build_error(result):
    """Show first 50 lines of error (not truncated!)."""
    error_msg = result.stderr.strip()

    if error_msg:
        print("\n❌ Cloud Build Error:")
        print("=" * 70)

        # Show first 50 lines of error
        for i, line in enumerate(error_msg.split('\n')[:50]):
            if line.strip():
                print(f"  {line}")

            if i == 49 and len(error_msg.split('\n')) > 50:
                print(f"\n  ... ({len(error_msg.split('\n')) - 50} more lines)")

        print("=" * 70)
```

**Common Error Patterns to Surface:**

```python
ERROR_PATTERNS = {
    "PERMISSION_DENIED": {
        "cause": "Missing IAM permissions",
        "fix": "Grant roles/cloudbuild.builds.builder to service account"
    },
    "QUOTA_EXCEEDED": {
        "cause": "API rate limit or quota exceeded",
        "fix": "Request quota increase or throttle builds"
    },
    "RESOURCE_EXHAUSTED": {
        "cause": "Build machine resources exhausted",
        "fix": "Increase worker pool machine type or disk size"
    },
    "TIMEOUT": {
        "cause": "Build exceeded timeout limit",
        "fix": "Increase timeout in cloudbuild.yaml or gcloud command"
    }
}
```

## Section 3: Performance Profiling

### Build Timing Analysis

**Extract Build Metrics:**

```bash
# Get detailed timing for each step
gcloud builds describe BUILD_ID --format=json | jq '.steps[] | {
  name: .name,
  status: .status,
  start: .timing.startTime,
  end: .timing.endTime,
  duration: .timing
}'

# Calculate step durations
gcloud builds describe BUILD_ID --format=json | \
  jq -r '.steps[] | "\(.name): \(.timing.startTime) → \(.timing.endTime)"'
```

**Performance Benchmarking:**

```python
import json
from datetime import datetime

def analyze_build_performance(build_id):
    """Extract performance metrics from build."""
    result = subprocess.run(
        ["gcloud", "builds", "describe", build_id, "--format=json"],
        capture_output=True, text=True
    )

    build_data = json.loads(result.stdout)

    steps = []
    for step in build_data.get('steps', []):
        timing = step.get('timing', {})
        start = datetime.fromisoformat(timing.get('startTime', '').replace('Z', '+00:00'))
        end = datetime.fromisoformat(timing.get('endTime', '').replace('Z', '+00:00'))
        duration = (end - start).total_seconds()

        steps.append({
            'name': step.get('name'),
            'status': step.get('status'),
            'duration_seconds': duration
        })

    # Sort by duration (slowest first)
    steps.sort(key=lambda x: x['duration_seconds'], reverse=True)

    print("Build Performance Analysis:")
    print("=" * 60)
    for step in steps:
        print(f"{step['name']:<40} {step['duration_seconds']:>8.1f}s  ({step['status']})")

    total_time = sum(s['duration_seconds'] for s in steps)
    print("=" * 60)
    print(f"{'Total Time':<40} {total_time:>8.1f}s")

    return steps

# Usage
analyze_build_performance("abc-123-def")
```

### Identifying Performance Bottlenecks

**Common Bottlenecks:**

1. **Docker Image Pull:**
   ```yaml
   # Problem: Pulling large base image every time
   steps:
     - name: 'gcr.io/cloud-builders/docker'
       args: ['pull', 'nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04']

   # Solution: Use Cloud Build caching
   options:
     machineType: 'E2_HIGHCPU_8'
     diskSizeGb: 100
     # Cache Docker layers
     volumes:
       - name: 'docker-cache'
         path: '/var/lib/docker'
   ```

2. **Slow Package Installation:**
   ```dockerfile
   # Problem: Installing packages from scratch
   RUN apt-get update && apt-get install -y python3.10

   # Solution: Use pre-built image with packages
   FROM gcr.io/my-project/python310-base:latest
   ```

3. **Sequential vs Parallel Steps:**
   ```yaml
   # Problem: Sequential steps (slow)
   steps:
     - name: 'gcr.io/cloud-builders/docker'
       id: 'build-image-1'
       args: ['build', '-t', 'image1', './dir1']

     - name: 'gcr.io/cloud-builders/docker'
       id: 'build-image-2'
       args: ['build', '-t', 'image2', './dir2']

   # Solution: Parallel execution (fast)
   steps:
     - name: 'gcr.io/cloud-builders/docker'
       id: 'build-image-1'
       args: ['build', '-t', 'image1', './dir1']
       # No waitFor = runs immediately

     - name: 'gcr.io/cloud-builders/docker'
       id: 'build-image-2'
       args: ['build', '-t', 'image2', './dir2']
       # No waitFor = runs in parallel with build-image-1
   ```

### Cloud Profiler Integration

From [Cloud Profiler documentation](https://docs.cloud.google.com/profiler/docs/concepts-profiling) (accessed 2025-02-03):

**Profiling Build Scripts:**

```python
# Enable profiling for Python build scripts
import googlecloudprofiler

def run_expensive_build_step():
    """Profile expensive operations."""
    try:
        googlecloudprofiler.start(
            service='cloud-build-step',
            service_version='1.0.0',
            verbose=3
        )
    except (ValueError, NotImplementedError) as exc:
        print(f"Profiler not started: {exc}")

    # Your expensive build logic here
    compile_pytorch_from_source()
    install_cuda_dependencies()

# Profiler automatically uploads data
# View in Cloud Console → Profiler
```

**Key Profiling Metrics:**
- CPU time per build step
- Memory allocation patterns
- Wall-clock time distribution
- I/O wait times

## Section 4: Common Error Solutions

### Build Error Quick Reference

| Error Message | Root Cause | Solution |
|---------------|------------|----------|
| `PERMISSION_DENIED` | Missing IAM role | Grant `roles/cloudbuild.builds.builder` |
| `TIMEOUT` | Build exceeded time limit | Increase `timeout` in config |
| `RESOURCE_EXHAUSTED` | Out of memory/disk | Use larger machine type |
| `INVALID_ARGUMENT` | Malformed cloudbuild.yaml | Validate YAML syntax |
| `NOT_FOUND` | Image/repo doesn't exist | Check image path, enable APIs |
| `NO_PUBLIC_EGRESS` | Worker pool blocks internet | Set `egressOption: PUBLIC_EGRESS` |

### Debug Mode Enablement

**Verbose Build Logs:**

```bash
# Enable debug verbosity
gcloud builds submit \
  --config=cloudbuild.yaml \
  --verbosity=debug \
  . 2>&1 | tee debug.log

# Show all gcloud API calls
gcloud builds submit \
  --config=cloudbuild.yaml \
  --log-http \
  .
```

**Build Step Debugging:**

```yaml
# cloudbuild.yaml with debug commands
steps:
  # Debug: Show environment
  - name: 'ubuntu'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        echo "=== Environment Debug ==="
        env | sort
        echo "=== Disk Space ==="
        df -h
        echo "=== Memory ==="
        free -h

  # Actual build step
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'myimage', '.']

  # Debug: Verify build output
  - name: 'gcr.io/cloud-builders/docker'
    args: ['images']
```

### Debugging Worker Pools

```bash
# List worker pools
gcloud builds worker-pools list --region=us-west2

# Describe pool configuration
gcloud builds worker-pools describe POOL_NAME \
  --region=us-west2 \
  --format=yaml

# Check egress setting
gcloud builds worker-pools describe POOL_NAME \
  --region=us-west2 \
  --format="value(privatePoolV1Config.networkConfig.egressOption)"

# View worker pool metrics
gcloud builds worker-pools describe POOL_NAME \
  --region=us-west2 \
  --format="value(privatePoolV1Config.workerConfig)"
```

## Section 5: Automated Debugging Workflows

### Health Check Script

```bash
#!/bin/bash
# cloud-build-health-check.sh

PROJECT_ID=$(gcloud config get-value project)
REGION="us-west2"

echo "Cloud Build Health Check"
echo "========================"
echo ""

# Check recent builds
echo "Recent Builds (last 5):"
gcloud builds list --limit=5 --region=$REGION \
  --format="table(id,status,createTime,duration)"
echo ""

# Check for failed builds in last 24h
FAILED_COUNT=$(gcloud builds list \
  --filter="status=FAILURE AND createTime>$(date -u -d '24 hours ago' +%Y-%m-%dT%H:%M:%S)" \
  --region=$REGION \
  --format="value(id)" | wc -l)

echo "Failed Builds (last 24h): $FAILED_COUNT"
echo ""

# Check service account permissions
SA_EMAIL="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
echo "Cloud Build Service Account: $SA_EMAIL"
gcloud projects get-iam-policy $PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:$SA_EMAIL" \
  --format="table(bindings.role)"
echo ""

# Check worker pools
echo "Worker Pools:"
gcloud builds worker-pools list --region=$REGION \
  --format="table(name,state,privatePoolV1Config.networkConfig.egressOption)"
echo ""

# Check quota usage
echo "Build Quota Usage:"
gcloud builds list --limit=100 --region=$REGION \
  --filter="createTime>=$(date -u -d '24 hours ago' +%Y-%m-%dT%H:%M:%S)" \
  --format="value(id)" | wc -l
echo " builds in last 24 hours"
```

### Failure Alerting

```python
# cloud_build_monitor.py
from google.cloud import logging_v2
from google.cloud import monitoring_v3
from datetime import datetime, timedelta

def check_build_failures(project_id, hours=1):
    """Alert on build failures in last N hours."""
    client = logging_v2.Client(project=project_id)

    # Query for failed builds
    filter_str = f'''
    resource.type="build"
    protoPayload.status.message:"FAILURE"
    timestamp>="{(datetime.utcnow() - timedelta(hours=hours)).isoformat()}Z"
    '''

    entries = client.list_entries(filter_=filter_str, page_size=100)

    failures = []
    for entry in entries:
        failures.append({
            'build_id': entry.resource.labels.get('build_id'),
            'timestamp': entry.timestamp,
            'message': entry.payload
        })

    if failures:
        print(f"⚠️  {len(failures)} build failures detected!")
        for f in failures[:5]:  # Show first 5
            print(f"  - Build {f['build_id']} at {f['timestamp']}")

    return failures

# Usage
failures = check_build_failures("my-project", hours=1)
```

## Sources

**Web Research:**

- [gcloud builds log reference](https://cloud.google.com/sdk/gcloud/reference/builds/log) - Google Cloud SDK documentation (accessed 2025-02-03)
- [Cloud Logging streaming and live tailing](https://docs.cloud.google.com/logging/docs/view/streaming-live-tailing) - Google Cloud documentation (accessed 2025-02-03)
- [Cloud Build troubleshooting](https://docs.cloud.google.com/build/docs/troubleshooting) - Google Cloud official troubleshooting guide (accessed 2025-02-03)
- [Stack Overflow: Streaming Cloud Build logs](https://stackoverflow.com/questions/74278775/is-it-possible-to-stream-cloud-build-logs-with-the-node-js-library) - Community discussion on programmatic log streaming (accessed 2025-02-03)
- [GitHub Issue: logs not streaming with logsBucket](https://github.com/GoogleCloudPlatform/cloud-builders/issues/562) - Known limitation discussion (accessed 2025-02-03)
- [Cloud Profiler concepts](https://docs.cloud.google.com/profiler/docs/concepts-profiling) - Performance profiling documentation (accessed 2025-02-03)
- [Stack Overflow: GCloud Build Failure - analyzer error](https://stackoverflow.com/questions/77181996/gcloud-build-failure-error-failed-to-initialize-analyzer-no-such-object) - Docker caching issue discussion (accessed 2025-02-03)

**Internal Sources:**

- arr-coc-0-1 project CLAUDE.md - Worker pool egress configuration, subprocess streaming pattern (2025-11-13)
- arr-coc-0-1 training/cli/launch/core.py - CloudBuild streaming implementation (2025-11-13)

**Additional References:**

- [Google Cloud Observability release notes](https://docs.cloud.google.com/stackdriver/docs/release-notes) - Real-time log streaming features
- [Cloud Build security and scanning](https://docs.cloud.google.com/build/docs/securing-builds/store-manage-build-logs) - Log storage best practices
