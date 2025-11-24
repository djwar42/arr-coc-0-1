# GCloud Authentication - Advanced Patterns

**Date**: 2025-02-03
**Category**: Authentication Security
**Scope**: Advanced scenarios, troubleshooting, security hardening, credential management

---

## Overview

Advanced GCP authentication patterns go beyond basic service account usage to cover sophisticated scenarios involving credential chaining, cross-project impersonation, debugging authentication failures, and security hardening strategies. This guide synthesizes best practices from Google Cloud documentation and production implementations.

**Why this matters:**
- Authentication failures are the #1 cause of deployment delays
- Improper credential management creates security vulnerabilities
- Understanding credential precedence prevents silent failures
- Advanced patterns enable zero-trust architectures

**Key topics covered:**
- Advanced authentication scenarios (multi-hop impersonation, credential chaining)
- Troubleshooting authentication failures (debugging ADC, token issues)
- Security hardening (key rotation, least privilege, monitoring)
- Token management (lifecycle, caching, refresh patterns)
- Audit logging (who accessed what, when, how)

---

## Section 1: Advanced Authentication Scenarios (~150 lines)

### Scenario 1: Multi-Hop Service Account Impersonation

From [Google Cloud - Service Account Impersonation](https://docs.cloud.google.com/iam/docs/service-account-impersonation) (accessed 2025-02-03):

**Problem**: User needs to impersonate Service Account B, but B can only be impersonated by Service Account A.

**Solution**: Chain impersonation through intermediate accounts.

```
User
  ↓ (impersonates)
Service Account A
  ↓ (impersonates)
Service Account B
  ↓ (access)
Target Resource
```

**Setup:**

```bash
# Grant user permission to impersonate SA-A
gcloud iam service-accounts add-iam-policy-binding \
  sa-a@project.iam.gserviceaccount.com \
  --member="user:engineer@company.com" \
  --role="roles/iam.serviceAccountTokenCreator"

# Grant SA-A permission to impersonate SA-B
gcloud iam service-accounts add-iam-policy-binding \
  sa-b@project.iam.gserviceaccount.com \
  --member="serviceAccount:sa-a@project.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountTokenCreator"

# Grant SA-B permission to access target resources
gcloud projects add-iam-policy-binding target-project \
  --member="serviceAccount:sa-b@project.iam.gserviceaccount.com" \
  --role="roles/storage.admin"
```

**Usage:**

```bash
# Impersonate SA-B through SA-A
gcloud storage ls gs://target-bucket \
  --impersonate-service-account=sa-b@project.iam.gserviceaccount.com \
  --impersonate-service-account=sa-a@project.iam.gserviceaccount.com
```

**Why multi-hop?**
- Enforce approval chains (user → approver SA → production SA)
- Separate dev/staging/prod access boundaries
- Implement break-glass access patterns (emergency SA → production SA)

### Scenario 2: Cross-Project Service Account Usage

From [Google Cloud - Cross-Project Service Account Usage](https://docs.cloud.google.com/iam/docs/attach-service-accounts) (accessed 2025-02-03):

**Problem**: Service in Project A needs to access resources in Project B using Project B's service account.

**Pattern 1: Attach SA from different project**

```bash
# Create VM in Project A using SA from Project B
gcloud compute instances create my-vm \
  --project=project-a \
  --zone=us-central1-a \
  --service-account=sa@project-b.iam.gserviceaccount.com \
  --scopes=cloud-platform
```

**Requirements:**
- User creating VM must have `iam.serviceAccountUser` on the SA
- SA must have `iam.serviceAccounts.actAs` permission in Project A

**Pattern 2: Impersonation across projects**

```bash
# In Project A, impersonate SA from Project B
gcloud compute instances list \
  --project=project-b \
  --impersonate-service-account=sa@project-b.iam.gserviceaccount.com
```

**Use case**: Centralized service account management (all SAs in one project, used across organization).

### Scenario 3: Application Default Credentials (ADC) Precedence Chain

From [Google Cloud - Application Default Credentials](https://docs.cloud.google.com/docs/authentication/application-default-credentials) (accessed 2025-02-03):

ADC searches for credentials in this order:

1. **GOOGLE_APPLICATION_CREDENTIALS** environment variable (path to JSON key file)
2. **gcloud auth application-default login** credentials (~/.config/gcloud/application_default_credentials.json)
3. **Attached service account** (if running on GCE, Cloud Run, Cloud Functions, GKE)
4. **Workload Identity Federation** credentials (if configured)

**Common pitfall**: Stale credentials in step 1 or 2 override valid attached SA in step 3!

```bash
# Check which credential ADC is using
gcloud auth application-default print-access-token

# If wrong credential, revoke stale ADC
gcloud auth application-default revoke

# Clear environment variable if set
unset GOOGLE_APPLICATION_CREDENTIALS
```

**Best practice for local development:**

```bash
# Use user impersonation instead of ADC keys
gcloud auth login  # Authenticate as yourself
gcloud config set auth/impersonate_service_account SA_EMAIL

# Now all gcloud commands impersonate SA (no keys needed!)
gcloud storage ls gs://bucket  # Uses your identity → impersonates SA
```

### Scenario 4: Workload Identity for GKE Pods

From [Google Cloud - Workload Identity for GKE](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/workload-identity) (accessed 2025-02-03):

**Problem**: Kubernetes pods need to access GCP APIs without service account keys.

**Solution**: Workload Identity maps Kubernetes service accounts to GCP service accounts.

```
Kubernetes Pod (ServiceAccount: app-sa in namespace default)
    ↓
GKE Workload Identity
    ↓
GCP Service Account (app-sa@project.iam.gserviceaccount.com)
    ↓
GCP Resources (GCS, BigQuery, etc.)
```

**Setup:**

```bash
# Enable Workload Identity on GKE cluster
gcloud container clusters create my-cluster \
  --workload-pool=PROJECT_ID.svc.id.goog

# Create GCP service account
gcloud iam service-accounts create app-sa

# Create Kubernetes service account
kubectl create serviceaccount app-sa -n default

# Bind KSA to GSA
gcloud iam service-accounts add-iam-policy-binding \
  app-sa@PROJECT_ID.iam.gserviceaccount.com \
  --role=roles/iam.workloadIdentityUser \
  --member="serviceAccount:PROJECT_ID.svc.id.goog[default/app-sa]"

# Annotate KSA with GSA
kubectl annotate serviceaccount app-sa \
  -n default \
  iam.gke.io/gcp-service-account=app-sa@PROJECT_ID.iam.gserviceaccount.com
```

**Pod manifest:**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-app
spec:
  serviceAccountName: app-sa  # Use annotated KSA
  containers:
  - name: app
    image: gcr.io/project/app
    # Application uses ADC - automatically gets GSA credentials!
```

**Key insight**: No keys, no secrets, no credential management! Pods automatically get short-lived tokens.

### Scenario 5: Temporary Token Generation for External Systems

From [Google Cloud - Creating Short-Lived Service Account Credentials](https://docs.cloud.google.com/iam/docs/create-short-lived-credentials-direct) (accessed 2025-02-03):

**Problem**: Need to give external system temporary access without creating service account keys.

**Solution**: Generate short-lived OAuth2 access tokens programmatically.

```python
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# User authenticates, then generates token for SA
credentials = service_account.Credentials.from_service_account_file(
    'key.json',  # Or use ADC
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

# Generate short-lived access token
request = Request()
credentials.refresh(request)
access_token = credentials.token  # Valid for 1 hour

# Give this token to external system
# They use it in Authorization: Bearer {access_token} header
```

**Alternative: Token creation API (no keys needed!)**

```bash
# Generate access token for SA (requires TokenCreator role)
gcloud auth print-access-token \
  --impersonate-service-account=sa@project.iam.gserviceaccount.com \
  --lifetime=3600s  # 1 hour (max: 1 hour for access tokens)

# Generate ID token (for services expecting OIDC tokens)
gcloud auth print-identity-token \
  --impersonate-service-account=sa@project.iam.gserviceaccount.com \
  --audiences=https://my-service.run.app
```

**Use cases:**
- Give contractor temporary API access (1-hour token, no key management)
- Testing authentication flows (generate tokens on demand)
- Break-glass emergency access (generate token manually, no key rotation)

---

## Section 2: Troubleshooting Authentication Failures (~150 lines)

### Debugging ADC (Application Default Credentials)

From [Google Cloud - Troubleshooting Application Default Credentials](https://docs.cloud.google.com/docs/authentication/troubleshoot-adc) (accessed 2025-02-03):

**Symptom**: Application fails with "Could not automatically determine credentials"

**Root causes:**

1. **No credentials in ADC search path**

```bash
# Check if ADC credentials exist
ls ~/.config/gcloud/application_default_credentials.json

# If missing, set up ADC
gcloud auth application-default login
```

2. **GOOGLE_APPLICATION_CREDENTIALS points to invalid file**

```bash
# Check environment variable
echo $GOOGLE_APPLICATION_CREDENTIALS

# If set but invalid, unset it
unset GOOGLE_APPLICATION_CREDENTIALS

# Or fix the path
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/valid/key.json
```

3. **Running outside GCP with no ADC setup**

```bash
# For local development, use user ADC
gcloud auth application-default login

# For CI/CD, use Workload Identity Federation (not keys!)
# Or set GOOGLE_APPLICATION_CREDENTIALS to service account key
```

**Debugging command:**

```bash
# Test which credential ADC will use
python3 -c "
import google.auth
credentials, project = google.auth.default()
print(f'Using: {credentials.service_account_email}')
print(f'Project: {project}')
"
```

**Common fix pattern:**

```bash
# 1. Clear stale credentials
gcloud auth application-default revoke
unset GOOGLE_APPLICATION_CREDENTIALS

# 2. Re-authenticate
gcloud auth application-default login

# 3. Verify
gcloud auth application-default print-access-token
# Should succeed and print token
```

### Debugging "Permission Denied" Errors

From [Google Cloud - IAM Troubleshooting](https://docs.cloud.google.com/iam/docs/troubleshooting-access) (accessed 2025-02-03):

**Error**: `ERROR: (gcloud.XXX) User does not have permission to access YYY`

**Debugging steps:**

**Step 1: Identify which credential is being used**

```bash
# Check active account
gcloud auth list

# Check effective account (with impersonation)
gcloud config get-value account

# Check impersonation setting
gcloud config get-value auth/impersonate_service_account
```

**Step 2: Check IAM permissions**

```bash
# Get IAM policy for resource
gcloud projects get-iam-policy PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:PRINCIPAL"

# Check if specific role is granted
gcloud projects get-iam-policy PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.role:ROLE AND bindings.members:PRINCIPAL"
```

**Step 3: Use Policy Troubleshooter**

From [Google Cloud - Policy Troubleshooter](https://docs.cloud.google.com/iam/docs/troubleshooting-access#policy-troubleshooter) (accessed 2025-02-03):

```bash
# Install Policy Troubleshooter (if using CLI)
gcloud policy-troubleshoot iam \
  --principal-email=PRINCIPAL \
  --resource-name=RESOURCE \
  --permission=PERMISSION

# Or use web UI:
# https://console.cloud.google.com/iam-admin/troubleshooter
```

**Step 4: Check for IAM conditions blocking access**

```bash
# View IAM policy with conditions
gcloud projects get-iam-policy PROJECT_ID --format=yaml

# Look for:
# - Time-based conditions (expired?)
# - IP range restrictions (wrong source IP?)
# - Resource name filters (accessing wrong resource?)
```

**Example debugging session:**

```bash
# Error: Permission denied on GCS bucket
$ gsutil ls gs://my-bucket
ERROR: (gcloud.storage) User does not have storage.buckets.list

# Step 1: Check which account is active
$ gcloud auth list
* account@example.com (active)

# Step 2: Check bucket IAM
$ gsutil iam get gs://my-bucket | grep account@example.com
# (no output - account not in policy!)

# Step 3: Grant permission
$ gsutil iam ch user:account@example.com:objectViewer gs://my-bucket

# Retry
$ gsutil ls gs://my-bucket
gs://my-bucket/file1.txt
gs://my-bucket/file2.txt
```

### Debugging Token Expiry Issues

From [Google Cloud - Service Account Token Lifetime](https://docs.cloud.google.com/iam/docs/create-short-lived-credentials-direct) (accessed 2025-02-03):

**Symptom**: Application works initially, then fails after ~1 hour with "Invalid authentication credentials"

**Root cause**: Access tokens expire after 1 hour and aren't being refreshed.

**Fix pattern 1: Use Cloud SDK (handles refresh automatically)**

```python
from google.cloud import storage

# Cloud SDK handles token refresh
client = storage.Client()  # Auto-refreshes tokens!
buckets = list(client.list_buckets())
```

**Fix pattern 2: Manual refresh with OAuth2 library**

```python
from google.oauth2 import service_account
from google.auth.transport.requests import Request

credentials = service_account.Credentials.from_service_account_file('key.json')

# Explicitly refresh before each request
def make_authenticated_request():
    if not credentials.valid:
        credentials.refresh(Request())  # Get new token

    # Use credentials.token in request headers
    headers = {'Authorization': f'Bearer {credentials.token}'}
    # ... make request ...
```

**Fix pattern 3: Long-running services with credential caching**

```python
import google.auth
from google.auth.transport.requests import AuthorizedSession

# Create session with auto-refresh
credentials, project = google.auth.default()
authed_session = AuthorizedSession(credentials)

# Session automatically refreshes tokens
response = authed_session.get('https://storage.googleapis.com/storage/v1/b')
# Tokens refreshed automatically on expiry!
```

### Debugging Workload Identity Federation

From [Google Cloud - Troubleshooting Workload Identity Federation](https://docs.cloud.google.com/iam/docs/troubleshooting-workload-identity-federation) (accessed 2025-02-03):

**Error**: `Error 400: Invalid JWT Signature` or `Error 403: Permission denied`

**Common causes:**

**1. Attribute mapping mismatch**

```bash
# Check provider configuration
gcloud iam workload-identity-pools providers describe PROVIDER_ID \
  --workload-identity-pool=POOL_ID \
  --location=global \
  --format=yaml

# Look for attribute-mapping and attribute-condition
# Ensure external token claims match expected mappings
```

**2. Principal not granted workloadIdentityUser role**

```bash
# Check SA IAM policy
gcloud iam service-accounts get-iam-policy SA_EMAIL

# Should contain:
# - role: roles/iam.workloadIdentityUser
#   members:
#   - principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/POOL_ID/*
```

**3. Audience mismatch in external token**

External JWT must have `aud` claim matching WIF provider:

```
aud: "//iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/POOL_ID/providers/PROVIDER_ID"
```

**Debugging with token exchange test:**

```bash
# Manually test token exchange
curl -X POST https://sts.googleapis.com/v1/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=urn:ietf:params:oauth:grant-type:token-exchange" \
  -d "audience=//iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/POOL_ID/providers/PROVIDER_ID" \
  -d "subject_token_type=urn:ietf:params:oauth:token-type:jwt" \
  -d "subject_token=EXTERNAL_JWT_TOKEN" \
  -d "requested_token_type=urn:ietf:params:oauth:token-type:access_token" \
  -d "scope=https://www.googleapis.com/auth/cloud-platform"

# Success: Returns GCP access token
# Failure: Returns error with specific reason
```

---

## Section 3: Security Hardening (~100 lines)

### Principle 1: Eliminate Service Account Keys

From [Google Cloud - Best Practices for Managing Service Account Keys](https://docs.cloud.google.com/iam/docs/best-practices-for-managing-service-account-keys) (accessed 2025-02-03):

**Goal**: Zero service account keys in use across entire organization.

**Implementation strategy:**

**1. Inventory existing keys**

```bash
# List all service accounts in project
gcloud iam service-accounts list --format="value(email)" > sa_list.txt

# Check keys for each SA
while read SA_EMAIL; do
  echo "Checking keys for: $SA_EMAIL"
  gcloud iam service-accounts keys list \
    --iam-account="$SA_EMAIL" \
    --filter="keyType=USER_MANAGED" \
    --format="table(name,validAfterTime)"
done < sa_list.txt
```

**2. Migrate workloads to keyless authentication**

- **GitHub Actions** → Workload Identity Federation
- **GitLab CI/CD** → Workload Identity Federation
- **GKE Pods** → Workload Identity
- **Cloud Run** → Attached service account
- **Cloud Functions** → Attached service account
- **Compute Engine VMs** → Attached service account
- **Local development** → `gcloud auth application-default login` (user credentials)

**3. Enforce with organization policy**

```bash
# Prevent service account key creation
gcloud resource-manager org-policies set-policy policy.yaml \
  --organization=ORG_ID

# policy.yaml:
# name: organizations/ORG_ID/policies/iam.disableServiceAccountKeyCreation
# spec:
#   rules:
#   - enforce: true
```

**4. Monitor for key creation attempts**

```bash
# Cloud Logging filter for key creation
protoPayload.methodName="google.iam.admin.v1.CreateServiceAccountKey"
AND severity="NOTICE"

# Set up alert when this occurs
```

### Principle 2: Least Privilege IAM Roles

From [Google Cloud - IAM Best Practices](https://docs.cloud.google.com/iam/docs/best-practices) (accessed 2025-02-03):

**Anti-pattern**: Using primitive roles (Owner, Editor, Viewer)

```bash
# ❌ BAD: Grant Editor role (thousands of permissions!)
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:app-sa@project.iam.gserviceaccount.com" \
  --role="roles/editor"
```

**Best practice**: Use predefined roles or custom roles with minimal permissions

```bash
# ✅ GOOD: Grant specific predefined roles
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:app-sa@project.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"  # Read-only GCS access

gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:app-sa@project.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataViewer"  # Read-only BigQuery access
```

**Custom role for precise control:**

```bash
# Create custom role with exact permissions needed
gcloud iam roles create customStorageUploader \
  --project=PROJECT_ID \
  --title="Custom Storage Uploader" \
  --description="Can upload objects to specific bucket" \
  --permissions="storage.objects.create,storage.objects.delete" \
  --stage=GA

# Grant custom role with resource condition
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:app-sa@project.iam.gserviceaccount.com" \
  --role="projects/PROJECT_ID/roles/customStorageUploader" \
  --condition="
    expression=resource.name.startsWith('projects/_/buckets/uploads-only/'),
    title=uploads-bucket-only
  "
```

### Principle 3: Time-Bound Access with IAM Conditions

From [Google Cloud - Configure Temporary Access](https://docs.cloud.google.com/iam/docs/configuring-temporary-access) (accessed 2025-02-03):

**Pattern**: Temporary elevated access that expires automatically.

```bash
# Grant emergency access for 4 hours
EXPIRY=$(date -u -d '+4 hours' +%Y-%m-%dT%H:%M:%SZ)

gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="user:oncall-engineer@company.com" \
  --role="roles/compute.admin" \
  --condition="
    expression=request.time < timestamp('${EXPIRY}'),
    title=emergency-access-4h,
    description=Break-glass access expires at ${EXPIRY}
  "

# Access automatically revoked after 4 hours - no cleanup needed!
```

**Automation with approval workflow:**

```python
def grant_temporary_access(user_email, role, hours=4):
    """Grant temporary access after approval"""
    import subprocess
    from datetime import datetime, timedelta

    # Calculate expiry
    expiry = (datetime.utcnow() + timedelta(hours=hours)).strftime('%Y-%m-%dT%H:%M:%SZ')

    # Grant with expiry condition
    subprocess.run([
        'gcloud', 'projects', 'add-iam-policy-binding', 'PROJECT_ID',
        '--member', f'user:{user_email}',
        '--role', role,
        '--condition', f"expression=request.time < timestamp('{expiry}'),title=temp-access"
    ])

    print(f"Granted {role} to {user_email} until {expiry}")
```

### Principle 4: Audit Logging and Monitoring

From [Google Cloud - Cloud Audit Logs](https://docs.cloud.google.com/logging/docs/audit) (accessed 2025-02-03):

**Key audit log types:**

1. **Admin Activity Logs** - Who created/modified/deleted resources (always enabled, free)
2. **Data Access Logs** - Who read/wrote data (disabled by default, costs $$)
3. **System Event Logs** - GCP-initiated changes (free)

**Enable data access logs for sensitive resources:**

```bash
# Enable BigQuery data access logs
gcloud projects get-iam-policy PROJECT_ID > policy.yaml

# Edit policy.yaml to add:
# auditConfigs:
# - auditLogConfigs:
#   - logType: DATA_READ
#   - logType: DATA_WRITE
#   service: bigquery.googleapis.com

gcloud projects set-iam-policy PROJECT_ID policy.yaml
```

**Monitor authentication events:**

```bash
# Query: Failed authentication attempts
protoPayload.status.code != 0
AND protoPayload.authenticationInfo.principalEmail:*

# Query: Service account key usage (vs. keyless auth)
protoPayload.authenticationInfo.serviceAccountKeyName:*

# Query: Impersonation events
protoPayload.authenticationInfo.principalEmail != protoPayload.authenticationInfo.serviceAccountDelegationInfo[0].principalEmail

# Create alert policies for suspicious patterns
```

---

## Section 4: Token Management (~100 lines)

### Access Token Lifecycle

From [Google Cloud - Understanding Service Account Tokens](https://docs.cloud.google.com/iam/docs/create-short-lived-credentials-direct) (accessed 2025-02-03):

**Token types:**

1. **Access Tokens** - OAuth2 tokens for API access (1-hour lifetime, max)
2. **ID Tokens** - OIDC tokens for service-to-service auth (1-hour lifetime, max)
3. **Self-signed JWT** - Service account signs its own JWT (up to 12 hours)

**Access token generation:**

```bash
# Generate access token (requires roles/iam.serviceAccountTokenCreator)
gcloud auth print-access-token \
  --impersonate-service-account=SA_EMAIL \
  --lifetime=3600s  # Max: 3600s (1 hour)
```

**ID token generation:**

```bash
# Generate ID token for Cloud Run service
gcloud auth print-identity-token \
  --impersonate-service-account=SA_EMAIL \
  --audiences=https://my-service-abc123.run.app \
  --include-email  # Include email in token claims
```

**Self-signed JWT (long-lived, for specific use cases):**

```python
import google.auth
from google.auth import jwt as google_jwt
import time

# Create self-signed JWT (up to 12 hours)
credentials = google.auth.jwt.Credentials.from_service_account_file(
    'key.json',
    audience='https://my-service.run.app',
    additional_claims={
        'target_audience': 'https://my-service.run.app',
        'exp': int(time.time()) + 43200  # 12 hours
    }
)

# JWT can be reused for 12 hours without refresh!
signed_jwt = credentials.token
```

### Token Caching Strategies

From [Google Cloud - Application Default Credentials](https://docs.cloud.google.com/docs/authentication/application-default-credentials) (accessed 2025-02-03):

**Problem**: Generating new token for every request is slow and rate-limited.

**Solution 1: Use Cloud SDK (built-in caching)**

```python
from google.cloud import storage

# SDK caches tokens automatically
client = storage.Client()  # Token cached for ~55 minutes
buckets = list(client.list_buckets())  # Uses cached token
```

**Solution 2: Manual caching with refresh**

```python
import google.auth
from google.auth.transport.requests import Request
import time

class TokenCache:
    def __init__(self):
        self.credentials, _ = google.auth.default()
        self.token = None
        self.expiry = 0

    def get_token(self):
        """Get cached token or refresh if expired"""
        if time.time() >= self.expiry - 60:  # Refresh 1 min before expiry
            request = Request()
            self.credentials.refresh(request)
            self.token = self.credentials.token
            self.expiry = self.credentials.expiry.timestamp()

        return self.token

# Use cached token
cache = TokenCache()
token = cache.get_token()  # First call: generates token
token = cache.get_token()  # Subsequent calls: returns cached token
```

**Solution 3: Redis-backed token cache (multi-process)**

```python
import redis
import google.auth
from google.auth.transport.requests import Request
import time
import json

class RedisTokenCache:
    def __init__(self, redis_client, sa_email):
        self.redis = redis_client
        self.sa_email = sa_email
        self.cache_key = f"gcp-token:{sa_email}"

    def get_token(self):
        """Get token from Redis cache or generate new one"""
        cached = self.redis.get(self.cache_key)

        if cached:
            data = json.loads(cached)
            if time.time() < data['expiry'] - 60:  # Valid for 1+ more min
                return data['token']

        # Generate new token
        credentials, _ = google.auth.default()
        request = Request()
        credentials.refresh(request)

        # Cache with TTL
        data = {
            'token': credentials.token,
            'expiry': credentials.expiry.timestamp()
        }
        ttl = int(data['expiry'] - time.time())
        self.redis.setex(self.cache_key, ttl, json.dumps(data))

        return credentials.token

# Share tokens across processes
r = redis.Redis(host='localhost', port=6379)
cache = RedisTokenCache(r, 'app-sa@project.iam.gserviceaccount.com')
token = cache.get_token()  # Shared across all workers
```

### Token Refresh Best Practices

From [Google Cloud - Service Account Token Lifecycle](https://docs.cloud.google.com/iam/docs/create-short-lived-credentials-direct) (accessed 2025-02-03):

**1. Refresh proactively (before expiry)**

```python
# ✅ GOOD: Refresh 5 minutes before expiry
if credentials.expiry - datetime.utcnow() < timedelta(minutes=5):
    credentials.refresh(request)

# ❌ BAD: Wait for 401 error, then refresh
try:
    make_request(token)
except Unauthorized:
    credentials.refresh(request)  # User saw error!
```

**2. Handle refresh failures gracefully**

```python
import time

def refresh_with_retry(credentials, request, max_retries=3):
    """Refresh with exponential backoff"""
    for attempt in range(max_retries):
        try:
            credentials.refresh(request)
            return
        except google.auth.exceptions.RefreshError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # 1s, 2s, 4s
            time.sleep(wait_time)
```

**3. Use AuthorizedSession for automatic refresh**

```python
from google.auth.transport.requests import AuthorizedSession
import google.auth

# Session handles refresh automatically
credentials, _ = google.auth.default()
session = AuthorizedSession(credentials)

# No manual refresh needed!
response = session.get('https://www.googleapis.com/storage/v1/b')
# Token refreshed if expired
```

---

## Section 5: Audit Logging for Authentication (~50 lines)

### Tracking Authentication Events

From [Google Cloud - Audit Logs for IAM](https://docs.cloud.google.com/logging/docs/audit/configure-data-access) (accessed 2025-02-03):

**Key queries for authentication monitoring:**

**1. Track service account impersonation:**

```bash
# Cloud Logging filter
protoPayload.serviceData.policyDelta.bindingDeltas.action="ADD"
AND protoPayload.serviceData.policyDelta.bindingDeltas.role="roles/iam.serviceAccountTokenCreator"

# Alert: New impersonation permission granted
```

**2. Track actual impersonation usage:**

```bash
# Filter: User impersonating service account
protoPayload.authenticationInfo.principalEmail="user@company.com"
AND protoPayload.authenticationInfo.serviceAccountDelegationInfo[0].principalEmail="sa@project.iam.gserviceaccount.com"

# Shows: user@company.com impersonated sa@project.iam.gserviceaccount.com
```

**3. Track service account key usage (detect keys still in use):**

```bash
# Filter: Authentication using service account keys
protoPayload.authenticationInfo.serviceAccountKeyName:*

# Alert: Key usage detected (migration to WIF incomplete!)
```

**4. Track failed authentication attempts:**

```bash
# Filter: Authentication failures
protoPayload.status.code != 0
AND protoPayload.authenticationInfo.principalEmail:*

# Alert: Multiple failures from same principal (brute force?)
```

### Example: Anomaly Detection Script

```python
from google.cloud import logging_v2

def detect_unusual_impersonation(project_id, hours=24):
    """Alert on unusual impersonation patterns"""
    client = logging_v2.Client(project=project_id)

    # Query: Impersonation events in last 24 hours
    filter_str = f'''
    protoPayload.authenticationInfo.serviceAccountDelegationInfo[0].principalEmail:*
    AND timestamp >= "{hours_ago}"
    '''

    entries = client.list_entries(filter_=filter_str)

    # Track: Who impersonated which SAs
    impersonations = {}
    for entry in entries:
        user = entry.payload['authenticationInfo']['principalEmail']
        sa = entry.payload['authenticationInfo']['serviceAccountDelegationInfo'][0]['principalEmail']

        impersonations.setdefault(user, set()).add(sa)

    # Alert: User impersonating more than 3 different SAs
    for user, sas in impersonations.items():
        if len(sas) > 3:
            print(f"⚠️  ALERT: {user} impersonated {len(sas)} different SAs in {hours}h")
            for sa in sas:
                print(f"   - {sa}")
```

---

## Sources

**Google Cloud Documentation:**
- [Service Account Impersonation](https://docs.cloud.google.com/iam/docs/service-account-impersonation) (accessed 2025-02-03)
- [Application Default Credentials](https://docs.cloud.google.com/docs/authentication/application-default-credentials) (accessed 2025-02-03)
- [Workload Identity for GKE](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/workload-identity) (accessed 2025-02-03)
- [Creating Short-Lived Service Account Credentials](https://docs.cloud.google.com/iam/docs/create-short-lived-credentials-direct) (accessed 2025-02-03)
- [Troubleshooting Application Default Credentials](https://docs.cloud.google.com/docs/authentication/troubleshoot-adc) (accessed 2025-02-03)
- [IAM Troubleshooting](https://docs.cloud.google.com/iam/docs/troubleshooting-access) (accessed 2025-02-03)
- [Policy Troubleshooter](https://docs.cloud.google.com/iam/docs/troubleshooting-access#policy-troubleshooter) (accessed 2025-02-03)
- [Troubleshooting Workload Identity Federation](https://docs.cloud.google.com/iam/docs/troubleshooting-workload-identity-federation) (accessed 2025-02-03)
- [Best Practices for Managing Service Account Keys](https://docs.cloud.google.com/iam/docs/best-practices-for-managing-service-account-keys) (accessed 2025-02-03)
- [IAM Best Practices](https://docs.cloud.google.com/iam/docs/best-practices) (accessed 2025-02-03)
- [Configure Temporary Access](https://docs.cloud.google.com/iam/docs/configuring-temporary-access) (accessed 2025-02-03)
- [Cloud Audit Logs](https://docs.cloud.google.com/logging/docs/audit) (accessed 2025-02-03)
- [Audit Logs for IAM](https://docs.cloud.google.com/logging/docs/audit/configure-data-access) (accessed 2025-02-03)
- [Cross-Project Service Account Usage](https://docs.cloud.google.com/iam/docs/attach-service-accounts) (accessed 2025-02-03)

**Related Files:**
- [gcloud-production/03-iam-advanced.md](../gcloud-production/03-iam-advanced.md) - Workload Identity Federation, IAM Conditions, Organization Policies

---

## Quick Reference Commands

```bash
# Multi-hop impersonation
gcloud COMMAND --impersonate-service-account=sa-b@project.iam.gserviceaccount.com \
  --impersonate-service-account=sa-a@project.iam.gserviceaccount.com

# Check ADC credential
gcloud auth application-default print-access-token

# Generate short-lived token
gcloud auth print-access-token --impersonate-service-account=SA_EMAIL --lifetime=3600s

# Debug authentication
gcloud auth list
gcloud config get-value auth/impersonate_service_account

# Check IAM permissions
gcloud projects get-iam-policy PROJECT_ID --flatten="bindings[].members" --filter="bindings.members:PRINCIPAL"

# Enable data access logs
# Edit policy with: gcloud projects get-iam-policy PROJECT_ID > policy.yaml
# Add auditConfigs section, then:
gcloud projects set-iam-policy PROJECT_ID policy.yaml

# Monitor authentication events (Cloud Logging)
protoPayload.authenticationInfo.principalEmail:*
protoPayload.authenticationInfo.serviceAccountKeyName:*  # Key usage
protoPayload.status.code != 0  # Failed auth
```
