# GCloud Production IAM - Advanced Topics

**Date**: 2025-02-03
**Category**: Production IAM
**Scope**: Workload Identity Federation, Key Management, IAM Conditions, Organization Policies

---

## Overview

Advanced IAM features in GCP provide sophisticated security controls beyond basic role assignments. This guide covers four critical production patterns: Workload Identity Federation (keyless authentication), service account key management best practices, IAM conditions (temporal/attribute-based access), and organization policies (governance constraints).

**Why these matter for training workloads:**
- WIF eliminates service account keys for external systems (GitHub Actions, GitLab CI/CD)
- Key management reduces credential exposure and rotation overhead
- IAM conditions enable just-in-time access and time-bound permissions
- Organization policies enforce security guardrails across entire GCP hierarchy

---

## Section 1: Workload Identity Federation (~200 lines)

### What is Workload Identity Federation?

From [Google Cloud Workload Identity Federation Documentation](https://docs.cloud.google.com/iam/docs/workload-identity-federation) (accessed 2025-02-03):

Workload Identity Federation (WIF) allows external workloads (outside GCP) to access GCP resources **without service account keys**. Instead, external identity providers (IdPs) like AWS, Azure, GitHub Actions, GitLab, or any OIDC-compliant provider issue short-lived tokens that GCP trusts.

**Key concepts:**
- **Workload Identity Pool**: Container for external identities from a specific IdP
- **Workload Identity Provider**: Configuration within a pool defining how to trust tokens from specific IdP
- **Token Exchange**: External JWT/OIDC token → GCP federated token → Service account impersonation

### Architecture Pattern

```
External System (GitHub Actions)
    ↓
Issues OIDC JWT Token (aud: GCP Workload Identity Provider)
    ↓
gcloud iam workload-identity-pools create-cred-config
    ↓
Exchanges JWT for GCP Federated Token
    ↓
Impersonates GCP Service Account
    ↓
Access GCP Resources (Cloud Run, Artifact Registry, etc.)
```

**No keys stored anywhere!** The entire flow uses short-lived tokens.

### Setup Example: GitLab CI/CD

From [Medium - Configure GCP Workload Identity Federation for GitLab](https://medium.com/google-cloud/configure-gcp-workload-identity-federation-for-gitlab-c526e6eb0517) by Rohan Singh (accessed 2025-02-03):

**Step 1: Create Workload Identity Pool and Provider**

```bash
# Create pool
gcloud iam workload-identity-pools create gitlab \
  --location=global \
  --description="GitLab CI/CD workload identity pool"

# Create provider
gcloud iam workload-identity-pools providers create-oidc middleman-provider \
  --location=global \
  --workload-identity-pool=gitlab \
  --issuer-uri="https://gitlab.com" \
  --attribute-mapping="
    google.subject=assertion.project_id + '::' + assertion.ref,
    attribute.aud=assertion.aud,
    attribute.project_id=assertion.project_id,
    attribute.project_path=assertion.project_path,
    attribute.namespace_path=assertion.namespace_path,
    attribute.ref=assertion.ref,
    attribute.ref_protected='assertion.ref_protected ? \"true\" : \"false\"',
    attribute.ref_type=assertion.ref_type,
    attribute.sha=assertion.sha
  " \
  --attribute-condition="attribute.project_id=='<GITLAB_REPO_PROJECT_ID>'"
```

**Attribute Mapping Explained:**
- `google.subject`: Composite identifier (project ID + Git ref) for granular IAM bindings
- `attribute.project_id`: GitLab project numerical ID (for filtering in conditions)
- `attribute.ref`: Git branch/tag (e.g., `main`, `feature/new-feature`)
- `attribute.ref_protected`: Boolean - is this a protected branch? (critical for prod deployments)
- `attribute.namespace_path`: GitLab group hierarchy (e.g., `my-org/my-team`)

**Attribute Condition**: Restricts pool to specific GitLab project(s) - only jobs from this project can authenticate.

**Step 2: Create Service Account and Grant Permissions**

```bash
# Create SA for impersonation
gcloud iam service-accounts create gitlab-sa \
  --description="Service account for GitLab CI/CD via WIF" \
  --display-name="GitLab WIF Service Account"

# Grant permissions in target projects (example: Cloud Run Admin)
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:gitlab-sa@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:gitlab-sa@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
```

**Step 3: Grant Workload Identity User Role**

```bash
# Allow federated identities from pool to impersonate SA
gcloud iam service-accounts add-iam-policy-binding \
  gitlab-sa@PROJECT_ID.iam.gserviceaccount.com \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/gitlab/*"
```

**Principal types:**
- `principalSet://.../*` - All identities in pool
- `principal://.../<subject>` - Specific identity (e.g., specific project + branch)
- `principalSet://.../<attribute>=<value>` - Identities matching attribute filter

### GitLab CI/CD Template

From [Medium - Configure GCP Workload Identity Federation for GitLab](https://medium.com/google-cloud/configure-gcp-workload-identity-federation-for-gitlab-c526e6eb0517) (accessed 2025-02-03):

```yaml
# .gcp-wif-auth.gitlab-ci.yml
.gcp-wif-auth:
  stage: gcp-wif-auth
  image: google/cloud-sdk:slim
  artifacts:
    paths:
    - $GOOGLE_CREDENTIALS
    - .gitlab-oidc-jwt
  id_tokens:
    GCP_OIDC_TOKEN:
      aud: "//iam.googleapis.com/${GCP_WI_PROVIDER_PATH}"
  variables:
    GCP_WI_PROVIDER_PATH: "projects/<PROJECT_NUMBER>/locations/global/workloadIdentityPools/gitlab/providers/middleman-provider"
    GCP_SERVICE_ACCOUNT_EMAIL: "gitlab-sa@PROJECT_ID.iam.gserviceaccount.com"
    GOOGLE_CREDENTIALS: ".gcp_temp_cred.json"
  before_script:
    # Store OIDC JWT from GitLab
    - echo "${GCP_OIDC_TOKEN}" > "${CI_PROJECT_DIR}/.gitlab-oidc-jwt"
    - unset GCP_OIDC_TOKEN

    # Exchange JWT for GCP federated token
    - >
      gcloud iam workload-identity-pools create-cred-config "${GCP_WI_PROVIDER_PATH}"
      --service-account="${GCP_SERVICE_ACCOUNT_EMAIL}"
      --output-file="${CI_PROJECT_DIR}/.gcp_temp_cred.json"
      --credential-source-file="${CI_PROJECT_DIR}/.gitlab-oidc-jwt"

    # Authenticate and set project
    - gcloud auth login --cred-file="${CI_PROJECT_DIR}/.gcp_temp_cred.json"
    - export GOOGLE_APPLICATION_CREDENTIALS="${CI_PROJECT_DIR}/.gcp_temp_cred.json"
    - gcloud config set project "${GCP_TARGET_PROJECT_ID}"

    - echo "--- GCP Authentication Successful! ---"
    - gcloud auth list
```

**Key insight**: `gcloud iam workload-identity-pools create-cred-config` creates ADC (Application Default Credentials) file that handles token exchange automatically.

### Multi-Project Strategy

From [Medium - Configure GCP Workload Identity Federation for GitLab](https://medium.com/google-cloud/configure-gcp-workload-identity-federation-for-gitlab-c526e6eb0517) (accessed 2025-02-03):

**Centralized WIF Project Pattern:**
- Create dedicated "WIF" project containing pools, providers, and central service accounts
- Central SA granted permissions across multiple target projects
- Reduces WIF configuration overhead (5 BUs = 5 WIF configs, not 100 project configs)

**Per-Project SA Pattern:**
- Each target project has its own dedicated service account
- More granular isolation but higher administrative overhead
- Better for strict security boundaries

**Hybrid approach**: Centralized WIF configuration + per-project SAs (best of both worlds).

### Best Practices

From [Google Cloud - Best Practices for Workload Identity Federation](https://docs.cloud.google.com/iam/docs/best-practices-for-using-workload-identity-federation) (accessed 2025-02-03):

1. **Use attribute conditions** to restrict pool to specific projects/repos
2. **Leverage ref_protected attribute** for production deployments (only protected branches)
3. **Map composite google.subject** for fine-grained IAM policies (project + branch)
4. **Rotate nothing** - tokens are short-lived by design (no key rotation needed!)
5. **Audit principal identities** in Cloud Logging - see which external identity accessed what
6. **Test with specific principals** first, then expand to principalSet

### Debugging WIF

```bash
# Get pool info
gcloud iam workload-identity-pools describe gitlab \
  --location=global \
  --format=yaml

# Test token exchange (requires valid external JWT)
gcloud auth login --cred-file=CREDENTIAL_CONFIG_FILE

# Check which federated identity is active
gcloud auth list

# View IAM policy for SA (who can impersonate?)
gcloud iam service-accounts get-iam-policy SA_EMAIL
```

---

## Section 2: Service Account Key Management (~150 lines)

### The Problem with Service Account Keys

From [Google Cloud - Best Practices for Managing Service Account Keys](https://docs.cloud.google.com/iam/docs/best-practices-for-managing-service-account-keys) (accessed 2025-02-03):

Service account keys are **long-lived credentials** that never expire by default. If leaked, they provide indefinite access until manually revoked.

**Key risks:**
- **Leaked in source code** - Accidentally committed to GitHub repos
- **Stored insecurely** - Hardcoded in CI/CD variables, config files, developer laptops
- **No automatic expiration** - Keys work forever unless you remember to rotate them
- **Hard to audit** - Which keys are actually in use? Which are forgotten?

**Google's #1 recommendation**: **Don't use service account keys if you can avoid them!**

### Alternatives to Service Account Keys

From [Google Cloud - Best Practices for Managing Service Account Keys](https://docs.cloud.google.com/iam/docs/best-practices-for-managing-service-account-keys) (accessed 2025-02-03):

1. **Workload Identity Federation** (external workloads) - covered in Section 1
2. **Workload Identity** (GKE) - Kubernetes pods authenticate as GCP service accounts
3. **Service Account Impersonation** - User/SA impersonates another SA temporarily
4. **Default service accounts** - Compute Engine VMs, Cloud Run, Cloud Functions use attached SAs
5. **ADC (Application Default Credentials)** - `gcloud auth application-default login` for local development

**Only use keys when:**
- On-premises workloads with no OIDC provider
- Legacy systems that require static credentials
- Third-party tools that don't support WIF

### Best Practices for Key Management

From [Google Cloud - Best Practices for Managing Service Account Keys](https://docs.cloud.google.com/iam/docs/best-practices-for-managing-service-account-keys) (accessed 2025-02-03):

**1. Use Short-Lived Keys with Expiration**

```bash
# Create key with expiry (max 10 years, but use hours/days!)
gcloud iam service-accounts keys create key.json \
  --iam-account=SA_EMAIL \
  --key-file-type=json \
  --expiration=2025-02-04T00:00:00Z
```

**Recommended expiry times:**
- Development: 1-7 days
- CI/CD (if WIF unavailable): 30-90 days
- Production: Avoid keys entirely, use WIF/Workload Identity

**2. Rotate Keys Regularly**

From [Community Tech Alliance - Best Practices for Managing GCP Service Account Keys](https://help.techallies.org/support/solutions/articles/154000222642-best-practices-for-managing-gcp-service-account-keys) (accessed 2025-02-03):

```bash
# List all keys for SA
gcloud iam service-accounts keys list \
  --iam-account=SA_EMAIL \
  --format="table(name,validAfterTime,validBeforeTime)"

# Create new key
gcloud iam service-accounts keys create new-key.json \
  --iam-account=SA_EMAIL

# Update application to use new key
# ... deploy new key to systems ...

# Delete old key
gcloud iam service-accounts keys delete KEY_ID \
  --iam-account=SA_EMAIL
```

**Rotation schedule:**
- High-privilege SAs: Every 30 days
- Medium-privilege SAs: Every 90 days
- Low-privilege SAs: Every 180 days

**3. Use Service Account Insights**

From [Google Cloud - Best Practices for Managing Service Account Keys](https://docs.cloud.google.com/iam/docs/best-practices-for-managing-service-account-keys) (accessed 2025-02-03):

```bash
# Check when key was last used
gcloud iam service-accounts keys get-iam-policy KEY_ID \
  --iam-account=SA_EMAIL

# View service account activity
gcloud logging read \
  "protoPayload.authenticationInfo.principalEmail=\"SA_EMAIL\"" \
  --limit=50 \
  --format=json
```

Service Account Insights (in Console) shows:
- Last authentication time for each key
- Unused keys (never authenticated or not used in 90+ days)
- Recommendations to delete stale keys

**4. Store Keys Securely**

From [Google Cloud - Secure Service Account Keys](https://cloud.google.com/distributed-cloud/hosted/docs/latest/gdch/platform/pa-user/iam/secure-service-account-keys) (accessed 2025-02-03):

**❌ DON'T:**
- Submit keys to source code repos (even private repos!)
- Store in plaintext environment variables
- Email keys or share via chat
- Store on developer laptops without encryption

**✅ DO:**
- Use Secret Manager for production keys
- Use CI/CD secret variables (encrypted at rest)
- Use expiry times to limit blast radius
- Delete keys immediately after migration to WIF

**5. Monitor for Key Creation**

```bash
# Alert on new key creation (Cloud Logging filter)
protoPayload.methodName="google.iam.admin.v1.CreateServiceAccountKey"
AND severity="NOTICE"
```

**Why monitor?**: Unexpected key creation may indicate:
- Developer bypassing WIF (needs training)
- Attacker with compromised credentials
- Automated system misconfiguration

### Key Rotation Automation Example

From [Community Tech Alliance - Best Practices for Managing GCP Service Account Keys](https://help.techallies.org/support/solutions/articles/154000222642-best-practices-for-managing-gcp-service-account-keys) (accessed 2025-02-03):

```bash
#!/bin/bash
# rotate-sa-keys.sh - Automated key rotation

SA_EMAIL="my-app@project.iam.gserviceaccount.com"
SECRET_NAME="my-app-sa-key"

# Create new key
NEW_KEY=$(mktemp)
gcloud iam service-accounts keys create "$NEW_KEY" \
  --iam-account="$SA_EMAIL"

# Store in Secret Manager
gcloud secrets versions add "$SECRET_NAME" \
  --data-file="$NEW_KEY"

# Wait for application to pick up new key (e.g., 5 minutes)
echo "Waiting for application to reload secret..."
sleep 300

# Delete old key (keep 2 most recent)
OLD_KEYS=$(gcloud iam service-accounts keys list \
  --iam-account="$SA_EMAIL" \
  --format="value(name)" \
  --filter="validAfterTime<$(date -d '30 days ago' -Iseconds)" \
  | head -n -1)

for KEY in $OLD_KEYS; do
  echo "Deleting old key: $KEY"
  gcloud iam service-accounts keys delete "$KEY" \
    --iam-account="$SA_EMAIL" \
    --quiet
done

# Clean up
rm "$NEW_KEY"
echo "Key rotation complete!"
```

**Schedule with Cloud Scheduler**: Run monthly for automated rotation.

---

## Section 3: IAM Conditions (Beta) (~150 lines)

### What are IAM Conditions?

From [Google Cloud - Overview of IAM Conditions](https://docs.cloud.google.com/iam/docs/conditions-overview) (accessed 2025-02-03):

IAM Conditions allow you to add **conditional logic** to IAM policy bindings. Grant access only when certain attributes match:
- **Time-based**: Access expires after specific date/time
- **Resource-based**: Access to specific resources only (e.g., specific buckets)
- **Request-based**: Access from specific IP ranges, during business hours

**Key use cases:**
- Temporary access for contractors (expires automatically)
- Emergency break-glass access (valid for 4 hours)
- Business hours only access (9 AM - 5 PM weekdays)
- Production access only from VPN IPs

### Architecture: Spotify's Gimme Tool

From [Spotify Engineering - Releasing Gimme: Managing Time Bound IAM Conditions](https://engineering.atspotify.com/2018/07/releasing-gimme-managing-time-bound-iam-conditions-in-google-cloud-platform) (accessed 2025-02-03):

Spotify built "Gimme" - a web interface for creating time-bound IAM conditions:

**Problem**: Engineers need temporary access for debugging, but manual revocation is error-prone (people forget!).

**Solution**: Self-service time-bound access:
1. Engineer requests access to resource via Gimme UI
2. Gimme creates IAM binding with expiry condition (e.g., 4 hours)
3. Engineer gets access immediately
4. Access **automatically expires** - no manual cleanup needed!

**Benefits:**
- True temporary access (not relying on business processes)
- Reduced attack surface (access expires by default)
- Audit trail (who requested what, when)
- No credential management (users authenticate with their own identity)

### Temporal Access Example

From [Google Cloud - Configure Temporary Access](https://docs.cloud.google.com/iam/docs/configuring-temporary-access) (accessed 2025-02-03):

```bash
# Grant temporary access (expires in 4 hours)
EXPIRY=$(date -u -d '+4 hours' +%Y-%m-%dT%H:%M:%SZ)

gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="user:engineer@company.com" \
  --role="roles/compute.admin" \
  --condition="
    expression=request.time < timestamp('${EXPIRY}'),
    title=temporary-access,
    description=Emergency access expires at ${EXPIRY}
  "
```

**Condition syntax**: Common Expression Language (CEL)
- `request.time` - Current request timestamp
- `timestamp('2025-02-04T00:00:00Z')` - Specific time
- Comparison operators: `<`, `>`, `<=`, `>=`, `==`, `!=`

**Verification:**

```bash
# View IAM policy with conditions
gcloud projects get-iam-policy PROJECT_ID \
  --format=yaml

# Output shows:
# bindings:
# - members:
#   - user:engineer@company.com
#   role: roles/compute.admin
#   condition:
#     expression: request.time < timestamp("2025-02-04T00:00:00Z")
#     title: temporary-access
```

### Advanced Condition Examples

From [Google Cloud - Configure Temporary Access](https://docs.cloud.google.com/iam/docs/configuring-temporary-access) (accessed 2025-02-03):

**1. Time Window (start AND end)**

```bash
# Access only valid between 2025-02-01 and 2025-02-05
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="user:contractor@external.com" \
  --role="roles/viewer" \
  --condition="
    expression=
      request.time >= timestamp('2025-02-01T00:00:00Z') &&
      request.time < timestamp('2025-02-05T00:00:00Z'),
    title=contractor-access-window
  "
```

**2. Business Hours Only**

```bash
# Access only 9 AM - 5 PM UTC, Monday-Friday
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="user:intern@company.com" \
  --role="roles/bigquery.user" \
  --condition="
    expression=
      request.time.getHours('UTC') >= 9 &&
      request.time.getHours('UTC') < 17 &&
      request.time.getDayOfWeek('UTC') >= 1 &&
      request.time.getDayOfWeek('UTC') <= 5,
    title=business-hours-only
  "
```

**3. Resource-Specific Access**

```bash
# Access to specific bucket only
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:data-pipeline@project.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer" \
  --condition="
    expression=resource.name.startsWith('projects/_/buckets/my-specific-bucket/'),
    title=single-bucket-access
  "
```

**4. IP Range Restriction**

```bash
# Access only from office IP range
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="user:employee@company.com" \
  --role="roles/editor" \
  --condition="
    expression=inIpRange(request.origin.ip, '203.0.113.0/24'),
    title=office-ip-only
  "
```

### Programmatic Condition Management

From [P0 Security - Granting Temporary Access in Google Cloud](https://www.p0.dev/blog/gcloud-access) (accessed 2025-02-03):

**Python example** using Google Cloud IAM API:

```python
from google.cloud import iam_v1
from google.protobuf.timestamp_pb2 import Timestamp
import time

def grant_temporary_access(project_id, member, role, hours=4):
    """Grant temporary access that expires in N hours"""

    client = iam_v1.IAMClient()
    resource = f"projects/{project_id}"

    # Get current policy
    policy = client.get_iam_policy(request={"resource": resource})

    # Calculate expiry time
    expiry_timestamp = Timestamp()
    expiry_timestamp.FromSeconds(int(time.time()) + hours * 3600)

    # Create condition
    condition = iam_v1.Expr(
        expression=f"request.time < timestamp('{expiry_timestamp.ToJsonString()}')",
        title="temporary-access",
        description=f"Expires in {hours} hours"
    )

    # Add binding with condition
    binding = iam_v1.Binding(
        role=role,
        members=[member],
        condition=condition
    )
    policy.bindings.append(binding)

    # Set updated policy
    client.set_iam_policy(request={"resource": resource, "policy": policy})
    print(f"Granted {member} role {role} until {expiry_timestamp.ToJsonString()}")
```

### Condition Limitations (Beta)

From [Google Cloud - Overview of IAM Conditions](https://docs.cloud.google.com/iam/docs/conditions-overview) (accessed 2025-02-03):

**Current limitations:**
- Not supported for **primitive roles** (Owner, Editor, Viewer)
- Not supported for **allUsers** or **allAuthenticatedUsers**
- Maximum 20 conditional bindings per role per resource
- Conditions evaluated at request time (slight latency overhead)
- Some services don't support conditions yet (check documentation)

**Best practices:**
- Use predefined roles (not primitive) for conditional access
- Test conditions thoroughly before production use
- Monitor for condition evaluation errors in Cloud Logging
- Document expiry times clearly in condition descriptions

---

## Section 4: Organization Policies (~100 lines)

### What are Organization Policies?

From [Orca Security - An Overview of GCP Organization Policies](https://orca.security/resources/blog/google-cloud-platform-gcp-organization/) (accessed 2025-02-03):

Organization Policies are **governance constraints** applied to the resource hierarchy (Organization → Folder → Project → Resource). They restrict **what configurations** can be applied, regardless of IAM permissions.

**Key difference from IAM:**
- **IAM**: Controls **who** can do **what** (permissions)
- **Organization Policies**: Controls **what configurations** are allowed (constraints)

**Hierarchy inheritance:**
```
Organization: "No public IPs allowed"
    ↓
Folder: "Allow public IPs for dev projects"  (can override)
    ↓
Project: "Allow public IPs for staging VMs"  (can further refine)
    ↓
Resource: Inherits effective policy
```

### Constraint Types

From [Google Cloud - Organization Policy Constraints](https://docs.cloud.google.com/resource-manager/docs/organization-policy/org-policy-constraints) (accessed 2025-02-03):

**1. Boolean Constraints** (enforced or not)

Example: `sql.restrictPublicIp`
- **True**: Prevent Cloud SQL instances from having public IP
- **False**: Allow Cloud SQL instances with public IP (default)

**2. List Constraints** (allowed/denied values)

Example: `compute.vmExternalIpAccess`
- **Allowed list**: Only VMs in this list can have external IPs
- **Denied list**: VMs in this list cannot have external IPs
- **Allow all**: Default behavior

### Common Organization Policies

From [Orca Security - An Overview of GCP Organization Policies](https://orca.security/resources/blog/google-cloud-platform-gcp-organization/) and [Google Cloud - Organization Policy Constraints](https://docs.cloud.google.com/resource-manager/docs/organization-policy/org-policy-constraints) (accessed 2025-02-03):

**Security-focused policies** (recommended for production):

1. **Disable automatic public IPs**
   - Constraint: `compute.vmExternalIpAccess`
   - Effect: Prevents VMs from getting public IPs (use Cloud NAT instead)

2. **Restrict service account key creation**
   - Constraint: `iam.disableServiceAccountKeyCreation`
   - Effect: Forces use of Workload Identity/WIF (no keys allowed!)

3. **Require VPC Service Controls**
   - Constraint: `compute.requireVpcFlowLogs`
   - Effect: All subnets must have flow logs enabled

4. **Disable serial port access**
   - Constraint: `compute.disableSerialPortAccess`
   - Effect: Prevents serial console access to VMs (security risk)

5. **Restrict public buckets**
   - Constraint: `storage.publicAccessPrevention`
   - Effect: Buckets cannot be made publicly accessible

6. **Define allowed resource locations**
   - Constraint: `gcp.resourceLocations`
   - Effect: Resources must be in specific regions (data residency compliance)

### Example: Restrict External IPs

From [Orca Security - An Overview of GCP Organization Policies](https://orca.security/resources/blog/google-cloud-platform-gcp-organization/) (accessed 2025-02-03):

```yaml
# Organization policy YAML
name: organizations/123456789/policies/compute.vmExternalIpAccess
spec:
  rules:
  - allowAll: false  # Deny all by default
  - values:
      allowedValues:
      - projects/my-project/zones/us-central1-a/instances/bastion-host
      - projects/my-project/zones/us-central1-a/instances/nat-gateway
    condition:
      expression: "!has(resource.labels.allow_external_ip)"  # Unless tagged
      title: "Allow specific VMs or tagged resources"
```

**Apply via gcloud:**

```bash
# Set organization policy
gcloud resource-manager org-policies set-policy policy.yaml \
  --organization=123456789

# Or at project level
gcloud resource-manager org-policies set-policy policy.yaml \
  --project=my-project
```

### Conditions with Resource Tags

From [Orca Security - An Overview of GCP Organization Policies](https://orca.security/resources/blog/google-cloud-platform-gcp-organization/) (accessed 2025-02-03):

**Scenario**: Allow external IPs only for resources tagged with `allow_external_ip: true`

```yaml
name: organizations/123456789/policies/compute.vmExternalIpAccess
spec:
  rules:
  - allowAll: false
  - allowAll: true
    condition:
      expression: "resource.matchTag('123456789/allow_external_ip', 'true')"
      title: "Allow if tagged"
```

**Create and apply tags:**

```bash
# Create tag key
gcloud resource-manager tags keys create allow_external_ip \
  --parent=organizations/123456789

# Create tag value
gcloud resource-manager tags values create true \
  --parent=tagKeys/allow_external_ip

# Tag a project
gcloud resource-manager tags bindings create \
  --tag-value=tagValues/true \
  --parent=//cloudresourcemanager.googleapis.com/projects/my-project
```

### Best Practices for Organization Policies

From [Medium - Comprehensive Guide to Organization Policies in Google Cloud](https://medium.com/google-cloud/comprehensive-guide-to-organization-policies-in-google-cloud-a81ff9e1c9eb) (accessed 2025-02-03):

1. **Start at Organization level** - Enforce baseline security for all resources
2. **Use "deny by default, allow exceptions"** - Safer than "allow all, deny some"
3. **Leverage resource tags** - Flexible exemptions without modifying policies
4. **Test in dev first** - Policies can break deployments if misconfigured
5. **Monitor policy violations** - Alert when someone tries prohibited action
6. **Document policy intent** - Use clear titles/descriptions for future maintainers

**Recommended starter policies** (from Medium article):
- Disable service account key creation (force WIF/Workload Identity)
- Restrict resource locations (data residency)
- Require VPC flow logs
- Disable serial port access
- Enable OS Login for VMs
- Restrict public IPs

---

## Sources

**Google Cloud Documentation:**
- [Workload Identity Federation](https://docs.cloud.google.com/iam/docs/workload-identity-federation) (accessed 2025-02-03)
- [Best Practices for Managing Service Account Keys](https://docs.cloud.google.com/iam/docs/best-practices-for-managing-service-account-keys) (accessed 2025-02-03)
- [Configure Temporary Access (IAM Conditions)](https://docs.cloud.google.com/iam/docs/configuring-temporary-access) (accessed 2025-02-03)
- [Organization Policy Constraints](https://docs.cloud.google.com/resource-manager/docs/organization-policy/org-policy-constraints) (accessed 2025-02-03)
- [Overview of IAM Conditions](https://docs.cloud.google.com/iam/docs/conditions-overview) (accessed 2025-02-03)
- [Secure Service Account Keys](https://cloud.google.com/distributed-cloud/hosted/docs/latest/gdch/platform/pa-user/iam/secure-service-account-keys) (accessed 2025-02-03)
- [Best Practices for Workload Identity Federation](https://docs.cloud.google.com/iam/docs/best-practices-for-using-workload-identity-federation) (accessed 2025-02-03)

**Community Articles:**
- [Medium - Configure GCP Workload Identity Federation for GitLab](https://medium.com/google-cloud/configure-gcp-workload-identity-federation-for-gitlab-c526e6eb0517) by Rohan Singh (accessed 2025-02-03)
- [Spotify Engineering - Releasing Gimme: Managing Time Bound IAM Conditions](https://engineering.atspotify.com/2018/07/releasing-gimme-managing-time-bound-iam-conditions-in-google-cloud-platform) (accessed 2025-02-03)
- [Orca Security - An Overview of GCP Organization Policies](https://orca.security/resources/blog/google-cloud-platform-gcp-organization/) (accessed 2025-02-03)
- [P0 Security - Granting Temporary Access in Google Cloud](https://www.p0.dev/blog/gcloud-access) (accessed 2025-02-03)
- [Community Tech Alliance - Best Practices for Managing GCP Service Account Keys](https://help.techallies.org/support/solutions/articles/154000222642-best-practices-for-managing-gcp-service-account-keys) (accessed 2025-02-03)
- [Medium - Comprehensive Guide to Organization Policies in Google Cloud](https://medium.com/google-cloud/comprehensive-guide-to-organization-policies-in-google-cloud-a81ff9e1c9eb) (accessed 2025-02-03)

**Additional References:**
- [Firefly - Setting Up Workload Identity Federation Between GitHub Actions and GCP](https://www.firefly.ai/academy/setting-up-workload-identity-federation-between-github-actions-and-google-cloud-platform) (accessed 2025-02-03)
- [Ventx - Understanding Workload Identity Federation and How to Use It](https://www.ventx.de/en/blog/gcp_workload_identity_federation/) (accessed 2025-02-03)
- [Sysdig - 24 Google Cloud Platform (GCP) Security Best Practices](https://www.sysdig.com/learn-cloud-native/24-google-cloud-platform-gcp-security-best-practices) (accessed 2025-02-03)

---

## Quick Reference Commands

```bash
# Workload Identity Federation
gcloud iam workload-identity-pools create POOL_ID --location=global
gcloud iam workload-identity-pools providers create-oidc PROVIDER_ID \
  --workload-identity-pool=POOL_ID --location=global \
  --issuer-uri=ISSUER_URL --attribute-mapping=...

# Service Account Keys
gcloud iam service-accounts keys list --iam-account=SA_EMAIL
gcloud iam service-accounts keys create key.json --iam-account=SA_EMAIL --expiration=DATE
gcloud iam service-accounts keys delete KEY_ID --iam-account=SA_EMAIL

# IAM Conditions (Temporary Access)
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member=PRINCIPAL --role=ROLE \
  --condition="expression=request.time < timestamp('DATE'),title=TITLE"

# Organization Policies
gcloud resource-manager org-policies set-policy policy.yaml --organization=ORG_ID
gcloud resource-manager org-policies describe CONSTRAINT --project=PROJECT_ID
```
