# GCP Zero Trust Architecture - BeyondCorp, IAP, and Context-Aware Access

**Date**: 2025-02-03
**Category**: Security
**Scope**: Zero Trust implementation, BeyondCorp Enterprise, Identity-Aware Proxy, Context-Aware Access

---

## Overview

Zero Trust is a security model that eliminates the concept of implicit trust within a network perimeter. Instead of assuming that users, devices, and applications inside the corporate network are trustworthy, Zero Trust requires continuous verification of all entities attempting to access resources, regardless of their location.

From [Medium - Zero Trust Architecture in GCP](https://medium.com/google-cloud/zero-trust-architecture-in-gcp-9b8ec1bbf578) (accessed 2025-02-03):

**Core principle**: "Trust No One, Verify Everything"

Zero Trust was introduced by John Kindervag (Forrester Research) in 2010 to address a critical flaw in traditional network security: their reliance on implicit trust. Traditional "castle-and-moat" security assumes everything inside the network perimeter is safe, creating vulnerabilities that can compromise entire systems if exploited.

**Why Zero Trust matters for AI/ML training:**
- Training workloads access sensitive data (datasets, model weights, hyperparameters)
- Compute resources are distributed (multi-region, multi-cloud, on-premise)
- External integrations (W&B, GitHub, artifact registries) require secure access
- Compromised credentials could exfiltrate training data or models

---

## Section 1: Zero Trust Principles (~100 lines)

### The Five Pillars of Zero Trust

From [Cloud4C - Zero Trust Security with Google Cloud](https://www.cloud4c.com/blogs/implementing-zero-trust-security-with-google-cloud) (accessed 2025-02-03):

**1. Verify Explicitly**
- Always authenticate and authorize based on all available data points
- Use multi-factor authentication (MFA)
- Analyze device health, user identity, location, and service access patterns

**2. Use Least Privilege Access**
- Grant minimum permissions required to complete a task
- Just-in-time (JIT) and just-enough-access (JEA)
- Risk-based adaptive policies

**3. Assume Breach**
- Design security assuming attackers are already inside
- Minimize blast radius through micro-segmentation
- Verify end-to-end encryption

**4. Micro-Segment Networks**
- Divide networks into isolated segments
- Prevent lateral movement between segments
- Apply granular access controls at each boundary

**5. Leverage Continuous Monitoring and Analytics**
- Real-time visibility into all access requests
- Anomaly detection using behavioral analytics
- Automated threat response

### How Zero Trust Differs from Traditional Security

From [Medium - Zero Trust Architecture in GCP](https://medium.com/google-cloud/zero-trust-architecture-in-gcp-9b8ec1bbf578) (accessed 2025-02-03):

| Traditional (Perimeter-Based) | Zero Trust |
|-------------------------------|------------|
| Trust inside network | Trust nothing by default |
| VPN for remote access | Context-aware access everywhere |
| Coarse-grained network policies | Micro-segmentation and least privilege |
| Periodic compliance checks | Continuous monitoring and validation |
| Network location determines access | Identity and context determine access |

**Key insight**: Traditional security fails in modern hybrid/multi-cloud environments where:
- Users work from anywhere
- Data resides in multiple clouds and SaaS applications
- Applications are containerized and distributed
- Attack surface is massive and constantly changing

---

## Section 2: BeyondCorp - Google's Zero Trust Implementation (~150 lines)

### What is BeyondCorp?

From [Medium - Zero Trust Architecture in GCP](https://medium.com/google-cloud/zero-trust-architecture-in-gcp-9b8ec1bbf578) (accessed 2025-02-03):

BeyondCorp is Google's implementation of Zero Trust, developed internally over a decade (starting 2011) and now offered as **BeyondCorp Enterprise** for Google Cloud customers.

**Core concept**: Move security perimeter from network to individual users and devices. Enable employees to work securely from any location without a traditional VPN.

**BeyondCorp vs VPN:**

```
Traditional VPN:
User → VPN Gateway → Corporate Network → All Resources (full access)

BeyondCorp:
User + Device → Identity Verification → Context-Aware Access → Specific Resource (minimal access)
```

**Benefits**:
- No VPN overhead (faster, simpler user experience)
- Granular access controls per application
- Device security posture verification
- Works for on-premise, cloud, and SaaS applications

### BeyondCorp Enterprise Components

From [Promevo - Implementing Zero Trust with Google](https://promevo.com/blog/implementing-zero-trust-with-google) (accessed 2025-02-03):

**1. Identity-Aware Proxy (IAP)**
- Application-level access control (Layer 7)
- Verifies user identity and device context before granting access
- No VPN required

**2. Cloud Identity**
- Centralized identity management
- Multi-factor authentication (MFA)
- Security key enforcement (phishing-resistant)

**3. Access Context Manager**
- Define access levels based on dynamic attributes
- Context factors: user identity, device status, location, IP address
- Centralized policy engine

**4. Endpoint Verification**
- Device trust assessment
- Chrome Enterprise integration
- Android Enterprise management

**5. Threat and Data Protection**
- Phishing and malware protection
- Data Loss Prevention (DLP)
- Chrome Enterprise security controls

### BeyondCorp Architecture for AI/ML Workloads

From [Medium - Zero Trust Architecture in GCP](https://medium.com/google-cloud/zero-trust-architecture-in-gcp-9b8ec1bbf578) (accessed 2025-02-03):

```
ML Engineer (Remote)
    ↓ [1. Authenticate with Cloud Identity + MFA]
Cloud Identity
    ↓ [2. Check device posture (Endpoint Verification)]
Access Context Manager
    ↓ [3. Evaluate access level (location, IP, device health)]
Identity-Aware Proxy
    ↓ [4. Verify authorization (IAM roles)]
Protected Resource:
    - Vertex AI Training Job
    - Cloud Storage (training data)
    - Artifact Registry (container images)
    - Secret Manager (API keys)
```

**Key insight**: Every access request is verified at multiple layers. Even if one layer is compromised, others prevent unauthorized access.

### Device Trust with Endpoint Verification

From [Promevo - Implementing Zero Trust with Google](https://promevo.com/blog/implementing-zero-trust-with-google) (accessed 2025-02-03):

**What it checks**:
- Device encryption enabled
- Screen lock configured
- Operating system version (up-to-date)
- Chrome browser version
- Security patches applied
- Corporate management enrollment

**Enforcement options**:
- **Block**: Deny access from non-compliant devices
- **Allow with warning**: Grant access but log anomaly
- **Require remediation**: Prompt user to fix issues before access

**Example policy**: Only allow access to production training data from:
- Corporate-managed Chromebooks or laptops
- Encrypted devices with latest OS patches
- Devices located in approved countries
- Connections from corporate IP ranges or specific remote locations

---

## Section 3: Identity-Aware Proxy (IAP) (~150 lines)

### What is IAP?

From [Medium - Zero Trust Architecture in GCP](https://medium.com/google-cloud/zero-trust-architecture-in-gcp-9b8ec1bbf578) (accessed 2025-02-03):

Identity-Aware Proxy (IAP) protects GCP-hosted applications by verifying user identity and context **before** granting access. It works at the application layer (Layer 7), providing fine-grained access control without requiring a VPN.

**How IAP works**:
1. User requests access to protected resource (e.g., Vertex AI Workbench)
2. IAP checks if user has valid session (OAuth 2.0 token)
3. If not authenticated, redirects to Google Sign-In
4. After authentication, IAP checks IAM permissions
5. If authorized, IAP forwards request to backend
6. Backend receives request with user identity in header (`X-Goog-Authenticated-User-Email`)

**Key features**:
- Centralized authentication using Google Identity
- Granular permissions based on identity and context
- Integrated with GCP logging and monitoring
- No application code changes required

### IAP for Different GCP Services

From [Medium - Zero Trust Architecture in GCP](https://medium.com/google-cloud/zero-trust-architecture-in-gcp-9b8ec1bbf578) (accessed 2025-02-03):

**1. IAP for App Engine**

```bash
# Enable IAP for App Engine app
gcloud app services update default \
  --project=PROJECT_ID

# Grant IAP access to specific user
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="user:ml-engineer@company.com" \
  --role="roles/iap.httpsResourceAccessor"
```

**Traffic flow**:
```
User → IAP → App Engine app
     ↓ (if not authenticated)
     Google Sign-In → MFA → IAP → App
```

**2. IAP for Cloud Run**

```bash
# Deploy Cloud Run service with IAP
gcloud run deploy training-dashboard \
  --image=gcr.io/project/dashboard:latest \
  --ingress=internal-and-cloud-load-balancing \
  --no-allow-unauthenticated

# Create backend service with IAP
gcloud compute backend-services create training-backend \
  --global \
  --iap=enabled

# Add Cloud Run NEG to backend
gcloud compute backend-services add-backend training-backend \
  --global \
  --network-endpoint-group=training-neg \
  --network-endpoint-group-region=us-central1
```

**3. IAP for Compute Engine**

```bash
# Enable IAP for VM instances behind load balancer
gcloud compute backend-services update web-backend \
  --global \
  --iap=enabled,oauth2-client-id=CLIENT_ID,oauth2-client-secret=SECRET

# Allow IAP traffic to VMs (firewall rule)
gcloud compute firewall-rules create allow-iap-traffic \
  --direction=INGRESS \
  --action=ALLOW \
  --rules=tcp:22,tcp:3389,tcp:80,tcp:443 \
  --source-ranges=35.235.240.0/20 \
  --target-tags=iap-protected
```

**4. IAP for GKE (Kubernetes)**

```yaml
# GKE Ingress with IAP
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: training-ingress
  annotations:
    kubernetes.io/ingress.class: "gce"
    kubernetes.io/ingress.global-static-ip-name: "training-ip"
spec:
  rules:
  - host: training.example.com
    http:
      paths:
      - path: /*
        pathType: ImplementationSpecific
        backend:
          service:
            name: training-service
            port:
              number: 8080
```

Then enable IAP on the backend service created by the Ingress.

**5. IAP for On-Premises Applications**

From [Medium - Zero Trust Architecture in GCP](https://medium.com/google-cloud/zero-trust-architecture-in-gcp-9b8ec1bbf578) (accessed 2025-02-03):

```
On-Premise App (HTTP/HTTPS)
    ↑
BeyondCorp Enterprise Connector (VM in on-prem or GCP)
    ↑
Google Front End (GFE)
    ↑
Identity-Aware Proxy
    ↑
User (anywhere)
```

**Setup**:
1. Install BeyondCorp Enterprise connector on VM
2. Configure connector to proxy traffic to on-premise app
3. Enable IAP on the connector's backend service
4. Users access app via IAP-protected URL (no VPN needed)

### IAP with Context-Aware Access

From [Cloud4C - Zero Trust Security with Google Cloud](https://www.cloud4c.com/blogs/implementing-zero-trust-security-with-google-cloud) (accessed 2025-02-03):

Combine IAP with Access Context Manager for dynamic, context-aware policies:

```bash
# Create access level (requires corporate IP or managed device)
gcloud access-context-manager levels create high_trust \
  --policy=POLICY_ID \
  --title="High Trust Access" \
  --basic-level-spec=access_level.yaml

# access_level.yaml
conditions:
  - ipSubnetworks:
    - 203.0.113.0/24  # Corporate IP range
    devicePolicy:
      requireScreenlock: true
      osConstraints:
        - osType: DESKTOP_CHROME_OS
          minimumVersion: "100.0.0"
```

Apply to IAP:

```bash
# Grant IAP access only if user meets access level
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="user:ml-engineer@company.com" \
  --role="roles/iap.httpsResourceAccessor" \
  --condition="
    expression=request.auth.claims.access_levels.exists(al, al == 'accessPolicies/POLICY_ID/accessLevels/high_trust'),
    title=require-high-trust-access
  "
```

**Result**: ML engineer can only access training dashboard from:
- Corporate office network OR
- Managed Chromebook with screen lock enabled

---

## Section 4: Context-Aware Access (~100 lines)

### What is Context-Aware Access?

From [Medium - Zero Trust Architecture in GCP](https://medium.com/google-cloud/zero-trust-architecture-in-gcp-9b8ec1bbf578) (accessed 2025-02-03):

Context-Aware Access (CAA) lets you define fine-grained access policies based on **dynamic attributes** beyond basic user identity. It acts as a central policy engine for evaluating requests based on:

**User attributes**:
- Identity (email, group membership)
- Authentication method (password, MFA, security key)
- Session duration

**Device attributes**:
- Operating system and version
- Encryption status
- Screen lock enabled
- Corporate management enrollment
- Security patch level

**Network attributes**:
- Source IP address or IP range
- Geographic location
- Network trust level

**Request attributes**:
- Time of day
- Resource being accessed
- Sensitivity level

### Access Levels

From [Cloud4C - Zero Trust Security with Google Cloud](https://www.cloud4c.com/blogs/implementing-zero-trust-security-with-google-cloud) (accessed 2025-02-03):

Access levels combine multiple conditions to define "who can access what under which circumstances."

**Example 1: Basic Access Level**

```yaml
# Employees from office or VPN
name: accessPolicies/POLICY_ID/accessLevels/office_access
title: "Office Access"
basic:
  conditions:
    - ipSubnetworks:
      - 203.0.113.0/24  # Office IP range
      - 198.51.100.0/24  # VPN IP range
```

**Example 2: Device Posture Access Level**

```yaml
# Managed devices with encryption
name: accessPolicies/POLICY_ID/accessLevels/managed_device
title: "Managed Device"
basic:
  conditions:
    - devicePolicy:
        requireScreenlock: true
        requireCorpOwned: true
        osConstraints:
          - osType: DESKTOP_CHROME_OS
            minimumVersion: "100.0.0"
          - osType: DESKTOP_WINDOWS
            minimumVersion: "10.0.19041"  # Windows 10 20H1
        requireAdminApproval: true
```

**Example 3: Combined Access Level**

```yaml
# High trust: Managed device + Office location + MFA
name: accessPolicies/POLICY_ID/accessLevels/high_trust
title: "High Trust Access"
basic:
  combiningFunction: AND
  conditions:
    - ipSubnetworks:
      - 203.0.113.0/24
    - devicePolicy:
        requireScreenlock: true
        requireCorpOwned: true
    - members:
      - user:*@company.com
```

### Applying Context-Aware Access to Resources

From [Cloud4C - Zero Trust Security with Google Cloud](https://www.cloud4c.com/blogs/implementing-zero-trust-security-with-google-cloud) (accessed 2025-02-03):

**1. IAP with Access Levels**

```bash
# Allow access to training workbench only from high trust contexts
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="user:ml-engineer@company.com" \
  --role="roles/iap.httpsResourceAccessor" \
  --condition="
    expression=
      request.auth.claims.access_levels.exists(
        al, al == 'accessPolicies/POLICY_ID/accessLevels/high_trust'
      ),
    title=require-high-trust
  "
```

**2. VPC Service Controls with Access Levels**

```yaml
# Create service perimeter protecting training data
name: accessPolicies/POLICY_ID/servicePerimeters/training_perimeter
title: "Training Data Perimeter"
status:
  resources:
    - projects/123456789
  restrictedServices:
    - storage.googleapis.com
    - bigquery.googleapis.com
  accessLevels:
    - accessPolicies/POLICY_ID/accessLevels/high_trust
  vpcAccessibleServices:
    enableRestriction: true
    allowedServices:
      - storage.googleapis.com
```

**Result**: Training data (Cloud Storage, BigQuery) only accessible from high trust contexts.

### Use Cases for Training Workloads

From [Medium - Zero Trust Architecture in GCP](https://medium.com/google-cloud/zero-trust-architecture-in-gcp-9b8ec1bbf578) (accessed 2025-02-03):

**1. Production Data Access**
- **Requirement**: Only managed devices from approved locations
- **Access Level**: Corporate IP + Encrypted device + Screen lock
- **Applied to**: Cloud Storage buckets with production training data

**2. Model Registry Access**
- **Requirement**: MFA + Device encryption
- **Access Level**: Security key authentication + Managed device
- **Applied to**: Artifact Registry (model artifacts)

**3. Hyperparameter Tuning**
- **Requirement**: Authenticated users from any location (remote work)
- **Access Level**: Valid Google Identity + MFA
- **Applied to**: Vertex AI Training jobs

**4. Emergency Access**
- **Requirement**: On-call engineer from personal device (incident response)
- **Access Level**: Security key + Approval workflow + Time-bound (4 hours)
- **Applied to**: Cloud Logging, Cloud Monitoring (read-only)

---

## Section 5: Security Command Center Integration (~50 lines)

### Security Command Center Overview

From [Medium - Zero Trust Architecture in GCP](https://medium.com/google-cloud/zero-trust-architecture-in-gcp-9b8ec1bbf578) (accessed 2025-02-03):

Security Command Center (SCC) is Google Cloud's centralized security and risk management platform. It provides:

**1. Asset Discovery**
- Inventory all GCP resources (VMs, buckets, databases, IAM principals)
- Continuous monitoring for new/changed resources

**2. Vulnerability Detection**
- Web Security Scanner (finds vulnerabilities in App Engine, Compute Engine apps)
- Container Scanning (detect vulnerabilities in container images)
- OS patch management findings

**3. Threat Detection**
- Event Threat Detection (suspicious activity in Cloud Logging)
- Anomaly Detection (unusual API calls, data exfiltration attempts)
- Malware Detection (Cloud Storage files)

**4. Compliance Monitoring**
- CIS GCP Foundation Benchmark
- PCI-DSS, HIPAA, ISO 27001 controls
- Custom compliance standards

**5. Risk Assessment**
- Security Health Analytics (misconfigurations)
- Resource exposure (public buckets, open firewall rules)
- IAM over-privileged principals

### SCC Integration with Zero Trust

**1. Identity-Aware Proxy Monitoring**

```bash
# Query IAP access logs
gcloud logging read \
  'protoPayload.serviceName="iap.googleapis.com"' \
  --limit=50 \
  --format=json

# Create alert for failed IAP access attempts
gcloud alpha logging sinks create iap-failed-access \
  --log-filter='
    protoPayload.serviceName="iap.googleapis.com"
    AND protoPayload.status.code!=0
  ' \
  --destination=pubsub.googleapis.com/projects/PROJECT/topics/security-alerts
```

**2. Context-Aware Access Violations**

SCC detects:
- Users accessing resources without meeting access level requirements
- Devices failing endpoint verification checks
- Suspicious login patterns (impossible travel, new device)

**3. VPC Service Controls Violations**

SCC alerts on:
- Attempts to access perimeter-protected resources from outside
- Data exfiltration attempts (copying data to external buckets)
- Unauthorized API calls

### Remediation with Security Command Center

From [Medium - Zero Trust Architecture in GCP](https://medium.com/google-cloud/zero-trust-architecture-in-gcp-9b8ec1bbf578) (accessed 2025-02-03):

**1. Automated Response**

```python
# Cloud Function triggered by SCC finding
def remediate_public_bucket(finding):
    """Remove public access from bucket when SCC detects it"""
    if finding['category'] == 'PUBLIC_BUCKET_ACL':
        bucket_name = finding['resourceName']
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Remove allUsers and allAuthenticatedUsers
        policy = bucket.get_iam_policy()
        policy.bindings = [
            b for b in policy.bindings
            if b['members'] not in ['allUsers', 'allAuthenticatedUsers']
        ]
        bucket.set_iam_policy(policy)

        print(f"Removed public access from {bucket_name}")
```

**2. Manual Remediation Workflows**

SCC provides step-by-step remediation guidance:
- Revoke over-privileged IAM roles
- Enable MFA for users
- Patch vulnerable VMs
- Update firewall rules

---

## Sources

**Source Documents:**
- [gcloud-production/03-iam-advanced.md](../gcloud-production/03-iam-advanced.md) - Workload Identity Federation, IAM Conditions, Organization Policies
- [gcloud-production/00-networking.md](../gcloud-production/00-networking.md) - VPC Service Controls, Private Google Access, Network Security

**Web Research:**

From Medium (accessed 2025-02-03):
- [Zero Trust Architecture in GCP](https://medium.com/google-cloud/zero-trust-architecture-in-gcp-9b8ec1bbf578) by RahulRaghav - Comprehensive overview of Zero Trust concepts, BeyondCorp implementation, IAP setup, Access Context Manager, Security Command Center integration

From Cloud4C (accessed 2025-02-03):
- [Implementing Zero Trust Security with Google Cloud](https://www.cloud4c.com/blogs/implementing-zero-trust-security-with-google-cloud) - Zero Trust principles, BeyondCorp Enterprise features, practical implementation patterns, use cases for different scenarios

From Promevo (accessed 2025-02-03):
- [Implementing Zero Trust with Google](https://promevo.com/blog/implementing-zero-trust-with-google) - BeyondCorp Enterprise components, benefits and challenges, key Google products, device trust with endpoint verification

**Additional References:**
- BeyondCorp Enterprise: https://cloud.google.com/beyondcorp-enterprise
- Identity-Aware Proxy: https://cloud.google.com/iap
- Access Context Manager: https://cloud.google.com/access-context-manager
- Security Command Center: https://cloud.google.com/security-command-center

---

## Quick Reference Commands

```bash
# Enable IAP for App Engine
gcloud app services update default --project=PROJECT_ID

# Grant IAP access
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="user:engineer@company.com" \
  --role="roles/iap.httpsResourceAccessor"

# Create access level
gcloud access-context-manager levels create LEVEL_NAME \
  --policy=POLICY_ID \
  --title="Access Level Title" \
  --basic-level-spec=access_level.yaml

# Enable IAP for Compute Engine backend
gcloud compute backend-services update BACKEND_NAME \
  --global \
  --iap=enabled,oauth2-client-id=ID,oauth2-client-secret=SECRET

# Query IAP logs
gcloud logging read \
  'protoPayload.serviceName="iap.googleapis.com"' \
  --limit=50 \
  --format=json

# Create VPC Service Controls perimeter
gcloud access-context-manager perimeters create PERIMETER_NAME \
  --policy=POLICY_ID \
  --resources=projects/PROJECT_NUMBER \
  --restricted-services=storage.googleapis.com,bigquery.googleapis.com \
  --access-levels=LEVEL_NAME
```
