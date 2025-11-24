# GCloud Config: Secrets Management with Secret Manager

**Topic**: Advanced secrets management patterns using GCP Secret Manager - rotation automation, audit logging, access patterns, and security best practices
**Last Updated**: 2025-02-03
**Scope**: Production-grade secrets management for GCP environments

---

## Section 1: Secret Manager Architecture (~100 lines)

### Overview

From [cloud-build-advanced/02-security-scanning.md](../cloud-build-advanced/02-security-scanning.md):
> "Secret Manager is a secure and convenient storage system for API keys, passwords, certificates, and other sensitive data. Secret Manager provides a central place and single source of truth to manage, access, and audit secrets across Google Cloud."

Secret Manager provides a fully managed service for storing, managing, and accessing secrets as binary blobs or text strings. Unlike storing secrets in environment variables or configuration files, Secret Manager offers:
- Automatic encryption at rest and in transit
- Fine-grained IAM access control
- Audit logging of all access
- Secret versioning and rotation
- Regional and global replication options

### Core Components

**Secrets**:
- Container for secret versions
- Metadata: labels, replication policy, rotation config
- Access controlled via IAM policies

**Secret Versions**:
- Immutable snapshots of secret data
- States: ENABLED, DISABLED, DESTROYED
- Can pin to specific version or use "latest"

**Replication Policies**:
- Automatic: Google manages replication globally
- User-managed: Specify regions for data residency

### Secret Manager vs. Environment Variables

| Aspect | Secret Manager | Environment Variables |
|--------|---------------|----------------------|
| **Security** | Encrypted, audited, IAM-controlled | Visible in process listings, logs |
| **Rotation** | Automated with versioning | Manual, requires redeployment |
| **Audit Trail** | Full Cloud Audit Logs | No audit trail |
| **Access Control** | Fine-grained IAM policies | All-or-nothing with service account |
| **Cost** | $0.06 per 10k access operations | Free |

**When to Use Secret Manager**:
- Database passwords and API keys
- Certificates and private keys
- OAuth tokens and credentials
- Any secret requiring rotation
- Secrets accessed across multiple services

**When Environment Variables Suffice**:
- Non-sensitive configuration (e.g., feature flags)
- Public API endpoints
- Version numbers or build IDs
- Values that don't require audit trails

### Recent Updates (2024-2025)

From search results ([Google Cloud Secret Manager documentation](https://docs.cloud.google.com/secret-manager/docs/secret-rotation), accessed 2025-02-03):

**Rotation Schedules (GA)**: Secret Manager now supports native rotation schedules that send Pub/Sub notifications at configured intervals, enabling automated rotation workflows without custom Cloud Scheduler setup.

**Regional Secrets**: Secrets can now be restricted to specific regions for data residency compliance, with CMEK (Customer-Managed Encryption Keys) support for regional secrets.

---

## Section 2: Rotation Automation (~150 lines)

### Rotation Strategy

**Why Rotate Secrets**:
- Limit exposure window if secret is compromised
- Comply with security policies (e.g., 90-day rotation)
- Reduce impact of leaked secrets in logs or code
- Meet regulatory requirements (PCI DSS, HIPAA)

**Rotation Frequency Recommendations**:
- **Critical secrets** (production DB passwords): 30-90 days
- **API keys** (third-party services): 90-180 days
- **Service account keys**: 90 days (or use Workload Identity instead)
- **Certificates**: Before expiration (typically 365 days)

From [Tutorial: Rotating Service Account Keys using Secret Manager](https://engineering.premise.com/tutorial-rotating-service-account-keys-using-secret-manager-5f4dc7142d4b) (accessed 2025-02-03):
> "It is recommended to rotate your keys on a regular basis as this reduces the risk to your system in the case that a key is leaked. Although Google provides Google-Managed Key Pairs for rotation, GCP does not include support for the rotation of service account keys or updating keys in external locations such as Github."

### Native Rotation Schedules

**Setup Rotation Schedule**:
```bash
# Create secret with rotation schedule
gcloud secrets create my-database-password \
  --replication-policy="automatic" \
  --rotation-period="2592000s" \  # 30 days
  --next-rotation-time="2025-03-01T00:00:00Z" \
  --topics="projects/PROJECT_ID/topics/secret-rotation"
```

**Rotation Event Payload**:
```json
{
  "name": "projects/PROJECT_ID/secrets/my-database-password",
  "rotationTime": "2025-03-01T00:00:00.000Z",
  "rotationPeriod": "2592000s"
}
```

**Rotation Workflow**:
1. Secret Manager sends Pub/Sub notification at `next-rotation-time`
2. Cloud Function (or Cloud Run job) triggered by Pub/Sub message
3. Function generates new secret value
4. Function adds new secret version
5. Function updates consuming services with new secret
6. Function disables old secret version after validation
7. Old version destroyed after grace period (e.g., 7 days)

### Cloud Function Rotation Implementation

**Python Example** (Rotation Cloud Function):
```python
import base64
import json
from google.cloud import secretmanager
from google.cloud import pubsub_v1
import os

PROJECT_ID = os.environ['PROJECT_ID']
secret_client = secretmanager.SecretManagerServiceClient()

def rotate_secret(event, context):
    """
    Triggered by Pub/Sub message from Secret Manager rotation schedule.
    """
    # Decode Pub/Sub message
    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    rotation_event = json.loads(pubsub_message)

    secret_name = rotation_event['name']

    # Generate new secret value (example: random password)
    new_value = generate_secure_password(32)

    # Add new secret version
    parent = secret_name
    payload = new_value.encode('UTF-8')

    response = secret_client.add_secret_version(
        request={
            "parent": parent,
            "payload": {"data": payload}
        }
    )

    print(f"Added new version: {response.name}")

    # Update consuming services (database, APIs, etc.)
    update_consuming_services(secret_name, new_value)

    # Disable old versions (keep last 2 versions enabled for rollback)
    disable_old_versions(secret_name, keep_count=2)

    return "Rotation complete"

def generate_secure_password(length):
    """Generate cryptographically secure random password."""
    import secrets
    import string

    alphabet = string.ascii_letters + string.digits + string.punctuation
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def update_consuming_services(secret_name, new_value):
    """
    Update services that consume this secret.
    Example: Update Cloud SQL password, restart services, etc.
    """
    # Implementation depends on service type
    # - Cloud SQL: ALTER USER password
    # - APIs: Update API key registration
    # - Services: Trigger rolling restart
    pass

def disable_old_versions(secret_name, keep_count=2):
    """Disable old secret versions, keeping last N versions."""
    versions = secret_client.list_secret_versions(
        request={"parent": secret_name}
    )

    enabled_versions = [
        v for v in versions
        if v.state == secretmanager.SecretVersion.State.ENABLED
    ]

    # Sort by create time (newest first)
    enabled_versions.sort(key=lambda v: v.create_time, reverse=True)

    # Disable versions beyond keep_count
    for version in enabled_versions[keep_count:]:
        secret_client.disable_secret_version(
            request={"name": version.name}
        )
        print(f"Disabled old version: {version.name}")
```

**Deploy Rotation Function**:
```bash
gcloud functions deploy secret-rotator \
  --runtime=python39 \
  --trigger-topic=secret-rotation \
  --entry-point=rotate_secret \
  --set-env-vars=PROJECT_ID=$PROJECT_ID \
  --service-account=secret-rotator@$PROJECT_ID.iam.gserviceaccount.com \
  --timeout=540s
```

**Required IAM Permissions** (for rotation service account):
```bash
# Grant permissions to rotation service account
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:secret-rotator@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.admin"

# If updating Cloud SQL passwords
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:secret-rotator@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/cloudsql.admin"
```

### Service Account Key Rotation

From [Tutorial: Rotating Service Account Keys](https://engineering.premise.com/tutorial-rotating-service-account-keys-using-secret-manager-5f4dc7142d4b):

**Plugin-Based Rotation Architecture**:
```python
# plugins/github.py - Update GitHub secrets
class GithubPlugin:
    schema = {
        "GH_ORG_FIELD": "",
        "GH_REPO_FIELD": "",
        "SECRET_NAME_FIELD": "GCP_SA_KEY",
    }

    def rotate_key(self, service_account_email, new_key_json):
        """Update GitHub secret with new service account key."""
        # 1. Create new service account key
        # 2. Update GitHub secret via API
        # 3. Delete old service account key
        # 4. Store new key version in Secret Manager
        pass

# plugins/storage.py - Store in GCS bucket
class StoragePlugin:
    def rotate_key(self, service_account_email, new_key_json):
        """Store new service account key in GCS."""
        # 1. Create new key
        # 2. Upload to GCS bucket
        # 3. Delete old key from GCS
        # 4. Update Secret Manager version
        pass
```

**Key Rotation Best Practices**:
- Rotate service account keys every 90 days
- Keep 2 valid keys during rotation (for zero-downtime)
- Delete old keys after 7-day grace period
- Use Workload Identity instead of keys when possible
- Never commit keys to version control
- Use Secret Manager labels to track key age

---

## Section 3: Audit Logging (~100 lines)

### Cloud Audit Logs Integration

From [Secret Manager Audit Logging documentation](https://docs.cloud.google.com/secret-manager/docs/audit-logging) (accessed 2025-02-03):
> "Google Cloud services generate audit logs that record administrative and access activities within your Google Cloud resources. Secret Manager audit logs include Admin Activity logs and Data Access logs."

**Audit Log Types**:

**Admin Activity Logs** (always enabled, no charge):
- Creating/deleting secrets
- Adding/disabling secret versions
- Updating IAM policies
- Modifying replication settings

**Data Access Logs** (must enable, charged):
- Reading secret versions (`AccessSecretVersion`)
- Listing secrets and versions
- Who accessed which secret, when, from where

### Enable Data Access Logging

**Enable for Secret Manager**:
```bash
# Create audit config file
cat > audit-config.yaml <<EOF
auditConfigs:
- service: secretmanager.googleapis.com
  auditLogConfigs:
  - logType: DATA_READ
  - logType: DATA_WRITE
  - logType: ADMIN_READ
EOF

# Apply audit config
gcloud projects set-iam-policy PROJECT_ID audit-config.yaml
```

**View Audit Logs**:
```bash
# Recent secret access
gcloud logging read "resource.type=secretmanager.googleapis.com/Secret" \
  --limit=50 \
  --format=json

# Specific secret access
gcloud logging read "resource.type=secretmanager.googleapis.com/Secret \
  AND protoPayload.resourceName=projects/PROJECT_ID/secrets/my-secret" \
  --limit=50
```

### Audit Log Analysis

**Query Patterns**:

**Who accessed a secret?**
```
resource.type="secretmanager.googleapis.com/Secret"
protoPayload.methodName="google.cloud.secretmanager.v1.SecretManagerService.AccessSecretVersion"
protoPayload.resourceName="projects/PROJECT_ID/secrets/my-secret"
```

**Failed access attempts**:
```
resource.type="secretmanager.googleapis.com/Secret"
protoPayload.status.code != 0
```

**Access from unexpected location**:
```
resource.type="secretmanager.googleapis.com/Secret"
protoPayload.methodName="AccessSecretVersion"
protoPayload.requestMetadata.callerIp != "10.0.0.0/8"
```

**Frequent access anomalies**:
```
resource.type="secretmanager.googleapis.com/Secret"
protoPayload.methodName="AccessSecretVersion"
| stats count by protoPayload.authenticationInfo.principalEmail
| where count > 1000
```

### Export Audit Logs to BigQuery

**Create Log Sink**:
```bash
# Create BigQuery dataset
bq mk --dataset --location=US secret_audit_logs

# Create log sink
gcloud logging sinks create secret-audit-sink \
  bigquery.googleapis.com/projects/$PROJECT_ID/datasets/secret_audit_logs \
  --log-filter='resource.type="secretmanager.googleapis.com/Secret"'
```

**Analyze in BigQuery**:
```sql
-- Top secret accessors
SELECT
  protopayload_auditlog.authenticationInfo.principalEmail AS user,
  resource.labels.secret_id,
  COUNT(*) AS access_count
FROM `project.secret_audit_logs.cloudaudit_googleapis_com_data_access_*`
WHERE protopayload_auditlog.methodName =
  'google.cloud.secretmanager.v1.SecretManagerService.AccessSecretVersion'
  AND _TABLE_SUFFIX BETWEEN '20250101' AND '20250131'
GROUP BY user, secret_id
ORDER BY access_count DESC
LIMIT 50;

-- Suspicious access times (non-business hours)
SELECT
  protopayload_auditlog.authenticationInfo.principalEmail AS user,
  resource.labels.secret_id,
  timestamp,
  protopayload_auditlog.requestMetadata.callerIp AS source_ip
FROM `project.secret_audit_logs.cloudaudit_googleapis_com_data_access_*`
WHERE protopayload_auditlog.methodName = 'AccessSecretVersion'
  AND (EXTRACT(HOUR FROM timestamp) < 6 OR EXTRACT(HOUR FROM timestamp) > 20)
  AND EXTRACT(DAYOFWEEK FROM timestamp) IN (1, 7)  -- Weekends
ORDER BY timestamp DESC;
```

### Audit Alerts with Cloud Monitoring

**Alert on Secret Access Failures**:
```yaml
# alert-policy.yaml
displayName: "Secret Access Denied"
conditions:
  - displayName: "Failed secret access"
    conditionThreshold:
      filter: 'resource.type="secretmanager.googleapis.com/Secret" AND protoPayload.status.code!=0'
      comparison: COMPARISON_GT
      thresholdValue: 5
      duration: 300s
notificationChannels:
  - projects/PROJECT_ID/notificationChannels/CHANNEL_ID
```

```bash
gcloud alpha monitoring policies create --policy-from-file=alert-policy.yaml
```

---

## Section 4: Access Patterns and Best Practices (~100 lines)

### IAM Access Control

**Principle of Least Privilege**:

From [cloud-build-advanced/02-security-scanning.md](../cloud-build-advanced/02-security-scanning.md) (Section 5):
> "Grant minimum required permissions only. Use conditions on IAM bindings (time-based, IP-based). Enable Cloud Audit Logs for IAM changes."

**Secret-Specific Roles**:
- `roles/secretmanager.secretAccessor` - Read secret versions only
- `roles/secretmanager.secretVersionManager` - Add/disable versions
- `roles/secretmanager.admin` - Full control (use sparingly)

**Grant Access to Specific Secret**:
```bash
# Allow service account to read specific secret
gcloud secrets add-iam-policy-binding my-database-password \
  --member="serviceAccount:app@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

**Conditional Access** (time-based):
```bash
gcloud secrets add-iam-policy-binding my-secret \
  --member="serviceAccount:temp@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" \
  --condition='expression=request.time < timestamp("2025-12-31T23:59:59Z"),title=temporary-access'
```

### Access Patterns by Service

**Cloud Build** (from [cloud-build-advanced/02-security-scanning.md](../cloud-build-advanced/02-security-scanning.md)):
```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        docker build \
          --build-arg DB_PASSWORD=$$DB_PASSWORD \
          -t my-app .
    secretEnv:
      - 'DB_PASSWORD'

availableSecrets:
  secretManager:
    - versionName: projects/${PROJECT_ID}/secrets/db-password/versions/latest
      env: 'DB_PASSWORD'
```

**CRITICAL**: Use double `$$` for secrets in Cloud Build (bash env var, not substitution).

**Cloud Run**:
```bash
# Mount secret as environment variable
gcloud run deploy my-service \
  --image=gcr.io/PROJECT_ID/my-app \
  --set-secrets=DB_PASSWORD=db-password:latest

# Mount secret as volume file
gcloud run deploy my-service \
  --image=gcr.io/PROJECT_ID/my-app \
  --set-secrets=/secrets/db-password=db-password:latest
```

**Cloud Functions**:
```bash
gcloud functions deploy my-function \
  --runtime=python39 \
  --set-secrets=DB_PASSWORD=db-password:latest
```

**Python Application**:
```python
from google.cloud import secretmanager

def access_secret(project_id, secret_id, version_id="latest"):
    """Access secret from Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()

    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})

    return response.payload.data.decode('UTF-8')

# Example usage
DB_PASSWORD = access_secret("my-project", "db-password")
```

### Secret Versioning Strategy

**Version Lifecycle**:
1. **ENABLED** - Active, accessible version
2. **DISABLED** - Temporarily unavailable (rollback safety)
3. **DESTROYED** - Permanently deleted (cannot recover)

**Best Practices**:
- Keep 2-3 recent versions ENABLED (rollback capability)
- DISABLE old versions after validation period (7 days)
- DESTROY versions after retention period (90 days)
- Use labels to track version metadata (e.g., `rotation_date`)

**Pin to Specific Version** (for stability):
```bash
# Instead of "latest"
gcloud run deploy my-service \
  --set-secrets=DB_PASSWORD=db-password:5
```

**Rollback to Previous Version**:
```bash
# Enable old version
gcloud secrets versions enable 4 --secret=db-password

# Disable current version
gcloud secrets versions disable 5 --secret=db-password
```

### Secret Naming Conventions

**Recommended Patterns**:
- `<service>-<environment>-<type>`: `api-prod-db-password`
- `<project>-<secret>`: `billing-service-api-key`
- Avoid: `secret1`, `password`, `key` (too generic)

**Labels for Organization**:
```bash
gcloud secrets create api-prod-db-password \
  --replication-policy="automatic" \
  --labels=env=production,service=api,type=database,team=backend
```

**Query by Label**:
```bash
gcloud secrets list --filter="labels.env=production AND labels.type=database"
```

### Security Best Practices Checklist

From [cloud-build-advanced/02-security-scanning.md](../cloud-build-advanced/02-security-scanning.md) (Section 5):

**Secret Management**:
- [ ] Use Secret Manager for all sensitive data
- [ ] Rotate secrets regularly (30-90 days)
- [ ] Enable Data Access logging for audit trails
- [ ] Grant minimum IAM permissions (secretAccessor only)
- [ ] Use Workload Identity instead of service account keys
- [ ] Never commit secrets to version control
- [ ] Destroy old secret versions after retention period
- [ ] Monitor unusual access patterns (alerts)
- [ ] Use secret versioning for rollback safety
- [ ] Label secrets for organization and cost tracking

**Rotation**:
- [ ] Configure automatic rotation schedules
- [ ] Test rotation procedures regularly
- [ ] Keep 2 versions enabled during rotation (zero-downtime)
- [ ] Validate consuming services after rotation
- [ ] Document rotation runbooks

**Access Control**:
- [ ] Use service accounts, not user accounts
- [ ] Grant access per-secret, not project-wide
- [ ] Use conditional IAM policies (time/IP restrictions)
- [ ] Review IAM policies quarterly
- [ ] Audit secret access logs monthly

**Monitoring**:
- [ ] Export audit logs to BigQuery for analysis
- [ ] Set up alerts for access failures
- [ ] Monitor access from unexpected IPs
- [ ] Track secret age and flag stale secrets
- [ ] Review access patterns for anomalies

---

## Section 5: Integration with Cloud Services (~50 lines)

### Cloud Build Integration

From [cloud-build-advanced/02-security-scanning.md](../cloud-build-advanced/02-security-scanning.md) (Section 4):

**Environment Variables** (recommended):
```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    entrypoint: 'bash'
    args:
      - '-c'
      - 'docker build --build-arg API_KEY=$$API_KEY -t my-app .'
    secretEnv: ['API_KEY']

availableSecrets:
  secretManager:
    - versionName: projects/${PROJECT_ID}/secrets/api-key/versions/latest
      env: 'API_KEY'
```

**Volume Mounts** (for file-based secrets):
```yaml
steps:
  - name: 'gcr.io/cloud-builders/gcloud'
    entrypoint: 'bash'
    args:
      - '-c'
      - 'gcloud auth activate-service-account --key-file=/secrets/sa-key.json'
    volumes:
      - name: 'secrets'
        path: '/secrets'

availableSecrets:
  secretManager:
    - versionName: projects/${PROJECT_ID}/secrets/sa-key/versions/latest
      env: 'SA_KEY_JSON'
```

### Cloud Run / Cloud Functions

**Runtime Injection** (preferred for ephemeral services):
```bash
# Cloud Run with secret as env var
gcloud run deploy my-service \
  --image=gcr.io/PROJECT_ID/my-app \
  --set-secrets=DB_PASSWORD=db-password:latest,API_KEY=api-key:latest

# Cloud Functions
gcloud functions deploy my-function \
  --runtime=python39 \
  --set-secrets=API_KEY=api-key:latest
```

**Benefits**:
- No secret in container image
- Automatic updates when secret rotates (if using "latest")
- IAM controls who can deploy with secrets

### Kubernetes / GKE Integration

**External Secrets Operator** (recommended):
```yaml
# ExternalSecret resource
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: db-credentials
spec:
  secretStoreRef:
    name: gcpsm-secret-store
    kind: SecretStore
  target:
    name: db-credentials-k8s
  data:
  - secretKey: password
    remoteRef:
      key: db-password
      version: latest
```

**Workload Identity** (for GKE access):
```bash
# Bind Kubernetes service account to GCP service account
gcloud iam service-accounts add-iam-policy-binding \
  gcp-sa@PROJECT_ID.iam.gserviceaccount.com \
  --member="serviceAccount:PROJECT_ID.svc.id.goog[NAMESPACE/KSA_NAME]" \
  --role="roles/iam.workloadIdentityUser"

# Grant secret access to GCP service account
gcloud secrets add-iam-policy-binding my-secret \
  --member="serviceAccount:gcp-sa@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

---

## Sources

**Source Documents**:
- [cloud-build-advanced/02-security-scanning.md](../cloud-build-advanced/02-security-scanning.md) - Section 4: Secret Manager Integration (lines 365-515)

**Google Cloud Documentation**:
- [Create Rotation Schedules in Secret Manager](https://docs.cloud.google.com/secret-manager/docs/secret-rotation) - Native rotation schedule configuration (accessed 2025-02-03)
- [Secret Manager Audit Logging](https://docs.cloud.google.com/secret-manager/docs/audit-logging) - Admin Activity and Data Access logs (accessed 2025-02-03)
- [Secret Manager Best Practices](https://docs.cloud.google.com/secret-manager/docs/best-practices) - Official best practices guide (accessed 2025-02-03)
- [Use Secrets from Secret Manager](https://docs.cloud.google.com/build/docs/securing-builds/use-secrets) - Cloud Build integration patterns (accessed 2025-02-03)

**Web Research**:
- [Tutorial: Rotating Service Account Keys using Secret Manager](https://engineering.premise.com/tutorial-rotating-service-account-keys-using-secret-manager-5f4dc7142d4b) - Premise Engineering, plugin-based rotation architecture (accessed 2025-02-03)
- [10 Best Practices For Cloud Secrets Management (2025 Guide)](https://beckcooper.medium.com/10-best-practices-for-cloud-secrets-management-2025-guide-ffed6858e76b) - Beck Cooper, Medium (accessed 2025-02-03)
- [How to Handle Secrets with Google Cloud Secret Manager](https://blog.gitguardian.com/how-to-handle-secrets-with-google-cloud-secret-manager/) - GitGuardian Blog (accessed 2025-02-03)
- [Using Secret Manager with Python](https://codelabs.developers.google.com/codelabs/secret-manager-python) - Google Codelabs (accessed 2025-02-03)
- [GCP Secret Manager: The Basics and a Quick Tutorial](https://configu.com/blog/gcp-secret-manager-the-basics-and-a-quick-tutorial/) - Configu (accessed 2025-02-03)

**Additional References**:
- [Secrets Management Simplified with Multi-Vault Integrations](https://blog.gitguardian.com/secrets-managers-integrations/) - GitGuardian, audit capabilities and access patterns (accessed 2025-02-03)
- [AWS Secrets Manager Monitor Documentation](https://docs.aws.amazon.com/secretsmanager/latest/userguide/monitoring.html) - Cross-cloud comparison (accessed 2025-02-03)
