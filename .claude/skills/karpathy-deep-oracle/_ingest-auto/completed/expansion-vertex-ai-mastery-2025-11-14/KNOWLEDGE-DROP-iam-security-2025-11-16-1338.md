# KNOWLEDGE DROP: IAM & Service Accounts Security

**Runner**: PART 16 executor
**Date**: 2025-11-16 13:38
**Status**: SUCCESS

---

## File Created

**gcp-vertex/15-iam-service-accounts-security.md** (~700 lines)

Comprehensive Vertex AI IAM and security knowledge covering:
- Predefined roles (aiplatform.admin, aiplatform.user, custom roles)
- Service account best practices (1 per workload, least privilege)
- IAM conditions (time-based, resource-based, IP restrictions)
- Workload Identity Federation binding (Kubernetes SA → GCP SA)
- Cross-project service account usage
- Cloud Audit Logs (Admin Activity, Data Access, who did what when)
- arr-coc-0-1 security configuration

---

## Key Insights

### 1. Predefined Roles Hierarchy
- `roles/aiplatform.admin` - Full control (admin only, never for workloads)
- `roles/aiplatform.user` - Training/pipelines (most common for automated workflows)
- `roles/aiplatform.predictor` - Inference only (read-only, production serving)
- `roles/aiplatform.viewer` - Monitoring (dashboards, observability)

Custom roles enable fine-grained least privilege.

### 2. Service Account Segregation
One service account per workload type:
- `ml-training-prod-sa` - Production training
- `ml-inference-prod-sa` - Production serving
- `ml-pipeline-orchestrator-sa` - Pipeline coordination
- `ml-experiments-sa` - Experiment tracking

Never share service accounts across environments or functions.

### 3. IAM Conditions Enable Context-Aware Access
**Time-based:**
- Temporary elevated access (expires on date)
- Business hours only access (9am-5pm UTC)
- Maintenance window permissions

**Resource-based:**
- Region restrictions (us-central1 only)
- Environment-specific access (production label filter)
- Bucket name pattern matching

**IP restrictions:**
- NOT supported in IAM conditions for service accounts
- Use VPC Service Controls instead

### 4. Workload Identity Federation (Keyless GKE)
Replaces service account keys with KSA → GSA binding:

```
Kubernetes Service Account (KSA)
        ↓ (iam.gke.io/gcp-service-account annotation)
Google Service Account (GSA)
        ↓ (IAM roles)
Vertex AI API
```

No keys to manage, automatic rotation, fine-grained per-pod identity.

### 5. Cloud Audit Logs Track Who Did What
**Admin Activity Logs** (always enabled, free):
- Training job creation
- Model deployment
- Endpoint updates

**Data Access Logs** (disabled by default, billable):
- Training data reads
- Model predictions
- Checkpoint writes

Export to BigQuery for compliance analysis (7-year retention for HIPAA).

### 6. Cross-Project Permissions
Training in Project A can access data in Project B:
1. Create service account in Project A
2. Grant GCS bucket permissions in Project B to Project A's service account
3. Launch training job with cross-project data paths

Service account impersonation enables temporary developer access.

### 7. arr-coc-0-1 Security Pattern
Separate service accounts:
- `arr-coc-training-sa` - Training jobs (read training data, write checkpoints/models)
- `arr-coc-inference-sa` - Inference endpoints (read models only, NO training data access)

Comprehensive audit logging:
- Vertex AI (Admin + Data Access)
- Cloud Storage (Admin + Data Access)
- Secret Manager (Admin Read for W&B keys)

---

## Web Research Sources

**Google Cloud Official:**
- Vertex AI access control with IAM (accessed 2025-11-16)
- IAM Conditions attribute reference (accessed 2025-11-16)
- Workload Identity Federation for GKE (August 30, 2024)
- Cloud Audit Logs overview (accessed 2025-11-16)

**Industry Research:**
- Stack Overflow: IP restrictions not supported for service accounts (accessed 2025-11-16)
- CyberArk: GKE Workload Identity patterns (August 2024)
- DoiT: Workload Identity naming changes (May 2024)
- Medium: Audit log importance (October 2024)

**Security Updates:**
- Google auto-disables leaked service account keys (May 2024)
- Workload Identity Federation renamed from "Workload Identity" (May 2024)

---

## Citations

**Source Documents:**
- [gcloud-iam/00-service-accounts-ml-security.md](../gcloud-iam/00-service-accounts-ml-security.md)

**Web Resources:**
- https://docs.cloud.google.com/vertex-ai/docs/general/access-control
- https://docs.cloud.google.com/iam/docs/conditions-overview
- https://docs.cloud.google.com/kubernetes-engine/docs/concepts/workload-identity
- https://docs.cloud.google.com/logging/docs/audit

**GitHub Examples:**
- salrashid123/k8s_federation_with_gcp - Workload Identity implementation

---

## Integration Points

**Connects to existing knowledge:**
- gcloud-iam/00-service-accounts-ml-security.md - IAM fundamentals
- practical-implementation/30-vertex-ai-fundamentals.md - Basic Vertex AI setup
- practical-implementation/69-gke-autopilot-ml-workloads.md - Workload Identity on GKE

**Extends knowledge with:**
- IAM conditions (time-based, resource-based constraints)
- Workload Identity Federation step-by-step setup
- Cloud Audit Logs (Admin Activity vs Data Access)
- arr-coc-0-1 specific security configuration

---

## Production Patterns Extracted

### Pattern 1: Multi-Environment Service Account Strategy
```
ml-{workload}-{environment}-sa pattern:
- ml-training-dev-sa
- ml-training-staging-sa
- ml-training-prod-sa
- ml-inference-prod-sa
```

### Pattern 2: Resource-Level IAM (Not Project-Level)
```
✅ GOOD: Bucket-specific permissions
gsutil iam ch serviceAccount:ml-sa@PROJECT.iam.gserviceaccount.com:objectViewer gs://specific-bucket

❌ BAD: Project-wide permissions
gcloud projects add-iam-policy-binding --role=roles/storage.admin
```

### Pattern 3: Workload Identity Setup Flow
```
1. Enable Workload Identity on GKE cluster
2. Create Google Service Account (GSA) with IAM roles
3. Create Kubernetes Service Account (KSA)
4. Bind KSA to GSA (workloadIdentityUser role)
5. Annotate KSA with GSA email
6. Use KSA in pod spec (no keys!)
```

### Pattern 4: Audit Log Monitoring Queries
```
# Training job creation
resource.type="aiplatform.googleapis.com/CustomJob"
protoPayload.methodName="CreateCustomJob"

# Permission denials
protoPayload.authorizationInfo.granted=false
```

---

## Next Steps

**For Oracle (Consolidation):**
- Add to INDEX.md under "GCP Vertex AI" section
- Cross-reference with gcloud-iam/00-service-accounts-ml-security.md
- Update arr-coc-0-1 documentation with security patterns

**For Users:**
- Implement service account segregation in production
- Enable Data Access audit logs for compliance
- Migrate GKE workloads to Workload Identity Federation
- Configure IAM conditions for time-based access

---

## File Stats

- **Lines**: ~700
- **Sections**: 8 major sections
- **Code examples**: 40+ bash/YAML/HCL/JSON snippets
- **Web sources**: 10+ official docs + industry articles
- **Cross-references**: 3 existing knowledge files
- **arr-coc-0-1 integration**: Production security configuration included

**PART 16 COMPLETE ✓**
