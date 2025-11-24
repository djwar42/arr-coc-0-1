# KNOWLEDGE DROP: Secret Manager & Credentials (PART 18)

**Timestamp**: 2025-11-16 14:38
**Runner**: Worker executing PART 18
**Status**: ✓ SUCCESS

---

## What Was Created

**File**: `gcp-vertex/17-secret-manager-credentials.md` (~700 lines)

Comprehensive guide to Google Cloud Secret Manager integration with Vertex AI and GKE for secure credential management in ML workloads.

---

## Coverage Summary

### Section 1: Secret Manager API Fundamentals
- Core concepts (secrets vs versions, resource hierarchy)
- Creating secrets (gcloud CLI, Python SDK)
- Accessing secrets with caching patterns
- Managing versions (add, disable, destroy)
- Replication policies (automatic vs user-managed)

### Section 2: Environment Variable Injection for Vertex AI
- Runtime secret access patterns (avoid hardcoding)
- Service account IAM permissions (secretAccessor role)
- Multi-secret management (environment-aware loading)
- Principle of least privilege (per-secret access)

### Section 3: Kubernetes Secret Mounting (GKE Integration)
- Secret Manager add-on for GKE
- Workload Identity setup (K8s SA → GCP IAM)
- CSI driver volume mounts (secrets as files)
- Automatic rotation with polling (120s interval)
- SecretProviderClass configuration

### Section 4: Automatic Rotation Policies
- Rotation schedules (30d, 90d periods)
- Cloud Functions automation workflow
- Cloud Scheduler integration
- Grace period pattern (zero-downtime rotation)
- Pub/Sub notification system

### Section 5: Customer-Managed Encryption Keys (CMEK)
- Cloud KMS setup for Secret Manager
- Key creation and rotation (90d automatic)
- IAM bindings for service agents
- Key destruction (cryptographic proof of deletion)
- Compliance use cases (HIPAA, PCI-DSS)

### Section 6: Secret Access Audit Logs
- Enabling Data Access logs
- Querying logs (gcloud logging read)
- Audit log schema (principalEmail, resourceName, timestamp)
- Alerting on suspicious access (>100 accesses/5min)
- Log sinks to Pub/Sub

### Section 7: arr-coc-0-1 Credential Management
- W&B API key storage and access
- HuggingFace token authentication
- Database credentials (PostgreSQL JSON)
- Multi-environment secrets (prod/staging/dev)
- Complete training script integration

---

## Key Insights

**Secret Manager Integration Patterns:**
1. **Vertex AI**: Access secrets at runtime via Python SDK (not env vars)
2. **GKE**: Mount secrets as files via CSI driver + Workload Identity
3. **Rotation**: Automated with Cloud Functions + Scheduler + Pub/Sub
4. **Audit**: Full access logging with principalEmail tracking

**Best Practices Discovered:**
- Cache secrets locally to avoid API rate limits (1,800/min/project)
- Use CMEK for compliance (HIPAA, PCI-DSS requirements)
- Implement grace periods (24h) for zero-downtime rotation
- Alert on high access rates (>100/5min = potential leak)
- Organize by environment (prod-wandb-key, staging-wandb-key)

**arr-coc-0-1 Specific:**
- W&B Launch integration via secret API key access
- HuggingFace model downloads with authenticated token
- PostgreSQL training data access with JSON credentials
- Environment-aware SecretManager class (staging/prod)

---

## Citations

**Source Documents:**
- [30-vertex-ai-fundamentals.md](../karpathy/practical-implementation/30-vertex-ai-fundamentals.md) - Service account IAM roles, Vertex AI architecture

**Web Research (accessed 2025-11-16):**
- [Google Cloud Blog: Automatic Password Rotation](https://cloud.google.com/blog/products/identity-security/how-to-use-google-clouds-automatic-password-rotation)
- [Grigor Khachatryan: Get Started with Automatic Password Rotation](https://grigorkh.medium.com/get-started-with-automatic-password-rotation-on-google-cloud-fdf9243bfb44)
- [Madhav Sake: Manage Secrets in GKE Part 1](https://medium.com/ankercloud-engineering/manage-secrets-in-gke-part-1-using-secret-manager-add-on-6a8f0e5f5b2d)

**Search Queries:**
- "Secret Manager Vertex AI Custom Jobs 2024"
- "automatic secret rotation GCP"
- "CMEK customer-managed encryption keys GCP"
- "Kubernetes secrets GKE integration"

---

## Stats

- **Lines**: 712
- **Sections**: 7 major sections
- **Code examples**: 40+ Python/Bash snippets
- **Web sources**: 3 detailed articles scraped
- **Source documents**: 1 cited (30-vertex-ai-fundamentals.md)
- **Topics covered**: API, Vertex AI, GKE, rotation, CMEK, auditing, arr-coc-0-1

---

## Verification

**File exists**: ✓
**Proper structure**: ✓ (Overview, 7 sections, Key Takeaways, Sources)
**Citations included**: ✓ (source docs + web research with URLs + access dates)
**Code examples**: ✓ (gcloud, Python SDK, YAML, Cloud Functions)
**arr-coc-0-1 integration**: ✓ (Section 7 with complete examples)
**Expected length**: ✓ (712 lines, target was ~700)

---

## Next Steps

- Mark PART 18 as complete in ingestion.md
- Oracle will consolidate all KNOWLEDGE DROP files
- Oracle will update INDEX.md after batch completion
