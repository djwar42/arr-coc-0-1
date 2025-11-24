# KNOWLEDGE DROP: GPU Resource Quotas & Governance Policies

**Dropped**: 2025-11-16
**Part**: PART 19
**File Created**: `gcp-gpu/18-gpu-quotas-governance-policies.md` (~750 lines)
**Batch**: 5/6 (Cost Optimization & Management)

---

## What Was Created

Comprehensive guide to GPU resource governance and quota management on GCP:

**8 Major Sections:**
1. Organization Policies for GPU Resources (GPU type restrictions, regional constraints, custom constraints)
2. Project-Level Quota Allocation (quota structure, team allocations, monitoring/alerting)
3. IAM Policies for GPU Resources (role-based access, conditional IAM)
4. Chargeback and Cost Allocation (label-based allocation, cost center reporting)
5. Approval Workflows for High-Cost GPU Requests (multi-stage approvals, IaC integration)
6. Resource Hierarchy Governance (org → folder → project structure)
7. arr-coc-0-1 Governance Model (project-specific implementation)
8. Best Practices and Common Patterns (governance checklist, cost control)

**Key Topics Covered:**
- Organization Policy constraints (GPU type restrictions, regional limits)
- Custom constraints with CEL (Common Expression Language)
- Project-level quota allocation across teams
- Quota monitoring and alerting (80% threshold alerts)
- IAM role-based and conditional access control
- Label-based cost allocation and chargeback
- BigQuery billing export for cost analysis
- Multi-stage approval workflows (self-service → team lead → director → executive)
- Cloud Functions for approval automation
- Resource hierarchy governance (organization, folders, projects)
- arr-coc-0-1 specific governance configuration

---

## Key Insights

### 1. GPU Quotas Start at Zero (Critical Discovery)

From web research (Medium article, 2023):
> "All cloud providers (GCP, AWS, Azure) start with default GPU quota of zero."

**Implication**: You MUST request quota increases before provisioning any GPU resources. This is intentional cost protection.

### 2. Organization Policy Inheritance

**Policies inherit DOWN the hierarchy:**
```
Organization Policy (Require cost labels)
  ↓ Inherited by folders
  ↓ Inherited by projects
```

**Quotas do NOT inherit** - they are per-project allocations.

### 3. Multi-Tier Approval Pattern

From Rafay documentation (June 2025):
> "The Org Admin is responsible for partitioning the organization's total GPU quota across internal teams and projects."

**Approval tiers:**
- < $100/week: Self-service (auto-approved)
- $100-500/week: Team lead approval
- $500-2000/week: Director + Finance approval
- > $2000/week: Executive + CFO approval

### 4. Label-Based Chargeback is Essential

**Required labels for GPU instances:**
- `cost-center`: Departmental chargeback
- `project-code`: Project-level tracking
- `environment`: dev/staging/prod budgeting
- `owner`: Individual accountability
- `gpu-approved-by`: Approval audit trail

### 5. Conditional IAM for Cost Control

**Examples:**
- Business hours only: `request.time.getHours() >= 9 && request.time.getHours() < 17`
- T4-only access for interns: `resource.acceleratorType.contains("nvidia-tesla-t4")`
- Time-limited access: `request.time < timestamp("2025-12-31T00:00:00Z")`

---

## Web Research Summary

### Sources Used

1. **Google Cloud Organization Policy Constraints** (Official docs, 2025-11-16)
   - Organization policy structure and enforcement
   - Custom constraints with CEL
   - Policy inheritance down resource hierarchy

2. **Rafay Multi-Tenant GPU Quota Management** (June 27, 2025)
   - Org-level quota partitioning across teams
   - Multi-level allocation (org → project → user)
   - Best practices for multi-tenant environments

3. **Medium: Things Cloud Providers Don't Tell You About GPUs** (2023)
   - Default GPU quota = 0 (all providers)
   - Quota request justification patterns

4. **CloudZero: GCP Cost Monitoring** (April 9, 2024)
   - Budget alert configuration (50%, 90%, 100% thresholds)
   - Cost tracking best practices

5. **Cloud Security Alliance: GCP Organization Policy Guide** (March 12, 2024)
   - Organization policy use cases
   - Governance frameworks

6. **Google Cloud Labeling Resources** (Official docs, 2025-11-16)
   - Label-based cost allocation
   - Billing export to BigQuery

---

## Existing Knowledge Integration

### Referenced Files

**From gcp-vertex/18-compliance-governance-audit.md:**
- Organization Policy patterns
- Compliance certification workflows
- Audit logging for governance

**From gcloud-iam/00-service-accounts-ml-security.md:**
- IAM role-based access control
- Service account best practices
- Conditional IAM policies

### Cross-References

**Will be referenced by:**
- PART 17: GPU Cost Optimization (budget allocation, committed use discounts)
- PART 18: GPU Monitoring & Observability (quota usage metrics)
- PART 21: GPU Production Deployment (governance in prod)

**References:**
- PART 1: Compute Engine GPU Instances (quota basics)
- PART 2: GPU Quota Management & Regional Availability (quota structure)

---

## arr-coc-0-1 Application

### Project-Specific Governance

**Quota allocation:**
- Dev: 4x T4 GPUs (auto-approved, within quota)
- Staging: 2x L4 + 2x A100 GPUs (team lead approval)
- Prod: 8x A100 GPUs (director + finance approval)

**Required labels:**
```yaml
cost-center: "ml-research"
project-code: "arr-coc-0-1"
environment: "[dev|staging|prod]"
owner: "[user-email]"
```

**Approval workflow:**
- Dev instances: < $200/week, auto-shutdown after 48 hours
- Staging instances: < $500/week, 4-hour SLA
- Prod instances: < $2000/week, 24-hour SLA, requires justification

---

## Code Examples Provided

1. **Organization Policy YAML** (GPU type restrictions)
2. **Custom Constraint with CEL** (require approval labels)
3. **Quota monitoring script** (Python with Compute API)
4. **BigQuery cost allocation query** (chargeback by cost center)
5. **Python chargeback report generator** (monthly billing summary)
6. **Cloud Functions approval workflow** (HTTP endpoint for requests)
7. **Terraform with approval gates** (IaC with preconditions)
8. **Auto-shutdown idle GPU instances** (cost control automation)

All examples are production-ready with full error handling and documentation.

---

## Completion Checklist

- [✓] Section 1: Organization Policies (GPU type restrictions, regional constraints, custom constraints)
- [✓] Section 2: Project-Level Quota Allocation (quota structure, team allocation, monitoring)
- [✓] Section 3: IAM Policies for GPU Resources (RBAC, conditional IAM)
- [✓] Section 4: Chargeback and Cost Allocation (labels, BigQuery billing)
- [✓] Section 5: Approval Workflows (multi-stage, Cloud Functions automation)
- [✓] Section 6: Resource Hierarchy Governance (org/folder/project structure)
- [✓] Section 7: arr-coc-0-1 Governance Model (project-specific config)
- [✓] Section 8: Best Practices (governance checklist, cost control patterns)
- [✓] **CITED**: gcp-vertex/18-compliance-governance-audit.md
- [✓] **CITED**: gcloud-iam/00-service-accounts-ml-security.md
- [✓] **CITED**: Web sources (Google Cloud docs, Rafay, Medium, CloudZero)
- [✓] All sources include access dates and URLs

---

## File Stats

- **Lines**: ~750
- **Sections**: 8 major
- **Code Examples**: 25+
- **Web Citations**: 6
- **Internal Citations**: 2
- **Command Examples**: 50+

---

## Next Steps

**PART 20**: GPU Benchmarking & Performance Testing (~700 lines)
- MLPerf benchmarks
- NCCL performance tests
- Nsight profiling
- A/B testing GPU configurations

**Remaining in Batch 5**: 1 more runner (PART 20)
**Batch 6**: PARTs 21-24 (Production & Advanced Patterns)

---

*Knowledge successfully extracted and integrated into karpathy-deep-oracle.*
