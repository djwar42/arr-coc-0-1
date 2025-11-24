# GCP Production Troubleshooting - Workflows, Root Cause Analysis, and Incident Response

## Overview

Production troubleshooting in GCP requires systematic workflows that combine observability tools, root cause analysis techniques, and structured incident response. This guide covers battle-tested approaches for diagnosing production issues, conducting effective root cause analysis, and managing incidents following SRE best practices.

**Key Production Debugging Capabilities:**
- Systematic troubleshooting workflows for GCP services
- Root cause analysis using Cloud Trace and Cloud Logging
- Incident response frameworks (IMAG, PagerDuty)
- Production debugging tools and techniques
- Post-incident analysis and continuous improvement

## Section 1: Production Troubleshooting Workflows

### Systematic Troubleshooting Approach

From Cloud Build debugging patterns (cloud-build-advanced/04-debugging.md):

**Step-by-Step Debugging Process:**

```bash
# 1. Get recent activity
gcloud builds list --limit=10 --region=us-west2

# 2. Identify failed resource
BUILD_ID="abc-123-def"

# 3. Get resource details
gcloud builds describe $BUILD_ID --region=us-west2 --format=yaml

# 4. Check resource status
gcloud builds describe $BUILD_ID \
  --format="value(status,statusDetail,timeout,timing)"

# 5. Retrieve full logs
gcloud builds log $BUILD_ID > failure.log

# 6. Extract errors
grep -i "error\|fail\|exception" failure.log | head -50

# 7. Check timing
gcloud builds describe $BUILD_ID \
  --format="value(steps[].name,steps[].status,steps[].timing)"

# 8. Analyze which step failed
gcloud builds log $BUILD_ID | grep -A 10 "FAILURE"
```

**Generic Pattern for Any GCP Service:**

```bash
# 1. Identify the failing service
gcloud services list --enabled | grep SERVICE_NAME

# 2. Check recent operations
gcloud [SERVICE] operations list --limit=10

# 3. Describe specific failure
gcloud [SERVICE] operations describe OPERATION_ID

# 4. Retrieve logs
gcloud logging read "resource.type=[RESOURCE_TYPE]" \
  --limit=100 \
  --format=json

# 5. Check quotas
gcloud compute project-info describe --project=PROJECT_ID

# 6. Verify IAM permissions
gcloud projects get-iam-policy PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:SA_EMAIL"
```

### Troubleshooting Decision Tree

From [GCP operational excellence documentation](https://docs.cloud.google.com/architecture/framework/operational-excellence/manage-incidents-and-problems) (accessed 2025-02-03):

```
Production Issue Detected
    ↓
Is customer impact confirmed?
    ↓ YES                           ↓ NO
    ↓                               → Monitor and investigate
    ↓
Declare incident (see Section 3)
    ↓
Assess severity:
    - P0: Critical outage
    - P1: Major functionality broken
    - P2: Degraded performance
    - P3: Minor issue
    ↓
Assemble response team
    ↓
Execute troubleshooting workflow:
    1. Gather symptoms
    2. Check recent changes
    3. Review monitoring/logs
    4. Form hypotheses
    5. Test hypotheses
    6. Mitigate impact
    7. Identify root cause
    ↓
Resolve and document
```

### Common GCP Production Issues

| Issue Type | Symptoms | First Steps | Tools |
|------------|----------|-------------|-------|
| **Permission Denied** | 403 errors, "does not have permission" | Check IAM bindings, service account roles | `gcloud projects get-iam-policy` |
| **Quota Exceeded** | 429 errors, "quota exceeded" | Check quota usage, request increase | `gcloud compute project-info describe` |
| **Resource Exhausted** | OOM, disk full, CPU spike | Check resource metrics, scale up | Cloud Monitoring dashboards |
| **Network Issues** | Timeouts, connection refused | Check VPC, firewall rules, DNS | `gcloud compute networks describe` |
| **Service Unavailable** | 503 errors, health check failures | Check backend health, load balancer config | `gcloud compute backend-services get-health` |
| **Data Corruption** | Inconsistent reads, checksum errors | Check storage logs, replication status | Cloud Logging, GCS versioning |

### Health Check Script

```bash
#!/bin/bash
# gcp-health-check.sh

PROJECT_ID=$(gcloud config get-value project)
REGION="us-west2"

echo "GCP Health Check - $(date)"
echo "================================"
echo ""

# Check recent Cloud Build failures
echo "Recent Build Failures (last 24h):"
FAILED_COUNT=$(gcloud builds list \
  --filter="status=FAILURE AND createTime>$(date -u -d '24 hours ago' +%Y-%m-%dT%H:%M:%S)" \
  --region=$REGION \
  --format="value(id)" | wc -l)
echo "  Failed builds: $FAILED_COUNT"

# Check service account permissions
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
SA_EMAIL="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
echo ""
echo "Cloud Build Service Account: $SA_EMAIL"
gcloud projects get-iam-policy $PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:$SA_EMAIL" \
  --format="table(bindings.role)"

# Check quota usage
echo ""
echo "GPU Quota Usage (last 24h):"
gcloud compute project-info describe --project=$PROJECT_ID \
  | grep -A 5 "NVIDIA_T4"

# Check recent errors in logs
echo ""
echo "Recent Error Log Entries (last 1h):"
gcloud logging read "severity>=ERROR timestamp>=$(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S)" \
  --limit=10 \
  --format="table(timestamp,resource.type,textPayload)"
```

## Section 2: Root Cause Analysis

### Five Whys Method

From [Google SRE Workbook - Incident Response](https://sre.google/workbook/incident-response/) (accessed 2025-02-03):

**Basic Five Whys Process:**

```
Problem: Cloud Build timeout after 20 minutes

Why #1: Why did the build timeout?
→ Because the Docker image push took too long

Why #2: Why did the push take too long?
→ Because we're pushing 70+ layers to Artifact Registry

Why #3: Why are we pushing 70+ layers?
→ Because we're not using layer caching effectively

Why #4: Why isn't layer caching working?
→ Because the base image changes frequently

Why #5: Why does the base image change frequently?
→ Because we're not pinning the base image version

Root Cause: Unpinned base image versions cause cache invalidation
Solution: Pin base image versions, use multi-stage builds
```

**Advanced Five Whys with Multiple Root Causes:**

```
Problem: GKE cluster creation failures (Case Study 2 from SRE Workbook)

Why #1: Why is cluster creation failing?
→ Certificate signing request failed

Why #2: Why did the CSR fail?
→ Docker image pull failed

Why #3: Why did the image pull fail?
→ DockerHub image was corrupted

Why #4: Why was the image corrupted?
→ GCR mirror cached the corrupted image

Why #5: Why did GCR mirror cache corruption?
→ No validation on mirror cache updates

Root Causes (Multiple):
1. External dependency (DockerHub) without fallback
2. GCR mirror lacked corruption detection
3. No generic rollback mechanism for images

Solutions:
1. Host all critical images in GCR directly
2. Add checksum validation to mirror
3. Build generic image rollback tool
```

### Root Cause Analysis Using Cloud Trace and Cloud Logging

From [Google Cloud blog - Using Cloud Trace and Cloud Logging for root cause analysis](https://cloud.google.com/blog/products/devops-sre/using-cloud-trace-and-cloud-logging-for-root-cause-analysis) (accessed 2025-02-03):

**Integrated RCA Workflow:**

```bash
# 1. Identify slow request in Cloud Trace
# Navigate to Cloud Trace → Trace List
# Filter by latency: >5000ms

# 2. Get trace ID from slow request
TRACE_ID="abc123def456"

# 3. Correlate with Cloud Logging using trace ID
gcloud logging read "trace=projects/PROJECT_ID/traces/$TRACE_ID" \
  --format=json \
  --limit=100

# 4. Extract error context
gcloud logging read "trace=projects/PROJECT_ID/traces/$TRACE_ID severity>=ERROR" \
  --format="table(timestamp,resource.type,jsonPayload.message)"

# 5. Analyze span timeline
# Cloud Trace UI shows:
# - Which service caused delay
# - Database query performance
# - External API call latency
# - Network overhead

# 6. Drill into specific service logs
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=my-service AND timestamp>='2025-02-03T10:00:00Z'" \
  --limit=100
```

**Root Cause Analysis Template:**

```
Incident: [Brief description]
Date: [YYYY-MM-DD HH:MM]
Severity: [P0/P1/P2/P3]
Duration: [X hours Y minutes]

SYMPTOMS OBSERVED:
- User reports: [What users reported]
- Monitoring alerts: [Which alerts fired]
- Error rate: [X% increase]
- Latency: [Xms → Yms]

TIMELINE:
- HH:MM - First alert received
- HH:MM - Incident declared
- HH:MM - Mitigation applied
- HH:MM - Root cause identified
- HH:MM - Incident resolved

ROOT CAUSE:
[Primary root cause using Five Whys]

CONTRIBUTING FACTORS:
- Factor 1: [Description]
- Factor 2: [Description]

TRIGGER:
[What triggered the incident]

RESOLUTION:
[How the incident was resolved]

LESSONS LEARNED:
- What went well: [List]
- What didn't go well: [List]

ACTION ITEMS:
- [ ] Action 1 - Owner - Due Date
- [ ] Action 2 - Owner - Due Date
```

### Fishbone Diagram (Ishikawa) for Complex Issues

```
                          Problem: Production Outage
                                    |
        ┌──────────────────────────┴──────────────────────────┐
        |                          |                           |
    People                     Process                    Technology
        |                          |                           |
  - On-call                  - No runbook               - Bug in code
    unavailable              - Unclear                  - Dependency
  - Lack of                    escalation                 failure
    training                   path                     - No monitoring
        |                          |                           |
        └──────────────────────────┴──────────────────────────┘
                                    |
                          [Root Cause Analysis]
```

**Fishbone Categories (6 Ms):**
1. **Manpower** (People) - Training, availability, expertise
2. **Method** (Process) - Procedures, runbooks, escalation
3. **Machine** (Technology) - Infrastructure, tools, systems
4. **Material** (Data) - Data quality, configuration, dependencies
5. **Measurement** (Monitoring) - Observability, alerts, metrics
6. **Mother Nature** (Environment) - External factors, weather, power

## Section 3: Incident Response Best Practices

### Incident Command System (ICS) Framework

From [Google SRE Workbook - Incident Response](https://sre.google/workbook/incident-response/) (accessed 2025-02-03):

**Three Cs of Incident Management:**
1. **Coordinate** - Organize response efforts
2. **Communicate** - Internal and external stakeholders
3. **Control** - Maintain incident state and decisions

**Core Roles:**

**Incident Commander (IC):**
- Commands and coordinates incident response
- Delegates roles as needed
- Makes high-level decisions
- Stays focused on the 3Cs (not technical details)
- Assumes all undelegated roles by default

**Operations Lead (OL):**
- Works to mitigate/resolve the incident
- Applies operational tools
- Reports progress to IC
- May lead a team of responders

**Communications Lead (CL):**
- Public face of incident response
- Provides periodic updates
- Manages external inquiries
- Coordinates with PR/support teams

### Incident Response Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DETECT & DECLARE                                         │
│    - Monitor alerts fire                                    │
│    - Confirm customer impact                                │
│    - Declare incident early                                 │
│    - Assign Incident Commander                              │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. ASSESS IMPACT                                            │
│    - Determine scope (users affected, regions, services)    │
│    - Assign severity (P0/P1/P2/P3)                         │
│    - Establish communication channels                       │
│    - Page additional responders if needed                   │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. MITIGATE (Priority #1)                                   │
│    - Stop the bleeding (generic mitigations first)          │
│    - Rollback recent changes                                │
│    - Drain problematic resources                            │
│    - Scale up/down as needed                                │
│    - Don't wait for full root cause                         │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. TROUBLESHOOT & RESOLVE                                   │
│    - Gather symptoms and logs                               │
│    - Form hypotheses                                        │
│    - Test hypotheses systematically                         │
│    - Identify root cause                                    │
│    - Apply targeted fix                                     │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. COMMUNICATE & CLOSE                                      │
│    - Update stakeholders                                    │
│    - Announce resolution                                    │
│    - Hand off to on-call if needed                          │
│    - Start postmortem process                               │
└─────────────────────────────────────────────────────────────┘
```

### Incident Severity Levels

From [GCP operational excellence - Manage incidents](https://docs.cloud.google.com/architecture/framework/operational-excellence/manage-incidents-and-problems) (accessed 2025-02-03):

| Severity | Impact | Response Time | Example |
|----------|--------|---------------|---------|
| **P0** | Critical outage, major customer impact | Immediate | Service completely down, data loss |
| **P1** | Significant degradation | <15 minutes | Core functionality broken, 50%+ error rate |
| **P2** | Partial degradation | <1 hour | Non-critical feature broken, 10-50% errors |
| **P3** | Minor issues | <4 hours | UI bug, low error rate, workaround exists |
| **P4** | Cosmetic/future work | Best effort | Documentation error, enhancement request |

### Incident Response Runbook Template

```markdown
# [SERVICE_NAME] Incident Response Runbook

## Severity Classification

**P0 Criteria:**
- [ ] Service completely unavailable
- [ ] Data loss or corruption
- [ ] Security breach

**P1 Criteria:**
- [ ] Core functionality broken
- [ ] Error rate >50%
- [ ] Significant user impact

## Incident Declaration

**When to declare:**
- Confirmed customer impact
- Multiple alerts firing
- Issue persists >15 minutes

**How to declare:**
1. Post in #incidents Slack channel
2. Use template: "[SEVERITY] [SERVICE] - [BRIEF DESCRIPTION]"
3. Page Incident Commander on-call
4. Create incident tracking document

## Quick Diagnostics

**Health Checks:**
```bash
# Check service health
gcloud run services describe SERVICE_NAME \
  --region=us-west2 \
  --format="value(status.conditions)"

# Check error rate
gcloud monitoring time-series list \
  --filter='metric.type="run.googleapis.com/request_count"' \
  --interval-start-time=$(date -u -d '10 minutes ago' +%Y-%m-%dT%H:%M:%SZ)

# Check recent deployments
gcloud run revisions list \
  --service=SERVICE_NAME \
  --region=us-west2 \
  --limit=5
```

## Generic Mitigations

**Rollback (fastest mitigation):**
```bash
# List recent revisions
gcloud run revisions list --service=SERVICE_NAME --region=us-west2

# Rollback to previous revision
gcloud run services update-traffic SERVICE_NAME \
  --region=us-west2 \
  --to-revisions=REVISION_ID=100
```

**Scale Up:**
```bash
# Increase instances
gcloud run services update SERVICE_NAME \
  --region=us-west2 \
  --min-instances=10 \
  --max-instances=100
```

**Drain Region:**
```bash
# Route traffic away from problematic region
gcloud compute url-maps set-default-service URL_MAP \
  --default-service=BACKEND_SERVICE_OTHER_REGION
```

## Root Cause Investigation

**Logs:**
```bash
# Recent errors
gcloud logging read "resource.type=cloud_run_revision AND severity>=ERROR" \
  --limit=100 \
  --format=json

# Specific timeframe
gcloud logging read \
  "resource.type=cloud_run_revision AND timestamp>='2025-02-03T10:00:00Z' AND timestamp<='2025-02-03T11:00:00Z'" \
  --format=json
```

**Traces:**
- Navigate to Cloud Trace → Trace List
- Filter by latency or errors
- Correlate trace IDs with logs

**Metrics:**
- Cloud Monitoring → Dashboard
- Check: request_count, latency, error_rate
- Compare to baseline

## Communication Templates

**Initial Notification:**
```
🚨 INCIDENT: [SERVICE_NAME] - [SEVERITY]

Impact: [Brief description of user impact]
Status: Investigating
Started: [HH:MM UTC]
IC: @incident-commander
CL: @comms-lead

Updates every 30 minutes in this thread.
```

**Update:**
```
UPDATE [HH:MM UTC]:
- Mitigation applied: [Description]
- Current status: [Degraded/Improving/Resolved]
- Next update: [HH:MM UTC]
```

**Resolution:**
```
✅ RESOLVED [HH:MM UTC]:
- Root cause: [Brief description]
- Resolution: [What fixed it]
- Duration: [X hours Y minutes]
- Postmortem: [Link when available]
```

## Escalation Paths

**Technical Escalation:**
- L1: On-call engineer
- L2: Senior engineer
- L3: Engineering manager
- L4: Director of Engineering

**Communication Escalation:**
- Internal: #incidents → #leadership
- External: Support team → PR team → Executive team

## Post-Incident

**Immediate Actions:**
- [ ] Announce resolution
- [ ] Create postmortem document
- [ ] Assign postmortem owner
- [ ] Schedule postmortem review meeting

**Postmortem Due:**
- P0: 24 hours
- P1: 48 hours
- P2: 1 week
```

### Generic Mitigation Strategies

From Google Home case study (SRE Workbook):

**Principle: Mitigate BEFORE full root cause analysis**

**Generic Mitigations (fastest → slowest):**

1. **Rollback** (~5 minutes)
   - Revert to known good version
   - Works when incident correlates with recent change

2. **Traffic Drain** (~10 minutes)
   - Route traffic away from problematic region/backend
   - Works when errors are localized

3. **Scale Up** (~15 minutes)
   - Increase resource capacity
   - Works for quota/resource exhaustion

4. **Kill Switch** (~5 minutes)
   - Disable problematic feature
   - Works when feature is non-critical

5. **Cache Invalidation** (~10 minutes)
   - Clear corrupt cache data
   - Works for cache poisoning

6. **Service Restart** (~30 minutes)
   - Full service restart
   - Last resort, causes brief downtime

**Decision Tree for Generic Mitigation:**

```
Is incident correlated with recent change?
    ↓ YES → ROLLBACK
    ↓ NO
    ↓
Are errors localized to region/datacenter?
    ↓ YES → DRAIN TRAFFIC
    ↓ NO
    ↓
Is service hitting resource limits?
    ↓ YES → SCALE UP
    ↓ NO
    ↓
Can we disable problematic feature?
    ↓ YES → KILL SWITCH
    ↓ NO
    ↓
Is cache suspect?
    ↓ YES → INVALIDATE CACHE
    ↓ NO
    ↓
Investigate root cause while monitoring
```

## Section 4: Production Debug Tools

### Cloud Debugger (Deprecated 2023-05-31)

**Note**: Cloud Debugger was deprecated on May 16, 2022 and shut down on May 31, 2023. Open source Snapshot Debugger remains available.

From [Cloud Debugger deprecation notice](https://docs.cloud.google.com/stackdriver/docs/deprecations/debugger-deprecation) (accessed 2025-02-03):

**Alternatives to Cloud Debugger:**
1. **Snapshot Debugger (Open Source)** - Community-maintained fork
2. **Cloud Logging** - Structured logging with severity levels
3. **Cloud Trace** - Distributed tracing for request flows
4. **Error Reporting** - Automatic error aggregation and alerting
5. **Local debugging** - Debug staging environments with production data

### Error Reporting

From [Error Reporting documentation](https://docs.cloud.google.com/error-reporting/docs) (accessed 2025-02-03):

**Error Reporting automatically:**
- Groups errors by stack trace
- Aggregates error counts
- Detects new errors
- Provides real-time notifications

**Sending Errors to Error Reporting:**

```python
# Python - Manual error reporting
from google.cloud import error_reporting

client = error_reporting.Client()

try:
    # Production code
    result = risky_operation()
except Exception as e:
    # Report to Error Reporting
    client.report_exception()
    # Also log for context
    logger.exception("Operation failed")
    raise
```

**Viewing Errors:**

```bash
# List errors via gcloud (if available)
gcloud error-reporting events list --service=SERVICE_NAME

# Query via Cloud Logging
gcloud logging read \
  "severity>=ERROR AND resource.type=cloud_run_revision" \
  --limit=100 \
  --format=json
```

**Error Grouping:**
- Errors with same stack trace → single group
- Shows frequency, first seen, last seen
- Alerts on new error groups
- Links to stack trace and logs

### Cloud Logging Advanced Queries

From cloud-build-advanced/04-debugging.md:

**Production Debugging Queries:**

```bash
# Errors in last hour
gcloud logging read \
  "severity>=ERROR AND timestamp>=$(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S)" \
  --limit=100

# Errors for specific service
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=my-service AND severity>=ERROR" \
  --limit=100

# Errors with specific message
gcloud logging read \
  "textPayload=~'timeout' AND severity>=ERROR" \
  --limit=100

# High-latency requests (via trace)
gcloud logging read \
  "jsonPayload.latency>5 AND resource.type=cloud_run_revision" \
  --limit=100
```

**Structured Logging Best Practices:**

```python
# Python - Structured logging
import structlog

logger = structlog.get_logger()

# Good: Structured fields
logger.info(
    "request_processed",
    user_id=user.id,
    request_id=request_id,
    duration_ms=duration,
    status_code=200
)

# Bad: String concatenation
logger.info(f"Processed request for user {user.id} in {duration}ms")
```

**Why structured logging matters:**
- Enables precise filtering in Cloud Logging
- Supports aggregation and analysis
- Makes correlation easier
- Improves log-based metrics

### Monitoring and Alerting Setup

```bash
# Create log-based metric for errors
gcloud logging metrics create error_rate \
  --description="Error rate metric" \
  --log-filter='severity>=ERROR'

# Create alert policy
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="High Error Rate" \
  --condition-display-name="Error rate > 1%" \
  --condition-threshold-value=0.01 \
  --condition-threshold-duration=300s
```

## Section 5: Post-Incident Analysis

### Postmortem Culture

From [Google SRE Workbook - Postmortem Culture](https://sre.google/workbook/postmortem-culture/) principles:

**Blameless Postmortems:**
- Focus on systems and processes, not people
- Assume everyone acted with good intent
- Learn from failures to prevent recurrence
- Share widely for organizational learning

**Postmortem Template:**

```markdown
# Postmortem: [Brief Incident Title]

**Date:** YYYY-MM-DD
**Authors:** [Names]
**Reviewers:** [Names]
**Status:** Draft | In Review | Final
**Severity:** P0 | P1 | P2 | P3

## Summary

[2-3 sentence summary of what happened]

## Impact

- **Duration:** [X hours Y minutes]
- **Users Affected:** [Number or percentage]
- **Revenue Impact:** [$X if applicable]
- **Services Affected:** [List]

## Timeline (all times UTC)

| Time | Event |
|------|-------|
| HH:MM | First alert fired |
| HH:MM | Incident declared |
| HH:MM | Mitigation attempted |
| HH:MM | Root cause identified |
| HH:MM | Incident resolved |

## Root Cause

### Five Whys

1. Why did X happen? → Because Y
2. Why did Y happen? → Because Z
3. [Continue to root cause]

### Technical Details

[In-depth technical explanation of root cause]

## Resolution

[What fixed the incident]

## Detection

**How we detected:**
- [Monitoring alert | Customer report | Manual discovery]

**Time to detect:** [X minutes]

**What went well:**
- [List]

**What could improve:**
- [List]

## Lessons Learned

### What Went Well

- Successfully used generic mitigation (rollback)
- Clear communication with stakeholders
- Fast escalation to appropriate teams

### What Didn't Go Well

- Delayed incident declaration
- Missing runbook for this scenario
- No monitoring for this failure mode

### Where We Got Lucky

- Incident occurred during business hours
- Expert was available and online
- Impact limited to single region

## Action Items

| Action Item | Owner | Bug | Due Date | Status |
|-------------|-------|-----|----------|--------|
| Add monitoring for X | @alice | BUG-123 | 2025-02-15 | Open |
| Create runbook for Y | @bob | BUG-124 | 2025-02-20 | In Progress |
| Fix root cause Z | @charlie | BUG-125 | 2025-03-01 | Open |

## Supporting Information

**Logs:** [Links to relevant logs]
**Traces:** [Links to traces]
**Monitoring:** [Links to dashboards]
**Discussion:** [Links to incident doc, Slack, etc.]
```

### Incident Review Meeting

**Agenda:**
1. Review timeline (10 min)
2. Discuss root cause (10 min)
3. Identify action items (15 min)
4. Assign owners and due dates (5 min)

**Ground Rules:**
- Blameless - focus on systems
- Constructive - focus on improvements
- Time-boxed - respect attendees' time
- Action-oriented - concrete next steps

### Follow-Up and Prevention

**Action Item Categories:**

1. **Prevent** - Stop this specific issue from recurring
   - Fix bug that caused incident
   - Add validation to catch this error

2. **Detect** - Catch similar issues faster
   - Add monitoring/alerting
   - Improve logging

3. **Mitigate** - Reduce impact when it happens
   - Build generic mitigation tool
   - Create runbook

4. **Learn** - Improve team knowledge
   - Share postmortem widely
   - Update training materials
   - Conduct drill/simulation

**Tracking Action Items:**

```bash
# Create tracking bugs for all action items
# Example using GitHub CLI:
gh issue create \
  --title "Add monitoring for NTP drift" \
  --body "Action item from postmortem POST-2025-02-03" \
  --label "postmortem-action,monitoring" \
  --assignee alice

# Link to postmortem document
# Set due date
# Track progress in project board
```

## Sources

**Source Documents:**
- [cloud-build-advanced/04-debugging.md](../cloud-build-advanced/04-debugging.md) - Cloud Build debugging patterns, log streaming, failure analysis

**Web Research:**
- [Google SRE Workbook - Incident Response](https://sre.google/workbook/incident-response/) - Comprehensive incident management guide, ICS framework, case studies (accessed 2025-02-03)
- [Google SRE Workbook - Postmortem Culture](https://sre.google/workbook/postmortem-culture/) - Blameless postmortem practices (accessed 2025-02-03)
- [GCP Operational Excellence - Manage Incidents](https://docs.cloud.google.com/architecture/framework/operational-excellence/manage-incidents-and-problems) - GCP-specific incident management recommendations (accessed 2025-02-03)
- [Cloud Trace and Cloud Logging for Root Cause Analysis](https://cloud.google.com/blog/products/devops-sre/using-cloud-trace-and-cloud-logging-for-root-cause-analysis) - Integrated RCA workflow using GCP observability tools (accessed 2025-02-03)
- [Error Reporting Documentation](https://docs.cloud.google.com/error-reporting/docs) - Automatic error aggregation and grouping (accessed 2025-02-03)
- [Cloud Debugger Deprecation](https://docs.cloud.google.com/stackdriver/docs/deprecations/debugger-deprecation) - Cloud Debugger sunset and alternatives (accessed 2025-02-03)
- [PagerDuty Incident Response](https://response.pagerduty.com/about/) - Industry-standard incident response framework (accessed 2025-02-03)
- [How to Implement Observability in GCP](https://www.nearsure.com/blog/how-to-implement-observability-in-gcp-tools-best-practices) - Observability tools and practices (accessed 2025-02-03)
- [Top Security Best Practices for GCP](https://www.darktrace.com/cyber-ai-glossary/top-security-best-practices-for-google-cloud-platform-gcp) - Security incident response (accessed 2025-02-03)
- [Performing Effective Root Cause Analysis](https://newrelic.com/blog/how-to-relic/performing-effective-root-cause-analysis) - RCA techniques and best practices (accessed 2025-02-03)

**Additional References:**
- [Incident Command System (ICS)](https://en.wikipedia.org/wiki/Incident_Command_System) - Origin of modern incident response frameworks
- [Google DiRT Program](https://queue.acm.org/detail.cfm?id=2371516) - Disaster recovery testing methodology
- [Incident Management Guide](https://sre.google/resources/practices-and-processes/incident-management-guide/) - SRE incident management primer
