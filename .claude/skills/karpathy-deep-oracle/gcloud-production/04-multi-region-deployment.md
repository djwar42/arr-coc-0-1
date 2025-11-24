# Multi-Region Deployment on GCP

## Overview

Multi-region deployment architecture distributes applications and data across multiple geographic regions to achieve high availability, disaster recovery, low latency for global users, and compliance with data residency requirements. This document covers deployment patterns, failover strategies, load balancing techniques, and disaster recovery planning for production GCP workloads.

**Key Benefits:**
- **High Availability**: 99.99%+ uptime through geographic redundancy
- **Disaster Recovery**: Automatic failover to healthy regions
- **Low Latency**: Serve users from nearest region
- **Data Sovereignty**: Comply with regional data regulations
- **Horizontal Scaling**: Distribute load across global infrastructure

From [Multi-regional deployment on Compute Engine](https://docs.cloud.google.com/architecture/multiregional-vms) (Google Cloud documentation, accessed 2025-02-03)

From [How to Build Highly Available Multi-regional Services](https://cloud.google.com/blog/topics/developers-practitioners/how-to-build-highly-available-multi-regional-services-with-cloud-run) (Google Cloud blog, accessed 2025-02-03)

---

## Multi-Region Deployment Patterns

### 1. Active-Active Pattern

Deploy identical application instances in multiple regions with traffic distributed across all regions.

**Architecture:**
```
┌─────────────────────────────────────────────────
│ Global Load Balancer (Cloud Load Balancing)
├─────────────────────────────────────────────────
│
├── Region 1 (us-central1)          ├── Region 2 (europe-west1)
│   ├── Backend Service             │   ├── Backend Service
│   ├── Instance Groups              │   ├── Instance Groups
│   ├── Cloud SQL (Primary)         │   ├── Cloud SQL (Replica)
│   └── GCS Bucket                  │   └── GCS Bucket
│
└── Region 3 (asia-southeast1)
    ├── Backend Service
    ├── Instance Groups
    ├── Cloud SQL (Replica)
    └── GCS Bucket
```

**Use Cases:**
- Global web applications requiring low latency worldwide
- E-commerce platforms with international customers
- SaaS applications with distributed user base
- Gaming platforms needing regional presence

**Configuration Example:**
```bash
# Create regional instance groups
gcloud compute instance-groups managed create app-ig-us \
  --region=us-central1 \
  --template=app-template \
  --size=3

gcloud compute instance-groups managed create app-ig-eu \
  --region=europe-west1 \
  --template=app-template \
  --size=3

# Create backend services
gcloud compute backend-services create global-backend \
  --protocol=HTTP \
  --health-checks=http-health-check \
  --global

# Add regional backends
gcloud compute backend-services add-backend global-backend \
  --instance-group=app-ig-us \
  --instance-group-region=us-central1 \
  --global

gcloud compute backend-services add-backend global-backend \
  --instance-group=app-ig-eu \
  --instance-group-region=europe-west1 \
  --global
```

From [Building a New Multi-Region Database Architecture in GCP](https://www.cloudthat.com/resources/blog/building-a-new-multi-region-database-architecture-in-google-cloud-platform-gcp) (CloudThat, accessed 2025-02-03)

### 2. Active-Passive Pattern

Primary region handles all traffic; secondary region(s) serve as hot standby.

**Architecture:**
```
┌─────────────────────────────────────
│ Cloud DNS (Failover Policy)
├─────────────────────────────────────
│
├── Primary Region (us-central1)
│   ├── Active: Serving Traffic
│   ├── Backend Services (100% traffic)
│   └── Primary Database
│
└── Secondary Region (us-east1)
    ├── Standby: Ready for Failover
    ├── Backend Services (0% traffic)
    └── Database Replica (read-only)
```

**Use Cases:**
- Disaster recovery for regional applications
- Cost optimization (pay for standby capacity only)
- Applications with strict data consistency requirements
- Legacy systems migrating to multi-region

**Failover Configuration:**
```bash
# Create Cloud DNS failover policy
gcloud dns record-sets create app.example.com \
  --type=A \
  --zone=production-zone \
  --routing-policy-type=FAILOVER \
  --routing-policy-primary-targets=PRIMARY_REGION_IP \
  --routing-policy-backup-targets=SECONDARY_REGION_IP \
  --enable-health-checking \
  --health-check-interval=10s
```

### 3. Multi-Regional with Locality-Based Routing

Route users to nearest region based on geographic location.

**Architecture:**
```
┌─────────────────────────────────────
│ Cloud DNS (Geo Routing Policy)
├─────────────────────────────────────
│
├── North America → us-central1
├── Europe → europe-west1
├── Asia → asia-southeast1
└── Australia → australia-southeast1
```

**Configuration Example:**
```bash
# Create geo-routing policy in Cloud DNS
gcloud dns record-sets create app.example.com \
  --type=A \
  --zone=production-zone \
  --routing-policy-type=GEO \
  --routing-policy-data="us-central1=10.0.1.10;europe-west1=10.0.2.10;asia-southeast1=10.0.3.10"
```

From [GCP Cross-region internal application load balancer](https://medium.com/google-cloud/gcp-cross-region-internal-application-load-balancer-why-and-how-f3a33226d690) (Medium/Google Cloud Community, accessed 2025-02-03)

---

## Load Balancing Strategies

### Global HTTP(S) Load Balancer

Distributes traffic across regions with intelligent routing and automatic failover.

**Key Features:**
- **Anycast IP**: Single global IP address for all regions
- **Cross-Region Failover**: Automatic traffic redirection to healthy backends
- **Proximity Routing**: Directs users to nearest region
- **SSL/TLS Termination**: Centralized certificate management
- **CDN Integration**: Cloud CDN caching at edge locations

**Setup:**
```bash
# Create global forwarding rule
gcloud compute forwarding-rules create global-lb-rule \
  --global \
  --target-http-proxy=http-proxy \
  --address=GLOBAL_STATIC_IP \
  --ports=80

# Create URL map
gcloud compute url-maps create global-url-map \
  --default-service=global-backend

# Create HTTP proxy
gcloud compute target-http-proxies create http-proxy \
  --url-map=global-url-map
```

**Traffic Distribution:**
- **Round Robin**: Equal distribution across healthy backends
- **Weighted**: Percentage-based traffic split (canary deployments)
- **Session Affinity**: Cookie-based sticky sessions

From [Failover for external Application Load Balancers](https://docs.cloud.google.com/load-balancing/docs/https/applb-failover-overview) (Google Cloud documentation, accessed 2025-02-03)

### Cross-Region Internal Load Balancer

Distributes internal traffic across regions for private workloads.

**Architecture:**
```
┌──────────────────────────────────────
│ Cross-Region Internal LB
├──────────────────────────────────────
│
├── Region 1 Frontend (10.10.152.10)
│   └── Backend: Region 1 VMs
│
└── Region 2 Frontend (10.10.151.11)
    └── Backend: Region 2 VMs
```

**Key Capabilities:**
- Proxy-based Layer 7 load balancing
- Private IP addressing (no external exposure)
- Automatic backend failover to nearest healthy region
- Integration with VPC peering and Shared VPC

**Configuration:**
```bash
# Create cross-region internal LB
gcloud compute forwarding-rules create internal-lb-asia-south1 \
  --load-balancing-scheme=INTERNAL_MANAGED \
  --network=vpc-network \
  --subnet=asia-south1-subnet \
  --address=10.10.152.10 \
  --ports=80 \
  --region=asia-south1 \
  --target-http-proxy=internal-proxy

gcloud compute forwarding-rules create internal-lb-asia-south2 \
  --load-balancing-scheme=INTERNAL_MANAGED \
  --network=vpc-network \
  --subnet=asia-south2-subnet \
  --address=10.10.151.11 \
  --ports=80 \
  --region=asia-south2 \
  --target-http-proxy=internal-proxy

# Create backend service with multi-region backends
gcloud compute backend-services create cross-region-backend \
  --load-balancing-scheme=INTERNAL_MANAGED \
  --protocol=HTTP \
  --health-checks=auto-health \
  --global

# Add regional backends
gcloud compute backend-services add-backend cross-region-backend \
  --instance-group=mumbai-instance-group \
  --instance-group-zone=asia-south1-c \
  --global

gcloud compute backend-services add-backend cross-region-backend \
  --instance-group=delhi-instance-group \
  --instance-group-zone=asia-south2-a \
  --global
```

**Failover Behavior:**
1. **Backend Failover**: Automatic routing to next closest healthy region
2. **Frontend Failover**: DNS-based failover to healthy frontend IP

From [GCP Cross-region internal application load balancer: why and how](https://medium.com/google-cloud/gcp-cross-region-internal-application-load-balancer-why-and-how-f3a33226d690) (Medium, accessed 2025-02-03)

### Regional Load Balancer with DNS Failover

Combine regional load balancers with Cloud DNS failover policies.

**Pattern:**
```
Cloud DNS Failover Policy
├── Primary: us-central1 LB (health check: PASS)
└── Backup: us-east1 LB (activated on primary failure)
```

**Configuration:**
```bash
# Create failover policy
gcloud dns record-sets create app.example.com \
  --type=A \
  --zone=prod-zone \
  --routing-policy-type=FAILOVER \
  --routing-policy-primary-targets=PRIMARY_LB_IP \
  --routing-policy-backup-targets=BACKUP_LB_IP \
  --enable-health-checking \
  --health-check-interval=10s \
  --health-check-timeout=5s
```

---

## Regional Failover Strategies

### 1. Health Check-Based Failover

Continuous monitoring with automatic traffic redirection on failure.

**Health Check Configuration:**
```bash
# Create HTTP health check
gcloud compute health-checks create http app-health-check \
  --check-interval=10s \
  --timeout=5s \
  --unhealthy-threshold=3 \
  --healthy-threshold=2 \
  --port=80 \
  --request-path=/health

# Create HTTPS health check with SSL
gcloud compute health-checks create https app-https-health \
  --check-interval=10s \
  --timeout=5s \
  --unhealthy-threshold=2 \
  --healthy-threshold=2 \
  --port=443 \
  --request-path=/healthz
```

**Thresholds:**
- **Check Interval**: 5-10 seconds (production)
- **Timeout**: 5 seconds
- **Unhealthy Threshold**: 2-3 consecutive failures
- **Healthy Threshold**: 2 consecutive successes

### 2. DNS-Based Failover

Cloud DNS failover policies for frontend IP failover.

**Failover Workflow:**
```
Primary Region Healthy
  └── DNS resolves to Primary LB IP (10.0.1.10)

Primary Region Fails
  ├── Health check detects failure (3 consecutive)
  ├── DNS switches to Backup LB IP (10.0.2.10)
  └── TTL: 60 seconds (propagation delay)
```

**Best Practices:**
- Set DNS TTL to 60 seconds for faster failover
- Use multiple backup regions (tiered failover)
- Test failover regularly (quarterly drills)
- Monitor DNS query patterns

### 3. Application-Level Failover

Circuit breaker patterns in application code.

**Example (Python):**
```python
import requests
from circuitbreaker import circuit

REGIONS = [
    'https://us-central1.example.com',
    'https://us-east1.example.com',
    'https://europe-west1.example.com'
]

@circuit(failure_threshold=3, recovery_timeout=60)
def call_api(region_url, endpoint):
    response = requests.get(f"{region_url}/{endpoint}", timeout=5)
    response.raise_for_status()
    return response.json()

def multi_region_call(endpoint):
    for region in REGIONS:
        try:
            return call_api(region, endpoint)
        except Exception as e:
            print(f"Region {region} failed: {e}")
            continue
    raise Exception("All regions unavailable")
```

---

## Data Replication Strategies

### Cloud Spanner (Global Database)

Fully managed, globally distributed database with strong consistency.

**Configuration:**
```bash
# Create multi-region instance
gcloud spanner instances create global-instance \
  --config=nam-eur-asia1 \
  --description="Global multi-region instance" \
  --nodes=3

# Create database
gcloud spanner databases create app-db \
  --instance=global-instance
```

**Replication Modes:**
- **Multi-Region**: Data replicated across 3+ regions automatically
- **Strong Consistency**: Linearizable reads and writes globally
- **Read Replicas**: Low-latency reads in additional regions

### Cloud SQL Cross-Region Replication

Asynchronous replication for MySQL and PostgreSQL.

**Setup:**
```bash
# Create primary instance
gcloud sql instances create primary-db \
  --database-version=POSTGRES_14 \
  --tier=db-n1-standard-4 \
  --region=us-central1 \
  --backup-location=us

# Create read replica in different region
gcloud sql instances create replica-db \
  --master-instance-name=primary-db \
  --tier=db-n1-standard-4 \
  --region=europe-west1 \
  --replica-type=READ_POOL
```

**Promotion to Primary (Failover):**
```bash
# Promote replica to standalone instance
gcloud sql instances promote-replica replica-db
```

### Cloud Storage Multi-Region Buckets

Geo-redundant object storage.

**Configuration:**
```bash
# Create multi-region bucket
gcloud storage buckets create gs://my-app-assets \
  --location=US \
  --storage-class=STANDARD

# Set lifecycle policy for cost optimization
gcloud storage buckets update gs://my-app-assets \
  --lifecycle-file=lifecycle.json
```

**Lifecycle Policy (lifecycle.json):**
```json
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
        "condition": {"age": 30}
      },
      {
        "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
        "condition": {"age": 90}
      },
      {
        "action": {"type": "Delete"},
        "condition": {"age": 365}
      }
    ]
  }
}
```

---

## Disaster Recovery Planning

### DR Metrics

**Key Metrics:**
- **RTO (Recovery Time Objective)**: Target time to restore service
- **RPO (Recovery Point Objective)**: Maximum acceptable data loss
- **RLO (Recovery Level Objective)**: Acceptable service level during DR

**Typical Targets:**
| Tier | RTO | RPO | Annual Downtime |
|------|-----|-----|-----------------|
| Tier 1 (Critical) | < 1 hour | < 5 minutes | < 4.38 hours |
| Tier 2 (Important) | < 4 hours | < 30 minutes | < 8.76 hours |
| Tier 3 (Standard) | < 24 hours | < 4 hours | < 43.8 hours |

From [Architecting disaster recovery for cloud infrastructure outages](https://docs.cloud.google.com/architecture/disaster-recovery) (Google Cloud documentation, accessed 2025-02-03)

### DR Patterns

**1. Backup and Restore (Lowest Cost, Highest RTO)**
```
Primary Region                 DR Region
├── Active Workload           ├── No resources (cold standby)
└── Automated Backups         └── Restore from backup on failure
    └── GCS, Snapshots            RTO: Hours to Days
                                  RPO: Hours
```

**Configuration:**
```bash
# Schedule automated snapshots
gcloud compute disks snapshot pd-disk \
  --snapshot-names=daily-snapshot-$(date +%Y%m%d) \
  --zone=us-central1-a

# Copy to DR region
gcloud compute snapshots copy daily-snapshot \
  --destination-region=us-east1
```

**2. Pilot Light (Moderate Cost, Medium RTO)**
```
Primary Region                 DR Region
├── Full Workload             ├── Minimal infrastructure
└── Database (Active)         └── Database Replica (Sync)
                                  + Launch templates ready
                                  RTO: Minutes to Hours
                                  RPO: Minutes
```

**3. Warm Standby (Higher Cost, Low RTO)**
```
Primary Region (100% traffic)  DR Region (0% traffic)
├── N nodes serving           ├── N/2 nodes running
└── Database (Primary)        └── Database (Replica)
                                  RTO: Minutes
                                  RPO: Seconds
```

**4. Hot Standby / Active-Active (Highest Cost, Lowest RTO)**
```
Region 1 (50% traffic)         Region 2 (50% traffic)
├── Full capacity             ├── Full capacity
└── Database (Multi-master)   └── Database (Multi-master)
                                  RTO: Seconds
                                  RPO: Zero (sync replication)
```

### DR Testing

**Test Scenarios:**
```bash
# Test 1: Regional failure simulation
gcloud compute instances stop --zone=us-central1-a --all

# Test 2: Database failover
gcloud sql instances promote-replica replica-db

# Test 3: DNS failover
gcloud dns record-sets update app.example.com \
  --type=A \
  --zone=prod-zone \
  --routing-policy-primary-targets=BACKUP_IP

# Test 4: Restore from backup
gcloud compute disks create restored-disk \
  --source-snapshot=daily-snapshot \
  --zone=us-east1-a
```

**Testing Cadence:**
- **Quarterly**: Full DR failover test
- **Monthly**: Backup restoration test
- **Weekly**: Health check validation
- **Daily**: Automated backup verification

From [Disaster recovery planning guide](https://docs.cloud.google.com/architecture/dr-scenarios-planning-guide) (Google Cloud documentation, accessed 2025-02-03)

---

## Quota Distribution Across Regions

**Best Practices:**
- Request quotas for all regions in DR plan
- Maintain 2x capacity in primary region
- Keep 1x capacity in DR regions (for failover)

**Quota Request:**
```bash
# Check current quotas
gcloud compute project-info describe \
  --project=PROJECT_ID \
  --format="value(quotas)"

# Request quota increase
gcloud alpha quotas update \
  --service=compute.googleapis.com \
  --quota=NVIDIA_T4_GPUS \
  --consumer=projects/PROJECT_ID \
  --region=us-east1 \
  --value=100
```

**Distribution Example:**
```
Primary Region (us-central1):
├── CPUs: 200
├── GPUs (T4): 50
└── Persistent Disks: 10 TB

DR Region (us-east1):
├── CPUs: 100 (50% of primary)
├── GPUs (T4): 25 (50% of primary)
└── Persistent Disks: 10 TB (full replica)

Additional Region (europe-west1):
├── CPUs: 100
├── GPUs (T4): 25
└── Persistent Disks: 5 TB
```

---

## Monitoring and Alerting

### Key Metrics

**Load Balancer Metrics:**
```bash
# Create alert policy for LB errors
gcloud alpha monitoring policies create lb-error-rate \
  --notification-channels=CHANNEL_ID \
  --condition-threshold-value=5 \
  --condition-threshold-duration=300s \
  --condition-metric=loadbalancing.googleapis.com/https/request_count \
  --condition-filter='metric.response_code_class="5xx"'
```

**Health Check Metrics:**
- `health_check_probe_count`: Number of health checks
- `health_check_healthy_count`: Healthy backends
- `health_check_unhealthy_count`: Unhealthy backends

**Failover Metrics:**
- Time to detect failure (< 30 seconds)
- Time to failover (< 60 seconds)
- Traffic loss during failover (< 1%)

### Observability Stack

**Cloud Monitoring Dashboard:**
```yaml
# Dashboard configuration
displayName: "Multi-Region Dashboard"
mosaicLayout:
  tiles:
    - widget:
        title: "Backend Health by Region"
        xyChart:
          dataSets:
            - timeSeriesQuery:
                timeSeriesFilter:
                  filter: 'resource.type="compute.googleapis.com/backend_service"'
                  aggregation:
                    perSeriesAligner: ALIGN_MEAN
                    groupByFields: ["resource.region"]
```

**Log-Based Metrics:**
```bash
# Create log-based metric for failover events
gcloud logging metrics create failover_events \
  --description="Count of failover events" \
  --log-filter='resource.type="gce_instance" AND
                jsonPayload.message=~"failover"'
```

---

## Cost Optimization

### Multi-Region Cost Strategies

**1. Committed Use Discounts (CUDs)**
```bash
# Purchase 1-year CUD for multi-region deployment
gcloud compute commitments create multi-region-cud \
  --region=us-central1 \
  --plan=12-month \
  --resources=vcpu=100,memory=400GB
```

**Savings: 37% for 1-year, 55% for 3-year**

**2. Right-Sizing DR Regions**
```
Production Region: 10 x n1-standard-8 (8 vCPU, 30 GB)
DR Region: 5 x n1-standard-8 (50% capacity)
Cost Savings: 50% on DR infrastructure
```

**3. Preemptible VMs for Non-Critical Workloads**
```bash
# Use preemptible VMs in less critical regions
gcloud compute instances create preemptible-worker \
  --zone=us-east1-b \
  --machine-type=n1-standard-4 \
  --preemptible \
  --instance-termination-action=STOP
```

**Savings: Up to 80% vs standard VMs**

**4. Storage Class Optimization**
```
Production Database: Cloud SQL (regional)
DR Database: Cloud SQL replica (read-only, lower tier)
Backups: GCS Nearline (30-day), Coldline (90+ days)
```

---

## Security Considerations

### Cross-Region Security

**VPC Service Controls:**
```bash
# Create service perimeter for multi-region resources
gcloud access-context-manager perimeters create multi-region-perimeter \
  --resources=projects/PROJECT_ID \
  --restricted-services=storage.googleapis.com,compute.googleapis.com \
  --vpc-allowed-services=ALLOW_ALL
```

**IAM Policies:**
- Use separate service accounts per region
- Grant minimum permissions for cross-region access
- Enable audit logging for all cross-region API calls

**Data Encryption:**
- Encrypt data at rest (CMEK for sensitive data)
- Encrypt data in transit (TLS 1.3)
- Use Cloud KMS multi-region keys

```bash
# Create multi-region KMS key
gcloud kms keyrings create multi-region-keyring \
  --location=us

gcloud kms keys create encryption-key \
  --keyring=multi-region-keyring \
  --location=us \
  --purpose=encryption
```

---

## Production Checklist

**Pre-Deployment:**
- [ ] Design multi-region architecture (active-active/active-passive)
- [ ] Request quotas in all target regions
- [ ] Set up VPC peering or Shared VPC
- [ ] Configure Cloud DNS geo-routing or failover policies
- [ ] Create load balancer backends in each region
- [ ] Set up database replication (Cloud SQL/Spanner)
- [ ] Configure backup retention policies

**Deployment:**
- [ ] Deploy application to primary region
- [ ] Deploy to DR regions (50-100% capacity)
- [ ] Configure health checks (10s interval)
- [ ] Set up global load balancer
- [ ] Enable Cloud CDN for static assets
- [ ] Configure SSL certificates (managed certs)

**Post-Deployment:**
- [ ] Test failover scenarios (quarterly)
- [ ] Monitor cross-region latency (< 100ms target)
- [ ] Set up alerting (PagerDuty, Slack)
- [ ] Document runbooks for failover procedures
- [ ] Review and optimize costs monthly
- [ ] Conduct DR drills with team

**Monitoring:**
- [ ] Backend health across regions
- [ ] Request latency by region
- [ ] Error rates (5xx) by region
- [ ] Database replication lag
- [ ] Quota utilization per region

---

## Common Issues and Solutions

**Issue: High cross-region latency (> 200ms)**

**Solution:**
```bash
# Enable Premium Network Tier for lower latency
gcloud compute project-info update \
  --default-network-tier=PREMIUM

# Use Cloud CDN for cacheable content
gcloud compute backend-services update backend \
  --enable-cdn
```

**Issue: Quota exhaustion during failover**

**Solution:**
- Pre-allocate 2x capacity in DR regions
- Request quotas proactively before scaling events
- Use managed instance groups with autoscaling

**Issue: Database replication lag > 5 seconds**

**Solution:**
```bash
# Promote replica to standalone (if lag persists)
gcloud sql instances promote-replica replica-db

# For Spanner: Increase node count
gcloud spanner instances update global-instance --nodes=5
```

**Issue: DNS propagation delay (TTL 300s)**

**Solution:**
- Reduce TTL to 60 seconds for production records
- Use health check-based failover (faster than DNS)
- Implement client-side retry logic

---

## Sources

**Google Cloud Documentation:**
- [Multi-regional deployment on Compute Engine](https://docs.cloud.google.com/architecture/multiregional-vms) (accessed 2025-02-03)
- [Failover for external Application Load Balancers](https://docs.cloud.google.com/load-balancing/docs/https/applb-failover-overview) (accessed 2025-02-03)
- [Architecting disaster recovery for cloud infrastructure outages](https://docs.cloud.google.com/architecture/disaster-recovery) (accessed 2025-02-03)
- [Disaster recovery planning guide](https://docs.cloud.google.com/architecture/dr-scenarios-planning-guide) (accessed 2025-02-03)

**Web Research:**
- [How to Build Highly Available Multi-regional Services with Cloud Run](https://cloud.google.com/blog/topics/developers-practitioners/how-to-build-highly-available-multi-regional-services-with-cloud-run) (Google Cloud Blog, accessed 2025-02-03)
- [Building a New Multi-Region Database Architecture in GCP](https://www.cloudthat.com/resources/blog/building-a-new-multi-region-database-architecture-in-google-cloud-platform-gcp) (CloudThat, accessed 2025-02-03)
- [GCP Cross-region internal application load balancer: why and how](https://medium.com/google-cloud/gcp-cross-region-internal-application-load-balancer-why-and-how-f3a33226d690) (Medium/Google Cloud Community, accessed 2025-02-03)

**Additional References:**
- Google Cloud regions and zones: https://cloud.google.com/compute/docs/regions-zones
- Cloud DNS routing policies: https://cloud.google.com/dns/docs/zones/manage-routing-policies
- Cloud Spanner multi-region configurations: https://cloud.google.com/spanner/docs/instances
