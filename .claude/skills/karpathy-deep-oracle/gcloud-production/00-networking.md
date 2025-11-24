# GCP Production Networking - VPC Service Controls, Private Access, and Shared VPC

Complete guide to production-grade GCP networking for AI/ML workloads with emphasis on security, private connectivity, and multi-project architectures.

---

## Section 1: Networking Overview (~100 lines)

### GCP Networking Architecture

**Global VPC - The GCP Advantage**

Unlike AWS and Azure where VPCs/VNets are regional, GCP VPCs are **global by default**. This fundamentally changes network architecture:

```bash
# Single VPC spans all regions globally
gcloud compute networks create production-vpc \
  --subnet-mode=custom \
  --bgp-routing-mode=global

# Add regional subnets as needed
gcloud compute networks subnets create us-central1-subnet \
  --network=production-vpc \
  --region=us-central1 \
  --range=10.0.0.0/20

gcloud compute networks subnets create europe-west1-subnet \
  --network=production-vpc \
  --region=europe-west1 \
  --range=10.1.0.0/20
```

**Benefits:**
- No VPC peering needed between regions (all one VPC)
- Internal IPs routable globally within VPC
- Simplified multi-region architectures
- Lower latency (traffic stays on Google's network)

**Subnets are Regional (Not Zonal)**

GCP subnets span all zones within a region - different from AWS where subnets are per-AZ:

```
VPC (Global): 10.0.0.0/16
├── us-central1 subnet: 10.0.0.0/20
│   ├── us-central1-a (zone)
│   ├── us-central1-b (zone)
│   └── us-central1-c (zone)
└── europe-west1 subnet: 10.1.0.0/20
    ├── europe-west1-b (zone)
    ├── europe-west1-c (zone)
    └── europe-west1-d (zone)
```

**Key Characteristics:**
- Subnets are **expandable** without downtime
- Support **secondary IP ranges** for GKE pods/services
- **Private Google Access** per subnet (VMs without external IPs can reach Google APIs)
- **VPC Flow Logs** enabled per subnet

### Firewall Rules - Distributed and Tag-Based

GCP firewall rules are **applied at VPC level** but **enforced at each VM instance** (distributed firewall):

```bash
# Allow HTTPS from internet to web servers (by tag)
gcloud compute firewall-rules create allow-https \
  --network=production-vpc \
  --action=allow \
  --direction=ingress \
  --rules=tcp:443 \
  --source-ranges=0.0.0.0/0 \
  --target-tags=web-server \
  --priority=1000

# Allow SSH via Identity-Aware Proxy (IAP)
gcloud compute firewall-rules create allow-iap-ssh \
  --network=production-vpc \
  --action=allow \
  --direction=ingress \
  --rules=tcp:22 \
  --source-ranges=35.235.240.0/20 \
  --priority=1000

# Allow internal traffic
gcloud compute firewall-rules create allow-internal \
  --network=production-vpc \
  --action=allow \
  --direction=ingress \
  --rules=all \
  --source-ranges=10.0.0.0/8 \
  --priority=2000
```

**Targeting Options:**
- **Network tags**: Apply to VMs with specific tags (e.g., `web-server`, `db-server`)
- **Service accounts**: Apply to VMs running with specific service accounts
- **Stateful**: Return traffic automatically allowed

From [VPC design considerations for Google Cloud](https://medium.com/@pbijjala/vpc-design-considerations-for-google-cloud-71ce67427256) (accessed 2025-02-03):
- Firewall rules are **eventual consistent** across regions
- Rules evaluated in **priority order** (0-65535, lower = higher priority)
- Default behavior: **Deny all ingress, allow all egress**

---

## Section 2: VPC Service Controls (~150 lines)

### What is VPC Service Controls?

VPC Service Controls (VPC-SC) creates **security perimeters** around Google Cloud resources to prevent data exfiltration. Even with proper IAM permissions, a compromised credential could copy data to external locations - VPC-SC prevents this.

From [VPC Service Controls Overview](https://docs.cloud.google.com/vpc-service-controls/docs/overview) (accessed 2025-02-03):
- Defines perimeters that **prevent access to Google-managed services** outside trusted boundaries
- Blocks data access from unauthorized networks/locations
- **Context-aware access** based on client attributes (identity, device, network origin)

### Core Concepts

**Service Perimeter** - Security boundary around projects and services:

```bash
# Create access policy (organization-level)
gcloud access-context-manager policies create \
  --organization=ORGANIZATION_ID \
  --title="Production Access Policy"

# Create access level (who can access)
gcloud access-context-manager levels create production_access \
  --policy=POLICY_ID \
  --title="Production Access" \
  --basic-level-spec=access_level.yaml

# access_level.yaml
conditions:
  - ipSubnetworks:
    - 10.0.0.0/8  # Internal VPC
    - 203.0.113.0/24  # Office IP
    members:
    - user:admin@example.com
```

**Service Perimeter** (create security boundary):

```bash
gcloud access-context-manager perimeters create production_perimeter \
  --policy=POLICY_ID \
  --title="Production Perimeter" \
  --resources=projects/PROJECT_NUMBER \
  --restricted-services=storage.googleapis.com,bigquery.googleapis.com \
  --access-levels=accessPolicies/POLICY_ID/accessLevels/production_access
```

### Use Cases

From [Configure VPC Service Controls for Gemini](https://developers.google.com/gemini-code-assist/docs/configure-vpc-service-controls) (accessed 2025-02-03):

**1. Protect Sensitive Data**

```yaml
# Perimeter protecting data services
restricted_services:
  - storage.googleapis.com
  - bigquery.googleapis.com
  - spanner.googleapis.com

# Only accessible from VPC or office
ip_subnetworks:
  - 10.0.0.0/8  # VPC CIDR
  - 203.0.113.0/24  # Office IP
```

**2. Prevent Data Exfiltration**

```bash
# Block copying data outside perimeter
gcloud access-context-manager perimeters update production_perimeter \
  --policy=POLICY_ID \
  --add-vpc-allowed-services=storage.googleapis.com

# Egress rules (strict)
--egress-policies=egress_policy.yaml
```

**3. Compliance Requirements (HIPAA, PCI-DSS)**

From [VPC Service Controls release notes](https://docs.cloud.google.com/vpc-service-controls/docs/release-notes) (accessed 2025-02-03):
- Supports **private IPs in Shared VPCs** (2024 update)
- Enhanced **ingress/egress controls**
- **Dry run mode** for testing before enforcement

### Dry Run Mode

**Test perimeter policies without blocking traffic:**

```bash
# Create perimeter in dry run mode
gcloud access-context-manager perimeters create test_perimeter \
  --policy=POLICY_ID \
  --title="Test Perimeter" \
  --perimeter-type=perimeter_type_regular \
  --perimeter-enforcement-mode=dry_run \
  --resources=projects/PROJECT_NUMBER \
  --restricted-services=storage.googleapis.com

# Monitor violations in logs
gcloud logging read "protoPayload.metadata.dryRun=true" \
  --format=json \
  --limit=50
```

**Benefits of dry run:**
- Identify unexpected traffic patterns
- Understand impact before enforcement
- Create **honeypot perimeters** to detect malicious probing

### VPC-SC with Vertex AI

From [Private connectivity to Vertex workloads](https://cloud.google.com/blog/products/networking/private-connectivity-to-vertex-workloads) (accessed 2025-02-03):

```bash
# Protect Vertex AI APIs within perimeter
gcloud access-context-manager perimeters create vertex_perimeter \
  --policy=POLICY_ID \
  --title="Vertex AI Perimeter" \
  --resources=projects/VERTEX_PROJECT \
  --restricted-services=aiplatform.googleapis.com \
  --access-levels=accessPolicies/POLICY_ID/accessLevels/vertex_access

# Allow access from specific VPC
--vpc-allowed-services=aiplatform.googleapis.com
```

**Connectivity options for Vertex AI:**
- **Private Service Connect (PSC)** for Google APIs
- **Private Google Access** (PGA)
- **Private Service Access (PSA)** via VPC peering
- **PSC Endpoints** for multi-tenancy

---

## Section 3: Private Google Access (~150 lines)

### What is Private Google Access?

Private Google Access (PGA) allows VMs **without external IPs** to reach Google APIs and services over Google's internal network (not the public internet).

From [GCP Networking Best Practices](https://quabyt.com/blog/gcp-networking-best-practices) (accessed 2025-02-03):
- Enabled **per subnet** (default is OFF)
- VMs use **internal IPs only** to access `*.googleapis.com`
- Reduces egress costs and improves security

### Enabling Private Google Access

```bash
# Enable PGA on existing subnet
gcloud compute networks subnets update us-central1-subnet \
  --region=us-central1 \
  --enable-private-ip-google-access

# Create new subnet with PGA enabled
gcloud compute networks subnets create private-subnet \
  --network=production-vpc \
  --region=us-central1 \
  --range=10.0.16.0/20 \
  --enable-private-ip-google-access
```

### DNS Configuration

**Default DNS (automatic):**

When PGA is enabled, Google Cloud DNS automatically resolves:
- `*.googleapis.com` → `199.36.153.8/30` (Private Google Access range)
- `*.gcr.io` → Private access IPs

**Custom DNS for restricted Google access:**

```bash
# Use restricted.googleapis.com for stricter controls
gcloud dns managed-zones create restricted-googleapis \
  --dns-name=googleapis.com \
  --description="Restricted Google APIs" \
  --networks=production-vpc

gcloud dns record-sets create restricted.googleapis.com \
  --zone=restricted-googleapis \
  --type=A \
  --ttl=300 \
  --rrdatas=199.36.153.4,199.36.153.5,199.36.153.6,199.36.153.7
```

**Three domain options:**

| Domain | IP Range | Use Case |
|--------|----------|----------|
| `private.googleapis.com` | `199.36.153.8/30` | Most Google APIs |
| `restricted.googleapis.com` | `199.36.153.4/30` | Only supported APIs (stricter) |
| `*.googleapis.com` (default) | Varies | Auto-configured when PGA enabled |

### Firewall Rules for PGA

```bash
# Allow egress to Google APIs
gcloud compute firewall-rules create allow-google-apis \
  --network=production-vpc \
  --action=allow \
  --direction=egress \
  --rules=tcp:443 \
  --destination-ranges=199.36.153.8/30 \
  --priority=1000

# Allow egress to restricted Google APIs
gcloud compute firewall-rules create allow-restricted-apis \
  --network=production-vpc \
  --action=allow \
  --direction=egress \
  --rules=tcp:443 \
  --destination-ranges=199.36.153.4/30 \
  --priority=1000
```

### Private Access for Hybrid Connectivity

From [Private networking patterns to Vertex AI workloads](https://asia.microfusion.cloud/news/private-networking-patterns-to-vertex-ai-workloads/) (accessed 2025-02-03):

**On-premises to Google APIs via Cloud Interconnect/VPN:**

```bash
# Advertise Private Google Access routes via Cloud Router
gcloud compute routers update interconnect-router \
  --region=us-central1 \
  --set-advertisement-mode=custom \
  --set-advertisement-groups=all_subnets \
  --add-advertisement-ranges=199.36.153.8/30

# Cloud VPN configuration
gcloud compute vpn-tunnels create on-prem-tunnel \
  --peer-address=ON_PREM_IP \
  --region=us-central1 \
  --ike-version=2 \
  --shared-secret=SECRET \
  --router=vpn-router \
  --interface=0
```

**Route advertisement:**
- Advertise `199.36.153.8/30` to on-premises
- On-prem routes traffic to Google APIs through VPN/Interconnect
- No public internet required

### Use Cases

**1. AI/ML Training Workloads**

```bash
# Vertex AI training without external IPs
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=training-job \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1 \
  --network=projects/PROJECT/global/networks/production-vpc \
  --enable-web-access=false  # No external IP

# VM accesses Cloud Storage privately
gsutil cp gs://training-data/dataset.csv /tmp/
```

**2. Data Engineering (BigQuery, Dataflow)**

```bash
# Dataflow job without public IPs
gcloud dataflow jobs run dataflow-job \
  --gcs-location=gs://dataflow-templates/latest/Word_Count \
  --region=us-central1 \
  --subnetwork=regions/us-central1/subnetworks/private-subnet \
  --disable-public-ips
```

---

## Section 4: Shared VPC (~100 lines)

### What is Shared VPC?

Shared VPC allows **centralized network management** across multiple projects. One "host" project owns the VPC, and multiple "service" projects attach to it.

From [Shared VPC Overview](https://docs.cloud.google.com/vpc/docs/shared-vpc) (accessed 2025-02-03):
- **Host project**: Contains VPC and subnets (network admins manage)
- **Service projects**: Attach to host VPC (teams deploy resources)
- **Within organization only** (requires GCP Organization)

### When to Use Shared VPC

**Multi-team/multi-project environments:**

```
Host Project (Network Admin)
├── Shared VPC: production-vpc
│   ├── us-central1-subnet (10.0.0.0/20)
│   └── europe-west1-subnet (10.1.0.0/20)
└── Service Projects:
    ├── Team A (ML Training)
    ├── Team B (Data Analytics)
    └── Team C (Web Services)
```

**Benefits:**
- **Centralized firewall management** (network team controls rules)
- **Shared services** (DNS, monitoring, NAT gateways)
- **Cost allocation** by project (billing per service project)
- **Simplified IP management** (single CIDR space)

### Setup Shared VPC

```bash
# 1. Enable Shared VPC in host project
gcloud compute shared-vpc enable HOST_PROJECT_ID

# 2. Associate service projects
gcloud compute shared-vpc associated-projects add SERVICE_PROJECT_1 \
  --host-project=HOST_PROJECT_ID

gcloud compute shared-vpc associated-projects add SERVICE_PROJECT_2 \
  --host-project=HOST_PROJECT_ID

# 3. Grant subnet access to service project
gcloud projects add-iam-policy-binding HOST_PROJECT_ID \
  --member="serviceAccount:SERVICE_ACCOUNT@SERVICE_PROJECT_1.iam.gserviceaccount.com" \
  --role="roles/compute.networkUser"

# 4. Grant specific subnet access
gcloud compute networks subnets add-iam-policy-binding us-central1-subnet \
  --project=HOST_PROJECT_ID \
  --region=us-central1 \
  --member="user:team-a-admin@example.com" \
  --role="roles/compute.networkUser"
```

### IAM Roles for Shared VPC

| Role | Scope | Permissions |
|------|-------|-------------|
| `roles/compute.xpnAdmin` | Organization/Folder | Enable Shared VPC, associate projects |
| `roles/compute.networkAdmin` | Host project | Manage VPC, subnets, firewall rules |
| `roles/compute.networkUser` | Subnet level | Use subnet for VM/GKE deployment |
| `roles/compute.securityAdmin` | Host project | Manage firewall rules only |

### Service Project Deployment

```bash
# Deploy VM in service project using host VPC
gcloud compute instances create my-vm \
  --project=SERVICE_PROJECT_1 \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --subnet=projects/HOST_PROJECT_ID/regions/us-central1/subnetworks/us-central1-subnet
```

**Key points:**
- VM created in **service project** (billing)
- Network interface from **host project** (connectivity)
- Firewall rules managed in **host project**

### Shared VPC vs VPC Peering

From [VPC design considerations for Google Cloud](https://medium.com/@pbijjala/vpc-design-considerations-for-google-cloud-71ce67427256) (accessed 2025-02-03):

| Feature | Shared VPC | VPC Peering |
|---------|------------|-------------|
| **Subnets shared** | Yes | No |
| **Firewall rules shared** | Yes | No (must configure on each side) |
| **Cross-organization** | No | Yes |
| **IP overlap** | Not allowed | Not allowed |
| **Number of connections** | Unlimited service projects | Max 25 peered VPCs |
| **Use case** | Multi-team within org | Cross-org connectivity |

---

## Section 5: Network Peering and Firewall (~100 lines)

### VPC Network Peering

VPC Peering connects two VPC networks (within or across organizations) for **private RFC 1918 connectivity**.

From [VPC design considerations for Google Cloud](https://medium.com/@pbijjala/vpc-design-considerations-for-google-cloud-71ce67427256) (accessed 2025-02-03):

**Key characteristics:**
- **Non-overlapping IP ranges** required
- **No transitive peering** (A ↔ B ↔ C, A cannot reach C)
- **Firewall rules not exchanged** (must configure separately)
- **Highest throughput, lowest cost** for VPC-to-VPC connectivity

```bash
# Create peering from VPC A to VPC B
gcloud compute networks peerings create vpc-a-to-vpc-b \
  --network=vpc-a \
  --peer-project=PROJECT_B \
  --peer-network=vpc-b \
  --auto-create-routes

# Create reverse peering from VPC B to VPC A
gcloud compute networks peerings create vpc-b-to-vpc-a \
  --network=vpc-b \
  --peer-project=PROJECT_A \
  --peer-network=vpc-a \
  --auto-create-routes
```

**Route exchange:**

```bash
# Enable custom route export/import
gcloud compute networks peerings update vpc-a-to-vpc-b \
  --network=vpc-a \
  --export-custom-routes \
  --import-custom-routes

# Export subnet routes only (default)
gcloud compute networks peerings update vpc-a-to-vpc-b \
  --network=vpc-a \
  --export-subnet-routes-with-public-ip \
  --no-import-subnet-routes-with-public-ip
```

### Peering Limitations

**No transitive peering:**

```
VPC A (10.0.0.0/16) ↔ VPC B (10.1.0.0/16) ↔ VPC C (10.2.0.0/16)

VPC A can reach VPC B ✓
VPC B can reach VPC C ✓
VPC A CANNOT reach VPC C ✗ (not transitive)
```

**Workaround for on-premises connectivity:**

```bash
# VPC A peers with VPC B (has Cloud Interconnect)
# VPC A can reach on-prem via VPC B (special exception)

# Enable on-premises access
gcloud compute networks peerings update vpc-a-to-vpc-b \
  --network=vpc-a \
  --import-custom-routes  # Import routes from VPC B
```

### Firewall Policy Hierarchy

From [VPC design considerations for Google Cloud](https://medium.com/@pbijjala/vpc-design-considerations-for-google-cloud-71ce67427256) (accessed 2025-02-03):

**Hierarchical firewall policies** apply at organization/folder level:

```bash
# Create organization-level policy
gcloud compute firewall-policies create org-policy \
  --organization=ORGANIZATION_ID \
  --description="Organization-wide firewall rules"

# Add rule to block all SSH except from IAP
gcloud compute firewall-policies rules create 1000 \
  --firewall-policy=org-policy \
  --action=deny \
  --direction=ingress \
  --src-ip-ranges=0.0.0.0/0 \
  --layer4-configs=tcp:22

gcloud compute firewall-policies rules create 900 \
  --firewall-policy=org-policy \
  --action=allow \
  --direction=ingress \
  --src-ip-ranges=35.235.240.0/20 \
  --layer4-configs=tcp:22

# Associate policy with organization
gcloud compute firewall-policies associations create \
  --firewall-policy=org-policy \
  --organization=ORGANIZATION_ID
```

**Evaluation order:**

```
1. Hierarchical firewall policy (Organization level) - Priority 0-1000
2. Hierarchical firewall policy (Folder level) - Priority 1001-2000
3. VPC firewall rules (VPC level) - Priority 0-65535
4. Implicit deny all (Priority 65535)
```

**Lower-level rules CANNOT override higher-level policies**

### Network Tags vs Resource Manager Tags

From [VPC design considerations for Google Cloud](https://medium.com/@pbijjala/vpc-design-considerations-for-google-cloud-71ce67427256) (accessed 2025-02-03):

| Feature | Network Tags | Resource Manager Tags |
|---------|--------------|----------------------|
| **Format** | Simple string | Key-value pairs |
| **Use in firewall** | VPC rules only | Hierarchical policies |
| **Access control** | None | IAM-based |
| **Example** | `web-server` | `env:production`, `team:ml` |

```bash
# Network tag (legacy, simple)
gcloud compute instances create vm1 \
  --tags=web-server,production

# Resource Manager tag (new, structured)
gcloud resource-manager tags values create production \
  --parent=PARENT_TAG_KEY

gcloud resource-manager tags bindings create \
  --tag-value=PARENT_TAG_KEY/production \
  --location=us-central1-a \
  --resource=//compute.googleapis.com/projects/PROJECT/zones/us-central1-a/instances/vm1
```

---

## Sources

**Source Documents:**
- None (web research only)

**Web Research:**

Google Cloud Official Documentation (accessed 2025-02-03):
- [VPC Service Controls Overview](https://docs.cloud.google.com/vpc-service-controls/docs/overview)
- [VPC Service Controls Release Notes](https://docs.cloud.google.com/vpc-service-controls/docs/release-notes)
- [Shared VPC Documentation](https://docs.cloud.google.com/vpc/docs/shared-vpc)
- [Configure VPC Service Controls for Gemini](https://developers.google.com/gemini-code-assist/docs/configure-vpc-service-controls)

Community Articles (accessed 2025-02-03):
- [VPC design considerations for Google Cloud](https://medium.com/@pbijjala/vpc-design-considerations-for-google-cloud-71ce67427256) - Pavan kumar Bijjala, Medium
- [GCP Networking Best Practices: Global VPC, Shared VPC, and Cloud Interconnect](https://quabyt.com/blog/gcp-networking-best-practices) - Quabyt
- [Private networking patterns to Vertex AI workloads](https://asia.microfusion.cloud/news/private-networking-patterns-to-vertex-ai-workloads/) - Microfusion Technology

Google Cloud Blog (accessed 2025-02-03):
- [Private connectivity to Vertex workloads](https://cloud.google.com/blog/products/networking/private-connectivity-to-vertex-workloads)

**Additional References:**
- [VPC Network Peering](https://cloud.google.com/vpc/docs/vpc-peering)
- [Private Google Access](https://cloud.google.com/vpc/docs/private-google-access)
- [Private Service Connect](https://cloud.google.com/vpc/docs/private-service-connect)
- [Hierarchical Firewall Policies](https://cloud.google.com/vpc/docs/firewall-policies)
