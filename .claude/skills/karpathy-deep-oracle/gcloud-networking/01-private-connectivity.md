# GCP Private Connectivity - Cloud Interconnect, HA VPN, and Private Service Connect

Comprehensive guide to private connectivity options for hybrid and multi-cloud architectures, focusing on production-grade connectivity between on-premises networks, GCP, and Google services.

---

## Section 1: Cloud Interconnect Overview (~150 lines)

### What is Cloud Interconnect?

Cloud Interconnect provides **direct physical connectivity** between your on-premises network and Google's network, bypassing the public internet entirely. This offers higher throughput, lower latency, and more predictable performance than VPN-based solutions.

From [Cloud Interconnect overview](https://docs.cloud.google.com/network-connectivity/docs/interconnect/concepts/overview) (accessed 2025-02-03):
- **Dedicated Interconnect**: Direct physical connection (10 Gbps or 100 Gbps) from your network to Google
- **Partner Interconnect**: Connectivity through supported service providers (50 Mbps to 50 Gbps)
- **Cross-Cloud Interconnect**: Direct connectivity to other cloud providers (AWS, Azure, Oracle)

### Dedicated Interconnect

**Use case**: High-bandwidth workloads requiring 10+ Gbps connectivity

```bash
# Prerequisites
# 1. Colocation facility in same metro as Google edge location
# 2. 10GBASE-LR or 100GBASE-LR fiber optic equipment
# 3. BGP routing capability

# Create Dedicated Interconnect
gcloud compute interconnects create my-interconnect \
  --interconnect-type=DEDICATED \
  --link-type=LINK_TYPE_ETHERNET_10G_LR \
  --location=us-east4-zone1 \
  --requested-link-count=2 \
  --admin-enabled

# Create VLAN attachment
gcloud compute interconnects attachments create my-attachment \
  --region=us-east4 \
  --interconnect=my-interconnect \
  --router=my-router \
  --vlan=100
```

**Key specifications**:
- **10 Gbps**: 10GBASE-LR (1310 nm), single-mode fiber
- **100 Gbps**: 100GBASE-LR4 (1310 nm), single-mode fiber
- **Maximum attachments**: 8 per interconnect link
- **BGP**: Required for route exchange

From [Announcement of pricing changes for Cloud Interconnect](https://cloud.google.com/network-connectivity/interconnect-pricing-update-notice) (accessed 2025-02-03):
- Effective February 1, 2024: Pricing changes for Dedicated Interconnect
- **Port fees**: $1,750/month for 10 Gbps, $8,750/month for 100 Gbps (regional pricing varies)
- **Data transfer**: Ingress free, egress varies by region and volume
- **No data transfer charges** for Cross-Cloud Interconnect to Oracle Cloud

### Partner Interconnect

**Use case**: Need connectivity but don't meet Dedicated Interconnect requirements

From [Google Cloud Interconnect Overview](https://docs.packetfabric.com/cloud/google/overview/) (accessed 2025-02-03):

```bash
# List available Partner Interconnect providers
gcloud compute interconnects locations list

# Create Partner Interconnect attachment (pairing key)
gcloud compute interconnects attachments partner create partner-attachment \
  --region=us-central1 \
  --edge-availability-domain=AVAILABILITY_DOMAIN_1 \
  --router=partner-router \
  --admin-enabled

# Get pairing key to provide to partner
gcloud compute interconnects attachments describe partner-attachment \
  --region=us-central1 \
  --format="value(pairingKey)"
```

**Bandwidth options**:
- 50 Mbps, 100 Mbps, 200 Mbps, 300 Mbps, 400 Mbps, 500 Mbps
- 1 Gbps, 2 Gbps, 5 Gbps, 10 Gbps, 20 Gbps, 50 Gbps

**Benefits over Dedicated**:
- No colocation facility required
- Lower bandwidth options available
- Faster provisioning (days vs weeks)
- Provider manages physical layer

### Redundancy and High Availability

From [Cloud Interconnect release notes](https://docs.cloud.google.com/network-connectivity/docs/interconnect/release-notes) (accessed 2025-02-03):

**99.99% SLA requirements**:
```
Topology: Two interconnects in different edge locations
    ↓
Each interconnect: 2 VLAN attachments
    ↓
Total: 4 attachments across 2 locations
    ↓
Cloud Router: Active/active BGP (all paths used)
```

**Configuration example**:
```bash
# Create two interconnects in different locations
gcloud compute interconnects create interconnect-1 \
  --location=us-east4-zone1 \
  --interconnect-type=DEDICATED

gcloud compute interconnects create interconnect-2 \
  --location=us-east4-zone2 \
  --interconnect-type=DEDICATED

# Create 2 attachments per interconnect (total 4)
for i in 1 2; do
  gcloud compute interconnects attachments create attach-ic1-${i} \
    --region=us-east4 \
    --interconnect=interconnect-1 \
    --router=ha-router \
    --vlan=$((100 + i))

  gcloud compute interconnects attachments create attach-ic2-${i} \
    --region=us-east4 \
    --interconnect=interconnect-2 \
    --router=ha-router \
    --vlan=$((200 + i))
done
```

**May 21, 2024 update**: Partner Interconnect now supports dual-stack (IPv4 + IPv6) configurations

---

## Section 2: HA VPN (~150 lines)

### What is HA VPN?

HA VPN (High Availability VPN) provides **99.99% SLA** for VPN connectivity between your VPC and on-premises network or another VPC. Unlike Classic VPN (99.9% SLA), HA VPN requires redundant tunnels.

From [Create an HA VPN gateway to a peer VPN gateway](https://docs.cloud.google.com/network-connectivity/docs/vpn/how-to/creating-ha-vpn) (accessed 2025-02-03):

**Key differences from Classic VPN**:

| Feature | HA VPN | Classic VPN |
|---------|--------|-------------|
| **SLA** | 99.99% | 99.9% |
| **Tunnels** | 2+ required | 1 minimum |
| **IP addresses** | 2 per gateway | 1 per gateway |
| **Dynamic routing** | Required (Cloud Router + BGP) | Optional (static or dynamic) |
| **Regional** | Yes | Yes |

### HA VPN Architecture

From [HA VPN topologies](https://docs.cloud.google.com/network-connectivity/docs/vpn/concepts/topologies) (accessed 2025-02-03):

**Recommended topology for 99.99% SLA**:
```
VPC Network (GCP)
    ↓
HA VPN Gateway (2 interfaces: interface 0, interface 1)
    ↓                      ↓
Tunnel 1              Tunnel 2
(interface 0)         (interface 1)
    ↓                      ↓
Peer VPN Gateway (2 interfaces)
    ↓
On-premises Network
```

**Active-active vs active-passive**:
- **Active-active**: Both tunnels carry traffic (ECMP routing)
- **Active-passive**: One tunnel primary, one standby (BGP MED/weight)

### Creating HA VPN Gateway

```bash
# 1. Create HA VPN gateway
gcloud compute vpn-gateways create ha-vpn-gateway \
  --network=production-vpc \
  --region=us-central1

# 2. Create Cloud Router (required for dynamic routing)
gcloud compute routers create ha-vpn-router \
  --region=us-central1 \
  --network=production-vpc \
  --asn=65001

# 3. Create external VPN gateway (represent on-prem peer)
gcloud compute external-vpn-gateways create on-prem-gateway \
  --interfaces=0=ON_PREM_IP_1,1=ON_PREM_IP_2

# 4. Create VPN tunnels (2 for 99.99% SLA)
gcloud compute vpn-tunnels create tunnel-0 \
  --region=us-central1 \
  --vpn-gateway=ha-vpn-gateway \
  --vpn-gateway-interface=0 \
  --peer-external-gateway=on-prem-gateway \
  --peer-external-gateway-interface=0 \
  --shared-secret=SECRET_0 \
  --router=ha-vpn-router \
  --ike-version=2

gcloud compute vpn-tunnels create tunnel-1 \
  --region=us-central1 \
  --vpn-gateway=ha-vpn-gateway \
  --vpn-gateway-interface=1 \
  --peer-external-gateway=on-prem-gateway \
  --peer-external-gateway-interface=1 \
  --shared-secret=SECRET_1 \
  --router=ha-vpn-router \
  --ike-version=2

# 5. Configure BGP sessions
gcloud compute routers add-interface ha-vpn-router \
  --interface-name=bgp-if-0 \
  --vpn-tunnel=tunnel-0 \
  --ip-address=169.254.0.1 \
  --mask-length=30 \
  --region=us-central1

gcloud compute routers add-bgp-peer ha-vpn-router \
  --peer-name=bgp-peer-0 \
  --peer-asn=65002 \
  --peer-ip-address=169.254.0.2 \
  --interface=bgp-if-0 \
  --region=us-central1

gcloud compute routers add-interface ha-vpn-router \
  --interface-name=bgp-if-1 \
  --vpn-tunnel=tunnel-1 \
  --ip-address=169.254.1.1 \
  --mask-length=30 \
  --region=us-central1

gcloud compute routers add-bgp-peer ha-vpn-router \
  --peer-name=bgp-peer-1 \
  --peer-asn=65002 \
  --peer-ip-address=169.254.1.2 \
  --interface=bgp-if-1 \
  --region=us-central1
```

### VPN-to-VPN Topology

From [Create HA VPN gateways to connect VPC networks](https://docs.cloud.google.com/network-connectivity/docs/vpn/how-to/creating-ha-vpn2) (accessed 2025-02-03):

**Connect two VPC networks** (same or different projects):

```bash
# Create HA VPN gateway in VPC A
gcloud compute vpn-gateways create vpc-a-gateway \
  --network=vpc-a \
  --region=us-central1 \
  --project=PROJECT_A

# Create HA VPN gateway in VPC B
gcloud compute vpn-gateways create vpc-b-gateway \
  --network=vpc-b \
  --region=us-central1 \
  --project=PROJECT_B

# Create tunnels from VPC A to VPC B
gcloud compute vpn-tunnels create tunnel-a-to-b-0 \
  --region=us-central1 \
  --vpn-gateway=vpc-a-gateway \
  --peer-gcp-gateway=projects/PROJECT_B/regions/us-central1/vpnGateways/vpc-b-gateway \
  --vpn-gateway-interface=0 \
  --peer-gcp-gateway-interface=0 \
  --shared-secret=SHARED_SECRET_0 \
  --router=router-a \
  --project=PROJECT_A

# Repeat for remaining 3 tunnels (4 total for full redundancy)
```

### Migration from Classic VPN

From [Move from Classic VPN to HA VPN](https://cloud.google.com/network-connectivity/docs/vpn/how-to/moving-to-ha-vpn) (accessed 2025-02-03):

**Migration steps**:
1. Create new HA VPN gateway (2 interfaces)
2. Create new Cloud Router (if needed)
3. Create HA VPN tunnels with higher BGP priority
4. Verify BGP routes learned
5. Gradually shift traffic to HA VPN tunnels
6. Delete Classic VPN tunnels after validation

**Zero-downtime migration**:
```bash
# Use BGP MED (Multi-Exit Discriminator) to prefer HA VPN
gcloud compute routers add-bgp-peer ha-router \
  --peer-name=ha-peer \
  --peer-asn=65002 \
  --peer-ip-address=169.254.0.2 \
  --advertised-route-priority=100 \
  --interface=ha-interface \
  --region=us-central1

# Classic VPN peer has higher MED (lower priority)
# Traffic gradually shifts to HA VPN as routes propagate
```

---

## Section 3: Private Service Connect (~100 lines)

### What is Private Service Connect?

Private Service Connect (PSC) allows **private consumption of services** across VPC networks without VPC peering. Consumers access services using internal IP addresses, and service producers can publish services without exposing them publicly.

From [Private Service Connect](https://docs.cloud.google.com/vpc/docs/private-service-connect) (accessed 2025-02-03):

**Key capabilities**:
- **Access Google APIs privately**: BigQuery, Cloud Storage, Vertex AI, etc.
- **Publish services privately**: Expose your own services to consumers
- **Third-party SaaS**: Connect to partner services (e.g., Confluent, Databricks, Elastic)
- **No VPC peering required**: Keeps networks isolated

### Architecture Patterns

**Consumer-side PSC endpoint**:
```
Consumer VPC (10.0.0.0/16)
    ↓
PSC Endpoint (internal IP: 10.0.1.10)
    ↓
Service Attachment (Producer VPC)
    ↓
Load Balancer → Backend VMs/services
```

**Producer-side service attachment**:
```
Producer VPC
    ↓
Internal Load Balancer (forwarding rule)
    ↓
Service Attachment (PSC configuration)
    ↓
Accepts connections from: Consumer projects/networks
```

### Accessing Google APIs via PSC

From [GCP Private Service Connect (PSC)](https://medium.com/google-cloud/gcp-private-service-connect-psc-service-publication-62eaf1d58651) (accessed 2025-02-03):

```bash
# Create PSC endpoint for Google APIs
gcloud compute addresses create psc-googleapis \
  --region=us-central1 \
  --subnet=private-subnet \
  --addresses=10.0.1.10

gcloud compute forwarding-rules create psc-googleapis-rule \
  --region=us-central1 \
  --network=production-vpc \
  --address=psc-googleapis \
  --target-google-apis-bundle=all-apis

# DNS configuration
gcloud dns managed-zones create googleapis-zone \
  --dns-name=googleapis.com \
  --networks=production-vpc \
  --visibility=private

gcloud dns record-sets create "*.googleapis.com." \
  --zone=googleapis-zone \
  --type=A \
  --ttl=300 \
  --rrdatas=10.0.1.10
```

**Supported API bundles**:
- `all-apis`: All Google APIs
- `vpc-sc`: VPC Service Controls compatible APIs only

### Publishing Services with PSC

From [Publish services by using Private Service Connect](https://docs.cloud.google.com/vpc/docs/configure-private-service-connect-producer) (accessed 2025-02-03):

```bash
# 1. Create internal load balancer
gcloud compute forwarding-rules create ilb-forwarding-rule \
  --region=us-central1 \
  --load-balancing-scheme=INTERNAL \
  --network=producer-vpc \
  --subnet=producer-subnet \
  --ip-protocol=TCP \
  --ports=80,443 \
  --backend-service=my-backend-service

# 2. Create service attachment
gcloud compute service-attachments create my-service-attachment \
  --region=us-central1 \
  --producer-forwarding-rule=ilb-forwarding-rule \
  --connection-preference=ACCEPT_AUTOMATIC \
  --nat-subnets=psc-nat-subnet

# 3. Grant access to consumer projects
gcloud compute service-attachments add-iam-policy-binding my-service-attachment \
  --region=us-central1 \
  --member="serviceAccount:CONSUMER_SA@consumer-project.iam.gserviceaccount.com" \
  --role="roles/compute.serviceAttachmentUser"
```

### PSC for Third-Party Services

From [Create a Google Cloud Private Service Connect endpoint](https://docs.confluent.io/cloud/current/networking/private-links/gcp-private-service-connect.html) (accessed 2025-02-03):

**Confluent Cloud example**:
```bash
# 1. Get service attachment from Confluent
# Service attachment: projects/confluent-prod/regions/us-central1/serviceAttachments/kafka-cluster-abc123

# 2. Create PSC endpoint in consumer VPC
gcloud compute addresses create confluent-psc-endpoint \
  --region=us-central1 \
  --subnet=kafka-subnet \
  --addresses=10.0.2.10

gcloud compute forwarding-rules create confluent-connection \
  --region=us-central1 \
  --network=production-vpc \
  --address=confluent-psc-endpoint \
  --target-service-attachment=projects/confluent-prod/regions/us-central1/serviceAttachments/kafka-cluster-abc123
```

**Other supported third-party integrations**:
- **Databricks**: [Enable Private Service Connect for your workspace](https://docs.databricks.com/gcp/en/security/network/classic/private-service-connect) (accessed 2025-02-03)
- **Elastic Cloud**: [Private connectivity with GCP Private Service Connect](https://www.elastic.co/docs/deploy-manage/security/private-connectivity-gcp) (accessed 2025-02-03)
- MongoDB Atlas, Snowflake, Neo4j (via partner service attachments)

---

## Section 4: Hybrid Connectivity Patterns (~100 lines)

### Interconnect + Private Google Access

From [Private Google Access](https://cloud.google.com/vpc/docs/private-google-access) (referenced in gcloud-production/00-networking.md):

**On-premises to Google APIs via Interconnect**:

```bash
# 1. Enable Private Google Access on VPC subnet
gcloud compute networks subnets update us-central1-subnet \
  --region=us-central1 \
  --enable-private-ip-google-access

# 2. Advertise Private Google Access routes via Cloud Router
gcloud compute routers update interconnect-router \
  --region=us-central1 \
  --set-advertisement-mode=custom \
  --set-advertisement-groups=all_subnets \
  --add-advertisement-ranges=199.36.153.8/30  # private.googleapis.com

# 3. On-premises routes traffic to 199.36.153.8/30 via Interconnect
# Traffic reaches Google APIs privately
```

**DNS configuration on-premises**:
```bash
# Configure on-prem DNS to resolve *.googleapis.com to 199.36.153.8/30
# OR use Cloud DNS forwarding zones

gcloud dns managed-zones create googleapis-fwd \
  --dns-name=googleapis.com \
  --networks=production-vpc \
  --visibility=private \
  --forwarding-targets=8.8.8.8,8.8.4.4  # Google Public DNS
```

### VPN + Private Service Connect

**Hybrid architecture** combining HA VPN and PSC:

```
On-premises Network
    ↓
HA VPN (2 tunnels)
    ↓
VPC Network
    ↓
PSC Endpoint → Google APIs (BigQuery, Vertex AI)
```

**Benefits**:
- On-premises apps access Google APIs via VPN
- No public internet exposure
- Centralized VPC manages all connectivity

**Configuration**:
```bash
# 1. Create HA VPN (as shown in Section 2)
# 2. Create PSC endpoint for Google APIs (as shown in Section 3)
# 3. Advertise PSC endpoint IP via BGP

gcloud compute routers add-advertised-ip-ranges interconnect-router \
  --region=us-central1 \
  --ip-range=10.0.1.10/32 \
  --description="PSC endpoint for Google APIs"

# 4. On-premises routes 10.0.1.10/32 via VPN
# 5. Configure on-prem DNS: *.googleapis.com → 10.0.1.10
```

### Multi-Region Hybrid Connectivity

From [Hybrid / On-prem connectivity options to GCP](https://discuss.google.dev/t/hybrid-on-prem-connectivity-options-to-gcp-partner-interconnect-or-cross-cloud-interconnect/172230) (accessed 2025-02-03):

**Best practices for global hybrid connectivity**:

```
On-premises (US East Coast)
    ↓
Partner Interconnect → us-east4 (primary)
    ↓
VPC Network (global)
    ↓
Partner Interconnect → us-west2 (backup)
    ↓
On-premises (US West Coast - DR site)
```

**Failover configuration**:
```bash
# Configure BGP MED for primary/backup
gcloud compute routers add-bgp-peer primary-router \
  --peer-name=primary-peer \
  --peer-asn=65002 \
  --advertised-route-priority=100 \
  --region=us-east4

gcloud compute routers add-bgp-peer backup-router \
  --peer-name=backup-peer \
  --peer-asn=65002 \
  --advertised-route-priority=200 \
  --region=us-west2

# Lower priority = higher preference
# Primary path used unless unavailable
```

### Interconnect vs VPN Decision Matrix

From [Creating a VPN Connection between GCP and AWS](https://medium.com/@oredata-engineering/achieving-high-availability-creating-a-vpn-connection-between-gcp-and-aws-5775c782b8b4) (accessed 2025-02-03):

| Factor | Dedicated Interconnect | Partner Interconnect | HA VPN |
|--------|----------------------|---------------------|---------|
| **Bandwidth** | 10 Gbps, 100 Gbps | 50 Mbps - 50 Gbps | Up to 3 Gbps per tunnel |
| **Latency** | Lowest (~1-2ms) | Low (~2-5ms) | Medium (~10-20ms) |
| **SLA** | 99.99% | 99.99% | 99.99% |
| **Setup time** | 4-6 weeks | 1-2 weeks | Hours |
| **Monthly cost** | $1,750+ | $100-$1,000 | $0.05/hour per tunnel |
| **Use case** | Large enterprise, >5 Gbps | Medium enterprise, 1-10 Gbps | Small/medium, <3 Gbps |

**When to use each**:
- **Dedicated Interconnect**: Mission-critical workloads, >10 Gbps sustained, low latency required
- **Partner Interconnect**: Don't have colocation, need 1-10 Gbps, faster setup
- **HA VPN**: Cost-sensitive, <3 Gbps, quick deployment, encrypted traffic required

---

## Section 5: Performance and Optimization (~50 lines)

### Interconnect Performance Tuning

From [Networking innovations at Google Cloud Next](https://cloud.google.com/blog/products/networking/networking-innovations-at-google-cloud-next25) (accessed 2025-02-03):

**TCP optimization for high-bandwidth links**:
```bash
# Enable TCP window scaling on on-premises side
sysctl -w net.ipv4.tcp_window_scaling=1
sysctl -w net.core.rmem_max=134217728
sysctl -w net.core.wmem_max=134217728
sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"

# Enable BBR congestion control (recommended)
sysctl -w net.core.default_qdisc=fq
sysctl -w net.ipv4.tcp_congestion_control=bbr
```

**VLAN attachment MTU**:
```bash
# Set MTU to 1500 (default) or 8896 (jumbo frames)
gcloud compute interconnects attachments create my-attachment \
  --mtu=8896 \
  --region=us-central1 \
  --interconnect=my-interconnect \
  --router=my-router

# Verify on-premises equipment supports same MTU
```

### VPN Performance Optimization

**Multi-tunnel bonding** (ECMP for bandwidth aggregation):

```bash
# Create 4 tunnels for ~12 Gbps total bandwidth
for i in {0..3}; do
  gcloud compute vpn-tunnels create tunnel-${i} \
    --region=us-central1 \
    --vpn-gateway=ha-vpn-gateway \
    --vpn-gateway-interface=$((i % 2)) \
    --peer-external-gateway=on-prem-gateway \
    --peer-external-gateway-interface=$((i / 2)) \
    --shared-secret=SECRET_${i} \
    --router=ha-vpn-router
done

# Configure Cloud Router for ECMP
gcloud compute routers update ha-vpn-router \
  --region=us-central1 \
  --advertisement-mode=custom \
  --set-custom-advertised-ip-ranges=10.0.0.0/16
```

**IPsec performance**:
- Use AES-128-GCM (faster than AES-256-CBC on modern CPUs)
- IKEv2 preferred over IKEv1 (faster rekey)
- Ensure on-premises VPN appliance supports hardware crypto offload

### Monitoring and Troubleshooting

```bash
# Check Interconnect status
gcloud compute interconnects describe my-interconnect \
  --format="value(state,operationalStatus)"

# Monitor VPN tunnel status
gcloud compute vpn-tunnels describe tunnel-0 \
  --region=us-central1 \
  --format="value(status,detailedStatus)"

# View BGP session status
gcloud compute routers get-status my-router \
  --region=us-central1 \
  --format="json"

# Check PSC endpoint connectivity
gcloud compute forwarding-rules describe psc-endpoint \
  --region=us-central1 \
  --format="value(pscConnectionStatus)"
```

**Common issues**:
- **MTU mismatch**: Causes fragmentation, packet loss
- **BGP flapping**: Unstable routing, check route priority
- **PSC connection failed**: Verify service attachment permissions

---

## Sources

**Source Documents:**
- [gcloud-production/00-networking.md](../gcloud-production/00-networking.md) - Private Google Access section (lines 242-379)

**Web Research:**

Google Cloud Official Documentation (accessed 2025-02-03):
- [Cloud Interconnect overview](https://docs.cloud.google.com/network-connectivity/docs/interconnect/concepts/overview)
- [Cloud Interconnect release notes](https://docs.cloud.google.com/network-connectivity/docs/interconnect/release-notes)
- [Announcement of pricing changes for Cloud Interconnect](https://cloud.google.com/network-connectivity/interconnect-pricing-update-notice)
- [Create an HA VPN gateway to a peer VPN gateway](https://docs.cloud.google.com/network-connectivity/docs/vpn/how-to/creating-ha-vpn)
- [HA VPN topologies](https://docs.cloud.google.com/network-connectivity/docs/vpn/concepts/topologies)
- [Create HA VPN gateways to connect VPC networks](https://docs.cloud.google.com/network-connectivity/docs/vpn/how-to/creating-ha-vpn2)
- [Move from Classic VPN to HA VPN](https://cloud.google.com/network-connectivity/docs/vpn/how-to/moving-to-ha-vpn)
- [Private Service Connect](https://docs.cloud.google.com/vpc/docs/private-service-connect)
- [Publish services by using Private Service Connect](https://docs.cloud.google.com/vpc/docs/configure-private-service-connect-services)
- [Private Service Connect Features and Benefits](https://cloud.google.com/private-service-connect)
- [Private Google Access](https://cloud.google.com/vpc/docs/private-google-access)

Third-Party Documentation (accessed 2025-02-03):
- [Google Cloud Interconnect Overview](https://docs.packetfabric.com/cloud/google/overview/) - PacketFabric
- [Create a Google Cloud Private Service Connect endpoint](https://docs.confluent.io/cloud/current/networking/private-links/gcp-private-service-connect.html) - Confluent
- [Enable Private Service Connect for your workspace](https://docs.databricks.com/gcp/en/security/network/classic/private-service-connect) - Databricks
- [Private connectivity with GCP Private Service Connect](https://www.elastic.co/docs/deploy-manage/security/private-connectivity-gcp) - Elastic

Community Articles (accessed 2025-02-03):
- [GCP Private Service Connect (PSC)](https://medium.com/google-cloud/gcp-private-service-connect-psc-service-publication-62eaf1d58651) - Ishaq Shaikh, Medium
- [Creating a VPN Connection between GCP and AWS](https://medium.com/@oredata-engineering/achieving-high-availability-creating-a-vpn-connection-between-gcp-and-aws-5775c782b8b4) - Oredata Engineering, Medium

Google Cloud Blog (accessed 2025-02-03):
- [Networking innovations at Google Cloud Next](https://cloud.google.com/blog/products/networking/networking-innovations-at-google-cloud-next25)
- [Private connectivity to Vertex workloads](https://cloud.google.com/blog/products/networking/private-connectivity-to-vertex-workloads)

Community Forums (accessed 2025-02-03):
- [Hybrid / On-prem connectivity options to GCP](https://discuss.google.dev/t/hybrid-on-prem-connectivity-options-to-gcp-partner-interconnect-or-cross-cloud-interconnect/172230) - Google Developer Forums

**Additional References:**
- [Cloud VPN Documentation](https://cloud.google.com/network-connectivity/docs/vpn)
- [Cross-Cloud Interconnect](https://cloud.google.com/network-connectivity/docs/interconnect/concepts/cross-cloud-interconnect)
- [Private Service Connect for Google APIs](https://cloud.google.com/vpc/docs/configure-private-service-connect-apis)
