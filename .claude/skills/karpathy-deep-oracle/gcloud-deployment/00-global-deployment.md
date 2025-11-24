# Global Deployment Architecture on GCP

## Overview

Global deployment on Google Cloud Platform enables applications to serve users worldwide with minimal latency through intelligent traffic distribution, CDN integration, and advanced load balancing strategies. This document covers global load balancing, Cloud CDN optimization, traffic management patterns, geo-routing, and disaster recovery for production workloads spanning multiple continents.

**Key Capabilities:**
- **Anycast IP**: Single global IP address routes to nearest backend
- **Edge Network**: Google's private fiber network (100+ PoPs worldwide)
- **Sub-100ms Latency**: 95% of internet users within 100ms of Google edge
- **Automatic Failover**: Health-based traffic redirection across continents
- **CDN Integration**: Cache static content at 200+ edge locations

From [Cloud Load Balancing](https://cloud.google.com/load-balancing) (Google Cloud, accessed 2025-02-03)

From [Traffic management overview for global external Application Load Balancers](https://docs.cloud.google.com/load-balancing/docs/https/traffic-management-global) (Google Cloud documentation, accessed 2025-02-03)

---

## Global HTTP(S) Load Balancer Architecture

### Anycast IP and Edge Network

**How It Works:**
```
User Request (anywhere in world)
    ↓
Anycast IP resolves to nearest Google PoP
    ↓
Google's private network routes to nearest healthy backend
    ↓
Response returns via same optimized path
```

**Benefits:**
- **Single IP Address**: No DNS changes needed for global distribution
- **Automatic Proximity Routing**: Users connect to nearest PoP
- **DDoS Protection**: Built-in Google Cloud Armor integration
- **Premium Tier**: Traffic stays on Google's network (lower latency)

**Architecture:**
```
Global Anycast IP: 203.0.113.1
├── Americas PoP (50+ locations)
│   ├── Routes to us-central1 backends (50ms)
│   └── Failover to us-east1 backends
│
├── Europe PoP (40+ locations)
│   ├── Routes to europe-west1 backends (40ms)
│   └── Failover to europe-north1 backends
│
└── Asia PoP (50+ locations)
    ├── Routes to asia-southeast1 backends (45ms)
    └── Failover to asia-northeast1 backends
```

### Load Balancer Components

**1. Global Forwarding Rule**
```bash
# Create global forwarding rule with anycast IP
gcloud compute forwarding-rules create global-https-rule \
  --global \
  --target-https-proxy=https-proxy \
  --address=global-static-ip \
  --ports=443

# Reserve anycast IP
gcloud compute addresses create global-static-ip \
  --global \
  --ip-version=IPV4
```

**2. URL Map (Traffic Routing)**
```bash
# Create URL map with path-based routing
gcloud compute url-maps create global-url-map \
  --default-service=default-backend

# Add path rules
gcloud compute url-maps add-path-matcher global-url-map \
  --path-matcher-name=api-matcher \
  --default-service=api-backend \
  --path-rules="/api/*=api-backend,/static/*=cdn-backend"
```

**3. Backend Services (Regional)**
```bash
# Create backend service per region
gcloud compute backend-services create us-backend \
  --protocol=HTTP \
  --health-checks=http-health \
  --global \
  --enable-cdn \
  --cache-mode=CACHE_ALL_STATIC

# Add regional instance groups
gcloud compute backend-services add-backend us-backend \
  --instance-group=us-central1-ig \
  --instance-group-zone=us-central1-a \
  --balancing-mode=RATE \
  --max-rate-per-instance=1000 \
  --global
```

From [External Application Load Balancer overview](https://docs.cloud.google.com/load-balancing/docs/https) (Google Cloud documentation, accessed 2025-02-03)

---

## Cloud CDN Integration

### CDN Architecture

**Edge Caching:**
```
User Request
    ↓
Nearest Edge PoP (200+ locations)
    ├── Cache HIT → Return cached content (< 10ms)
    └── Cache MISS → Fetch from origin backend
        ├── Cache at edge
        └── Return to user
```

**Cache Hierarchy:**
```
Edge PoP Layer (200+ locations)
├── Tier 1: Hot content (frequently accessed)
├── Tier 2: Warm content (periodically accessed)
└── Cache MISS → Origin backend
```

### Enable Cloud CDN

```bash
# Enable CDN on backend service
gcloud compute backend-services update cdn-backend \
  --enable-cdn \
  --cache-mode=CACHE_ALL_STATIC \
  --default-ttl=3600 \
  --max-ttl=86400 \
  --client-ttl=3600 \
  --global

# Configure cache key policy
gcloud compute backend-services update cdn-backend \
  --cache-key-include-protocol \
  --cache-key-include-host \
  --cache-key-include-query-string \
  --global
```

### CDN Optimization Best Practices

**1. Cache Static Content**
```
Cacheable:
├── Images (.jpg, .png, .webp)
├── CSS/JS bundles
├── Fonts (.woff2, .ttf)
├── Videos (.mp4, .webm)
└── API responses (with proper headers)

Non-Cacheable:
├── User-specific content (dashboards)
├── Dynamic API endpoints
├── POST/PUT/DELETE requests
└── Content with Set-Cookie headers
```

**2. Cache-Control Headers**
```nginx
# Static assets (1 year)
Cache-Control: public, max-age=31536000, immutable

# API responses (5 minutes)
Cache-Control: public, max-age=300, s-maxage=300

# User-specific content (no cache)
Cache-Control: private, no-cache, no-store, must-revalidate
```

**3. Custom Cache Keys**
```bash
# Cache by device type (mobile/desktop)
gcloud compute backend-services update cdn-backend \
  --cache-key-include-query-string \
  --cache-key-query-string-whitelist="device,version" \
  --global
```

**4. Negative Caching**
```bash
# Cache 404 errors to reduce origin load
gcloud compute backend-services update cdn-backend \
  --negative-caching \
  --negative-caching-policy="404=60,500=10" \
  --global
```

**5. Signed URLs for Private Content**
```python
import datetime
import hashlib
import base64

def sign_url(url, key_name, key, expiration_time):
    """Generate signed URL for Cloud CDN"""
    encoded_url = base64.urlsafe_b64encode(url.encode()).decode()
    expires = int((datetime.datetime.now() +
                   datetime.timedelta(hours=1)).timestamp())

    to_sign = f"URLPrefix={encoded_url}&Expires={expires}&KeyName={key_name}"
    signature = base64.urlsafe_b64encode(
        hashlib.sha256((to_sign + key).encode()).digest()
    ).decode()

    return f"{url}?URLPrefix={encoded_url}&Expires={expires}&KeyName={key_name}&Signature={signature}"
```

From [Content delivery best practices | Cloud CDN](https://docs.cloud.google.com/cdn/docs/best-practices) (Google Cloud documentation, accessed 2025-02-03)

From [Best Practices and Tips for Faster Content Delivery using Google Cloud CDN](https://www.cloudthat.com/resources/blog/best-practices-and-tips-for-faster-content-delivery-using-google-cloud-cdn) (CloudThat, accessed 2025-05-29)

---

## Traffic Management Strategies

### 1. Traffic Splitting (Canary Deployments)

**Use Case**: Gradually roll out new version to small percentage of users

```bash
# Create two backend services (v1 and v2)
gcloud compute backend-services create app-v1 --global
gcloud compute backend-services create app-v2 --global

# Configure weighted traffic split (90% v1, 10% v2)
gcloud compute url-maps add-path-matcher global-url-map \
  --path-matcher-name=canary-matcher \
  --default-service=app-v1 \
  --global

# Update traffic split via URL map
gcloud compute url-maps set-default-service global-url-map \
  --default-service=traffic-split-backend

# Configure backend with traffic weights
gcloud compute backend-services update traffic-split-backend \
  --global \
  --traffic-split=app-v1=0.9,app-v2=0.1
```

**Progressive Rollout:**
```
Phase 1: 95% v1, 5% v2   (1 hour, monitor errors)
Phase 2: 80% v1, 20% v2  (2 hours, check metrics)
Phase 3: 50% v1, 50% v2  (4 hours, validate SLOs)
Phase 4: 0% v1, 100% v2  (full rollout)
```

### 2. Header-Based Routing

**Use Case**: Route beta users to new version

```bash
# Create URL map with header matching
gcloud compute url-maps add-path-matcher global-url-map \
  --path-matcher-name=beta-matcher \
  --default-service=production-backend \
  --header-action="X-User-Type:beta->beta-backend" \
  --global
```

**Application Code (Add Beta Header):**
```python
import requests

# Beta users get routed to beta backend
headers = {"X-User-Type": "beta"} if user.is_beta else {}
response = requests.get("https://api.example.com/endpoint", headers=headers)
```

### 3. Path-Based Routing

**Use Case**: Route different services to different backends

```bash
# Create path-based routing rules
gcloud compute url-maps add-path-matcher global-url-map \
  --path-matcher-name=service-router \
  --default-service=web-backend \
  --path-rules="/api/*=api-backend,/static/*=cdn-backend,/admin/*=admin-backend" \
  --global
```

**URL Routing:**
```
https://example.com/          → web-backend
https://example.com/api/v1    → api-backend
https://example.com/static/*  → cdn-backend (Cloud CDN enabled)
https://example.com/admin/*   → admin-backend (restricted access)
```

### 4. Traffic Mirroring

**Use Case**: Test new backend with production traffic without impacting users

```bash
# Configure traffic mirroring policy
gcloud compute url-maps add-path-matcher global-url-map \
  --path-matcher-name=mirror-matcher \
  --default-service=production-backend \
  --route-action="mirror-backend=test-backend,mirror-percentage=10" \
  --global
```

**Workflow:**
```
User Request
    ├── 100% → Production Backend (responses returned to user)
    └── 10% → Test Backend (responses discarded, logs collected)
```

From [Traffic management overview for global external Application Load Balancers](https://docs.cloud.google.com/load-balancing/docs/https/traffic-management-global) (Google Cloud documentation, accessed 2025-02-03)

---

## Geo-Based Traffic Management

### Cloud DNS Geo Routing

**Use Case**: Route users to nearest region based on location

```bash
# Create geo-routing policy
gcloud dns record-sets create app.example.com \
  --type=A \
  --zone=production-zone \
  --routing-policy-type=GEO \
  --routing-policy-data="us-east1=34.95.111.10;europe-west1=34.107.200.20;asia-southeast1=34.142.100.30" \
  --enable-health-checking
```

**Geographic Routing Rules:**
```
User Location        → Region          → IP Address
├── North America   → us-central1     → 34.95.111.10
├── South America   → southamerica-east1 → 34.95.200.50
├── Europe          → europe-west1    → 34.107.200.20
├── Middle East     → europe-west1    → 34.107.200.20
├── Asia            → asia-southeast1 → 34.142.100.30
└── Australia       → australia-southeast1 → 34.116.100.40
```

### Traffic Director (Service Mesh)

**Global Traffic Management for Microservices**

```yaml
# Traffic Director configuration
apiVersion: networking.gke.io/v1
kind: ServicePolicy
metadata:
  name: global-traffic-policy
spec:
  targetRef:
    kind: Service
    name: my-service
  trafficPolicy:
    loadBalancer:
      simple: ROUND_ROBIN
    outlierDetection:
      consecutiveErrors: 5
      interval: 30s
      baseEjectionTime: 30s
    connectionPool:
      tcp:
        maxConnections: 1000
      http:
        http1MaxPendingRequests: 100
        http2MaxRequests: 1000
```

**Multi-Region Service Discovery:**
```
Traffic Director Control Plane
├── Region 1 (us-central1)
│   ├── Service A (3 instances)
│   └── Service B (2 instances)
│
├── Region 2 (europe-west1)
│   ├── Service A (3 instances)
│   └── Service B (2 instances)
│
└── Region 3 (asia-southeast1)
    ├── Service A (2 instances)
    └── Service B (2 instances)

Client Request → Nearest healthy service instance
```

From [How Traffic Director provides global load balancing for open service mesh](https://cloud.google.com/blog/products/networking/traffic-director-global-traffic-management-for-open-service-mesh) (Google Cloud Blog, accessed 2019-04-17)

---

## Advanced Load Balancing Features

### 1. Session Affinity

**Use Case**: Route users to same backend for session persistence

```bash
# Enable cookie-based session affinity
gcloud compute backend-services update backend \
  --session-affinity=GENERATED_COOKIE \
  --affinity-cookie-ttl=3600 \
  --global
```

**Affinity Types:**
- `NONE`: No affinity (default, best for stateless)
- `CLIENT_IP`: Based on source IP (simple, works with L4)
- `GENERATED_COOKIE`: HTTP cookie (L7 only, most flexible)
- `CLIENT_IP_PROTO`: IP + protocol (L3/L4)

### 2. Connection Draining

**Use Case**: Gracefully remove backends without dropping connections

```bash
# Set connection draining timeout
gcloud compute backend-services update backend \
  --connection-draining-timeout=60 \
  --global
```

**Workflow:**
```
Backend becomes unhealthy
    ├── New connections → route to healthy backends
    └── Existing connections → continue for 60s (draining)
        └── After 60s → forcefully terminate
```

### 3. Custom Request/Response Headers

**Use Case**: Add headers for debugging or routing

```bash
# Add custom headers to requests
gcloud compute url-maps add-path-matcher global-url-map \
  --path-matcher-name=header-matcher \
  --default-service=backend \
  --request-header-action="add=X-Region:us-central1,add=X-LB-Version:v2" \
  --response-header-action="add=X-Cache-Status:HIT,remove=Server" \
  --global
```

**Common Headers:**
```
Request Headers:
├── X-Forwarded-For: Client IP
├── X-Forwarded-Proto: Original protocol (http/https)
├── X-Cloud-Trace-Context: Distributed tracing
└── X-Client-Region: Client's geographic region

Response Headers:
├── X-Cache-Status: HIT/MISS/BYPASS
├── X-Served-By: Backend region
└── X-Response-Time: Backend processing time
```

### 4. CORS Configuration

**Use Case**: Enable cross-origin requests for SPAs

```bash
# Configure CORS policy
gcloud compute backend-services update backend \
  --global \
  --enable-cors \
  --cors-allow-origins="https://example.com,https://app.example.com" \
  --cors-allow-methods="GET,POST,PUT,DELETE" \
  --cors-allow-headers="Content-Type,Authorization" \
  --cors-expose-headers="X-Custom-Header" \
  --cors-max-age=3600
```

---

## Performance Optimization

### Network Tier Selection

**Premium Tier (Default - Best Performance)**
- Traffic stays on Google's global network
- Lower latency (20-30% faster)
- Single global anycast IP
- Best for: Production apps, global users

**Standard Tier (Cost Optimized)**
- Traffic leaves Google network at regional edge
- Regional IPs (no anycast)
- 25% cheaper than Premium
- Best for: Regional apps, cost-sensitive workloads

```bash
# Set project default network tier
gcloud compute project-info update \
  --default-network-tier=PREMIUM

# Create regional IP with Standard tier
gcloud compute addresses create regional-ip \
  --region=us-central1 \
  --network-tier=STANDARD
```

**Performance Comparison:**
| Metric | Premium Tier | Standard Tier |
|--------|-------------|---------------|
| Global Anycast | Yes | No (regional IPs) |
| Latency (avg) | 50-70ms | 80-120ms |
| Network Path | Google backbone | Public internet |
| Cost | Higher | 25% lower |
| DDoS Protection | Built-in | Limited |

### HTTP/2 and HTTP/3 (QUIC)

**Enable Modern Protocols:**
```bash
# Enable HTTP/2
gcloud compute target-https-proxies create https-proxy \
  --url-map=global-url-map \
  --ssl-certificates=ssl-cert \
  --quic-override=ENABLE

# HTTP/3 automatically enabled with QUIC
```

**Benefits:**
- **HTTP/2**: Multiplexing, header compression, server push (40% faster)
- **HTTP/3 (QUIC)**: 0-RTT connection, better mobile performance (60% faster on lossy networks)

### TLS Early Data (0-RTT)

```bash
# Enable TLS 1.3 early data
gcloud compute target-https-proxies update https-proxy \
  --tls-early-data=STRICT
```

**Warning**: Only use for idempotent requests (GET, HEAD) - NOT for state-changing operations

---

## Monitoring and Observability

### Key Metrics

**Load Balancer Metrics:**
```bash
# Create dashboard
gcloud monitoring dashboards create --config-from-file=dashboard.yaml
```

**dashboard.yaml:**
```yaml
displayName: "Global LB Dashboard"
mosaicLayout:
  tiles:
    - widget:
        title: "Request Count by Region"
        xyChart:
          dataSets:
            - timeSeriesQuery:
                timeSeriesFilter:
                  filter: 'resource.type="https_lb_rule" AND metric.type="loadbalancing.googleapis.com/https/request_count"'
                  aggregation:
                    perSeriesAligner: ALIGN_RATE
                    groupByFields: ["resource.region"]

    - widget:
        title: "Backend Latency (p50, p95, p99)"
        xyChart:
          dataSets:
            - timeSeriesQuery:
                timeSeriesFilter:
                  filter: 'metric.type="loadbalancing.googleapis.com/https/backend_latencies"'
                  aggregation:
                    perSeriesAligner: ALIGN_DELTA
                    crossSeriesReducer: REDUCE_PERCENTILE_50

    - widget:
        title: "Error Rate (5xx)"
        xyChart:
          dataSets:
            - timeSeriesQuery:
                timeSeriesFilter:
                  filter: 'metric.type="loadbalancing.googleapis.com/https/request_count" AND metric.response_code_class="500"'
```

**Critical Metrics:**
- `request_count`: Total requests (by region, response code)
- `backend_latencies`: P50/P95/P99 latencies
- `total_latencies`: End-to-end user latency
- `backend_request_count`: Requests per backend
- `serving_capacity`: Backend capacity utilization

### Cloud CDN Metrics

```bash
# CDN cache hit ratio
gcloud monitoring time-series list \
  --filter='metric.type="loadbalancing.googleapis.com/https/external/regional/backend_latencies"' \
  --format=json
```

**CDN Metrics:**
- `cache_hit_count`: Successful cache hits
- `cache_miss_count`: Cache misses (origin requests)
- `total_bytes_sent`: Bandwidth served from CDN
- `cache_fill_bytes`: Data cached from origin

**Target Cache Hit Ratio:**
- Static assets: 95%+ (images, CSS, JS)
- API responses: 60-80% (with proper caching)
- Overall: 80%+ (good), 90%+ (excellent)

### Logging

```bash
# Enable load balancer logging
gcloud compute backend-services update backend \
  --enable-logging \
  --logging-sample-rate=0.1 \
  --global
```

**Log Analysis (BigQuery):**
```sql
-- Top 10 slowest endpoints
SELECT
  httpRequest.requestUrl,
  AVG(httpRequest.latency) as avg_latency,
  COUNT(*) as request_count
FROM `project.dataset.https_lb_logs`
WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
GROUP BY httpRequest.requestUrl
ORDER BY avg_latency DESC
LIMIT 10;

-- CDN cache hit ratio by path
SELECT
  httpRequest.requestUrl,
  COUNTIF(jsonPayload.cacheHit = true) / COUNT(*) as cache_hit_ratio
FROM `project.dataset.https_lb_logs`
WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
GROUP BY httpRequest.requestUrl
HAVING COUNT(*) > 100
ORDER BY cache_hit_ratio ASC;
```

---

## Cost Optimization

### 1. CDN Caching (Reduce Origin Traffic)

**Impact**: 60-80% cost reduction on bandwidth

```
Without CDN:
├── 1TB origin traffic → $0.12/GB → $120/month

With CDN (90% cache hit ratio):
├── 100GB origin traffic → $0.12/GB → $12/month
├── 900GB CDN egress → $0.08/GB → $72/month
└── Total: $84/month (30% savings)
```

### 2. HTTP/2 and Compression

```nginx
# Enable gzip compression
gzip on;
gzip_types text/plain text/css application/json application/javascript;
gzip_min_length 1000;
```

**Savings**: 70% bandwidth reduction for text content

### 3. Image Optimization

```bash
# Serve modern formats (WebP, AVIF)
Cache-Control: public, max-age=31536000
Vary: Accept

# Backend serves appropriate format
if (request.headers['Accept'].includes('image/avif')) {
  return image.avif  // 50% smaller than JPEG
} else if (request.headers['Accept'].includes('image/webp')) {
  return image.webp  // 30% smaller than JPEG
} else {
  return image.jpg
}
```

### 4. Regional Backend Optimization

**Strategy**: Place backends in regions with lowest egress costs

**Egress Pricing (per GB):**
```
Lowest Cost:
├── us-central1 (Iowa): $0.12
├── us-west1 (Oregon): $0.12
└── europe-west4 (Netherlands): $0.12

Higher Cost:
├── asia-south1 (Mumbai): $0.15
├── southamerica-east1 (São Paulo): $0.19
└── australia-southeast1 (Sydney): $0.19
```

**Optimization**: Use CDN to minimize egress, place origins in cheapest regions

---

## Security Best Practices

### 1. Cloud Armor (DDoS Protection)

```bash
# Create security policy
gcloud compute security-policies create global-armor-policy \
  --description="Global DDoS protection"

# Add rate limiting rule
gcloud compute security-policies rules create 1000 \
  --security-policy=global-armor-policy \
  --expression="origin.region_code == 'CN'" \
  --action=rate-based-ban \
  --rate-limit-threshold-count=1000 \
  --rate-limit-threshold-interval-sec=60 \
  --ban-duration-sec=600

# Attach to backend service
gcloud compute backend-services update backend \
  --security-policy=global-armor-policy \
  --global
```

### 2. SSL/TLS Configuration

```bash
# Use Google-managed SSL certificates (automatic renewal)
gcloud compute ssl-certificates create managed-cert \
  --domains=example.com,www.example.com \
  --global

# Attach to HTTPS proxy
gcloud compute target-https-proxies create https-proxy \
  --url-map=global-url-map \
  --ssl-certificates=managed-cert \
  --ssl-policy=modern-ssl-policy
```

**SSL Policy:**
```bash
# Create modern SSL policy (TLS 1.2+)
gcloud compute ssl-policies create modern-ssl-policy \
  --profile=MODERN \
  --min-tls-version=1.2
```

### 3. IAP (Identity-Aware Proxy)

```bash
# Enable IAP for admin backends
gcloud compute backend-services update admin-backend \
  --global \
  --iap=enabled \
  --oauth2-client-id=CLIENT_ID \
  --oauth2-client-secret=CLIENT_SECRET
```

---

## Disaster Recovery

### Global Failover Strategies

**1. Active-Active (Lowest RTO)**
```
All regions actively serving traffic
├── Region 1 failure → automatic failover to regions 2, 3
├── RTO: < 60 seconds
└── RPO: 0 (synchronous replication)
```

**2. Active-Passive (Cost Optimized)**
```
Primary region serves traffic
├── Secondary regions on standby (scaled to 0)
├── Health check failure → scale up secondaries
├── RTO: 5-10 minutes
└── RPO: Minutes (database replication lag)
```

### Health Check Configuration

```bash
# Aggressive health checks for fast failover
gcloud compute health-checks create https global-health-check \
  --check-interval=5s \
  --timeout=3s \
  --unhealthy-threshold=2 \
  --healthy-threshold=1 \
  --request-path=/health \
  --port=443
```

**Failover Timeline:**
```
Backend becomes unhealthy
    ├── T+0s: First failed health check
    ├── T+5s: Second failed health check (marked unhealthy)
    ├── T+10s: Load balancer stops routing to backend
    └── T+15s: Traffic fully migrated to healthy backends
```

---

## Production Checklist

**Pre-Deployment:**
- [ ] Reserve global static IP (anycast)
- [ ] Create SSL certificates (Google-managed or custom)
- [ ] Configure backend services in all target regions
- [ ] Set up health checks (5s interval, 3s timeout)
- [ ] Enable Cloud CDN with proper cache policies
- [ ] Configure Cloud Armor security policies
- [ ] Set up Cloud DNS with geo-routing
- [ ] Test traffic splitting/canary deployment

**Deployment:**
- [ ] Deploy application to all regions
- [ ] Verify health checks passing
- [ ] Enable load balancer with 10% traffic
- [ ] Monitor error rates and latency
- [ ] Gradually increase traffic (canary)
- [ ] Enable Cloud CDN
- [ ] Configure logging (10% sample rate)

**Post-Deployment:**
- [ ] Monitor CDN cache hit ratio (target 80%+)
- [ ] Set up alerting (p99 latency, error rate)
- [ ] Test failover scenarios
- [ ] Review logs in BigQuery
- [ ] Optimize caching policies
- [ ] Review costs (CDN savings, egress costs)

**Monitoring:**
- [ ] Request count by region
- [ ] P50/P95/P99 latency
- [ ] Error rate (4xx, 5xx)
- [ ] CDN cache hit ratio
- [ ] Backend capacity utilization
- [ ] Network egress costs

---

## Common Issues and Solutions

**Issue: High CDN cache miss ratio (< 50%)**

**Solution:**
```bash
# Analyze cache misses
bq query --use_legacy_sql=false '
SELECT
  httpRequest.requestUrl,
  jsonPayload.cacheFillBytes,
  COUNT(*) as miss_count
FROM `project.dataset.https_lb_logs`
WHERE jsonPayload.cacheHit = false
GROUP BY 1, 2
ORDER BY miss_count DESC
LIMIT 20'

# Fix: Add Cache-Control headers, enable cache key customization
gcloud compute backend-services update backend \
  --cache-key-include-query-string \
  --cache-key-query-string-whitelist="version,locale"
```

**Issue: High latency from specific regions**

**Solution:**
```bash
# Check backend distribution
gcloud compute backend-services describe backend --global

# Add backends in underserved regions
gcloud compute instance-groups managed create asia-ig \
  --region=asia-southeast1 \
  --template=app-template \
  --size=3

# Verify Premium Network Tier enabled
gcloud compute project-info describe
```

**Issue: Uneven traffic distribution across backends**

**Solution:**
```bash
# Check balancing mode
gcloud compute backend-services describe backend --global

# Change to RATE-based balancing
gcloud compute backend-services update backend \
  --global \
  --balancing-mode=RATE \
  --max-rate-per-instance=1000
```

**Issue: Slow SSL handshake**

**Solution:**
```bash
# Enable TLS 1.3 and early data
gcloud compute target-https-proxies update https-proxy \
  --tls-early-data=STRICT

# Enable HTTP/3 (QUIC)
gcloud compute target-https-proxies update https-proxy \
  --quic-override=ENABLE
```

---

## Sources

**Google Cloud Documentation:**
- [Cloud Load Balancing](https://cloud.google.com/load-balancing) (accessed 2025-02-03)
- [Traffic management overview for global external Application Load Balancers](https://docs.cloud.google.com/load-balancing/docs/https/traffic-management-global) (accessed 2025-02-03)
- [External Application Load Balancer overview](https://docs.cloud.google.com/load-balancing/docs/https) (accessed 2025-02-03)
- [Content delivery best practices | Cloud CDN](https://docs.cloud.google.com/cdn/docs/best-practices) (accessed 2025-02-03)
- [Multi-regional deployment on Compute Engine](https://docs.cloud.google.com/architecture/multiregional-vms) (accessed 2025-02-03)

**Web Research:**
- [How Traffic Director provides global load balancing for open service mesh](https://cloud.google.com/blog/products/networking/traffic-director-global-traffic-management-for-open-service-mesh) (Google Cloud Blog, accessed 2019-04-17)
- [Best Practices and Tips for Faster Content Delivery using Google Cloud CDN](https://www.cloudthat.com/resources/blog/best-practices-and-tips-for-faster-content-delivery-using-google-cloud-cdn) (CloudThat, accessed 2023-05-29)
- [A Comprehensive Guide to Load Balancing in Google Cloud Platform (GCP)](https://srivastavayushmaan1347.medium.com/a-comprehensive-guide-to-load-balancing-in-google-cloud-platform-gcp-90b4b3905169) (Medium, accessed 2023)

**Related Documentation:**
- See [gcloud-production/04-multi-region-deployment.md](../gcloud-production/04-multi-region-deployment.md) for regional deployment patterns
- See [gcloud-production/00-networking.md](../gcloud-production/00-networking.md) for VPC configuration
- See [gcloud-production/02-billing-cost-analysis.md](../gcloud-production/02-billing-cost-analysis.md) for cost optimization
