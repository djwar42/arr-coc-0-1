# GCloud Output Formatting and Data Extraction

## Overview

The gcloud CLI provides powerful output formatting and filtering capabilities that transform command output for automation, scripting, and data processing. This document covers the `--format`, `--filter`, and `--flatten` flags, plus integration with external tools like `jq`.

**Key Concept**: gcloud's output system operates in two phases:
1. **Filtering** (`--filter`) - Server or client-side data filtering
2. **Formatting** (`--format`) - Transform filtered results into desired output format

---

## Section 1: Output Formats (~150 lines)

### Available Format Types

The `--format` flag controls output structure. All gcloud commands support these formats:

```bash
# Core formats
--format=json          # JSON output (machine-readable)
--format=yaml          # YAML output (human and machine-readable)
--format=csv           # CSV output (spreadsheet-compatible)
--format=table         # Formatted table (default for interactive)
--format=value(KEY)    # Extract single value
--format=text          # Plain text key-value pairs
--format=list          # Bulleted list format
--format=flattened     # Flatten nested structures
```

### JSON Format

**Use Case**: Machine processing, API integration, scripting

```bash
# List compute instances as JSON
gcloud compute instances list --format=json

# Output structure
[
  {
    "name": "instance-1",
    "zone": "us-central1-a",
    "machineType": "n1-standard-1",
    "status": "RUNNING",
    "networkInterfaces": [...]
  }
]
```

**Advantages**:
- Full structure preservation
- Easy parsing with `jq`, Python, or other tools
- Ideal for CI/CD pipelines

**Tip**: Combine with `jq` for powerful extraction:
```bash
gcloud compute instances list --format=json | jq '.[].name'
```

### YAML Format

**Use Case**: Configuration files, human-readable output, GitOps

```bash
# Describe instance in YAML
gcloud compute instances describe INSTANCE_NAME --format=yaml

# Output
name: instance-1
zone: https://www.googleapis.com/compute/v1/projects/PROJECT/zones/us-central1-a
machineType: n1-standard-1
status: RUNNING
networkInterfaces:
  - network: default
    accessConfigs:
      - natIP: 34.123.45.67
```

**Advantages**:
- More readable than JSON for humans
- Comments supported (when editing)
- Direct use in Kubernetes/Terraform configs

### CSV Format

**Use Case**: Spreadsheet import, quick data analysis, reporting

```bash
# Export instance data to CSV
gcloud compute instances list \
  --format="csv(name,zone.basename(),machineType.basename(),status)"

# Output
name,zone,machineType,status
instance-1,us-central1-a,n1-standard-1,RUNNING
instance-2,us-west1-b,n1-standard-2,TERMINATED
```

**Key Features**:
- Specify columns with projection syntax
- No header row: `--format="csv[no-heading](...)`
- Custom separator: Not directly supported (pipe through sed/awk)

### Table Format

**Use Case**: Interactive terminal display, human readability

```bash
# Default table output
gcloud compute instances list

# Custom table columns
gcloud compute instances list \
  --format="table(name,zone.basename(),status,networkInterfaces[0].accessConfigs[0].natIP:label=EXTERNAL_IP)"
```

**Output**:
```
NAME        ZONE           STATUS   EXTERNAL_IP
instance-1  us-central1-a  RUNNING  34.123.45.67
instance-2  us-west1-b     RUNNING  35.234.56.78
```

**Table Customization**:
- `:label=CUSTOM_HEADER` - Custom column header
- `:align=right` - Right-align column
- `:sort=2` - Sort by column index

### Value Format

**Use Case**: Extract single field for shell variables

```bash
# Get just the project ID
PROJECT=$(gcloud config get-value project --format="value()")

# Get specific field from list
ZONE=$(gcloud compute instances list \
  --filter="name=instance-1" \
  --format="value(zone.basename())")

echo "Instance is in zone: $ZONE"
```

**Multiple Values**:
```bash
# Get multiple fields as tab-separated
gcloud compute instances list \
  --format="value(name,zone.basename(),status)"
```

### Flattened Format

**Use Case**: Discover available fields, debug nested structures

```bash
# Show all available fields with their JSON paths
gcloud compute instances describe INSTANCE_NAME --format=flattened

# Output
name:                              instance-1
zone:                              us-central1-a
machineType:                       n1-standard-1
networkInterfaces[0].network:      default
networkInterfaces[0].networkIP:    10.128.0.2
networkInterfaces[0].accessConfigs[0].natIP: 34.123.45.67
```

**Tip**: Use flattened format to derive JSON paths for projections!

---

## Section 2: Filter Expressions (~200 lines)

### Filter Syntax Basics

The `--filter` flag uses a specialized expression language to select resources:

```bash
# Basic equality
gcloud compute instances list --filter="name=instance-1"

# Pattern matching
gcloud compute instances list --filter="name~'^web-.*'"

# Comparison operators
gcloud compute instances list --filter="creationTimestamp>'2024-01-01'"

# Multiple conditions (AND)
gcloud compute instances list --filter="zone:us-central1-a AND status=RUNNING"

# OR conditions
gcloud compute instances list --filter="status=RUNNING OR status=STOPPED"
```

### Comparison Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `=` | Equals | `name=my-instance` |
| `!=` | Not equals | `status!=TERMINATED` |
| `:` | Has substring | `zone:us-central1` |
| `~` | Regex match | `name~'^prod-.*'` |
| `<` | Less than | `creationTimestamp<'2024-01-01'` |
| `<=` | Less than or equal | `diskSizeGb<=100` |
| `>` | Greater than | `diskSizeGb>500` |
| `>=` | Greater than or equal | `creationTimestamp>='2024-01-01'` |

### Logical Operators

```bash
# AND (both conditions must be true)
--filter="status=RUNNING AND zone:us-central1"

# OR (either condition must be true)
--filter="status=RUNNING OR status=STOPPED"

# NOT (negate condition)
--filter="NOT status=TERMINATED"

# Complex combinations
--filter="(zone:us-central1 OR zone:us-west1) AND status=RUNNING"
```

### Nested Field Filtering

Access nested fields using dot notation:

```bash
# Filter by nested field
gcloud compute instances list \
  --filter="networkInterfaces[0].accessConfigs[0].natIP=34.123.45.67"

# Filter by metadata labels
gcloud compute instances list \
  --filter="labels.env=production"

# Check if field exists
gcloud compute instances list \
  --filter="scheduling.preemptible=true"
```

### Advanced Filter Examples

**Find instances by label**:
```bash
gcloud compute instances list \
  --filter="labels.environment=production AND labels.team=backend"
```

**Find recently created resources**:
```bash
# Instances created in last 24 hours
gcloud compute instances list \
  --filter="creationTimestamp>$(date -u -d '24 hours ago' '+%Y-%m-%dT%H:%M:%S')"
```

**Pattern matching for naming conventions**:
```bash
# All instances starting with "web-" or "api-"
gcloud compute instances list \
  --filter="name~'^(web|api)-.*'"
```

**Multi-region filtering**:
```bash
# Instances in any us-central1 zone
gcloud compute instances list \
  --filter="zone:us-central1"
```

**Complex production resource filtering**:
```bash
gcloud compute instances list \
  --filter="labels.env=production AND status=RUNNING AND zone:(us-central1 OR us-east1)"
```

### Filter Gotchas

**String Quoting**:
```bash
# Correct: Quote the entire filter expression
--filter="name=instance-1"

# Wrong: Don't quote individual values unless they contain spaces
--filter='name="instance-1"'  # May cause issues

# Exception: Values with spaces need quotes
--filter="description:'my test instance'"
```

**Regex Escaping**:
```bash
# Regex patterns need proper escaping
--filter='name~"^prod-\d+"'  # Single quotes to protect backslashes
```

**Date Formats**:
```bash
# ISO 8601 format required for timestamps
--filter="creationTimestamp>'2024-01-15T10:30:00Z'"
```

---

## Section 3: Flatten and Projection (~150 lines)

### Understanding Projections

Projections select specific fields from output using dot notation:

```bash
# Select specific fields in table format
gcloud compute instances list \
  --format="table(name, zone.basename(), status)"

# Select nested fields
gcloud compute instances list \
  --format="table(name, networkInterfaces[0].networkIP)"
```

**Projection Syntax**:
- `field` - Top-level field
- `nested.field` - Nested field
- `array[0].field` - Array element field
- `field.basename()` - Extract basename from URL

### Flatten Flag

The `--flatten` flag expands nested arrays into separate rows:

```bash
# Without flatten: One row per instance
gcloud compute instances list

# With flatten: One row per disk attached to instance
gcloud compute instances list \
  --flatten="disks[]" \
  --format="table(name, disks.source.basename())"
```

**Example Output**:
```
NAME        DISK
instance-1  boot-disk
instance-1  data-disk-1
instance-1  data-disk-2
instance-2  boot-disk
```

### Combining Flatten with Projections

**List all network interfaces across instances**:
```bash
gcloud compute instances list \
  --flatten="networkInterfaces[]" \
  --format="table(name, networkInterfaces.networkIP, networkInterfaces.accessConfigs[0].natIP)"
```

**Expand IAM policy bindings**:
```bash
gcloud projects get-iam-policy PROJECT_ID \
  --flatten="bindings[].members[]" \
  --format="table(bindings.role, bindings.members)"
```

**Output**:
```
ROLE                        MEMBER
roles/owner                 user:admin@example.com
roles/editor                user:dev1@example.com
roles/editor                user:dev2@example.com
roles/viewer                serviceAccount:sa@project.iam.gserviceaccount.com
```

### Projection Functions

Built-in functions transform field values:

```bash
# basename() - Extract last component from URL
--format="table(name, zone.basename())"
# Output: us-central1-a (instead of full URL)

# scope() - Extract scope from resource URL
--format="table(name, scope())"

# date() - Format timestamp
--format="table(name, creationTimestamp.date('%Y-%m-%d'))"

# size() - Get array/list length
--format="table(name, disks.size())"
```

### Complex Projection Examples

**Multi-level nested extraction**:
```bash
gcloud compute instances list \
  --format="table(
    name,
    zone.basename(),
    networkInterfaces[0].networkIP:label=INTERNAL_IP,
    networkInterfaces[0].accessConfigs[0].natIP:label=EXTERNAL_IP,
    status
  )"
```

**Conditional field display**:
```bash
# Show preemptible status
gcloud compute instances list \
  --format="table(
    name,
    zone.basename(),
    scheduling.preemptible.yesno(yes='PREEMPTIBLE', no='STANDARD'):label=TYPE
  )"
```

**Derive JSON paths from flattened output**:
```bash
# Step 1: Use flattened to see all available fields
gcloud compute instances describe INSTANCE_NAME --format=flattened

# Step 2: Use discovered paths in projections
gcloud compute instances list \
  --format="csv(name, networkInterfaces[0].accessConfigs[0].natIP)"
```

---

## Section 4: jq Integration (~150 lines)

### Why jq with gcloud?

While gcloud has powerful built-in formatting, `jq` provides:
- **More complex transformations** than gcloud projections
- **Data manipulation** (math, string operations, conditionals)
- **Multi-stage processing** pipelines
- **Familiar JSON processing** for developers

**Basic Pattern**:
```bash
gcloud COMMAND --format=json | jq 'EXPRESSION'
```

### Essential jq Patterns for gcloud

**Extract array of names**:
```bash
gcloud compute instances list --format=json | jq '.[].name'
# Output
"instance-1"
"instance-2"
"instance-3"
```

**Filter in jq (alternative to --filter)**:
```bash
gcloud compute instances list --format=json | \
  jq '.[] | select(.status == "RUNNING") | .name'
```

**Transform to custom JSON structure**:
```bash
gcloud compute instances list --format=json | \
  jq '.[] | {
    instance: .name,
    zone: (.zone | split("/")[-1]),
    ip: .networkInterfaces[0].networkIP,
    running: (.status == "RUNNING")
  }'
```

**Output**:
```json
{
  "instance": "instance-1",
  "zone": "us-central1-a",
  "ip": "10.128.0.2",
  "running": true
}
```

### jq for Data Aggregation

**Count instances by zone**:
```bash
gcloud compute instances list --format=json | \
  jq 'group_by(.zone) | map({
    zone: (.[0].zone | split("/")[-1]),
    count: length
  })'
```

**Sum disk sizes**:
```bash
gcloud compute disks list --format=json | \
  jq '[.[].sizeGb | tonumber] | add'
```

**Find max/min values**:
```bash
# Find largest disk
gcloud compute disks list --format=json | \
  jq 'max_by(.sizeGb | tonumber) | {name, sizeGb}'
```

### jq for Complex Filtering

**Multiple conditions**:
```bash
gcloud compute instances list --format=json | \
  jq '.[] | select(
    .status == "RUNNING" and
    (.zone | contains("us-central1")) and
    .machineType | contains("n1-standard")
  )'
```

**Regex filtering**:
```bash
# Find instances matching naming pattern
gcloud compute instances list --format=json | \
  jq '.[] | select(.name | test("^(web|api)-prod-\\d+$"))'
```

**Nested field filtering**:
```bash
# Find instances with external IPs
gcloud compute instances list --format=json | \
  jq '.[] | select(
    .networkInterfaces[0].accessConfigs != null and
    (.networkInterfaces[0].accessConfigs | length) > 0
  )'
```

### jq for Data Transformation

**Flatten nested arrays**:
```bash
gcloud compute instances list --format=json | \
  jq '.[] | {
    name,
    disks: [.disks[].source | split("/")[-1]]
  }'
```

**Combine multiple gcloud outputs**:
```bash
# Combine instances and disks data
instances=$(gcloud compute instances list --format=json)
disks=$(gcloud compute disks list --format=json)

jq -n --argjson inst "$instances" --argjson dsks "$disks" '{
  instances: $inst,
  disks: $dsks,
  summary: {
    total_instances: ($inst | length),
    total_disks: ($dsks | length)
  }
}'
```

**CSV conversion with custom fields**:
```bash
gcloud compute instances list --format=json | \
  jq -r '.[] | [
    .name,
    (.zone | split("/")[-1]),
    .status,
    .networkInterfaces[0].networkIP // "N/A"
  ] | @csv'
```

### jq Performance Tips

**Use select early** (filter before processing):
```bash
# Bad: Process all, then filter
gcloud compute instances list --format=json | \
  jq 'map(transform) | .[] | select(.status == "RUNNING")'

# Good: Filter first
gcloud compute instances list --format=json | \
  jq '.[] | select(.status == "RUNNING") | transform'
```

**Prefer gcloud --filter** (server-side filtering):
```bash
# Better: Let gcloud filter on server
gcloud compute instances list --filter="status=RUNNING" --format=json | \
  jq '...'
```

---

## Section 5: Scripting Patterns and Best Practices (~100 lines)

### Scripting Best Practices

**1. Always specify --format for scripts**:
```bash
# Don't rely on default format (may change)
gcloud compute instances list

# Do specify explicit format
gcloud compute instances list --format=json
```

**2. Use value() for shell variables**:
```bash
#!/bin/bash
PROJECT=$(gcloud config get-value project --format="value()")
ZONE=$(gcloud compute instances list \
  --filter="name=$INSTANCE_NAME" \
  --format="value(zone.basename())")

echo "Operating in project: $PROJECT, zone: $ZONE"
```

**3. Handle empty results**:
```bash
# Check if results exist
instances=$(gcloud compute instances list --format=json)
if [ "$instances" = "[]" ]; then
  echo "No instances found"
  exit 0
fi
```

**4. Error handling with --format**:
```bash
#!/bin/bash
set -e  # Exit on error

# Capture stderr
result=$(gcloud compute instances list --format=json 2>&1) || {
  echo "Error: $result"
  exit 1
}

echo "$result" | jq '...'
```

### Common Scripting Patterns

**Loop through resources**:
```bash
#!/bin/bash
gcloud compute instances list --format="value(name)" | while read instance; do
  echo "Processing $instance..."
  gcloud compute instances describe "$instance" --format=json | \
    jq '.status'
done
```

**Batch operations with parallel**:
```bash
# Install: apt-get install parallel
gcloud compute instances list --format="value(name)" | \
  parallel -j 10 "gcloud compute instances describe {} --format=json"
```

**Create lookup tables**:
```bash
# Build associative array of instance -> zone
declare -A instance_zones
while IFS=$'\t' read -r name zone; do
  instance_zones["$name"]="$zone"
done < <(gcloud compute instances list --format="value(name,zone.basename())")

# Use lookup
echo "Instance web-1 is in ${instance_zones[web-1]}"
```

**Export to multiple formats**:
```bash
#!/bin/bash
INSTANCE_NAME="my-instance"

# Export as JSON for processing
gcloud compute instances describe "$INSTANCE_NAME" \
  --format=json > instance.json

# Export as YAML for human review
gcloud compute instances describe "$INSTANCE_NAME" \
  --format=yaml > instance.yaml

# Export key fields as CSV
gcloud compute instances list \
  --filter="name=$INSTANCE_NAME" \
  --format="csv(name,zone.basename(),status,machineType.basename())" \
  > instance.csv
```

### Performance Optimization

**1. Limit fields retrieved**:
```bash
# Slow: Retrieve all fields
gcloud compute instances list --format=json

# Fast: Retrieve only needed fields
gcloud compute instances list --format="json(name,zone,status)"
```

**2. Use server-side filtering**:
```bash
# Slow: Client-side filtering
gcloud compute instances list --format=json | jq '.[] | select(.zone | contains("us-central1"))'

# Fast: Server-side filtering
gcloud compute instances list --filter="zone:us-central1" --format=json
```

**3. Cache results when iterating**:
```bash
# Cache expensive query
instances=$(gcloud compute instances list --format=json)

# Reuse cached data
echo "$instances" | jq '...'
echo "$instances" | jq '...'  # No re-query
```

### Debugging Tips

**Discover available fields**:
```bash
# See all fields with flattened
gcloud compute instances describe INSTANCE_NAME --format=flattened

# Or as JSON with pretty print
gcloud compute instances describe INSTANCE_NAME --format=json | jq '.'
```

**Test filters incrementally**:
```bash
# Start simple
gcloud compute instances list --filter="name=instance-1"

# Add conditions incrementally
gcloud compute instances list --filter="name=instance-1 AND status=RUNNING"
```

**Validate jq expressions**:
```bash
# Test jq without gcloud
echo '[{"name":"test","value":42}]' | jq '.[].name'

# Then apply to gcloud output
gcloud compute instances list --format=json | jq '.[].name'
```

---

## Sources

**Web Research:**

From [Scripting gcloud CLI commands](https://docs.cloud.google.com/sdk/docs/scripting-gcloud) (Google Cloud SDK docs, accessed 2025-02-03):
- Output format options: json, yaml, csv, table, value, text, list, flattened
- Projection syntax and field selection
- Value extraction for shell variables

From [Filtering and formatting fun with gcloud](https://cloud.google.com/blog/products/management-tools/filtering-and-formatting-fun-with) (Google Cloud Blog, accessed 2025-02-03):
- Filter expression syntax and operators
- Combining --filter with --format
- Deriving JSON paths from flattened output
- Table format customization

From [gcloud beta topic formats](https://cloud.google.com/sdk/gcloud/reference/beta/topic/formats) (Google Cloud SDK Reference, accessed 2025-02-03):
- Complete format specification reference
- Projection functions (basename, scope, date, size)
- Advanced formatting options

From [Bash hacks gcloud, kubectl, jq](https://medium.com/google-cloud/bash-hacks-gcloud-kubectl-jq-etc-c2ff351d9c3b) (Medium article by Daz Wilkin, accessed 2025-02-03):
- gcloud + jq integration patterns
- Shell scripting best practices
- Flatten flag usage examples

From [jq 1.8 Manual](https://jqlang.org/manual/) (jq official documentation, accessed 2025-02-03):
- jq filter syntax and operators
- Data aggregation functions (group_by, max_by, add)
- Array and object transformation
- CSV output formatting (@csv)

From [How To Transform JSON Data with jq](https://www.digitalocean.com/community/tutorials/how-to-transform-json-data-with-jq) (DigitalOcean tutorial, accessed 2025-02-03):
- jq basic patterns for API data extraction
- Complex filtering with select()
- Multi-stage pipeline processing

**Additional References:**
- [gcloud compute usage tips](https://docs.cloud.google.com/compute/docs/gcloud-compute/tips) - Google Cloud official docs
- [gcloud topic projections](https://fig.io/manual/gcloud/topic/projections) - Fig.io CLI reference
- [The gcloud CLI cheat sheet](https://docs.cloud.google.com/sdk/docs/cheatsheet) - Google Cloud SDK docs
