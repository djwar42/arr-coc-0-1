# Dolphie - MySQL/MariaDB Real-Time Monitoring Tool

**Source**: [GitHub - charles-001/dolphie](https://github.com/charles-001/dolphie)
**Accessed**: 2025-11-02
**Version**: 6.10.7
**Stars**: 988
**License**: GPL-3.0-or-later

---

## Overview

Dolphie is a **Textual-based TUI** providing real-time analytics for MySQL/MariaDB and ProxySQL databases. It serves as a "single pane of glass" for comprehensive database monitoring with advanced features like record/replay, daemon mode, and multi-host tab management.

**Tagline**: "Your single pane of glass for real-time analytics into MySQL/MariaDB & ProxySQL"

### Core Capabilities

- **Real-time monitoring** - Live dashboard with 1-second default refresh
- **Multi-panel system** - Dashboard, processlist, graphs, replication, metadata locks, DDL, performance schema metrics, statement summaries
- **ProxySQL support** - Hostgroup summaries, query rules, command stats
- **Record & Replay** - SQLite-based session recording with ZSTD compression for post-incident analysis
- **Daemon Mode** - Headless continuous recording for always-on monitoring
- **Multi-host tabs** - Connect to multiple databases simultaneously with hostgroup configurations
- **Advanced credential management** - Profiles, mysql_config_editor integration, environment variables

---

## Key Features

### Multi-Panel Dashboard System

**Available Panels** (specified via `--panels` flag):

1. **dashboard** - System metrics, server stats, global variables
2. **processlist** - Real-time query execution monitor
3. **graphs** - Visual metrics using plotext (customizable markers)
4. **replication** - Replica status and lag monitoring
5. **metadata_locks** - InnoDB lock analysis
6. **ddl** - DDL operation tracking
7. **pfs_metrics** - Performance Schema metrics
8. **statements_summary** - Query performance analysis
9. **proxysql_hostgroup_summary** - ProxySQL hostgroup stats
10. **proxysql_mysql_query_rules** - ProxySQL query routing
11. **proxysql_command_stats** - ProxySQL command statistics

**Default startup**: `['dashboard', 'processlist']`

### System Utilization Display

When Dolphie runs on the **same host** as the monitored server:

```
╔═══════════════════════════════
║ System Utilization
╠═══════════════════════════════
║ Uptime: 45d 12h 30m
║ CPU: 45.2% (16 cores)
║ Load: 2.1, 1.8, 1.5 (1m, 5m, 15m)
║ Memory: 62% (48GB / 78GB)
║ Swap: 2GB / 16GB
║ Network: ↓ 120 Mbps ↑ 45 Mbps
```

**Powered by**: psutil library for local system metrics

### Record & Replay Architecture

**Recording Mode** (`--record`, `--replay-dir`):
- Data stored in **SQLite database**
- **ZSTD compression** for efficient storage
- Records all panel data in real-time
- Supports live session or daemon mode recording

**Replay Mode** (`--replay-file`):
- Navigate recorded sessions like live monitoring
- Step backward/forward, play/pause, jump to timestamps
- Full review and troubleshooting capabilities
- Some commands restricted, but core functionality preserved

**Use Case**: Post-incident forensics, performance issue investigation, historical analysis

### Daemon Mode

**Purpose**: Headless, always-on monitoring with continuous recording

**Activation**: `--daemon` flag (auto-enables `--record`)

**Characteristics**:
- Removes Textual TUI (no display)
- Creates log file for messages
- Resource-efficient passive monitoring
- Retains 10 minutes of metrics for graphing
- Performance schema deltas reset every 10 minutes

**Recommended Setup**:
- Use `systemctl` for process management
- See [dolphie.service example](https://github.com/charles-001/dolphie/blob/main/examples/dolphie.service)
- See [daemon config example](https://github.com/charles-001/dolphie/blob/main/examples/dolphie-daemon.cnf)

**Control Options**:
- `--daemon-log-file` - Log file path
- `--daemon-panels` - Which panels to query (controls load)
- `--replay-retention-hours` - Data retention (default 48 hours)
- `--refresh-interval` - Collection frequency (default 1 second)

**Example Log Output**:
```
[INFO] Starting Dolphie in daemon mode with a refresh interval of 1s
[INFO] Log file: /var/log/dolphie/dolphie.log
[INFO] Connected to MySQL with Process ID 324
[INFO] Replay SQLite file: /var/lib/dolphie/replays/localhost/daemon.db (24 hours retention)
[INFO] Connected to SQLite
[INFO] Replay database metadata - Host: localhost, Port: 3306, Source: MySQL (Percona Server), Dolphie: 6.3.0
[INFO] ZSTD compression dictionary trained with 10 samples (size: 52.56KB)
[WARNING] Read-only mode changed: R/W -> RO
[INFO] Global variable innodb_io_capacity changed: 1000 -> 2000
```

**Storage Warning**: Replay files can consume significant disk space on busy servers. Adjust `--replay-retention-hours` and `--refresh-interval` accordingly.

---

## Installation

**Python Requirements**: 3.9+

### PyPI
```bash
pip install dolphie
```

### Poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry install
```

### Homebrew
```bash
brew install dolphie
```

### Docker
```bash
docker pull ghcr.io/charles-001/dolphie:latest
docker run -dit --name dolphie ghcr.io/charles-001/dolphie:latest
docker exec -it dolphie dolphie --tab-setup
```

---

## Architecture & Technology Stack

### Core Dependencies

**From [pyproject.toml](https://github.com/charles-001/dolphie/blob/main/pyproject.toml)**:

```toml
textual = "^6.4.0"        # TUI framework
rich = "^14.2.0"          # Terminal formatting
pymysql = "^1.1.2"        # MySQL database connector
plotext = "^5.3.2"        # Terminal-based plotting
zstandard = "^0.25.0"     # Compression for replay files
loguru = "^0.7.3"         # Logging
orjson = "^3.10.15"       # Fast JSON serialization
psutil = "^7.1.2"         # System metrics
sqlparse = "^0.5.3"       # SQL query parsing
requests = "^2.32.5"      # HTTP (version checks)
myloginpath = "^0.0.4"    # MySQL credential file support
packaging = "^25.0"       # Version comparison
```

### Database Integration Patterns

**Connection Methods** (precedence order):
1. **URI string** - `mysql://user:password@host:port`
2. **Command-line options** - `-u`, `-p`, `-h`, `-P`, `-S`
3. **Credential profiles** - Config file sections `[credential_profile_<name>]`
4. **Environment variables** - `DOLPHIE_USER`, `DOLPHIE_PASSWORD`, etc.
5. **Dolphie config** - `~/.dolphie.cnf`, `/etc/dolphie.cnf`
6. **mysql_config_editor** - `~/.mylogin.cnf` (encrypted credentials)
7. **MySQL my.cnf** - `~/.my.cnf` with `[client]` section

**SSL/TLS Support**:
- `--ssl-mode` - REQUIRED/VERIFY_CA/VERIFY_IDENTITY
- `--ssl-ca`, `--ssl-cert`, `--ssl-key` - Certificate paths

**Socket Connections**:
- `-S` / `--socket` - Unix socket file (takes precedence over TCP)

### Real-Time Data Display Architecture

**Refresh Cycle** (default 1 second):
1. **Data Collection** - Query MySQL/ProxySQL for metrics
2. **Processing** - Parse results, calculate deltas
3. **Rendering** - Update Textual widgets
4. **Recording** (if enabled) - Compress and write to SQLite

**Graph Rendering**:
- Uses **plotext** for terminal-based charts
- Customizable markers via `--graph-marker` (default: braille)
- See [marker options](https://tinyurl.com/dolphie-markers)

**Performance Schema Integration**:
- Automatic delta calculation for PFS metrics
- 10-minute reset cycle in daemon mode
- Supports custom queries via panels

**Processlist Features**:
- Real-time query execution tracking
- Option to show only active transactions (`--show-trxs-only`)
- Additional columns toggle (`--additional-columns`)
- Thread killing capability (requires SUPER privilege)

---

## Usage Patterns

### Basic Connection
```bash
# URI connection
dolphie mysql://user:password@host:3306

# Command-line credentials
dolphie -u root -p password -h localhost -P 3306

# Using credential profile
dolphie -C production

# Socket connection
dolphie -S /var/run/mysqld/mysqld.sock
```

### Tab Setup Modal
```bash
# Start with interactive tab setup
dolphie --tab-setup

# Or specify tab setup file
dolphie --tab-setup-file ~/my_dolphie_hosts
```

### Recording Sessions
```bash
# Record live session
dolphie -h localhost --record --replay-dir /var/lib/dolphie/replays

# Replay recorded session
dolphie --replay-file /var/lib/dolphie/replays/localhost/session-2025-11-02.db
```

### Daemon Mode
```bash
# Start daemon with recording
dolphie -h localhost --daemon \
  --daemon-log-file /var/log/dolphie/dolphie.log \
  --replay-dir /var/lib/dolphie/replays \
  --replay-retention-hours 48

# Control which panels run (reduce load)
dolphie -h localhost --daemon \
  --daemon-panels processlist,metadata_locks
```

### Multi-Host Monitoring (Hostgroups)
```bash
# Connect to hostgroup defined in config
dolphie -H cluster1

# Config file example:
[cluster1]
1={"host": "host1", "tab_title": "[yellow]host1[/yellow] :ghost:", "credential_profile": "dev"}
2={"host": "host2", "tab_title": "[blue]host2[/blue]", "credential_profile": "dev"}
3={"host": "host3:3307", "tab_title": "[red]production[/red]", "credential_profile": "prod"}
```

**Tab Naming Features**:
- Rich color syntax: `[red]text[/red]`
- Emoji support: `:ghost:`, `:dolphin:`
- Custom ports: `host:3307`

---

## Credential Management

### Credential Profiles

**Config File Format** (`~/.dolphie.cnf`):
```ini
[credential_profile_dev]
user = dev_user
password = dev_password
host = dev.example.com
port = 3306

[credential_profile_prod]
mycnf_file = /secure/path/to/prod.cnf
ssl_mode = VERIFY_IDENTITY
ssl_ca = /etc/ssl/ca.pem
```

**Usage**: `dolphie -C prod` (omit `credential_profile_` prefix)

**Supported Options**:
- host, port, user, password, socket
- ssl_mode, ssl_ca, ssl_cert, ssl_key
- mycnf_file, login_path

### Hostgroups Configuration

**Purpose**: Connect to multiple hosts at once with tab management

**Config Example**:
```ini
[cluster1]
1={"host": "host1", "tab_title": "[yellow]host1[/yellow] :ghost:", "credential_profile": "dev"}
2={"host": "host2", "tab_title": "[blue]host2[/blue] :ghost:", "credential_profile": "dev"}
3={"host": "host3:3307", "tab_title": "[red]production[/red]", "credential_profile": "prod"}
4={"host": "host4"}
```

**JSON Fields**:
- `host` - Hostname with optional `:port`
- `tab_title` - Rich-formatted tab label (colors, emojis)
- `credential_profile` - Reference to credential profile

**Usage**: `dolphie -H cluster1`

---

## MySQL/MariaDB Support

### Supported Versions

**MySQL/Percona**:
- 5.6, 5.7, 8.x, 9.x
- AWS RDS/Aurora
- Azure MySQL

**MariaDB**:
- 5.5, 10.0, 11.0+
- AWS RDS
- Azure MariaDB

**ProxySQL**:
- 2.6+ (use `admin` user, not `stats` for full features)

### Required Privileges

**Least Privilege**:
1. `PROCESS` (if using processlist via `P` command)
2. `SELECT` on `performance_schema` + heartbeat table (if used)
3. `REPLICATION CLIENT` / `REPLICATION SLAVE`

**Recommended**:
1. `PROCESS`
2. Global `SELECT` (for EXPLAIN queries, database listing)
3. `REPLICATION CLIENT` / `REPLICATION SLAVE`
4. `SUPER` (required for killing queries)

### Replication Monitoring

**Standard**: Uses `SHOW REPLICA STATUS` for lag (`Seconds_Behind_Master`)

**pt-heartbeat Integration**:
```bash
# Use heartbeat table for more accurate lag measurement
dolphie -h replica --heartbeat-table percona.heartbeat
```

**Benefits**: More reliable than `Seconds_Behind_Master` in complex topologies

---

## Advanced Features

### Host Cache File
```bash
# Resolve IPs to hostnames when DNS unavailable
dolphie --host-cache-file ~/dolphie_host_cache

# File format (one per line):
192.168.1.10=db-server-1
192.168.1.11=db-server-2
```

### Global Variable Notifications

Dolphie alerts when global variables change.

**Exclude frequent changes**:
```bash
dolphie --exclude-notify-vars=innodb_buffer_pool_size,max_connections
```

### Panel Customization
```bash
# Start with specific panels
dolphie --panels dashboard,processlist,graphs,replication

# All available panels:
# dashboard, processlist, graphs, replication, metadata_locks, ddl,
# pfs_metrics, statements_summary, proxysql_hostgroup_summary,
# proxysql_mysql_query_rules, proxysql_command_stats
```

### Graph Marker Options
```bash
# Use different plotting markers
dolphie --graph-marker braille   # Default, highest resolution
dolphie --graph-marker dot       # Larger dots
dolphie --graph-marker hd        # High density

# See all options: https://tinyurl.com/dolphie-markers
```

---

## Real-World Use Cases

### Post-Incident Forensics
```bash
# Run daemon mode on production servers
dolphie -h prod-db-01 --daemon \
  --replay-dir /var/lib/dolphie/replays \
  --replay-retention-hours 72 \
  --refresh-interval 1

# After incident, replay session
dolphie --replay-file /var/lib/dolphie/replays/prod-db-01/daemon.db
# Navigate to exact timestamp of incident, review processlist, locks, metrics
```

### Performance Troubleshooting
```bash
# Record live session during performance test
dolphie -h test-db --record --replay-dir ~/replays

# Share replay file with team
dolphie --replay-file ~/replays/test-db/session-2025-11-02-1430.db
```

### Multi-Region Monitoring
```bash
# Monitor all regions simultaneously
dolphie -H global_cluster

# Config:
[global_cluster]
1={"host": "us-east.db", "tab_title": "[green]US East[/green] :earth_americas:", "credential_profile": "prod"}
2={"host": "eu-west.db", "tab_title": "[blue]EU West[/blue] :earth_africa:", "credential_profile": "prod"}
3={"host": "ap-south.db", "tab_title": "[yellow]AP South[/yellow] :earth_asia:", "credential_profile": "prod"}
```

### Replication Health Dashboard
```bash
# Focus on replication across cluster
dolphie -H replicas --panels dashboard,replication,processlist

[replicas]
1={"host": "replica-1", "tab_title": "Replica 1"}
2={"host": "replica-2", "tab_title": "Replica 2"}
3={"host": "replica-3", "tab_title": "Replica 3"}
```

---

## Textual Integration Insights

### Why Dolphie Uses Textual

1. **Rich widgets** - Built-in tables, graphs, status bars
2. **Reactive updates** - Efficient re-rendering for real-time data
3. **Tab management** - Native multi-tab support for hostgroups
4. **Modal dialogs** - Tab Setup modal for interactive configuration
5. **Keyboard commands** - Process switching (`P`), killing queries
6. **Theming** - Consistent visual experience

### Entry Point

**From [pyproject.toml](https://github.com/charles-001/dolphie/blob/main/pyproject.toml)**:
```toml
[tool.poetry.scripts]
dolphie = "dolphie.App:main"
```

Application lives in `dolphie/App.py` with `main()` entry point.

### Daemon Mode Implementation

Daemon mode **removes Textual TUI** entirely:
- No `App.run()` call
- Direct database query loop
- Log-based output (loguru)
- Headless operation

**Why**: Resource efficiency for always-on monitoring without display overhead

---

## Sources

**Primary Documentation**:
- [GitHub Repository](https://github.com/charles-001/dolphie) - Main documentation
- [README.md](https://github.com/charles-001/dolphie/blob/main/README.md) - Comprehensive usage guide
- [pyproject.toml](https://github.com/charles-001/dolphie/blob/main/pyproject.toml) - Dependency information

**Configuration Examples**:
- [dolphie.service](https://github.com/charles-001/dolphie/blob/main/examples/dolphie.service) - systemctl service config
- [dolphie-daemon.cnf](https://github.com/charles-001/dolphie/blob/main/examples/dolphie-daemon.cnf) - Daemon mode config

**Additional Resources**:
- [Homebrew Formula](https://formulae.brew.sh/formula/dolphie)
- [PyPI Package](https://pypi.org/project/dolphie/)
- [Docker Image](https://ghcr.io/charles-001/dolphie)

---

## Key Takeaways for Textual Development

1. **Record/Replay Architecture** - SQLite + ZSTD for session persistence
2. **Daemon Mode Pattern** - TUI-optional design for headless operation
3. **Multi-Host Tabs** - JSON-based hostgroup configuration for complex setups
4. **Real-Time Graphs** - Terminal plotting with plotext integration
5. **Credential Flexibility** - Multiple authentication sources with precedence chain
6. **System Metrics Integration** - psutil for local host monitoring
7. **Database-Specific Features** - Performance Schema, replication lag, metadata locks
8. **Production-Ready** - Used for real-world MySQL/MariaDB monitoring with 988 GitHub stars
