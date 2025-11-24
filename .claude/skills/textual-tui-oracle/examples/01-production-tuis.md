# Production TUI Applications: Real-World Examples

**Domain**: Enterprise TUI Applications
**Created**: 2025-11-02
**Source**: Web research (Janssen + Doppler documentation)

## Overview

This document examines two production-grade TUI applications: Janssen TUI (identity/auth server configuration) and Doppler TUI (secrets management). Both demonstrate professional approaches to complex system administration through terminal interfaces.

---

## Janssen TUI - Identity Server Configuration

**URL**: https://jans.io/docs/head/janssen-server/config-guide/config-tools/jans-tui/
**Purpose**: Text-based administrative interface for Janssen Project (OpenID/OAuth/UMA server)
**Language**: Python
**Access Date**: 2025-11-02

### Core Architecture

**Technology Stack:**
- Built with Python 3
- OAuth Device Flow for authentication
- REST API backend (Jans Config API)
- Distributed as `.pyz` self-executable file (via shiv)
- Plugin-based extensible architecture

**Installation Locations:**
```bash
# Standard VM installation
sudo /opt/jans/jans-cli/jans_cli_tui.py

# Standalone with pip
pip3 install https://github.com/JanssenProject/jans/archive/refs/heads/main.zip#subdirectory=jans-cli-tui

# Execute standalone
jans-cli-tui
```

**Configuration Storage:**
- Client credentials: `~/.config/jans-cli.ini` (encoded format)
- Command logs: `<log-dir>/cli_cmd.log` (all write operations logged)
- Tokens and auth data stored securely in config file

### UI Structure & Navigation

**Main Panel Organization:**

```
╔═══════════════════════════════════════════════════════════
║ JANSSEN TUI - Main Panel
╠═══════════════════════════════════════════════════════════
║
║  • Auth Server         ← OAuth/OpenID/UMA configuration
║  • FIDO                ← FIDO 2 and U2F settings
║  • SCIM                ← Identity management config
║  • [Plugin Sections]   ← user-mgt, scim, fido2, admin-ui
║
╚═══════════════════════════════════════════════════════════
```

**Navigation Patterns:**
- Hierarchical menu system (main → section → detail)
- List views with enter to drill down
- Upper-right action buttons ("Add Client", "Edit", etc.)
- Tab navigation between panels
- Return/back navigation to previous screens

### Feature Areas

#### 1. Auth Server Management

**Client Management:**
```
List View:
╔═══════════════════════════════════════════════════════════
║ Existing Clients
╠═══════════════════════════════════════════════════════════
║ Client ID                    │ Name           │ Type
║ ────────────────────────────────────────────────────────
║ 1000.abc-123-def             │ Web App 1      │ OAuth
║ 2000.xyz-789-ghi             │ Mobile App     │ OpenID
║                              │                │
║ [Add Client]                 │                │ [View]
╚═══════════════════════════════════════════════════════════

Detail View (press Enter on client):
╔═══════════════════════════════════════════════════════════
║ Client Details: Web App 1
╠═══════════════════════════════════════════════════════════
║ Client ID:        1000.abc-123-def
║ Client Secret:    [REDACTED]
║ Grant Types:      authorization_code, refresh_token
║ Redirect URIs:    https://example.com/callback
║ Scopes:          openid, profile, email
║ Token Endpoint:   client_secret_basic
║
║ [Edit] [Delete] [Back]
╚═══════════════════════════════════════════════════════════
```

**Configuration Sections:**
- OpenID Connect client configuration
- OAuth scope management
- JSON Web Key (JWK) management
- Authentication method configuration
- Server property configuration
- Logging configuration
- SSA (Software Statement Assertion) config
- Agama project configuration
- Attribute configuration
- Cache configuration
- Rate limiting
- UMA resource management
- Session management

#### 2. FIDO Configuration

**Two-Tier Configuration System:**

```
Dynamic Configuration:
╔═══════════════════════════════════════════════════════════
║ FIDO Dynamic Configuration
╠═══════════════════════════════════════════════════════════
║ Issuer:                    https://example.com
║ Base Endpoint:             /fido2/restv1
║ Clean Service Interval:    60
║ Clean Service Batch Size:  100
║ Use Super Gluu:            [✓] Enabled
║ Authenticator Certs Folder: /etc/certs
║
║ [Save] [Reset] [Back]
╚═══════════════════════════════════════════════════════════

Static Configuration:
╔═══════════════════════════════════════════════════════════
║ FIDO Static Configuration
╠═══════════════════════════════════════════════════════════
║ Base Directory:   /etc/jans/conf/fido2
║ Server Metadata:  [View File]
║ MDS TOC Certs:    [Manage Certificates]
║ MDS TOC URL:      https://mds.fidoalliance.org
║
║ [View] [Back]
╚═══════════════════════════════════════════════════════════
```

#### 3. SCIM Configuration

**Identity Management Settings:**

```
╔═══════════════════════════════════════════════════════════
║ SCIM Configuration
╠═══════════════════════════════════════════════════════════
║ User Management:
║   Max Count:           200
║   Bulk Max Operations: 30
║   Bulk Max Payload:    3072000
║
║ Protection:
║   Mode:               OAuth
║   Test Mode:          [✗] Disabled
║
║ Schema Extensions:
║   User Extension:     [View Schema]
║   Group Extension:    [View Schema]
║
║ [Save] [Reset] [Back]
╚═══════════════════════════════════════════════════════════
```

### Plugin Architecture

**Plugin System:**
- Plugins loaded dynamically from `/opt/jans/jans-cli/plugins`
- Numeric priority-based folder ordering
- Configuration in `jans-cli.ini`:
  ```ini
  jca_plugins = user-mgt,scim,fido2,admin-ui
  ```

**Plugin Capabilities:**
- Extend main menu with new sections
- Add custom configuration panels
- Integrate with Config API endpoints
- Custom UI components and workflows

### Command Line Integration

**Logging All Operations:**

Every write operation (POST/PUT/PATCH) logged to `cli_cmd.log`:

```bash
# Example: Creating a user via TUI generates this command log
/usr/bin/python3 /opt/jans/jans-cli/cli/config_cli.py \
  --operation-id post-user \
  --data '{
    "customObjectClasses": ["top", "jansPerson"],
    "customAttributes": [
      {"name": "middleName", "multiValued": false, "values": [""]},
      {"name": "sn", "multiValued": false, "values": ["Watts"]},
      {"name": "nickname", "multiValued": false, "values": [""]}
    ],
    "mail": "ewatts@foo.org",
    "userId": "ewatts",
    "displayName": "Emelia Watts",
    "givenName": "Emelia",
    "userPassword": "TopSecret",
    "jansStatus": "active"
  }'
```

**Use Cases:**
- Audit trail for compliance
- Replay commands for automation
- Script generation from UI actions
- Debugging configuration changes
- Convert UI workflows to CLI scripts

### Authentication Flow

**OAuth Device Flow:**
1. User launches TUI
2. If no credentials in `~/.config/jans-cli.ini`, prompt for:
   - Client ID (e.g., `2000.eac308d1-95e3-4e38-87cf-1532af310a9e`)
   - Client Secret (e.g., `GnEkCqg4Vsks`)
3. Device flow activation code displayed
4. User authenticates via browser
5. Token stored in config file (encoded)
6. Subsequent runs use stored token

**Security Features:**
- Encoded credential storage
- Role-based access control (RBAC) via introspection script
- Token refresh handling
- Secure secret display (redacted in logs)

### Deployment Modes

**1. VM Installation (Standard):**
```bash
# Installed during Janssen setup
/opt/jans/jans-cli/jans_cli_tui.py

# Automatic Config API client provisioning
# Credentials in setup.properties.last
```

**2. Standalone Remote Installation:**

**Build PYZ (Python Zipapp):**
```bash
pip3 install shiv
wget https://github.com/JanssenProject/jans/archive/refs/heads/main.zip -O jans-main.zip
unzip jans-main.zip
cd jans-main/jans-cli-tui/
make zipapp

# Execute
./jans-cli-tui.pyz
```

**Direct pip install:**
```bash
pip3 install https://github.com/JanssenProject/jans/archive/refs/heads/main.zip#subdirectory=jans-cli-tui
jans-cli-tui
```

**3. Container Deployment:**
- Docker container includes TUI
- Kubernetes operator support
- Access via `kubectl exec`

### Production Patterns

**Form Handling:**
- Multi-field input forms with validation
- Required/optional field indicators
- Inline help text and tooltips
- Save/Reset/Cancel button pattern
- Unsaved changes warning on exit

**List Management:**
- Sortable columns
- Filter/search capability
- Pagination for large datasets
- Bulk operations support
- Context menu for row actions

**Configuration Editing:**
- View-only vs edit mode toggle
- Dirty state tracking (unsaved changes indicator)
- Atomic save operations (all or nothing)
- Rollback on error
- Confirmation prompts for destructive actions

---

## Doppler TUI - Secrets Management

**URL**: https://docs.doppler.com/docs/tui
**Purpose**: Terminal-based secrets viewer and editor for Doppler SecretOps platform
**Access Date**: 2025-11-02

### Core Architecture

**Technology Stack:**
- Built into Doppler CLI (Go-based)
- Uses same authentication as CLI (service tokens, login session)
- Direct API integration with Doppler platform
- Vim-inspired modal editing interface

**Launch:**
```bash
doppler tui
```

**Authentication:**
- Uses existing CLI authentication (`doppler login`)
- Service token support (`DOPPLER_TOKEN` env var)
- Same scoping as CLI (project/config context)

### UI Layout

**Three-Panel Interface:**

```
╔═══════════════════╦═══════════════════╦═══════════════════════════════════
║ [1] CONFIGS       ║ [2] PROJECTS      ║ [3] SECRETS
║                   ║                   ║
║ • dev             ║ • backend         ║ Name          │ Value
║ • staging         ║ • frontend        ║ ──────────────┼──────────────────
║ • production      ║ • mobile-app      ║ DATABASE_URL  │ postgres://...
║                   ║                   ║ API_KEY       │ sk_live_...
║ [Active: dev]     ║ [Active: backend] ║ JWT_SECRET    │ [REDACTED]
║                   ║                   ║
╚═══════════════════╩═══════════════════╩═══════════════════════════════════

Filter: [/] to activate
Status: [Normal Mode] │ [Edit Mode] │ [Filter Mode]
Help: Press [?] for keybinds
```

### Navigation System

**Keybind Philosophy:**
- Vim-inspired modal interface
- Normal mode for navigation
- Edit mode for text entry
- Filter mode for searching

**Global Keybinds:**
```
1         Focus Configs panel
2         Focus Projects panel
3         Focus Secrets panel
/         Focus Filter (search)
q         Exit TUI
?         Show help/keybinds
```

**Panel Navigation (Configs/Projects):**
```
j         Move cursor down
k         Move cursor up
Enter     Select current item
```

**Secrets Panel Navigation:**
```
j/k       Move cursor up/down between secrets
h/l       Toggle between secret name and value
J/K       Scroll current selection (multi-line values)
```

### Editing Workflow

**Modal Editing (Vim-style):**

**Normal Mode → Edit Mode:**
```
e         Enter edit mode (modify current secret)
a         Add new secret
d         Delete current secret
u         Undo changes (before save)
y         Copy current selection to clipboard
```

**Edit Mode → Normal Mode:**
```
Esc       Exit editing mode
Tab       Toggle between name and value fields
```

**Saving Changes:**
```
s         Open save confirmation prompt
```

**Save Prompt:**
```
╔═══════════════════════════════════════════════════════════
║ Save Changes?
╠═══════════════════════════════════════════════════════════
║ Modified: 3 secrets
║ Added:    1 secret
║ Deleted:  1 secret
║
║ This will update the 'dev' config in 'backend' project
║
║ [Enter] Confirm │ [Esc/q] Cancel
╚═══════════════════════════════════════════════════════════
```

### UI Patterns

#### 1. Secret Display

**Multi-line Value Scrolling:**
```
Name: PRIVATE_KEY
╔═══════════════════════════════════════════════════════════
║ -----BEGIN RSA PRIVATE KEY-----
║ MIIEpAIBAAKCAQEAx...
║ [Scroll: J/K]
║ [Line 3 of 27]
╚═══════════════════════════════════════════════════════════
```

**Value Visibility:**
- Sensitive values can be masked/revealed
- Copy to clipboard without displaying on screen
- Redacted display for long values in list view

#### 2. Filtering System

**Filter Activation:**
```
Press [/]
╔═══════════════════════════════════════════════════════════
║ Filter: data█
╠═══════════════════════════════════════════════════════════
║ Matching Secrets (3):
║
║ DATABASE_URL
║ DATABASE_PASSWORD
║ DATABASE_HOST
║
║ [Enter] Apply │ [Esc] Cancel
╚═══════════════════════════════════════════════════════════
```

**Filter Behavior:**
- Real-time filtering as you type
- Partial match on secret names
- Case-insensitive search
- Press Enter to apply, Esc to cancel
- Cleared filter shows all secrets

#### 3. State Management

**Dirty State Tracking:**
```
╔═══════════════════════════════════════════════════════════
║ Secrets (Modified) *
╠═══════════════════════════════════════════════════════════
║ • DATABASE_URL *     │ postgres://...  [Modified]
║   API_KEY            │ sk_live_...
║ + NEW_SECRET *       │ value123        [Added]
║ - OLD_SECRET         │ [Deleted]
║
║ [s] Save │ [u] Undo
╚═══════════════════════════════════════════════════════════
```

**Indicators:**
- `*` Modified indicator on secrets
- `+` Added secret indicator
- `-` Deleted secret (strikethrough)
- Unsaved changes count in status bar

#### 4. Context Awareness

**Scope Display:**
```
Current Context:
  Workplace:  Acme Corp
  Project:    backend-api
  Config:     production

[Switch: 1=Configs, 2=Projects]
```

### Common Workflows

#### Workflow 1: Quick Value Update

**Optimized Path (4 steps):**
```
1. Press [/]                    ← Focus filter
2. Type "DATABASE"              ← Filter to target
3. Press [Enter]                ← Apply filter
4. Press [e]                    ← Edit mode
5. Update value
6. Press [Esc]                  ← Back to normal
7. Press [s] → [Enter]          ← Save and confirm
```

**Time:** ~10 seconds for single secret update

#### Workflow 2: Bulk Secret Addition

```
1. Press [a]                    ← Add new secret
2. Enter name: "API_URL"
3. Press [Tab]
4. Enter value: "https://..."
5. Press [Esc]                  ← Normal mode
6. Press [a] again              ← Add another
   (repeat steps 2-5)
7. Press [s] → [Enter]          ← Save all at once
```

#### Workflow 3: Secret Migration

```
1. Focus source config (panel 1)
2. Select secrets with [j/k]
3. Press [y] to copy
4. Switch to target config
5. Press [a] and paste
6. Press [s] to save
```

### Comparison with GUI Dashboard

**TUI Advantages:**
- Fast keyboard-only workflow
- Works over SSH/remote connections
- Scriptable/automatable context switching
- Lower latency for quick edits
- No context switching from terminal

**TUI Limitations (per docs):**
- May not contain all latest features
- Some advanced dashboard features unavailable
- Visual features (charts, graphs) not present
- Limited bulk operations vs API

### Integration with CLI

**Shared Context:**
```bash
# Set up scope via CLI
doppler setup --project backend --config dev

# Launch TUI (inherits scope)
doppler tui

# TUI respects CLI configuration
cat ~/.doppler/.doppler.yaml
```

**Service Token Usage:**
```bash
# Use service token for auth
DOPPLER_TOKEN=dp.st.dev.xyz123 doppler tui
```

### Security Features

**Access Control:**
- Respects user permissions (read/write)
- Role-based secret visibility
- Audit logging (all changes logged server-side)
- No local secret caching (always live from API)

**Safe Operations:**
- Confirmation prompts for destructive actions
- Undo before save
- Atomic saves (all or nothing)
- Network error handling with retry

---

## Common Production TUI Patterns

### 1. Authentication Strategies

**Janssen Approach:**
- OAuth Device Flow for initial auth
- Long-lived token storage
- Role-based access via server-side introspection
- Admin credentials required

**Doppler Approach:**
- Reuse CLI authentication session
- Service token support for CI/CD
- Session-based for interactive use
- Automatic token refresh

**Pattern:** Both avoid password entry in TUI, delegate to secure auth flows

### 2. Configuration Persistence

**Janssen:**
- Config file: `~/.config/jans-cli.ini`
- Encoded credentials
- Operation logs: `cli_cmd.log`

**Doppler:**
- Uses CLI config: `~/.doppler/.doppler.yaml`
- No local secret cache (always live)
- Stateless operation (server is source of truth)

**Pattern:** Minimal local state, prefer server-side persistence

### 3. Form Input Patterns

**Multi-field Forms:**
```
╔═══════════════════════════════════════════════════════════
║ Add New Client
╠═══════════════════════════════════════════════════════════
║ Client Name:      [Web Application___________] *required
║ Client ID:        [auto-generated____________]
║ Grant Types:      [☑] authorization_code
║                   [☐] implicit
║                   [☑] refresh_token
║ Redirect URI:     [https://__________________|] +Add More
║
║ [Tab] Next Field │ [Shift+Tab] Previous │ [Enter] Save
╚═══════════════════════════════════════════════════════════
```

**Validation Patterns:**
- Inline validation messages
- Required field indicators (`*`)
- Format hints (placeholders)
- Prevent save until valid

### 4. List Operations

**Common List UI:**
```
╔═══════════════════════════════════════════════════════════
║ Items (234)                    [Search: ___] [Filter ▼]
╠═══════════════════════════════════════════════════════════
║ Name              │ Type      │ Status   │ Modified
║ ──────────────────┼───────────┼──────────┼──────────────
║ prod-client       │ OAuth     │ Active   │ 2 days ago
║ dev-app           │ OpenID    │ Active   │ 1 week ago
║ test-service      │ Internal  │ Inactive │ 3 months ago
║
║ [j/k] Navigate │ [Enter] View │ [/] Search │ [a] Add
╚═══════════════════════════════════════════════════════════
```

**Features:**
- Sortable columns
- Filter/search
- Status indicators
- Last modified timestamps
- Row selection with keyboard

### 5. Action Confirmation

**Destructive Operations:**
```
╔═══════════════════════════════════════════════════════════
║ ⚠ Confirm Delete
╠═══════════════════════════════════════════════════════════
║ Are you sure you want to delete this client?
║
║ Client: prod-client (ID: 1000.abc-123-def)
║
║ This action CANNOT be undone.
║
║ Type 'DELETE' to confirm: [___________]
║
║ [Enter] Confirm │ [Esc] Cancel
╚═══════════════════════════════════════════════════════════
```

**Best Practices:**
- Clear warning message
- Show what will be deleted
- Require typed confirmation for critical ops
- Easy cancel option (Esc)

### 6. Help and Discoverability

**Context-Sensitive Help:**
```
Press [?] at any time for help

╔═══════════════════════════════════════════════════════════
║ Keyboard Shortcuts - Secrets View
╠═══════════════════════════════════════════════════════════
║ Navigation:
║   j/k     Move up/down
║   h/l     Switch name/value
║   J/K     Scroll multi-line values
║
║ Editing:
║   e       Edit current secret
║   a       Add new secret
║   d       Delete secret
║   u       Undo changes
║
║ Other:
║   s       Save changes
║   y       Copy to clipboard
║   /       Filter/search
║   ?       Show this help
║   q       Quit
║
║ [Press any key to close]
╚═══════════════════════════════════════════════════════════
```

**Inline Hints:**
```
Status Bar: [Normal Mode] │ [s] Save │ [?] Help │ [q] Quit
```

### 7. Error Handling

**Network Errors:**
```
╔═══════════════════════════════════════════════════════════
║ ⚠ Connection Error
╠═══════════════════════════════════════════════════════════
║ Failed to connect to Janssen Config API
║
║ Error: Connection refused (localhost:8443)
║
║ Possible causes:
║ • Config API service is down
║ • Network connectivity issues
║ • Incorrect API endpoint configuration
║
║ [r] Retry │ [c] Check Config │ [q] Quit
╚═══════════════════════════════════════════════════════════
```

**Validation Errors:**
```
╔═══════════════════════════════════════════════════════════
║ ✗ Validation Failed
╠═══════════════════════════════════════════════════════════
║ Cannot save client configuration
║
║ Issues:
║ • Redirect URI must be a valid HTTPS URL
║ • At least one grant type must be selected
║
║ [Enter] Fix Issues │ [Esc] Cancel
╚═══════════════════════════════════════════════════════════
```

### 8. Plugin/Extension Patterns

**Janssen Plugin Loading:**
```python
# Plugin directory structure
/opt/jans/jans-cli/plugins/
├── 001-user-mgt/
│   ├── __init__.py
│   ├── plugin.py
│   └── views/
├── 002-scim/
│   ├── __init__.py
│   ├── plugin.py
│   └── views/
└── 003-fido2/
    ├── __init__.py
    └── plugin.py

# Plugin registration in jans-cli.ini
[DEFAULT]
jca_plugins = user-mgt,scim,fido2,admin-ui
```

**Plugin Capabilities:**
- Add menu items to main panel
- Register custom views/panels
- Hook into API client
- Custom keybinds per plugin context

---

## Configuration Management Patterns

### Janssen Configuration Approach

**Hierarchical Configuration:**
```
Server Level (Global)
  ├── Auth Server Properties
  ├── Logging Configuration
  └── Cache Configuration
      │
      ├── Project Level (Per Component)
      │   ├── FIDO Dynamic Config
      │   ├── SCIM Settings
      │   └── Client Configurations
      │       │
      │       └── Instance Level (Per Client)
      │           ├── Client Properties
      │           ├── Grant Types
      │           └── Scopes
```

**Configuration Storage:**
- Backend: REST API + Database (LDAP/MySQL/PostgreSQL/Couchbase)
- TUI acts as frontend to API
- All changes via API (consistent with web UI and CLI)

**Edit → Save Pattern:**
1. Fetch config from API (GET)
2. Display in form
3. User edits (validation on input)
4. Save button pressed
5. POST/PUT to API
6. Refresh view from API (confirm save)
7. Log command to `cli_cmd.log`

### Doppler Configuration Approach

**Project → Config → Secrets Hierarchy:**
```
Workplace (Org Level)
  ├── Project: backend-api
  │   ├── Config: dev
  │   │   └── Secrets: key-value pairs
  │   ├── Config: staging
  │   │   └── Secrets: inherited + overrides
  │   └── Config: production
  │       └── Secrets: inherited + overrides
  └── Project: frontend
      └── Configs...
```

**Config Inheritance:**
- Base config defines defaults
- Branch configs inherit and override
- TUI shows effective values (computed)
- Changes affect only selected config

**Live Updates:**
- No local cache of secrets
- Always fetch from API
- Changes immediately visible to other clients
- Real-time sync across TUI sessions

### Common Configuration Patterns

#### 1. Two-Phase Commit

Both Janssen and Doppler use:
```
1. Edit Phase (local state)
   - Dirty flag tracking
   - Validation on input
   - Undo capability

2. Save Phase (commit to server)
   - Atomic save (all or nothing)
   - Server-side validation
   - Error handling with rollback
   - Confirmation dialog
```

#### 2. Optimistic UI Updates

```python
# Pattern:
def save_secret(name, value):
    # 1. Update local display (optimistic)
    ui.update_secret(name, value)
    ui.mark_saving()

    # 2. Send to server
    try:
        api.update_secret(name, value)
        ui.mark_saved()
    except Exception as e:
        # 3. Rollback on error
        ui.revert_secret(name)
        ui.show_error(e)
```

#### 3. Read-Only vs Edit Modes

**Janssen:**
- View mode: Display with navigation only
- Edit mode: Forms become editable
- Clear mode indicator in UI
- Different keybinds per mode

**Doppler:**
- Normal mode: Navigation + copy
- Edit mode: Text input enabled
- Modal interface (Vim-style)
- Explicit enter/exit edit mode

#### 4. Batch Operations

**Janssen:**
```
Bulk Import:
1. Upload JSON/YAML file
2. Preview changes in table
3. Confirm import
4. Show progress bar
5. Report success/failures
```

**Doppler:**
```
Multiple Edits:
1. Edit secret A
2. Edit secret B
3. Add secret C
4. Delete secret D
5. Press 's' (save once)
6. Atomic commit (all changes together)
```

**Pattern:** Allow multiple changes, commit atomically

---

## Production Deployment Patterns

### Remote Access Considerations

**Janssen:**
- **VM Installation**: Direct server access required
- **Standalone Mode**: Install on admin workstation
- **SSH Access**: Run over SSH with no issues
- **Network**: Must reach Config API endpoint
- **Latency**: Optimized for LAN/datacenter speeds

**Doppler:**
- **Local Install**: Install CLI on workstation
- **SSH Friendly**: Works over high-latency connections
- **API First**: Always talks to cloud API
- **Offline**: No offline mode (requires connectivity)

### Multi-User Scenarios

**Janssen:**
- Role-based access control (RBAC)
- Admin user required for TUI
- Concurrent edits possible (last write wins)
- Audit logs track who made changes
- No real-time collaboration features

**Doppler:**
- Team-based access (workspace → project → config)
- Fine-grained permissions per user
- Concurrent edits handled server-side
- Activity logs for compliance
- No collaborative editing in TUI

**Pattern:** Single-user TUI sessions, rely on server-side conflict resolution

### CI/CD Integration

**Janssen:**
```bash
# Command log replay in CI
# TUI generates CLI commands in cli_cmd.log
cat cli_cmd.log | while read cmd; do
  $cmd  # Replay for automation
done

# Example: Automated client creation
python3 /opt/jans/jans-cli/cli/config_cli.py \
  --operation-id post-user \
  --data "$(cat user_config.json)"
```

**Doppler:**
```bash
# Service token in CI
export DOPPLER_TOKEN="${{ secrets.DOPPLER_SERVICE_TOKEN }}"

# TUI not used in CI (CLI instead)
doppler secrets set API_KEY="new_value" --project backend --config production

# TUI for interactive admin tasks only
```

**Pattern:** TUI for interactive admin, CLI/API for automation

---

## Textual Implementation Notes

### UI Components Likely Used

Based on observed patterns, these Textual widgets are likely employed:

**Janssen TUI:**
```python
# Main layout
from textual.app import App
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, Button, Input, DataTable

# Likely structure:
class JanssenTUI(App):
    """Main TUI application"""

    # Three-column layout
    # Left: Navigation tree
    # Center: List view (DataTable)
    # Right: Detail form (Input fields + Buttons)

    # Widgets used:
    # - Tree for menu navigation
    # - DataTable for client lists
    # - Input for form fields
    # - Button for actions
    # - Modal screens for confirmations
```

**Doppler TUI:**
```python
from textual.app import App
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Input, Static
from textual.binding import Binding

class DopplerTUI(App):
    """Secrets management TUI"""

    # Modal interface (vim-style)
    BINDINGS = [
        Binding("j", "cursor_down", "Down"),
        Binding("k", "cursor_up", "Up"),
        Binding("e", "edit_mode", "Edit"),
        Binding("s", "save", "Save"),
        # ... etc
    ]

    # Widgets:
    # - DataTable for secrets (two columns: name, value)
    # - Input for filter
    # - Static for status bar
    # - Custom modal for save confirmation
```

### Key Textual Features Demonstrated

**1. Reactive State:**
```python
# Track dirty state
class SecretsView(Widget):
    dirty = reactive(False)  # Triggers UI update

    def watch_dirty(self, dirty: bool) -> None:
        """Update save button state when dirty changes"""
        self.query_one("#save_btn").disabled = not dirty
```

**2. Keybind Management:**
```python
# Modal keybinds (different per mode)
class SecretEditor(Widget):
    mode = reactive("normal")  # or "edit"

    def on_key(self, event: Key) -> None:
        if self.mode == "normal":
            # Navigation keys
            if event.key == "j": self.cursor_down()
            elif event.key == "e": self.enter_edit_mode()
        elif self.mode == "edit":
            # Text input enabled
            if event.key == "escape": self.exit_edit_mode()
```

**3. DataTable with Custom Rendering:**
```python
# Janssen client list with status indicators
table = DataTable()
table.add_column("Client ID", key="id")
table.add_column("Name", key="name")
table.add_column("Status", key="status")

# Add row with styled status
table.add_row(
    "1000.abc-123",
    "Web App",
    Text("Active", style="green"),  # Rich Text styling
)
```

**4. Screen/Modal Management:**
```python
# Confirmation modal
class ConfirmSave(ModalScreen):
    """Save confirmation dialog"""

    def compose(self) -> ComposeResult:
        yield Container(
            Static("Save changes?"),
            Static(f"Modified: {self.changes} secrets"),
            Horizontal(
                Button("Confirm", id="confirm"),
                Button("Cancel", id="cancel"),
            ),
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            self.dismiss(True)  # Return True to caller
        else:
            self.dismiss(False)
```

**5. API Integration:**
```python
# Async API calls in Textual
class ConfigView(Widget):
    async def save_config(self) -> None:
        """Save config to API"""
        self.loading = True
        try:
            await api_client.update_config(self.config_data)
            self.notify("Saved successfully", severity="information")
        except APIError as e:
            self.notify(f"Error: {e}", severity="error")
        finally:
            self.loading = False
```

**6. Logging Integration:**
```python
# Command logging (Janssen pattern)
import logging

logger = logging.getLogger("cli_cmd")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("cli_cmd.log")
logger.addHandler(handler)

def log_command(operation, data):
    """Log TUI action as CLI command"""
    cmd = f"/usr/bin/python3 /opt/jans/jans-cli/cli/config_cli.py --operation-id {operation} --data '{data}'"
    logger.info(cmd)

# Call on every save
async def on_save(self):
    await api.save(data)
    log_command("post-user", json.dumps(data))
```

---

## Design Principles from Production TUIs

### 1. Keyboard-First Design

**Principle:** Every operation must be completable without mouse

**Implementation:**
- Single-key shortcuts for common actions
- Vim-style navigation (hjkl)
- Tab for form navigation
- Enter for confirm, Esc for cancel
- Number keys for panel switching (1, 2, 3)

**Evidence:**
- Doppler: Full vim-style modal interface
- Janssen: Keyboard navigation throughout
- Both: No mouse requirements

### 2. Progressive Disclosure

**Principle:** Show only what's needed, hide complexity

**Implementation:**
- List view → Detail view (drill-down)
- Summary in lists, full details on selection
- Expandable sections in forms
- Filter to reduce visual noise
- Help hidden behind `?` key

**Evidence:**
- Janssen: Client list shows summary, Enter for full details
- Doppler: Secrets list, scroll for multi-line values

### 3. Immediate Feedback

**Principle:** User should always know system state

**Implementation:**
- Status bar showing current mode
- Loading indicators during API calls
- Dirty state indicators (`*` modified)
- Success/error notifications
- Progress bars for long operations

**Evidence:**
- Doppler: `[Normal Mode]` vs `[Edit Mode]` in status
- Janssen: Modified indicators on forms
- Both: Clear save/error messages

### 4. Fail-Safe Operations

**Principle:** Prevent data loss, allow recovery

**Implementation:**
- Undo before save
- Confirmation for destructive actions
- Atomic saves (all or nothing)
- Error handling with rollback
- Explicit save required (no auto-save)

**Evidence:**
- Doppler: `u` for undo, save confirmation dialog
- Janssen: Confirm deletes, save/reset/cancel buttons
- Both: Changes not committed until explicitly saved

### 5. Consistent Navigation

**Principle:** Same keys do same things everywhere

**Implementation:**
- `j/k` always means down/up
- `q` always means quit
- `?` always shows help
- `Esc` always cancels/goes back
- `Enter` always confirms/selects

**Evidence:**
- Both apps use identical base keybinds
- Context-specific keys clearly documented
- Help shows available keys per screen

### 6. Offline-First for Local, API-First for Remote

**Janssen Pattern (Local Config):**
- Config file stores auth
- Can work with cached data
- Logs operations for replay

**Doppler Pattern (Remote Secrets):**
- No local secret storage
- Always live API calls
- Network required (by design)

**Principle:** Choose based on data sensitivity and usage pattern

### 7. Auditability

**Principle:** All changes must be traceable

**Implementation:**
- Janssen: Command log (`cli_cmd.log`)
- Doppler: Server-side activity logs
- Both: Who, what, when for every change
- Logs consumable by external systems

---

## Common Production TUI Features Summary

### Must-Have Features

1. **Authentication & Authorization**
   - Secure credential storage
   - Token-based auth
   - Role-based access control
   - Session management

2. **List Management**
   - Sortable columns
   - Filter/search
   - Pagination
   - Keyboard navigation

3. **Form Handling**
   - Multi-field input
   - Validation (inline + on save)
   - Required field indicators
   - Tab navigation between fields

4. **State Management**
   - Dirty state tracking
   - Undo capability
   - Atomic saves
   - Conflict resolution

5. **Error Handling**
   - Network error recovery
   - Validation error display
   - Graceful degradation
   - Clear error messages

6. **Help System**
   - Context-sensitive help (`?`)
   - Inline hints (status bar)
   - Keybind reference
   - Tooltips/placeholders

7. **Audit/Logging**
   - Operation logging
   - Command replay capability
   - Activity tracking
   - Integration with external logs

### Nice-to-Have Features

1. **Plugin/Extension System**
   - Dynamic loading
   - Custom panels
   - Hook into app lifecycle

2. **Multi-Panel Layout**
   - Independent panel focus
   - Panel switching (numbers)
   - Context per panel

3. **Clipboard Integration**
   - Copy values with `y`
   - Paste into forms
   - OS clipboard support

4. **Bulk Operations**
   - Multi-select
   - Batch edit
   - Import/export

5. **Live Updates**
   - Real-time data refresh
   - Change notifications
   - Multi-user awareness

6. **Advanced Search**
   - Regex support
   - Filter by field
   - Saved searches

---

## Sources

**Primary Documentation:**

1. **Janssen TUI Documentation**
   - URL: https://jans.io/docs/head/janssen-server/config-guide/config-tools/jans-tui/
   - Access Date: 2025-11-02
   - Content: Architecture, installation, navigation, plugin system, configuration management
   - Technology: Python, OAuth Device Flow, REST API backend

2. **Doppler TUI Documentation**
   - URL: https://docs.doppler.com/docs/tui
   - Access Date: 2025-11-02
   - Content: Keybinds, workflows, editing patterns, filtering, modal interface
   - Technology: Go (Doppler CLI), Vim-inspired UI, live API integration

**Key Insights:**

- **Janssen**: Complex enterprise identity server configuration via TUI, plugin architecture, command logging for automation, OAuth-based auth flow
- **Doppler**: Lightweight secrets editor, vim-modal interface, keyboard-optimized workflows, no local caching

**Production Patterns Observed:**
- Modal editing (normal vs edit mode)
- Three-panel layouts (navigation + list + detail)
- Atomic saves with confirmation
- Keyboard-first design
- Plugin/extension systems
- Command logging for CI/CD integration
- API-backed state (TUI as frontend)

---

## Cross-References

**Related Oracle Documentation:**

- `architecture/00-application-structure.md` - Overall app architecture patterns
- `architecture/01-reactive-programming.md` - Reactive state management in forms
- `architecture/02-css-styling.md` - Styling list views, forms, modals
- `widgets/00-input-textarea.md` - Form input handling
- `widgets/01-button.md` - Action buttons (Save/Cancel/Delete)
- `widgets/02-datatable.md` - List views (clients, secrets, configs)
- `widgets/06-tree.md` - Navigation trees (Janssen menu structure)
- `layout/00-grid-system.md` - Multi-panel layouts
- `layout/01-dock-system.md` - Panel docking and focus
- `patterns/00-modal-dialogs.md` - Confirmation dialogs (if exists)

**Recommended Next Steps:**

1. Study Janssen TUI source code for plugin architecture
2. Analyze Doppler CLI TUI implementation for vim-modal patterns
3. Review Textual DataTable for list management
4. Implement atomic save pattern with undo
5. Create reusable confirmation modal component
