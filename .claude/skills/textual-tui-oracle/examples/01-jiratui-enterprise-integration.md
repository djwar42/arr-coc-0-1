# JiraTUI - Jira Client Shell Integration

## Overview

JiraTUI is a production-ready Text User Interface (TUI) for interacting with Atlassian Jira directly from the shell. Built with Textual and Rich, it demonstrates enterprise API integration, CLI/TUI hybrid architecture, and professional configuration management patterns.

**Repository**: https://github.com/whyisdifficult/jiratui
**Stars**: 1,000+ (November 2025)
**License**: MIT
**Platform Support**: Linux, macOS, Windows
**Python**: 3.10, 3.11, 3.12, 3.13
**Documentation**: https://jiratui.readthedocs.io

From [GitHub README](https://github.com/whyisdifficult/jiratui) (accessed 2025-11-02):
- Supports Jira REST API v3 (Cloud) and v2 (Data Center/on-premises)
- Full CLI for automation + interactive TUI for exploration
- YAML-based configuration with XDG specification compliance
- 1,000+ stars, actively maintained with 72+ commits

## Key Features

### Enterprise Integration
- **Jira Cloud Platform API** (REST v3) - default
- **Jira Data Center API** (on-premises, REST v2) - configurable
- Personal Access Token (PAT) authentication
- JQL (Jira Query Language) expression support
- Custom field mapping (sprints, custom metadata)

### Hybrid CLI/TUI Architecture
- **CLI commands** - automation, scripting, CI/CD pipelines
- **Interactive TUI** - visual exploration, rapid navigation
- Shared configuration and API controller
- Rich table rendering in terminal

### Configuration Management
- YAML config files (XDG specification)
- Environment variable overrides
- Per-user settings and defaults
- Pre-defined JQL expressions

## Installation

From [README Installation section](https://github.com/whyisdifficult/jiratui):

```bash
# Recommended: uv
uv tool install jiratui

# Alternative: pip
pip install jiratui

# Alternative: pipx
pipx install jiratui

# Arch Linux (AUR)
yay -S jiratui-git

# Homebrew
brew install jiratui
```

## Configuration

### XDG Specification Compliance

JiraTUI follows the [XDG Base Directory specification](https://specifications.freedesktop.org/basedir-spec/latest/) for config and log files.

**Config file resolution order**:
1. `$JIRA_TUI_CONFIG_FILE` (env variable) - if set, use this file
2. `$XDG_CONFIG_HOME/jiratui/config.yaml` - if XDG_CONFIG_HOME set
3. `$HOME/.config/jiratui/config.yaml` - fallback

**Find your config location**:
```bash
jiratui config
# Output: Using: /home/user/.config/jiratui/config.yaml
```

### Example Configuration

From [jiratui.example.yaml](https://github.com/whyisdifficult/jiratui/blob/main/jiratui.example.yaml):

```yaml
# API Credentials (required)
jira_api_username: 'foo@bar.com'
jira_api_token: '12345'  # Personal Access Token
jira_api_base_url: 'https://<your-hostname>.atlassian.net'
jira_base_url: 'https://<your-hostname>.atlassian.net'

# User Settings
jira_user_group_id: '12345'
jira_account_id: '098765'

# Search Configuration
search_results_per_page: 65
search_issues_default_day_interval: 15
show_issue_web_links: True
default_project_key_or_id: 'MY-PROJECT-KEY'
ignore_users_without_email: True

# Custom Fields
custom_field_id_sprint: 'customfield_<MY-PROJECT-SPRINT-FIELD-ID>'

# Pre-defined JQL Expressions
pre_defined_jql_expressions:
  1:
    label: "Work in the current sprint"
    expression: 'sprint in openSprints()'

jql_expression_id_for_work_items_search: 1
```

### Choosing Jira Platform

**Jira Cloud (default)**:
```yaml
# Uses Jira Cloud API v3 by default
jira_api_base_url: 'https://your-instance.atlassian.net'
```

**Jira Data Center (on-premises)**:
```yaml
cloud: False  # Switches to Data Center API
jira_api_base_url: 'https://jira.your-company.com'
```

**API Version Override** (Cloud only):
```yaml
jira_api_version: 2  # Use v2 instead of v3 (cloud only)
```

**Important**: When `cloud: False`, JiraTUI automatically uses the correct API version and ignores `jira_api_version`.

## CLI Interface

### Command Structure

From [cli.py](https://github.com/whyisdifficult/jiratui/blob/main/src/jiratui/cli.py):

```
jiratui
├── ui              # Launch interactive TUI
├── issues          # Work item operations
│   ├── search      # Search issues
│   ├── metadata    # Show issue metadata
│   └── update      # Update issue fields
├── comments        # Comment operations
│   ├── add         # Add comment
│   ├── list        # List comments
│   ├── show        # Show comment text
│   └── delete      # Delete comment
├── users           # User operations
│   ├── search      # Search users by email/name
│   └── groups      # Search user groups
├── config          # Show config file location
├── version         # Show version
├── themes          # List available themes
└── completions     # Generate shell completions (bash/zsh/fish)
```

### CLI Usage Examples

**Search issues in a project**:
```bash
jiratui issues search --project-key SCRUM
```

Output (Rich table rendering):
```
| Key     | Type | Created          | Status (ID)   | Reporter          | Assignee          | Summary                                    |
|---------|------|------------------|---------------|-------------------|-------------------|--------------------------------------------|
| SCRUM-1 | Bug  | 2025-07-31 15:55 | To Do (10000) | lisa@simpson.com  | bart@simpson.com  | Write 100 times "I will be a good student" |
| SCRUM-2 | Task | 2025-06-30 15:56 | To Do (10000) | homer@simpson.com | homer@simpson.com | Eat donuts                                 |
```

**Search specific issue**:
```bash
jiratui issues search --key SCRUM-1
```

**Search with filters**:
```bash
jiratui issues search \
  --project-key SCRUM \
  --assignee-account-id 098765 \
  --limit 100 \
  --created-from 2025-01-01 \
  --created-until 2025-12-31
```

**Get issue metadata** (useful before updates):
```bash
jiratui issues metadata SCRUM-1
```

**Update issue fields**:
```bash
# Update summary
jiratui issues update SCRUM-1 --summary "New summary text"

# Update assignee
jiratui issues update SCRUM-1 --assignee-account-id 098765

# Unassign issue
jiratui issues update SCRUM-1 --assignee-account-id ""

# Update due date
jiratui issues update SCRUM-1 --due-date 2025-12-31

# Update status (use --meta first to get status IDs)
jiratui issues update SCRUM-1 --status-id 10001

# Update priority
jiratui issues update SCRUM-1 --priority-id 3
```

**Comment operations**:
```bash
# Add comment
jiratui comments add SCRUM-1 "This is a comment"

# List comments (paginated, 10 per page)
jiratui comments list SCRUM-1 --page 1

# Show specific comment
jiratui comments show SCRUM-1 12345

# Delete comment
jiratui comments delete SCRUM-1 12345
```

**User operations**:
```bash
# Search users by email or name (get account IDs)
jiratui users search "bart@simpson.com"
jiratui users search "Bart Simpson"

# Search user groups
jiratui users groups --group-names "developers,admins"
jiratui users groups --group-ids "12345,67890"

# Count users in a group
jiratui users groups --group-id 12345
```

**Shell completions**:
```bash
# Generate completions
jiratui completions bash > ~/.local/share/bash-completion/completions/jiratui
jiratui completions zsh > ~/.zfunc/_jiratui
jiratui completions fish > ~/.config/fish/completions/jiratui.fish
```

## Interactive TUI

### Launch TUI

```bash
# Basic launch
jiratui ui

# Pre-select project
jiratui ui --project-key SCRUM

# Pre-select assignee (your account ID)
jiratui ui --assignee-account-id 098765

# Open specific work item
jiratui ui --work-item-key SCRUM-42

# Use pre-defined JQL expression
jiratui ui --jql-expression-id 1

# Use custom theme
jiratui ui --theme monokai

# Search on startup
jiratui ui --search-on-startup

# Focus specific item on startup (requires --search-on-startup)
jiratui ui --search-on-startup --focus-item-on-startup 3

# Custom config file
JIRA_TUI_CONFIG_FILE=/path/to/custom-config.yaml jiratui ui
```

### TUI Keybindings

From [app.py](https://github.com/whyisdifficult/jiratui/blob/main/src/jiratui/app.py):

```python
BINDINGS = [
    Binding(key='f1,ctrl+question_mark,ctrl+shift+slash',
            action='help',
            description='Help'),
    Binding(key='f2',
            action='server_info',
            description='Server',
            tooltip='Show details of the Jira server'),
    Binding(key='f3',
            action='config_info',
            description='Config',
            tooltip='Show the settings in the configuration file'),
    Binding(key='ctrl+q',
            action='quit',
            description='Quit',
            key_display='^q',
            tooltip='Quit',
            show=True),
]
```

**Key patterns**:
- `F1` / `Ctrl+?` / `Ctrl+Shift+/` - Help screen
- `F2` - Server info (Jira instance details)
- `F3` - Config info (view active configuration)
- `Ctrl+Q` - Quit (with confirmation if configured)

### TUI Screens

From app.py implementation:
1. **MainScreen** - Primary work item search/view interface
2. **HelpScreen** - Context-sensitive help with anchor navigation
3. **ServerInfoScreen** - Jira server version and instance details
4. **ConfigFileScreen** - Display active configuration settings
5. **QuitScreen** - Confirmation dialog before exit

## Architecture Patterns

### Application Structure

From [app.py](https://github.com/whyisdifficult/jiratui/blob/main/src/jiratui/app.py):

```python
class JiraApp(App):
    """Main Textual application class"""

    CSS_PATH = 'css/jt.tcss'
    TITLE = 'JiraTUI'
    DEFAULT_THEME = 'textual-dark'

    def __init__(
        self,
        settings: ApplicationConfiguration,
        project_key: str | None = None,
        user_account_id: str | None = None,
        jql_expression_id: int | None = None,
        work_item_key: str | None = None,
        user_theme: str | None = None,
        focus_item_on_startup: int | None = None,
    ):
        super().__init__()
        self.config = settings
        self.api = APIController()  # Shared API controller

        # Initialize state from parameters or config
        self.initial_project_key = project_key or settings.default_project_key_or_id
        self.initial_user_account_id = user_account_id or settings.jira_account_id
        self.initial_jql_expression_id = jql_expression_id
        self.initial_work_item_key = work_item_key
        self.focus_item_on_startup = focus_item_on_startup

        self._setup_logging()
        self._setup_theme(user_theme)
```

**Key architectural patterns**:
- Shared `APIController` - single API instance across screens
- Configuration injection - `ApplicationConfiguration` passed to constructor
- State initialization - CLI args override config defaults
- Deferred screen mounting - `on_mount()` pushes MainScreen
- Worker-based async - `run_worker()` for background API calls

### Configuration Management

From app.py initialization:

```python
# Global configuration context
CONFIGURATION.set(settings)

# Config file resolution
if jira_tui_config_file := os.getenv('JIRA_TUI_CONFIG_FILE'):
    config_file = Path(jira_tui_config_file).resolve()
else:
    config_file = get_config_file()  # XDG spec resolution
```

**Pattern**: Global configuration singleton with context manager for access:
```python
CONFIGURATION.get().log_level
CONFIGURATION.get().jira_api_base_url
CONFIGURATION.get().theme
```

### Logging Setup

From app.py `_setup_logging()`:

```python
def _setup_logging(self) -> None:
    self.logger = logging.getLogger(LOGGER_NAME)
    self.logger.setLevel(CONFIGURATION.get().log_level or logging.WARNING)

    # Log file resolution (XDG spec)
    if jira_tui_log_file := os.getenv('JIRA_TUI_LOG_FILE'):
        log_file = Path(jira_tui_log_file).resolve()
    elif config_log_file := CONFIGURATION.get().log_file:
        log_file = Path(config_log_file).resolve()
    else:
        log_file = get_log_file()  # XDG spec default

    # JSON structured logging
    fh = logging.FileHandler(log_file)
    fh.setFormatter(JsonFormatter('%(asctime)s %(levelname)s %(message)s ...'))
    self.logger.addHandler(fh)
```

**Patterns**:
- JSON structured logging via `pythonjsonlogger`
- XDG specification for log file location
- Environment variable override support
- Configurable log levels

### Theme Management

```python
def _setup_theme(self, user_theme: str | None = None) -> None:
    if input_theme := (user_theme or CONFIGURATION.get().theme):
        try:
            self.theme = input_theme
        except InvalidThemeError:
            self.logger.warning(
                f'Unknown theme {input_theme}. Using default: {self.DEFAULT_THEME}'
            )
            self.theme = self.DEFAULT_THEME
    else:
        self.theme = self.DEFAULT_THEME
```

**Available themes**:
```bash
jiratui themes
# Lists all built-in Textual themes
```

Use with:
```bash
jiratui ui --theme monokai
```

### CLI Architecture

From [cli.py](https://github.com/whyisdifficult/jiratui/blob/main/src/jiratui/cli.py):

**Command handler pattern**:
```python
# Shared command handler for all CLI operations
handler = CommandHandler()

# Search with status feedback
with console.status('Searching work items...'):
    try:
        response = handler.search_issues(
            project_key=project_key,
            assignee_account_id=assignee_account_id,
            limit=limit,
            created_from=created_from.date() if created_from else None,
            created_until=created_until.date() if created_until else None,
        )
    except CLIException as e:
        console.print(str(e))
        renderer = CLIExceptionRenderer()
        renderer.render(console, e.get_extra_details())
    except Exception as e:
        console.print(f'An unknown error occurred: {str(e)}')
    else:
        render = JiraIssueSearchRenderer()
        render.render(console, response)
```

**CLI patterns**:
1. **Click framework** - command groups, options, arguments
2. **Rich console** - table rendering, status spinners, styled output
3. **CommandHandler** - shared business logic (CLI + TUI)
4. **Renderer pattern** - separate presentation from logic
5. **Exception hierarchy** - `CLIException` with extra details
6. **Async support** - `asyncio.run()` for async handlers

### Renderer Pattern

Multiple specialized renderers for different data types:

```python
# From cli.py imports
from jiratui.commands.render import (
    CLIExceptionRenderer,           # Error details
    JiraIssueCommentRenderer,       # Single comment
    JiraIssueCommentsRenderer,      # Comment list
    JiraIssueCommentTextRenderer,   # Comment text
    JiraIssueMetadataRenderer,      # Issue metadata
    JiraIssueSearchRenderer,        # Search results table
    JiraUserGroupRenderer,          # User groups
    JiraUserRenderer,               # User list
    ThemesRenderer,                 # Theme list
)

# Usage
render = JiraIssueSearchRenderer()
render.render(console, response)
```

**Pattern benefits**:
- Separation of concerns (data vs presentation)
- Reusable across commands
- Consistent Rich table formatting
- Easy to test renderers independently

## API Integration Patterns

### Authentication

**Personal Access Token (PAT)**:
```yaml
jira_api_username: 'user@example.com'
jira_api_token: 'your-personal-access-token'
```

**Pattern**: Token-based authentication for both Cloud and Data Center APIs.

### API Controller

From app.py:
```python
self.api = APIController()  # Initialized once in app
```

Shared across:
- All TUI screens (passed to MainScreen, ServerInfoScreen, etc.)
- CLI CommandHandler
- Background workers

**API lifecycle**:
```python
async def action_quit(self) -> None:
    await self.api.api.client.close_async_client()
    await self.api.api.async_http_client.close_async_client()
    self.app.exit()
```

### Async HTTP Client Pattern

```python
# Background worker for server info
self.run_worker(self._set_application_title_using_server_info())

async def _set_application_title_using_server_info(self) -> None:
    response_server_info: APIControllerResponse = await self.api.server_info()
    if response_server_info.success and response_server_info.result:
        self.server_info = response_server_info.result
        self.title = f'{self.title} - {self.server_info.base_url_or_server_title}'
```

**Patterns**:
- Async API calls with `await`
- Workers for background operations
- Response wrapper objects (`APIControllerResponse`)
- Success/failure handling with structured responses

### JQL Expression Support

```yaml
pre_defined_jql_expressions:
  1:
    label: "Work in the current sprint"
    expression: 'sprint in openSprints()'
  2:
    label: "My open issues"
    expression: 'assignee = currentUser() AND status != Done'
  3:
    label: "Critical bugs"
    expression: 'type = Bug AND priority = Highest'
```

**Usage**:
```bash
# Use pre-defined JQL expression ID 1
jiratui ui --jql-expression-id 1
```

**Pattern**: Named JQL expressions in config for common queries.

## Configuration Settings Reference

### Required Settings
```yaml
jira_api_username: 'user@example.com'
jira_api_token: 'token'
jira_api_base_url: 'https://instance.atlassian.net'
```

### Optional Settings

**Platform selection**:
```yaml
cloud: True  # Default: True (Cloud API), False (Data Center API)
jira_api_version: 3  # Default: 3 (only for cloud: True)
```

**User defaults**:
```yaml
jira_account_id: '098765'  # Your account ID for default assignee
default_project_key_or_id: 'SCRUM'  # Default project selection
```

**Search settings**:
```yaml
search_results_per_page: 65
search_issues_default_day_interval: 15
search_on_startup: False  # Auto-search when UI launches
```

**Display settings**:
```yaml
show_issue_web_links: True
ignore_users_without_email: True
```

**Custom fields**:
```yaml
custom_field_id_sprint: 'customfield_10020'  # Sprint field ID
```

**Logging**:
```yaml
log_level: 'WARNING'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
log_file: '/path/to/custom.log'  # Override XDG default
```

**TUI settings**:
```yaml
theme: 'monokai'  # Textual theme name
tui_title: 'My Jira'  # Custom title
tui_custom_title: 'Team SCRUM'  # Overrides tui_title
tui_title_include_jira_server_title: True  # Append server name
confirm_before_quit: True  # Show quit confirmation
```

**JQL expressions**:
```yaml
pre_defined_jql_expressions:
  1:
    label: "Description"
    expression: 'JQL query string'
jql_expression_id_for_work_items_search: 1  # Default JQL
```

## Enterprise Integration Lessons

### 1. Multi-Platform API Support

**Challenge**: Support both Jira Cloud (v3 API) and Jira Data Center (v2 API, on-premises).

**Solution**: Configuration-driven API version selection:
```yaml
cloud: False  # Automatically uses correct API version
jira_api_version: 2  # Manual override (cloud only)
```

**Pattern**: Single codebase, runtime API selection based on deployment target.

### 2. Configuration Management

**Challenge**: Enterprise users need flexible config location, env overrides, per-user settings.

**Solution**: XDG specification + environment variable overrides:
1. Check `$JIRA_TUI_CONFIG_FILE` (highest priority)
2. Check `$XDG_CONFIG_HOME/jiratui/config.yaml`
3. Fallback to `$HOME/.config/jiratui/config.yaml`

**Pattern**: Standards-compliant with escape hatches for special needs.

### 3. Hybrid CLI/TUI Design

**Challenge**: Different users have different workflows (automation vs exploration).

**Solution**: Shared business logic, dual interfaces:
- **CLI**: Automation, scripting, CI/CD
- **TUI**: Visual exploration, rapid navigation

**Pattern**: `CommandHandler` + `APIController` shared by both interfaces.

### 4. Authentication Strategy

**Challenge**: Enterprise security requirements (no password storage).

**Solution**: Personal Access Token (PAT) only:
```yaml
jira_api_token: 'your-PAT'  # No passwords
```

**Pattern**: Modern token-based auth, revocable, scoped permissions.

### 5. Rich Error Handling

**Challenge**: API failures need actionable feedback for users.

**Solution**: Exception hierarchy with extra details:
```python
try:
    response = handler.search_issues(...)
except CLIException as e:
    console.print(str(e))
    renderer = CLIExceptionRenderer()
    renderer.render(console, e.get_extra_details())
```

**Pattern**: Structured errors with human-readable rendering.

### 6. Async HTTP Client Management

**Challenge**: Long-running API calls block UI.

**Solution**: Async workers with proper lifecycle:
```python
self.run_worker(self._fetch_server_info())
# ... later
await self.api.client.close_async_client()
```

**Pattern**: Background workers + explicit client cleanup.

### 7. Structured Logging

**Challenge**: Enterprise debugging requires machine-readable logs.

**Solution**: JSON structured logging:
```python
from pythonjsonlogger.json import JsonFormatter
fh.setFormatter(JsonFormatter('%(asctime)s %(levelname)s %(message)s ...'))
```

**Pattern**: JSON logs for log aggregation systems (ELK, Splunk, etc.).

### 8. Shell Integration

**Challenge**: Terminal users expect shell completions.

**Solution**: Native Click completion support:
```bash
jiratui completions bash > ~/.local/share/bash-completion/completions/jiratui
```

**Pattern**: First-class shell integration for professional CLI tools.

## Code Examples

### Example 1: Launch TUI with Pre-selected Context

```bash
#!/bin/bash
# Launch JiraTUI for current sprint work assigned to me

MY_ACCOUNT_ID="5b10ac8d82e05b22cc7d4ef5"
PROJECT_KEY="SCRUM"
JQL_EXPRESSION_ID=1  # "sprint in openSprints()"

jiratui ui \
  --project-key "$PROJECT_KEY" \
  --assignee-account-id "$MY_ACCOUNT_ID" \
  --jql-expression-id "$JQL_EXPRESSION_ID" \
  --search-on-startup \
  --focus-item-on-startup 1 \
  --theme monokai
```

### Example 2: CLI Automation Script

```bash
#!/bin/bash
# Daily standup report: my open issues

MY_EMAIL="developer@company.com"

# Get my account ID
ACCOUNT_ID=$(jiratui users search "$MY_EMAIL" | grep -o 'Account ID: [^,]*' | cut -d' ' -f3)

# Search my open issues from last 7 days
jiratui issues search \
  --project-key SCRUM \
  --assignee-account-id "$ACCOUNT_ID" \
  --created-from $(date -d '7 days ago' +%Y-%m-%d) \
  --limit 50
```

### Example 3: CI/CD Integration

```bash
#!/bin/bash
# Update Jira issue when deployment succeeds

ISSUE_KEY="SCRUM-42"
STATUS_ID=10001  # "In Production" status

# Update issue status
if jiratui issues update "$ISSUE_KEY" --status-id "$STATUS_ID"; then
  # Add deployment comment
  COMMENT="Deployed to production by CI/CD pipeline $CI_PIPELINE_ID"
  jiratui comments add "$ISSUE_KEY" "$COMMENT"
  echo "Jira issue $ISSUE_KEY updated successfully"
else
  echo "Failed to update Jira issue $ISSUE_KEY"
  exit 1
fi
```

### Example 4: Custom Config Workflow

```bash
#!/bin/bash
# Use different configs for different Jira instances

# Production Jira
JIRA_TUI_CONFIG_FILE=~/.config/jiratui/prod.yaml jiratui ui

# Staging Jira
JIRA_TUI_CONFIG_FILE=~/.config/jiratui/staging.yaml jiratui ui

# Local on-premises Jira (Data Center)
JIRA_TUI_CONFIG_FILE=~/.config/jiratui/datacenter.yaml jiratui ui
```

### Example 5: Programmatic Issue Creation (concept)

While JiraTUI focuses on browsing/updating (no issue creation in current CLI), the pattern would be:

```bash
# If issue creation were supported, it would follow this pattern:
jiratui issues create \
  --project-key SCRUM \
  --type Bug \
  --summary "Login button not responding" \
  --description "Steps to reproduce: ..." \
  --assignee-account-id "$MY_ACCOUNT_ID" \
  --priority-id 3
```

## Key Takeaways for Textual Developers

### 1. Dual Interface Pattern
- Build **both** CLI and TUI for maximum utility
- Share business logic via handler/controller classes
- CLI for automation, TUI for exploration

### 2. Configuration Best Practices
- Follow XDG specification for config/log files
- Support environment variable overrides
- Provide `config` command to show active config location

### 3. Enterprise API Integration
- Use async HTTP clients for non-blocking API calls
- Implement proper client lifecycle (initialization + cleanup)
- Use workers for background operations
- Structured error handling with user-friendly rendering

### 4. Professional CLI Design
- Rich table rendering for readable output
- Status spinners for long operations
- Exception hierarchy with actionable error messages
- Shell completion support

### 5. Theme and Customization
- Allow theme selection via CLI and config
- Graceful fallback for invalid themes
- Support custom titles and branding

### 6. Logging for Production
- JSON structured logging for enterprise log aggregation
- XDG-compliant log file locations
- Configurable log levels
- Environment variable override for log file path

### 7. State Management
- Initialize from CLI args → config → sensible defaults
- Pass initial state to screens/widgets
- Use global config singleton with context access

### 8. Screen Architecture
- Modular screens (Main, Help, ServerInfo, ConfigInfo, Quit)
- Context-sensitive help with anchor navigation
- Confirmation dialogs for destructive actions

## Performance Considerations

### API Call Optimization
- **Pagination**: `search_results_per_page: 65` (configurable)
- **Field selection**: Request only needed fields
- **JQL expressions**: Pre-defined queries for common searches
- **Async workers**: Background API calls don't block UI

### Rendering Performance
- **Rich tables**: Efficient terminal rendering
- **Lazy loading**: Paginated results (not all data at once)
- **Status feedback**: Spinners during long operations

## Testing and Development

From repository structure:
```
jiratui/
├── .github/workflows/       # GitHub Actions CI
│   ├── test.yaml           # Automated tests
│   └── codeql.yaml         # Security scanning
├── src/jiratui/            # Main source
│   ├── app.py              # Textual application
│   ├── cli.py              # Click CLI
│   ├── api_controller/     # API integration
│   ├── commands/           # Command handlers
│   ├── config.py           # Configuration
│   └── widgets/            # Textual widgets
├── docs/                   # Sphinx documentation
├── pyproject.toml          # Poetry/uv project
├── ruff.toml               # Linting config
├── .pre-commit-config.yaml # Pre-commit hooks
└── Makefile                # Development tasks
```

**Development setup**:
```bash
# Clone repository
git clone https://github.com/whyisdifficult/jiratui.git
cd jiratui

# Install with uv (recommended)
uv sync

# Install pre-commit hooks
pre-commit install

# Run tests
make test

# Lint
make lint
```

## Sources

**GitHub Repository:**
- [whyisdifficult/jiratui](https://github.com/whyisdifficult/jiratui) - Main repository (accessed 2025-11-02)
- [README.md](https://github.com/whyisdifficult/jiratui/blob/main/README.md) - Installation and features
- [app.py](https://github.com/whyisdifficult/jiratui/blob/main/src/jiratui/app.py) - Textual application implementation
- [cli.py](https://github.com/whyisdifficult/jiratui/blob/main/src/jiratui/cli.py) - Click CLI implementation
- [jiratui.example.yaml](https://github.com/whyisdifficult/jiratui/blob/main/jiratui.example.yaml) - Configuration example

**Documentation:**
- [JiraTUI Documentation](https://jiratui.readthedocs.io) - Official documentation (accessed 2025-11-02)

**API References:**
- [Jira Cloud REST API v3](https://developer.atlassian.com/cloud/jira/platform/rest/v3/intro/)
- [Jira Data Center REST API v2](https://developer.atlassian.com/server/jira/platform/rest/v11001/intro/)
- [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/latest/)

**Related Resources:**
- [Textual Framework](https://textual.textualize.io/)
- [Rich Library](https://rich.readthedocs.io/)
- [Click CLI Framework](https://click.palletsprojects.com/)
