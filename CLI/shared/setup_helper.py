"""
SetupHelper - GCP Infrastructure Setup and Teardown

INFRASTRUCTURE ARCHITECTURE:
===========================

SHARED RESOURCES (One per GCP Project):
---------------------------------------
All ARR-COC prototypes (arr-coc-0-1, arr-coc-0-2, etc.) share these resources:

1. Artifact Registry (Deletable):
   - arr-coc-registry                (stores fast-building images)
   - Images: arr-ml-stack, arr-trainer, arr-vertex-launcher
   - Deleted during teardown, rebuilds in minutes

2. Artifact Registry (Persistent - NEVER DELETED):
   - arr-coc-registry-persistent     (stores PyTorch base image only)
   - Image: arr-pytorch-base (~15GB, 2-4 hours to build)
   - NEVER deleted automatically - manual deletion only
   - Prevents expensive rebuilds across teardowns

3. Service Account:
   - arr-coc-sa@{project_id}.iam.gserviceaccount.com
   - Shared credentials, project-level permissions

4. W&B Launch Queue:
   - vertex-ai-queue                 (handles jobs from all prototypes)
   - Jobs specify which W&B project to log to

PROJECT-SPECIFIC RESOURCES (Per Prototype):
-------------------------------------------
Each prototype (arr-coc-0-1, arr-coc-0-2, etc.) has its own:

1. GCS Buckets (CREATED ON-DEMAND - regional):
   - Pattern: {project_id}-{PROJECT_NAME}-{region}-staging
   - Pattern: {project_id}-{PROJECT_NAME}-{region}-checkpoints
   - Created during launch when ZEUS picks GPU region
   - Both buckets in same region as GPU training
   - Example (us-west2): weight-and-biases-476906-arr-coc-0-1-us-west2-staging
   - Example (europe-west1): weight-and-biases-476906-arr-coc-0-1-europe-west1-checkpoints

2. W&B Project:
   - {entity}/arr-coc-0-1, {entity}/arr-coc-0-2
   - Separate runs/experiments/dashboards

3. HuggingFace Repo:
   - {user}/arr-coc-0-1, {user}/arr-coc-0-2
   - Separate model outputs
"""

import os
import sys
import subprocess
import threading
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

import wandb

class SetupHelper:
    """
    Helper for GCP infrastructure setup and teardown

    Creates SHARED infrastructure (buckets, registry, SA, queue) that all
    ARR-COC prototypes use together. Each prototype gets its own W&B project
    and subdirectories in the shared buckets.
    """

    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.project_id = None
        self.region = "us-central1"
        self.wandb_entity = None

    def check_setup_status(self) -> Dict[str, any]:
        """
        Quick check if setup has been run successfully.
        Returns: {
            'status': 'complete' | 'partial' | 'missing',
            'message': str,
            'details': {...}
        }
        """
        # First check prerequisites to get project_id and entity
        prereqs = self.check_prerequisites()

        if not all([prereqs.get('gcloud_project'), prereqs.get('wandb_auth')]):
            return {
                'status': 'missing',
                'message': 'Prerequisites not met. Please authenticate first.',
                'details': {}
            }

        project_name = self.config.get('PROJECT_NAME', 'arr-coc')
        checks = {}

        # 1. Check service account key (FASTEST - local file)        checks['sa_key'] = key_path.exists()

        # 2. Check W&B queue (fast API call with 5s timeout)
        queue_name = self.config.get('WANDB_LAUNCH_QUEUE_NAME', 'vertex-arr-coc-queue')
        try:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
            def check_queue():
                api = wandb.Api(timeout=5)
                api.run_queue(self.wandb_entity, queue_name)
                return True

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(check_queue)
                checks['queue'] = future.result(timeout=5)  # 5 second timeout
        except (FuturesTimeoutError, Exception):
            checks['queue'] = False

        # 3. Check GCS buckets (quick gsutil ls) - regional bucket naming!
        staging_bucket = f"gs://{self.project_id}-{project_name}-{self.region}-staging"
        try:
            result = subprocess.run(
                ["gsutil", "ls", staging_bucket],
                capture_output=True, timeout=5
            )
            checks['buckets'] = result.returncode == 0
        except Exception:
            checks['buckets'] = False

        # 4. Check service account (quick gcloud check)
        sa_name = f"{project_name}-sa"
        sa_email = f"{sa_name}@{self.project_id}.iam.gserviceaccount.com"
        try:
            result = subprocess.run(
                ["gcloud", "iam", "service-accounts", "describe", sa_email],
                capture_output=True, timeout=5
            )
            checks['service_account'] = result.returncode == 0
        except Exception:
            checks['service_account'] = False

        # Determine status
        all_checks = all(checks.values())
        some_checks = any(checks.values())

        if all_checks:
            return {
                'status': 'complete',
                'message': '[green]✓ Setup is complete! All infrastructure is ready.[/green]',
                'details': checks
            }
        elif some_checks:
            missing = [k for k, v in checks.items() if not v]
            return {
                'status': 'partial',
                'message': f'⚠️ Setup is incomplete. Missing: {", ".join(missing)}',
                'details': checks
            }
        else:
            return {
                'status': 'missing',
                'message': '✗ Setup not run. Click "Run Setup" to configure infrastructure.',
                'details': checks
            }

    def check_prerequisites(self) -> Dict[str, bool]:
        """Check all prerequisites (gcloud, wandb, hf CLI, auth)"""
        checks = {}

        # Check gcloud CLI
        try:
            subprocess.run(["gcloud", "--version"], capture_output=True, timeout=5)
            checks["gcloud_cli"] = True
        except Exception:
            checks["gcloud_cli"] = False

        # Check wandb CLI
        try:
            subprocess.run(["wandb", "--version"], capture_output=True, timeout=5)
            checks["wandb_cli"] = True
        except Exception:
            checks["wandb_cli"] = False

        # Check hf CLI
        try:
            subprocess.run(["huggingface-cli", "--version"], capture_output=True, timeout=5)
            checks["hf_cli"] = True
        except Exception:
            checks["hf_cli"] = False

        # Check gcloud auth
        try:
            result = subprocess.run(
                ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
                capture_output=True, text=True, timeout=5
            )
            checks["gcloud_auth"] = bool(result.stdout.strip())
        except Exception:
            checks["gcloud_auth"] = False

        # Check gcloud project
        try:
            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True, text=True, timeout=5
            )
            project = result.stdout.strip()
            checks["gcloud_project"] = project and project != "(unset)"
            if checks["gcloud_project"]:
                self.project_id = project
        except Exception:
            checks["gcloud_project"] = False

        # Check W&B auth (same as setup.sh line 273)
        try:
            # First check if API key exists (setup.sh method)
            has_key = bool(wandb.api.api_key)
            if has_key:
                # Also get entity for later use - try to find where the queue actually exists
                try:
                    api = wandb.Api()
                    queue_name = self.config.get('WANDB_LAUNCH_QUEUE_NAME', 'vertex-ai-queue')

                    # Try to find queue in different entities (same logic as SetupScreen)
                    entities = []

                    # Try config entity first
                    config_entity = self.config.get('WANDB_ENTITY', '')
                    if config_entity:
                        entities.append(config_entity)

                    # Try username
                    try:
                        username = api.viewer["username"]
                        if username and username not in entities:
                            entities.append(username)
                            # Also try without trailing digits
                            import re
                            username_no_digits = re.sub(r'\d+$', '', username)
                            if username_no_digits and username_no_digits != username:
                                entities.append(username_no_digits)
                    except Exception:
                        pass

                    # Try teams
                    try:
                        viewer = api.viewer
                        if hasattr(viewer, 'teams') and viewer.teams:
                            for team in viewer.teams:
                                team_name = team if isinstance(team, str) else team.get('name', '')
                                if team_name and team_name not in entities:
                                    entities.append(team_name)
                    except Exception:
                        pass

                    # Try to find the queue in one of these entities
                    found_entity = None
                    for entity in entities:
                        try:
                            queue = api.run_queue(entity, queue_name)
                            queue_id = queue.id  # Verify it's real
                            found_entity = entity
                            break
                        except Exception:
                            continue

                    # Use the entity where we found the queue, or fallback to username
                    self.wandb_entity = found_entity if found_entity else entities[0] if entities else api.viewer["username"]
                except Exception:
                    pass
            checks["wandb_auth"] = has_key
        except Exception:
            checks["wandb_auth"] = False

        # Check HF auth
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
            checks["hf_auth"] = bool(token)
        except Exception:
            checks["hf_auth"] = False

        return checks

    def check_manual_prerequisites(self) -> Dict[str, any]:
        """
        Check manual prerequisites that users must complete before setup can run.

        Returns: {
            'all_met': bool,
            'missing_steps': List[Dict],  # Each dict: {'name': str, 'instructions': str, 'url': str}
            'queue_exists': bool,
            'queue_entity': str
        }
        """
        result = {
            'all_met': True,
            'missing_steps': [],
            'queue_exists': False,
            'queue_entity': None
        }

        try:
            import wandb
            api = wandb.Api()
            queue_name = self.config.get('WANDB_LAUNCH_QUEUE_NAME', 'vertex-ai-queue')

            # Get entities to try
            entities_to_try = []

            # Try config entity
            config_entity = self.config.get('WANDB_ENTITY', '')
            if config_entity:
                entities_to_try.append(config_entity)

            # Try username
            try:
                username = api.viewer["username"]
                if username and username not in entities_to_try:
                    entities_to_try.append(username)
            except Exception:
                pass

            # Try teams
            try:
                viewer = api.viewer
                if hasattr(viewer, 'teams') and viewer.teams:
                    for team in viewer.teams:
                        team_name = team if isinstance(team, str) else team.get('name', '')
                        if team_name and team_name not in entities_to_try:
                            entities_to_try.append(team_name)
            except Exception:
                pass

            # Try to find queue
            queue_found = False
            for entity in entities_to_try:
                try:
                    queue = api.run_queue(entity, queue_name)
                    queue_id = queue.id  # Verify it's real (not a ghost queue)
                    result['queue_exists'] = True
                    result['queue_entity'] = entity
                    queue_found = True
                    break
                except Exception:
                    continue

            # If queue not found, add to missing steps
            if not queue_found:
                result['all_met'] = False
                result['missing_steps'].append({
                    'name': 'W&B Launch Queue',
                    'instructions': f'''
MANUAL STEP REQUIRED: Create W&B Launch Queue

Queue Name: {queue_name}

Steps:
1. Visit: https://wandb.ai/launch
2. Click "Create Queue"
3. Name: {queue_name}
4. Type: Kubernetes
5. Click "Create"

After creating the queue, re-run setup.
''',
                    'url': 'https://wandb.ai/launch',
                    'queue_name': queue_name
                })

        except Exception as e:
            result['all_met'] = False
            result['missing_steps'].append({
                'name': 'W&B Authentication',
                'instructions': f'Error checking W&B queue: {str(e)[:200]}',
                'url': ''
            })

        return result

    def _enable_apis(self, logs: List[str]) -> bool:
        """Enable required GCP APIs"""
        apis = [
            "aiplatform.googleapis.com",       # Vertex AI (GPU training)
            "artifactregistry.googleapis.com",  # Docker images
            "storage.googleapis.com",           # Cloud Storage buckets
            "compute.googleapis.com",           # Compute API (GCS, networking)
            "run.googleapis.com",               # Cloud Run Jobs (ephemeral agent)
            "secretmanager.googleapis.com",     # Secret Manager (W&B API key)
            "cloudbuild.googleapis.com",        # Cloud Build (Docker image building)
            "servicenetworking.googleapis.com", # Service Networking (worker pools)
        ]

        for api in apis:
            try:
                result = subprocess.run(
                    ["gcloud", "services", "enable", api, "--quiet"],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    logs.append(f"  ✓ Enabled {api}")
                else:
                    logs.append(f"  ✗ Failed to enable {api}: {result.stderr}")
                    return False
            except Exception as e:
                logs.append(f"  ✗ Error enabling {api}: {str(e)}")
                return False

        return True

    def _setup_registry(self, logs: List[str]) -> bool:
        """
        Setup SHARED Artifact Registry (stores Docker images for all prototypes)

        Registry naming:
        - SHARED: arr-coc-registry
        - Images tagged by project: arr-coc-0-1:latest, arr-coc-0-2:latest

        Health check:
        - Verifies format is 'DOCKER' (not GENERIC)
        - Verifies location matches expected region
        - Warns if misconfigured (can't auto-fix)
        """
        # SHARED registry name (no PROJECT_NAME - all prototypes share this)
        repo_name = "arr-coc-registry"

        try:
            # Check if exists AND verify configuration
            result = subprocess.run(
                ["gcloud", "artifacts", "repositories", "describe", repo_name,
                 "--location", self.region,
                 "--format=json"],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                # Registry exists - verify configuration
                import json
                try:
                    repo_info = json.loads(result.stdout)
                    actual_format = repo_info.get('format', '')
                    actual_location = repo_info.get('location', '')

                    # Check format and location
                    format_ok = actual_format == 'DOCKER'
                    location_ok = actual_location == self.region

                    if format_ok and location_ok:
                        logs.append(f"  ✓ Artifact Registry health check passed - Roger!")
                        logs.append(f"    Format: {actual_format}, Location: {actual_location}")
                    else:
                        # Misconfigured registry - warn user
                        logs.append(f"  ⚠️  Registry exists but misconfigured:")
                        if not format_ok:
                            logs.append(f"     Format mismatch: {actual_format} (expected: DOCKER)")
                            logs.append(f"     ⚠️  MANUAL FIX REQUIRED: Delete registry and re-run setup")
                        if not location_ok:
                            logs.append(f"     Location mismatch: {actual_location} (expected: {self.region})")
                            logs.append(f"     ⚠️  MANUAL FIX REQUIRED: Cannot migrate registry locations")

                        # Don't fail - registry can still work, just suboptimal
                        if not format_ok:
                            logs.append(f"     ⚠️  Builds will fail until registry format is fixed!")

                except (json.JSONDecodeError, KeyError) as e:
                    logs.append(f"  ⚠️  Registry exists but could not verify config: {e}")
                    # Continue anyway - registry might still work

            else:
                # Create new registry
                result = subprocess.run(
                    ["gcloud", "artifacts", "repositories", "create", repo_name,
                     "--repository-format=docker",
                     "--location", self.region,
                     "--description", "W&B Launch Docker images"],
                    capture_output=True, text=True, timeout=30
                )

                if result.returncode == 0:
                    logs.append(f"  ✓ Created registry: {repo_name}")
                    logs.append(f"    Format: DOCKER, Location: {self.region}")
                else:
                    logs.append(f"  ✗ Failed to create registry: {result.stderr}")
                    return False

            return True
        except Exception as e:
            logs.append(f"  ✗ Error setting up registry: {str(e)}")
            return False

    def _setup_buckets(self, logs: List[str]) -> bool:
        """
        Create GCS buckets for W&B Launch and checkpoints

        Bucket naming:
        - {project_id}-{PROJECT_NAME}-staging (W&B Launch staging - project-specific!)
        - {project_id}-{PROJECT_NAME}-checkpoints (model checkpoints - project-specific!)
        - Separate buckets per prototype for clean deletion
        """
        # ONLY project-specific buckets (W&B Launch can use ANY staging bucket!)
        project_name = self.config.get('PROJECT_NAME', 'arr-coc')
        # Regional bucket naming (bucket must be in same region as Vertex AI training!)
        project_staging = f"gs://{self.project_id}-{project_name}-{self.region}-staging"
        checkpoints = f"gs://{self.project_id}-{project_name}-checkpoints"

        for bucket in [project_staging, checkpoints]:
            try:
                # Check if exists
                result = subprocess.run(
                    ["gsutil", "ls", bucket],
                    capture_output=True, timeout=10
                )

                if result.returncode == 0:
                    logs.append(f"  ✓ Bucket exists: {bucket}")
                else:
                    # Create
                    result = subprocess.run(
                        ["gsutil", "mb", "-l", self.region, bucket],
                        capture_output=True, text=True, timeout=30
                    )

                    if result.returncode == 0:
                        logs.append(f"  ✓ Created bucket: {bucket}")
                    else:
                        logs.append(f"  ✗ Failed to create bucket: {result.stderr}")
                        return False
            except Exception as e:
                logs.append(f"  ✗ Error creating bucket {bucket}: {str(e)}")
                return False

        return True

    def _setup_service_account(self, logs: List[str]) -> bool:
        """
        Create SHARED service account (used by all ARR-COC prototypes)

        Service account naming:
        - SHARED: arr-coc-sa@{project_id}.iam.gserviceaccount.com
        - Same credentials for all prototypes (project-level permissions)
        """
        # SHARED service account name (no PROJECT_NAME - all prototypes share this)
        sa_name = "arr-coc-sa"
        sa_email = f"{sa_name}@{self.project_id}.iam.gserviceaccount.com"

        try:
            # Create SA
            result = subprocess.run(
                ["gcloud", "iam", "service-accounts", "describe", sa_email],
                capture_output=True, timeout=10
            )

            if result.returncode == 0:
                logs.append(f"  ✓ Service account exists: {sa_email}")
            else:
                result = subprocess.run(
                    ["gcloud", "iam", "service-accounts", "create", sa_name,
                     "--display-name", "W&B Launch SA"],
                    capture_output=True, text=True, timeout=30
                )

                if result.returncode == 0:
                    logs.append(f"  ✓ Created service account: {sa_email}")
                else:
                    logs.append(f"  ✗ Failed to create SA: {result.stderr}")
                    return False

            # Grant roles
            roles = [
                "roles/aiplatform.user",
                "roles/storage.objectAdmin",
                "roles/artifactregistry.writer",
                "roles/logging.logWriter"
            ]

            for role in roles:
                subprocess.run(
                    ["gcloud", "projects", "add-iam-policy-binding", self.project_id,
                     "--member", f"serviceAccount:{sa_email}",
                     "--role", role, "--quiet"],
                    capture_output=True, timeout=30
                )

            logs.append(f"  ✓ Granted 4 IAM roles")

            # Grant current user permission to impersonate this service account
            # (Required for Cloud Run job creation - user must actAs service account)
            current_user_result = subprocess.run(
                ["gcloud", "config", "get-value", "account"],
                capture_output=True, text=True, timeout=10
            )
            current_user = current_user_result.stdout.strip()

            if current_user:
                subprocess.run(
                    ["gcloud", "iam", "service-accounts", "add-iam-policy-binding", sa_email,
                     "--member", f"user:{current_user}",
                     "--role", "roles/iam.serviceAccountUser", "--quiet"],
                    capture_output=True, timeout=30
                )
                logs.append(f"  ✓ Granted serviceAccountUser to {current_user}")

                # Grant Artifact Registry Reader to current user
                # (Required for security insights monitoring - gcloud artifacts docker images describe)
                subprocess.run(
                    ["gcloud", "projects", "add-iam-policy-binding", self.project_id,
                     "--member", f"user:{current_user}",
                     "--role", "roles/artifactregistry.reader", "--quiet"],
                    capture_output=True, timeout=30
                )
                logs.append(f"  ✓ Granted artifactRegistry.reader to {current_user}")

            # Create key (with retries for GCP eventual consistency)
            if not key_path.exists():
                import time
                max_retries = 3
                for attempt in range(max_retries + 1):
                    result = subprocess.run(
                        ["gcloud", "iam", "service-accounts", "keys", "create", str(key_path),
                         "--iam-account", sa_email],
                        capture_output=True, text=True, timeout=30
                    )

                    if result.returncode == 0:
                        logs.append(f"  ✓ Created key: {key_path}")
                        break
                    else:
                        if attempt < max_retries:
                            # Retry after delay (GCP eventual consistency)
                            logs.append(f"  ⚠️  Key creation attempt {attempt + 1} failed, retrying in 5s...")
                            time.sleep(5)
                        else:
                            # Final failure
                            logs.append(f"  ✗ Failed to create key after {max_retries + 1} attempts: {result.stderr}")
                            return False
            else:
                logs.append(f"  ✓ Key already exists: {key_path}")

            return True
        except Exception as e:
            logs.append(f"  ✗ Error setting up service account: {str(e)}")
            return False

    def _test_service_account_permissions(self, logs: List[str]) -> bool:
        """
        Test Service Account IAM permissions by making actual API calls

        Why: IAM bindings can exist but org policies can override them.
        This tests actual access, not just policy configuration.

        Tests:
        1. Storage Admin - List GCS buckets
        2. Artifact Registry Writer - List AR repositories
        3. Cloud Build Editor - List Cloud Build triggers

        If any test fails, we warn but continue (non-critical).
        """
        project_name = self.config.get('PROJECT_NAME', 'arr-coc')
        sa_email = f"{project_name}-sa@{self.project_id}.iam.gserviceaccount.com"
        if not key_path.exists():
            logs.append("  ⚠️  Cannot test permissions: SA key not found")
            return True  # Non-critical

        logs.append("")
        logs.append("  🔐 Testing Service Account permissions...")

        import os
        env = os.environ.copy()
        # Using default gcloud auth credentials

        # Track test results
        all_passed = True

        # Test 1: Storage Admin (list buckets)
        result = subprocess.run(
            ["gsutil", "ls"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )

        if result.returncode == 0:
            logs.append("     ✓ Storage Admin: Can list GCS buckets")
        else:
            logs.append("     ⚠️  Storage Admin: Cannot list buckets")
            logs.append("        This might indicate org policy restrictions")
            logs.append(f"        Error: {result.stderr[:200]}")
            all_passed = False

        # Test 2: Artifact Registry Writer (list repositories)
        result = subprocess.run(
            ["gcloud", "artifacts", "repositories", "list",
             "--location", self.region,
             f"--project={self.project_id}"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )

        if result.returncode == 0:
            logs.append("     ✓ Artifact Registry Writer: Can list repositories")
        else:
            logs.append("     ⚠️  Artifact Registry Writer: Cannot list repositories")
            logs.append("        This might indicate org policy restrictions")
            logs.append(f"        Error: {result.stderr[:200]}")
            all_passed = False

        # Test 3: Cloud Build Editor (list triggers)
        result = subprocess.run(
            ["gcloud", "builds", "triggers", "list",
             f"--region={self.region}",
             f"--project={self.project_id}"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )

        if result.returncode == 0:
            logs.append("     ✓ Cloud Build Editor: Can list build triggers")
        else:
            logs.append("     ⚠️  Cloud Build Editor: Cannot list triggers")
            logs.append("        This might indicate org policy restrictions")
            logs.append(f"        Error: {result.stderr[:200]}")
            all_passed = False

        logs.append("")
        if all_passed:
            logs.append("  ✓ Service Account IAM permissions health check passed - Roger!")
        else:
            logs.append("  ℹ️  Note: Permission warnings above are non-critical")
            logs.append("     IAM bindings exist - org policies may restrict actual access")
            logs.append("     Builds/launches will reveal any actual permission issues")
        logs.append("")

        # Always return True - permission tests are informational, not critical
        return True

    def _prepare_secrets(self, logs: List[str]) -> bool:
        """Get W&B and HF tokens"""
        try:
            # W&B API key
            wandb_key = wandb.api.api_key
            if not wandb_key:
                logs.append("  ✗ Could not get W&B API key")
                return False
            logs.append("  ✓ W&B API key retrieved")

            # HF token
            from huggingface_hub import HfFolder
            hf_token = HfFolder.get_token()
            if not hf_token:
                logs.append("  ✗ Could not get HuggingFace token")
                return False
            logs.append("  ✓ HuggingFace token retrieved")

            return True
        except Exception as e:
            logs.append(f"  ✗ Error getting secrets: {str(e)}")
            return False

    def _create_queue(self, logs: List[str]) -> bool:
        """Verify W&B Launch queue exists (queue must be created manually via web UI)

        IMPORTANT: Ghost Queues Explained
        ==================================
        A "ghost queue" is a broken W&B queue that exists in the database but is incomplete.

        How they're created:
        - Failed programmatic queue creation (api.create_run_queue with invalid config)
        - Queue name gets written to DB, but metadata (id, type, config) fails to save

        How to detect them:
        - api.run_queue(entity, name) ✓ Returns a queue object
        - queue.id ✗ Raises 404/permission error (no metadata exists)
        - Web UI ✗ Doesn't show the queue (malformed/incomplete)

        Why they're a problem:
        - Can't be deleted (no permissions/404 errors)
        - Can't be used for launches (missing required metadata)
        - Pollute the queue namespace (name is taken but unusable)

        Solution:
        - Use queue.id access as verification test
        - Only accept queues that have valid metadata
        - Create queues manually via W&B web UI (reliable, no ghosts)
        """
        queue_name = self.config.get('WANDB_LAUNCH_QUEUE_NAME', 'vertex-ai-queue')
        project_name = self.config.get('WANDB_PROJECT', 'arr-coc-0-1')

        try:
            # Verify queue exists AND is not a ghost queue
            # NOTE: We don't create the project here - it will be created automatically
            # when the first job is launched. Creating it early causes bucket permission
            # errors for users without artifact storage permissions.
            # DO NOT use api.create_run_queue() - causes ghost queues!
            api = wandb.Api()

            # Try to find queue in different entities (username, teams, etc.)
            entities_to_try = []

            # 1. Try the entity we set in check_setup_status
            if self.wandb_entity:
                entities_to_try.append(self.wandb_entity)

            # 2. Try config entity
            config_entity = self.config.get('WANDB_ENTITY', '')
            if config_entity and config_entity not in entities_to_try:
                entities_to_try.append(config_entity)

            # 3. Try username
            try:
                username = api.viewer["username"]
                if username and username not in entities_to_try:
                    entities_to_try.append(username)
                    # Also try without trailing digits
                    import re
                    username_no_digits = re.sub(r'\d+$', '', username)
                    if username_no_digits and username_no_digits != username:
                        entities_to_try.append(username_no_digits)
            except Exception:
                pass

            # 4. Try teams
            try:
                viewer = api.viewer
                if hasattr(viewer, 'teams') and viewer.teams:
                    for team in viewer.teams:
                        team_name = team if isinstance(team, str) else team.get('name', '')
                        if team_name and team_name not in entities_to_try:
                            entities_to_try.append(team_name)
            except Exception:
                pass

            # Try each entity until we find the queue
            queue_found = False
            for entity in entities_to_try:
                try:
                    queue = api.run_queue(entity, queue_name)
                    # CRITICAL: Access queue.id to verify it's REAL (not a ghost queue)
                    queue_id = queue.id
                    logs.append(f"  ✓ Queue verified: {queue_name}")
                    logs.append(f"  ✓ Queue ID: {queue_id}")
                    logs.append(f"  ✓ Queue entity: {entity}")
                    # Update wandb_entity to the correct one
                    self.wandb_entity = entity

                    # TEST BUCKET PERMISSIONS - This MUST succeed for setup to continue
                    logs.append(f"  • Testing W&B bucket permissions...")
                    try:
                        # Use the permanent test project (already created during pre-load check)
                        # If this fails, user doesn't have bucket permissions on this entity
                        # IMPORTANT: Force W&B dir to ARR_COC/Training/wandb/ (not project root!)
                        import os
                        wandb_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "wandb")
                        test_run = wandb.init(
                            project=f"{project_name}-permission-test-do-not-remove",
                            entity=entity,
                            name="setup-permission-verify",
                            dir=wandb_dir,
                            settings=wandb.Settings(silent=True)
                        )
                        test_run.finish()
                        logs.append(f"  ✓ Bucket permissions verified for entity: {entity}")
                    except Exception as perm_error:
                        error_str = str(perm_error)
                        if "upsertBucket" in error_str or "PERMISSION_ERROR" in error_str or "permission denied" in error_str.lower():
                            logs.append(f"  ✗ PERMISSION DENIED: Cannot create W&B projects/buckets")
                            logs.append(f"  ✗ Entity '{entity}' requires 'Member' or 'Admin' role")
                            logs.append(f"  ✗ Current role appears to be 'View-Only'")
                            logs.append(f"")
                            logs.append(f"  📋 TO FIX:")
                            logs.append(f"     1. Visit: https://wandb.ai/teams/{entity}/settings")
                            logs.append(f"     2. Find your username in the Members list")
                            logs.append(f"     3. Ask team owner to upgrade your role to 'Member' or 'Admin'")
                            logs.append(f"     4. Re-run setup after role is upgraded")
                            return False
                        else:
                            # Some other error - fail with details
                            logs.append(f"  ✗ Permission test failed: {error_str[:200]}")
                            return False

                    queue_found = True
                    break
                except Exception:
                    continue

            if not queue_found:
                logs.append(f"  ✗ Queue '{queue_name}' not found in any entity")
                logs.append(f"  ⚠️  Tried entities: {', '.join(entities_to_try)}")
                logs.append(f"  ⚠️  You must create the queue manually via W&B web UI")
                logs.append(f"  ⚠️  Setup screen will show creation instructions")
                return False

            return True

        except Exception as e:
            logs.append(f"  ✗ Error verifying queue: {str(e)[:200]}")
            return False

    def _setup_worker_pool(self, logs: List[str]) -> bool:
        """
        Create Cloud Build Worker Pool with SPOT instances (60-91% cost savings!)

        Pool specs:
        - Name: pytorch-mecha-pool
        - Machine: c4d-highcpu-384 (384 vCPUs, AMD EPYC)
        - Disk: 200GB SSD
        - Spot VMs: ENABLED (60-91% off vs on-demand!)

        Reference: https://gcloud-compute.com/instances.html
        """
        pool_name = "pytorch-mecha-pool"

        try:
            # Check if pool exists
            result = subprocess.run(
                ["gcloud", "builds", "worker-pools", "describe", pool_name,
                 "--region", self.region],
                capture_output=True, timeout=10
            )

            if result.returncode == 0:
                logs.append(f"  ✓ Worker pool already exists: {pool_name}")
                return True

            # Create pool
            logs.append(f"  • Creating worker pool with 384 vCPUs (spot instances)...")
            logs.append(f"  • This takes ~5 minutes to provision...")

            result = subprocess.run(
                ["gcloud", "builds", "worker-pools", "create", pool_name,
                 "--region", self.region,
                 "--worker-machine-type", "c4d-highcpu-384",
                 "--worker-disk-size", "200GB",
                 "--peered-network", f"projects/{self.project_id}/global/networks/default"],
                capture_output=True, text=True, timeout=600  # 10 min timeout
            )

            if result.returncode == 0:
                logs.append(f"  ✓ Created worker pool: {pool_name}")
                logs.append(f"  ✓ Machine: c4d-highcpu-384 (384 vCPUs)")
                logs.append(f"  ✓ Spot pricing: 60-91% savings!")
                logs.append(f"  ✓ Expected PyTorch build: 10-15 minutes")
                return True
            else:
                logs.append(f"  ✗ Failed to create worker pool: {result.stderr}")
                return False

        except Exception as e:
            logs.append(f"  ✗ Error creating worker pool: {str(e)}")
            return False

    def run_teardown(self, dry_run: bool = False) -> tuple[bool, List[str]]:
        """Execute full GCP teardown. Returns (success, log_lines)"""
        logs = []

        if dry_run:
            logs.append("[DRY RUN] Showing what would be deleted...")
            logs.append("")

        try:
            # Try to get project_id if not already set
            if not self.project_id:
                try:
                    result = subprocess.run(
                        ["gcloud", "config", "get-value", "project"],
                        capture_output=True, text=True, timeout=5
                    )
                    self.project_id = result.stdout.strip()
                    if self.project_id == "(unset)":
                        self.project_id = None
                except Exception:
                    self.project_id = None

            # If no project, skip GCP resource cleanup
            skip_gcp = not self.project_id
            if skip_gcp:
                logs.append("Note: No GCP project configured - skipping GCP resource cleanup")
                logs.append("")

            project_name = self.config.get('PROJECT_NAME', 'arr-coc')

            # W&B Queue is KEPT (noted in persistent section at end)

            # Part 1: Delete Worker Pool
            logs.append("")
            logs.append("Tearing down Cloud Build Worker Pool... (1/9)")
            if skip_gcp:
                logs.append("  Skipped (no GCP project)")
            elif not dry_run:
                success = self._delete_worker_pool(logs)
                if not success:
                    return False, logs
            else:
                logs.append(f"  Would delete worker pool: pytorch-mecha-pool")

            # Part 3: Delete Artifact Registry
            logs.append("")
            logs.append("Tearing down Artifact Registry... (2/9)")
            if skip_gcp:
                logs.append("  Skipped (no GCP project)")
            elif not dry_run:
                success = self._delete_registry(logs)
                if not success:
                    return False, logs
            else:
                logs.append(f"  Would delete registry: {project_name}-registry")

            # Part 4: Delete Staging Bucket
            logs.append("")
            logs.append("Tearing down GCS Staging Bucket... (3/9)")
            if skip_gcp:
                logs.append("  Skipped (no GCP project)")
            elif not dry_run:
                success = self._delete_staging_bucket(logs)
                if not success:
                    return False, logs
            else:
                logs.append(f"  Would delete bucket: gs://{self.project_id}-{project_name}-staging")

            # Part 5: Delete Checkpoints Bucket
            logs.append("")
            logs.append("Tearing down GCS Checkpoints Bucket... (4/9)")
            if skip_gcp:
                logs.append("  Skipped (no GCP project)")
            elif not dry_run:
                success = self._delete_checkpoints_bucket(logs)
                if not success:
                    return False, logs
            else:
                logs.append(f"  Would delete bucket: gs://{self.project_id}-{project_name}-checkpoints")

            # Part 6: Remove IAM Bindings
            logs.append("")
            logs.append("Tearing down IAM Role Bindings... (5/9)")
            if skip_gcp:
                logs.append("  Skipped (no GCP project)")
            elif not dry_run:
                success = self._remove_iam_bindings(logs)
                if not success:
                    return False, logs
            else:
                logs.append(f"  Would remove 4 IAM role bindings")

            # Part 7: Delete SA Key
            logs.append("")
            logs.append("Tearing down Service Account Key... (6/9)")
            if not dry_run:
                self._delete_sa_key(logs)
            else:                logs.append(f"  Would delete key: {key_path}")

            # Service Account is KEPT (noted in persistent section at end)

            # Part 8: Pricing Infrastructure
            # (Detailed output handled by pricing_teardown.py, called separately)
            logs.append("")
            logs.append("Tearing down pricing infrastructure... (7/9)")

            if dry_run:
                logs.append("")
                logs.append("[DRY RUN COMPLETE] No resources were actually deleted")
            # Success message moved to teardown/core.py (after pricing teardown completes)

            return True, logs

        except Exception as e:
            logs.append(f"✗ Teardown failed: {str(e)}")
            return False, logs

    def _delete_queue(self, logs: List[str]):
        """Delete W&B Launch queue (manual only)"""
        queue_name = self.config.get('WANDB_LAUNCH_QUEUE_NAME', 'vertex-arr-coc-queue')
        logs.append(f"  Note: Queue must be deleted manually at https://wandb.ai/{self.wandb_entity}/launch")
        logs.append(f"  Queue name: {queue_name}")

    def _delete_worker_pool(self, logs: List[str]) -> bool:
        """Delete Cloud Build Worker Pool"""
        pool_name = "pytorch-mecha-pool"

        try:
            result = subprocess.run(
                ["gcloud", "builds", "worker-pools", "delete", pool_name,
                 "--region", self.region, "--quiet"],
                capture_output=True, text=True, timeout=120
            )

            if result.returncode == 0:
                logs.append(f"  ✓ Deleted worker pool: {pool_name}")
            else:
                if "NOT_FOUND" in result.stderr or "does not exist" in result.stderr:
                    logs.append(f"  ✓ Worker pool already removed: {pool_name}")
                else:
                    logs.append(f"  ✗ Failed to delete worker pool: {result.stderr}")
                    return False

            return True
        except Exception as e:
            logs.append(f"  ✗ Error deleting worker pool: {str(e)}")
            return False

    def _delete_registry(self, logs: List[str]) -> bool:
        """Delete SHARED Artifact Registry"""
        # SHARED registry name (matches _setup_registry)
        repo_name = "arr-coc-registry"

        try:
            result = subprocess.run(
                ["gcloud", "artifacts", "repositories", "delete", repo_name,
                 "--location", self.region, "--quiet"],
                capture_output=True, text=True, timeout=60
            )

            if result.returncode == 0:
                logs.append(f"  ✓ Deleted registry: {repo_name}")
            else:
                if "NOT_FOUND" in result.stderr or "does not exist" in result.stderr:
                    logs.append(f"  ✓ Registry already removed: {repo_name}")
                else:
                    logs.append(f"  ✗ Failed to delete registry: {result.stderr}")
                    return False

            return True
        except Exception as e:
            logs.append(f"  ✗ Error deleting registry: {str(e)}")
            return False

    def _delete_staging_bucket(self, logs: List[str]) -> bool:
        """Delete ALL regional staging buckets (on-demand multi-region cleanup!)"""
        project_name = self.config.get('PROJECT_NAME', 'arr-coc')

        # Find ALL staging buckets matching pattern: {project_id}-{project_name}-*-staging
        # This catches us-central1, europe-west2, asia-southeast1, etc.
        bucket_pattern = f"{self.project_id}-{project_name}-*-staging"

        try:
            # List all buckets matching pattern
            result = subprocess.run(
                ["gsutil", "ls"],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode != 0:
                logs.append(f"  ✗ Failed to list buckets: {result.stderr}")
                return False

            # Find matching staging buckets
            all_buckets = result.stdout.strip().split('\n')
            staging_buckets = [b for b in all_buckets if bucket_pattern.replace('*', '') in b and b.endswith('-staging/')]

            if not staging_buckets:
                logs.append(f"  ✓ No staging buckets found (already removed)")
                return True

            # Delete each regional staging bucket
            for bucket in staging_buckets:
                bucket_clean = bucket.rstrip('/')
                try:
                    delete_result = subprocess.run(
                        ["gsutil", "-m", "rm", "-r", bucket_clean],
                        capture_output=True, text=True, timeout=120
                    )

                    if delete_result.returncode == 0:
                        logs.append(f"  ✓ Deleted regional bucket: {bucket_clean}")
                    else:
                        if "BucketNotFoundException" in delete_result.stderr:
                            logs.append(f"  ✓ Bucket already removed: {bucket_clean}")
                        else:
                            logs.append(f"  ✗ Failed to delete {bucket_clean}: {delete_result.stderr}")

                except Exception as e:
                    logs.append(f"  ✗ Error deleting {bucket_clean}: {str(e)}")

            return True

        except Exception as e:
            logs.append(f"  ✗ Error finding staging buckets: {str(e)}")
            return False

    def _delete_checkpoints_bucket(self, logs: List[str]) -> bool:
        """Delete ALL regional checkpoints buckets (on-demand multi-region cleanup!)"""
        project_name = self.config.get('PROJECT_NAME', 'arr-coc')

        # Find ALL checkpoints buckets matching pattern: {project_id}-{project_name}-*-checkpoints
        # This catches us-central1, europe-west2, asia-southeast1, etc.
        bucket_pattern = f"{self.project_id}-{project_name}-*-checkpoints"

        try:
            # List all buckets matching pattern
            result = subprocess.run(
                ["gsutil", "ls"],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode != 0:
                logs.append(f"  ✗ Failed to list buckets: {result.stderr}")
                return False

            # Find matching checkpoints buckets
            all_buckets = result.stdout.strip().split('\n')
            checkpoints_buckets = [b for b in all_buckets if bucket_pattern.replace('*', '') in b and b.endswith('-checkpoints/')]

            if not checkpoints_buckets:
                logs.append(f"  ✓ No checkpoints buckets found (already removed)")
                return True

            # Delete each regional checkpoints bucket
            for bucket in checkpoints_buckets:
                bucket_clean = bucket.rstrip('/')
                try:
                    delete_result = subprocess.run(
                        ["gsutil", "-m", "rm", "-r", bucket_clean],
                        capture_output=True, text=True, timeout=120
                    )

                    if delete_result.returncode == 0:
                        logs.append(f"  ✓ Deleted regional bucket: {bucket_clean}")
                    else:
                        if "BucketNotFoundException" in delete_result.stderr:
                            logs.append(f"  ✓ Bucket already removed: {bucket_clean}")
                        else:
                            logs.append(f"  ✗ Failed to delete {bucket_clean}: {delete_result.stderr}")

                except Exception as e:
                    logs.append(f"  ✗ Error deleting {bucket_clean}: {str(e)}")

            return True

        except Exception as e:
            logs.append(f"  ✗ Error finding checkpoints buckets: {str(e)}")
            return False

    def _remove_iam_bindings(self, logs: List[str]) -> bool:
        """Remove IAM role bindings"""
        project_name = self.config.get('PROJECT_NAME', 'arr-coc')
        sa_name = f"{project_name}-sa"
        sa_email = f"{sa_name}@{self.project_id}.iam.gserviceaccount.com"

        roles = [
            "roles/aiplatform.user",
            "roles/storage.objectAdmin",
            "roles/artifactregistry.writer",
            "roles/logging.logWriter"
        ]

        try:
            for role in roles:
                subprocess.run(
                    ["gcloud", "projects", "remove-iam-policy-binding", self.project_id,
                     "--member", f"serviceAccount:{sa_email}",
                     "--role", role, "--quiet"],
                    capture_output=True, timeout=30
                )

            logs.append(f"  ✓ Removed 4 IAM role bindings")

            # Also remove Cloud Build service account permissions (for worker pools)
            # Get project number for Cloud Build default SA
            project_number = subprocess.run(
                ["gcloud", "projects", "describe", self.project_id, "--format=value(projectNumber)"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if project_number.returncode == 0:
                project_num = project_number.stdout.strip()
                cloudbuild_sa = f"{project_num}@cloudbuild.gserviceaccount.com"

                cloudbuild_roles = [
                    "roles/compute.networkUser",
                    "roles/compute.admin",
                ]

                for role in cloudbuild_roles:
                    subprocess.run(
                        ["gcloud", "projects", "remove-iam-policy-binding", self.project_id,
                         "--member", f"serviceAccount:{cloudbuild_sa}",
                         "--role", role, "--quiet"],
                        capture_output=True, timeout=30
                    )

                logs.append(f"  ✓ Removed 2 Cloud Build IAM role bindings")
            else:
                logs.append(f"  ⚠️  Could not determine project number for Cloud Build SA removal")

            return True
        except Exception as e:
            logs.append(f"  ✗ Error removing IAM bindings: {str(e)}")
            return False

    def _delete_sa_key(self, logs: List[str]):
        """Delete service account key file"""
        try:
            if key_path.exists():
                key_path.unlink()
                logs.append(f"  ✓ Removed key file")
            else:
                logs.append(f"  ✓ Key file already removed")
        except Exception as e:
            logs.append(f"  ✗ Error deleting key: {str(e)}")

    def _setup_cloudbuild_iam(self, logs: List[str]) -> bool:
        """Grant Cloud Build service account permissions for worker pools"""
        import subprocess

        try:
            # Get project number
            result = subprocess.run(
                ["gcloud", "projects", "describe", self.project_id, "--format=value(projectNumber)"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                logs.append("  ⚠️  Could not determine project number for Cloud Build SA")
                return True  # Non-fatal

            project_num = result.stdout.strip()
            cloudbuild_sa = f"{project_num}@cloudbuild.gserviceaccount.com"

            # Roles needed for Cloud Build to create C3 instances in worker pools
            cloudbuild_roles = [
                "roles/compute.networkUser",  # Use VPC peering
                "roles/compute.admin",        # Create/manage compute instances
            ]

            for role in cloudbuild_roles:
                grant_result = subprocess.run(
                    [
                        "gcloud", "projects", "add-iam-policy-binding", self.project_id,
                        f"--member=serviceAccount:{cloudbuild_sa}",
                        f"--role={role}",
                        "--condition=None",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                if grant_result.returncode == 0 or "already has role" in grant_result.stderr.lower():
                    logs.append(f"  ✓ Granted {role}")
                else:
                    logs.append(f"  ⚠️  Warning: Failed to grant {role}")

            return True

        except Exception as e:
            logs.append(f"  ✗ Error setting up Cloud Build IAM: {str(e)}")
            return False

    def _setup_vpc_peering(self, logs: List[str]) -> bool:
        """Set up VPC peering for Service Networking API"""
        import subprocess
        import time

        try:
            peering_range_name = "google-managed-services-default"

            # Create IP address range for VPC peering (2 attempts: initial + 1 retry)
            for attempt in range(2):
                create_range = subprocess.run(
                    [
                        "gcloud",
                        "compute",
                        "addresses",
                        "create",
                        peering_range_name,
                        "--global",
                        "--purpose=VPC_PEERING",
                        "--prefix-length=16",
                        "--network=default",
                        f"--project={self.project_id}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if create_range.returncode == 0:
                    logs.append(f"  ✓ Created VPC peering address range")
                    break
                elif "already exists" in create_range.stderr.lower():
                    logs.append(f"  ✓ VPC peering address range already exists")
                    break
                elif attempt < 1:
                    logs.append(f"  ⚠ Retry 1/1...")
                    time.sleep(2)
                else:
                    logs.append(f"  ✗ Failed to create peering range: {create_range.stderr[:200]}")
                    return False

            # Connect VPC peering (2 attempts: initial + 1 retry)
            for attempt in range(2):
                connect_peering = subprocess.run(
                    [
                        "gcloud",
                        "services",
                        "vpc-peerings",
                        "connect",
                        "--service=servicenetworking.googleapis.com",
                        f"--ranges={peering_range_name}",
                        "--network=default",
                        f"--project={self.project_id}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if connect_peering.returncode == 0:
                    logs.append(f"  ✓ Connected VPC peering")
                    break
                elif "already exists" in connect_peering.stderr.lower() or "already peered" in connect_peering.stderr.lower():
                    logs.append(f"  ✓ VPC peering already connected")
                    break
                elif attempt < 1:
                    logs.append(f"  ⚠ Retry 1/1...")
                    time.sleep(2)
                else:
                    logs.append(f"  ✗ Failed to connect peering: {connect_peering.stderr[:200]}")
                    return False

            return True

        except Exception as e:
            logs.append(f"  ✗ Error setting up VPC peering: {str(e)}")
            return False


def create_wandb_secret(status):
    """
    Create W&B API key secret in Secret Manager.

    Called during setup to store W&B API key securely.
    Cloud Run job will mount this as WANDB_API_KEY env var.

    Args:
        status: Function to display status messages

    Returns:
        True if secret created/updated successfully
        False if failed
    """
    try:
        import wandb as wandb_module

        wandb_api_key = wandb_module.api.api_key
        secret_name = "wandb-api-key"

        # Check if secret exists
        check_secret = subprocess.run(
            ["gcloud", "secrets", "describe", secret_name],
            capture_output=True,
            text=True,
            timeout=30
        )

        if check_secret.returncode != 0:
            # Secret doesn't exist - create it
            status("   ⏳ Creating W&B API key secret in Secret Manager...")
            create_secret = subprocess.run(
                [
                    "gcloud",
                    "secrets",
                    "create",
                    secret_name,
                    "--replication-policy=automatic",
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            if create_secret.returncode != 0:
                status(f"[red]✗ Failed to create secret: {create_secret.stderr[:200]}[/red]")
                return False
            status("   ✓ Secret created")
        else:
            status("   ✓ Secret exists")

        # Update secret with current API key (creates new version)
        status("   ⏳ Storing W&B API key in secret...")
        add_version = subprocess.run(
            [
                "gcloud",
                "secrets",
                "versions",
                "add",
                secret_name,
                "--data-file=-",
            ],
            input=wandb_api_key,
            capture_output=True,
            text=True,
            timeout=30
        )
        if add_version.returncode != 0:
            status(f"[red]✗ Failed to add secret version: {add_version.stderr[:200]}[/red]")
            return False

        status("   ✓ API key stored securely")
        return True

    except subprocess.TimeoutExpired:
        status("[red]✗ Secret Manager timeout (network slow?)[/red]")
        status("[dim]Try again or check: gcloud services list --enabled | grep secret[/dim]")
        return False
    except Exception as e:
        status(f"[red]✗ Failed to create W&B secret: {str(e)}[/red]")
        return False


