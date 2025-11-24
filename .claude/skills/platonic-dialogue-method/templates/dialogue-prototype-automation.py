#!/usr/bin/env python3
"""
Complete Dialogue Prototype Automation Script

Creates GitHub repo + HuggingFace Space with full privacy automation.
No manual steps required!

Usage:
    python AUTOMATION_SCRIPT.py project-name "Description"
"""

import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi


def run_command(cmd, description):
    """Run shell command with error handling."""
    print(f"\n{'=' * 60}")
    print(f"⚡ {description}")
    print(f"{'=' * 60}")
    print(f"$ {cmd}\n")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        sys.exit(1)

    print(f"✅ Success: {description}")
    return result


def create_dialogue_prototype(project_name, description, dialogue_number):
    """
    Complete automation for dialogue prototype creation.

    Args:
        project_name: e.g., "arr-coc-0-1"
        description: GitHub repo description
        dialogue_number: e.g., "46" for Part 46
    """

    base_path = Path(
        f"RESEARCH/PlatonicDialogues/{dialogue_number}-mvp-be-doing/code/{project_name}"
    )

    print(f"""
╔══════════════════════════════════════════════════════════════
║ DIALOGUE PROTOTYPE AUTOMATION
╠══════════════════════════════════════════════════════════════
║ Project: {project_name}
║ Description: {description}
║ Path: {base_path}
╚══════════════════════════════════════════════════════════════
    """)

    # 1. Create folder structure
    run_command(f"mkdir -p {base_path}", "Create folder structure")

    # 2. Initialize git
    run_command(f"cd {base_path} && git init", "Initialize git repository")

    run_command(
        f"cd {base_path} && git config user.name '{project_name.upper()} Project'",
        "Set git user name",
    )

    run_command(
        f"cd {base_path} && git config user.email '{project_name}@alfrednorth.com'",
        "Set git user email",
    )

    # 3. Create initial files (README, requirements, etc.)
    # Note: This assumes files were already created
    # In practice, you'd generate them here

    # 4. Create GitHub repo (private)
    run_command(
        f"cd {base_path} && gh repo create {project_name} "
        f"--private "
        f'--description "{description}" '
        f"--source=. "
        f"--remote=origin",
        "Create private GitHub repository",
    )

    # 5. Commit and push
    run_command(f"cd {base_path} && git add .", "Stage files")

    commit_msg = f"""Initial commit: {project_name.upper()} structure

{description}

Born from Platonic Dialogue Part {dialogue_number}"""

    run_command(
        f"cd {base_path} && git commit -m '{commit_msg}'", "Create initial commit"
    )

    run_command(f"cd {base_path} && git push -u origin main", "Push to GitHub")

    # 6. Create HuggingFace Space
    run_command(
        f"huggingface-cli repo create {project_name} "
        f"--type space "
        f"--space_sdk gradio "
        f"-y",
        "Create HuggingFace Gradio Space",
    )

    # 7. Sync with HuggingFace Space
    print(f"\n{'=' * 60}")
    print("⚡ Syncing with HuggingFace Space")
    print(f"{'=' * 60}\n")

    try:
        api = HfApi()

        # Get username
        whoami = api.whoami()
        username = whoami["name"]

        # Add HF remote
        run_command(
            f"cd {base_path} && git remote add hf https://huggingface.co/spaces/{username}/{project_name}",
            "Add HuggingFace remote",
        )

        # Pull from HF (it auto-creates files like .gitignore)
        run_command(
            f"cd {base_path} && git pull hf main --rebase --allow-unrelated-histories || true",
            "Pull from HuggingFace (merging auto-generated files)",
        )

        # Push to HuggingFace
        run_command(f"cd {base_path} && git push hf main", "Push to HuggingFace Space")

        # 8. Make HuggingFace Space PRIVATE via API
        print(f"\n{'=' * 60}")
        print("⚡ Setting HuggingFace Space to PRIVATE via API")
        print(f"{'=' * 60}\n")

        # Update to private
        api.update_repo_settings(
            repo_id=f"{username}/{project_name}", private=True, repo_type="space"
        )

        # Verify
        info = api.repo_info(repo_id=f"{username}/{project_name}", repo_type="space")

        if info.private:
            print(f"✅ HuggingFace Space is now PRIVATE")
        else:
            print(f"⚠️  Warning: Space privacy toggle may have failed")

    except Exception as e:
        print(f"❌ Failed to sync with HuggingFace: {e}")
        print("⚠️  You may need to manually sync/set privacy at:")
        print(f"   https://huggingface.co/spaces/{username}/{project_name}/settings")

        # 9. Check Space status (wait for build)
        print(f"\n{'=' * 60}")
        print("⚡ Checking Space build status")
        print(f"{'=' * 60}\n")

        import time

        space_id = f"{username}/{project_name}"

        # Wait for Space to finish building
        max_wait = 600  # 10 minutes
        start_time = time.time()

        while (time.time() - start_time) < max_wait:
            runtime = api.get_space_runtime(space_id)

            if runtime.stage == "RUNNING":
                print(f"\n✅ Space is RUNNING!")
                domains = runtime.raw.get("domains", [])
                if domains:
                    print(f"   URL: https://{domains[0]['domain']}")
                break

            elif runtime.stage == "RUNTIME_ERROR":
                print(f"\n❌ Space build FAILED with runtime error:")
                print("-" * 60)
                error_msg = runtime.raw.get("errorMessage", "No error message")
                print(error_msg)
                print("-" * 60)
                break

            elif runtime.stage == "BUILDING":
                elapsed = int(time.time() - start_time)
                print(f"🔨 Still building... ({elapsed}s elapsed)")
                time.sleep(10)  # Check every 10 seconds

            else:
                print(f"⚠️  Unknown stage: {runtime.stage}")
                break

        if (time.time() - start_time) >= max_wait:
            print(f"\n⏱️  Timeout: Space still building after {max_wait}s")
            print(
                f"   Check status manually at: https://huggingface.co/spaces/{space_id}"
            )

    except Exception as e:
        print(f"❌ Failed to check Space status: {e}")

    # Summary
    print(f"""

╔══════════════════════════════════════════════════════════════
║ ✅ DIALOGUE PROTOTYPE COMPLETE
╠══════════════════════════════════════════════════════════════
║
║ GitHub (private):
║   https://github.com/{username}/{project_name}
║
║ HuggingFace Space (private):
║   https://huggingface.co/spaces/{username}/{project_name}
║
║ Local path:
║   {base_path}
║
║ Next: Check Space status with:
║   python spacecheck.py {username}/{project_name}
║
╚══════════════════════════════════════════════════════════════
    """)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python AUTOMATION_SCRIPT.py project-name 'Description'")
        sys.exit(1)

    project_name = sys.argv[1]
    description = sys.argv[2]
    dialogue_num = "46"  # Default, can be parameterized

    create_dialogue_prototype(project_name, description, dialogue_num)
