"""
Infrastructure Core Logic - Docker Image Security Scanning.

This module handles Docker image security scanning for all 4 arr- images.
TUI and CLI both use these functions to display security status.

Moved from ../monitor/core.py to infra/core.py (security belongs in infrastructure!)
"""

from typing import Dict, Optional
import subprocess

from ..shared.types import StatusCallback
from ..shared.api_helpers import run_gcloud_with_retry


def check_image_security_cached(config, status, force_refresh=False):
    """Wrapper for check_image_security() - NO CACHING (removed for always-fresh data)

    Args:
        config: Training configuration
        status: Status callback
        force_refresh: Ignored (kept for compatibility, but always does fresh check now)
    """
    # ALWAYS do fresh check - caching was causing confusion with stale data
    # Security check takes ~5-10 seconds for 4 images, which is acceptable
    return check_image_security(config, status)
def check_image_security(config: dict, status: StatusCallback) -> Optional[Dict]:
    """
    Check ALL FOUR image security (SLSA level, vulnerabilities)

    ALWAYS returns dict with scan status for latest images:
    - arr-pytorch-base: PyTorch compiled from source (MECHA build)
    - arr-ml-stack: ML foundation stack
    - arr-trainer: ARR-COC training code
    - arr-vertex-launcher: W&B Launch agent

    Shows scan pending if GCP hasn't scanned the latest image yet (10-30 min delay)

    Args:
        config: Training configuration dict
        status: Status callback (not used - silent check)

    Returns:
        Dict with security info for ALL FOUR images, ALWAYS (never None)
        Includes: scan_available, image_digest, vulnerabilities, SLSA
    """
    try:
        import os
        import json

        # Get project ID from config FIRST (most reliable)
        project_id = config.get('GCP_PROJECT_ID')

        # Fall back to environment variables
        if not project_id:
            project_id = os.environ.get('GOOGLE_CLOUD_PROJECT') or os.environ.get('GCP_PROJECT') or os.environ.get('GCLOUD_PROJECT')

        # Last resort: try gcloud (but this often times out in TUI)
        if not project_id:
            try:
                project_result = run_gcloud_with_retry(
                    ["gcloud", "config", "get-value", "project"],
                    max_retries=3,
                    timeout=10,
                    operation_name="get GCP project ID",
                )
                project_id = project_result.stdout.strip()
            except subprocess.TimeoutExpired:
                pass  # Silent fail - will use config fallback below

        if not project_id:
            # Can't get project ID - return minimal status
            return {
                'error': 'Cannot determine GCP project ID',
                'pytorch': {'scan_available': False},
                'base': {'scan_available': False},
                'training': {'scan_available': False},
                'launcher': {'scan_available': False}
            }

        region = config.get('GCP_ROOT_RESOURCE_REGION', 'us-central1')
        registry_name = config.get('ARTIFACT_REGISTRY_NAME', 'arr-coc-registry')
        persistent_registry = 'arr-coc-registry-persistent'  # PyTorch base image only

        # Check ALL FOUR images: pytorch-base → ml-stack → trainer → vertex-launcher
        results = {}

        # Map display names to actual image names and registries
        image_mapping = {
            'pytorch': ('arr-pytorch-base', persistent_registry),  # PyTorch in persistent registry
            'base': ('arr-ml-stack', registry_name),               # ML foundation stack
            'training': ('arr-trainer', registry_name),            # ARR-COC training code
            'launcher': ('arr-vertex-launcher', registry_name)     # W&B Launch agent
        }

        for image_type in ['pytorch', 'base', 'training', 'launcher']:
            actual_image_name, target_registry = image_mapping[image_type]
            image_url = f"{region}-docker.pkg.dev/{project_id}/{target_registry}/{actual_image_name}:latest"

            # Get vulnerabilities + SLSA level
            result = run_gcloud_with_retry(
                [
                    "gcloud", "artifacts", "docker", "images", "describe",
                    image_url,
                    "--show-package-vulnerability",
                    "--format=json"
                ],
                max_retries=3,
                timeout=60,  # Increased from 30s - first image (arr-pytorch-base) needs more time for gcloud initialization
                operation_name="describe Docker image with vulnerabilities",
            )

            if result.returncode != 0:
                # :latest tag missing or no image found
                results[image_type] = {'scan_available': False, 'image_url': image_url, 'error': 'Image not found (no :latest tag)'}
                continue

            data = json.loads(result.stdout) if result.stdout else {}

            # Extract image digest (the actual SHA256 being scanned)
            image_digest = data.get('image_summary', {}).get('digest', 'unknown')
            # Strip "sha256:" prefix if present (we'll add our own "hash:" label in display)
            if image_digest.startswith('sha256:'):
                image_digest = image_digest[7:]  # Remove "sha256:" (7 chars)

            slsa_level = data.get('image_summary', {}).get('slsa_build_level', 0)

            # Scan staleness detection: REMOVED (caused false positives)
            #
            # We previously compared :latest tag digest vs newest-by-upload-time digest
            # to detect stale scans after rebuilds. This broke because:
            # 1. gcloud describe :latest returns digest A (from tag)
            # 2. gcloud list --sort-by=~UPDATE_TIME returns digest B (newest upload)
            # 3. These CAN differ even for same logical image (tag race conditions)
            #
            # After full teardown/rebuild with arr- prefix rename, digests mismatched
            # constantly, showing "awaiting scan" when scans existed.
            #
            # LESSON LEARNED: If GCP has scan results, trust them. The :latest tag
            # is authoritative - if it has a scan, that's the current image state.
            # Don't try to be clever with time-based comparisons.
            scan_is_outdated = False  # Trust GCP's scan results

            # Check if vulnerability scan data is available
            vuln_summary = data.get('package_vulnerability_summary', {})

            # If no vulnerability summary, scan is pending
            if not vuln_summary or 'vulnerabilities' not in vuln_summary:
                results[image_type] = {
                    'scan_available': False,
                    'scan_pending': True,
                    'scan_outdated': scan_is_outdated,
                    'image_url': image_url,
                    'image_digest': image_digest[:12],  # Short digest (first 12 chars)
                    'slsa_level': slsa_level,
                    'message': '⏳ Security scan pending (GCP takes 10-30 min after push)'
                }
                continue

            # Scan data available! Process vulnerabilities
            vulns = vuln_summary.get('vulnerabilities', {})

            critical = len(vulns.get('CRITICAL', []))
            high = len(vulns.get('HIGH', []))
            medium = len(vulns.get('MEDIUM', []))
            low = len(vulns.get('LOW', []))

            # Extract detailed vulnerability information for ALL severities
            detailed_vulns = {
                'critical': [],
                'high': [],
                'medium': [],
                'low': []
            }

            # Process CRITICAL vulnerabilities
            for vuln in vulns.get('CRITICAL', []):
                # GCP API structure: vuln['noteName'] = "projects/goog-vulnz/notes/CVE-XXXX"
                note_name = vuln.get('noteName', '')
                cve_id = note_name.split('/')[-1] if note_name else 'N/A'

                # Vulnerability details nested under 'vulnerability'
                vuln_data = vuln.get('vulnerability', {})
                cvss_score = vuln_data.get('cvssScore', 0.0)
                description = vuln_data.get('longDescription', '')[:200]

                # Package issue is an ARRAY (can have multiple affected packages)
                package_issues = vuln_data.get('packageIssue', [])
                if package_issues:
                    pkg = package_issues[0]  # Take first package
                    detailed_vulns['critical'].append({
                        'cve_id': cve_id,
                        'cvss_score': cvss_score,
                        'package': pkg.get('affectedPackage', 'unknown'),
                        'current_version': pkg.get('affectedVersion', {}).get('fullName', 'unknown'),
                        'fixed_version': pkg.get('fixedVersion', {}).get('fullName', 'Not available'),
                        'description': description,
                    })

            # Process HIGH vulnerabilities
            for vuln in vulns.get('HIGH', []):
                note_name = vuln.get('noteName', '')
                cve_id = note_name.split('/')[-1] if note_name else 'N/A'

                vuln_data = vuln.get('vulnerability', {})
                cvss_score = vuln_data.get('cvssScore', 0.0)
                description = vuln_data.get('longDescription', '')[:200]

                package_issues = vuln_data.get('packageIssue', [])
                if package_issues:
                    pkg = package_issues[0]
                    detailed_vulns['high'].append({
                        'cve_id': cve_id,
                        'cvss_score': cvss_score,
                        'package': pkg.get('affectedPackage', 'unknown'),
                        'current_version': pkg.get('affectedVersion', {}).get('fullName', 'unknown'),
                        'fixed_version': pkg.get('fixedVersion', {}).get('fullName', 'Not available'),
                        'description': description,
                    })

            # Process MEDIUM vulnerabilities
            for vuln in vulns.get('MEDIUM', []):
                note_name = vuln.get('noteName', '')
                cve_id = note_name.split('/')[-1] if note_name else 'N/A'

                vuln_data = vuln.get('vulnerability', {})
                cvss_score = vuln_data.get('cvssScore', 0.0)
                description = vuln_data.get('longDescription', '')[:200]

                package_issues = vuln_data.get('packageIssue', [])
                if package_issues:
                    pkg = package_issues[0]
                    detailed_vulns['medium'].append({
                        'cve_id': cve_id,
                        'cvss_score': cvss_score,
                        'package': pkg.get('affectedPackage', 'unknown'),
                        'current_version': pkg.get('affectedVersion', {}).get('fullName', 'unknown'),
                        'fixed_version': pkg.get('fixedVersion', {}).get('fullName', 'Not available'),
                        'description': description,
                    })

            # Process LOW vulnerabilities
            for vuln in vulns.get('LOW', []):
                note_name = vuln.get('noteName', '')
                cve_id = note_name.split('/')[-1] if note_name else 'N/A'

                vuln_data = vuln.get('vulnerability', {})
                cvss_score = vuln_data.get('cvssScore', 0.0)
                description = vuln_data.get('longDescription', '')[:200]

                package_issues = vuln_data.get('packageIssue', [])
                if package_issues:
                    pkg = package_issues[0]
                    detailed_vulns['low'].append({
                        'cve_id': cve_id,
                        'cvss_score': cvss_score,
                        'package': pkg.get('affectedPackage', 'unknown'),
                        'current_version': pkg.get('affectedVersion', {}).get('fullName', 'unknown'),
                        'fixed_version': pkg.get('fixedVersion', {}).get('fullName', 'Not available'),
                        'description': description,
                    })

            results[image_type] = {
                'scan_available': True,
                'scan_pending': False,
                'scan_outdated': scan_is_outdated,
                'image_url': image_url,
                'image_digest': image_digest[:12],  # Short digest
                'slsa_level': slsa_level,
                'critical': critical,
                'high': high,
                'medium': medium,
                'low': low,
                'total_vulns': critical + high + medium + low,
                'detailed_vulns': detailed_vulns,  # NEW: Detailed CVE info for critical/high
            }

        # Return combined results for all 4 images
        # ALWAYS return data (never None) so monitor always shows security section
        return {
            'pytorch': results.get('pytorch', {'scan_available': False, 'error': 'Not checked'}),
            'base': results.get('base', {'scan_available': False, 'error': 'Not checked'}),
            'training': results.get('training', {'scan_available': False, 'error': 'Not checked'}),
            'launcher': results.get('launcher', {'scan_available': False, 'error': 'Not checked'}),
            'console_url': f"https://console.cloud.google.com/artifacts/docker/{project_id}/{region}/{registry_name}?project={project_id}",
        }

    except Exception as e:
        # Return error state but still show security section
        return {
            'error': f'Security check failed: {str(e)[:100]}',
            'base': {'scan_available': False},
            'training': {'scan_available': False},
            'launcher': {'scan_available': False},
        }
def format_security_summary_core(security_data: Dict) -> str:
    """
    Format security scan data into display-ready text (UI-agnostic)

    Returns plain text with Rich markup that both CLI and TUI can use.
    CLI strips markup, TUI renders it.

    Args:
        security_data: Dict with keys 'base', 'training', 'launcher'
                      Each contains: scan_pending, scan_outdated, scan_available,
                                    critical, high, medium, low, total_vulns,
                                    image_digest, slsa_level

    Returns:
        Formatted text string ready for display
    """
    lines = []

    # Image display name mapping (key → actual arr- image name)
    image_display_map = {
        'pytorch': 'arr-pytorch-base',
        'base': 'arr-ml-stack',
        'training': 'arr-trainer',
        'launcher': 'arr-vertex-launcher'
    }

    # Check if ALL FOUR images are scanned, current, and clean
    all_clean = True
    for image_key in ['pytorch', 'base', 'training', 'launcher']:
        img = security_data.get(image_key, {})
        if img.get('scan_pending') or img.get('scan_outdated') or not img.get('scan_available') or img.get('total_vulns', 0) > 0:
            all_clean = False
            break

    # If all clean, show brief success message and return early
    if all_clean:
        lines.append("  [green]✓ all 4 images secure roger ok[/green]")
        return "\n".join(lines)

    # Display each image compactly (issues found or scans pending)
    for idx, image_key in enumerate(['pytorch', 'base', 'training', 'launcher']):
        # Add separator (not before first image)
        if idx > 0:
            lines.append("             ─────────────────────────────────────────────")

        # Get actual image name for display
        image_display_name = image_display_map.get(image_key, image_key)

        # Color mapping for image names
        color_map = {"pytorch": "magenta", "base": "cyan", "training": "yellow", "launcher": "green"}
        img_color = color_map.get(image_key, "white")
        img = security_data.get(image_key, {})
        
        if img.get('scan_pending'):
            # GCP is actively scanning THIS exact image right now
            lines.append(f"  ([bold {img_color}]{image_display_name:20s}[/bold {img_color}]) ⏳ GCP is currently scanning this image (10-30 min)")
            lines.append(f"             Digest: {img.get('image_digest', 'unknown')}")

        elif img.get('scan_outdated'):
            # Scan is outdated (shouldn't happen now, but keep for safety)
            lines.append(f"  ([bold {img_color}]{image_display_name:20s}[/bold {img_color}]) ⏳ Fresh image built - awaiting first security scan (10-30 min)")
            lines.append(f"             Image: {img.get('image_digest', 'unknown')}")
            
        elif img.get('scan_available'):
            # Scan complete AND current - show results
            total = img.get('total_vulns', 0)
            crit = img.get('critical', 0)
            high = img.get('high', 0)
            med = img.get('medium', 0)
            low = img.get('low', 0)

            # First line: standardized format (like LAUNCHER)
            if total == 0:
                status = "✅ CLEAN (no vulnerabilities)"
            else:
                # Build severity description
                severity_parts = []
                if crit > 0:
                    severity_parts.append("CRITICAL")
                if high > 0:
                    severity_parts.append("HIGH")
                if med > 0:
                    severity_parts.append("Medium")
                if low > 0:
                    severity_parts.append("Low")

                severity_desc = "/".join(severity_parts) if severity_parts else "Unknown"

                # Add CRITICAL warning if needed
                if crit > 0:
                    status = f"⚠️  {total} total ({severity_desc}) - CRITICAL"
                else:
                    status = f"✓ {total} total ({severity_desc})"

            lines.append(f"  ([bold {img_color}]{image_display_name:20s}[/bold {img_color}]) {status}")

            if total > 0:
                # Second line: details (hide zero counts!)
                hash_display = img.get('image_digest', 'unknown')

                # Build vulnerability breakdown (only show non-zero counts)
                vuln_parts = []
                if crit > 0:
                    vuln_parts.append(f"🔴 {crit}")
                if high > 0:
                    vuln_parts.append(f"🟠 {high}")
                if med > 0:
                    vuln_parts.append(f"🟡 {med}")
                if low > 0:
                    vuln_parts.append(f"🔵 {low}")

                vuln_display = " ".join(vuln_parts)
                lines.append(f"             {vuln_display} | SLSA: {img.get('slsa_level', 0)}/3 | hash:{hash_display}")
        else:
            # No scan available - check if image exists or not
            if img.get('error'):
                # Image not found in registry
                lines.append(f"  ([bold {img_color}]{image_display_name:20s}[/bold {img_color}]) [dim]❌ Image not found in registry[/dim]")
                lines.append(f"             [dim]Run setup/launch to build images[/dim]")
            else:
                # Brand new image - never scanned before
                lines.append(f"  ([bold {img_color}]{image_display_name:20s}[/bold {img_color}]) ⏳ Fresh image built - awaiting first security scan (10-30 min)")
                lines.append(f"             Image: {img.get('image_digest', 'unknown')}")
    
    return "\n".join(lines)


def should_show_security_core(security_data: Dict) -> bool:
    """
    Determine if security section should be displayed

    ALWAYS SHOW security section to confirm scans completed!

    Shows:
    - Fresh images awaiting scans
    - Scans pending
    - Scans outdated
    - ANY vulnerabilities (even 1 low severity)
    - Success message when all clean

    Returns:
        True (always show security section)
    """
    # ALWAYS show security section (even when all clean - users want confirmation!)
    return True
