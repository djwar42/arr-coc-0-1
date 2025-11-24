#!/bin/bash
# W&B Launch Agent Wrapper - Semi-Persistent Runner
#
# This wrapper monitors the W&B Launch agent and:
# 1. Exits immediately on fatal errors (quota, permissions, etc.)
# 2. Exits after idle timeout (no job submissions)
# 3. Tracks jobs run and lifetime for monitoring
#
# Version: 2.5 (Idle timeout increased to 30 minutes - 2025-11-18)
#
# Features:
# - Idle timeout (auto-shutdown when no activity)
# - Jobs run counter (tracks submissions)
# - Fatal error detection (fast bailout on quota/permission errors)
# - Lifetime tracking (for monitoring tables.)

set -euo pipefail

# Configuration
IDLE_TIMEOUT=1800  # Idle timeout in seconds (30 minutes)
IDLE_TIMEOUT_MINUTES=$((IDLE_TIMEOUT / 60))  # Calculate minutes for display

# Create log file for the agent
LOG_FILE="/tmp/wandb-agent.log"
touch "$LOG_FILE"

# Semi-persistent runner state
LAST_ACTIVITY=$(date +%s)
JOBS_RUN=0
RUNNER_START=$(date +%s)
LAST_SUBMISSION=""  # Track last submission to avoid double-counting

echo "🚀 Starting Semi-Persistent W&B Launch Agent..."
echo "📝 Logs: $LOG_FILE"
echo "⏱️  Idle timeout: ${IDLE_TIMEOUT_MINUTES} minutes"
echo "📊 Runs: 0"

# Start W&B agent (sitecustomize.py applies patch on Python startup!)
# CRITICAL SOLUTION (2025-11-16): sitecustomize.py is auto-loaded by Python
# Python imports sitecustomize.py BEFORE wandb modules load → patch persists!
# Source: https://docs.python.org/3/library/site.html#module-sitecustomize
echo ""
echo "🚀 Starting W&B Launch agent..."
echo "   (sitecustomize.py will apply spot patch on Python startup)"
echo ""

wandb launch-agent "$@" 2>&1 | tee "$LOG_FILE" &
AGENT_PID=$!

echo "✓ W&B agent started (PID: $AGENT_PID)"
echo "⏳ Monitoring for fatal errors and idle timeout..."

# Helper function to print final stats (reusable across all exit paths)
print_final_stats() {
    local current_time=$(date +%s)
    local lifetime=$((current_time - RUNNER_START))
    echo ""
    echo "📊 Final runner stats:"
    echo "   • Runs: $JOBS_RUN"
    echo "   • Lifetime: $((lifetime / 60))m $((lifetime % 60))s"
    echo ""
}

# Helper function to show error context with pattern highlighting
show_error_context() {
    local pattern="$1"
    local description="$2"  # Human-readable description of what we're looking for

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔍 BAILOUT TRIGGER: $description"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Show ALL matching lines with surrounding context
    echo "📍 All occurrences of pattern in last 100 lines:"
    echo ""
    tail -100 "$LOG_FILE" | grep -n -E "$pattern" | sed 's/^/  ► /'

    echo ""
    echo "━━━ Full Log Context (last 100 lines) ━━━"
    tail -100 "$LOG_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# Monitor loop - check logs every 5 seconds
while kill -0 "$AGENT_PID" 2>/dev/null; do
    sleep 5

    # Calculate idle time and runner lifetime
    CURRENT_TIME=$(date +%s)
    IDLE_TIME=$((CURRENT_TIME - LAST_ACTIVITY))
    LIFETIME=$((CURRENT_TIME - RUNNER_START))

    # Check for job submission (reset idle timer and increment counter)
    # Pattern: W&B Launch outputs "running N out of a maximum" when jobs are active
    # We detect job submission by checking if "running N" increased
    if tail -20 "$LOG_FILE" | grep -qE "Launch agent received job|running [1-9].*out of a maximum"; then
        # Get the latest submission line
        CURRENT_SUBMISSION=$(tail -20 "$LOG_FILE" | grep -E "Launch agent received job|running [1-9].*out of a maximum" | tail -1)

        # Check if this is a NEW submission (different from last one we saw)
        if [ -n "$CURRENT_SUBMISSION" ] && [ "$CURRENT_SUBMISSION" != "$LAST_SUBMISSION" ]; then
            JOBS_RUN=$((JOBS_RUN + 1))
            LAST_ACTIVITY=$CURRENT_TIME
            IDLE_TIME=0
            LAST_SUBMISSION="$CURRENT_SUBMISSION"  # Remember this submission

            echo ""
            echo "✅ Job submitted to Vertex AI! (Runs: $JOBS_RUN)"
            echo "⏱️  Idle timer reset to 0s (${IDLE_TIMEOUT_MINUTES}min until auto-shutdown)"
            echo "📊 Runner lifetime: $((LIFETIME / 60))m $((LIFETIME % 60))s"
            echo ""
        fi
    fi

    # Check idle timeout
    if [ $IDLE_TIME -gt $IDLE_TIMEOUT ]; then
        echo ""
        # Show completion message with green tick if jobs > 0, yellow/warning if 0 jobs
        if [ "$JOBS_RUN" -gt 0 ]; then
            echo "✓ Runner completed $JOBS_RUN runs"
        else
            echo "✗ Runner completed 0 runs"
        fi
        echo "📊 Final stats:"
        echo "   • Runs: $JOBS_RUN"
        echo "   • Runner lifetime: $((LIFETIME / 60))m $((LIFETIME % 60))s"
        echo "   • Idle time: $((IDLE_TIME / 60))m $((IDLE_TIME % 60))s"
        echo "✓ Graceful shutdown - killing agent (PID: $AGENT_PID)"
        kill "$AGENT_PID" 2>/dev/null || true
        echo "✓ Semi-persistent runner finished cleanly"
        exit 0
    fi

    # Periodic status update (every 5 minutes)
    if [ $((LIFETIME % 300)) -lt 5 ]; then
        echo "[$(date '+%H:%M:%S')] Runner alive: ${LIFETIME}s lifetime, ${IDLE_TIME}s idle, Runs: $JOBS_RUN"
    fi

    # Check for fatal errors in recent logs
    if tail -100 "$LOG_FILE" | grep -q "Machine type.*is not supported"; then
        echo "🚨 FATAL ERROR DETECTED: Machine type not supported!"
        show_error_context "Machine type.*is not supported" "Machine type incompatible with GPU (pattern: 'Machine type.*is not supported')"
        echo "❌ Killing agent (PID: $AGENT_PID) - this error will not self-resolve"
        kill "$AGENT_PID" 2>/dev/null || true
        print_final_stats
        exit 1
    fi

    if tail -100 "$LOG_FILE" | grep -q "InvalidArgument: 400"; then
        echo "🚨 FATAL ERROR DETECTED: Invalid argument (400)!"
        show_error_context "InvalidArgument: 400" "GCP API rejected request with HTTP 400 (pattern: 'InvalidArgument: 400')"
        echo "❌ Killing agent (PID: $AGENT_PID) - config error, will not self-resolve"
        kill "$AGENT_PID" 2>/dev/null || true
        print_final_stats
        exit 1
    fi

    if tail -100 "$LOG_FILE" | grep -q "PermissionDenied: 403"; then
        echo "🚨 FATAL ERROR DETECTED: Permission denied (403)!"
        show_error_context "PermissionDenied: 403" "Missing IAM permissions (pattern: 'PermissionDenied: 403')"
        echo "❌ Killing agent (PID: $AGENT_PID) - IAM permissions missing"
        kill "$AGENT_PID" 2>/dev/null || true
        print_final_stats
        exit 1
    fi

    if tail -100 "$LOG_FILE" | grep -q "NotFound: 404"; then
        echo "🚨 FATAL ERROR DETECTED: Resource not found (404)!"
        show_error_context "NotFound: 404" "Required GCP resource does not exist (pattern: 'NotFound: 404')"
        echo "❌ Killing agent (PID: $AGENT_PID) - resource doesn't exist"
        kill "$AGENT_PID" 2>/dev/null || true
        print_final_stats
        exit 1
    fi

    if tail -100 "$LOG_FILE" | grep -q "QuotaExceeded\|ResourceExhausted"; then
        echo "🚨 FATAL ERROR DETECTED: Quota exceeded!"
        show_error_context "QuotaExceeded\|ResourceExhausted" "GCP quota limit reached (pattern: 'QuotaExceeded|ResourceExhausted')"
        echo "❌ Killing agent (PID: $AGENT_PID) - quota limit reached"
        kill "$AGENT_PID" 2>/dev/null || true
        print_final_stats
        exit 1
    fi

    # General error patterns (avoid false positives with specific context)
    # Only trigger if the same error appears 3+ times in last 50 lines
    if tail -50 "$LOG_FILE" | grep -c "FAILED\|FAILURE\|FATAL\|Traceback (most recent call last)" | grep -q "[3-9]\|[1-9][0-9]"; then
        echo ""
        echo "🚨 FATAL ERROR DETECTED: Repeated failures in agent logs!"
        echo "❌ Killing agent (PID: $AGENT_PID) - persistent error detected"
        kill "$AGENT_PID" 2>/dev/null || true
        print_final_stats
        exit 1
    fi

    # Check for unhandled exceptions (Python stack traces)
    if tail -50 "$LOG_FILE" | grep -A3 "Traceback (most recent call last)" | grep -q "Error:\|Exception:"; then
        echo ""
        echo "🚨 FATAL ERROR DETECTED: Unhandled Python exception!"
        echo "❌ Killing agent (PID: $AGENT_PID) - exception in agent code"
        kill "$AGENT_PID" 2>/dev/null || true
        print_final_stats
        exit 1
    fi

    # Check for GCP API errors we haven't seen before (5xx errors, other 4xx)
    if tail -50 "$LOG_FILE" | grep -qE "HttpError: <HttpError [45][0-9]{2}"; then
        echo ""
        echo "🚨 FATAL ERROR DETECTED: HTTP error from GCP API!"
        echo "❌ Killing agent (PID: $AGENT_PID) - API error detected"
        kill "$AGENT_PID" 2>/dev/null || true
        print_final_stats
        exit 1
    fi

    # NOTE: We removed the generic 500/503 catch-all check here because:
    # 1. It caused false positives (matched config values like 'MAX_TRAIN_SAMPLES': '500')
    # 2. The HttpError check above (line 119) already catches structured 5xx errors
    # 3. The repeated failures check (line 104) catches persistent unstructured errors
    # If GCP returns "HttpError: <HttpError 500>" we catch it. If it's unstructured,
    # repeated failures check catches it. This prevents false positives.

    # W&B Launch specific errors (from Bright Data research)
    if tail -100 "$LOG_FILE" | grep -qE "Failed to initialize|wandb.*ERROR.*Failed|Unable to connect.*wandb"; then
        echo "🚨 FATAL ERROR DETECTED: W&B agent initialization failure!"
        show_error_context "Failed to initialize|wandb.*ERROR.*Failed|Unable to connect.*wandb" "W&B connection/initialization failed (pattern: 'Failed to initialize|wandb.*ERROR.*Failed|Unable to connect.*wandb')"
        echo "❌ Killing agent (PID: $AGENT_PID) - cannot connect to W&B"
        kill "$AGENT_PID" 2>/dev/null || true
        print_final_stats
        exit 1
    fi

    # Container/Image pull errors (common GCP failure mode)
    if tail -100 "$LOG_FILE" | grep -qE "ImagePullBackOff|ErrImagePull|Failed to pull image"; then
        echo "🚨 FATAL ERROR DETECTED: Container image pull failure!"
        show_error_context "ImagePullBackOff|ErrImagePull|Failed to pull image" "Docker image pull failed from Artifact Registry (pattern: 'ImagePullBackOff|ErrImagePull|Failed to pull image')"
        echo "❌ Killing agent (PID: $AGENT_PID) - cannot pull Docker image"
        kill "$AGENT_PID" 2>/dev/null || true
        print_final_stats
        exit 1
    fi
done

# Agent exited (either completed jobs or error)
wait "$AGENT_PID"
EXIT_CODE=$?

echo ""
echo "✓ W&B agent exited (exit code: $EXIT_CODE)"
print_final_stats
exit $EXIT_CODE
