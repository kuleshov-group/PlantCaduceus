#!/usr/bin/env bash
#
# watchdog.sh -- auto-restart wrapper for src/run_finetune_expression.sh
#
# Monitors the training log for NaN/Inf in loss or grad_norm, kills the run
# early (before a poisoned checkpoint can be saved), and resumes from the
# last checkpoint written *before* any NaN was observed in this run's log.
#
# Usage (run from ~/plantcad, on the HOST, not inside the container):
#   ./watchdog.sh
#
# Config via env vars (all optional, sane defaults below):
#   DATA_PREFIX, MODEL_PATH, NUM_EPOCHS  -- forwarded to the training script
#   MAX_RETRIES     -- give up after this many restarts (default 5)
#   CHECK_INTERVAL  -- seconds between log checks (default 30)
#   SANITY          -- 1 for a fast 30-step smoke test instead of a full run (default 0)

trap 'docker compose kill plantcad 2>/dev/null; exit 130' INT TERM

set -euo pipefail
cd "$(dirname "$0")"

DATA_PREFIX="${DATA_PREFIX:-./data/expression_dataset_1024}"
MODEL_PATH="${MODEL_PATH:-./model/PlantCAD2-Small-l24-d0768}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
MAX_RETRIES="${MAX_RETRIES:-5}"
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"
SANITY="${SANITY:-0}"

RUN_LOG_DIR="./watchdog_logs"
mkdir -p "$RUN_LOG_DIR"

attempt=0
resume_checkpoint=""   # empty = start fresh; set after a verified-clean checkpoint exists
resume_run_name=""

log() { echo "[watchdog $(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Returns 0 (true) if the given log file contains a NaN/Inf loss or grad_norm line
has_nan() {
    local logfile="$1"
    grep -Eq "'grad_norm': (nan|inf)|'loss': (nan|inf)|'eval_loss': (nan|inf)" "$logfile" 2>/dev/null
}

# Prints the highest checkpoint-N directory under a run dir, or empty if none
latest_checkpoint() {
    local run_dir="$1"
    ls -1 "$run_dir" 2>/dev/null | grep -E '^checkpoint-[0-9]+$' \
        | sort -t- -k2 -n | tail -1
}

launch_and_monitor() {
    local run_name="plantcad2_lettuce_expression_$(date +%Y%m%d_%H%M%S)"
    local logfile="${RUN_LOG_DIR}/${run_name}.log"

    log "Starting attempt $((attempt+1))/$MAX_RETRIES -- run_name=$run_name"
    [ -n "$resume_checkpoint" ] && log "Resuming from: $resume_checkpoint"

    # Launch training as a background job we can signal later. Docker Compose
    # itself becomes the process we track; we kill via `docker compose kill`
    # so the container is actually torn down, not just the host-side client.
    (
        cd "$(dirname "$0")"
        RUN_NAME="$run_name" \
        DATA_PREFIX="$DATA_PREFIX" \
        MODEL_PATH="$MODEL_PATH" \
        NUM_EPOCHS="$NUM_EPOCHS" \
        RESUME_FROM="$resume_checkpoint" \
        SANITY="$SANITY" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        docker compose run --rm \
            -e RUN_NAME -e DATA_PREFIX -e MODEL_PATH -e NUM_EPOCHS -e RESUME_FROM -e SANITY -e CUDA_VISIBLE_DEVICES\
            plantcad \
            bash -c "pip install --quiet fire tensorboard && bash src/run_finetune_expression.sh"
    ) > "$logfile" 2>&1 &
    local compose_pid=$!

    log "Launched (host pid $compose_pid). Watching $logfile"

    while kill -0 "$compose_pid" 2>/dev/null; do
        if has_nan "$logfile"; then
            log "NaN/Inf detected in $logfile -- killing this attempt NOW"
            # Tear the container down at the Docker level, not just the host
            # process, so nothing keeps running (or writing a checkpoint) after we return.
            docker compose kill plantcad 2>/dev/null || true
            kill "$compose_pid" 2>/dev/null || true
            wait "$compose_pid" 2>/dev/null || true
            echo "NAN_DETECTED"
            return
        fi
        sleep "$CHECK_INTERVAL"
    done

    wait "$compose_pid"
    local exit_code=$?
    if [ "$exit_code" -eq 0 ] && grep -q "^Done\. Run directory:" "$logfile"; then
        echo "COMPLETED"
    else
        # Died for a reason other than NaN (crash, OOM, disconnect, etc.)
        echo "CRASHED"
    fi

    # Record run_name globally for the caller to inspect checkpoints
    echo "$run_name" > "${RUN_LOG_DIR}/.last_run_name"
}

while [ "$attempt" -lt "$MAX_RETRIES" ]; do
    result=$(launch_and_monitor | tail -1)
    run_name=$(cat "${RUN_LOG_DIR}/.last_run_name" 2>/dev/null || echo "")
    run_dir="./model/${run_name}"

    case "$result" in
        COMPLETED)
            log "Training completed successfully. Run dir: $run_dir"
            exit 0
            ;;
        NAN_DETECTED|CRASHED)
            attempt=$((attempt+1))
            log "Attempt failed ($result). $attempt/$MAX_RETRIES used."

            # Only trust a checkpoint from THIS run if its own log never saw NaN
            # up to the point it was saved -- since detection fires before the
            # next save interval, any checkpoint on disk from a NaN'd run is
            # suspect. Simplest safe rule: if this attempt ever saw NaN, do NOT
            # resume from anything it produced -- fall back to the last
            # checkpoint verified clean from a PRIOR attempt (if any), or fresh.
            ckpt=$(latest_checkpoint "$run_dir") || ckpt=""
            if [ "$result" = "CRASHED" ] && [ -n "$ckpt" ]; then
                resume_checkpoint="${run_dir}/${ckpt}"
                resume_run_name="$run_name"
                log "Non-NaN crash with a checkpoint present -- will resume from $resume_checkpoint"
            elif [ "$result" = "NAN_DETECTED" ]; then
                log "NaN'd run -- discarding any checkpoints it wrote, retrying fresh"
                resume_checkpoint=""
                resume_run_name=""
            fi

            if [ "$attempt" -lt "$MAX_RETRIES" ]; then
                backoff=$(( 30 * attempt ))
                log "Backing off ${backoff}s before retry..."
                sleep "$backoff"
            fi
            ;;
    esac
done

log "Max retries ($MAX_RETRIES) exhausted without a successful run. Giving up."
log "This many consecutive failures suggests a systematic issue, not the usual"
log "~40%% random NaN rate -- worth escalating rather than retrying further."
exit 1
