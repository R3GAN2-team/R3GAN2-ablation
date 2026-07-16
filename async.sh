#!/usr/bin/env bash
set -Eeuo pipefail

# Put this file next to async_eval.sh.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

JOB_NAME="async-eval-auto-in2x"
POLL_SECONDS=300
LOG_FILE="slurm.out"
LOCK_FILE=".async_eval_resubmitter.lock"

# Prevent accidentally running two resubmission loops at once.
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "Another async-eval resubmitter is already running in $(pwd)." >&2
    exit 1
fi

submit_eval() {
    local job_id
    job_id="$(
        sbatch --parsable \
            --job-name="${JOB_NAME}" \
            --partition=gpu \
            --gres=gpu:2 \
            --cpus-per-task=8 \
            --mem=96G \
            --constraint="l40s" \
            --time=01:00:00 \
            --exclude=gpu3005,gpu2709 \
            --output="${LOG_FILE}" \
            --error="${LOG_FILE}" \
            --open-mode=truncate \
            async_eval.sh
    )"
    job_id="${job_id%%;*}"
    echo "[$(date --iso-8601=seconds)] Submitted job ${job_id}."
}

job_is_active() {
    squeue --noheader --user="${USER}" --name="${JOB_NAME}" --format="%A" | grep -q .
}

echo "Watching every ${POLL_SECONDS}s. Press Ctrl-C to stop."
echo "Working directory: $(pwd)"

while true; do
    if job_is_active; then
        active_ids="$(
            squeue --noheader --user="${USER}" --name="${JOB_NAME}" \
                --format="%A %T" | paste -sd ', ' -
        )"
        echo "[$(date --iso-8601=seconds)] Evaluation still active: ${active_ids}"
    else
        # The prior job has left squeue, so it has finished, failed, or been cancelled.
        rm -f -- "${LOG_FILE}"
        echo "[$(date --iso-8601=seconds)] Previous evaluation is no longer active; deleted ${LOG_FILE}."
        submit_eval
    fi

    sleep "${POLL_SECONDS}"
done
