#!/usr/bin/env bash
set -euo pipefail

# Submit the 240-run missing-dataset reviewer extension.
# Sixty array tasks each execute four configurations sequentially on one GPU.
REPO_DIR="${REPO_DIR:-$(pwd)}"
REVIEWER_VENV="${REVIEWER_VENV:-${HOME}/.venvs/coldread}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/reviewer_results}"
PARTITION="${PARTITION:-a100_normal_q}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

if [[ ! -f "${REPO_DIR}/examples/reviewer_experiments.py" ]]; then
    echo "ERROR: REPO_DIR does not point to the repository root: ${REPO_DIR}" >&2
    exit 1
fi

mkdir -p "${REPO_DIR}/slurm_logs" "${OUTPUT_ROOT}"

python "${REPO_DIR}/tests/smoke_reviewer_missing_datasets.py"
python "${REPO_DIR}/examples/reviewer_experiments.py" \
    --study efficiency_missing_datasets \
    --dry-run

submission=$(sbatch --parsable \
    --partition="${PARTITION}" \
    --time="${TIME_LIMIT}" \
    --array="0-59%${MAX_PARALLEL}" \
    --export="ALL,REPO_DIR=${REPO_DIR},REVIEWER_VENV=${REVIEWER_VENV},OUTPUT_ROOT=${OUTPUT_ROOT}" \
    "${REPO_DIR}/scripts/arc/reviewer_missing_datasets_packed4.sbatch")

job_id="${submission%%;*}"
echo "Submitted 240 missing-dataset runs as sixty packed tasks."
echo "Job ID: ${job_id}"
