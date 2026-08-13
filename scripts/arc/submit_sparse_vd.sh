#!/usr/bin/env bash
set -euo pipefail

# Submit only the six Sparse Variational Dropout reviewer runs. This separate
# launcher prevents accidental resubmission of the earlier 80-run matrix.
REPO_DIR="${REPO_DIR:-$(pwd)}"
REVIEWER_VENV="${REVIEWER_VENV:-}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"

if [[ ! -f "${REPO_DIR}/examples/reviewer_experiments.py" ]]; then
    echo "ERROR: REPO_DIR does not point to the repository root: ${REPO_DIR}" >&2
    exit 1
fi

mkdir -p "${REPO_DIR}/slurm_logs" "${REPO_DIR}/reviewer_results"

# Smoke-check the exact matrix before asking Slurm to allocate GPUs.
python "${REPO_DIR}/examples/reviewer_experiments.py" --study sparse_vd --dry-run

sbatch \
    --time="${TIME_LIMIT}" \
    --array="0-5%${MAX_PARALLEL}" \
    --export="ALL,STUDY=sparse_vd,REPO_DIR=${REPO_DIR},REVIEWER_VENV=${REVIEWER_VENV}" \
    "${REPO_DIR}/scripts/arc/reviewer_array.sbatch"

echo "Submitted 6 Sparse Variational Dropout reviewer runs."
