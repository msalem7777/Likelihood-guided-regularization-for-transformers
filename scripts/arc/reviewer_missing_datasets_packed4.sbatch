#!/usr/bin/env bash
#SBATCH --account=iwave2
#SBATCH --partition=a100_normal_q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=nn_p3_missing
#SBATCH --output=slurm_logs/nn_p3_missing_%A_%a.out
#SBATCH --error=slurm_logs/nn_p3_missing_%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:?REPO_DIR must point to the repository root}"
REVIEWER_VENV="${REVIEWER_VENV:?REVIEWER_VENV must point to the Python virtual environment}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/reviewer_results}"
PACK_SIZE=4
LAST_RUN_INDEX=239

cd "${REPO_DIR}"
source "${REVIEWER_VENV}/bin/activate"
mkdir -p "${REPO_DIR}/slurm_logs" "${OUTPUT_ROOT}"

echo "Host: $(hostname)"
echo "Study: efficiency_missing_datasets"
echo "Packed task: ${SLURM_ARRAY_TASK_ID}"
echo "Commit: $(git rev-parse HEAD)"
echo "Python: $(command -v python)"
python --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

start_index=$((SLURM_ARRAY_TASK_ID * PACK_SIZE))
end_index=$((start_index + PACK_SIZE - 1))
if (( end_index > LAST_RUN_INDEX )); then
    end_index=${LAST_RUN_INDEX}
fi

for run_index in $(seq "${start_index}" "${end_index}"); do
    result_file=$(printf "%s/efficiency_missing_datasets/run_%03d.csv" "${OUTPUT_ROOT}" "${run_index}")
    if [[ -s "${result_file}" ]]; then
        echo "Skipping completed matrix index ${run_index}: ${result_file}"
        continue
    fi

    run_log=$(printf "%s/slurm_logs/efficiency_missing_run_%02d_%s.log" "${REPO_DIR}" "${run_index}" "${SLURM_ARRAY_JOB_ID}")
    echo "Starting matrix index ${run_index}; detailed log: ${run_log}"
    python examples/reviewer_experiments.py \
        --study efficiency_missing_datasets \
        --run-index "${run_index}" \
        --output-root "${OUTPUT_ROOT}" \
        2>&1 | tee "${run_log}"
done

echo "Packed task ${SLURM_ARRAY_TASK_ID} complete: indices ${start_index}-${end_index}."
