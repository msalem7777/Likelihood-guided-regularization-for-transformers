#!/usr/bin/env bash
#SBATCH --account=iwave2
#SBATCH --partition=l40s_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=nn_p3_review
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

: "${STUDY:?STUDY must be exported by the submit script}"
: "${REPO_DIR:?REPO_DIR must be exported by the submit script}"

cd "$REPO_DIR"
mkdir -p slurm_logs reviewer_results

# Optional: point REVIEWER_VENV at an existing Python virtual environment.
# If it is unset, the job uses whatever `python` is provided by the submitted environment.
if [[ -n "${REVIEWER_VENV:-}" ]]; then
    # shellcheck disable=SC1090
    source "${REVIEWER_VENV}/bin/activate"
fi

echo "Host: $(hostname)"
echo "Study: ${STUDY}"
echo "Matrix index: ${SLURM_ARRAY_TASK_ID}"
echo "Python: $(command -v python)"
python --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python examples/reviewer_experiments.py \
    --study "$STUDY" \
    --run-index "$SLURM_ARRAY_TASK_ID" \
    --output-root reviewer_results
