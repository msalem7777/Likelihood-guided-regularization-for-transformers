#!/usr/bin/env bash
set -euo pipefail

# Submit all currently implemented reviewer studies as independent one-GPU array tasks.
# The scheduler may queue tasks above your account/QoS GPU limit; MAX_PARALLEL_PER_STUDY
# keeps each individual study from occupying an unlimited number of GPUs at once.

REPO_DIR="${REPO_DIR:-$(pwd)}"
MAX_PARALLEL_PER_STUDY="${MAX_PARALLEL_PER_STUDY:-4}"
REVIEWER_VENV="${REVIEWER_VENV:-}"

mkdir -p "$REPO_DIR/slurm_logs" "$REPO_DIR/reviewer_results"

submit_study() {
    local study="$1"
    local last_index="$2"

    echo "Submitting ${study}: array 0-${last_index}%${MAX_PARALLEL_PER_STUDY}"
    sbatch \
        --array="0-${last_index}%${MAX_PARALLEL_PER_STUDY}" \
        --export="ALL,STUDY=${study},REPO_DIR=${REPO_DIR},REVIEWER_VENV=${REVIEWER_VENV}" \
        scripts/arc/reviewer_array.sbatch
}

# 20 runs: 10 seeds × 2 external-field settings.
submit_study cifar100_variance 19

# 51 runs: the complete two-condition efficiency comparison.
submit_study efficiency 50

# 240 runs: all ten remaining paper dataset/size conditions.
submit_study efficiency_missing_datasets 239

# 15 runs
submit_study warmstart 14

# 18 runs
submit_study sensitivity 17

# 36 runs: all twelve paper dataset/size conditions × 3 seeds.
submit_study sparse_vd 35

echo "Submitted the complete 380-run reviewer matrix."
