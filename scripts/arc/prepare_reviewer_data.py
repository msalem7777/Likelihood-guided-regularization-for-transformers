"""Download the datasets needed by the reviewer experiment matrix exactly once.

Run this before launching the SLURM arrays so simultaneous workers never race while
extracting the same torchvision archive.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples.reviewer_experiments import ensure_data


def main() -> None:
    for dataset in ("mnist", "cifar100"):
        print(f"Preparing {dataset}...")
        ensure_data(dataset)
    print("Reviewer datasets are ready.")


if __name__ == "__main__":
    main()
