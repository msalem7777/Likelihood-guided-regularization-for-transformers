#!/usr/bin/env python3
"""Run the complete 380-entry reviewer matrix strictly one run at a time.

This launcher contains no scientific experiment logic. It delegates every
matrix entry to ``examples/reviewer_experiments.py`` and adds resume handling,
per-run logs, and machine-readable progress.

Examples
--------
Validate the configured matrix without training::

    python scripts/run_reviewer_serial.py --dry-run

Start or resume the complete serial run::

    python scripts/run_reviewer_serial.py

Display current completion status::

    python scripts/run_reviewer_serial.py --status-only
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
REVIEWER_RUNNER = REPOSITORY_ROOT / 'examples' / 'reviewer_experiments.py'


@dataclass(frozen=True)
class Study:
    """One named reviewer study and its exact number of matrix entries."""

    name: str
    count: int


STUDIES = (
    Study('cifar100_variance', 20),
    Study('efficiency', 51),
    Study('efficiency_missing_datasets', 240),
    Study('warmstart', 15),
    Study('sensitivity', 18),
    Study('sparse_vd', 36),
)
EXPECTED_TOTAL_RUNS = 380


def result_path(output_root: Path, study: Study, run_index: int) -> Path:
    """Return the CSV path written by one successful matrix entry."""
    return output_root / study.name / f'run_{run_index:03d}.csv'


def run_is_complete(output_root: Path, study: Study, run_index: int) -> bool:
    """A run is complete only when its result CSV exists and is nonempty."""
    path = result_path(output_root, study, run_index)
    return path.is_file() and path.stat().st_size > 0


def count_completed(output_root: Path) -> int:
    """Count completed result CSVs across the complete matrix."""
    return sum(
        run_is_complete(output_root, study, run_index)
        for study in STUDIES
        for run_index in range(study.count)
    )


def write_progress(path: Path, payload: dict[str, object]) -> None:
    """Write progress atomically so interruptions cannot truncate the JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    temporary_path.replace(path)


def validate_matrix() -> None:
    """Confirm both the launcher total and every runner-side study count."""
    configured_total = sum(study.count for study in STUDIES)
    if configured_total != EXPECTED_TOTAL_RUNS:
        raise RuntimeError(
            f'Launcher defines {configured_total} runs; '
            f'expected {EXPECTED_TOTAL_RUNS}.'
        )

    for study in STUDIES:
        command = [
            sys.executable,
            str(REVIEWER_RUNNER),
            '--study',
            study.name,
            '--dry-run',
        ]
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f'Unable to validate study {study.name}:\n'
                f'{completed.stderr}'
            )

        expected_marker = f'Selected runs: {study.count} / {study.count}'
        if expected_marker not in completed.stdout:
            raise RuntimeError(
                f'Study {study.name} no longer contains {study.count} runs.'
            )


def print_status(output_root: Path) -> None:
    """Print current matrix completion without starting a run."""
    completed = count_completed(output_root)
    percentage = 100.0 * completed / EXPECTED_TOTAL_RUNS
    print(
        f'Completed {completed}/{EXPECTED_TOTAL_RUNS} '
        f'({percentage:.2f}%).'
    )


def run_serial(
    output_root: Path,
    logs_root: Path,
    progress_path: Path,
) -> int:
    """Run every incomplete entry serially and return a process exit code."""
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    failures: list[dict[str, object]] = []
    started_at = time.time()

    for study in STUDIES:
        for run_index in range(study.count):
            if run_is_complete(output_root, study, run_index):
                continue

            log_stem = f'{study.name}_run_{run_index:03d}'
            stdout_path = logs_root / f'{log_stem}.out'
            stderr_path = logs_root / f'{log_stem}.err'
            command = [
                sys.executable,
                str(REVIEWER_RUNNER),
                '--study',
                study.name,
                '--run-index',
                str(run_index),
                '--output-root',
                str(output_root),
            ]

            completed_before = count_completed(output_root)
            print(
                f'[{completed_before + 1}/{EXPECTED_TOTAL_RUNS}] '
                f'{study.name} run {run_index:03d}',
                flush=True,
            )

            with stdout_path.open('w', encoding='utf-8') as stdout_handle, \
                    stderr_path.open('w', encoding='utf-8') as stderr_handle:
                result = subprocess.run(
                    command,
                    cwd=REPOSITORY_ROOT,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    check=False,
                )

            successful = (
                result.returncode == 0
                and run_is_complete(output_root, study, run_index)
            )
            if not successful:
                failures.append({
                    'study': study.name,
                    'run_index': run_index,
                    'return_code': result.returncode,
                    'stdout': str(stdout_path),
                    'stderr': str(stderr_path),
                })

            completed_now = count_completed(output_root)
            write_progress(progress_path, {
                'updated_at': datetime.now().astimezone().isoformat(
                    timespec='seconds'
                ),
                'completed': completed_now,
                'total': EXPECTED_TOTAL_RUNS,
                'elapsed_seconds': time.time() - started_at,
                'failures': failures,
            })

    print_status(output_root)
    if failures:
        print(f'{len(failures)} run(s) failed. See {progress_path}.')
        return 1

    print('All 380 reviewer runs completed successfully.')
    return 0


def parse_args() -> argparse.Namespace:
    """Parse launcher-only options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-root',
        default='reviewer_results_local_serial',
    )
    parser.add_argument(
        '--logs-root',
        default='local_logs_serial',
    )
    parser.add_argument(
        '--progress-file',
        default='reviewer_results_local_serial/progress.json',
    )
    parser.add_argument('--status-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main() -> int:
    """Validate, report status, or execute the complete serial matrix."""
    args = parse_args()
    output_root = (REPOSITORY_ROOT / args.output_root).resolve()
    logs_root = (REPOSITORY_ROOT / args.logs_root).resolve()
    progress_path = (REPOSITORY_ROOT / args.progress_file).resolve()

    if not REVIEWER_RUNNER.is_file():
        print(f'Reviewer runner not found: {REVIEWER_RUNNER}', file=sys.stderr)
        return 2

    if args.status_only:
        print_status(output_root)
        return 0

    validate_matrix()
    if args.dry_run:
        print('Validated all 380 reviewer matrix entries; no training started.')
        return 0

    return run_serial(output_root, logs_root, progress_path)


if __name__ == '__main__':
    raise SystemExit(main())
