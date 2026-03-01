#!/usr/bin/env python3
"""Run the full training/inference pipeline end-to-end.

Usage:
    python run_pipeline.py
    python run_pipeline.py --python .venv\\Scripts\\python.exe
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent

PIPELINE = [
    ROOT / "notebooks" / "02_classical_baselines.py",
    ROOT / "notebooks" / "03_quantum_reservoir.py",
    ROOT / "notebooks" / "04_quantum_autoencoder.py",
    ROOT / "notebooks" / "05_final_predictions.py",
]


def run_step(python_bin: str, script: Path) -> None:
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")

    print("=" * 80)
    print(f"Running: {script.relative_to(ROOT)}")
    print("=" * 80)

    t0 = time.time()
    subprocess.run([python_bin, str(script)], cwd=str(ROOT), check=True)
    dt = time.time() - t0
    print(f"Done: {script.name} ({dt:.1f}s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full EPFL Quantum Hack pipeline")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use (default: current interpreter)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Project root: {ROOT}")
    print(f"Python: {args.python}")

    total_start = time.time()
    for step in PIPELINE:
        run_step(args.python, step)

    total_dt = time.time() - total_start
    print("=" * 80)
    print(f"Pipeline completed successfully in {total_dt/60:.1f} minutes")
    print("Artifacts generated in: trained_models/ and data/results.xlsx")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print("=" * 80)
        print(f"Pipeline failed at command: {exc.cmd}")
        print(f"Exit code: {exc.returncode}")
        print("=" * 80)
        raise
