from __future__ import annotations

import argparse
from pathlib import Path

from src.models.quantum.quantum_reservoir_lstm import (
    evaluate_notebook_baseline,
    load_default_train_template,
    evaluate_quantum_reservoir_lstm,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the quantum-inspired reservoir + LSTM hybrid model."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root. Defaults to the current repo.",
    )
    parser.add_argument(
        "--save-submission",
        type=Path,
        default=None,
        help="Optional path for writing the QR+LSTM submission workbook.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    train_df, template_df = load_default_train_template(project_root)

    baseline = evaluate_notebook_baseline(train_df)
    qr_summary, qr_model = evaluate_quantum_reservoir_lstm(train_df)
    submission = qr_model.build_submission(template_df)

    print("== Benchmark ==")
    print(
        f"{baseline.name}: factor_mse={baseline.factor_mse:.10f} "
        f"surface_mse={baseline.surface_mse:.10f} [{baseline.details}]"
    )
    print(
        f"{qr_summary.name}: factor_mse={qr_summary.factor_mse:.10f} "
        f"surface_mse={qr_summary.surface_mse:.10f} [{qr_summary.details}]"
    )

    baseline_gain = (baseline.surface_mse - qr_summary.surface_mse) / baseline.surface_mse
    print(f"gain_vs_notebook={baseline_gain:.2%}")

    missing_mask = submission["Type"].astype(str).str.lower().str.contains("missing")
    value_cols = [c for c in submission.columns if c not in ("Type", "Date")]
    remaining_nans = int(submission.loc[:, value_cols].isna().sum().sum())
    print("\n== Submission Sanity ==")
    print(f"future_rows={int((submission['Type'] == 'Future prediction').sum())}")
    print(f"missing_rows={int(missing_mask.sum())}")
    print(f"remaining_missing_values={remaining_nans}")

    if args.save_submission is not None:
        output_path = args.save_submission.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_excel(output_path, index=False)
        print(f"saved_submission={output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
