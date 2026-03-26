#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_SUBJECTS = ["sub-06", "sub-09", "sub-11", "sub-14"]

EXPECTED_CUE_EPOCHS_COLUMNS = {
    "subject",
    "trial_id",
    "condition",
    "onset_s",
    "tr_s",
    "tr_relative",
    "time_from_cue_s",
    "nacc_psc",
    "vta_psc",
}

EXPECTED_TRIAL_SUMMARY_COLUMNS = {
    "subject",
    "trial_id",
    "condition",
    "onset_s",
    "tr_s",
    "window_start_tr",
    "window_stop_tr",
    "nacc_psc_window_mean",
    "vta_psc_window_mean",
}

EXPECTED_PNGS = ["qc_nacc.png", "qc_vta.png"]


def parse_args() -> argparse.Namespace:
    repo_root_default = Path(__file__).resolve().parents[3]
    exercise_root_default = repo_root_default / "exercises" / "E08_neuroimaging_nacc"

    parser = argparse.ArgumentParser(
        description="Validate lightweight Exercise 8 outputs before publishing them to students."
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=EXPECTED_SUBJECTS,
        help="Classroom subject labels to validate.",
    )
    parser.add_argument(
        "--derived-root",
        type=Path,
        default=exercise_root_default / "derived",
        help="Path to the live derived outputs root.",
    )
    return parser.parse_args()


def validate_columns(path: Path, expected: set[str]) -> list[str]:
    df = pd.read_csv(path, nrows=5)
    actual = set(df.columns)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    messages: list[str] = []
    if missing:
        messages.append(f"Missing columns in {path.name}: {missing}")
    if extra:
        messages.append(f"Unexpected columns in {path.name}: {extra}")
    return messages


def validate_cue_epochs(path: Path, subject: str) -> list[str]:
    df = pd.read_csv(path)
    messages: list[str] = []

    if df.empty:
        messages.append(f"{path.name} is empty")
        return messages

    if set(df["condition"].dropna().unique()) - {"reward", "neutral"}:
        messages.append(f"Unexpected condition labels in {path.name}")

    subject_values = set(df["subject"].dropna().astype(str).unique())
    if subject_values != {subject}:
        messages.append(f"{path.name} subject values do not match {subject}: {sorted(subject_values)}")

    trial_counts = df.groupby("trial_id").size()
    if (trial_counts < 2).any():
        messages.append(f"Some trial_id values in {path.name} have fewer than 2 rows")

    return messages


def validate_trial_summary(path: Path, subject: str) -> list[str]:
    df = pd.read_csv(path)
    messages: list[str] = []

    if df.empty:
        messages.append(f"{path.name} is empty")
        return messages

    subject_values = set(df["subject"].dropna().astype(str).unique())
    if subject_values != {subject}:
        messages.append(f"{path.name} subject values do not match {subject}: {sorted(subject_values)}")

    if df["trial_id"].duplicated().any():
        messages.append(f"Duplicate trial_id values found in {path.name}")

    if df[["nacc_psc_window_mean", "vta_psc_window_mean"]].isna().any().any():
        messages.append(f"Missing PSC window means found in {path.name}")

    if set(df["condition"].dropna().unique()) - {"reward", "neutral"}:
        messages.append(f"Unexpected condition labels in {path.name}")

    return messages


def validate_subject_dir(derived_root: Path, subject: str) -> list[str]:
    subject_dir = derived_root / subject
    messages: list[str] = []

    if not subject_dir.exists():
        return [f"Missing subject directory: {subject_dir}"]

    cue_epochs_path = subject_dir / "cue_epochs.csv"
    trial_summary_path = subject_dir / "trial_summary.csv"

    for required_path in [cue_epochs_path, trial_summary_path]:
        if not required_path.exists():
            messages.append(f"Missing required file: {required_path}")

    for png_name in EXPECTED_PNGS:
        png_path = subject_dir / png_name
        if not png_path.exists():
            messages.append(f"Missing QC image: {png_path}")

    if not cue_epochs_path.exists() or not trial_summary_path.exists():
        return messages

    messages.extend(validate_columns(cue_epochs_path, EXPECTED_CUE_EPOCHS_COLUMNS))
    messages.extend(validate_columns(trial_summary_path, EXPECTED_TRIAL_SUMMARY_COLUMNS))
    messages.extend(validate_cue_epochs(cue_epochs_path, subject))
    messages.extend(validate_trial_summary(trial_summary_path, subject))
    return messages


def main() -> None:
    args = parse_args()
    all_messages: list[str] = []

    for subject in args.subjects:
        subject_messages = validate_subject_dir(args.derived_root, subject)
        if subject_messages:
            all_messages.extend(subject_messages)
        else:
            print(f"OK: {subject}")

    if all_messages:
        print("Validation failed:")
        for message in all_messages:
            print(f"- {message}")
        raise SystemExit(1)

    print("All requested subject outputs passed validation.")


if __name__ == "__main__":
    main()