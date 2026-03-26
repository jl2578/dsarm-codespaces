#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image, plotting
from nilearn.image import resample_to_img
from nilearn.masking import apply_mask


SUBJECT_MAP = {
    "sub-06": "sub-s006",
    "sub-09": "sub-s009",
    "sub-11": "sub-s011",
    "sub-14": "sub-s014",
}

EPOCH_PRE_TRS = 2
EPOCH_POST_TRS = 14
WINDOW_LAG_TRS = 5
WINDOW_WIDTH_TRS = 2
VTA_MNI = (-4, -15, -9)
VTA_RADIUS_MM = 5


def parse_args() -> argparse.Namespace:
    repo_root_default = Path(__file__).resolve().parents[3]
    exercise_root_default = repo_root_default / "exercises" / "E08_neuroimaging_nacc"

    parser = argparse.ArgumentParser(
        description=(
            "Generate lightweight Exercise 8 outputs from raw fMRI inputs. "
            "This script is intended for instructor-side preprocessing only."
        )
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["sub-06"],
        help="Classroom subject labels to process. Use one or more of: sub-06 sub-09 sub-11 sub-14.",
    )
    parser.add_argument(
        "--exercise-root",
        type=Path,
        default=exercise_root_default,
        help="Path to the Exercise 8 folder.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=exercise_root_default / "data",
        help=(
            "Root folder containing raw OpenNeuro subject folders, e.g. data/sub-s006/func/... "
            "This script does not download data."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=exercise_root_default / "derived",
        help="Folder where lightweight outputs will be written.",
    )
    return parser.parse_args()


def ensure_supported_subjects(subjects: list[str]) -> None:
    unsupported = [subject for subject in subjects if subject not in SUBJECT_MAP]
    if unsupported:
        allowed = ", ".join(SUBJECT_MAP)
        raise ValueError(f"Unsupported subject(s): {unsupported}. Allowed subjects: {allowed}")


def build_vta_mask_from_bold(bold_path: Path, out_path: Path) -> Path:
    bold_img = nib.load(str(bold_path))
    affine = bold_img.affine
    shape = bold_img.shape[:3]

    ijk = np.indices(shape).reshape(3, -1).T
    xyz = nib.affines.apply_affine(affine, ijk)
    center = np.array(VTA_MNI, dtype=float)
    dist_mm = np.linalg.norm(xyz - center, axis=1)

    mask_data = (dist_mm <= VTA_RADIUS_MM).reshape(shape).astype("uint8")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask_data, affine), str(out_path))
    return out_path


def extract_psc(bold_path: Path, mask_path: Path, roi_name: str) -> dict[str, np.ndarray | float | str]:
    bold_img = image.load_img(str(bold_path))
    mask_resampled = resample_to_img(
        source_img=str(mask_path),
        target_img=bold_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    roi_voxels = apply_mask(bold_img, mask_resampled)
    raw_ts = roi_voxels.mean(axis=1)
    baseline_raw = float(raw_ts.mean())
    psc_ts = 100.0 * (raw_ts - baseline_raw) / baseline_raw

    return {
        "roi_name": roi_name,
        "raw_ts": raw_ts,
        "psc_ts": psc_ts,
        "baseline_raw": baseline_raw,
    }


def label_cue_condition(trial_type: str) -> str | None:
    text = str(trial_type).lower()
    if "reward" in text or "gain" in text:
        return "reward"
    if "neutral" in text or "triangle" in text:
        return "neutral"
    return None


def build_cue_tables(events_df: pd.DataFrame, tr_s: float, nacc_psc: np.ndarray, vta_psc: np.ndarray, subject: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cue_events = events_df.copy()
    cue_events["condition"] = cue_events["trial_type"].map(label_cue_condition)
    cue_events = cue_events[cue_events["condition"].notna()].reset_index(drop=True)
    cue_events["trial_id"] = np.arange(1, len(cue_events) + 1)

    epoch_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []

    for row in cue_events.itertuples(index=False):
        onset_s = float(row.onset)
        onset_tr = int(round(onset_s / tr_s))
        window_start_tr = onset_tr + WINDOW_LAG_TRS
        window_stop_tr = window_start_tr + WINDOW_WIDTH_TRS

        nacc_window = nacc_psc[window_start_tr:window_stop_tr]
        vta_window = vta_psc[window_start_tr:window_stop_tr]

        summary_rows.append(
            {
                "subject": subject,
                "trial_id": row.trial_id,
                "condition": row.condition,
                "onset_s": onset_s,
                "tr_s": tr_s,
                "window_start_tr": window_start_tr,
                "window_stop_tr": window_stop_tr,
                "nacc_psc_window_mean": float(np.nanmean(nacc_window)),
                "vta_psc_window_mean": float(np.nanmean(vta_window)),
            }
        )

        for tr_relative in range(-EPOCH_PRE_TRS, EPOCH_POST_TRS):
            tr_index = onset_tr + tr_relative
            if tr_index < 0 or tr_index >= len(nacc_psc):
                continue

            epoch_rows.append(
                {
                    "subject": subject,
                    "trial_id": row.trial_id,
                    "condition": row.condition,
                    "onset_s": onset_s,
                    "tr_s": tr_s,
                    "tr_relative": tr_relative,
                    "time_from_cue_s": tr_relative * tr_s,
                    "nacc_psc": float(nacc_psc[tr_index]),
                    "vta_psc": float(vta_psc[tr_index]),
                }
            )

    cue_epochs = pd.DataFrame(epoch_rows)
    trial_summary = pd.DataFrame(summary_rows)
    return cue_epochs, trial_summary


def save_qc_images(mean_bold_path: Path, nacc_mask_path: Path, vta_mask_path: Path, subject_dir: Path) -> None:
    qc_nacc_path = subject_dir / "qc_nacc.png"
    qc_vta_path = subject_dir / "qc_vta.png"

    display = plotting.plot_roi(str(nacc_mask_path), bg_img=str(mean_bold_path), draw_cross=False)
    display.savefig(str(qc_nacc_path), dpi=150)
    display.close()

    display = plotting.plot_roi(str(vta_mask_path), bg_img=str(mean_bold_path), draw_cross=False)
    display.savefig(str(qc_vta_path), dpi=150)
    display.close()


def process_subject(subject: str, exercise_root: Path, raw_root: Path, output_root: Path) -> None:
    raw_subject = SUBJECT_MAP[subject]
    func_dir = raw_root / raw_subject / "func"

    bold_candidates = sorted(func_dir.glob("*task-mid*_bold.nii*"))
    events_candidates = sorted(func_dir.glob("*task-mid*_events.tsv"))

    if not bold_candidates:
        raise FileNotFoundError(f"No MID BOLD file found for {subject} in {func_dir}")
    if not events_candidates:
        raise FileNotFoundError(f"No MID events file found for {subject} in {func_dir}")

    bold_path = bold_candidates[0]
    events_path = events_candidates[0]
    subject_dir = output_root / subject
    subject_dir.mkdir(parents=True, exist_ok=True)

    nacc_mask_path = exercise_root / "data" / "nacc_bilateral_mask.nii"
    if not nacc_mask_path.exists():
        raise FileNotFoundError(f"Missing NAcc mask: {nacc_mask_path}")

    vta_mask_path = subject_dir / "vta_sphere_mask.nii.gz"
    build_vta_mask_from_bold(bold_path, vta_mask_path)

    nacc = extract_psc(bold_path, nacc_mask_path, "nacc")
    vta = extract_psc(bold_path, vta_mask_path, "vta")

    bold_img = image.load_img(str(bold_path))
    mean_bold = image.mean_img(bold_img)
    mean_bold_path = subject_dir / "mean_bold.nii.gz"
    mean_bold.to_filename(str(mean_bold_path))

    events_df = pd.read_csv(events_path, sep="\t")
    zooms = bold_img.header.get_zooms()
    tr_s = float(zooms[3])

    cue_epochs, trial_summary = build_cue_tables(
        events_df=events_df,
        tr_s=tr_s,
        nacc_psc=nacc["psc_ts"],
        vta_psc=vta["psc_ts"],
        subject=subject,
    )

    cue_epochs.to_csv(subject_dir / "cue_epochs.csv", index=False)
    trial_summary.to_csv(subject_dir / "trial_summary.csv", index=False)
    save_qc_images(mean_bold_path, nacc_mask_path, vta_mask_path, subject_dir)


def main() -> None:
    args = parse_args()
    ensure_supported_subjects(args.subjects)

    for subject in args.subjects:
        process_subject(
            subject=subject,
            exercise_root=args.exercise_root,
            raw_root=args.raw_root,
            output_root=args.output_root,
        )
        print(f"Wrote packaged outputs for {subject} -> {args.output_root / subject}")


if __name__ == "__main__":
    main()