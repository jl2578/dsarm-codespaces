# Exercise 8 Instructor Workflow

This folder contains the maintainer-only workflow for regenerating the lightweight packaged data used by the live student notebook.

## Live Student Workflow

Students use:

- `exercises/E08_neuroimaging_nacc/Exercise8.ipynb`
- `exercises/E08_neuroimaging_nacc/derived/`
- `exercises/E08_neuroimaging_nacc/data/`

The `figs/` folder is only an output location for figures generated while running the notebook.

## Instructor Files

- `exercises/E08_neuroimaging_nacc/instructor/preprocess_exercise8.py`
- `exercises/E08_neuroimaging_nacc/instructor/validate_derived_outputs.py`
- `exercises/E08_neuroimaging_nacc/instructor/archive/Exercise8_legacy_raw.ipynb`

## Derived Outputs

For each classroom subject (`sub-06`, `sub-09`, `sub-11`, `sub-14`), the preprocessing script writes:

- `cue_epochs.csv`
- `trial_summary.csv`
- `qc_nacc.png`
- `qc_vta.png`
- `vta_sphere_mask.nii.gz`

These are written to:

```text
exercises/E08_neuroimaging_nacc/derived/<subject>/
```

Example:

```text
exercises/E08_neuroimaging_nacc/derived/sub-06/
```

## Assumptions

- Raw OpenNeuro files already exist locally under `exercises/E08_neuroimaging_nacc/data/sub-sXXX/func/`
- The committed NAcc mask exists at `exercises/E08_neuroimaging_nacc/data/nacc_bilateral_mask.nii`
- This script is run by an instructor or maintainer, not by students

The script does not download data. It only reads existing raw inputs and refreshes the live lightweight outputs.

## Run One Subject

From the repository root:

```bash
python3 exercises/E08_neuroimaging_nacc/instructor/preprocess_exercise8.py --subjects sub-06
```

## Run Multiple Subjects

```bash
python3 exercises/E08_neuroimaging_nacc/instructor/preprocess_exercise8.py --subjects sub-06 sub-09 sub-11 sub-14
```

## Validate Outputs

After generating derived files, validate that each subject folder has the expected files and columns:

```bash
python3 exercises/E08_neuroimaging_nacc/instructor/validate_derived_outputs.py --subjects sub-06
```

To validate all four classroom subjects:

```bash
python3 exercises/E08_neuroimaging_nacc/instructor/validate_derived_outputs.py
```

## Output Schema

### cue_epochs.csv

One row per cue trial per peri-cue timepoint.

Columns:

- `subject`
- `trial_id`
- `condition`
- `onset_s`
- `tr_s`
- `tr_relative`
- `time_from_cue_s`
- `nacc_psc`
- `vta_psc`

### trial_summary.csv

One row per cue trial.

Columns:

- `subject`
- `trial_id`
- `condition`
- `onset_s`
- `tr_s`
- `window_start_tr`
- `window_stop_tr`
- `nacc_psc_window_mean`
- `vta_psc_window_mean`

## Processing Choices

These match the live notebook logic so the storage/compute optimization does not change the analysis definition.

- PSC baseline: whole-run mean ROI signal
- Epoch window: `-2` to `+13` TR relative to cue onset
- Trial summary window: start at `+5` TR, width `2` TR
- VTA sphere center: `(-4, -15, -9)`
- VTA sphere radius: `5 mm`

## Regeneration Workflow

1. Run preprocessing for the subject set you need to refresh.
2. Run the validator against `derived/`.
3. Smoke-test `Exercise8.ipynb` on one of the refreshed subjects.