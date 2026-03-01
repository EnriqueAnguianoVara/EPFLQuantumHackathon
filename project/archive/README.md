# Archive Folder

This folder stores legacy or exploratory material kept for traceability.

## Purpose

- Preserve experiments that are not required for the official submission flow.
- Keep the root repository cleaner for judges and reviewers.
- Avoid accidental use of out-of-scope scripts and assets.

## Scope

The official runnable workflow is:

1. `python run_pipeline.py`
2. `streamlit run app.py`

Anything under `archive/` is outside that workflow and is not required to
reproduce the official benchmark and submission outputs.

## Current Contents

- `archive/legacy/`: legacy scripts and model prototypes moved from `src/`.
- `archive/Quandela_folder/`: challenge resources preserved as reference files.
