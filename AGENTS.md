# AGENTS.md

## Project Overview
- This repository studies how ENSO background states affect the vertical tilt structure of MJO-related convection and circulation.
- Code lives in `src/`; lightweight path configuration lives in `config/`; exploratory work lives in `notebooks/`; generated figures live in `outputs/`; logs live in `logs/`.
- Large input datasets are stored outside the repository, mainly under `E:\Datas`.
- Assume this repo is analysis-first: preserve reproducibility, avoid ad hoc rewrites, and prefer small, explicit changes.

## Communication Defaults
- Respond in Chinese unless the user explicitly asks for English.
- Be concise and technical; lead with the result, then list key evidence or changed files.
- When assumptions are necessary, state them explicitly.
- If a command or edit could be expensive, destructive, or trigger large recomputation, warn first.

## Repository Map
- `config/paths.py`: canonical local path definitions for project root, data root, outputs, figures, tables, and logs.
- `src/00_*`: data acquisition scripts, including ERA5 download helpers.
- `src/01_*`: preprocessing and Lanczos band-pass filtering for OLR / ERA5 fields.
- `src/02_*`: mvEOF analysis and related reconstruction utilities.
- `src/03_*`: daily tilt computation, q-based tilt diagnostics, verification, and plotting.
- `src/04_*`: event-level tilt statistics.
- `src/05_*`: ENSO-group comparison for tilt or phase-speed metrics.
- `src/06*`: mechanism / correlation / composite / diagnostic analysis modules.
- `src/07*`: 2D latitude-longitude correlation and group composite analysis.
- `notebooks/`: ad hoc tests and exploratory scripts, not the primary production pipeline.
- `outputs/figures/`: generated figures; treat as derived artifacts unless the user explicitly asks to edit or regenerate them.

## Likely Core Pipeline
Use this as the default mental model unless the user specifies a different branch of analysis.
1. `src/00_download_data_new.py` or related download helpers prepare raw inputs under `E:\Datas`.
2. `src/01_lanczos_bandpass.py` filters OLR / ERA5 daily fields into MJO-relevant variability.
3. `src/02_mvEOF.py` performs mvEOF analysis and reconstructs MJO-related fields.
4. `src/03_compute_tilt_daily.py` computes daily tilt indices using reconstructed omega and tracked convection centers.
5. `src/04_tilt_statistics.py` aggregates daily tilt into event-level statistics.
6. `src/05_tilt_by_enso.py` groups event statistics by ENSO phase and generates summary figures.
7. `src/06*` and `src/07*` extend into mechanism diagnosis, correlations, composites, and 2D analyses.

## Environment And Execution
- Preferred environment is the Conda env declared in `environment.yml` (`python=3.11`, `numpy`, `pandas`, `xarray`, `scipy`, `matplotlib`, `dask`, `netcdf4`, `jupyter`).
- Prefer reading path definitions from `config/paths.py` before hardcoding new paths.
- Many existing scripts still contain hardcoded Windows paths under `E:\Datas`; preserve current conventions unless the user asks for path refactoring.
- Assume PowerShell on Windows unless the environment context says otherwise.

## Working Rules
- Read relevant scripts before proposing pipeline changes; filenames are sequential but there are parallel experimental variants (`03b_*`, `05b_*`, `06*`, `07*`).
- Prefer minimal, localized edits; do not rename or reorganize the analysis pipeline without an explicit request.
- Do not overwrite user outputs in `outputs/` or large external data under `E:\Datas` unless the user explicitly asks.
- Treat netCDF, CSV, and generated figures as potentially expensive to regenerate; prefer inspecting metadata or code paths first.
- When summarizing this project, distinguish clearly between production-like scripts in `src/` and exploratory files in `notebooks/`.

## Code Change Guidance
- Follow the existing script-oriented style unless the user asks for refactoring.
- Keep edits ASCII unless the target file already uses non-ASCII and changing that is justified.
- Add comments only when they clarify non-obvious logic.
- Preserve unrelated user changes; never revert them unless explicitly requested.
- If you discover unexpected modifications during work, stop and ask how to proceed.

## Validation Guidance
- For documentation-only tasks, no execution is required.
- For code changes, run the smallest validation that meaningfully checks the change.
- Prefer targeted validation over full-pipeline runs; many scripts likely depend on external datasets not present in the repo.
- If validation is skipped because data or environment prerequisites are unavailable, say so explicitly.

## Output Expectations
- When reporting code changes, list the touched files and the reason for each change.
- When reporting analysis structure, mention script order, inputs, outputs, and whether a file looks primary or exploratory.
- When giving next steps, keep them concrete, short, and ordered.

## Things To Avoid By Default
- Do not bulk-edit all `*_new.py` / `*_new2.py` variants unless the user clearly wants all variants updated.
- Do not assume every numbered script is current; inspect the specific file the user mentions.
- Do not trigger large downloads, full historical recomputation, or figure regeneration without confirmation.
- Do not treat `README.md` as perfectly authoritative if it conflicts with the current `src/` tree; prefer the actual repository state.
