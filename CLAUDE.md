# CXR Report Generation

## What this is
Encoder-decoder for chest X-ray → radiology report (CNN encoder → classification head + sequence decoder).
A learning project. The goal is to UNDERSTAND, not to finish fast.

The step-by-step plan is `CXR_Report_Project_Roadmap.pdf` (local only, gitignored). It is written for
TensorFlow/Keras — translate its steps to PyTorch.

## About me
Just finished the Deep Learning Specialization (deeplearning.ai, TensorFlow/Keras). First real project,
new to PyTorch — explain PyTorch-specific things when they first appear.

## How I want you to work
- One step at a time. Don't write whole modules unless I explicitly ask.
- Explain WHY before the code, briefly.
- When there are alternatives, tell me them and the trade-off — don't choose silently.
- If you see a bug in my code, show me where it is before fixing it.
- Ask before adding dependencies.

## Stack
PyTorch / torchvision (CUDA, cu130 wheels), numpy, pandas, scikit-learn. Python 3.11, Windows, NVIDIA GPU.

## Conventions
- Logic lives in `src/`; notebooks are for exploration only.
- Hyperparameters ONLY from `configs/*.yaml` — never hardcoded.
- Every experiment writes a row to `experiments/results.csv`.
- Seeds: python `random`, numpy, torch — set at every entry point.
- `data/raw/` is never modified.
- Split by patient, never by image.

## Status
Phase: 0 — Steps 0.1 (inventory), 0.2 (split), 0.3 (labels) done; next: decide on stratified re-split, then Step 0.4 (metrics module)
What works: environment (torch 2.14.1+cu130, GPU detected), IU X-Ray in `data/raw/`, `src/data/inventory.py` (reports 3,955×9, images 7,470×5), `src/data/splits.py` (`python -m src.data.splits`: report-level 70/15/15 from `configs/base.yaml`, seed 42 → `data/interim/{train,val,test}.csv` = 2,695/577/579 reports, 5,231/1,120/1,119 images), `src/data/labels.py` (`python -m src.data.labels`: MeSH `major` → CheXpert 14 via `configs/label_map.yaml` → `data/interim/labels.csv`, 3,955 rows; `labeled`=0 for the 95 `No Indexing` reports; write-up in `reports/labels.md`), `python -m pytest` (3 tests pass)
What doesn't work: —
Open decisions: no patient ID — `IM-####` is an export counter (`notebooks/02_splits.ipynb`), so split is by report (limitation: one person's reports may cross sets); random split spreads rare categories unevenly (Pneumothorax 20/1/5, Consolidation 19/8/3 train/val/test) — keep, or re-split stratified by label? (`reports/labels.md` §6); 1,007 labeled reports are abnormal but all-zero (findings outside the 14); frontal vs lateral not labelled (Phase 1)
