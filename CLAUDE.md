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
Phase: 0 — Step 0.1: inventory module done (`python -m src.data.inventory`); next: write `reports/inventory.md`, then Step 0.2 (patient-level split)
What works: environment (torch 2.14.1+cu130, GPU detected), IU X-Ray in `data/raw/`, `src/data/inventory.py` builds reports (3,955×9) + images (7,470×5) tables
What doesn't work: —
Open decisions: no patient ID in the XML (Step 0.2); frontal vs lateral not labelled (Phase 1)
