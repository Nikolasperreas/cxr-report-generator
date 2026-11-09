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
Phase: 1 — Phase 0 done (Steps 0.1–0.5; Step 0.6 hand-labelling of 50 val X-rays skipped, decided 2026-11-03); Step 1.1 (input pipeline) done; Step 1.3 in progress (model + train/evaluate loops done; next: 1.3c training script + `configs/small_cnn.yaml` + results row; small CNN trained WITHOUT augmentation → its results row is the reference), then Step 1.2 (augmentation: small rotation/translation/zoom/brightness-contrast, NO flips; runs: + augmentation, + flip, each compared to the 1.3 row) — order swapped 2026-11-04 so each run changes one thing
TODO when the first CNN result comes in: create the four-gap sheet (Bayes proxy from published radiologist agreement, train/dev/test error, avoidable bias, variance, data mismatch, dev-test overfit), row 1 = numpy baseline incl. its one-time test score, row 2 = the CNN
What works: environment (torch 2.14.1+cu130, GPU detected), IU X-Ray in `data/raw/`, `src/data/inventory.py` (reports 3,955×9, images 7,470×5), `src/data/splits.py` (`python -m src.data.splits`: report-level 70/15/15 from `configs/base.yaml`, seed 42 → `data/interim/{train,val,test}.csv` = 2,695/577/579 reports, 5,231/1,120/1,119 images), `src/data/labels.py` (`python -m src.data.labels`: MeSH `major` → CheXpert 14 via `configs/label_map.yaml` → `data/interim/labels.csv`, 3,955 rows; `labeled`=0 for the 95 `No Indexing` reports; write-up in `reports/labels.md`), `src/evaluation/metrics_cls.py` (`classification_metrics(y_true, y_prob, threshold, class_names)` → one row per category + `macro` + `micro`; columns positives, tp, fp, fn, tn, precision, recall, f1, auc; precision 0/0 → 0, nothing to grade → NaN, skipped by macro; threshold scalar or one per category), `python -m pytest` (8 tests pass: 3 splits, 5 metrics vs toy + sklearn), `src/data/small_images.py` (`python -m src.data.small_images` → `data/processed/{split}_32.npz` with X float32 (m, 1024), Y (m, 14), image_ids; labeled reports only: 5,114/1,099/1,085 images), `src/utils.py` (`set_seeds`: random, numpy, torch), `src/models/baseline_numpy.py` (logistic regression, 14 outputs, rows = examples, full-batch GD, gradients / m), `src/training/train_baseline.py` (`python -m src.training.train_baseline --comment "..."`, settings from `configs/baseline_numpy.yaml`, appends a row to `experiments/results.csv`), `src/data/pipeline.py` (`CXRDataset(split_df, labels_df, categories, size, transform=None)`: labeled reports only, whole split cached in RAM as uint8 (m, 1, size, size), squashed, 1 channel, ~11 s / 260 MB for train at 224; `__getitem__` → float 0..1, then transform; `make_loader(split, size, batch_size, transform=None)`: shuffle only for train, num_workers 0; same image order and labels as the Step 0.5 arrays; batch plot in `notebooks/06_pipeline.ipynb`: upright, not mirrored, frontal + lateral of a report come as a pair with the same labels, text markers burned into some images), `src/models/encoders.py` (`conv_block` = Conv3x3 no bias → BatchNorm → ReLU → MaxPool2; `SmallCNN(channels, n_classes)`: one block per entry of channels [16, 32, 64, 128, 256], global average pool → features (N, 256), Linear head → logits (N, 14); 396,414 params), `src/training/loops.py` (`train_one_epoch(model, loader, loss_fn, optimizer, device)` → average train loss per example; `evaluate(model, loader, loss_fn, device)` → (loss, y_true, y_prob) as numpy, eval mode + no_grad); notebook checks in `06_pipeline.ipynb`: overfit one val batch of 16 → loss 0.694 → 0.0034 in 101 steps; 3 epochs (batch 32, Adam 1e-3, BCEWithLogits, ~3 s/epoch): train 0.218 → 0.197, val loss 0.196, val macro F1 0.019, macro AUC 0.681 at threshold 0.5 (exploration, not a results row)
First result (baseline, lr 0.01, 2000 steps): val macro F1 0.021, micro F1 0.111, macro AUC 0.617; train ≈ val → high bias, no variance; final cost 0.1973 vs 0.205 for "always predict each category's frequency" → pixels add very little
Decided: macro F1 is the headline metric, micro reported next to it (`README.md`); `results.csv` columns = timestamp, model, config_path, seed, config (whole config as JSON), final_cost, train/val macro F1, micro F1, macro AUC, comment; test set is never scored in training scripts, only once per phase on the chosen model; loops live in `src/training/loops.py`, shared by every PyTorch model; `evaluate` returns probabilities and the caller computes metrics (so threshold tuning and error analysis can reuse them)
What doesn't work: —
Open decisions: no patient ID — `IM-####` is an export counter (`notebooks/02_splits.ipynb`), so split is by report (limitation: one person's reports may cross sets); random split KEPT (decided 2026-10-29) although rare categories are uneven (Pneumothorax 20/1/5, Consolidation 19/8/3 train/val/test; `reports/labels.md` §6) → show positives next to every per-class score, no Pneumothorax threshold tuning on val; 1,007 labeled reports are abnormal but all-zero (findings outside the 14); frontal vs lateral not labelled (Phase 1); GPU nondeterminism: same seed + same code gives slightly different results (two 3-epoch runs: val macro F1 0.014 vs 0.019, macro AUC 0.660 vs 0.681), `torch.backends.cudnn.deterministic = True` made two runs identical → decide in Step 1.3c (speed cost not measured yet)
