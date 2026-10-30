# CXR Report Generation

Chest X-ray → radiology report: a CNN encoder feeding a 14-finding classification head and a text
decoder, built in PyTorch on the IU X-Ray dataset. A learning project.

## Metrics: macro vs micro

The classifier answers 14 yes/no questions per X-ray (the CheXpert categories).
`src/evaluation/metrics_cls.py` computes precision, recall, F1 and AUC for each category, then
averages them in two ways:

- **Macro** averages the 14 category scores: every category gets one vote.
- **Micro** adds up the TP / FP / FN counts of all categories first and computes one score: every
  case gets one vote.

With these labels the two tell different stories, because the categories are very uneven.
No Finding, Lung Opacity and Cardiomegaly hold 64% of all positive labels (2,503 of 3,919), while
Pneumothorax, Enlarged Cardiomediastinum and Consolidation have 27–30 reports each. Two fake models
scored on the 3,860 labeled reports show the effect:

| Fake model | Macro F1 | Micro F1 |
|---|---|---|
| Perfect on the 3 most common categories, never says yes to the other 11 | 0.21 | 0.78 |
| Perfect on the 11 rarer categories, never says yes to the 3 most common | 0.79 | 0.53 |

The first model never finds a pneumothorax, edema, pneumonia or fracture, yet micro F1 rates it
0.78. Micro mostly measures the common categories.

**Macro is the main number for this project.** The rare categories include the most dangerous
findings, and a model that ignores them must not look good. Micro is reported next to it as the
overall, case-level view.

**Macro's weakness is noise.** A category with few cases can swing its F1, and with it the macro
average, on a single case. Pneumothorax has 27 labeled reports and only 1 in the validation set.
So every per-category score is reported with its number of positives, and a change in macro is
checked against the per-category rows before it is trusted.
