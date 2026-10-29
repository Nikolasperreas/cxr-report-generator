# IU X-Ray: Labels from MeSH (Step 0.3)

Category counts are produced by `python -m src.data.labels`, which also writes `data/interim/labels.csv`
(not in this repository; rerun to recreate it). The per-split table and the term counts come from
`notebooks/03_labels.ipynb`. The mapping itself is `configs/label_map.yaml`.

## 1. Summary

- Every report gets **14 yes/no labels**, the CheXpert categories, built from its human-assigned MeSH `major` terms.
- **3,860 of 3,955 reports are labeled.** The other 95 are tagged `No Indexing` (never annotated); they keep a row
  with `labeled = 0`, meaning "unknown", not "no findings".
- **The labels are very uneven:** `No Finding` 36%, `Lung Opacity` 18.6%, and four categories under 50 reports
  each (0.7–1.1%).
- **1,007 labeled reports (26%) are abnormal but match none of the 14**, so their label row is all zeros.
- `normal` is 36% of labeled reports, not the 60–70% the roadmap expected.

## 2. Source: MeSH `major` terms

| | `mesh_major` (used) | `mesh_automatic` |
|---|---|---|
| Made by | human annotators | software that scanned the text |
| Empty lists | 0 of 3,955 | 1,773 of 3,955 |
| Looks like | `Cardiomegaly/mild`, `Pulmonary Atelectasis/base/left` | `emphysemas`, `scarring`, `Emphysema` |

- **Structure:** `Finding/location/severity`, e.g. `Pleural Effusion/bilateral/small`. There are 1,719 distinct terms
  but only 122 distinct first parts ("heads").
- **The head is not always the finding.** Some heads are organs and the finding is in a later part:
  `Lung/hypoinflation` (245) and `Lung/hyperdistention` (164) are opposites; `Cardiac Shadow/enlarged` is cardiomegaly;
  `Costophrenic Angle/blunted` is a sign of pleural effusion.
- **The text is not clean:** `Thoracic vertebrae` next to `Thoracic Vertebrae`, `Subcutaneous  Emphysema` with two
  spaces, and trailing spaces. Terms and patterns are lowercased and their spaces normalised before comparing.
- **No negations:** MeSH terms only list what is present, so "no effusion" problems do not arise here.
  They will for the text labeler in Phase 4.
- `normal` never appears together with another term.

## 3. Categories and rulebook

**Why the CheXpert 14.** Two alternatives were measured with draft mappings:

| | CheXpert 14 (chosen) | 14 most common IU groups |
|---|---|---|
| Smallest category | 27 reports | 126 reports |
| Abnormal reports left all-zero | ~1,000 | ~220 |
| Keeps rare, dangerous findings (pneumothorax, edema, pneumonia, fracture) | yes | no |
| Same categories as the Phase 4 clinical F1 labeler and published work | yes | no |

The roadmap says not to drop rare categories to make the problem cleaner, so the CheXpert set was chosen.
Adding extra IU categories later (granuloma, degenerative spine…) only means adding entries to the rulebook.

**Matching rule.** A pattern `Head` matches every term with that head (`Cardiomegaly` matches `Cardiomegaly/mild`).
A pattern `Head/word` also needs `word` somewhere after the head (`Costophrenic Angle/blunted` matches
`Costophrenic Angle/right/blunted/mild`). Heads are compared for equality, never as substrings, so the pattern
`Fibrosis` does not match `Cystic Fibrosis`. One term can turn on two categories: `Hydropneumothorax` (air and fluid)
counts as both Pneumothorax and Pleural Effusion.

**Judgment calls:**

| Terms | Decision | Reason |
|---|---|---|
| `Calcified Granuloma`, `Granulomatous Disease`, `Granuloma` | **not** Lung Lesion | healed, harmless spots; including them would turn Lung Lesion (133) into ~500 mostly harmless cases |
| `Pulmonary Congestion` | Edema | overfilled lung vessels, usually the stage before edema |
| `Costophrenic Angle/blunted`, `Sulcus/blunted`, `Pleura/blunted` | Pleural Effusion | the classic X-ray sign of a small effusion |
| `Markings/interstitial`, `Lung Diseases, Interstitial`, `Pulmonary Fibrosis`, `Fibrosis` | Lung Opacity | streaky, whiter lung; `Markings/bronchovascular` (normal vessels) excluded |
| `Cysts` | only `Cysts/lung` → Lung Lesion | 2 of the 5 cyst terms are bone cysts in the upper arm |

## 4. Unlabeled and all-zero reports

- **`No Indexing` (95 reports, 92 with images):** always the only term. Several read as normal
  ("Both lungs are clear and expanded… No active disease"), so all zeros would be a wrong label. They get
  `labeled = 0`; the classifier should use only `labeled == 1`, while the report generator can still use their text.
- **`Technical Quality of Image Unsatisfactory`:** 80 of its 86 reports also have real findings and are labeled
  normally; the 6 where it is the only term end up all zeros. Too few to matter.
- **1,007 all-zero labeled reports:** the findings are real but outside the 14. Most common heads among them:
  `Lung` (hypo/hyperinflation, 246), `Calcified Granuloma` (185), `Thoracic Vertebrae` (degenerative, 155),
  `Calcinosis` (133), `Spine` (96), `Scoliosis` (61), `Granulomatous Disease` (61), `Aorta` (tortuous, 61).
  For the classifier these images look like "not normal, but none of the 14".

## 5. Frequency per category (3,860 labeled reports)

| Category | Reports | % |
|---|---|---|
| No Finding | 1,391 | 36.0 |
| Lung Opacity | 717 | 18.6 |
| Cardiomegaly | 395 | 10.2 |
| Support Devices | 379 | 9.8 |
| Atelectasis | 332 | 8.6 |
| Pleural Effusion | 204 | 5.3 |
| Lung Lesion | 133 | 3.4 |
| Edema | 100 | 2.6 |
| Fracture | 84 | 2.2 |
| Pleural Other | 58 | 1.5 |
| Pneumonia | 42 | 1.1 |
| Consolidation | 30 | 0.8 |
| Enlarged Cardiomediastinum | 27 | 0.7 |
| Pneumothorax | 27 | 0.7 |

## 6. Frequency per split

Labeled reports in the Step 0.2 split (reports with images only): train 2,634, val 565, test 560
(unlabeled: 61 / 12 / 19).

| Category | Train | Val | Test | Train % | Val % | Test % |
|---|---|---|---|---|---|---|
| No Finding | 956 | 200 | 223 | 36.3 | 35.4 | 39.8 |
| Lung Opacity | 487 | 114 | 83 | 18.5 | 20.2 | 14.8 |
| Cardiomegaly | 269 | 44 | 51 | 10.2 | 7.8 | 9.1 |
| Atelectasis | 220 | 49 | 46 | 8.4 | 8.7 | 8.2 |
| Support Devices | 219 | 55 | 47 | 8.3 | 9.7 | 8.4 |
| Pleural Effusion | 143 | 28 | 21 | 5.4 | 5.0 | 3.8 |
| Lung Lesion | 89 | 21 | 16 | 3.4 | 3.7 | 2.9 |
| Edema | 69 | 13 | 10 | 2.6 | 2.3 | 1.8 |
| Fracture | 60 | 11 | 12 | 2.3 | 1.9 | 2.1 |
| Pleural Other | 35 | 9 | 10 | 1.3 | 1.6 | 1.8 |
| Pneumonia | 27 | 8 | 5 | 1.0 | 1.4 | 0.9 |
| Pneumothorax | 20 | **1** | 5 | 0.8 | 0.2 | 0.9 |
| Consolidation | 19 | 8 | **3** | 0.7 | 1.4 | 0.5 |
| Enlarged Cardiomediastinum | 16 | 6 | 4 | 0.6 | 1.1 | 0.7 |

- Every category has at least one positive in every split, but **Pneumothorax has a single positive in val** and
  Consolidation only 3 in test. Per-class scores for these will be close to meaningless, and tuning a decision
  threshold for Pneumothorax on val (Phase 1.6) is not possible with one example.
- The common categories are spread reasonably. The largest gap, Lung Opacity at 14.8% in test vs 18.5% in train,
  is about 2 standard errors; among 14 categories, one gap of that size is expected by chance.

## 7. Open questions

| Question | Where it gets decided |
|---|---|
| Rare categories are spread unevenly by the random split (Pneumothorax: 1 in val). Keep it, or re-split stratified by label? | before Phase 1 |
| Add frequent IU findings outside the 14 (granuloma, degenerative spine, hyperinflation) as extra categories? | after Phase 1 error analysis |
| Rare categories (27–42 reports) will give noisy per-class metrics: how to report them | Steps 0.4 and 1.6 |
| Text labeler for the clinical F1 must handle negations ("no pneumothorax") | Phase 4 |
