# IU X-Ray: Data Inventory (Step 0.1)

Most numbers are produced by `python -m src.data.inventory`; the rest come from `notebooks/01_eda.ipynb`.
Dataset: Indiana University Chest X-ray Collection (Open-i, U.S. National Library of Medicine),
licence CC BY-NC-ND 4.0. The data itself is not in this repository.

## 1. Summary

- **3,337 of 3,955 reports (84%) are usable** for report generation: they have Findings text *and* at least one image.
- The dataset is clean at the file level: no duplicate report IDs, no image shared between reports,
  no orphan images, no missing image files.
- Images were resized by the provider to a **fixed width of 512 px**; heights vary (362–873 px, 304 distinct sizes),
  so every image will need a resize strategy before training.
- **There is no patient ID** in the reports, and **frontal vs lateral views are not labelled**. Both need a decision later.

## 2. Reports

| | Count |
|---|---|
| XML report files | 3,955 |
| Report numbers | 1–3999, with 44 numbers missing (so 3,955, not 3,999) |
| Duplicate report IDs | 0 |

Every report has all four text sections (`COMPARISON`, `INDICATION`, `FINDINGS`, `IMPRESSION`) and a `MeSH` block.
A section can be present but **empty**:

| Section | Empty | % of reports |
|---|---|---|
| comparison | 622 | 15.7% |
| indication | 90 | 2.3% |
| **findings** | **530** | **13.4%** |
| impression | 34 | 0.9% |

**Text `"None."` is not the same as an empty section.** An empty XML element is Python `None` (pandas `NaN`).
The text `"None."` was written by the radiologist and means "no earlier X-ray to compare with".
1,402 comparisons are literally `"None."` or `"None"`, on top of the 622 empty ones.

**Images per report:**

| Images | Reports | % |
|---|---|---|
| 0 | 104 | 2.6% |
| 1 | 446 | 11.3% |
| 2 | 3,208 | 81.1% |
| 3 | 181 | 4.6% |
| 4 | 15 | 0.4% |
| 5 | 1 | 0.0% |
| **total images referenced** | **7,470** | |

**MeSH terms:** `<major>` terms (most likely assigned by people, structured as `finding/location/side`) are never empty.
1,391 reports (35%) are tagged only `normal`. `<automatic>` terms (most likely extracted by software) are empty in
1,773 reports, including all 1,391 `normal` ones; a normal report has no findings for software to extract.

## 3. Images

| | Value |
|---|---|
| Files in `images/` | 7,471 |
| PNG images | 7,470 |
| Non-PNG files | `Thumbs.db` |
| Mode | RGB for all 7,470 |
| Width | always 512 px |
| Height | min 362, median 512, mean 533, max 873 px |
| Distinct sizes | 304 |

- **`Thumbs.db`** is a hidden Windows thumbnail cache that was packaged with the dataset by accident.
  It is why images are always counted with `*.png`, never `*`. It stays in `data/raw/` (raw data is never modified).
- **RGB, but really grayscale:** in the images we checked, the three channels are identical. Storing one channel would
  use a third of the memory; a pretrained CNN expecting 3 channels can get the channel copied at load time.
- **Width is always exactly 512**, so the provider resized every image to that width while keeping its proportions.
  That is why only the height varies.
- **Most common heights:** 624 (1,983 images), 512 (1,528) and 420 (1,424).

**Shape:**

| Shape | Images | % |
|---|---|---|
| portrait (taller than wide) | 3,332 | 44.6% |
| landscape (wider than tall) | 2,610 | 34.9% |
| square (512 × 512) | 1,528 | 20.5% |

Shape is a **hint about the view, not a rule**. In a random sample of 6, all 3 portrait images were lateral (side) views
and 2 of the 3 landscape images were frontal, but `CXR751_IM-2305-2001` is landscape *and* lateral.
Square images give no hint at all. Example: report `CXR1` has a frontal image (`-4001`, 512×420)
and a lateral one (`-3001`, 512×624).

## 4. Linking reports and images

| Check | Result |
|---|---|
| Image IDs referenced by reports | 7,470 (all unique: no image belongs to two reports) |
| Orphan images (file, but no report mentions it) | 0 |
| Missing files (mentioned in a report, but no file) | 0 |

Unique images per report say nothing about **patients**: one patient could have several reports,
each with its own images.

## 5. Usable samples

A report is usable for report generation if it has Findings text **and** at least one image:

```
3,955 reports − 530 (no findings) − 104 (no images) + 16 (both problems) = 3,337
```

The 16 reports with both problems are subtracted twice by the first two terms, so they are added back once.
The same number comes from filtering the table directly.

**Possible fallback:** 502 of the 530 reports without Findings still have an Impression; 28 have neither.

## 6. What the report text looks like

From five random usable reports (seed 42: CXR1345, CXR74, CXR1774, CXR2443, CXR2706):

- **`XXXX` everywhere:** words removed for anonymisation, sometimes in the middle of medical phrases
  ("XXXX are normal", "multiple XXXX deformities").
- **Negations dominate:** radiologists mostly list what is *not* there ("No pneumonia, effusions, edema, pneumothorax…").
  A naive keyword search would tag these reports with exactly the diseases they rule out.
- **Standard phrases repeat:** "Heart size normal", "Lungs are clear", "No pleural effusion or pneumothorax".
  A model can produce fluent, plausible text by copying these phrases without looking at the image.

## 7. Open questions for later steps

| Question | Where it gets decided |
|---|---|
| No patient ID in the XML: what unit do we split by? | Step 0.2 |
| MeSH `major` vs `automatic`, and handling negations when building labels | Step 0.3 |
| Frontal vs lateral is not labelled; shape is only a hint | Phase 1 |
| Resizing rectangular images to 224×224: pad, crop or squash? | Phase 1 |
| Store 1 channel instead of 3 | Phase 1 |
| 530 reports without Findings: use Impression as a fallback? | Phase 3 |
| What to do with `XXXX` in the text | Phase 3 |
