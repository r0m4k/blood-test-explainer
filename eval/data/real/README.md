# Real lab-report eval set

Hand-collected, publicly available **sample** lab reports (fake patients, no PHI), used to
measure extraction accuracy on real-world formats — the credibility number for the OpenBMB
before/after. These are for **evaluation** (and demo), not the primary training source: the
model trains on the synthetic generator (`train/synth_reports.py`); these tell us how well it
generalizes to messy real layouts.

## Files (13 reports, varied labs/countries/panels)
| File | Lab / format | Notes |
|---|---|---|
| 01_sterling_accuris.pdf | Sterling (IN), 19 pages | huge multi-panel; first pages = CBC |
| 02_cbc_umc_johndoe.pdf | US CBC, 1 page | **labeled** ✓ (macrocytic-anemia picture) |
| 03_lpl_wm17s.pdf | Dr Lal PathLabs (IN), 7p | multi-panel |
| 04_pk0016_urine.pdf | urine + chem, 13p | |
| 05_gribbles_cbm.pdf | Gribbles (AU/MY), 3p | |
| 06_drlogy_cbc.pdf | Drlogy CBC, 1 page | **labeled** ✓ |
| 07_sample_505271.pdf | US, 2p | |
| 08_investigation_scanned.pdf | **image-only / scanned** | no text layer → good *vision*-modality test |
| 09_pathkind_pl02.pdf | Pathkind (IN), 8p | |
| 11_functionaldx.pdf | FunctionalDX, 24p | interpretive; heavy |
| 12_lpl_k017.pdf | LPL (IN), 2p | |
| 15_gribbles_crp.pdf | Gribbles, 9p | pediatric |
| 16_zrt_female_hormones.pdf | ZRT hormones, 5p | estradiol/progesterone/cortisol etc. |

Removed: a veterinary (dog) report — wrong domain. Reference tables (PSAP, NBME) moved to
`kb/references/` (they're KB material, not reports — see below).

## Labels
`labels.jsonl` — one row per report: `{"image": "<file>.pdf", "tests": [...], "notes": []}`.
The extractor reads PDFs directly (it renders pages to images), so `image` points at the PDF.
**Currently labeled: 02 and 06** (full, verified). The rest are best labeled via bootstrap:

```bash
# 1) draft labels with the current extractor, then hand-correct into gold
EXTRACTOR_BACKEND=transformers python eval/run_eval.py \
    --labels eval/data/real/labels.jsonl --run
```
To add a report: run the extractor on it, copy the predicted `tests` into a new `labels.jsonl`
row, and correct any mistakes against the PDF. Faster and more accurate than typing from scratch.

## Run the eval
```bash
python eval/run_eval.py --labels eval/data/real/labels.jsonl --run
```
Use it twice (base vs fine-tuned GGUF) for the before/after.
