# Blood Test Explainer — Remaining Work

**For:** Dimitris + agents  
**Repo:** `r0m4k/blood-test-explainer`  
**Space:** `build-small-hackathon/blood-test-explainer`  
**Last updated:** 2026-06-13  
**Suggested order:** 1 → 2 → 3 & 4 (parallel) → 5 → 6

---

## Status snapshot

| Area | Now |
|---|---|
| Space / app | Live on Transformers (`openbmb/MiniCPM-V-4.6`) |
| Knowledge graph | 107 markers in `kb/cbc_knowledge_graph.json` |
| Marker videos | All 107 have `video_url`; ~44 unique YouTube IDs (many reused) |
| Real eval labels | 2/13 reports fully labeled in `eval/data/real/labels.jsonl` |
| Fine-tune pipeline | `train/modal_finetune.py` → merge → Hub push |
| Article / demo video | Not started |

---

## 1. Insert the custom model

**Owner:** Dimitris (Modal + HF Space vars)

- [ ] Confirm fine-tuned Transformers repo on Hub (e.g. `dimitriskalligaridis/blood-test-minicpmv-4_6`) loads with `transformers[torch]==5.7.0`
- [ ] If not published yet: finish labeling → `modal run train/modal_finetune.py::main --real-labels eval/data/real/labels_train.jsonl` → `modal run train/modal_finetune.py::merge --repo-id <owner>/<name>`
- [ ] Set HF Space variables:
  ```bash
  EXTRACTOR_BACKEND=transformers
  ZEROGPU_MODEL_ID=<fine-tuned-repo>
  ```
- [ ] Rebuild Space; test 2–3 PDFs from `eval/data/real/`
- [ ] Run before/after eval: `modal run train/modal_eval.py::compare --finetuned-id <repo>` → save `eval/before_after.json`
- [ ] *(Optional, Llama badge only)* GGUF via `scripts/convert_to_gguf.sh` + `LLAMACPP_VISION=1` vars (see `README.md`)

**Done when:** Space uses custom model; we have a before/after metric for the article.

---

## 2. Fine-tune app wording

**Owner:** Dimitris or copy agent

**Edit:** `app.py` (hero, upload hints, status, disclaimers), `src/pipeline_trace.py` (step copy), `README.md` (Space card)

- [ ] One clear pitch: upload → extract → explain → prepare for clinician conversation
- [ ] Badge claims match reality (Well-Tuned only after custom model is live)
- [ ] Consistent “educational, not diagnosis” disclaimer
- [ ] Less dev jargon in user-facing text (“pipeline phase”, etc.)
- [ ] Align hero badges with hackathon criteria (OpenBMB, Modal, HF, off-grid)

**Done when:** Hero + upload + report readable in under 60 seconds.

---

## 3. Enlarge the knowledge graph

**Owner:** Agent task (Dimitris to review)

**Tools:** `src/markers.py`, `kb/knowledge_base.py`, `scripts/expand_lab_knowledge_graph.py`, `kb/cbc_knowledge_graph.json`

- [ ] Expand canonical markers in `src/markers.py` (target: 150–200 common lab markers)
- [ ] For each marker: description, importance, food/exercise/supplement guidance, age/sex stats (cite MedlinePlus / `kb/references/`)
- [ ] Add IDs to `MARKER_IDS` in `scripts/expand_lab_knowledge_graph.py`
- [ ] Run `python scripts/expand_lab_knowledge_graph.py`
- [ ] Run `pytest tests/test_report_pipeline.py`
- [ ] Spot-check 10 markers in UI after a real PDF upload

**Done when:** KG covers target marker list; multi-panel PDFs enrich correctly.

---

## 4. Marker video review (per marker)

**Owner:** Agent task (Dimitris to review)

**Tools:** `kb/marker_videos.json`, `scripts/expand_lab_knowledge_graph.py`, `app.py` (`_youtube_embed_html`)

- [ ] Replace generic reused YouTube URLs with marker- or category-specific explainers
- [ ] Prefer: MedlinePlus, NHS, Cleveland Clinic, Osmosis-style education
- [ ] Avoid: treatment promises, irrelevant content
- [ ] Use category fallback when no single-marker video exists (CBC, liver, lipids, thyroid, etc.)
- [ ] Regenerate graph; QA embeds on high / low / normal marker cards

**Done when:** ≥80% markers have unique or category-specific videos; no empty `video_url`.

---

## 5. Create an article

**Owner:** Dimitris (+ Roman review)

**Publish to:** HF blog / Devpost / LinkedIn (pick one primary)

- [ ] Problem → approach (vision extract + deterministic KB, not LLM medical facts)
- [ ] Fine-tune story + before/after numbers from `eval/before_after.json`
- [ ] Architecture: Gradio + ZeroGPU, no hosted API
- [ ] 2 screenshots + Space link
- [ ] Limitations + disclaimer
- [ ] Links: Space, model repo, GitHub

**Blocked by:** #1 (custom model live), #2 (copy pass), metrics from eval.

---

## 6. Demo video (Laytimely-style)

**Owner:** Dimitris

- [ ] Script (~400–600 words): hook → upload → trace → report → one marker → disclaimer
- [ ] AI voiceover (same stack as Laytimely)
- [ ] Screen record Space or local app; strong PDF (`02_cbc_umc_johndoe.pdf` or `06_drlogy_cbc.pdf`)
- [ ] Show trace hover, marker card, embedded YouTube
- [ ] Royalty-free background music under voice (−18 to −24 dB)
- [ ] Captions + title/end cards with Space URL
- [ ] Publish (YouTube unlisted or HF README embed); link in article + submission

**Blocked by:** #1, #2, ideally #3/#4 so demo looks polished.

---

## Submission checklist

- [ ] Custom model on Space (`ZEROGPU_MODEL_ID`)
- [ ] Before/after eval documented
- [ ] Copy + badges accurate
- [ ] KG + videos polished
- [ ] Article published
- [ ] Demo video with AI voice + music
- [ ] README / Space card matches final story

---

## Key paths

| Path | Purpose |
|---|---|
| `train/modal_finetune.py` | LoRA train + merge + Hub push |
| `train/modal_eval.py` | Base vs fine-tuned comparison |
| `eval/data/real/` | Real PDFs + labels |
| `scripts/expand_lab_knowledge_graph.py` | Regenerate KB JSON |
| `kb/marker_videos.json` | Video catalog |
| `README.md`, `RUNBOOK.md`, `DEPLOY.md` | Deployment + llama.cpp docs |

## Agent notes

- Default extraction: `EXTRACTOR_BACKEND=transformers` — do not change unless badge work requires llama.cpp.
- Do not commit model weights, tokens, or PHI.
- Push to `origin` (GitHub) and `space` (HF) after merged changes on `main`.
- Workflow details: `RUNBOOK.md`
