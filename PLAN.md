# Blood Test Explainer ("Pulse") — Plan, revised against current repo

> Grounded in the actual code as of 2026-06-09. Two developers, parallel.
> Target prizes: **OpenAI + OpenBMB + Modal** + all badges.
> ⚠️ Confirm the real submission deadline first — teams are already final-submitting.

> 2026-06-10 deployment update: the active HF Space path is now **Gradio ZeroGPU + official
> OpenBMB Transformers model**. The Docker/llama.cpp serving plan was replaced because ZeroGPU is
> Gradio-only and the Docker build was OOM-killed on free CPU hardware. For the fine-tuned model,
> keep this ZeroGPU architecture and replace only `ZEROGPU_MODEL_ID`.

---

## 1. Where we are (DONE — solid foundation)

- **Polished Gradio app** (`app.py`): light clinical theme, animated "formation" hero,
  loading/empty states, report cards with status pills, tabs (Report / Values / JSON / Raw),
  responsive. This is genuinely good and is a real head start on the "premium document" wow.
- **Extraction works** (`src/openbmb_client.py`, `src/document_processing.py`): OpenBMB
  **MiniCPM-V-4.6** (vision) via OpenAI-compatible API. PDF→images (PyMuPDF), images, and
  text files supported; base64 image payloads; robust JSON-repair parsing.
- **Schema:** `{marker, value, unit, reference_range, status, source_text, confidence}` + notes.
- **Env/secret handling** ready (`.env` local, Space secret). Codex is contributing.
- Extraction prompt is deliberately **extraction-only** ("do not diagnose/interpret").

## 2. Blunt gap analysis — what's missing and which prize it unblocks

The current app is a beautiful **extractor that calls an external API**. As-is it does not win.

| Missing piece | Why it matters | Unblocks |
|---|---|---|
| **Interpretation + cross-marker reasoning** | Right now MiniCPM looks like OCR/plumbing. The model must *reason* (e.g. "ALT+AST+GGT all high → liver-enzyme pattern") to be visibly central. | **OpenBMB** "central model" + req #5 |
| **Cited knowledge base (~40 markers)** | Reliable reference ranges + "what it measures" + "questions for your doctor", grounded not hallucinated. | medical credibility |
| **Run the model LOCALLY (llama.cpp)** | The current external API call **fails off-grid / "fully offline."** | **off-grid badge** + offline requirement |
| **Fine-tune + GGUF on Modal** | No fine-tune yet. Needed for the badge + the OpenBMB before/after story. | **fine-tune + quantization badges, Modal prize** |
| **Agent trace** | Single API call now; needs a visible multi-step pipeline. | req #5 |
| **Eval harness + before/after** | No metrics yet. | **OpenBMB** proof |
| **Traces dataset + model card + multi-repo** | Top teams (compliment-forest) do this. | competitiveness |
| **Video (local run) + social + README lines** | Required submission artifacts. | **general pool** |

## 3. The biggest technical fork: getting OFF the external API

Off-grid + "fully offline" require the model to **run inside the Space**, no external calls.
The README already flags this ("API-backed extractor is temporary"). Options:

- **Recommended (hybrid):** keep MiniCPM-V for extraction but run it **locally in the Space**
  (MiniCPM-V GGUF via llama.cpp multimodal, or via transformers on **ZeroGPU**), and **fine-tune
  a small MiniCPM *text* model** for the **interpretation + cross-marker reasoning** layer (easier
  to fine-tune, this is the "model is the star" part, runs offline via llama.cpp). Earns off-grid
  + fine-tune + quantization together.
- **Simpler fallback:** drop vision; do **local OCR/text extraction** (PyMuPDF text + tesseract)
  → one fine-tuned MiniCPM text model does structuring **and** interpretation. Fully offline,
  easiest fine-tune; weaker on scans/photos.
- **Decide in Phase B.** Main technical risk = running MiniCPM-V locally; the fallback de-risks it.

## 4. Parallel plan from here (2 devs, shared stack, contract-first)

**Shared contract to agree first (extend the existing dataclass):** add `Interpretation`
(`marker, plain, meaning, questions[], citation`) and a `cross_marker: list[str]` + `summary`
to the result object. Both tracks build against it; Track B can use a fixture until Track A ships.

**Module ownership (avoid collisions):**
- **Track A (model/data):** `src/openbmb_client.py` (+ interpretation), new `src/kb/`,
  `src/reasoning/`, `train/`, `eval/`. Owns: extraction-offline, fine-tune, KB, reasoning, eval.
- **Track B (product/UI):** `app.py`, report rendering, **interpretation cards**, **cross-marker
  insight section**, **doctor-questions**, **agent-trace panel**, `.html` download, sample report, deploy.
- **Co-owned:** the result dataclass (the contract).

### Phase A — DONE ✓ (extraction MVP + UI shell)

### Phase B — Interpretation + KB + reasoning (the win-maker)
- **Track A:** build the **cited KB** (~40 markers: CBC, metabolic, lipid, thyroid, key vitamins);
  add an **interpretation pass** (grounded in KB + the user's value) and the **cross-marker
  reasoning** (W1). Decide the offline path (§3).
- **Track B:** extend the report to render interpretation per marker + a **cross-marker insights**
  block + **"questions for your doctor"**; add the **agent-trace** panel (Ingest → extract →
  normalize → KB lookup → reason → render) streaming via generator `yield`.
- **Milestone:** real report → extracted + explained + reasoned document.

### Phase C — Offline + fine-tune (the badges)
- **Track A:** synthetic data (Claude) for extraction/interpretation; **LoRA fine-tune on Modal**
  → merge → **GGUF Q4_K_M**; wire local **llama.cpp** inference; **before/after chart**.
- **Track B:** deploy to the **org Space** with the local model; verify **zero external calls**;
  graceful failure; downloadable `.html`; bundle a **sample report**.
- **Milestone:** runs fully offline on a stranger's report; fine-tune metrics in hand.

### Phase D — Robustness + artifacts
- International units (mg/dL↔mmol/L), messy formats, unknown markers.
- Publish **agent traces as a HF dataset** + **model card with GGUF/eval metrics**; multi-repo.

### Phase E — Submission
- Record **2-min video locally** (so "nothing left my laptop" is literally true); **social post**;
  README with every eligibility line; flip repo public if needed; final eval numbers. Freeze + submit.

## 5. The two weaknesses we engineer around
- **W1 — model must be the star:** cross-marker reasoning is a first-class component, not an add-on.
- **W2 — medical credibility:** every fact from a **cited KB**; "questions for your doctor," never
  diagnosis (the current prompt's restraint is the right instinct — keep it).
- Privacy is only honest run locally → record the **video locally**; the Space ships a sample report.

## 6. Submission checklist
- [ ] Gradio app, HF Space under the **org**.
- [ ] All models <32B; the model <4B; **fine-tuned MiniCPM declared central**.
- [ ] **Runs fully offline** (no external API — the current OpenBMB call must be replaced) under **llama.cpp** (GGUF).
- [ ] **Demo video** (local) + **social post** linked in README.
- [ ] **Public repo** + dense **Codex-attributed commits**.
- [ ] README states: MiniCPM central, **Modal for fine-tuning**, off-grid, fine-tuned, quantized.
- [ ] Before/after chart + **traces dataset** + **model card** published.
- [ ] Space runs without our hardware (sample report bundled).

## 7. Prize map (one line each)
- **OpenAI $10K** — build via Codex (already a contributor); public repo.
- **OpenBMB $10K** — MiniCPM central (extraction **+ reasoning**); before/after chart.
- **Modal $20K credits** — LoRA fine-tune (+ eval/data-gen) on Modal.
- **General pool** — Gradio Space + premium document + video + post.
- **Badges** — off-grid (local model, no API), fine-tune (LoRA), quantization (GGUF Q4_K_M).
