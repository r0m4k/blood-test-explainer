# Message to Dimitris (copy-paste)

Hi Dimitris — here’s what’s left before submission. Full checklist for you and your agents: **`docs/REMAINING_WORK.md`**.

**Priority order:** custom model → copy pass → KB + videos (agents) → article → demo video.

1. **Custom model** — Fine-tuned repo is live on Hub and wired as the code default; confirm Space after redeploy, run `modal run train/modal_eval.py::compare` for before/after numbers.
2. **Copy** — Tighten hero, trace steps, and README; fix badge claims (Well-Tuned only after model swap).
3. **Knowledge graph** — Expand beyond 107 markers via `markers.py` + `expand_lab_knowledge_graph.py` (agent task).
4. **Videos** — Replace reused YouTube URLs with marker/category-specific explainers (agent task).
5. **Article** — Problem → architecture → fine-tune proof → limitations + Space link.
6. **Demo video** — Laytimely-style: AI voice + screen recording + background music; record after 1–2 are done.

**Current baseline:** App defaults to `build-small-hackathon/blood-test-minicpmv-4_6-medreason`; KG has 107 markers; only 2/13 real eval reports labeled; article and demo not started.

Ping me when the model is on the Space or if you want me on copy/KB review.
