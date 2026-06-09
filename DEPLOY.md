# Deploying the offline Space (Docker)

The submission Space runs the model **on-device** via a Docker image: it builds current
llama.cpp, **bakes the MiniCPM-V GGUF into the image at build time**, and launches
`llama-server` + the Gradio app. No external API at runtime → **off-grid**.

## 1. Make the Space a Docker Space (under the org)
The Space `build-small-hackathon/blood-test-explainer` must use **`sdk: docker`**. Set its
`README.md` frontmatter to exactly:

```yaml
---
title: Blood Test Explainer
emoji: 📊
colorFrom: green
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
```

## 2. Get these files into the Space repo
The Space needs `Dockerfile`, `start.sh`, `.dockerignore`, `app.py`, and `src/` (plus the
sample reports under `eval/data/real/`). If your Space mirrors this GitHub repo, push this
branch's contents. If the Space is a separate HF git repo, copy these files into it and push.

## 3. HF builds it
First build takes ~10–15 min (compiles llama.cpp + downloads the ~1.6 GB model). When it's
up, upload a report → it extracts fully offline.

## 4. Ship the fine-tuned model (later)
Once you've fine-tuned and converted to GGUF, upload it to an HF model repo, then rebuild the
Space with the build-args pointing at it:

```
MODEL_REPO=<you>/minicpmv-lab-gguf  MODEL_FILE=<your-model>.Q4_K_M.gguf  MMPROJ_FILE=mmproj-model-f16.gguf
```
(In a Space, set these as build-time variables, or edit the `ARG` defaults in the Dockerfile.)

## 5. Verify off-grid (badge)
- Model is baked into the image at build time; `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`.
- The only network at runtime is `127.0.0.1:8080` (llama-server in the same container).
- Grep check: nothing on the request path hits an external host — the API backend (the only
  outbound HTTP) is disabled when `EXTRACTOR_BACKEND=local`.

## Hardware
Free **CPU** Space works (first inference is slower than your M3, ~30–60s). Upgrade the Space
hardware if you want it snappier for live demos; the demo **video** should be recorded locally
where it's fast.
