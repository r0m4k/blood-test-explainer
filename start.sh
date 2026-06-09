#!/usr/bin/env bash
# Launch the on-device model server, wait for it, then start the Gradio UI.
# Everything stays on localhost — no external network — so this is fully off-grid.
set -euo pipefail

echo "Starting llama-server (on-device MiniCPM-V)..."
llama-server \
  -m "${LOCAL_MODEL_PATH}" \
  --mmproj "${LOCAL_MMPROJ_PATH}" \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 4096 &

# Wait until the model is loaded and the server answers /health (model load can take a while).
echo "Waiting for llama-server to be ready..."
for _ in $(seq 1 240); do
  if curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "llama-server is ready."
    break
  fi
  sleep 2
done

echo "Starting Gradio app on port ${GRADIO_SERVER_PORT:-7860}..."
exec python app.py
