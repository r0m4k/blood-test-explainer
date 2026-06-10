# Offline Blood Test Explainer — MiniCPM-V 4.6 served on-device by llama.cpp, no external APIs.
# HF Docker Space. The model is baked into the image at build time, so there is zero network
# at runtime (off-grid badge).
#
# Do not change this workflow. Keep Docker + build-time model download + local llama-server.
# When the fine-tuned model is ready, only replace MODEL_REPO / MODEL_FILE / MMPROJ_FILE below.
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 1) Build current llama.cpp (its server supports MiniCPM-V 4.6). GGML_NATIVE=OFF for portability.
WORKDIR /opt
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git && \
    cmake -S llama.cpp -B llama.cpp/build \
        -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF && \
    cmake --build llama.cpp/build --config Release -j --target llama-server && \
    cp llama.cpp/build/bin/llama-server /usr/local/bin/llama-server && \
    rm -rf /opt/llama.cpp

# 2) Bake the model + vision projector into the image (no runtime download).
#    Override these build-args to ship your fine-tuned model instead of the base.
ARG MODEL_REPO=openbmb/MiniCPM-V-4.6-gguf
ARG MODEL_FILE=MiniCPM-V-4_6-Q4_K_M.gguf
ARG MMPROJ_FILE=mmproj-model-f16.gguf
RUN pip install --no-cache-dir "huggingface_hub[cli]" && \
    mkdir -p /models && \
    hf download "$MODEL_REPO" "$MODEL_FILE" "$MMPROJ_FILE" --local-dir /models

# 3) App + runtime deps (no llama-cpp-python: we use the llama-server backend).
WORKDIR /app
RUN pip install --no-cache-dir \
        gradio==6.17.3 requests==2.32.5 pillow==12.0.0 pymupdf==1.26.6 json-repair==0.60.1
COPY . /app

ENV EXTRACTOR_BACKEND=local \
    LLAMA_SERVER_URL=http://127.0.0.1:8080/v1/chat/completions \
    LLAMA_SERVER_MODEL=minicpm-v \
    LOCAL_MODEL_PATH=/models/MiniCPM-V-4_6-Q4_K_M.gguf \
    LOCAL_MMPROJ_PATH=/models/mmproj-model-f16.gguf \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 7860
CMD ["bash", "start.sh"]
