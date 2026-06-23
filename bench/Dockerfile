# Run the Zero-TVM throughput bench on a cloud GPU instance — no Mac required.
#
#   docker build -f bench/Dockerfile -t zerotvm-bench .
#   docker run --gpus all --rm zerotvm-bench        # prints bench/results.json
#
# Requirements on the host:
#   - an NVIDIA GPU + the NVIDIA Container Toolkit (so the container sees the
#     driver's Vulkan ICD — that's what Chrome's WebGPU binds to)
#   - outbound network to huggingface.co (the ~2 GB Phi-3 weights)
#
# The container measures and prints results.json. Copy it into your repo and run
# `npm run bench:sync -- --write` (no GPU needed) to update BENCH.md + the docs.
#
# NOTE: a cloud NVIDIA GPU is a different baseline than the M2 Pro in BENCH.md,
# so absolute tok/s will differ. The Zero-TVM-vs-WebLLM gap is the portable part.
# This image is assembled from standard NVIDIA-Vulkan + Puppeteer patterns; it
# has not been run in the build sandbox (no GPU there), so treat the first run
# as the smoke test.

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Expose the driver's graphics/Vulkan capability to the container.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    NVIDIA_VISIBLE_DEVICES=all \
    DEBIAN_FRONTEND=noninteractive

# Node 22, the Vulkan loader + tools, Xvfb, Chrome's runtime libs, and the
# HuggingFace CLI (to prime the weight mirror Vite serves at /local-weights).
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg git xvfb python3-pip \
      libvulkan1 vulkan-tools \
      libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 \
      libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2 \
      libpango-1.0-0 libcairo2 libatspi2.0-0 fonts-liberation \
  && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
  && apt-get install -y --no-install-recommends nodejs \
  && pip3 install --no-cache-dir -U "huggingface_hub[cli]" \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci                       # deps + Chrome download (puppeteer postinstall)
COPY . .

# xvfb-run gives Chrome a virtual display; the NVIDIA GPU drives WebGPU.
ENTRYPOINT ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24"]
CMD ["bash", "bench/cloud-bench.sh"]
