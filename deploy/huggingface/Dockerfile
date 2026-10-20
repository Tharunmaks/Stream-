# ExpertCloud on a free Hugging Face Space (Docker, CPU basic: 2 vCPU, 16 GB RAM).
# Create a Space with the Docker SDK and add this file and README.md. Settings > Variables and secrets: set
# EXPERTCLOUD_ADMIN_TOKEN (and EXPERTCLOUD_JOIN_TOKEN) so the tokens stay the same across restarts.
# To get newer code: Settings > Factory rebuild.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends clang make git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user
ARG BRANCH=claude/expertstream-moe-android-oegt59
RUN git clone --depth 1 -b $BRANCH https://github.com/tharunmaks/stream- app
WORKDIR /home/user/app
RUN make -j4 es_serve es_get
RUN pip install --no-cache-dir --user fastapi uvicorn websockets httpx psutil
EXPOSE 7860
CMD ["bash", "deploy/cloud_start.sh"]
