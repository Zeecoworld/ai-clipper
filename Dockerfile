FROM python:3.11-slim

ARG WHISPER_MODEL=base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    WHISPER_MODEL=${WHISPER_MODEL} \
    WHISPER_CACHE=/opt/whisper-models \
    AUTOCLIP_WORK_DIR=/tmp/autoclip

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg fonts-dejavu-core fontconfig curl \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the speech model into the image so cold starts don't download it.
RUN mkdir -p ${WHISPER_CACHE} && python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8', download_root='${WHISPER_CACHE}')"

COPY . .

RUN useradd -m appuser && mkdir -p /tmp/autoclip && chown -R appuser /app /tmp/autoclip ${WHISPER_CACHE}
USER appuser

EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD curl -fs http://localhost:${PORT:-10000}/healthz || exit 1

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-10000}"]
