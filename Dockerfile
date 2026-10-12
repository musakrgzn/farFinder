FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    QT_QPA_PLATFORM=xcb \
    QT_X11_NO_MITSHM=1 \
    FARFINDER_MODEL_PATH=/app/models/face_landmarker.task

# Libraries needed by the GUI-enabled OpenCV wheel, Qt xcb, and MediaPipe.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgl1 libglib2.0-0 libgomp1 libportaudio2 \
    libx11-6 libxext6 libxrender1 libsm6 libice6 \
    libxcb1 libxcb-xinerama0 libxcb-cursor0 libxcb-icccm4 \
    libxcb-image0 libxcb-keysyms1 libxcb-render-util0 \
    libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt \
    && python -m pip check

COPY scripts/download_model.py /app/scripts/download_model.py
RUN python /app/scripts/download_model.py

ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --gid "${APP_GID}" farfinder \
    && useradd --uid "${APP_UID}" --gid farfinder --no-create-home farfinder \
    && mkdir -p /app/data \
    && chown farfinder:farfinder /app/data

COPY farfinder ./farfinder
COPY tests ./tests
USER farfinder
CMD ["python", "-m", "farfinder"]
