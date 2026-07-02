FROM python:3.11-slim

# System libs:
#   libportaudio2 / libsndfile1 — sounddevice + audio file I/O
#   libgomp1                    — runtime for scipy/matplotlib
#   libpulse0 + libasound2-plugins — bridge so PortAudio -> ALSA -> PulseAudio,
#       letting audio reach the WSLg PulseServer (host speakers) when the
#       socket is mounted with `run-sandbox.sh --audio`.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libportaudio2 \
        libsndfile1 \
        libgomp1 \
        libpulse0 \
        libasound2-plugins \
    && rm -rf /var/lib/apt/lists/*

# Route ALSA's default device through the PulseAudio plugin, so PortAudio
# (used by sounddevice) plays to PULSE_SERVER when one is configured.
RUN printf 'pcm.!default { type pulse }\nctl.!default { type pulse }\n' > /etc/asound.conf

# Claude Code CLI — installed via the native binary installer (no Node.js).
# Drops a self-contained `claude` binary under /root/.local/bin. curl + CA
# certs are needed to fetch it at build time and to reach the API at runtime.
# `bash -s latest` pins the latest release channel (the installer defaults to
# stable). CLAUDE_INSTALL_DATE is injected by run-sandbox.sh as today's date so
# this layer is re-run (at most) once per calendar day, keeping the binary current.
ARG CLAUDE_INSTALL_DATE=unknown
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        git \
    && curl -fsSL https://claude.ai/install.sh | bash -s latest \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:${PATH}"

# Headless matplotlib by default (no display inside the sandbox).
ENV MPLBACKEND=Agg

WORKDIR /workspace

# Install Python deps in a layer that is cached independently of the source,
# so editing project files does not trigger a reinstall.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Source is mounted at runtime (see run-sandbox.sh), not baked in.
CMD ["bash"]
