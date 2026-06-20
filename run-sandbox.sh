#!/usr/bin/env bash
#
# run-sandbox.sh — build and enter a Docker sandbox for this project.
#
# The container can ONLY see the contents of this folder: the directory holding
# this script is bind-mounted at /workspace and nothing else from the host is
# exposed. Capabilities are dropped, privilege escalation is blocked, and the
# network is disabled by default.
#
# Usage:
#   ./run-sandbox.sh                 # interactive bash shell in the sandbox
#   ./run-sandbox.sh python irrational.py
#   ./run-sandbox.sh --audio         # shell with host-speaker audio (WSLg PulseAudio)
#   ./run-sandbox.sh --ui            # Gradio UI (network + port 7860 + audio)
#   NETWORK=1 ./run-sandbox.sh ...   # enable networking without the UI helper
#   AUDIO=1   ./run-sandbox.sh ...   # enable host audio without the --audio flag
#
# Secrets: the OpenAI key lives OUTSIDE this folder (so it's never on the
# sandbox mount). If OPENAI_API_KEY isn't already exported, it's read from
# ~/.config/irrational/openai_api_key and passed in as an env var — but only
# when networking is enabled, since that's the only time it could be used or
# leak anywhere.
#
set -euo pipefail

# Absolute path to the folder containing this script — the only host path the
# container is allowed to touch.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="irrational-sandbox"
KEY_FILE="${OPENAI_API_KEY_FILE:-$HOME/.config/irrational/openai_api_key}"
ANTHROPIC_KEY_FILE="${ANTHROPIC_API_KEY_FILE:-$HOME/.config/irrational/anthropic_api_key}"

cd "$PROJECT_DIR"

# Build (cached after the first run; only re-installs deps if requirements.txt changes).
docker build -t "$IMAGE" "$PROJECT_DIR"

# --- parse convenience flags ----------------------------------------------
CMD=("$@")
case "${1:-}" in
  --ui)
    NETWORK=1; PUBLISH_UI=1; AUDIO=1
    CMD=(python app.py)
    ;;
  --audio)
    AUDIO=1
    CMD=("${@:2}")   # drop --audio; remaining args (if any) become the command
    ;;
  --claude)
    NETWORK=1        # Claude Code needs to reach api.anthropic.com
    CMD=("${@:2}")   # drop --claude; default below launches the `claude` REPL
    if [[ ${#CMD[@]} -eq 0 ]]; then CMD=(claude); fi
    ;;
esac

# --- assemble run options -------------------------------------------------
RUN_OPTS=(
  --rm -it
  --mount "type=bind,source=$PROJECT_DIR,target=/workspace"
  --workdir /workspace
  --cap-drop ALL
  --security-opt no-new-privileges
)

# Network: off by default for isolation.
if [[ "${NETWORK:-0}" == "1" ]]; then
  :  # use Docker's default bridge network
else
  RUN_OPTS+=(--network none)
fi

# Host audio: mount the WSLg PulseAudio socket and point clients at it.
PULSE_SOCK="/mnt/wslg/PulseServer"
if [[ "${AUDIO:-0}" == "1" ]]; then
  if [[ -S "$PULSE_SOCK" ]]; then
    RUN_OPTS+=(
      --mount "type=bind,source=$PULSE_SOCK,target=/tmp/PulseServer"
      -e "PULSE_SERVER=unix:/tmp/PulseServer"
    )
    echo "Audio: routing to host speakers via WSLg PulseAudio."
  else
    echo "WARNING: --audio requested but $PULSE_SOCK not found; audio will be silent." >&2
  fi
fi

# Gradio UI port.
if [[ "${PUBLISH_UI:-0}" == "1" ]]; then
  RUN_OPTS+=(--publish 127.0.0.1:7860:7860 -e GRADIO_SERVER_NAME=0.0.0.0)
  echo "Gradio UI will be available at http://127.0.0.1:7860"
fi

# Pass the OpenAI key in only when networking is on (the only time it's usable).
if [[ "${NETWORK:-0}" == "1" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" && -r "$KEY_FILE" ]]; then
    OPENAI_API_KEY="$(tr -d '[:space:]' < "$KEY_FILE")"
  fi
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    RUN_OPTS+=(-e OPENAI_API_KEY)
    export OPENAI_API_KEY
  fi

  # Same for the Anthropic key, so `claude` can authenticate non-interactively.
  if [[ -z "${ANTHROPIC_API_KEY:-}" && -r "$ANTHROPIC_KEY_FILE" ]]; then
    ANTHROPIC_API_KEY="$(tr -d '[:space:]' < "$ANTHROPIC_KEY_FILE")"
  fi
  if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
    RUN_OPTS+=(-e ANTHROPIC_API_KEY)
    export ANTHROPIC_API_KEY
  fi
fi

# Default command is an interactive shell.
if [[ ${#CMD[@]} -eq 0 ]]; then
  CMD=(bash)
fi

echo "Sandbox: only $PROJECT_DIR is mounted (at /workspace). Network: ${NETWORK:-0}. Audio: ${AUDIO:-0}."
exec docker run "${RUN_OPTS[@]}" "$IMAGE" "${CMD[@]}"
