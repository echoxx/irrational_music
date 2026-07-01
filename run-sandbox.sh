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

# Docker's daemon runs in its own WSL2 distro and cannot bind-mount Windows
# drive letters exposed under /mnt (Google Drive, etc.).  Depending on the WSL
# version these show up as 9p/v9fs or DrvFs and any bind-mount fails with
# "no such device".  When the project lives on such a path we transparently
# mirror it to a local ext4 copy under $HOME that Docker *can* mount, run the
# sandbox there, and sync changes back on exit.  Set SANDBOX_LOCAL_DIR to
# override the mirror location, or point it at an existing local copy to skip
# the auto-mirror entirely.

# Returns 0 if Docker can bind-mount paths on the given dir's filesystem.
_fs_is_mountable() {
  case "$(stat --file-system -c '%T' "$1" 2>/dev/null)" in
    v9fs|9p|drvfs|cifs|fuseblk|nfs|nfs4) return 1 ;;
    "") return 1 ;;   # unreadable / unreachable (e.g. disconnected cloud drive)
    *) return 0 ;;
  esac
}

MIRROR_SYNC=0
if [[ -n "${SANDBOX_LOCAL_DIR:-}" ]]; then
  # Explicit override: trust it, use as-is.
  DOCKER_DIR="$(cd "$SANDBOX_LOCAL_DIR" && pwd)"
elif _fs_is_mountable "$PROJECT_DIR"; then
  DOCKER_DIR="$PROJECT_DIR"
else
  # Source is on a filesystem Docker can't mount — mirror it to local ext4.
  if ! command -v rsync >/dev/null; then
    echo "ERROR: '$PROJECT_DIR' is on a filesystem Docker can't bind-mount" >&2
    echo "  (a Windows/cloud drive under /mnt), and rsync isn't installed to mirror it." >&2
    echo "  Install rsync, or set SANDBOX_LOCAL_DIR to a local copy of the project." >&2
    exit 1
  fi
  _mirror_slug="$(echo "$PROJECT_DIR" | tr '[:upper:]' '[:lower:]' | sed 's|[^a-z0-9]|-|g; s|-\+|-|g; s|^-||; s|-$||')"
  DOCKER_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/irrational-sandbox/$_mirror_slug"
  MIRROR_SYNC=1
  mkdir -p "$DOCKER_DIR"
  # .git is excluded: it's slow to copy over 9p and writing it back could corrupt
  # the Drive-hosted repo; the sandbox gets the working tree, not git history.
  echo "Source is on a non-mountable filesystem; mirroring to $DOCKER_DIR ..."
  rsync -a --delete --exclude='.git/' "$PROJECT_DIR/" "$DOCKER_DIR/"
fi

# Derive a unique image name from the *source* project path so two different
# project sandboxes launched with the same script name never clobber an image.
_dir_slug="$(echo "$PROJECT_DIR" | tr '[:upper:]' '[:lower:]' | sed 's|[^a-z0-9]|-|g; s|-\+|-|g; s|^-||; s|-$||')"
IMAGE="sandbox-${_dir_slug}"
KEY_FILE="${OPENAI_API_KEY_FILE:-$HOME/.config/irrational/openai_api_key}"
ANTHROPIC_KEY_FILE="${ANTHROPIC_API_KEY_FILE:-$HOME/.config/irrational/anthropic_api_key}"

cd "$DOCKER_DIR"

# Build (cached after the first run; only re-installs deps if requirements.txt changes).
# CLAUDE_INSTALL_DATE is today's date — busts only the Claude install layer so
# the binary is re-fetched (at most once per day) without invalidating pip deps.
docker build \
  --build-arg "CLAUDE_INSTALL_DATE=$(date +%Y%m%d)" \
  -t "$IMAGE" "$DOCKER_DIR"

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
    CLAUDE_PERSIST=1 # keep the OAuth login across --rm runs (see below)
    CMD=("${@:2}")   # drop --claude; default below launches the `claude` REPL
    if [[ ${#CMD[@]} -eq 0 ]]; then CMD=(claude); fi
    ;;
esac

# --- assemble run options -------------------------------------------------
RUN_OPTS=(
  --rm -it
  --mount "type=bind,source=$DOCKER_DIR,target=/workspace"
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

# Persist Claude Code's OAuth login across --rm runs.  The container is
# ephemeral, so without this you'd re-authenticate every time.  We keep Claude's
# config in a host dir OUTSIDE the project folder (so it's never on the
# /workspace mount) and relocate Claude's config dir there via CLAUDE_CONFIG_DIR.
# First `--claude` run: log in once; every run after is already authenticated.
if [[ "${CLAUDE_PERSIST:-0}" == "1" ]]; then
  CLAUDE_CFG_DIR="${CLAUDE_CONFIG_DIR_HOST:-$HOME/.config/irrational/claude-config}"
  mkdir -p "$CLAUDE_CFG_DIR"
  chmod 700 "$CLAUDE_CFG_DIR"
  RUN_OPTS+=(
    --mount "type=bind,source=$CLAUDE_CFG_DIR,target=/claude-config"
    -e CLAUDE_CONFIG_DIR=/claude-config
  )
  echo "Claude Code: persisting login in $CLAUDE_CFG_DIR (log in once on the first run)."
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

if [[ "$MIRROR_SYNC" == "1" ]]; then
  echo "Sandbox: local mirror of $PROJECT_DIR is mounted (at /workspace). Network: ${NETWORK:-0}. Audio: ${AUDIO:-0}."
else
  echo "Sandbox: only $DOCKER_DIR is mounted (at /workspace). Network: ${NETWORK:-0}. Audio: ${AUDIO:-0}."
fi

if [[ "$MIRROR_SYNC" == "1" ]]; then
  # Run (not exec) so we can sync sandbox changes — recordings, presets, edits —
  # back to the real project afterwards.  No --delete: only propagate additions
  # and changes, never remove files from the source.
  rc=0
  docker run "${RUN_OPTS[@]}" "$IMAGE" "${CMD[@]}" || rc=$?
  echo "Syncing sandbox changes back to $PROJECT_DIR ..."
  rsync -a --exclude='.git/' "$DOCKER_DIR/" "$PROJECT_DIR/"
  exit "$rc"
else
  exec docker run "${RUN_OPTS[@]}" "$IMAGE" "${CMD[@]}"
fi
