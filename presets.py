"""Save/load named control presets for the Gradio UI.

Presets are a flat {name: {param: value}} dict persisted to presets.json in
the project folder (gitignored — they're personal). Values are whatever the
UI controls hold (strings, numbers, lists), so everything is JSON-native.
"""

import json
import os
import threading

PRESETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets.json")

_io_lock = threading.Lock()


def load_presets() -> dict:
    with _io_lock:
        if not os.path.exists(PRESETS_PATH):
            return {}
        try:
            with open(PRESETS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}


def save_preset(name: str, config: dict) -> dict:
    """Add/overwrite one preset; returns the full updated presets dict."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Preset name must not be empty")
    presets = load_presets()
    presets[name] = config
    with _io_lock:
        with open(PRESETS_PATH, "w", encoding="utf-8") as f:
            json.dump(presets, f, indent=2, sort_keys=True)
    return presets


def delete_preset(name: str) -> dict:
    """Remove a preset if present; returns the full updated presets dict."""
    presets = load_presets()
    presets.pop(name, None)
    with _io_lock:
        with open(PRESETS_PATH, "w", encoding="utf-8") as f:
            json.dump(presets, f, indent=2, sort_keys=True)
    return presets
