"""Settings, stored as JSON in the per-user config directory."""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
from pathlib import Path

from padmidi import APP_NAME
from padmidi.modes import MODE_CLASSES

log = logging.getLogger(__name__)

DEFAULTS: dict = {
    "midi": {
        # "" = automatic: a virtual "PadMIDI" port on macOS, the first
        # loopMIDI / loopback port on Windows.
        "output": "",
        # Mixer feedback from the DAW. "" = automatic, "off" = disabled.
        "input": "",
    },
    "hotkeys": {
        "toggle": "<ctrl>+<alt>+m",
        "next_mode": "<ctrl>+<alt>+n",
    },
    "overlay": {
        "position": "bottom",  # bottom, top, bottom-left, bottom-right
        "width": 520,
        "opacity": 92,
        "show_when_inactive": False,
    },
    "touch": {
        "backend": "auto",  # auto or mouse
        "pressure_min": 0.15,
        "pressure_max": 1.05,
        "lock_cursor": True,
    },
    "mode": "keys",
    "modes": {cls.key: copy.deepcopy(cls.defaults) for cls in MODE_CLASSES},
}


def config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / APP_NAME


def config_path() -> Path:
    return config_dir() / "config.json"


def merge(defaults: dict, user: dict) -> dict:
    out = copy.deepcopy(defaults)
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path: Path | None = None) -> dict:
    path = path or config_path()
    try:
        with open(path, encoding="utf-8") as f:
            return merge(DEFAULTS, json.load(f))
    except FileNotFoundError:
        return copy.deepcopy(DEFAULTS)
    except Exception:
        log.exception("bad config file %s, using defaults", path)
        return copy.deepcopy(DEFAULTS)


def save(cfg: dict, path: Path | None = None) -> None:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, path)
