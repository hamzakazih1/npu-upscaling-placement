"""Raw multitouch input from the trackpad.

Every backend produces whole frames: the full list of fingers currently on the
pad, with ``x``/``y`` normalised to 0..1 (x left to right, y bottom to top)
and ``pressure`` 0..1. The engine diffs consecutive frames into
began/moved/ended events, so backends stay simple.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger(__name__)


@dataclass
class Contact:
    id: int
    x: float
    y: float
    pressure: float = 0.6


FrameCallback = Callable[[list[Contact]], None]


class TouchBackend:
    name = "none"
    #: True when ``pressure`` is measured rather than a constant.
    pressure_supported = False

    def start(self, callback: FrameCallback) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        pass


def normalise_pressure(raw: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.6
    return max(0.0, min(1.0, (raw - lo) / (hi - lo)))


def create_backend(settings: dict) -> TouchBackend | None:
    """The native backend for this OS, or None if there is no usable trackpad."""
    choice = settings.get("backend", "auto")
    if choice == "mouse":
        return None
    try:
        if sys.platform == "darwin":
            from padmidi.touch.macos import MacMultitouchBackend

            return MacMultitouchBackend(settings)
        if sys.platform == "win32":
            from padmidi.touch.windows import WindowsPrecisionTouchpadBackend

            return WindowsPrecisionTouchpadBackend(settings)
    except Exception:
        log.exception("native touch backend unavailable, falling back to mouse")
    return None
