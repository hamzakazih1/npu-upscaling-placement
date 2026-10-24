"""Freezing the pointer while the trackpad is a MIDI controller.

While PadMIDI is active the fingers are playing, so the cursor must not
wander across the DAW. The pointer is parked over the overlay (so stray taps
land on PadMIDI rather than on a track) and pinned there.
"""

from __future__ import annotations

import ctypes
import logging
import sys

log = logging.getLogger(__name__)


class CursorLock:
    def __init__(self) -> None:
        self.locked = False
        self._impl = None
        try:
            if sys.platform == "win32":
                self._impl = _WindowsLock()
            elif sys.platform == "darwin":
                self._impl = _MacLock()
        except Exception:
            log.exception("cursor lock unavailable")

    def lock(self, x: int, y: int) -> None:
        """Pin the pointer at physical/global coordinates ``x``, ``y``."""
        if self._impl:
            try:
                self._impl.lock(x, y)
                self.locked = True
            except Exception:
                log.exception("cursor lock failed")

    def refresh(self) -> None:
        """Re-assert the lock (Windows drops ClipCursor on focus changes)."""
        if self.locked and self._impl:
            self._impl.refresh()

    def unlock(self) -> None:
        if self._impl and self.locked:
            try:
                self._impl.unlock()
            except Exception:
                log.exception("cursor unlock failed")
        self.locked = False


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _WindowsLock:
    def __init__(self) -> None:
        self.user32 = ctypes.WinDLL("user32")
        self.rect = None

    def lock(self, x: int, y: int) -> None:
        # Qt has already moved the pointer; read back its physical position.
        pt = _POINT()
        self.user32.GetCursorPos(ctypes.byref(pt))
        self.rect = _RECT(pt.x, pt.y, pt.x + 1, pt.y + 1)
        self.user32.ClipCursor(ctypes.byref(self.rect))

    def refresh(self) -> None:
        if self.rect is not None:
            self.user32.ClipCursor(ctypes.byref(self.rect))

    def unlock(self) -> None:
        self.rect = None
        self.user32.ClipCursor(None)


class _MacLock:
    def __init__(self) -> None:
        self.cg = ctypes.CDLL("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
        self.cg.CGAssociateMouseAndMouseCursorPosition.argtypes = [ctypes.c_bool]

    def lock(self, x: int, y: int) -> None:
        # Decouple mouse movement from the cursor: the pointer stays put while
        # the trackpad keeps delivering raw contacts.
        self.cg.CGAssociateMouseAndMouseCursorPosition(False)

    def refresh(self) -> None:
        self.cg.CGAssociateMouseAndMouseCursorPosition(False)

    def unlock(self) -> None:
        self.cg.CGAssociateMouseAndMouseCursorPosition(True)
