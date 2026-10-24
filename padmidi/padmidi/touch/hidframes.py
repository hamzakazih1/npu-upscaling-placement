"""Assembling Windows Precision Touchpad HID reports into touch frames.

A precision touchpad report carries a fixed number of finger slots plus a
"contact count". Devices with more fingers than slots use hybrid mode: the
first report of a frame carries the total contact count, the following
reports carry a count of 0 and the remaining fingers. Lifted fingers are
reported once more with the tip switch off.

Kept free of ctypes so it can be unit tested on any OS.
"""

from __future__ import annotations

from dataclasses import dataclass

from padmidi.touch import Contact


@dataclass
class RawSlot:
    contact_id: int
    tip: bool
    x: float  # normalised 0..1, HID orientation (y down)
    y: float
    pressure: float | None = None
    confident: bool = True


class FrameAssembler:
    def __init__(self) -> None:
        self._expected = 0
        self._pending: list[RawSlot] = []

    def feed(self, contact_count: int | None, slots: list[RawSlot]) -> list[Contact] | None:
        """Feed one report; returns the finished frame when one completes."""
        if contact_count is None:
            # No contact count usage: every report is a complete frame.
            return self._finish(slots)
        if contact_count > 0:
            self._expected = contact_count
            self._pending = []
        elif self._expected == 0:
            return None  # stray continuation report
        room = self._expected - len(self._pending)
        self._pending.extend(slots[: max(0, room)])
        if len(self._pending) >= self._expected:
            frame = self._finish(self._pending)
            self._expected = 0
            self._pending = []
            return frame
        return None

    @staticmethod
    def _finish(slots: list[RawSlot]) -> list[Contact]:
        out = []
        for s in slots:
            if not s.tip or not s.confident:
                continue
            out.append(
                Contact(
                    id=s.contact_id,
                    x=min(1.0, max(0.0, s.x)),
                    y=min(1.0, max(0.0, 1.0 - s.y)),  # HID y grows downwards
                    pressure=0.6 if s.pressure is None else s.pressure,
                )
            )
        return out


def normalise(value: int, lo: int, hi: int) -> float:
    if hi <= lo:
        return 0.0
    return (value - lo) / (hi - lo)
